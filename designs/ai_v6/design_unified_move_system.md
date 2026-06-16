# Design — Unified Move System (ai_v6)

**Status:** design + build (NOT shipped, NOT run). Successor to
`design_unified_damage_system.md`. Retrain-class. `MODEL_CONFIG_VERSION` 23 → 24;
`ARCH_SIGNATURE` stays `gen3_wish_wired_v1` (every new piece is flag-gated and byte-identical
when OFF, like v19–v23 — the new tensors are GPU-side, non-persistent, recomputable buffers).

## Goal

Move toward **one unified representation of every move's outcome** so the model makes better
choices. Today move knowledge is fragmented across three places that share no backbone:

| Place | Holds | Limit |
|---|---|---|
| Obs move slot (`moves.py`, 11d) | id, BP, type, category, acc, **`has_secondary`/`has_recoil` bits** | no *what* / *how much* |
| Move embedding (`nn.Embedding(400,16)`) | a pure learned lookup by move-num | no structure; Rock Slide ⊥ HP Rock |
| Damage-op effect block (6 scalars) | belief-weighted MAX of **binary** flags `{recovery,status,phaze,boost,hazard,protect}` | no per-status split, no chance, no flinch, no priority, no Serene Grace |

And belief grading is **rigid per-move-ID BCE** (multi-hot over 400 moves) — "Rock Slide" and
"Hidden Power Rock" are orthogonal classes, so guessing one for the other scores as a total miss.

The user's three asks:
1. A **latent that represents all moves** (status / recovery / side-effect) so the model
   understands outcomes uniformly, and the damage op is enriched to indicate flinch etc.
2. **Grade move guesses in that latent** so Rock Slide ≈ Hidden Power Rock.
3. For **our attacks**, expand "can cause status" to *what* status, with **probability**, and
   handle **Serene Grace** (doubles secondary chance) on either side.

## The backbone: one `MOVE_ATTR` table, three consumers

A single new static buffer `MOVE_ATTR[n_moves, A]` (built once in `damage_tables.py` from the
newly-extracted move fields) is the spine. It feeds:

```
        data unlock: extract secondary{chance,status,volatile,boosts}, priority, drain, recoil
                                        │
                            MOVE_ATTR[n_moves, A]   ← single source of truth for "what a move does"
                          /             │              \
       (1) richer op       (2) MoveLatentEncoder        (3) belief-grading target
       per-status P,        context-free f(emb⊕attr)     cosine in latent space →
       Serene Grace         → mechanics-grounded         Rock Slide ≈ HP Rock
       identity
```

DRY by construction: the op's flinch chance, the latent's "this is a flinch move" axis, and the
grading similarity all read the **same** columns. The `_DMG_IDX_*` layout stays the op's single
source; `MOVE_ATTR` is the attribute single source.

## Stage 0 — Data unlock (retrain-class; land at a clean boundary)

`build_moves` currently drops secondary details by an explicit past decision ("the status is
incidental"). We reverse that. Upstream poke-env static data already has everything.

**New `gen3_moves.json` fields** (obs values UNCHANGED — these ride GPU-side, not the obs vector):

```json
"thunderbolt": { ...existing...,
  "priority": 0,
  "secondaryEffects": {"par": 10},     // column → percent (only nonzero)
  "drainFraction": 0.0,
  "recoilFraction": 0.0 }
```

- `secondaryEffects`: dict over **10 columns** `[par, brn, frz, slp, psn, tox, confusion, flinch,
  foe_statdrop, self_boost]` → trigger percent. Normalized from `secondary` + `secondaries`:
  `status`→its column, `volatileStatus`→flinch/confusion, foe `boosts`→`foe_statdrop`,
  `self.boosts`→`self_boost`. Tri Attack's `onHit` (20% par/brn/frz split) is a curated override
  `_SECONDARY_ONHIT` (the Belly-Drum/Refresh callback-invisible precedent).
- `priority` (int), `drainFraction` = drain[0]/drain[1], `recoilFraction` = recoil[0]/recoil[1].

**Facade** (`gen3_data/moves.py`): `MoveData` gains `priority:int`, `drain_fraction:float`,
`recoil_fraction:float`, `secondary_effects: Tuple[Tuple[str,int],...]` (hashable; frozen-safe) +
accessor `secondary_chance(col)->float`. `damage_tables` reads the raw JSON for the buffer build
(the established pattern).

Guards: regenerate via `sync.py`; `extractor_parity_test` + the **obs golden** must stay green
(obs unchanged → golden unchanged; only the json schema grows).

## Stage 1 — Richer damage op (`DamageOperator`)

Replace the 6 binary effect scalars with **per-status probabilities**, computed in the op (the
multiply lives in the differentiable physics; the ReLU head stays additive — the accuracy-fold
precedent):

- **New buffers** (`damage_tables.py`, non-persistent): `MOVE_SECONDARY[n_moves, 10]` (per-effect
  chance ∈[0,1]), `MOVE_PRIORITY[n_moves]`, `MOVE_DRAIN[n_moves]`, `MOVE_RECOIL[n_moves]`,
  `ABILITY_SECONDARY_MULT[n_abilities]` = **2.0 Serene Grace, 0.0 Shield Dust, 1.0 else**.
- **Incoming effect block**: per-effect probability `P_k = chance_k × ability_mult`, gathered over
  the opp-active's candidate moves and **belief-weighted** (`w_m = sigmoid(move_belief_logits)`):
  expected `Σ_m w_m·P_mk` for magnitude **and** a `max_m w_m·1[P_mk>0]` "threat-exists" bit. Gated
  by the **opp active's** `ABILITY_SECONDARY_MULT` (their Serene Grace / our reading of it).
  Structural flags `{recovery(magnitude via drain/heal), phaze, hazard, protect}` retained.
- **NO speed coupling.** Flinch is exposed as raw `P(flinch) = flinch_chance × serene` — the op
  does **not** multiply by `p_first`. The transformer/Pokémon encoder already sees speed and
  learns "flinch only helps if I move first" itself (provide-the-fact, learn-the-interaction).
  `p_outspeed` stays a separate pure-speed per-mon feature.
- **Priority** is a **raw attribute** (in `MOVE_ATTR`/latent and an outgoing feature), **not**
  folded into `p_outspeed` — ordering is a per-matchup thing attention handles.
- **Outgoing (our 4 moves, action-aligned)**: per-move effect head
  `{P(any-status), P(flinch), self-heal-frac, priority}`, gated by **our** active's ability
  (your Serene Grace). Analogue of the outgoing-damage win.

Layout: extend `_DMG_EFFECT` (incoming), add an outgoing-effect group; update `out_dim`,
`decode_damage_block`, `last_raw_block`/`last_damage_block`. Exact column counts are a build
detail pinned by tests; the `_DMG_IDX_*` constants remain the single source.

## Stage 2 — `MoveLatentEncoder`

A **context-free** module: `concat(move_embedding(id), MOVE_ATTR[id]) → Linear→ReLU→Linear →
move_latent` (new const `MOVE_LATENT_DIM`). Unlike the existing 32-d move-network output (mixed
with HP/turn/matchup context), this is a stable *move identity* grounded in mechanics. Routed
**into** the move network as the move's representation (augments the raw embedding input). Because
the attribute vector dominates, Rock Slide and HP Rock land adjacent.

**Hidden Power** gets the same typed expansion the op already does (reuse `HP_TYPE_IDX`) so
HP-Rock's latent actually carries Rock — the exact case the goal names.

## Stage 3 — Latent belief grading (`instrumented_ppo.py`)

Keep the per-move-ID BCE (the op needs an actionable per-ID distribution to gather buffers) and
**add** a SimSiam-style aux mirroring the proven species latent head:
`expected_latent = sigmoid(move_logits) @ MoveLatentEncoder.weight`, graded by **cosine toward the
stop-grad latent of the true (unrevealed) moves** + a VICReg variance floor. New coef
`--move-belief-latent-coef` (requires move-belief on + move-latent on). Hard BCE still teaches
exact identity; the latent aux relaxes the penalty for near-misses so the belief generalizes to
the right *region* of move space — and the op gets a better damage/effect estimate on never-seen
exact moves.

**Leak-safe** (the iron rule): the move-latent target + predicted latents are stashed for the loss
only, **never** in pi/vf; `is_grad_enabled`-gated so rollout skips it.

## Flags / versioning / safety

- Umbrella `--unified-moves` desugars to the three (`--damage-effects`, `--move-latent`,
  `--move-belief-latent-coef`), consistent with `--unified-damage`. All OFF by default.
- `MODEL_CONFIG_VERSION` 23→24; `ARCH_SIGNATURE` unchanged. Migration block + `check_compatible`
  gates (`damage_effects`/`move_latent` structural like `damage_op`; the coef forward-behavior
  like `move_belief_coef`) + `from_policy_kwargs` reads + `current_model_version` threading +
  `_run_arch_toggles` to **all 4 opp-load sites** (pool/stable/eval-sentinel/distill) so a
  belief-ON self-play run doesn't FATAL. Coefs read-back on resume.

## ML / AI principles

- **Provide raw known facts, let the model learn the value/interaction.** Chance, status-type,
  priority, drain are GIGO-proof facts → provide. Never hand-code "paralysis is worth X" or
  couple flinch to speed in the op.
- **Compute deterministic / learn epistemic / represent aleatoric.** Secondary chance is a known
  coin-flip (represent as P); Serene Grace doubling is deterministic given ability (compute);
  which-move is epistemic (learn) — and **marginalize over the move belief**, don't mean-field.
- **Grade in the structured metric, not one-hot.** The species latent head already proved
  latent-space grading works (role geometry AUC 0.8–1.0); moves have *more* explicit structure.
- **Multiply in the op, keep the head additive** (chance × ability), but **leave board-state
  interactions (speed/ordering) to attention** — they are per-matchup, not within-move physics.

## Risks / honest cons

1. **Redundancy** — the 400×16 embedding can already learn move similarity from gradient; the
   structured latent mostly accelerates/regularizes it. Marginal value unmeasured (role-probe
   caution: "decodable ≠ helps").
2. **Grading-helps-loss ≠ helps-policy** — the op still reads the per-ID distribution; softer
   grading improves belief *generalization*, policy benefit is plausible, not proven.
3. **Learnable-but-inconsequential** — verify secondary-driven craters exist in the loss corpus
   (prober `falsify-scan`) before crediting Stage 3.

## A/B gate (the verdict)

Fresh run vs the current arch: belief precision↑ (per-ID + latent cosine) **AND**
surprise-secondary / wrong-effect crater share↓ (prober `falsify-scan`) **AND** win-rate
non-regress. Serene Grace is a genuine lever here — gen3 Jirachi (Serene Grace) is ~25% OU usage
(only gen4 Jirachi is Ubers), so Serene-Grace-doubled Body Slam/Thunderbolt/Fire Punch para/burn
is common, not a footnote. Update the `research_state/` lever file + ledger row in the same pass.

## Test plan

- `damage_tables_test`: Thunderbolt 10% par, Rock Slide 30% flinch, Zap Cannon 100% par, Crunch
  20% foe_statdrop, Meteor Mash 20% self_boost, Serene Grace 2×, Shield Dust 0×, Tri Attack split.
- `damage_op_test`: per-status incoming probs (belief-weighted), outgoing per-move effects, Serene
  Grace doubling both sides, NO speed in flinch, decode round-trip.
- `move_latent_test`: HP-Rock ≈ Rock Slide cosine; context-free; HP typed expansion.
- belief-latent aux: leak-safe (never in pi/vf), grades near-moves as near; gradient-flow + fuzz.
- migration (v24 OFF byte-identical, resume coef read-back); extractor-parity + obs golden after
  regen; full-stack bridge smoke roundtrip (`[ModelVersion] Round-trip smoke test PASSED`).

## Further-unification backlog (redundancies this work exposed)

Building `MOVE_ATTR` as a single backbone made it clear that move info is now encoded in several
overlapping places. Ranked by payoff / risk:

1. **`MOVE_EFFECT_FLAGS` ⊂ `MOVE_ATTR` (cheap DRY win).** The op builds a separate
   `MOVE_EFFECT_FLAGS[n,6]` (recovery/status/phaze/boost/hazard/protect) AND `MOVE_ATTR` already carries
   is_heal/is_boost/is_protect/is_phaze/is_hazard (5 of 6) + the 10 secondary cols. Only the
   *primary-status* flag is missing. Add one `primary_status` column to `MOVE_ATTR`, drop
   `MOVE_EFFECT_FLAGS`, have the op read effect flags from `MOVE_ATTR` columns → one source for every
   per-move flag. Contained to `damage_tables` + the op effect block.
2. **Static move facts are TRIPLE-encoded (big payoff, real risk → A/B-gated).** The same static
   properties live in (a) the CPU obs move slot (BP/type/category/accuracy/never_miss/has_secondary), (b)
   the CPU reactive move-effect block (is_boost/heal/protect/phaze/hazard/inflicts_status/cures_*), and (c)
   `MOVE_ATTR` (a superset). The obs blocks also carry DYNAMIC state (PP, known flag, live
   `status_will_land`). End-state: source the STATIC facts ONCE via the move latent and keep only the
   dynamic fields in the obs (~12–15 dims/move-slot saved). Risk: forces the model through a LEARNED
   bottleneck for facts it now reads raw (raw BP is trivially linear-readable; a latent of it is not) →
   only pursue if the move-latent A/B shows the latent is strongly used. Retrain-class + obs perf gate.
3. **CPU incoming-damage block already A/B-redundant with the op (cleanup, already wired).** The 51-dim
   CPU `incoming_damage` obs block and the op's incoming block compute P(KO) two ways. `--mask-incoming-
   damage-obs` already A/Bs removing the CPU block from the model's view; if the A/B shows the op subsumes
   it, DELETE the CPU block (−51 obs dims + the obs-side damage code). The cleanest unification — needs the
   run to confirm, then the deletion.
4. **Two `p_outspeed` computations + the speed↔status disconnect.** `p_outspeed` is computed separately in
   the op's incoming (per-defender) and outgoing (our-active) blocks (same logistic, duplicated). More
   importantly it is a point estimate that IGNORES paralysis (gen3 cuts speed to 25%) and boosts — and the
   op now KNOWS which moves cause paralysis (the new secondary data) but the speed calc doesn't consume the
   resulting status. Unify into one shared `p_outspeed` that folds active-mon status/boosts.

## v2 op follow-ups (named rough edges)

- **Fixed-damage type immunity ("Seismic Toss into Ghosts").** Seismic Toss / Night Shade are fixed-damage
  (BP 0 in data → the op treats them as non-damaging, so it never tells the policy "100 fixed damage,
  EXCEPT 0 vs Ghost by Normal-type immunity"). Model fixed-damage in the op (level-HP, type-gated to 0 vs
  the immune type). The move latent now carries the type, so the immunity is learnable, but the explicit
  damage signal is the real fix.
- **Speed + status** (item 4 above): fold paralysis (×0.25 speed, gen3) + boosts into `p_outspeed`, so
  "I paralyze them → I outspeed next turn" is priced (the secondary-par data is now available to drive it).

## v25 — Spread/speed belief + the disable-redundant master flag (BUILT, `gen3_unified_spread_belief_v1`)

The THIRD belief leg (moves ✓, species ✓, **stats**). Same proven pattern as the move belief: a usage
PRIOR ⊕ a learned head, reinjected into the opp token, consumed by the op (replacing hand-coded constants).

- **Prior** (`damage_tables.build_opp_spread_prior` → `[n_species, 5, 2]`): usage-weighted `(mean, std)` of
  each species' realized L100/IV31 stat for `{atk,def,spa,spd,spe}`, from the Smogon spreads
  (`gen3_data.priors.spreads`, already on disk). Aerodactyl reads 392 speed, Blissey 50 atk / 306 spd,
  Tyranitar atk ±47 (DD vs CB) — far better than the flat de-timid/neutral constant. HP skipped (the op
  keeps a neutral maxhp × the obs HP fraction). Non-persistent buffer.
- **`SpreadBelief`** (mirrors `MoveBelief`): `believed = prior_mean + delta·prior_std` (zero-init head →
  cold-start == prior, the clean A/B baseline), the delta reinjected as a small residual into revealed opp
  slots. Predicts **derived stats** (not EVs+nature) so the op consumes the value directly — keeping "the
  op does the multiplicative physics, the head reasons additively." Stash `last_spread_belief [B,6,5]`.
- **Op consumption** — the `DamageOperator` `forward` (opp attacker atk/spa + `p_outspeed` opp speed) and
  `_outgoing_block` (opp defender def/spd + speed) gather the believed stats at `ctx.opp_active_local` and
  use them in place of the de-timid `252/×1.1` / neutral-0-EV constants. `spread_belief=None` → legacy
  constants (byte-identical). So the op's opponent stats are a **learned belief, not a fixed guess** — a
  unification (the op stops reading hand-coded constants). The offensive/defensive stats get RL gradient
  via the damage rolls; speed gets weak gradient via `p_outspeed` → the supervision below is its accelerant.
- **Disable-redundant master flag** `--unified-obs`: flips three forward-behavior obs masks in `ObsUnpack`
  (clone-once, zero the region from the model's view, reward/PBRS untouched, offsets from named
  `reactive_layout` entries): `mask_incoming_damage_obs` (51-dim, → the op), `mask_active_move_scalars_obs`
  (move_power+multiplier 8-dim, → the op outgoing block, requires `--damage-outgoing`),
  `mask_move_effects_obs` (44-dim, → MOVE_ATTR/the move latent). Granular flags underneath. The pure-unified
  run = `--unified-moves both --spread-belief --unified-obs`.
- **Versioning** v25: `spread_belief` (structural, check_compatible) + `spread_belief_coef` (training-only)
  + the 2 masks (forward-behavior). Threaded through every site; OFF byte-identical; arch_toggles to the 4
  opp-load sites. Verified: 2647 unit + the spread-belief tests + full-stack `--unified-obs` smoke roundtrip.

### Staged next (designed, not yet built)

1. **Speed SUPERVISION** (`--spread-belief-coef`, the flag exists; the loss is the remaining piece) —
   observed MOVE ORDER (`TurnView.we_moved_first`, public + already in the TurnDelta) is a leak-safe
   inequality label on the hidden opp speed (we know our speed exactly). Build: a training-only
   `speed_first`+`speed_mask` obs key (`belief_labels.py` + `Gen3Env`, mirroring `belief_moves`), masked to
   CLEAN turns (both sides used a non-priority move, neither paralyzed — the scout's confound guards), and a
   masked BCE of the believed `p_outspeed` toward it (`instrumented_ppo`). This is what makes SPEED learn
   well (the RL gradient on speed alone is weak). The observed-DAMAGE leg (back-solve atk/spa/def/spd) is a
   noisier follow-on.
2. **`MOVE_EFFECT_FLAGS` ⊂ `MOVE_ATTR`** — add a `primary_status` column to `MOVE_ATTR`, have the op read
   the 6 effect flags from `MOVE_ATTR`, drop the separate buffer (one source for every per-move flag).
3. **Fixed-damage type immunity** ("Seismic Toss into Ghosts") — model Seismic Toss / Night Shade as
   level-HP damage in the op, type-gated to 0 vs the immune type.
