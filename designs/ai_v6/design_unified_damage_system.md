# Unified Damage System (ai_v6)

**Goal:** collapse the three disparate damage/move systems into ONE differentiable GPU engine
driven by ONE move posterior, with a clean DRY interface, and extend it to the directions and
representation the owner needs. *Next run's goal: not have the 3 disparate damage systems.*

## The three systems today (what we unify)

| | System | Where | Source of moves | Differentiable | Consumer |
|---|---|---|---|---|---|
| A | CPU incoming block (`gen3_incoming_crit_split_v1`, 51d) | `observation/incoming_damage*.py` | fixed usage prior | no | obs (model) **and** reward PBRS (`live_view`) |
| B | GPU `DamageOperator` (incoming only, 54d) | `model/features_extractor.py` + `damage_tables.py` | learned move belief | **yes** | both projection heads |
| C | Move-belief fusion (belief/known/priors) | `MoveBelief` (`--move-belief-mode`/`--move-prior-fusion`) | one posterior already | yes | reinjected into tokens + the op |

System C's prior-fusion (v20) **already** unifies belief/known/priors into one tensor
`last_move_belief_logits` — `posterior = prior(species) + head_delta`, revealed pinned to
`_REVEAL_LOGIT`. The remaining work is to (1) make that posterior THE only path, (2) drive ONE
engine off it for all directions, (3) bound it with a learnset + rarity cap, (4) keep the CPU block
only as the reward potential (model reads the GPU engine via `--mask-incoming-damage-obs`), and
(5) collapse the 4 flags into one coherent surface.

## Decisions (owner-confirmed)

- **Directions:** incoming + safe-switch + **outgoing**, each flag-guarded.
- **Damage representation:** per channel `[low_roll, high_roll, crit, pko, accuracy]` (5 feats).
  The 3 rolls are damage IF it lands (fraction of MAX HP); `pko = acc·P(KO|hit)` is the **exact realized
  KO-this-turn probability** (accuracy and the damage roll are independent events, so the product is
  correct, not an approximation); `accuracy` is the dominant threat's hit rate. `{pko, accuracy}`
  together = the full miss/survive/KO distribution. **Why this shape (the ReLU argument):** ReLU heads
  can't easily *learn* a product, so the operator does every multiplication and hands the head
  pre-computed, additive-friendly scalars — folding accuracy into `pko` in the op is the ReLU-CORRECT
  choice (the alternative, exposing `acc` and `ko_hit` separately, would force the head to multiply).
  `accuracy` is additionally exposed because it's the one piece of the outcome distribution not
  recoverable from the rest, and it matches how our own moves' accuracy already rides the obs move-block.
- **Outgoing is PER-MOVE, action-aligned.** Justification (owner, from human-replay analysis): the
  model picks the wrong move between two **same-effectiveness** moves (Earthquake vs Meteor Mash vs
  a Rock; Earthquake vs Surf vs a Rock) — the type-multiplier obs feature can't break the tie, only
  resolved damage (BP, STAB, physical-vs-special hitting Def vs SpD) can. So outgoing exposes each of
  our 4 moves' resolved damage in **request-slot order** (action logit `6+k`, like
  `gen3_move_slot_align_v1`), so the policy head compares move A vs move B directly. A collapsed
  "best move" scalar would NOT fix this failure.
- **Build outgoing regardless** (no falsifier pre-gate). The same-effectiveness-tie evidence already
  shows it is not redundant with the trunk's move-type representation.
- **DRY, with nice interfaces** — the loud requirement. See below.

## DRY architecture — one kernel, one layout spec

The damage math exists in two runtimes that genuinely cannot share code (numpy CPU for obs+reward,
fast + non-diff; torch GPU for the op, batched + differentiable). DRY is achieved by sharing the
**spec**, not the kernels, and by role-parameterizing each kernel:

1. **`damage_layout.py` (NEW, single source of truth for the feature contract).** Named field
   offsets for a per-(defender,channel) slot `[IDX_LOW, IDX_HIGH, IDX_CRIT, IDX_PKO]`, the channel
   order, `per_mon`/effect widths, and the gen3 damage constants currently duplicated
   (`_L_TERM=42`, roll min `0.85`, `_CRIT_P=1/16`, clamp caps). Imported by the CPU core, the GPU
   op, the reward reader, and the prober. *No magic offsets anywhere.* (Today `incoming_damage.IDX_*`
   and the op's hardcoded `torch.stack` order are two separate layouts — this merges them.)
2. **One role-parameterized kernel per runtime.** CPU `incoming_damage._threat(attacker, defender,
   screens, hp_denom)` and GPU `UnifiedDamageOperator._threat_block(attacker_stats, attacker_types,
   move_weights, defender_stats, defender_types, defender_abilities, defender_hp, defender_screens)`
   — both compute the SAME `[low, high, crit, pko]` per channel; all directions are calls with
   swapped roles, never copy-pasted blocks.
3. **One direction descriptor.** A small frozen dataclass `Direction(attacker_slot, defender_slots,
   defender_screen_idx, hp_denom, move_weight_source)` so incoming/outgoing/safe-switch are three
   descriptors fed to one dispatch, not three forward branches.

## The one posterior (unchanged)

`last_move_belief_logits [B,6,M]` from `MoveBelief` under always-on prior fusion. Per move m:
revealed → certain (`sigmoid(10)`, zero grad); unrevealed known-species → `prior(species)+delta`
(gradient flows here — the surprise-OHKO lever); hidden-species slot → floor prior + delta.

## The engine — three directions

| Direction | Attacker | Defender(s) | Move weights | Shape | Built when |
|---|---|---|---|---|---|
| **incoming** | opp active (revealed species; hidden spread → de-timid OFFENSE) | our 6 (real spread) | `sigmoid(posterior[active])`, channel-MAX | per-mon `[phys(4), spec(4), p_outspeed, provenance]` ×6 + 6 effect | `incoming`/`both` |
| **safe-switch** | opp active | our **bench** (5) | same as incoming | the bench rows of the incoming call (`[low,high,crit,pko]` reduced to `[high,pko,p_outspeed]` per channel — the "is this pivot safe" core) | `both` + `--unified-damage-safe-switch` |
| **outgoing** | our active (real spread; **our burn** halves phys) | opp active (revealed types; **opp-side** screens; **opp Def/SpD = prior MEAN**, not max-bulk; opp ability = revealed-or-prior immunity) | our 4 **revealed moves**, one-hot (certain), **request-slot order**, legality-masked (Choice/Disable/Taunt/PP) | per-move `[low,high,crit,pko]` ×4 (action-aligned) + p_outspeed | `both` |

All gated to a clean zero when `has_opp==0`, per fainted defender, and (outgoing) when our active is
absent/fainted. Appended LAST to both `pi_parts`/`vf_parts` so off-by-default widths auto-size.
GPU-operator outputs, **not** CPU obs blocks → zero obs-dim change, no `ARCH_SIGNATURE` bump,
OFF byte-identical, obs-perf gate untouched, and differentiable in the belief where it matters.

### Faithfulness fixes the review caught (do NOT ship outgoing without these)
- opp-side screens are `screen_feature[:,1]`/`[:,3]` (NOT our `[:,0]`/`[:,2]`).
- our-active **burn** halves our physical outgoing (public — read the status one-hot).
- opp defender spread = `priors.stat_distribution` **mean** (max-bulk systematically under-prices our
  KOs — the worst bias exactly where revenge-kill tempo matters).
- mask outgoing moves to currently-legal slots (Choice-lock/Disable/Taunt/PP) so we don't price a
  phantom move the action mask forbids.
- our-active paralysis (public) quarters our speed in `p_outspeed`.

**Deferred to v2 (owner-confirmed):** recoil self-damage (outgoing) and drain/lifesteal (Giga Drain —
opp sustain on incoming + our sustain on outgoing). Both follow the accuracy pattern (a `frac·damage`
product the op should compute, not the head), but both need a `tools/` data field we don't yet store
(`recoil_fraction` / `drain_fraction`). The model still sees `has_recoil` as a bool in the obs move-block.
Weather, burn, defender boost stages, fixed-damage/OHKO moves remain v2 as before.

## Learnset gate (LEGALITY-only — revised)

- **`gen3_data.learnset` (NEW facade)** + **`data/pokemon/gen3_learnset.json`** (derived by
  `tools/.../sync.py::build_learnset`, gen3-legal = any learnset method code starting with `"3"`;
  validate each move exists in `gen3_moves.json`; spot-checked vs known gen3OU sets).
- **Legality-only gate at BUFFER-BUILD time** (perf gate untouched): `build_move_prior_logits` gains
  `learnset_gate` + `floor` params. With the gate ON: a move a species **cannot learn** → `logit(eps)≈0`
  (impossible — this is the only thing pruned, and it removes the phantom-threat noise the old flat 0.02
  floor invented, e.g. "a special attacker might have Explosion"); a **legal** move keeps its **true**
  Smogon usage (a rare tech stays rare-but-present — naturally negligible in the op's hard-max, yet
  liftable by the learned head + pinned certain the moment it's revealed); a legal-but-unobserved move
  gets the small `floor` base. **No rarity cap.** (An earlier draft pruned legal sub-`floor` moves to ~0;
  that crippled surprise-move anticipation — the very thing the move belief exists to learn — and is
  removed. Because any move with recorded usage is necessarily legal, the legality mask only ever bites
  the ABSENT cells.) HP's typed usages sum into `num` 237 (legal iff bare `'hiddenpower'` is learnable).
- **Default OFF = byte-identical** to today's 0.02 floor (the gate is a version-checked
  forward-behavior change, not a silent prior tightening — review blocker #1).
- **Revealed moves are unaffected either way** — `MoveBelief` pins a revealed move to `_REVEAL_LOGIT`
  (certain), overriding the prior; the gate only shapes the *unrevealed* guess.

## Flag collapse + versioning (`MODEL_CONFIG_VERSION` 22 → 23)

| New flag | Replaces | Class |
|---|---|---|
| `--unified-damage {off,incoming,both}` | `--damage-op` + `--move-belief-mode` + `--move-prior-fusion` | structural (widens projections) → `check_compatible` |
| `--guess-unrevealed-moves` | the `revealed` vs `both` axis of `--move-belief-mode` | forward-behavior → `check_compatible` |
| `--unified-damage-safe-switch` | (new) | structural |
| `--move-candidate-floor` (float, 0.02) | (new — the learnset/rarity gate enable + threshold) | forward-behavior |
| `--mask-incoming-damage-obs` | retained unchanged | forward-behavior |

**Wiring (DRY, single source of truth):** `--unified-damage` **desugars** at arg-parse into the
existing `(move_belief_mode, damage_op, move_prior_fusion)` extractor kwargs via one
`_unified_to_kwargs()` used at BOTH fresh-build and resume, so the recorded `ModelVersion` and the
built module can never disagree (review major). Keep the legacy ModelVersion fields for
`_migrate_config` back-compat. NO `ARCH_SIGNATURE` bump (OFF builds no module). Thread the new
fields through `current_model_version` / `arch_toggles_from_model` at all **4 opp-load sites**
(pool / stable / eval-sentinel / distill) so a unified-ON self-play run doesn't FATAL on its own
sentinels.

## Leak safety (unchanged, re-pinned)

The engine reads only `last_move_belief_logits` (the model's prediction) + public obs. Privileged
labels stay training-only Dict keys consumed by the aux losses, never in `pi/vf`. New `no_leak` fuzz:
op output bit-identical with/without the privileged keys present.

## The reward coupling (why the CPU block stays)

A PBRS potential must be a fixed model-independent function of state, so the reward keeps reading the
CPU `incoming_damage` core from `live_view`. The CPU **obs block** migrates to the 3-roll+P(KO)
representation too (one definition, shared with the GPU op via `damage_layout.py`); the model's
*view* of it is what `--mask-incoming-damage-obs` removes. So "3 systems" → ONE GPU engine (model
view) + the SAME CPU core retained solely as the reward potential. (Caveat: don't claim the mask is a
*faithful* replacement until the op carries the CPU block's `recovery_known` provenance — review
major; either add a provenance channel or exclude the 3 recovery scalars from the mask.)

## Phased plan (green tests at each gate)

- **P0 — learnset facade + rarity-bounded prior (data layer).** `build_learnset` + `gen3_learnset.json`
  + `gen3_data/learnset.py` + export; extend `build_move_prior_logits(learnset_gate, rarity_floor)`
  (default OFF = byte-identical). Tests: `learnset_test.py`, parity, prior-floor unit (illegal +
  <2% → ~0; legal ≥2% → real prior; HP split handled). **No model/version change.**
- **P1 — `damage_layout.py` + role-parameterize the op** (rename `UnifiedDamageOperator`,
  `_threat_block`), incoming byte-identical on the torch==numpy oracle (still 600/600).
- **P2 — 3-roll + P(KO) representation** (CPU core + GPU op together, via `damage_layout`), update the
  oracle, the reward reader, width tests. Retrain-class for the CPU obs block (bump `ARCH_SIGNATURE`).
- **P3 — flag collapse → `--unified-damage` (config v23)**: ModelVersion fields, `_migrate_config`,
  `check_compatible` gates, desugar, 4-site threading. Wire `move_candidate_floor` to enable the P0 gate.
- **P4 — outgoing (per-move, action-aligned)** with all the faithfulness fixes + legality mask + the
  outgoing torch==numpy oracle (roles swapped, opp defender, opp screens, our burn).
- **P5 — safe-switch bench matrix**, bit-identical to the incoming bench rows.
- **P6 — no-leak fuzz, obs-perf gate, doc sync, single-variable A/B ladder** (unify-incoming →
  +learnset → +safe-switch → +outgoing; separate belief-precision from wr so the differentiable lever
  is credited to the arm that adds it).

## Honesty gate

Wired + differentiable + clean ≠ helps the policy. The fresh-run A/B must show belief precision↑
AND surprise-OHKO crater share↓ (`falsify-scan`) AND wr/ELO non-regress. Outgoing carries no belief
gradient (our moves are certain) but is justified by the same-effectiveness-tie evidence; its gate is
the move-selection mix on those ties, not belief precision.

## Deferred op-coverage gaps (per-move outcome) — BACKLOG, to remember

The differentiable op is NOT yet gen3-complete on a few per-(move, defender) outcomes that are damage/
decision-relevant. Captured here so they aren't lost (surfaced by the 2026-06-17 per-move-spec panel).
None are in v1; revisit when enriching the per-move top-K row or before deleting the CPU obs:

- **Absorb abilities = HEAL, not 0.** Water Absorb / Volt Absorb / (Flash Fire's boost) make the move read
  a *negative* outcome for the attacker = a **great-switch** signal for us. The op currently folds these to
  damage 0 (immune), losing the "this pivot HEALS off their move" read. Highest-value gap.
- **Substitute-break flag.** Per (move, defender): does the hit's damage ≥ the defender's Sub HP (breaks it)
  vs chip it. Changes whether a Sub pivot is safe.
- **Multi-hit moves** (Rock Blast / Bullet Seed / Double Kick etc.): n-hit damage + Sub-break interaction.
- **OHKO moves** (Fissure / Horn Drill / Sheer Cold / Guillotine): fixed-KO-if-hit, accuracy-gated,
  level/ability (Sturdy) immunity — currently mispriced by the roll formula.
- **Leech Seed vs Substitute (gen3 nuance):** the v27 status-landing blocks ALL status moves on a Sub, but
  gen3 Leech Seed's interaction needs a dedicated rule; also Leech-Seed-already-seeded + **Yawn** (sleep
  next turn) are uncovered.

## Accuracy is a STANDALONE per-move signal (decision, 2026-06-17)

Keep `accuracy` exposed per move **in addition to** folding it into `pko` (`acc·P(KO|hit)`) — they answer
different questions. `pko` is the miss-OR-survive collapsed KO odds (matters when the move WOULD kill);
raw `accuracy` is the distinct miss signal for **chip and status moves that never KO but can still whiff**
(Hypnosis / Will-O-Wisp / Focus Blast risk). NOT a redundant double-expose — it is the provide-the-fact-
once principle applied to two genuinely different downstream uses. (Overrides the per-move-spec critic's
"don't double-expose accuracy" note.)
