# Design — Hidden-Attribute Inference Trackers: Choice-Band Elimination + Speed-from-Turn-Order (ai_v5)

> **PROPOSED 2026-06-06 — research-backed, no code yet.** Grounds the Phase-2 items deferred
> by `design_incoming_damage_obs.md` (§8 open decisions: the CB worst-case channel and the
> speed-uncertainty refinement). Two inference systems (Choice-Band elimination and
> speed-from-turn-order), both the **Hidden Power tracker pattern** reused: a per-episode,
> per-opponent-species posterior over a hidden attribute, seeded from a Smogon usage prior, narrowed
> in `EpisodeTracker.record()` *before* the encoder runs, threaded into the encoder as a plain
> `tracker=` kwarg, and fed to the model as a **calibrated belief + a separate known/evidence flag —
> never an argmax**.
>
> Every mechanic below is verified against the project's **bundled Showdown source**
> (`deps/pokemon-showdown/data/mods/gen3/` + the `gen <= N` gates in `sim/`), the sim our env
> actually runs — not wikis (which disagreed). Two scope decisions from the requestor are baked in:
> **(1)** speed signals are limited to equal-priority move order + same-effect residual order
> (Leftovers-vs-Leftovers, Sandstorm-chip-vs-Sandstorm-chip); **(2)** **Quick Claw is out of scope**
> — banned in gen3ou as of ~2026-06 (deps/teams not yet re-scraped; see §5 caveat) — which removes
> the only stochastic turn-order confounder and turns the speed update into clean elimination.
>
> **Recommended order:** System A (build now) → System B (next). System B carries its own
> **observation-evidence scalar** (the former "System C", folded in — §4.7) so the model can tell a
> *confirmed speed tie* from an *unobserved* opponent. Each system is retrain-class (ARCH bump +
> golden-fixture regen + the obs-build benchmark gate); total obs footprint is **+2 trailing scalars**
> (CB belief + speed-evidence) plus an in-place sharpening of the existing `p_outspeed`.

---

## 1. Motivation

The incoming-damage / OHKO belief block (`gen3_incoming_damage_v1`) prices *what the opponent can do
to us*, but two of its belief inputs are deliberately coarse:

- **Choice Band is invisible to the block.** It is never revealed by a protocol message, so the
  block's physical worst-case (`atk_tail`) captures spread investment but **not** the CB ×1.5. The
  deferred "CB worst-case channel" (`design_incoming_damage_obs.md` §8) needs a calibrated *belief*
  about whether the opponent is even holding CB before it can fire without over-pessimising every
  physical attacker.
- **`p_outspeed` ignores every speed fact the battle has already revealed.** It marginalises the
  *static* usage prior `priors.stat_distribution(species,'spe')` every decision and never folds in
  the turn-order evidence the game keeps handing us (who moved first, whose Leftovers ticked first).
  Speed is the gate on the revenge-KO / "do I switch or eat it" decisions the loss forensics flagged
  as a critic blind spot.

Both are the same shape as the **Hidden Power tracker**, which already narrows a hidden attribute
from in-battle observation and feeds a calibrated belief to the model. This doc specifies three
systems built on that template.

## 2. The unifying pattern (Hidden Power tracker, generalised)

`HiddenPowerTracker` (`src/agents/training/hidden_power_tracker.py`) is the prior art. Its shape,
which all three systems copy:

| Aspect | How HP does it | What the new trackers reuse |
|---|---|---|
| **Ownership / lifecycle** | One instance, owned by `EpisodeTracker` (`episode_tracker.py:102`), exposed via a read-only `@property`, `reset()` per battle (`:383`). | Same — a sibling tracker field, reset alongside. |
| **Update timing** | `EpisodeTracker.record()` runs the narrowing from protocol-truth events **strictly before** the encoder runs (`gen3_env.py:71` record → `:74` encode). | Same seam: the obs at turn N carries narrowing from turns 1..N. |
| **Threading** | Passed as a plain `hp_tracker=` kwarg through `state_encoder.encode` → the per-mon block **and** the matchup math. No globals. | Same — `cb_tracker=` / `speed_belief=` kwargs into `incoming_damage_encoder`. |
| **Prior** | Per-species usage prior via the `gen3_data` facade, flat fallback. | `priors.items()` for CB; `priors.stat_distribution(...,'spe')` for speed — both already exist. |
| **Narrowing** | Pure elimination over a discrete candidate set; **monotone** → idempotent → not included in snapshot/restore. | Same monotone-elimination property (CB: zero a probability; speed: drop bins) ⇒ **no rollback bookkeeping**. |
| **Obs representation** | Full distribution + a separate `revealed/known` flag (disambiguates *not-seen* / *ambiguous* / *ruled-out*). | A calibrated scalar/dist + a separate evidence/known flag — never argmax. |
| **Validation** | A bridge fuzz test with ground-truth opponent teams (invariant + ground-truth + protocol cross-check). | Same — ground-truth-item / ground-truth-speed teams. |

The seams are **not HP-specific**: `record()` already has the live read-model and the opponent's
last event in hand; the encoder already accepts a thread-through handle and writes per-opp-slot
blocks; the incoming-damage block already carries the precise belief *inputs* each system needs
(`priors.items()` exposes P(CB); `stat_distribution(spe)` feeds `p_outspeed`; turn order is already
folded into `TurnView.we_moved_first`). None of these are greenfield.

---

## 3. System A — Choice-Band elimination belief  **[build now · effort M · value high]**

### 3.1 What proves a mon is NOT Choice Band (Gen 3, Showdown sim)

- **(HARD) Item already revealed as anything else.** Items are mutually exclusive (one held item).
  Any reveal — Leftovers/Sitrus passive heal, Berry/Power-Herb consumption (`-enditem` →
  `consumed_item`), Trick/Knock-Off/Thief/Frisk — sets `mon.item` / `mon.consumed_item`, both already
  read by `ItemsEncoder`. Check **both** (the consumed-Berry case is special-cased in poke-env's heal
  handler, `abstract_battle.py:471-476`, but `consumed_item` is still set).
- **(HARD) Used ≥2 distinct moves in one uninterrupted stint.** In the Showdown sim, CB locks the
  holder into its first move via the `choicelock` volatile (`data/conditions.ts:324-348`) until it
  switches; selecting a second distinct move is impossible under any Choice item. **CB is the only
  Choice item in Gen 3** (Choice Scarf/Specs are `gen: 4`), so any Choice-lock behaviour
  unambiguously implies CB — no disambiguation needed.

**Non-proofs to avoid (verified gotchas):**
- A **lone status move is NOT an elimination.** CB locks into the *first* move, which can itself be a
  status move (`conditions.ts:332-347`). The predicate is "two *distinct* moves in one stint," any
  category.
- **No protocol tag exists** for the opponent's CB lock — Showdown's `[from] lockedmove` suffix is
  only for *our own* forced moves. The opponent's lock is enforced silently in their move request,
  which we never see, so the tell must be **reconstructed from cumulative moves**, not detected.
- The move-lock is a **Showdown-sim fact, not cartridge Gen-3** (real Gen-3 CB only boosts Atk; the
  lock is a later-gen mechanic Showdown applies to gen3). Valid only because we exclusively play the
  sim — **must be documented in the tracker** so a future "this is wrong for gen3" refactor doesn't
  delete it.

### 3.2 Observability — what's free vs new

- **Free:** the per-species P(CB) prior — `priors.items(species)['choiceband']` is already built
  (`data/pokemon/gen3_item_priors.json`, 199/216 species; e.g. Aerodactyl 0.76, Metagross 0.31) and
  exposed but consumed by nothing. Both HARD-via-item eliminations are free from existing
  `mon.item` / `mon.consumed_item` state.
- **New (small):** the "≥2 distinct moves in **one stint**" qualifier. `len(opp_active.moves)` is
  cumulative across the *whole battle* (conflates the soft cross-switch case), so a tiny
  **reset-on-switch distinct-move counter** is needed. The MOVE / SWITCH / DRAG primitives
  (`move_id` delegation-aware, `actor_species`) are all in the event log via `events_since(0)` — no
  battle-layer change. Exclude **delegated** moves (Sleep Talk / Metronome pick a different move but
  do not break the lock — gate on `from_move` / `called_via`).

### 3.3 Tracker design — `ChoiceBandTracker`

Mirrors `HiddenPowerTracker` exactly:

- **State:** `dict[opp_species → p_cb: float]`, seeded lazily from `priors.items(species).get('choiceband', 0.0)`;
  plus a `dict[opp_species → set[move_id]]` of distinct **selected** moves since that mon's last
  switch-in.
- **Narrowing** (in `EpisodeTracker.record()`, the slot where `_maybe_observe_hidden_power` runs):
  set `p_cb = 0.0` on any of — (a) `mon.item` revealed to a non-CB id, (b) `mon.consumed_item` set,
  (c) the distinct-move set reaching size 2. Reset the distinct-move set on the opponent's
  `SWITCH` / `DRAG`. Pure monotone-down elimination ⇒ idempotent ⇒ excluded from snapshot/restore
  (inherits the HP property).
- **Accessors:** `p_cb(species)` (prior if unobserved, 0.0 if eliminated), `is_known(species)`
  (eliminated **or** item revealed), `reset()`.

### 3.4 Obs representation — one scalar, provide-vs-learn

Append **one** per-opp-active scalar `p_cb_live ∈ [0,1]` to the incoming-damage block's trailing
region (after the recovery scalars): `INCOMING_RECOVERY_DIM 3 → 4`, `INCOMING_DMG_DIM 33 → 34`,
`PER_MON` unchanged. The block is already routed to both heads via `non_matchup_rest`.

**Emit the calibrated P(CB) and let the critic learn the ×1.5 — do not bake the multiplier.** This
is the [[feedback_provide_vs_learn]] choice: representation (B), a raw belief fact, over
representation (A), scaling `atk_tail ×1.5` inside `_attacker_threat`. (A) is the fused-mechanic
variant that depends on wiring the deferred worst-case physical sub-channel; (B) ships independently
and is strictly more in-house-style. **Do not** route this into the per-mon item block — that block
is *identity*, not *belief*. If (A) is ever added, the ×1.5 is **physical-channel only** (CB does not
touch SpA).

### 3.5 Validation

A bridge fuzz test (`choice_band_tracker_fuzz_test.py`) with fixed opponent teams of **known** items
(some CB, some not), forcing varied move usage. Assert: (1) **invariant** — `p_cb` is monotone
non-increasing and hits 0 exactly when an elimination condition is observed; (2) **ground truth** —
a true-CB mon's `p_cb` never wrongly reaches 0; a non-CB mon's reaches 0 once it reveals its item or
its second distinct move; (3) **protocol cross-check** — each elimination corroborated by an archived
`|-item|` / `|-enditem|` / second `|move|` line.

---

## 4. System B — Speed inference from turn-order  **[build next · effort L · value high]**

### 4.1 Signals (scoped per the requestor)

Exactly two, both speed-ordered in Gen 3:

1. **Equal-priority move order** — `TurnView.we_moved_first` (`turn_view.py:259`, already folded from
   the event-log `seq` order and already in the obs as `move_order=[we_first,opp_first]`). Speed
   evidence **only** when both used moves of equal priority.
2. **Same-effect residual order**, limited to **Leftovers-vs-Leftovers** and
   **Sandstorm-chip-vs-Sandstorm-chip**. Gen-3 end-of-turn residuals dispatch in speed order *within
   one effect-type bucket* (`battle.ts` `speedSort`/`comparePriority`, `handler.speed =
   pokemon.speed`). The value of the residual signal: it reveals speed on turns where **neither mon
   attacked** (double switch, both used status). Recovered from the `[from]` reason of the ordered
   `-heal`/`-damage` events (there is no `|upkeep|` boundary event — it's dropped as CONTROL — so the
   phase is inferred from the `[from]` vocabulary, mirroring `_classify_faint_cause`).

**Explicitly out of scope:** cross-effect residual order (Sandstorm always precedes Leftovers
regardless of speed — it's effect ordering, not speed), and all other residual types (Leech Seed /
poison / burn). Keeping it to the two clean, common buckets avoids over-reading.

**Dropped entirely:** **Intimidate / switch-in order.** Verified correction — Gen-3 simultaneous
switch-ins resolve in **player-slot order (P1 before P2), not speed** (the speed-sorted switch-in
path is gated `gen >= 5`; Smogon ADV consensus). It carries **zero** speed information.

### 4.2 Quick Claw is out of scope

Quick Claw was banned in gen3ou (~2026-06); we design as if it is absent. This removes the only
stochastic turn-order confounder. **Consequence:** a single "they moved first at equal priority" is
now *clean* evidence of speed order (no ~20% silent-jump leak), so the update is **monotone
elimination** (HP-style), not a soft reweight. Gen 3 also has **no Trick Room, Tailwind, or
Custap/Lagging-Tail** (all gen4+), so after equal-priority gating and folding observed
boosts/paralysis, **move priority is the only remaining confounder.** This makes speed inference
nearly as clean as HP elimination.

> **Caveat (write into the tracker docstring):** deps/Showdown still implement Quick Claw and our
> teams aren't re-scraped, so a QC holder *could* appear in training until that lands. Under pure
> elimination a QC fluke would wrongly drop a valid speed bin. Mitigation is cheap and optional —
> keep the elimination "soft" by retaining a small floor on the just-eliminated bins (a 1-line knob),
> or simply accept the rare mis-narrow until teams update. Default: design for QC-absent; leave the
> floor knob at 0.

### 4.3 The latent and the update

The hidden quantity is **which speed bin** the opponent's set occupies — a *discrete categorical*
(`stat_distribution(species,'spe')` returns `[(spe_value, weight)]`, one bin per nature/EV-realised
stat). So the posterior is a **reweighting of those bins**, and `p_outspeed` (`incoming_damage.py:105`)
runs **unchanged** on the narrowed dist. This is a discrete-likelihood filter, **not Beta-Binomial**
(that estimates one Bernoulli rate; we are localising a categorical value).

Per **informative** turn, compare *effective* speeds (fold the turn's observed boosts and paralysis —
gen3 paralysis ×0.25 — into both sides; our speed is exact):

- **we moved first** ⇒ `our_eff ≥ their_eff` ⇒ **eliminate bins with `their_eff > our_eff`.**
  (Bins where `their_eff == our_eff` survive — a true tie goes either way; bins `<` survive.)
- **they moved first** ⇒ `their_eff ≥ our_eff` ⇒ **eliminate bins with `their_eff < our_eff`.**

The **tie bin survives both directions** and is only squeezed out when a *different* turn's modifiers
break the tie and contradict it — so exact speed ties (randomly shuffled, `prng.shuffle`) are handled
correctly with no special casing. Same-effect residual order feeds the identical inequality
(residual dispatch uses `pokemon.speed`, i.e. post-boost/para effective speed). Monotone elimination
⇒ idempotent ⇒ excluded from snapshot/restore.

### 4.4 Confounder gates (an `is_feasible`-style pre-check)

Skip a turn (no update) when the order is not clean speed evidence:

- **Unequal move priority** — needs a **move-priority table** (lives in `gen3_data`/mechanics, not the
  battle layer). Gate move-order comparisons to equal-priority turns; treat Pursuit-on-a-switching
  target as its own case (skip).
- `we_moved_first is None` (one/both switched) — fall back to the residual signal if available.
- Residual: compare **only within the same effect bucket** (both Leftovers, or both taking sand);
  ignore cross-type order.

That's the whole gate list in Gen 3 with QC out — materially shorter than the multi-gen case.

### 4.5 Tracker design — `SpeedBelief`

- **State:** `dict[opp_species → mask: np.ndarray]` over that species' `spe_dist` bins (the static
  prior stays `lru_cached`; the per-battle **narrowing mask** is **not** cached — caching would leak
  belief across battles, exactly the HP split). `reset()` per episode.
- **`observe(species, we_first, ctx)`** with `ctx` = (our exact effective speed, both sides'
  boosts/para, the move priorities / residual-effect bucket) — applies the §4.3 elimination after the
  §4.4 feasibility gate. Raises on full elimination with a bug-vs-data-gap message (HP precedent).
- **Accessors:** `posterior_dist(species)` (the reweighted `[(spe,weight)]`), `n_observed(species)`
  (the count of informative order-observations folded in — the evidence scalar of §4.7), and an
  `is_observed(species)` flag.

### 4.6 Consumption

Thread `speed_belief=` into `incoming_damage_encoder._attacker_threat` along the identical path the
`hp_tracker` rides (`gen3_env` → `state_encoder.encode` → `reactive.encode` → `encode_block`), and
replace the static `spe_dist` handed to `p_outspeed` with the **posterior-reweighted** dist. The
marginalisation math is reused verbatim.

### 4.7 Obs representation — sharpen + one evidence scalar

Two parts, **+1 dim total**:

1. **Sharpen `p_outspeed` in place — +0 dims.** The per-our-mon 5-tuple is unchanged; only the
   *value* of the existing `p_outspeed` slot changes (it now reflects the narrowed belief).
2. **One per-opp-active `opp_speed_evidence` scalar — +1 dim** in the trailing region (beside the
   recovery / CB scalars): a saturating normalisation of `n_observed` (the count of informative
   order-observations folded into the current opponent's speed belief — e.g. `min(n, K)/K`).
   `INCOMING_DMG_DIM` 34 → 35.

**Why the evidence scalar earns its dim — the speed-tie case.** Under monotone elimination,
`p_outspeed ≈ 0.5` is ambiguous between *(a)* "no order ever observed — the prior just sits near
50/50" and *(b)* "narrowed hard to a genuine speed tie with us." Those call for opposite play (in
(b) the model *knows* it is a coin flip and must not bank on outspeeding). A point estimate can't
separate them; `opp_speed_evidence` read alongside `p_outspeed` does — `evidence ≈ 0` ⇒ case (a),
high `evidence` with `p_outspeed ≈ 0.5` ⇒ case (b). This is the speed analog of the HP block's
`revealed` flag beside its distribution. Both parts are retrain-class (re-meaning a block is
explicitly retrain-class in `model/CLAUDE.md`).

### 4.8 Why a raw count, not a baked confidence bound

The "Wilson lower bound / a 4.9 with many reviews beats a 5.0 with one" analogy (the original
"System C") is the right *intuition* — shrink toward the prior when evidence is thin — but the wrong
*mechanism*: because the latent is a **discrete categorical** speed bin, the posterior **already
encodes the uncertainty**, and a Wilson/Beta bound answers a different question ("what fraction of
the time does this beat a fixed reference"). So we do **not** fuse a confidence bound into one scalar
(that would double down on the very point-estimate anti-pattern `p_outspeed` already commits). We
emit the **raw observation count** — a clean fact ([[feedback_provide_vs_learn]]) — and let the net
learn the risk weighting itself. The count's one weakness (a turn's information depends on how close
the speeds were) is moot because the model also sees `p_outspeed`; given both it can learn the
combination.

### 4.9 Validation

A bridge fuzz test (`speed_belief_tracker_fuzz_test.py`) with fixed opponent teams of **known**
Speed spreads (`GROUND_TRUTH` dict). Assert: (1) **invariant** — after each observe, the true speed
bin is never eliminated and the surviving set is consistent with every observed order; (2) **ground
truth** at battle end — the posterior concentrates on the true bin (or a tie-class containing it);
(3) **protocol cross-check** — each update corroborated by the archived move-order / residual lines;
(4) **evidence monotonicity** — `opp_speed_evidence` is non-decreasing within a stint, reads high for
a hard-pinned speed, and ~0 for an unobserved opponent.

---

## 5. Mechanics grounding (verified vs the bundled gen3 sim)

| Fact | Verdict | Source |
|---|---|---|
| End-of-turn residuals dispatch in **speed order within one effect-type bucket** | ✅ confirmed | `sim/battle.ts:404` `comparePriority`, `:1003` `handler.speed = pokemon.speed`, `:2832` residual phase |
| Cross-effect residual order is fixed by effect (sand before Leftovers), **not** speed | ✅ confirmed | same comparator: `order` (effect-type) sorts before `speed` |
| Switch-in / **Intimidate order is player-slot (P1→P2), not speed** in Gen 3 → **drop it** | ✅ corrected | speed-sorted switch-in gated `gen >= 5` (`battle.ts:1024`); Smogon ADV consensus |
| **Quick Claw = 20% (1/5)** in Gen 3 (not 60/256 = Gen 2) and **silent** (no protocol line) | ✅ corrected | `battle.ts:1795` `gen===3 → randomChance(1,5)`; gen3 mod strips `-activate` (`mods/gen3/items.ts:301`, `scripts.ts:25`). **Out of scope (banned).** |
| Exact speed ties are **randomly shuffled** each turn → no stable order | ✅ confirmed | `battle.ts:455` `prng.shuffle` |
| **Choice Band is the only Choice item** in Gen 3 (Scarf/Specs are `gen: 4`) | ✅ confirmed | item dex `gen` fields |
| CB **locks into the first move** (sim) until switch; the first move **can be a status move** | ✅ confirmed | `data/conditions.ts:324-348` `choicelock` |
| One item per mon (mutual exclusion) | ✅ trivially | item system |
| **No Trick Room / Tailwind / Custap** in Gen 3 (all gen4+) → fewer speed confounders | ✅ confirmed | gen-gated; absent from gen3 mod |

> **Discipline that this doc inherits (and that the recent Sandstorm-SpD regression violated):**
> verify gen-specific mechanics against `deps/pokemon-showdown/data/mods/gen3/` + the `gen <= N`
> gates in `sim/`, **never** general/modern-gen knowledge. The mechanics research here found the
> wikis disagreed on Quick Claw (20% vs 23.4%) and the bundled source resolved it definitively.

> **Distribution-shift caveats:** (a) `priors.items()` is normalised sum→1 over *observed* items only,
> so P(CB) is conditioned on the species being seen — re-verify against the `data/teams` pool the
> agent actually faces (a self-play regime shifts this further). (b) The Quick-Claw ban means
> deps/teams will be re-scraped; until then a QC holder is out-of-distribution for System B (§4.2).

---

## 6. Observation layout & retrain impact

Both systems' additions live in the incoming-damage block's **trailing (opp-active) region** — the
natural home, since the block already carries the speed/offensive belief inputs and is routed to both
heads via `non_matchup_rest`. Minimal footprint:

| Change | `PER_MON` | trailing (`RECOVERY`) | `INCOMING_DMG_DIM` | obs total |
|---|---|---|---|---|
| today (`gen3_incoming_damage_v1`) | 5 | 3 | 33 | 3390 |
| + System A (`p_cb_live`) | 5 | 4 | 34 | 3391 |
| + System B (sharpen `p_outspeed` in place **+** `opp_speed_evidence`) | 5 | 5 | 35 | 3392 |

- **Retrain-class** each: bump `ARCH_SIGNATURE` (`model_version.py`, currently
  `gen3_incoming_damage_v1`), regenerate `golden_obs_fixture.json`, pass the **obs-build benchmark**
  (<10% calls/encode, `observation/CLAUDE.md`). `PER_MON` / `RECOVERY` stay the single source of
  truth in `incoming_damage.py`, imported by `constants.py`.
- **Per-decision cost is cheap and off the hot path:** the narrowing runs in `record()` (per
  decision, but not in the obs encode loop); the encoder read is a dict-get + a reuse of the existing
  `p_outspeed` marginalisation. Per-species static work (`priors.items`, `stat_distribution`) stays
  `lru_cached`; the per-battle narrowing masks are **not** cached. Expected benchmark impact: well
  inside budget (comparable to the existing belief loop).

---

## 7. Phasing & recommended order

1. **System A — CB elimination (build now).** Highest value-to-effort, no hard dependency: the prior
   is already built/exposed, two of three eliminations are free, and gen3's single-Choice-item fact
   removes all ambiguity. Only new state is the reset-on-switch distinct-move counter. One ARCH bump,
   +1 scalar.
2. **System B — speed inference (next).** High value (sharpens the OHKO/revenge-kill belief the loss
   analyses flag). Effort L: a `SpeedBelief` tracker + a move-priority table + the (now short)
   confounder gate. With QC out the update is clean elimination. Carries its own **+1
   `opp_speed_evidence` scalar** (the former System C — §4.7/§4.8) so the net can read a confirmed
   speed tie vs an unobserved opponent. Sharpen + scalar = +1 dim, one ARCH bump. Sequence after A so
   the simpler belief lands first.

---

## 8. Risks & open questions

- **Sim-vs-cartridge CB lock (A).** The 2-distinct-moves tell is a Showdown-sim fact; document it in
  the tracker to prevent a "this is wrong for gen3" deletion.
- **Move-priority table (B)** is load-bearing — a wrong priority misclassifies a turn as
  equal-priority and corrupts the belief for the rest of the battle. Must be exhaustive for gen3 OU
  movepools and validated by the fuzz test.
- **Effective-speed comparison (B)** must fold the turn's observed boosts/paralysis on **both** sides
  before comparing to candidate bins; an unaccounted modifier is a permanent mis-narrow (the
  monotone-elimination property means errors don't self-heal — hence the `is_feasible` guard before
  every update).
- **QC re-scrape (B).** Until teams/deps update, keep the optional soft-floor knob (default 0) so a
  stray QC holder degrades gracefully rather than corrupting a bin.
- **Open — does the policy attend to it?** As with the incoming-damage feature, these sharpen the
  *critic's* inputs; whether they move the *policy* (the under-switching gap) is only measurable by a
  retrain. Reuse the incoming-damage gates: a calibration fuzz (belief vs ground truth) + a
  saliency/`CRITIC_BLINDSPOT` check on the trained model.
- **Open — A's representation (B) vs (A).** Ship the calibrated `p_cb` scalar first; only wire the
  fused ×1.5 worst-case physical channel if a post-retrain falsifier shows the critic isn't learning
  the relationship itself.

---

## 9. References

- **Prior art:** `src/agents/training/hidden_power_tracker.py` (`:46` class, `:81` observe/narrow,
  `:150` `is_feasible`, `:163-192` accessors, `:194` reset); `episode_tracker.py:102,114,156-174,383`
  (ownership / record-before-encode / reset; snapshot deliberately excludes the tracker);
  `gen3_env.py:52,71,74,139`; `hidden_power_tracker_fuzz_test.py` (the bridge-fuzz template).
- **Event log:** `battle/battle_event.py` (EventKind, `seq`, `actor_species`, `raw`, `from_cause`);
  `battle/turn_view.py:245` `_compute_move_order` / `:259` `we_moved_first`, `:39` `FAINT_CAUSE_VOCAB`
  (residual `[from]` mapping pattern); `battle/gen3_battle.py:314` `_parse_from`, `:476` DAMAGE/HEAL
  builder (residual cause in `reason`); `battle/strict_view.py:80` `events_since`.
- **Belief inputs / consumption:** `gen3_data/priors.py:42` `items()`, `:67` `stat_distribution`;
  `observation/incoming_damage.py:105` `p_outspeed`, `:30` `PER_MON`/`RECOVERY`;
  `observation/incoming_damage_encoder.py:115` `_attacker_threat`; `observation/constants.py:98`
  (incoming block layout); `observation/items.py:27` `ItemsEncoder`.
- **Mechanics (bundled sim):** `deps/pokemon-showdown/sim/battle.ts:404,455,1003,1024,1795,2832`;
  `data/mods/gen3/items.ts:301`, `scripts.ts:25`, `conditions.ts:52` (Sandstorm `onModifySpD:
  undefined`), `data/conditions.ts:324` (`choicelock`).
- **Parent design / deferred items:** `designs/ai_v5/design_incoming_damage_obs.md` §8;
  recovery-scalar precedent folded from `designs/ai_v5/design_stall_recovery_obs.md`.
- **Philosophy:** [[feedback_provide_vs_learn]] (raw facts + flag, let the model learn the
  relationship); [[feedback_prober_self_improvement]] (validate beliefs via the prober).
