# Implementation: Step 4 — Incoming-Damage / OHKO Belief Observation

A per-our-mon **calibrated belief about being KO'd** by the opponent active — the obs signal the
critic was tail-blind to (it priced "2× effective", never "this OHKOs"). Targets the
critic-tail-blindness pathology the `run_20260601_193826` loss forensics pinned.

> **Status: BUILT & shipped** (commit `34b4724`). This is an as-built record. The forward design
> (the reasoning, the go/no-go, the rejected alternatives) is `design_incoming_damage_obs.md`; **this
> doc records what actually landed, where it deviated from that plan, and what's deferred.** The
> post-retrain efficacy gate (Gate 2) is still pending a training run on the new arch.

---

## What shipped (one paragraph)

A **33-dim incoming-damage belief block** at reactive offset **50** (before the matchups → routed to
**both** policy and value heads via `non_matchup_rest`, the same lane the trapping bits and the
move-effect block use). Per our 6 team mons, slot-aligned: `[phys_expdmg_frac, spec_expdmg_frac,
phys_pko, spec_pko, p_outspeed]` (5 × 6 = 30), then 3 opponent-active recovery scalars
`[recovery_rate, cures_status, recovery_known]`. Obs dim **3357 → 3390**; `REACTIVE_DIM` **338 → 371**
(matchups shifted to reactive-offset 83); `ARCH_SIGNATURE` `gen3_move_effects_v1` →
**`gen3_incoming_damage_v1`**. Retrain-class (old checkpoints fail loudly).

---

## The reframe (why this is a belief, not a calc)

Exact damage is impossible under hidden info (move / spread / item unknown) and isn't the goal —
**calibration** is. The human read is "probably OHKOs unless it's the defensive set." So per defender
we compute a `P(KO)` integrated over the hidden-set prior (revealed moves are certainties; the rest
is Smogon usage), and let the model learn the soft gate (how to weight phys/spec given the board).
The closed-form roll math does the part the net *can't* learn (the KO threshold nonlinearity); the
priors supply what isn't derivable in-battle.

---

## Two modules — a deep split (math core vs battle glue)

The feature is split so the hard, testable math is poke-env-free and the battle reads live behind one
narrow door:

| File | Role | Touches poke-env? |
|---|---|---|
| `agents/observation/incoming_damage.py` | **Pure math core** — the gen3 damage formula, the 16-roll → `P(KO)` closed form, `p_outspeed` over a Speed distribution, and the `Candidate`/`Defender`/`AttackerThreat` belief dataclasses + `compute_team_block`. No poke-env, no torch → unit-tests without a battle. | no |
| `agents/observation/incoming_damage_encoder.py` | **Battle → belief glue**, single public entry `encode_block(battle, our_team, live)`. Owns the *only* poke-env / `gen3_data` reads (revealed ∪ prior candidate moves, offensive-stat distributions, screens/weather via the `LiveView` read-model). Everything else is `_`-prefixed. | yes |
| `agents/observation/reactive.py` | Just calls `encode_incoming_block(battle, our_team, live)` and writes the slice. The extraction helpers no longer live here. | — |

`constants.py` imports `PER_MON` / `RECOVERY` **from `incoming_damage.py`** (single source of truth,
the same pattern `VOLATILE_DIM` uses) so the obs layout and the math core can't disagree.

---

## The three belief paths

### 1. Speed → `P(outspeed)` (a probability, not a bit)

Our Speed is exact; the opponent's is **hidden** (nature + EVs vary per set), so we emit
`P(outspeed) = P(their_spe < our_spe) + ½·P(tie)` ∈ [0,1], marginalised over the opp active's Speed
**distribution** from the spread priors (`priors.stat_distribution(species, "spe")` → each usage
spread becomes a concrete L100/IV31 Spe value, usage-weighted). Observed boosts and paralysis
(gen3 par = `_PARA_SPEED` = ×0.25) fold in on **both** sides; a Speed tie → 0.5 (the gen3 coin flip);
an unknown opponent (empty distribution) → 0.5 (max uncertainty). The probability *is* the
uncertainty — `≈0.5` means set-dependent, `≈0.95` near-certain. (`incoming_damage.py:p_outspeed`.)

### 2. Attack magnitude → tail `P(KO)` + mean expected-chip

The same offensive distribution is collapsed to **two summary numbers** (so multimodal sets are
handled without a second damage calc on the hot path): **tail** = the 85th percentile
(`_OFFENSIVE_TAIL_Q = 0.85`, the worst-case investment) and **mean** = the usage-weighted mean
(`_offensive_stat` in the encoder; no-prior fallback = `priors.gen3_stat(base, 252, 1.0)`). Routed
by gen3 **TYPE-category** (physical types → Atk vs Def, special → SpA vs SpD):
- **`P(KO)`** is computed at the **tail** stat: `gen3_damage_max(...)` → `p_ko(dmax, hp_remaining)`,
  which counts the 16 rolls (R∈85..100) that reach the KO threshold — a closed form, no sampling.
- **expected-chip** uses the **mean**: `exp = dmax · (mean/tail) · _MEAN_ROLL(0.925) / hp_max`
  (damage is ~linear in Atk, so scaling the tail damage by mean/tail approximates the mean-stat
  hit), clamped to 1.5.
- Each channel takes the **max over candidate moves** of `p_in_set · (pko / exp)` — candidates being
  **revealed (P=1) ∪ top prior** moves (`_candidates` / `_prior_candidates`; the
  `_MAX_CANDIDATES_PER_CHANNEL = 4` cap applies to the **prior** moves per channel, with revealed
  damaging moves appended on top, and a `_PRIOR_MOVE_MIN_P = 0.12` usage floor). So the belief is
  "the scariest plausible move in their set," not a coverage sum.

### 3. Fixed-damage branch (the unpriced stall KOs)

Seismic Toss / Night Shade (=100), Dragon Rage (=40), Sonic Boom (=20) read `basePower=0` in the dex
(bucketed STATUS) and would otherwise price as **zero threat** — exactly the unpriced KOs on stall
mons. `FIXED_DAMAGE` tags them with constant damage; `_channel_threat` resolves them ignoring
Atk/Def/roll but **respecting type immunity** (Seismic Toss = 0 vs a Ghost). HP-relative (Super Fang/
Endeavor), reflective (Counter/Mirror Coat) and unreliable-OHKO moves (Sheer Cold/…) are **deferred**
— they need live HP / our-damage context the obs hot path doesn't carry.

---

## Mechanic corrections (the damage formula is gen3-faithful for KO-relevant hits)

`gen3_damage_max` and `_channel_threat` apply, all knowable from *our* board (correctness, not
uncertainty):

- **Explosion / Self-Destruct halve the target's Def** (`_HALVE_DEF_MOVES = {explosion,
  selfdestruct}` in the encoder → `Candidate.halves_defense` → `cdef = max(1, defense // 2)`). Found
  by the correctness review — without it Metagross Explosion vs a wall under-read the KO.
- **No Sandstorm SpD boost** — the ×1.5 SpD for Rock-types is a **gen4+ mechanic** that gen3 does
  *not* have (the Showdown gen3 mod nulls `onModifySpD` in `data/mods/gen3/conditions.ts`). The
  correctness review flagged it as a "missing" modifier using gen4 mechanics, it was briefly added,
  then **removed** once checked against the gen3 sim. gen3 sandstorm's only combat effect is residual
  chip (folded into HP elsewhere), so it does not touch the per-hit damage belief.
- **Reflect ×0.5 phys**, **Light Screen ×0.5 spec** (read from `live.ours.side_conditions`), **Burn
  ×0.5 phys**, **rain/sun BP** on Water/Fire (`weather_damage_mult`), **STAB ×1.5**, type
  effectiveness (incl. our defender's real ability — Levitate/Wonder Guard/etc. — via
  `effective_multiplier_by_types`).
- **Our Substitute up → `P(KO)=0`** for that mon (the hit eats the Sub); a fainted/absent slot →
  all-zero 5-tuple.
- **`Defender.status` is the raw `Status` enum** (not a lowercased string) so it feeds the
  effectiveness primitive (which keys Flash Fire off `Status.FRZ`) correctly, and paralysis folds
  into `p_outspeed` via `Status.PAR`. (The string→enum bug was a review find.)

### Recovery scalars (the Suicune-Rest discriminator)

`recovery_rate` (a clamped sum of per-move usage priors over the canonical
`gen3_mechanics.RECOVERY_MOVES` — revealed→1 — i.e. an expected-recovery-coverage proxy in [0,1]),
`cures_status` (P it runs **Rest** specifically — Rest
cures the Toxic clock *and* full-heals → an unbreakable wall for a chip team), and `recovery_known`
(prior-vs-confirmed bit). Folds in the `cures_status` ask from `design_stall_recovery_obs.md`.

---

## Data — the new priors

`tools/smogon_stats_downloader/compute_priors.py` was extended to emit three new files from the
existing aggregated Smogon stats (`gen3_smogon_stats.json` → `data/pokemon/`):

| File | Contents | Consumed by |
|---|---|---|
| `gen3_move_priors.json` | `{species → {move_id: P(move in set)}}` (`Moves/RawCount`, **not** sum-1) | the candidate pools |
| `gen3_spread_priors.json` | `{species → [[nature, [hp,atk,def,spa,spd,spe], weight], …]}` (top-25 raw) | `stat_distribution` (atk/spa/spe magnitude + outspeed) |
| `gen3_item_priors.json` | `{species → {item_id: P(item)}}` (sum-1, **includes `choiceband`**) | **nothing yet** — staged for the v2 CB worst-case channel |

Reached via the `gen3_data.priors` facade: new accessors `moves()` / `items()` / `spreads()`, the
`stat_distribution(species, stat)` distribution (lru-cached), and `gen3_stat(base, ev, mult)` — the
**single source of truth** for the L100/IV31 stat formula (`2·base + 31 + ev//4 + 5`, integer nature
mult), which the no-prior fallback reuses.

---

## Where it deviated from the design (`design_incoming_damage_obs.md` §5)

The shipped feature is **scoped tighter** than the elaborate §5 plan — deliberately, to keep the math
core narrow and poke-env-free:

| Design §5 item | As-built |
|---|---|
| Reuse `opponents.py:_estimate_damage_fraction` (§5.2) | **Wrote a fresh `gen3_damage_max`** in the pure math core instead. The heuristic-bot helper is poke-env-coupled and not unit-testable in isolation; a clean closed form was the better module boundary. |
| Log-space inductive bias — `log(A/Def)`, `log HP`, `log BP`, STAB bit, immune-bit gating (§5.3) | **Not built.** The closed-form `p_ko` already does the threshold nonlinearity, so the block ships the raw expected-fraction + `P(KO)` pair and lets the net relate them. Log features deferred. |
| Turn-shifted (Future Sight scoping) + end-of-turn chip folded into switch-target `HP_remaining` (§5.2c) | **Not built.** Prices the immediate per-turn channel only; bench-survival residual deferred. |
| Speed bit — listed as Phase 2 (§8) | **Built in v1** (`P(outspeed)`), per the §5.3b owner decision. |
| CB worst-case channel (§8) | **Data built** (`gen3_item_priors.json` + `priors.items()`), **consumption deferred to v2.** v1's tail captures spread investment but **not** the CB item ×1.5 — so CB attackers are under-priced ~1.5× on the physical channel. |
| Per-mon-token placement (§5.4 / Phase 2) | **Not done** — rides `non_matchup_rest` to both heads (the v1 plan). |

---

## Gates (all green at ship)

| Gate | Result |
|---|---|
| Full unit suite (`not integration and not e2e`) | **1834 passed** (1820 pre-rebase; +14 from rebased-in ELO tests), 2 skipped |
| Math unit tests (`incoming_damage_test.py`) | pass — incl. Explosion-Def-halve, the gen3-no-sandstorm-SpD-boost guard, and status-enum-paralysis cases |
| Golden-obs parity (byte-exact, `gen3_data_obs_parity_integration_test.py`) | fixture regenerated (3390-dim, 991 decisions) → **passes** |
| obs-build benchmark | **6,853 calls/encode** — +7.7% vs the pre-feature 6.36k (the feature's justified cost), +0.04% vs the post-feature baseline; well under the 10% hard gate. Held via per-species `lru_cache` on candidates + stat distributions. |
| Model roundtrip + `--debug` smoke (bridge) | `[ModelVersion] Round-trip smoke test PASSED`; episodes complete, eval ran, no crash |
| **Incoming-damage invariants fuzz** (`poke_env_gaps/incoming_damage_fuzz_test.py`, NEW) | 2722 decisions over 40 bridge battles / 36 distinct opp species, **0 invariant violations**, 100% non-zero blocks |

**Not built (deferred):**
- **Gate 1 — model-free calibration fuzz** (the bridge counterfactual oracle: `P(KO available)` vs
  the belief, reliability curve sliced by knownness). The shipped fuzz validates *invariants* (no
  raise / finiteness / ranges / Sub-and-fainted zeroing / belief fires), not *calibration*. The
  oracle version is the open Gate-1 work.
- **Gate 2 — post-retrain efficacy** (saliency rises ≥10–50× from ~0.002 **and** the
  CRITIC_BLINDSPOT turning-point count drops on a fresh forensics pass). Pending a training run on
  the new arch — this is the *sufficient* condition the design flagged as unresolvable without a
  retrain (the binding risk is policy-side under-switching).

---

## Module map

| File | Change |
|---|---|
| `agents/observation/incoming_damage.py` | **NEW** — pure belief math core |
| `agents/observation/incoming_damage_encoder.py` | **NEW** — battle→belief glue, single `encode_block` |
| `agents/observation/incoming_damage_test.py` | **NEW** — math unit tests |
| `agents/training/poke_env_gaps/incoming_damage_fuzz_test.py` | **NEW** — bridge invariants fuzz |
| `agents/observation/reactive.py` | calls `encode_incoming_block`; extraction helpers removed |
| `agents/observation/constants.py` | `INCOMING_*` dims (PER_MON/RECOVERY imported from the math core), offsets shifted |
| `agents/gen3_data/priors.py` | `moves()`/`items()`/`spreads()` + `stat_distribution` + public `gen3_stat` |
| `tools/smogon_stats_downloader/compute_priors.py` | emits `gen3_{move,spread,item}_priors.json` |
| `data/pokemon/gen3_{move,spread,item}_priors.json` | **NEW** prior data |
| `agents/model/model_version.py` | `ARCH_SIGNATURE = gen3_incoming_damage_v1` |
| `src/main/prober/engine_test.py` | offset pins updated (matchups now at om/tm 1501/1645) |
| docs | root + `observation/` + `model/` + `prober/` `CLAUDE.md` |

---

## Future (v2, gated on a proven Phase-1 residual)

- **CB worst-case channel** — consume `gen3_item_priors.json`: a CB-on physical channel weighted by
  `P(CB | species, unrevealed)`, using the "≥2 distinct moves in one stint = not Choice-locked" tell.
- **HP-relative / reflective / OHKO candidates** — Super Fang, Counter/Mirror Coat (need our-damage
  context), Sheer Cold/Fissure (≈0.30 guaranteed-KO).
- **Log-space stats + immune-bit gating** (§5.3) if the raw pair under-attends post-retrain.
- **Residual-into-HP for switch targets** (Spikes/Sandstorm/Toxic chip) — the Claydol bench-survival
  case (§5.2c).
- **The model-free calibration oracle** (Gate 1) and **per-mon-token placement** (Phase 2).
