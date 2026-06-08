# Implementation: Step 5 — Markovian / PBRS reward redesign + the no-progress clock

Make **every** reward term either **PBRS** (objective-neutral, telescoping) or **Markovian-w.r.t.-the-
observation** (a clean, obs-keyed bias), so the critic faces no irreducible per-term variance and the
material spine stops biasing the optimum toward dominant-over-clutch wins. Targets the
clutch-vs-dominant return skew (clutch win +26 vs dominant +47, a faint costing −5.82 even in *won*
games) and gives the anti-spam family a single obs-keyed counter the model can state-condition on.

> **Status: BUILT** (`ARCH_SIGNATURE = gen3_markovian_progress_v1`, obs 3390 → 3391), **not yet
> trained.** This is an as-built record. The forward design (the two axes, the §9 adversarial-review
> ledger, the rejected alternatives) is `design_markovian_reward_and_features.md`; **this doc records
> what landed, the staging that keeps the default run a single-variable change, where it deviated
> from the plan, and the post-implementation code-review refactor pass.** The post-retrain efficacy
> gate (clutch-conversion ↑ / useless-turn-rate ↓ / ELO non-regressed) is pending a training run.

---

## What shipped (one paragraph)

A **reward registry** (`RewardBreakdown._REGISTRY` mapping each field → `RewardClass` ∈
{TERMINAL, PBRS, BIAS}) that drives the fold one-treatment-per-class; an always-on **material PBRS
`Φ_mat`** that replaces the unconditional `hp_ours/hp_opp/faint_ours/faint_opp` base spine (so every
win returns +30 / loss −30); a **`--bias-additivity`** accumulate-refund knob (λ ∈ [0,1], default 1.0
= byte-identical to today's additive biases); and the **no-progress clock** — a `turns_since_progress`
reactive obs scalar at `vec[14]` plus a `no_progress_tax` reward term, both keyed on **one**
`EpisodeTracker`-owned `ProgressClock` (the `HiddenPowerTracker` precedent). Obs dim **3390 → 3391**;
`REACTIVE_SCALAR_DIM` **14 → 15** (`REACTIVE_DIM` 371 → 372, matchups shift +1); `ARCH_SIGNATURE`
`gen3_incoming_damage_v2` → **`gen3_markovian_progress_v1`**; `MODEL_CONFIG_VERSION` **3 → 4** (three
new resume-immutable reward hparams). The shipped belief PBRS field `pbrs_material` is **renamed
`pbrs_belief`** (it was the incoming-KO belief PBRS, never material). Retrain-class (old checkpoints
fail loudly).

---

## The single-variable staging (the load-bearing rollout decision)

The full bias redesign is implemented but **gated behind `RewardConfig.bias_redesign`
(`--bias-redesign`, default OFF)** so the *first* run is a clean attribution experiment:

| | `bias_redesign = OFF` (default) | `bias_redesign = ON` |
|---|---|---|
| Material | `Φ_mat` PBRS (always-on, the clutch-fix) | same |
| Anti-spam family (`repetition`/`bouncing`/`dead_matchup`/`struggle` taxes) | **active, as today** | **suppressed** — the clock subsumes them |
| `no_progress_tax` | **0** (clock tracks the obs scalar only) | charged (`−no_progress_penalty` per gated no-op) |
| `roar`/`status`/`se_switch`/`switch_base` reframes | as today | (deferred — see Deviations) |

So at the default the **only** reward-behavior change vs the live baseline is `Φ_mat`. Crucially the
`turns_since_progress` obs scalar is present in **both** arms (the clock always tracks it), so the
material-only run and the redesign run **share one architecture** and can A/B by resume.

---

## Material PBRS `Φ_mat` (`reward_manager.py:_compute_phi_mat`)

```
Φ_mat(s) = MAT_HP_WEIGHT·(Σ our_hp_frac − Σ opp_hp_frac) + MAT_ALIVE_WEIGHT·(n_alive_ours − n_alive_opp)
           over the DECLARED team size (unrevealed opp mons = full-HP-alive)
MAT_HP_WEIGHT = 2.0   (= old HP_VALUE; reproduces the per-turn hp density exactly)
MAT_ALIVE_WEIGHT = 1.25   (= old FAINT_BASE 0.5 + FAINT_MATERIAL_PENALTY 0.75; the stated invariant)
```

`bd.pbrs_material = γ·Φ_mat(s′) − Φ_mat(s)`, `Φ_mat(terminal)=0` → telescopes to the policy-invariant
constant `−Φ_mat(s_0)`. **Declared-team summation** is the key correctness choice: unrevealed opponent
mons count as full-HP-alive, so `Φ_mat(s_0) ≈ 0` (6−6 HP, 6−6 alive), there are **no opp-reveal
discontinuities** (the opp sum only ever decreases from a 6.0 baseline), and the per-episode constant
has near-zero cross-episode variance (it would otherwise land in the unnormalized value-loss target —
the `project_popart` pathology). The old asymmetric `−0.75 FAINT_MATERIAL_PENALTY` is **removed** (a
faint's discrete cost is the alive-term drop, a state potential, not a standalone bias). The `+2.0`
explosion literal is **deleted** (the survive-Explosion credit rides `Φ_mat`); `explosion_block` stays.

---

## Bias-additivity accumulate-refund (`reward_manager.py:_fold_bias_refund`)

`--bias-additivity` λ ∈ [0,1] (`RewardConfig.bias_additivity`, default **1.0**) dials the BIAS class
from fully additive to fully telescoping, **per run, not annealed**:

```
bd.bias_refund = −(1−λ)·(γ·acc′ − acc)      # acc = running Σ of BIAS-class fields; acc′ = acc + this window's bias_sum
episode BIAS contribution = Σ bias + Σ refund ≈ λ·acc
```

Implemented as **accumulate-and-refund** (not per-turn scaling) so **λ=1 is a structural byte-no-op**
(the refund is identically 0 → the BIAS class is byte-identical to today) and each per-turn bias value
is recorded un-scaled for telemetry. The accumulator-potential spread (`γ·acc′−acc`) keeps the refund
low-variance instead of a terminal lump. PBRS/TERMINAL ignore the flag.

---

## The no-progress clock (`progress_clock.py`, NEW)

`ProgressClock` is an episode-scoped `turns_since_progress` counter **owned by `EpisodeTracker`** (NOT
`LiveView` — it is cross-turn state) and read by **both** the obs encoder (`value()` → the `vec[14]`
log-saturated scalar) and the reward (`last_penalty` → `no_progress_tax`), so obs and reward key on
**one** value. Updated at `record()`/`embed_battle` time (poke-env runs `embed_battle` *before*
`calc_reward`, so updating there keeps the obs fresh). Ternary per decision window:

| outcome | trigger | clock |
|---|---|---|
| **PROGRESS** | our-attributed damage ≥ `PROGRESS_DMG_EPS` (3%, via `our_damaging_event`/`opp_target_hp_delta` — **not** net opp-HP, so passive Sandstorm/Leech can't reset it) / status landed / hazard layer added / forced opp commit | `n = 0` |
| **DENIED** | miss / Protect-block / cant (para/sleep/freeze/flinch) / a productive heal (HP up via an `is_heal` move — `Φ_mat` prices it) | freeze `n` (no charge) |
| **NO_OP** | a deliberate wheel-spin (immune attack, capped setup, …) | `n = min(n+1, 10)` + charge `−no_progress_penalty`, **unless** no switch is legal (trapped-vs-wall) |

Gated off on forced-switch windows. Encoded as `log(1+min(n,10))/log(11)` (the global-turn-clock form).
`gen3_env.embed_battle` + `inference/player.embed_battle` both call the shared
`EpisodeTracker.update_progress_clock(battle, legal)` helper (folds the window delta once, advances the
clock, returns the delta for the env's `calc_reward` reuse) — so eval / self-play opponents / `play.py`
see the same scalar the trainer did.

**Constants** (`progress_clock.py`): `PROGRESS_DMG_EPS = 0.03`, `PROGRESS_CLOCK_CAP = 10`. The penalty
magnitude lives on `RewardConfig.no_progress_penalty` (0.15), set once onto the clock by the env (single
source of truth — no scattered literal). `stall_tax` was **re-tuned gentle** (it covers the *defensive*
stalls the offense-centric clock can't see): `START_TURN` 60→100, `PER_TURN` 0.05→0.02, `RAMP` 20→40,
`MAX` 0.5→0.15 (cumulative ~−10 to the turn-250 forfeit, was ~−21).

---

## The registry + the fold (`reward_manager.py`)

`RewardBreakdown._REGISTRY` is the single source of truth (field → `RewardClass`); a coverage test
pins it 1:1 against the dataclass fields (only `bias_refund`, the fold mechanism, is excluded) and a
second test pins `_GROUPS` (the to_dict display buckets) against it. The fold drives **BIAS
generically** (`_fold_bias_refund` sums `registry_fields(BIAS)`); TERMINAL + the two PBRS terms are
**explicit named folds** (`_fold_material_pbrs` / `_fold_belief_pbrs`) because each PBRS term carries
its own `_prev_phi_*` telescoping state a generic loop can't hold. `process_turn_reward` reads as a
phase sequence over these helpers; the telescoping math (`γ·Φ′−Φ` + the `Φ(terminal)=0` zeroing — the
§2.3 dominant-win footgun) lives in **one** `_pbrs_step` helper.

---

## Where it deviated from the design

- **Staging via `bias_redesign`** (above) — the design described the staging conceptually (§1.3); the
  implementation made it a single resume-immutable flag and made the obs scalar arch-shared so the
  arms are resume-compatible. This is the cleanest realization of the "single-variable first run".
- **Deferred reframes (no default-run effect).** The Markovian-purity reframes that remove hidden
  `self._prev_*` state — `roar` (§3 #9, drop `_prev_opp_boosts`), `status` (#29, the transition-event
  reframe), `se_switch` (#25, drop the `_last_opp_seen_by` gate), `switch_base` (#18, drop the
  `last_switch_turn` spam-gate) — are **not yet built**. They don't change the default run (today's
  terms still fire) and are clean follow-ons; the `Φ_hazard`/`Φ_status` telescoping forms (§2.6–2.7)
  likewise activate only at `bias_additivity → 0`, which is a later A/B arm.
- **`MODEL_CONFIG_VERSION` not `ARCH` for the reward hparams.** `bias_additivity` / `mat_alive_weight`
  / `bias_redesign` are value-meaning (resume-immutable) but NOT weight-shape — recorded on
  `ModelVersion` and enforced via `check_reward_config` on the training-resume path only (the
  `vf_coef` pattern), excluded from `check_compatible`. Only the obs scalar bumped `ARCH_SIGNATURE`.

---

## Code-review refactor pass (post-implementation)

A 3-reviewer pass (DRY / deep-interfaces / complexity) over the diff found **zero correctness bugs**
and a thin layer of cheap debt, all cleaned up:

| Fix | Effect |
|---|---|
| Deleted dead/duplicate constants (`NO_PROGRESS_PENALTY`, a stray `PROGRESS_CLOCK_CAP`, a doubled `STRUGGLE_*` pair) in `reward_manager.py` | removed silent-drift footguns |
| Moved `no_progress_penalty` onto `ProgressClock` (attr, set from config); dropped it from `update()` | killed the obs-path/reward-param leak **and** 2 of 3 scattered `0.15` literals |
| Extracted `_pbrs_step` + `_fold_material_pbrs`/`_fold_belief_pbrs`/`_apply_progress_clock`/`_fold_bias_refund`/`_log_turn` | `process_turn_reward` 174 → ~120 lines; the telescoping math lives in ONE place |
| `EpisodeTracker.update_progress_clock()` helper | DRYs the identical clock-wiring copy-pasted in `gen3_env` + `inference/player` |
| Pinned `_GROUPS` with a test; narrowed `except Exception`→`(AttributeError, TypeError)` in the clock | closes the third-parallel-list drift gap; a real delta-math bug can't hide as "not healed" |

Explicitly **declined** as premature: forcing PBRS/TERMINAL through a generic dispatcher; removing the
vestigial `explosion` field (a test guards it == 0); an `EncodeContext` for the 4 `encode()` kwargs; a
generic immutable-hparam registry. Two deliberate couplings (`_is_progress` vs `_last_attack_had_effect`;
`_prev_spikes` vs `_prev_opp_spikes`) kept with cross-reference comments.

---

## Gates (all green at ship)

| Gate | Result |
|---|---|
| Full unit suite (`not integration and not e2e`) | **1920 passed**, 2 skipped (1919 pre-change + 1 new `_GROUPS`-pin test) |
| New reward-redesign tests (`reward_redesign_test.py`, NEW) | registry coverage, `Φ_mat` telescoping + terminal-zeroing (the 6-0-win-must-not-bonus guard), bias λ=1 byte-no-op + parameterized blend (λ∈{0,0.5,1}), the full ProgressClock predicate (miss/heal/prevented freeze, our-attributed dmg, forced-switch + trapped gates) — all pass |
| Golden-obs parity (byte-exact) | fixture regenerated (**3391**-dim, 991 decisions) → passes (deterministic) |
| obs-build benchmark | **~7,173 calls/encode** — *below* the ~7.3k `gen3_incoming_damage_v2` reference (the clock scalar is one `math.log` + one array write); well under the 10% gate |
| Model roundtrip + `--debug` smoke (bridge, GPU + CPU `--bias-redesign`) | `[ModelVersion] Round-trip smoke test PASSED`; episodes complete; no NaN; `model_config.json` records `arch=gen3_markovian_progress_v1` / `total_dim=3391` / `config_version=4` / the reward hparams |
| Inference player tests | pass (the clock-wiring helper) |

**Not built (deferred):** the efficacy gate (post-retrain: clutch-conversion ↑, useless-turn-rate ↓,
ELO non-regressed — pre-registered in design §7.4); the offline reward-replay falsifier over
`eval_traces` (design §7.1a); the deferred reframes (above). The full `learn()`→eval "Training complete"
literal wasn't captured (GPU saturated by the live run; CPU+bridge too slow for the smoke timeout) — but
round-trip + training-entry are confirmed and the PPO update math is unchanged (just +1 auto-discovered
obs dim).

---

## Module map

| File | Change |
|---|---|
| `agents/training/progress_clock.py` | **NEW** — the `ProgressClock` (ternary predicate, `value()`, `last_penalty`) |
| `agents/training/reward_redesign_test.py` | **NEW** — registry / `Φ_mat` / bias-additivity / ProgressClock tests |
| `agents/training/reward_manager.py` | registry + `RewardClass`/`RewardConfig`; `Φ_mat`; bias-refund; `pbrs_material`→`pbrs_belief` rename; the `_fold_*`/`_pbrs_step`/`_log_turn` helpers; `Φ_mat`/clock constants; gentle `stall_tax` |
| `agents/training/episode_tracker.py` | owns the `ProgressClock`; `update_progress_clock()` helper |
| `agents/training/gen3_env.py` | clock wiring (penalty from config) + the `_pending_delta` fold-reuse cache |
| `agents/inference/player.py` | clock wiring in `embed_battle` (eval / self-play / play parity) |
| `agents/observation/constants.py` | `REACTIVE_SCALAR_DIM` 14→15 (offsets cascade); stale-comment cleanup |
| `agents/observation/reactive.py` | writes `vec[14]` from `progress_clock.value()`; `get_layout` entry |
| `agents/observation/state_encoder.py` | threads `progress_clock` into `encode()` |
| `agents/model/model_version.py` | `ARCH_SIGNATURE = gen3_markovian_progress_v1`; `bias_additivity`/`mat_alive_weight`/`bias_redesign` fields + `check_reward_config`; `MODEL_CONFIG_VERSION` 3→4 + migration |
| `agents/model/snapshot.py` | `enforce_reward_config` on the resume path |
| `src/main/train_rl_agent.py` | `--bias-additivity`/`--mat-alive-weight`/`--no-progress-penalty`/`--bias-redesign`; `RewardConfig` threaded to the reward factory; `PBRS_GAMMA == model.gamma` assert |
| `agents/training/golden_obs_fixture.json` | regenerated (3391-dim) |
| `agents/training/reward_manager_test.py`, `reward_invariants_e2e_test.py` | updated for the removed base spine + the new fields |
| `src/main/prober/engine_test.py` | offset pins re-pinned (om/tm 1502/1646, incoming 1469) |
| docs | root + `observation/` + `training/` `CLAUDE.md`; the forward design doc's as-built note |

---

## Future / next

- **Train it.** The first run is the single-variable `Φ_mat` clutch-fix (`bias_redesign` OFF); judge by
  the design §7.4 pre-registration (clutch-conversion ↑, useless-turn-rate ↓, ELO non-regressed — a
  PBRS-only ELO regression is a bug signal, not an outcome).
- **A/B arms** (later, by resume — same arch): `--bias-redesign` (clock + anti-spam collapse);
  `--bias-additivity < 1` (telescope the biases / the `Φ_hazard`/`Φ_status` no-double-count forms).
- **Build the deferred reframes** (roar/status/se_switch/switch_base) + the offline reward-replay
  falsifier (design §7.1a) before relying on the bias-redesign arm.
- **Pair with `--use-popart`** (design §6.3) — `Φ_mat`'s declared-team formulation already makes
  `Φ_mat(s_0)≈0`, but the value-scale guards (`grad/value_share`) decide.
