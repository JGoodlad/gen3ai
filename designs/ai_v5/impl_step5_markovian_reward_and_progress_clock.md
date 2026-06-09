# Implementation: Step 5 — Markovian / PBRS reward redesign + the no-progress clock

Make **every** reward term either **PBRS** (objective-neutral, telescoping) or **Markovian-w.r.t.-the-
observation** (a clean, obs-keyed bias), so the critic faces no irreducible per-term variance and the
material spine stops biasing the optimum toward dominant-over-clutch wins. Targets the
clutch-vs-dominant return skew (clutch win +26 vs dominant +47, a faint costing −5.82 even in *won*
games) and gives the anti-spam family a single obs-keyed counter the model can state-condition on.

> **Status: BUILT + TRAINED (both arms); the `bias_redesign=ON` arm regressed on its first run and was
> fixed.** (`ARCH_SIGNATURE = gen3_markovian_progress_v1`, obs 3390 → 3391.) The `bias_redesign=OFF`
> single-variable arm ran as `ai_v5_5_popart` (PopArt validated, under-switching improved — see
> [[project_popart]]). The `bias_redesign=ON` arm ran as `ai_v5_6` and **lost to random** (a
> switch-bounce farm) with a meaningless eval reward; both were root-caused and fixed —
> see **"First `bias_redesign=ON` run (`ai_v5_6`) — regressions found + fixed"** below. The forward
> design (the two axes, the §9 adversarial-review ledger, rejected alternatives) is
> `design_markovian_reward_and_features.md` (its two AS-BUILT 2026-06-08 notes record the same fixes).

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
| Anti-spam family (`repetition`/`dead_matchup`/`struggle` taxes) | **active, as today** | **suppressed** — the clock subsumes them |
| `switch_bouncing_tax` | **active** | **KEPT active** (as-built fix: the clock does NOT subsume *switch*-spam — `ai_v5_6` bounce-farmed) |
| `no_progress_tax` | **0** (clock tracks the obs scalar only) | charged (`−no_progress_penalty` per gated no-op) |
| `switch_base`/`se_switch`/`status` reframes | as today (hidden-state forms) | **obs-keyed forms** — but the `switch_base` **spam-gate is KEPT** + a back-to-back switch zeros the whole switch family (as-built fix) |

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
generically** (`_fold_bias_refund` sums `registry_fields(BIAS)`); TERMINAL + the three PBRS terms are
**explicit named folds** (`_fold_material_pbrs` / `_fold_belief_pbrs` / `_fold_status_pbrs`) because each
PBRS term carries its own `_prev_phi_*` telescoping state a generic loop can't hold. `process_turn_reward`
reads as a phase sequence over these helpers; the telescoping math (`γ·Φ′−Φ` + the `Φ(terminal)=0`
zeroing — the §2.3 dominant-win footgun) lives in **one** `_pbrs_step` helper.

---

## Where it deviated from the design

- **Staging via `bias_redesign`** (above) — the design described the staging conceptually (§1.3); the
  implementation made it a single resume-immutable flag and made the obs scalar arch-shared so the
  arms are resume-compatible. This is the cleanest realization of the "single-variable first run".
- **Obs-keyed reframes — BUILT, gated on `bias_redesign`.** The Markovian-purity reframes that key on
  obs-present quantities instead of hidden `self._prev_*`/`_last_*` state ship gated, so the default
  run stays byte-identical: `switch_base` (#18) drops its `last_switch_turn` spam-gate → a flat
  per-voluntary-switch bias (the clock handles spam); `se_switch` (#25) drops its `_last_opp_seen_by`
  once-per-matchup gate (the SE-threat fact is in the matchup obs); `status` (#29) keys on the
  TurnDelta transition events (`status_applied`/`status_cured`) — `+` on landing a status / a self-cure,
  `−` on being statused / the opp curing (e.g. Rest) — instead of the `_prev_*_statused` count diff.
  **`roar` (#9) needed no change:** `_prev_opp_boosts` is the decision-time opp-boosts the model saw
  in its obs (the boosts the Roar phazed away); at reward time `live` shows the reset board, so the
  snapshot is the only — and already obs-recoverable — source.
- **`Φ_status` — BUILT as a third always-on-style PBRS (non-damaging only), gated on `bias_redesign`.**
  The event-form `status` reframe drops the *standing* value of a held non-damaging status (par/slp/frz —
  tempo, not in `Φ_mat`); `Φ_status` (`_compute_phi_status` / `_fold_status_pbrs`, `pbrs_status`) =
  `STATUS_TEMPO_WEIGHT·(opp_tempo_statused − our_tempo_statused)` restores it as a telescoping
  (`Φ_status(s_0)=0`, terminal-zeroed → net-zero, policy-invariant) potential. It is **gated on
  `bias_redesign`** (the default count-diff `status` BIAS already pays the standing value → double-count
  otherwise) and covers **non-damaging statuses only** (Toxic/burn/poison ride `Φ_mat`'s chip → disjoint,
  no double-bridge). It adds **no** resume-immutable field (rides `bias_redesign`). This is the §2.7 /
  §7.4 hedge shipped pre-emptively as a potential. NB: the design's *other* §2.6–2.7 telescoping form,
  `Φ_hazard` (spikes), still activates only at `bias_additivity → 0` (a later A/B arm) — it was not
  separately built.
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
| Full unit suite (`not integration and not e2e`) | **1930 passed**, 2 skipped (1919 baseline + the `_GROUPS`-pin + 4 reframe tests + 6 `Φ_status` tests) |
| New reward-redesign tests (`reward_redesign_test.py`, NEW) | registry coverage, `Φ_mat` telescoping + terminal-zeroing (the 6-0-win-must-not-bonus guard), **`Φ_status` (non-damaging-only, gated-off-by-default, side-symmetry, application/cure fold, telescopes-to-zero)**, bias λ=1 byte-no-op + parameterized blend (λ∈{0,0.5,1}), the bias_redesign reframes, the full ProgressClock predicate (miss/heal/prevented freeze, our-attributed dmg, forced-switch + trapped gates) — all pass |
| Golden-obs parity (byte-exact) | fixture regenerated (**3391**-dim, 991 decisions) → passes (deterministic) |
| obs-build benchmark | **~7,173 calls/encode** — *below* the ~7.3k `gen3_incoming_damage_v2` reference (the clock scalar is one `math.log` + one array write); well under the 10% gate |
| Model roundtrip + `--debug` smoke (bridge, GPU + CPU `--bias-redesign`) | `[ModelVersion] Round-trip smoke test PASSED`; episodes complete; no NaN; `model_config.json` records `arch=gen3_markovian_progress_v1` / `total_dim=3391` / `config_version=4` / the reward hparams |
| Inference player tests | pass (the clock-wiring helper) |

**Not built (deferred):** the efficacy gate (post-retrain: clutch-conversion ↑, useless-turn-rate ↓,
ELO non-regressed — pre-registered in design §7.4); the offline reward-replay falsifier over
`eval_traces` (design §7.1a). The full `learn()`→eval "Training complete"
literal wasn't captured (GPU saturated by the live run; CPU+bridge too slow for the smoke timeout) — but
round-trip + training-entry are confirmed and the PPO update math is unchanged (just +1 auto-discovered
obs dim).

---

## Module map

| File | Change |
|---|---|
| `agents/training/progress_clock.py` | **NEW** — the `ProgressClock` (ternary predicate, `value()`, `last_penalty`) |
| `agents/training/reward_redesign_test.py` | **NEW** — registry / `Φ_mat` / `Φ_status` / bias-additivity / ProgressClock / reframe tests |
| `agents/training/reward_manager.py` | registry + `RewardClass`/`RewardConfig`; `Φ_mat`; **`Φ_status` (`_compute_phi_status`/`_fold_status_pbrs`, `pbrs_status`, `STATUS_TEMPO_WEIGHT`/`_TEMPO_STATUSES`)**; bias-refund; `pbrs_material`→`pbrs_belief` rename; the `_fold_*`/`_pbrs_step`/`_log_turn` helpers; `Φ_mat`/clock constants; gentle `stall_tax` |
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

## First `bias_redesign=ON` run (`ai_v5_6`) — regressions found + fixed

The staging worked as intended for the OFF arm (`ai_v5_5_popart`: clean `Φ_mat` clutch-fix, PopArt
validated, under-switching improved). Turning `bias_redesign` **ON for the first time** exposed three
problems — two design assumptions that were falsified, plus a latent infra bug. All fixed; the design
doc carries the full as-built detail in its two **AS-BUILT 2026-06-08** notes.

1. **The no-progress charge was a no-op in the OFF arm → stalls ran free (commit `d7aa983`).** With
   `--no-bias-redesign`, `_apply_progress_clock` early-returns, so the clock only fed the obs scalar and
   charged 0 — and `ai_v5_5` produced 2247 **self-play mirror Recover/Rest heal-wars** to the 250-turn
   cap. Even with the charge ON, the `DENIED` freeze (productive heal) let a *mutual* heal-war run
   uncharged. **Fix:** a 5th PROGRESS branch (our-owned residual chipping the opp net-down → a *winning*
   Toxic/Leech stall is never taxed) + a `HEAL_FREEZE_GRACE` streak cap (a *sustained* heal-war charges)
   + a new **`--draw-penalty`** terminal (the cap is a forfeit-LOSS, so a timeout is detected by
   `turn ≥ cap` and can be priced worse than a clean loss). `MODEL_CONFIG_VERSION 6→7`. Guarded by
   `progress_clock_fuzz_test.py`.

2. **Switch-bounce farm — §3 #18/#19 falsified (commit `cf043dc`).** The ON arm **lost to random** (94%
   switches, ELO ~516): the per-switch reframes (`se_switch`+`escape`+`switch_base` ≈ +0.95/switch) dwarf
   the clock's flat −0.15, so dropping the `switch_base` spam-gate (#18) and declaring `switch_bouncing_tax`
   "subsumed by the clock" (#19) made bouncing net-positive. **Fix:** `switch_base` spam-gated in both arms;
   under `bias_redesign` a back-to-back switch zeros the **whole** switch-reward family (cycle-agnostic);
   `switch_bouncing_tax` is **no longer suppressed**. #18/#19 are as-built **reversed** — the clock
   subsumes repetition/struggle/dead-matchup but **not** *switch*-spam.

3. **Silent eval mismeasurement → single-source `RewardConfig` (commit `cf043dc`).** Eval scored with a
   DEFAULT `RewardConfig()` (`bias_redesign=False`), so the `ai_v5_6` eval reward was meaningless (~−108,
   the old escalating bounce tax the policy never trained against). A hand-threaded config was silently
   missed on the eval path — hidden until now because every prior run used the defaults. **Fix:**
   `RewardConfig.from_args` (single construction) + `from_dict` (single reconstruction from
   `model_config.json`); the eval worker builds the reward factory once from `model_config.json` and
   threads it to every `EvalRLPlayer`; `EvalRLPlayer`/`build_eval_players` now **require**
   `reward_fn_factory` (a missing config is a loud error). Adding a reward flag is now 2 places, not ~8.

**Net:** the ON arm is now safe to run. Re-run is a fresh `--bias-redesign --draw-penalty -35`; judge by
the §7.4 pre-registration below (and the new guards: stall count ↓, switch-bounce absent, win-rate vs
`staller`/`staller_v2` not dropped).

---

## Future / next

- **Train it.** The first run is the single-variable `Φ_mat` clutch-fix (`bias_redesign` OFF); judge by
  the design §7.4 pre-registration (clutch-conversion ↑, useless-turn-rate ↓, ELO non-regressed — a
  PBRS-only ELO regression is a bug signal, not an outcome).
- **A/B arms** (later, by resume — same arch): `--bias-redesign` (clock + anti-spam collapse + the
  Markovian-purity reframes + the `Φ_status` standing-tempo potential); `--bias-additivity < 1`
  (telescope the biases / the `Φ_hazard` no-double-count form).
- **The Markovian-purity reframes are built** (roar/status/se_switch/switch_base) **plus `Φ_status`**
  (the non-damaging-tempo standing potential that restores what the event-form `status` drops), all
  gated on `bias_redesign` so the default run stays byte-identical. On the first `bias_redesign` run,
  watch the §7.4 guard (status-application rate must not collapse). Still future: the offline
  reward-replay falsifier (design §7.1a) to confirm the bias-redesign arm before relying on it.
- **Pair with `--use-popart`** (design §6.3) — `Φ_mat`'s declared-team formulation already makes
  `Φ_mat(s_0)≈0`, but the value-scale guards (`grad/value_share`) decide.
