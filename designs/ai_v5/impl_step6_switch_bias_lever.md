# Implementation: Step 6 — Belief-risk-scaled switch BIAS lever (the under-switch fix)

Give the policy a reward signal that actually re-ranks **stay vs switch** when the incoming-KO belief
is high, because the shipped potential-based shaping (`pbrs_belief`, Step-5 PBRS class) is
objective-neutral by construction and — verified on `run_20260607_102632` — left the under-switch
pathology unchanged (switch-mass still inverts vs P(KO); stay-and-die ≈ 61%, == the V1 control).

> **Status: BUILT** (reward-only; **no obs/arch change** → `ARCH_SIGNATURE` unchanged at
> `gen3_incoming_damage_v2`, obs **3390**; `MODEL_CONFIG_VERSION 4 → 5`), **not yet trained.** As-built
> record. Forward design + the verified null result it corrects: `design_reward_switching.md §7`.
> Efficacy gate (P(KO)≥0.8 switch-mass no longer inverted; stay-and-die ↓; no pivot-spam) is pending a
> fresh `--switch-bias-weight 1.5` run.

---

## What shipped (one paragraph)

Two new **BIAS-class** reward terms on the Step-5 registry — `stay_risk_tax` and `escape_risk_bonus` —
gated by a new resume-immutable hparam `RewardConfig.switch_bias_weight` (`--switch-bias-weight`,
**default 0.0 = OFF → the default run is byte-unchanged**). When on, the policy is taxed
`max(−w·risk, −2.0)` for STAYING into a high imminent-KO spot it could have escaped, and rewarded
`w·0.5·risk` for escaping such a spot to a safe pivot. `risk = max(phys_pko,spec_pko)·(1−P(outspeed))`
read from the incoming-damage belief block (the same snapshot the Step-5 escape/stay re-gate uses).
Unlike `pbrs_belief` (kept, as a free credit-assignment aid) these are **additive, action-conditioned**
biases → they change the objective the actor optimises. Being BIAS-class they ride `--bias-additivity`,
so a fixed weight at **λ=1 vs λ=0** is a clean causal A/B for "is it the objective tilt that helps."

## Constants (`reward_manager.py`)

| Constant | Value | Meaning |
|---|---|---|
| `RewardConfig.switch_bias_weight` | `0.0` (OFF) | lever magnitude `w`; recommended experimental `1.5` |
| `SAFE_PIVOT_PKO_MAX` | `0.35` | a non-fainted bench mon with incoming P(KO) ≤ this is a "safe pivot" |
| `STAY_RISK_TAX_FLOOR` | `−2.0` | per-turn clamp (below faint ≈ −3.25 via Φ_mat, far below ±30) |
| `ESCAPE_RISK_FRACTION` | `0.5` | escape reward = `w·0.5·risk` — asymmetric (< the tax) → no farm surface |

## The two terms + their gates (`_compute_stay_risk_tax`, `_apply_switch_outcome`)

`stay_risk_tax` fires only when ALL hold (the red-team-hardened gate):
- a switch was **LEGAL** this decision (`_cur_can_switch`, snapshot in `record_action` from
  `ctx.mask[:SWITCH_END]`) — **never tax a trapped stay** (the highest-severity review finding;
  Arena-Trap/Magnet-Pull states otherwise get the loudest false penalty);
- we did **not** switch (`our_switch_to is None`); the move did **not** fizzle to RNG
  (`not our_failed_to_move` — flinch/full-para/sleep/freeze, mirroring the progress-clock FREEZE); the
  move did **not** KO the opp (`not opp_fainted` → staying won the exchange);
- decision-time `risk_active ≥ SWITCH_RISK_THRESHOLD (0.5)` **and** a safe pivot existed
  (`_prev_safe_pivot`). Staying-and-**fainting** is included (the exact pathology).

`escape_risk_bonus` fires on a voluntary switch out of a `risk ≥ 0.5` spot **and** `_prev_safe_pivot`
(symmetry — reward escaping *to safety*, not sacrificing a fresh mon into the same threat; this also
closes the no-safe-pivot 3-mon rotation farm).

**Snapshots / timing.** `_belief_potential_and_risk` now returns a 3rd value, `min_bench_pko` (lowest
**raw** incoming P(KO) over non-fainted, HP>0 bench mons — raw because a switch-in always eats the
turn's hit; speed can't save it). `_fold_belief_pbrs` snapshots `_prev_safe_pivot = min_bench_pko ≤
0.35` and `_prev_active_ko_risk` at end of turn (= next turn's decision board); both are read in
`process_turn_reward` **before** `_fold_belief_pbrs` overwrites them. `_cur_can_switch` is set in
`record_action` (this decision). All three describe the decision the policy actually made.

## Files changed

- **`src/agents/training/reward_manager.py`** — constants; `RewardConfig.switch_bias_weight`;
  `RewardBreakdown.stay_risk_tax` / `escape_risk_bonus` (+ `_REGISTRY` → BIAS, + `_GROUPS`);
  `_compute_stay_risk_tax`; `escape_risk_bonus` in `_apply_switch_outcome`; `_cur_can_switch` snapshot
  in `record_action` (+ `import SWITCH_END`); `_belief_potential_and_risk` 3-tuple + `_fold_belief_pbrs`
  safe-pivot snapshot; init/reset of the two new `_prev_*`/`_cur_*` fields.
- **`src/agents/model/model_version.py`** — `MODEL_CONFIG_VERSION 4→5`; `switch_bias_weight` field +
  `from_kwargs` + `check_reward_config` (resume-immutable, FATAL on drift) + `_migrate_config` v5
  default 0.0. **Excluded from `check_compatible`/`_WEIGHT_FIELDS`** (frozen eval/pool/distill loads
  accept any value — it's value-meaning, not weight-shape).
- **`src/main/train_rl_agent.py`** — `--switch-bias-weight` arg → `RewardConfig(switch_bias_weight=…)`.
- **Tests** — `reward_redesign_test.py::TestSwitchBias` (16 cases: OFF-by-default, risk scaling, floor,
  the four gates incl. trapped + RNG-fizzle + no-safe-pivot, escape asymmetry + safe-pivot gate,
  registry membership, 3-tuple, terminal-zeroing snapshot, end-to-end fold); `reward_manager_test.py`
  `_phi` helper updated for the 3-tuple; `snapshot_test.py` adds direct `check_reward_config` coverage
  (match / switch_bias_weight-mismatch-FATAL / `check_compatible`-ignores).

## Deviation from the design / decisions

- **Kept `pbrs_belief`** rather than replacing it: it's policy-invariant (harmless) and aids credit
  assignment; the behavioural pull is the new BIAS terms. ("instead of PBRS" = *the lever* moved to
  BIAS, not a deletion.)
- **Flag-gated, default OFF** to preserve the Step-5 single-variable-attribution discipline — the lever
  is opt-in and A/B-able, never silently in the default run.
- **`record_action`-mask gate for trapping** (vs `delta.attempted_switch_rejected`): the mask is the
  decision-time server-authoritative legality, so it's correct even on the turn a trap is first applied
  (before any reject event exists).

## Verification

- **1069** training+model unit tests pass; the reward + snapshot suites (incl. the 16 new switch-bias
  cases + resume-immutability) green.
- **Bridge `--debug` smoke, lever ON (`--switch-bias-weight 1.5`):** `[ModelVersion] Round-trip smoke
  test PASSED` (config_version 5 serialise/reload), episodes complete, the `record_action`/mask path +
  the new belief 3-tuple run on real battles with no crash.
- **Adversarial red-team** (1 agent, 8 axes): verdict SHIP-WITH-FIXES → all must-fix + the high-value
  MEDIUM hardenings (trap gate, RNG-fizzle gate, escape safe-pivot gate, bench-HP guard) applied; the
  residual LOW notes are documented above.
