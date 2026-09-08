---
name: project_popart
description: "PopArt value-target normalization — flag-guarded (--use-popart), version-checked, shipped; the fix for value swamping the shared trunk"
metadata: 
  node_type: memory
  type: project
  originSessionId: c7a2c4b5-8ff9-4a00-b8ce-a07be32d63b9
---

> **Archived 2026-09-08** — PopArt is SHIPPED and is OFF by construction under the ai_v12 win-prob critic; mechanism in designs/learning/popart_value_scale_and_currencies.md. Preserved verbatim; nothing below is current.

PopArt (van Hasselt 2016) value-target normalization — built to fix the value gradient **swamping**
the shared trunk (diagnosed via `grad/value_share`≈0.997 on run_20260606; see
[[project_throughput_profile]]). `vf_coef` alone couldn't fix it because the cause is the return
*scale* (γ=0.9999 → return_std≈23.6, abs_max≈225): balancing via `vf_coef` would need ~0.002
(critic-starving). PopArt normalizes the value loss by σ² → value gradient O(1) at a normal `vf_coef`.

**Status (2026-06-06):** built + verified in worktree. Built ON TOP of the committed grad_balance
feature (commit 74373ea). **SHIPPED + RUN — see the run-outcome block at the bottom.**

**Design (opt-in, default off):**
- `src/agents/model/popart.py` — `PopArtNormalizer` (pure/torch): running `(mu,sigma)` buffers,
  `normalize`/`denormalize`, `update(returns, value_net)` does the **POP** weight-surgery
  (`W'=(σ_old/σ_new)W`, `b'=(σ_old·b+μ_old−μ_new)/σ_new`) so de-normalized outputs are preserved
  across a stats update (verified output-preserving to 2.4e-7). `_DEFAULT_BETA=0.1` EMA,
  `_SIGMA_FLOOR=1e-2` (module constants; only flag is on/off).
- `policy.py` (Gen3DualHeadMaskablePolicy) — `__init__(use_popart)` builds `self.popart` AFTER
  super().__init__ (value_net must exist); `_denorm` wraps the 3 value sites
  (forward/evaluate_actions/predict_values) so GAE/advantages stay real-unit.
- `instrumented_ppo.train()` — once per call (pre-epochs) `popart.update(rollout_buffer.returns,
  value_net)`; value loss = `MSE(normalize(returns), normalize(values))`. Mutually exclusive with
  vf-clipping (`--use-popart` auto-disables value clipping (override + notice)).
- **Version-checked:** `ModelVersion.use_popart` (config_version bumped 2→3, migration default
  False) → recorded in model_config.json; **dedicated** `check_compatible` block raises a clear
  error if a resume toggles it (can't flip mid-run — value head parameterization differs).
- **TB+TUI diagnostics:** `popart/mu`, `popart/sigma` (track return_mean/std), `popart/value_weight_norm`.

**Verification:** POP-invariance unit test (2.4e-7), roundtrip PASSED (serialize/version-check/policy
reconstruction with popart buffers), 685 training-dir tests pass, explicit --clip-range-vf none required (errors otherwise) + migration + check_compatible raise
on toggle. Live --debug smoke (4 train() calls): mu/sigma track return_mean/std EXACTLY,
`train/value_loss` normalized to O(1) (1.07→0.32), `grad/value_norm` O(1) not σ², **`grad/value_share`
falling 0.9965→0.9742** as `policy_norm` grows (0.003→0.029). The value_share→~0.4 payoff is a
training-time emergent (policy_norm must grow off the random init) — confirm on the real GPU run.

To run: `--use-popart --clip-range-vf none`. See [[project_outgoing_damage_design]] for the other
open obs work; grad_balance metrics ([[project_throughput_profile]] context) are the live gauge.

**UPDATE 2026-06-08 — RUN OUTCOME, reviewed @~36M (`models/ai_v5_5_popart_N_0607`, PopArt + native
self-play, the single-variable run = markovian impl with `--bias-redesign` OFF, `--switch-bias-weight 0`).
PopArt did its job, confirmed on the real GPU run:** `popart/sigma` stable ~20 (carries the return
scale), `value_loss` held **O(1) ~0.12** (not exploding), `grad/value_norm_shared` collapsed 4→0.24 and
**`grad/value_share` fell 1.0→~0.39** (value no longer swamps the shared trunk — the training-time
payoff the smoke only hinted at, now REALIZED), `value_policy_cosine`≈0 (heads not fighting). Skill rose
cleanly: **`eval/elo` 1400→~1850** (monotone, tight CI), explained_variance ~0.84, entropy stable (no
collapse), no crashes. Mean bot win-rate 0.27→**0.84** and still inching up. The run was reviewed via a
multi-agent workflow (4 plot-vision agents + forensics + adversarial verification) — full picture in
[[project_incoming_damage_outcome]] (threat-response: critic reads strongly, policy weak; surprise-OHKO
belief-UNDER-READ is the majority failure) and [[project_anti_stall_fix]] (the 2247 self-play mirror
stalls + the shipped fix). **Net: PopArt is validated and is now standard for these runs.**
