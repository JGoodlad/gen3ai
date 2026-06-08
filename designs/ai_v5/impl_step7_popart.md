# Implementation: Step 7 — PopArt value-target normalization (the value-swamping fix)

Stop the value loss from **swamping the shared trunk**. With γ=0.9999 the returns run to ±hundreds
(`run_20260606`: `return_std`≈23.6, `abs_max`≈225), so the value MSE gradient dwarfs the policy
gradient on the dual-head extractor's shared transformer body — measured directly by the Step-pre
`grad_balance.py` diagnostics as `grad/value_share`≈**0.997** (value is ~600× the policy pull). The
policy under-updates (~2× slower learning vs the prior run; verified by forensics). `vf_coef` alone
can't fix it: balancing a σ≈23.6 scale needs `vf_coef`≈0.002, which starves the critic. PopArt fixes
the value **scale** adaptively instead.

> **Status: BUILT & SHIPPED** (commit `0f97e9d`). Opt-in **`--use-popart`** (default OFF → the default
> run is byte-unchanged). **No obs/arch change** → `ARCH_SIGNATURE` unchanged at
> `gen3_markovian_progress_v1`, obs **3390**; `MODEL_CONFIG_VERSION 5 → 6` (new `use_popart` field).
> As-built record. The forward design is `design_markovian_reward_and_features.md §6.3` (the "PopArt
> pairing" sub-design that motivated it). Efficacy gate (`grad/value_share` falls to ~0.4 and the
> policy-learning curve catches the baseline) is pending a fresh `--use-popart --clip-range-vf none`
> training run — the `--debug` smoke confirmed the *mechanism* (4 train() calls), not convergence.

---

## What shipped (one paragraph)

PopArt (van Hasselt et al. 2016) value-target normalization, behind **`--use-popart`**. A
`PopArtNormalizer` keeps a running `(mu, sigma)` of the value targets (the returns); the value head
emits **normalized** values and the PPO loss trains in normalized space
(`MSE(normalize(returns), normalize(values))`), so the value gradient into the shared trunk stays
O(1) instead of O(σ²). The **POP** half rewrites the value head's output `Linear` on every stats
update (`W'=(σ_old/σ_new)·W`, `b'=(σ_old·b+μ_old−μ_new)/σ_new`) so the **de-normalized** prediction is
unchanged for every input — making the stats update a no-op on the learned value function (the
corruption-free property that naive running-std normalization lacks). The policy **de-normalizes** at
all three value sites (`forward`/`evaluate_actions`/`predict_values`), so GAE / advantages / the
rollout buffer stay real-unit — the policy path is untouched. `use_popart` is version-checked in
`ModelVersion` (config v6, recorded in `model_config.json`, a dedicated `check_compatible` error so it
cannot be toggled on a resumed model — it changes the value-head state_dict). New TB/TUI diagnostics:
`popart/mu`, `popart/sigma`, `popart/value_weight_norm`. Mutually exclusive with vf-clipping —
`--use-popart` **requires an explicit `--clip-range-vf none`** (clipping is unnecessary with value
normalization and would clip in un-normalized units).

## Constants (`agents/model/popart.py`)

| Constant | Value | Meaning |
|---|---|---|
| `_DEFAULT_BETA` | `0.1` | per-`train()`-call EMA decay for the running `(mu, second-moment)` (~10-update window) |
| `_SIGMA_FLOOR` | `1e-2` | lower bound on `sigma` (divide-by-zero guard on a near-constant return batch) |
| `MODEL_CONFIG_VERSION` | `5 → 6` | `use_popart` field added; migration defaults old configs to `False` |
| `ARCH_SIGNATURE` | unchanged | `gen3_markovian_progress_v1` — PopArt is config-only, not an obs/arch change |

## The two halves (ART + POP) — `PopArtNormalizer`

- **ART (Adaptively Rescale Targets).** `update(returns, value_net)` is called **once per `train()`**
  (before the gradient epochs) and advances `(mu, sigma)` by EMA over the rollout's returns; the value
  loss is then computed on `(target − mu)/sigma` so it — and its gradient into the shared trunk —
  stays O(1). `mu`/`sigma`/`nu` (second moment)/`initialized` are registered **buffers** → they ride
  the policy `state_dict` and save/restore across checkpoints automatically.
- **POP (Preserve Outputs Precisely).** The same `update` rescales `value_net`'s weight + bias so its
  **de-normalized** output is identical before/after the `(mu, sigma)` change — verified output-
  preserving to **2.4e-7** in `popart_test.py`. The rescale runs under `no_grad` and touches `value_net`
  outside the optimizer; momentum staleness is negligible because the EMA keeps `σ_old/σ_new ≈ 1` each
  call (the standard PopArt approximation; optimizer state intentionally not rescaled).

Pure (torch-only, no SB3) → the math unit-tests without a training loop. The load-bearing test is
**POP invariance**.

## Policy + PPO integration

| File | Change |
|---|---|
| `agents/model/policy.py` | `Gen3DualHeadMaskablePolicy.__init__(use_popart=…)` builds `self.popart` **after** `super().__init__` (which builds `value_net`); `_denorm` wraps the 3 value sites so callers see real-unit values. `popart=None` (identity `_denorm`) when off. |
| `agents/training/instrumented_ppo.py` | `train()` reads `self.policy.popart`; once per call `popart.update(rollout_buffer.returns, value_net)`; value loss = `MSE(popart.normalize(returns), popart.normalize(values))`. Logs `popart/{mu,sigma,value_weight_norm}`. Coexists with the Step-pre `grad_balance` probe (whose `grad/value_share` is the efficacy signal). |
| `main/train_rl_agent.py` | `--use-popart` (`BoolFlag`); startup error if combined with a non-None `--clip-range-vf`; `policy_kwargs["use_popart"]` threaded on both fresh + resume paths. |

## Version-check (the "structural toggle" pattern, distinct from `vf_coef`/reward)

`vf_coef` (Step-pre) and the reward hparams (Step 5/6) are **value-meaning** — they don't change the
forward, so they ride resume-only `check_vf_coef` / `check_reward_config` and are excluded from
`check_compatible`. `use_popart` is different: it changes the value head's **structure** (adds
`mu/sigma` buffers + normalized output), so a mismatch breaks the `state_dict` on **every** load
(resume / eval / pool / distill). It therefore lives in **`check_compatible`** with a dedicated,
tailored error. The litmus test, now documented in `src/agents/model/CLAUDE.md`: **value-meaning →
resume-only `check_*`; structural → `check_compatible`.** A toggle-on-resume raises `ModelVersionError`
→ `train_rl_agent` exits `FATAL_CONFIG`, so the launcher gives up instead of restart-looping.

## Why `--use-popart` requires `--clip-range-vf none`

PopArt replaces value clipping — clipping is unnecessary with value normalization (and the broader
PPO literature finds `clip_range_vf` little-to-negative regardless). The constraint is **explicit and
required** (errors otherwise) rather than a silent override, so a PopArt run's command self-documents
`--clip-range-vf none`. The real hazard it guards: the value sites return *de-normalized* values, so an
active clip would clip in **un-normalized** units (`clip_range_vf` vs σ) and cripple the critic.

## Gates (all green at ship)

| Gate | Result |
|---|---|
| POP-invariance unit test (`popart_test.py`) | de-normalized outputs identical across a stats update to **2.4e-7** |
| Full unit suite (`not integration and not e2e`) | **1964 passed**, 2 skipped (combined tree after rebasing onto 13 upstream commits) |
| Model roundtrip + `--debug --use-popart --clip-range-vf none` smoke (bridge) | `Round-trip smoke test PASSED`; 4 `train()` calls clean; `popart/mu,sigma` track `train/return_mean,return_std` EXACTLY; `train/value_loss` normalized to O(1) (**1.07→0.32**); `grad/value_norm` O(1) (not σ²); `grad/value_share` **falling 0.9965→0.9742** as `policy_norm` grows (0.003→0.029) |
| Version-check | migration v5→v6 defaults `use_popart=False`; `check_compatible` raises on toggle; `vf_coef` differs ⇒ `check_compatible` does NOT raise (resume-only) |
| Mutual-exclusion guard | `--use-popart` without `--clip-range-vf none` errors (exit 2); with it, proceeds |

**Pending (the efficacy gate):** a real training run. On `run_20260606`'s scale (σ≈23.6) PopArt should
take `grad/value_share` from ≈1.0 toward ~0.4 at a normal `vf_coef`, and the policy-learning curve
should close the ~2× gap to the baseline.

## Module map

| File | Change |
|---|---|
| `agents/model/popart.py` | **NEW** — `PopArtNormalizer` (ART + POP, pure torch) |
| `agents/model/popart_test.py` | **NEW** — POP-invariance + stats-tracking + sigma-floor + EMA + state_dict tests |
| `agents/model/policy.py` | `__init__(use_popart)` + `_denorm` at the 3 value sites |
| `agents/model/model_version.py` | `use_popart` field (v6) + factory + migration + dedicated `check_compatible` block |
| `agents/training/instrumented_ppo.py` | popart update + normalized value loss + `popart/*` metrics |
| `main/train_rl_agent.py` | `--use-popart` (BoolFlag) + clip-range-vf guard + `policy_kwargs` (fresh + resume) |
| `main/launcher/format.py` | `popart/*` display order + labels |
| docs | `model/` + `training/` + `launcher/` `CLAUDE.md`; this doc; `todo.md` |

## Forward design / context

- Forward design: `design_markovian_reward_and_features.md §6.3` (PopArt pairing **[RT-6]**) — the
  markovian reward redesign explicitly pairs with PopArt because its material-density return-target
  structure adds to the trunk this run is already fighting.
- The diagnostics that motivated + measure it (`grad/value_share`, `grad/value_policy_logratio`,
  `train/return_*`) are the Step-pre `grad_balance.py` block (see `src/agents/training/CLAUDE.md`).
