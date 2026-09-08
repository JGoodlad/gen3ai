# Training — the PPO step (`instrumented_ppo/`)

Moved verbatim out of `src/agents/training/CLAUDE.md` on **2026-09-08**, in the second pass of the
topic split. `src/agents/training/CLAUDE.md` keeps the **FOLD ORDER contract** itself — the
numbered per-minibatch sequence and the 7-before-8 stash hazard — because that is the thing a fold
edit must not get wrong. This file owns the package's module map and the two source-pin rules
around it.

---

## The package map

**`instrumented_ppo` is a PACKAGE** (2026-08-23; it was a single 2,152-line file, the last entry
on the size ratchet's grandfathered list). `__init__.py` is a pure re-export hub, so every
`from agents.training.instrumented_ppo import <name>` resolves unchanged:

| module | holds |
|---|---|
| `ppo.py` | `InstrumentedMaskablePPO` + `train()` — the vendored upstream override and **the whole fold sequence**, plus `train_step_source()` (below) |
| `train_setup.py` | **the pre-loop half of `train()`** — `_align_opp_intent_labels`, `_resolve_fold_flags` (→ `FoldFlags`: which terms this call folds, plus the counterfactual buffer's one per-rollout poll) and `_train_probe_setup` (→ `ProbeSetup`: the once-per-call diagnostics, PopArt's advance, the two gradient samplers). Both containers are `NamedTuple`s carrying the SAME names the fold uses, so `train()` unpacks them back into the locals the loop was written against |
| `metrics_export.py` | **the ~400-line `self.logger.record` tail** — diagnostics, no gradient, one method per TB prefix group, each taking the accumulators this call filled |
| `rollout_probes.py` | `collect_rollouts`, the entropy-boost schedule (`_annealed_entropy_boost` and its two accessors) and `_winprob_start_metrics` — per-ROLLOUT work that is not part of the fold at all |
| `hparams.py` | every after-construction knob `train_rl_agent` sets (`value_tail_weight`, the belief/intent/cf coefficients, `grad_accum_steps`, …) with the rationale comment each carries, plus `_excluded_save_params` |
| `noise_scale.py` | the McCandlish gradient-noise-scale estimator + the rate-limited NSR advisor + `noise_ratio_sample`, the read seam `--adaptive-batch` steers by |
| `noise_scale_terms.py` | the PER-LOSS-TERM half of it — is the total reading the POLICY gradient's, or the dense aux heads'? |
| `distill_terms.py` | search-teacher AWR · OPD · the exploiter-distillation family (policy KL — or the top-K/action-CE form with the advantage gate, `_gated_action_distill_loss` — value MSE, the FitNets hint) |
| `value_terms.py` | the win-prob BCE · the value-dist HL-Gauss CE · `_value_loss_from_se` |
| `aux_terms.py` | the `belief_bank` / `td_aux` / `cf_terms` delegates |
| `constants.py` | `_VALUE_TAIL_FRAC` · `_WIN_CONTESTED_TAU` · `_NOISE_SCALE_EMA_DECAY` |


## The two source-pin rules

🚨 **A SOURCE-LEVEL PIN THAT SAYS "in `train()`" SHOULD READ `ppo.train_step_source()`** — `train()`
concatenated with the three setup methods and the seven export methods. The fold, its setup and its
export are ONE train step; which of the three modules a given line sits in is a decomposition
detail, and a pin that depends on it breaks on a move that changed nothing. Five test files read it
(`instrumented_ppo_winprob_critic_test`, `vf_coef_scale_readout_test`,
`instrumented_ppo_noise_scale_terms_test`, `winprob_pbrs_test`, and the hub contract's own
docstring). **The fold's own ORDERING pins stay on `inspect.getsource(train)`**, where
straight-line source order is the thing being checked. The same rule holds for a MONKEYPATCH: a
test that stubs a diagnostic must patch the module that now owns the call
(`train_setup.advantage_density_metrics` / `train_setup.shared_trunk_parameters` /
`metrics_export.live_gauge_metrics`), because patching `ppo` would silently stub nothing and the
byte-identity test would then compare two identical arms and pass. Per minibatch:

The upstream-drift hash check (`_verify_upstream_unchanged` + `_EXPECTED_UPSTREAM_TRAIN_HASH`)
stays in the HUB on purpose: `instrumented_ppo_test` patches that global on the module object it
imports, so moving it into a submodule would have left the patch reaching a different global than
the function reads — a test that still passes, for the wrong reason.

## Rollout collection: sync barrier vs `--async-rollout` (`async_vec_env.py`)

> `InstrumentedMaskablePPO.collect_rollouts` is what dispatches to it, which is why the detail
> sits beside the PPO step. The leaf keeps the seam rule (`env_method` PULL, not an info-dict
> thread) and the headline FPS number.

The default `SubprocVecEnv.step()` is a **per-step barrier** — the trainer waits for the slowest of
N env workers every step, so a slow battle turn / heavy opponent forward / oversubscription jitter
stalls the whole batch and the GPU policy-forward never overlaps CPU env-stepping. `--async-rollout`
swaps in **`AsyncSubprocVecEnv`** (per-env `send_step`/`poll_ready`/`recv_step` over the pipes +
**drain-safe `env_method`** — the eval callback's `set_self_play_target`/
`opponent_default_stats` fire mid-collection, so the override stashes in-flight step results before
any barrier RPC to avoid a pipe desync) and **`collect_rollouts_async`**, dispatched by
`InstrumentedMaskablePPO.collect_rollouts` when `model._async_rollout` is set.

The collector keeps every worker continuously in-flight, batch-forwards whichever envs are READY
(dynamic batch), and writes each env's transition into **its own buffer column**
(`MaskableDictRolloutBuffer`); collection ends when every column has `n_steps`. It is **exactly
on-policy** — PPO freezes the policy during collection, so this is a *scheduling* change (overlap
forward with stepping, drop the max-latency barrier), NOT an APPO-style algorithm change. Bookkeeping
(`num_timesteps`, GH-#633 timeout bootstrap, `_update_info_buffer`, `_last_*` carry-over, per-column
GAE) mirrors the stock loop exactly. The per-decision **mask rides in the Dict obs**
(`obs["action_mask"]`, = `last_ctx.mask`), so no per-env `env_method` and no wrapper change.

**Measured FPS (bridge, GPU forward, steady-state, heuristic opponents):** +20% at `--n-envs 16`;
**+14% at the production `--n-envs 64` (1489→1695)**; `--async-rollout --n-envs 32` matches `sync@64`
FPS with half the envs (≈half the env/bridge RAM). Off by default (stock `SubprocVecEnv`), ignored
under `--debug`. Caveat: benchmarked with heuristic opponents — re-bench under `--self-play` for the
production-regime number. Full design + benchmark table: `designs/ai_v5/design_async_rollout.md`.
