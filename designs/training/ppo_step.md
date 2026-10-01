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
| `ppo.py` | `InstrumentedMaskablePPO` + `train()` — the vendored upstream override: the call to region R1 (`micro_step`, fold steps 1-3a) and **the eager tail of the fold sequence** (3b onward), plus `train_step_source()` (below, which includes R1's source) |
| `train_setup.py` | **the pre-loop half of `train()`** — `_align_opp_intent_labels`, `_resolve_fold_flags` (→ `FoldFlags`: which terms this call folds, plus the counterfactual buffer's one per-rollout poll) and `_train_probe_setup` (→ `ProbeSetup`: the once-per-call diagnostics, PopArt's advance, the two gradient samplers). Both containers are `NamedTuple`s carrying the SAME names the fold uses, so `train()` unpacks them back into the locals the loop was written against. Also R1's `_micro_static` / `_micro_var` / `_micro_region` |
| `metrics_export.py` | **the ~400-line `self.logger.record` tail** — diagnostics, no gradient, one method per TB prefix group, each taking the accumulators this call filled |
| `rollout_probes.py` | `collect_rollouts` — the ENV-CORE DISPATCH (the Rust collector, the async collector, or `loop.OwnedLoop._collect_python`) plus PBRS / frozen-φ after it — the entropy-boost schedule (`_annealed_entropy_boost` and its two accessors) and `_winprob_start_metrics` — per-ROLLOUT work that is not part of the fold at all |
| `loop.py` | **THE LOOP IS OURS** (`gen3_owned_ppo_loop_v1`, [`design_own_ppo_loop.md`](../endstate/design_own_ppo_loop.md) stage 1): `OwnedLoop` — `learn()` written out as the declared `LOOP_PHASES` table (`setup → training_start → [collect → progress → dump → update]* → training_end`), plus `_setup_learn`, `dump_logs`, `_update_current_progress_remaining`, `_update_info_buffer` and the Python env core's `_collect_python`, each vendored from sb3 / sb3-contrib operation for operation and hash-pinned at import (`UPSTREAM_SOURCE_SHA256`). Three CONTRACTS: the dump comes BEFORE the update (update k's `train/*` is stamped after rollout k+1 — every TB series in the archive, and the KL controller's logger read, rest on it); `collect` / `update` go THROUGH THE ATTRIBUTE (K6's freeze guard and the compile sentinel wrap them as instance attributes); the Python collect keeps sb3's LOCAL NAMES and fires `on_step` before `rollout_buffer.add` (callbacks read `self.locals`, the win-prob labeller reads `buf.pos`). `GEN3AI_PPO_LOOP=sb3_reference` is a TEST SEAM (the equivalence A/B) that defers every vendored method to upstream; it goes with the Python core (stage 3). Pinned EXACT against upstream by `own_ppo_loop_test.py` (hook trace, buffers, dumps, params, resume) and, on real games, `own_ppo_loop_parity_test.py` (`sim`) |
| `hparams.py` | every after-construction knob `train_rl_agent` sets (`value_tail_weight`, the belief/intent/cf coefficients, `grad_accum_steps`, …) with the rationale comment each carries, plus `_excluded_save_params` |
| `noise_scale.py` | the McCandlish gradient-noise-scale estimator + the rate-limited NSR advisor + `noise_ratio_sample`, the read seam `--adaptive-batch` steers by |
| `noise_scale_terms.py` | the PER-LOSS-TERM half of it — is the total reading the POLICY gradient's, or the dense aux heads'? |
| `distill_terms.py` | search-teacher AWR · OPD · the exploiter-distillation family (policy KL — or the top-K/action-CE form with the advantage gate, `_gated_action_distill_loss` — value MSE, the FitNets hint) |
| `value_terms.py` | the win-prob BCE · the value-dist HL-Gauss CE · `_value_loss_from_se` |
| `aux_terms.py` | the `belief_bank` / `td_aux` / `cf_terms` delegates |
| `constants.py` | `_VALUE_TAIL_FRAC` · `_WIN_CONTESTED_TAU` · `_NOISE_SCALE_EMA_DECAY` |
| `learner_gates.py` | **K9 — the learner's in-loop GIGO gates** (`designs/training/learner_gates.md`): the python-core behaviour gate on the first micro-batch's own forward (`behaviour_gate_mode` picks it or Lane G's pre-loop probe, never both) and the fail-closed non-finite checks — the buffer (before PopArt), the assembled loss per micro-batch (naming the term), the approx-KL, and `clip_grad_norm_checked` at every optimizer step. All raise before the optimizer moves anything |
| `device_batches.py` | **K8 — the DEVICE-RESIDENT MICRO-BATCH** (`gen3_device_batches_v1`): `install(buffer)` at the top of `train()` puts an instance `_get_samples` on a CUDA buffer that gathers every micro-batch from ONE device copy of the flattened arrays (made at the first micro-batch — after the buffer's own `get()` flattened them and a fork buffer appended its rows); the buffer's own `get()` still draws the permutation, so the RNG stream and the batches are bit-identical to the host path; `uninstall` after the epoch loop drops the copy (the update's peak grows by it, ~1.1 GB at production shape; the quiescent floor does not). A CPU buffer (incl. the rust core's host-side one) keeps the host path |
| `intent_fold.py` | **K8 — the OPPONENT-INTENT fold as one STATIC, `fullgraph`-traceable function** (`intent_fold` / `intent_fold_tensors` → `IntentFoldOut`): the α/β cross-entropies, the set-valued partial credit and every `opp_intent/*` metric as `(value, weight)` pairs — no host read, no boolean-mask indexing, no `bincount`, no value-dependent branch; a term the legacy block would skip is an exact 0.0 with `present=False`, a metric it would not emit has weight 0 and value 0. Called by `micro_step` (R1). `instrumented_ppo_intent_fold_test.py` holds a VERBATIM copy of the old inline block + the `opp_intent` functions as the ORACLE: float64 metrics/terms to 1e-12 (the set-valued loss, which the legacy code computes in float32 by an explicit `.float()`, to float32 rounding), gradients bit-equal, and a `fullgraph=True` trace on torch 2.8 |
| `micro_step.py` | **K8 — the LEARNER MICRO-STEP, the compile REGION R1** (`gen3_learner_micro_step_v1`): `evaluate_actions` (functional masking) + fold steps 1-3a as ONE static-shape function → `MicroOut` (the loss, the probe terms + their noise groups + presence, every diagnostic as `(value, weight)`); `pack`/`unpack` are the ONE host read per micro-batch. `MicroStatic` (resolved per call by `TrainSetup._micro_static`) is every flag and coefficient it branches on or multiplies by; `_micro_var` the per-update tensors (the entropy boosts' factors, the strata weights). `train()` folds the eager tail onto `MicroOut.loss` |
| `phase_hook.py` | **BENCHMARK-ONLY** segment marks inside `train()` (`gen3_learner_phase_hook_v1`): `PHASE_HOOK` is None in production, read ONCE per call via `current()`, every site a guarded one-liner; the names and what each books are in its docstring. The one consumer is `agents/training/learner_benchmark.py`, which installs a `cuda.synchronize`-bracketed segment timer (or drives `torch.profiler` windows off the `epoch_end` mark). Pinned by `learner_benchmark_test.py` (guarded, named, and byte-identical parameters with a hook installed) |


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

## The policy's GAE λ (`--policy-gae-lambda`, `gen3_policy_gae_lambda_v1`, config v123)

**`--policy-gae-lambda FLOAT`, default `0.80`** — the λ `MaskableRolloutBuffer.compute_returns_and_advantage`
uses to turn rewards and recorded values into the ADVANTAGES the clipped surrogate is trained on
(and the `returns` the scalar value loss regresses toward). Until 2026-09-26 it was a literal
`0.80` at both `main/train/model_build.py` sites (the fresh `InstrumentedMaskablePPO(gae_lambda=…)`
and the resume path's `model.gae_lambda = …`); both now read `args.policy_gae_lambda`, so the
default is byte-identical. Range `[0, 1]` (a parser error outside it).

- **It is NOT `--win-prob-lambda`.** That is the CRITIC's λ-RETURN target for the win-prob BCE,
  computed in a separate post-collection pass over the recorded values
  ([`critic_and_value_losses.md`](critic_and_value_losses.md)). This one never touches the BCE
  target; the two can be set independently.
- **Recorded and inherited** — the `td_aux_coef` provenance class: a `ModelVersion` field
  (`policy_gae_lambda`, so `model_config.json`), `_resolve`-inherited on a flagless resume, never
  compared by `check_compatible`. A pre-v123 config migrates to 0.80 (the only possible past).
  SB3 also pickles `gae_lambda` in the zip, but the resume path OVERWRITES it with the resolved
  flag — so the recorded field, not the zip, is what a resume trains at. `metadata.json`'s
  per-checkpoint `gae_lambda` and the one-shot `hparams/gae_lambda` TB scalar read the live
  `model.gae_lambda`, as before.

## Per-epoch PPO diagnostics (`gen3_ppo_per_epoch_diag_v1`)

`train()` logs **`train/approx_kl_epoch_<k>`** and **`train/clip_fraction_epoch_<k>`** for
`k = 0 … n_epochs−1` — one pair per epoch the update actually ran (a `target_kl` early stop leaves
fewer; the tripping epoch is recorded, partial). Each is the MEAN of that epoch's per-minibatch
numbers — the same `approx_kl_div` and `clip_fraction` values the loop already computes for the
stock tags, sliced by epoch, so it adds no forward, no device sync, and no measurable cost.

⚠️ **The two stock tags pool DIFFERENTLY, which is why the per-epoch pair is worth having.**
`train/approx_kl` is the LAST epoch's mean only (stock SB3 resets its list every epoch), while
`train/clip_fraction` pools EVERY epoch's minibatches. So `approx_kl` == `approx_kl_epoch_<last>`,
and (with equal-size epochs) `clip_fraction` == the mean of the `clip_fraction_epoch_*` series.
Epoch 0 starts on the rollout policy, so `approx_kl_epoch_0` is the smallest in a healthy update;
the per-epoch curve is how the policy drifts across the `n_epochs` passes.

## The optional-telemetry cadence (`--diagnostics-every N`, `gen3_diagnostics_cadence_v1`, config v124)

M5 Lane K2. The learner benchmark measured one production update at 58.5 s, of which the OPTIONAL
probes are **12.7%** (the per-term noise-scale probe alone 11.5%;
`designs/research_state/measurements/learner_bench_2026-09-28/README.md`). None of them changes a
number the optimizer sees, so they run on **every Nth update**; the module that owns the rule is
`instrumented_ppo/diagnostics_cadence.py`.

| gated probe | its tags (written ONLY on a diagnostics update — a skipped update leaves a GAP) |
|---|---|
| per-term noise sampler (`noise_scale_terms.py`) | `train/noise_scale_<g>`, `train/noise_scale_ratio_<g>`, `train/noise_scale_share_<g>`, `train/noise_per_term_ms` |
| grad balance (`grad_balance_metrics`) | `grad/*`, and the two scalars read off it: `train/cf_grad_share`, `train/cf_evidential_grad_share` |
| effective rank (`rank_probe`) | `rank/{trunk,value_cls,policy,vf_feat}_*` |
| edge / cell liveness | `edge/*`, `cell/*` |

**Every update, unchanged:** the loss terms, `train/approx_kl{,_epoch_k}`, the clip fractions,
`train/grad_norm`, `train/train_ms`, the TOTAL `train/noise_scale{,_ratio}` (two grad-norm reads the
accumulation makes anyway — and `--adaptive-batch total`'s input), `signal/*`, the head metrics.
`--capacity-telemetry` (off in production) keeps its own `--capacity-*-every` cadences.

- **Which updates.** `rollout_index = num_timesteps // (n_steps · n_envs)`; an update is a
  diagnostics update iff `N == 1`, or `rollout_index % N == 0`, or it is **the first update of the
  process**. The phase rides the restored step counter, so it survives a restart. The first-update
  rule is the DECLARED LIFECYCLE's: `compile_control` locks after the first update, so every
  signature the probes use (the rank probe's no-grad train-mode forward, their
  `autograd.grad(retain_graph=True)`) must be seen by then — a probe first reached on update N
  would be a post-lock recompile, a typed FATAL. Its latch `_diagnostics_ran_in_process` is in
  `_excluded_save_params`, so every process re-arms it.
- **Load-bearing exemptions** (derived in `model_build.apply_training_hparams` from the SAME
  predicates `main.train.callbacks` registers the consumer with): `--rank-tripwire warn|abort`
  (production: `warn`) keeps `rank/*` EVERY update — the tripwire's skip / baseline / half-life /
  persistence constants are counted in READINGS and were validated at one per update;
  `--adaptive-batch policy` keeps the per-term sampler every update — the controller steps K off
  that EMA and its warm-up counts its samples.
- **Default and provenance.** A FRESH run resolves to **10** (≈ 90% of the 12.7% back; one reading
  per ~1.3M env steps at 64 × 2048). `ModelVersion.diagnostics_every`, `_resolve`-inherited on a
  flagless resume, never compared by `check_compatible` — the `policy_gae_lambda` class. A pre-v124
  config migrates to **1**, so a live or resumed older run keeps every-update series until the flag
  is NAMED. `main.checkargs` reports it through its parser sweep.
- ⚠️ **Readers that window by READING COUNT now span N× the updates.** The vf_coef restart rule's
  "last 20 `grad/value_policy_logratio`" (`scripts/ops/restart_read.sh`, `main.ops.tb_read`,
  `main.ops.vf_framings`) is 20 readings = 200 rollouts at N = 10; the readers print the step span.
  They already treat an absent tag as "no reading", never 0.
- **THE GUARANTEE — bit-identical learning** (`diagnostics_cadence_test.py`): on the production
  extractor surface (every edge family and pointer cell), one update with every probe ON and one
  with all of them SKIPPED leave identical parameters, AdamW state, loss / KL / clip / grad-norm
  scalars and torch / numpy / python RNG state; a second ON arm is the reproducibility control, and
  the ON arm must emit all five families. Mutation-checked 2026-09-29: a `th.rand(1)` inside the rank
  probe fails the RNG assertion, a 1e-7 parameter nudge fails the parameter assertion. No probe
  consumed an RNG before this change (the rank forward is `no_grad` at dropout 0), so none needed an
  isolated generator.
- **The measurement.** `learner_benchmark`'s `diag_skipped` config is a skipped update at the run's
  own flags (so the tripwire's `rank/*` still runs — the honest production saving). CPU `--tiny`
  exercises the path; the GPU number is `python3 -m agents.training.learner_benchmark run --device
  cuda` on an idle GPU (**UNVERIFIED** until that read lands).

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
