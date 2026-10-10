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
| `train_setup.py` | **the pre-loop half of `train()`** — `_align_opp_intent_labels`, `_resolve_fold_flags` (→ `FoldFlags`: which terms this call folds, plus the counterfactual buffer's one per-rollout poll) and `_train_probe_setup` (→ `ProbeSetup`: the once-per-call diagnostics and the gradient sampler). Both containers are `NamedTuple`s carrying the SAME names the fold uses, so `train()` unpacks them back into the locals the loop was written against. Also R1's `_micro_static` / `_micro_region` |
| `metrics_export.py` | **the ~400-line `self.logger.record` tail** — diagnostics, no gradient, one method per TB prefix group, each taking the accumulators this call filled |
| `rollout_probes.py` | `collect_rollouts` — the Rust collector's entry (`_collect_rust`; a learner without `_rust_collector` raises — the async collector and the Python collect are deleted) and `_winprob_start_metrics` — per-ROLLOUT work that is not part of the fold at all |
| `loop.py` | **THE LOOP IS OURS** (`gen3_owned_ppo_loop_v1`, [`design_own_ppo_loop.md`](../endstate/design_own_ppo_loop.md) stages 1-3): `OwnedLoop` — `learn()` written out as the declared `LOOP_PHASES` table (`setup → training_start → [collect → progress → dump → update]* → training_end → final_dump`), plus `_setup_learn`, `dump_logs`, `_update_current_progress_remaining` and `_update_info_buffer`, each vendored from sb3 / sb3-contrib operation for operation and hash-pinned at import (`UPSTREAM_SOURCE_SHA256`); `_setup_model` forces the OWNED rollout buffer and `_wrap_env` accepts only a `TrainerVecEnv`. CONTRACTS: the dump comes BEFORE the update (update k's `train/*` is stamped after rollout k+1 — every TB series in the archive rests on it); the `final_dump` writes the LAST update's pending scalars at `num_timesteps` (`gen3_final_update_dump_v1`, P3 — on a normal end two `train/*` points at the final step); `learn` > `collect` | `update` are the DECLARED HOOK POINTS (`agents/training/loop_hooks.py`, `gen3_declared_loop_hooks_v1`: K6's freeze guard and the compile sentinel register there, outermost first by `HOOK_OWNERS`, and the table freezes at training start), and the methods are still called through the attribute. Stage 3 (deletion pass U4) owned the rest of sb3's runtime beside it — `agents/training/rollout_buffer.py` (the buffer, sb3's `get()` permutation), `train_logger.py` (the logger), `loop_callbacks.py` (the callback protocol, events and step locals declared), `trainer_env.py` (the env base) — and deleted the `GEN3AI_PPO_LOOP=sb3_reference` seam with its lockstep test. Pinned by `owned_loop_test.py` (dump order, final dump, env refusal, protocol), `rollout_buffer_test.py` / `train_logger_test.py` (EXACT against sb3's while installed) and `loop_hooks_test.py` (the hook table) |
| `strict_load.py` | **THE STRICT CHECKPOINT LOAD** (`gen3_strict_checkpoint_load_v1`): the `StrictCheckpointLoad` mixin (`set_parameters` always strict — a missing or unexpected key raises, sb3's non-strict "SB3 < 1.7.0" retry is `StrictLoadError`; `OwnedLoop` inherits it) and `StrictMaskablePPO` (a plain `MaskablePPO` + the mixin — the class every READER of a checkpoint builds through `snapshot.load_checkpoint_strict`). Held closed by `src/strict_checkpoint_load_gate_test.py`. Its `load` never unpickles `NEVER_UNPICKLED` (the ride-along optimizers) and raises `CudaContextOnCpuLoad` when a CPU load initialises CUDA (`gen3_cpu_load_no_cuda_v1`) |
| `hparams.py` | every after-construction knob `train_rl_agent` sets (the belief/intent/cf coefficients, `grad_accum_steps`, …) with the rationale comment each carries, plus `_excluded_save_params` |
| `noise_scale.py` | the McCandlish gradient-noise-scale estimator + the rate-limited NSR advisor + `noise_ratio_sample`, the read seam `--adaptive-batch` steers by |
| `noise_scale_terms.py` | the PER-LOSS-TERM half of it — is the total reading the POLICY gradient's, or the auxiliary heads'? |
| `value_terms.py` | the win-prob BCE |
| `aux_terms.py` | the `belief_bank` delegate |
| `constants.py` | `_WIN_CONTESTED_TAU` · `_NOISE_SCALE_EMA_DECAY` |
| `learner_gates.py` | **K9 — the learner's in-loop GIGO gates** (`designs/training/learner_gates.md`): `behaviour_gate_mode` (K9(b) is Lane G's pre-loop probe — the ONE implementation; the python core's in-loop variant was deleted, U4) and the fail-closed non-finite checks — the buffer, the assembled loss per micro-batch (naming the term), the approx-KL, and `clip_grad_norm_checked` at every optimizer step. All raise before the optimizer moves anything |
| `device_batches.py` | **K8 — HOW THE MICRO-BATCHES REACH THE DEVICE** (`gen3_device_batches_v1`, `gen3_device_batch_mode_v1`, `--device-batch`): `install(buffer, mode=...)` at the top of `train()`, `uninstall` after the epoch loop; it serves the owned `rollout_buffer.RolloutBuffer` and builds its `RolloutSamples` (U4). `staged` (DEFAULT since 2026-10-01): the buffer's own `get()` is run unchanged (its `_get_samples` hands back the index slice), a prefetch thread gathers each slice on the host (`np.take` into a pinned block, 2 micro-batches ahead) and it is copied non-blocking on the COMPUTE stream (`gen3_staged_compute_stream_v1`: the old per-update side stream stranded +1.86 GiB of cache at N = 256 under `expandable_segments`, `learner_lifecycle.md` "The staged batch's stream") — ~2 micro-batches on the card. `resident` (K8.6): ONE device copy of the flattened arrays per update (made at the first micro-batch; +~1.1 GB of update peak at production shape). `host`: the buffer's own per-micro-batch path. Every mode draws the same permutation and serves BIT-IDENTICAL micro-batches (`instrumented_ppo_device_batches_test`, incl. a whole update per mode); the quiescent floor is unchanged. A CPU buffer keeps the host path |
| `intent_fold.py` | **K8 — the OPPONENT-INTENT fold's ENTRY POINT** (`intent_fold` → `IntentFoldOut`), called by `micro_step` (R1): it dispatches to `flat_intent_fold` when the extractor published `flat_intent_logits` and returns `None` (the block skipped — no term, no metric) otherwise; plus the STATIC metric primitives the fold is written with (`_rate`, `_Sink`'s `(value, weight)` pairs, `info_gain_nats_static`, `_scale`) — no host read, no boolean-mask indexing, no `bincount`, no value-dependent branch; a metric not emitted has weight 0 and value 0. `instrumented_ppo_intent_fold_test.py` pins the dispatch |
| `flat_intent_fold.py` | **X5 U4 — the FLAT opponent pointer's fold, THE opponent-intent loss**: one CE over the flat list at `flat_intent_targets`' column (OTHER labels supervised), `--intent-label-bot-weight` on the bot rows at the supervised-row-count denominator, the `flat_*` metrics pooled and per opponent class, `label_bot_frac`, and `opp_intent/other_label_rate` (F-X5-8). Static; `flat_intent_test.py` traces it `fullgraph=True` on the real stash shapes, `intent_label_bot_weight_test.py` pins the weight |
| `micro_step.py` | **K8 — the LEARNER MICRO-STEP, the compile REGION R1** (`gen3_learner_micro_step_v1`): `evaluate_actions` (functional masking) + fold steps 1-3a as ONE static-shape function → `MicroOut` (the loss, the probe terms + their noise groups + presence, every diagnostic as `(value, weight)`); `pack` bundles a micro-batch's diagnostics, verdict inputs and term values as ONE flat DEVICE tensor and `unpack` reads its host copy — the read itself is `host_reads`'. `MicroStatic` (resolved per call by `TrainSetup._micro_static`) is every flag and coefficient it branches on or multiplies by. `train()` folds the eager tail onto `MicroOut.loss` |
| `host_reads.py` | **T25 item 1 — THE UPDATE'S DEFERRED HOST READS** (`gen3_batched_host_reads_v1`): `HostReadQueue` holds every device→host read `train()` can defer (a micro-batch's `pack`, epoch 0's calibration columns, the noise scale's small-batch norm) with the callback that consumes it, and makes them in ONE transfer at the next optimizer step beside its gradient norm (`learner_gates.clip_grad_norm_checked(reads=...)`) — callbacks in push order, float32 only. A micro-batch is read at once only when something before its backward needs it (`ppo.train`'s `_read_now`: the KL early stop, the capacity probes, the per-term noise tagger, the grad-balance probe still waiting); the deferred read makes its own K9(c) verdicts (`ppo._micro_reader`). Every logged value bit-identical; `host_sync_guard_test.py` bounds the syncs (`designs/training/learner_gates.md` "The deferred reads") |
| `phase_hook.py` | **BENCHMARK-ONLY** segment marks inside `train()` (`gen3_learner_phase_hook_v1`): `PHASE_HOOK` is None in production, read ONCE per call via `current()`, every site a guarded one-liner; the names and what each books are in its docstring. The one consumer is `agents/training/learner_benchmark.py`, which installs a `cuda.synchronize`-bracketed segment timer (or drives `torch.profiler` windows off the `epoch_end` mark). Pinned by `learner_benchmark_test.py` (guarded, named, and byte-identical parameters with a hook installed) |


## The two source-pin rules

🚨 **A SOURCE-LEVEL PIN THAT SAYS "in `train()`" SHOULD READ `ppo.train_step_source()`** — `train()`
concatenated with the three setup methods and the seven export methods. The fold, its setup and its
export are ONE train step; which of the three modules a given line sits in is a decomposition
detail, and a pin that depends on it breaks on a move that changed nothing. Five test files read it
(`instrumented_ppo_winprob_critic_test`, `vf_coef_scale_readout_test`,
`instrumented_ppo_noise_scale_terms_test`, and the hub contract's own
docstring). **The fold's own ORDERING pins stay on `inspect.getsource(train)`**, where
straight-line source order is the thing being checked. The same rule holds for a MONKEYPATCH: a
test that stubs a diagnostic must patch the module that now owns the call
(`train_setup.advantage_density_metrics` / `train_setup.shared_trunk_parameters`; the third example,
`metrics_export.live_gauge_metrics`, was retired with the in-training scaffolding gauge, P11d), because patching `ppo` would silently stub nothing and the
byte-identity test would then compare two identical arms and pass. Per minibatch:

The upstream-drift hash check (`_verify_upstream_unchanged` + `_EXPECTED_UPSTREAM_TRAIN_HASH`)
stays in the HUB on purpose: `instrumented_ppo_test` patches that global on the module object it
imports, so moving it into a submodule would have left the patch reaching a different global than
the function reads — a test that still passes, for the wrong reason.

## The policy's GAE λ (`--policy-gae-lambda`, `gen3_policy_gae_lambda_v1`, config v123)

**`--policy-gae-lambda FLOAT`, default `0.80`** — the λ the rollout buffer's window GAE (`RolloutBuffer.compute_returns_and_advantage`) and the complete-game `store.game_gae`
uses to turn rewards and recorded values into the ADVANTAGES the clipped surrogate is trained on
(and the `returns` the scalar value loss regresses toward). Until 2026-09-26 it was a literal
`0.80` at both `main/train/model_build.py` sites (the fresh `InstrumentedMaskablePPO(gae_lambda=…)`
and the resume path's `model.gae_lambda = …`); both now read `args.policy_gae_lambda`, so the
default is byte-identical. Range `[0, 1]` (a parser error outside it).

- **It never touches the BCE target.** The critic's win-prob BCE target is the terminal outcome
  ([`critic_and_value_losses.md`](critic_and_value_losses.md)); a critic λ-return (`--win-prob-lambda`)
  existed until deletion pass L2 and is gone, so this is the only λ in the loop.
- **Recorded and inherited** — the `training_coef` provenance class: a `ModelVersion` field
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
| grad balance (`grad_balance_metrics`) | `grad/*` |
| effective rank (`rank_probe`) | `rank/{trunk,value_cls,policy,vf_feat}_*` — `vf_feat` measures `value_pooled` (128-wide; the extractor's whole value half since the version break's part 2 — before it, the 512-wide post-ReLU value projection, so the series breaks there) |
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

## Rollout collection — the Rust complete-game collector is the ONLY collector

> The Python collectors are DELETED (deletion pass U3): the per-step-barrier `SubprocVecEnv` path, the async-wave collector (`AsyncSubprocVecEnv`, `collect_rollouts_async`, and the async-rollout flag) and `OwnedLoop._collect_python`. `InstrumentedMaskablePPO.collect_rollouts` / `OwnedLoop` collect only through the Rust collector (`rust_rollout/`; a `RolloutProbes.collect_rollouts` with no `_rust_collector` raises). Design and hazards: [`rust_collector.md`](rust_collector.md). The async design record, with its dated FPS table (+14% at `--n-envs 64`, heuristic opponents) is history: `designs/ai_v5/design_async_rollout.md`.

## From the training leaf (moved 2026-10-10)

> These sections headed `src/agents/training/CLAUDE.md` until its 2026-10-10 cleanup; moved here as they
> stood (minus statements verified FALSE). Where an earlier section of this doc says the same in more
> detail, both are current; fix both in the same pass.

### The PPO step (`instrumented_ppo/`) — and the FOLD ORDER contract

`instrumented_ppo` is a PACKAGE whose `__init__.py` is a pure re-export hub; the module map and the
two source-pin rules (a pin that says "in `train()`" should read `ppo.train_step_source()`; a
monkeypatch follows the SYMBOL) are in
[`designs/training/ppo_step.md`](ppo_step.md). **The contract below stays
here, because it is the thing a fold edit must not get wrong.**

**THE FOLD SEQUENCE is TWO straight lines** (K8, `gen3_learner_micro_step_v1`): steps 1 to 3a below
are the body of ONE function, `instrumented_ppo/micro_step.micro_step` — the compile REGION R1 under
`--compile-trainer` (`fullgraph=True`), eager otherwise — and `train()` folds the steps after 3a onto
R1's loss as the DECLARED EAGER TAIL, in order. Each part is straight-line source and
`instrumented_ppo_hub_contract_test.py` pins both orders. 🚨 **R1 is a static-shape program**: no
host read (`.item()`, `float(t)`, `bool(t)`), no boolean-mask indexing / `nonzero` / `bincount`, no
Python branch on a tensor value, no numpy — a diagnostic is a `(value, weight)` pair of 0-d tensors
(weight 1.0 exactly where the old fold appended to its list), and `train()` reads ALL of a micro-batch's
diagnostics as ONE packed device tensor (`micro_step.pack`). 🚨 **That read is DEFERRED to the next optimizer step's
one transfer** (`instrumented_ppo/host_reads.py`, `gen3_batched_host_reads_v1`, T25 item 1) unless something
before the backward needs it, and the K9(c) loss / KL verdicts ride it; a new per-micro-batch host read in
`train()` (an `.item()`, a `float(t)`, an `as_numpy`) goes through the queue too, or it is a full GPU-queue drain
×480 per production update — `host_sync_guard_test.py` FAILS on one. The belief losses' static twins are
`belief_bank_static.py`; the opponent-intent block's is `instrumented_ppo/intent_fold.py`, which dispatches to
X5's flat-pointer fold (`flat_intent_fold.py`; the blob α / β fold was deleted at the X5 version break, v144) — the
legacy `belief_bank` functions stay as the REFERENCE they are pinned equal to (float64 to 1e-12). A new term on the production surface belongs in R1, written to these rules; anything else
joins the tail in contract order. What moved out of `train()` before K8 is everything AROUND the
sequence: the pre-loop setup (`train_setup`, incl. R1's static flags `_micro_static`), the metrics
export (`metrics_export`) and the per-rollout probes (`rollout_probes`).

🚨 **THE LOOP AROUND `train()` IS OURS TOO** (`gen3_owned_ppo_loop_v1`, `src/agents/training/instrumented_ppo/loop.py`;
[`designs/endstate/design_own_ppo_loop.md`](../endstate/design_own_ppo_loop.md)): `learn()`
is the declared `LOOP_PHASES` table, and `_setup_learn` / `dump_logs` are
vendored from sb3 operation for operation (hash-pinned). **PPO stage 3 (deletion pass U4) took the rest of
sb3's RUNTIME off it — each owned module holds sb3's arithmetic for our one layout and REFUSES any other:**
the rollout buffer (`rollout_buffer.py`, `gen3_owned_rollout_buffer_v1`: host numpy, sb3's `get()`
permutation — the K9 golden is its bar), the logger (`train_logger.py`, `gen3_owned_logger_v1`: the same
tags, steps, stdout table and `name_to_value` bus; its writers pair values with exclusions KEY FOR KEY —
`paired()` raises on a key only one side holds, where sb3's `zip(strict=True)` checked lengths only), the callback protocol (`loop_callbacks.py`,
`gen3_owned_callbacks_v1`: declared events, and the per-step locals declared in `STEP_LOCALS` — an
undeclared key, an sb3 callback or a bare function is refused) and the env base (`trainer_env.py`: the
learner's env must be a `TrainerVecEnv` — `RustVecEnv`, or `testkit.ToyVecEnv` in a test). The
`GEN3AI_PPO_LOOP=sb3_reference` seam and its lockstep test against upstream are gone with it. Four things
an edit must not break: the **dump stays BEFORE the update** (update k's `train/*` is stamped after
rollout k+1 — the archive's TB convention); the **final dump** (`gen3_final_update_dump_v1`, P3) writes the
LAST update's pending scalars at `learn()`'s end, at the current step (so on a normal end the `train/*`
tags carry two points at the final step — update k-1's, then update k's), and the abort / graceful-restart
path (`main/train/lifecycle.py`) runs the loop's own `dump_logs()` before it saves — at a SAFE POINT only
(`main/train/deferred_abort.py`, `gen3_deferred_abort_v1`: a stop signal just records the request, and
`GracefulRestartCallback` runs it at every loop event, never inside an update or a dump; a new loop event
or a new place that dumps must not become a place the abort can run mid-update) — so a test reads what
an update logged from a recorded dump (`testkit.record_dumps`), never from `name_to_value` after `learn()`;
**`learn` > `collect` | `update` are the DECLARED HOOK POINTS** (`agents/training/loop_hooks.py`,
`gen3_declared_loop_hooks_v1`: K6's freeze guard and the compile sentinel REGISTER there, outermost first
by `HOOK_OWNERS`; the table freezes at training start and a late, duplicate or undeclared hook is
FATAL_CONFIG — never reassign a learner's bound method to hook it); and the **`step` event fires before
the buffer row is written**, with the declared step locals (`infos`, `dones`). `owned_loop_test.py` pins
the dump order, the final dump, the env refusal and the protocol; `rollout_buffer_test.py` /
`train_logger_test.py` hold the buffer and the logger EXACT against sb3's while sb3 is installed (stage 4
drops it — `design_own_ppo_loop.md` §3.4). 🚨 **REGIME BOUNDARY
(`gen3_eval_dump_isolation_v1`, 2026-10-01):** an eval cycle's mid-rollout `logger.dump(step)` used to
CLEAR the previous update's `train/*`, so the KL→LR controller (and RankTripwire) skipped one reading per eval cycle — 5% of N0's updates. Both eval callbacks'
`_collect_pending` now run under `logger_scope.isolated_dump`, so the cycle dumps only its own scalars.
A live-controller run from that commit onward is not comparable with an earlier one on its LR / dose
trajectory (`designs/training/step_size_and_batch.md`). A new callback that dumps the logger
mid-rollout MUST use the same decorator. **Seeding is OURS too** (`gen3_owned_seeding_v1`, `OwnedLoop.set_random_seed`):
sb3's draws in sb3's order and NO cuDNN flag — sb3 set the process-wide `cudnn.deterministic=True` on
every CUDA construction and load (a nominal regime boundary: 0 cuDNN kernels run in a production update).
🚨 **No global RNG is SEEDED after the freeze** (`gen3_no_global_reseed_v1`, `global_rng_guard.py`): `LearnerFreeze` arms a guard
on `random.seed` / `numpy.random.seed` / `torch.manual_seed` and kin, so a seed is `GlobalReseedError` (FATAL_CONFIG) naming its
site. An opponent / reader load (`InferenceMaskablePPO`) never seeds and builds inside `isolated_global_rng()`. Until 2026-10-02
every load re-seeded to the snapshot's seed, which replayed the minibatch permutation. A stream that must repeat owns a generator. Static twin: `src/global_rng_seed_gate_test.py`. Detail:
`designs/training/learner_lifecycle.md` "No global reseed after the freeze".

**K9 — the learner's GIGO gates** ([`designs/training/learner_gates.md`](learner_gates.md)).
🚨 **`learner_golden_test.py` pins what ONE update computes** — exact post-update parameter bytes and
every loss, per torch build — so ANY change to the fold, a term, a coefficient default or the step
fails the routine gate until someone re-records deliberately: `python -m agents.training.learner_golden
record --reason "..."` under EVERY interpreter with an entry (never a routine step). **Its ONE entry is X5's
fixed-mass surface** (the production model since the X5 version break, v144): the former `arms.fixed_mass` entry
MOVED VERBATIM to the default slot, then was re-recorded ONCE at the end of the break (the seed-18 buffer rebuilt for
the 2845-dim observation, the NAME-KEYED perturbation, fp64 reference, K9(b) read and coverage re-taken;
`learner_golden_fixed_mass_test.py` holds the X5-specific checks); the blob entry and buffer are deleted
(`learner_gates.md`). Every non-finite
loss / gradient / buffer value / KL is `main.exit_codes.NonFiniteLearnerError` (tagged `[Learner]
FATAL`; exit 4, the launcher does NOT restart) BEFORE the optimizer moves anything
(`instrumented_ppo/learner_gates.py`) — never a `nan_to_num`, a NaN-mask on a trained
quantity or a skipped step; a new term must reach the assembled loss (or carry its own check), and a
`where(isfinite)` on a label is a NaN hide unless it means `-inf` (use `isneginf`).

🚨 **K6 — THE LEARNER FREEZES at the first rollout of `learn()`** (`learner_lifecycle.py`, [`designs/training/learner_lifecycle.md`](learner_lifecycle.md)): after it, a NEW optimizer, `nn.Parameter`, module, buffer or optimizer-state entry anywhere in the learner's graph is `LazyAcquisitionError` (`[LearnerLifecycle] FATAL`, exit 3, not restarted) naming the object and its construction site. Build anything the steady state uses at STARTUP — in `_build` / `_setup_model` / an `__init__`, or a function marked `@lifecycle_decl.startup_builder` that the startup path runs — never on the first update. Adam/AdamW state is declared at startup (`declare_optimizer_state`, bit-identical to torch's lazy init). Every run logs a CUDA memory LEDGER by startup step (`cuda_ledger.py`, `<run_dir>/cuda_ledger.json`) and per-update peaks (`lifecycle/cuda_*_peak_*`, `lifecycle/device_batch_mib`); a rust-core pool refresh is a DECLARED LOAD (pool on the CPU; `checked_slot_load` refuses a load that allocates). On CUDA the MEMORY half rides the same attach (`CudaMemoryWatch` over `cuda_memory_trend.py`): a sample after every rollout and update, the OOM projection logged at every window (`[CudaMemTrend]`, TB `lifecycle/cuda_*`), and only a SUSTAINED leak projecting an OOM inside 25 updates stops — `CudaMemoryLeakError`, checkpoint (`final_model_exception.zip`) then exit 6 (`FATAL_CUDA_LEAK`), which the launcher restarts from that checkpoint at most twice per session; a step-up or fragmentation never does. The STATIC twin is `src/learner_lifecycle_gate_test.py` (routine, EMPTY allowlist): a
construction in a training-step path outside a `@startup_builder` / `__init__` / `_build` / `_setup_model`
fails the gate before it can fail a run. 🚨 **A CUDA STREAM / GRAPH / GRAPH POOL is a startup acquisition
too** (the gate's `cuda_resource` kind, `gen3_staged_compute_stream_v1`), and for it a bare `__init__` is
NOT an exemption — only `@startup_builder` / `_setup_model` / `_build`: the caching allocator keeps a
cache per stream, so the staged batch's per-update side stream (built in a helper's `__init__`) stranded
+1.86 GiB of reserved over 31 updates of sizing arm B, invisible to every segment counter.

**`train()` carries BENCHMARK-ONLY phase marks** (`gen3_learner_phase_hook_v1`): ~14 lines of
`if _ph is not None: _ph("<phase>")`, `_ph` read ONCE per call from `instrumented_ppo/phase_hook.py`
(None in production). A new mark must use exactly that guarded one-line form and a name in
`phase_hook.PHASES`; `learner_benchmark_test.py` pins the guard, the names, and that installing a
hook changes no parameter. The one consumer is `learner_benchmark.py` (where an update's wall time
goes — `designs/ops/testing.md` → Benchmarks), and its workers measure the LEARN LOOP's update only: the
trainer's own startup `train()` (the CUDA fit check's dry update) runs production's code (`learn_loop_only`).

Per minibatch (1 to 3a inside R1):

1. the upstream PPO loss (`policy_grad_coef·policy_loss + ent_coef·entropy + vf_term` — `--policy-grad-coef`
   scales ONLY the clipped surrogate, never entropy/value/aux; at the 1.0 default the UNSCALED
   `policy_loss` tensor is used, byte-identical to upstream, and 0.0 removes the policy-gradient
   term alone. Training-only, the `training_coef` provenance
   class: recorded, `_resolve`-inherited on a flagless resume, never gated)
2. the belief bank — the hidden-team set BCE (X5), opponent intent (the flat pointer's CE), move / spread /
   nature-EV / HP-type / item belief, move-latent
3. (3a) the win-prob BCE — the last R1 term
4. (retired — the value-dist HL-Gauss CE was deleted with the dist head; the numbering below is unchanged)
5. (retired — the distill family was deleted with distillation, config v133; the numbering is unchanged)
6. (retired — search-teacher AWR and OPD were deleted with the search teacher, config v133)
7. (retired — the TD-consistency auxiliary was deleted, P11c; the numbering is unchanged)
8. (retired — the counterfactual block (cf-winprob, cf-evidential, cf-twin, cf-shadow, q-winprob) was deleted with the cf training half, config v134; the numbering is unchanged)

**No flag combination reorders these.** Each term is guarded by its own `if <x>_on:`; a term that
is off contributes nothing and moves no one. **A tail fold that runs its OWN extractor
forward CLOBBERS the minibatch's stashes** (`last_win_prob_logits`, `last_spread_belief`, …)
that steps 2-4 read. Moving a stash-reading fold below such a fold does not crash — it silently scores
the wrong states. `instrumented_ppo_hub_contract_test.py` pins R1's order and that the R1 call
precedes every eager-tail fold by reading the source, along with the mixin base list (a dropped mixin removes a
whole family of loss terms without breaking an import) and `MaskablePPO` staying LAST in the MRO
(or `_excluded_save_params`'s `super()` stops reaching upstream and checkpoints start pickling a
`threading.Lock`). It also walks the package's own import graph TRANSITIVELY from the hub, so a
module reached only through `train_setup` still counts as reachable — requiring a direct edge from
`__init__`/`ppo` would forbid a decomposition rather than check one.
