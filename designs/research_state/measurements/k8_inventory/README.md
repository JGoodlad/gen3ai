# K8 compile inventory — the WHOLE PPO update under dynamo (2026-09-30)

**Verdict.** Today only the feature extractor is compiled. On torch 2.8 (the target), **60.3% of the
production update's wall time is kernel time inside compiled graphs.** The rest is 10.9% eager
kernels, 2.2% copies, and ~26.6% GPU-idle host time. Most of the gap is ONE thing: `train()`'s
inline loss fold runs eager, launching ~1,770 eager kernels and ~890 eager backward kernels per
micro-batch.

Dynamo cannot get into the fold today. `train()` is lost at its first statement on both torches,
and handing the whole update to dynamo fragments it into **514 graphs (2.5.1) / 646 graphs (2.8)**
across ~200 break sites. So K8's regions require a FUNCTIONAL rewrite of the micro-step, not a
`torch.compile` around `train()`.

The extractor alone, the K8 test (`fullgraph=True`):
- **2.8:** compiles in train/grad, train/no-grad and eval/no-grad.
- **2.5.1:** REFUSES in every mode at `forward_guard.py:57`, a break `torch._dynamo.explain` cannot
  see.
- **Both torches:** the trainer-hosted rollout forward also broke at the `nn.MultiheadAttention`
  fast path (eval/no-grad). The cause is UNVERIFIED, since a standalone compile does not reproduce
  it.

Tag: MEASURED (one checkpoint, one pinned buffer; CPU traces on both torches; GPU time on 2.8 under
CPU contention — see "Caveats"). Every number below comes from the result files named in
"Provenance", and the tables in `tables.md` are generated from them by the tool.

## What was measured, and how

`python -m main.compile_inventory` (`src/main/compile_inventory/`) runs the REAL trainer in a
subprocess. It uses arm C's final checkpoint, `ai_v14_02_lbat_ctrl/final_model.zip` (sha256
`06efae2c44d6…`), with C's own recorded flags, and restores the learner benchmark's pinned
98,304-row rollout buffer
(`~/gen3ai_archive/learner_bench/20260928_135948_cuda/rollout_buffer.pkl`, collected by that
checkpoint). The first `train()` is then replaced by one of two stages:

- **trace** (CPU, `aot_eager`, never timed). Two VIEWS of the same update:
  - `today` compiles only `features_extractor.forward`, as `--compile-trainer` does.
  - `whole_step` hands `train()` itself to dynamo, so every frame it reaches is compiled or
    reported: the extractor, heads, masking, every loss-term helper, the optimizer, the buffer
    and logging.

  Each view runs four calls:
  1. update 1, the process's first update, with every diagnostic probe;
  2. update 2, a `--diagnostics-every` SKIPPED update;
  3. a rollout-shaped EVAL/no-grad forward at n_envs = 48 rows;
  4. update 3, skipped again and profiled.

  The capture reads every graph, break, skipped frame, eager fallback and recompile from dynamo's
  own log artifacts AND from each graph's `compile_subgraph_reason` (`capture.py` says why
  `explain` alone is not enough). Real rows: 4 steps × 48 envs = 192 rows = 3 micro-batches of
  64, accum 2, 2 epochs. The graph structure does not depend on the batch size. The torch 2.8
  `today` view was run at micro 512 (before the memory fix below).
- **time** (CUDA, the PRODUCTION compiled learner). The real `--compile-trainer` compile and its
  TF32 parity gate run first, at matmul `high` (TF32), on all 98,304 rows (micro 2048 × 48,
  accum 32, 10 epochs). Then:
  - one warm-up epoch;
  - one full update bracketed with `torch.cuda.synchronize()` per phase (`PhaseTimer`);
  - one full un-bracketed update (`train/train_ms`);
  - one epoch under `torch.profiler` (CPU + CUDA, no stacks), with every `train()` phase as a
    `record_function` range.

  Each kernel is then attributed to compiled or eager by the marker enclosing its launch
  (`CompiledFunction`, `CompiledFunctionBackward`, a `triton_*` op), and to the phase whose range
  encloses the launch (`trace_classify.py`). The production prewarm and lock were skipped so a
  unit fits the 20-min GPU budget. The warm-up compiles the one signature the timed updates use.

All timed updates are `diag_skipped`. C predates the diagnostics cadence and so inherits
`diagnostics_every 1`. Fresh runs default to 10, so 9 of 10 updates are of this kind.

## (a) Graphs and breaks, per area, both torches

Notation per cell: graphs / ops in graphs / break sites (events) / recompiles. The area is taken
from the SOURCE FILE of the graph root or break site, which is the same on both torches (`tables.md`
has the full cells and every site).

| area | 2.5.1 `today` | 2.8 `today` | 2.5.1 `whole_step` | 2.8 `whole_step` |
|---|---|---|---|---|
| extractor | 4 / 33,317 / 2 (5) / 5 | 3 / 24,933 / 1 (1) / 2 | 10 / 41,539 / 2 (15) / 8 | 1 / 8,311 / 0 / 0 (the rest is inlined into `evaluate_actions`) |
| heads (`policy.py`, pointer head, sb3 policy glue) | — | — | 9 / 144 / 0 / 4 | 7 / 25,090 / 0 / 1 (`evaluate_actions` = ONE 8,342-op graph: extractor + MLPs + pointer head + distribution construction) |
| masking + distribution (sb3) | — | — | 8 / 19 / 3 (19) / 7 | 8 / 19 / 1 (17) / 7 |
| `train()` fold (inline) | — | — | 0 graphs: lost at line 176 | 0 graphs: lost at line 176 |
| loss terms (belief bank, win-prob, intent, labels) | — | — | 335 / 1,117 / 123 (492) / 330 | 413 / 1,266 / 104 (357) / 234 |
| diagnostics (rank, grad balance, calibration, scaffolding, metrics) | — | — | 117 / 4,262 / 75 (154) / 64 | 141 / 2,423 / 71 (154) / 48 |
| optimizer (AdamW, foreach) | — | — | 3 / 15,120 / 1 / 0 | 3 / 15,120 / 0 / 2 |
| rollout buffer (`_get_samples`, `swap_and_flatten`) | — | — | 9 / 93 / 0 / 7 | 9 / 93 / 0 / 7 |
| **total** | **4 graphs / 2 break sites / 5 recompiles** | **3 / 1 / 2** | **514 / 206 / 420** (12 skipped frames, 6 eager fallbacks) | **646 / 178 / 299** (13 skipped frames, 4 eager fallbacks) |

The constructs, by area (dynamo's own reason in brackets):

- **`fullgraph=True`, the K8 test itself** (`python -m main.compile_inventory fullgraph`: the production
  surface, the committed real-obs rows, CPU; `fullgraph_torch*.json`). Every extractor candidate is
  tested: `forward`, `Module.__call__`, `extract_features`, in train/grad, train/no-grad and
  eval/no-grad.
  - **2.5.1 REFUSES every one** at `forward_guard.py:57` (`WeakKeyDictionary.get`).
  - **2.8 compiles every one.**
  - The rollout call `policy(obs, action_masks=…)` refuses on 2.8 only at sb3's
    `apply_masking:69`; on 2.5.1 it refuses at the guard first.
- **Extractor, production scope.**
  - **Both torches:** `encoders.py:287 move_self_attn`, an `nn.MultiheadAttention` that takes its
    native fast path in eval/no-grad ["`aten._native_multi_head_attention` … does not support
    fake tensors"]. Rollout/eval only; the train graph is unaffected. It shows in the
    TRAINER-hosted eval forward (C's checkpoint, real rows) and NOT in the standalone
    `fullgraph` check above. **UNVERIFIED** why; fix 5b removes it either way.
  - **2.5.1 only:** `forward_guard.py:57` [`WeakKeyDictionary.get` inline in skipfiles], 4 events
    in update 1. It sits at the frame's first instruction, so no graph ends in it and `explain`
    reports **0 breaks**. That is why "1 graph / 0 breaks on both torches" (`compile_flags.md`,
    Lane K1, and `ecd2be00`'s message) holds on 2.8 only; on 2.5.1, `fullgraph=True` would refuse
    here.
  - Recompiles are the `grad_mode` guard (train/grad, train/no-grad for the rank probe,
    eval/no-grad). On 2.5.1 there is also a batch-size recompile of `_forward_unguarded`
    (B=48 rollout, then B=64 probe), because the guard break makes that function its own frame
    shared by two callers.
  - The ObservationDebugger break is REMOVED (owner 2026-09-30, `ecd2be00`): this tree has no
    debugger API, and the `rate_limiter` break is gone.
- **`train()` itself (both torches).** The first break is `ppo.py:176 _t_train0 = time.perf_counter()`
  [skipped function]. The resume frame then fails to convert ["WON'T CONVERT
  torch_dynamo_resume_in_train_at_176"] inside `opp_intent_labels.py:139
  align_labels_to_predictions`, which does numpy label alignment, so the WHOLE fold runs eager.
  The static scan of the minibatch loop (`attribution.scan_fold_host_syncs`, AST, not dynamo)
  lists what a region would have to hoist: 11 `.item()`, 49 `float()`/`int()`/`bool()` host reads,
  7 Python branches on a value, 3 host-copy lines (`.cpu().numpy()`) and 25 inner Python loops.
  The committed result predates the scan's de-duplication, so `tables.md` lists 6 host copies,
  each `.cpu().numpy()` counted twice.
- **Masking.** `sb3_contrib …/distributions.py:69 apply_masking` calls `self.__dict__.pop("probs")`
  and re-runs `torch.distributions.Categorical.__init__` ["call_method … `__dict__` pop" (2.5.1) /
  "Unsupported method call" (2.8)]. That is 17–19 events per update view; on 2.5.1 `entropy` and
  `sample` also break.
- **Loss terms.** The constructs are:
  - `.item()` used for metrics: `belief_bank.py:85/125/126`, `value_terms.py:236–256`, and
    `opp_intent.py:313–364 _alpha_subset_metrics`, which also recompiles on `len(out) == N` as its
    metric dict grows.
  - Dynamic-shape ops: `aten.bincount` in `opp_intent.py:233 info_gain_nats` and `:516
    intent_losses`; `aten.nonzero` in `signal_metrics`.
  - Python container mutation inside traced code: `list.append` in `belief_bank.py:202`, and
    `dict.update` in `opp_intent.py:516`.
  - `bool(tensor)` vocabulary checks in `belief_bank.py:184 _vocab_check`.
  - A generator: `belief_bank.compute`, skipped as a frame.
- **Diagnostics.**
  - `rank_metrics.py:44` (data-dependent `_local_scalar_dense`) plus a numpy float64 SVD.
  - `grad_balance` (`torch.autograd.grad` is a dynamo-skipped function).
  - `calibration` and `scaffolding` (numpy, loops over bins).
  - `metrics_export` (loops over TB tags).
- **Optimizer.** AdamW's foreach step traces cleanly into one graph per frame (5,040 ops). It sits
  between torch's OWN deliberate `graph_break()` calls in `Optimizer._use_grad`
  (`torch/optim/optimizer.py:90/93` on 2.5.1, `:80/83` on 2.8). The `len(self.state) != 252`
  recompiles are a harness artifact (the tool restores the checkpoint's smaller optimizer state
  before each call). Their production analogue is the first step after a load.

## (b) + (d) What runs compiled and what runs eager — GPU time, torch 2.8, TF32

The full update (unbracketed `train/train_ms`) takes **53.23 s**, and 57.62 s with a sync per
phase. The profiled epoch, scaled ×10 epochs:

| | s / update | share of the 53.23 s update |
|---|---|---|
| kernels inside compiled graphs | 32.1 | **60.3%** |
| eager kernels | 5.8 | 10.9% |
| memcpy / memset | 1.2 | 2.2% |
| GPU idle (host-bound: Python, launches, numpy, host reads) | ~14.2 | ~26.6% |

Per phase. "Bracketed" is the synchronised segment wall over the full update; the kernel columns
come from the profiled epoch, so they are per EPOCH:

| phase | bracketed s (share) | compiled kernel ms / epoch | eager kernel ms / epoch | eager kernels / epoch | compiled share of kernel time |
|---|---|---|---|---|---|
| backward | 25.93 (45.0%) | 2,192 | 248 | 42,708 (~890 per micro) | 89.8% |
| forward (`evaluate_actions`) | 11.73 (20.4%) | 1,008 | 24 | 3,696 (77 per micro: heads, masking) | 97.7% |
| probes | 8.89 (15.4%) | 9 | 16 | 253 | 37.8% — host-bound, see fix 3 |
| loss (the fold) | 7.84 (13.6%) | 0 | 262 | **84,924 (~1,770 per micro)** | **0%** |
| batch (`rollout_buffer.get`) | 2.95 (5.1%) | 0 | 0 | 0 (66 ms of H2D copies) | — |
| logging, kl, optim, epoch_end, setup | ≤ 0.13 each | — | — | — | — |

Over a whole update, **84.7% of kernel time** is in compiled graphs. The loss fold and its
backward are where the eager kernels, the launches and the idle GPU are. They launch
~1,770 + ~890 tiny kernels per micro-batch (262 + 248 ms of kernel time per epoch, but 7.84 s
bracketed for the fold).

The bracketed update makes 66,784 host scalar reads (0.54 s blocked), about 139 per micro-batch.

On CPU, counted as top-level ATen ops in the profiled steady update, compiled graphs run 49.8%
(2.5.1 `today`) / 70.2% (2.8 `today`, micro 512) of ops, and 53.0% / 67.1% in the `whole_step`
view. Op counts are a proxy only; the GPU time is the measurement.

**torch 2.5.1, the same stage on a QUIET box (load ~8, HEAD `8925e87b`):**
- bracketed full update 53.80 s;
- compiled kernels 32.3 s per update = **83.6% of kernel time, 60.0% of the bracketed update**
  (2.8: 55.7% of its bracketed 57.62 s, at load 75–95);
- the fold's eager kernels are identical (84,924 per epoch), the eager backward is 44,340;
- probes take 4.92 s here against 8.89 s under contention, so the rank probe's host SVD scales with
  load.

The two torches are within noise on kernel time. Their bracketed totals differ by the host load,
not the torch.

## (c) Recompiles

- **Production scope (`today`).** 2 recompiles on 2.8 and 5 on 2.5.1, all named above. All are
  first-time signatures that the K6 prewarm already declares, except the 2.5.1 batch-size
  recompile, which goes away with fix 5a.
- **`whole_step`.** 299 recompiles (2.8) and 420 (2.5.1), almost all in loss-term RESUME frames:
  - "Cache line invalidated because `L['___stack1']` got deallocated" (resume frames keyed on
    stack objects);
  - `len(out) == N` (metric dicts growing);
  - numpy dtype/rank guards in `swap_and_flatten`;
  - `masks is None` in `apply_masking`.

  The train→eval switch adds 7–9 recompiles. 2–4 frames hit the recompile limit (8) and now run
  EAGER. Steady state (update 3) still recompiles 13–15 times.

  This is what "compile the update as it stands" would mean, and it is why the regions are
  declared rather than discovered.

## Memory — the 2026-09-30 OOM incident, and RSS per stage

Three host OOM kills on 2026-09-30 (15:33, 16:44, 16:54; 74–82 GB anonymous RSS each) tore down the
tmux scope and every session in it. The cause was **this tool's CPU
`torch.profiler(with_stack=True)`**, exported and then `json.load`ed inside the worker.

Under a 12 GB no-swap scope (`systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0`),
the same unit (torch 2.8, `whole_step`, micro 64) held 1.7–1.9 GB through the whole dynamo trace,
then went **4.6 → 11.05 GB in < 9 s inside the profiler block**. The scope's OOM killer took only
the worker (kernel log 17:05:30, pid 249863).

The tool now does four things. A stack profile REFUSES unless `--allow-stack-profile` is given AND
the process's cgroup chain has a finite `memory.max` (`memcap.py`). Classification runs in a
separate step after the worker exits. The trace JSON is read by streaming. And every unit runs
capped.

Peak RSS per stage (MB) of the runs this readout uses (`rss_peak_mb_by_stage`):

| stage | 2.5.1 trace (today + whole step) | 2.8 trace (whole step) | 2.8 time (CUDA, full buffer) | 2.5.1 time |
|---|---|---|---|---|
| startup → first update | 873 | 960 | 9,149 (the 98,304-row buffer + the compile; finer stages were added after this unit) | — |
| `today` updates 1–3 / rollout | 1,192 / 1,186 / 1,233 | — | — | — |
| `today` profiled update + export (no stacks) | 2,365 / 2,503 | — | — | — |
| `whole_step` update 1 (the whole trace, every probe's backward) | 2,591 | 1,711 | — | — |
| `whole_step` update 2 / rollout / update 3 | 2,612 / 2,666 / 2,666 | 1,848 / 1,922 / 1,921 | — | — |
| `whole_step` profiled update + export (no stacks) | 3,300 / 3,300 | 2,464 / 2,464 | — | — |
| time: warm-up / bracketed update / profiled epoch | — | — | — | 6,241 / 6,719 / 8,035 |
| classify (separate process, streaming) | 812 | 695 | 1,173 | 1,553 |
| the capped `with_stack` repro (2.8, `whole_step`) | — | **> 11,051: OOM-killed in the profiler block** | — | — |

## Fix plan, ranked by measured gain

Classes: **H** trivially hoistable (guard, logging, `.item()`) · **F** functional rewrite ·
**D** data-dependent control flow (made static or split into a declared region) · **U** unsupported
op. Gains are from the 2.8 TF32 phase table against the 53.2 s update. They are ESTIMATES, and
they are the prediction K8's acceptance read will test.

| # | item | class | effort | evidence | expected saving |
|---|---|---|---|---|---|
| 1 | **The micro-step as a function, compiled as ONE region (R1).** `micro_step(policy, batch, flags) -> (loss, terms, metrics)`: the policy forward plus the whole fold, with the flags resolved before the region (static) and every host read hoisted to ONE batched read after it. The fold order contract is preserved by construction, and K9's learner golden is the parity gate. | **F** | L (the fold is ~1,250 lines and its contract is source order) | `train()` lost at line 176; the fold has 84,924 eager kernels per epoch; the scan's 11 `.item()` + 49 host reads + 7 value branches | the loss phase 7.8 s → ~1–2 s, plus the eager backward (2.5 s of kernels and ~430k launches per update) compiled: **≈ −8 to −10 s (15–19%)** |
| 1a | masking without `torch.distributions`: masked `log_softmax`, `log_prob`, entropy as plain tensor ops (inside R1) | **F** | S | `apply_masking`'s `__dict__.pop` + re-init: 17–19 events per update view | inside #1 (77 eager forward kernels per micro) |
| 1b | loss-term metrics returned as tensors, never `.item()`; no list/dict mutation inside the traced code | **H** | S–M | belief bank 58 sites, `value_terms` 16, intent 53 | inside #1 |
| 1c | `bincount`/`nonzero` replaced by fixed-size `scatter_add`/one-hot sums (`info_gain_nats`, `intent_losses`, `signal_metrics`) | **D** → static | S | "Dynamic shape operator" | inside #1 |
| 1d | `_vocab_check`'s `bool(tensor)` asserts and `time.perf_counter` moved out of the region (startup or an eager check after it) | **H** | XS | `belief_bank.py:184`, `ppo.py:176` | inside #1 |
| 2 | **Optimizer region (R3)**: `clip_grad_norm_` (norm kept as a tensor) + AdamW step + `zero_grad`; state initialised at startup | **H** (torch's own break is the boundary) | S | 1 graph per frame, 15,120 ops, traces cleanly | ~0 s: optim is 0.04 s per update with accum 32. It is DECLARED for the lifecycle (0 lazy compiles), not for speed |
| 3 | **Rank probe on the device**: `torch.linalg.svdvals` in fp32 on the GPU instead of a numpy float64 SVD, or on K2's cadence instead of every update | **H** (diagnostic) | XS | one probes segment = 3,434 ms per update, of which ~50 ms is torch ops; bracketed probes 8.89 s (15.4%) | **≈ −3 s** (load-dependent; UNVERIFIED how much of the bracketed 8.9 s it is beyond the 3.4 s) |
| 4 | **Device-resident batch**: index the GPU copy instead of numpy fancy indexing + H2D per micro | not dynamo (a declared eager boundary) | M (Lane G's collector is the natural owner) | batch 2.95 s (5.1%), 1.53 s of host ops per epoch | ≈ −2.5 s |
| 5a | extractor `forward_guard` lookup moved outside the traced region (resolved by the caller), or 2.8 only | **H** | XS | 2.5.1 break invisible to `explain`; the B=48/64 recompile | 0 s; required for `fullgraph=True` on 2.5.1 |
| 5b | MHA fast path off inside the eval region (`torch.backends.mha.set_fastpath_enabled(False)`), or attention through `F.scaled_dot_product_attention` | **U** | S | eval/no-grad break on both torches | 0 s on the learner; required for a `fullgraph=True` rollout/eval region (T2) |
| 6 | win-prob start metrics' eager extractor forward (`rollout_probes._winprob_start_metrics`, `type(fe).forward`): DECLARED eager, or padded to a fixed bucket through the eval region | **D** (variable row count) | XS–S | 11,820 eager host ops per update in `logging` | ≈ 0.1 s |
| 7 | diagnostics (grad balance's `autograd.grad`, the per-term noise probe, calibration, scaffolding, liveness, metrics export): stay EAGER, declared, on K2's cadence | — | 0 | `autograd.grad` is dynamo-skipped; numpy loops | — |
| 8 | label alignment (numpy, once per update) and the buffer's `swap_and_flatten`: stay eager, outside the regions | — | 0 | "WON'T CONVERT" at `align_labels_to_predictions` | — |

Sum of #1 + #3 + #4: 53.2 s → **≈ 38–41 s (−23 to −28%)**, with #1 alone ≈ 43–45 s.

## Proposed K8 region table — PROPOSED

| region | callables | declared signatures | why the boundary is here | predicted graphs |
|---|---|---|---|---|
| R0 `rollout_forward` | `policy.forward` (extractor + heads + functional masking + sampling) — T2's inference service | eval / no-grad × T2's buckets | the env step is the boundary; already compiled and CUDA-graph-captured by T2 | 1 per bucket (after 5b) |
| R1 `learner_micro_step` | `micro_step`: `evaluate_actions` + the whole loss fold (every ON term, TD-aux/CF own forwards included when on) → `(loss, terms, metrics)`; its backward compiled by AOTAutograd from the same graph | train / grad × B = micro (2048) | the accumulation group and the optimizer step happen per GROUP, not per micro; the probes read the returned per-term tensors; host reads are batched after it | **1** (+ its AOT backward) |
| R2 `probe_forward` | the rank probe's no-grad train-mode forward — only while `--rank-tripwire` keeps it every update | train / no-grad × B = micro | a separate grad mode is a separate graph by construction | 1 |
| R3 `optimizer_step` | `clip_grad_norm_` + `AdamW.step` + `zero_grad` | × 1 (state initialised at startup) | runs once per group; torch itself breaks around `step` | 1 per optimizer frame (AdamW → Adam, measured 1 each) |
| *eager, declared* | batch build, label alignment, flag resolution, the KL early-stop read (returned by R1, read once), the batched metric read, logging, every cadence diagnostic, the win-prob start metrics | — | host-side or data-dependent by nature; none is on the hot path once #3/#4 land | 0 |

Predicted compiled share of the update wall after R1 + R3 (+ #3, #4): **~80–85%** (today 60.3%).

## Acceptance criteria for K8 — PROPOSED

1. **graphs == declared (region × signature)**, counted by this tool's capture over update 1, update
   2, the eval forward and update 3 — today `whole_step` is 646 (2.8).
2. **0 undeclared breaks**: every region is compiled `fullgraph=True`; a break inside one is a
   startup FATAL; the tool's capture reports 0 break sites inside the regions on BOTH torches (the
   2.5.1 `forward_guard` break counts — `explain` does not see it).
3. **compiled wall-time share ≥ 80%** of the un-profiled update (`--stage time`: compiled kernel
   time per update / `train/train_ms`, torch 2.8, TF32, diag-skipped, on an IDLE box) — today
   60.3%.
4. **0 recompiles after freeze** (`compile_control`'s after-freeze counter; the capture's update 2,
   eval and update 3 report 0 recompiles).
5. **update time ≤ 0.80 × today's** on the learner benchmark (same buffer, flags, TF32,
   diag-skipped; today 53.2 s on 2.8 under contention — re-baseline on an idle box first).
6. **K6 parity held** for every region (the real-obs gate: legal log-probs, V, gradient cosine ≥
   0.9999, the TF32 rule) and **K9's learner golden unchanged** (one fp32 update, compiled R1 vs
   eager).

## Changes that add components (not exercised by this inventory)

- **Lane K9 (`laneK-k9`, uncommitted)** — `instrumented_ppo/learner_gates`: `check_buffer_finite`
  (once per update), `check_behaviour_first_micro` (a host read on the first micro),
  `check_loss_finite` / `check_kl_finite` (per micro), `clip_grad_norm_checked`. They READ R1's and
  R3's outputs, so they sit outside the regions; a per-micro finiteness read must be folded into
  the one batched host read, or it adds a sync per micro.
- **Ride-along heads — LANDED mid-measurement (`80146b74`, between the 2.8 and 2.5.1 GPU units;
  flag-gated and OFF under C's flags, so the inventory does not exercise them, and their new
  `ridealong` phase reads 0.01 s)** — `_ridealong_update` after each
  `evaluate_actions` on epoch 0: its own forward on detached stashes, backward, clip and a
  SEPARATE lazily-built Adam, plus a new `ridealong` phase. That is a fourth region candidate
  (`ridealong_step`, train/grad × micro, epoch 0) with its own optimizer region; its lazily built
  optimizer is a lazy acquisition the declared lifecycle forbids.

## Caveats

- The 2.8 time unit ran with the box at **load 75–95** (other agents). Host-bound phases (probes,
  batch, loss, idle) are inflated; kernel times are not. Criteria 3 and 5 need an idle-box baseline.
- The production prewarm and lock were skipped in the time stage (the compile and its parity gate
  ran); the warm-up compiled the timed signature.
- One checkpoint (C) and C's flags; C predates `--diagnostics-every`, so the timed updates force
  the skipped variant. TD-aux and the CF block are OFF in C's flags and so are not in this
  inventory.
- CPU trace geometry (192 rows, micro 64) — graph structure is batch-independent, recompiles on
  batch size are not exercised beyond the eval/train pair.

## Provenance

| result | dir under `~/gen3ai_archive/k8_inventory/` | copied here |
|---|---|---|
| 2.5.1 trace, `today` + `whole_step`, micro 64, HEAD `89679ddb` | `20260930_170750_trace_cpu_torch2.5.1` | `trace_torch2.5.1.json.gz` |
| 2.8 trace, `whole_step`, micro 64, HEAD `89679ddb` | `20260930_170750_trace_cpu_torch2.8.0` | `trace_torch2.8.0_whole_step.json.gz` |
| 2.8 trace, `today`, micro 512, HEAD `ecd2be00` | `20260930_155733_trace_cpu_torch2.8.0` | `trace_torch2.8.0_today.json.gz` |
| 2.8 time, CUDA TF32, HEAD `89679ddb` + this tool, load 75–95 | `20260930_174406_time_cuda_torch2.8.0` | `time_torch2.8.0.json` |
| 2.5.1 time, CUDA TF32, HEAD `8925e87b` (+ ride-along heads, off) + this tool, load ~8 | `20260930_180918_time_cuda_torch2.5.1` | `time_torch2.5.1.json` |
| the capped OOM repro | `20260930_170136_trace_cpu_torch2.8.0` (`rss.jsonl`) | — |
| `fullgraph=True` verdicts, both torches, HEAD `8925e87b` + this tool | `fullgraph/` | `fullgraph_torch2.5.1.json`, `fullgraph_torch2.8.0.json` |

`tables.md` is `python -m main.compile_inventory report <the dirs above>`, verbatim.
