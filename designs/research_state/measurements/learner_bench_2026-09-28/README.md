# Learner-only PPO update benchmark — 2026-09-28 (T13)

**Status: INSTRUMENT BUILT; GPU READ PENDING.** The read runs on an idle 3080 Ti after the T32b arm
ends. Its report JSON goes in this directory (`--publish`), and this README gets the numbers.

## The question

Where does the ~57 s of one arm-C PPO update go? That is `train/train_ms`: median 56.9 s, n=80,
over 98,304 samples, 10 epochs, and 480 micro-batches of 2048 samples with 32-step grad-accum.
Answers split three ways:

- **compute-bound**: a faster card helps;
- **launch/sync-bound**: fix the host side first;
- **probe cost**: diagnostics, not learning.

These are the facts already known from C's TensorBoard. They are inputs, and the benchmark does
not re-derive them:

- The noise-scale probe is 5.35 s of the update.
- The E5 two-point fit gives ~4.9 s per epoch plus ~7.5 s fixed.
- The update is ~0.31 of the 182 s iteration.

## The instrument

`src/agents/training/learner_benchmark.py`. The `designs/ops/testing.md` → Benchmarks entry
describes how to run it.

1. **A real buffer, never synthetic.** A worker runs the trainer in-process. Its argv is C's
   recorded `original_command`, re-pointed as a fork of C's `final_model.zip` into
   `~/gen3ai_archive/learner_bench/<stamp>/run_main`. It lets one `collect_rollouts` and its
   label callbacks run. It then pickles the pre-`train()` buffer and intercepts `train()`.
2. **Same work every repeat.** Before each call it restores all of these:
   - the weights;
   - the optimizer state;
   - the model's plain counters and EMAs;
   - the pristine buffer, because `get()` flattens it in place and the intent-label alignment
     shifts it in place;
   - the RNG seeds.

   The per-repeat `train/loss`, approx-KL and parameter checksum are recorded as evidence. On CPU
   they are bit-identical across repeats, and also across processes that reuse the saved buffer.
3. **Configs.** Each config gets 1 warm-up call and then K = 5 measured calls:
   - baseline;
   - baseline with sync-bracketed phase marks (`instrumented_ppo/phase_hook.py`);
   - the per-term noise probe OFF;
   - all optional telemetry OFF: the noise probe plus the grad-balance, rank, edge and cell
     probes;
   - n_epochs halved;
   - TF32 (`--matmul-precision high`), in a separate worker. It runs only if the TF32
     compile-parity gate exists at HEAD, and then the real startup gate decides.
4. **One profiled call** with 3 epochs, one `torch.profiler` window each. Window 0 also holds the
   once-per-call probes. Epoch 2 is the steady state. For each window it reports:
   - GPU busy %, as the union of device intervals over the window span;
   - the top 10 kernels;
   - kernel launches, sync calls and `aten::_local_scalar_dense` reads per micro-batch.

## What the CPU `--tiny` exercise already shows (code path, NOT a measurement)

Setup: 128 samples, batch 32, accum 2, a contended CPU. The report is at
`~/gen3ai_archive/learner_bench/20260928_091410_cpu_tiny/learner_bench.json`.

**UNVERIFIED on the production geometry:** the steady-state epoch windows issue **~270
host-blocking scalar reads (`aten::_local_scalar_dense`) per micro-batch**. The count does not
depend on the device. On cuda each read is a stream sync, and 480 micro-batches per update puts
that at ~130k syncs. That makes the launch/sync-bound hypothesis the first thing the GPU read has
to confirm or kill.

The tiny timings themselves are noise, and none of them is quotable.

## Hazards for the reader

- **Bracketed phases add syncs.** Each phase's % is reported against both the bracketed and the
  un-bracketed train_ms. On a sync-heavy update the phase table overstates the device-side
  phases, so the profiler windows are the source for the launch/sync question.
- **The noise-probe window.** The per-term noise probe samples only epoch 0's first accumulation
  group. So "noise probe OFF" saves a fixed cost per update, not a per-epoch one.
