# Learner-only PPO update benchmark — 2026-09-28 (T13)

**Status: READ DONE 2026-09-28 13:59–14:49 PT** on an idle 3080 Ti (no trainer; box load 1.8–5.0 on 16 cpus, CPU reads at nice 19), HEAD `201b1128`, arm C's `final_model.zip` (sha256 `06efae2c…`). Report: `learner_bench.json` in this directory. Results are in §Results at the end.

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


## Results (2026-09-28, K = 5 repeats per config after 1 warm-up; repeats agree to ±1%)

| config | train_ms median | vs baseline |
|---|---:|---:|
| baseline (production: fp32, compiled, noise probe on) | **58.5 s** | — |
| noise-scale probe OFF | 51.8 s | **−11.5%** |
| all optional telemetry OFF | 51.1 s | −12.7% |
| 5 epochs (E5) | 33.9 s | **−42.1%** |
| TF32 (`--matmul-precision high`) | 54.4 s | −7.0% |

The live arm-C `train/train_ms` (56.9 s) is reproduced, so the live timer is honest.

**Phase breakdown** (cuda-synchronized brackets; bracketing cost +0.6%):
backward 28.7 s (49%) · forward 13.1 s (22%) · loss assembly 6.9 s (12%) · noise-scale probe 7.1 s (12%) · minibatch get 2.3 s (4%) · logging + KL + epoch end + capacity < 0.2 s.
Two-point epoch fit: 4.93 s per epoch (102.8 ms per 2,048-sample micro-batch), 9.2 s fixed.

**The host-sync hypothesis is REFUTED.** 66,852 blocking scalar reads per update (≈ 240 per micro-batch) cost **1.0 s in total (1.7%)**. The profiler shows the GPU **99.2–99.5% busy** through every epoch window, so the host never starves it. Moving the diagnostics on-device is hygiene, not speed.

**What the GPU is doing instead:** ≈ 4,270 kernel launches per micro-batch in steady epochs (≈ 24 µs of GPU time each), dominated by Inductor's fused `triton_` kernels (~1,000 per micro-batch, ~31% of kernel time), the memory-efficient attention BACKWARD (~7%), and fp32 SGEMMs. Matmuls are a minority, which is why TF32 buys only 7%. Reading: the update is **GPU-bound on many small, mostly elementwise/memory-bound kernels**, not on tensor-core math and not on the host. Epoch 0 carries the noise-scale probe (≈ 8,100 launches per micro-batch vs 4,270).

**Implications (feed M5 Lane K and T13):**
1. The noise probe on a cadence (every Nth update) recovers ~11% of the update ≈ 3.5% of wall time.
2. Fewer, larger kernels are the next lever: a larger micro-batch at the same effective batch (e.g. 8,192 × accum 8 instead of 2,048 × 32; same math up to float order) — NOT measured yet; add it as a benchmark config.
3. Loss assembly at 12% is worth a look (aux/belief heads).
4. For the GPU purchase (T13): a workload of small memory-bound kernels scales with clocks, cache and bandwidth far more than with tensor-core FLOPs — the clock sweep will confirm.
