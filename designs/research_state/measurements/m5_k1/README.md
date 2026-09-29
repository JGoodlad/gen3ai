# M5 Lane K1: learner speed A/B, torch 2.5.1 vs torch 2.8 (2026-09-29)

**Verdict: on this workload torch 2.8 (split OFF) is about 1-2% faster than 2.5.1 (split ON). The gap is
small and within about 2x the run-to-run spread. The learning is the same: the per-config loss agrees
to 5 decimals.** This is the first 2.8 run of a real `train()` to finish. The first attempt crashed
at the grad-balance probe (Lane K1b; fixed by `gen3_donated_buffer_off_v1`, see
`designs/training/compile_flags.md` "Lane K1b").

## Setup

- **Tool:** `python -m agents.training.learner_benchmark run --device cuda --seed 0`. It runs only the
  learner-side PPO update (`train()`), K=5 measured calls per config after 1 warm-up call.
- **Checkpoint:** `ai_v14_02_lbat_ctrl/final_model.zip` (sha256 `06efae2c44d6…`), using its own
  recorded flags.
- **Rollout buffer:** both arms use the same saved buffer, 2048 steps × 48 envs, captured by the
  2.5.1 run. The 2.8 runs load it with `--buffer`.
- **Hardware:** RTX 3080 Ti on an idle box. No trainer was running, and the GPU had no other
  compute apps at launch.
- **Code:** 2.5.1 ran at `964ddf41`. 2.8 ran at `964ddf41` plus the K1b fix.

## Files

| file | torch | notes |
|---|---|---|
| `learner_bench_20260929_061323_cuda_torch251.json` | 2.5.1+cu121 (`gen3ai_stable`), CUDA trunk split ON | the baseline |
| `learner_bench_20260929_105141_cuda_torch28.json` | 2.8.0+cu126 (`gen3ai_torch28`), split OFF | **the 2.8 row of record** (profiler off) |
| `learner_bench_20260929_095539_cuda_torch28_replicate.json` | same | replicate. Its main-worker rows agree with the row of record to within ±1%. Its **TF32 row is CONTENDED**: the routine test gate ran on the CPU during that worker, giving 57.46 s [54.92–60.62]. Do not quote it |

## train_ms per update: median over K=5 [min–max]

| config | 2.5.1 | 2.8 (row of record) | 2.8 / 2.5.1 | train/loss 2.5.1 → 2.8 |
|---|---|---|---|---|
| baseline | 58.45 s [58.31–59.77] | 57.21 s [57.07–58.57] | 0.979 | 0.37724 → 0.37721 |
| diag_skipped | 51.27 s [50.66–52.06] | 50.47 s [49.81–50.61] | 0.984 | 0.37723 → 0.37721 |
| epochs_half | 32.97 s [32.82–33.23] | 33.08 s [32.36–33.71] | 1.003 | 0.41338 → 0.41338 |
| tf32 | 54.46 s [53.97–54.78] | 53.42 s [53.03–54.12] | 0.981 | 0.37718 → 0.37719 |
| noise_probe_off | 51.49 s [50.94–51.79] | 50.72 s [50.45–51.37] | 0.985 | 0.37722 → 0.37722 |
| telemetry_off | 50.82 s [50.51–51.33] | 49.84 s [49.80–50.72] | 0.981 | 0.37724 → 0.37720 |

**Per-phase time (bracketed baseline, s per update, 2.5.1 → 2.8):**

| phase | 2.5.1 | 2.8 |
|---|---|---|
| backward | 28.33 | 27.31 |
| forward | 13.03 | 13.02 |
| noise_probe | 7.00 | 6.77 |
| loss | 6.83 | 6.88 |
| batch | 2.32 | 2.62 |

Nearly all of the gain is in the backward pass. The forward pass is unchanged, even though 2.8 runs
the extractor as one unsplit graph.

## What turning off the donated buffer costs

This was measured on torch 2.8 with `/tmp/k1b_don.py`. That script is not committed; the method is
below.

- **Method:** production extractor, compiled forward plus backward. Each arm used its own Inductor
  cache key. 40 timed steps after 5 warm-up steps, and each arm was run twice.
- **Result:** no speed cost.
  - Batch 64 (the production micro-batch, 2048/32): 10.16 / 9.93 ms with donation vs 10.11 / 10.17 ms
    without.
  - Batch 2048: 75.00 / 74.93 ms with donation vs 74.79 / 74.69 ms without.
- **Peak memory** rises by 1–11 MiB out of about 3.4 GiB (+0.3%).

## What this does NOT say

- **Run-to-run spread:** it is about ±1%, and each arm here is a single 5-call cell. The 2.8 speed-up
  is at most a small effect. Neither arm has a CI on its median.
- **Batch-1 eval graph:** on 2.8 the batch-1 CUDA eval graph does not lower (a Triton
  `CompilationError`). This benchmark never reaches that graph. Details are in the K1b section of
  `designs/training/compile_flags.md`.
