# T25 perf phase on the END-STATE graph: results (2026-10-10)

The plan, the lever ranking and the protocol are in [`PLAN.md`](PLAN.md). This file holds what the GPU windows
MEASURED. Box: RTX 3080 Ti, torch 2.8.0+cu126, fp32 'highest'. The training chain was paused, so no other GPU
process ran; load1 was 1.3–2.4 (quiet). Tags are **MEASURED** unless marked otherwise.

## Window A: the COST ATTRIBUTION (09:13–09:49, lease `perf_T25`)

**Method.** `scripts/update_bench.py`, one process per config, with the closing test's ride-along flags
(`scripts/configs.py`). Each process:
- builds the learner for the config's argv at the production shape (98,304 rows, 2,048 × 32, 10 epochs);
- runs the trainer's own `arm_compile_sentinel` (the R1 startup parity gate, the prewarm, the lock);
- times the real `train()` on the update-fit fixture.

A full PLAIN update is estimated as median(1 epoch) + 9 × the marginal epoch, where the marginal epoch is (3-epoch
call − 1-epoch call) / 2. The code is PINNED to an extracted tree of `6e0a1a7a` (E's exact KO = the 16-roll sum).
Every config **passed the R1 startup gate**.

**Calibration against real launches (MEASURED both sides).** R: bench 51.4 s vs the launch's 51.59 s
(`gpu_checks_endstate_2026-10-09`). E: bench 53.5 s vs ~53.8 s (the closing test's precondition-E launch, descriptive).
Both agree within 0.6 %, so the tiled-fixture bench reads the real update.

| step | config | plain update (s) | Δ vs previous | marginal epoch (s) | diagnostic 1-epoch call (s) | compile + gate + prewarm (s) | peak reserved (MiB) |
|---|---|---|---|---|---|---|---|
| P | `--arch production` | **44.95** (real launch; the bench's P ran all-diagnostic, see F-PA-2) | — | (4.44, distorted) | 13.06 | 159.7 | 6,560 |
| S | + `--token-encoding static` | 44.4 | ≈ 0 vs P | 4.31 | 12.70 | 161.2 | 6,098 |
| SF | + the recovery facts, depth 2 (`static_recovery --trunk-layers 2`) | 44.5 | +0.1 | 4.33 | 12.72 | 164.4 | 6,084 |
| R | + the THIRD trunk round (`--arch static_recovery`) | 51.4 | **+6.9 (+15.5 %)** | 5.02 | 14.63 | 162.9 | 6,758 |
| EK | + the rest of the end state, KO = `ramp` | 53.2 | +1.8 (+3.5 %) | 5.20 | 14.71 | 196.7 | 6,624 |
| E | + exact KO (the 16-roll sum) = `--arch endstate` | 53.5 | **+0.3 (+0.6 %)** | 5.22 | 14.81 | 277.5 | 6,696 |

**Read.**
- The third trunk round is **~80 %** of the end state's extra update time over production: 6.9 of 8.6 s.
- The recovery facts and the static encoding cost nothing measurable.
- The rest of the end state costs 1.8 s, and the exact KO 0.3 s. The 16-term KO loop is one fused kernel on the
  GPU, so it was already cheap. The closed form (now THE `exact`) can save at most that 0.3 s; window B's E_base
  reads it.
- A diagnostic update (every 10th: the per-term noise sampler, grad balance, liveness) costs +7.2 to +8.4 s, about
  0.8 s per update amortised.
- A plain update's first-epoch overhead is ~1.2 s.

**E's profile** (one epoch under torch.profiler; this epoch was a DIAGNOSTIC one, F-PA-2):
- GPU busy 93.2 %; 9,092 kernels per micro-batch.
- Kernel time by class: GEMM 35.8 %, Triton pointwise 26.4 %, attention 15.1 %, Triton reductions 14.8 %, aten other
  7.8 % (`results/window_A/E.json` `profile`).
- The single top kernel is the fp32 memory-efficient attention BACKWARD (`fmha_cutlassB_f32`): 1.92 s of 14.1 s.
  Then the fp32 SIMT GEMMs (`ampere_sgemm_128x64_nn` 1.20 s, `_128x128_nt` 1.12 s).
- This is the shape the 2026-10-03 production profile had. The bf16 lever (A) attacks the GEMM and attention
  classes, together ~51 % of kernel time.
- The raw trace is LOCAL (`~/gen3ai_archive/perf_phase_2026-10-10/traces/`).

**Not measured in window A:** T2's per-step cost per architecture. The template deepcopy failed on E (F-PA-3), and T2
was dropped from P and R to fit the deadline. It is window C.

## Findings (standing rule 7)

- **F-PA-1 (harness, fixed).** The first attempt died in E's first `train()`: the ride-along heads' Adam state was
  still on the CPU. The bench builds the learner on CPU and moves it; a launch builds it on the device. The bench now
  moves every optimizer's state.
- **F-PA-2 (harness, fixed; two readings are distorted).** The diagnostics cadence's phase is the ROLLOUT index
  (`num_timesteps // rows`). The first E and P runs never advanced it, so every call was a diagnostic update. Their
  1-epoch calls carry ~7–11 s of probes, and their marginal epoch is distorted: E read 4.07 there vs 5.22 plain,
  because the diagnostic overhead differs between a 1-epoch and a 3-epoch call. The bench now times plain updates at
  phase 1–9 plus one diagnostic call. E was re-run plain (`E_plain.json`); P's plain bench value is NOT measured, and
  the real launch's 44.95 s stands in.
- **F-PA-3 (harness, fixed).** The trained policy does not `deepcopy`: its per-forward stashes hold non-leaf graph
  tensors. The T2 template is now a fresh build carrying the learner's state dict.
- **F-PA-4.** E's compile + gate + prewarm (277 s) is 1.7× production's (160 s). That is the learner side of every
  launch and restart, on top of T2's build. It feeds the owner's compile-cache decision (PLAN.md §3).

## Files

- `scripts/`: `update_bench.py`, `configs.py`, `window.sh`, `window_watch.sh` (the file-based handoff),
  `cpu_flops.py`, `module_flops.py`, `cpu_flops_all.sh`.
- `results/cpu_flops_*.json`, `results/module_flops_endstate.json`: the CPU FLOP reads (PLAN.md §0).
- `results/window_A/{P,S,SF,R,EK,E,E_plain}.json` + `logs/`: window A. `results/window_A_attempt1_device_bug/` is the
  dead first attempt (F-PA-1).
