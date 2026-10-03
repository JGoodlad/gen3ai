# The fp32 'highest' UPDATE BENCHMARK, working again (deletion pass lane C, 2026-10-02)

**This is a benchmark READ, not a baseline and not a gate.** The owner retired the wall-clock guard
(`compiled_perf_guard_test`, deleted in the same commit): a timing test is noisy, needs an idle box and goes stale — its
only banked number was TF32's, which K2 retired. Performance is checked by deterministic functional tests of what makes the
update fast (`designs/ops/testing.md` "Performance-shape tests") plus THIS benchmark, run on demand at milestones: after a
torch upgrade, before a baseline run, on suspicion. The numbers below go stale; re-run, do not compare against them blindly.

    scripts/ops/gpu_lock.sh timeout 1260 scripts/ops/mem_cap.sh 48 python -m main.compile_inventory run \
        --stage time --device cuda --keep-prewarm --unbracketed --out-root <dir>        # ≈ 6.5 min, idle GPU

**The read (MEASURED, n = 5, one idle box):** one production PPO update through the K8 regions takes **40.17 s** (range
40.11–40.23 s, relative spread **0.31%**), **89.2%** of its wall compiled (0.890–0.893), on the pinned 98,304-row buffer
(micro-batch 2,048 × accumulation 32, 10 epochs), arm C's final checkpoint, torch 2.8.0+cu126, RTX 3080 Ti, the Rust env core.

## Why the benchmark could not run, and the cause (found, not guessed)

K2 retired TF32 and could not re-measure at fp32: the time stage, run as the guard ran it, timed **0.32 s on a freshly
collected 4,096-row buffer** and never reused the pinned one. K2 could not say why (UNVERIFIED); it is now traced:

- Since `gen3_update_fit_v1` (2026-10-01) the trainer runs ONE REAL `model.train()` at startup, before `learn()` —
  `update_fit.dry_update`, the CUDA first-update fit check, on a tiled FIXTURE of the buffer's declared shape. The worker
  runs with `--n-envs 2`, so that fixture is 2 × 2,048 = **4,096 rows**.
- Both benchmark workers (`learner_benchmark`, `main.compile_inventory`) replaced `InstrumentedMaskablePPO.train` for the
  WHOLE process, so the first `train()` call that ever happened was the fit check's. The time stage took it as the
  measurement, wrote `time_result.json` and `os._exit`-ed — before `learn()` reached `collect_rollouts`, where the tool's
  `_bench_collect` restores the pinned buffer. (K2's "Rust collection of 4,096 fresh rows" is, in the log I reproduced, the
  `[RUST ENV] collector … update at >= 4,096 completed-game rows` startup banner; no collection ran. The Rust core was
  never at fault.)
- **What is verified, and how.** (a) A time stage run without `--keep-prewarm` (so the regions are not installed) raised
  "not compiled" immediately after the `[CudaLedger]` startup table — the stage's `train` fired at startup, before any
  `reused buffer` line, which only `learn()` can print. (b) The fit check's own log line reads `dry first update (4,096
  fixture rows, 1 epoch …)`. (c) After the fix the same command reaches `reused buffer … (2048x48)` and times the
  98,304-row update. K2's "0.32 s on 4,096 rows" fits (a)+(b); I did not rerun the UNFIXED tree to completion under
  `--keep-prewarm`, so that exact number is K2's, not re-measured here.
- A second obstacle sat behind the first: the K6 learner freeze (`gen3_no_global_reseed_v1`) makes a global RNG SEED after
  the freeze a `FATAL_CONFIG`, and every repeat seeds (`seed_all`) to replay the same minibatch permutations. The workers
  now run inside `global_rng_guard.isolated_global_rng()`.

The fix: `learner_benchmark.learn_loop_only` / `install_worker_hooks` — the tool's `train` is live only from the learn
loop's first `collect_rollouts`; the startup update runs production's own code. The time stage also REFUSES to run unless
the pinned buffer was restored (`buffer_restored`, recorded in `time_result.json`), so this failure can no longer produce a
number. A time stage without `--keep-prewarm` is refused up front (the regions are installed by the compile sentinel the old
skip-the-prewarm variant replaced).

## The units

Interleaved **B P B B B B**, then **L1**; one unit per `gpu_lock` hold, load1 < 0.5 and no GPU process before each
(`results/*_box.txt`). Driver `run.sh` (resumable), `run_loaded.sh`, reader `summarize.py`, copier `collect.sh`.

- **B1–B5** — the tree as committed (main `cbd20111` + the lane C change; no learner code differs from main).
- **P1** — a throwaway worktree with ONE planted regression: `compile_regions.install`'s R1 dispatcher runs the EAGER
  micro-step once the lock is held, uncounted (a change that routes AROUND the region's compiled route and its counters).
  P2 was skipped; one planted read is enough.
- **L1** — B under 6 CPU-bound bystander processes for the whole unit.

| unit | un-bracketed update | bracketed | compiled share of the update wall | `train/loss` |
|---|---|---|---|---|
| B1 | 40.189 s | 40.760 s | 0.8913 | 0.384206 |
| P1 (planted: R1 eager) | **116.481 s (2.90×)** | 116.608 s | **0.000** | 0.384300 |
| B2 | 40.140 s | 40.771 s | 0.8924 | 0.384207 |
| B3 | 40.234 s | 40.864 s | 0.8902 | 0.384214 |
| B4 | 40.182 s | 40.845 s | 0.8920 | 0.384215 |
| B5 | 40.110 s | 40.750 s | 0.8933 | 0.384211 |
| L1 (6 bystanders) | 40.189 s (+0.04%) | 40.774 s | 0.8865 | 0.384206 |

The five B units agree to the digit on `train/loss` (0.38421) and `approx_kl` (0.00870): the same program ran each time, and
it is the K8 acceptance's own update (its TF32 read logged `train/loss` 0.38423). A run on the pre-L3 tree (`5c02513a`, the
first after the fix) read 40.08 s. The regions' update is GPU-bound: six CPU bystanders moved it +0.04%.

## What the planted regression shows (and why a timing gate is the wrong tool for it)

R1 on its eager route is **not** caught by the run-level defences when it routes AROUND them (no counter, no inventory
entry changes after the lock): the time stage finished `ok` at 2.90× with a compiled share of 0.000 and its loss matching to
4 digits. The retired guard failed it (120.3 s against a 42.2 s bar), but a timing test finds it only at a milestone, on an
idle box, against a number that goes stale. The same regression is a deterministic failure of
`update_performance_shape_test::test_a_real_update_runs_every_micro_batch_through_R1_compiled_and_none_eager`, in every
routine gate (`designs/ops/testing.md`).
