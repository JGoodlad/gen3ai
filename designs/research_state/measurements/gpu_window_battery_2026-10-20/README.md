# FP32_ALT GPU-window battery (staged 2026-10-20, NOT RUN)

The 2026-10-19 battery's T62 gate read **FP32_ALT**: T62's float64 τ solve cost 82-127 % of a training cycle on this
card ([`../gpu_window_battery_2026-10-19/README.md`](../gpu_window_battery_2026-10-19/README.md), "T62 VERDICT").
FP32_ALT is built in `gen3_tau_fp32_alt_v1` (config v160): a split residual in fp32, no float64 op. Its CPU evidence is
in [`../tau_fp32_alt_2026-10-20/`](../tau_fp32_alt_2026-10-20/).

This battery measures FP32_ALT's GPU cost against the pre-T62 naive fp32 sum. It has two arms only: fp64 was measured
by the 2026-10-19 battery. The training session runs it in a short window before X35's production arms. Production GO
waits for its read.

## Run it (the training session, with the run PAUSED and the GPU clear)

```bash
cd /home/goodlad/dev/gen3ai                                  # main, at a commit that carries FP32_ALT and this directory
export GEN3AI_GPU_LEASE_TOKEN_FILE=<the lease token file>     # or GEN3AI_GPU_LEASE_TOKEN=<token>
B=designs/research_state/measurements/gpu_window_battery_2026-10-20
$B/run_battery.sh --out /tmp/gpu_battery_fp32alt --dry-run          # CPU only: every check + every command
$B/run_battery.sh --out /tmp/gpu_battery_fp32alt --budget-min 45    # the window; re-run the same line to resume
/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 $B/read_battery.py /tmp/gpu_battery_fp32alt
```

**Exit codes:** `0` DONE · `1` HOLD (`battery_status.json` and the `HOLD:` line in `battery.log` give the reason; the
next cell did NOT run) · `10` BUDGET_STOP (re-run the same line to resume) · `2` usage error, or a refused `--out`
(never under `models/` or `data/`).

The battery never stops, resumes or signals a training run. The caller pauses and resumes the run. On a HOLD, the
window is the orchestrator's call (`TRAINING_RUN_SOP.md` §3, rule 3). The driver is the 2026-10-19 battery's, with
only the cells changed: the same `gpu_lock` + `mem_cap` + wall cap per cell, HOLD semantics, `--dry-run`,
`--budget-min`, `--only` and resumability.

## The cells, in run order

Each cell runs as `gpu_lock.sh mem_cap.sh --name battery-<cell> <GB> timeout --kill-after=60 <cap> <cmd>`. A cell is
skipped on a re-run once its `cells/<cell>.state.json` reads `ok`.

| # | cell | what it measures | mem cap | wall cap | ETA |
|---|---|---|---|---|---|
| 1 | `tau_dec` | **Rollout.** T2's decision forward (`engine.decide`, inductor `dynamic=False`, CUDA graph: the `graph` backend's build) at buckets 64 and 256, `fp32_naive` (old) vs `fp32_alt` (new), in one process on the same seeded fresh build at X26's surface. 8 interleaved blocks of 50 replays per arm (ABAB). Also the agreement of the two arms: max \|d log π\|, max \|d V\|, greedy agreement. The method is `t62_dec`'s | 16 GB | 45 min | 16 min |
| 2-5 | `tau_u_naive_a`, `tau_u_alt_a`, `tau_u_alt_b`, `tau_u_naive_b` | **Learner.** The frozen suite's `update_bf16` cell (X26's argv, the K9 learner, the compiled bf16 `train()` at 384 × 256 rows, micro-batch 2,048 × accumulation 32, the 10-epoch estimate), the arm forced per process, run **ABBA**. Only the cell's name differs from the suite; `reps` is the suite's 2 | 24 GB each | 40 min each | 6 min each |

**Total ETA: about 40 min** (`tau_dec` 16, the learner 24), plus about 1.5 min of CPU checks. **Worst case at every
wall cap:** 205 min.

### Why 40 min and not 30: compile time is fixed

The 2026-10-19 walls (`/home/goodlad/gen3ai_archive/gpu_battery_2026-10-19/`) show where the time goes:

- `t62_dec` took 15.8 min. 15.1 min of it was compiling (447 s + 460 s for the two arms, both buckets). The 8 timed
  blocks took seconds. Fewer blocks saves almost nothing, so `tau_dec` keeps 8 blocks and its noise.
- The `t62_u_*` cells took 6.1-9.4 min. 4.2-5.8 min of each was compile + gate + prewarm. Timing at `reps` 4 was
  about 75 s (fp32) and 165 s (fp64). At the suite's `reps` 2, a naive or alt cell should take 5-7 min.

Getting to 30 min would need a change of method. Two options, for the orchestrator to choose:

- run one learner pair (`--only tau_dec,tau_u_naive_a,tau_u_alt_a`), about 28 min. This loses the second pair, and with
  it the drift term in the band;
- drop bucket 64 from `tau_dec`, saving about 7 min. This loses the SERIAL opponent bound.

### Noise at these settings

- **Learner.** At `reps` 4 the two fp32 cells read 41.9 and 41.4 s (1.2 % apart), and each rep spread was 3 % or less.
  `reps` 2 takes the median of two, so expect about 1-2 % per cell. One percent of the update is about 0.56 % of the
  cycle. The ABBA pair disagreement puts this noise into the band.
- **Rollout.** The 2026-10-19 per-block deltas were tight: B64 3.35-3.78 ms, B256 14.94-15.06 ms. The SERIAL bound is
  the sensitive term. It multiplies the B64 delta by 20 opponent slots and 416.5 host steps, so 0.27 ms of B64 delta
  alone is 3 % of the cycle.

## The τ forcing (`tau_force.py`)

A model's solve is `features_extractor.hypothesis_builder.tau_solve`, and no model accepts `fp32_naive`. So each cell
forces its arm at the one call site every model path shares. `hypothesis_set.fixed_mass_presence` calls
`fixed_size_tau` through the module global. The cell replaces that global with a wrapper, before anything is built or
compiled. The wrapper ignores the caller's `solve` and passes `solve=<arm>`.

The wrapper counts its calls without changing the compiled program. An eager call adds to `EAGER[arm]`. A dynamo
trace runs a `comptime` callback, which counts at trace time and puts nothing in the graph. (A plain counter
mutation inside a traced function recompiles on every call. This was probed on 2026-10-20 with torch 2.8.)

Each cell writes a `tau_force` block into its result. The cell FAILS, with exit 4 and status `fail`, if the wrapper was
replaced mid-cell or an arm called it 0 times. The reader also re-checks this block.

**`--check`** (CPU, about 45 s per script; the four learner cells share one run):

1. The FP32_ALT API is present: `TAU_SOLVES`, `TAU_SOLVE_NAIVE`, `fixed_size_tau(…, n_iter, solve)`, and no
   `TAU_SOLVE_DTYPE`.
2. The seeded fresh model is built at X26's surface. For each arm, the cell's own path must reach the wrapper:
   - the decision cell: T2's eager decision forward, B8;
   - the learner cells: `evaluate_actions_functional` forward + backward, B8.
3. A dynamo trace of `fixed_mass_presence` (`fullgraph=True`) per arm must count one trace and must not re-trace on a
   second call.

## The FP32_ALT gate (`read_battery.py`, deterministic)

The 2026-10-19 T62 cost computation, with `fp32_naive` as old and `fp32_alt` as new:

1. **Learner.** Take each ABBA pair's delta of `update_s_est` (`alt_a − naive_a`, `alt_b − naive_b`) as a fraction of
   the mean naive estimate. Scale it onto X26's live update time: `train_s` = 42.24 s.
2. **Rollout.** Take each bucket's per-block deltas (`alt − naive`) over X26's 416.5 host steps per cycle. There are
   two bounds:
   - SERIAL: the trainee's B256 forward plus 20 opponent B64 forwards;
   - TRAINEE-ONLY: B256 only.
3. **The band.** `cost_hi` is the larger learner delta plus SERIAL at the max per-block deltas. `cost_lo` is the
   smaller learner delta plus TRAINEE-ONLY at the min B256 delta. Both are a % of X26's 74.8 s live cycle, the same
   `LIVE_CYCLE` as the T62 read.
4. **The verdict:**
   - **CONFIRMED** iff `cost_hi ≤ 3 %`;
   - **OVER** iff `cost_lo > 3 %`;
   - **INCONCLUSIVE** otherwise. The band straddles the bar, so the orchestrator decides (standing rule 8: never
     decided by chance);
   - **INCOMPLETE** if any of the five cells is not `ok`, or lacks a clean `tau_force` record for its arm.

The reader also prints the agreement of alt vs naive and the compile sentinel's gate verdict per learner cell.
fp64 vs fp32_naive is not re-measured. The 2026-10-19 battery's `t62_dec` is that read.

## Hazards known when staging

- **The naive arm is forced through the wrapper, not checked out at bd3126e02^.** It runs the same arithmetic as the
  pre-T62 sum (`fixed_size_tau`'s `fp32_naive` branch). The traced graph may still differ from bd3126e02^'s by
  dtype-promotion no-ops. Its timing was not compared with the 2026-10-19 `old` arm.
- **The compile sentinel's startup parity gate** reads the committed K9 golden. An arm whose solve differs from the
  golden's may be REFUSED there. It then continues with install + prewarm only, and the timing is still the compiled
  update's. A REFUSED gate is not a HOLD. The reader prints `gate` per cell.
- **The network is a seeded FRESH build at X26's surface**, not X26's weights. τ's cost is a fixed 64-step bisection
  on static shapes, so the weights do not move it. X35's architecture may differ from X26's.
- **The rollout bounds leave out the eval cycle and T2's host-side work.** The cycle denominator is X26's live cycle at
  258.6M (2026-10-18), not X35's.
- **The learner cells run without a CPU pin.** Any other CPU job on the box adds noise. The ABBA disagreement carries
  that noise into the band.
- **`tau_dec`'s ETA is the 2026-10-19 compile time.** `fp32_alt` has more ops per bisection step than the naive sum,
  so its compile may take longer. The 45 min wall cap leaves room.
