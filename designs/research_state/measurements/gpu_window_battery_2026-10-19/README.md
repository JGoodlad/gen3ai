# GPU-window battery (staged 2026-10-19, NOT RUN)

One script collects every pending GPU-only measurement in the next GPU window. That window is X34's planned 24 h
restart (about 2026-10-20 11:35) or the gap between X34 and X35. The training session holds the GPU lease and runs
it. Nothing here has touched a GPU yet: it was staged on the CPU, with a `--dry-run` that came back green.

## Run it (the training session, with the run PAUSED and the GPU clear)

```bash
cd /home/goodlad/dev/gen3ai                                  # main, at a commit that carries this directory
export GEN3AI_GPU_LEASE_TOKEN_FILE=<the lease token file>     # or GEN3AI_GPU_LEASE_TOKEN=<token>
B=designs/research_state/measurements/gpu_window_battery_2026-10-19
$B/run_battery.sh --out /tmp/gpu_battery_2026-10-20 --dry-run          # CPU only: every check + every command
$B/run_battery.sh --out /tmp/gpu_battery_2026-10-20 --budget-min 100   # the window; re-run the same line to resume
/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 $B/read_battery.py /tmp/gpu_battery_2026-10-20
```

**Exit codes:**
- `0`: DONE.
- `1`: HOLD. `battery_status.json` and the `HOLD:` line in `battery.log` give the reason. The next cell did NOT run.
- `10`: BUDGET_STOP. The next cell's ETA would overrun `--budget-min`. Re-run the same line to resume.
- `2`: usage error, or `--out` was refused (it must not be under `models/` or `data/`).

The battery never stops, resumes or signals a training run. The caller pauses and resumes the run. On a HOLD, the
window is the orchestrator's call (`TRAINING_RUN_SOP.md` §3, rule 3).

## Run note: the `frozen_guard` cell (cell 2) failed on its TEST threshold, not on the guard

On the first run (2026-10-19, under the training session's lease) both parametrisations failed on
`cuda_reserved_bytes_after_freeze >= 256 * 2**20`: the counter read 251,658,240 B = 240 MiB. Every precondition
passed (`segments_now == 0`, a healthy flush reserved 0 B / 0 growths, the old segment counter read 0), so the
reserved-bytes counter DID see the new mapping. The bar was wrong: under `expandable_segments` the allocator reuses
pages that are already mapped and free (it extends the segment's free tail and maps only the shortfall), so 16 MiB of
the 256 MiB request was not a new mapping. The test now asserts `>= reserved_growth.new_mapping_floor(request,
reusable)`, where `reusable = memory_reserved() - memory_allocated()` is read inside the guard just before the request
(`gen3_expandable_floor_v1`; CPU-tested in `service_test.py`). The cell must be re-run on a card to see 2 passed.

**`RESERVED_GROWTH_POISONS` stays OFF** by the orchestrator's ruling, for a separate reason: the overlap false-FATAL
found in the core review. This cell passing does not arm it.

**Slow-tier recording is off for the pytest cells** (`frozen_guard`, `x45_cuda_test`). They run from the MAIN checkout,
where the conftest's recorder would rewrite main's committed `designs/ops/slow_tier_status.json` and dirty main
mid-window. Each cell's command carries `GEN3AI_SKIP_SLOW_STATUS_RECORD=1`, and `--dry-run` FAILs if a pytest cell
lacks it. The cost: these two cells' verdicts are NOT banked in the status file; read them from the junit XML.

## The cells, in run order

Each cell runs as `gpu_lock.sh mem_cap.sh --name battery-<cell> <GB> timeout --kill-after=60 <cap> <cmd>`. A cell is
skipped on a re-run once its `cells/<cell>.state.json` reads `ok`, or `skip` (its precondition says it cannot run at
this commit).

| # | cell | what it measures | mem cap | wall cap | ETA |
|---|---|---|---|---|---|
| 1 | `t62_dec` | **T62, rollout.** T2's decision forward (`engine.decide`, inductor `dynamic=False`, CUDA graph: the `graph` backend's build) at buckets 64 and 256. It compares the pre-T62 fp32 τ solve with the fp64 one, both in one process on the same weights and rows. Timing is interleaved over 8 blocks of 50 replays per arm, and the cell also checks that the two arms agree on log π, V and the greedy action | 16 GB | 45 min | 20 min |
| 2 | `frozen_guard` | **bf4eb59d's arming check:** `service_cuda_test.py::test_expandable_segments_steady_state_reserves_nothing_and_a_new_mapping_is_seen`, eager + graph, with `GEN3AI_TEST_ALLOW_GPU=1`. It must show 2 passed with 0 skipped: **a SKIP is a HOLD, never a pass** | 16 GB | 65 min | 10 min |
| 3-6 | `t62_u_old_a`, `t62_u_new_a`, `t62_u_new_b`, `t62_u_old_b` | **T62, learner.** The frozen suite's `update_bf16` cell (`gpu_bench_update`: X26's argv, the K9 learner, the compiled bf16 `train()` at 384 × 256 rows, micro-batch 2,048 × accumulation 32, the 10-epoch estimate). τ is solved in fp32 (old) or fp64 (new), run in **ABBA** order. Two fields differ from the suite: the cell's name and `reps` 4 (the suite uses 2) | 24 GB each | 40 min each | 12 min each |
| 7 | `x45_cuda_test` | **X45's CUDA test** (landed `ad821b615`): `grouped_cuda_test.py` with `GEN3AI_TEST_ALLOW_GPU=1` (graph-backend parity, weight swap, frozen steady state). It must show 1 passed with 0 skipped: **a SKIP is a HOLD**. It runs before the benchmark cells | 24 GB | 30 min | 5 min |
| 8-11 | `x45_p20_b3`, `x45_p20_b5`, `x45_p5_b3`, `x45_p5_b5` | **X45 sweep** (§1.6): `grouped_benchmark.py --opp-slots P --buckets B --json`, with P in {20, 5} and B in {`8,64,256` (b3), `8,16,32,64,256` (b5)}. It compares the grouped (vmapped) opponent forward against per-slot forwards (OFF / ON interleaved). `read_battery.py` prints per-slot vs grouped ms/step, the ratio and the peak memory. **SKIP** at a commit that lacks `grouped_benchmark.py`, `grouped_cuda_test.py` or `--t2-grouped-opponents` | 24 GB each | 30 min each | 7 / 10 / 7 / 10 min |
| 12 | `x14_ab` | **X14 (optional):** off-policy overlap A/B. It runs only once the X14 unit commits `x14_cell.sh` here. The hook takes `<out-json>` (`{"status": "ok", …}`) and supports `--check`, which exits 0 when ready, 3 for SKIP (not at this commit) and anything else for FAIL. Absent means SKIP | 24 GB | 60 min | 30 min |

**Total ETA:** about **117 min** for cells 1-11 (T62 78, `x45_cuda_test` 5, the X45 sweep 34); X14 (cell 12) adds
about 30 min when staged and SKIPs today. **Worst case at every wall cap:** 420 min for cells 1-11 (cells 1-6 270, cells 7-11 30 each), 480 min with X14. Order is
value per GPU-minute: first the T62 gate input that is cheap and least noisy, then the cheap pass / fail arming
check, then the learner pairs, then the in-build units. With a short window, `--budget-min` stops cleanly between
cells.

**CPU heavy slots (verified 2026-10-20, core review D-3).** Rule 5b's slots apply only while a `train_rl_agent.py
--run-dir` process is live (`utils.heavy_slots.wanted`). A cell runs as `gpu_lock.sh mem_cap.sh …`, and a job
that already runs under `gpu_lock` (flock holder or lease owner) takes **no** CPU slot (`GEN3AI_GPU_JOB`,
`utils.gpu_lock.gpu_job`; pinned by `heavy_slots_test.py` and
`mem_cap_test.py::test_gpu_lock_outside_mem_cap_takes_no_cpu_slot_beside_a_full_house`). So the battery cannot
stall on a slot whether or not a run is live, and never holds the GPU idle behind a queue (exit 87 cannot occur).

**HOLD conditions:**
- a cell exits non-zero;
- a wall timeout (124 / 137);
- an OOM at the cell's cap (86);
- `gpu_lock` refuses the cell (5 / 6);
- no `ok` result was written;
- a precondition FAILs (as opposed to a SKIP);
- the GPU is not clear before a cell (`nvidia-smi --query-compute-apps`: a benchmark never shares the card);
- no lease token is set.

## Decision rules (`read_battery.py`, deterministic)

**The T62 gate:** adopt the float64 τ solve in X35's pins iff it costs **≤ 3 %** of a training cycle. Otherwise X35
uses the fp32 complementary-constraint alternative, recorded in `designs/endstate/design_x5_belief_tokens.md` (the
T62 decision-record row).

The reader works out the cost in four steps:

1. **Learner.** It takes each ABBA pair's delta of `update_s_est` (`new_a − old_a`, `new_b − old_b`) as a fraction of
   the mean old estimate. It then scales that fraction onto X26's live update time: `LIVE_CYCLE.train_s` = 42.24 s,
   from the `gpu_bench_suite` calibration.
2. **Rollout.** It starts from each bucket's per-block deltas (`new − old`) and sums them over X26's 416.5 host steps
   per cycle. It works out two bounds:
   - **SERIAL**: the trainee's bucket-256 forward plus 20 opponent bucket-64 forwards, so every opponent delta adds;
   - **TRAINEE-ONLY**: the lanes hide every opponent delta.
3. **The cost band.** `cost_hi` is the larger learner delta plus SERIAL at the largest per-block deltas. `cost_lo` is
   the smaller learner delta plus TRAINEE-ONLY at the smallest per-block delta. Both are expressed over X26's 74.8 s
   live cycle.
4. **The verdict:**
   - **ADOPT** iff `cost_hi ≤ 3 %`;
   - **FP32_ALT** iff `cost_lo > 3 %`;
   - **INCONCLUSIVE** otherwise. The band straddles the bar, and standing rule 8 says a check is never decided by
     chance: the orchestrator decides, for example with a collect-phase A/B;
   - **INCOMPLETE** if any of the five T62 cells is not `ok`.

**Frozen guard:** PASS iff the junit shows 2 tests, 0 failures, 0 errors and 0 skipped. A PASS clears
`RESERVED_GROWTH_POISONS` to be armed (gen3_t2_reserved_after_freeze_v1, warn-first until then).

**X45 / X14:** this battery reports them and does not decide them. Their own units hold the decision rules.

## Hazards known when staging

- The **old arm patches `hypothesis_set.TAU_SOLVE_DTYPE` to float32**. It does not check out bd3126e02^. The two are
  equal only because the τ scores reach `fixed_size_tau` in float32: bf16 autocast is confined to the trunk rounds
  (`trunk_precision.py`), and the T62 equivalence read uses the same patch. This was not re-verified on the GPU.
- The **compile sentinel's startup parity gate** reads the K9 golden that was re-recorded under fp64. The fp32 learner
  arm may therefore be REFUSED there and continue with install + prewarm only. The reader shows `gate` per cell. A
  REFUSED gate is expected on `old` and is not a HOLD.
- The **network is a seeded FRESH build at X26's surface** (`gpu_bench_suite.PRODUCTION_ARGV` at the run's commit),
  not X26's weights. τ's cost is a fixed 64-step bisection on static shapes, so the weights do not move it. X35's
  architecture may differ from X26's; the gate is read at X26's.
- The **rollout bounds leave out the eval cycle and T2's host-side work**, and the cycle denominator is X26's live
  cycle at 258.6M (2026-10-18), not X35's.
- The learner cells run without a CPU pin. In a window the trainer is paused, but any other CPU job on the box adds
  noise. The ABBA pair disagreement is what puts that noise into the decision band.
