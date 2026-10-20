# FP32_ALT + X14 GPU-window battery (staged 2026-10-20, NOT RUN)

The 2026-10-19 battery's T62 gate read **FP32_ALT**: T62's float64 τ solve cost 82-127 % of a training cycle on this
card ([`../gpu_window_battery_2026-10-19/README.md`](../gpu_window_battery_2026-10-19/README.md), "T62 VERDICT").
FP32_ALT is built in `gen3_tau_fp32_alt_v1` (config v160): a split residual in fp32, no float64 op. Its CPU evidence is
in [`../tau_fp32_alt_2026-10-20/`](../tau_fp32_alt_2026-10-20/).

This battery has three parts, in this order, in one short GPU window before X35's production arms:

1. **FP32_ALT's tau on the card** (`tau_cuda_parity`, ~9 min, FIRST: cheap, and it decides what the cost read MEANS).
   Every earlier "two programs land on one tau" number compared CPU programs; the production pair is T2's compiled CUDA
   rollout (Triton) against the CUDA learner. The review's F2
   ([`../tau_fp32_alt_2026-10-20/REVIEW.md`](../tau_fp32_alt_2026-10-20/REVIEW.md)) found it unmeasured. A miss of the
   preregistered bar HOLDS the battery before any cost is read. Section "The tau parity cell".
2. **FP32_ALT's GPU cost** against the pre-T62 naive fp32 sum (two arms only: fp64 was measured by the 2026-10-19
   battery). Production GO waits for this read.
3. **The X14 overlap A/B** (`x14_off`, `x14_on`): two forks of X26 at 380,000,195, 15 updates each, at the battery's pin.
   The first attempt (2026-10-20, pin `397cce81`) was deferred because that pin carried T62's fp64 τ solve
   ([`../x14_overlap_2026-10-19/README.md`](../x14_overlap_2026-10-19/README.md) §4 finding 13). This is the re-run.

4. **T2's frozen guard on the card** (`t2_guard`, added 2026-10-20, LAST so it never delays the FP32_ALT read): the
   2026-10-19 `frozen_guard` arming check re-run on its fixed bar, plus the overlap STREAM ATTRIBUTION proof
   (`gen3_t2_stream_attribution_v1`). Skip it with `--only` if the window is short; it decides nothing below.

The training session runs it with the X35 arms paused. **The first three parts need ~125 min** (8 cells), the fourth
~12 min more (9 cells; section "The cells").

## Run it (the training session, with the run PAUSED and the GPU clear)

```bash
cd /home/goodlad/dev/gen3ai                                  # main, at a commit that carries FP32_ALT and this directory
export GEN3AI_GPU_LEASE_TOKEN_FILE=<the lease token file>     # or GEN3AI_GPU_LEASE_TOKEN=<token>
B=designs/research_state/measurements/gpu_window_battery_2026-10-20
# BEFORE the window, on the CPU (any time; ~7 min at nice 19 beside a live run; the window re-runs only the light checks):
nice -n 19 taskset -c 5-7,13-15 $B/run_battery.sh --out /tmp/gpu_battery_fp32alt --dry-run
# the window; re-run the SAME line to resume:
$B/run_battery.sh --out /tmp/gpu_battery_fp32alt --budget-min 135
/home/goodlad/miniconda3/envs/gen3ai_torch28/bin/python3 $B/read_battery.py /tmp/gpu_battery_fp32alt
```

**Options.** `--only a,b` runs named cells. `--learner-gate on|off` (default `on`): the learner cells' startup gate, see
"The learner fairness rule". `--pin <commit-ish>`: the commit the X14 arms run at (default: main HEAD when the first cell
started, recorded in `battery_status.json`; it must contain `d4af9f1e` and `02cac1a9`).

**Exit codes:** `0` DONE · `1` HOLD (`battery_status.json` and the `HOLD:` line in `battery.log` give the reason; the
next cell did NOT run) · `10` BUDGET_STOP (re-run the same line to resume) · `2` usage error, or a refused `--out`
(never under `models/` or `data/`).

**A cell's own exit code** (the HOLD line names it): `3` precondition · `4` the τ forcing was violated · `9` **`tau_cuda_parity` missed a pass bar** (the compiled rollout and the learner do not land on one τ on the card) · `8` **a learner
arm did not run the production compiled path** (its startup gate was not PASS) · `11` the X14 launcher exited non-zero ·
`12` the X14 run is not the arm that was specified · `124`/`137` wall cap · `86` OOM at the cell's cap · `5`/`6` `gpu_lock`.

The battery never stops, resumes or signals a training run. The caller pauses and resumes the run. On a HOLD, the
window is the orchestrator's call (`TRAINING_RUN_SOP.md` §3, rule 3). Useful moves after one: re-run the same line
(a failed cell re-runs, an `ok` one is skipped); `--learner-gate off` after exit 8 (re-runs the four learner cells,
+24 min); `--only x14_off,x14_on` to go on to the A/B without the learner pairs.

## The cells, in run order

Each cell runs as `gpu_lock.sh mem_cap.sh --name battery-<cell> <GB> timeout --kill-after=60 <cap> <cmd>`. A cell is
skipped on a re-run once its `cells/<cell>.state.json` reads `ok`.

| # | cell | what it measures | mem cap | wall cap | ETA |
|---|---|---|---|---|---|
| 1 | `tau_cuda_parity` | **τ on the card (REVIEW F2).** Read 1: the CUDA case of `tau_solve_test.py`'s compiled-vs-eager test (a pytest subprocess; leads 23 / 40 / 100, k = 1 / 3). Read 2: the compiled CUDA decision forward against the CUDA learner (`evaluate_actions`, train mode, eager), taken-action \|d log π\|, on X26's three τ-class violators (+ the scan's 32 worst, each event at its nearest checkpoint) and the 915 fixture rows. Read 3: τ itself, compiled vs eager, on those same rows' captured inputs. Section "The tau parity cell" | 16 GB | 25 min | 9 min |
| 2 | `tau_dec` | **Rollout.** T2's decision forward (`engine.decide`, inductor `dynamic=False`, CUDA graph: the `graph` backend's build) at buckets 64 and 256, `fp32_naive` (old) vs `fp32_alt` (new), in one process on the same seeded fresh build at X26's surface. 8 interleaved blocks of 50 replays per arm (ABAB). Also the agreement of the two arms: max \|d log π\|, max \|d V\|, greedy agreement. The method is `t62_dec`'s | 16 GB | 45 min | 16 min |
| 3-6 | `tau_u_naive_a`, `tau_u_alt_a`, `tau_u_alt_b`, `tau_u_naive_b` | **Learner.** The frozen suite's `update_bf16` cell (X26's argv, the K9 learner, the compiled bf16 `train()` at 384 × 256 rows, micro-batch 2,048 × accumulation 32, the 10-epoch estimate), the arm forced per process, run **ABBA**. Only the cell's name (and, see below, the path rule) differs from the suite; `reps` is the suite's 2 | 24 GB each | 40 min each | 6 min each |
| 7 | `x14_off` | **X14, the serial arm.** A fork of X26 `rb_x26_s2601` @ 380,000,195 through the real launcher (`x14_cell.py`), 15 updates, `--behaviour-check warn`, eval and ladder games off, at the battery's pin. It is the cold-compile arm | 40 GB | 75 min | 45 min |
| 8 | `x14_on` | **X14, the overlap arm.** The same argv plus `--overlap-collect`. Its compile cache is warm from `x14_off` | 40 GB | 55 min | 25 min |
| 9 | `t2_guard` | **T2's frozen guard** (pytest, `GEN3AI_TEST_ALLOW_GPU=1`, `-s`): `service_cuda_test.py`'s `test_expandable_segments_steady_state_...` (eager + graph; the 2026-10-19 cell's test on its fixed `new_mapping_floor` bar) and `test_overlap_attribution_books_only_t2s_own_streams` (expandable + default allocator: T2 flushing on a worker thread's own stream, the "learner" allocating 256 MiB on the default stream INSIDE a guard window). It must show **4 passed, 0 skipped: a SKIP is a HOLD**. The probe's JSON (incl. the worker stream's first-flush growth, recorded not judged) is in `logs/t2_guard.log` | 16 GB | 45 min | 12 min |

**Total ETA: about 125 min** = `tau_cuda_parity` 9 + `tau_dec` 16 + learner 24 + `x14_off` 45 + `x14_on` 25, plus about 5 min
of CPU checks between cells (the GPU idles for them); `t2_guard` adds about 12 (its ETA is the 2026-10-19 `frozen_guard`
cell's 10 plus the two attribution probes' one-bucket graph builds; UNVERIFIED until it runs). **Worst case at every wall
cap:** 360 min, 405 with `t2_guard`. With `--budget-min 135` the battery
stops cleanly before a cell that would overrun.

Why the ETAs are what they are:

- **Compile time is fixed.** On 2026-10-19 `t62_dec` took 15.8 min, 15.1 of it compiling; the `t62_u_*` cells took
  6.1-9.4 min, 4.2-5.8 of it compile + gate + prewarm. Fewer timed blocks saves almost nothing.
- **The X14 OFF arm's startup is long.** The first attempt's first process took **23 min** from launch to its first
  rollout (`~/gen3ai_archive/x14_ab_runs/rb_x14ab_off_s2601/launcher_child.log`: learner compile 281 s, the rest the
  inference service's buckets and pool seeding on a cold shared compile cache); a restart with a warm cache took 5.8 min.
  FP32_ALT changes the compiled τ graphs, so the OFF arm should be treated as cold again: 23 min + 15 cycles of about
  70-100 s (update ≈ 42 s + collect ≈ 27-55 s) = 41-48 min. The ON arm shares the cache.
- **The README's old estimate for the A/B** (≈ 10 min startup + 18 / 14 min of updates) assumed a warm cache.
- To save time at the cost of method, drop bucket 64 from `tau_dec` (about 7 min; it loses the SERIAL opponent bound) or
  run one learner pair with `--only tau_dec,tau_u_naive_a,tau_u_alt_a` (about 12 min; it loses the drift term in the band).

### Noise at these settings

- **Learner.** At `reps` 4 the two fp32 cells read 41.9 and 41.4 s (1.2 % apart), and each rep spread was 3 % or less.
  `reps` 2 takes the median of two, so expect about 1-2 % per cell. One percent of the update is about 0.56 % of the
  cycle. The ABBA pair disagreement puts this noise into the band.
- **Rollout.** The 2026-10-19 per-block deltas were tight: B64 3.35-3.78 ms, B256 14.94-15.06 ms. The SERIAL bound is
  the sensitive term. It multiplies the B64 delta by 20 opponent slots and 416.5 host steps, so 0.27 ms of B64 delta
  alone is 3 % of the cycle.

## The tau parity cell (`tau_cuda_parity_cell.py`, `tau_parity_read.py`)

**The question.** FP32_ALT's property is that "two programs land on one τ": T2's compiled rollout and the eager learner
used to land at different points of the naive sum's ulp(1) / slope ≈ 1e-2 band (the X26 τ class), and K9(b) read the 1e-3
gap. The proof says any summation order lands within 2c, and a fast `exp` is the same class of perturbation as the rounding
of x = a + τ. But every measurement was a CPU pair. The review (F2) asks for the CUDA pair; this is that cell.

**The reads** (one process tree, one card, ~9 min with the first Triton compile cold):

1. **Synthetic rows, as a test.** `pytest src/agents/model/tau_solve_test.py -k 'compiled_by_inductor and cuda'` with
   `GEN3AI_TEST_ALLOW_GPU=1 GEN3AI_SKIP_SLOW_STATUS_RECORD=1`, inside the cell's `gpu_lock.sh` / lease (the test file's
   CPU case was parametrized over `device`, the CUDA case is `slow`). Rows: 6 saturated rows, leads 23 / 40 / 100 nats,
   k = 1 / 3. Compiled Triton `fixed_size_tau` against eager, both on the card. The test writes its numbers to
   `$TAU_CUDA_PARITY_RECORD` before it asserts. The shared helper is `tau_solve_test.compiled_vs_eager_tau`.
2. **The network.** The compiled decision forward (`engine.decide`, inductor `dynamic=False`, bucket 256, the rollout's
   build) against the learner forward (`policy.evaluate_actions`, train mode, eager, fp32 `highest`, grad on, as the K9(b)
   probe runs it; the same 256-row chunks), on the SAME taken action's log π:
   - the 3 X26 τ-class violators, each dump (the violator + the scan's 32 worst) at its event's NEAREST checkpoint
     (`taus.EVENTS`: u32180 @ 316000231, u32320 @ 317000041, u35950 @ 353000151), the three **violators** being row 0;
   - the **915 fixture rows** (`play_rows(24, seed=6201)`, T62's) at P\* (X26 checkpoint_280000170), the taken action being
     the compiled forward's greedy legal action.
   The compiled function is built ONCE and the weights are loaded into it in place, so the one Triton build serves all sets.
3. **τ on the same rows.** Each chunk's species-presence τ inputs are CAPTURED from an eager CUDA decision forward
   (`taus.Tap`, call 0) and solved in one batch by the SAME helper as read 1.

**The bars (preregistered 2026-10-20 before any CUDA datum; deterministic comparisons of seeded inputs; `tau_parity_read.py`).**

| read | bar | why |
|---|---|---|
| 1 | every row ≤ **4 ulp** of τ | the CPU test's bar is 2 (measured 0 on every violator); Triton adds a per-thread serial prefix and a fast `exp` |
| 3 | every LIVE row (0 < k < n) of every set ≤ **4 ulp**, the ulp taken at the row's scale max(\|τ\|, max over candidates \|score\|) | the rounding of x = a + τ is at the scale of the largest \|x\|, not of τ (a τ near 0 has a tiny ulp) |
| 2 | the 3 violators' taken-action \|d log π\| ≤ **1e-4** | K9(b)'s fatal bar. The CPU analogue read 2.4e-6 / 6.7e-6 / 1.4e-5 |
| 2 | the fixture rows' taken-action **p99** ≤ 1e-4 | CPU p99 was 1.0e-5; the fixture MAX is reported, not judged (F4 of the FP32_ALT README: a non-τ fp32 tail at 4e-4 exists at some weights) |
| all | the forcing record clean (the compiled forward TRACED the wrapper, the learner / capture reached it eagerly); every value finite | a vacuous compiled arm is not a pass |

A miss is **exit 9, a HOLD**: the proof's worst case (F3) is 2c ≈ 4.8e-5 for ANY summation order, so a miss between the bar
and that is a number for the orchestrator to read, not a refutation of FP32_ALT. `read_battery.py` re-judges the stored
record (`TAU ON CUDA` section: PASS / FAIL / INCOMPLETE). It is not a cost, so it does not change the cost verdict; with
a FAIL or INCOMPLETE the cost verdict is a cost only, and "FP32_ALT lands on one τ on the card" stays UNVERIFIED.

**Reported:** τ's max |Δ| (absolute and in ulp), p99, and the share of rows bit-equal; per set the taken-action |Δ log π|
max / p99 / median and the count over 1e-4; the violators' own values; the fixture row count (the 915 of T62's fixture:
the record carries `fixture_rows` beside `fixture_rows_expected`, and a different count is a changed fixture, reported not
failed).

**What the cell cannot see.**
- The τ INSIDE the compiled decision forward is a graph-internal value. Read 3 solves the same inputs with the same
  function compiled STANDALONE, which can be fused and tiled differently. Read 2 measures the embedded programs through
  their log π.
- The rollout's CUDA graph is not captured: a graph replays the kernels it captured, so it cannot change the numerics. The
  cell calls the compiled function directly.
- The learner is the eager fp32 forward, not the compiled bf16-trunk R1 of training. K9(b) probes the same eager forward.
- **The CUDA path of this cell has not been executed** (the card was busy with X35 when it was staged); the CPU `--prove`
  ran every line except the Triton compile and the pytest CUDA case. See "Hazards".

**On a CPU the cell SKIPS cleanly:** no visible CUDA device ⇒ result status `skip`, exit 0, and the battery's reader treats
that as NOT ok (a window with no card never counts as a pass); the pytest CUDA case skips by its own `_cuda_skip_reason`
(naming why) and the cell reads "no record" as a failure.

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

## The learner fairness rule (`learner_path.py`)

**The builder's finding (`d4af9f1e`).** The compile sentinel's startup parity gate might REFUSE the naive arm, because
the K9 learner golden is now recorded under fp32_alt. The suite's rule for a refusal is that the cell "continues with
install + prewarm only". Then the two arms would run different code paths (a gated, locked, canary-armed R1 against a
bare one) and the cost comparison would be invalid.

**What the code does (checked 2026-10-20).**

- The gate (`compile_regions.gate_regions` → `r1_rungs`) compares **no recorded output** of the golden. It reads the
  golden's real labelled **rows as inputs**. It then computes the compiled R1, the eager R1 and the fp32 reference
  **live, in the cell's own process, under the cell's own forced solve, on the arm's own weights**. The clip relocation
  also reads the arm's own forward. Re-recording the golden under fp32_alt changed no input to it.
- The symmetric case already ran on 2026-10-19: the fp32 "old" arm against a golden recorded under fp64. **Both `old`
  cells' gates PASSED**, with the same lines as the fp64 arms
  (`/home/goodlad/gen3ai_archive/gpu_battery_2026-10-19/cells/t62_u_old_{a,b}.json`).
- So there is no golden-equality check to bypass and no per-arm golden to supply. A refusal is still *possible*: the gate
  fails an arm whose compiled and eager programs disagree on a golden row. **UNVERIFIED on the card** until the window.

**What the cells do now.** The guarantee no longer rests on the gate passing.

- Every learner cell records its compiled-path **markers** (`path` in its result): the gate verdict, and five lines the
  production `arm_compile_sentinel` prints (R1 installed, `parity PASS`, `inventory == declaration: R1 1 graph`,
  `prewarmed 1 production signature`, `CompileCanary armed`), plus whether the compile trace reached the τ wrapper.
- **A cell whose gate is not PASS FAILS** (exit 8, status `fail`): the battery HOLDs, naming the verdict and the marker,
  and no number is produced from a different path. The reader independently refuses a verdict (INCOMPLETE) unless all four
  cells ran the same mode with violation-free, **identical** markers.
- **The fallback, `--learner-gate off`.** It skips the gate by config in **all four** cells (`gate: "off"`, the suite's
  own switch). R1 is still compiled, installed and prewarmed in each, the timed `train()` is the same compiled graph;
  the lock and the canary are not armed. Identical by construction, and it keeps compile on. Its cost: the cells no
  longer test the fp32_alt gate on the card, and the numbers differ from the 2026-10-19 gated ones by the lock/canary
  bookkeeping (none of it inside a timed `train()`). A recorded `ok` of one mode is not reused under the other.

**The CPU proof (`--dry-run`).** `tau_update_cell.py --check --prove` runs the **real** `arm_compile_sentinel` per arm on
the CPU. R1's backend is `eager` (dynamo still traces `fullgraph=True` through the τ wrapper; no inductor build), on the
cell's own learner (seeded production-surface K9 build, bf16 trunk, the cell's argv). For each arm it asserts: gate
`PASS`, all five line markers true, the compile trace reached the wrapper, and the markers **equal the other arm's**.
The dry-run record is in "Dry-run record" below. It is a CPU stand-in: the CUDA gate's verdict is not proved by it.

## The X14 cells (`x14_cell.py`, `x14_read.py`)

**The recipe** (the first attempt's recorded command with the FP32_ALT pin and `--behaviour-check warn`):

```
--pin-commit <pin> --model <main>/models/rb_x26_s2601/checkpoints/checkpoint_380000195_steps.zip \
--steps 381474755 --device cuda --run-name rb_x14ab2_{off,on}_s2601 --restart-interval-hours 24 \
--eval-freq 1000000000 --snapshot-ladder-games 0 --behaviour-check warn [--overlap-collect]
```

`--steps` is absolute: 380,000,195 + 15 × 98,304. `GEN3AI_MODELS_DIR=~/gen3ai_archive/x14_ab_runs_2026-10-21`
(created by the cell; refused if under `models/` or `data/`). A run dir of the arm's name already there is **moved aside**
to `<name>.prev-<time>`, never deleted. The parent checkpoint is only read.

**Each arm is its own cell and its own process tree**, under its own `mem_cap` scope and wall cap. The cell starts the
launcher headless (stdin off a TTY), forwards a SIGTERM to it by PID, and reads the run after it exits.

**The pin.** The battery records main HEAD when its first cell starts. Both arms run at it; `cells/x14_pin.json` makes
the second arm refuse another commit. The cell refuses a pin that does not contain `d4af9f1e` (FP32_ALT: an update of
~42 s, not T62's ~97 s) and `02cac1a9` (the decoupled update normalises the raw advantage; `--overlap-collect` is a
recorded recipe row). The other cells run the checkout's working tree, not the pin; the reader warns if the cells ran at
different checkout commits.

**The read** (`x14_read.py`, printed by `read_battery.py`):

- **The unit is one dump** (one rollout end). A restart writes a new TensorBoard event file, so there is **one segment
  per process**. In each process the first 3 dumps are dropped ("the updates after the 3rd per process") and the rest are
  kept: 12 of 15 in an uninterrupted arm.
- **fps = the cycle's wall fps**: `(step_k − step_{k−1}) / (wall_k − wall_{k−1})` from the event wall times.
  **`time/fps` is not used to decide**: SB3 logs it as `(num_timesteps − at_start) / elapsed` since the process's
  `learn()` began (the archived OFF arm reads 1820, 923, 803 … 675 over 11 dumps), so over 15 dumps it is dominated by
  the first rollouts. Its ratio is printed beside the wall one (the old README's number).
- **The decision:** ratio = median wall fps(ON) / median wall fps(OFF); **≥ ×1.10 CLEARS** the bar for a learning A/B,
  below it is BELOW, within 1e-9 relative of it is AT_BAR (standing rule 8: the orchestrator decides). Fewer than 6 kept
  dumps in an arm is INCOMPLETE. The 25-75 % band is printed, and a band straddling the bar is said so.
- **Also printed:** OFF update, collect and the max hideable share of the cycle; ON update, the EXPOSED collect (the
  `rollout/collect_ms` of an ON run), `overlap/join_wait_ms`, `overlap/play_ms`, `overlap/prox_forward_ms`, IS ESS and
  truncated share, the T2 reserved-growth warning count, the update-peak reserved MiB.
- **The K9(b) section-7 ceiling read of each arm** (`main.x35.batch.judge_k9b`: the same function and ceilings as X35's
  AM-6) over the fork's **own** rows, all of them: a judged max ≥ 1e-3, a p99 ≥ 1e-4, any non-finite value or the child
  log's non-finite line **FLAGS** the arm. `warn` means report: **a flag does not fail the cell or HOLD the battery**.
  The OFF arm reads `behaviour/*`; the ON arm reads `behaviour_snapshot/*` (and `behaviour/*` where present). The
  count of judged rows at or over the fatal 1e-4 bar is printed. A read error is `UNREAD`, also a flag.
- **The cell FAILS (HOLD)** when the launcher exits non-zero (11), or the run is not the arm that was specified (12): it
  stopped short of 381,474,755, the ON arm logged no `overlap/join_wait_ms` (a vacuous arm), the OFF arm did, the fork
  step is not 380,000,195, or the run's recorded `pin_history` is not the battery's pin.

**What the dry run proves on the CPU** (`x14_cell.py --check --prove`): the pin carries both commits; the checkpoint and
the archive are valid; `main.checkargs` and `main.launcher --dry-run` accept each arm's exact argv (the launcher's dry
run against an empty scratch dir; nothing is launched); the reader on synthetic runs (burn-in per process, ratio, ceilings,
vacuity); and the cell's run machinery against a **fake launcher** (a script that writes a synthetic run dir): exit 0 for a
complete OFF and ON arm, 11 for a launcher failure, 12 for a short run, a never-overlapping ON arm, an overlapping OFF arm
and a wrong recorded pin, 0 for a K9(b) flag (reported, not failed), 143 with the SIGTERM forwarded, 3 for a second arm at
another pin, and the prior run dir moved aside.

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
   - **INCOMPLETE** if any of the five cells is not `ok`, lacks a clean `tau_force` record for its arm, or the four
     learner cells did not run the identical compiled path.

The reader also prints the agreement of alt vs naive, the compile sentinel's gate verdict per learner cell and the shared
path markers. fp64 vs fp32_naive is not re-measured: the 2026-10-19 battery's `t62_dec` is that read. The X14 read is
independent of this gate: either can be INCOMPLETE while the other reads.

## The `t2_guard` cell (added 2026-10-20, `gen3_t2_stream_attribution_v1`)

**PASS iff the junit shows 4 tests, 0 failures, 0 errors, 0 skipped.** It decides no production question; it is the
card proof two backlog steps wait on (`designs/ops/TECH_DEBT_BACKLOG.md`, the frozen-guard row): (1) the 2026-10-19
arming check on its fixed bar, and (2) the overlap attribution — (a) T2's own growth is still seen, (b) a learner
growth on the default stream inside a worker-thread flush window is booked FOREIGN, not to T2 (with
`expandable_segments` True and False). A PASS does NOT by itself clear `RESERVED_GROWTH_POISONS`: the orchestrator also
reads the probe's `worker_warmup` line in `logs/t2_guard.log` (the overlap worker stream's FIRST flushes; a non-zero
own growth there is a real T2 acquisition after the freeze that arming would make fatal).

## Hazards known when staging

- **The naive arm is forced through the wrapper, not checked out at bd3126e02^.** It runs the same arithmetic as the
  pre-T62 sum (`fixed_size_tau`'s `fp32_naive` branch). The traced graph may still differ from bd3126e02^'s by
  dtype-promotion no-ops. Its timing was not compared with the 2026-10-19 `old` arm.
- **The CUDA gate verdict for either arm is UNVERIFIED** (the CPU proof is a stand-in). The fairness rule turns a refusal
  into a HOLD, never a number.
- **The network is a seeded FRESH build at X26's surface**, not X26's weights. τ's cost is a fixed 64-step bisection on
  static shapes, so the weights do not move it. X35's architecture may differ from X26's.
- **The rollout bounds leave out the eval cycle and T2's host-side work.** The cycle denominator is X26's live cycle at
  258.6M (2026-10-18), not X35's.
- **The learner cells run without a CPU pin.** Any other CPU job on the box adds noise. The ABBA disagreement carries
  that noise into the band.
- **`tau_dec`'s ETA is the 2026-10-19 compile time.** `fp32_alt` has more ops per bisection step than the naive sum,
  so its compile may take longer. The 45 min wall cap leaves room.
- **The X14 cap `mem 40 GB` is a guess.** A production trainer's resident set is unmeasured here (the live toy arm reads
  5.6 GB). An OOM at the cap is exit 86, a HOLD; raise the cap in `battery.py` if it recurs.
- **The X14 arms are real training runs.** Any watcher that keys on trainer processes sees them. The launcher's startup
  prune only removes DEAD launchers' pinned worktrees (`designs/launcher/pinning_and_worktrees.md`), so the paused X35 arm's
  worktree is untouched.
- **X14's own caveats stand** ([`../x14_overlap_2026-10-19/README.md`](../x14_overlap_2026-10-19/README.md) §4): CUDA
  concurrency and GIL contention are unverified on a card; `RESERVED_GROWTH_POISONS` stays OFF (until 2026-10-20 T2's
  reserved-growth guard booked a learner reservation growth inside a worker flush as its own; at a pin carrying
  `gen3_t2_stream_attribution_v1` it books only T2's own streams, and the ON arm's T2 reserved-growth warning count is
  T2's own growth — the `t2_guard` cell proves the mechanism on the card); `rollout/collect_ms` on an ON run is
  the exposed collection; 15 updates give a throughput read, not a learning read.

- **The `tau_cuda_parity` CUDA path is UNEXECUTED when staged.** The CPU `--prove` runs the whole pipeline on the CPU
  with the compile backend `eager` (the real CPU inductor kernel only for read 3's shared helper), so a bug in anything
  CUDA-specific (a device mismatch, the Triton compile, the in-place weight load into a compiled function, memory at bucket
  256 beside a learner forward with grad on) would surface in the window as exit 1 / HOLD. Its mem cap (16 GB) and wall cap
  (25 min) are guesses from `tau_dec` (16 GB; compile ~3.8 min per build). It is first in the order so that a bug costs 25
  min at most. Its Triton compile is COLD (nothing warms it earlier), so `tau_dec` after it should compile faster.
- **`tau_cuda_parity` calls the compiled decision function directly, without a CUDA graph** (numerics identical by
  construction; **UNVERIFIED** on the card).
- **The fixture is rebuilt by playing battles** (`play_rows`, Rust env, seeded): the row count is recorded against T62's
  915 and a different count is reported, not failed.

## Dry-run record (2026-10-20, CPU, `CUDA_VISIBLE_DEVICES=""`, `nice -n 19 taskset -c 5-7,13-15`, under `mem_cap.sh 24`)

`run_battery.sh --out /tmp/... --dry-run` at `d4af9f1e2` + this change: **all checks ok in 404 s** (peak 1.04 GB), exit 0,
"runnable cells: ETA 110 min, worst case 335 min". What it showed:

- **The same compiled-path markers in both τ arms** (the real `arm_compile_sentinel`, CPU eager-backend stand-in, B16,
  bf16 trunk, the cell's learner): `fp32_naive` and `fp32_alt` each had `region_installed`, `parity_pass`,
  `inventory_one_graph`, `prewarmed`, `canary_armed` all True, gate PASS (116 s / 134 s), and the compile trace reached
  the τ wrapper in each. The `learner_path` rules refuse a REFUSED gate, a missing marker, an untraced τ, a gate-mode
  mismatch and gate-off-with-a-PASS verdict.
- **Both X14 arms' argv launch**: `main.checkargs` accepts them and `main.launcher --dry-run` says "would launch" for
  `rb_x14ab2_off_s2601` and `rb_x14ab2_on_s2601` (pin `d4af9f1e2` carries `d4af9f1e` and `02cac1a9`; checkpoint present;
  `--steps 381,474,755`; K9(b) `warn`; eval and ladder off). The archive dir is created at run time.
- **The cell's run machinery on a FAKE launcher**: exit 0 for a complete OFF and ON arm, 11 launcher failure, 12 for a
  short run / a never-overlapping ON arm / an overlapping OFF arm / a wrong recorded pin, 0 with a flag for a K9(b)
  ceiling (reported, not failed), 143 for a forwarded SIGTERM and for a launcher that ignores it (the cell kills its
  process group after the escalation), 3 for a second arm at another pin, and the prior run dir moved aside.
- **The reader** on synthetic runs: burn-in per process (9 + 12 kept dumps from 15 + 15), the wall-fps ratio (not the
  cumulative `time/fps`), the ceilings, the vacuity checks, AT_BAR; and on the archived 2026-10-20 OFF arm
  (`rb_x14ab_off_s2601`: 2 processes, 15 dumps, 9 kept, wall fps median 640 from a 153 s cycle = 98 s update + 55 s
  collect; K9(b) PASS).

**Not proved, and so UNVERIFIED until the window:** the CUDA verdict of either learner arm's gate; that a real launcher run
reaches the cell's post-run read (the fake launcher writes the same run-dir shape the first attempt's real one had);
`mem 40 GB` for a production trainer; `ON` arm behaviour on a card.

### Dry-run record: the `tau_cuda_parity` cell (2026-10-20, CPU, `CUDA_VISIBLE_DEVICES=""`, `nice -n 19 taskset -c 5-7,13-15`)

Three runs; the box was contended (the heavy-slot queue was 13 deep, so the FULL `run_battery.sh --dry-run` could not get a
slot in an hour and was withdrawn before it started):

- **`tau_cuda_parity_cell.py --check --prove`, under `mem_cap.sh 24` in a heavy slot: exit 0, 1.69 GB peak.** The
  verdict passes a clean record and refuses 8 planted misses; the pytest CUDA case is collected and **skips cleanly with no
  card** (`1 skipped, 30 deselected`, exit 0, no record, which the verdict reads as a miss); the cell run with no CUDA
  writes status `skip` and exits 0; the WHOLE network pipeline ran on the CPU (compile backend `eager`, bucket 32, 62
  fixture rows from 2 battles, the 3 events at their nearest checkpoints) in 620 s, with the shared τ helper compiling the
  real CPU inductor kernel on 115 live rows: τ max **0.31 ulp**, violators' |d log π| 9.1e-6 / 2.9e-6 / 2.9e-6, fixture
  p99 1.2e-5, the forcing record clean, verdict PASS on that record.
- **The battery dry run with the LIGHT checks** (every cell's `check` in place of its `dry_check`; the learner and X14
  `--prove` runs are the previous record's and this unit does not touch them): `all checks ok`, **runnable cells: ETA 119
  min, worst case 360 min** (was 110 / 335), the new cell first, its command
  `gpu_lock.sh mem_cap.sh --name battery-tau_cuda_parity 16 timeout --kill-after=60 1500 python3 -u tau_cuda_parity_cell.py --out …`.
- **The reader**: `read_battery.py --selftest` ok (including the parity verdict: PASS / FAIL on a planted violator /
  INCOMPLETE when absent or skipped), and the table plus the `TAU ON CUDA` section print from a synthetic record.
- **Static tier** (`utils.static_gates --paths`, 261 tests) green; `tau_solve_test.py`: 30 passed, 1 skipped (the CUDA
  case, with its reason), the CPU compiled case 2 ulp bar unchanged.

**Not proved, UNVERIFIED until the window:** the Triton compile and the CUDA pytest case; the in-place weight load into a
compiled function on a card; the cell's mem cap (16 GB) and wall cap (25 min); the 9 min ETA (the cold compile is
~4 min of it, from `t62_dec`'s 3.8 min per build).
