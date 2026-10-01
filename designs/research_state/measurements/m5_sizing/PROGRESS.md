# M5 SIZING STUDY — PROGRESS (resume point)

Registration: [`REGISTRATION.md`](REGISTRATION.md) (order constraint 5 of
`designs/endstate/program_rust_core.md`). Worktree `/home/goodlad/dev/gen3ai-wt/m5-sizing`, branch
`m5-sizing`. Lane J owns it (`src/main/rust_core_m5/` + this directory).

## Units

| # | unit | status |
|---|---|---|
| 1 | harness extension: `fanout.py --real-flushes` (the trainee / opponent split on REAL flushes) + `--t2-buckets` | SHIPPED `a73a8feb` |
| 1b | F-SZ-1 fix: the battery's untaught unit driver on main + its test | SHIPPED `bfb8e7e2` |
| 2 | the registration + its scripts (`bucket_rule.py`, `part_t.sh`, `pps_check.py`) | REGISTERED (this commit) |
| 3 | Part T: (T-c) update cost; (T-b)/(T-a) at N = 48, 128, 256, 512, 1024, 2048; (T-a′) at N\* | first pass DONE 08:36 (T-c dropped, D-1; 512–2048 BUSY, quiet re-runs after arm A, D-5); T-a′ after N\* |
| 4 | Part L: pre-flight at each arm's N; arms A, B, C, A′; U + G-A reads | pre-flight N = 48 PASSED; arm A RUNNING (08:51) |
| 5 | verdict: Decision records, endstate docs, `production_config.json` iff a production value changes, ledger | NOT STARTED |

## How to run

```bash
K=designs/research_state/measurements/m5_sizing
bash $K/scripts/part_t.sh update          # (T-c), once
bash $K/scripts/part_t.sh n 256           # (T-b) then (T-a) at one N; resumable via ~/.cache/gen3ai/tmp/sizing/part_t/status
bash $K/scripts/part_t.sh nstar <N*>      # (T-a') fewer active snapshots at N*
python3 $K/scripts/bucket_rule.py rule $K/results/split_n256.json
python3 $K/scripts/pps_check.py --out $K/results/pps_check.json
```

## Pre-registration reads (declared in the registration's header)

- Untaught-8 operating range (run with this study's then-copy of the driver, the same change as
  `bfb8e7e2`), 25 games / team, opponent N0 @24M: N0 @4.8M **24.0 %** (48/200);
  N0 @9.48M **45.0 %** (90/200). Rows in `~/.cache/gen3ai/tmp/sizing/pilot/rows/` (scratch).
- SmallRL guard operating range: N0 @9.48M, away, greedy, 100 games: **38 / 100** (regime verified).
- N = 256 instrument smoke of the throughput harness. It ran end to end; T2 startup at 4 buckets was
  327 s. Its throughput number is discarded.
- Noise-scale history of N0 / C_fix / K2 (REGISTRATION §6).

## Findings

- **F-SZ-1 (FIXED on main, `bfb8e7e2`):** the battery's `n0_endofrun_2026-09-27/scripts/gu_unit.py`
  called `untaught_meter._strip_debugger`, which `ecd2be00` deleted along with the
  ObservationDebugger, so every untaught unit died at load. The driver now uses the models as loaded,
  and `src/main/untaught_unit_script_test.py` pins it (routine: a static symbol check with teeth;
  `sim`: one real battle). Ten older measurement scripts under `designs/research_state/measurements/`
  (`arch_transfer_2026-09-05/*`, `teacher_content_2x2_2026-09-04/*`, `ext_first_team_audit`) still
  call it. They are era-pinned history, not live drivers, and are left as they are.
- **F-SZ-2 (a trap for any fork at N > 48, UNVERIFIED as a failure):** on a resume, SB3 restores the
  checkpoint's n_steps (2,048). `load(…, env=…)` then builds a rollout buffer of n_steps × N rows
  before `RustCollector._ensure_buffer` resizes it to target / N: 4.2M rows (≈ 46 GB of obs) at
  N = 2048. `np.zeros` is lazily paged, so it may never be resident. Not exercised here (the arms are
  FRESH), and not tested.
- **F-SZ-3:** on the Rust core one SB3 "vec-env call" is N trainee decisions. So the default
  checkpoint cadence (50,000 vec calls) is 50,000 × N steps: 102M at N = 2048, i.e. never in an 8M arm.
  The arms type `--checkpoint-every-steps 2000000`. The production-default fix is routed to another
  agent by the orchestrator.
- **F-SZ-4 (MAJOR, reported to the orchestrator 2026-10-01 05:30; K8's to fix):** (T-c)'s
  `compile_inventory --stage time --matmul-precision highest` on C's trained checkpoint FATALed in the
  R1 region gate: "the compiled TRAIN graph's gradient DISAGREES with eager on 7 parameter(s)", max
  per-param rel err 2.47e-3 (`history_events.itemtr_emb.weight`) against a bar of 1e-3. The
  extractor parity on 64 real rows passed in the same process at 1.15e-5 against the trained bar of
  0.2.
  - `compile_regions._r1_verdict` always uses the FRESH-weights bar (`_MAX_PARAM_GRAD_REL` = 1e-3).
    `compile_trainer` documents healthy trained-weight noise up to 9.3e-4 and keeps
    `_MAX_PARAM_GRAD_REL_TRAINED` = 0.2 for it, and the in-run canary uses that one.
  - K8's acceptance ran at TF32, where the TF32 rule applies instead.
  - Exposure: any fp32 launch whose R1 gate sees trained weights (fork, resume, launcher restart).
  - Miscompile vs bar mis-selection: UNVERIFIED.

## Deviations from the registration (declared as they happen)

- **D-1 (2026-10-01 05:30): (T-c) dropped** because of F-SZ-4. U_E10 for Part T's E2E rule is the
  median `train_ms` of arm A's updates 3..82 instead (§5.2's descriptor, the same quantity measured on
  the production path). **Consequence:** N\* is chosen after arm A has run, so the order is Part T,
  then A, then N\*, then B, C, A′. Arm A does not depend on N\*.
- **D-2 (orchestrator 2026-10-01, standing rule for Part L):** if an arm crash-restarts before
  F-SZ-4's fix lands and the restart FATALs on that cause, the arm is INCONCLUSIVE and is re-run
  (fresh) after the fix. It is never resumed with the gate weakened. The K8 owner queues GPU work
  between this study's units, so no hold may exceed its registered length.
- **D-3 (orchestrator 2026-10-01 ~07:10):** no learning arm launches before the R1 bar fix
  (`compile_regions._r1_verdict`, F-SZ-4) is on main. The K8 owner's noise control found that a FRESH
  fp32 launch at B = 2,048 also trips the old 1e-3 bar (2.4e-2 on unperturbed fresh weights), so arm A
  would FATAL at startup on the pre-fix main. Every arm is pinned at or after that fix and its pin is
  recorded here. If the fix has not landed when Part T ends, the study WAITS: it does not lower the
  gate and does not use `--no-compile-trainer`, because either changes the measured system.
- **C-1 (declared confound, orchestrator 2026-10-01, from the PPO-loop scoping agent):** the eval
  callbacks call `logger.dump(step)` mid-rollout. On every eval cycle (every 2M steps) that clears the
  previous update's `train/*` values, so (a) the KL→LR controller skips its step for that update, and
  (b) `train/*` points are stamped at the eval step. All four arms share it equally, so the arm
  comparisons stay fair. But no `train/*` series (`train_ms`, approx-KL, clip fraction, noise scale)
  is read as aligned at an eval-cycle step: the arm readers drop the update straddling each eval
  cycle. Size (orchestrator): ONE KL-controller reading is lost per eval cycle, ≈ 5 % of updates,
  identical across arms at matched eval cadence. Fixed by stage 2 of the PPO-loop work, after these arms.
- **D-4 (2026-10-01): arms RELEASED** by the orchestrator. `c0664251` (the R1 bars selected by weight
  regime: fresh 0.100, trained 9.9e-3, from a 25-state noise control) and `2b0793f5` (R0 converts
  the action masks to bool) are on main. Every arm is pinned at or after `2b0793f5`. Pins per arm are
  in the Part L table below.
- **D-5 (2026-10-01 08:30): BUSY Part T reads.** From ~07:40 a peer's pytest-xdist session (6+ workers,
  load1 10–12) shared the box. The N = 1024 serial blocks fell from 10.1–11.1k to 4.3–7.4k decisions/s
  inside one run. The harness's busy-box check ran only at the start (load1 2.4), so nothing was
  flagged. N = 512's late blocks also fell (10.2–10.5k to 8.4–8.8k), cause unknown.
  - The 512 / 1024 / 2048 reads are KEPT, labelled BUSY, and re-run as `part_t.sh abq <N>` after
    arm A. Each re-run waits (bounded) for load1 < 3 and holds the `gate_lock` slot (orchestrator).
  - The harness now judges the box PER BLOCK (`f9349f95`): a block with > 1 bystander CPU is re-run.
  - Owner addendum (DESCRIPTOR, not a registered arm), "one thread or the box?":
    - per-phase CPU (env-core CPUs during its step; the host main thread's share), from the same
      re-runs;
    - `part_t.sh threads 1024`: serial at 8 / 12 / 16 core threads.

## Part T — first pass (2026-10-01 05:31–08:36; `scripts/summarize_t.py`)

D = 98,304, 95 % self-play vs a 20-snapshot pool, T = 8 core threads, fp32, torch 2.8, 6 rounds × 20 s.
CIs are bootstrap 95 % over blocks (per arm) and over rounds (the ratio). **512–2048 are BUSY (D-5): not
quotable until the quiet re-runs.**

| N | buckets (rule) | serial decisions/s [CI] | overlapped | overlap / serial [CI] | ms / step | core ms | GPU wait ms | opponent share of step | slots / flush |
|---|---|---|---|---|---|---|---|---|---|
| 48 | 8, 48 | 3,880 [3,635, 4,038] | 2,811 | 0.726 [0.696, 0.780] | 11.4 | 2.0 | 7.3 | 59 % | 19.5 |
| 128 | 8, 16, 128 | 7,875 [7,548, 8,071] | 5,534 | 0.704 [0.688, 0.732] | 14.9 | 3.5 | 8.8 | 48 % | 20.0 |
| 256 | 8, 16, 32, 256 | 10,472 [10,430, 10,506] | 8,880 | 0.848 [0.845, 0.851] | 22.4 | 7.0 | 11.2 | 39 % | 20.0 |
| 512 (BUSY) | 8, 32, 512 | 9,468 [8,829, 10,103] | 10,598 | 1.123 [1.061, 1.198] | 49.9 | 22.3 | 19.4 | 36 % | 20.0 |
| 1024 (BUSY) | 8, 64, 512 | 8,570 [6,452, 10,349] | 12,832 | 1.564 [1.317, 1.917] | 122.4 | 73.4 | 32.8 | 17 % | 20.0 |
| 2048 (BUSY) | 8, 112, 128, 512 | 9,218 [8,453, 10,076] | 13,597 | 1.473 [1.449, 1.498] | 206.1 | 116.6 | 60.6 | 17 % | 20.0 |

- N = 48 on torch 2.8 reproduces Lane G's 2.5.1 read: 3,780 [3,650, 3,941]; overlap 0.712.
- The real-flush split at N = 48 reproduces Lane G's fan-out read: whole 8.63 ms, trainee part
  1.94 ms, opponents 6.69 ms over 19.5 slots.
- The bucket rule's second pass ran at 256, 512, 1024 and 2048 (the measured p95 differed from the
  provisional prediction). At 2048 the rule added a fourth bucket (128) to meet the time bound.

## Part L — arms

| arm | run | pin | launched | status |
|---|---|---|---|---|
| pre-flight N = 48 | `~/gen3ai_archive/m5_sizing_preflight/n48` | worktree at `f609e956` | 08:35 | PASSED: one update, `train_ms` 47 s (first update), peak RSS 13.7 GB, T2 up 164 s (31 slots: 24 pool + trainee + 6 eval) |
| A | `sizing_A_n48_e10_s1001` | `f9349f95` | 08:51 (GPU 08:56) | DONE 10:24, 8,062,355 steps / 82 updates; descriptors `results/arm_A_descriptors.json`; meters PENDING |

## D-6 — the MEMORY constraint on N\* (declared 2026-10-01 ~09:20, before any N\* > 48 is launched)

The orchestrator asked for this as a declared rule after arm A's first memory read.

**Arm A's first read (N = 48, `lifecycle/cuda_*` at update 5):** reserved 9,490 MiB = demand (peak
reserved), device free 1,200 MiB, ceiling (reserved + free − 512 MiB) 10,200 MiB, quiescent allocated
floor 851 MiB. `cuda_ooms` 0, alloc retries 0. The compositor holds ~0.5 GB of the 12 GiB card.

**Breakdown, approximate (from what is cheap; not a measurement of each part):**
- The learner alone, K8's memory read (`k6_k8/memory/`, TF32, no T2 in process): floor ~168 MiB,
  peak allocated ~4.0 GiB on a plain update and ~5.2 GiB on a diagnostics update, peak reserved
  ~5.6 GiB.
- Arm A's floor is 851 MiB, ~680 MiB above the learner's. That fits T2's 31 slots × ~16 MB (~0.5 GB)
  plus the staging arenas.
- The remaining ~3.9 GiB of reserved over the learner's 5.6 GiB is T2's captured-graph pool, the
  device-resident rollout batch (98,304 × 2,761 × 4 B ≈ 1.1 GB of obs alone) and allocator slack.
  How it splits among those is UNMEASURED.
- **Lever map:** the update peak (micro-batch activations, diagnostics probes) is the biggest
  single part. T2's share grows with the largest bucket, because one shared graph pool is sized by
  the biggest capture.

**The rule.**
- **(a) Startup + first-update memory PRE-FLIGHT at every candidate N\* before its arm.** This is
  REGISTRATION §5.1's pre-flight extended. It brings up everything the declared lifecycle acquires
  (T2 slots × buckets including b_tr, the eval slots, the learner with its prewarm and the
  device-resident batch) and runs one full update. Recorded: reserved, peak (demand), device free,
  floor.
- **(b) Headroom = ceiling − demand ≥ 1.0 GiB** (K6's 512 MiB margin, doubled). PUSHBACK on the
  suggested 1.5 GB: arm A at N = 48, today's production shape, has ~0.7 GiB under K6's ceiling and
  1.2 GiB device free. A 1.5 GB-free rule would reject production itself, so the rule is relative
  to what runs.
  - **A candidate must also not need more than +0.5 GiB of demand over arm A's N = 48 demand at its
    pre-flight**, so that it is no closer to the edge than production is.
  - The first eval cycle's memory is read off arm A at 2M steps. If an eval cycle raises demand,
    the margin in (b) absorbs it, and that number is recorded here.
- **(c) A candidate that fails** (a) or (b) is not chosen. N\* becomes the largest throughput-eligible
  N that passes, and the trade-off is REPORTED: fewer or smaller T2 buckets (b_tr < 512), bf16
  opponents (X19), fewer snapshot slots, or a smaller update micro-batch. No N is picked that only
  fits on paper.
- **F-SZ-5 (eval wall, 2026-10-01; O9's input):** `eval/duration_sec` (and the `[NNNs]` on the
  `[SELFPLAY EVAL]` line) is the SUM of per-unit durations (`selfplay_callback.py:730`, units timed in
  `rust_eval/executor.py:471`). The units run concurrently on the eval core's 64 envs, so arm A's
  354–392 s is not wall time.
  - The cycle's own line gives the wall: 900–1,000 games in 13.7–14.4 s.
  - The blocking cost, read off arm A's update cadence (TensorBoard wall times of `train/train_ms`):
    median update cycle 54.7 s; the cycles straddling the three eval steps 72.2 / 71.1 / 87.2 s.
    So eval plus its collection costs ≈ 17–33 s per 2M steps, ≈ 1.5–2 % of wall at N = 48,
    consistent with Lane H's 16.5 s.
  - The O9 table uses this wall delta, never `duration_sec`. The tag's name misleads (the
    orchestrator's first estimate read it as 27 % of wall). Renaming it is Lane H's call.
- **D-7 (orchestrator 2026-10-01 ~10:05):** the post-arm-A drain is extended from 45 to ≤ 80 min. Six
  short GPU units from other lanes (~70 min, several of them real trainer launches with env workers)
  run first, so the quiet Part T re-runs then go back to back on an idle box. "Drained" = the GPU lock
  free on 3 consecutive checks 60 s apart. Expected delay to Part T's end: ~+35 min.

### Arm A descriptors (N = 48, E10; `scripts/read_arm.py`, train/* points at eval steps dropped, C-1)

- `train_ms` median **41.0 s** [IQR 40.1–42.7] = **U_E10** (D-1). Update cycle median 56.1 s.
- KL controller: lr 3e-4 → 4.32e-4. approx-KL mean 0.011 (last 0.027). Clip fraction mean 0.16 (last 0.25).
- Staleness: current-version share 98.86 %, so stale **1.14 %**. Age-1 probe: |r − 1| 0.10, 22 % outside the
  clip band. Implied E[L²]/E[L] = 2D · 0.0114 / 48 = **46.9**.
  - Predicted stale share = N · 46.9 / (2D): 256 → 6.1 %, 512 → 12.2 %, 1024 → 24.4 %, 2048 → 48.8 %.
- Noise scale (policy B_noise, fresh 1–8M): 3.8k–9.2k (37k at 0.2M is EMA warm-up). Ratio policy 0.09 at 8M:
  over-batched at B_eff 65,536.
- Memory: demand 9,494 → 9,514 MiB; device free 1,196 → 1,176 MiB; 0 OOM / retries / segments after freeze.
  - **F-SZ-6 (memory, cause UNVERIFIED):** the floor steps ~+33 MiB per pool snapshot (851 → 882 at ~4.6M →
    914 at ~6.4M; promotions at 4.0 / 6.0 / 8.0M). A full window of 20 would add ~+0.65 GiB, which is
    today's headroom under K6's margin. Routed to the K8 memory audit.
- The K6 canary never ran (100-update cadence > 82 updates).
- Eval: four cycles, wall delta ≈ 17–33 s each (F-SZ-5).
- `rust_env/trainee_decisions_per_s` fell 7.9k → 2.6k as the pool filled (bots in-core early, then the
  T2 fan-out).
- **D-8 (2026-10-01 ~11:50): re-pin after main's bad commit `aebae9a1`** (11:23–11:29; it reverted ~10
  commits, repaired by `277f318f`). Nothing of this study was built or pinned in that window.
  - Every remaining arm and quiet re-run is pinned at `277f318f`, and the worktree is rebased onto it
    (Rust core rebuilt). The quiet chain now also re-runs N = 256, so every decision-relevant N
    (256–2048) is read on one tree. 48 and 128 stay at the first-pass tree (e6d69d5c-based), declared.
  - **Arm A's pin `f9349f95` precedes `f6b32f5a`** (eval-dump isolation: C-1 fixed). That commit is a
    REGIME BOUNDARY for live-KL-controller runs: A's controller lost 4 readings out of 82 updates,
    while B / C / A′ at `277f318f` lose none. **Proposed (default): re-run A at `277f318f` as
    `sizing_A2_n48_e10_s1001`, the control for B.** The old A is kept as a same-seed cross-pin
    descriptor.
- **D-9 (orchestrator 2026-10-01 ~11:55): A2 APPROVED.** Old A is a DESCRIPTOR only. **The A vs A2
  same-seed pair mixes the pin effect with CUDA nondeterminism.** The owned-loop agent showed
  byte-identity on the Rust core on CPU only, so the pair is not a pure pin read. Order: quiet re-runs
  → A2 → B → C → A′. The GPU lock is released between EVERY arm, with a ≥ 2 min gap so queued holders
  get it.

### Projected arm-boundary times (2026-10-01, updated as they move)

| unit | start | end (projected) | GPU lock released after |
|---|---|---|---|
| quiet re-runs 256 / 512 / 1024 / 2048 (relaunched after `95af710e`) + threads 1024 | 12:18 | ~13:20 | each unit |
| A2 (N = 48, E10, `277f318f`) | ~13:25 | ~15:10 | yes (2 min gap) |
| memory pre-flight at N\* = 256 (≤ 20 min hold) | ~15:15 | ~15:35 | yes |
| B (N\*, E10) | ~15:40 | ~17:10 | yes (2 min gap): **the switch-prep holds fit here (~17:10–17:50)** |
| C (N\*, E5) | ~17:55 | ~19:10 | yes |
| A′ (N = 48, E10, seed 1002) | ~19:15 | ~21:00 | yes |
- **Orphan unit (benign):** killing the first chain_q instance at 11:50 left its child `part_t.sh abq 512`
  waiting for the quiet precondition, and it ran at 12:06 on the rebased `277f318f` tree. The gate slot
  serialises it with the new chain's units, and the new chain skips 512 on its ok row. Lesson: kill a
  chain's process GROUP, not only the chain's bash.
- **F-SZ-7 (fixed, `95af710e`):** on `277f318f` the quiet re-runs at 512 and then 256 died at build with
  `LazyAcquisitionError: pool snapshot … was loaded onto cuda:0`. Since gen3_declared_slot_load_v1 a
  pool refresh is a DECLARED LOAD into its T2 slot, so the snapshot must stay on the CPU. The trainer
  does this, but the harness's `_build_rust_collector` passed the arm's device. Fixed to
  `device="cpu"`, pinned by an AST test. No timed quantity changes: slot loads happen at build. The
  quiet chain was relaunched at ~12:18 (`chain_q2.sh`), with A2 behind it.

## Part T — QUIET re-runs (2026-10-01 12:18–13:00, `277f318f` + `95af710e`; 0 BUSY blocks)

| N | buckets | serial decisions/s [CI] | overlap / serial [CI] | ms / step (serial) | core ms | env-core CPUs during its step (T = 8) | host main thread (cores) |
|---|---|---|---|---|---|---|---|
| 256 | 8, 16, 32, 256 | 10,303 [10,272, 10,335] | 0.855 [0.851, 0.860] | 22.8 | 7.2 | 7.0 | 0.68 |
| 512 | 8, 32, 512 | 10,248 [9,984, 10,415] | 1.060 [1.046, 1.085] | 45.8 | 17.9 | 7.3 | 0.61 |
| 1024 | 8, 64, 512 | 11,230 [11,133, 11,318] | 1.236 [1.209, 1.256] | 83.5 | 38.3 | 7.5 | 0.54 |
| 2048 | 8, 112, 128, 512 | 11,347 [10,989, 11,573] | 1.412 [1.385, 1.464] | 165.5 | 78.9 | 7.5 | 0.52 |

- The BUSY first-pass reads understated the large-N core (1024: 73 → 38 ms; 2048: 117 → 79 ms).
- **The owner's question:** the 8-thread core pool runs at 7.0–7.5 CPUs during its step, i.e.
  SATURATED. The host main thread uses 0.5–0.7 core. So at large N the pipeline is core-thread-bound
  and serial, not box-bound. The threads read (8 / 12 / 16) follows.
- **N\* by the registered rule** (U_E10 = 41.0 s from arm A; r = the better of serial and overlapped):
  - E2E = 1,482 (48), 1,838 (128), 1,946 (256), 1,964 (512), 2,045 (1024), 2,086 (2048).
  - Threshold 0.95 × 2,086 = 1,981.
  - 512 misses by 0.9 % in E2E, which is 5.1 % in r, outside the 3 % null band.
  - 1024 passes, with predicted staleness 24.4 % ≤ 25 %. 2048 is ineligible on staleness (48.8 %).
  - **N\* = 1024, subject to D-6's memory pre-flight.**
- **FINDING for the verdict:** 1024's margin is OVERLAP (O5: 1.236 [1.209, 1.256] ⇒ ADOPT at 1024).
  The overlapped collector is a harness arm only, not yet a training path. On SERIAL rates alone,
  E2E = 1,945 / 1,940 / 1,976 / 1,986 (256 / 512 / 1024 / 2048), which would give N\* = 256.
  Arm B at 1024 measures learning per sample correctly (overlap does not change the rows), but it
  collects serially in training.

- **D-10 (orchestrator decision, 2026-10-01 ~13:20): N\* = 256 for arms B and C.** This is a declared
  deviation from the registered rule's definition of r. (The orchestrator's message called it "D-8";
  D-8 was already used for the re-pin.)
  1. **r must be a rate the TRAINING path can achieve.** The overlapped collector exists only in the
     harness, and training collects serially. The registered rule on SERIAL r gives N\* = 256:
     E2E 1,945 against a max of 1,986 (0.95 bar = 1,887).
  2. **Even with overlap, 1024 buys ~5 % E2E** (2,045 vs 1,946), for 24.4 % predicted staleness
     against a 25 % cap. That is 4× the stale share at 256, and in the fresh regime 22 % of stale rows
     already fall outside the clip band at 1.14 %. It also costs more memory against a ~1 GiB margin,
     and it needs a training-path overlapped collector that does not exist.
  3. **The update dominates:** U_E10 = 41 s against ~9.5 s of collection per 98,304 rows at
     N = 256. Past 256, more envs barely move E2E. The throughput lever is the UPDATE: epochs (arm C)
     and the batch (B_eff 65,536 is ~7–15× the fresh run's B_noise of 3.8k–9.2k).
  4. **1024 + the overlapped collector is recorded as a FUTURE option** (O5 = 1.236 [1.209, 1.256] at
     1024). Prerequisites: build the overlapped training collector, a staleness correction, and the
     memory headroom. Not adopted now.

  The memory pre-flight therefore runs at 256, then B (256, E10), then C (256, E5). A2 and A′ are
  unchanged. Buckets at 256: (8, 16, 32, 256).
- **D-6 CORRECTION (2026-10-01 ~13:30, before any candidate's memory was read).** As written, (b)
  "ceiling − demand ≥ 1.0 GiB" is mis-specified: K6's ceiling (reserved + device free − 512 MiB) already
  subtracts K6's own 512 MiB margin. Arm A at N = 48 reads ceiling − demand = 10,177 − 9,514 = 663 MiB,
  so the rule as written would reject production itself, which is exactly what the pushback meant to
  avoid. **Intended and now applied: ceiling − demand ≥ 512 MiB**, i.e. ≥ 1 GiB of device headroom at
  peak (K6's margin doubled). Arm A passes at 663. The relative clause (demand ≤ the N = 48 control's
  demand + 512 MiB) is unchanged. Encoded in `scripts/mem_rule.py`.
  - The pre-flight now runs 5 updates (491,520 steps): a 1-update pre-flight logs no `lifecycle/cuda_*`
    sample.

### The owner's "one thread or the box?" — threads addendum (N = 1024, serial, quiet, 6 rounds; `results/throughput_threads_n1024.json`)

| core threads | decisions/s [CI] | vs 8 [CI] | core ms / step | env-core CPUs during its step | host main thread |
|---|---|---|---|---|---|
| 8 | 11,228 [11,159, 11,300] | 1 | 38.4 | 7.4 | 0.54 |
| 12 | 11,489 [11,199, 11,667] | 1.023 [0.993, 1.041] | 35.4 | 10.2 | 0.57 |
| 16 | 11,671 [11,610, 11,738] | 1.040 [1.035, 1.045] | 33.2 | 12.1 | 0.59 |

**Reading (DESCRIPTOR):** the core's thread pool is saturated at 8 threads, but it does not scale past
them. 2× the threads buys −14 % core time and +4 % decisions/s; the CPUs used rise 7.4 → 12.1 for that.
On this 8-core / 16-thread box the hyperthreads add little to the core step (SMT, cause otherwise
UNVERIFIED). The host thread is far from saturated (0.5–0.6 core). So at large N the pipeline is bound
by the env core's per-step work on 8 physical cores plus the serial GPU wait. It is neither a single
host thread nor the whole box. More core threads are not a lever worth taking (+4 %). Overlapping the
core step with inference (O5) and a cheaper core step are.
