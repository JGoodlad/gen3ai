# T24: training throughput over a run, and what a periodic restart buys (2026-10-16)

The question is backlog item **T24** (`designs/ops/TASK_BACKLOG.md`): the launcher kills and relaunches the training
child every `--restart-interval-hours` (6 h since the owner's 2026-10-04 ruling; it was 3 h). The restart was
written to reclaim Python allocator (pymalloc) fragmentation in the old Python env core. Does it buy anything on the
Rust env core, and what does it cost?

This was a READ-ONLY analysis: no GPU, and no change to `models/` or `data/`. The live X26 run (`rb_x26_s2601`) was
only read: its TensorBoard events, its logs, its shared compile-cache stamps and `/proc/<pid>/status`. All CPU work
ran under rule 5b: `taskset -c 5-7,13-15`, `nice -n 19`, one job at a time.

## Bottom line

1. **Over a run, the UPDATE does not drift at all; the ROLLOUT gets slower, for reasons a restart cannot touch.**
   - **The update does not drift.**
     - The update time (`train/train_ms`) is flat to within ±3 % from the first hourly bin to the last, in all 77
       processes that ran > 1 h (median change 0.0 %).
     - It held 41.9-42.2 s on X26 for 6 h, and 9.2 s on the toy runs for 5.5 h.
     - There were no recompiles after the compile lock in any process. The card's reserved memory did not move
       (+0 MiB median, +2 MiB max).
   - **The rollout ("collect") gets slower.** On X26 the collect time per 100k steps went from 9.9 s to 27.4 s in
     6 h, and fps fell from 1,888 to 1,417.
   - **The causes are all tied to the TRAINING STEP, and none to the process's age.**
     - **The self-play switch-on** at ~4M steps: games get longer and an opponent network must run. It adds
       +5.5 s per 100k steps.
     - **The self-play pool's size.** Each frozen snapshot is one more opponent network to serve on the GPU.
       - The rollout's GPU wait (`gpu_wait`) rises by ~1 ms per host step per snapshot: 20.4 ms at 1 snapshot,
         29.5 ms at 10, 39.0 ms at 16-17. The pool's cap is 20.
       - The toy runs, which had no restart, show the same shape: fps falls until the pool reaches 20 at ~45M steps,
         then stays FLAT for the remaining ~3 h.
     - **The snapshot-ladder updater** (F1). This is a detached CPU job, started at every promotion, that plays the
       new snapshot against EVERY snapshot in the pool.
       - Its run time grows with the pool: 114 s at 1 pair, 1,109 s at 10, 1,756 s at 14.
       - Rollouts collected while it runs are slower (+16 % like-for-like up to 22M).
       - **Since ~28M it outlives the gap between promotions.**
2. **A restart recovers nothing measurable.** X26's first timed restart (09:09, ~33M steps) is a clean paired read:
   - **Fps.** Steady fps was 1,417 in the hour before and 1,364 in the ~45 min after (medians, eval cycles excluded).
   - **Update time.** 42.13 s → 42.23 s.
   - **Env core's CPU time.** 20.5 → 20.4 ms per host step.
   - **GPU wait.** It ROSE 34.6 → 39.0 ms, because the pool grew from 14-15 to 16-17 snapshots across the seam,
     which is the slope above.
   - **Host memory.** The trainer's resident memory (RSS) was FLAT at 9.49-9.71 GB from 4.1 h to 6.1 h of age (a
     +0.4 MB/h fit over 56 samples). The fresh child settled HIGHER, at 11.4 GB after 25 min.
   - **Process age at a matched step.** A matched-step comparison over 16 runs found no effect.
     - The bf16 A/B arms ran four seeds as two processes and four as one.
     - At the same training step, a process ~1 h older was −1.5 % / −1.8 % slower (median over ten 1M-step bins,
       per arm).
     - That is inside the ±3 % bin-to-bin noise, with identical GPU wait and update time.
3. **A restart costs ~5.5 min when its compile cache hits; on X26 it cost 22.5 min because it did not (F8).**
   - **X26's restart end to end: 1,352 s.** That is from the old child's last rollout (09:12:42) to the new child's
     first rollout (09:35:14).
     - The interval-triggered stop at a rollout boundary, the save and the relaunch: ~70 s.
     - The new child's startup: 1,282 s, against 265 s at the same run's launch at 03:09.
   - **The extra ~17 min was a full COLD compile**, caused by a cache-key defect (F8).
   - **What a warm restart costs.** It should cost ~70 s + ~265 s ≈ **5.5 min, 1.5 % of a 6 h interval**, ≈ 1.3 h over
     X26's ~14 restarts. A restart that misses the cache costs 6.3 %.
4. **Restarts did not visibly disturb learning.** Across 14 seams × 6 instruments (83 jumps):
   - the jump across a seam was a median 0.46 of the pre-seam spread;
   - only one exceeded 2: an episode-length jump of 2.43 after a crash rolled the run back 550k steps;
   - X26's largest jump was 1.28, in clip fraction.

   That is what chance gives at this count.
5. **Recommendation to the owner** (the interval is his call): **lengthen the interval to 24 h**, or set it to 0 once
   T7's memory-triggered restart exists.
   - **Why:** a restart buys nothing measurable and costs ≥ 1.5 % of the wall at 6 h (6.3 % on a cache miss). At
     24 h it costs ~0.4 %, while a backstop still bounds any slow leak not yet seen.
   - **Evidence strength: MODERATE.**
     - The "no benefit" side rests on one paired production restart at 6 h of age, 2 h of flat RSS before it, the
       toy runs (5.5 h with no restart, flat after the pool's cap), and the matched-step comparison (≤ 3 h processes).
     - No production process has yet run past 6.1 h.
   - **Fix F8 regardless:** it makes the FIRST restart of every `--arch production` run a ~20-min cold compile.
   - **The bigger throughput lever is not the restart but the ladder updater (F1).**

## Data

- **The runs:** every `models/rb_*` run with TensorBoard events, 97 runs and 118 child processes (116 reached a
  rollout).
  - One event file is one child process. A fork's inherited parent files are excluded.
  - Series read: `rollout/collect_ms`, `train/train_ms`, the Rust env core's per-host-step phases
    (`rust_env/*_ms_per_host_step`), `rollout/ep_len_mean`, `train/selfplay_fraction`,
    `eval/pool_snapshot_count`, `eval/wall_sec`, the compile and CUDA lifecycle scalars, and the learning instruments.
- **The startup lines:** each child's lines in `launcher_child.full.log` (`reset + prewarm took`,
  `dry first update`, the DiskGuard compile-cache line, how the child ended).
- **X26 only:**
  - `snapshot_ladder/games.jsonl`, the updater's timestamps;
  - the shared compile cache's entry stamps (`~/.cache/gen3ai/compile_shared/*/stamp.json`, read only);
  - `/proc/<pid>/status` of the old (209527) and new (401777) trainer child, sampled every 2 min
    (`rss_sampler.sh` → `rss_x26.log`).

**Before X26, no timed 6 h restart had happened in the Rustboro era.** Every multi-process run is one of three kinds:
- a human second leg: the bf16 A/B seeds 2001-2004, 0→5M then 5M→15M, launched 9-15 h later;
- a crash restart: `rb_st_static_*`, `rb_hib_moves_s3101`, `rb_es_end_s2001`;
- the one manual SIGTERM: `rb_toy_ent_s4001` at 3.8M.

Every process before X26 lasted ≤ 3.4 h, except the toys at 5.5-5.7 h. X26's first child ran 6.1 h and was restarted
on the interval at 09:09.

### The cycle, and the eval inside the collect

**What a cycle is made of.** Between two consecutive `collect_ms` events the wall is `collect[i] + train[i-1]`; the
remainder is ~0.04 s median.

**The in-loop eval runs INSIDE the collect window.** The eval cycle's `collect_ms` is 20-40 s above its neighbours.
- Each eval cycle is therefore marked by its `eval/wall_sec` event and excluded from the steady statistics.
- The eval's share of the wall is 1.9-3.5 % per hour on X26 and 3.2-4.2 % on the toys, which agrees with T33's
  ~3.5 %.

SB3's `time/fps` is cumulative since the process started, so it is not used.

## Results

### The fps trajectory over a run (question 1)

X26 (`rb_x26_s2601`, production, pin `9c033489`), per 1M steps. Collect is seconds per 100k steps; gpu_wait and core
are ms per host step:

| steps | process age | pool | collect | gpu_wait | core | ep len | train s | fps |
|---|---|---|---|---|---|---|---|---|
| 0-1M | 0.1 h | 0 | 9.9 | 8.3 | 11.0 | 32.8 | 41.9 | 1,888 |
| 4-5M | 0.7 h | 1 | 18.0 | 20.4 | 14.1 | 36.7 | 42.1 | 1,639 |
| 10-11M | 1.8 h | 4 | 17.9 | 18.2 | 16.4 | 41.3 | 41.9 | 1,648 |
| 16-17M | 2.8 h | 7 | 19.9 | 22.1 | 16.9 | 41.5 | 41.9 | 1,592 |
| 20-21M | 3.5 h | 9 | 22.8 | 28.2 | 17.3 | 41.8 | 41.9 | 1,525 |
| 23-24M | 4.1 h | 10 | 24.1 | 29.5 | 18.6 | 43.8 | 42.1 | 1,480 |
| the last hour before the restart (~29-33M) | 5.1-6.1 h | 14-15 | 27.4 | 34.6 | 20.5 | 42.9 | 42.1 | 1,417 |

What each column does:
- **The update** never moves.
- **The episode length** rises from 30 to 42 decisions with the self-play switch, then is flat.
- **gpu_wait** tracks the pool's size (+1.0 ms per snapshot), not the process's age (next sections).
- **The env core's CPU time** alternated by 1M-step bin up to ~24M, at ~11-12 ms then ~16-18 ms. The slow bins are
  the ones just after a promotion, while the ladder updater is playing. Once the updater runs continuously (F1) the
  alternation stops at ~20 ms.

**The toy runs, with no restart.** `rb_toy_ent_s4002` / `s4003` ran 100M steps in 5.5-5.7 h:

| steps | age | pool | fps (s4002) |
|---|---|---|---|
| 0-5M | 0.1 h | 0 | 6,174 |
| 20-25M | 1.2 h | 9 | 5,328 |
| 40-45M | 2.3 h | 19 | 5,028 |
| 60-65M | 3.5 h | 20 | 4,900 |
| 80-85M | 4.6 h | 20 | 4,986 |
| 95-100M | 5.5 h | 20 | 5,103 |

The toy's fps is flat from the pool's cap at ~45M to the end, 3.2 h later, with the update unchanged at 9.2 s. This is
the cleanest evidence of no age-driven drift. The caveat: the toy's GPU-side opponent cost is small (gpu_wait
2.4 → 4.9 ms).

**Candidate causes, with the evidence for each:**

| candidate | verdict | evidence |
|---|---|---|
| games get longer | real, early, one-off | episode length 30 → 42 at the self-play switch (~4M), flat after |
| the self-play pool's opponent inference | **real, grows until the pool's cap of 20** | gpu_wait +1.0 ms per snapshot (X26); toy flat after the cap; unchanged by X26's restart |
| the ladder updater (detached, CPU) | **real, grows with the pool** | collect +16 % while it runs (up to 22M); its wall 114 → 1,756 s (F1) |
| eval pauses | real, small | 1.9-3.5 % of the wall (in-loop); the T33 figure holds |
| host memory growth / fragmentation | **not detected** | update flat; matched-step age effect within noise; X26 RSS flat over 4.1-6.1 h of age (+0.4 MB/h); the restart left fps unchanged after the pool's share is taken out |
| compile recompilations | **none** | `compile/recompiles_after_lock` 0 in every process-hour |
| thermal / clock | not measurable from the record | no `nvidia-smi` log exists; the update, which is GPU-bound, is flat to ±3 %, so a sustained clock loss would have shown there |

### Process age at a matched step (question 2's evidence) — `age.json`

**The comparison** (`age_vs_step.py`). In `rb_bf_e_*` and `rb_bf_eb_*`, seeds 2001-2004 restarted at 5M and seeds
2005-2008 did not. At 5-15M the single-process seeds are ~1 h (bf_e) or ~0.8 h (bf_eb) older, at the same step,
recipe and pool.

| arm | single-process fps vs restarted, median over 10 bins | gpu_wait, single vs restarted | train s |
|---|---|---|---|
| bf_e | −1.5 % (bins: −2.9 … +3.3 %) | equal within 1 ms in every bin | equal |
| bf_eb | −1.8 % (bins: −5.3 … +0.5 %) | equal within 1 ms | equal |

**Reading it.** The small deficit is in the env core's CPU time. That time oscillates between ~10 and ~15 ms from bin
to bin in the restarted group, and sits at ~14 ms in the single group. Box co-load explains it better than age does.
**NOT DETECTED** at a ±3 % floor.

**A regression that does not settle it.** An OLS fit over 6,807 steady self-play cycles in 71 runs, with the run as a
fixed effect:
- **The pool's coefficient:** collect per 100k steps rises +0.45 s per snapshot.
- **Process age's coefficient:** **−1.49 s per hour**, the wrong sign for a leak.
- **Why it does not identify an age effect:** age and pool size are collinear within a process, and the iid standard
  errors are optimistic over autocorrelated cycles. Reported only because the sign is not the one a slowdown with
  age needs.

### The restart's own cost (question 2)

| component | measured | source |
|---|---|---|
| compile prewarm, same-run restart | **71.5 s median** (54.7-89 s warm; cold: 263.9 s `rb_hib_moves_s3101`, 255.4 s X26 (F8)) | 19 restarts, `summary.json` |
| compile prewarm, first launch | bimodal: 17 of 93 ≤ 120 s (the shared cache already held the key), 76 cold at 165-420 s | `summary.json` |
| dry first update (the update-fit gate) | 13.9 s median (12.8-18.3) | 111 launches |
| the first 3 cycles' excess over the steady cycle | −20 … +18 s (noise) | seams |
| old child's last rollout → new child's first rollout | 209 s median over 12 launcher seams: 103 s (the toy's SIGTERM), 150-260 s (crash restarts, the crash included), **1,352 s (X26's timed restart, a cold compile: F8)** | `summary.json` |
| X26: child attach → first rollout | 265 s at the launch (warm shared cache) vs 1,282 s at the restart (cold) | log + TB |
| steps lost | 0 on a SIGTERM restart (it saves at the safe point); 0.26-0.84M on a crash (it resumes from the last 1M checkpoint) | seams |

### X26's first timed restart

**What happened.**
- The log reads `[ABORT] restart interval (21600s) elapsed — clean rollout-boundary restart`, then a save of
  `final_model_interrupted.zip` at 32.94M steps.
- The new child attached at 09:13:52, and its first rollout started at 09:35:14 at 33.14M. No steps were lost.

**Before vs after** (`result.json`, the X26 seam). Before is the last hour; after is the ~45 min following the first 3
cycles.

| | before | after |
|---|---|---|
| steady fps | 1,417 | 1,364 |
| collect s / 100k | 27.4 | 29.9 |
| update s | 42.13 | 42.23 |
| gpu_wait ms / host step | 34.6 | 39.0 |
| core ms / host step | 20.5 | 20.4 |
| pool snapshots | 14-15 | 16-17 |
| trainer RSS | 9.49-9.71 GB, flat over 4.1-6.1 h of age | 11.4 GB from 25 min on |

**Why the startup compiled cold (F8).**
- **What the key reads.** The run's shared compile-cache key (`compile_cache_shared.plan` → `run_fields(args)`)
  hashes `COMPILE_ARG_KEYS`, which include `batch_size` read from the ARGV.
- **Why the value changed.**
  - On the fresh launch, `--arch production` supplied the recipe's `--batch-size 2048`.
  - A restart relaunches in the RESUME role, which strips `--arch` (`checkpoint.resume_child_args`). So
    `args.batch_size` falls back to the parser's default of 4096 (`main/train/parser/hyperparameters.py:13`).
  - The trainer itself INHERITS the saved 2048: on a resume `--batch-size` is inert.
- **The evidence.** The launch's entry `ae959cd1…` and the restart's `5d99a269…` differ in exactly one field,
  `run.compile_args.batch_size`: 2048 vs 4096.
- **The result.** The shared cache missed, and the restart rebuilt the whole set.
  - It wrote 6,613 Triton kernel dirs against the launch entry's 6,615, and 979 Inductor dirs against 983, compiling
    from 09:14 to 09:34.
  - The run's log shows `compile cache 2.39 GiB = 2.39 GiB - 0.0 MiB already on disk`.
- **The next restart.** The entry now exists, so X26's next restart should hit it. That assumes the 20 GiB LRU prune
  does not evict it between children; it is leased while a child is live. That it survives is UNVERIFIED.

### Restarts and learning (question 3)

**The method.** On 14 seams with ≥ 10 updates on each side, compare the mean of the last 5 updates before the seam
with the first 5 after it, in units of the standard deviation of the last 20 before it:

| | |
|---|---|
| tags | approx_kl, entropy loss, clip fraction, value loss, `behaviour/excluded_frac` (K9(b)), episode length |
| jump/sd | median 0.46, max 2.43; 1 of 83 > 2 (X26's largest: clip fraction 1.28) |

**Reading it.** The only value > 2 is the episode length after `rb_st_static_s1001`'s crash rolled the run back 548k
steps. Eval win rate cannot be compared at a seam, because there is one eval per 2M steps.

**FINDING.** A resumed child's startup K9(b) gate reports a BEHAVIOUR-POLICY MISMATCH on the dry update's fixture
rows. For example, `rb_bf_eb_s2001`'s second leg read max |Δlog π| 0.0279, excluded share 0, non-fatal. The first
rollout's own K9(b) is clean (`behaviour/excluded_frac` is continuous across the seam). Not investigated here.

## FINDINGS (rule 7)

- **F1: THE SNAPSHOT-LADDER UPDATER NOW OUTRUNS ITS PROMOTION INTERVAL ON X26, AND OVERLAPS ITSELF.**
  - **What it does.** `selfplay_callback._spawn_snapshot_ladder_update` spawns one detached CPU job per promotion.
    `snapshot_ladder.promotion_plan` plays the new snapshot against EVERY snapshot in the pool.
  - **Its wall on X26** (`ladder_x26.json`):

    | pairs | 1 | 10 | 12 | 13 | 14-15 |
    |---|---|---|---|---|---|
    | wall (s) | 114 | 1,109 | 1,362 | 1,622 | 1,735-1,756 |

  - **Against the gap between promotions** (1,135-1,561 s): from ~28M steps on, each updater outlives the gap. At the
    pool's cap (19 pairs) a linear extrapolation gives ~2,200 s.
  - **Its effect on the trainer:**
    - It is not niced or pinned, so it competes with the Rust env core on the trainer's cores.
    - Collect was 24.1 s per 100k steps with an updater running vs 17.6 s idle, over the whole run (a figure that
      mixes in the pool's growth). Like-for-like up to 22M: +16 %.
    - The cost of overlapping updaters is UNVERIFIED.
  - **ESTIMATE:** X26's fps falls from ~1,400 now to ~1,300 at the pool's cap (~42M steps), then is flat.
  - **Options for the orchestrator**, all out of this unit's scope: nice and pin the updater, cap its round-robin to
    the K nearest snapshots plus the sentinels, or serialise the updaters.
- **F2: no production process has run past 6.1 h** (X26's first child). A slow leak that only shows after 6 h cannot
  be excluded; lengthening the interval is how to observe one.
- **F3: no run records host-memory telemetry.** There is no RSS scalar in TensorBoard or in the log. The RSS here is a
  2-min `/proc` sample of X26 alone: 4.1-6.1 h of the old child and the first ~55 min of the new one. T7
  (memory-triggered restarts) needs an RSS scalar first.
- **F4: no thermal or clock record exists** (no run logs `nvidia-smi`). Clock throttling is excluded only indirectly,
  by the flat update time.
- **F5:** the startup K9(b) mismatch on a resumed child's dry-update fixture rows (question 3).
- **F6: the pre-era side note was not measured.**
  - Pre-Rustboro runs restarted every 3 h on the Python env core.
  - The 2026-06-05 memo (`research_state/memory_archive/project_env_worker_memory_leak.md`) found their "memory
    explosion" was a retained-object leak, not fragmentation. It was fixed at the source; the restart had been masking
    it.
  - Those runs' TensorBoard was not re-read: a different env core, which the brief ranks as a side note only.
- **F7: `eval/duration_sec` is not a wall.** Its 345-832 s on X26 are the SUMMED unit time of the eval shards
  (`eval_collect.py:290`). Read alone, it overstates the eval's cost ~20×; the wall is `eval/wall_sec`.
- **F8: THE SHARED COMPILE-CACHE KEY READS AN INERT ARGV VALUE, so the FIRST restart of an `--arch production` run
  compiles cold.**
  - **The defect.** `run_fields(args)` hashes `args.batch_size`. On a restart that is the parser's default 4096,
    while the launch read 2048 and the trainer inherits 2048 from the checkpoint.
  - **The cost on X26:** +17 min of startup (1,282 s vs 265 s), +4.7 % of that 6 h interval. The miss also writes a
    second 3.3 GB entry per run into the 20 GiB shared cache, which raises LRU pressure.
  - **Fix direction (not done here).** Key from the EFFECTIVE values the trainer runs with, after the resume's
    inheritance from `model_config.json`, or drop the keys that are inert on a resume. Every key in
    `COMPILE_ARG_KEYS` is a candidate for the same split; only `batch_size` differed on X26.

## Reproduce

```bash
D=designs/research_state/measurements/fps_and_restarts_2026-10-16
export PYTHONPATH=$PWD/src CUDA_VISIBLE_DEVICES=""
P="nice -n 19 taskset -c 5-7,13-15 $GEN3AI_PYTHON"
$P $D/extract.py /home/goodlad/dev/gen3ai/models /tmp/fpsr/cache.json   # ~3 min; the raw series (21 MB, not committed)
$P $D/analyze.py /home/goodlad/dev/gen3ai/models /tmp/fpsr/cache.json $D/result.json
$P $D/summary.py $D/result.json $D/summary.json
$P $D/age_vs_step.py /tmp/fpsr/cache.json $D/age.json
$P $D/ladder_load.py /home/goodlad/dev/gen3ai/models/rb_x26_s2601 /tmp/fpsr/cache.json rb_x26_s2601 > $D/ladder_x26.json
$P $D/report.py $D/result.json procs|seams|hourly <run>...                 # the tables
```

**What each output holds:**
- **`result.json`**, per run and per process: the whole-process summary, the hourly trajectory, the first cycles,
  and every seam's before / after hour, gap, prewarm and learning jumps.
- **`rss_x26.log`**: the raw `/proc` sample.
- **`result.json` and `ladder_x26.json`** were read at 10:15 PDT. X26 is live, so a rerun extends them.
