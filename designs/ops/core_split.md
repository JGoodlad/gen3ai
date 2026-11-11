# The core split: which CPUs the trainer gets and which the agents get

**Status (2026-11-10): the A/B has RUN; the split stays `5+3`.** The trainer is on CPUs `0-4,8-12` (5 physical
cores), agent work on `5-7,13-15` (3 physical cores, root `CLAUDE.md` rule 5b). The A/B of §5 ran once, on the fresh
launch `rb_x35_x67ent_s6701`, and no narrower arm passed the 3 % bar (§5, "The result"); the orchestrator kept `5+3`
by the registered rule, and an operator rule that freezes the split reads `5+3`. A re-measure is worth its 10 minutes
only if agent CPU becomes the binding limit again. When a split is adopted, rewrite this status to state the split as
it is.

**In one paragraph.** The trainer uses 1.8 of its 10 threads on average and never more than 3.4 in any second, while
the 6 agent threads are 93 % busy. **4 + 4** (trainer `0-3,8-11`, agents `4-7,12-15`) is predicted to cost the trainer
0 to 1 % of its steps per second for a third more agent CPU; **3 + 5** (trainer `0-2,8-10`, agents `3-7,11-15`) about
3 % for two thirds more. A 10-minute A/B on a live fresh launch decided by a rule fixed in advance (§5): it measured
**+2.88 %** for 4 + 4 (between the pass and fail lines) and **+8.71 %** for 3 + 5, so neither passed and `5+3` stays.

The box: one Ryzen 7 9800X3D, 8 physical cores, 16 threads; CPU n and n+8 are SMT siblings; one shared L3 cache. A split
is always made of WHOLE physical cores.

## 1. The owner: `utils.core_split`

Both lists live in ONE module, `src/utils/core_split.py`, and no other file may spell one
(`src/core_split_literal_gate_test.py`, a static gate, fails on a literal; §6 lists what it exempts).

```bash
python -m utils.core_split show                    # the active split, a queued one, live trainers and their CPUs, an A/B marker
python -m utils.core_split agent-cpus              # the agent list in taskset's form: nice -n 19 taskset -c "$(… agent-cpus)" <cmd>
python -m utils.core_split trainer-cpus            # the trainer list
python -m utils.core_split trainer-cpus --launch   # a LAUNCH KIT's read: a queued split becomes active first (--dry-run: writes nothing)
python -m utils.core_split set 4+4 --next-launch   # queue a split for the next launch (--note "<why>")
python -m utils.core_split set 5+3                 # set it now: REFUSED while a trainer is live
```

- **The splits** are `5+3` (the default), `4+4` and `3+5`: trainer physical cores + agent physical cores, the trainer
  taking the lowest. Python reads `core_split.agent_cpus()` / `trainer_cpus()` when it LAUNCHES a job, never at
  import, so a long-lived supervisor follows a change without a restart.
- **The state** is one JSON file, `~/.local/state/gen3ai/core_split.json`, under the home directory the password
  database names for the user: `$HOME` and `$XDG_STATE_HOME` are not read, so two readers with different environments
  cannot name two files (`$GEN3AI_CORE_SPLIT_STATE` names another; the test suite seals it to a path that does not
  exist). No file is the default split, so a box that never ran `set` behaves as before the owner existed. A file that
  cannot be parsed is an error in every reader, never a silent default.
- **A live trainer** is a process whose python RUNS `-m main.launcher` (or that package by its path; not `--dry-run` / `--help`) or the trainer
  script, read from its command line the way CPython reads it; a word in some other argv (`grep -m main.launcher`, an
  editor on the script) is not one.
- **The order is held by two refusals.** A live launcher keeps its CPUs until a new launcher starts (a child it
  restarts inherits them), so agents that widen under it land on its cores. `set` is therefore refused while a
  training launcher or trainer is live, and `set --next-launch` only queues: the split becomes active when a launch
  kit reads `trainer-cpus --launch` with no trainer live, so the new tree starts narrow and the agents widen in the
  same write. With a trainer live that read prints the active list and the queue stands. A CPU-only run an agent
  started inside the agent list is agent work: it blocks nothing as long as the new split keeps its CPUs with the
  agents.
- **Beside a live run, the ship gate reads the trainer's own affinity** (`core_split.agent_ids_beside`): its CPUs are
  the online CPUs in neither the trainer's affinity nor the active trainer list. A trainer narrowed for 8 s by the
  A/B frees nothing. A live run that is agent work (`core_split.is_agent_work`, the one rule `blocking_trainers` and
  `status` read too: its whole tree inside the agent list, or started with CUDA hidden) is not a training run for
  rule 5b. `core_split.training_runs` / `split_runs` / `agent_run_pids` are the ONE answer to "is a training run
  live?" that every shape or slot decision reads: the ship gate, `heavy_slots.wanted`, the gate lock's warning, the
  launch kit's `kit live` and the arm chain's second-GO refusal (`live.launches`). `utils.paths.refuse_live_run_dir`
  asks a different question (is this run DIRECTORY being written?) and keeps the raw `procfs.live_training_runs`;
  `src/utils/live_runs_test.py` pins the callers.
- **Who reads it.** The trainer's pin: `launch_x35.sh` (the launch kit; it HOLDs when the read fails, and logs the list it got and any note about a queued
  split). The agent list: the arm chain's defaults
  (`src/main/ops/arm_chain/spec.py`; a chain file's `cpu` block still overrides them), `main.x35.kakuna` (also the
  default behind `$GEN3AI_ANCHOR_CORES`), `main.team_strength.kakuna_lane`, `main.reads.runner`,
  `main.human_data.x59s_ops`, `utils.ship_gate`, `utils.gate_lock`, `utils.gate_lock_status`,
  `utils.rebuild_main_rust` (and `scripts/ops/rebuild_main_rust.sh`), `utils.history_rewrite`.
- **What does NOT read it yet** is §6.

## 2. What the trainer uses

Measured on `rb_x60t_s3507` on 2026-11-09 (256 envs, 8 env-worker threads, pinned `0-4,8-12`) with
`python -m main.ops.trainer_cpu_profile`, which only reads `/proc`. Record:
[`core_split_trainer_cpu_2026-11-09`](../research_state/measurements/core_split_trainer_cpu_2026-11-09/README.md).
1,200 s at 1 s (about 17 PPO cycles), then 215 s at 4 ms.

| phase | share of the time | busy threads, mean / p95 / max | who | busiest physical core, p95 (of 2.0) |
|---|---|---|---|---|
| rollout (collect) | 35 % | 2.82 / 3.07 / 3.42 | env workers 1.98, learner thread 0.72, launcher 0.12 | 1.27 |
| update | 63 % | 1.17 / 1.35 / 1.41 | learner thread 0.57, autograd thread 0.42, launcher 0.11 | 1.07 |
| all | | 1.76 / 2.96 / 3.42 | | 1.26 |

- Steps per second from the run's log: 1,368 over the window, 1,390 over the run. One PPO cycle is 70.7 s: 25.0 s
  collecting, 44.2 s updating.
- The compile-worker pool, the CUDA threads and the data path used 0.00 of a thread in steady state.
- The run-queue wait was 0.013 s per second in the rollout and 0 in the update: the trainer never waits for a CPU.
- **The rollout is a strict alternation, not a steady load.** 17.2 times a second all 8 env workers run together for
  16 ms (a "burst"), then the learner thread runs alone for about 42 ms (inference on the GPU, and Python). Workers
  and learner were runnable together in 0.0 % of 17,949 readings. So the trainer's instantaneous demand is 8 threads
  for 27 % of the rollout and 1 thread the rest of the time.
- **The update needs exactly one thread** (mean runnable 1.00, peak 2).
- Over the same window the 6 agent threads were 5.57 busy (median 5.99) and saturated in 60 % of the seconds, while
  the trainer's 10 threads were 1.93 busy with every tenant counted.

## 3. What is known about steps per second against cores

**No measurement exists.** What does:

| evidence | what it says | limit |
|---|---|---|
| rule 5b's history (commits of 2026-10-10 and 10-11) | "about 15 %", "25 %" and "30 %" lost to agent work | no record behind any figure, and the trainer was NOT pinned then: agents ran on its cores. It measures sharing, not core count |
| the 2026-10-12 ruling (`designs/endstate/design_bf16_strength_ab.md`) | an unpinned run beside agent work ran at contention factor p90 1.82; the pin fixed it | the same: sharing |
| X65-F1 ([record](../research_state/measurements/contention_trainer_vs_kakuna_2026-11-05/README.md)) | a pinned trainer ran 1,405 steps/s with 6 search loops on the agent cores, against 1,360 unloaded; the search lost 13 % | the one controlled read; 5 trainer cores only |
| env-core thread scaling ([record](../research_state/measurements/rust_core_m5_transport_2026-09-26/README.md)) | 1 / 2 / 4 / 8 worker threads: 14.5k / 28.6k / 51.6k / 84.3k decisions per second | quiet box, no pin, no learner; shows the workers do scale to 8 |
| `utils.cpu_meter` | a thread whose SMT sibling is busy runs 1.70 to 1.83 times slower | the constant the model below uses |

## 4. The model: what a narrower trainer costs

The update is insensitive: it needs one thread. The learner's part of the rollout is insensitive for the same reason.
Only the worker bursts can stretch, and a burst ends when its slowest worker ends.

- **5 cores (today).** 8 workers on 5 cores: at least 3 cores carry two workers, and those workers run at the
  SMT-shared rate. They set the burst length.
- **4 cores.** 8 workers on 8 threads: every worker runs at the SMT-shared rate, which is the rate that already sets
  the burst. Predicted burst change 0 to +2.5 %; steps per second **0 to -1 %**.
- **3 cores.** 8 workers on 6 threads: the burst is about 8/6 = 1.33 times longer (16 to 21 ms). Collect time
  +9 % (workers are 27.4 % of it), cycle time +3.2 %; steps per second **about -3 %**, say -3 to -5 % with what the
  model leaves out. Lowering `--rust-env-threads` to 6 gives the same 1.33, so it is no remedy.

Not in the model, all **UNVERIFIED:** the start-up of each launch and restart on fewer cores (the compile pool is idle
only in steady state); the eval rollouts (about 1 iteration in 20); and what five loaded agent cores do to the trainer
through the shared L3 and the all-core clock (X65-F1 saw nothing with three).

## 5. The 10-minute A/B: `main.ops.core_split_ab`

It ran once (2026-11-10; the result closes this section), on the real run, by the session that launched it (procedure:
`TRAINING_RUN_SOP.md` §2). It only narrows that run's own tree inside the CPUs it already has, so nothing else on the
box moves, and it changes the run's timing, not its arithmetic. Agents stay on the active agent list throughout.

```bash
A="$(python -m utils.core_split agent-cpus)"
nice -n 19 taskset -c "$A" python -m main.ops.core_split_ab run --launcher-pid <pid> --out-dir <dir> --check   # refusals + plan, changes nothing
setsid nohup nice -n 19 taskset -c "$A" python -m main.ops.core_split_ab run --launcher-pid <pid> --out-dir <dir> \
  > <dir>.log 2>&1 &                                           # ~9 min: detached, then ONE watcher on its pid
python -m main.ops.core_split_ab verdict <dir>/record.json     # re-read a record
python -m main.ops.core_split_ab restore                       # only after the driver was SIGKILLed
```

The run is longer than a tool's foreground limit, and a driver cut mid-segment by its caller is exactly the case the
restore exists for: start it detached and wait on its pid (after `setsid … &`, `$!` is not the driver: read the pid
of the `python` process). **Never start it under anything that ends a job with SIGKILL** (`mem_cap.sh --max-s`,
`timeout -s KILL`): a killed driver restores nothing and tells nobody. **The watcher ends with `restore`**
(`timeout 3300 bash -c "while [ -d /proc/$P ]; do sleep 5; done"; python -m main.ops.core_split_ab restore`): after
a clean exit it finds no marker and does nothing, after a kill it puts the lists back, and under a live driver it
refuses.

- **Arms.** A = the tree's own 5 cores `0-4,8-12` (control), B = 4 cores `0-3,8-11`, C = 3 cores `0-2,8-10`.
- **Unit.** An 8-second segment of one rollout. A rollout is about 24 s (23.8 s on the live run of 2026-11-09), so
  it holds three segments, one per arm. The order rotates over 6 rollouts (ABC, BCA, CAB, CBA, ACB, BAC), so each arm
  sits in each position twice. The update between them runs on arm A. Six cycles are 7.1 minutes. A rollout ends with
  its third segment, or at once when no worker was runnable for 0.3 s; the open segment ends THERE (the third one of
  a 23.8 s rollout measures 6.8 s, not 7), never at the time planned for it, and a segment that never started makes
  its arms INCONCLUSIVE.
- **When to start it.** After the run's third logged iteration (the driver refuses earlier) and right after an eval
  has logged, so no eval falls inside: an eval comes about once in 20 iterations. The driver does not wait for an
  eval itself.
- **Mechanism.** The driver sets the affinity of every thread of the launcher's tree at each segment boundary and
  writes the switch times down. It finds a rollout's start from the run state of the TRAIN env's workers, read every
  4 ms: the first runnable worker after 2 s with none, confirmed by 4 bursts in half a second (a blip inside an update
  is not a rollout). The tree holds a second env process, the eval env, with workers of its own: the train env is the
  env process with the most CPU time, and the driver refuses a tree where the runner-up has half of it or more.
- **A segment counts only when its list was verified.** A switch reports whether its last walk found every thread on
  the arm's list, and each segment's end reads every thread's list back before the next switch. A set the kernel
  refused, or a list something else changed mid-segment, makes that arm INCONCLUSIVE; it is never booked as the arm.
- **What it may touch.** One launcher's tree, by `sched_setaffinity` only, and only to a subset of that tree's own
  CPUs. It refuses a pid that is not a `main.launcher`, a tree that is not uniformly on the `5+3` trainer list, a run
  with fewer than three logged iterations, and a second A/B. It never signals, renices or widens anything.
- **Restore.** The tree's original list goes back when the A/B ends, on any error, and on every catchable signal
  that would end or stop the driver (the driver's `_HANDLED`: SIGTERM, SIGINT, SIGHUP, SIGQUIT, SIGUSR1/2, the
  terminal stops, the timers, SIGXCPU, SIGIO, SIGPWR; a SIGHUP ignored under `nohup` stays ignored; the restore
  cannot be interrupted by a second signal). It is tried up to three
  times and counts only when a walk AFTER the last set found every thread on the original list. It reaches every
  process a narrowing touched, in the tree or not: each is named (pid and start time) in a marker,
  `core_split_ab.json` beside the state file, BEFORE it is narrowed (and a child born on a narrow list is named at
  that segment's end), so a trainer whose launcher died mid-segment, or a child re-parented away, gets its list back
  too, and a reused pid is never touched. The driver stops with "the
  launcher is gone" within a second of that. The marker is removed only after a verified restore; until then
  `utils.core_split show` reports it. A SIGKILL or an OOM kill cannot be caught: `restore` reads the marker and puts
  the lists back, and refuses while the driver that wrote it is alive (SIGTERM that driver; SIGKILL it only if it does
  not exit).
- **Meter.** Its own recording, `<dir>/burst.jsonl`, in `main.ops.trainer_cpu_profile burst`'s format (that tool's
  `summarize` reads it), on one clock with the switches. Per segment, drop the first second after a switch, then take
  the mean time from one burst's start to the next (57 ms today; an 8 s segment holds about 120). That period IS the
  rollout's speed: it carries the burst and anything the narrower mask does to the learner's part. Per rollout, form
  period(B) / period(A) and period(C) / period(A); the result is the mean of each ratio over the 6 rollouts and its
  standard error.
- **Projection.** Loss of steps per second = 0.353 x (ratio - 1), 0.353 being the collect share of a cycle.
- **The control** is `update_wall_s` of the six updates against the three before. The run's logger table lands within
  half a second of a rollout's END and carries the update BEFORE that rollout (checked on `rb_x60t_s3507`, three
  cycles), so the first table after the A/B's first rollout starts is the last "before" and the next six are the
  A/B's.
- **The rule (orchestrator, 2026-11-09; bar 3 %).** An arm PASSES at a projected loss of 2.5 % or less and FAILS at
  3.5 % or more. Between the two it is INCONCLUSIVE and counts as a fail. It is also INCONCLUSIVE when the standard
  error of its projected loss exceeds 0.5 %, when an eval landed in the measured window (an eval line in the log, or
  a new row in the run's `eval_results.jsonl`), when the control moved by more than 10 % (three times its standard
  error: `update_wall_s` varies 4.7 % per iteration), or when the recording is thin (next bullet). **Adopt the
  narrowest arm that passes**; with none passing, the split stays `5+3`. The model puts arm C at 3.2 %, inside the
  band: expect the A/B, not the model, to decide 3 + 5.
- **A late reading is excluded, never tolerated.** The driver runs at nice 19 beside busy agent jobs. A
  burst-to-burst interval holding a sampling gap over 12 ms (shorter than a burst, so none hides in an accepted gap)
  is dropped; a segment with over 25 % of its intervals dropped, or fewer than 40 left, makes its arm INCONCLUSIVE.
  A replay of the 215 s recording of §2 through the driver dropped 0 of about 120 intervals per segment.
- **Known noise.** With no narrowing at all, the period drifts 1 to 2.7 % with a segment's position inside a rollout
  (the same replay). The rotation puts each arm in each position twice, which balances it; what is left shows in the
  standard error, which the rule bounds.
- **Why the bar is 3 %.** 3 % is 43 minutes of GPU steps a day; two more agent cores are 48 core-hours a day on a box
  where CPU, not memory, is what agent work waits for.
- **Why not whole iterations.** `collect_ms` varies 3.3 % from one iteration to the next. Resolving a 1 % change in it
  needs about 22 iterations per arm, 80 minutes for three arms. That version (three hand switches per block, by the
  launching session on its own tree) is the fallback.
- **What it cannot see.** The freed cores sit idle during the A/B, so it is optimistic about shared-cache and clock
  effects, and it does not cover start-up or evals. §7's check after the switch covers those.
- **The result (2026-11-10, `rb_x35_x67ent_s6701`, launcher 209033, 16:37:58 to 16:47:41 PST, started right after
  the 284,000,191 eval; no eval inside).** `RESULT: no narrower arm passed. Keep the split as it is (5+3).`
  Arm B (4 trainer cores, `0-3,8-11`): INCONCLUSIVE, projected loss **+2.88 % ± 0.33 %** (period ratio 1.0817 over 6
  rollouts), between the pass and fail lines. Arm C (3 trainer cores, `0-2,8-10`): INCONCLUSIVE, **+8.71 % ± 0.73 %**
  (period ratio 1.2468), its standard error over the 0.5 % limit and its loss far past the fail line. Control
  (`update_wall_s`, during vs before): +3.7 %. The driver restored `0-4,8-12`, verified on 68 threads. Decision
  (orchestrator, 2026-11-10): KEEP `5+3`, no `set`. The model had put C at 3.2 %; the measurement is nearly three
  times that, so §4's model understates what a narrower trainer costs. B sits just under the bar, so a re-measure is
  worth it only if agent CPU becomes the binding limit again.

## 6. What does not read the owner

| what | state | who acts |
|---|---|---|
| **the pin enforcer** (the orchestrator's session script: while a GPU lease is held it re-pins every nice-19 process of the user to the agent list every 30 s) | READS the owner (patched and restarted 2026-11-09 20:50, checked 2026-11-10: `agent-cpus` once per cycle, a cycle whose read fails is skipped, the list it pinned to is in its log). Before that it spelled the `5+3` agent list twice and would have put every agent job back on it within 30 s of an adopted split | the orchestrator: restart it after any edit to the script, and BEFORE the launch that adopts a split. Its skip list needs no entry for the A/B driver: the trainer's tree runs at nice 10, which the enforcer never touches |
| **a session's own launch script** outside the repo | spells the trainer list in its `taskset` | the launching session: `taskset -c "$(python -m utils.core_split trainer-cpus --launch)"`, and HOLD when the read prints nothing |
| **frozen measurement kits** under `designs/research_state/` | keep their literals: they are records, and the gate exempts that tree, EXCEPT `LIVE_KITS` (`launch_x35.sh` and `drive_look.sh` read the owner; the gate scans them) and every code file in a measurement directory dated 2026-11-10 or later (`core_split_literal_gate_test.born_live`: a kit is a tool the day it is written; an armed kit cannot be edited). **Still spelling a list, written before that day** (2026-11-10): X66 `bank_build.sh`, `gate.sh` (agent list) and `gpu_students.sh` (trainer list), X67 `cross/drive_cross.sh` (agent), X65 `gpu_arms.sh` (trainer), X34 `launch_x34.sh` (both) | whoever runs another kit again converts it and names it in `LIVE_KITS` first; the X66 / X67 kits after their launches |
| **a chain file's `cpu` block** | an explicit list in a chain file overrides the owner | drop the block from a chain that should follow the split |
| **a supervisor started before the owner landed** (a play lane on an old pin, an arm-chain supervisor) | holds the `5+3` agent list in its own code until it restarts. That list lies inside both wider agent sets, so it keeps working, on fewer cores than it could use | restart it to use the new cores |

The gate also exempts the append-only `designs/CHANGELOG.md` and any file whose name carries its date (a review or
audit of that day). `utils.heavy_slots` and `mem_cap.sh` hold no core list.

## 7. The switch, the check, the rollback

**Order matters: the trainer is narrow before the agents are wide.** Agents that widen first land on cores the
trainer still owns, and priority alone has not protected a run. A live tree is never re-pinned in place: a split
changes only at a launch, where one write does both sides.

1. **Launch on `5+3` and run the A/B** (§5). Keep `<dir>/record.json`; the verdict names the split to adopt, or says
   to keep `5+3`. Let the run log 10 iterations without an eval: they are step 4's baseline.
2. **Queue it:** `python -m utils.core_split set <split> --next-launch --note "A/B <dir>"`. Nothing changes yet. The
   orchestrator patches and restarts the pin enforcer (§6) now.
3. **It becomes active at the next launch through a kit**: the next run, or this run stopped and resumed (a resume is
   a launch; the 6-hourly restart of the child inside a live launcher is not). `python -m utils.core_split show` then
   names the new split and the tree on its trainer list. Jobs the agents start from then on take the wider list; the
   enforcer moves the running ones.
4. **Check, with the agent cores loaded:** every thread of the tree shows the new list; then 20 minutes of
   `trainer_cpu_profile sample` and 215 s of `burst`. Pass: the rollout's run-queue wait stays under 0.05 s per second
   (0.013 today); the burst-to-burst period is within the A/B's projection plus 1 %; over at least 30 iterations without
   an eval, the median `collect_ms` against step 1's baseline projects (ratio less 1, times 0.353) to a loss inside the
   bar, and the mean `update_wall_s` is within 5 % of the baseline's. Check an arm chain's registered throughput floor
   against the projected loss before step 2. Time the launch from its start to its first trainer line against the last
   one on 5 cores.
5. **Roll back** when a check fails, when work the orchestrator marked BLOCKING is starved, or on the orchestrator's
   word, **agents first**: stop the run at a safe point, KEEPING the GPU lease (`TRAINING_RUN_SOP.md` §4); `python -m utils.core_split set
   5+3` (accepted now that no trainer is live), so every new agent job takes the narrow list; wait until an enforcer cycle
   (30 s; it runs while the GPU lease is held) adds no `pinned` line to its log, so no running job is left on the cores
   the trainer is about to get; resume the run through its kit. No code is reverted: the split is state.

## 8. What it buys

| split | agent cores | agent threads | change | predicted cost in steps per second | measured (§5, 2026-11-10) |
|---|---|---|---|---|---|
| today | 3 | 6 | | | |
| 4 + 4 | 4 | 8 | +33 %, +48 thread-hours a day | 0 to -1 % | -2.88 % ± 0.33 % (not adopted) |
| 3 + 5 | 5 | 10 | +67 %, +96 thread-hours a day | about -3 % (-3 to -5 %) | -8.71 % ± 0.73 % (not adopted) |

- **In Kakuna terms.** A 20-game Kakuna unit cost a mean 231 CPU-seconds on 11-09 (median 179; 1.10 threads busy
  while it runs; systemd's accounting of 293 unit scopes). One unit stream running without a pause is one lane-hour per
  hour and takes 1.1 threads. The extra threads are therefore **+44 Kakuna lane-hours a day at 4 + 4 and +87 at
  3 + 5** (about +15,000 and +30,000 games a day), IF all of the new capacity went to Kakuna and nothing else held the
  lane back. It is an upper bound.
- **Cores are not what holds the Kakuna lane back today.** On 11-09 its units ran for 147 to 179 s (median, by six
  hours) and WAITED for a heavy slot for 145 s in the morning and 1,477 s after 16:00; the lane used 18.8 CPU-hours
  in 18 hours, one thread on average. Its play pin predates memory admission, so its units see two slots while slots
  2 and 3 stand free (`testing.md`, "Mixed versions"). A new play pin or lane restart is worth more to this lane
  than any core; the split helps it by the unit's run time only.
- **Everything CPU-bound on the agent cores gains the ratio**, less what the shared cache takes (X65-F1: 13 %): the
  ship gate (its `-n` is half the agent threads, so 3 becomes 4 or 5 by itself), measurement batteries, the S lane.
  Memory admission allows 4 heavy jobs at once; 4 or 5 cores are what lets four jobs run rather than share three.

## 9. Considered, not adopted

- **Lending idle trainer cores by scheduler class.** During the update (63 % of the time) all but one trainer core
  is idle. Agents in the idle scheduling class on those cores, with core-scheduling cookies so no agent thread
  shares a physical core with a running trainer thread, would use that time and yield within the kernel's own
  latency. It is the largest prize here and the class of change that caused the rule-5b incident: the 17 bursts a
  second each wake 8 workers onto cores an agent has just used. It needs its own A/B and its own design.
- **Flipping affinities by phase.** A daemon that moves every agent process twice per 70 s cycle has to predict the
  phase boundary and churns dozens of processes; a late flip lands on a rollout.
- **Fewer than 3 trainer cores.** The burst would stretch by 2 times on 2 cores (-10 % of steps per second by the
  model).

Revisit the first of these if a static split is adopted and CPU still binds.
