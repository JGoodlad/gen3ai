# The core split: which CPUs the trainer gets and which the agents get

**Status (2026-11-09): PROPOSED, not adopted.** The live split is unchanged: the trainer on CPUs `0-4,8-12` (5 physical
cores), agent work on `5-7,13-15` (3 physical cores, root `CLAUDE.md` rule 5b). The split changes only on the
orchestrator's decision. This document holds the measurement behind the proposal, the proposal, the A/B that decides
it, the change list and the rollback. When a split is adopted, rewrite this document to state the split as it is.

**The proposal in one paragraph.** The trainer uses 1.8 of its 10 threads on average and never more than 3.4 in any
second, while the 6 agent threads are 93 % busy. Move the boundary at the next fresh launch: **4 + 4** (trainer
`0-3,8-11`, agents `4-7,12-15`) is predicted to cost the trainer 0 to 1 % of its steps per second for a third more
agent CPU; **3 + 5** (trainer `0-2,8-10`, agents `3-7,11-15`) is predicted to cost about 3 % for two thirds more. No
measurement of steps per second against core count exists, so a 10-minute A/B on that launch picks between them by
a rule fixed in advance (§5).

The box: one Ryzen 7 9800X3D, 8 physical cores, 16 threads; CPU n and n+8 are SMT siblings; one shared L3 cache. A split
is always made of WHOLE physical cores.

## 1. Where the split is set today

Nothing owns it. Each side is a literal, repeated:

- **The trainer's cores** are set by the launch kit, not by the launcher: `taskset -c 0-4,8-12` in front of
  `python -m main.launcher` in `designs/research_state/measurements/x35_breakers_stage_2026-10-17/launch_x35.sh`
  (pinned by `src/main/x35/launch_x35_test.py`) and in each session's own launch script. The launcher only sets the
  priority (the tree runs at nice 10). A live launcher keeps its CPUs until a new launcher starts; a child it restarts
  inherits them.
- **The agents' cores** are the string `5-7,13-15` in rule 5b, in 13 code files and 6 tests (§6), and twice in the
  orchestrator's pin enforcer, a session script that, while a GPU lease is held, re-pins every nice-19 process of the
  user to exactly that list every 30 s.
- **The arm chain** reads both lists from a chain file's `cpu` block, defaulting to today's
  (`src/main/ops/arm_chain/spec.py`). It uses them for its GO-time CPU gate and for the prefix of its own agent jobs;
  it does not pin the trainer.
- `utils.ship_gate` is the one reader that derives the agent set: beside a live run it takes every online CPU the
  trainer is not pinned to.

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

## 5. The 10-minute A/B (designed, not run)

It runs on the next fresh launch, on the real run, by the session that launched it, on the orchestrator's go. It only
narrows the run's own tree inside the CPUs it already has, so nothing else on the box moves, and it changes the run's
timing, not its arithmetic. Agents stay on `5-7,13-15` throughout.

- **Arms.** A = 5 cores `0-4,8-12` (control), B = 4 cores `0-3,8-11`, C = 3 cores `0-2,8-10`.
- **Unit.** An 8-second segment of one rollout. A rollout is 25 s, so it holds three segments, one per arm. The order
  rotates over 6 rollouts (ABC, BCA, CAB, CBA, ACB, BAC), so each arm sits in each position twice. The update between
  them runs on arm A. Six cycles are 7.1 minutes; start after the third logged iteration, right after an eval
  iteration, so no eval falls inside.
- **Mechanism.** `taskset -a -cp <list> <pid>` on every process of the launcher's tree at each segment boundary, and
  the switch times written down. This needs a small driver (§6, step 2), which finds a rollout's start from the
  workers' run state as `burst` does: 18 timed switches are not hand work.
- **Meter.** `python -m main.ops.trainer_cpu_profile burst` recording the whole A/B. Per segment, drop the first
  second after a switch, then take the mean time from one burst's start to the next (57 ms today; an 8 s segment holds
  about 120). That period IS the rollout's speed: it carries the burst and anything the narrower mask does to the
  learner's part. Per rollout, form period(B) / period(A) and period(C) / period(A); the result is the mean of each
  ratio over the 6 rollouts and its standard error.
- **Projection.** Loss of steps per second = 0.353 x (ratio - 1), 0.353 being the collect share of a cycle. The
  control is `update_wall_s` of the six iterations against the three before.
- **The rule, fixed before the run.** The orchestrator sets the bar first; the recommendation is **3 %**. An arm
  PASSES at a projected loss of 2.5 % or less and FAILS at 3.5 % or more. Between the two it is INCONCLUSIVE and counts
  as a fail. It is also INCONCLUSIVE when the standard error of its projected loss exceeds 0.5 %, when an eval landed
  in a measured rollout, or when the control moved by more than 10 % (three times its standard error: `update_wall_s`
  varies 4.7 % per iteration). **Adopt the narrowest arm that passes.**
  The model puts arm C at 3.2 %, inside the band: expect the A/B, not the model, to decide 3 + 5.
- **Why the bar is 3 %.** 3 % is 43 minutes of GPU steps a day; two more agent cores are 48 core-hours a day on a box
  where CPU, not memory, is what agent work waits for.
- **Why not whole iterations.** `collect_ms` varies 3.3 % from one iteration to the next. Resolving a 1 % change in it
  needs about 22 iterations per arm, 80 minutes for three arms. That version needs no driver (three hand switches per
  block) and is the fallback.
- **What it cannot see.** The freed cores sit idle during the A/B, so it is optimistic about shared-cache and clock
  effects, and it does not cover start-up or evals. §7's check after the switch covers those.

## 6. What must change

| step | what | where |
|---|---|---|
| 1 | **One owner for the two lists** (a build unit, not done): one module holding the trainer and agent lists, a one-line CLI for shell scripts, and the rule `utils.ship_gate` already applies (beside a live run, agents = online CPUs minus the trainer's). Every site below reads it | new, under `src/utils/` |
| 2 | the A/B driver (§5): timed `taskset` on one launcher's tree, refusing any pid outside it and any list outside the tree's own CPUs | new, under `src/main/ops/` |
| 3 | the trainer's pin | the `taskset` literal of the launch kit in use (`launch_x35.sh` and `src/main/x35/launch_x35_test.py` for the X35 kit; each session launch script) |
| 4 | the arm chain's defaults and every live chain file's `cpu` block; a running supervisor holds its values until it restarts, and its CPU gate names any agent job outside its agent list a hog | `src/main/ops/arm_chain/spec.py`, `cli.py`, `cpugate.py`, `runner_test.py` |
| 5 | the agent list in code | `src/main/x35/kakuna.py` (also the default behind `$GEN3AI_ANCHOR_CORES`), `src/main/team_strength/kakuna_lane.py`, `src/main/reads/runner.py`, `src/main/human_data/x59s_ops.py`, `src/main/ops/rss_watch.py`, `src/utils/ship_gate.py` (its lease-only list), `src/utils/gate_lock_status.py`, `src/utils/rebuild_main_rust.py`, `scripts/ops/rebuild_main_rust.sh`, `src/utils/history_rewrite/__main__.py`, and their tests |
| 6 | the pin enforcer: its list (twice) and its skip list. Left alone it puts every agent job back on `5-7,13-15` within 30 s whenever a lease is held, and the switch silently does nothing | the orchestrator's session script; the orchestrator edits and restarts it |
| 7 | the rule and its echoes | root `CLAUDE.md` rule 5b; `designs/ops/testing.md`, `TRAINING_RUN_SOP.md`, `ORCHESTRATOR_SOP.md`; `designs/tools/reads.md`; `designs/training/eval_and_rating.md`; the `designs/endstate/` docs that quote a command |

`utils.heavy_slots` and `mem_cap.sh` hold no core list. A play pin cut before the switch keeps `5-7,13-15`, which lies
inside both proposed agent sets, so an old lane keeps working (the enforcer widens it). Frozen measurement kits keep
their literals: they are records.

## 7. The switch, the check, the rollback

**Order matters. Narrow the trainer first, widen the agents second.** Agents that widen first land on cores the
trainer still owns, and priority alone has not protected a run. A rollback runs in the reverse order: agents back to
`5-7,13-15` first (and one enforcer cycle to confirm no nice-19 process is outside it), then the trainer widened.

1. Launch on today's split. Run the A/B (§5), read the rule, then let the run log 10 iterations without an eval on
   the old list: they are step 4's baseline.
2. The launching session narrows its tree to the adopted list (one `taskset -a -cp` per process of the tree), and
   the launch kit's literal changes in the same shipped change, so the next launch starts narrow.
3. The agent side ships (steps 4 to 7 of §6); the orchestrator restarts the enforcer and any chain supervisor.
4. **Check, with the agent cores loaded:** every thread of the tree shows the new list; then 20 minutes of
   `trainer_cpu_profile sample` and 215 s of `burst`. Pass: the rollout's run-queue wait stays under 0.05 s per second
   (0.013 today); the burst-to-burst period is within the A/B's projection plus 1 %; over at least 30 iterations without
   an eval, the median `collect_ms` against step 1's baseline projects (ratio less 1, times 0.353) to a loss inside the
   bar, and the mean `update_wall_s` is within 5 % of the baseline's. Check an arm chain's registered throughput floor against the projected loss before
   step 2. Time the next restart from its start to its first trainer line against the last one on 5 cores.
5. **Roll back** when a check fails, when work the orchestrator marked BLOCKING is starved, or on the orchestrator's
   word: the reverse order above, then one revert of the shipped change.

## 8. What it buys

| split | agent cores | agent threads | change | predicted cost in steps per second |
|---|---|---|---|---|
| today | 3 | 6 | | |
| 4 + 4 | 4 | 8 | +33 %, +48 thread-hours a day | 0 to -1 % |
| 3 + 5 | 5 | 10 | +67 %, +96 thread-hours a day | about -3 % (-3 to -5 %) |

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

## 9. Considered, not proposed

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
