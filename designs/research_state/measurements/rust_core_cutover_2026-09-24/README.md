# Rust Core M6 — the CUTOVER stress (interim record)

<!-- A MEASUREMENT record (2026-09-24) for the Rust Core Program's M6
(designs/endstate/program_rust_core.md §2 M6, §3 "The CUTOVER tier — PRE-REGISTERED targets").
Code: src/main/rust_core_cutover/. This record is INTERIM: the stress is running; the readiness
report replaces the "progress" sections with verdicts read at the registered counts. -->

Box: 16 cores shared with the live training queue (`ai_v13_27_popr2_loop`, pinned `6eb9c776`, GPU)
and other agents; every stress unit `nice 19`, one torch/BLAS thread, CPU only, no port, no
`models/` write, no `data/` change.

## 1. Where it runs and how to read it

* **Out dir (durable rows):** `~/gen3ai_archive/cutover_stress_2026-09-24/run/` — `units/` (one
  atomic row per unit), `divergences/` (every divergent battle's input log, re-runnable alone),
  `fuzz/` (each fuzz unit's run dir + repros), `logs/`, `control.json` (the cap / pause / stop
  knobs), `governor.json` (training fps rows + throttle events), `timeline.jsonl`,
  `registration.json` (+ `amendments`).
* **Pins:** `~/gen3ai_archive/cutover_stress_2026-09-24/pins/<sha12>/` — a `git archive` export
  with its own self-check + release binaries (not a worktree; a binary built in a worktree panics
  once the worktree is removed). Every row is stamped with its pin.
* **Progress:** `cd <pin> && PYTHONPATH=src python -m main.rust_core_cutover status --out ../../run`
  (`--json` for the machine form). It is PROGRESS, not a verdict: divergences are listed by
  CUTOVER / READING / VIEW-ROAD class (`verdict.py`).
* **Operate:** edit `control.json` (`cap`, `pause`, `stop`); kill only by the PID in
  `driver.pid`; restart with `run --out … --detach` from a pin (resumes; adopts still-running units).

## 2. Kill + resume, proven on the real run (2026-09-24 21:58–21:59)

1. SIGTERM the driver ALONE (pid 1569865) while 4 units ran: all 4 kept running (units are their own
   sessions). A new driver (pid 1605676) ADOPTED the 3 still running (`running/*.pid` + argv check),
   started ONE new unit to fill the cap of 4, and the adopted `pool_random.00012` finished with its
   row — no duplicate spawn.
2. SIGTERM that driver and SIGKILL one of its units (`pool_policy.00008`, pid 1605697) together: no
   row was on disk; the next driver (pid 1605891) re-ran `pool_policy.00008` from scratch.

Pinned by `driver_test.py` (atomic rows define done, a half-written tmp is not done, adoption by
argv, a changed registration refused unless amended visibly).

## 3. The training arm's cost (the governor, pre-registered in §3)

MARGINAL fps per PPO iteration (98,304 steps), read-only from the live arm's `launcher_child.log`:

| window | iterations | median fps | vs pre-stress |
|---|---|---|---|
| pre-stress OFF window (21:18–21:33, the driver's first 15 min, no unit) | 6 | **573** | — |
| stress at cap 6 (21:42–21:53) | 4 | 473 | **−17.5%** → the throttle fired (0.82 < 0.85): cap 6 → 4 at 21:53 |
| stress at cap 4 (21:56–22:15) | 7 | 520 | **−9.2%** (within the 15% bar) |

Honest limits: the arm's own iteration-to-iteration spread is ±5–8% (492–607 fps with no stress);
the 4 iterations before the OFF window (21:05–21:15, ~500 fps) overlapped the routine gate this
agent ran and the child's post-restart warm-up, so they are not the baseline; the periodic 20-min
OFF windows (every 4 h) and an arm switch re-baseline it.

## 4. Progress at the interim report

See `status`. At 22:15: ladder full tier 2,160 (A) / 2,080 (B) of 22,813 teams covered; ~630 k
slice-O decisions byte-equal across all parity streams; **0 CUTOVER-class divergences** in
slices E / V / T / O; the READING and VIEW-ROAD classes seen so far are the ones M2–M4 recorded
(R3-residue own PP after a faint, the Transform `stats` class, the view road's volatile / item /
opp-PP projection); the soak's first child played 10,000 episodes with 0 replacements, child RSS
flat at 9.6 MiB and env RSS 437 → 446 MB (+2.0%).

Fuzzer findings so far (under triage, not yet explained): a protocol divergence — the port omits
`[fatigue]` on the lock-end confusion `|-start|…|confusion|[fatigue]` (`grep fatigue
src/rust_sim/src` is empty; Showdown's `conditions.ts` confusion.onStart adds it for `lockedmove`),
and a `seed` (RNG-consumption) divergence at decision 79 of a ladder battle
(`fz_proto_ladder.00000/divergences/rmuggvoke_ab_3_15`).

## 5. Slice N and finding F1

Two `Gen3Env`s in LOCKSTEP (python vs core obs source; same seed, teams, opponent RNG, trainee
actions; a phantom step is driven with action 0 as `MaskableAgentWrapper` does), every obs key,
reward and episode end compared per step:

| corpus | episodes | steps | core decisions | phantom steps | differences |
|---|---|---|---|---|---|
| pool, seeded-random | 60 | 4,530 | 4,273 | 228 (5.0%) | 41, all `turns_since_progress` after a phantom |
| pool, `production` policy | 4 | 137 | 117 | 20 (14.6%) | 2, the same class |
| ladder full tier, seeded-random | 20 | 1,860 | 1,755 | 105 (5.6%) | 18: 11 clock + 7 opp recency, all after a phantom |
| procedural, seeded-random | 10 | 784 | 742 | 42 (5.4%) | 8, clock, after a phantom |
| pool, seeded-random, the phantom record SUPPRESSED in the Python env (diagnostic; run before the harness drove phantom steps with action 0) | 60 | 4,903 | 4,639 | — | **0** |

**F1:** on a step where the trainee is not asked to move (a `wait` request while the opponent
force-switches; poke-env's `battle1` re-embed), the live env still records a trainee DECISION
(`EpisodeTracker.record` + `update_progress_clock`, then the wrapper's `step(0)` → `advance(0)`);
the core and the replay oracle do not. A Python-side TRAINING-INPUT bug, reported with its rate, not
fixed (backlog (a), P0, the orchestrator's call). The cutover as built would change these inputs.

## 6. The `__OBS__` cost

Release `sim_bridge`, load ~27, `nice 19` (UNVERIFIED on an idle box): +~135 µs per frame (the
parse + tracker fold ~137 µs, encode ~30 µs, frame JSON ~19 µs) against a Python production-shape
encode of 0.12–0.16 ms (M4). The throughput A/B (§3 target 10,
`python -m main.rust_core_cutover.throughput_ab`) is the verdict.
