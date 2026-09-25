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

**MILESTONE tiers at `318bdcb8`** (recorded in `designs/ops/slow_tier_status.json`): slice N 4 / 4 (pool 60, `production` policy 15, ladder 30, procedural 20 episodes — no unexplained difference; F1 only), the parity MILESTONE 10 / 10 with the new parse-path encode gate in `core_events`.

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

## 6. The `__OBS__` cost and the throughput A/B (§3 target 10) — NOT MET

Release `sim_bridge`, load ~27, `nice 19` (UNVERIFIED on an idle box): +~135 µs per frame (the
parse + tracker fold ~137 µs, encode ~30 µs, frame JSON ~19 µs) against a Python production-shape
encode of 0.12–0.16 ms (M4).

`python -m main.rust_core_cutover.throughput_ab --pairs 8 --decisions 400` from pin `318bdcb8`
(interleaved A B / B A, `--pin-battles --seed 0`, the rust release bridge, concurrently with the
stress at cap 4; rows `~/gen3ai_archive/cutover_stress_2026-09-24/throughput_ab/`):

| per decision | python obs (median of 8) | core obs | core − python, per pair |
|---|---|---|---|
| full cycle (our CPU + the child, serialized) | 1.07 ms | 1.19 ms | **median +7.5%**, mean +16.2%, 95% CI [+3.0%, +38.2%] (one +90% pair) |
| our controllable (Python) CPU | 0.69 ms | 0.58 ms | mean −12.2%, CI [−21.1%, +2.4%] |

**Verdict: NON-REGRESSION NOT MET** (the CI's upper end +38% > +3%; even the lower end is above
the bar). Mechanism: under `core` the Python side still runs the tracker fold, `live_view()`, the
labels and the reward (they are not the obs), so the core's parse + tracker fold in the child is
ADDED work while only the encode (~0.1 ms) is removed — the tracker fold is paid twice. Training is
CPU-bound at `--n-envs 48` on 16 cores, so the cycle total is the quantity that predicts fps.

## 7. Readiness against the pre-registered targets (2026-09-25 04:30 PT) — NOT READY

**New training-input boundary: `c080f5f0`** (folds into `b9d74f65`, no run having trained between —
round 2 is pinned to `6eb9c776`): `52887823` (five engine classes: `[fatigue]`, Safeguard self-
confusion, Imprison re-cast, Defense Curl / Rage stat index, Mimic self-overwrite readers — change
Rust-bridge play; 0 of 813 pool files affected), `7f781602` (F1: a decision is recorded only when the
env asks the trainee to move), `c080f5f0` (a move's target is its dex target class, in the reading and
the event window — 20.2% of window MOVE rows change target). The stress runs from pin `c080f5f0`
since 04:19; its earlier rows ran at `0478c14c` / `318bdcb8` / `e26dd14c` (per-row stamp).

| # | target | done / registered | CUTOVER-class | state |
|---|---|---|---|---|
| 1 | ladder full tier, both slot orders | 3,840 + 3,840 / 11,407 + 11,407 battles (7,680 of 22,813 teams per pass) | **2 open**: `ladderA_3459` / `ladderB_3459` refused — the core's Hidden-Power belief eliminates every Lunatone candidate where Python's does not (under triage) | open |
| 2 | pool × 12 partners | 2,920 / 8,628 | 1, FIXED (`pool_110_5`, a Snatch-stolen move's owner, `6b91d710`) | running |
| 3 | `production` policy: pool / ladder / procedural | 980 / 2,876 · 280 / 800 · 180 / 500 | 0 | running |
| 4 | procedural | 1,360 / 4,000 | 0 | running |
| 5 | protocol + byte-fuzz corpora | 1 / 1 | 0 | MET |
| 6 | slice N | pool 1,000 / 3,000 · policy 340 / 1,000 · ladder 350 / 1,000 · procedural 175 / 500 episodes | every difference is F1 at `318bdcb8` (FIXED `7f781602`); since the fix: **0** (70 stress episodes, 130 checked by hand, MILESTONE 4 / 4) | running |
| 7 | the four A/B fuzzers | state 3,400 / 10,000 ladder · 1,100 / 3,000 ourandom · 400 / 1,000 pool; bytes 3,400 · 1,100 · 400; bridge 1,700 / 5,000 · 700 / 2,000 pool · 400 / 1,000 trapping; sim-bridge 700 / 2,000 · 350 / 1,000 | 14 hard: 2 fixed classes (`[fatigue]`, Mimic), 2 recorder artifacts (recorder fixed), 3 speed-tie block swaps (now a PROVEN allowlist entry), **7 untriaged** (5 `seed`, 1 `request`, 1 bridge `seed`, all at `318bdcb8`); plus a Rollout-turn Perish faint the allowlist proof's run found | open |
| 8 | soak | 2 / 6 transport + 2 / 4 core-obs children × 10,000 episodes | 0 errors, 0 replacements; child RSS flat (9.6–13.6 MiB); env RSS +2.0–2.3% (bar 10%) | running |
| 9 | the first two minutes of a real launch, `--obs-source core` | 2 launches (CPU, 8 envs, throwaway dirs deleted) | `e26dd14c`: preload compiled, 10 PPO iterations / 40,960 steps, no core-obs error; `c080f5f0`: 2 iterations, killed by PID | **MET** |
| 10 | throughput ≤ +3% (CI upper) | 8 pairs | median +7.5%, CI [+3.0, +38.2]% | **NOT MET** — the Rust frame cost is being profiled |

The stress's cost to training, cap 2: 0.93× (`ai_v13_28`, 87 iterations) and 0.93× (`ai_v13_29`, 5).

**poke-env reading findings** (READING class, both paths agree, none a cutover blocker; M6
triage): the R3 residue (an opposing Aerodactyl's unrevealed Pressure; 0.91 per 1,000 decisions,
1.31 on ladder, 0 on pool; reaches the obs as own PP); Sleep Talk into Pressure (mechanism confirmed;
0 in this stress); a Mimic'd Hidden Power keeps the bare id; a pending Yawn dropped on another sleep;
Transform `stats` (not in the obs; feeds Φ_belief); Skill Swap's ability; and one misfiled class —
`ours.types` after a Conversion-2 faint is a PORT gap (`types_override` not reset at a faint), poke-env
is right. Backlog rows in `designs/ops/TECH_DEBT_BACKLOG.md` §2 (a) / (c).

Other findings: `--eval-battles 0` makes the final eval divide by zero (`final_eval.py:115`) and the
launcher's crash restart then refuses `--arch production` on the resume (a pre-existing pair, found
by the smoke); `slice_n_test`'s COMMIT tier failed ONCE order-dependently in a worker's combined run
(`opp_team moves+5/+16/+27`, suspected the `_category_val` cache in `observation/moves.py`) while
passing in three routine gates here — UNVERIFIED.
