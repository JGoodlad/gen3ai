# M5 Lane E — opponent routing: PROGRESS (resume point)

Lane E of `designs/endstate/program_rust_core.md` §2 M5 (the lane-table row and the "Lane E BUILT"
paragraph, which is the design of record). Owns `src/rust_env/src/opponents.rs`,
`src/agents/training/rust_env_opponents.py` (+ `_test.py`), the gate harness
`src/agents/training/rust_env_opponents_parity.py` (+ `_test.py`), the benchmark
`src/agents/training/rust_env_opponents_benchmark.py`, `src/rust_env/tests/opponents_test.rs`, and its
rows of `columns.py` (`ep_opp`, `opp_route`, `opp_slot`) + the spec key `opponents` (hand-offs below).

## How to test

```bash
export PYTHONPATH=$PYTHONPATH:src
python3 -m pytest src/agents/training/rust_env_opponents_test.py -q            # unit, routine, ~2 s
python3 -m pytest src/agents/training/rust_env_opponents_parity_test.py -q -m "not slow"   # the gate, COMMIT, ~70 s
(cd src/rust_env && CARGO_TARGET_DIR=$PWD/target cargo test --profile selfcheck --features emission-selfcheck --test opponents_test --lib)
# the GPU MILESTONE (idle GPU, the lock, fresh compile caches):
GEN3AI_TEST_ALLOW_GPU=1 flock /home/goodlad/.claude/jobs/gpu.lock python3 -m pytest \
    src/agents/training/rust_env_opponents_parity_test.py -q -m slow -k gpu
# the gate as a script (JSON out), and the production-shape benchmark:
python -m agents.training.rust_env_opponents_parity --pool fresh:2 --n-envs 4 --episodes 8
flock /home/goodlad/.claude/jobs/gpu.lock python -m agents.training.rust_env_opponents_benchmark \
    --snapshots /home/goodlad/dev/gen3ai/models/ai_v14_06_lbat_ctrl_fix/snapshots --n-envs 48
```

The harness rebuild rule: a stale `.so` is REFUSED by the stamp (Lane 0 gate ⑤) — rebuild with
`cargo build --lib --profile selfcheck --features emission-selfcheck` (the test fixture does it).

## Units

| # | unit | status |
|---|---|---|
| 1 | CORE routing: the spec's `opponents` ROUTE TABLE (`external` / `policy` slot / `bot`), `ep_opp` in (read with the teams at every start), `opp_route` / `opp_slot` out; out-of-table = `CallerError`; Rust tests (auto-resets, park, thread invariance) | BUILT |
| 2 | HOST: `OpponentPlan` (route table from the production inputs), `EpisodeOpponentSampler` (= `_select_episode_opponent` draw for draw), `SlotFamily` (refresh = LOAD into a free slot; in-use never overwritten; `SlotCapacityExceeded`), `PolicyOpponentServer` (T2 adapter; `sample_actions` = `RLPlayer`'s multinomial bit for bit), `RustEnvOpponents` (staging one episode ahead, live pushes) | BUILT |
| 3 | BOTS in the core (Lane F's `Bot::decide`, wired after 02b0ab14): bot routes with a DECLARED seed rule (`stream_seed`, Python twin `bot_stream_seed`), BaitBot's `p_bait` declared; a bot decides only at real decisions and is never exposed | BUILT |
| 4 | The opponent-side STALL rule: a POLICY route's p2 decision at turn >= threshold is never exposed (today's `RLPlayer` stall check precedes any forward); p2 forfeits if p1 has no decision open | BUILT |
| 5 | FINDING fixed at the source: `gen3_no_phantom_opponent_poll_v1` (SingleAgentWrapper polls the opponent only when it will send the order) | BUILT (training-input change) |
| 6 | THE GATE: `rust_env_opponents_parity` record (core + T2) / replay (Gen3Env + the per-env RLPlayer path, CUDA-less child); COMMIT (routine) + compiled (`slow`) + GPU MILESTONE | BUILT; results below |
| 7 | Throughput at production shape (48 envs, pool of 20, production mix incl. in-core bots) | MEASURED (descriptor) |

## Gate results

| tier | T2 backend | per-env path | pool | decisions | divergences | max \|Δ legal logp\| |
|---|---|---|---|---|---|---|
Assertion rule (declared, `judge_flips`): an argmax flip with top-2 margin < 2 x the tier's bar is a counted TIE; any other flip, or any |Δ| above the bar, is FATAL. Bars: eager 1e-5, compiled CPU 1e-5, GPU 1e-3.

| COMMIT greedy (routine) | eager CPU | eager CPU RLPlayer | 2 perturbed fresh | 615 (12 eps, trial) | 0 | 9.5e-7 |
| COMMIT sampled, stall threshold 6, refresh at step 20 | eager CPU | eager CPU | 2 + 1 refreshed | 259 (49 eps) | 0 (0 near-ties) | 9.5e-7 |
| compiled (trial) | eager CPU | `--compile-opponents` CPU | 2 fresh | 386 | 0 | 1.4e-6 |
| GPU MILESTONE sampled (`gate_gpu_sampled.json`) | graph CUDA, lanes 8, buckets 2/8/16 | compiled CPU | ai_v14_06_lbat_ctrl_fix 6 + 1 refreshed | 2,052 (64 eps, 16 envs) | 0 (2 near-ties, both equal) | 2.7e-5 |
| GPU MILESTONE greedy (`gate_gpu_greedy.json`) | graph CUDA, lanes 8, buckets 2/8/16 | compiled CPU | ai_v14_06_lbat_ctrl_fix 6 + 1 refreshed | 1,725 (66 eps, 16 envs) | 0 (5 near-ties, all equal) | 4.5e-5 |

Lifecycle: every run's T2 `compiles/captures/cuda_segments_after_freeze` delta 0 across the refresh
(a LOAD: `svc.counters["loads"]` = startup snapshots + refreshes); the core's `*_AFTER_FREEZE` 0.

GPU runs: the refresh at step 120 LOADED the held-back snapshot into route 6 with every
`*_after_freeze` delta 0 (`svc_loads` 7 = 6 + 1), but no episode on it completed before the run's 64
episodes were reached — the "refreshed snapshot is played" assertion is covered by the CPU COMMIT
tier only.

## Throughput at production shape (`bench_48env_pool20.json`, 2026-09-29, DESCRIPTOR)

48 envs, T = 8, release core; pool = the last 20 snapshots of `ai_v14_06_lbat_ctrl_fix` in 24 slots
(20 + 4 spare), 8 lanes, buckets 2/4/8/16, graph backend, torch 2.5.1; self-play fraction 0.9 + the
8 training-roster bots IN THE CORE; p1 a seeded random policy (the trainee's forward is NOT in
these numbers — Lane G). ⚠️ Lane S's process (1700054) held ~5.7 GB on the GPU and was computing
intermittently during the run (`busy_box_warnings`): no ratio is claimed from these numbers.

| per env step (600 timed) | mean | p95 |
|---|---|---|
| opponent SERVE (gather + T2 flush + sampling) | 5.90 ms | 6.18 ms |
| — of which T2 flush (≈ 40 rows over ≈ 17.6 slots) | 1.15 ms | |
| — of which SAMPLING (per-row generators, Python loop) | 5.09 ms | |
| core STEP (48 envs, bots in core) | 1.65 ms | 1.94 ms |
| p1 random + staging | 0.21 ms | |

* ≈ 5,970 trainee decisions / s end to end at this shape (without the trainee forward).
* Rows per slot per flush: 1 (3,690), 2 (3,408), 3 (1,497), 4 (1,286), 5 (461), 6 (212) — buckets 2
  and 4 carry almost everything; 8 / 16 are nearly idle at a pool of 20 (input to the SIZING study).
* Startup 425 s (build 416 s = 4 bucket compiles + 24 × 4 captures; parity 8.8 s); a slot LOAD
  0.81 s mean (the in-place copy + the per-load parity verify), the initial 20-snapshot admit 16 s.
* Every lifecycle counter 0 (T2 and core); 736 episodes ended, 0 quarantined; 68 bot / 716 pool.

## Findings

- **F-LE-1 (FIXED at the source, `gen3_no_phantom_opponent_poll_v1`, a TRAINING-INPUT change):**
  `SingleAgentWrapper.step` polled `opponent.choose_move` on every step whose `battle2` was not a
  `wait`, including steps whose p2 order is never sent (`agent2_to_move` False). For a POLICY
  opponent that phantom poll embedded the stale request and RECORDED A DECISION: its progress clock
  (`turns_since_progress`, reactive offset 2, obs cell 1606) read one step high at the following
  decisions — measured 5 of 281 opponent rows in the first trial, every one directly after a
  phantom (4 phantoms / 281 decisions) — and it drew a sample from its generator. For a BOT it drew
  from its RNG (F-LF-2). The opponent-side twin of F1 (`gen3_no_phantom_decision_v1`). Fix: poll
  only when the order will be sent (the default order otherwise, as for `wait`); the sim sees
  nothing new. Regression test fails on revert. Hand-off: Lane F's COMMIT bot bank (`src/rust_env/tests/fixtures/bots/commit_corpus.json.gz`) re-recorded with `python -m utils.rust_env.bot_corpus --commit-tier --write` (its phantom polls are gone); `bots_gate_test.py` green on it. Closes F-LF-2. Every pre-fix run's self-play opponent saw the
  corrupted clock on ~1–2% of its decisions (not re-measured at scale).
- **F-LE-2 (built into the core):** today's `RLPlayer` FORFEITS at the stall threshold
  (`Gen3Player._handle_stall`) and does so BEFORE any forward — Lane D's core modelled only p1's
  forfeit. The core now hides a policy route's p2 decision at turn >= threshold (measured: before
  the fix, every low-threshold episode had one extra served p2 decision — 42/42). The p2-forfeit
  branch itself (p2 decides alone at turn >= threshold) is **UNVERIFIED: argued unreachable** in
  gen-3 singles (p1 decides at every turn start; one-sided forced switches precede `|turn|`); not
  observed in 49 low-threshold episodes.
- **F-LE-3 (declared, for Lane G):** a draw happens ONE EPISODE EARLIER than today (staging for the
  auto-reset); with no live push the route sequence equals the wrapper's (pinned), a push reaches an
  env one episode later. The pool is scanned once per generation by the host, not per worker reset.
  Every stream is seeded (today's defaults are unseeded, so no stream identity with a live run is
  possible or claimed).
- **F-LE-4 (for Lane G):** re-stage `ep_opp` AND `ep_team` / `ep_seed` for every env whose `episode`
  column MOVED (an auto-reset, a quarantine restart, a refused start and the parked env's later
  start all consume the staged row; `done` alone misses the parked start). `RustEnvOpponents.
  after_op` returns exactly that set and cross-checks `opp_route` against what it staged.
- **F-LE-5 (for Lane G):** per-opponent PINNED TEAMS (stable / exploiter `team_strs`) are carried on
  `Route.team_strs`; mapping them to `ep_team` indices needs those teams in the core's startup team
  table — G's staging. `opp_class` per episode is `RustEnvOpponents.opp_class(cols)` (Lane C's
  host-filled key).
- **F-LE-6 (for Lane G / H):** the exploiter LADDER swaps weights between episodes; the plan gives
  the exploiter TWO slots when `--exploiter-ladder` is declared so a rung LOADS into the idle one
  (`set_exploiter_rung`). The temperature push is immediate (sampling is the host's).
- **F-LE-8 — CORRECTED 2026-09-30 (Lane G + the coordinator): the "5.1 ms of SAMPLING" below was
  MISATTRIBUTED.** `serve()` billed the GPU forward's completion wait (`ticket.host()` → the event sync)
  to `sample_s`; the draws themselves cost 0.13 / 0.25 / 0.38 ms at 8 / 40 / 48 rows. The timer is now
  split — `ServeStats.wait_s` (the wait) vs `draw_s` (the draws), `sample_s` = their sum kept as a
  property — and the real cost at this shape is the opponent forward FAN-OUT (~18 slot replays per
  step); Lane G's T2 fan-out read (`main.rust_core_m5.fanout`) measures it. The keyed draw Lane G
  adopted is still right, for REPLAYABILITY, and saves ~0.2 ms, no more. The original entry, as
  believed at the time: SAMPLING is 5.1 of the 5.9 ms opponent
  serve: `sample_actions` draws each row's Exp(1) variates from that env's own `torch.Generator` in
  a Python loop, to stay bit-identical to today's per-player `torch.multinomial` (what makes the
  sampled gate EXACT). Options: one batched draw from a single seeded stream per step (distribution-
  equal, not stream-equal — the gate would drop to a stated distribution test), or the draw moved
  into T2's graph. Not done here (it changes the gate's contract; G's call).
- **F-LE-9 (for Lane G / the SIZING study):** at 48 envs and a pool of 20, 90 % of per-slot row
  counts are 1–4; buckets 8 / 16 cost startup (~100 s compile each) and serve almost nothing.
- **F-LE-10 (= Lane J's F-LJ-6, FIXED 2026-09-30):** `test_slow_compiled_per_env_path` failed whenever it
  ran after other tests. ROOT CAUSE: leaked PROCESS state — `record()` called `torch.set_num_threads(4)`
  and never restored it, so the next `run()` built its fresh snapshots at 4 threads instead of the
  process's 8; a fresh policy's orthogonal init (QR) is thread-sensitive (94 of 721 tensors differ, up to
  7.5e-6), so the weights moved by rounding, the battles did not, and at the one EXACT tie of the run
  (env 3, episode 3, decision 40: two switch targets at logp -2.0050578 each, margin 0.0) eager's argmax
  (8) and the compiled path's (9, off by 2.4e-7) disagreed. Bisected by diffing the torch globals and the
  recordings of the two orders (identical rows and actions, 1,720 of 1,881 logp rows differing by <= 1.4e-6,
  94 weight tensors differing). The eager forward itself is bit-reproducible at a fixed thread count and
  differs between 1 / 4 / 8 threads (probe). FIX: `declared_torch_state` (pin, verify precision / dtype /
  threads on entry AND exit, restore; `GateStateError` otherwise) around the snapshot build (1 thread),
  the recording (4) and the replay child (1); the tie rule DECLARED in the assertion (`judge_flips`).
  Regression tests fail on revert (the build pin; the restore). VERIFIED: the compiled configuration after
  the file's other tests and alone record byte-identical battles (sha `b48b83d0…`), 0 flips; a full-order
  session of the whole file under the GPU lock: 10 passed (1,881 / 2,052 / 1,725 decisions, 0 flips).
- **F-LE-7:** a stable / exploiter opponent of ANOTHER architecture (same obs family, other weight
  shapes) needs its own T2 slot group — `PolicyOpponentServer` takes the service as built; grouping
  by state-dict signature at startup is G's (T2's `SlotArchMismatch` refuses a wrong load).
