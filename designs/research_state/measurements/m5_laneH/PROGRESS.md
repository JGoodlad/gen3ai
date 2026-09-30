# M5 Lane H — eval on the Rust core: PROGRESS (resume point)

Lane H of `designs/endstate/program_rust_core.md` §2 M5 (the lane row, the "Lane H BUILT" paragraph,
the Decision record). The design of record is `designs/training/eval_and_rating.md` → "Eval on the
Rust env core". Worktree `/home/goodlad/dev/gen3ai-wt/m5-laneH`, branch `m5-laneH`.

**Owns:** `src/agents/training/rust_eval/` (seeds, traces, executor, build, launch, parity, benchmark +
their tests), the eval callbacks' `--env-core rust` branch (`eval_callback.py`, `selfplay_callback.py`,
`eval_callback_rust_test.py`), `eval_sharding/units.game_range`, the eval worker's per-game seed mode
(`main/eval_worker.py` `seed_rule = "per_game"`), `--rust-eval-envs`. **Hand-offs (other lanes' files):**
the core's three seams below (Lanes 0 / A / B / E files), `rust_rollout/build.py`'s `extra_slots`
(Lane G), `rust_env_setup.py`'s eval declaration (Lane G), the prober's core-trace expander
(`src/main/prober/core_trace.py` + session wiring) and `obs_materializer._next_tag`'s prefix fix.

## How to test

```bash
export PYTHONPATH=$PYTHONPATH:src
python3 -m pytest src/agents/training/rust_eval/ src/agents/training/eval_callback_rust_test.py \
    src/main/prober/core_trace_test.py src/main/prober/core_trace_integration_test.py -q -m "not slow"   # ~80 s
python3 -m pytest src/agents/training/rust_eval/parity_test.py -q -m slow -k cpu                      # CPU milestone, ~15 min
GEN3AI_TEST_ALLOW_GPU=1 scripts/ops/gpu_lock.sh python3 -m pytest \
    src/agents/training/rust_eval/parity_test.py -q -m slow -k gpu                                    # GPU milestone
python -m agents.training.rust_eval.parity --games 25 --sentinels 5 --n-envs 64 --out gate.json      # the gate as a script
```

## Units

| # | unit | status |
|---|---|---|
| 1 | THE CORE SEAMS (`src/rust_env`): a bot route's `"streams": "episode"` (the bot re-seeded at every episode start from the route seed and the episode's battle seed — `opponents::episode_stream_seed`, Python twin `rust_env_opponents.episode_bot_stream_seed`); `Core::finished` (every episode that ENDED in the last op: env, episode, winner, end turn, forfeit side, input-log script — FFI `rust_env_finished_json`, process control `FINISHED`); the pure record writer `rust_env_trace_json` (`crate::trace`: a script → both sides' `gen3_core_event_v1` records, round-trip and re-parse checked, + the reconstruction record) | BUILT |
| 2 | THE PER-GAME SEED RULE (`gen3_eval_game_seed_v1`, `rust_eval/seeds.py`) shared by both paths; the eval worker's `seed_rule = "per_game"` | BUILT |
| 3 | THE EXECUTOR (`rust_eval/executor.py`): units on envs in plan order, filler, the trainee greedy through T2 at `Priority.EVAL`, sentinels greedy / keyed sample, the stall rule, `Core::finished` bookkeeping, per-unit `ShardResult`s, the lifecycle check | BUILT |
| 4 | THE DECLARATION + WIRING: eval slots in the ONE T2 declaration (`build_collector(extra_slots=)`), the eval core (`rust_eval/build.py`), `--rust-eval-envs`, both callbacks' in-process branch (`rust_eval/launch.py`), `RustEvalUnavailable` | BUILT |
| 5 | CORE TRACES (`gen3_core_trace_v1`, `rust_eval/traces.py`) + the prober's expander (`main.prober.core_trace`, cross-checked against the record) | BUILT |
| 6 | THE GATE (`rust_eval/parity.py`): COMMIT routine, MILESTONE `slow`, GPU MILESTONE; the M5 registry row | BUILT |
| 7 | Eval wall-clock per cycle, both paths, production eval shape (`rust_eval/eval_benchmark.py`) | MEASURED (below) |

## The gate — the same seed set on both paths

| tier | trainee / sentinels | Rust T2 vs Python forward | games equal | ties | fatal | trainee decisions | max \|Δ logp(chosen)\| (bar) | metrics | traces checked / diffs |
|---|---|---|---|---|---|---|---|---|---|
| COMMIT (routine) — 9 bots + 2 sentinels × 2 games, shard 1, 8 envs | perturbed fresh | eager CPU vs eager CPU | 22 / 22 | 0 | 0 | 1,038 | 8.4e-7 (1e-5) | EQUAL | 22 / 0 |
| MILESTONE CPU (`gate_milestone_cpu.json`) — 9 bots + 5 sentinels × 25, shard 25, 64 envs (the production eval shape at 25 games) | perturbed fresh | eager CPU vs eager CPU | **350 / 350** | 0 | 0 | 17,508 | 1.28e-6 (1e-5) | **EQUAL** | 40 / 0 |
| MILESTONE GPU (`gate_milestone_gpu.json`) — 9 bots + 5 sentinels × 20, shard 25, 64 envs | `ai_v14_06_lbat_ctrl_fix` final + 5 of its pool snapshots (32M … 82M) | T2 `graph` CUDA (lanes 6, buckets 8/48) vs the Python worker's COMPILED CPU extractor | **279 / 280** | 1 (margins 4.8e-7 / 9.5e-7, the game still ended equal) | 0 | 9,242 | 1.48e-5 (1e-3) | **EQUAL** | 40 / 0 |

Every game's winner, end turn and every trainee action; the pooled metrics (win rates, reward means,
episode lengths, exact W/L counts, draws, trace-selection tuples; TD tails within 1e-4); the same trace
FILES kept on both sides (same games, same names), and each Rust core trace expanding in the prober to
the Python trace's decisions (turn, phase, chosen, both actives). Lifecycle: every core and T2
`*_after_freeze` counter 0 after every cycle. Teeth: a Python path on another seed set is FATAL; one
tampered shard record is a metric difference; the tie rule excuses only a flip under its margin; the
default `"env"` bot-stream rule demonstrably depends on history (`core_seams_integration_test`).

**The M5 registry** (`python -m main.rust_core_m5 gates --tier milestone --lanes H --gpu`, 2026-09-30,
`m5_laneJ/results/gates_milestone_lanes_H.json`): **H PASS 29/0/0/0, GPU part PASS**; with E (30/0, GPU PASS)
and G (55/0) from the same day's E,G,H run, `python -m main.rust_core_m5 verdict` reads **M5 GATE: MET**
(every lane PASS; slice N and the depth-3 slice PASS; the throughput A/B MEASURED).

**On the real launch path** (a CPU fork of `ai_v14_06_lbat_ctrl_fix`'s 82.9M checkpoint with the parent's
argv, `--env-core rust --self-play --n-sentinels 3 --eval-games 4`, archive dir): T2 declared one slot
group of 29 slots (24 pool + trainee + the eval trainee + 3 sentinels); the `SelfPlayCallback` cycle played
48 games on the eval core in 26 s (CPU eager T2), recorded bots 90.6 % / pool 58.3 %, promoted, persisted
the snapshot; `python -m main.prober.query summary|scan|turns|analyze|falsify` all ran on its traces
(`analyze` re-ran the model at the current architecture: its argmax agreed). A `--debug --debug-eval`
PerOpponent smoke: two cycles of 18 games in ~5 s each.

## Eval wall-clock per cycle (DESCRIPTOR; `bench_*.json`)

`python -m agents.training.rust_eval.eval_benchmark` (`bench_eval_cycle_production_shape.json`, 2026-09-30).
**Shape:** the production self-play eval cycle, which is 9 roster bots + 5 pool sentinels × 100 games
(`EVAL_GAMES`), shard 25, so 56 units and 1,400 games. The trainee is `ai_v14_06_lbat_ctrl_fix` final;
the sentinels are 5 of its pool snapshots.
**Arms, interleaved R P P R:**
- **Python** is today's live cycle: 10 work-stealing `main.eval_worker` processes (`--eval-workers 5` × 2
  under `--self-play`), CPU, the compiled extractor, the rust bridge, unseeded.
- **Rust** is the eval core: 64 envs, 8 threads, the process front end, the release build. T2 runs
  `graph` on CUDA, lanes 6, buckets 8 / 48. The GPU lock is held for the Rust arms only.
**Regime:** contention factor 1.0 at every arm's start. load1 was 0.8–1.0 before the first arms; the
Python arms drove it to 8–10 themselves. No other process of note ran on the box, and no training run
was live.

| arm | wall per cycle (s) | games | trainee decisions | notes |
|---|---|---|---|---|
| Rust R1 (2 cycles) | 16.28 / 16.60 | 1,400 / 1,400 | 46,389 / 47,017 | in-cycle split, cycle 1: T2 serve + host (act) 4.2 s, core 2.7, finish 4.0, traces 3.9 (221 kept), loads 0.45; the rest (~3.7 s) is loading the 5 sentinel zips from disk |
| Python P1 | 116.0 | 1,400 | — | all 10 workers exit 0, no missing shard |
| Python P2 | 118.6 | 1,400 | — | |
| Rust R2 (2 cycles) | 16.23 / 16.71 | 1,400 / 1,400 | 46,389 / 47,017 | the same games as R1 (same cycle seeds), bit for bit in counts |

- **Rust ≈ 16.5 s vs Python ≈ 117 s per cycle, about 7.1× less wall-clock.** Per-arm spread is under
  3 %; no CI is claimed on two blocks per arm.
- The two cost different resources. The Rust cycle BLOCKS the trainer for ~16.5 s, i.e. ~1 % of a
  ~2M-step eval interval (at Lane G's production rate: 3,780 decisions/s plus ~20 updates of ~62 s, ≈ 30 min). The Python cycle runs BESIDE training, on 10 CPU
  processes the rollout would otherwise use.
- T2's startup here, 40 s for the benchmark's own 6-slot service, is paid once per process. In training,
  eval's slots join the trainer's ONE service: +2 + `--n-sentinels` slots, captures only, no new compiled
  bucket. That share of the trainer's T2 startup was NOT measured.
- Every lifecycle counter was 0 across all 4 cycles. T2 counters: 12 loads (2 cycles × 6 slots) and
  0 compiles / captures / segments after the freeze.

## Findings

- **F-LH-1 (declared change of stream):** a Rust eval game is seeded by the GAME (`gen3_eval_game_seed_v1`);
  today's live Python eval is unseeded (global `random`, OS-entropy teams). Same distributions, a declared
  stream — no live Python cycle can be reproduced, which is why the gate adds a per-game-seeded mode to the
  Python worker rather than comparing against a live cycle.
- **F-LH-2 (declared change of timing):** the Rust cycle is IN PROCESS and BLOCKING between two host steps
  (today: non-blocking subprocesses). Its cost is the wall-clock above; the end state's background FILLER
  interleave (T2's `EVAL` priority riding the rollout's flushes) is NOT built — a cutover-era decision
  if the blocking cost matters at the sizing study's N.
- **F-LH-3 (a core refusal fails the cycle):** a QUARANTINED eval battle raises `EvalCoreError`; the callback
  logs it and records nothing for that cycle (a crashed worker's shape). None observed.
- **F-LH-4 (traces carry no auxiliary heads):** `win_probs` NaN, no belief / intent / value-dist rows in the
  npz; no `_replay.html` (the prober renders the protocol from the expansion). `analyze` re-runs the model.
- **F-LH-5 (readers OUTSIDE the prober see meta only):** `cf_audit`, the search teacher's selection /
  generation, `harvest`, `probe_replay`, `audit_states`, `scaffolding_gauge` (G7 stall rate uses meta only —
  fine), `ops/eval_trace_gen`, `ops/conditioning_meters`, `ops/quota_match`, `mechanic_usage_baseline`,
  `search_dividend/*` read `*_summary.json` directly and would see ZERO invocations on a Rust-eval run. They
  must switch to `main.prober.core_trace.load_summary` (or refuse a core trace) before any Rust-eval run's
  traces feed them — a cutover item (program M7).
- **F-LH-6 (prober, fixed):** `obs_materializer._next_tag` passed a non-Showdown battle tag through, and
  poke-env silently refuses such a room: every counterfactual view on a core trace read "replay desync".
  Now prefixed; well-formed tags unchanged.
- **F-LH-7 (T2, known — Lane T2's):** a failed per-load parity check POISONS the whole service; eval LOADS
  the trainee and every sentinel each cycle, so a sentinel that fails parity would take training down with
  it. Correct by the T2 contract (refuse), noted because eval adds loads.
- **F-LH-8 (the reward the prober re-scores):** the expanded summary's per-turn `outcome.reward` is scored
  with `model_config.json`'s `RewardConfig` (the Python worker's source); the Rust cycle's reward is the
  core's terminal from the COLLECTOR's resolved `RewardConfig.from_args` — the same config on every run
  whose argv and `model_config.json` agree (the gate derives both from `model_config.json`).
- **F-LH-9:** the expansion replays each core trace once per prober process (0.03–0.7 s; cached in memory).
- **F-LH-10 (not exercised):** fixed / stable opponents on the Rust eval path (their slots are declared and
  reused from the training plan; no gate game used one); the sampled-sentinel regime
  (`--no-eval-sentinel-greedy`: keyed draw keyed by the game — distribution-equal only, no gate); a p2
  forfeit in a trace expansion; the launcher's restart loop around a Rust-eval run (F-LG-6's gap).

- **F-LH-11 (harness hazard, Lane J's):** `python -m main.rust_core_m5 gates … --gpu` takes
  `/home/goodlad/.claude/jobs/gpu.lock` ITSELF around its GPU pytest. Wrapping the whole command in
  `flock` on the same lock DEADLOCKS: the inner flock waits forever, at 0 % CPU, on a lock its parent holds
  (happened 2026-09-30 and cost one ~30-min CPU pass). The class fix is the re-entrant `utils.gpu_lock` /
  `scripts/ops/gpu_lock.sh` (`40d2998c`: never a bare flock); `rust_core_m5` migrates to it after Lane H.

## Resume point (2026-09-30)

Units 1–7 built; the registry row BUILT. What remains before a CUTOVER (none of it gating H):

1. F-LH-5: the non-prober trace readers onto `load_summary` (or a loud refusal).
2. F-LH-2: background (filler) eval, if the SIZING study's N makes the blocking cycle expensive.
3. F-LH-10's unexercised paths: a gate row with a fixed opponent and one with the sampled regime.
4. The launcher restart / pin path on a Rust-core run (Lane G's F-LG-6, now also eval's).
