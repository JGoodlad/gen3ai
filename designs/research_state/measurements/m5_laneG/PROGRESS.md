# M5 Lane G — training integration: PROGRESS (resume point)

Lane G of `designs/endstate/program_rust_core.md` §2 M5 (the lane row, the "Lane G" paragraph, order
constraints 5 and 6, the Decision record). The design of record is
[`designs/training/rust_collector.md`](../../../training/rust_collector.md). Worktree
`/home/goodlad/dev/gen3ai-wt/m5-laneG`, branch `m5-laneG`.

**Owns:** `src/agents/training/rust_rollout/` (store, trigger, teams, collector, build, consistency,
parity, testkit + their tests), `src/agents/training/keyed_draw.py`, `src/agents/training/rust_vec_env.py`,
`src/main/train/rust_env_setup.py`, `src/main/train/parser/env_core.py`, `designs/training/rust_collector.md`.
**Hand-offs:** `rust_env_opponents.PolicyOpponentServer` split into `submit` / `complete` (serve = both,
byte-identical) + a `keyed` sampling mode (Lane E's file); `win_prob_callback.backfill_terminal_labels`
(the callback's scan as a function, one owner); the trainer wiring (unit 2's list).

## How to test

```bash
export PYTHONPATH=$PYTHONPATH:src
python3 -m pytest src/agents/training/keyed_draw_test.py src/agents/training/rust_rollout/ \
    src/agents/training/rust_vec_env_test.py -q          # unit + integration (CPU, ~30 s warm)
```

## Units

| # | unit | status |
|---|---|---|
| 1 | THE COLLECTOR + BUFFER: the row arena, complete-game GAE (sb3 bit for bit), the complete-game and window fills, the sample-count trigger + adaptive hook, team / seed staging, the host loop (one T2 flush for trainee + opponents), the keyed draw for the trainee, version pinning (declared, off), `RustVecEnv` + its surface table | SHIPPED `11073fe1` |
| 2 | THE TRAINER WIRING behind `--env-core rust` (python stays the default): parser family (`parser/env_core.py`), combination refusals (`env_core_rust_*`), the startup hook BEFORE the trainer compile, `collect_rollouts` routing + the T2 LOAD after each update, K9(b) + the staleness probe (`consistency.py`), `rollout/collect_ms` on both cores, `rust_env/*` tags, `metadata.json`'s `env_core` | SHIPPED `ac67fa6c` |
| 3 | THE PARITY GATES: slice N at the ROLLOUT level (window mode vs today's Python path, `rust_rollout/parity.py`) + the learner-level check; COMMIT routine, MILESTONE `slow` | SHIPPED `a2b26acb` |
| 4 | THE SAMPLING CHANGE for policy opponents (F-LE-8, keyed default) + buckets (F-LE-9) + Lane J's hooks (learner sampling, production mix, complete-game collector; G's row BUILT) + the owner's registered 95 / 5 throughput A/B (Python / Rust serial / Rust overlapped, keyed before / after) + the fan-out levers (T2 fan-out read, fewer active snapshots) + the core-respawn recovery + the ragged-micro-batch rule and K10(c)'s dose count (coordinator, 2026-09-30) | BUILT (this commit) |

## Measurements so far (descriptors)

- CPU, T2 eager, N = 48, T = 8, proc front end (release), fresh perturbed policy, p2 random external
  (2026-09-29, `/tmp` script): **1,319 trainee decisions/s**; per host step the T2 flush 31 ms (eager CPU,
  the whole cost), the core 2.2 ms, the collector's own work (submit + draw + write + post) 0.8 ms; the
  fill of 6,144 rows 16 ms; every lifecycle counter 0. Not a production-shape read (no GPU, no policy
  opponents) — unit 4's benchmark is.

## The GPU smoke on the real launch path (unit 2's gate; 2026-09-30, `/tmp/laneG/gpu_smoke1.log`)

`train_rl_agent.py` (NOT through the launcher — F-LG-6) forked from `ai_v14_06_lbat_ctrl_fix` final
(83.07M) with the parent's full argv + `--env-core rust --n-envs 48 --steps 83420000 --eval-freq 1e9`
(eval pushed out: the eval workers are Lane H's and would share the CPU), under the GPU lock, fresh
Inductor / Triton caches, `--compile-trainer` on, self-play 90 % against the parent's 20-snapshot pool
in T2 (24 pool slots + the trainee = one slot group of 25, buckets (8, 48), 8 lanes, graph backend),
the 8 training bots in the core, opponent sampling KEYED. Idle box otherwise (load ~2).

- **Startup:** T2 205 s (2 bucket compiles + 50 captures + the per-slot × bucket parity gate); the core
  0.1 s (717 teams validated by use); then the learner's compile. Every `*_after_freeze` 0 (checked
  after each update).
- **4 updates, each 98,304 rows:** collection **21.9 s** per update = **4,510 trainee decisions/s**
  (44.1 trainee rows per host step; per host step: T2 flush 1.16 ms serving the trainee AND ~41
  opponent rows, core 1.61 ms, the collector's own work 0.48 ms; the fill 235 ms per update);
  `train_ms` 58–66 s. 2,170–2,230 games per update, 0 quarantined, 0 cut.
- **K9(b):** max |Δ log π| on current-version rows **9.1e-6 – 1.45e-5** (bar 1e-4), T2's CUDA graph vs
  the compiled learner (1,024–2,048 rows probed per update).
- **Staleness (the owner's MEASURE):** after the first update **1.1–1.4 % of rows are one version old**
  (the games straddling an update), none older; one game split per update (36–69 rows carried).
  Age-1 rows at the start of the update: ratio mean 0.995–1.000, mean |r − 1| 0.045–0.068, outside the
  clip band (0.15) **6–12 %**, approx-KL 0.003–0.006. Age-0 rows: ratio 1 ± 9e-7. Nothing here calls
  for version pinning (declared, OFF).
- **Against today's path (NOT an A/B — a descriptor):** the parent run's own log (the Python path,
  2026-09-28, 48 envs, same recipe) reads ~198 s per iteration with ~60 s of `train_ms`, i.e. ~138 s of
  collection per 98,304 decisions ≈ **712 decisions/s**; that run also had its eval workers and whatever
  else shared the box. The interleaved A/B is unit 4 (Lane J's harness).

## The rollout-level slice N + the learner-level check (unit 3; `rust_rollout/parity.py`)

RECORD in Rust (the collector in WINDOW mode, a fixed perturbed-fresh production policy, p2 an external
seeded-random route), REPLAY in Python (`InstrumentedMaskablePPO.collect_rollouts` over a `DummyVecEnv`
of production-surface `Gen3Env`s in `Monitor(MaskableAgentWrapper)`, `WinProbLabelCallback` registered;
the one substitution: the policy forward's SAMPLE is the keyed draw from the replay's own log-probs).

| tier | rows | games | win-labelled rows | divergences | max \|Δ\| values / log-probs / adv / returns | learner: max \|Δ param\| (update moves) | scalars within allowance |
|---|---|---|---|---|---|---|---|
| COMMIT (routine) — 4 envs × 48 × 2 windows, pool | 384 | 8 | 151 | 0 | 3.0e-7 / 6.0e-7 / 2.9e-7 / 1.2e-7 | 1.2e-7 (3.0e-4) | 234 / 234 |
| MILESTONE pool — 8 × 128 × 3, team offset 11 (`parity_milestone_pool.json`, re-run 2026-09-30 under the tie rule) | 3,072 | 48 | 2,126 | 0 (0 flips; near-boundary rows at 2e-5: 1 Rust / 1 Python) | 3.6e-7 / 7.2e-7 / 3.5e-7 / 1.2e-7 | 6.0e-8 (3.0e-4) | 249 / 249 (worst 2 % of allowance) |
| MILESTONE ladder — 8 × 128 × 3, team offset 11 (`parity_milestone_ladder.json`, re-run 2026-09-30 under the tie rule) | 3,072 | 45 | 1,933 | 0 (0 flips; near-boundary 2 / 2) | 3.6e-7 / 6.0e-7 / 3.6e-7 / 3.6e-7 | 6.0e-8 (3.0e-4) | 250 / 250 (worst 1.7 %) |

EXACT on every observation key (the row, the mask, all 18 core labels, `opp_class`, `win_target`,
`win_mask`), actions, masks, rewards, episode starts; every recorded trainee action reproduced by the
replay's own keyed draw. Since 2026-09-30 a disagreeing trainee action is judged by Lane E's margin rule
(`judge_flips`: a TIE only if both sides' keyed-draw margins are below 2 × 1e-5, the recorded action played
on; any other flip fatal) with ONE near-boundary threshold (2e-5) on both sides — the re-run milestones
met no flip, and the two sides counted the SAME near-boundary rows (1 / 1 pool, 2 / 2 ladder).
Teeth: one Python label cell moved fails the slice (per key); the comparator refuses one bit and a
float past its bar. **The learner check is ONE optimizer step** (one epoch, the buffer in 4 accumulated
micro-batches): with 8 optimizer steps it read 3.4e-4 (0.15 of the update's movement) at the milestone —
iterated PPO steps amplify a ≤ 7e-7 input difference; the same learner on the same buffer twice is
bit-identical (F-LG-8).

## Unit 4 — the owner's registered throughput comparison (95 % self-play / 5 % bots, 48 envs)

**Regime (every row stamped with it):** Lane J's harness (`python -m main.rust_core_m5 throughput`,
`--opponent production --collector complete_game --inference learner`), 48 envs, a 20-snapshot pool of
`ai_v14_06_lbat_ctrl_fix` (T2: 24 pool slots + the trainee = 25 slots, one slot group, buckets (8, 48),
8 lanes, `graph` backend, fp32 `highest` — the run's own precision), the 8 training-floor bots (in the
core on Rust; `create_training_env_random`'s workers on Python), trainee = that run's final checkpoint,
release builds, the GPU under the lock, FRESH per-run Inductor / Triton caches, 50 untimed warm-up steps
per arm, then 6 rounds of interleaved 20 s blocks (rotating order). CIs: percentile bootstrap over
blocks (per arm) and over rounds (ratios). No bystander in the measured trees; load1 ~2 at the start,
driven up by the Python arm's 48 workers. `results/throughput_production_sp95_n48.json` (Lane J's dir).

| arm | trainee decisions / s [95 % CI] | ms / step [95 % CI] | CPU µs / decision | GPU util |
|---|---|---|---|---|
| Python (today's path) | 743 [674, 783] | 65.4 [61.3, 72.8] | 12,993 | 6 % |
| Rust SERIAL, keyed opponent draw | **3,780** [3,650, 3,941] | **11.7** [11.2, 12.1] | **496** | 67 % |
| Rust serial, generator draw (before the keyed sampler) | 3,769 [3,593, 3,921] | 11.7 [11.2, 12.3] | 493 | 68 % |
| Rust OVERLAPPED, keyed | 2,690 [2,643, 2,732] | 16.3 [16.1, 16.6] | 723 | 86 % |
| Rust overlapped, generator | 2,662 [2,586, 2,726] | 16.5 [16.1, 17.0] | 731 | 85 % |

- **Rust serial vs Python: 5.11× [4.72, 5.63] decisions/s, 0.038× [0.037, 0.039] CPU per decision.**
  (The 90 % self-play read the day before: 5.10× [4.93, 5.26], `throughput_production_sp90_n48.json`.)
- **Keyed vs generator: 0.997× [0.957, 1.030] — no measurable difference.** In situ the opponents' draw
  costs 0.16 (keyed) vs 0.20 ms (generator) a step. The isolated micro-benchmark
  (`keyed_draw_benchmark.py`, 41 rows, idle box, 10 interleaved blocks × 400 calls,
  `keyed_draw_benchmark_r41.json`) agrees: keyed 0.049 ms vs generator 0.068 ms a call, 0.67×
  [0.61, 0.72]. That is a 0.02 ms saving. F-LE-8 corrected: its "5.1 ms of sampling" was the host
  waiting for the forward (the timers are now split: `gpu_wait` vs `opp_draw` / `draw`, and Lane E's
  `ServeStats.wait_s` vs `draw_s`). The keyed draw stays the default for REPLAYABILITY, not speed.
- **Overlap does NOT matter — it HURTS: 0.712× [0.690, 0.734] of serial.** Two halves (24 envs each, one
  shared T2 service), each half's core step on a worker thread while the other half's inference runs.
  It hides the core entirely (1.87 → 0.004 ms unhidden), but each half flushes its own opponent slots, and
  the forward's cost follows the number of distinct SLOT replays, not rows: the GPU wait goes from 7.1 to
  12.2 ms a step, GPU utilisation from 67 % to 86 %.
- **Where a Rust serial step goes (ms, the owner's five components; they sum to the step):** core step
  1.87 · T2 forward launch + flush 1.59 · **GPU wait 7.07** · sampling 0.25 (opponents 0.16, trainee 0.09)
  · host glue 0.88 (the complete-game fill amortised 0.58, arena writes 0.13, post 0.14, residual 0.03).
  The forward is 74 % of the step. Python: the learner's forward 9.8 ms, the workers' vec step (env +
  per-worker CPU opponent forwards + their sampling, 48 processes in parallel) 55.6 ms.
- Startup (fresh caches): Python 628 s (48 workers compile their opponents), the first Rust arm 430 s
  (T2's bucket compiles + 50 captures + the per-slot parity gate); every later Rust arm 20–41 s on the
  shared service. Every core and T2 `*_after_freeze` counter 0; 0 quarantines, 0 respawns.

### The T2 fan-out read (`main.rust_core_m5.fanout`; `t2_fanout_sp95_n48.json`)

One flush = submits + `flush()` + every `host()`, on REAL rows harvested from the production collector,
configs interleaved in seeded random order, 12 rounds × 50 flushes each, 95 % CIs over rounds (all within
±0.02 ms). The collector itself met **19.5 distinct opponent slots per step** (p10–p90 19–20) with 1–3
rows per slot in 90 % of cases.

| per flush (ms), T2's default 8 lanes | S = 1 | 2 | 4 | 8 | 12 | 18 |
|---|---|---|---|---|---|---|
| the trainee's 48 rows + 40 opponent rows over S slots | 3.76 | 4.58 | 5.70 | 4.40 | 6.32 | **8.43** |
| the 40 opponent rows alone | 1.92 | 2.72 | 3.96 | 2.73 | 4.49 | 6.60 |

- The trainee's 48 rows alone: 1.92 ms. At S = 18 the opponents' fan-out adds 6.5 ms. That matches the
  A/B's 8.7 ms a step of forward + wait.
- **A grouped forward across slots could save at most 4.7 ms per flush** (S = 18 → S = 1), i.e. 40 % of
  the serial step, a ~1.67× ceiling. It still reads every slot's weights, so it cannot beat the one-slot
  cost.
- **Lanes already do most of what concurrency can:** with 1 lane S = 18 costs 24.2 ms (1.20 ms per extra
  slot); with 8 lanes it costs 8.4 ms (0.28 ms per extra slot).
- **Buckets matter at the margin:** 4 slots × 10 rows pay the 48 bucket (5.70 ms), which is dearer than
  8 slots × 5 rows in the 8 bucket (4.40 ms). A 16 bucket is a SIZING-study input.

### The FEWER-ACTIVE-SNAPSHOTS candidate, end to end (`results/throughput_production_sp95_n48_active_snapshots.json`)

The same harness and regime (quiet box: load1 0.2 at the start). Four Rust serial keyed arms share one T2
service and differ only in how many of the pool's 20 snapshots the envs are routed to: all 20, or the K
newest (`_p<K>`; the arm's pool is a temp dir holding only those K, and the build refuses a pool that
holds more). The arms met 17.5 / 8.0 / 4.0 / 1.0 distinct opponent slots per flush.

| active snapshots | decisions / s [95 % CI] | ms / step | T2 launch + wait (ms) | vs all 20 |
|---|---|---|---|---|
| 20 (production) | 4,116 [4,092, 4,146] | 10.74 | 1.34 + 7.21 | 1 |
| 8 | 6,530 [6,510, 6,548] | 6.79 | 0.87 + 3.76 | **1.59× [1.58, 1.60]** |
| 4 | 5,913 [5,889, 5,932] | 7.49 | 0.68 + 4.70 | 1.44× [1.43, 1.44] |
| 1 | 7,151 [6,976, 7,258] | 6.21 | 0.57 + 3.46 | 1.74× [1.70, 1.76] |

- **One active snapshot is the end-to-end ceiling of ANY fan-out fix at N = 48: 1.74×.** That includes a
  grouped forward: one replay serves every opponent row. **Eight active snapshots take 80 % of it: 1.59×.**
  8 slots fit T2's 8 lanes, and ~5.6 rows per slot stay in the 8 bucket. Four do worse than eight: ~11
  rows per slot pay the 48 bucket.
- **The cost is the opponent DISTRIBUTION, so this is not adopted — it is a decision to bring back.**
  There are two ways to have K active:
  - **Shrink the pool window to K.** Fewer and more recent past selves. Diversity drops and cycling risk
    rises. A different experiment, not a speed knob.
  - **Rotate a K-subset of the same pool.** Change the subset faster than an update (~2,000 host steps at
    N = 48; e.g. every ~200 steps, longer than a ~45-decision game). Each update then still meets the
    whole pool in about today's proportions. Each EPISODE draws only from the active subset, a declared
    change of the per-episode rule: `EpisodeOpponentSampler` is today rule for rule
    `MaskableAgentWrapper._select_episode_opponent`, and Lane E's gate pins that.
  - The measured gain shrinks with N: rows per slot grow, so the fan-out amortises. The SIZING study
    re-reads it at its N.
- **The null read** (`results/throughput_production_sp95_n48_null_identical_arms.json`): an earlier run of
  the same four arms was VOID as a fewer-snapshots read. `SnapshotPool` applies `max_snapshots` only when
  ADDING, so every arm still routed to ~18 slots. It is kept as a NULL read: four arms of IDENTICAL
  configuration, differing only in their run seeds. Their ratios were 1.001 [0.995, 1.007],
  0.974 [0.969, 0.982] and 0.991 [0.985, 0.997]. **Arm-to-arm variation of ~3 % is NOT carried by a
  per-arm CI**, so no difference under ~3 % between Rust arms is a finding. The keyed-vs-generator read is
  far inside that; overlap and fewer snapshots are far outside it.

## Findings

- **F-LG-1 (design, declared):** the complete-game fill SPLITS at most one game per update (the one
  straddling the D-th row); its tail is trained on at the next update, one version older. Rows are never
  dropped; the split share is logged (`staleness/games_split`).
- **F-LG-2 (declared change of stream):** trainee sampling is the keyed draw, battle seeds are keyed hashes,
  teambuilder copies are seeded per env — none reproduces a Python-path stream (today's are unseeded).
- **F-LG-3:** a CUT game (quarantine / respawn) releases its rows uncounted in any buffer (no outcome, no
  next state); counted in `staleness/rows_cut_total`. None observed so far.
- **F-LG-4 (T2, for Lane T2):** the startup CONCURRENT gate puts one full chunk of the largest bucket per
  slot into ONE flush, so it overflows the declared `max_rows_per_flush` when n_slots × largest bucket
  exceeds it — a raw NumPy broadcast `ValueError` inside `startup()`, not a typed refusal (25 slots ×
  48 > 1,024). The collector now declares the arena at n_slots × the largest bucket; T2's `validate()`
  should refuse the combination by name.
- **F-LG-5 (Lane E's file, hand-off):** `PolicyOpponentServer` refused a service with MORE slots than the
  plan's policy routes; the trainee's slot(s) now follow them in the same service (one flush), so the
  check is `<`, not `!=`.
- **F-LG-6:** the real-launch smoke ran `train_rl_agent.py` directly, not through `main.launcher`
  (whose resume PINS to the checkpoint's commit — before Lane G there is no `--env-core`: a Rust-core
  resume must pin a commit that has it). The launcher's restart / crash loop around a Rust-core run is
  untested.
- **F-LG-8 (the learner check's scope, declared):** "one update on each buffer lands on the same weights" holds
  for ONE optimizer step (Δ 6e-8); over several steps PPO's clip gates and Adam amplify the paths' ≤ 7e-7
  log-prob / value rounding (the two paths forward different batch compositions, and CPU matmul rounds
  by batch shape) into 3.4e-4. Exact float equality would need batch-invariant forwards — not pursued.
- **F-LG-7:** T2 startup costs ~205 s at 25 slots × 2 buckets (the per-slot × bucket eager parity gate
  dominates); K3's per-run cache does not cover the parity gate. A launcher restart pays it again.
- **F-LG-9 (`SnapshotPool`, a hazard beyond this lane):** `max_snapshots` is applied only when a snapshot is
  ADDED (`_evict`), never at a scan. A pool dir holding more files than the window serves ALL of them. That
  is how the first fewer-snapshots read went void. In training the pool owns its dir and evicts on add, so
  it holds; any tool that points a `SnapshotPool` at a larger dir gets the whole dir.
- **F-LG-10 (measurement):** Rust arms of IDENTICAL configuration differ by up to ~3 % in decisions/s with
  CIs that exclude 1 (the null read). The per-arm CI does not carry the arm's own routing state, so no
  Rust-arm difference under ~3 % is a finding.
- **F-LG-11 (fixed):** the harness stamped an overlapped arm's T2 buckets from the half's DECLARATION
  (8, 24). The shared service serves (8, 48). The stamp now reads the service, and the committed JSON is
  corrected with a warning.
- **F-LG-12 (the coordinator's pad + row-weight mask, raised not built):** it conflicts with what is
  built, because the collector never emits a ragged micro-batch (the target is on the lcm(micro, N)
  quantum, and the straddling game is split, not trimmed). Only the last ACCUMULATION group can be short,
  and the learner already normalises it by its real rows. That invariant is pinned by `trigger_test`
  (two tests), `_ensure_buffer`'s refusal and `instrumented_ppo_test[4-3-12]`. K10(c)'s dose count is
  fixed. Still open, on today's PYTHON path only and not this lane's: a rollout that does not divide by
  `--batch-size` leaves a short MICRO-batch (the eager path's documented "bounded mis-weighting";
  `--compile-trainer` refuses it). Also open: every aux / belief term that normalises by a per-micro
  subset count is a mean of means under accumulation, on both paths. That was not audited here.
- **F-LG-13 (noise, known):** tearing down the Python arm prints
  `OSError: Can not reset player's battles while they are still running` from each
  `create_training_env_random` worker's close (a window ends mid-battle). It does not affect the numbers.
- **F-LG-14 (unverified):** one run of `rust_core_m5/throughput_test.py` failed ONE test while the 95 / 5
  A/B was compiling beside it (169 s wall), and it passed on the two reruns. Which test failed was not
  captured, so treat it as possibly load-sensitive.

## Resume point (2026-09-30)

Units 1–4 are built; unit 4 ships with this commit. What remains before a CUTOVER, none of it this
lane's to decide alone:

1. **Lane H** — eval on the core. Eval still runs Python workers.
2. **The launcher's restart / pin path on a Rust-core run** (F-LG-6). A resume must pin a commit that has
   `--env-core`; the crash loop around it is untested.
3. **The adaptive-batch controller → `SampleTrigger.set_target`.** The hook and its band exist; nothing
   calls it yet.
4. **The SIZING study** (order constraint 5): N, buckets (a 16 bucket), overlap at every N, the
   grouped-forward rule, the rotating-subset question.
5. **The refused-flag list** (`env_core_rust_*`): each is a later build or a deliberate drop.
6. **T2's startup** (~205–430 s with fresh caches, F-LG-7) under K3's hermetic cache.
7. K9(b) runs under `rust` only (off on the Python path, by decision).
