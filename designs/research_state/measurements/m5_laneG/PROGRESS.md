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
| 3 | THE PARITY GATES: slice N at the ROLLOUT level (window mode vs today's Python path, `rust_rollout/parity.py`) + the learner-level check; COMMIT routine, MILESTONE `slow` | BUILT (this commit) |
| 4 | THE SAMPLING CHANGE for policy opponents (F-LE-8) + buckets (F-LE-9), measured | — |

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
| MILESTONE pool — 8 × 128 × 3 (`parity_milestone_pool.json`) | 3,072 | 48 | 2,126 | 0 | 3.6e-7 / 7.2e-7 / 3.5e-7 / 1.2e-7 | 6.0e-8 (3.0e-4) | 249 / 249 (worst 2 % of allowance) |
| MILESTONE ladder — 8 × 128 × 3 (`parity_milestone_ladder.json`) | 3,072 | 45 | 1,933 | 0 | 3.6e-7 / 6.0e-7 / 3.6e-7 / 3.6e-7 | 6.0e-8 (3.0e-4) | 250 / 250 (worst 1.7 %) |

EXACT on every observation key (the row, the mask, all 18 core labels, `opp_class`, `win_target`,
`win_mask`), actions, masks, rewards, episode starts; every recorded trainee action reproduced by the
replay's own keyed draw (1 near-boundary row on the Python side in each milestone, no disagreement).
Teeth: one Python label cell moved fails the slice (per key); the comparator refuses one bit and a
float past its bar. **The learner check is ONE optimizer step** (one epoch, the buffer in 4 accumulated
micro-batches): with 8 optimizer steps it read 3.4e-4 (0.15 of the update's movement) at the milestone —
iterated PPO steps amplify a ≤ 7e-7 input difference; the same learner on the same buffer twice is
bit-identical (F-LG-8).

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
