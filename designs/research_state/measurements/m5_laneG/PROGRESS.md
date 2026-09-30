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
| 1 | THE COLLECTOR + BUFFER: the row arena, complete-game GAE (sb3 bit for bit), the complete-game and window fills, the sample-count trigger + adaptive hook, team / seed staging, the host loop (one T2 flush for trainee + opponents), the keyed draw for the trainee, version pinning (declared, off), `RustVecEnv` + its surface table | BUILT (this commit) |
| 2 | THE TRAINER WIRING behind `--env-core rust` (python stays the default): parser family, combination refusals, the startup hook before the trainer compile, `collect_rollouts` routing + the T2 load after each update, K9(b) + the staleness probe, `metadata.json`'s `env_core` | NEXT |
| 3 | THE PARITY GATES: slice N at the ROLLOUT level (window mode vs today's Python path) + the learner-level check | — |
| 4 | THE SAMPLING CHANGE for policy opponents (F-LE-8) + buckets (F-LE-9), measured | — |

## Measurements so far (descriptors)

- CPU, T2 eager, N = 48, T = 8, proc front end (release), fresh perturbed policy, p2 random external
  (2026-09-29, `/tmp` script): **1,319 trainee decisions/s**; per host step the T2 flush 31 ms (eager CPU,
  the whole cost), the core 2.2 ms, the collector's own work (submit + draw + write + post) 0.8 ms; the
  fill of 6,144 rows 16 ms; every lifecycle counter 0. Not a production-shape read (no GPU, no policy
  opponents) — unit 4's benchmark is.

## Findings

- **F-LG-1 (design, declared):** the complete-game fill SPLITS at most one game per update (the one
  straddling the D-th row); its tail is trained on at the next update, one version older. Rows are never
  dropped; the split share is logged (`staleness/games_split`).
- **F-LG-2 (declared change of stream):** trainee sampling is the keyed draw, battle seeds are keyed hashes,
  teambuilder copies are seeded per env — none reproduces a Python-path stream (today's are unseeded).
- **F-LG-3:** a CUT game (quarantine / respawn) releases its rows uncounted in any buffer (no outcome, no
  next state); counted in `staleness/rows_cut_total`. None observed so far.
