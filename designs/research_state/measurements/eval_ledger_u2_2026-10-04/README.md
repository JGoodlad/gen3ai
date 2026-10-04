# Eval U2: the in-loop eval cycle's storage-only proof (2026-10-04)

**Question.** Eval unit U2 moved the trainer's in-loop eval cycle (`eval_launch.launch_rust_eval_cycle`, both eval
callbacks) and the SPRT promotion onto the eval COUNT ledger (`agents/training/cycle_ledger.py`). The X5 constraint
(design_evaluation.md §0c rule 6, review M3) requires it to be STORAGE ONLY: the same in-loop cycle on the same seed
must play the same games before and after, and the ledger rows must BE those games.

**Verdict: IDENTICAL** (MEASURED, `compare.txt`, exit 0): 48 games per side compared, 12 ledger rows checked.

## What was run (`run.sh`, CPU only, under `scripts/ops/mem_cap.sh`, peak ~1.0 GB)

- **Trees:** BEFORE = a detached worktree at `e5f393cb` (eval U1 on main, no in-loop ledger), its own Rust self-check
  build. AFTER = the U2 tree (its working tree on base `e5f393cb`, before the U2 commit — so `after.json`'s `commit`
  field reads the base; the U2 files' sha256 at the time of the run: `cycle_ledger.py f0ed81cc…`, `eval_launch.py
  110a9865…`, `sprt_promotion.py 5702d992…`, `rust_eval/executor.py 208c6d7c…`, `rust_eval/launch.py b3ad82bf…`).
- **The cycle** (`digest_proof.py`, self-contained so it runs unchanged on both trees): seeded PERTURBED fresh
  production-arch policies (`rust_eval.parity.build_models`: a trainee + 2 pool sentinels; `build_fixed_models`: 1
  fixed opponent on a pinned team), CPU eager, 4 envs; opponents = 3 roster bots + the 2 sentinels + the fixed one, 4
  games each; one UNMIRRORED cycle with greedy sentinels and one MIRRORED cycle with sentinels sampled at T = 1.0;
  cycle seed = the in-loop rule on run seed 7, step 4,000.
- Per cycle, per tree: (1) the in-loop path exactly as the callbacks call it (AFTER: with a `CycleLedger` on a scratch
  root and the trainee zip as the snapshot) → the merged shard results the collect reads; AFTER also reads back every
  ledger row; (2) the same plan through `run_rust_eval_cycle` with the executor's FULL game log → the per-game outcome
  vector `(game, W/L/D, end turn)` per opponent, its digest at the 2e-3 near-tie margin and over every game.

## Result

| check | result |
|---|---|
| (1) the in-loop merged shard results (W, finished, draws, pentanomial per opponent), BEFORE vs AFTER | identical |
| (2) the full per-game outcome vectors and digests, BEFORE vs AFTER (48 games) | identical |
| (3) every AFTER ledger row's `outcome_digest`, near-tie list, `outcome_digest_all` and W / L / D vs its opponent's full log, and vs the merged shard results | equal (12 rows) |

The two cycles' trainee decision counts match too (1,982 unmirrored, 2,063 mirrored, in both `run.log` halves).

**Caveat (F-ED-23).** On these near-uniform perturbed policies 40 of 48 games held a decision inside the 2e-3 bar,
so the margin-filtered digest covers few games; check (2) compares EVERY game's outcome directly, and the rows'
`outcome_digest_all` covers every game. The in-pytest half of this check, on a smaller plan, is
`src/agents/training/cycle_ledger_integration_test.py` (`sim` + `slow` tier).

## Also here

- `foldbench.py` — F-ED-22: the ledger writer's per-claim cost as the archive's requests stream grows (11.9 ms at
  1,700 events / 5 files; 37.1 ms at 5,100 events / 10 files; ≈ 7 µs per event, linear).
