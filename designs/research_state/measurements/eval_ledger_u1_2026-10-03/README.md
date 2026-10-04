# Eval U1: the `main.h2h` storage-only proof (2026-10-03)

**Question.** Eval unit U1 moved `main.h2h` from ledger v1 (`gen3_eval_count_row_v1`, rows in a caller-named
directory) onto ledger v2 (`gen3_eval_count_row_v2`, a ledger root, every batch under a claim for a request). X5's
registration pins the h2h protocol (`design_evaluation.md` §0c rule 6, review M3(c)), so the migration must be
STORAGE-ONLY: the same seeded batch must play the same games before and after.

**Verdict: IDENTICAL** (MEASURED, `compare.json`, exit 0).

## What was run (`run.sh`, CPU only, under `scripts/ops/mem_cap.sh`, peak 1.0 GB)

- **Players:** the M5 sizing finals `sizing_A_n48_e10_s1001` (A, sha `808ad275…`) against `sizing_B_n256_e10_s1001`
  (B, sha `cfdfb9f2…`), the trained checkpoints X5 P0 used, read-only from main's `models/`.
- **Batches:** 2 batches × 50 mirrored pairs (200 games), schedule seed 7, the default (order-independent) schedule
  key. Compute: CPU, T2 `eager`, 16 envs, 4 core threads, 4 torch threads.
- **Trees:** BASE = `origin/main` at `cc2c4830` (ledger v1), in a throwaway worktree with its own Rust build. U1 = the
  U1 branch.

Steps:
1. `h2h_digest_proof.py games` in each tree: `H2HEngine.play_batch` at the CLI's own seeds, dumping every game's
   `(game, result, swapped, teams, end_turn, winner, forfeit, near-tie counts)`. The outcome digest is computed by a
   LOCAL copy of the v2 algorithm, so the base tree is hashed by exactly the same code.
2. `python -m main.h2h play` (the CLI) in each tree with the same arguments: v1 rows from BASE, v2 rows from U1.
3. `h2h_digest_proof.py compare`.

## Result

| check | result |
|---|---|
| (a) the two trees' per-game vectors (`base_games.json.gz`, `u1_games.json.gz`) | **byte-identical** (`cmp`) |
| (b) the outcome digests over the non-near-tie games | equal: `b528c51a…`, `3e59cba0…` |
| (c) the U1 CLI rows' `compute.outcome_digest` vs the BASE games' digest | equal, batch for batch; near-tie indices equal |
| (d) BASE v1 rows vs U1 v2 rows | same W/L/D (47/53/0, 45/55/0), pentanomials ([11,0,31,0,8], [12,0,31,0,7]), team counters, cycle seeds |
| the U1 proof ledger | `python -m main.eval_ledger audit`: OK (2 rows, 2 claims, 0 live) |

- **Near ties.** 15 and 13 of each batch's 100 games hold a decision inside the digest margin (2e-3, the GPU bar).
  The digest lists them by index rather than hashing them. On this one device and configuration the vectors are
  byte-identical INCLUDING those games (check (a)).
- **The pooled read:** A scores 46.0 % [40.0, 52.0] against B over 100 pairs. It is not a result; the plumbing is.

## Scope and limits (FINDINGS)

- **One device, one configuration (CPU eager).** The GPU `graph` path was not run here (U1 is a CPU unit). F-P0-7
  measured CPU-eager vs GPU-graph identity on 200 games before U1; U1 changes nothing on the forward path, so this
  is DEFERRED rather than open, but it was not re-measured.
- **The bot round robin** (`bot_base_ratings_2026-10-03/bot_rr.py`), the second migrated writer, was checked the same
  way at the row level (no per-game dump): 4 non-random edges (`heuristic` vs `heuristic2` / `staller` /
  `staller_v2` / `aggressive`) × 10 mirrored pairs, schedule seed 0, BASE vs U1. All four rows carry the SAME W/L/D,
  pentanomial, team counters, cycle seeds and bot shas (e.g. `heuristic:aggressive` 11/6/3, [1,1,3,2,3]); the U1
  ledger audits clean (4 rows, 4 claims). The `random` edges were skipped as uninformative (0/8 every time).
