# Untaught meter

**Teams** /home/goodlad/dev/gen3ai/designs/research_state/measurements/reuse_batch_2026-09-03/offline_collateral_kl/untaught_teams.json (8 clusters) · **opponent** `/home/goodlad/dev/gen3ai/models/ai_v9_29_rev1_0823/snapshots/snapshot_000024000000.zip` · **200 games/team** · seed 0 · concurrency 1

## Levels — cluster mean over teams (equal weight)

| ref | win rate | CI95 | wins/finished | timeouts |
|---|---:|---|---:|---:|
| `fold_p1M` | 47.62pp | [44.56, 50.50] | 762/1600 | 0 |
| `fold_p3M` | 51.56pp | [46.19, 56.69] | 825/1600 | 0 |
| `fold_p6M` | 51.31pp | [46.00, 56.31] | 821/1600 | 0 |
| `armW` | 46.19pp | [42.81, 49.12] | 739/1600 | 0 |
| `armWb` | 49.88pp | [46.81, 53.00] | 798/1600 | 0 |

## Deltas

| ref | Δ vs baseline | verdict |
|---|---|---|
| `fold_p1M` | — | — |
| `fold_p3M` | — | — |
| `fold_p6M` | — | — |
| `armW` | — | — |
| `armWb` | — | — |

> ⚠️ **No continuation control.** A delta against a FROZEN baseline credits an arm with progress the baseline would have made anyway — ledger 2026-09-06 (cell 2) measured a plain continuation moving this meter +3.45pp [+0.46, +6.48] on its own. Pass `--control` with continuation arms at matched depth.

## Per-team win rate

| team | `fold_p1M` | `fold_p3M` | `fold_p6M` | `armW` | `armWb` |
|---|---:|---:|---:|---:|---:|
| `U_61590463` | 50.00 | 44.50 | 48.50 | 48.50 | 58.00 |
| `U_92832108` | 43.50 | 44.50 | 37.50 | 39.00 | 53.00 |
| `U_ce35b736` | 53.00 | 58.50 | 60.50 | 49.50 | 42.50 |
| `U_9909f2e9` | 40.00 | 39.00 | 45.50 | 38.00 | 48.00 |
| `U_9d5f8458` | 51.50 | 56.00 | 56.00 | 47.00 | 46.00 |
| `U_f7ba5702` | 44.50 | 52.50 | 54.50 | 48.00 | 48.00 |
| `U_90b94599` | 47.00 | 63.00 | 60.50 | 48.00 | 51.50 |
| `U_dbf81d8e` | 51.50 | 54.50 | 47.50 | 51.50 | 52.00 |
