# Untaught meter

**Teams** /home/goodlad/dev/gen3ai/designs/research_state/measurements/reuse_batch_2026-09-03/offline_collateral_kl/untaught_teams.json (8 clusters) · **opponent** `/home/goodlad/dev/gen3ai/models/ai_v9_29_rev1_0823/snapshots/snapshot_000024000000.zip` · **200 games/team** · seed 0 · concurrency 1

## Levels — cluster mean over teams (equal weight)

| ref | win rate | CI95 | wins/finished | timeouts |
|---|---:|---|---:|---:|
| `split_p3M` | 57.88pp | [55.56, 60.06] | 926/1600 | 0 |
| `split_p6M` | 58.56pp | [55.19, 61.87] | 937/1600 | 0 |
| `split_p12M` | 58.62pp | [54.81, 61.38] | 938/1600 | 0 |
| `armW` | 46.19pp | [42.81, 49.12] | 739/1600 | 0 |

## Deltas

| ref | Δ vs baseline | verdict |
|---|---|---|
| `split_p3M` | — | — |
| `split_p6M` | — | — |
| `split_p12M` | — | — |
| `armW` | — | — |

> ⚠️ **No continuation control.** A delta against a FROZEN baseline credits an arm with progress the baseline would have made anyway — ledger 2026-09-06 (cell 2) measured a plain continuation moving this meter +3.45pp [+0.46, +6.48] on its own. Pass `--control` with continuation arms at matched depth.

## Per-team win rate

| team | `split_p3M` | `split_p6M` | `split_p12M` | `armW` |
|---|---:|---:|---:|---:|
| `U_61590463` | 55.00 | 54.50 | 61.50 | 48.50 |
| `U_92832108` | 52.50 | 51.00 | 47.00 | 39.00 |
| `U_ce35b736` | 61.50 | 59.00 | 62.00 | 49.50 |
| `U_9909f2e9` | 55.00 | 53.50 | 55.50 | 38.00 |
| `U_9d5f8458` | 61.00 | 63.00 | 60.00 | 47.00 |
| `U_f7ba5702` | 58.50 | 60.50 | 62.00 | 48.00 |
| `U_90b94599` | 62.00 | 61.00 | 59.00 | 48.00 |
| `U_dbf81d8e` | 57.50 | 66.00 | 62.00 | 51.50 |
