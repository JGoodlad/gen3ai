# Untaught meter

**Teams** /home/goodlad/dev/gen3ai/designs/research_state/measurements/reuse_batch_2026-09-03/offline_collateral_kl/untaught_teams.json (8 clusters) · **opponent** `/home/goodlad/dev/gen3ai/models/ai_v9_29_rev1_0823/snapshots/snapshot_000024000000.zip` · **200 games/team** · seed 0 · concurrency 1

## Levels — cluster mean over teams (equal weight)

| ref | win rate | CI95 | wins/finished | timeouts |
|---|---:|---|---:|---:|
| `wcont_p3M` | 58.50pp | [55.62, 61.06] | 936/1600 | 0 |
| `wcont_p6M` | 60.44pp | [57.06, 63.50] | 967/1600 | 0 |
| `wcont_p12M` | 61.69pp | [59.25, 64.25] | 987/1600 | 0 |
| `armW` | 46.19pp | [42.81, 49.12] | 739/1600 | 0 |

## Deltas

| ref | Δ vs baseline | verdict |
|---|---|---|
| `wcont_p3M` | — | — |
| `wcont_p6M` | — | — |
| `wcont_p12M` | — | — |
| `armW` | — | — |

> ⚠️ **No continuation control.** A delta against a FROZEN baseline credits an arm with progress the baseline would have made anyway — ledger 2026-09-06 (cell 2) measured a plain continuation moving this meter +3.45pp [+0.46, +6.48] on its own. Pass `--control` with continuation arms at matched depth.

## Per-team win rate

| team | `wcont_p3M` | `wcont_p6M` | `wcont_p12M` | `armW` |
|---|---:|---:|---:|---:|
| `U_61590463` | 57.50 | 59.00 | 59.00 | 48.50 |
| `U_92832108` | 50.00 | 51.50 | 56.50 | 39.00 |
| `U_ce35b736` | 58.00 | 65.50 | 65.00 | 49.50 |
| `U_9909f2e9` | 59.50 | 56.00 | 61.50 | 38.00 |
| `U_9d5f8458` | 60.00 | 66.00 | 66.00 | 47.00 |
| `U_f7ba5702` | 64.50 | 64.00 | 67.00 | 48.00 |
| `U_90b94599` | 57.00 | 62.50 | 60.00 | 48.00 |
| `U_dbf81d8e` | 61.50 | 59.00 | 58.50 | 51.50 |
