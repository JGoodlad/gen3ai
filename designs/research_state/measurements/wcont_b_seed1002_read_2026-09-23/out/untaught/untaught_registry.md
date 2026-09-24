# Untaught meter

**Teams** /home/goodlad/dev/gen3ai-wt/wcontb_read/designs/research_state/measurements/reuse_batch_2026-09-03/offline_collateral_kl/untaught_teams.json (8 clusters) · **opponent** `/home/goodlad/dev/gen3ai/models/ai_v9_29_rev1_0823/snapshots/snapshot_000024000000.zip` · **200 games/team** · seed 0 · concurrency 1

## Levels — cluster mean over teams (equal weight)

| ref | win rate | CI95 | wins/finished | timeouts |
|---|---:|---|---:|---:|
| `wcontb_p3M` | 59.44pp | [57.31, 61.13] | 951/1600 | 0 |
| `wcontb_p6M` | 58.75pp | [56.62, 60.88] | 940/1600 | 0 |
| `wcontb_p12M` | 59.13pp | [56.75, 61.56] | 946/1600 | 0 |
| `armW` | 46.19pp | [42.81, 49.12] | 739/1600 | 0 |
| `wcont_p12M` | 61.69pp | [59.25, 64.25] | 987/1600 | 0 |

## Deltas

| ref | Δ vs baseline | verdict |
|---|---|---|
| `wcontb_p3M` | — | — |
| `wcontb_p6M` | — | — |
| `wcontb_p12M` | — | — |
| `armW` | — | — |
| `wcont_p12M` | — | — |

> ⚠️ **No continuation control.** A delta against a FROZEN baseline credits an arm with progress the baseline would have made anyway — ledger 2026-09-06 (cell 2) measured a plain continuation moving this meter +3.45pp [+0.46, +6.48] on its own. Pass `--control` with continuation arms at matched depth.

## Per-team win rate

| team | `wcontb_p3M` | `wcontb_p6M` | `wcontb_p12M` | `armW` | `wcont_p12M` |
|---|---:|---:|---:|---:|---:|
| `U_61590463` | 59.00 | 61.50 | 54.50 | 48.50 | 59.00 |
| `U_92832108` | 60.50 | 53.50 | 57.50 | 39.00 | 56.50 |
| `U_ce35b736` | 60.00 | 57.00 | 65.50 | 49.50 | 65.00 |
| `U_9909f2e9` | 59.50 | 58.00 | 54.00 | 38.00 | 61.50 |
| `U_9d5f8458` | 61.00 | 58.00 | 60.50 | 47.00 | 66.00 |
| `U_f7ba5702` | 59.00 | 60.00 | 60.50 | 48.00 | 67.00 |
| `U_90b94599` | 63.50 | 64.50 | 60.00 | 48.00 | 60.00 |
| `U_dbf81d8e` | 53.00 | 57.50 | 60.50 | 51.50 | 58.50 |
