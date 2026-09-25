# Untaught meter

**Teams** /home/goodlad/dev/gen3ai-wt/pin-6eb9c776-popr2/designs/research_state/measurements/reuse_batch_2026-09-03/offline_collateral_kl/untaught_teams.json (8 clusters) · **opponent** `/home/goodlad/dev/gen3ai/models/ai_v9_29_rev1_0823/snapshots/snapshot_000024000000.zip` · **200 games/team** · seed 0 · concurrency 1

## Levels — cluster mean over teams (equal weight)

| ref | win rate | CI95 | wins/finished | timeouts |
|---|---:|---|---:|---:|
| `popr2_loop` | 60.75pp | [58.44, 63.31] | 972/1600 | 0 |
| `popr2_ctrl` | 62.06pp | [59.00, 65.19] | 993/1600 | 0 |
| `plateau_b1` | 60.19pp | [58.50, 61.87] | 963/1600 | 0 |

## Deltas

| ref | Δ vs baseline | verdict |
|---|---|---|
| `popr2_loop` | — | — |
| `popr2_ctrl` | — | — |
| `plateau_b1` | — | — |

> ⚠️ **No continuation control.** A delta against a FROZEN baseline credits an arm with progress the baseline would have made anyway — ledger 2026-09-06 (cell 2) measured a plain continuation moving this meter +3.45pp [+0.46, +6.48] on its own. Pass `--control` with continuation arms at matched depth.

## Per-team win rate

| team | `popr2_loop` | `popr2_ctrl` | `plateau_b1` |
|---|---:|---:|---:|
| `U_61590463` | 59.00 | 58.50 | 58.00 |
| `U_92832108` | 58.00 | 54.50 | 56.00 |
| `U_ce35b736` | 61.50 | 66.50 | 58.50 |
| `U_9909f2e9` | 58.00 | 60.00 | 60.50 |
| `U_9d5f8458` | 67.50 | 69.50 | 64.00 |
| `U_f7ba5702` | 56.00 | 65.00 | 61.00 |
| `U_90b94599` | 63.50 | 62.50 | 63.00 |
| `U_dbf81d8e` | 62.50 | 60.00 | 60.50 |
