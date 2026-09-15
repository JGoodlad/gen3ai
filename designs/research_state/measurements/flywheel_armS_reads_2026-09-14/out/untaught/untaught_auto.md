# Untaught meter

**Teams** /home/goodlad/dev/gen3ai/designs/research_state/measurements/reuse_batch_2026-09-03/offline_collateral_kl/untaught_teams.json (8 clusters) · **opponent** `/home/goodlad/dev/gen3ai/models/ai_v9_29_rev1_0823/snapshots/snapshot_000024000000.zip` · **200 games/team** · seed 0 · concurrency 1

## Levels — cluster mean over teams (equal weight)

| ref | win rate | CI95 | wins/finished | timeouts |
|---|---:|---|---:|---:|
| `armS` | 54.50pp | [52.25, 56.62] | 872/1600 | 0 |
| `winprob75M` | 58.25pp | [56.44, 60.44] | 932/1600 | 0 |

## Deltas

| ref | Δ vs baseline | verdict |
|---|---|---|
| `armS` | — | — |
| `winprob75M` | — | — |

> ⚠️ **No continuation control.** A delta against a FROZEN baseline credits an arm with progress the baseline would have made anyway — ledger 2026-09-06 (cell 2) measured a plain continuation moving this meter +3.45pp [+0.46, +6.48] on its own. Pass `--control` with continuation arms at matched depth.

## Per-team win rate

| team | `armS` | `winprob75M` |
|---|---:|---:|
| `U_61590463` | 56.50 | 61.00 |
| `U_92832108` | 49.50 | 59.00 |
| `U_ce35b736` | 56.00 | 64.50 |
| `U_9909f2e9` | 54.50 | 55.50 |
| `U_9d5f8458` | 55.50 | 57.50 |
| `U_f7ba5702` | 55.00 | 57.00 |
| `U_90b94599` | 49.50 | 55.00 |
| `U_dbf81d8e` | 59.50 | 56.50 |
