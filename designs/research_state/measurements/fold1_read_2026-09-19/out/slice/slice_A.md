# Untaught meter

**Teams** /home/goodlad/dev/gen3ai-wt/fold1_read_20260919/designs/research_state/measurements/fold1_read_2026-09-19/scripts/harness/taught_slice_teams.json (2 clusters) · **opponent** `/home/goodlad/dev/gen3ai/models/ai_v9_29_rev1_0823/snapshots/snapshot_000024000000.zip` · **800 games/team** · seed 0 · concurrency 1

## Levels — cluster mean over teams (equal weight)

| ref | win rate | CI95 | wins/finished | timeouts |
|---|---:|---|---:|---:|
| `fold_p6M` | 48.44pp | [43.12, 53.75] | 775/1600 | 0 |
| `armW` | 48.12pp | [47.00, 49.25] | 770/1600 | 0 |

## Deltas

| ref | Δ vs baseline | verdict |
|---|---|---|
| `fold_p6M` | — | — |
| `armW` | — | — |

> ⚠️ **No continuation control.** A delta against a FROZEN baseline credits an arm with progress the baseline would have made anyway — ledger 2026-09-06 (cell 2) measured a plain continuation moving this meter +3.45pp [+0.46, +6.48] on its own. Pass `--control` with continuation arms at matched depth.

## Per-team win rate

| team | `fold_p6M` | `armW` |
|---|---:|---:|
| `U_f6229d2c` | 43.12 | 49.25 |
| `U_9eb3abdc` | 53.75 | 47.00 |
