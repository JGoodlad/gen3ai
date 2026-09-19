# Untaught meter

**Teams** /home/goodlad/dev/gen3ai-wt/fold1_read_20260919/designs/research_state/measurements/fold1_read_2026-09-19/scripts/harness/taught_slice_teams.json (2 clusters) · **opponent** `/home/goodlad/dev/gen3ai/models/ai_v9_29_rev1_0823/snapshots/snapshot_000024000000.zip` · **800 games/team** · seed 0 · concurrency 1

## Levels — cluster mean over teams (equal weight)

| ref | win rate | CI95 | wins/finished | timeouts |
|---|---:|---|---:|---:|
| `fold_p3M` | 53.06pp | [46.75, 59.38] | 849/1600 | 0 |
| `armWb` | 54.75pp | [54.00, 55.50] | 876/1600 | 0 |

## Deltas

| ref | Δ vs baseline | verdict |
|---|---|---|
| `fold_p3M` | — | — |
| `armWb` | — | — |

> ⚠️ **No continuation control.** A delta against a FROZEN baseline credits an arm with progress the baseline would have made anyway — ledger 2026-09-06 (cell 2) measured a plain continuation moving this meter +3.45pp [+0.46, +6.48] on its own. Pass `--control` with continuation arms at matched depth.

## Per-team win rate

| team | `fold_p3M` | `armWb` |
|---|---:|---:|
| `U_f6229d2c` | 46.75 | 54.00 |
| `U_9eb3abdc` | 59.38 | 55.50 |
