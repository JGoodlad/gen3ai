# Untaught meter

**Teams** /home/goodlad/dev/gen3ai-wt/split_read/designs/research_state/measurements/split_lossoff_read_2026-09-20/scripts/harness/taught_slice_teams.json (2 clusters) · **opponent** `/home/goodlad/dev/gen3ai/models/ai_v9_29_rev1_0823/snapshots/snapshot_000024000000.zip` · **800 games/team** · seed 0 · concurrency 1

## Levels — cluster mean over teams (equal weight)

| ref | win rate | CI95 | wins/finished | timeouts |
|---|---:|---|---:|---:|
| `split_p3M` | 64.00pp | [62.75, 65.25] | 1024/1600 | 0 |

## Deltas

| ref | Δ vs baseline | verdict |
|---|---|---|
| `split_p3M` | — | — |

> ⚠️ **No continuation control.** A delta against a FROZEN baseline credits an arm with progress the baseline would have made anyway — ledger 2026-09-06 (cell 2) measured a plain continuation moving this meter +3.45pp [+0.46, +6.48] on its own. Pass `--control` with continuation arms at matched depth.

## Per-team win rate

| team | `split_p3M` |
|---|---:|
| `U_f6229d2c` | 62.75 |
| `U_9eb3abdc` | 65.25 |
