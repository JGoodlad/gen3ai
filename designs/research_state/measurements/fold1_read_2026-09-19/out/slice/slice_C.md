# Untaught meter

**Teams** /home/goodlad/dev/gen3ai-wt/fold1_read_20260919/designs/research_state/measurements/fold1_read_2026-09-19/scripts/harness/taught_slice_teams.json (2 clusters) · **opponent** `/home/goodlad/dev/gen3ai/models/ai_v9_29_rev1_0823/snapshots/snapshot_000024000000.zip` · **800 games/team** · seed 0 · concurrency 1

## Levels — cluster mean over teams (equal weight)

| ref | win rate | CI95 | wins/finished | timeouts |
|---|---:|---|---:|---:|
| `t1_big5` | 47.44pp | [37.25, 57.63] | 759/1600 | 0 |
| `t2_ddtar` | 37.69pp | [27.62, 47.75] | 603/1600 | 0 |

## Deltas

| ref | Δ vs baseline | verdict |
|---|---|---|
| `t1_big5` | — | — |
| `t2_ddtar` | — | — |

> ⚠️ **No continuation control.** A delta against a FROZEN baseline credits an arm with progress the baseline would have made anyway — ledger 2026-09-06 (cell 2) measured a plain continuation moving this meter +3.45pp [+0.46, +6.48] on its own. Pass `--control` with continuation arms at matched depth.

## Per-team win rate

| team | `t1_big5` | `t2_ddtar` |
|---|---:|---:|
| `U_f6229d2c` | 57.63 | 27.62 |
| `U_9eb3abdc` | 37.25 | 47.75 |
