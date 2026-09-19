# Untaught meter

**Teams** /home/goodlad/dev/gen3ai-wt/fold1cont_read/designs/research_state/measurements/fold1_cont_read_2026-09-19/scripts/harness/taught_slice_teams.json (2 clusters) · **opponent** `/home/goodlad/dev/gen3ai/models/ai_v9_29_rev1_0823/snapshots/snapshot_000024000000.zip` · **800 games/team** · seed 0 · concurrency 1

## Levels — cluster mean over teams (equal weight)

| ref | win rate | CI95 | wins/finished | timeouts |
|---|---:|---|---:|---:|
| `cont_p12M` | 53.19pp | [44.00, 62.38] | 851/1600 | 0 |
| `cont_p9M` | 50.31pp | [42.25, 58.38] | 805/1600 | 0 |

## Deltas

| ref | Δ vs baseline | verdict |
|---|---|---|
| `cont_p12M` | — | — |
| `cont_p9M` | — | — |

> ⚠️ **No continuation control.** A delta against a FROZEN baseline credits an arm with progress the baseline would have made anyway — ledger 2026-09-06 (cell 2) measured a plain continuation moving this meter +3.45pp [+0.46, +6.48] on its own. Pass `--control` with continuation arms at matched depth.

## Per-team win rate

| team | `cont_p12M` | `cont_p9M` |
|---|---:|---:|
| `U_f6229d2c` | 44.00 | 42.25 |
| `U_9eb3abdc` | 62.38 | 58.38 |
