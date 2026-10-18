# Ladder resemblance and `ladder_weights_v1` (2026-10-18)

**What this is.** A zero-game read for
[`../../../endstate/design_diversity_teams_opponents.md`](../../../endstate/design_diversity_teams_opponents.md)
§5.0 and §5.3. It asks how much the ladder's teams resemble the SAMPLE teams, and turns the answer into the frozen
strata weights behind THE headline L. It plays no game.

## Inputs (read-only)

- **The ladder proxy:** Metamon's `hl_05_26/gen3ou`, the archived copy
  (`~/gen3ai_archive/metamon_cache_2026-09-24/teams/hl_05_26/gen3ou`). It holds 22,862 replay-derived teams from the
  1400+ ladder and smogtours, with unrevealed parts filled by usage sampling.
- **The sample reference:** home72 (`../n0_endofrun_2026-09-27/teams_home72`, our `data/teams/sample`) ∪ `away`
  (Metamon's `competitive/gen3ou`). That is 92 files and 79 distinct six-species cores.
- **heldout72** (`../heldout_teams_kakuna_2026-10-18/teams_heldout72`) and the parallel agent's `heldout72_hpfix`
  copy (`~/dev/gen3ai-reads/heldout72hp_kakuna_2026-10-18/teams_heldout72_hpfix`), for their stratum membership.

## Result

| best overlap with ONE sample team | 0 | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|---|
| **species** | 0.03 % | 0.21 % | 2.06 % | 12.64 % | 34.30 % | 33.17 % | **17.61 %** |
| **mon sets** (species + moves) | 21.53 % | 62.40 % | 14.18 % | 1.67 % | 0.20 % | 0.01 % | 0 % |

**The strata and `ladder_weights_v1`** (per file, so popular teams weigh by how often they were played):

| stratum | definition | weight |
|---|---|---|
| CORE | all 6 species of one sample team | **0.1761** |
| NEAR | 4–5 | **0.6747** |
| FAR | ≤ 3 | **0.1493** |

**heldout72 / heldout72_hpfix strata:** 16 CORE / 44 NEAR / 12 FAR teams, i.e. 22 / 61 / 17 %. That is ladder-like.
The membership lists are in the JSON.

**Descriptive, post hoc.** Each model's banked heldout72 score, split by the stratum of the team OUR side piloted:

| model | CORE | NEAR | FAR |
|---|---|---|---|
| X26 | 80 games, 0.425 | 244, 0.430 | 76, **0.250** |
| K2 | 80, 0.350 | 244, 0.381 | 76, 0.289 |

Kakuna's team is the same random heldout72 mix in every column, so the columns differ in our piloted team only. They
are confounded with team strength. The intervals are wide: Wilson, games not independent, 12–44 teams per column.

## Cross-check

The orchestrator's quick script `~/.cache/gen3ai/hl_vs_samples.py` was re-run 2026-10-18 on the same inputs, with a
lighter parser and 93 reference files. It read species 6 = 17.6 %, 5 = 33.2 %, 4 = 34.3 %, 3 = 12.6 %, and mon
sets ≥ 4 = 0.2 %. That agrees with this module to 0.1 pp.

## Freezing and versioning

`ladder_weights_v1.json` is a DRAFT (`status`) until X37a's registration commit freezes it, BEFORE any
deployment-cell game. The JSON records:

- every input's digest (each file's name and sha256);
- the matcher (`gen3_team_overlap_v1`);
- the strata boundaries.

A change of any input is a NEW version (`ladder_weights_v2`, …). Every L is quoted with its version, and versions
are never pooled.

## Known weaknesses

These are in the design doc, §5.3:

- the proxy is not today's ladder at our rating;
- unrevealed team MEMBERS are usage-filled, so the strata are partly fill-driven;
- the mon-set overlap is understated by the fills;
- the reference set defines the strata.

## Files and provenance

- `scripts/ladder_weights.py` writes `ladder_weights_v1.json`. The run was CPU-only (`CUDA_VISIBLE_DEVICES=""`)
  under `nice -n 19 taskset -c 5-7,13-15`, with X26 live, in about 3 s.
- Nothing under `models/` or `data/` was written. No game was played.
