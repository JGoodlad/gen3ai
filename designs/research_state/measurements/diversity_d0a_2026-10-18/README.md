# D0a — the three Kakuna team sets, described with ZERO games (2026-10-18)

**What this is.** A post-hoc, descriptive read for
[`../../../endstate/design_diversity_teams_opponents.md`](../../../endstate/design_diversity_teams_opponents.md) §2.4. It
plays no game. It asks one question: what is heldout72, compared with home72, `away` and our 719-team pool?
It looks at entity coverage, mon-set overlap, Hidden Power, archetype mix and game length. It also splits the banked
2026-10-17/18 Kakuna games by OUR piloted team's covariates.

**Nothing here is a verdict.** The splits are many-way and chosen after seeing the data. They are confounded with team
strength: an incoherent set is weaker for whoever pilots it. The Wilson intervals treat games as independent, which is
optimistic because teams repeat.

## The sets

| | home72 | away | heldout72 | pool (719) |
|---|---|---|---|---|
| archetype mix (`team_archetypes.classify_team`): hyper offense / offense / balance / semi-stall / stall | 11 / 19 / 16 / 12 / 14 | 1 / 5 / 5 / 5 / 4 | **7 / 18 / 25 / 18 / 4** | 139 / 155 / 206 / 129 / 90 |
| teams with any species / move / item / ability that no pool team carries | 0 | 0 | **9 / 72** | — |
| mons with a species no pool team carries | 0 | 0 | 7 / 432 | — |
| move slots holding a move no pool mon carries | 0 | 0 | 7 | — |
| species-move pairs absent from the pool | 0 | 0 | 203 (**160 of them Hidden Power**) | — |
| mons with an item / ability no pool team carries | 0 / 0 | 0 / 0 | 0 / 1 | — |
| HP Dark mons | 0 | 0 | 164 / 432 | — |
| mon sets seen in the pool (`gen3_team_overlap_v1`) | 432 / 432 | 119 / 120 | 104 / 432 | — |
| teams whose six SPECIES equal some pool team's | 72 | 19 | 21 | — |
| median game length, X26 / K2 (turns) | 30 / 33 | 37 / 36 | 40 / 41 | — |

## heldout72 games split by OUR team's covariates

Score counts ties as 0.5, so X26's home72 reads 0.5775 here and 0.575 (wins only) in the source read.

| split | X26 (games, score) | K2 (games, score) |
|---|---|---|
| HP Dark mons on our team: 0–1 / 2 / ≥ 3 | 94, 0.447 / 144, 0.410 / 162, 0.352 | 94, 0.426 / 144, 0.389 / 162, 0.290 |
| our team has ≥ 1 species the pool lacks: no / yes | 368, 0.408 / 32, 0.250 | 368, 0.375 / 32, 0.156 |
| non-HP species-move pairs absent from the pool: 0–1 / 2–3 / ≥ 4 | 340, 0.400 / 34, 0.441 / 26, 0.269 | 340, 0.353 / 34, 0.618 / 26, 0.077 |
| most mon sets shared with ONE pool team: 0 / 1 / 2 | 74, 0.338 / 224, 0.411 / 102, 0.402 | 74, 0.351 / 224, 0.335 / 102, 0.412 |
| same six species as a pool team: yes / no | 108, 0.426 / 292, 0.384 | 108, 0.352 / 292, 0.360 |

Every pair of intervals overlaps. `d0a.json` has the Wilson intervals and the archetype splits.

## How to read it

- **heldout72 is not stall-heavy.** Only 4 of its 72 teams are stall, against 90 of the pool's 719. Its games run
  longer anyway.
- **Entity coverage is small.** Only 9 of 72 teams carry anything the pool never had. Most of the "unseen" content is
  the Hidden Power artifact.
- **Scores fall with HP Dark on our team, in BOTH models.** That is consistent with the artifact making weaker sets.
  This read cannot separate it from familiarity. The test is the parallel `heldout72_hpfix` re-read, which is PENDING
  and is not duplicated here.
- **No familiarity gradient is visible at the mon-set level INSIDE heldout72.**
- **The peer's per-game team is NOT used.** The banked peer CSV's `Team File` column does not align one-to-one with
  the peer's `team_draws`. In unit `X26_greedy_hl72_00`, draws[1..9] are logged, the last is repeated, and draws[0]
  never appears. Facing-side splits would be GIGO (design doc FINDING F-DV-6).

## Files and provenance

- `scripts/d0a_team_covariates.py` writes `d0a.json`. The run was CPU-only (`CUDA_VISIBLE_DEVICES=""`) under
  `nice -n 19 taskset -c 5-7,13-15`, with X26 live.
- **Inputs, all read-only:**
  - this checkout's `data/teams` (719 teams, through `TeamLoader` with an absolute base);
  - `../n0_endofrun_2026-09-27/teams_home72`;
  - `~/dev/metamon/cache/teams/competitive/gen3ou`;
  - `../heldout_teams_kakuna_2026-10-18/teams_heldout72`;
  - the `rows.jsonl` of `../k2_x26_kakuna_both_teamsets_2026-10-17/` and `../heldout_teams_kakuna_2026-10-18/`.
- Nothing under `models/` or `data/` was written. No game was played.
