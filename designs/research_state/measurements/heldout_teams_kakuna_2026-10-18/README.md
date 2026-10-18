# heldout72 — a held-out gen3ou team set, and X26 / K2 vs Kakuna on it — 2026-10-18

**Why (orchestrator FINDING, 2026-10-18 07:30).** The Kakuna reads called `away` (Metamon's `competitive/gen3ou`)
"Metamon's teams", a set our models never trained on. It is not. Every Kakuna read so far ran on teams our training
pool already holds, so none of them says how X26 plays a team it has not seen. `heldout72` is the set that can.

## The team sets, overlapped (`main.anchors.team_overlap`, `gen3_team_overlap_v1`)

A mon set = normalised species + sorted moves, Hidden Power typed (a typeless one takes its IVs' type); the strict tier
adds item, ability, nature and EVs. "≥ 4 shared" = the read team shares ≥ 4 mon sets with ONE reference team.

| read set (n) | vs | exact team (strict) | same 6 species | mon sets seen (strict) | ≥ 4 shared |
|---|---|---:|---:|---:|---:|
| **away** = `competitive` (20) | our training pool (719) | **19 (19)** | 19 | 119 / 120 (115) | 19 |
| away (20) | `hl_05_26` (22,862) | 0 (0) | 19 | 103 / 120 (44) | 3 |
| **home72** (72) | our training pool | **72 (72)** | 72 | 432 / 432 | 72 |
| home72 (72) | `competitive` | 8 (7) | 12 | 173 / 432 (61) | 8 |
| home72 (72) | `hl_05_26` | 0 (0) | 57 | 380 / 432 (149) | 13 |
| **heldout72** (72) | our training pool | **0 (0)** | 21 | 104 / 432 (27) | **0** |
| heldout72 (72) | `competitive` | 0 (0) | 9 | 39 / 432 (3) | 0 |
| heldout72 (72) | home72 | 0 (0) | 13 | 78 / 432 (16) | 0 |
| pool (719) | `competitive` | 23 (22) | 38 | 1,449 / 4,314 (266) | 28 |

The pool holds 23 entries that are exact copies of 19 `competitive` teams (four duplicates). `home72` is
`data/teams/sample`, which is part of the pool. The orchestrator's "19 of 21" counted `competitive`'s `index.csv` as a
21st team; the set has 20 teams (`main.anchors.peers.TEAM_FILE_GLOBS` excludes the CSV, as every read does).

## The draw rule (written 2026-10-18 07:44 PT, BEFORE any team was drawn)

**Source.** Metamon's `hl_05_26/gen3ou` team set, the archived copy at
`~/gen3ai_archive/metamon_cache_2026-09-24/teams/hl_05_26/gen3ou`: 22,862 team files (`team_NNNNNN.gen3ou_team`;
the directory's `index.csv` is not a team).

**Order.** The file names, sorted, shuffled ONCE by `random.Random(20261018)`. Candidates are taken in that order,
and the first 72 that pass every check below are the set. A candidate that fails is dropped and the next one is
taken ("redraw by rule"); nothing is hand-picked.

**Rejection rules** (matching = `main.anchors.team_overlap`, `gen3_team_overlap_v1`: a mon set = normalised species
+ sorted moves, Hidden Power typed):

1. **≥ 1 EXACT team match** (all six mon sets) with any team of our 719-team training pool (`data/teams`), or of
   Metamon's `competitive/gen3ou` (`away`), or of a team already drawn.
2. **≥ 4 of 6 mon sets in common with any ONE team** of the training pool or of `competitive`.
3. Not exactly six mons after sanitising.
4. **Rejected by the pinned Showdown team validator** (`utils.bridge.team_validator.validate_teams_locally`,
   `gen3ou`, Showdown pin `e0551883f`), on the sanitised text.

**Sanitising** (the `n0_endofrun_2026-09-27/scripts/make_home72.py` convention, so all three clients read the same
complete paste): nickname-free; every Hidden Power spelled `Hidden Power [Type]` with the `IVs:` line of
`utils.gen3_utils.GEN3_HP_IVS[Type]` (a typeless `Hidden Power` takes the type its IVs give, under the gen-3
formula); one blank line between mons and one newline at the end.

## The draw (run 2026-10-18 07:45 PT, `scripts/make_heldout72.py`)

- **80 candidates examined, 72 accepted, 0 rejected by any rule.** The last 8 of the second validation batch were not
  needed (`draw_log.jsonl`). Every one of the first 80 passed: overlap with our pool is rare in `hl_05_26` (above).
- **The validator accepted all 72** (`validate_teams_locally`, gen3ou, pin `e0551883f`); its teeth were checked on the
  same call (a Mewtwo swap is rejected: "tagged Uber"). `Gen3Teambuilder` keeps all 72.
- **Files:** `teams_heldout72/team_NNNNNN.txt` (the source file's stem). `heldout72_manifest.json` records, per team, the
  source file, its sha256 and the written file's sha256, plus the overlap check above.
- **Installed for Metamon** as `~/dev/metamon/cache/teams/gen3ai_heldout72/gen3ou/team_NNNNNN.gen3ou_team`, byte copies.
  The read's own copy is `~/dev/gen3ai-reads/heldout72_kakuna_2026-10-18/teams_heldout72/`. The queue (`scripts/hq.py`)
  sha256-checks every file on BOTH sides, and the config, against the manifest before every unit.
- **Config:** `anchors_heldout72_kakuna.json` (sha256 `b178f3bc…616d`): the home72 Kakuna config with `home` = heldout72 on
  every side; run with `--teamset home`.

🚨 **HIDDEN POWER IS DARK ON 164 OF 432 MONS.** `hl_05_26` spells Hidden Power without a type in 21,905 of its 22,862
files, with all IVs 31. Under the gen-3 formula that is Hidden Power **Dark**, and that is what Metamon itself plays from
these files. heldout72 spells it `Hidden Power [Dark]` with explicit IVs, so all three clients play the same move. The
real players' types were not recorded (a Salamence with HP Dark is unusual). The set is faithful to Metamon's file, not
to the original players' teams. 4 mons carry a typed Hidden Power (Bug 1, Fire 1, Flying 2).

## The Kakuna caveat — what Kakuna trained on (Metamon checkout `0a00a759`)

- **The `hl_05_26` SET did not exist when Kakuna was trained.** It entered the repo in commit `a790b624` (2026-05-21,
  "may 2026 metamon-dev sync"); Kakuna is dated "Dec 2025" (`README.md:288`), and its class first appears in commit
  `534c7c42` (2025-12-17). That conclusion is from dates; no file says it directly.
- **Kakuna may still have played teams like these.** Kakuna "was trained on `metamon-parsed-replays`, `pac-base`, and
  `pac-exploratory`" (`README.md:825`). `hl_05_26` is built from ladder and tournament REPLAYS (`README.md:867`, "Subset of `gl_05_26` that
  restricts to higher-rated games and tournament-server replays"; `metamon/env/wrappers.py:45`, `gl_05_26` = "last-year
  revealed teams, NaiveUsagePredictor fill"), with the
  unrevealed fields filled by sampling usage stats (`metamon/backend/team_prediction/predictor.py:122-128`). Replays from
  before Kakuna's data cut may be in its parsed-replays training data. **UNVERIFIED** (the parsed-replays version and
  date range Kakuna used are not named in the checkout).
- **Kakuna's self-play team sets are not named anywhere.** "Trajectories were generated by the ladder self-play launcher
  with various team sets and model pools" (`README.md:817`); `README.md:288` says "Trained on diverse teams". The
  launcher's example config defaults to `competitive` (`metamon/rl/evaluate/ladder_self_play/example_config.yaml:3`),
  which is an example, not a record. `modern_replays` / `modern_replays_v2` (replay-predicted sets, the latter dated
  September 2025, `metamon/env/wrappers.py:41-42`) existed then; whether Kakuna's self-play used them: **UNVERIFIED**.
- **`competitive` is the set Metamon says it overfit to**: "because they were once the main evaluation and deployment
  set, Metamon has overfit to them" (`README.md:865`). Kakuna's published GXE is measured "with 'competitive' TeamSet"
  (`metamon/rl/pretrained.py:885`).
- **So:** heldout72 is held out from US by construction (0 exact, 0 sharing ≥ 4 mon sets). From Kakuna it is held out as
  a SET, and its teams are of the kind (high-ladder, replay-derived, usage-filled) Kakuna's data is built from.

## The read — X26 and K2 vs Kakuna on heldout72

RUNNING at the set's commit (the queue `scripts/hq.py`, 40 units: X26 and K2 on paired seeds 0–19, round-robin). The
results land in this section in the read's own commit.
