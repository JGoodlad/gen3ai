# Team promotion — 40 teams, seed `20260830`

> ⚠️ **DEMO DRAW — not the launch draw.** Produced by `python -m main.promote_teams --draw-only
> --seed 20260830` to exercise and document the tool. **Nothing was promoted**; no
> file under `data/teams/` was touched. The real draw happens at launch with a fresh seed and the
> training session watching, and writes its manifest into `data/teams/sample/`. Reproduce it
> exactly with `--dry-run --seed 20260830`; `promote_teams_test.py` re-derives it on every run.
>
> **RE-DRAWN 2026-08-31** at the same seed, after `promotion_exclusions.json`'s `rev4_pending`
> block was repaired against recorded run provenance. The eligible COUNT is unchanged (693), but 4
> teams left the exclusion set and 4 entered it — and because the draw is a seeded shuffle of the
> *sorted* eligible list, swapping 4 of its 693 members moves 21 of the 40 drawn positions. A
> seeded draw is reproducible against a FIXED eligible set; it is not stable across a change to
> that set, and this is what that instability looks like.

Generated 2026-08-31T04:34:04+00:00 at `153de08e77` by `python -m main.promote_teams`.

**Selection is UNIFORM RANDOM over the eligible pool** (owner ruling, ledger 2026-08-30) — not ranked, not curated. That is what makes the fleet's result an unbiased estimate of pool-wide transferability. Archetype and folder composition below are **REPORTED, never corrected**; correcting them would put the selection confound back.

| | |
|---|---|
| pool | 719 |
| exclusions | 26 |
| eligible | 693 |
| drawn | 40 |
| candidates considered | 40 |
| replaced (failed validation) | 0 |
| validated | True (`gen3ou`) |

Keys are `sha1(team_text.strip())[:10]` — the strip-normalized convention shared by `team_archetypes.team_sha`, `MatchupSpec` pins and `TeamWinRateCallback`. (An unstripped `sha1(text)` is a recorded derived-key defect and is NOT used here.)

## Exclusions applied (before the draw)

| category | teams |
|---|---|
| `taught_F5` | 9 |
| `taught_F6` | 12 |
| `rev4_pending` | 24 |
| `held_out_instruments` | 2 |
| **union (deduped)** | **26** |

Source: `/home/goodlad/dev/gen3ai/.claude/worktrees/simplify-p0/designs/ai_v12/promotion_exclusions.json`.

## The draw

| # | sha | source | action | archetype | folder |
|---|---|---|---|---|---|
| 1 | `16d681e970` | `data/teams/others/giraffe/73bea8a147236023.txt` | copy | stall | giraffe |
| 2 | `a3bc0b0470` | `data/teams/others/mcmegan/3e98a75821c859bd.txt` | copy | semi_stall | mcmegan |
| 3 | `7ad65bac76` | `data/teams/others/giraffe/6b0e85bcbb852991.txt` | copy | balance | giraffe |
| 4 | `6e12f4e836` | `data/teams/others/giraffe/39f39aa240c34f11.txt` | copy | balance | giraffe |
| 5 | `ff78d9fb04` | `data/teams/others/giraffe/62ab6c32ece9cd5e.txt` | copy | hyper_offense | giraffe |
| 6 | `e4b2368267` | `data/teams/others/yak_attack/e8ae949961ef5084.txt` | copy | balance | yak_attack |
| 7 | `3cf4751c64` | `data/teams/others/giraffe/04a94b0076814cce.txt` | copy | hyper_offense | giraffe |
| 8 | `6ef6e77850` | `data/teams/others/giraffe/637604269761c864.txt` | copy | hyper_offense | giraffe |
| 9 | `aa0cffc5ed` | `data/teams/others/giraffe/d984ea2cd55b5fe8.txt` | copy | semi_stall | giraffe |
| 10 | `a577a735b7` | `data/teams/others/yak_attack/82be38ccdf48af18.txt` | copy | balance | yak_attack |
| 11 | `1c1420342b` | `data/teams/others/giraffe/0ceb14cbaf3692df.txt` | copy | offense | giraffe |
| 12 | `b9380b1d7f` | `data/teams/others/giraffe/1c9f8bd535d8dccd.txt` | copy | balance | giraffe |
| 13 | `683908e124` | `data/teams/others/giraffe/45eec47ae82b85f9.txt` | copy | balance | giraffe |
| 14 | `8cd386c07d` | `data/teams/others/johnnyg2/39ca8ea07e3d8c99.txt` | copy | balance | johnnyg2 |
| 15 | `abae242d0e` | `data/teams/others/giraffe/032aaccaad105519.txt` | copy | hyper_offense | giraffe |
| 16 | `3ceea3aaaf` | `data/teams/others/giraffe/e81c37f726e5f3db.txt` | copy | offense | giraffe |
| 17 | `13388c8b10` | `data/teams/others/giraffe/fff27ca1c7fb8550.txt` | copy | hyper_offense | giraffe |
| 18 | `3977f6236e` | `data/teams/others/johnnyg2/1f48afa08a0d8f51.txt` | copy | semi_stall | johnnyg2 |
| 19 | `53e60239d2` | `data/teams/others/yak_attack/a5c671838fce89a9.txt` | copy | offense | yak_attack |
| 20 | `f4faa56a39` | `data/teams/others/giraffe/ed07e11e26b443fc.txt` | copy | balance | giraffe |
| 21 | `1defb7ba39` | `data/teams/others/yak_attack/77c8eb66c36ecec9.txt` | copy | hyper_offense | yak_attack |
| 22 | `eabfc3f133` | `data/teams/others/giraffe/2c181c09152a5477.txt` | copy | balance | giraffe |
| 23 | `dc94160296` | `data/teams/others/giraffe/55f9b006e460f138.txt` | copy | hyper_offense | giraffe |
| 24 | `3895ace219` | `data/teams/others/giraffe/763ea745c9767c15.txt` | copy | offense | giraffe |
| 25 | `37f1678a79` | `data/teams/others/yak_attack/78be29a6a0f1f2d0.txt` | copy | semi_stall | yak_attack |
| 26 | `cf5a05b36e` | `data/teams/others/johnnyg2/5d781fa29f556ba0.txt` | copy | stall | johnnyg2 |
| 27 | `187e48f614` | `data/teams/others/yak_attack/c793251cfe179043.txt` | copy | semi_stall | yak_attack |
| 28 | `4b1a1c2e58` | `data/teams/others/johnnyg2/be96c6d78622e609.txt` | copy | offense | johnnyg2 |
| 29 | `fc511de22b` | `data/teams/others/yak_attack/35f69502d4cc9151.txt` | copy | balance | yak_attack |
| 30 | `a0ac7f0f71` | `data/teams/others/giraffe/c477910ef0fec296.txt` | copy | offense | giraffe |
| 31 | `a3a1c1f5db` | `data/teams/others/giraffe/3a5e3c967c8012f6.txt` | copy | balance | giraffe |
| 32 | `f7874cfe2f` | `data/teams/others/schmuck_nick/dea061e41de8742f.txt` | copy | semi_stall | schmuck_nick |
| 33 | `68886bc169` | `data/teams/others/johnnyg2/9d1c0faf49bc0e1e.txt` | copy | offense | johnnyg2 |
| 34 | `747985a50d` | `data/teams/others/giraffe/9c2477a1ef31908d.txt` | copy | offense | giraffe |
| 35 | `98b691a908` | `data/teams/others/giraffe/e2e9f9f6a2a8c4a5.txt` | copy | hyper_offense | giraffe |
| 36 | `3f95b25e9a` | `data/teams/others/giraffe/29c41e122ed50a6b.txt` | copy | stall | giraffe |
| 37 | `ed562de44b` | `data/teams/others/giraffe/56ee62531c304648.txt` | copy | hyper_offense | giraffe |
| 38 | `009e3d0244` | `data/teams/others/giraffe/995ccaf788125f1e.txt` | copy | offense | giraffe |
| 39 | `bd865559ab` | `data/teams/others/giraffe/55a5648767ea3d21.txt` | copy | hyper_offense | giraffe |
| 40 | `30b097e497` | `data/teams/others/johnnyg2/a2159752826fa15e.txt` | copy | hyper_offense | johnnyg2 |

## Replacements

None — every drawn team validated on the first pass.

## Composition (REPORTED, not corrected)

| archetype | n |
|---|---|
| balance | 11 |
| hyper_offense | 11 |
| offense | 9 |
| semi_stall | 6 |
| stall | 3 |

| source folder | n |
|---|---|
| `giraffe` | 25 |
| `yak_attack` | 7 |
| `johnnyg2` | 6 |
| `mcmegan` | 1 |
| `schmuck_nick` | 1 |

A random draw reproduces the pool's own composition in expectation. A skew here is a property of the draw, and rebalancing it would put the selection confound back.
