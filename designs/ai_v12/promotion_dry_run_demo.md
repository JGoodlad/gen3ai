# Team promotion — 40 teams, seed `20260830`

> ⚠️ **DEMO DRAW — not the launch draw.** Produced by `python -m main.promote_teams --draw-only
> --seed 20260830` on 2026-08-30 to exercise and document the tool. **Nothing was promoted**; no
> file under `data/teams/` was touched. The real draw happens at launch with a fresh seed and the
> training session watching, and writes its manifest into `data/teams/sample/`. Reproduce it
> exactly with `--dry-run --seed 20260830`; `promote_teams_test.py` re-derives it on every run.

Generated 2026-08-30T04:05:47+00:00 at `56bfd48661` by `python -m main.promote_teams`.

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

Source: `/home/goodlad/dev/gen3ai/.claude/worktrees/agent-a9d86d34a29a263f2/designs/ai_v12/promotion_exclusions.json`.

## The draw

| # | sha | source | action | archetype | folder |
|---|---|---|---|---|---|
| 1 | `16d681e970` | `data/teams/others/giraffe/73bea8a147236023.txt` | copy | stall | giraffe |
| 2 | `a3bc0b0470` | `data/teams/others/mcmegan/3e98a75821c859bd.txt` | copy | semi_stall | mcmegan |
| 3 | `7ad65bac76` | `data/teams/others/giraffe/6b0e85bcbb852991.txt` | copy | balance | giraffe |
| 4 | `6e12f4e836` | `data/teams/others/giraffe/39f39aa240c34f11.txt` | copy | balance | giraffe |
| 5 | `ff78d9fb04` | `data/teams/others/giraffe/62ab6c32ece9cd5e.txt` | copy | hyper_offense | giraffe |
| 6 | `e43ea0314c` | `data/teams/others/giraffe/bfb86d7025728d54.txt` | copy | hyper_offense | giraffe |
| 7 | `3c7640632e` | `data/teams/others/giraffe/2aaccd0c78c1172b.txt` | copy | hyper_offense | giraffe |
| 8 | `6ef6e77850` | `data/teams/others/giraffe/637604269761c864.txt` | copy | hyper_offense | giraffe |
| 9 | `a9f6bcf79c` | `data/teams/others/johnnyg2/c93248f100dbd588.txt` | copy | semi_stall | johnnyg2 |
| 10 | `a577a735b7` | `data/teams/others/yak_attack/82be38ccdf48af18.txt` | copy | balance | yak_attack |
| 11 | `1c1420342b` | `data/teams/others/giraffe/0ceb14cbaf3692df.txt` | copy | offense | giraffe |
| 12 | `b904dbe059` | `data/teams/others/giraffe/104e2a85a6e9af88.txt` | copy | balance | giraffe |
| 13 | `683908e124` | `data/teams/others/giraffe/45eec47ae82b85f9.txt` | copy | balance | giraffe |
| 14 | `8cd386c07d` | `data/teams/others/johnnyg2/39ca8ea07e3d8c99.txt` | copy | balance | johnnyg2 |
| 15 | `ab54c04e46` | `data/teams/others/giraffe/80582acc8106c4df.txt` | copy | balance | giraffe |
| 16 | `3bdc4b63dd` | `data/teams/others/giraffe/cdc15fa0e32c3451.txt` | copy | balance | giraffe |
| 17 | `13388c8b10` | `data/teams/others/giraffe/fff27ca1c7fb8550.txt` | copy | hyper_offense | giraffe |
| 18 | `3895ace219` | `data/teams/others/giraffe/763ea745c9767c15.txt` | copy | offense | giraffe |
| 19 | `525408901e` | `data/teams/others/mcmegan/d07725903ef50648.txt` | copy | balance | mcmegan |
| 20 | `f4d8046665` | `data/teams/others/yak_attack/6467a4f18bf4f571.txt` | copy | semi_stall | yak_attack |
| 21 | `1defb7ba39` | `data/teams/others/yak_attack/77c8eb66c36ecec9.txt` | copy | hyper_offense | yak_attack |
| 22 | `eaa88395e7` | `data/teams/others/giraffe/492e282ad33e0f7b.txt` | copy | hyper_offense | giraffe |
| 23 | `dc1b87ad2a` | `data/teams/others/yak_attack/803bab977c23bb23.txt` | copy | semi_stall | yak_attack |
| 24 | `382e285965` | `data/teams/others/yak_attack/de6092df9621942b.txt` | copy | balance | yak_attack |
| 25 | `37d5d7dca4` | `data/teams/others/giraffe/b20a4344ae6ac088.txt` | copy | hyper_offense | giraffe |
| 26 | `ceab5d099e` | `data/teams/others/yak_attack/d088ac1235f6bf46.txt` | copy | stall | yak_attack |
| 27 | `187e48f614` | `data/teams/others/yak_attack/c793251cfe179043.txt` | copy | semi_stall | yak_attack |
| 28 | `4ad093b5b5` | `data/teams/others/johnnyg2/9bfb2769871d0bbd.txt` | copy | semi_stall | johnnyg2 |
| 29 | `faf49668ea` | `data/teams/others/yak_attack/2ed5a563f0c780b5.txt` | copy | stall | yak_attack |
| 30 | `a0ac7f0f71` | `data/teams/others/giraffe/c477910ef0fec296.txt` | copy | offense | giraffe |
| 31 | `a3a1c1f5db` | `data/teams/others/giraffe/3a5e3c967c8012f6.txt` | copy | balance | giraffe |
| 32 | `f75c5fef49` | `data/teams/others/giraffe/3674ba5bb3f0001f.txt` | copy | offense | giraffe |
| 33 | `68886bc169` | `data/teams/others/johnnyg2/9d1c0faf49bc0e1e.txt` | copy | offense | johnnyg2 |
| 34 | `747985a50d` | `data/teams/others/giraffe/9c2477a1ef31908d.txt` | copy | offense | giraffe |
| 35 | `98b691a908` | `data/teams/others/giraffe/e2e9f9f6a2a8c4a5.txt` | copy | hyper_offense | giraffe |
| 36 | `3ec9787dfd` | `data/teams/others/yak_attack/48bdf3938e72aa6b.txt` | copy | stall | yak_attack |
| 37 | `eb308a7fd8` | `data/teams/others/yak_attack/0dfb219a4ccf3278.txt` | copy | semi_stall | yak_attack |
| 38 | `009e3d0244` | `data/teams/others/giraffe/995ccaf788125f1e.txt` | copy | offense | giraffe |
| 39 | `bca128e264` | `data/teams/others/giraffe/edf65aba70eab245.txt` | copy | hyper_offense | giraffe |
| 40 | `2fde3f49fc` | `data/teams/others/yak_attack/79b6997959cfa712.txt` | copy | offense | yak_attack |

## Replacements

None — every drawn team validated on the first pass.

## Composition (REPORTED, not corrected)

| archetype | n |
|---|---|
| balance | 11 |
| hyper_offense | 10 |
| offense | 8 |
| semi_stall | 7 |
| stall | 4 |

| source folder | n |
|---|---|
| `giraffe` | 23 |
| `yak_attack` | 11 |
| `johnnyg2` | 4 |
| `mcmegan` | 2 |

A random draw reproduces the pool's own composition in expectation. A skew here is a property of the draw, and rebalancing it would put the selection confound back.
