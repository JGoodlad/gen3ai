# Smogon forum teams: a third-source candidate set (2026-10-18)

**What this is.** Competitive ADV OU (gen 3 OU) teams collected from Smogon's public ADV forum and
the paste sites its resource posts link to. It is a candidate for the **T2(b)** set of
[`design_diversity_teams_opponents.md`](../../../endstate/design_diversity_teams_opponents.md) §3.1:
human-built, competition-grade teams unseen by our training pool. Nothing here is excluded yet. The
design doc's freeze rule picks the frozen panel later. **Nothing was written under `data/`** (it is
frozen while X26 is live, and a file under `data/teams/` would silently widen the training pool:
F-DV-2). Nothing was installed into Metamon's cache, and no games were played.

| | count |
|---|---|
| candidate teams read | 491 |
| dropped (Showdown validator: 26 · not exactly 6 mons: 2) | 28 |
| duplicates of an earlier candidate (same six mon sets) | 26 |
| **kept** (validator-legal, distinct): tier A 430 · tier B 7 | **437** |
| kept, **clean against our 719-team pool** (tier A/B) | **219** |
| kept, clean against EVERY reference below (tier A/B) | 214 |
| **independent core**: clean against every reference AND ≤ 3 mon sets shared with every other core team | **166** (A 160, B 6) |

The brief's target was ≥ 72 validated tier-A/B teams clean against the pool. **219 meet it, and 166
form an independent core.** Read the source-concentration and set-novelty findings (F4, F5) before
treating 166 as 166 independent samples.

## Files

- `teams/sftNNN.txt`: one file per KEPT team (437). Showdown export, nickname-free, one blank line
  between mons. Every Hidden Power is written `[Type]` with an explicit `IVs:` line.
- `manifest.json`: one row per candidate (kept, dropped and duplicate). Each row records the
  provenance (`source_url`, `thread_url`, `thread_title`, `post_author`, `post_date`, `credited`
  builder, `team_name`, plus the paste's own `paste_title` / `paste_author` / `paste_date`), the
  `tier`, the `category`, the validator result, the `norm_flags`, the `sha256` of the team file, the
  per-reference `overlap` (`exact` / `near` / `clean`, max shared mon sets, the nearest reference
  team), `within_set_max_shared` and `independent_core`. `counts` and `overlap_summary` hold every
  number in this README.
- `scripts/`:
  - `fetch.py`: the throttled fetcher (≥ 2.5 s between requests, cached by URL, never retries).
  - `sources.py`: the source register (provenance as read on the page).
  - `inline_teams.py`: the 7 transcribed inline teams (F2).
  - `pagetext.py`: pulls the text out of the pastebin and pokepast.es pages.
  - `build.py`: extract → normalise → validate → dedupe → overlap → write.

Rebuild (the cache is not committed; `fetch.py` refills it from the register):

```bash
export PYTHONPATH=$PWD/src CUDA_VISIBLE_DEVICES=""
nice -n 19 taskset -c 5-7,13-15 python designs/research_state/measurements/smogon_forum_teams_2026-10-18/scripts/build.py <cache_dir>
```

## Method

1. **Search.** WebSearch, then a read of each Smogon page with WebFetch. The forum serves a JavaScript
   browser check to plain HTTP clients (F1). Team TEXT came from the paste site itself:
   - the pokepast.es `/json` body;
   - or the pastebin.com page's source listing.
   Only provenance came through the page reader, except the 7 transcribed teams (F2).
2. **Normalise** exactly as `n0_endofrun_2026-09-27/scripts/make_home72.py` does:
   - strip nicknames (a `(M)`/`(F)` tag is kept);
   - spell every Hidden Power `[Type]` with the canonical `IVs:` line from `utils.gen3_utils.GEN3_HP_IVS`;
   - use one blank line between mons.

   It also makes two changes, both SPELLING only and both flagged in `norm_flags`:
   - **(a)** a Hidden Power whose paste already carries an `IVs:` line that gives the bracketed type
     KEEPS that line (home72 always replaced it). No kept team needed a mismatched line replaced.
   - **(b)** pre-2015 export spelling is modernised: `Trait:` becomes `Ability:`; `Spd` / `SAtk` /
     `SDef` become `Spe` / `SpA` / `SpD`; `Adamant Nature (+Atk, -SAtk)` becomes `Adamant Nature`.
     19 teams, all from the Teams-Through-the-Ages pastes and one transcription.

   No set content was changed. A typeless Hidden Power would have been typed only from the page's own
   `IVs:` line and dropped otherwise; none occurred.
3. **Validate**: `utils.bridge.team_validator.validate_teams_locally("gen3ou", …)`, the pinned
   `deps/pokemon-showdown` (e0551883), the same validator the anchors and `make_home72` use. A
   validator infrastructure error aborts the build; it is never counted as a verdict.
4. **Dedupe** on the six mon sets (`main.anchors.team_overlap`, species + sorted moves, typed Hidden
   Power). The register is ordered sample teams → history archive → competition winners → team
   analyses → transcriptions → dumps, and the first copy wins.
5. **Overlap** with `main.anchors.team_overlap` (`gen3_team_overlap_v1`, fda1efda). `exact` means the
   six mon sets equal one reference team's; `near` means ≥ 4 shared with one reference team (the
   heldout72 rule); otherwise `clean`.

## Sources searched

Etiquette:
- public pages only: no login, no posting, no contact with anyone;
- one request at a time, ≥ 2 s apart;
- robots.txt read: smogon.com has no disallow for general agents; pastebin.com disallows `/raw/` (F3);
  pokepast.es has none;
- only team text and provenance are stored.

| source | URL | used | yield |
|---|---|---|---|
| ADV OU Sample Teams (mielke, post 1, 2021-07-16; vapicuno, post 2 archive + dump links) | https://www.smogon.com/forums/threads/adv-ou-sample-teams.3687813/ | 32 + 22 pokepastes, 5 dumps | 50 kept; 46 are pool `exact`/`near`. `data/teams/sample` IS this thread |
| ADV Beginner's Lounge resource post (lists the dumps and the team analyses) | https://www.smogon.com/forums/threads/adv-beginners-lounge-rules-faq-resources-question-answer.3687817/ | the index | — |
| thelinearcurve dumps 2019 / 2020 (all, viable) / 2021 | https://pastebin.com/jyFh4AF3 · https://pastebin.com/tUytkX29 · https://pastebin.com/QbwUxw0P · https://pastebin.com/6xCb1crp | multi-team pastes | 2019: **every kept team is in our pool**; 2020: 56 clean; 2021: 10 clean |
| Ibidem 2017 dump | https://pastebin.com/i7HzJQmC | 116 teams | 101 pool-clean |
| Golden Sun "OU Team dump" (2016-07-03) | https://www.smogon.com/forums/threads/team-dump.3576354/ → 6 pastebins | 6 | 6 pool-clean. Its Dropbox folder was NOT fetched |
| ADV Teams Through The Ages (M Dragon, 2021-07-04; Taylor, 2022-06-05) | https://www.smogon.com/forums/threads/adv-teams-through-the-ages.3687203/ | 32 pokepastes | 30 kept, all pool-clean (NetBattle-era national metagames, Smogon Tour era) |
| ADV OU Teambuilding Competition, round winners (Hall of Fame) | https://www.smogon.com/forums/threads/adv-ou-teambuilding-competition.3770678/ | 7 | 7 kept, tier **B** (themed constraints) |
| Resource-listed team analyses / RMTs (11 threads) | e.g. https://www.smogon.com/forums/threads/indestructible.3694811/ (others in `manifest.json`) | 3 pokepastes + 7 transcribed | 9 kept; Late Game Zapdos fails Speed Pass Clause |
| CALLOUS Invitational 7 (teams, replays, usage) | https://www.smogon.com/forums/threads/callous-invitational-7-teams-replays-and-usage-statistics.3722746/ | — | teams shown as sprites + replays only; its pastes are usage statistics. Not collected (F6) |
| ADV OU Teambuilding Compendium | https://www.smogon.com/forums/threads/adv-ou-teambuilding-compendium.3777674/ | — | a role list, no teams |
| ADV OU Set and Team Sharing (11 pages) | https://www.smogon.com/forums/threads/adv-ou-set-and-team-sharing.3648620/ | — | NOT harvested (F7) |
| ADV OU Team Dump (2016) | https://www.smogon.com/forums/threads/adv-ou-team-dump.3584528/ | — | HTTP 403 to the reader, twice (F1) |
| set dumps (Jhonx 2021, thelinearcurve 2019/2020 sets) | https://pastebin.com/qjNWD0bJ · https://pastebin.com/haMgma5w · https://pastebin.com/ubq1NEVR | — | single SETS, not teams; skipped |

**Tiers.**
- **A**: the sample teams, the curated dumps, the Teams-Through-the-Ages archive and the analyses that
  the ADV resource post lists (moderator-curated archive entries).
- **B**: competition winners chosen by a judging panel.
- **C**: none collected.

`category` keeps the distinction inside A (`sample_team`, `sample_team_archived`,
`team_archive_history`, `curated_team_dump`, `resource_listed_team_analysis`).

## Drop reasons (28)

Validator (26):
- Speed Pass Clause / One Boost Passer Clause (Baton Pass chains), 13 including duplicates across dumps;
- Soundproof banned (Mr. Mime / Exploud), 4;
- Baton Pass + Mean Look, 5;
- Sand Veil banned, 1;
- Swagger banned, 1;
- Cursed Body (a gen-4 ability on Gengar, a dump typo), 2;
- an illegal Skarmory egg-move combination (Drill Peck + Whirlwind), 1;
- a 0-EV Zapdos ("did you forget to EV it?"), 1.

These are current Smogon clauses applied to older teams, plus two data errors in the dumps. Not
6 mons (2):
- a 20-mon pokepaste with no team headers ("double dogs / Flygon TSS"); splitting it would be a guess;
- a single-set paste ("Modest Zapdos").

Every row and its validator message is in `manifest.json`.

## Overlap (`gen3_team_overlap_v1`)

**All 437 kept teams:**

| reference | n ref | exact | ≥ 4 shared with one team | same 6 species | mon sets seen |
|---|---|---|---|---|---|
| training pool (`data/teams`) | 719 | **198** | 218 | 243 | 2323 / 2622 |
| `data/teams/sample` | 72 | 41 | 46 | 68 | 1711 / 2622 |
| Metamon `competitive/gen3ou` | 20 | 16 | 23 | 33 | 1174 / 2622 |
| `hl_05_26/gen3ou` | 22,862 | 0 | 18 | 307 | 2266 / 2622 |
| heldout72 | 72 | 0 | 0 | 23 | 549 / 2622 |

**The independent core (166):**

| reference | exact | ≥ 4 shared | same 6 species | mon sets seen | strict mon sets seen |
|---|---|---|---|---|---|
| training pool | 0 | 0 | 22 | **764 / 996** | 289 / 996 |
| `data/teams/sample` | 0 | 0 | 9 | 549 / 996 | 93 / 996 |
| Metamon `competitive` | 0 | 0 | 5 | 343 / 996 | 20 / 996 |
| `hl_05_26` | 0 | 0 | 105 | 835 / 996 | 179 / 996 |
| heldout72 | 0 | 0 | 3 | 203 / 996 | 16 / 996 |

Per team, against the pool, by the brief's labels (kept, A/B):

| label | count |
|---|---|
| exact | 198 |
| near (≥ 4 of 6) | 20 |
| clean | 219 |

## Named teams

The owner's "Jirachi sand spam" and "jirachee … spire" were not found under those names.

- **"Jirachee" is a Smogon ADV player.** Their "Regice Spikes Offense" is a sample team:
  https://pokepast.es/ce35b7368c3d692e (`sft011`, pool **exact**).
- **"Superman"** is the ADV archetype: Skarmory + Flygon + a special wall (Blissey or **Jirachi**), often
  with **sand**. It is the likeliest reading of "Jirachi sand":
  - **Gengar + Jirachi "Superman" TSS**, credited to ADV Community: https://pokepast.es/82e97fe2bfbb6f69
    (`sft002`, pool exact).
  - **Steelwork** by Foggi (2017-07-19), Jirachi + Tyranitar sand + Skarmory + Flygon + Aerodactyl, the
    resource post's "2017 Foggi Superman (Steelwork)":
    https://www.smogon.com/forums/threads/steelwork.3609872/ (`sft090`, **clean**, transcribed).
  - **Moltres "Superman" TSS** by Golden Sun: https://pokepast.es/9f27f5d3e34021a7 (`sft003`, pool exact).
  - **Molt Superman** (2018): https://pokepast.es/26c7b11a3462fd02 (`sft032`, pool exact).
- **WishJira Skarm Beatdown** by Astamatitos: https://pokepast.es/e0d97b0ed592889d (`sft029`, pool exact).
- Other famous named teams, all **clean**:
  - VenuPert (eden), `sft091`;
  - Explosion is an art / Boom Spam (Foggi), `sft092`;
  - Ozymandias / Smurf Rain (Smurf.), `sft093`;
  - querencia (McMeghan), `sft094`;
  - Special Offense + Spikes (UD), `sft095`;
  - Indestructible (EG Jhonx~);
  - Timeless Adventure (Teclis);
  - DDTar Offense (McMeghan).

## Findings

- **F1. Plain HTTP is refused by smogon.com/forums.** The forum serves a JavaScript "checking your
  browser" page to plain HTTP clients. It was not circumvented. All forum reads went through the
  WebFetch reader, a summarising model. Provenance fields (post author, date, team names, link lists)
  are therefore as that reader reported them. Two independent reads of the Sample Teams thread agreed
  on every link. Attribution inside Teams Through the Ages is partly illegible: `credited` is `""`
  where the reader could not tie a builder to a link. One thread (ADV OU Team Dump, 3584528) returned
  HTTP 403 to the reader twice and was not read.
- **F2. Seven teams are TRANSCRIBED, not copied.** They were posted inline with no paste link, and
  `transcribed: true` marks them. Each was read twice with differently worded prompts (a verbatim
  block, then a per-field list), and kept only if both reads agreed on species, item, ability,
  nature, EVs, IVs and moves. All seven agreed. A summarising reader can still corrupt a digit
  identically twice. The validator checks legality, not fidelity.
- **F3. A robots.txt slip, corrected.** Four pastebin `/raw/` pages were fetched in the SAME command
  that first read pastebin's robots.txt, which disallows `/raw/`. Those four responses were deleted
  unread from the cache. Every dump was then re-fetched as the allowed paste page (`/<id>`).
- **F4. Source concentration.** 116 of the 166 core teams come from two curators: thelinearcurve (60)
  and Ibidem (56). The core is therefore two people's teambuilding plus about 50 others. A frozen
  panel should cap per-curator share; that is the freeze rule's call.
- **F5. "Clean" is team-level novelty, not set-level.** 764 of the core's 996 mon sets (77 %) appear,
  as species + moves, on some pool team, and 835 (84 %) appear in `hl_05_26`. With item, ability,
  nature and EVs also matched, only 289 (29 %) appear in the pool. ADV's standard sets are few, so
  any expert-built set is mostly made of known sets. What these teams add is unseen COMBINATIONS.
- **F6. Tournament teams are not available as text.** SPL / CALLOUS / ADV Cup threads publish teams
  as sprites plus replays, and their pastes are usage statistics. Rebuilding teams from replays gives
  partial, `hl_05_26`-like information, which this unit did not do.
- **F7. Not exhausted.** The following were left un-harvested because the target was met:
  - the 11-page Set and Team Sharing thread;
  - the ADV Rate My Team forum beyond the resource-listed analyses;
  - Golden Sun's Dropbox folder;
  - the Teambuilding Competition's non-winning entries;
  - the 20-mon "double dogs" paste.

  Each would add tier-B/C teams.
- **F8. Our pool already holds whole dumps.** Every kept team of thelinearcurve's 2019 dump and 152
  of the 341 kept dump teams are exact pool copies. `data/teams/others/` evidently imported these
  same Smogon dumps. That is why the set's pool-clean fraction is only about half.
- **F9. 169 kept teams have a ≥ 4-shared near-twin inside this set** (dump variants). The independent
  core removes them greedily, in register order.
- **F10. Era and rules.** The history archive's NetBattle-era teams (2004-2010) are famous but predate
  the modern ADV metagame. Current clauses dropped 13 Baton Pass teams that were legal when built.
  `category` and `post_date` let the freeze rule weigh both.
