# Smogon forum teams v2: the mega sweep (2026-10-18)

**What this is.** An owner-requested sweep, systematic rather than sampled, of every ADV OU (gen 3 OU)
team on Smogon's public pages. It extends
[v1](../smogon_forum_teams_2026-10-18/README.md). v1's manifest is unchanged, and the
`gen3ai_forum72` panel frozen from v1's core stays as it was. v2 re-reads v1's register as its first
source (`origin: v1_register`), so v2 is a superset of v1.

- Every legal team is kept with its tier (A / B / C), its provenance, its curator and its overlap.
  Nothing is excluded here; a panel's freeze rule does that.
- **Nothing is under `data/`** (F-DV-2).
- Nothing went into Metamon's cache, no games were played and the GPU was not used. CPU jobs ran
  under `nice -n 19 taskset -c 5-7,13-15`.

| | count |
|---|---|
| forum / site pages read or recorded by the crawlers (5 shards) | 2,338 page rows (incl. listing pages and `skipped:` rows) |
| team references recorded | 2,974 rows → 2,683 distinct paste URLs + 133 inline teams + others |
| candidate teams (after splitting multi-team pastes) | **11,450** |
| dropped: not exactly 6 mons (usage stats, set dumps, boxes) 1,173 · validator 648 · format tag not gen3ou 328 · typeless HP with no IVs line 8 | 2,157 |
| duplicates (same six mon sets as an earlier candidate) | 3,963 |
| **kept** (validator-legal, distinct): A 671 · B 262 · C 4,397 | **5,330** |
| kept that are new vs v1 (not exact or near a v1 team) | 4,688 |
| kept and **clean against our 719-team pool**: A 407 · B 245 · C 3,580 | **4,232** |
| **independent core, tiers A/B** (clean vs pool, `competitive`, `hl_05_26`, `sample`, heldout72; ≤ 3 shared mon sets with every other core team) | **559** (533 verbatim) |
| **independent core, tiers A/B/C** | **2,995** (2,894 verbatim) |
| team references that could not be read (`manifest_unread.jsonl`) | 143 |

**Verbatim vs transcribed.**
- `fidelity: verbatim` means the paste site's own text: 5,221 kept teams.
- `transcribed` means a model read the team off a forum page: 109 kept teams (F2).
- **Build a panel from the verbatim cores**: `independent_core_AB_verbatim` and
  `independent_core_ABC_verbatim` in `manifest_teams.jsonl`.

## Files

- `teams/sfv2_NNNN.txt`: one file per KEPT team (5,330). Normalised exactly as v1 (`make_home72`
  conventions). Transcribed rows also get the format-only `scripts/legacy_export.py` pre-pass.
- `manifest.json`: `counts`, plus `overlap_summary` against every reference (all / tier / core).
- `manifest_teams.jsonl`: one line per candidate (kept, dropped, duplicate). Each line has:
  - provenance: `source_url`, `thread_url`, `thread_title`, `post_author`, `post_date`, `post_no`,
    `page_url`, and `seen_in` (every post that linked the paste);
  - `curator`, `tier`, `tier_reason`, `category`, `origin` (the shard);
  - for kept teams: `sha256`, `validator`, `norm_flags`, `fidelity`, the per-reference `overlap`,
    `within_set_*`, the four `independent_core_*` flags and `also_seen_in` (duplicates' provenance).
- `manifest_unread.jsonl`: every reference that could not be turned into text, with its reason.
- `crawl/shards/<shard>/{queue,pages,teams}.jsonl`: the RESUMABLE crawl state. Each file has every URL
  queued, every page read or skipped with its status, and every team reference.
- `crawl/spoink_resolved.json`: where each `spo.ink` short link points. spo.ink is Smogon's own link
  shortener; one GET was made per link, redirects were not followed, and the Location header was
  recorded.
- `scripts/`:
  - `crawl.py`: the state tool, the shared per-host throttle (≥ 2 s, flock) and `inline-check`.
  - `fetch_refs.py`: the paste fetcher.
  - `legacy_export.py`: the format-only canonicaliser.
  - `build_v2.py`: imports v1's own normaliser, validator wrapper and references.
  - `fetch.py` / `pagetext.py`: copies of v1's.

Rebuild (the page cache is local, at `~/.cache/gen3ai/smogon_crawl`; `fetch_refs.py` refills it from
the crawl state):

```bash
export PYTHONPATH=$PWD/src CUDA_VISIBLE_DEVICES=""
S=designs/research_state/measurements/smogon_forum_teams_v2_2026-10-18/scripts
python3 -I $S/fetch_refs.py ~/.cache/gen3ai/smogon_crawl
nice -n 19 taskset -c 5-7,13-15 python $S/build_v2.py ~/.cache/gen3ai/smogon_crawl
```

## Method

1. **Crawl.** Five crawler agents (opus-medium) each owned one shard. Successor agents resumed
   from the on-disk queue after stops for context or budget. Every Smogon read went through WebFetch,
   because the forum JS-checks plain HTTP clients (v1 F1), and was preceded by the shared throttle
   (≥ 2 s per host across all agents). Each agent:
   - enumerated the listing pages;
   - read every page of every in-scope thread, under the per-shard coverage rules below;
   - recorded every pokepast.es / pastebin.com / other paste link and every inline six-mon team, with
     its provenance.
2. **Fetch.** `fetch_refs.py` fetched pokepast.es `/json` and pastebin.com `/<id>` pages (never
   `/raw/`), throttled per host, after a robots.txt check on every host:

   | host | robots.txt | action |
   |---|---|---|
   | smogon.com | no disallow | read |
   | pokepast.es | none (404) | read |
   | pastebin.com | disallows `/raw/` | `/<id>` pages only |
   | dropbox.com | disallows `/sh/` | Golden Sun's folder NOT fetched |
   | spo.ink | redirects to `smogon.com/spoink` | resolved by one GET per link |
   | docs.google.com, pastebin.pl, pokemonperfect.com | not checked | not fetched |
3. **Build.** `build_v2.py`:
   - splits multi-team pastes on `=== [format] name ===` headers. A paste of six-mon groups separated
     by `--` lines is split only when every segment is exactly six (this resolved v1's 20-mon "double
     dogs" paste into 3 teams);
   - normalises as v1;
   - validates with the pinned `gen3ou` validator. A validator infrastructure error aborts the build;
     it is never counted as a verdict;
   - dedupes on the six mon sets (first in tier order A → B → C, v1 first within a tier);
   - measures overlap with `main.anchors.team_overlap` (`gen3_team_overlap_v1`) against the five
     references plus v1's set;
   - flags the cores greedily, in that same order.

## Shards and coverage

| shard | scope | pages | team refs | coverage rule |
|---|---|---|---|---|
| `adv_main` | every thread listed in the ADV forum (`/forums/forums/adv/`): Set and Team Sharing (11 pp.), Sample Teams, Teams Through the Ages, Teambuilding Competition (all entries with placings), Viability Ranking (46 pp.), Beginner's Lounge, Ladder Achievements, Battle of the Week, CALLOUS / SPL / World Cup threads, etc. | 194 | 1,624 | every page |
| `adv_rmt` | ADV Rate My Team (6 listing pages), ADV Replay Review | 260 | 320 | every page |
| `adv_archive` | ADV Archive, ADV Tournament Archive, and the ADV Circuit Tournaments link (Old Gen Metagames Circuits, ADV prefix, 301 threads) | 1,266 rows | 970 | every page, EXCEPT tournament round / signup / finals threads: page 1 of all ~320 read, later pages read only if page 1 had a team (F6) |
| `old_rmt` | Gen 3 threads in the general RMT areas: Team Showcase, RMT Archive, Past Gen Teams (prefix 67), RMT Projects, and all 372 listing pages of "2007-2022 Rate My Team" (title filter) | 463 | 29 | prefix filter where it exists; else title filter (F7) |
| `tours_misc` | Smogon-wide tournaments (SPL, WCoP, Classic, Old Gen Circuits, RoA tournaments, Unofficial tours), The Smog / RS articles, the RS dex | 155 | 31 | listing title filter for team / dump / usage threads (F6) |

**Tiers** (each row has a `tier_reason`):
- **A**: official sample teams, staff-curated archives, teams posted as used in a tournament, team
  dumps by tournament players.
- **B**: an RMT or team showcase with substantive positive feedback from other posters, or a judged
  competition entry.
- **C**: everything else, mostly the Set and Team Sharing thread.

## Totals

**By tier (kept):** A 671 · B 262 · C 4,397.

**By source thread (kept, top):**

| thread | tier | kept |
|---|---|---|
| ADV OU Set and Team Sharing | C | 3,925 |
| ADV OU Sample Teams (v1 register incl. the dumps) | A | 385 |
| ADV OU Metagame Discussion | C | 141 |
| ADV OU Set and Team Sharing | A | 128 |
| Teambuilding Competition (all rounds) | B | 92 |
| Ladder Achievements | B | 69 |
| JI3 win and even more free teams | A | 63 |
| Secret Santa 2025 / 2021 | C | 51 / 31 |
| Johnald Teamdump March 2026 | B | 46 |
| Teams Through The Ages | A | 37 |
| SPL XIV / SPL Review / SPL XI postseason | A | 9 / 9 / 9 |

Eight single pokepastes are multi-team dumps of 196–548 teams each.

**By curator (kept, top 15):**

| curator | kept |
|---|---|
| giraffefromholland | 1,350 |
| Zpanther | 765 |
| Shitrock enjoyer | 630 |
| SPFunkM | 549 |
| thelinearcurve | 228 |
| johnnyg2 | 164 |
| Fruhdazi | 124 |
| Ibidem | 108 |
| SEA | 86 |
| Endill | 66 |
| L3W | 63 |
| Johnald | 51 |
| McMeghan | 50 |
| Rainer24 | 49 |
| M Dragon | 37 |

**Curators of the cores:**
- **A/B verbatim core:** thelinearcurve 60, Ibidem 56, Endill 56, Johnald 34, M Dragon 26, Spreek 19,
  mikmer 16, Siglut 13, CALLOUS 13, … (`counts.core_AB_verbatim_by_curator_top15`).
- **A/B/C core:** giraffefromholland 593, Zpanther 487, SPFunkM 317, Shitrock enjoyer 282, …

## Overlap (`gen3_team_overlap_v1`)

| reference | set | n ref | exact | ≥ 4 shared | same 6 species | mon sets seen | strict seen |
|---|---|---|---|---|---|---|---|
| training pool | all 5,330 | 719 | **711** | 1,098 | 1,569 | 26,175 / 31,980 | 12,214 |
| training pool | core A/B 559 | 719 | 0 | 0 | 63 | 2,418 / 3,354 | 758 |
| training pool | core A/B/C 2,995 | 719 | 0 | 0 | 428 | 13,554 / 17,970 | 4,418 |
| `data/teams/sample` | all | 72 | 71 | 286 | 379 | 19,760 / 31,980 | 6,063 |
| `data/teams/sample` | core A/B/C | 72 | 0 | 0 | 123 | 9,871 / 17,970 | 2,305 |
| Metamon `competitive` | all | 20 | 19 | 40 | 97 | 9,836 / 31,980 | 873 |
| Metamon `competitive` | core A/B/C | 20 | 0 | 0 | 32 | 5,375 / 17,970 | 404 |
| `hl_05_26` | all | 22,862 | 0 | 202 | 2,960 | 26,628 / 31,980 | 8,777 |
| `hl_05_26` | core A/B/C | 22,862 | 0 | 0 | 1,484 | 14,675 / 17,970 | 4,905 |
| heldout72 | all | 72 | 0 | 0 | 192 | 8,648 / 31,980 | 1,157 |
| heldout72 | core A/B/C | 72 | 0 | 0 | 79 | 4,616 / 17,970 | 642 |
| v1 set | all | 437 | 437 | 642 | 970 | — | — |
| v1 set | core A/B | 437 | 166 | 167 | 191 | — | — |

The 166 v1 teams in the A/B core are v1's own independent core, re-taken in register order.

Per team, against the pool (kept):

| tier | clean | near | exact |
|---|---|---|---|
| A | 407 | 31 | 233 |
| B | 245 | 15 | 2 |
| C | 3,580 | 341 | 476 |

## What could not be read

| item | reason |
|---|---|
| 68 pastebin / pokepaste links | HTTP 404 (deleted pastes); `~/.cache/gen3ai/smogon_crawl/fetch_refs_errors.jsonl` |
| 47 inline teams | the two transcriptions disagreed, or the team was not transcribed (non-OU / prose) |
| 28 links on other hosts, recorded and not fetched | 13 spo.ink links that resolve to forum posts or have no redirect; 5 Google docs/sheets; 4 pastebin.pl; 2 Dropbox (robots.txt disallows `/sh/`); 1 pokemonperfect.com; 2 malformed pokepast.es ids; 1 malformed pastebin |
| the RS strategy dex (`/dex/rs/formats/ou/`) | JavaScript-rendered; WebFetch saw no teams (UNVERIFIED that none exist) |
| forum search (`/search/?q=…`) | HTTP 403 to the reader, once; not retried |
| ~20 teams posted only as screenshots (UD, VR thread page 27; El Clasico post 1010) | images; not transcribed |
| tournament teams shown as sprite icons (CALLOUS, SPL XIII, World Cup II, SPL 9 weeks 3-9) | species only, no sets |

## Findings

- **F1. Our training pool is largely this forum.** 711 of the pool's 719 teams appear here as exact
  copies. The curators include `johnnyg2` and `SchmuckNick`, which are pool collection names in
  `data/teams/others/`, and `giraffefromholland`, which is presumably the "giraffe" collection. The
  `Set and Team Sharing` thread is the pool's main upstream. So "clean vs pool" here selects teams
  posted after the pool was built, or not imported into it.
- **F2. Transcription is weak evidence (109 kept teams).**
  - The WebFetch reader caches pages and is near-deterministic, so read 2 often copies read 1. In one
    case (Sepulo, "Hotshot") two reads produced two different whole teams, so the reader can
    confabulate.
  - One crawler built some read-2 files by `sed` from read 1. It verified the differences by eye
    first, and the rows' notes say so, but not every such row is identifiable.
  - `inline-check` compares sets in order and canonicalises spelling. It still passed one case with a
    stray "(McM edit)" move suffix and one with an added gender tag.
  - **Use the verbatim cores (533 A/B, 2,894 A/B/C) for any panel.**
- **F3. Reader errors in provenance.**
  - Post numbers drift on long pages; some rows hold forum post ids, or 0.
  - Some pages needed 2–4 offset reads.
  - A few hash errors were caught and re-read: a 15-character id, `…c15` vs `…c13`, and a garbled
    `901e4486971e31.1That'shat`.
  - Two pastes carry conflicting credits (`ad85129d1352b49c` is "Star's team" in one place and
    "BKC's SpinMie" in another; `2ad2fcb7e3f831c8` likewise).
  - About 410 links from one compilation post (jimmy the card, Set and Team Sharing page 10) are
    flagged "unverified" for their per-team labels. The pastes themselves are fetched verbatim.
  - The listing reader's page and thread counts were occasionally wrong; pages were re-read or
    recorded `skipped` with a reason.
- **F4. Curator concentration is severe in tier C.** Four curators (giraffefromholland, Zpanther,
  Shitrock enjoyer, SPFunkM) hold 3,294 of the 5,330 kept teams and 1,679 of the 2,995 A/B/C core,
  mostly from single multi-team pokepastes of 196–548 teams. A frozen panel must cap curator share,
  as forum72 did (cap 14).
- **F5. Tiering is a model's judgement.** B means "positive feedback from other posters" as the
  reader saw it. Praise in the VR thread came as reactions, not replies, so those teams are C. Some
  B calls rest on one short reply.
  - Four staff-archived RMTs were tiered B in `adv_rmt` and A in `old_rmt`. Dedupe keeps the A copy;
    the B one lands in `also_seen_in`.
  - 85 competition entries are B. Round 8 entries are C (not judged yet). Round 7 placings exist
    only as an image.
- **F6. Tournament rounds hold replays, not teams; their later pages were not all read.**
  - Of ~320 ADV tournament round / signup / finals threads, page 1 of every one was read. Only the
    2023 Circuit Championship Finals had paste links (2 pastebins; tier A, content unverified as
    teams).
  - Several hundred later pages are recorded `skipped:page1-has-no-team-refs`. A team posted late in
    such a thread would be missed.
  - Smogon-wide tournament forums were filtered by listing title (team / dump / usage). Round threads
    there were not opened, beyond 6 sampled ones that held none.
  - Never rebuilt from replays (v1 F6).
- **F7. Old general RMT forums were title-filtered.**
  - "2007-2022 Rate My Team" (May 2007 – Aug 2012, 372 listing pages, ~15,000 threads) has no Gen 3
    prefix. Every listing page was read, but only threads whose TITLE named Gen 3 / ADV / RSE were
    opened. In that era titles rarely named the generation, so untitled ADV teams there are missed.
  - Many title hits were DPP teams. One "Successful ADV Team" used Bullet Punch and Stealth Rock;
    those are skipped.
  - Past Gen Teams' own rules (2025-10-21) moved ADV team posts to ADV Rate My Team, which is fully
    crawled.
- **F8. Non-OU content is excluded by construction.** Excluded content:
  - ADV Lower Tiers subforum: not crawled; it is not OU;
  - 69 non-OU archive threads, skipped by title;
  - pastes tagged another format: 205 `gen9`, 75 `gen3ou-box` (sets boxes), 22 `gen3`, and others;
  - in-cartridge FR/LG threads.
- **F9. Rule slips by crawlers, all disclosed and harmless.**
  - One crawler fetched one pokepast.es page directly, unthrottled for that host, to check what a
    stats paste was.
  - Crawlers used throwaway `/tmp` helper scripts to build JSON rows. Every throttle call came before
    its fetch.
  - The first `adv_main` brief went out with its common instructions unexpanded. The crawler found the
    contract in `/tmp/crawl_common.txt` and the correction followed.
- **F10. Usage-statistics pastes are numerous and sometimes tier-labelled A by the crawler.** None
  survive the build, because none parse as six-mon teams. Their rows are in `manifest_teams.jsonl` as
  `dropped: not_6_mons`.
- **F11. 13 spo.ink links resolve back into forum posts on smogon.com.** They were not followed.
  Their teams, if any, are on pages the shards read (UNVERIFIED for each one).
