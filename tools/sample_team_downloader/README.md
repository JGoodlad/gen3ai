# Sample Teams Sync Tool

Syncs `data/teams/sample/` to **exactly** the teams linked from the first post of Smogon's
[ADV OU sample-teams thread](https://www.smogon.com/forums/threads/adv-ou-sample-teams.3687813/).
It is the folder's **only writer**: `data/teams/sample/` is the CURATED set (and the training bias),
nothing else. Teams promoted to be exploiter trainees live in `data/teams/promoted/`
(`python -m main.promote_teams`); see the role table in `tools/CLAUDE.md`.

## Usage
From the project root:
```bash
npm run sync-teams                                        # the real sync
python tools/sample_team_downloader/sync.py --dry-run     # print the plan, write nothing
python tools/sample_team_downloader/sync.py --accept-count-change   # the post's team count moved
```

## What it guarantees
- **Structural pairing.** Each team's title is read from the line that follows *its own* link
  (`<a href=pokepast><sprites></a><br>Title – by Author<br>`). The previous version zipped a list of
  links with a list of " – by " lines, so one extra link or one missing title silently shifted every
  later team's name, author and category onto the wrong paste.
- **A team link is a pokepast link that shows sprites and is not inside a spoiler/quote.** A
  plain-text pokepast link (a variant in a description) is ignored. A team link with no
  `Title – by Author` line after it is a **refusal** — the post's structure changed.
- **It refuses to touch a manifest it does not own.** Every entry in `sample/teams.json` must have
  exactly the shape this tool writes; a promoted or hand-added entry aborts the sync.
- **The team count is pinned** to the committed manifest's. A post that now links a different number
  of teams fails unless `--accept-count-change` is passed.
- **All-or-nothing.** Every paste is downloaded before anything is written; one failed download aborts
  with the tree untouched. A no-op sync is byte-identical.
- **A paste the thread drops is RETIRED, not deleted.** It moves to `data/teams/superseded/` (its own
  `teams.json`, not in the pool, still a legal exploiter trainee) and `data/teams/relocations.json`
  maps its old path, because archived runs recorded that path and its fingerprint. The 2026-09-23
  sync retired Curse RestLax `0972146213a667c9` (replaced by `04e417ef9822abe9`; the only difference
  is that the old paste spells out Dugtrio's `IVs: 30 SpD / 30 Spe` for Hidden Power Bug, the new one
  leaves Showdown to fill in the default Bug IVs — `30 Atk / 30 Def / 30 SpD`).
- **Named encodings**: overrides `requests`' ISO-8859-1 fallback with `apparent_encoding` and writes
  with an explicit `encoding="utf-8"` (see `tools/CLAUDE.md`).

## Data structure
1. `data/teams/sample/<paste id>.txt` — the raw PokePaste exports.
2. `data/teams/sample/teams.json` — the manifest `utils.team_loader.TeamLoader` loads (`file` is
   relative to `data/`):

```json
{
  "url": "https://pokepast.es/f6229d2c867e21d6",
  "name": "Big 5 + Starmie (Beerlover)",
  "author": "UD",
  "category": "Balance",
  "id": "f6229d2c867e21d6",
  "file": "teams/sample/f6229d2c867e21d6.txt",
  "source": "https://pokepast.es/f6229d2c867e21d6"
}
```

## Tests
`sample_team_downloader_test.py` — offline, on `fixtures/first_post_2026-09-23.html` (the real first
post, scripts stripped) plus synthetic posts: the committed manifest equals the tool's output on the
saved post; a non-team link does not shift titles; a missing title refuses; a foreign manifest entry,
a count change and a failed download each write nothing; a replaced paste is retired with a
relocation.

## Dependencies
`requests`, `beautifulsoup4`, `lxml`.
