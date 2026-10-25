# Sample Teams Sync Tool

Ingests the official ADV OU sample teams from the Smogon forum thread into `data/teams/sample/`.

- Downloads the PokePaste links in parallel (`ThreadPoolExecutor`, 3 workers).
- **Ordered zip**: pairs the post's PokePaste links with its `" – by "` title lines in document order, 1 to 1, so
  names, authors and categories stay attached to the right paste; a count mismatch prints a warning and syncs only
  the matched prefix.
- **Named encodings**: overrides `requests`' ISO-8859-1 fallback with `apparent_encoding` and writes with an explicit
  `encoding="utf-8"`, so a UTF-8 nickname is not baked in as mojibake (`tools/CLAUDE.md`).
- Strips forum artifacts (trailing dashes, "by" indicators).

## Usage

From the project root:
```bash
npm run sync-teams
```

🚨 **`data/teams/sample/` has a second writer.** `python -m main.promote_teams` promotes pool teams into the same
folder and appends them to the same manifest, while this sync rebuilds `teams.json` **from the forum thread alone**,
so a re-sync drops every promoted entry. Check for `data/teams/sample/PROMOTION_MANIFEST.json` first and re-promote
from its recorded seed afterwards.

## Output

1. `data/teams/sample/`: the raw PokePaste `.txt` files (ID-based filenames).
2. `data/teams/sample/teams.json`: the index the teams are loaded through (`src/utils/team_loader/`, which discovers
   every `data/teams/**/teams.json`). `file` is relative to `data/`.

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

Tests: `sample_team_downloader_test.py`.
