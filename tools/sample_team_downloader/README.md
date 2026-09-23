# Sample Teams Sync Tool

This tool automates the ingestion of official ADV OU sample teams from the Smogon community into a structured, validated data repository.

## Features
- **High Performance**: Utilizes parallel downloads (`ThreadPoolExecutor(max_workers=3)`) to sync the thread's 32 teams in under 6 seconds.
- **Ordered Zip Extraction**: Zips the post's PokePaste links to its `" – by "` title lines in document order, 1-to-1, so names, authors and categories stay attached to the right paste; a count mismatch prints a warning and syncs only the matched prefix.
- **Rich Metadata**: Generates a `data/teams/sample/teams.json` index with separate fields for `name`, `author`, `category`, and `url`.
- **Named encodings**: Overrides `requests`' ISO-8859-1 fallback with `apparent_encoding` and writes with an explicit `encoding="utf-8"`, so a UTF-8 nickname is not baked in as mojibake (see `tools/CLAUDE.md`).
- **Automatic Cleanup**: Strips forum artifacts (trailing dashes, "by" indicators) for a clean dataset.

## Usage
Run the sync script from the project root:
```bash
npm run sync-teams
```

🚨 **`data/teams/sample/` has a second writer.** `python -m main.promote_teams` promotes pool teams
into the same folder and appends them to the same manifest, while this sync rebuilds
`teams.json` **from the forum thread alone** — so a re-sync drops every promoted entry. Check for
`data/teams/sample/PROMOTION_MANIFEST.json` first and re-promote from its recorded seed afterwards.

## Data Structure
The sync tool populates two locations:
1. `data/teams/sample/`: Contains the raw PokePaste `.txt` files (ID-based filenames).
2. `data/teams/sample/teams.json`: The metadata index this folder's teams are loaded through (`src/utils/team_loader/`, which discovers every `data/teams/**/teams.json`). `file` is relative to `data/`.

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

## Dependencies
- `requests`: HTTP client for forum scraping and PokePaste raw downloads.
- `beautifulsoup4`: HTML parsing and DOM traversal.
- `lxml`: High-speed XML/HTML processing backend.
