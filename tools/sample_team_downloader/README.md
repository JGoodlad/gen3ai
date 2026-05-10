# Sample Teams Sync Tool

This tool automates the ingestion of official ADV OU sample teams from the Smogon community into a structured, validated data repository.

## Features
- **High Performance**: Utilizes parallel downloads (ThreadPoolExecutor) to sync 30+ teams in under 6 seconds.
- **Ordered Zip Extraction**: Implements a robust 1-to-1 matching strategy to ensure 100% accuracy in team names, authors, and categories.
- **Rich Metadata**: Generates a centralized `data/teams/teams.json` index with separate fields for `name`, `author`, `category`, and `url`.
- **Automatic Cleanup**: Strips forum artifacts (trailing dashes, "by" indicators) for a clean dataset.

## Usage
Run the sync script from the project root:
```bash
npm run sync-teams
```

## Data Structure
The sync tool populates two locations:
1. `data/teams/`: Contains the raw PokePaste `.txt` files (ID-based filenames).
2. `data/teams/teams.json`: The central metadata index used by the application.

```json
{
  "id": "f6229d2c867e21d6",
  "name": "Big 5 + Starmie (Beerlover)",
  "author": "UD",
  "category": "Balance",
  "url": "https://pokepast.es/f6229d2c867e21d6",
  "file": "teams/f6229d2c867e21d6.txt"
}
```

## Dependencies
- `requests`: HTTP client for forum scraping and PokePaste raw downloads.
- `beautifulsoup4`: HTML parsing and DOM traversal.
- `lxml`: High-speed XML/HTML processing backend.
