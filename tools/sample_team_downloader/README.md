# Sample Teams Sync Tool

This tool extracts ADV OU sample teams from the official Smogon thread and saves them as raw `.txt` files for use in the AI environment.

## Source
- Thread: [ADV OU Sample Teams](https://www.smogon.com/forums/threads/adv-ou-sample-teams.3687813/)
- Only teams from the **first post** are extracted.

## Usage
Run the sync script from the project root:
```bash
npm run sync-teams
```
Or directly using the Python venv:
```bash
PYTHONPATH=src deps/venv/bin/python3.11 tools/sample_team_downloader/sync.py
```

## Output
Teams are saved to the `data/teams/` directory. Each file is named based on the team description found in the thread.

## Dependencies
- `requests`: For fetching thread and team data.
- `beautifulsoup4`: For parsing the Smogon forum HTML.
- `lxml`: Fast HTML parser.
