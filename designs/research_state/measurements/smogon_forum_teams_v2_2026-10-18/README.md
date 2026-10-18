# Smogon forum teams v2: the mega sweep (IN PROGRESS checkpoint, 2026-10-18)

**Status: CHECKPOINT.** The crawl is still running. This README is rewritten when the crawl is done.
Counts are in `manifest.json` (`counts`); one row per candidate is in `manifest_teams.jsonl`; refs
that could not be read are in `manifest_unread.jsonl`.

v1 (`../smogon_forum_teams_2026-10-18/`, manifest unchanged) is re-read here as the first source
(`origin: v1_register`). The crawl state lives in `crawl/shards/<shard>/{queue,pages,teams}.jsonl`
and is resumable (`scripts/crawl.py`). Pastes are fetched by `scripts/fetch_refs.py`; the set is built
by `scripts/build_v2.py`. Nothing here is under `data/`.
