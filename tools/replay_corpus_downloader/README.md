# Replay corpus downloader

Downloads the PUBLIC Pokemon Showdown replay logs of one format into the layout the project's human-replay
readers expect:

```
<out>/<YYYY-MM-DD>/battle-<format>-<id>.log        replays/showdown/gen3ou/2026-05-18/battle-gen3ou-2612838147.log
```

The corpus lives only in the MAIN checkout (`<main>/replays/showdown/gen3ou/`, gitignored, 376k replays at the time
of writing: 109 date folders, 2026-05-18 -> 2026-09-23). It is read by `python -m main.live.replay_scan` (the Rust
reader's bulk drift scan), `main.ladder_drift_scan` (a pre-session check on a fresh download) and the ai_v11
human-replay census (`designs/ai_v11/design_human_replay_objectives.md` §5).

## 🚨 PROVENANCE — this is a RE-WRITE, not the original

The script that produced the existing corpus was **never committed and could not be recovered** (searched
2026-10-08: the home directory, `~/.cache/gen3ai`, `~/dev`, the readable shell history, the P4 handoff, every Claude
session transcript under `~/.claude/projects/`, the Gemini/Antigravity logs, VS Code local history and
`git log --all -S`). What is known about it comes from the **files**, which this tool reproduces:

* names and folders: `battle-<format>-<id>.log` under a `YYYY-MM-DD` folder (the id is the public replay id minus
  the leading `battle-`);
* the date folder is the **America/Los_Angeles calendar date of the battle's START** — the `|t:|` line of the log.
  100 of 100 sampled files agree (32 of them differ from the UTC date), so it is not a UTC folder;
* the content is the raw replay `.log` (`|init|battle`, `|title|`, `|player|` with ratings, …).

What is **not** known: the original's source (this tool uses `replay.pokemonshowdown.com`'s `search.json` and
`<id>.log`), its politeness, or why the corpus has holes (no folders for 2026-06-16..06-20 and 2026-08-03..08-17,
among others). **Nothing here has been run against the public server** — it was tested only against a fake fetcher.
Run it first with `--list-only` and a small `--max-replays`, and read what it writes.

## Usage

```bash
export PYTHONPATH=$PYTHONPATH:src
# a small, polite first run: list, then write 200
python -m tools.replay_corpus_downloader.sync --out <main>/replays/showdown/gen3ou --list-only --max-replays 20
python -m tools.replay_corpus_downloader.sync --out <main>/replays/showdown/gen3ou --since 2026-09-23 --max-replays 200
```

* **Resumable**: every id already under `--out` (any date folder) is skipped; a file is written through a `.part`
  and renamed, so a killed run leaves no torn `.log`.
* **Polite by construction**: one request at a time, `--sleep` (default 1.0 s, below 0.5 s refused), back-off on
  429 / 5xx, a hard stop on a persistent failure, `--max-replays` (default 1000). It identifies itself in the
  `User-Agent`.
* `--since YYYY-MM-DD` ends the newest-first walk at that Los Angeles date.
* Private replays are skipped.

## Before you run it for real

* It is network load on a public server the project does not own: ask first, keep volumes small, and never run it
  from a launcher, a gate or a test (the test file uses a fake fetcher and makes no request).
* A bulk download changes `replays/`, not `data/`; no pinned run reads it. Still, the corpus is the input of banked
  censuses: a re-measurement should name the date range it used.

Tests: `tools/replay_corpus_downloader/replay_corpus_downloader_test.py` (fake server: layout, date convention,
skip/resume, pagination seam, `--since`, `--max-replays`, `--list-only`, failure stop).
