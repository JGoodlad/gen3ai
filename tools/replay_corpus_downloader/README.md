# Replay corpus downloader

Downloads the PUBLIC Pokemon Showdown replay logs of one format into the layout the project's human-replay
readers expect:

```
<out>/<YYYY-MM-DD>/battle-<format>-<id>.log        replays/showdown/gen3ou/2026-05-18/battle-gen3ou-2612838147.log
```

The corpus lives only in the MAIN checkout (`<main>/replays/showdown/gen3ou/`, gitignored): 376,410 spectator logs to 2026-09-23, then this tool's
archive pull, run by the `gen3ai-replay-*` systemd user units (`service.py`, `systemd/`; status, units and how to stop them:
[`designs/tools/replay_corpus.md`](../../designs/tools/replay_corpus.md)). It is read by `python -m main.live.replay_scan` (the Rust
reader's bulk drift scan), `main.ladder_drift_scan` (a pre-session check on a fresh download) and the ai_v11
human-replay census (`designs/ai_v11/design_human_replay_objectives.md` §5).

## 🚨 PROVENANCE — this is a RE-WRITE, and a different mechanism

The existing corpus was produced by `src/main/collect_replays.py`: a LIVE SPECTATOR daemon on poke-env's `BattleSpectator` that saved each public
battle's log when it finished, run by hand in a tmux window. It WAS committed (`77ef01cac`, 2026-05-18) and was deleted with poke-env
(`5d85d8a20`, 2026-10-06): `git show 5d85d8a20^:src/main/collect_replays.py`. (The 2026-10-08 note that it "was never committed" was wrong: the
search missed it because the file was already deleted.) It stopped twice because a reboot killed its tmux window; see
[`designs/tools/replay_corpus.md`](../../designs/tools/replay_corpus.md) for the status, the cause and the restart plan.

This tool is the replacement for the same on-disk layout, from the public replay ARCHIVE instead of a live spectator. Known from the files, which it
reproduces:

* names and folders: `battle-<format>-<id>.log` under a `YYYY-MM-DD` folder (the id is the public replay id minus
  the leading `battle-`);
* the date folder is the **America/Los_Angeles calendar date of the battle's START** — the `|t:|` line of the log
  (100 of 100 sampled files agree; 32 of them differ from the UTC date). The collector used `datetime.now()` at the save, the same date but for a
  battle that crosses midnight;
* the content is the raw replay `.log` (`|init|battle`, `|title|`, `|player|` with ratings, …).

Measured against the spectator files of the same ids (2026-10-20, 50 replays): the archive log equals the spectator log from `|j|` to
`|win|`, lacks its `|init|battle` / `|title|` header and adds post-game lines (`|raw|` rating changes, sometimes `|l|` / `|c|`); the Rust reader
parses it cleanly. The archive holds only UPLOADED replays, about a tenth of what the spectator saw.

## Usage

```bash
export PYTHONPATH=$PYTHONPATH:src
# by hand, into a SCRATCH dir (the corpus itself is the units' job): list, then write 50 of one day
python -m tools.replay_corpus_downloader.sync --out /tmp/scratch --proxy socks5h://127.0.0.1:1080 --list-only --max-replays 20
python -m tools.replay_corpus_downloader.sync --out /tmp/scratch --proxy socks5h://127.0.0.1:1080 --since 2026-09-22 --until 2026-09-23 --max-replays 50
```

* **Resumable**: every id already under `--out` (any date folder) is skipped; a file is written through a `.part`
  and renamed, so a killed run leaves no torn `.log`.
* **Polite by construction**: one request at a time, `--sleep` (default 1.0 s, below 1.0 s refused), back-off on
  429 / 5xx, a hard stop on a persistent failure, `--max-replays` (default 1000), one downloader per machine (an
  `flock`; a second exits 75). It identifies itself in the `User-Agent`.
* **Proxied, fail closed**: `--proxy` (required, `socks5h://` or `socks5://`) carries every request; an unreachable
  proxy exits 3 before any request, and the session ignores `*_PROXY` / `NO_PROXY`. A credential in the URL is never printed.
* `--since YYYY-MM-DD` ends the newest-first walk at that Los Angeles upload date; `--until` (exclusive) starts it there.
* Private replays are skipped.

## Before you run it for real

* It is network load on a public server the project does not own (the owner approved the proxied download on
  2026-10-20, nothing else): keep volumes bounded, and never run it from a launcher, a gate or a test (the tests use
  a fake fetcher and a fake loopback SOCKS5 server and make no outside request).
* A bulk download changes `replays/`, not `data/`; no pinned run reads it (evidence: `designs/tools/replay_corpus.md`). Pass an ABSOLUTE `--out` or run from the main checkout: a relative one resolves against the cwd. Still, the corpus is the input of banked
  censuses: a re-measurement should name the date range it used.

Tests: `replay_corpus_downloader_test.py` (fake server: layout, date convention, skip/resume, pagination seam,
`--since` / `--until`, `--max-replays`, `--list-only`, failure stop), `proxy_test.py` (the proxy and the lock),
`service_test.py` (backfill phases, the daily pull and alarm).

## Is the corpus still growing? (read-only alarm)

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m tools.replay_corpus_downloader.freshness [--out <corpus>] [--max-age-days 3]   # exit 0 fresh, 1 STALE, 2 no corpus
```

It lists the date folders and writes nothing, so it is safe at any time. Tests: `freshness_test.py`.
