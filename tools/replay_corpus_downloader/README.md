# Replay corpus downloader

Downloads the PUBLIC Pokemon Showdown replay logs of one format from the replay archive into the layout the
project's human-replay readers expect:

```
<out>/<YYYY-MM-DD>/battle-<format>-<id>.log        replays/showdown/gen3ou/2026-05-18/battle-gen3ou-2612838147.log
```

The corpus lives only in the MAIN checkout (`<main>/replays/showdown/gen3ou/`, gitignored). It grows through the
`gen3ai-replay-*` systemd user units (`service.py`, `systemd/`); the status, the units, how to stop them, and how the
archive differs from the earlier live-spectator files are in [`designs/tools/replay_corpus.md`](../../designs/tools/replay_corpus.md).
Readers: `python -m main.live.replay_scan` (the Rust reader's bulk drift scan) and `main.ladder_drift_scan`.

Layout facts the tool reproduces: the id in the name is the public replay id minus the leading `battle-`; the date
folder is the **America/Los_Angeles calendar date of the battle's START** (the `|t:|` line), not the UTC date or the
download date; the content is the raw replay `.log`. The archive holds only UPLOADED replays.

## Usage

```bash
export PYTHONPATH=$PYTHONPATH:src
# by hand, into a SCRATCH dir (the corpus itself is the units' job): list, then write 50 of one day
python -m tools.replay_corpus_downloader.sync --out /tmp/scratch --proxy socks5h://127.0.0.1:1080 --list-only --max-replays 20
python -m tools.replay_corpus_downloader.sync --out /tmp/scratch --proxy socks5h://127.0.0.1:1080 --since 2026-09-22 --until 2026-09-23 --max-replays 50
```

* **Resumable**: every id already under `--out` (any date folder) is skipped; a file is written through a `.part`
  and renamed, so a killed run leaves no torn `.log`. Private replays are skipped.
* **Polite by construction**: one request at a time, `--sleep` (below 1.0 s refused), back-off on 429 / 5xx, a hard
  stop on a persistent failure, `--max-replays` (default 1000), one downloader per machine (an `flock`; a second
  exits 75). It identifies itself in the `User-Agent`.
* **Proxied, fail closed**: `--proxy` (required, `socks5h://` or `socks5://`) carries every request; an unreachable
  proxy exits 3 before any request, and the session ignores `*_PROXY` / `NO_PROXY`. A credential in the URL is never printed.
* `--since YYYY-MM-DD` ends the newest-first walk at that Los Angeles upload date; `--until` (exclusive) starts it there.

## Before you run it for real

* It is network load on a public server the project does not own (the owner approved the proxied download and
  nothing else): keep volumes bounded, and never run it from a launcher, a gate or a test (the tests use a fake
  fetcher and a fake loopback SOCKS5 server and make no outside request).
* A bulk download changes `replays/`, not `data/`. Pass an ABSOLUTE `--out` or run from the main checkout: a relative
  one resolves against the cwd. The corpus is the input of banked censuses, so a re-measurement should name the date
  range it used.

## Is the corpus still growing? (read-only alarm)

```bash
python -m tools.replay_corpus_downloader.freshness [--out <corpus>] [--max-age-days N]   # exit 0 fresh, 1 STALE, 2 no corpus
```

It lists the date folders and writes nothing, so it is safe at any time.

Tests: `replay_corpus_downloader_test.py`, `proxy_test.py`, `service_test.py`, `freshness_test.py` (all offline).
