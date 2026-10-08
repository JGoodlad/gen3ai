"""Protocol-drift gate for LIVE play: parse real gen3ou replays from the public replay archive through the reader
the live client runs.

Why this exists
---------------
Every battle we have ever parsed came from a Showdown pinned in ``deps/pokemon-showdown``. The public server runs
current master, and the live reader REFUSES any keyword it does not know BY DESIGN (``core_events::LineError::
UnknownKeyword``) — a tripwire that is correct for a closed local sim and fatal on an open one: the refusal is a T28
halt (``main.live.halt``), and the game is lost on the timer.

So before a live session, MEASURE the drift instead of arguing about it. The replay archive is a public read-only HTTP
endpoint — no account, no websocket, no rules exposure — and its logs are the same protocol stream a live battle room
carries, minus the ``|request|`` frames.

Since P6 of the poke-env retirement (2026-10-08) every check runs on the RUST stack (the Python battle layer it used
to replay through — ``battle_event.classify``, ``Gen3Battle``, ``gen3_effects`` — is the client nobody runs). Four
checks, all run:

* **read** (keyword + structural) — each replay is read from BOTH seats by the live reader (``main.live.replay_scan``'s
  ``scan_one``: ``live_reader`` → ``pokesim::side_reader``, the live client's own line filter): an unknown keyword or
  an argument SHAPE the chain cannot fold is a refusal, named with the line;
* **encoder (replayed)** — after every turn the seat's reading is ENCODED (``PROBE``): an effect, volatile or status
  the encoder cannot classify surfaces here rather than mid-battle;
* **encoder (source)** — replays only show what a day's games happened to do, so the effect lines are ALSO derived
  from the Showdown source the public server runs (``agents.observation.gen3_effect_sources``' text scan) and each is
  read + encoded by the same reader (``main.live.effect_scan``). ``--showdown DIR`` names a checkout; by default a
  sparse shallow clone of master is kept under ``--cache``. ``--no-effects`` skips it (offline);
* **format spec** — the same master checkout's gen3ou entry, gen-3 ``Standard``, Uber tier and clause bodies against
  ``agents.gen3_data.format_spec`` (``main.format_drift``): a ban or clause the ladder added or dropped since the
  spec was written FAILS (``--no-format-spec`` skips it).

Run::

    python src/main/ladder_drift_scan.py --n 60
    python src/main/ladder_drift_scan.py --n 200 --format gen3ou --cache /tmp/psreplays

Exit 0 = clean, exit 1 = drift found (with the offending lines named). The bulk corpus read (P4 gate (b), 376,410
replays) is ``python -m main.live.replay_scan``; this script is the pre-session check on a fresh download.

(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""

import argparse
import collections
import json
import os
import subprocess
import sys
import time
from typing import List

SEARCH_URL = "https://replay.pokemonshowdown.com/search.json"
LOG_URL = "https://replay.pokemonshowdown.com/{id}.log"
USER_AGENT = "gen3ai-ladder-drift-scan"

def _fetch(url: str, timeout: int = 30) -> str:
    """GET via curl. Deliberately not `requests`/`urllib`: the archive 403s a bare
    urllib User-Agent, and curl is the one HTTP client this repo can assume."""
    proc = subprocess.run(
        ["curl", "-sS", "--max-time", str(timeout), "-A", USER_AGENT, url],
        capture_output=True, text=True, check=False,
    )
    return proc.stdout


def fetch_replay_ids(format_id: str, n: int, pause: float = 0.4) -> List[str]:
    ids: List[str] = []
    page = 1
    while len(ids) < n and page <= 20:
        body = _fetch(f"{SEARCH_URL}?format={format_id}&page={page}")
        try:
            rows = json.loads(body)
        except ValueError:
            print(f"[drift] search page {page} was not JSON: {body[:120]!r}", file=sys.stderr)
            break
        if not rows:
            break
        ids.extend(r["id"] for r in rows)
        page += 1
        time.sleep(pause)
    return ids[:n]


def download_logs(ids: List[str], cache_dir: str, pause: float = 0.25) -> List[str]:
    os.makedirs(cache_dir, exist_ok=True)
    paths = []
    for rid in ids:
        path = os.path.join(cache_dir, f"{rid}.log")
        if os.path.exists(path) and os.path.getsize(path) > 500:
            paths.append(path)
            continue
        body = _fetch(LOG_URL.format(id=rid))
        if body and "|player|" in body:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(body)
            paths.append(path)
        time.sleep(pause)
    return paths


SHOWDOWN_GIT = "https://github.com/smogon/pokemon-showdown.git"


def fetch_showdown_master(dest: str) -> str:
    """A sparse, shallow checkout of Showdown master (sim/, data/, config/) at ``dest`` — the
    code the PUBLIC server runs — refreshed if it already exists. Returns the resolved commit."""
    if os.path.isdir(os.path.join(dest, ".git")):
        subprocess.run(["git", "-C", dest, "pull", "-q", "--depth", "1"], check=True)
    else:
        subprocess.run(["git", "clone", "-q", "--depth", "1", "--filter=blob:none", "--sparse",
                        SHOWDOWN_GIT, dest], check=True)
        subprocess.run(["git", "-C", dest, "sparse-checkout", "set", "sim", "data", "config"],
                       check=True)
    return subprocess.run(["git", "-C", dest, "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()


def effects_source_check(showdown_root: str) -> int:
    """The ENCODER (source) check, on the Rust reader (``main.live.effect_scan``)."""
    from main.live import effect_scan

    return effect_scan.check(showdown_root)


def scan(paths: List[str]) -> int:
    """The READ + ENCODER (replayed) checks: every replay, both seats, through the live reader (refusals and
    per-turn encode failures are findings)."""
    from main.live.reader import LiveReader, ReaderRefusal
    from main.live.replay_scan import norm, scan_one

    kinds: collections.Counter = collections.Counter()
    refusals: collections.Counter = collections.Counter()
    classified: collections.Counter = collections.Counter()
    encoder: collections.Counter = collections.Counter()
    example: dict = {}
    clean = probes = total_lines = 0
    reader = LiveReader()
    try:
        for path in paths:
            try:
                r = scan_one(reader, path, ("p1", "p2"), True)
            except ReaderRefusal as exc:  # the reader process itself died: a finding, and a fresh reader
                r = {"lines": 0, "kw": collections.Counter(), "probes": 0, "probe_failures": [],
                     "refusals": [{"seat": "?", "kind": exc.kind, "message": str(exc)}]}
                reader.close()
                reader = LiveReader()
            total_lines += r["lines"]
            probes += r["probes"]
            kinds.update(r["kw"])
            for x in r["refusals"]:
                if x.get("class"):  # the one spectator-only refusal class (replay_scan.VIEWER_SEAT_RENAMED)
                    classified[x["class"]] += 1
                    continue
                key = f"{x['kind']}: {norm(x['message'])}"
                refusals[key] += 1
                example.setdefault(key, (path, x["seat"], x["message"]))
            for x in r["probe_failures"]:
                key = norm(x["message"])
                encoder[key] += 1
                example.setdefault(key, (path, x["seat"], f"turn {x['turn']}: {x['message']}"))
            if not r["refusals"] and not r["probe_failures"]:
                clean += 1
    finally:
        reader.close()

    print(f"[drift] replays={len(paths)}  protocol_lines={total_lines}  distinct_keywords={len(kinds)}")
    print(f"[drift] keyword census: {dict(kinds.most_common())}")
    print(f"[drift] read clean (both seats): {clean}/{len(paths)}"
          + (f"; spectator-only refusals (classified): {dict(classified)}" if classified else ""))
    print(f"[drift] encoder (replayed): {probes} per-turn encodes checked")
    bad = False
    if refusals:
        bad = True
        print("\n[drift] ✗ the live reader REFUSED (an unknown keyword or an argument shape it cannot fold — the live "
              "client would halt):")
        for key, count in refusals.most_common(10):
            path, seat, msg = example[key]
            print(f"  [{count}x] {key}\n     first: {path} ({seat}): {msg[:200]}")
        print("[drift]   fix: classify the keyword in src/rust_sim/src/core_events/schema.rs (Rust-owned, P1) / fold "
              "the shape, with a revert-failing test.")
    if encoder:
        bad = True
        print("\n[drift] ✗ encoder (replayed) failures — the live encode would RAISE here:")
        for key, count in encoder.most_common(10):
            path, seat, msg = example[key]
            print(f"  [{count}x] {key}\n     first: {path} ({seat}): {msg[:200]}")
    if not bad:
        print("\n[drift] ✓ no drift: every replay read clean from both seats, every turn encoded.")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--format", default="gen3ou", help="Showdown format id")
    ap.add_argument("--n", type=int, default=60, help="how many replays to scan")
    ap.add_argument("--cache", default="/tmp/psreplays",
                    help="where downloaded .log files live (re-used across runs)")
    ap.add_argument("--offline", action="store_true",
                    help="scan whatever is already in --cache, download nothing")
    ap.add_argument("--showdown", default=None,
                    help="Showdown checkout for the encoder (source) check (default: a sparse "
                         "shallow clone of master under <--cache>/showdown-master)")
    ap.add_argument("--no-effects", action="store_true",
                    help="skip the encoder (source) check")
    ap.add_argument("--no-format-spec", action="store_true",
                    help="skip the FORMAT-SPEC check (gen3ou's bans / clauses vs agents.gen3_data.format_spec)")
    args = ap.parse_args()

    effects_rc = format_rc = 0
    if not (args.no_effects and args.no_format_spec):
        root = args.showdown
        if root is None:
            root = os.path.join(args.cache, "showdown-master")
            if args.offline and not os.path.isdir(root):
                print("[drift] --offline and no cached Showdown master: pass --showdown or "
                      "--no-effects --no-format-spec", file=sys.stderr)
                return 2
            if not args.offline:
                print(f"[drift] Showdown master @ {fetch_showdown_master(root)}")
        if not args.no_effects:
            effects_rc = effects_source_check(root)
        if not args.no_format_spec:
            # the FORMAT-SPEC check: a new / removed ban or clause on the ladder's gen3ou FAILS
            # (design_format_spec.md §7; offline twin: `python -m main.format_drift check`)
            from pathlib import Path

            from main import format_drift
            format_rc = format_drift.check(Path(root), label=f"Showdown at {root}")
    effects_rc = max(effects_rc, format_rc)

    if args.offline:
        paths = sorted(
            os.path.join(args.cache, f)
            for f in os.listdir(args.cache) if f.endswith(".log")
        )[: args.n]
    else:
        ids = fetch_replay_ids(args.format, args.n)
        print(f"[drift] archive returned {len(ids)} replay ids for {args.format}")
        paths = download_logs(ids, args.cache)
    if not paths:
        print("[drift] no replays to scan (network blocked? empty cache?)", file=sys.stderr)
        return 2
    return max(scan(paths), effects_rc)


if __name__ == "__main__":
    sys.exit(main())
