"""Protocol-drift gate for LIVE-LADDER play: parse real gen3ou replays from the public
replay archive through our own ``Gen3Battle``.

Why this exists
---------------
Every battle we have ever parsed came from a Showdown pinned in ``deps/pokemon-showdown``
(currently a 2026-05 commit). The public server runs current master, and
``agents.battle.battle_event.classify`` raises on any keyword it does not know BY DESIGN
— a tripwire that is correct for a closed local sim and fatal on an open one: the raise
kills the parse task, no choice is ever sent, and the battle is lost on the timer.

So before a first rated game, MEASURE the drift instead of arguing about it. The replay
archive is a public read-only HTTP endpoint — no account, no websocket, no rules
exposure — and its logs are the same protocol stream a live battle room carries, minus
the ``|request|`` frames (which the Player layer consumes before a battle ever sees them,
and which this scan therefore skips exactly as the Player does).

Two levels of check, both run:

* **keyword** — every ``|<kw>|`` classified; an unclassified or ``UNSUPPORTED`` one is
  the fatal case;
* **structural** — the whole log replayed into a real ``Gen3Battle``, which catches an
  argument-SHAPE change that a keyword census cannot see (a new positional field, a
  ``[from]`` form we do not strip).

Run::

    python src/main/ladder_drift_scan.py --n 60
    python src/main/ladder_drift_scan.py --n 200 --format gen3ou --cache /tmp/psreplays

Exit 0 = clean, exit 1 = drift found (with the offending keywords / tracebacks named).

Measured 2026-08-23: **59 replays, 22 794 protocol lines, 56 distinct keywords, ZERO
unknown, ZERO unsupported, 59/59 structurally clean.** Re-run it before going live —
that is a reading of one day's ladder, not a proof about every future one.

(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""

import argparse
import collections
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from typing import List

SEARCH_URL = "https://replay.pokemonshowdown.com/search.json"
LOG_URL = "https://replay.pokemonshowdown.com/{id}.log"
USER_AGENT = "gen3ai-ladder-drift-scan"

# Consumed by `Player._handle_battle_message` BEFORE the battle sees them, so a battle
# parser is never asked about them and this scan must not ask either.
HANDLED_BY_PLAYER = frozenset(
    {"t:", "expire", "uhtmlchange", "request", "showteam", "win", "tie", "error", "bigerror"}
)


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


def split_lines(path: str):
    """Yield the split protocol lines a Player would hand to `battle.parse_message`."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh.read().split("\n"):
            if not raw:
                continue
            parts = (raw if raw.startswith("|") else "|" + raw).split("|")
            if len(parts) < 2 or parts[1] in HANDLED_BY_PLAYER:
                continue
            yield parts


def scan(paths: List[str]) -> int:
    from agents.battle.battle_event import (
        UnknownMessageType,
        UnsupportedMessageType,
        classify,
    )
    from agents.battle.gen3_battle import Gen3Battle

    logging.disable(logging.CRITICAL)  # replays are noisy; we only care about raises
    quiet = logging.getLogger("ladder_drift_scan")

    kinds: collections.Counter = collections.Counter()
    unknown: collections.Counter = collections.Counter()
    unsupported: collections.Counter = collections.Counter()
    structural: collections.Counter = collections.Counter()
    example: dict = {}
    clean = 0
    total_lines = 0

    for path in paths:
        for parts in split_lines(path):
            kw = parts[1]
            kinds[kw] += 1
            total_lines += 1
            try:
                classify(kw)
            except UnknownMessageType:
                unknown[kw] += 1
            except UnsupportedMessageType:
                unsupported[kw] += 1

        tag = "battle-drift-" + os.path.basename(path).rsplit(".", 1)[0]
        battle = Gen3Battle(tag, "p1", quiet, gen=3)
        battle._player_role = "p1"
        try:
            for parts in split_lines(path):
                battle.parse_message(parts)
            clean += 1
        except Exception as exc:  # noqa: BLE001 — every failure is a finding, not a crash
            key = f"{type(exc).__name__}: {str(exc)[:100]}"
            structural[key] += 1
            example.setdefault(key, (path, traceback.format_exc()))

    print(f"[drift] replays={len(paths)}  protocol_lines={total_lines}  "
          f"distinct_keywords={len(kinds)}")
    print(f"[drift] keyword census: {dict(kinds.most_common())}")
    print(f"[drift] structurally clean: {clean}/{len(paths)}")

    bad = False
    if unknown:
        bad = True
        print(f"\n[drift] ✗ UNCLASSIFIED keywords (would raise UnknownMessageType and "
              f"WEDGE the battle): {dict(unknown)}")
        print("[drift]   fix: classify each in agents/battle/battle_event.MESSAGE_POLICY, "
              "and — if it is not battle content — add it to "
              "poke_env.battle.abstract_battle.AbstractBattle.MESSAGES_TO_IGNORE too.")
    if unsupported:
        bad = True
        print(f"\n[drift] ✗ UNSUPPORTED keywords seen in a gen3 game: {dict(unsupported)}")
    if structural:
        bad = True
        print("\n[drift] ✗ structural parse failures:")
        for key, count in structural.most_common(10):
            path, tb = example[key]
            print(f"  [{count}x] {key}\n     first: {path}")
            for line in tb.splitlines()[-4:]:
                print("     ", line.strip()[:160])
    if not bad:
        print("\n[drift] ✓ no drift: every keyword classified, every replay parsed clean.")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--format", default="gen3ou", help="Showdown format id")
    ap.add_argument("--n", type=int, default=60, help="how many replays to scan")
    ap.add_argument("--cache", default="/tmp/psreplays",
                    help="where downloaded .log files live (re-used across runs)")
    ap.add_argument("--offline", action="store_true",
                    help="scan whatever is already in --cache, download nothing")
    args = ap.parse_args()

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
    return scan(paths)


if __name__ == "__main__":
    sys.exit(main())
