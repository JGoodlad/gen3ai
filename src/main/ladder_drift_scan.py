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

Four checks, all run:

* **keyword** — every ``|<kw>|`` classified; an unclassified or ``UNSUPPORTED`` one is
  the fatal case;
* **structural** — the whole log replayed into a real ``Gen3Battle``, which catches an
  argument-SHAPE change that a keyword census cannot see (a new positional field, a
  ``[from]`` form we do not strip);
* **encoder (replayed)** — a known keyword can still carry an effect the OBSERVATION
  ENCODER has never classified (``-activate|…|move: Heal Bell`` crashed the encode on
  2026-09-24 with every keyword known). After every line that can put an effect on a mon
  (``-start`` / ``-activate`` / ``-singleturn`` / ``-singlemove`` / ``move`` / ``-prepare``)
  each mon's volatiles go through ``gen3_effects.encode_volatiles``, and every ``|cant|``
  reason through ``normalize_cant_reason`` — the two crash-don't-drop tables;
* **encoder (source)** — replays only show what a day's games happened to do, so the
  effect-id class is ALSO derived from the Showdown source the public server runs
  (``gen3_effect_sources``: every ``add('-start'|'-activate'|'-singleturn'|'-singlemove', …)``
  the gen3 format executes, each executed on a real ``Gen3Battle``) and every id is required
  to be classified. ``--showdown DIR`` names a checkout; by default a sparse shallow clone of
  master is kept under ``--cache``. ``--no-effects`` skips it (offline).

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


SHOWDOWN_GIT = "https://github.com/smogon/pokemon-showdown.git"
#: The lines after which a mon's effects can have GROWN (poke-env's start_effect call sites).
EFFECT_KEYWORDS = frozenset({"-start", "-activate", "-singleturn", "-singlemove", "move", "-prepare"})


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
    """The ENCODER (source) check: derive every effect id the gen3 sim at ``showdown_root``
    can announce onto a mon, and require each to be classified by ``gen3_effects``."""
    from pathlib import Path

    from agents.observation import gen3_effect_sources as S

    root = Path(showdown_root)
    try:
        chain = S.mod_chain(root)
        if chain != S.GEN3_MOD_CHAIN:
            print(f"[drift] ✗ gen3's mod chain changed: {chain} (the scan walks "
                  f"{S.GEN3_MOD_CHAIN}) — update gen3_effect_sources.GEN3_MOD_CHAIN")
            return 1
        derived = S.derive_encoder_ids(root)
    except S.UnresolvedDynamicEffect as exc:
        print(f"[drift] ✗ encoder (source): {exc}")
        return 1
    bad = S.unclassified(derived)
    pending = sorted({eff for kw, eff, _ in derived.get("unknown", [])
                      if (kw, eff) in S.PENDING_OWNER_LINES})
    print(f"[drift] encoder (source): {len(derived)} effect ids derived from {root}"
          + (f"; owner-pending (still RAISE): {pending}" if pending else ""))
    if bad:
        print("[drift] ✗ UNCLASSIFIED effect ids (encode_volatiles would RAISE mid-battle):")
        for vid, srcs in sorted(bad.items()):
            print(f"     {vid}: {srcs[:3]}")
        print("[drift]   fix: classify each in agents/observation/gen3_effects.py (a slot, or "
              "NOT_A_VOLATILE with where its information lives).")
        return 1
    print("[drift] ✓ encoder (source): every derived effect id is classified.")
    return 0


def scan(paths: List[str]) -> int:
    from agents.battle.battle_event import (
        UnknownMessageType,
        UnsupportedMessageType,
        classify,
    )
    from agents.battle.gen3_battle import Gen3Battle
    from agents.battle.live_view import _id
    from agents.observation.gen3_effects import (
        UnknownCantReasonError,
        UnknownVolatileError,
        encode_volatiles,
        normalize_cant_reason,
    )

    logging.disable(logging.CRITICAL)  # replays are noisy; we only care about raises
    quiet = logging.getLogger("ladder_drift_scan")

    kinds: collections.Counter = collections.Counter()
    unknown: collections.Counter = collections.Counter()
    unsupported: collections.Counter = collections.Counter()
    structural: collections.Counter = collections.Counter()
    encoder: collections.Counter = collections.Counter()
    encoder_checks = 0
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
                # the ENCODER (replayed) check — separate from the structural one, so an encode
                # failure is named as such and does not stop the replay
                try:
                    if parts[1] == "cant" and len(parts) > 3:
                        encoder_checks += 1
                        normalize_cant_reason(parts[3])
                    elif parts[1] in EFFECT_KEYWORDS:
                        for mon in (*battle.team.values(), *battle.opponent_team.values()):
                            if mon.effects:
                                encoder_checks += 1
                                encode_volatiles([_id(e) for e in mon.effects])
                except (UnknownVolatileError, UnknownCantReasonError) as exc:
                    key = f"{type(exc).__name__}: {str(exc)[:100]}"
                    encoder[key] += 1
                    example.setdefault(key, (path, "|".join(parts)))
            clean += 1
        except Exception as exc:  # noqa: BLE001 — every failure is a finding, not a crash
            key = f"{type(exc).__name__}: {str(exc)[:100]}"
            structural[key] += 1
            example.setdefault(key, (path, traceback.format_exc()))

    print(f"[drift] replays={len(paths)}  protocol_lines={total_lines}  "
          f"distinct_keywords={len(kinds)}")
    print(f"[drift] keyword census: {dict(kinds.most_common())}")
    print(f"[drift] structurally clean: {clean}/{len(paths)}")
    print(f"[drift] encoder (replayed): {encoder_checks} volatile/cant encodes checked")

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
    if encoder:
        bad = True
        print("\n[drift] ✗ encoder (replayed) failures — the obs encode would RAISE here:")
        for key, count in encoder.most_common(10):
            path, line = example[key]
            print(f"  [{count}x] {key}\n     first: {path}  at  {line[:160]}")
    if not bad:
        print("\n[drift] ✓ no drift: every keyword classified, every replay parsed clean, "
              "every replayed effect and cant reason encoded.")
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
    args = ap.parse_args()

    effects_rc = 0
    if not args.no_effects:
        root = args.showdown
        if root is None:
            root = os.path.join(args.cache, "showdown-master")
            if args.offline and not os.path.isdir(root):
                print("[drift] --offline and no cached Showdown master: pass --showdown or "
                      "--no-effects", file=sys.stderr)
                return 2
            if not args.offline:
                print(f"[drift] Showdown master @ {fetch_showdown_master(root)}")
        effects_rc = effects_source_check(root)

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
