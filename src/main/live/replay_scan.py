"""P4 gate (b) — the PUBLIC REPLAY corpus through the RUST reader (the Rust drift scan).

``main.ladder_drift_scan`` feeds public replays through the PYTHON layer (``battle_event.classify``, ``Gen3Battle``,
``gen3_effects``) — a client nobody will run after P6. This scan feeds them through the reader the live client
runs (``main.live.reader`` → ``live_reader`` → ``pokesim::side_reader``), with the SAME line filter the live client
applies (``""`` and the declared :data:`main.live.reader.ROOM_SKIP` dropped, everything else fed — an unknown
keyword is a REFUSAL, never a skip):

* **read** — each replay is read from BOTH seats as a spectator-style chain (``OPEN`` with no team: a public replay
  carries no request and no private line), fed turn by turn; any refusal is a finding, with the replay, the seat
  and the reader's own message (which names the line);
* **encode** — after every turn the seat's current reading is ENCODED (``PROBE``, the drift scan's check only —
  the live path encodes at decisions): an effect, volatile or status the encoder cannot classify surfaces here
  rather than mid-battle.

The corpus is the project's own download (``<main checkout>/replays/showdown/gen3ou/<date>/*.log``, the public
replay archive's gen3ou logs). Run OFFLINE — this scan downloads nothing::

    python -m main.live.replay_scan --corpus <dir> --workers 8 --out <dir> [--limit N] [--seats p1,p2]

Exit 0 = zero refusals and zero encoder failures; 1 = findings (named in ``summary.json``).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from main.live.reader import LiveReader, ReaderRefusal, is_room_line, keyword

#: A message normalized for grouping (digits and quoted payloads folded), so one root cause is one class.
_NORM = [(re.compile(r'"[^"]*"'), '"…"'), (re.compile(r"\d+"), "N")]


def norm(msg: str) -> str:
    out = msg[:240]
    for rx, rep in _NORM:
        out = rx.sub(rep, out)
    return out


def battle_lines(text: str) -> List[str]:
    """The lines the LIVE client would feed (the same filter as ``LiveBattle.on_lines``)."""
    return [ln for ln in text.split("\n") if ln != "" and not is_room_line(ln)]


def turn_chunks(lines: List[str]) -> List[List[str]]:
    """Split at every ``|turn|`` (each chunk ENDS with its turn line), so a refusal is localized to a turn."""
    out: List[List[str]] = [[]]
    for ln in lines:
        out[-1].append(ln)
        if ln.startswith("|turn|"):
            out.append([])
    return [c for c in out if c]


def seat_names(lines: List[str]) -> Dict[str, List[str]]:
    """Every distinct non-empty name each seat's ``|player|`` lines carried, in order of appearance."""
    out: Dict[str, List[str]] = {}
    for ln in lines:
        parts = ln.split("|")
        if ln.startswith("|player|") and len(parts) > 3 and parts[3] and parts[3] not in out.setdefault(parts[2], []):
            out[parts[2]].append(parts[3])
    return out


#: The ONE refusal class a spectator read can meet that live play cannot (root-caused 2026-10-07, P4 gate (b)):
#: the reading's identity is the player NAME (poke-env's ``_player_role`` rule, ``BoardReading::player``), so when the
#: VIEWER's own seat is renamed mid-battle (a guest logging in: ``|player|p2|<new name>|``) the reading flips its
#: role and adds the other side's mons to its own team. A live client's own name never changes inside a battle, and
#: an OPPONENT's rename leaves the role alone (``live_integration_test``). Classified, never hidden.
VIEWER_SEAT_RENAMED = "spectator-only: the viewer's seat was renamed mid-battle (identity is the name)"


def scan_one(reader: LiveReader, path: str, seats: Iterable[str], probe: bool) -> Dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = battle_lines(text)
    all_names = seat_names(lines)
    names = {seat: ns[0] for seat, ns in all_names.items()}
    res: Dict[str, Any] = {"path": path, "lines": len(lines), "refusals": [], "probe_failures": [],
                           "probes": 0, "kw": collections.Counter(keyword(ln) for ln in lines)}
    for seat in seats:
        reader.open(seat, names.get(seat, seat), None)
        try:
            for chunk in turn_chunks(lines):
                reader.feed(chunk)
                if probe:
                    err = reader.probe()
                    res["probes"] += 1
                    if err is not None:
                        res["probe_failures"].append({"seat": seat, "turn": reader.turn, "message": err})
                        break
        except ReaderRefusal as exc:
            rec = {"seat": seat, "kind": exc.kind, "message": str(exc)}
            if len(all_names.get(seat, [])) > 1:
                rec["class"] = VIEWER_SEAT_RENAMED
                rec["names"] = all_names[seat]
            res["refusals"].append(rec)
    return res


_READER: Optional[LiveReader] = None


def _worker(args: Tuple[str, Tuple[str, ...], bool]) -> Dict[str, Any]:
    global _READER
    if _READER is None:
        _READER = LiveReader()
    path, seats, probe = args
    try:
        return scan_one(_READER, path, seats, probe)
    except ReaderRefusal as exc:  # the process itself died: respawn for the next replay, report this one
        _READER.close()
        _READER = None
        return {"path": path, "lines": 0, "refusals": [{"seat": "?", "kind": exc.kind, "message": str(exc)}],
                "probe_failures": [], "probes": 0, "kw": collections.Counter()}


def corpus_files(corpus: str, limit: Optional[int]) -> List[str]:
    files: List[str] = []
    for root, _dirs, names in os.walk(corpus):
        files.extend(os.path.join(root, n) for n in names if n.endswith(".log"))
    files.sort()
    return files[:limit] if limit else files


def default_corpus() -> Optional[str]:
    from utils.git import get_main_repo_root
    try:
        p = Path(get_main_repo_root()) / "replays" / "showdown" / "gen3ou"
    except Exception:  # noqa: BLE001
        return None
    return str(p) if p.is_dir() else None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.live.replay_scan", description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", default=None, help="a directory of replay .log files (default: the main "
                                                    "checkout's replays/showdown/gen3ou)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seats", default="p1,p2")
    ap.add_argument("--no-probe", action="store_true", help="skip the per-turn encoder probe")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    corpus = a.corpus or default_corpus()
    if not corpus:
        print("[replay_scan] no corpus (pass --corpus)", file=sys.stderr)
        return 2
    files = corpus_files(corpus, a.limit)
    seats = tuple(s for s in a.seats.split(",") if s)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    kw: collections.Counter = collections.Counter()
    refusals: collections.Counter = collections.Counter()
    classified: collections.Counter = collections.Counter()
    probe_fail: collections.Counter = collections.Counter()
    examples: Dict[str, Dict[str, Any]] = {}
    n_lines = n_probes = n_clean = 0
    with Pool(a.workers) as pool, open(out / "findings.jsonl", "w") as fh:
        for i, r in enumerate(pool.imap_unordered(_worker, ((f, seats, not a.no_probe) for f in files), chunksize=64)):
            n_lines += r["lines"]
            n_probes += r["probes"]
            kw.update(r["kw"])
            if not r["refusals"] and not r["probe_failures"]:
                n_clean += 1
            for x in r["refusals"]:
                key = f"{x['kind']}: {norm(x['message'])}"
                if x.get("class"):
                    classified[x["class"]] += 1
                    examples.setdefault("classified: " + x["class"], {"path": r["path"], **x})
                    continue
                refusals[key] += 1
                examples.setdefault(key, {"path": r["path"], **x})
            for x in r["probe_failures"]:
                key = norm(x["message"])
                probe_fail[key] += 1
                examples.setdefault("probe: " + key, {"path": r["path"], **x})
            if r["refusals"] or r["probe_failures"]:
                fh.write(json.dumps({k: v for k, v in r.items() if k != "kw"}) + "\n")
            if (i + 1) % 20000 == 0:
                print(f"[replay_scan] {i + 1}/{len(files)} ({time.time() - t0:.0f}s) refusals={sum(refusals.values())} "
                      f"probe_failures={sum(probe_fail.values())}", flush=True)
    from utils.bridge.sim_bridge_bin import resolve_live_reader_bin
    summary = {
        "gate": "P4 (b) public replays through the Rust reader",
        "corpus": corpus, "replays": len(files), "clean_replays": n_clean, "seats": list(seats),
        "protocol_lines": n_lines, "encoder_probes": n_probes,
        "distinct_keywords": len(kw), "keywords": dict(kw.most_common()),
        "refusals": dict(refusals.most_common()), "probe_failures": dict(probe_fail.most_common()),
        "classified_refusals": dict(classified.most_common()),
        "examples": examples, "seconds": round(time.time() - t0, 1),
        "live_reader": resolve_live_reader_bin(),
        "verdict": "PASS" if files and not refusals and not probe_fail else "FAIL",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: summary[k] for k in ("replays", "clean_replays", "protocol_lines", "encoder_probes",
                                              "distinct_keywords", "classified_refusals", "seconds", "verdict")},
                     indent=1))
    for k, v in list(refusals.most_common())[:15]:
        print(f"  REFUSAL x{v}: {k}")
    for k, v in list(probe_fail.most_common())[:15]:
        print(f"  ENCODE  x{v}: {k}")
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
