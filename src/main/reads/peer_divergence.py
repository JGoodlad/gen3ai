"""AM-DL-PEER's classifier, in ``src`` (the deviation-only live read's amendment,
``designs/research_state/measurements/dlive_read_2026-10-28/PREREG.md``; its frozen script
``scripts/peer_divergence.py`` is the record and is never imported by a tool): did a replay of a deal leave the other
game of that deal because the PEER (Kakuna) chose differently on an identical view, before a named turn?

FINDING DL-7 (the dlive record): Kakuna in a lone one-game replay sometimes chooses differently from the same game
inside its 20-game unit, on a byte-identical view. A read that pairs two games of one deal (``searched_play``'s
G-PREFIX mismatches, :mod:`main.reads.rules_searched_play`) classifies each divergence with this module.

It reads two front-end captures (``<cell>/capture/<battle>.json``: the ``START`` document, the ``commands`` both sides
sent, the ``chunks`` each side was relayed) and NEVER a result: every ``|win|`` / ``|tie`` line is removed before any
comparison, and no game row or summary is opened.

The test, per side s (ours / the peer's, named from the START document: ours ``Gen3AIAnchor*``, the peer
``MetaKakuna*``):

* ``j_s``: the first relayed chunk of s's stream that differs between the two games. ``k_s``: the number of s's choice
  requests (``|request|`` without ``"wait"``) in s's stream BEFORE ``j_s``: the choices s made on a view identical in
  both games.
* s DIVERGED ON AN IDENTICAL VIEW iff s's first ``k_s`` commands (s's ``CHOOSE`` lines, in order) are not all equal.
  Its divergence turn is the last ``|turn|N`` of s's stream before the request its first differing command answered.

Classes (:func:`classify`, the amendment's conditions (a)-(c) with branch turn ``T``):

* ``identical``: neither stream nor either side's commands differ;
* ``peer_first``: the PEER diverged on an identical view (a), OUR first ``k_ours`` commands are all equal (b), and the
  peer's divergence turn is below ``T`` (c);
* ``peer_first_late``: (a) and (b) hold, (c) does not;
* ``ours_first``: our side diverged on an identical view;
* ``unattributable``: the streams differ and neither side chose differently on an identical view, or the capture
  cannot be read this way.

:func:`is_peer_drop` is the dlive read's drop rule (a BRANCH needs ``peer_first``; a NULL ``peer_first`` or
``peer_first_late``). Identical in behaviour to the dlive script (``peer_divergence_test`` runs both on the same
captures).

THE OPERATOR'S LOOK (read-only, outcome-blind): when a searched read's lane HOLDS on a G-PREFIX mismatch,

    python -m main.reads.peer_divergence prefix --state DIR --group G [--spec SPEC]

prints one JSON line per planned pair of the group whose two captures differ before the search's first turn: the deal,
the two units, the class, the peer's divergence turn, whether our side diverged, and whether it is a peer drop
(:func:`classify_pair`, the very function ``rules_searched_play`` classifies with). It reads the plans and the
captures only: never a ``games.jsonl``, a summary or a result line, and it prints no protocol line.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

OURS, PEER = "Gen3AIAnchor", "MetaKakuna"
RESULT_PREFIXES = ("|win|", "|tie")
CLASSES = ("identical", "peer_first", "peer_first_late", "ours_first", "unattributable")


def _clean(text: str) -> str:
    return "\n".join(x for x in str(text).split("\n") if not x.startswith(RESULT_PREFIXES))


def sides(cap: Mapping[str, Any]) -> Dict[str, str]:
    """``{"ours": "p1"|"p2", "peer": ...}`` from the START document's player names."""
    start = cap["commands"][0]
    if not start.startswith("START "):
        raise ValueError("the capture's first command is not START")
    doc = json.loads(start[len("START "):])
    names = {p: str(doc[p]["name"]) for p in ("p1", "p2")}
    ours = [p for p, n in names.items() if n.startswith(OURS)]
    peer = [p for p, n in names.items() if n.startswith(PEER)]
    if len(ours) != 1 or len(peer) != 1 or ours == peer:
        raise ValueError(f"cannot name the sides from {names}")
    return {"ours": ours[0], "peer": peer[0]}


def _stream(cap: Mapping[str, Any], side: str) -> List[str]:
    return [_clean(c[1]) for c in cap["chunks"] if c[0] == side]


def _cmds(cap: Mapping[str, Any], side: str) -> List[str]:
    pre = f"CHOOSE {side} "
    return [c[len(pre):] for c in cap["commands"][1:] if c.startswith(pre)]


def _is_choice(chunk: str) -> bool:
    if not chunk.startswith("|request|"):
        return False
    try:
        req = json.loads(chunk[len("|request|"):])
    except ValueError:
        return False
    return not req.get("wait")


def _last_turn(chunks: Sequence[str]) -> int:
    t = 0
    for c in chunks:
        for line in c.split("\n"):
            if line.startswith("|turn|"):
                t = int(line.split("|")[2])
    return t


def _side_view(raw: Mapping[str, Any], br: Mapping[str, Any], side: str) -> Dict[str, Any]:
    a, b = _stream(raw, side), _stream(br, side)
    j = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
    if j is None and len(a) != len(b):
        j = min(len(a), len(b))
    prefix = a if j is None else a[:j]
    req_at = [i for i, c in enumerate(prefix) if _is_choice(c)]
    k = len(req_at)
    ca, cb = _cmds(raw, side), _cmds(br, side)
    m = next((i for i in range(min(k, len(ca), len(cb))) if ca[i] != cb[i]), None)
    if m is None and k > min(len(ca), len(cb)):
        m = min(len(ca), len(cb))                     # a side that answered fewer requests on an identical view
    turn = _last_turn(prefix[:req_at[m]]) if m is not None and m < k else None
    return {"stream_diff": j, "choices_on_identical_view": k, "first_diff_command": m, "diverged": m is not None,
            "divergence_turn": turn, "commands_equal": ca == cb}


def classify(raw: Mapping[str, Any], br: Mapping[str, Any], branch_turn: int) -> Dict[str, Any]:
    """The class of the pair of captures ``raw`` / ``br`` (one deal) against branch turn ``branch_turn`` (module
    docstring), with each side's view."""
    try:
        sd = sides(raw)
        if sides(br) != sd or raw["commands"][0] != br["commands"][0]:
            return {"class": "unattributable", "why": "the START documents differ"}
        ours, peer = _side_view(raw, br, sd["ours"]), _side_view(raw, br, sd["peer"])
    except (KeyError, ValueError, IndexError, TypeError) as exc:
        return {"class": "unattributable", "why": f"{type(exc).__name__}: {exc}"}
    out: Dict[str, Any] = {"sides": sd, "ours": ours, "peer": peer, "branch_turn": int(branch_turn)}
    if (ours["stream_diff"] is None and peer["stream_diff"] is None and ours["commands_equal"]
            and peer["commands_equal"]):
        out["class"] = "identical"
    elif peer["diverged"] and not ours["diverged"]:
        late = peer["divergence_turn"] is None or peer["divergence_turn"] >= int(branch_turn)
        out["class"] = "peer_first_late" if late else "peer_first"
    elif ours["diverged"]:
        out["class"] = "ours_first"
    else:
        out["class"] = "unattributable"
    return out


def is_peer_drop(kind: str, cls: str) -> bool:
    """AM-DL-PEER rules 1 and 3 (with the 2026-10-31 clarification): a branch needs (a)-(c), a null (a)-(b)."""
    if kind == "branch":
        return cls == "peer_first"
    if kind == "null":
        return cls in ("peer_first", "peer_first_late")
    raise ValueError(f"unit kind {kind!r}")


# ------------------------------------------------------------------------------------------------ a searched read's pairs
def classify_pair(raw_cell: Path, searched_cell: Path, from_turn: int) -> Dict[str, Any]:
    """One deal's two per-game cells classified (raw first, branch turn ``from_turn``): ``{class, class_why,
    peer_divergence_turn, ours_diverged, peer_drop}``; an unreadable capture is class ``unreadable``."""
    from main.reads.prefix import PrefixError, capture_of

    try:
        c = classify(capture_of(Path(raw_cell)), capture_of(Path(searched_cell)), int(from_turn))
        row: Dict[str, Any] = {"class": c["class"], "class_why": c.get("why"),
                               "peer_divergence_turn": (c.get("peer") or {}).get("divergence_turn"),
                               "ours_diverged": (c.get("ours") or {}).get("diverged")}
    except (PrefixError, KeyError) as exc:
        row = {"class": "unreadable", "class_why": str(exc)}
    row["peer_drop"] = row["class"] == "peer_first"
    return row


def prefix_report(spec: Any, state: Path, group: str) -> List[Dict[str, Any]]:
    """Every planned pair of ``group`` whose captures DIFFER before the search's first turn (G-PREFIX's comparison,
    ``main.reads.prefix.compare``), classified: outcome-blind (plans and captures only; the comparison's line text is
    not carried)."""
    from main.reads import prefix as PX

    out = []
    for p in PX.deal_pairs(spec, Path(state), group):
        (st_s, us), (st_r, ur) = p["searched"], p["raw"]
        rc, sc = Path(st_r) / "cells" / ur.name, Path(st_s) / "cells" / us.name
        try:
            res = PX.compare(PX.capture_of(rc), PX.capture_of(sc), p["from_turn"])
        except PX.PrefixError:
            continue                          # not played yet (or not one capture): not a pair G-PREFIX checks
        if res["ok"]:
            continue
        out.append({"deal": p["deal"], "searched": us.name, "raw": ur.name, "from_turn": p["from_turn"],
                    **{k: v for k, v in classify_pair(rc, sc, p["from_turn"]).items() if k != "class_why"}})
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.reads.peer_divergence", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="verb", required=True)
    a = sub.add_parser("prefix", help="classify each G-PREFIX mismatch of a searched group (read-only, outcome-blind)")
    a.add_argument("--state", required=True, help="the read's state dir")
    a.add_argument("--group", required=True, help="the per_game anchor group")
    a.add_argument("--spec", default=None, help="the read's spec (must be the state's own)")
    args = ap.parse_args(argv)
    from main.reads.runner import load_state
    from main.reads.spec import SpecError, load

    try:
        spec = load_state(Path(args.state))
        if args.spec is not None and load(args.spec).sha256 != spec.sha256:
            print(json.dumps({"error": f"{args.spec} is not the state's spec ({spec.sha256[:12]})"}))
            return 2
    except (SpecError, OSError, RuntimeError) as exc:
        print(json.dumps({"error": str(exc)}))
        return 2
    if args.group not in spec.anchor or not spec.anchor[args.group].per_game:
        print(json.dumps({"error": f"{args.group!r} is not a per_game anchor group of the spec"}))
        return 2
    rows = prefix_report(spec, Path(args.state), args.group)
    for r in rows:
        print(json.dumps(r, sort_keys=True))
    print(json.dumps({"mismatches": len(rows), "peer_drops": sum(1 for r in rows if r["peer_drop"])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
