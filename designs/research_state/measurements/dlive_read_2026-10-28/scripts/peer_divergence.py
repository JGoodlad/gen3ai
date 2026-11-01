"""AM-DL-PEER's classifier (PREREG, Amendment AM-DL-PEER): did a forced-branch replay leave its raw game because the
PEER (Kakuna) chose differently on an identical view, before the branch turn?

It reads two front-end captures (``<cell>/capture/<battle>.json``: the ``START`` document, the ``commands`` both sides
sent, and the ``chunks`` each side was relayed) and NEVER a result: every ``|win|`` / ``|tie`` line is removed before
any comparison, no game row or summary is opened, and only indices, turns and sides are printed.

The test, per side s (ours / the peer's, named from the START document: ours ``Gen3AIAnchor*``, the peer
``MetaKakuna*``):

* ``j_s``: the first relayed chunk of s's stream that differs between the raw game and the replay (``|t:|`` is
  normalised in the captures). ``k_s``: the number of s's choice requests (``|request|`` without ``"wait"``) in
  s's stream BEFORE ``j_s``: the choices s made on a view identical in both games.
* s DIVERGED ON AN IDENTICAL VIEW iff s's first ``k_s`` commands (s's ``CHOOSE`` lines, in order; a side's commands
  answer its requests one for one) are not all equal. Its divergence turn is the last ``|turn|N`` of s's stream before
  the request its first differing command answered.

Classes (the amendment's conditions (a)-(c)):

* ``identical``: neither stream nor either side's commands differ (a null that replayed its game);
* ``peer_first``: the PEER diverged on an identical view (a), OUR first ``k_ours`` commands are all equal (b), and the
  peer's divergence turn is below the branch turn ``T`` (c): a PEER DROP;
* ``peer_first_late``: (a) and (b) hold, (c) does not;
* ``ours_first``: our side diverged on an identical view (a forced branch does this at its own turn T by design);
* ``unattributable``: the streams differ and neither side chose differently on an identical view, or the capture
  cannot be read this way.

    python peer_divergence.py classify --raw RAW_CAPTURE.json --branch BRANCH_CAPTURE.json --turn T
    python peer_divergence.py state --branch-state DIR [--all]      # every uncounted or G-BRANCH-failed attempt
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

OURS, PEER = "Gen3AIAnchor", "MetaKakuna"
RESULT_PREFIXES = ("|win|", "|tie")


def _clean(text: str) -> str:
    return "\n".join(x for x in str(text).split("\n") if not x.startswith(RESULT_PREFIXES))


def sides(cap: Mapping[str, Any]) -> dict[str, str]:
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


def _stream(cap: Mapping[str, Any], side: str) -> list[str]:
    return [_clean(c[1]) for c in cap["chunks"] if c[0] == side]


def _cmds(cap: Mapping[str, Any], side: str) -> list[str]:
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


def _side_view(raw: Mapping[str, Any], br: Mapping[str, Any], side: str) -> dict[str, Any]:
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


def classify(raw: Mapping[str, Any], br: Mapping[str, Any], branch_turn: int) -> dict[str, Any]:
    try:
        sd = sides(raw)
        if sides(br) != sd or raw["commands"][0] != br["commands"][0]:
            return {"class": "unattributable", "why": "the START documents differ"}
        ours, peer = _side_view(raw, br, sd["ours"]), _side_view(raw, br, sd["peer"])
    except (KeyError, ValueError, IndexError, TypeError) as exc:
        return {"class": "unattributable", "why": f"{type(exc).__name__}: {exc}"}
    out: dict[str, Any] = {"sides": sd, "ours": ours, "peer": peer, "branch_turn": int(branch_turn)}
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


def _load(p: Path) -> dict[str, Any]:
    return dict(json.loads(Path(p).read_text()))


def _attempt_dirs(state: Path, unit: str) -> list[Path]:
    return [d for d in [state / "cells" / unit, *sorted((state / "cells").glob(f"{unit}.partial.*"))] if d.is_dir()]


def classify_state(state: Path, everything: bool = False) -> list[dict[str, Any]]:
    """Every attempt of every planned unit that is not counted or that recorded a G-BRANCH failure (all units with
    ``everything``), classified against its raw game's capture."""
    from main.value_search.live_branch import _g_branch_fails
    from main.x35 import kakuna as KA

    units = {u.name: u for u in KA.load_plan(state)}
    out = []
    for line in (state / "branch_plan.jsonl").read_text().splitlines():
        m = json.loads(line)
        u = units[m["unit"]]
        done, gb = KA.is_done(state, u), _g_branch_fails(state, u.name)
        if not everything and done and not gb:
            continue
        raw = _load(Path(m["raw_state"]) / "cells" / m["cell"] / "capture" / f"{m['battle']}.json")
        for d in _attempt_dirs(state, u.name):
            caps = sorted((d / "capture").glob("*.json"))
            if len(caps) != 1:
                out.append({"unit": u.name, "kind": m["kind"], "panel": m["panel"], "attempt": d.name,
                            "class": "unattributable", "why": f"{len(caps)} captures"})
                continue
            c = classify(raw, _load(caps[0]), int(m["turn"]))
            out.append({"unit": u.name, "kind": m["kind"], "panel": m["panel"], "attempt": d.name, "counted": done,
                        "g_branch_fails": len(gb), **c})
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="verb", required=True)
    a = sub.add_parser("classify")
    a.add_argument("--raw", required=True)
    a.add_argument("--branch", required=True)
    a.add_argument("--turn", type=int, required=True)
    b = sub.add_parser("state")
    b.add_argument("--branch-state", required=True)
    b.add_argument("--all", action="store_true")
    args = ap.parse_args(argv)
    if args.verb == "classify":
        print(json.dumps(classify(_load(Path(args.raw)), _load(Path(args.branch)), args.turn), sort_keys=True))
        return 0
    for r in classify_state(Path(args.branch_state), everything=args.all):
        print(json.dumps(r, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
