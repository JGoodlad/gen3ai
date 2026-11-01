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

A PEER DROP (:func:`is_peer_drop`): a BRANCH attempt classed ``peer_first``; a NULL attempt classed ``peer_first`` or
``peer_first_late`` ((a) and (b) at any turn: AM-DL-PEER's clarification of 2026-10-31, since a null forces pi's own
action and any later divergence is not ours either). Every other divergence of a failed or changed unit stays VOID.

    python peer_divergence.py classify --raw RAW_CAPTURE.json --branch BRANCH_CAPTURE.json --turn T
    python peer_divergence.py state --branch-state DIR [--all]      # every null, and every uncounted or
                                                                    # G-BRANCH-failed branch attempt
    python peer_divergence.py selftest                              # synthetic captures, one per class
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


def is_peer_drop(kind: str, cls: str) -> bool:
    """AM-DL-PEER rules 1 and 3 (with the 2026-10-31 clarification): a branch needs (a)-(c), a null (a)-(b)."""
    if kind == "branch":
        return cls == "peer_first"
    if kind == "null":
        return cls in ("peer_first", "peer_first_late")
    raise ValueError(f"unit kind {kind!r}")


def _load(p: Path) -> dict[str, Any]:
    return dict(json.loads(Path(p).read_text()))


def _attempt_dirs(state: Path, unit: str) -> list[Path]:
    return [d for d in [state / "cells" / unit, *sorted((state / "cells").glob(f"{unit}.partial.*"))] if d.is_dir()]


def classify_state(state: Path, everything: bool = False) -> list[dict[str, Any]]:
    """Every attempt of every NULL, and of every branch that is not counted or that recorded a G-BRANCH failure (every
    unit with ``everything``), classified against its raw game's capture, with :func:`is_peer_drop`. A null is always
    classified: a counted null can still change its game after its branch turn."""
    from main.value_search.live_branch import _g_branch_fails
    from main.x35 import kakuna as KA

    units = {u.name: u for u in KA.load_plan(state)}
    out = []
    for line in (state / "branch_plan.jsonl").read_text().splitlines():
        m = json.loads(line)
        u = units[m["unit"]]
        done, gb = KA.is_done(state, u), _g_branch_fails(state, u.name)
        if not everything and m["kind"] == "branch" and done and not gb:
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
                        "g_branch_fails": len(gb), "peer_drop": is_peer_drop(m["kind"], c["class"]), **c})
    return out


def _synthetic(turns: int, *, peer_at: int | None = None, ours_at: int | None = None,
               engine_at: int | None = None) -> dict[str, Any]:
    """A capture of ``turns`` turns, ours p1 and the peer p2. ``peer_at`` / ``ours_at``: the turn from which that side
    chooses differently (each later turn's protocol then differs); ``engine_at``: the protocol differs from that turn
    with every command equal."""
    start = 'START {"formatid": "gen3ou", "p1": {"name": "Gen3AIAnchor1", "team": "x"}, ' \
            '"p2": {"name": "MetaKakuna1", "team": "y"}, "seed": [1, 2, 3, 4]}'
    cmds, chunks = [start], []
    diverged = False
    for t in range(1, turns + 1):
        for side in ("p1", "p2"):
            chunks.append([side, f'|request|{{"active": [{{}}], "rqid": {t}, "v": {int(diverged)}}}'])
        ours = "move b" if ours_at is not None and t >= ours_at else "move a"
        peer = "move d" if peer_at is not None and t >= peer_at else "move c"
        cmds += [f"CHOOSE p1 {ours}", f"CHOOSE p2 {peer}"]
        diverged = diverged or ours != "move a" or peer != "move c" or (engine_at is not None and t >= engine_at)
        for side in ("p1", "p2"):
            chunks.append([side, f"|\n|move|{ours}|{peer}|{int(diverged)}\n|turn|{t + 1}"])
    for side in ("p1", "p2"):
        chunks.append([side, "|win|Gen3AIAnchor1"])
    return {"tag": "battle-gen3ou-1", "seed": [1, 2, 3, 4], "commands": cmds, "chunks": chunks}


def selftest() -> int:
    """One synthetic case per class, each with its peer-drop verdict for a branch and for a null (branch turn 10)."""
    raw = _synthetic(20)
    cases = [  # (replay, class, branch drop, null drop)
        (_synthetic(20), "identical", False, False),
        (_synthetic(20, peer_at=3), "peer_first", True, True),
        (_synthetic(20, peer_at=12), "peer_first_late", False, True),
        (_synthetic(20, peer_at=10), "peer_first_late", False, True),
        (_synthetic(20, ours_at=10), "ours_first", False, False),
        (_synthetic(20, ours_at=3, peer_at=3), "ours_first", False, False),
        (_synthetic(20, engine_at=4), "unattributable", False, False),
    ]
    bad = 0
    for rep, cls, b_drop, n_drop in cases:
        got = classify(raw, rep, 10)["class"]
        ok = got == cls and is_peer_drop("branch", got) == b_drop and is_peer_drop("null", got) == n_drop
        bad += not ok
        print(json.dumps({"want": cls, "got": got, "branch_drop": is_peer_drop("branch", got),
                          "null_drop": is_peer_drop("null", got), "ok": ok}))
    print(json.dumps({"selftest": "PASS" if not bad else "FAIL", "cases": len(cases)}))
    return 0 if not bad else 1


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
    sub.add_parser("selftest")
    args = ap.parse_args(argv)
    if args.verb == "selftest":
        return selftest()
    if args.verb == "classify":
        print(json.dumps(classify(_load(Path(args.raw)), _load(Path(args.branch)), args.turn), sort_keys=True))
        return 0
    for r in classify_state(Path(args.branch_state), everything=args.all):
        print(json.dumps(r, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
