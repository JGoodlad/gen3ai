"""The VIEWER's decision points on a spectator log (X50 rung 1, `design_x50_rung1_intent_labels.md` §3).

A point is ``(pos, kind)``: the decision opens after ``lines[:pos]`` (the battle lines the live client would
feed, ``main.live.replay_scan.battle_lines``), where Showdown sends the viewer a non-``wait`` request:

* ``turn``: right after each ``|turn|N`` line;
* ``forced``: the viewer's replacement after its active FAINTED or its Baton Pass resolved, placed at the START
  of the chunk that carries the replacement ``|switch|``: back from that switch to the chunk's ``|t:|`` line, and
  before the bare ``|`` that opens the chunk (a double KO's two replacements share one chunk). A replacement with
  no new chunk since its trigger was CHOSEN before (a switch that continues after a Pursuit KO): no point.

No point at the opponent's forced switch (the viewer's request is ``wait``) and none after ``|win|`` / ``|tie|``.
A log with no ``|t:|`` chunk marks at all is a :class:`PointError`, never a guess. Pure text; the points are checked against real request positions on our own traces
(``points_test.py``, and ``run.trace_job`` on every trace it reads).
"""
from __future__ import annotations

from typing import List, Tuple

TURN, FORCED = "turn", "forced"


class PointError(ValueError):
    """A decision point that cannot be placed deterministically."""


def _ident_side(field: str) -> str:
    return field[:2]


def decision_points(lines: List[str], seat: str) -> List[Tuple[int, str]]:
    """The viewer ``seat``'s (``"p1"`` / ``"p2"``) decision points over ``lines``, in order."""
    if not any(ln.startswith("|t:|") for ln in lines):
        raise PointError("a log with no |t:| chunk marks: forced decisions cannot be placed")
    out: List[Tuple[int, str]] = []
    pending_at = -1          # index of the viewer's faint / Baton Pass line awaiting its replacement
    for i, ln in enumerate(lines):
        parts = ln.split("|")
        kw = parts[1] if len(parts) > 1 else ""
        if kw in ("win", "tie"):
            break
        if kw == "turn":
            pending_at = -1
            out.append((i + 1, TURN))
            continue
        who = _ident_side(parts[2]) if len(parts) > 2 else ""
        if who != seat:
            continue
        if kw == "faint" or (kw == "move" and len(parts) > 3 and parts[3] == "Baton Pass"):
            if pending_at < 0:
                pending_at = i
            continue
        if kw == "switch" and pending_at >= 0:
            j = i - 1
            while j > pending_at and not lines[j].startswith("|t:|"):
                j -= 1
            if j <= pending_at:
                # no new chunk since the trigger: no request was sent — a switch chosen earlier that continues
                # after its user fainted (gen 2-4 Pursuit), never a forced decision
                pending_at = -1
                continue
            pos = j - 1 if lines[j - 1] == "|" else j
            if out and pos <= out[-1][0]:
                raise PointError(f"{seat}: forced point {pos} not after the previous point {out[-1][0]}")
            out.append((pos, FORCED))
            pending_at = -1
    return out
