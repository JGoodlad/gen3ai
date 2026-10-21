"""A SPECTATOR view of one of our own battles, from its two seat streams (X50 rung 1, design §3).

An eval trace (``models/<run>/eval_traces/step_*/<opp>/*.p{1,2}.jsonl.gz``, the core record) holds each seat's
own stream: its private lines (its own HP in absolute numbers) and its ``|request|`` lines. A public replay holds
neither. The view drops every request and merges the two streams (:func:`spectator_lines`): the public version of
each line, and no seat's private line. Anything that does not pair is a :class:`SpectateError`.
"""
from __future__ import annotations

import bisect
import difflib
import gzip
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


class SpectateError(ValueError):
    """Two seat streams that do not merge into one public log."""


def read_stream(path: Path) -> Tuple[Dict, List[str]]:
    """(the record header, the stream's protocol lines) of one ``*.pN.jsonl.gz`` core record."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        header = json.loads(fh.readline())
        lines = [json.loads(ln)["text"] for ln in fh if ln.strip()]
    return header, lines


def _kw(line: str) -> str:
    parts = line.split("|")
    return parts[1] if len(parts) > 1 else ""


def _names(line: str) -> str:
    parts = line.split("|")
    return parts[2][:2] if len(parts) > 2 and parts[2][:2] in ("p1", "p2") else ""


_HP = re.compile(r"\d+/\d+")


def _norm(line: str) -> str:
    """A line with every ``x/y`` HP figure folded, so a seat's private version pairs with the public one."""
    return _HP.sub("HP", line)


def _private_any(line: str) -> bool:
    """A line a seat may hold alone whatever it names: its refused choice, or a mechanics hint."""
    return line.startswith("|error|") or line.startswith("|-hint|")


@dataclass
class SpectatorView:
    """The public log, and for each merged line the index + 1 of its source line in each seat's request-free
    stream (a dropped private line counts toward the NEXT public line), to map own positions onto the log."""
    lines: List[str]
    consumed: Dict[str, List[int]]

    def public_pos(self, seat: str, own_pos: int) -> int:
        """The public-log position of a point that sits after ``own_pos`` lines of ``seat``'s own stream."""
        return bisect.bisect_right(self.consumed[seat], own_pos)


def spectator_view(p1: List[str], p2: List[str]) -> SpectatorView:
    """The public log of a battle from its two seat streams (requests dropped).

    The two request-free streams are ALIGNED (``difflib``, HP figures folded): a paired line is taken as is when
    the two agree, else from the stream of the side it does NOT name (the public version); a line only ONE stream
    holds must be that seat's PRIVATE line (one naming its own side, e.g. ``|-ability|p1a: Zapdos|Pressure|[silent]``,
    or its ``|error|`` / ``|-hint|``) and is dropped. Anything else is a :class:`SpectateError`."""
    a = [ln for ln in p1 if not ln.startswith("|request|")]
    b = [ln for ln in p2 if not ln.startswith("|request|")]
    sm = difflib.SequenceMatcher(None, [_norm(x) for x in a], [_norm(y) for y in b], autojunk=False)
    out: List[str] = []
    ca: List[int] = []
    cb: List[int] = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for x, y, i, j in zip(a[i1:i2], b[j1:j2], range(i1, i2), range(j1, j2)):
                if x == y:
                    out.append(x)
                else:
                    side = _names(x)
                    if side not in ("p1", "p2"):
                        raise SpectateError(f"lines {i} / {j} differ but name no side: {x[:80]!r} vs {y[:80]!r}")
                    out.append(y if side == "p1" else x)
                ca.append(i + 1)
                cb.append(j + 1)
            continue
        for i in range(i1, i2):
            if not (_names(a[i]) == "p1" or _private_any(a[i])):
                raise SpectateError(f"p1 line {i} has no partner and is not p1's private line: {a[i][:80]!r}")
        for j in range(j1, j2):
            if not (_names(b[j]) == "p2" or _private_any(b[j])):
                raise SpectateError(f"p2 line {j} has no partner and is not p2's private line: {b[j][:80]!r}")
    return SpectatorView(out, {"p1": ca, "p2": cb})


def spectator_lines(p1: List[str], p2: List[str]) -> List[str]:
    """:func:`spectator_view`'s public log."""
    return spectator_view(p1, p2).lines


def request_points(stream: List[str]) -> List[Tuple[int, bool]]:
    """Each non-``wait`` request of a seat's own stream as (its position in the request-free stream, forced)."""
    out: List[Tuple[int, bool]] = []
    n = 0
    for ln in stream:
        if ln.startswith("|request|"):
            body = ln[len("|request|"):]
            if body:
                req = json.loads(body)
                if not req.get("wait"):
                    out.append((n, bool(req.get("forceSwitch"))))
            continue
        n += 1
    return out
