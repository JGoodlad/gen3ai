"""KEPT INPUT LOGS of a RAW anchor cell (``main.anchors --keep-input-logs FROM-TO``): the S-U gate test's states
(``designs/research_state/measurements/su_gate_2026-10-27/``; the S-U review's "gate test", 7aafa527).

A raw cell plays exactly as without the flag: our side is the checkpoint, greedy, on the in-process core slot. At each
of OUR decisions in the turn window [FROM, TO] with >= 2 legal actions, the recorder writes the battle's TRUE input log
cut honestly for that decision (:func:`main.anchors.search_side.honest_log`, the live searched player's own cut) with
the decision's identity, so the registered search can be re-run OFFLINE at that decision exactly as the live searcher
would have run it there (:mod:`main.value_search.sign_gate`):

* ``key``: the live searcher's keying (``search_side.KEY_SCHEMA | deal | turn | decision index``), so the worlds, their
  screens and the playouts' dice are the ones a searched game of this deal draws at this decision;
* ``tokens`` and ``frame_sha256`` (the live core row's base64, hashed): the offline replay of ``log`` must reproduce
  the decision's tokens and the row byte for byte (the live guard G12, ``LiveSearcher._check_root``), so a kept log
  that does not replay to the decision we played is refused by name, never scored;
* ``pi_argmax``: the checkpoint's greedy choice we PLAYED; the offline forward must put it first.

One JSON line a recorded decision (schema :data:`SCHEMA`) in ``<out>/input_logs.jsonl``. Nothing is recorded outside
the window, and the recorder never changes a choice: the raw cell's games are the raw arm's, decision for decision.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

SCHEMA = "gen3_input_log_v1"
_WINDOW = re.compile(r"^(\d{1,3})-(\d{1,3})$")


class InputLogError(ValueError):
    """A window that is not ``FROM-TO`` with 1 <= FROM <= TO."""


def parse_window(text: str) -> Tuple[int, int]:
    """``"10-30"`` -> ``(10, 30)``: both turns inclusive."""
    m = _WINDOW.match(str(text).strip())
    if not m:
        raise InputLogError(f"{text!r}: a turn window is FROM-TO (e.g. 10-30)")
    lo, hi = int(m.group(1)), int(m.group(2))
    if not 1 <= lo <= hi:
        raise InputLogError(f"{text!r}: needs 1 <= FROM <= TO")
    return lo, hi


def canonical(text: str) -> str:
    lo, hi = parse_window(text)
    return f"{lo}-{hi}"


def frame_sha256(b64: str) -> str:
    return hashlib.sha256(str(b64).encode()).hexdigest()


class InputLogRecorder:
    """Records OUR decisions of a raw cell in ``window`` (module docstring) to ``out``."""

    def __init__(self, window: str, out: Path) -> None:
        self.window = canonical(window)
        self.lo, self.hi = parse_window(self.window)
        self.out = Path(out)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.stats: Dict[str, Any] = {"window": self.window, "decisions": 0, "in_window": 0, "recorded": 0,
                                      "one_legal": 0}

    def record(self, front: Any, tag: str, d: Dict[str, Any], mask: Any, conn: Any, argmax: int) -> None:
        """One decision of ours (``d`` its core frame, ``conn`` our connection) whose greedy choice ``argmax`` we play."""
        from main.anchors.search_side import KEY_SCHEMA, LiveSearchError, deal_key, honest_log

        self.stats["decisions"] += 1
        turn, n = int(d.get("turn") or 0), int(d["n"])
        if not self.lo <= turn <= self.hi:
            return
        self.stats["in_window"] += 1
        legal = [int(i) for i in np.flatnonzero(np.asarray(mask))]
        if len(legal) < 2:
            self.stats["one_legal"] += 1
            return
        battle = front.battles.get(tag)
        if battle is None:
            raise LiveSearchError(f"{tag}: the front end holds no such battle")
        side = next((s for s, c in battle.conns.items() if c is conn), None)
        if side is None:
            raise LiveSearchError(f"{tag}: our connection is not a side of the battle")
        log, at, update = honest_log(battle, side)
        deal = deal_key(battle.start_doc, side)
        rec = {"schema": SCHEMA, "battle": tag, "index": getattr(battle, "index", None), "side": side, "turn": turn,
               "n": n, "n_legal": len(legal), "mask": [int(x) for x in np.asarray(mask).tolist()],
               "pi_argmax": int(argmax), "deal": deal, "key": f"{KEY_SCHEMA}|{deal}|t{turn}|n{n}", "update": update,
               "at": at, "tokens": {str(k): str(v) for k, v in (d.get("tokens") or {}).items()},
               "frame_sha256": frame_sha256(d["frame"]["b64"]), "log": log}
        with open(self.out, "a") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
            f.flush()
        self.stats["recorded"] += 1

    def summary(self) -> Dict[str, Any]:
        return dict(self.stats, path=str(self.out))
