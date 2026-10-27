"""A UNIT'S ATTEMPTS, read back from its kakuna state's own log (``designs/tools/reads.md`` §3, §6).

``kakuna.MAX_ATTEMPTS`` bounds the attempts a supervisor makes at one unit, but only in its own memory: a finished
supervisor leaves no record of who gave up. Two different reasons leave a unit without a counted result:

* **stop-cut**: the lane reached its registered stop before the unit's turn (or before its retry);
* **EXHAUSTED**: every attempt the lane makes at a unit ran and none counted (a crash, a kill, a peer fault, a refusal of
  the count): more time would not have helped, and the unit is a defect of the play, not of the schedule.

``INCOMPLETE_BY_STOP`` (``record``) is a claim about the first reason only, so the records must tell them apart. The
evidence is ``<kakuna state>/units.jsonl``, which ``kakuna._play`` appends to for every attempt: one ``end`` event per
child, ``counted`` saying whether the unit counted then. A short-lane overrun plays the unit twice inside ONE attempt (the
short child's ``end``, a ``short_lane_overrun`` event, then the long child's ``end``): the overrun event nets the first
``end`` out, so an attempt is one ``run_unit`` call. An attempt whose supervisor was killed has no ``end`` and is not
counted as a failure (it was interrupted, not failed).

An attempt count is PERSISTED history, so it is the number of failed attempts over every supervisor life of the state; a
later supervisor still retries an exhausted unit (its own ``fails`` start at zero), and a unit that counts later is no
longer exhausted. This module is OFF the play path (``main.reads.playpath``).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence


def failed_attempts(kakuna_state: Path) -> Dict[str, int]:
    """``{unit name: attempts that ENDED uncounted}`` from the state's ``units.jsonl`` (a damaged line is skipped, as
    ``kakuna._overran`` skips it; no log = no attempt). A unit that counted later still has its earlier failures here."""
    try:
        lines = (Path(kakuna_state) / "units.jsonl").read_text().splitlines()
    except OSError:
        return {}
    failed: Dict[str, int] = {}
    for line in lines:
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if not isinstance(ev, dict):
            continue
        name, kind = ev.get("unit"), ev.get("event")
        if not isinstance(name, str):
            continue
        if kind == "end" and not ev.get("counted"):
            failed[name] = failed.get(name, 0) + 1
        elif kind == "short_lane_overrun":          # the short child's `end` was half an attempt: the long job is the rest
            failed[name] = failed.get(name, 0) - 1
    return {n: k for n, k in failed.items() if k > 0}


def exhausted_units(kakuna_state: Path, units: Optional[Sequence] = None) -> List[str]:
    """The names of the planned units that are NOT counted and whose failed attempts reach ``kakuna.MAX_ATTEMPTS``: the
    retries are spent. ``units``: the state's plan (default: loaded from ``plan.json``)."""
    from main.x35 import kakuna as KA

    st = Path(kakuna_state)
    plan = KA.load_plan(st) if units is None else list(units)
    failed = failed_attempts(st)
    return sorted(u.name for u in plan if failed.get(u.name, 0) >= KA.MAX_ATTEMPTS and not KA.is_done(st, u))
