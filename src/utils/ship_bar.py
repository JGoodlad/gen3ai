"""THE 8 s BAR — a routine test that costs more than the ship gate's re-tier allows is NAMED (and, far
over, FAILS the gate). The root ``conftest.py`` feeds it every report; ``scripts/ops/ship_gate.sh``
turns it on (``$GEN3AI_SHIP_BAR=1``).

WHY (2026-10-19). The ship gate is ~15 min beside a live run only because every test whose setup + call
+ teardown cost >= 8 s in that shape was tiered ``slow`` (``designs/ops/testing.md`` "The SHIP GATE").
Nothing held the line afterwards: new tests drifted over (``world_legality_integration_test``, 25-30 s),
and each one is minutes of every ship gate, forever. The 30 s quiet-box tier-budget guard in the root
conftest is a backstop for MINUTES-long tests; this is the ship gate's own, tighter line.

WHAT IS MEASURED. Per test, the SUM of its setup + call + teardown durations (a module fixture's cost
lands on the test that pays it, as the tiering rule counts it), on the xdist CONTROLLER (or a serial
session), and — only for a test at or over the bar — the contention over that test's own window
(``utils.cpu_meter``, the root conftest's meter: external load only, the session's own tree
subtracted). Exempt: the COST markers (``slow``, ``e2e``, ``benchmark``), ``static`` (the static
gates run in every tier by definition) and ``linchpin`` (kept routine at any cost, each with a reason).

THE DECISION (:func:`judge`), deterministic by construction (standing rule 8):

* the bar is :data:`BAR_S` (8 s) at the contention the rule was MEASURED at — the ship-gate shape beside
  a live run, whose windowed factor read x2.7-x4.9 (:data:`REF_FACTOR` = 2.7, the low end, so the bar
  never TIGHTENS below 8 s): ``threshold = BAR_S * max(1, factor / REF_FACTOR)``. A box busier than
  the one the rule was measured on stretches it; a quiet box never shrinks it;
* ``total < threshold`` — fine;
* ``threshold <= total < FAIL_MARGIN x threshold`` — WARN: named in a loud list, never a verdict. This
  is rule 8's exclusion band: a test near the bar may read either side of it run to run;
* ``total >= FAIL_MARGIN x threshold`` (16 s at the reference contention) — FAIL the session. A test
  that honours the 8 s rule cannot read twice its contention-stretched bar: the meter under-reads by at
  most ~25 % (its calibration) and the shape's own factor spread (x2.7-x4.9, 1.8x) is inside 2x even if
  the scaling explained none of it. Such a test is mis-tiered whichever side of 16 s it lands;
* contention UNREADABLE for the window ⇒ WARN at most (a window we cannot read is not a measurement).

WHY FAIL AND NOT ONLY WARN: a warn-only list is what let the tree drift (nobody reads it), and the fail
line is placed where no compliant test can reach it, so failing there is deterministic for every test
that obeys the rule. The fix is the re-tier's: mark it ``slow`` with its measured cost in a
``# ship gate (YYYY-MM-DD): N s beside a live run`` comment (the WHOLE file when a module fixture
moves), or make it faster.

Knobs (tests and deliberate experiments only): ``$GEN3AI_SHIP_BAR_S`` (the bar),
``$GEN3AI_SHIP_BAR_REPORT`` (a JSON of every test at or over half the bar, for calibration).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

ENABLE_ENV = "GEN3AI_SHIP_BAR"
BAR_ENV = "GEN3AI_SHIP_BAR_S"
REPORT_ENV = "GEN3AI_SHIP_BAR_REPORT"
BAR_S = 8.0
REF_FACTOR = 2.7
FAIL_MARGIN = 2.0
EXEMPT_MARKERS = ("slow", "e2e", "benchmark", "static", "linchpin")


def enabled() -> bool:
    return os.environ.get(ENABLE_ENV, "") not in ("", "0")


def bar_s() -> float:
    raw = os.environ.get(BAR_ENV)
    return float(raw) if raw else BAR_S


def threshold(factor: float, bar: Optional[float] = None) -> float:
    return (bar_s() if bar is None else bar) * max(1.0, factor / REF_FACTOR)


def judge(total_s: float, factor: Optional[float], bar: Optional[float] = None) -> str:
    """``"ok"`` / ``"warn"`` / ``"fail"`` for one test's setup+call+teardown seconds (see the docstring)."""
    b = bar_s() if bar is None else bar
    if total_s < b:
        return "ok"
    if factor is None:
        return "warn"
    t = threshold(factor, b)
    if total_s < t:
        return "ok"
    return "fail" if total_s >= FAIL_MARGIN * t else "warn"


@dataclass
class _Acc:
    total: float = 0.0
    start: Optional[float] = None
    stop: Optional[float] = None
    pid: Optional[int] = None
    exempt: bool = False


@dataclass
class ShipBar:
    """Accumulates one session's reports; ``window(start, stop, pid)`` returns a contention reading
    (an object with ``.factor`` and ``.describe()``) or None."""

    window: Callable[[float, float, Optional[int]], object]
    acc: Dict[str, _Acc] = field(default_factory=dict)
    judged: List[Tuple[str, str, float, Optional[float], str]] = field(default_factory=list)
    report_rows: List[dict] = field(default_factory=list)

    def add(self, report) -> None:
        a = self.acc.setdefault(report.nodeid, _Acc())
        a.total += float(getattr(report, "duration", 0.0) or 0.0)
        st, sp = getattr(report, "start", None), getattr(report, "stop", None)
        if st is not None:
            a.start = st if a.start is None else min(a.start, st)
        if sp is not None:
            a.stop = sp if a.stop is None else max(a.stop, sp)
        a.pid = getattr(report, "gen3ai_pid", a.pid)
        if any(m in report.keywords for m in EXEMPT_MARKERS):
            a.exempt = True
        if report.when == "teardown":
            self._finish(report.nodeid, self.acc.pop(report.nodeid))

    def _finish(self, nodeid: str, a: _Acc) -> None:
        b = bar_s()
        if a.exempt or a.total < b / 2:
            return
        reading = None
        if a.total >= b and a.start is not None and a.stop is not None:
            try:
                reading = self.window(a.start, a.stop, a.pid)
            except Exception:            # noqa: BLE001 — an unreadable window is a WARN, never a crash
                reading = None
        factor = getattr(reading, "factor", None)
        verdict = judge(a.total, factor, b)
        desc = reading.describe() if reading is not None and hasattr(reading, "describe") else (
            "contention UNREADABLE for this window" if a.total >= b else "")
        self.report_rows.append({"nodeid": nodeid, "total_s": round(a.total, 3), "factor": factor,
                                 "verdict": verdict})
        if verdict != "ok":
            self.judged.append((nodeid, verdict, a.total, factor, desc))

    @property
    def failed(self) -> List[tuple]:
        return [j for j in self.judged if j[1] == "fail"]

    def lines(self) -> List[str]:
        if not self.judged:
            return []
        b = bar_s()
        out = [f"{len(self.failed)} routine test(s) FAIL and {len(self.judged) - len(self.failed)} WARN the "
               f"ship gate's {b:.0f} s bar (setup + call + teardown; bar x max(1, factor / {REF_FACTOR}); FAIL at "
               f">= {FAIL_MARGIN:.0f}x, the band between is rule 8's margin — named, never a verdict):"]
        for nodeid, verdict, total, factor, desc in sorted(self.judged, key=lambda j: -j[2]):
            f = f"x{factor:.2f}" if factor is not None else "x?"
            out.append(f"  {total:7.1f}s  {verdict.upper():4}  {f}  {nodeid}")
            if desc:
                out.append(f"           window: {desc}")
        out.append("Mark each FAIL `slow` with a `# ship gate (<date>): N s beside a live run` comment (the "
                   "WHOLE file when a module fixture's cost would move onto the next test), or make it faster "
                   "(designs/ops/testing.md \"The SHIP GATE\" — the 8 s bar). A WARN that repeats is the same "
                   "test, earlier.")
        return out

    def write_report(self) -> Optional[str]:
        path = os.environ.get(REPORT_ENV)
        if not path:
            return None
        rows = sorted(self.report_rows, key=lambda r: -r["total_s"])
        with open(path, "w") as f:
            json.dump({"schema": "gen3_ship_bar_report_v1", "bar_s": bar_s(), "ref_factor": REF_FACTOR,
                       "fail_margin": FAIL_MARGIN, "rows": rows}, f, indent=1)
        return path
