"""THE 8 s BAR — a routine test that costs more than the ship gate's re-tier allows is NAMED (and, far
over, FAILS the gate). The root ``conftest.py`` feeds it every report; ``scripts/ops/ship_gate.sh``
turns it on (``$GEN3AI_SHIP_BAR=1``).

WHY (2026-10-19). The ship gate is ~15 min beside a live run only because every test whose setup + call
+ teardown cost >= 8 s in that shape was tiered ``slow`` (``designs/ops/testing.md`` "The SHIP GATE").
Nothing held the line afterwards: new tests drifted over (``world_legality_integration_test``, 25-30 s),
and each one is minutes of every ship gate, forever. The 30 s quiet-box tier-budget guard in the root
conftest is a backstop for MINUTES-long tests; this is the ship gate's own, tighter line.

WHAT IS MEASURED. Per test, the SUM of its setup + call + teardown durations MINUS the setup and teardown
of every fixture wider than a function (see SCOPE below), on the xdist CONTROLLER (or a serial session), and
— only for a test at or over the bar — the contention over that test's own window
(``utils.cpu_meter``, the root conftest's meter: external load only, the session's own tree
subtracted). Exempt: the COST markers (``slow``, ``e2e``, ``benchmark``), ``static`` (the static
gates run in every tier by definition) and ``linchpin`` (kept routine at any cost, each with a reason).

SCOPE (2026-10-19, after a ship gate failed ``live_side_test`` on a cost that was not the test's). pytest
charges a module- / class- / package- / session-scoped fixture's setup to whichever test happens to request it
FIRST, and its teardown to the LAST test of the scope, so a verdict that sums per test depends on which tests
a module holds, their order and — under xdist — which worker got them (rule 8). So the cost is attributed by
SCOPE, timed where pytest runs it (the root conftest's ``pytest_fixture_setup`` wrapper, plus a pair of
finalizer markers around the fixture's own teardown):

* a FUNCTION-scoped fixture is part of the test that asks for it (it counts toward that test, as before);
* a fixture of any wider scope is taken OFF the test and charged to its SCOPE OWNER — the module (a class- or
  package-scoped one: its class / package; the session: ``(session)``) — as one ROW per owner: the sum of its
  fixtures' setup and teardown seconds, each fixture's cost being its MAX over xdist workers (``--dist load``
  can split a module over workers, each paying the fixture; a max does not depend on the split);
* an owner row is judged ONCE, by :func:`judge`, with the SAME bar and margins as a test (a module's shared
  setup is as much of the ship gate's wall as a test of that cost: the tiering rule marks the WHOLE file
  ``slow`` when a module fixture is what costs), over the seconds-weighted contention of its fixtures' own
  windows. An owner is exempt when the test that carried the cost was (``slow`` / ``e2e`` / ...).

The same tree, the same verdicts, whatever the order or the worker split: ``ship_bar_test.py`` proves it in
child sessions. NOT covered: a one-off cost paid INSIDE a test body (a lazy import, a lazy ``cargo build``)
is the test's by construction; ``ship_gate.py`` warms the Rust binaries before the session for that reason.

THE DECISION (:func:`judge`), deterministic by construction (standing rule 8):

* the bar is :data:`BAR_S` (8 s) at the contention the rule was MEASURED at — the ship-gate shape beside
  a live run, whose windowed factor read x2.7-x4.9 (:data:`REF_FACTOR` = 2.7, the low end, so the bar
  never TIGHTENS below 8 s): ``threshold = BAR_S * max(1, factor / REF_FACTOR)``. A box busier than
  the one the rule was measured on stretches it; a quiet box never shrinks it;
* ``total < threshold`` — fine;
* ``threshold <= total < FAIL_MARGIN x threshold`` — WARN: named in a loud list, never a verdict. This
  is rule 8's exclusion band: a test near the bar may read either side of it run to run;
* ``FAIL_MARGIN x threshold <= total < FAIL_LINE x threshold`` — still WARN: THE EXCLUSION BAND AROUND THE
  FAIL LINE (rule 8). The nominal line is 2x the stretched bar (16 s at the reference contention), but a
  read within :data:`FAIL_BAND` (25 %) of it is not a verdict: ``compile_cache_test``'s ``[passing]`` case
  read 16.7 s against a 16.0 s line under heavy contention and failed a ship gate (2026-10-19) while its
  sibling case read 8.4 s stretch-normalised — a coin flip on a test that obeys the 8 s rule;
* ``total >= FAIL_LINE x threshold`` (:data:`FAIL_LINE` = ``FAIL_MARGIN * (1 + FAIL_BAND)`` = 2.5x, 20 s at the
  reference contention) — FAIL the session. WHERE THE 25 % COMES FROM, with its data: (a) the contention
  meter under-reads the probe by up to ~25 % in the 8-core band (``designs/ops/testing.md``, the meter's
  calibration), so the stretch ``factor / REF_FACTOR`` can itself be 1.25x too small and the true threshold
  1.25x higher; (b) the MEASURED spread of a test's stretch-normalised read (``total / max(1, factor /
  REF_FACTOR)``) over the day's ship gates (2026-10-19, 18 WARN / FAIL rows in /tmp gate logs): the
  tests that sit just over the bar read 8.0-11.3 s normalised (same test repeated: ``cpu_inductor_tile2d``
  8.8-11.3 = 1.28x, ``learner_gates`` rollout-weights 10.0-14.3 = 1.43x), the worst of them 14.3 s = 1.8x the
  bar, which x1.25 for (a) is 2.2x < 2.5x; and the one outlier that failed the gate was 16.7 s = 2.1x. The
  tests that are truly mis-tiered read 22.1 s (live_side) / 23.4 / 31.6 / 33.6 / 56.5 s = 2.8x-7x, all still
  past the 2.5x line. A test that honours the 8 s rule cannot reach 2.5x its stretched bar through noise;
  a test past it is mis-tiered whichever side of 20 s it lands;
* contention UNREADABLE for the window ⇒ WARN at most (a window we cannot read is not a measurement).

WHY FAIL AND NOT ONLY WARN: a warn-only list is what let the tree drift (nobody reads it), and the fail
line (with its band) is placed where no compliant test can reach it, so failing there is deterministic for every test
that obeys the rule. The fix is the re-tier's: mark it ``slow`` with its measured cost in a
``# ship gate (YYYY-MM-DD): N s beside a live run`` comment (the WHOLE file when a module fixture
moves), or make it faster.

Knobs (tests and deliberate experiments only): ``$GEN3AI_SHIP_BAR_S`` (the bar),
``$GEN3AI_SHIP_BAR_REPORT`` (a JSON of every test at or over half the bar, for calibration).
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

ENABLE_ENV = "GEN3AI_SHIP_BAR"
BAR_ENV = "GEN3AI_SHIP_BAR_S"
REPORT_ENV = "GEN3AI_SHIP_BAR_REPORT"
BAR_S = 8.0
REF_FACTOR = 2.7
FAIL_MARGIN = 2.0
#: Rule 8's exclusion band AROUND the 2x line: the contention meter's documented ~25 % under-read (see the
#: docstring for the data); a read inside [FAIL_MARGIN, FAIL_LINE) x threshold is WARN, never FAIL.
FAIL_BAND = 0.25
FAIL_LINE = FAIL_MARGIN * (1.0 + FAIL_BAND)
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
    return "fail" if total_s >= FAIL_LINE * t else "warn"


#: Fixture scopes wider than a function: their setup / teardown is the SCOPE OWNER's, never one test's.
SESSION_OWNER = "(session)"


class ScopedTimer:
    """Worker-side half: times the setup and the teardown of every WIDE-scoped fixture and parks the events
    until the next report is built, which carries them to the controller (``report.gen3ai_scoped``, a list of
    JSON-safe dicts: ``scope``, ``owner``, ``fixture``, ``phase``, ``s``, ``start``, ``stop``, ``pid``).

    The root conftest drives it: its ``pytest_fixture_setup`` wrapper calls :meth:`begin` / :meth:`end`, and
    ``pytest_runtest_makereport`` calls :meth:`drain`. Teardown is timed by two finalizer markers: one added
    BEFORE the fixture function runs (so it runs LAST in pytest's LIFO ``FixtureDef.finish``, after the
    generator's own teardown) and one added AFTER (so it runs FIRST, right after the dependents' ``finish``,
    which registered later still): the span between them is that fixture's OWN teardown.
    """

    def __init__(self) -> None:
        self.pending: List[dict] = []

    def _event(self, scope: str, owner: str, fixture: str, phase: str, t0: float, t1: float) -> None:
        self.pending.append({"scope": scope, "owner": owner, "fixture": fixture, "phase": phase,
                             "s": max(0.0, t1 - t0), "start": t0, "stop": t1, "pid": os.getpid()})

    def begin(self, fixturedef: Any, request: Any) -> tuple:
        scope = str(fixturedef.scope)
        owner = SESSION_OWNER if scope == "session" else (getattr(request.node, "nodeid", "") or SESSION_OWNER)
        name = str(fixturedef.argname)
        span: Dict[str, float] = {}

        def end_marker() -> None:           # runs LAST: after the fixture's own teardown
            if "t0" in span:
                self._event(scope, owner, name, "teardown", span["t0"], time.time())

        fixturedef.addfinalizer(end_marker)
        return (scope, owner, name, span, fixturedef, time.time())

    def end(self, token: tuple, ok: bool) -> None:
        scope, owner, name, span, fixturedef, t0 = token
        self._event(scope, owner, name, "setup", t0, time.time())
        if ok:
            fixturedef.addfinalizer(lambda: span.__setitem__("t0", time.time()))   # runs FIRST

    def drain(self) -> List[dict]:
        out, self.pending = self.pending, []
        return out


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
    #: (scope, owner, fixture, phase) -> [(seconds, start, stop, pid)] over every worker's copy, non-exempt only
    scoped: Dict[Tuple[str, str, str, str], List[Tuple[float, float, float, Optional[int]]]] = field(
        default_factory=dict)
    _finalized: bool = False

    def add(self, report: Any) -> None:
        a = self.acc.setdefault(report.nodeid, _Acc())
        exempt = any(m in report.keywords for m in EXEMPT_MARKERS)
        wide = 0.0
        for e in getattr(report, "gen3ai_scoped", None) or ():
            wide += float(e["s"])
            if not exempt:
                key = (e["scope"], e["owner"], e["fixture"], e["phase"])
                self.scoped.setdefault(key, []).append((float(e["s"]), float(e["start"]), float(e["stop"]),
                                                        e.get("pid")))
        a.total += max(0.0, float(getattr(report, "duration", 0.0) or 0.0) - wide)
        st, sp = getattr(report, "start", None), getattr(report, "stop", None)
        if st is not None:
            a.start = st if a.start is None else min(a.start, st)
        if sp is not None:
            a.stop = sp if a.stop is None else max(a.stop, sp)
        a.pid = getattr(report, "gen3ai_pid", a.pid)
        if exempt:
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

    def finalize(self) -> None:
        """Judge every scope owner's row ONCE (idempotent; call after the last report). See SCOPE above."""
        if self._finalized:
            return
        self._finalized = True
        b = bar_s()
        owners: Dict[Tuple[str, str], Dict[Tuple[str, str], Tuple[float, float, float, Optional[int]]]] = {}
        for (scope, owner, fixture, phase), evs in self.scoped.items():
            worst = max(evs, key=lambda e: e[0])       # a max over workers: independent of the split
            owners.setdefault((scope, owner), {})[(fixture, phase)] = worst
        for (scope, owner), fx in sorted(owners.items()):
            total = sum(w[0] for w in fx.values())
            if total < b / 2:
                continue
            factor: Optional[float] = None
            desc = ""
            if total >= b:
                readings = []
                for (_f, _p), (s_, st, sp, pid) in fx.items():
                    if s_ < b / 8:
                        continue
                    try:
                        readings.append((s_, self.window(st, sp, pid)))
                    except Exception:            # noqa: BLE001 — an unreadable window is a WARN, never a crash
                        readings.append((s_, None))
                factors: List[Tuple[float, Optional[float]]] = [(s_, getattr(r, "factor", None))
                                                                for s_, r in readings]
                known = [(s_, f) for s_, f in factors if f is not None]
                if factors and len(known) == len(factors):
                    wsum = sum(s_ for s_, _ in known)
                    factor = sum(s_ * f for s_, f in known) / wsum if wsum else None
                big = max(readings, key=lambda x: x[0], default=None)
                if big is not None and big[1] is not None and hasattr(big[1], "describe"):
                    desc = big[1].describe()
                else:
                    desc = "contention UNREADABLE for this window"
            verdict = judge(total, factor, b)
            parts = ", ".join(f"{f}/{p} {w[0]:.1f}s"
                              for (f, p), w in sorted(fx.items(), key=lambda kv: -kv[1][0]))
            label = f"{owner} [{scope}-scoped fixtures: {parts}]"
            self.report_rows.append({"nodeid": label, "scope": scope, "owner": owner, "total_s": round(total, 3),
                                     "factor": factor, "verdict": verdict})
            if verdict != "ok":
                self.judged.append((label, verdict, total, factor, desc))

    @property
    def failed(self) -> List[tuple]:
        self.finalize()
        return [j for j in self.judged if j[1] == "fail"]

    def lines(self) -> List[str]:
        self.finalize()
        if not self.judged:
            return []
        b = bar_s()
        out = [f"{len(self.failed)} routine test(s) / scope owner(s) FAIL and "
               f"{len(self.judged) - len(self.failed)} WARN the "
               f"ship gate's {b:.0f} s bar (setup + call + teardown; bar x max(1, factor / {REF_FACTOR}); FAIL at "
               f">= {FAIL_LINE:g}x; the band below it, from {FAIL_MARGIN:g}x, is rule 8's exclusion margin — named, never a verdict):"]
        for nodeid, verdict, total, factor, desc in sorted(self.judged, key=lambda j: -j[2]):
            f = f"x{factor:.2f}" if factor is not None else "x?"
            out.append(f"  {total:7.1f}s  {verdict.upper():4}  {f}  {nodeid}")
            if desc:
                out.append(f"           window: {desc}")
        out.append("A row naming `[<scope>-scoped fixtures: ...]` is a SCOPE OWNER's shared fixture cost (charged to "
                   "its module / class / session, never to one test): mark the WHOLE file `slow`, or make the "
                   "fixture cheaper. Mark each other FAIL `slow` with a `# ship gate (<date>): N s beside a live "
                   "run` comment, or make it faster "
                   "(designs/ops/testing.md \"The SHIP GATE\" — the 8 s bar). A WARN that repeats is the same "
                   "test, earlier.")
        return out

    def write_report(self) -> Optional[str]:
        path = os.environ.get(REPORT_ENV)
        if not path:
            return None
        self.finalize()
        rows = sorted(self.report_rows, key=lambda r: -r["total_s"])
        with open(path, "w") as f:
            json.dump({"schema": "gen3_ship_bar_report_v1", "bar_s": bar_s(), "ref_factor": REF_FACTOR,
                       "fail_margin": FAIL_MARGIN, "fail_band": FAIL_BAND, "fail_line": FAIL_LINE, "rows": rows}, f, indent=1)
        return path
