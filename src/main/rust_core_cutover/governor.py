"""The live training arm's MARGINAL fps (read-only), and the stress's PRE-REGISTERED throttle.

**What is read.** Every live ``train_rl_agent.py`` process (found in ``/proc``; its ``--run-dir``
names the run) and that run's ``launcher_child.log`` — never written, never signalled. Each PPO
iteration prints ``iterations`` / ``time_elapsed`` / ``total_timesteps``; a launcher child restart
prints ``===== child attached … (pid N) =====`` and restarts ``time_elapsed``. An iteration's
marginal fps is ``Δtotal_timesteps / Δtime_elapsed`` between consecutive rows of ONE child (a
98,304-step rollout at ~500 fps is ~200 s, so the integer seconds cost ~0.5%); the first
:data:`SKIP_FIRST` iterations of a child are dropped (startup, compile, warm-up).

**Wall time of an iteration.** ``time_elapsed`` counts from the child's ``learn()``, whose wall
origin the log does not print; it is estimated per child as ``min(seen_wall − time_elapsed)`` over
the rows this governor watched appear (polled every few seconds, so the estimate is late by at most
one poll). A row whose wall interval cannot be placed is not classified.

**Classification.** The driver appends ``(t, running units, cap)`` to ``timeline.jsonl`` at every
change. An iteration is ``off`` if no stress unit ran at any point of its wall interval, ``on`` if
at least one ran throughout, ``mixed`` otherwise (not used).

**The throttle (pre-registered 2026-09-24, before the stress started).** Per run: the BASELINE is
the median fps of its most recent :data:`MIN_OFF` or more ``off`` iterations (an OFF window); the
ON reading is the median of its last :data:`ON_WINDOW` ``on`` iterations completed since the last
cap change. If ON < :data:`THROTTLE_RATIO` × BASELINE (a drop of more than 15%), the cap is cut by
:data:`CUT_STEP` (floor :data:`CAP_FLOOR`) and judged again only after :data:`ON_WINDOW` more ``on``
iterations. The governor never RAISES the cap (a raise is a written decision in ``control.json``).
OFF windows: :data:`FIRST_OFF_S` at the driver's first start (the pre-stress measurement), then
:data:`OFF_S` every :data:`OFF_EVERY_S`, and immediately when a run with no baseline appears (an arm
switch). During an OFF window the driver starts no unit; running units finish (they are minutes long)
and the window's clock starts when the last one has exited.
"""
from __future__ import annotations

import json
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

SKIP_FIRST = 2
MIN_OFF = 4
ON_WINDOW = 5
THROTTLE_RATIO = 0.85
CUT_STEP = 2
CAP_FLOOR = 2
FIRST_OFF_S = 15 * 60
OFF_S = 20 * 60
OFF_EVERY_S = 4 * 3600

_ATTACH = re.compile(r"===== child attached (\S+ \S+) \(pid (\d+)\) =====")
_ROW = re.compile(r"\|\s+(iterations|time_elapsed|total_timesteps)\s+\|\s+(\d+)\s+\|")


@dataclass
class Iteration:
    child: int            # the child pid the row belongs to (0 before any attach line is seen)
    iteration: int
    elapsed: int
    steps: int


def parse_log(text: str) -> List[Iteration]:
    """Every complete ``(iterations, time_elapsed, total_timesteps)`` triple, in log order, tagged
    with the child that printed it."""
    out: List[Iteration] = []
    child = 0
    cur: Dict[str, int] = {}
    for line in text.splitlines():
        m = _ATTACH.search(line)
        if m:
            child, cur = int(m.group(2)), {}
            continue
        m = _ROW.search(line)
        if not m:
            continue
        cur[m.group(1)] = int(m.group(2))
        if m.group(1) == "total_timesteps" and {"iterations", "time_elapsed"} <= cur.keys():
            out.append(Iteration(child, cur["iterations"], cur["time_elapsed"], cur["total_timesteps"]))
            cur = {}
    return out


def marginals(its: Sequence[Iteration]) -> List[Tuple[int, int, int, float]]:
    """``(child, elapsed_start, elapsed_end, fps)`` per consecutive pair of one child's rows,
    minus the first :data:`SKIP_FIRST` iterations of each child."""
    out = []
    by_child: Dict[int, List[Iteration]] = {}
    for it in its:
        by_child.setdefault(it.child, []).append(it)
    for child, rows in by_child.items():
        rows = [r for r in rows if r.iteration > SKIP_FIRST]
        for a, b in zip(rows, rows[1:]):
            if b.elapsed > a.elapsed and b.steps > a.steps and b.iteration == a.iteration + 1:
                out.append((child, a.elapsed, b.elapsed, (b.steps - a.steps) / (b.elapsed - a.elapsed)))
    return out


def live_runs() -> Dict[str, int]:
    """``{run_dir (absolute): pid}`` of every live ``train_rl_agent.py`` process."""
    out = {}
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            argv = (p / "cmdline").read_bytes().split(b"\0")
            cwd = (p / "cwd").resolve()
        except OSError:
            continue
        args = [a.decode(errors="replace") for a in argv if a]
        if not any(a.endswith("train_rl_agent.py") for a in args[:3]) or "--run-dir" not in args:
            continue
        rd = Path(args[args.index("--run-dir") + 1])
        out[str(rd if rd.is_absolute() else cwd / rd)] = int(p.name)
    return out


def classify(t0: float, t1: float, timeline: Sequence[Tuple[float, int]]) -> str:
    """``off`` / ``on`` / ``mixed`` for a wall interval, from ``(t, running)`` change points."""
    running_at = [n for t, n in timeline if t <= t0][-1:] or [0]
    levels = running_at + [n for t, n in timeline if t0 < t < t1]
    if all(n == 0 for n in levels):
        return "off"
    if all(n > 0 for n in levels):
        return "on"
    return "mixed"


@dataclass
class RunState:
    #: per child pid: min(seen_wall - elapsed), the learn() wall origin estimate
    origin: Dict[int, float] = field(default_factory=dict)
    seen: int = 0
    #: classified iterations: [child, elapsed_end, wall_end, fps, class]
    rows: List[list] = field(default_factory=list)
    last_cut_at: float = 0.0


class Governor:
    """Polls the live runs, classifies their iterations, and decides the cap."""

    def __init__(self, out: Path):
        self.path = out / "governor.json"
        self.state: Dict[str, RunState] = {}
        self.events: List[dict] = []
        if self.path.exists():
            d = json.loads(self.path.read_text())
            for run, s in d.get("runs", {}).items():
                self.state[run] = RunState(origin={int(k): v for k, v in s["origin"].items()},
                                           seen=s["seen"], rows=s["rows"], last_cut_at=s["last_cut_at"])
            self.events = d.get("events", [])

    def save(self) -> None:
        d = {"runs": {r: {"origin": s.origin, "seen": s.seen, "rows": s.rows,
                          "last_cut_at": s.last_cut_at} for r, s in self.state.items()},
             "events": self.events[-500:]}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(d))
        tmp.replace(self.path)

    def poll(self, timeline: Sequence[Tuple[float, int]], now: Optional[float] = None) -> None:
        now = time.time() if now is None else now
        for run in live_runs():
            log = Path(run) / "launcher_child.log"
            try:
                its = parse_log(log.read_text(errors="replace"))
            except OSError:
                continue
            st = self.state.setdefault(run, RunState())
            # Only the LIVE child (the last one the log names) can be placed on the wall clock:
            # rows of an earlier child were written before this poll could see them, so
            # `now - elapsed` says nothing about when (the ring log also trims attach lines).
            live = its[-1].child if its else None
            its = [it for it in its if it.child == live]
            for it in its:
                o = now - it.elapsed
                if it.child not in st.origin or o < st.origin[it.child]:
                    st.origin[it.child] = o
            have = {(r[0], r[1]) for r in st.rows}
            for child, e0, e1, fps in marginals(its):
                if (child, e1) in have or child not in st.origin:
                    continue
                st.rows.append([child, e1, 0.0, round(fps, 2), "", e0])
            # (Re)place every recent row with its child's CURRENT origin estimate: the estimate only
            # moves earlier as rows are seen closer to their writing, so a row first placed from a
            # late sighting is corrected on the next poll (bounded to the last 2 h of rows).
            for r in st.rows:
                if len(r) < 6 or r[0] not in st.origin:
                    continue
                w0, w1 = st.origin[r[0]] + r[5], st.origin[r[0]] + r[1]
                if r[4] and now - w1 > 7200:
                    continue
                r[2], r[4] = round(w1, 1), classify(w0, w1, timeline)
            st.rows = st.rows[-2000:]

    def baseline(self, run: str) -> Optional[float]:
        off = [r[3] for r in self.state[run].rows if r[4] == "off"] if run in self.state else []
        return statistics.median(off[-8:]) if len(off) >= MIN_OFF else None

    def on_reading(self, run: str) -> Optional[float]:
        st = self.state.get(run)
        if st is None:
            return None
        on = [r[3] for r in st.rows if r[4] == "on" and r[2] > st.last_cut_at]
        return statistics.median(on[-ON_WINDOW:]) if len(on) >= ON_WINDOW else None

    def needs_baseline(self) -> List[str]:
        return [r for r in live_runs() if self.baseline(r) is None]

    def decide(self, cap: int, now: Optional[float] = None) -> int:
        """The cap after applying the throttle to every live run (never above ``cap``)."""
        now = time.time() if now is None else now
        new = cap
        for run in live_runs():
            base, on = self.baseline(run), self.on_reading(run)
            if base is None or on is None or on >= THROTTLE_RATIO * base:
                continue
            cut = max(CAP_FLOOR, new - CUT_STEP)
            self.events.append({"t": now, "run": run, "baseline": base, "on": on,
                                "ratio": round(on / base, 3), "cap_from": new, "cap_to": cut})
            self.state[run].last_cut_at = now
            new = cut
        return new

    def report(self) -> List[dict]:
        out = []
        for run, st in self.state.items():
            off = [r[3] for r in st.rows if r[4] == "off"]
            on = [r[3] for r in st.rows if r[4] == "on"]
            row = {"run": run, "off_n": len(off), "on_n": len(on),
                   "off_median": round(statistics.median(off), 1) if off else None,
                   "on_median": round(statistics.median(on), 1) if on else None}
            if off and on:
                row["on_over_off"] = round(statistics.median(on) / statistics.median(off), 3)
            out.append(row)
        return out
