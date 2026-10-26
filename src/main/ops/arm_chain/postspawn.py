"""After the GO: did the launch take, is it the arm we registered, and is it running at a sane rate?

Three checks, each a HOLD on failure (the chain never signals a run; a HOLD pages the orchestrator, who decides):

* :func:`wait_spawned` — the launch wrapper's ``post-spawn pool-seed check PASS`` line for THIS run (the kit's own
  T54 check, which no dry run can preview), or its ``HOLD:`` line, or the wrapper gone without either;
* :func:`banner_guard` — the arm's registered startup banner in the child log (D3's ``[TEAM-MIX] d3_setmut_v1@ALL``),
  and none of its VOID lines (``[TEAM-MIX] FATAL``, ``does NOT inherit its parent's --team-mix``), by a deadline:
  a fork that silently dropped its lever trains the parent's recipe under the arm's name;
* :func:`throughput_gate` — within the first hour, a MARGINAL steps/s from consecutive checkpoint mtimes (the SOP's
  standard meter; never the cumulative or integer-derived fps) of at least the arm's bar, from checkpoints this run
  wrote (step above the fork step).
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

from main.ops.arm_chain.launch import Hold
from main.ops.arm_chain.procwatch import ProcRef, alive
from main.ops.arm_chain.spec import Banner, Throughput
from main.ops.arm_chain.statuslog import StatusLog, Trigger

_CKPT = re.compile(r"checkpoint_(\d+)_steps\.zip")


@dataclass
class Clock:
    sleep: Callable[[float], None] = time.sleep
    monotonic: Callable[[], float] = time.monotonic


def wait_spawned(log: StatusLog, passed: Trigger, held: Trigger, wrapper: ProcRef, timeout_s: float,
                 poll_s: float = 15.0, proc: str = "/proc", clock: Optional[Clock] = None) -> str:
    c = clock or Clock()
    t0 = c.monotonic()
    while True:
        m = log.find(passed)
        if m:
            return m.group(0)
        h = log.find(held)
        if h:
            raise Hold(f"the launch wrapper HELD: {h.group(1)[:220]}")
        if not alive(wrapper, proc):
            raise Hold(f"the launch wrapper (pid {wrapper.pid}) is gone without a pool-seed PASS or a HOLD line")
        if c.monotonic() - t0 > timeout_s:
            raise Hold(f"no post-spawn pool-seed PASS within {timeout_s / 60:.0f} min of the GO")
        c.sleep(poll_s)


def _scan_banner(text: str, b: Banner) -> Tuple[Optional[str], Optional[str]]:
    """(void line, expect line) found in ``text``; VOID wins when both are present."""
    void = next((ln for ln in text.splitlines() if any(v in ln for v in b.void)), None)
    ok = next((ln for ln in text.splitlines() if b.expect in ln), None)
    return void, ok


def banner_guard(child_logs: List[str], b: Banner, wait_for_log_s: float = 3600.0, poll_s: float = 10.0,
                 clock: Optional[Clock] = None) -> str:
    """Wait (bounded) for the child log, then for the banner; a VOID line or no banner by the deadline is a HOLD."""
    c = clock or Clock()
    t0 = c.monotonic()
    while not any(os.path.exists(p) for p in child_logs):
        if c.monotonic() - t0 > wait_for_log_s:
            raise Hold(f"no child log ({', '.join(os.path.basename(p) for p in child_logs)}) within "
                       f"{wait_for_log_s / 60:.0f} min of the GO — the banner cannot be checked")
        c.sleep(poll_s)
    t1 = c.monotonic()
    while True:
        text = ""
        for p in child_logs:
            try:
                text += open(p, errors="replace").read() + "\n"
            except OSError:
                pass
        void, ok = _scan_banner(text, b)
        if void:
            raise Hold(f"VOID — {void.strip()[:200]}")
        if ok:
            return ok.strip()[:200]
        if c.monotonic() - t1 > b.within_min * 60:
            raise Hold(f"VOID — no {b.expect!r} banner in the child's first {b.within_min:g} min")
        c.sleep(poll_s)


def new_checkpoints(run_dir: str, fork_step: int) -> List[Tuple[int, float]]:
    """``(step, mtime)`` of the run's own checkpoints (step > the fork step), oldest first."""
    ck = os.path.join(run_dir, "checkpoints")
    out = []
    try:
        names = os.listdir(ck)
    except OSError:
        return []
    for n in names:
        m = _CKPT.fullmatch(n)
        if m and int(m.group(1)) > fork_step:
            out.append((int(m.group(1)), os.path.getmtime(os.path.join(ck, n))))
    return sorted(out)


def marginal_rates(cks: List[Tuple[int, float]]) -> List[float]:
    return [(s1 - s0) / (t1 - t0) for (s0, t0), (s1, t1) in zip(cks, cks[1:]) if t1 > t0]


def throughput_gate(run_dir: str, fork_step: int, tp: Throughput, poll_s: float = 120.0,
                    clock: Optional[Clock] = None, already_s: float = 0.0) -> str:
    """PASS as soon as one consecutive pair of the run's own checkpoints reaches the bar; HOLD at the deadline,
    ``tp.within_min`` after the GO (``already_s`` = how long ago the GO was)."""
    c = clock or Clock()
    t0 = c.monotonic()
    while True:
        rates = marginal_rates(new_checkpoints(run_dir, fork_step))
        if rates and max(rates) >= tp.min_steps_per_s:
            return f"marginal {max(rates):,.0f} steps/s >= {tp.min_steps_per_s:,.0f} (checkpoint mtimes, {len(rates)} pair(s))"
        if c.monotonic() - t0 + already_s > tp.within_min * 60:
            best = f"best {max(rates):,.0f} steps/s over {len(rates)} pair(s)" if rates else "fewer than two new checkpoints"
            raise Hold(f"the first-{tp.within_min:g}-min throughput gate failed: {best} < {tp.min_steps_per_s:,.0f}")
        c.sleep(poll_s)
