"""After the GO: did the launch take, is it the arm we registered, and is it running at a sane rate?

Three checks, each a HOLD on failure. A VOID banner raises :class:`Void` and a failed throughput gate
:class:`ThroughputFail`: the chain then applies the arm's REGISTERED action (``on_void`` / ``on_throughput_fail``:
``stop`` = SIGTERM the launcher through the identity gate, then HOLD; ``page`` = HOLD only).

* :func:`wait_spawned` — the launch wrapper's ``post-spawn pool-seed check PASS`` line for THIS run (the kit's own
  T54 check, which no dry run can preview), or its ``HOLD:`` line, or the wrapper gone without either;
* :func:`banner_guard` — the arm's registered startup banner in the child log (D3's ``[TEAM-MIX] d3_setmut_v1@ALL``),
  and none of its VOID lines (``[TEAM-MIX] FATAL``, ``does NOT inherit its parent's --team-mix``), by a deadline:
  a fork that silently dropped its lever trains the parent's recipe under the arm's name;
* :func:`throughput_gate` — within ``within_min`` of the FIRST TRAINER LINE (:func:`wait_first_trainer_line`; or
  the first own checkpoint), a MARGINAL steps/s from consecutive checkpoint mtimes (the SOP's standard meter; never
  the cumulative or integer-derived fps) of at least the arm's bar, from checkpoints this run wrote (step above the
  fork step). Counting from the GO failed healthy cold starts: ~22-23 min passed between the GO and the first
  trainer line (operator review 10-27).

The banner and throughput checks also watch the wrapper's END line (``ended``; review 10-26, finding 6): a run that
dies in its first hour is judged at once on its now-complete logs and checkpoints, never left to the gate's deadline.
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


class Void(Hold):
    """The arm's registered banner is VOID (a VOID line, or no expected banner by its deadline)."""


class ThroughputFail(Hold):
    """The first-hour throughput gate failed (or the trainer never printed its first line)."""


#: The launcher's own header in ``launcher_child.log`` (``child._open_child_log``): written at the SPAWN, before the
#: trainer prints anything, so it is never "the first trainer line".
_CHILD_HEADER = re.compile(r"^=+ child attached .*=+$")


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


Ended = Optional[Callable[[], Optional[str]]]


def banner_guard(child_logs: List[str], b: Banner, wait_for_log_s: float = 3600.0, poll_s: float = 10.0,
                 clock: Optional[Clock] = None, ended: Ended = None) -> str:
    """Wait (bounded) for the child log, then for the banner; a VOID line or no banner by the deadline is a HOLD. A
    run that ENDED is judged on its complete logs at once."""
    c = clock or Clock()
    t0 = c.monotonic()
    while not any(os.path.exists(p) for p in child_logs):
        e = ended() if ended else None
        if e and not any(os.path.exists(p) for p in child_logs):
            raise Void(f"VOID — the run ENDED ({e[:120]}) before it wrote a child log")
        if c.monotonic() - t0 > wait_for_log_s:
            raise Hold(f"no child log ({', '.join(os.path.basename(p) for p in child_logs)}) within "
                       f"{wait_for_log_s / 60:.0f} min of the GO — the banner cannot be checked")
        c.sleep(poll_s)
    t1 = c.monotonic()
    while True:
        e = ended() if ended else None   # read BEFORE the logs: an END seen here means the logs below are complete
        text = ""
        for p in child_logs:
            try:
                text += open(p, errors="replace").read() + "\n"
            except OSError:
                pass
        void, ok = _scan_banner(text, b)
        if void:
            raise Void(f"VOID — {void.strip()[:200]}")
        if ok:
            return ok.strip()[:200]
        if e:
            raise Void(f"VOID — the run ENDED ({e[:120]}) with no {b.expect!r} banner")
        if c.monotonic() - t1 > b.within_min * 60:
            raise Void(f"VOID — no {b.expect!r} banner in the child's first {b.within_min:g} min")
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


def first_trainer_line(child_log: str) -> Optional[str]:
    """The first line the TRAINER wrote to ``launcher_child.log`` (blank lines and the launcher's ``child attached``
    header skipped), or None. Reads at most the first 64 KiB."""
    try:
        with open(child_log, errors="replace") as f:
            head = f.read(65536)
    except OSError:
        return None
    for ln in head.splitlines():
        if ln.strip() and not _CHILD_HEADER.match(ln.strip()):
            return ln
    return None


def wait_first_trainer_line(child_log: str, run_dir: str, fork_step: int, deadline_s: float,
                            now: Callable[[], float] = time.time, poll_s: float = 30.0,
                            clock: Optional[Clock] = None, ended: Ended = None) -> Tuple[float, str]:
    """``(epoch, what)`` of the first trainer line — the wall time it was SEEN (late by at most ``poll_s``) — or of
    the first own checkpoint's mtime when one exists first. ``deadline_s`` (from now) with neither is a
    :class:`ThroughputFail`."""
    c = clock or Clock()
    t0 = c.monotonic()
    while True:
        e = ended() if ended else None
        cks = new_checkpoints(run_dir, fork_step)
        if cks:
            return min(t for _, t in cks), f"the first own checkpoint (step {cks[0][0]:,})"
        ln = first_trainer_line(child_log)
        if ln is not None:
            return now(), f"the first trainer line ({ln.strip()[:80]!r})"
        if e:
            raise ThroughputFail(f"the run ENDED ({e[:120]}) with no trainer line and no own checkpoint")
        if c.monotonic() - t0 > deadline_s:
            raise ThroughputFail(f"no trainer line in {os.path.basename(child_log)} and no own checkpoint "
                                 f"{deadline_s / 60:.0f} min on: the trainer never started")
        c.sleep(poll_s)


def throughput_gate(run_dir: str, fork_step: int, tp: Throughput, poll_s: float = 120.0,
                    clock: Optional[Clock] = None, already_s: float = 0.0, ended: Ended = None) -> str:
    """PASS as soon as one consecutive pair of the run's own checkpoints reaches the bar; :class:`ThroughputFail` at
    the deadline, ``tp.within_min`` after the FIRST TRAINER LINE (``already_s`` = how long ago that line was)."""
    c = clock or Clock()
    t0 = c.monotonic()
    while True:
        e = ended() if ended else None
        rates = marginal_rates(new_checkpoints(run_dir, fork_step))
        if rates and max(rates) >= tp.min_steps_per_s:
            return f"marginal {max(rates):,.0f} steps/s >= {tp.min_steps_per_s:,.0f} (checkpoint mtimes, {len(rates)} pair(s))"
        best = f"best {max(rates):,.0f} steps/s over {len(rates)} pair(s)" if rates else "fewer than two new checkpoints"
        if e:
            raise ThroughputFail(f"the run ENDED ({e[:120]}) before the throughput gate passed: {best} "
                                 f"< {tp.min_steps_per_s:,.0f}")
        if c.monotonic() - t0 + already_s > tp.within_min * 60:
            raise ThroughputFail(f"the throughput gate failed {tp.within_min:g} min after the first trainer line: {best} "
                                 f"< {tp.min_steps_per_s:,.0f}")
        c.sleep(poll_s)
