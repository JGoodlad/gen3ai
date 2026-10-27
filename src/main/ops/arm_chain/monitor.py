"""``arm_chain monitor --chain FILE``: ONE long-running follower for the alert Monitor, so no re-arm is ever needed.

THE CLASS BUG (review 2026-10-26, finding 4): the alert Monitor was ``tail -F $(watch-files …) | grep -E <filter>``,
and ``watch-files`` lists the logs that exist when it is CALLED. Armed at the cut-over, before the first GO, it
followed ``status.txt`` alone: no run's Traceback / Child crashed / WEDGE line could ever page.

Each pass this follower re-reads the file set (:func:`chain_files`: the status file, each unfinished arm's fixed log
paths, and every live launcher's logs from ``live.discover``), follows every file ``tail -F`` style (:class:`Follower`)
and prints each line the chain's :func:`canary.monitor_filter` matches, prefixed ``[<file name>]``. It also probes the
supervisor (``liveness``) and, once per case, appends ``ARMCHAIN[<name>]: SUPERVISOR DEAD|STALE: …`` to the status
file (which the filter matches, so it prints in the same pass): a dead supervisor with an arm mid-flight pages.

It is bounded (``--max-hours``) and prints an ``EXPIRING`` line on its last pass. It never writes anything but that
alert line and its once-only key file (``<work_dir>/monitor_alerted.txt``).
"""
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Tuple

from main.ops.arm_chain import canary, live, liveness
from main.ops.arm_chain import spec as spec_mod
from main.ops.arm_chain.chain import TERMINAL, own_tag, watch_paths
from main.ops.arm_chain.statuslog import StatusError, StatusLog


def chain_files(spec: spec_mod.ChainSpec, state: Mapping[str, object], runs: List[live.LiveRun],
                alert: Callable[[str, str], None]) -> List[str]:
    """The status file, each unfinished arm's launcher log and ``launcher_child.full.log`` (fixed per item, whether
    or not they exist yet), and every live launcher's logs (a run outside the chain is watched too)."""
    items = state.get("items", {}) if isinstance(state, Mapping) else {}
    out = [spec.status_file]
    for it in spec.arms():
        rec = items.get(it.key) if isinstance(items, Mapping) else None
        if isinstance(rec, Mapping) and rec.get("phase") in TERMINAL:
            continue
        out += watch_paths(spec, it)
    out += canary.watch_files(runs, alert)
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


@dataclass
class _Tail:
    fd: int
    inode: int
    offset: int
    partial: bytes = b""


@dataclass
class Follower:
    """``tail -F`` over a file set that changes between polls. A file present at the FIRST poll starts at its end
    (``-n0``); one that appears later starts at 0 (its every line is new). A path whose inode changed (rotation by
    rename) has its old file drained first, then the new one read from 0; a file that shrank restarts at 0."""

    tails: Dict[str, _Tail] = field(default_factory=dict)
    polled: bool = False

    def _open(self, path: str, at_end: bool) -> Optional[_Tail]:
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            return None
        st = os.fstat(fd)
        return _Tail(fd, st.st_ino, st.st_size if at_end else 0)

    @staticmethod
    def _drain(t: _Tail) -> List[str]:
        chunks = []
        while True:
            b = os.pread(t.fd, 1 << 20, t.offset)
            if not b:
                break
            chunks.append(b)
            t.offset += len(b)
        data = t.partial + b"".join(chunks)
        *lines, t.partial = data.split(b"\n")
        return [ln.decode("utf-8", errors="replace") for ln in lines]

    def poll(self, paths: List[str]) -> List[Tuple[str, str]]:
        first = not self.polled
        self.polled = True
        out: List[Tuple[str, str]] = []
        for gone in [p for p in self.tails if p not in paths]:
            os.close(self.tails.pop(gone).fd)
        for p in paths:
            t = self.tails.get(p)
            if t is None:
                t = self._open(p, at_end=first)
                if t is None:
                    continue
                self.tails[p] = t
            try:
                st = os.stat(p)
            except OSError:
                st = None
            if st is not None and st.st_ino != t.inode:      # rotated: finish the old file, then the new from 0
                out += [(p, ln) for ln in self._drain(t)]
                os.close(t.fd)
                nt = self._open(p, at_end=False)
                if nt is None:
                    del self.tails[p]
                    continue
                self.tails[p] = t = nt
            elif os.fstat(t.fd).st_size < t.offset:          # truncated in place
                t.offset, t.partial = 0, b""
            out += [(p, ln) for ln in self._drain(t)]
        return out

    def close(self) -> None:
        for t in self.tails.values():
            os.close(t.fd)
        self.tails.clear()


def read_state(spec: spec_mod.ChainSpec) -> Dict[str, object]:
    try:
        return dict(json.loads(open(spec.state_path).read()))
    except (OSError, ValueError):
        return {}


def run_monitor(chain_path: str, out: Callable[[str], None] = print, interval_s: float = 10.0,
                max_hours: float = 24.0, proc: str = "/proc", sleep: Callable[[float], None] = time.sleep,
                now: Callable[[], float] = time.time, passes: Optional[int] = None,
                live_runs: Optional[Callable[[str], List[live.LiveRun]]] = None) -> int:
    spec = spec_mod.load(chain_path)
    discover = live_runs or (lambda md: live.discover(md, proc))
    fol = Follower()

    def emit(msg: str) -> None:
        try:
            StatusLog(spec.status_file, own_tag(spec)).append(msg)
        except (OSError, StatusError) as e:     # a full disk: the page still reaches the Monitor
            out(f"ARMCHAIN-MONITOR[{spec.name}]: {msg} (status file unwritable: {e})")

    total = passes if passes is not None else max(1, int(max_hours * 3600 / interval_s))
    out(f"ARMCHAIN-MONITOR[{spec.name}]: following the chain's files every {interval_s:g} s for "
        f"{total * interval_s / 3600:.1f} h (no re-arm at a GO)")
    try:
        # the files as they are NOW start at their end (``tail -n0``), before any alert this monitor writes
        fol.poll(chain_files(spec, read_state(spec), discover(spec.models_dir), lambda k, m: None))
        for i in range(total):
            try:
                spec = spec_mod.load(chain_path)     # items appended to the chain file are followed too
            except spec_mod.SpecError as e:
                out(f"ARMCHAIN-MONITOR[{spec.name}]: the chain file does not load ({e}); following the last good one")
            pattern = re.compile(canary.monitor_filter(spec.name))
            os.makedirs(spec.work_dir, exist_ok=True)
            alert = canary.Alerter(os.path.join(spec.work_dir, "monitor_alerted.txt"), emit)
            state = read_state(spec)
            lv = liveness.probe(state, spec.state_path + ".lock", now(), proc,
                                {it.key: os.path.join(spec.job_dir, f"{it.run}.pid") for it in spec.arms()})
            if lv.loud:
                alert(lv.key, f"SUPERVISOR {lv.verdict}: {lv.detail}")
            files = chain_files(spec, state, discover(spec.models_dir), alert)
            for path, line in fol.poll(files):
                if pattern.search(line):
                    out(f"[{os.path.basename(path)}] {line}")
            if i + 1 == total:
                out(f"ARMCHAIN-MONITOR[{spec.name}]: EXPIRING after {total} passes: re-arm it")
                break
            sleep(interval_s)
    finally:
        fol.close()
    return 0
