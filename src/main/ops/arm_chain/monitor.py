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

IT NEVER BLOCKS THE ARM END (the 10-27 03:14 incident: the thin planner refuses while ANY process has the run dir open,
and the first ``monitor`` held ``launcher_child.full.log`` open for its whole life, so d3 3504's thin HELD and the GPU
idled). Two rules, each pinned by ``monitor_test``:

- **No file is held open between reads.** Each pass opens, reads to EOF and closes every file; the follower keeps only
  ``(dev, inode, offset)``. A rotation by rename is followed by finding the old generation by its inode among the
  path's siblings (``launcher_child.full.log.1``), so ``tail -F``'s "finish the old file" survives without a handle.
- **An arm past its run phase (ENDED or later) loses its run-dir log** (:data:`POST_RUN`): one last drain on the
  first pass that reads the phase, then the monitor never opens a file inside that run dir again, unless
  ``live.discover`` finds a launcher in it (a by-hand resume: the run is running again, and the thin refuses a run a
  live launcher names whatever the monitor does). The thin, the K9(b) read, the ship and any delete never see it.
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
from main.ops.arm_chain.chain import ARM_PHASES, TERMINAL, own_tag, watch_paths
from main.ops.arm_chain.statuslog import StatusError, StatusLog


#: The phases at and after the arm's run ended: the monitor follows no file inside such an arm's run dir (the thin
#: refuses while any process has the run dir open; 10-27 03:14).
POST_RUN = tuple(dict.fromkeys(ARM_PHASES[ARM_PHASES.index("ENDED"):] + TERMINAL))


def _inside(path: str, dirs: List[str]) -> bool:
    r = os.path.realpath(path)
    return any(r == d or r.startswith(d + os.sep) for d in dirs)


def chain_files(spec: spec_mod.ChainSpec, state: Mapping[str, object], runs: List[live.LiveRun],
                alert: Callable[[str, str], None]) -> List[str]:
    """The status file, each unfinished arm's launcher log and ``launcher_child.full.log`` (fixed per item, whether
    or not they exist yet), and every live launcher's logs (a run outside the chain is watched too). An arm at a
    :data:`POST_RUN` phase contributes NO path inside its run dir (the arm end's thin refuses while any process holds
    that dir); only a live launcher there, which the thin refuses anyway, brings its log back."""
    items = state.get("items", {}) if isinstance(state, Mapping) else {}
    out = [spec.status_file]
    for it in spec.arms():
        rec = items.get(it.key) if isinstance(items, Mapping) else None
        phase = rec.get("phase") if isinstance(rec, Mapping) else None
        if phase in TERMINAL:
            continue
        run_dir = os.path.realpath(os.path.join(spec.models_dir, it.run))
        out += [p for p in watch_paths(spec, it) if not (phase in POST_RUN and _inside(p, [run_dir]))]
    out += canary.watch_files(runs, alert)
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


@dataclass
class _Pos:
    """Where a followed path was read up to: the file's identity, the byte offset, and an unterminated last line."""
    dev: int
    inode: int
    offset: int
    partial: bytes = b""


def _drain(fd: int, pos: _Pos, final: bool = False) -> List[str]:
    chunks = []
    while True:
        b = os.pread(fd, 1 << 20, pos.offset)
        if not b:
            break
        chunks.append(b)
        pos.offset += len(b)
    data = pos.partial + b"".join(chunks)
    *lines, pos.partial = data.split(b"\n")
    if final and pos.partial:                  # a generation that will never grow again: its last line is whole
        lines.append(pos.partial)
        pos.partial = b""
    return [ln.decode("utf-8", errors="replace") for ln in lines]


@dataclass
class Follower:
    """``tail -F`` over a file set that changes between polls, holding NO file open between reads (each poll opens,
    reads to EOF and closes). A file present at the FIRST poll starts at its end (``-n0``); one that appears later
    starts at 0 (its every line is new). A path whose inode changed (rotation by rename) has its old generation
    drained first, found by inode among the path's siblings, then the new one read from 0; a file that shrank
    restarts at 0. A path dropped from the set is drained one last time, then forgotten."""

    pos: Dict[str, _Pos] = field(default_factory=dict)
    polled: bool = False

    @staticmethod
    def _rotated(path: str, pos: _Pos) -> List[str]:
        """The rest of the old generation of ``path`` (renamed beside it, ``<name>.1``), read without keeping it open.
        Gone (deleted, or moved out of the directory): its unread tail is lost, the one thing a held fd would keep."""
        d, base = os.path.split(path)
        try:
            names = [e.name for e in os.scandir(d or ".") if e.name.startswith(base + ".") and e.inode() == pos.inode]
        except OSError:
            return []
        for n in names:
            try:
                fd = os.open(os.path.join(d, n), os.O_RDONLY)
            except OSError:
                continue
            try:
                st = os.fstat(fd)
                if (st.st_dev, st.st_ino) == (pos.dev, pos.inode):
                    return _drain(fd, pos, final=True)
            finally:
                os.close(fd)
        return []

    def _read(self, path: str, at_end: bool) -> List[str]:
        pos = self.pos.get(path)
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            return []                          # absent for now; a rotation's old generation is drained at reappearance
        try:
            st = os.fstat(fd)
            out: List[str] = []
            if pos is None:
                pos = _Pos(st.st_dev, st.st_ino, st.st_size if at_end else 0)
            elif (st.st_dev, st.st_ino) != (pos.dev, pos.inode):   # rotated: finish the old file, then the new from 0
                out += self._rotated(path, pos)
                pos = _Pos(st.st_dev, st.st_ino, 0)
            elif st.st_size < pos.offset:                           # truncated in place
                pos.offset, pos.partial = 0, b""
            out += _drain(fd, pos)
            self.pos[path] = pos
            return out
        finally:
            os.close(fd)

    def poll(self, paths: List[str]) -> List[Tuple[str, str]]:
        first = not self.polled
        self.polled = True
        out: List[Tuple[str, str]] = []
        for gone in [p for p in self.pos if p not in paths]:   # one last read, then never opened again
            out += [(gone, ln) for ln in self._read(gone, at_end=False)]
            del self.pos[gone]
        for p in paths:
            out += [(p, ln) for ln in self._read(p, at_end=first)]
        return out


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
    fol = Follower()      # holds no file open between passes (it can never block a thin, a ship or a delete)

    def emit(msg: str) -> None:
        try:
            StatusLog(spec.status_file, own_tag(spec)).append(msg)
        except (OSError, StatusError) as e:     # a full disk: the page still reaches the Monitor
            out(f"ARMCHAIN-MONITOR[{spec.name}]: {msg} (status file unwritable: {e})")

    total = passes if passes is not None else max(1, int(max_hours * 3600 / interval_s))
    out(f"ARMCHAIN-MONITOR[{spec.name}]: following the chain's files every {interval_s:g} s for "
        f"{total * interval_s / 3600:.1f} h (no re-arm at a GO)")
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
    return 0
