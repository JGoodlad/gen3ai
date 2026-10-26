"""Live-run discovery from the LAUNCHERS' own argv in ``/proc`` — never from a run-name glob.

THE CLASS BUG (orchestrator 2026-10-24 ~13:45): the canary logger and the alert watch listed runs by name patterns,
and three new families (``rb_x35t_*``, ``rb_x35_*``, ``rb_x26c_*``) were each missed until a version bump added
their glob. Here a live run is whatever a live ``python -m main.launcher`` (not ``--dry-run``) is running:

1. the run dir of a TRAINER descendant of that launcher (``train_rl_agent.py --run-dir D``) — authoritative;
2. else the launcher's argv: ``--run-dir``; ``--run-name`` under ``models/``; else a checkpoint-shaped ``--model``'s
   run (the RESTART case; a fork without ``--run-name`` would resolve to its PARENT, so this source is labelled);
3. else UNRESOLVED, reported with the reason — never dropped.

A trainer with no live launcher ancestor is reported too (an orphan, or a launch outside the launcher).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional

from utils import procfs

_TRAINER = "train_rl_agent" + ".py"


@dataclass(frozen=True)
class LiveRun:
    pid: int
    start_ticks: int
    elapsed_s: int
    run_dir: Optional[str]
    source: str           # "child" | "argv:--run-dir" | "argv:--run-name" | "argv:--model" | "orphan-trainer"
    reason: str = ""      # why UNRESOLVED (run_dir None)
    stdout: str = ""      # /proc/<pid>/fd/1 target

    def line(self) -> str:
        if self.run_dir is None:
            return f"{self.pid} {self.elapsed_s} UNRESOLVED {self.reason}"
        return f"{self.pid} {self.elapsed_s} {self.run_dir} [{self.source}]"


def _argv(pid: int, proc: str) -> List[str]:
    raw = procfs.read_text(pid, "cmdline", proc)
    return [a for a in raw.split("\0") if a]


def _flag(argv: List[str], name: str) -> Optional[str]:
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(name + "="):
            return a.split("=", 1)[1]
    return None


def is_launcher(argv: List[str]) -> bool:
    if "--dry-run" in argv:
        return False
    return any(a == "-m" and i + 1 < len(argv) and argv[i + 1] == "main.launcher" for i, a in enumerate(argv))


def _abs(p: str, pid: int, proc: str) -> str:
    if os.path.isabs(p):
        return p
    cwd = procfs.readlink(pid, "cwd", proc)
    return os.path.join(cwd, p) if cwd else p


def _uptime(proc: str) -> float:
    try:
        with open(os.path.join(proc, "uptime")) as f:
            return float(f.read().split()[0])
    except (OSError, ValueError, IndexError):
        return 0.0


def discover(models_dir: str, proc: str = "/proc") -> List[LiveRun]:
    procs = procfs.snapshot(proc)
    up = _uptime(proc)
    argvs = {pid: _argv(pid, proc) for pid in procs}
    launchers = [pid for pid, a in argvs.items() if is_launcher(a)]
    trainers: Dict[int, str] = {}
    for pid, a in argvs.items():
        if any(x.endswith(_TRAINER) for x in a[:3]):
            rd = _flag(a, "--run-dir")
            if rd:
                trainers[pid] = os.path.realpath(_abs(rd, pid, proc))
    owned: Dict[int, str] = {}
    for t, rd in trainers.items():
        for anc in procfs.ancestors(t, procs):
            if anc in launchers:
                owned.setdefault(anc, rd)
                break
    out: List[LiveRun] = []
    for pid in sorted(launchers):
        p = procs[pid]
        el = int(up - p.starttime / procfs.HZ) if up else 0
        a = argvs[pid]
        stdout = procfs.readlink(pid, "fd/1", proc)
        common = dict(pid=pid, start_ticks=p.starttime, elapsed_s=el, stdout=stdout)
        if pid in owned:
            out.append(LiveRun(run_dir=owned[pid], source="child", **common))  # type: ignore[arg-type]
            continue
        rd: Optional[str] = None
        src = ""
        if _flag(a, "--run-dir"):
            rd, src = _abs(_flag(a, "--run-dir") or "", pid, proc), "argv:--run-dir"
        elif _flag(a, "--run-name"):
            rd, src = os.path.join(models_dir, _flag(a, "--run-name") or ""), "argv:--run-name"
        elif _flag(a, "--model"):
            m = os.path.realpath(_abs(_flag(a, "--model") or "", pid, proc))
            if os.path.basename(os.path.dirname(m)) == "checkpoints":
                rd, src = os.path.dirname(os.path.dirname(m)), "argv:--model"
        if rd is None:
            out.append(LiveRun(run_dir=None, source="", reason="no trainer child yet and no --run-dir / --run-name / "
                               "checkpoint-shaped --model in the launcher's argv", **common))  # type: ignore[arg-type]
        elif not os.path.isdir(rd):
            out.append(LiveRun(run_dir=None, source=src, reason=f"{rd} is not a directory (yet)",
                               **common))  # type: ignore[arg-type]
        else:
            out.append(LiveRun(run_dir=os.path.realpath(rd), source=src, **common))  # type: ignore[arg-type]
    claimed = set(owned)
    for t, rd in sorted(trainers.items()):
        if not any(anc in claimed or anc in launchers for anc in procfs.ancestors(t, procs)):
            p = procs[t]
            out.append(LiveRun(pid=t, start_ticks=p.starttime, elapsed_s=int(up - p.starttime / procfs.HZ) if up else 0,
                               run_dir=rd, source="orphan-trainer", stdout=procfs.readlink(t, "fd/1", proc)))
    return out
