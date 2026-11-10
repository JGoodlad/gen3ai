"""A planted ``/proc`` for the tests of the live-run rule (``utils.core_split.split_runs`` and every reader of it).

Test support only (like ``utils.lock_listing_testing``): production code never imports it. The tests of the heavy slots,
the gate lock's warning, the ship gate, the launch kit's pre-check and the arm chain's second-GO refusal all need "a box
with an agent's training smoke on it", and each must be able to fail deterministically when its reader goes back to the
raw ``procfs.live_training_runs`` - so they share ONE box (:func:`incident_box`, the 2026-11-09 incident) and one way to
make the readers see it (:func:`plant`).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import pytest

from utils import core_split, procfs

#: ``{pid: (ppid, argv, CUDA_VISIBLE_DEVICES or None)}`` as ``/proc`` would show it (``None``: the variable is unset).
Table = Dict[int, Tuple[int, List[str], Optional[str]]]
Affinity = Callable[[int], Optional[List[int]]]

TRAINER_SCRIPT = "train_rl_agent" + ".py"
ONLINE = list(range(16))
TRAINER = list(core_split.DEFAULT.trainer_ids)
AGENTS = list(core_split.DEFAULT.agent_ids)
#: the run dirs of the two runs of :func:`incident_box`: the agent's CPU smoke, and the pinned GPU run
SMOKE, REAL = "/m/smoke_b1", "/m/rb_x"
#: their trainer pids
SMOKE_PID, REAL_PID = 111, 201


def fake_proc(root: Path, table: Table) -> str:
    """Write ``table`` under ``root`` as a fake ``/proc`` (``cmdline``, ``stat``, ``environ`` per pid); returns its path."""
    root.mkdir(parents=True, exist_ok=True)
    for pid, (ppid, argv, cuda) in table.items():
        d = root / str(pid)
        d.mkdir()
        (d / "cmdline").write_bytes("\0".join(argv).encode() + b"\0")
        (d / "stat").write_text(f"{pid} (python3) S {ppid} {pid} {pid} " + " ".join(["0"] * 30) + "\n")
        env = ["LANG=C"] + ([] if cuda is None else [f"CUDA_VISIBLE_DEVICES={cuda}"])
        (d / "environ").write_bytes("\0".join(env).encode() + b"\0")
    return str(root)


def incident_box(root: Path, *, smoke_cuda: Optional[str] = "0", real: bool = True) -> str:
    """The incident's box: an agent's training smoke (wrapper 100 > launcher 110 > trainer 111 > worker 112) and,
    with ``real``, the pinned GPU run (launcher 200 > trainer 201)."""
    table: Table = {
        100: (1, ["python3", "-m", "main.b1.resume_check", "smoke"], smoke_cuda),
        110: (100, ["python3", "-m", "main.launcher", "--debug", "--steps", "10000"], smoke_cuda),
        111: (110, ["python3", f"/wt/src/main/{TRAINER_SCRIPT}", "--run-dir", SMOKE], smoke_cuda),
        112: (111, ["python3", "-c", "from multiprocessing.spawn import spawn_main"], smoke_cuda)}
    if real:
        table[200] = (1, ["python3", "-m", "main.launcher", "--steps", "15000000", "--device", "cuda"], "0")
        table[201] = (200, ["python3", f"/repo/src/main/{TRAINER_SCRIPT}", "--run-dir", REAL], "0")
    return fake_proc(root, table)


def affinities(smoke: Sequence[int] = tuple(AGENTS), **over: Optional[Sequence[int]]) -> Affinity:
    """Each fake process's affinity: the smoke's tree on ``smoke`` (the agent cores), the GPU run on the trainer's.
    ``p<pid>=`` overrides one (``None``: unreadable)."""
    table: Dict[int, Optional[List[int]]] = {100: list(ONLINE), 110: list(smoke), 111: list(smoke), 112: list(smoke),
                                             200: list(TRAINER), 201: list(TRAINER)}
    table.update({int(k[1:]): (None if v is None else list(v)) for k, v in over.items()})
    return lambda pid: table[pid]


#: the real readers, captured at import (before any test patches them), so a test may plant a second box
_RAW, _RULE = procfs.live_training_runs, core_split.training_runs


def plant(monkeypatch: pytest.MonkeyPatch, proc: str, affinity: Affinity) -> None:
    """Make the box's live-run readers see ``proc``: the raw ``procfs.live_training_runs`` AND the rule
    (``core_split.training_runs``) read the planted tree, with ``affinity``. A reader that is changed back to the raw one
    then reports the smoke as a run, which is how its test fails on a revert (never by reading the real ``/proc``)."""
    monkeypatch.setattr(procfs, "live_training_runs", lambda p="/proc": _RAW(proc))
    monkeypatch.setattr(core_split, "training_runs", lambda p="/proc", a=None: _RULE(proc, affinity))
