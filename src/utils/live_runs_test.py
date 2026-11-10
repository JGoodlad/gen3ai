"""THE LIVE-RUN QUESTION OF RULE 5b: an agent's own CPU-only run is agent work, not a training run (``utils.core_split``).

2026-11-09 the ship gate refused beside an agent's smoke (it read the smoke as a second trainer). 2026-11-10 the same
defect was found in the other readers of "is a training run live": the heavy slots (``wanted``: every heavy job took a
slot beside a smoke), the gate lock's warning, the launch kit's pre-check (``kit live`` HELD a launch) and the arm chain's
second-GO refusal. The rule is ONE (``core_split.split_runs`` / ``training_runs`` / ``agent_run_pids``), and each reader
asks it. Every reader test plants the incident's box (``utils.fake_proc_testing``: an agent's smoke beside the pinned
GPU run) so that a reader changed back to the raw ``procfs.live_training_runs`` fails DETERMINISTICALLY.

Test seams are the planted ``/proc`` and an ``affinity`` function: nothing here reads the real box.
"""
from __future__ import annotations

import ast
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pytest

from main.ops.arm_chain import live
from main.ops.arm_chain.live_test import FakeProc, launcher_argv
from main.x35 import kit
from utils import core_split
from utils import fake_proc_testing as FP
from utils import gate_lock as GL
from utils import heavy_slots as HS
from utils.paths import repo_path


@pytest.fixture(autouse=True)
def _no_lease(monkeypatch: pytest.MonkeyPatch) -> None:
    """The box's real GPU lease must not decide a unit test's answer."""
    monkeypatch.setattr(HS, "lease_reason", lambda: None)


# --- the rule ---------------------------------------------------------------------------------------------------------
def test_the_smoke_is_agent_work_and_the_pinned_gpu_run_is_not(tmp_path: Path) -> None:
    proc = FP.incident_box(tmp_path)
    runs = {FP.SMOKE: FP.SMOKE_PID, FP.REAL: FP.REAL_PID}
    assert core_split.split_runs(runs, proc=proc, affinity=FP.affinities()) == (
        {FP.REAL: FP.REAL_PID}, {FP.SMOKE: FP.SMOKE_PID})
    assert core_split.agent_run_pids([FP.SMOKE_PID, FP.REAL_PID, 110], proc=proc, affinity=FP.affinities()) == {
        FP.SMOKE_PID, 110}, "a LAUNCHER pid is classified by its tree too (the arm chain asks by launcher)"
    assert core_split.agent_run_pids([], proc=proc, affinity=FP.affinities()) == set()


@pytest.mark.parametrize("over,why", [
    ({"p112": FP.ONLINE}, "a worker of the smoke runs on the trainer's cores"),
    ({"p110": FP.ONLINE}, "its launcher is unpinned: the next child inherits every CPU"),
    ({"p111": None}, "the trainer's affinity cannot be read"),
    ({"p112": None}, "a LIVE descendant's affinity cannot be read"),
    ({"p111": [], "p110": [], "p112": []}, "an empty affinity proves nothing"),
])
def test_a_run_counts_as_training_unless_its_WHOLE_tree_sits_inside_the_agent_list(
        tmp_path: Path, over: Dict[str, Optional[Sequence[int]]], why: str) -> None:
    proc = FP.incident_box(tmp_path, real=False)
    runs = {FP.SMOKE: FP.SMOKE_PID}
    assert core_split.split_runs(runs, proc=proc, affinity=FP.affinities(**over)) == (runs, {}), why
    (tmp_path / "112" / "stat").unlink()                         # the descendant EXITED after it was listed: no evidence
    gone = core_split.split_runs(runs, proc=proc, affinity=FP.affinities(p112=None))
    assert gone == ({}, runs)


def test_the_agent_list_is_the_ACTIVE_splits_read_from_the_owner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One rule, the owner's (`core_split.is_agent_work`): a smoke on the 4+4 agent cores is agent work only while
    4+4 is the active split. Fails if the rule spells its own list or its own subset test."""
    proc, four = FP.incident_box(tmp_path, real=False), core_split.parse_split("4+4")
    wide, runs = FP.affinities(smoke=four.agent_ids), {FP.SMOKE: FP.SMOKE_PID}
    assert core_split.split_runs(runs, proc=proc, affinity=wide) == (runs, {})
    monkeypatch.setattr(core_split, "active", lambda path=None: four)
    assert core_split.split_runs(runs, proc=proc, affinity=wide) == ({}, runs)
    seen: List[object] = []
    monkeypatch.setattr(core_split, "is_agent_work", lambda aff, *s: seen.append(aff) or False)
    assert core_split.split_runs(runs, proc=proc, affinity=wide) == (runs, {}) and seen


@pytest.mark.parametrize("cuda,agent_work", [("", True), ("-1", True), ("0", False), (None, False)])
def test_a_run_started_with_cuda_hidden_is_agent_work_wherever_it_runs(
        tmp_path: Path, cuda: Optional[str], agent_work: bool) -> None:
    """Rule 5: an agent without a lease runs with CUDA hidden, so such a run is never the GPU's trainer, pinned or
    not. With CUDA visible (or unset) an unpinned run is a training run."""
    proc = FP.incident_box(tmp_path, smoke_cuda=cuda)
    training, agent = core_split.split_runs({FP.SMOKE: FP.SMOKE_PID, FP.REAL: FP.REAL_PID}, proc=proc,
                                            affinity=FP.affinities(smoke=FP.ONLINE))
    assert (FP.SMOKE in agent, FP.SMOKE in training) == (agent_work, not agent_work) and FP.REAL in training


def test_training_runs_reads_the_planted_box_and_keeps_every_run_when_it_cannot_classify(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    proc = FP.incident_box(tmp_path)
    assert core_split.training_runs(proc, FP.affinities()) == {FP.REAL: FP.REAL_PID}

    def broken(*_a: object, **_k: object) -> None:
        raise OSError("a /proc read failed")
    monkeypatch.setattr(core_split, "split_runs", broken)
    both = core_split.training_runs(proc, FP.affinities())
    assert both == {FP.REAL: FP.REAL_PID, FP.SMOKE: FP.SMOKE_PID}, "the safe answer is LIVE"


# --- the readers ------------------------------------------------------------------------------------------------------
def test_the_heavy_slots_are_not_wanted_for_an_agents_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """THE DEFECT (2026-11-10): ``heavy_slots.wanted`` took the raw live-run list, so an agent's CPU smoke on the agent
    cores made every heavy job take a slot (and queue for it). Fails on a revert to ``procfs.live_training_runs``."""
    FP.plant(monkeypatch, FP.incident_box(tmp_path / "smoke", real=False), FP.affinities())
    assert HS.wanted() is None, "a smoke alone is no reason for rule 5b"
    assert "asked for" in (HS.wanted(force=True) or ""), "--slot still asks"
    FP.plant(monkeypatch, FP.incident_box(tmp_path / "both"), FP.affinities())
    why = HS.wanted()
    assert why is not None and "rb_x" in why and "smoke_b1" not in why, "the GPU run is named, the smoke is not"


def test_the_gate_locks_warning_names_the_trainer_not_the_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """THE DEFECT (2026-11-10): ``gate_lock.live_run_warning`` read the raw list, so beside a smoke it warned of a
    'training run' (and, with the smoke's affinity standing in for the trainer's, 'its cpus OVERLAP the trainer's')."""
    FP.plant(monkeypatch, FP.incident_box(tmp_path / "smoke", real=False), FP.affinities())
    assert GL.live_run_warning() is None
    FP.plant(monkeypatch, FP.incident_box(tmp_path / "both"), FP.affinities())
    warning = GL.live_run_warning(my_cpus=FP.AGENTS, my_nice=19, trainer_cpus=FP.TRAINER)
    assert warning is not None and "rb_x (pid 201)" in warning and "smoke_b1" not in warning
    assert "THIS gate" not in warning, "the gate's own cpus are the agent cores: nothing overlaps"


def test_the_kit_live_precheck_does_not_hold_a_launch_for_a_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                                                 capsys: pytest.CaptureFixture) -> None:
    """THE DEFECT (2026-11-10): ``kit live`` (GO pre-check 1) exited 1 while an agent's CPU smoke was live, so a launch
    HELD for it. The GPU run still holds it."""
    FP.plant(monkeypatch, FP.incident_box(tmp_path / "smoke", real=False), FP.affinities())
    assert kit.live_runs() == {} and kit.main(["live"]) == 0
    FP.plant(monkeypatch, FP.incident_box(tmp_path / "both"), FP.affinities())
    assert kit.live_runs() == {FP.REAL: FP.REAL_PID} and kit.main(["live"]) == 1
    assert f"LIVE {FP.REAL} pid {FP.REAL_PID}" in capsys.readouterr().out


def test_the_chains_second_go_refusal_sees_a_launcher_not_an_agents_smoke(tmp_path: Path) -> None:
    """THE DEFECT (2026-11-10): ``arm_chain`` refused a GO beside ANY live launcher, an agent's CPU smoke included.
    ``live.launches`` is ``discover`` less the agent work; the monitors keep ``discover``."""
    fp, models = FakeProc(tmp_path), str(tmp_path / "models")
    (tmp_path / "models").mkdir()
    for rd in ("smoke_b1", "rb_gpu_s1", "smoke_hidden"):
        (tmp_path / "models" / rd).mkdir()
    fp.add(300, launcher_argv("--run-name", "smoke_b1"))                 # on the agent cores
    fp.add(310, launcher_argv("--run-name", "rb_gpu_s1"))                # on the trainer's
    fp.add(320, launcher_argv("--run-name", "smoke_hidden"))             # unpinned, but started with CUDA hidden
    (Path(fp.root) / "320" / "environ").write_bytes(b"CUDA_VISIBLE_DEVICES=\0")
    aff = {300: FP.AGENTS, 310: FP.TRAINER, 320: FP.ONLINE}
    assert sorted(r.pid for r in live.discover(models, fp.root)) == [300, 310, 320]
    assert [r.pid for r in live.launches(models, fp.root, aff.get)] == [310]
    assert [r.pid for r in live.launches(models, fp.root, lambda pid: None)] == [300, 310], \
        "an unreadable affinity keeps a launcher a launch (only the CUDA-hidden one is agent work by its environment)"


# --- the callers ------------------------------------------------------------------------------------------------------
#: ``(file, function)`` that may call the RAW ``procfs.live_training_runs``: the rule itself, the ship gate (which needs
#: the agent work too, to name it), and the one reader of a different question (is this run DIRECTORY being written?).
RAW_CALLERS = {("src/utils/core_split.py", "training_runs"), ("src/utils/ship_gate.py", "resolve_shape"),
               ("src/utils/paths.py", "refuse_live_run_dir")}


class _Callers(ast.NodeVisitor):
    """``(enclosing function, line)`` of every reference to ``live_training_runs`` (an attribute or an import)."""

    def __init__(self) -> None:
        self.stack: List[str] = ["<module>"]
        self.found: List[str] = []

    def _fn(self, node: ast.AST) -> None:
        self.stack.append(getattr(node, "name"))
        self.generic_visit(node)
        self.stack.pop()

    visit_FunctionDef = visit_AsyncFunctionDef = _fn  # type: ignore[assignment]

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr == "live_training_runs":
            self.found.append(self.stack[-1])
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if any(a.name == "live_training_runs" for a in node.names):
            self.found.append(self.stack[-1])


def _raw_callers() -> List[Tuple[str, str]]:
    root, out = repo_path(), set()
    files = subprocess.run(["git", "-C", str(root), "ls-files", "src", "scripts"], capture_output=True, text=True,
                           check=True).stdout.split()
    for rel in files:
        if not rel.endswith(".py") or rel.endswith(("_test.py", "_testing.py")):
            continue
        try:
            tree = ast.parse((root / rel).read_text(encoding="utf-8"))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        v = _Callers()
        v.visit(tree)
        out.update((rel, fn) for fn in v.found)
    return sorted(out)


def test_only_the_declared_callers_read_the_raw_live_run_list() -> None:
    """The class (2026-11-09, 2026-11-10): a reader that decides a SHAPE or a SLOT from ``procfs.live_training_runs``
    counts an agent's smoke as a training run. A new caller asks ``core_split.training_runs``; a caller with a genuine
    reason to see every run (a directory being written) is declared in :data:`RAW_CALLERS` with that reason."""
    found = set(_raw_callers())
    assert found <= RAW_CALLERS, (
        "a new reader of procfs.live_training_runs: ask core_split.training_runs (the rule-5b question) or, if it is the "
        "run-directory question, declare it in RAW_CALLERS: " + repr(sorted(found - RAW_CALLERS)))
    assert found, "the scan found no caller at all: it is blind"


def test_the_caller_scan_has_teeth() -> None:
    """It sees an attribute reference, a from-import (deferred or not) and a module-level use; not a mere mention."""
    src = ("from utils.procfs import live_training_runs\n"
           "x = procfs.live_training_runs\n"
           "def a():\n    return procfs.live_training_runs()\n"
           "def b():\n    from utils.procfs import live_training_runs\n    return live_training_runs()\n"
           "def c():\n    return 'live_training_runs'\n")
    v = _Callers()
    v.visit(ast.parse(src))
    assert sorted(v.found) == ["<module>", "<module>", "a", "b"]
