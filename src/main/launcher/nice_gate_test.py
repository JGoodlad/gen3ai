"""The launcher's INHERITED-NICE PREFLIGHT (`main/launcher/nice_gate.py`).

THE INCIDENT (2026-11-10). A waiter armed under the agent-CPU prefix (`nice -n 19`) would have passed nice 19 to every GPU
training launch; the launcher only raises its nice toward 10 and a process cannot lower its own, so the runs would have
trained at nice 19. What is pinned here: a real GPU launch (`--device cuda`, no `--debug`) from a shell above the ceiling
is refused FATAL_CONFIG naming the inherited nice, BEFORE `_apply_nice` and before `run`; `--dry-run` prints the same
refusal; cpu / auto / `--debug` launches, a nice-0 or nice-10 shell, and a typed `--nice` ceiling are never refused.
Every case sets the inherited nice EXPLICITLY (monkeypatched `os.getpriority`), so the caller's own nice never decides.
"""
import importlib
import os
import subprocess
import sys

import pytest

import main.launcher.dry_run as dry_run_mod
import main.launcher.nice_gate as gate
from main.exit_codes import TrainExitCode

from main.launcher.submodule_gate_test import _FRESH, _make_repo  # the throwaway-repo harness

launcher_run = importlib.import_module("main.launcher.run")  # the package also exports a function named `run`


_REAL_INHERITED_NICE = gate.inherited_nice   # the root conftest pins the caller's nice to 0 for every other file


@pytest.fixture(autouse=True)
def _real_reading(monkeypatch):
    monkeypatch.setattr(gate, "inherited_nice", _REAL_INHERITED_NICE)


def test_the_gate_default_is_the_launchers_default():
    assert gate.DEFAULT_NICE == launcher_run.DEFAULT_NICE


# ---------------------------------------------------------------------------------------
# the pure verdict
# ---------------------------------------------------------------------------------------

@pytest.mark.parametrize("inherited", [11, 15, 19])
@pytest.mark.parametrize("dev", ["cuda", "cuda:0", "CUDA"])
def test_a_deprioritised_shell_refuses_a_real_cuda_launch(inherited, dev):
    v = gate.verdict_for_launch(dev, debug=False, nice_arg=10, inherited=inherited)
    assert v.applies and v.refused
    assert f"nice {inherited}" in v.reason and "nice-0" in v.reason
    assert len(v.lines()) == 1 and "✗ REFUSED" in v.lines()[0]


@pytest.mark.parametrize("inherited", [-5, 0, 5, 10])
def test_a_shell_at_or_below_the_ceiling_never_refuses(inherited):
    v = gate.verdict_for_launch("cuda", debug=False, nice_arg=10, inherited=inherited)
    assert v.applies and not v.refused
    assert "✓" in v.lines()[0]


@pytest.mark.parametrize("dev,debug", [("cpu", False), ("auto", False), (None, False), ("", False),
                                       ("cuda", True), ("cuda:0", True)])
def test_cpu_auto_and_debug_launches_are_never_refused(dev, debug):
    v = gate.verdict_for_launch(dev, debug=debug, nice_arg=10, inherited=19)
    assert not v.applies and not v.refused
    assert v.lines()[0].startswith("inherited nice : n/a")


def test_a_typed_nice_is_the_ceiling_when_it_is_above_the_default():
    assert not gate.verdict_for_launch("cuda", debug=False, nice_arg=19, inherited=19).refused
    assert gate.verdict_for_launch("cuda", debug=False, nice_arg=15, inherited=19).refused
    # `--nice 0` disables the launcher's own renice, so the inherited value stands: held to the default ceiling.
    assert gate.verdict_for_launch("cuda", debug=False, nice_arg=0, inherited=19).refused
    assert not gate.verdict_for_launch("cuda", debug=False, nice_arg=0, inherited=10).refused


def test_an_unreadable_nice_never_refuses(monkeypatch):
    def boom(*a, **k):
        raise OSError("no getpriority")
    monkeypatch.setattr(os, "getpriority", boom)
    assert gate.inherited_nice() is None
    assert not gate.verdict_for_launch("cuda", debug=False, nice_arg=10).refused


def test_the_default_reading_is_the_processes_real_nice():
    assert gate.inherited_nice() == os.nice(0)


# ---------------------------------------------------------------------------------------
# the REAL path: `main()` refuses before `_apply_nice` and before `run`
# ---------------------------------------------------------------------------------------

def _main(monkeypatch, argv, inherited):
    calls = []
    monkeypatch.setattr(os, "getpriority", lambda *a, **k: inherited)
    monkeypatch.setattr(launcher_run, "_apply_nice", lambda n: calls.append(("nice", n)))
    monkeypatch.setattr(launcher_run, "run", lambda *a, **k: calls.append(("run",)))
    monkeypatch.setattr(sys, "argv", ["launcher", *argv])
    return calls


@pytest.mark.parametrize("argv", [["--device", "cuda"], ["--device=cuda"], ["--device", "cuda:0"]])
@pytest.mark.parametrize("inherited", [11, 19])
def test_main_refuses_FATAL_CONFIG_and_does_nothing(argv, inherited, monkeypatch, capsys):
    calls = _main(monkeypatch, ["--steps", "10", *argv], inherited)
    with pytest.raises(SystemExit) as e:
        launcher_run.main()
    assert e.value.code == int(TrainExitCode.FATAL_CONFIG)
    err = capsys.readouterr().err
    assert f"nice {inherited}" in err and "nice-0" in err
    assert calls == [], "a refusal comes BEFORE the renice and before the session"


@pytest.mark.parametrize("argv,inherited", [
    (["--device", "cuda"], 0),                         # the nice-0 driver: unchanged path, launcher renices to 10
    (["--device", "cuda"], 10),
    (["--device", "cuda", "--nice", "19"], 19),        # a deliberate typed ceiling
    (["--device", "cpu"], 19),
    ([], 19),                                          # no --device (auto)
    (["--device", "cuda", "--debug"], 19),             # an agent smoke
])
def test_main_proceeds_when_nothing_is_contradicted(argv, inherited, monkeypatch):
    calls = _main(monkeypatch, ["--steps", "10", *argv], inherited)
    launcher_run.main()
    assert [c[0] for c in calls] == ["nice", "run"]
    assert calls[0][1] == (19 if "19" in argv else launcher_run.DEFAULT_NICE)


def test_a_real_process_under_nice_19_is_refused_end_to_end():
    """No monkeypatching: a genuinely niced subprocess (the incident's shape) refuses before creating anything."""
    code = ("import os, sys; os.nice(19 - os.nice(0)); sys.argv = ['launcher', '--device', 'cuda', '--steps', '10'];"
            "from main.launcher.run import main; main()")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}   # never reach a GPU, whatever else happens
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    assert p.returncode == int(TrainExitCode.FATAL_CONFIG), p.stderr[-500:]
    assert "nice 19" in p.stderr and "nice-0" in p.stderr


# ---------------------------------------------------------------------------------------
# --dry-run reports the same verdict
# ---------------------------------------------------------------------------------------

def _dry(root, monkeypatch, tmp_path, flag, inherited, nice=0):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dry_run_mod, "get_repo_root", lambda *a, **k: root)
    monkeypatch.setattr(os, "getpriority", lambda *a, **k: inherited)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    lines = []
    # --allow-desktop-gpu: the host's own display state must not decide this file's verdicts.
    code = dry_run_mod.dry_run(
        _FRESH + ["--allow-desktop-gpu"] + flag, interval_hours=0.0, pin=True, sync_to_main=False, pin_commit=None,
        grace_minutes=1.0, max_crash_restarts=0, nice=nice, out=lines.append)
    return code, "\n".join(lines)


@pytest.mark.parametrize("inherited", [11, 19])
def test_dry_run_refuses_a_cuda_launch_from_a_deprioritised_shell(inherited, tmp_path, run_archive, monkeypatch):
    root, _sha = _make_repo(tmp_path, "repo", state="ok")
    code, text = _dry(root, monkeypatch, tmp_path, ["--device", "cuda"], inherited)
    assert code == int(TrainExitCode.FATAL_CONFIG), text
    assert "inherited nice : ✗ REFUSED" in text and f"nice {inherited}" in text
    assert "would NOT launch" in text


@pytest.mark.parametrize("flag,inherited", [(["--device", "cuda"], 0), (["--device", "cuda"], 10),
                                            (["--device", "cpu"], 19), (["--device", "cuda", "--debug"], 19)])
def test_dry_run_passes_when_nothing_is_contradicted(flag, inherited, tmp_path, run_archive, monkeypatch):
    root, _sha = _make_repo(tmp_path, "repo", state="ok")
    code, text = _dry(root, monkeypatch, tmp_path, flag, inherited)
    assert code == 0, text
    assert "inherited nice :" in text and "✗ REFUSED" not in text.split("inherited nice :")[1].split("\n")[0]
