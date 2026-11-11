"""The launcher's INHERITED-NICE PREFLIGHT (`main/launcher/nice_gate.py`).

THE INCIDENT (2026-11-10). A waiter armed under the agent-CPU prefix (`nice -n 19`) would have passed nice 19 to every GPU
training launch; the launcher only raises its nice toward 10 and a process cannot lower its own, so the runs would have
trained at nice 19. What is pinned here: a real GPU launch (`--device cuda`, no `--debug`) from a shell above the ceiling
is refused FATAL_CONFIG naming the inherited nice, BEFORE `_apply_nice` and before `run`; `--dry-run` prints the same
refusal; cpu / `--debug` launches, a nice-0 or nice-10 shell, and a typed `--nice` ceiling are never refused.
THE RESOLVED DEVICE (2026-11-10 follow-up): `--device auto` / no `--device` on a box where CUDA is usable IS a GPU launch
and is refused like `--device cuda`; where it resolves to CPU (no driver, or `CUDA_VISIBLE_DEVICES=""` hiding every GPU,
the agent no-lease shell) it is not. Every case sets the inherited nice EXPLICITLY (monkeypatched `os.getpriority`) and
the box's GPU state EXPLICITLY (the `gpu_box` fixture), so neither the caller's nice nor the host decides.
"""
import importlib
import os
import subprocess
import sys

import pytest

import main.launcher.dry_run as dry_run_mod
import main.launcher.nice_gate as gate
from utils import desktop_gpu
from main.exit_codes import TrainExitCode

from main.launcher.submodule_gate_test import _FRESH, _make_repo  # the throwaway-repo harness

launcher_run = importlib.import_module("main.launcher.run")  # the package also exports a function named `run`


_REAL_INHERITED_NICE = gate.inherited_nice   # the root conftest pins the caller's nice to 0 for every other file


@pytest.fixture(autouse=True)
def _real_reading(monkeypatch):
    monkeypatch.setattr(gate, "inherited_nice", _REAL_INHERITED_NICE)


def _gpu_box(monkeypatch, present: bool):
    """The box has (or lacks) an NVIDIA driver, whatever the host is (the root conftest already says "none")."""
    monkeypatch.setattr(desktop_gpu, "nvidia_gpu_visible", lambda: present)


@pytest.fixture
def gpu_box(monkeypatch):
    _gpu_box(monkeypatch, True)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


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


@pytest.mark.parametrize("dev,debug", [("cpu", False), ("cuda", True), ("cuda:0", True), ("auto", True), (None, True)])
def test_cpu_and_debug_launches_are_never_refused(dev, debug, gpu_box):
    v = gate.verdict_for_launch(dev, debug=debug, nice_arg=10, inherited=19)
    assert not v.applies and not v.refused
    assert v.lines()[0].startswith("inherited nice : n/a")


@pytest.mark.parametrize("dev", ["auto", None, "", "AUTO"])
@pytest.mark.parametrize("inherited", [11, 19])
def test_an_auto_launch_that_resolves_to_cuda_is_refused_like_an_explicit_one(dev, inherited):
    v = gate.verdict_for_launch(dev, debug=False, nice_arg=10, inherited=inherited, cuda_available=lambda: True)
    assert v.applies and v.refused and v.device == "cuda"
    assert f"nice {inherited}" in v.reason and "--device auto resolved to cuda" in v.reason
    assert "✗ REFUSED" in v.lines()[0]


@pytest.mark.parametrize("dev", ["auto", None, ""])
def test_an_auto_launch_that_resolves_to_cpu_is_never_refused(dev):
    v = gate.verdict_for_launch(dev, debug=False, nice_arg=10, inherited=19, cuda_available=lambda: False)
    assert not v.applies and not v.refused and v.device == "cpu"
    assert v.lines()[0].startswith("inherited nice : n/a") and "resolves to cpu" in v.lines()[0]


def test_an_auto_launch_at_or_below_the_ceiling_passes_and_says_what_it_resolved_to():
    v = gate.verdict_for_launch("auto", debug=False, nice_arg=10, inherited=0, cuda_available=lambda: True)
    assert v.applies and not v.refused and "resolved to cuda" in v.lines()[0]


def test_the_default_resolver_reads_the_driver_and_the_launchers_env(monkeypatch):
    """`gpu_usable`: a driver AND an env that does not hide every device (the child inherits the launcher's env)."""
    _gpu_box(monkeypatch, True)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    assert gate.gpu_usable()
    for hidden in ("", "-1", "NoDevFiles"):
        assert not gate.gpu_usable({"CUDA_VISIBLE_DEVICES": hidden}), hidden
    assert gate.gpu_usable({"CUDA_VISIBLE_DEVICES": "0"})
    _gpu_box(monkeypatch, False)
    assert not gate.gpu_usable({})
    # ... and the verdict's own default (no `cuda_available` passed) is that resolver
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    _gpu_box(monkeypatch, True)
    assert not gate.verdict_for_launch("auto", debug=False, nice_arg=10, inherited=19).refused
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert gate.verdict_for_launch("auto", debug=False, nice_arg=10, inherited=19).refused


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


@pytest.mark.parametrize("argv", [["--device", "cuda"], ["--device=cuda"], ["--device", "cuda:0"],
                                  ["--device", "auto"], ["--device=auto"], []])     # [] is `--device auto`, the default
@pytest.mark.parametrize("inherited", [11, 19])
def test_main_refuses_FATAL_CONFIG_and_does_nothing(argv, inherited, monkeypatch, capsys, gpu_box):
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
    (["--device", "cuda", "--debug"], 19),             # an agent smoke
    (["--debug"], 19),                                 # auto + --debug is CPU, GPU box or not
    ([], 0),                                           # the nice-0 driver with no --device: auto resolves to cuda, passes
])
def test_main_proceeds_when_nothing_is_contradicted(argv, inherited, monkeypatch, gpu_box):
    calls = _main(monkeypatch, ["--steps", "10", *argv], inherited)
    launcher_run.main()
    assert [c[0] for c in calls] == ["nice", "run"]
    assert calls[0][1] == (19 if "19" in argv else launcher_run.DEFAULT_NICE)


@pytest.mark.parametrize("argv", [[], ["--device", "auto"]])
def test_main_proceeds_for_an_auto_launch_that_resolves_to_cpu(argv, monkeypatch):
    """No driver on the box, or every GPU hidden (the agent no-lease shell): `auto` is a CPU run, so nice 19 is fine."""
    calls = _main(monkeypatch, ["--steps", "10", *argv], 19)
    _gpu_box(monkeypatch, False)
    launcher_run.main()
    assert [c[0] for c in calls] == ["nice", "run"]
    calls.clear()
    _gpu_box(monkeypatch, True)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    launcher_run.main()
    assert [c[0] for c in calls] == ["nice", "run"]


def test_a_real_process_under_nice_19_is_refused_end_to_end():
    """No monkeypatching: a genuinely niced subprocess (the incident's shape) refuses before creating anything."""
    code = ("import os, sys; os.nice(19 - os.nice(0)); sys.argv = ['launcher', '--device', 'cuda', '--steps', '10'];"
            "from main.launcher.run import main; main()")
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}   # never reach a GPU, whatever else happens
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    assert p.returncode == int(TrainExitCode.FATAL_CONFIG), p.stderr[-500:]
    assert "nice 19" in p.stderr and "nice-0" in p.stderr


def test_a_real_auto_process_under_nice_19_is_refused_end_to_end(tmp_path):
    """The gap's own shape, no monkeypatching of the launcher: nice 19, NO `--device`, a driver on PATH (a stub
    `nvidia-smi`, never the box's), no CUDA_VISIBLE_DEVICES. The stub is never run: the verdict reads PATH only."""
    bindir = tmp_path / "bin"
    bindir.mkdir()
    (bindir / "nvidia-smi").write_text("#!/bin/sh\nexit 0\n")
    (bindir / "nvidia-smi").chmod(0o755)
    code = ("import os, sys; os.nice(19 - os.nice(0)); sys.argv = ['launcher', '--steps', '10'];"
            "from main.launcher.run import main; main()")
    env = {k: v for k, v in os.environ.items() if k != "CUDA_VISIBLE_DEVICES"}
    env["PATH"] = f"{bindir}{os.pathsep}{env.get('PATH', '')}"
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=120)
    assert p.returncode == int(TrainExitCode.FATAL_CONFIG), p.stderr[-500:]
    assert "nice 19" in p.stderr and "--device auto resolved to cuda" in p.stderr


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
@pytest.mark.parametrize("flag", [["--device", "cuda"], ["--device", "auto"], []])
def test_dry_run_refuses_a_cuda_launch_from_a_deprioritised_shell(flag, inherited, tmp_path, run_archive, monkeypatch,
                                                                  gpu_box):
    root, _sha = _make_repo(tmp_path, "repo", state="ok")
    code, text = _dry(root, monkeypatch, tmp_path, flag, inherited)
    assert code == int(TrainExitCode.FATAL_CONFIG), text
    assert "inherited nice : ✗ REFUSED" in text and f"nice {inherited}" in text
    assert "would NOT launch" in text


def test_dry_run_passes_an_auto_launch_that_resolves_to_cpu(tmp_path, run_archive, monkeypatch):
    root, _sha = _make_repo(tmp_path, "repo", state="ok")
    _gpu_box(monkeypatch, False)
    code, text = _dry(root, monkeypatch, tmp_path, [], 19)
    assert code == 0, text
    assert "inherited nice : n/a" in text and "resolves to cpu" in text


@pytest.mark.parametrize("flag,inherited", [(["--device", "cuda"], 0), (["--device", "cuda"], 10),
                                            (["--device", "cpu"], 19), (["--device", "cuda", "--debug"], 19)])
def test_dry_run_passes_when_nothing_is_contradicted(flag, inherited, tmp_path, run_archive, monkeypatch, gpu_box):
    root, _sha = _make_repo(tmp_path, "repo", state="ok")
    code, text = _dry(root, monkeypatch, tmp_path, flag, inherited)
    assert code == 0, text
    assert "inherited nice :" in text and "✗ REFUSED" not in text.split("inherited nice :")[1].split("\n")[0]
