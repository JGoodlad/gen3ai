"""The launcher's half of the DISK GUARD (`utils.disk_guard`): it refuses a launch that cannot fit BEFORE
anything exists, it does NOT restart a child that stopped on a full disk (`FATAL_DISK`, 8), and it honours
`--allow-low-disk` for a pinned child whose trainer has no such flag.

Every test fails on revert of the part it names.
"""
from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from main.exit_codes import TrainExitCode
from main.launcher.disk_gate import consume_if_absent_at_pin, strip_opt_out, verdict_for_launch
from main.launcher.nonfinite_exit_test import _drive
from main.launcher.state import LauncherState
from utils import disk_guard as dg
from utils.disk_guard import GiB

launcher_run = importlib.import_module("main.launcher.run")
_ARCH_OK = ["--allow-nonproduction-arch", "--allow-nonproduction-recipe"]


# --- the restart loop really stops --------------------------------------------------------------

def test_a_disk_stop_is_never_restarted(tmp_path):
    code, spawned, events = _drive(tmp_path, int(TrainExitCode.FATAL_DISK))
    assert code == 8 and spawned == 1, (code, spawned)
    assert any("Disk full (guard) — will NOT restart" in e for e in events), events


def test_the_launcher_classifies_code_8_as_fatal_and_names_the_fix():
    lines = ["...", "[DiskGuard] 🛑 12.0 MiB free on / is below ONE more checkpoint (60.0 MiB)"]
    reason = launcher_run._fatal_config_reason(int(TrainExitCode.FATAL_DISK), lines)
    assert reason and "disk" in reason[0] and "resume with --model" in reason[0]
    assert "below ONE more checkpoint" in reason[-1]
    assert launcher_run._fatal_config_reason(int(TrainExitCode.CRASH), ["plain traceback"]) is None


# --- the launch is refused before anything exists ------------------------------------------------

def _prepare(argv, monkeypatch):
    monkeypatch.setattr(launcher_run.atexit, "register", lambda *a, **k: None)
    return launcher_run._prepare_session(
        LauncherState(0.0), argv, interval_hours=0.0, pin=False, sync_to_main=False, pin_commit=None,
        grace_minutes=1.0, max_crash_restarts=0)


def test_prepare_session_refuses_a_launch_that_does_not_fit_and_creates_nothing(
        tmp_path, run_archive, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dg, "free_bytes", lambda path: 1 * GiB)
    with pytest.raises(SystemExit) as e:
        _prepare(["--steps", "15000000", "--run-name", "rb_too_big", *_ARCH_OK], monkeypatch)
    assert e.value.code == int(TrainExitCode.FATAL_CONFIG)
    err = capsys.readouterr().err
    assert "not enough free disk" in err and "= REQUIRED" in err and "SHORT by" in err
    assert not (run_archive / "rb_too_big").exists(), "a refusal must leave no run dir behind"


def test_prepare_session_launches_when_it_fits_and_when_the_flag_tolerates(
        tmp_path, run_archive, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(dg, "free_bytes", lambda path: 2000 * GiB)
    ctx = _prepare(["--steps", "15000000", "--run-name", "rb_roomy", *_ARCH_OK], monkeypatch)
    assert ctx.run_dir.endswith("rb_roomy")
    monkeypatch.setattr(dg, "free_bytes", lambda path: 1 * GiB)
    ctx = _prepare(["--steps", "15000000", "--run-name", "rb_tolerated", "--allow-low-disk", *_ARCH_OK],
                   monkeypatch)
    assert ctx.run_dir.endswith("rb_tolerated")


# --- the pinned child -----------------------------------------------------------------------------

class _Report:
    """A pinned parser that is authoritative and does NOT declare `--allow-low-disk`."""

    authoritative = True
    available = True
    options = ["--steps"]

    def __init__(self, declares=()):
        self._declares = set(declares)

    def declares(self, name):
        return name in self._declares or name == "--steps"


def test_the_opt_out_is_consumed_when_the_pinned_trainer_does_not_know_it():
    argv = ["--steps", "1", "--allow-low-disk"]
    out, stripped = consume_if_absent_at_pin(argv, _Report())
    assert stripped and out == ["--steps", "1"]
    # ... and kept when the pinned trainer declares it (its own guard honours it)
    out, stripped = consume_if_absent_at_pin(argv, _Report(declares=["--allow-low-disk"]))
    assert not stripped and out == argv
    # ... and untouched with no report / no flag
    assert consume_if_absent_at_pin(argv, None) == (argv, False)
    assert consume_if_absent_at_pin(["--steps", "1"], _Report()) == (["--steps", "1"], False)


def test_strip_opt_out_removes_only_the_flag():
    assert strip_opt_out(["--a", "1", "--allow-low-disk", "--b"]) == ["--a", "1", "--b"]


def test_the_launch_verdict_reads_the_flag_from_the_argv_even_without_a_resolved_namespace(tmp_path):
    ns = SimpleNamespace(steps=15_000_000, checkpoint_every_steps=1_000_000, eval_freq=None, model=None,
                         self_play=True, device="cuda", debug=False, allow_low_disk=False)

    def tiny(_p):
        return 1 * GiB

    run = str(tmp_path / "x")
    assert verdict_for_launch(["--steps", "1"], run, ns=ns, free_fn=tiny).refused
    assert verdict_for_launch(["--steps", "1", "--allow-low-disk"], run, ns=ns, free_fn=tiny).status == dg.OPTED_OUT
