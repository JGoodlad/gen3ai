"""The ship gate's SHAPE: it runs the routine gate's whole set, and only the shape follows the box."""
from __future__ import annotations

import pytest

from utils import ship_gate as SG

_ONLINE = list(range(16))
_TRAINER = [0, 1, 2, 3, 4, 8, 9, 10, 11, 12]


def test_quiet_box_runs_the_quiet_policy_unpinned_at_normal_priority() -> None:
    shape = SG.resolve_shape({}, online=_ONLINE)
    assert (shape.cpus, shape.nice, shape.workers) == (None, 0, SG.QUIET_WORKERS)
    argv = SG.build_argv(shape, "py")
    assert argv[0].endswith("gate_lock.sh") and "nice" not in argv and "taskset" not in argv


def test_a_live_run_takes_rule_5b_on_every_cpu_the_trainer_does_not_hold() -> None:
    shape = SG.resolve_shape({"/m/rb_x": 42}, online=_ONLINE, trainer_cpus=_TRAINER)
    assert shape.cpus == [5, 6, 7, 13, 14, 15] and shape.nice == 19
    assert shape.workers == int(6 * SG.LIVE_WORKERS_PER_CPU)
    argv = SG.build_argv(shape, "py")
    assert argv[:5] == ["nice", "-n", "19", "taskset", "-c"] and argv[5] == "5-7,13-15"
    assert "rb_x" in shape.describe()


def test_a_trainer_pinned_to_every_cpu_is_refused_not_overlapped() -> None:
    with pytest.raises(SystemExit, match="no agent cores"):
        SG.resolve_shape({"/m/rb_x": 42}, online=_ONLINE, trainer_cpus=_ONLINE)


@pytest.mark.parametrize("live", [False, True])
def test_it_runs_the_WHOLE_routine_set_never_a_selection(live: bool) -> None:
    """The owner's rule (2026-09-30): the gate runs everything. The argv names all of src/ and the
    routine gate's marker expression verbatim, under the gate lock and the memory cap."""
    shape = (SG.resolve_shape({"/m/r": 1}, online=_ONLINE, trainer_cpus=_TRAINER) if live
             else SG.resolve_shape({}, online=_ONLINE))
    argv = SG.build_argv(shape, "py", ["--durations=20"])
    i = argv.index("pytest")
    assert argv[i + 1:i + 4] == ["src/", "-m", "not slow and not e2e"]
    assert SG.MARKERS == "not slow and not e2e"
    assert argv[-1] == "--durations=20"
    lock, cap = argv.index(next(a for a in argv if a.endswith("gate_lock.sh"))), argv.index(str(SG.MEM_CAP_GB))
    assert lock < cap < i


def test_dry_run_prints_and_runs_nothing(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.setattr(SG, "resolve_shape", lambda: SG.Shape({}, None, 0, 6))
    monkeypatch.setattr(SG.subprocess, "call", lambda *a, **k: pytest.fail("a dry run ran the gate"))
    assert SG.main(["--dry-run"]) == 0
    assert "not slow and not e2e" in capsys.readouterr().err
    assert SG.main(["--nope"]) == 2
    assert SG.main([]) == 2                         # no argument never starts a gate
    assert SG.main(["--", "-x"]) == 2


def test_the_gate_runs_with_no_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}
    monkeypatch.setattr(SG, "resolve_shape", lambda: SG.Shape({}, None, 0, 6))
    monkeypatch.setattr(SG.subprocess, "call", lambda cmd, cwd, env: seen.update(env=env) or 0)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    assert SG.main(["run"]) == 0
    assert seen["env"]["CUDA_VISIBLE_DEVICES"] == ""
