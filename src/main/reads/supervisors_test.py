"""``main.reads.supervisors``: ONE supervisor per lane per state PATH (the D3 reader's duplicate, 2026-10-26: a state
renamed aside took its in-state lock with it, and a second supervisor started on the re-planned state at the same path
while the first lived). Each guard here fails on revert: the path lock outside the state, the ``/proc`` scan, the
dir-identity stop, ``plan``'s refusal of a fresh state some live supervisor names, and ``status``'s live pids."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from main.reads import layout as LY
from main.reads import runner as RN
from main.reads import supervisors as SV
from main.reads.runner_test import CODE, _spec, _step, _zip
from utils import procfs


def _fake_supervisor(cwd: Path, argv_tail: list, module: str = "main.reads") -> subprocess.Popen:
    """A REAL process whose argv is a supervisor's (``python -c <sleep> -m main.reads run ...``; ``module`` may be the
    standalone ``main.x35.kakuna``): ``-c`` takes the code, the rest is only argv, so nothing runs but the sleep. Waits
    (bounded) until ``/proc`` shows the argv."""
    p = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)", "-m", module, *argv_tail],
                         cwd=str(cwd), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    t_end = time.monotonic() + 20
    while module not in procfs.cmdline(p.pid):
        if time.monotonic() > t_end:
            p.kill()
            p.wait()
            pytest.fail("the fake supervisor's argv never appeared in /proc")
        time.sleep(0.02)
    return p


def _stop(p: subprocess.Popen) -> None:
    p.kill()                                         # our own child, by its pid
    p.wait(timeout=30)


def test_parse_argv_finds_only_run_and_resume_with_their_state_and_lane():
    assert SV.parse_argv(["python", "-m", "main.reads", "run", "--state", "S", "--lane", "anchor", "--hours", "2"]) == (
        "run", "S", "anchor")
    assert SV.parse_argv(["py", "-u", "-m", "main.reads", "resume", "--lane=s", "--state=~/x"]) == ("resume", "~/x", "s")
    assert SV.parse_argv(["py", "-m", "main.reads", "s-unit", "--state", "S", "--group", "g"]) is None
    assert SV.parse_argv(["py", "-m", "main.reads", "status", "--state", "S"]) is None
    assert SV.parse_argv(["bash", "-c", "python -m main.reads run --state S --lane anchor"]) is None


def test_the_path_lock_survives_the_state_being_renamed_aside(tmp_path):
    """THE INCIDENT: the in-state lock moves with the renamed dir, so a second supervisor on a new dir at the same
    path gets it; the path lock (outside the state) refuses that second one."""
    state = tmp_path / "state"
    state.mkdir()
    first = SV.Hold(state, "anchor")
    old_lock = RN._lock(state, "anchor")
    try:
        state.rename(tmp_path / "state.aside")
        state.mkdir()                                # re-planned at the same path
        second_in_state = RN._lock(state, "anchor")  # the old guard alone lets the duplicate start ...
        second_in_state.close()
        with pytest.raises(SV.SupervisorError, match="path lock"):
            SV.Hold(state, "anchor")                 # ... the path lock does not
        SV.Hold(state, "s").close()                  # one supervisor per LANE: the other lane is free
        assert SV.lock_path(state, "anchor").parent == Path(os.environ[SV.LOCK_DIR_ENV])
        assert json.loads(SV.lock_path(state, "anchor").read_text())["pid"] == os.getpid()
    finally:
        old_lock.close()
        first.close()
    SV.Hold(state, "anchor").close()                 # released with the first supervisor


def test_the_proc_scan_sees_a_supervisor_that_took_no_path_lock(tmp_path):
    """A supervisor of OLDER code takes only the in-state lock; the /proc scan still sees it (its relative --state
    resolved against ITS cwd), and status lists its pid."""
    state = tmp_path / "rd" / "state"
    state.mkdir(parents=True)
    p = _fake_supervisor(tmp_path / "rd", ["run", "--state", "state", "--lane", "anchor", "--hours", "1"])
    try:
        assert SV.live_supervisors(state) == {"anchor": [p.pid]}
        assert SV.live_supervisors(state, lane="s") == {}
        assert SV.live_supervisors(tmp_path / "other") == {}
        with pytest.raises(SV.SupervisorError, match=rf"pid\(s\) \[{p.pid}\]"):
            SV.Hold(state, "anchor")
    finally:
        _stop(p)
    assert SV.live_supervisors(state) == {}, "a reaped process is not live (read from /proc, never kill -0)"
    SV.Hold(state, "anchor").close()


def test_a_running_supervisor_stops_when_its_dir_is_moved_from_the_path(tmp_path, run_archive_env, monkeypatch):
    spec = _spec(tmp_path, run_archive_env)
    _zip(run_archive_env / "rb_new", age=3600)
    state = tmp_path / "state"
    RN.plan(spec, state, code=CODE, step_of=_step)
    monkeypatch.setattr(RN, "checkout_code", lambda root=None: CODE)
    played = []

    def unit_then_rename(st, u):
        played.append(u.name)
        if len(played) == 1:
            state.rename(tmp_path / "state.aside")
            state.mkdir()                            # someone re-plans at the path
        return 1

    with pytest.raises(SV.SupervisorError, match="moved or replaced"):
        RN.run(state, "anchor", time.time() + 3600, run_unit=unit_then_rename, sleep=lambda s: None)
    assert len(played) == 1, "no unit is played into the dir that now sits at the path"
    assert list(state.iterdir()) == [], "the stopped supervisor wrote nothing into the new dir"


def test_plan_refuses_a_fresh_state_that_a_live_supervisor_names(tmp_path, run_archive_env):
    spec = _spec(tmp_path, run_archive_env)
    state = tmp_path / "state"
    p = _fake_supervisor(tmp_path, ["run", "--state", str(state), "--lane", "anchor", "--hours", "1"])
    try:
        with pytest.raises(RN.RunError, match="moved from it"):
            RN.plan(spec, state, code=CODE, step_of=_step)
        assert not state.exists()
    finally:
        _stop(p)
    RN.plan(spec, state, code=CODE, step_of=_step)
    q = _fake_supervisor(tmp_path, ["resume", f"--state={state}", "--lane=s", "--hours", "1"])
    try:
        RN.plan(spec, state, code=CODE, step_of=_step)      # an existing plan re-settles alongside its own lanes
        assert RN.status(state)["supervisors"] == {"anchor": [], "s": [q.pid]}
    finally:
        _stop(q)
    assert RN.status(state)["supervisors"] == {"anchor": [], "s": []}
    assert LY.spec_file(state).is_file()


# ------------------------------------------------ the standalone Kakuna queue (python -m main.x35.kakuna run), 2026-10-26
def _kakuna_state(tmp_path: Path, name: str = "kstate") -> Path:
    """A kakuna state with a one-unit plan (``away`` x unit seed 0), nothing played."""
    from main.x35 import kakuna as KA

    state = tmp_path / name
    state.mkdir()
    units = KA.plan_units([{"label": "a", "arm": "A", "seed": 1, "path": "/m.zip"}], hash_models=False,
                          teamsets=("away",), unit_seeds=(0,))
    (state / "plan.json").write_text(json.dumps({"units": [u.__dict__ for u in units]}))
    return state


def test_parse_argv_reads_the_standalone_kakuna_queue_as_a_lane_of_its_own():
    assert SV.parse_argv(["py", "-m", "main.x35.kakuna", "run", "--state", "S", "--stop-at-epoch", "9"]) == (
        "run", "S", SV.KAKUNA_LANE)
    assert SV.parse_argv(["py", "-m", "main.x35.kakuna", "run", "--state=S"]) == ("run", "S", "kakuna")
    for verb in ("plan", "status", "smoke", "gk", "d1h"):             # only `run` plays a queue
        assert SV.parse_argv(["py", "-m", "main.x35.kakuna", verb, "--state", "S"]) is None
    assert SV.parse_argv(["py", "-m", "main.x35.kakuna", "run"]) == ("run", None, "kakuna")


def _stub_play(monkeypatch):
    """`kakuna run` with no game: a unit 'plays' by writing a counted summary."""
    from main.x35 import kakuna as KA
    from main.x35.kakuna_test import GOOD

    played = []

    def run_unit(st, u):
        played.append(u.name)
        d = st / "cells" / u.name
        d.mkdir(parents=True)
        (d / "summary.json").write_text(json.dumps(GOOD))
        return 0

    monkeypatch.setattr(KA, "run_unit", run_unit)
    return played


def test_the_standalone_kakuna_queue_is_refused_by_the_path_lock_after_its_state_was_renamed_aside(tmp_path,
                                                                                                   monkeypatch):
    """THE INCIDENT for `kakuna run`: its in-state `supervisor.lock` moved with the renamed dir, so a second queue on
    the re-planned dir at the same path found it free. The path lock (outside the state) refuses the second queue."""
    from main.x35 import kakuna as KA

    played = _stub_play(monkeypatch)
    state = _kakuna_state(tmp_path)
    first = SV.Hold(state, SV.KAKUNA_LANE)               # the first queue's claims
    try:
        state.rename(tmp_path / "kstate.aside")
        state = _kakuna_state(tmp_path)                  # a new plan at the same path: its supervisor.lock is free
        with pytest.raises(KA.KakunaError, match="path lock"):
            KA.cmd_run(state, None)
        assert played == [] and not (state / "units.jsonl").exists(), "the refused queue played and wrote nothing"
    finally:
        first.close()
    assert KA.cmd_run(state, None) == 0 and played == ["a__away__00"]       # released with the first queue


def test_the_standalone_kakuna_queue_is_refused_by_the_proc_scan_of_a_queue_that_took_no_path_lock(tmp_path,
                                                                                                    monkeypatch):
    """An older `kakuna run` took only the in-state lock; the /proc scan still sees it (its relative --state resolved
    against ITS cwd). The scan sees nobody on another state, and a reaped process is not live."""
    from main.x35 import kakuna as KA

    played = _stub_play(monkeypatch)
    (tmp_path / "w").mkdir()
    state = _kakuna_state(tmp_path / "w")
    p = _fake_supervisor(tmp_path / "w", ["run", "--state", state.name, "--stop-at-epoch", "9999999999"],
                         module="main.x35.kakuna")
    try:
        assert SV.live_supervisors(state) == {SV.KAKUNA_LANE: [p.pid]}
        assert SV.live_supervisors(state, lane="anchor") == {} and SV.live_supervisors(tmp_path / "other") == {}
        with pytest.raises(KA.KakunaError, match=rf"pid\(s\) \[{p.pid}\]"):
            KA.cmd_run(state, None)
        assert played == [] and not (state / "units.jsonl").exists()
    finally:
        _stop(p)
    assert SV.live_supervisors(state) == {}
    assert KA.cmd_run(state, None) == 0 and played == ["a__away__00"]


def test_a_standalone_kakuna_queue_stops_when_its_dir_is_moved_from_the_path(tmp_path, monkeypatch):
    from main.x35 import kakuna as KA

    state = _kakuna_state(tmp_path)
    played = []

    def unit_then_rename(st, u):
        played.append(u.name)
        state.rename(tmp_path / "kstate.aside")
        state.mkdir()                                    # someone re-plans at the path
        return 1                                         # the unit did not count: the queue would try again

    monkeypatch.setattr(KA, "run_unit", unit_then_rename)
    with pytest.raises(KA.KakunaError, match="moved or replaced"):
        KA.cmd_run(state, None)
    assert len(played) == 1, "no unit is played into the dir that now sits at the path"
    assert list(state.iterdir()) == [], "the stopped queue wrote nothing into the new dir"
