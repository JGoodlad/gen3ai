"""NESTED supervisors (``main.reads.supervisors``, ``designs/tools/reads.md`` §3). The anchor lane of a reads state ``S``
plays its units on the kakuna states ``S/anchor/<group>/<label>/``; a standalone ``kakuna run`` on one of them plays the
SAME units, and neither the path lock nor the ``/proc`` scan of either looked at a parent or child path (the duplicate-play
class the D3 reader hit, by another route). Each guard fails on revert: the ``/proc`` scan in both directions, the
ancestor's path lock, the claims the lane holds on every kakuna state it plays, and what ``status`` lists. No game is played;
a "supervisor" is a real process carrying a supervisor's argv."""
from __future__ import annotations

import json
import time

import pytest

from main.reads import layout as LY
from main.reads import runner as RN
from main.reads import supervisors as SV
from main.reads.anchor_pool_test import write_played
from main.reads.lanectl_test import NO_SLEEP, lane  # noqa: F401  (the fixture)
from main.reads.supervisors_test import _fake_supervisor, _kakuna_state, _stop
from main.x35 import kakuna as KA
from utils.filelock import try_exclusive


def _sub(state):
    """The kakuna state the lane plays for ``old``."""
    return LY.anchor_state(state, "u", "old")


def test_a_kakuna_queue_inside_a_live_reads_anchor_lane_is_refused_by_the_proc_scan(lane, tmp_path):  # noqa: F811
    sub = _sub(lane)
    p = _fake_supervisor(tmp_path, ["run", "--state", str(lane), "--lane", "anchor", "--hours", "1"])
    try:
        assert SV.live_supervisors(sub) == {}, "the paths differ: the equal-path scan alone sees nothing"
        assert SV.nested_supervisors(sub, SV.KAKUNA_LANE) == [(p.pid, "anchor", SV.canonical(lane))]
        with pytest.raises(SV.SupervisorError, match=rf"pid {p.pid} \(anchor supervisor of"):
            SV.Hold(sub, SV.KAKUNA_LANE)
        with pytest.raises(KA.KakunaError, match="lies inside a state whose reads ANCHOR lane plays its units"):
            KA.cmd_run(sub, None)
        assert not (sub / "units.jsonl").exists(), "the refused queue played and wrote nothing"
        (tmp_path / "unrelated").mkdir()
        SV.Hold(tmp_path / "unrelated", SV.KAKUNA_LANE).close()       # a queue on any other path is free
        assert SV.nested_supervisors(sub, "s") == [] and SV.nested_supervisors(sub, "anchor") == []
    finally:
        _stop(p)
    SV.Hold(sub, SV.KAKUNA_LANE).close()                              # a reaped supervisor is not live


def test_a_reads_s_lane_plays_no_kakuna_unit_so_it_does_not_exclude_a_queue_inside_its_state(lane, tmp_path):  # noqa: F811
    p = _fake_supervisor(tmp_path, ["run", "--state", str(lane), "--lane", "s", "--hours", "1"])
    try:
        assert SV.nested_supervisors(_sub(lane), SV.KAKUNA_LANE) == []
        SV.Hold(_sub(lane), SV.KAKUNA_LANE).close()
    finally:
        _stop(p)


def test_an_anchor_lane_is_refused_while_a_kakuna_queue_runs_inside_its_state(lane, tmp_path):  # noqa: F811
    sub = _sub(lane)
    p = _fake_supervisor(tmp_path, ["run", "--state", str(sub), "--stop-at-epoch", "9999999999"],
                         module="main.x35.kakuna")
    try:
        assert SV.nested_supervisors(lane, "anchor") == [(p.pid, SV.KAKUNA_LANE, SV.canonical(sub))]
        with pytest.raises(SV.SupervisorError, match="standalone kakuna queue plays a state INSIDE"):
            SV.Hold(lane, "anchor")
        SV.Hold(lane, "s").close()                                    # the S lane plays no kakuna unit
        played = []
        with pytest.raises(SV.SupervisorError, match="standalone kakuna queue plays a state INSIDE"):
            RN.run(lane, "anchor", time.time() + 3600, run_unit=lambda st, u: played.append(u.name) or 0,
                   sleep=NO_SLEEP)
        assert played == [] and not LY.supervisor_pid_file(lane, "anchor").exists()
    finally:
        _stop(p)


def test_an_ancestors_anchor_path_lock_refuses_a_kakuna_queue_inside_it(lane):  # noqa: F811
    """No /proc process is needed: the lane's own path lock, held, is the evidence (the lock side of 'the lock + the
    /proc scan')."""
    sub = _sub(lane)
    first = SV.Hold(lane, "anchor")
    try:
        assert SV.held_ancestor_locks(sub) == [(SV.canonical(lane), SV._holder(SV.lock_path(lane, "anchor")))]
        # the sub-state's own kakuna path lock is free, so only the ANCESTOR's lock can refuse this queue
        free = SV.acquire(sub, SV.KAKUNA_LANE)
        free.close()
        with pytest.raises(SV.SupervisorError, match="lies inside .* whose anchor lane holds its path lock"):
            SV.Hold(sub, SV.KAKUNA_LANE)
        with pytest.raises(KA.KakunaError, match="whose anchor lane holds its path lock"):
            KA.cmd_run(sub, None)
    finally:
        first.close()
    assert SV.held_ancestor_locks(sub) == []
    SV.Hold(sub, SV.KAKUNA_LANE).close()


def test_the_anchor_lane_holds_every_kakuna_state_it_plays_and_releases_them_at_its_end(lane):  # noqa: F811
    """While a unit is in flight, a queue on the state is refused by the path lock the lane holds for it, and the in-state
    `supervisor.lock` (all a queue of OLDER code takes) is held too; both are free again once the lane ended."""
    sub, other = _sub(lane), LY.anchor_state(lane, "u", "new")
    seen = {}

    def unit(st, u):
        if not seen:
            for s in (sub, other):               # both models' states are held, not only the one being played
                with pytest.raises(SV.SupervisorError, match="holds the path lock"):
                    SV.acquire(s, SV.KAKUNA_LANE)
                assert try_exclusive(s / "supervisor.lock", "a+") is None
            holder = json.loads(SV.lock_path(sub, SV.KAKUNA_LANE).read_text())
            seen["via"] = holder["via"]
        write_played(st, u)
        return 0

    assert RN.run(lane, "anchor", time.time() + 3600, run_unit=unit, sleep=NO_SLEEP) == 0
    assert seen["via"] == "main.reads anchor lane"
    for s in (sub, other):
        SV.acquire(s, SV.KAKUNA_LANE).close()
        lock = try_exclusive(s / "supervisor.lock")
        assert lock is not None
        lock.close()


def test_a_standalone_queue_already_on_a_state_the_lane_plays_refuses_the_lane_before_anything_is_played(lane, tmp_path):  # noqa: F811
    sub = _sub(lane)
    p = _fake_supervisor(tmp_path, ["run", "--state", str(sub), "--stop-at-epoch", "9999999999"],
                         module="main.x35.kakuna")
    try:
        # the lane's own Hold scan sees the queue first; the per-state claim is the second line (a queue the scan misses)
        claims = SV.SubStateClaims()
        with pytest.raises(SV.SupervisorError, match=rf"already plays .*pid\(s\) \[{p.pid}\]"):
            claims.claim(sub)
        assert sub not in claims
        SV.acquire(sub, SV.KAKUNA_LANE).close()                       # the refused claim held nothing
    finally:
        _stop(p)


def test_a_queue_of_older_code_holding_only_the_in_state_lock_refuses_the_lane_and_the_lane_frees_its_locks(lane):  # noqa: F811
    old = try_exclusive(_sub(lane) / "supervisor.lock", "w")
    assert old is not None
    played = []
    try:
        with pytest.raises(RN.RunError, match=r"anchor u/old: another supervisor holds .*supervisor.lock"):
            RN.run(lane, "anchor", time.time() + 3600, run_unit=lambda st, u: played.append(u.name) or 0,
                   sleep=NO_SLEEP)
    finally:
        old.close()
    assert played == [] and not LY.supervisor_log(lane, "anchor").exists(), "refused before it wrote anything"
    SV.Hold(lane, "anchor").close()                                   # the refused lane released its path lock
    SV.acquire(_sub(lane), SV.KAKUNA_LANE).close()                    # and the kakuna path lock it took first
    assert RN.run(lane, "anchor", time.time() + 3600, run_unit=lambda st, u: write_played(st, u) or 0,
                  sleep=NO_SLEEP) == 0                                # a later lane plays


def test_a_model_that_settles_during_the_run_is_claimed_before_its_first_unit(tmp_path, run_archive_env, monkeypatch):
    """`new` settles on the second loop turn (its final is not quiet at first): its state did not exist when the lane
    started, so the lane claims it then, before it plays a unit of it."""
    from main.reads.runner_test import CODE, _spec, _step, _zip

    spec = _spec(tmp_path, run_archive_env)
    _zip(run_archive_env / "rb_new", age=10)                          # not quiet yet
    state = tmp_path / "state"
    RN.plan(spec, state, code=CODE, step_of=_step)
    monkeypatch.setattr(RN, "checkout_code", lambda root=None: CODE)
    new = LY.anchor_state(state, "u", "new")
    assert not new.exists()
    held = {}
    settle = RN.settle
    monkeypatch.setattr(RN, "settle", lambda sp, st: settle(sp, st, step_of=_step))

    def sleep(_s):
        _zip(run_archive_env / "rb_new", age=3600)                    # it settles while the lane waits

    def unit(st, u):
        if u.name.startswith("new__") and "new" not in held:
            held["new"] = try_exclusive(new / "supervisor.lock", "a+") is None
        write_played(st, u)
        return 0

    assert RN.run(state, "anchor", time.time() + 3600, run_unit=unit, sleep=sleep) == 0
    assert held == {"new": True}, "the lane held the late state's lock while it played its unit"


def test_status_lists_a_nested_kakuna_queue_and_a_supervisor_that_cannot_be_held(lane, tmp_path):  # noqa: F811
    from main.reads import lanectl as LC

    sub = _sub(lane)
    assert RN.status(lane)["nested_kakuna"] == [] and RN.status(lane)["not_honoring_hold"] == {"anchor": [], "s": []}
    q = _fake_supervisor(tmp_path, ["run", "--state", str(sub), "--stop-at-epoch", "9999999999"],
                         module="main.x35.kakuna")
    p = _fake_supervisor(tmp_path, ["run", "--state", str(lane), "--lane", "anchor", "--hours", "1"])
    try:
        st = RN.status(lane)
        assert st["nested_kakuna"] == [{"pid": q.pid, "state": SV.canonical(sub)}]
        # a supervisor of code that predates `hold` writes no pid file: a hold would not stop it
        assert st["not_honoring_hold"] == {"anchor": [p.pid], "s": []}
        LC.write_pidfile(lane, "anchor", stop_epoch=time.time() + 60, pid=p.pid)     # now it "records itself"
        assert RN.status(lane)["not_honoring_hold"] == {"anchor": [], "s": []}
    finally:
        _stop(p)
        _stop(q)


def test_a_lone_standalone_queue_is_untouched_by_the_nested_rules(tmp_path):
    """The pre-existing standalone path stays free: a queue on its own state (no reads state around it) still runs."""
    state = _kakuna_state(tmp_path)
    SV.Hold(state, SV.KAKUNA_LANE).close()
    assert SV.nested_supervisors(state, SV.KAKUNA_LANE) == [] and SV.held_ancestor_locks(state) == []
