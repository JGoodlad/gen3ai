"""``main.reads.lanectl``: the supervisor's OWN pid file (the `setsid nohup ... &` `$!` is the wrapper's pid, gone in
seconds: belief-prior-mix 71c76dfb FINDING 2) and the HOLD / RESUME verbs (FINDING 5: a hold was killing the supervisor
by pid by hand). Each guard here fails on revert. The units are stubbed (no game is played); the one process test runs a
REAL supervisor behind a real `setsid` fork on a state that has nothing to play (it waits for a model that never settles)."""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from main.reads import api, cli, lanectl as LC, layout as LY
from main.reads import runner as RN
from main.reads import supervisors as SV
from main.reads.anchor_pool_test import write_played
from main.reads.pipeline_test import CODE as S_CODE, FakeSlot, _refpop, _spec as _s_spec
from main.reads.runner_test import CODE, _spec, _step, _zip
from main.reads.supervisors_test import _fake_supervisor, _stop
from utils import procfs

NO_SLEEP = lambda s: None  # noqa: E731


@pytest.fixture
def lane(tmp_path, run_archive_env, monkeypatch):
    """A planned state of 8 anchor units (2 models x 1 team set x 4 unit seeds); no git, no game."""
    spec = _spec(tmp_path, run_archive_env)
    doc = json.loads(spec.read_text())
    doc["anchor"]["u"]["unit_seeds"] = "0-3"
    spec.write_text(json.dumps(doc))
    _zip(run_archive_env / "rb_new", age=3600)
    state = tmp_path / "state"
    RN.plan(spec, state, code=CODE, step_of=_step)
    monkeypatch.setattr(RN, "checkout_code", lambda root=None: CODE)
    monkeypatch.setenv("GEN3AI_HEAVY_SLOTS", "3")
    return state


def _events(state: Path, lane_name: str = "anchor"):
    return [json.loads(x) for x in LY.supervisor_log(state, lane_name).read_text().splitlines()]


def _run(state, unit, **kw):
    return RN.run(state, "anchor", time.time() + 3600, run_unit=unit, sleep=NO_SLEEP, **kw)


# ------------------------------------------------------------------------------------------------ item 2: the pid
def test_the_supervisor_writes_its_own_pid_and_start_ticks_and_status_prints_them(lane):
    seen = {}

    def unit(st, u):
        seen.setdefault("file", json.loads(LY.supervisor_pid_file(lane, "anchor").read_text()))
        seen.setdefault("status", RN.status(lane)["supervisor_pids"]["anchor"])
        write_played(st, u)
        return 0

    assert _run(lane, unit) == 0
    me = procfs.read_stat(os.getpid())
    assert me is not None
    assert seen["file"]["pid"] == os.getpid() and seen["file"]["start_ticks"] == me.starttime
    assert seen["file"]["lane"] == "anchor" and seen["file"]["stop_epoch"] > time.time()
    assert seen["status"]["pid"] == os.getpid() and seen["status"]["start_ticks"] == me.starttime
    assert seen["status"]["alive"] is True and seen["status"]["stale"] is False
    assert not LY.supervisor_pid_file(lane, "anchor").exists(), "a supervisor that ends takes its pid file with it"
    assert RN.status(lane)["supervisor_pids"] == {"anchor": None, "s": None}


def test_a_pid_file_that_outlived_its_supervisor_or_names_a_reused_pid_is_not_alive(lane):
    me = procfs.read_stat(os.getpid())
    assert me is not None
    LC.write_pidfile(lane, "s", stop_epoch=time.time() + 60)
    assert LC.pid_status(lane, "s")["alive"] is True
    doc = json.loads(LY.supervisor_pid_file(lane, "s").read_text())
    doc["start_ticks"] += 1                     # the same pid number, a different process: PID REUSE
    LY.supervisor_pid_file(lane, "s").write_text(json.dumps(doc))
    st = LC.pid_status(lane, "s")
    assert st["alive"] is False and st["stale"] is True and LC.live_pid(lane, "s") is None
    child = subprocess.Popen([sys.executable, "-c", "pass"])
    child.wait()
    LC.write_pidfile(lane, "s", stop_epoch=time.time() + 60, pid=os.getpid())
    doc = json.loads(LY.supervisor_pid_file(lane, "s").read_text())
    doc["pid"] = child.pid                      # a reaped process: gone
    LY.supervisor_pid_file(lane, "s").write_text(json.dumps(doc))
    assert LC.pid_status(lane, "s")["alive"] is False
    LY.supervisor_pid_file(lane, "s").write_text("{not json")
    assert LC.pid_status(lane, "s")["alive"] is False and "damaged" in LC.pid_status(lane, "s")["error"]


def test_a_successors_pid_file_is_never_removed_by_the_supervisor_it_replaced(lane):
    LC.write_pidfile(lane, "anchor", stop_epoch=time.time() + 60)
    assert LC.remove_pidfile(lane, "anchor", pid=os.getpid() + 1) is False
    assert LY.supervisor_pid_file(lane, "anchor").exists()
    assert LC.remove_pidfile(lane, "anchor") is True and not LY.supervisor_pid_file(lane, "anchor").exists()


@pytest.mark.skipif(shutil.which("setsid") is None, reason="needs util-linux setsid")
def test_the_pid_verb_finds_the_real_supervisor_behind_a_setsid_fork_not_the_wrapper(tmp_path, run_archive_env):
    """THE INCIDENT: `setsid nohup python -m main.reads run ... &` -- `$!` is the wrapper (a process-group leader makes
    setsid FORK, and its parent exits at once). The supervisor, started for real here, has nothing to play (its only
    model never settles), so it waits; `pid` returns ITS pid, which is not the wrapper's."""
    doc = {"schema": "gen3_read_spec_v1", "read": "demo_pid_read_2026-10-26",
           "models": {m: {"arm": m, "seed": 1, "ref": {"kind": "final", "run": f"models/rb_{m}", "min_step": 100,
                                                       "quiet_s": 60}} for m in ("aa", "bb")},
           "anchor": {"u": {"opponent": "metamon:Kakuna", "teamsets": ["heldout72_hpfix"], "unit_seeds": "0-1",
                            "models": ["aa", "bb"]}},
           "decision": {"rule": "anchor_trajectory", "params": {"on": "aa", "off": "bb", "anchor_group": "u",
                                                                "goal_bar": 0.5, "min_unit_frac": 0.9}}}
    spec = tmp_path / "read_spec.json"
    spec.write_text(json.dumps(doc))
    state = tmp_path / "state"
    RN.plan(spec, state, code=RN.checkout_code(), step_of=_step)
    wrapper = subprocess.Popen(
        ["setsid", sys.executable, "-m", "main.reads", "run", "--state", str(state), "--lane", "anchor", "--hours", "1"],
        env=RN.child_env(), stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        start_new_session=True)
    sup = None
    try:
        wrapper.wait(timeout=60)                              # the wrapper is gone; `$!` named it
        assert wrapper.poll() == 0
        sup = api.pid(state, "anchor", wait_s=180)
        assert sup != wrapper.pid and procfs.pid_alive(sup) and "main.reads" in procfs.cmdline(sup)
        st = RN.status(state)
        assert st["supervisor_pids"]["anchor"]["pid"] == sup and st["supervisor_pids"]["anchor"]["alive"]
        assert st["supervisors"]["anchor"] == [sup], "the /proc scan and the supervisor's own record agree"
        assert LC.hold(state, "anchor", by="t", why="probe")["supervisors"]["honoring"] == [sup]
    finally:
        started = SV.live_supervisors(state, exclude=SV.self_and_ancestors()).get("anchor", [])
        for pid in started:                                   # the supervisor this test started, by its pid
            os.kill(pid, signal.SIGTERM)
        t_end = time.monotonic() + 30
        while any(procfs.pid_alive(p) for p in started) and time.monotonic() < t_end:
            time.sleep(0.1)
        wrapper.poll()
    assert LC.pid_status(state, "anchor")["alive"] is False, "a killed supervisor leaves a pid file that reads stale"


def test_the_pid_verb_waits_a_bounded_time_and_refuses_when_nothing_is_alive(lane, capsys):
    assert cli.main(["pid", "--state", str(lane), "--lane", "anchor"]) == 2
    assert "no live supervisor" in capsys.readouterr().err
    t0 = time.monotonic()
    assert cli.main(["pid", "--state", str(lane), "--lane", "anchor", "--wait", "1"]) == 2
    assert 0.9 <= time.monotonic() - t0 < 30
    capsys.readouterr()
    assert cli.main(["pid", "--state", str(lane), "--lane", "anchor", "--wait", "601"]) == 2
    assert "always bounded" in capsys.readouterr().err
    LC.write_pidfile(lane, "anchor", stop_epoch=time.time() + 60)
    assert cli.main(["pid", "--state", str(lane), "--lane", "anchor"]) == 0
    assert capsys.readouterr().out.strip() == str(os.getpid())


# ------------------------------------------------------------------------------------------------ item 1: hold / resume
def test_a_hold_starts_no_new_unit_and_the_lane_runs_again_only_after_resume(lane):
    played = []

    def unit(st, u):
        played.append(u.name)
        if len(played) == 1:
            LC.hold(lane, "anchor", by="orchestrator", why="N1-P look 1 first")
        write_played(st, u)
        return 0

    assert _run(lane, unit) == 0
    assert played == ["old__heldout72_hpfix__00"], "the unit in flight finished; nothing new was started"
    assert RN.status(lane)["anchor"]["u"]["old"] == {"counted": 1, "planned": 4, "exhausted": 0}
    ev = _events(lane)
    assert [e["event"] for e in ev] == ["start", "held", "held_stopped"]
    assert ev[1]["by"] == "orchestrator" and ev[2]["left"] == 7
    assert not LY.supervisor_pid_file(lane, "anchor").exists()
    assert RN.status(lane)["holds"]["anchor"]["why"] == "N1-P look 1 first" and RN.status(lane)["holds"]["s"] is None
    with pytest.raises(LC.LaneError, match=r"HELD by orchestrator .*resume --state .* --by WHO"):
        _run(lane, unit)                       # `run` refuses a held lane, before it takes any lock
    assert len(played) == 1
    rel = LC.release(lane, "anchor", by="orchestrator", why="go")
    assert rel["released"] is True and LC.read_hold(lane, "anchor") is None
    assert _run(lane, unit) == 0
    assert len(played) == 8 and RN.status(lane)["anchor"]["u"]["new"] == {"counted": 4, "planned": 4, "exhausted": 0}, \
        "the rest of the queue resumed, no counted unit replayed"
    log = LC.load_holds(lane)
    assert [e["event"] for e in log] == ["hold", "release"] and log[1]["held_by"] == "orchestrator"


def test_a_hold_lets_every_unit_in_flight_finish_at_concurrency_two(lane):
    barrier = threading.Barrier(2, timeout=30)
    played, lock = [], threading.Lock()

    def unit(st, u):
        with lock:
            played.append(u.name)
            first = len(played) == 1
        if first:
            LC.hold(lane, "anchor", by="o", why="w")
        barrier.wait()                          # both units are in flight before either ends
        write_played(st, u)
        return 0

    assert _run(lane, unit, concurrency=2) == 0
    assert sorted(played) == ["new__heldout72_hpfix__00", "old__heldout72_hpfix__00"]
    assert RN.status(lane)["anchor"]["u"] == {"old": {"counted": 1, "planned": 4, "exhausted": 0}, "new": {"counted": 1, "planned": 4, "exhausted": 0}}
    assert [e["event"] for e in _events(lane)][-2:] == ["held", "held_stopped"]


def test_the_s_lane_honors_a_hold_between_its_units(tmp_path, monkeypatch):
    refpop, rdoc = _refpop(tmp_path)
    spec = _s_spec(tmp_path, refpop, rdoc["content_sha256"])
    state = tmp_path / "state"
    monkeypatch.setattr(RN, "checkout_code", lambda root=None: S_CODE)
    RN.plan(spec, state, code=S_CODE, step_of=lambda _p: 300_000_000)
    slot = FakeSlot(state, rdoc)

    def run_s(argv):
        rc = slot(argv)
        if len(slot.calls) == 1:
            LC.hold(state, "s", by="o", why="w")
        return rc

    assert RN.run(state, "s", time.time() + 7200, run_s=run_s, sleep=NO_SLEEP) == 0
    assert len(slot.calls) == 1, "6 units were pending; the hold stopped the lane after the one in flight"
    assert [e["event"] for e in _events(state, "s")] == ["start", "unit", "unit_end", "held", "held_stopped"]


def test_a_supervisor_waiting_for_a_model_to_settle_looks_for_a_hold_every_few_seconds(tmp_path, run_archive_env,
                                                                                       monkeypatch):
    spec = _spec(tmp_path, run_archive_env)                  # `new` has no final yet: the lane waits on it
    state = tmp_path / "state"
    RN.plan(spec, state, code=CODE, step_of=_step)
    monkeypatch.setattr(RN, "checkout_code", lambda root=None: CODE)
    slept = []

    def sleep(s):
        slept.append(s)
        assert len(slept) < 4, "the lane never stopped for the hold"
        LC.hold(state, "anchor", by="o", why="w")

    def unit(st, u):
        write_played(st, u)
        return 0

    assert RN.run(state, "anchor", time.time() + 3600, run_unit=unit, sleep=sleep) == 0
    assert slept == [LC.HOLD_POLL_S], "one short step, not one 300 s sleep a hold would wait out"
    assert [e["event"] for e in _events(state)][-2:] == ["held", "held_stopped"]


def test_the_hold_verb_is_recorded_once_and_says_which_supervisors_will_not_honor_it(lane):
    # a live supervisor of OLDER code: it has no supervisor.pid and never reads hold.json
    old = _fake_supervisor(lane.parent, ["run", "--state", str(lane), "--lane", "anchor", "--hours", "1"])
    try:
        doc = LC.hold(lane, "anchor", by="o", why="first")
        assert doc["held"] and not doc["already"]
        assert doc["supervisors"] == {"live": [old.pid], "honoring": [], "not_honoring": [old.pid]}
        assert any("predates the hold" in w for w in doc["warnings"])
        LC.write_pidfile(lane, "anchor", stop_epoch=time.time() + 60, pid=old.pid)   # ... now it records itself
        again = LC.hold(lane, "anchor", by="someone else", why="second")
        assert again["already"] is True and (again["by"], again["why"]) == ("o", "first")
        assert again["supervisors"]["honoring"] == [old.pid] and again["warnings"] == []
        with pytest.raises(LC.LaneError, match=rf"pid\(s\) \[{old.pid}\] are still alive"):
            LC.release(lane, "anchor", by="o")                # a live supervisor is still draining: do not clear
        assert LC.read_hold(lane, "anchor") is not None
    finally:
        _stop(old)
    assert [e["event"] for e in LC.load_holds(lane)] == ["hold"], "the repeat wrote nothing"
    assert LC.release(lane, "anchor", by="o")["released"] is True
    assert LC.release(lane, "anchor")["released"] is False, "releasing a lane that is not held is a no-op"


@pytest.mark.parametrize("kw,needle", [({"by": " ", "why": "w"}, "--by"), ({"by": "o", "why": ""}, "--why")])
def test_a_hold_needs_who_and_why(lane, kw, needle):
    with pytest.raises(LC.LaneError, match=needle):
        LC.hold(lane, "anchor", **kw)
    assert LC.read_hold(lane, "anchor") is None


def test_a_hold_on_a_lane_the_spec_does_not_have_or_an_unplanned_state_is_refused(lane, tmp_path):
    with pytest.raises(LC.LaneError, match="no s group"):
        LC.hold(lane, "s", by="o", why="w")
    with pytest.raises(LC.LaneError, match="no spec.json"):
        LC.hold(tmp_path / "nowhere", "anchor", by="o", why="w")
    with pytest.raises(LC.LaneError, match="none of"):
        LC.hold(lane, "kakuna", by="o", why="w")


def test_releasing_a_held_lane_needs_a_name(lane):
    LC.hold(lane, "anchor", by="o", why="w")
    with pytest.raises(LC.LaneError, match="needs --by"):
        LC.release(lane, "anchor")
    assert LC.read_hold(lane, "anchor") is not None


def test_a_damaged_hold_file_is_a_refusal_never_a_free_lane(lane):
    LY.hold_file(lane, "anchor").parent.mkdir(parents=True, exist_ok=True)
    LY.hold_file(lane, "anchor").write_text("{broken")
    with pytest.raises(LC.LaneError, match="damaged hold file"):
        _run(lane, lambda st, u: 0)
    assert "damaged hold file" in RN.status(lane)["holds"]["anchor"]["error"], "status does not crash on it"


def test_the_cli_hold_and_resume_round_trip(lane, monkeypatch, capsys):
    S = str(lane)
    assert cli.main(["hold", "--state", S, "--lane", "anchor", "--by", "orch", "--why", "wait for N1-P"]) == 0
    out = capsys.readouterr()
    assert json.loads(out.out)["held"] is True and "no live supervisor" in out.err
    assert cli.main(["run", "--state", S, "--lane", "anchor", "--hours", "1"]) == 2
    assert "REFUSED" in capsys.readouterr().err
    assert cli.main(["resume", "--state", S, "--lane", "anchor", "--hours", "1"]) == 2     # a held lane needs --by
    assert "needs --by" in capsys.readouterr().err and LC.read_hold(lane, "anchor") is not None
    calls = []
    monkeypatch.setattr(api, "run", lambda *a, **k: calls.append((a, k)) or 0)
    assert cli.main(["resume", "--state", S, "--lane", "anchor", "--hours", "1", "--by", "orch", "--why", "go"]) == 0
    out = capsys.readouterr().out
    assert '"released_hold"' in out and len(calls) == 1 and LC.read_hold(lane, "anchor") is None
    assert cli.main(["resume", "--state", S, "--lane", "anchor", "--hours", "1"]) == 0        # not held: plain `run`
    assert len(calls) == 2 and "released_hold" not in capsys.readouterr().out
