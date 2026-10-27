"""The 2026-10-26 adversarial review's MAJORs (2-4) and the chain-side MINORs (6-11, 13), each pinned by a test that
fails when its fix is reverted (``designs/ops/arm_chain_review_2026-10-26.md``; the runner's finding 5 is in
``runner_test.py``, the ship's finding 12 in ``ship_test.py``), plus restart compatibility with the state the
``93a05fe91`` supervisor writes (``testdata/state_93a05fe9_*.json``, written by that code on these fakes).

``integration``: real fake launches (``testing_fakes``), real ``/proc``, real flocks. ~15 s in all.
"""
from __future__ import annotations

import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
from typing import Any, Dict, Iterator, List

import pytest

from main.ops.arm_chain import canary, cli, liveness, monitor
from main.ops.arm_chain import chain as C
from main.ops.arm_chain.chain_test import (FakeShipper, _Died, _recording_kill, _void_arm, deps, phases, runner_for,
                                           status, world_then_wait)
from main.ops.arm_chain.procwatch import ProcRef, ref_of, wait_gone
from main.ops.arm_chain.testing_fakes import World, commit_all

pytestmark = pytest.mark.integration

TESTDATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata")


@pytest.fixture
def w(tmp_path) -> Iterator[World]:
    yield from world_then_wait(tmp_path)


def _hold_reason(ch: C.Chain, key: str = "arm:rb_a1_s1") -> str:
    return ch.state["items"][key]["hold"]["reason"]


def _chain_file(w: World, items: List[Dict[str, Any]]) -> str:
    p = os.path.join(w.root, "chain.json")
    with open(p, "w") as f:
        json.dump(w.chain_dict(items), f)
    return p


class _Lock:
    """Hold the chain's lock the way a live supervisor does (an exclusive flock on its own open file)."""

    def __init__(self, path: str, pid_text: str = ""):
        self.fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
        fcntl.flock(self.fd, fcntl.LOCK_EX)
        if pid_text:
            os.ftruncate(self.fd, 0)
            os.write(self.fd, pid_text.encode())

    def release(self) -> None:
        os.close(self.fd)


# -- MAJOR 2: supervisor liveness ------------------------------------------------------------------------------------

def test_a_live_supervisor_reads_alive_with_its_heartbeat_and_what_it_is_doing(w):
    d = deps(w, runner_for(w), FakeShipper(w.repo))
    spec = w.spec([w.arm("a1", 1, w.pin_a)])
    ch = C.Chain(spec, d)
    seen: List[Any] = []
    real = ch._arm_running

    def running(it: Any) -> None:
        seen.append(C.Chain(spec, d).liveness())     # what `status` prints, from a second process's view
        real(it)
    ch._arm_running = running  # type: ignore[method-assign]
    assert ch.run() == C.EXIT_OK, status(w)
    lv = seen[0]
    assert (lv.verdict, lv.pid, lv.loud) == ("ALIVE", os.getpid(), False)
    assert "arm:rb_a1_s1 RUNNING" in lv.detail
    sup = json.loads(open(spec.state_path).read())["supervisor"]
    assert sup["pid"] == os.getpid() and sup["exit"] == 0 and sup["start_ticks"] == ref_of(os.getpid()).start_ticks


def test_a_beat_before_a_blocking_step_declares_its_bound(w):
    """The heartbeat must never read STALE during a healthy long step: the kit's dry run (<= 1,500 s) is declared."""
    seen: List[Dict[str, Any]] = []
    runner = runner_for(w)

    def is_dry(argv: Any) -> bool:
        return any(a.endswith("launch_x35.sh") for a in argv) and "GO" not in argv

    def dry(argv: Any, out: str) -> int:
        seen.append(json.loads(open(os.path.join(w.job, "arm_chain", "t.state.json")).read())["supervisor"])
        return runner.real.run(argv, cwd=w.repo, env=runner.calls[-1][1], timeout_s=60, out_path=out)
    runner.handlers.insert(0, (is_dry, dry))
    d = deps(w, runner, FakeShipper(w.repo))
    ch = C.Chain(w.spec([w.arm("a1", 1, w.pin_a)]), d)
    ch.launcher.runner = ch.runner                      # the production wiring: the launcher runs through the beat
    assert ch.run() == C.EXIT_OK, status(w)
    sup = seen[0]
    assert sup["doing"].startswith("arm:rb_a1_s1 PENDING: running ") and "launch_x35.sh" in sup["doing"]
    assert sup["stale_after_epoch"] - sup["last_beat_epoch"] >= 1500


def _stop_fake_launcher(w: World, run: str) -> None:
    """End this test's OWN fake launcher (a ``sleep``), through the identity gate."""
    from main.ops.arm_chain.procwatch import ref_from_pidfile, signal_if
    ref = ref_from_pidfile(os.path.join(w.job, f"{run}.pid"))
    if ref is not None:
        signal_if(ref, C.launcher_ere(run), signal.SIGTERM)
        assert wait_gone(ref, 30, poll_s=0.1)


def test_a_dead_supervisor_with_a_live_run_is_shown_loudly(w):
    w.knobs("rb_a1_s1", FAKE_RUN_S=60)
    d = deps(w, runner_for(w), FakeShipper(w.repo))
    spec = w.spec([w.arm("a1", 1, w.pin_a)])
    ch = C.Chain(spec, d)

    def die(_it: Any) -> None:
        raise _Died()
    ch._arm_running = die  # type: ignore[method-assign]
    with pytest.raises(_Died):
        ch.run()
    try:
        head = C.Chain(spec, d).status_lines()[0]
    finally:
        _stop_fake_launcher(w, "rb_a1_s1")
    assert head.startswith("!!! supervisor: DEAD") and "arm:rb_a1_s1 RUNNING is mid-flight" in head
    assert "arm:rb_a1_s1's launcher pid" in head and "is ALIVE" in head
    assert wait_gone(ProcRef.from_json(ch.state["items"]["arm:rb_a1_s1"]["data"]["wrapper"]), 30, poll_s=0.1)


def test_a_wedged_supervisor_reads_stale_and_a_held_chain_reads_dead_quietly(w):
    spec = w.spec([w.arm("a1", 1, w.pin_a)])
    d = deps(w, runner_for(w), FakeShipper(w.repo))
    ch = C.Chain(spec, d)
    now = time.time()
    me = ref_of(os.getpid())
    ch.state["history"].append({"at": "x", "item": "arm:rb_a1_s1", "from": "PENDING", "to": "PRECHECKED", "note": ""})
    ch.state["items"]["arm:rb_a1_s1"]["phase"] = "RUNNING"
    ch.state["supervisor"] = {"pid": me.pid, "start_ticks": me.start_ticks, "last_beat_epoch": now - 2000,
                              "stale_after_epoch": now - 100, "doing": "arm:rb_a1_s1 RUNNING"}
    lock = _Lock(spec.state_path + ".lock")
    try:
        lv = liveness.probe(ch.state, spec.state_path + ".lock", now)
    finally:
        lock.release()
    assert (lv.verdict, lv.loud) == ("STALE", True) and "33 min ago" in lv.detail
    ch.state["items"]["arm:rb_a1_s1"]["hold"] = {"reason": "x", "at": "x", "phase": "RUNNING"}
    lv = liveness.probe(ch.state, spec.state_path + ".lock", now)
    assert (lv.verdict, lv.loud) == ("DEAD", False)


def test_an_exception_that_escapes_the_supervisor_leaves_one_hold_line(w, monkeypatch):
    """Review 10-26, findings 1-2: an exception outside the per-item try (here ``_run_locked`` itself, as a full disk
    under ``_hold`` would) killed the supervisor with no HOLD line."""
    p = _chain_file(w, [w.arm("a1", 1, w.pin_a)])

    def boom(self: Any, max_items: Any) -> int:
        raise OSError(28, "No space left on device")
    monkeypatch.setattr(C.Chain, "_run_locked", boom)
    with pytest.raises(OSError):
        cli.main(["run", p])
    lines = [ln for ln in status(w).splitlines() if "ARMCHAIN[t]: HOLD:" in ln]
    assert len(lines) == 1 and "HOLD: supervisor died: OSError" in lines[0]
    assert json.loads(open(os.path.join(w.job, "arm_chain", "t.state.json")).read())["supervisor"]["exit"] == \
        "died: OSError"


def test_the_supervisor_turns_a_sigterm_into_its_hold_line(w, monkeypatch):
    p = _chain_file(w, [w.arm("a1", 1, w.pin_a)])
    seen = {}

    def look(self: Any, max_items: Any) -> int:
        seen["term"], seen["hup"] = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGHUP)
        return C.EXIT_OK
    before = signal.getsignal(signal.SIGTERM)
    monkeypatch.setattr(C.Chain, "_run_locked", look)
    old_hup = signal.signal(signal.SIGHUP, signal.SIG_IGN)       # as under ``nohup``
    try:
        assert cli.main(["run", p]) == 0
        assert seen == {"term": cli._on_signal, "hup": signal.SIG_IGN}     # an ignored SIGHUP stays ignored
    finally:
        signal.signal(signal.SIGHUP, old_hup)
    assert signal.getsignal(signal.SIGTERM) == before
    with pytest.raises(cli.SupervisorSignalled, match="SIGTERM"):
        cli._on_signal(signal.SIGTERM, None)


# -- MAJOR 3: clear-hold resolutions -----------------------------------------------------------------------------------

def test_a_kit_hold_at_the_go_is_relaunched_fresh_never_retried_forever(w):
    """Review 10-26, finding 3: a HOLD in the kit's GO-time pre-checks HELD again on every retry, forever."""
    w.knobs("rb_a1_s1", FAKE_GO_HOLD=1)
    d = deps(w, runner_for(w), FakeShipper(w.repo))
    spec = w.spec([w.arm("a1", 1, w.pin_a)])
    ch = C.Chain(spec, d)
    assert ch.run() == C.EXIT_HOLD and phases(ch)["arm:rb_a1_s1"] == "LAUNCHED"
    first = _hold_reason(ch)
    assert "fake GO-time pre-check refusal" in first
    ch.clear_hold("arm:rb_a1_s1", by="orchestrator", why="kit live was transient")       # retry: the same evidence
    assert C.Chain(spec, d).run() == C.EXIT_HOLD
    ch = C.Chain(spec, d)
    assert _hold_reason(ch) == first
    w.knobs("rb_a1_s1")
    msg = ch.clear_hold("arm:rb_a1_s1", by="orchestrator", why="kit live was transient", resolution="relaunch")
    assert "resolution relaunch at LAUNCHED: RELAUNCH" in msg and msg in status(w)
    assert phases(ch)["arm:rb_a1_s1"] == "PENDING"
    assert C.Chain(spec, d).run() == C.EXIT_OK, status(w)
    ch = C.Chain(spec, d)
    assert phases(ch)["arm:rb_a1_s1"] == "DONE"
    assert len([c for c in w.calls() if c.startswith("GO")]) == 2
    attempts = ch.state["items"]["arm:rb_a1_s1"]["data"]["attempts"]
    assert len(attempts) == 1 and "fake GO-time" in attempts[0]["held"]["reason"]
    assert any(h["note"].startswith("cleared by orchestrator (relaunch at LAUNCHED)") for h in ch.state["history"])


def test_relaunch_is_refused_once_the_kit_launched(w):
    w.knobs("rb_a1_s1", FAKE_NO_PASS=1)
    d = deps(w, runner_for(w), FakeShipper(w.repo))
    spec = w.spec([w.arm("a1", 1, w.pin_a)])
    ch = C.Chain(spec, d)
    assert ch.run() == C.EXIT_HOLD and phases(ch)["arm:rb_a1_s1"] == "LAUNCHED"
    with pytest.raises(C.ChainError, match="models/rb_a1_s1 EXISTS"):
        ch.clear_hold("arm:rb_a1_s1", by="orchestrator", why="x", resolution="relaunch")
    assert C.Chain(spec, d).state["items"]["arm:rb_a1_s1"]["hold"] is not None
    with pytest.raises(C.ChainError, match="must be one of"):
        ch.clear_hold("arm:rb_a1_s1", by="orchestrator", why="x", resolution="skip")


def test_accept_at_launched_advances_and_every_later_gate_still_runs(w):
    w.knobs("rb_a1_s1", FAKE_NO_PASS=1)
    d = deps(w, runner_for(w), FakeShipper(w.repo))
    spec = w.spec([w.arm("a1", 1, w.pin_a)])
    ch = C.Chain(spec, d)
    assert ch.run() == C.EXIT_HOLD
    msg = ch.clear_hold("arm:rb_a1_s1", by="orchestrator", why="pool seeded by hand", resolution="accept")
    assert "ACCEPTED: LAUNCHED -> SPAWNED" in msg
    assert C.Chain(spec, d).run() == C.EXIT_OK, status(w)
    ch = C.Chain(spec, d)
    assert phases(ch)["arm:rb_a1_s1"] == "DONE"
    assert [a["phase"] for a in ch.state["items"]["arm:rb_a1_s1"]["data"]["accepted"]] == ["LAUNCHED"]
    assert "SPAWNED -> BANNER_OK" in status(w) and "ENDED -> COMPLETE" in status(w)


def test_a_paged_banner_is_accepted_past_its_gate(w):
    """Review 10-26, finding 9: a ``page`` arm the orchestrator accepts had no path to its K9(b), thin and records."""
    w.knobs("rb_a1_s1", FAKE_RUN_S=3, FAKE_BANNER="[TEAM-MIX] FATAL: the lever did not load")
    d = deps(w, runner_for(w), FakeShipper(w.repo))
    spec = w.spec([_void_arm(w, "page")])
    ch = C.Chain(spec, d)
    assert ch.run() == C.EXIT_HOLD and phases(ch)["arm:rb_a1_s1"] == "SPAWNED"
    ch.clear_hold("arm:rb_a1_s1", by="orchestrator", why="the banner check is wrong for this arm",
                  resolution="accept")
    assert phases(ch)["arm:rb_a1_s1"] == "BANNER_OK"
    assert C.Chain(spec, d).run() == C.EXIT_OK, status(w)


def test_accept_is_refused_for_a_stopped_run(w):
    w.knobs("rb_a1_s1", FAKE_RUN_S=30, FAKE_BANNER="[TEAM-MIX] FATAL: the lever did not load")
    killed: List[Any] = []
    d = deps(w, runner_for(w), FakeShipper(w.repo))
    d.kill = _recording_kill(killed)
    ch = C.Chain(w.spec([_void_arm(w, "stop")]), d)
    assert ch.run() == C.EXIT_HOLD and len(killed) == 1
    with pytest.raises(C.ChainError, match="registered stop SIGTERMed"):
        ch.clear_hold("arm:rb_a1_s1", by="orchestrator", why="x", resolution="accept")


# -- MAJOR 4: monitor coverage -----------------------------------------------------------------------------------------

def test_a_go_writes_a_monitor_rearm_line_the_filter_matches(w):
    d = deps(w, runner_for(w), FakeShipper(w.repo))
    assert C.Chain(w.spec([w.arm("a1", 1, w.pin_a)]), d).run() == C.EXIT_OK, status(w)
    line = next(ln for ln in status(w).splitlines() if "MONITOR-REARM" in ln)
    assert re.search(canary.monitor_filter("t"), line)
    assert os.path.join(w.job, "rb_a1_s1.launcher.log") in line
    assert os.path.join(w.models, "rb_a1_s1", "launcher_child.full.log") in line


def test_the_monitor_follows_logs_that_appear_after_it_started_with_no_rearm(w):
    """Armed before the GO, the follower prints the new run's alerts (and only lines written after it started)."""
    p = _chain_file(w, [w.arm("a1", 1, w.pin_a)])
    st = os.path.join(w.job, "status.txt")
    with open(st, "a") as f:
        f.write("[t0] ARMCHAIN[t]: HOLD: an OLD hold, before the monitor started\n")
    out: List[str] = []
    child = os.path.join(w.models, "rb_a1_s1", "launcher_child.full.log")

    def between(_s: float) -> None:
        n = len([o for o in out if not o.startswith("ARMCHAIN-MONITOR")])
        if n == 0 and not os.path.exists(child):
            os.makedirs(os.path.dirname(child))
            with open(os.path.join(w.job, "rb_a1_s1.launcher.log"), "w") as f:
                f.write("ok\nTraceback (most recent call last):\n")
            with open(child, "w") as f:
                f.write("trainer line\nWEDGE: no progress for 600 s\n")
            with open(st, "a") as f:
                f.write("[t1] ARMCHAIN[t]: HOLD: arm:rb_a1_s1 at SPAWNED: new\n")
        elif os.path.exists(child) and not os.path.exists(child + ".1"):
            os.rename(child, child + ".1")                     # the full log rotates by rename
            with open(child, "w") as f:
                f.write("Child crashed (exit 1)\n")
    monitor.run_monitor(p, out=out.append, interval_s=0.01, passes=4, sleep=between, live_runs=lambda md: [])
    body = [o for o in out if not o.startswith("ARMCHAIN-MONITOR")]
    assert body == ["[status.txt] [t1] ARMCHAIN[t]: HOLD: arm:rb_a1_s1 at SPAWNED: new",
                    "[rb_a1_s1.launcher.log] Traceback (most recent call last):",
                    "[launcher_child.full.log] WEDGE: no progress for 600 s",
                    "[launcher_child.full.log] Child crashed (exit 1)"], out
    assert out[-1].startswith("ARMCHAIN-MONITOR[t]: EXPIRING")


def test_the_monitor_pages_a_dead_supervisor_once(w):
    p = _chain_file(w, [w.arm("a1", 1, w.pin_a)])
    spec = w.spec([w.arm("a1", 1, w.pin_a)])
    ch = C.Chain(spec, deps(w, runner_for(w), FakeShipper(w.repo)))
    ch.state["items"]["arm:rb_a1_s1"]["phase"] = "RUNNING"
    ch.state["history"].append({"at": "x", "item": "arm:rb_a1_s1", "from": "BANNER_OK", "to": "RUNNING", "note": ""})
    ch._save()
    out: List[str] = []
    monitor.run_monitor(p, out=out.append, interval_s=0.01, passes=3, sleep=lambda s: None, live_runs=lambda md: [])
    pages = [o for o in out if "SUPERVISOR DEAD" in o]
    assert len(pages) == 1 and "arm:rb_a1_s1 RUNNING is mid-flight" in pages[0]
    assert pages[0].startswith("[status.txt] ") and re.search(canary.monitor_filter("t"), pages[0])
    assert status(w).count("SUPERVISOR DEAD") == 1


def test_watch_files_for_a_chain_lists_each_unfinished_arms_fixed_paths(w, capsys):
    p = _chain_file(w, [w.arm("a1", 1, w.pin_a), w.arm("a2", 2, w.pin_b)])
    spec = w.spec([w.arm("a1", 1, w.pin_a), w.arm("a2", 2, w.pin_b)])
    ch = C.Chain(spec, deps(w, runner_for(w), FakeShipper(w.repo)))
    ch.state["items"]["arm:rb_a1_s1"]["phase"] = "DONE"
    ch._save()
    assert cli.main(["watch-files", "--chain", p, "--models-dir", w.models]) == 0
    files = capsys.readouterr().out.split()
    assert files[:3] == [os.path.join(w.job, "status.txt"), os.path.join(w.job, "rb_a2_s2.launcher.log"),
                         os.path.join(w.models, "rb_a2_s2", "launcher_child.full.log")]
    assert not any("rb_a1_s1" in f for f in files)


# -- MINORs ------------------------------------------------------------------------------------------------------------

def test_a_run_that_ends_during_the_banner_guard_fails_at_once(w):
    """Finding 6: a run that died early was noticed only at the gate's deadline."""
    clock = {"t": 0.0}
    d = deps(w, runner_for(w), FakeShipper(w.repo))
    d.monotonic = lambda: clock["t"]
    real_sleep = d.sleep

    def sleep(s: float) -> None:
        clock["t"] += s
        real_sleep(s)
    d.sleep = sleep
    arm = w.arm("a1", 1, w.pin_a, banner={"expect": "[TEAM-MIX] ok", "void": ["TEAM-MIX] FATAL"], "within_min": 60})
    ch = C.Chain(w.spec([arm]), d)
    assert ch.run() == C.EXIT_HOLD
    assert "the run ENDED (a1 s1: END rb_a1_s1 launcher_rc=0) with no '[TEAM-MIX] ok' banner" in _hold_reason(ch)
    assert clock["t"] < 15 * 60


def test_a_go_beside_a_live_launch_is_refused(w):
    """Finding 7: the GO never checked for another live GO wrapper or launcher."""
    d = deps(w, runner_for(w), FakeShipper(w.repo))
    d.live_runs = lambda: ["4242 60 /models/rb_other_s9 [argv:--run-name]"]
    ch = C.Chain(w.spec([w.arm("a1", 1, w.pin_a)]), d)
    assert ch.run() == C.EXIT_HOLD
    assert "refusing the GO of rb_a1_s1" in _hold_reason(ch) and "rb_other_s9" in _hold_reason(ch)
    assert [c for c in w.calls() if c.startswith("GO")] == []
    # another arm's GO wrapper (by argv), alive in its pre-checks
    script = os.path.join(w.repo, "kit", "launch_x35.sh")
    other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)", script, "a9", "9", "GO"])
    try:
        d.live_runs = lambda: []
        ch.clear_hold("arm:rb_a1_s1", by="orchestrator", why="x")
        ch2 = C.Chain(w.spec([w.arm("a1", 1, w.pin_a)]), d)
        assert ch2.run() == C.EXIT_HOLD
        assert f"GO wrapper(s) [{other.pid}]" in _hold_reason(ch2)
    finally:
        other.kill()      # this test's own stand-in process, by its pid
        other.wait()
    assert [c for c in w.calls() if c.startswith("GO")] == []


def test_a_supervisor_that_died_after_its_sigterm_never_sends_a_second(w):
    """Finding 8: a restart between the SIGTERM and its record re-read VOID and SIGTERMed again."""
    w.knobs("rb_a1_s1", FAKE_RUN_S=60, FAKE_BANNER="[TEAM-MIX] FATAL: the lever did not load")
    killed: List[Any] = []
    d = deps(w, runner_for(w), FakeShipper(w.repo))
    d.stop_wait_s = 0.5

    def record_then_die(pid: int, sig: int) -> None:   # a launcher still saving: the SIGTERM has not ended it yet
        killed.append((pid, sig))
        raise _Died()
    d.kill = record_then_die
    spec = w.spec([_void_arm(w, "stop")])
    try:
        with pytest.raises(_Died):
            C.Chain(spec, d).run()
        d.kill = lambda pid, sig: killed.append((pid, sig))
        ch = C.Chain(spec, d)
        assert ch.run() == C.EXIT_HOLD
    finally:
        _stop_fake_launcher(w, "rb_a1_s1")
    assert len(killed) == 1 and "already SIGTERMed" in _hold_reason(ch)
    assert ch.state["items"]["arm:rb_a1_s1"]["data"]["stopped"]["gone"] is False


def test_clear_hold_and_abandon_refuse_under_a_live_supervisor_and_saves_are_fsynced(w, monkeypatch):
    """Finding 10: the CLI wrote the state without the supervisor's lock, through a shared tmp file, unsynced."""
    w.knobs("rb_a1_s1", FAKE_RC=1)
    d = deps(w, runner_for(w), FakeShipper(w.repo))
    spec = w.spec([w.arm("a1", 1, w.pin_a)])
    ch = C.Chain(spec, d)
    assert ch.run() == C.EXIT_HOLD
    lock = _Lock(spec.state_path + ".lock", "4242\n")
    try:
        for verb in (ch.clear_hold, ch.abandon):
            with pytest.raises(C.ChainError, match="a supervisor holds"):
                verb("arm:rb_a1_s1", by="orchestrator", why="x")
    finally:
        lock.release()
    synced: List[int] = []
    real_fsync = os.fsync
    monkeypatch.setattr(os, "fsync", lambda fd: (synced.append(fd), real_fsync(fd))[1])
    ch.abandon("arm:rb_a1_s1", by="orchestrator", why="x")
    assert len(synced) >= 2 and not os.path.exists(spec.state_path + ".tmp")


def test_ship_records_survives_a_chain_file_edit_to_a_started_item(w, monkeypatch):
    """Finding 11: ``ship-records`` loaded the state, which refuses an edited started item, and died with no result."""
    sh = FakeShipper(w.repo)
    d = deps(w, runner_for(w), sh)
    assert C.Chain(w.spec([w.arm("a1", 1, w.pin_a)]), d).run() == C.EXIT_OK
    p = _chain_file(w, [w.arm("a1", 1, w.pin_a, go_by="orchestrator, edited after the start")])
    with pytest.raises(C.ChainError, match="CHANGED"):
        cli._chain(type("A", (), {"chain": p})())
    monkeypatch.setattr(cli, "default_shipper", lambda s, r, e: sh)
    assert cli.main(["ship-records", p, "--item", "arm:rb_a1_s1"]) == 0
    spec = w.spec([w.arm("a1", 1, w.pin_a)])
    assert json.loads(open(C.ship_result_path(spec, spec.arms()[0])).read())["ok"] is True


def test_a_ship_worker_with_no_request_writes_a_result_and_pages(w):
    spec = w.spec([w.arm("a1", 1, w.pin_a)])
    assert C.ship_worker(spec, "arm:rb_a1_s1", FakeShipper(w.repo)) == C.EXIT_HOLD
    assert json.loads(open(C.ship_result_path(spec, spec.arms()[0])).read())["ok"] is False
    assert "SHIP-HOLD: arm:rb_a1_s1: no readable ship request" in status(w)


@pytest.mark.parametrize("agree", [True, False])
def test_the_dose_record_takes_the_comparator_the_stages_records_agree_on(w, agree):
    """Finding 13: a chain file with no ``x_dose_comparator`` silently dropped the ``c0_comparator`` block."""
    block = {"seeds": [3501, 3502, 3503], "wall_min": [265, 241, 240], "steps_per_s": [1384, 1418, 1424]}
    os.makedirs(os.path.join(w.repo, w.stage_rel, "dose"))
    for i, b in enumerate([block, block if agree else {**block, "seeds": [1]}]):
        with open(os.path.join(w.repo, w.stage_rel, "dose", f"rb_old_s{i}.json"), "w") as f:
            json.dump({"run": f"rb_old_s{i}", "c0_comparator": b}, f)
    commit_all(w.repo, "dose records")
    sh = FakeShipper(w.repo)
    assert C.Chain(w.spec([w.arm("a1", 1, w.pin_a)]), deps(w, runner_for(w), sh)).run() == C.EXIT_OK, status(w)
    dose = json.loads(open(sh.shipped[0][os.path.join(w.stage_rel, "dose", "rb_a1_s1.json")]).read())
    if agree:
        assert dose["c0_comparator"] == block
        assert "c0_comparator: the block all 2 dose record(s) in the stage carry" in status(w)
    else:
        assert "c0_comparator" not in dose and "carry 2 different blocks" in status(w)


# -- restart compatibility with the state 93a05fe91's supervisor writes ------------------------------------------------

def _fixture_world(w: World, name: str, items: List[Dict[str, Any]]) -> Dict[str, Any]:
    fx = json.loads(open(os.path.join(TESTDATA, f"state_93a05fe9_{name}.json")).read())
    sub = lambda t: t.replace("@ROOT@", w.root).replace("@PIN_A@", w.pin_a).replace("@PIN_B@", w.pin_b)  # noqa: E731
    st = os.path.join(w.job, "status.txt")
    with open(st, "w") as f:
        f.write(sub(fx["status_txt"]))
    text = sub(json.dumps(fx["state"])).replace('"@INODE@"', str(os.stat(st).st_ino))
    state = json.loads(text)
    os.makedirs(os.path.join(w.job, "arm_chain"), exist_ok=True)
    with open(os.path.join(w.job, "arm_chain", "t.state.json"), "w") as f:
        json.dump(state, f, indent=1, sort_keys=True)
    assert state["schema"] == "arm_chain_state_v1" and "supervisor" not in state
    return state


def _fake_run_files(w: World, run: str, pin: str) -> None:
    """What the fake kit's GO wrote for the run (the fixture carries the state and the status file, not the run)."""
    ck = os.path.join(w.models, run, "checkpoints")
    os.makedirs(ck, exist_ok=True)
    t = time.time()
    for i, s in enumerate((2000, 3000, 4000, 4999)):
        p = os.path.join(ck, f"checkpoint_{s}_steps.zip")
        open(p, "w").close()
        open(os.path.join(ck, f"checkpoint_{s}_steps.json"), "w").close()
        os.utime(p, (t - 400 + i * 100, t - 400 + i * 100))
    open(os.path.join(w.models, run, "final_model.zip"), "w").close()
    with open(os.path.join(w.job, f"{run}.launcher.log"), "w") as f:
        f.write(f"  launching with --pin-commit {pin}\n  Last step   : 5000\n  FPS         : 1,400\n  Restarts    : 0\n")


def _two(w: World, first: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [first, w.arm("a2", 2, w.pin_b)]


def test_a_93a05fe9_state_resumed_in_running_completes_and_keeps_every_old_key(w):
    old = _fixture_world(w, "running", _two(w, w.arm("a1", 1, w.pin_a)))
    _fake_run_files(w, "rb_a1_s1", w.pin_a)
    d = deps(w, runner_for(w), FakeShipper(w.repo))
    spec = w.spec(_two(w, w.arm("a1", 1, w.pin_a)))
    head = C.Chain(spec, d).status_lines()[0]
    assert head.startswith("!!! supervisor: DEAD") and "arm:rb_a1_s1 RUNNING" in head
    assert C.Chain(spec, d).run() == C.EXIT_OK, status(w)
    new = json.loads(open(spec.state_path).read())
    assert set(phases(C.Chain(spec, d)).values()) == {"DONE"} and "chain COMPLETE" in status(w)
    assert new["schema"] == "arm_chain_state_v1" and new["supervisor"]["exit"] == 0
    a1_old, a1_new = old["items"]["arm:rb_a1_s1"]["data"], new["items"]["arm:rb_a1_s1"]["data"]
    assert all(a1_new[k] == v for k, v in a1_old.items())
    assert new["history"][:len(old["history"])] == old["history"]


def test_a_93a05fe9_go_hold_at_launched_is_relaunched(w):
    _fixture_world(w, "launched_go_hold", _two(w, w.arm("a1", 1, w.pin_a)))
    d = deps(w, runner_for(w), FakeShipper(w.repo))
    spec = w.spec(_two(w, w.arm("a1", 1, w.pin_a)))
    ch = C.Chain(spec, d)
    assert ch.status_lines()[0].startswith("supervisor: DEAD")   # held: quiet
    ch.clear_hold("arm:rb_a1_s1", by="orchestrator", why="transient", resolution="relaunch")
    assert C.Chain(spec, d).run() == C.EXIT_OK, status(w)
    assert [c.split()[0] for c in w.calls()] == ["DRY", "GO", "DRY", "GO"]


@pytest.mark.parametrize("name,first", [("launched_no_pass", None), ("spawned_void_page", "banner")])
def test_a_93a05fe9_post_spawn_hold_is_accepted_and_completes(w, name, first):
    arm = (w.arm("a1", 1, w.pin_a) if first is None else
           w.arm("a1", 1, w.pin_a, banner={"expect": "[TEAM-MIX] ok", "void": ["TEAM-MIX] FATAL"], "within_min": 1}))
    _fixture_world(w, name, _two(w, arm))
    _fake_run_files(w, "rb_a1_s1", w.pin_a)
    d = deps(w, runner_for(w), FakeShipper(w.repo))
    spec = w.spec(_two(w, arm))
    ch = C.Chain(spec, d)
    if name == "launched_no_pass":
        with pytest.raises(C.ChainError, match="EXISTS"):
            ch.clear_hold("arm:rb_a1_s1", by="orchestrator", why="x", resolution="relaunch")
    ch.clear_hold("arm:rb_a1_s1", by="orchestrator", why="checked by hand", resolution="accept")
    assert C.Chain(spec, d).run() == C.EXIT_OK, status(w)
    assert set(phases(C.Chain(spec, d)).values()) == {"DONE"}


def test_a_93a05fe9_state_with_an_async_ship_resumes_and_an_old_supervisor_reads_alive(w):
    _fixture_world(w, "async_ship", _two(w, w.arm("a1", 1, w.pin_a)))
    d = deps(w, runner_for(w), FakeShipper(w.repo))
    spec = w.spec(_two(w, w.arm("a1", 1, w.pin_a)))
    ch = C.Chain(spec, d)
    lines = ch.status_lines()
    assert "ship: c0ffee123" in next(ln for ln in lines if ln.startswith("arm:rb_a1_s1"))
    # a 93a05fe9 supervisor holds the lock and writes only its pid there, no heartbeat
    lock = _Lock(spec.state_path + ".lock", f"{os.getpid()}\n")
    try:
        lv = ch.liveness()
        assert (lv.verdict, lv.pid) == ("ALIVE", os.getpid()) and "writes no heartbeat" in lv.detail
        assert ch.run() == C.EXIT_BUSY
    finally:
        lock.release()
    assert C.Chain(spec, d).run() == C.EXIT_OK, status(w)
    assert set(phases(C.Chain(spec, d)).values()) == {"DONE"} and "chain COMPLETE" in status(w)
