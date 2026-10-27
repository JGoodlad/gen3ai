"""The arm chain end to end: a real supervisor, a real bash stand-in for ``launch_x35.sh``, real ``/proc`` waits.

Each test pins a class bug of the job-directory scripts the chain replaces (``designs/ops/arm_chain.md`` §5). The
"launcher" is a ``sleep`` named ``python -m main.launcher …`` that lives ~1 s; the kit's K9(b) and the thin planner
are scripted (``testing_fakes``); the record ship is a recording fake (``ship_test.py`` covers the real one).

``integration``: it spawns its own short-lived processes and runs git in a temp checkout. ~10 s in all.
"""
from __future__ import annotations

import fcntl
import json
import os
import subprocess
from typing import Any, Dict, Iterator, List, Mapping

import pytest

from main.ops.arm_chain import chain as C
from main.ops.arm_chain.cpugate import GateResult
from main.ops.arm_chain.launch import X35Launcher
from main.ops.arm_chain.procwatch import ProcRef, wait_gone
from main.ops.arm_chain.ship import ShipResult
from main.ops.arm_chain.spec import ChainSpec
from main.ops.arm_chain.testing_fakes import FakeRunner, World, k9b_handler, make_world, planner_handler

pytestmark = pytest.mark.integration


class FakeShipper:
    def __init__(self, repo: str):
        self.repo = repo
        self.shipped: List[Dict[str, str]] = []

    def ship(self, files: Mapping[str, str], message: str, tag: str) -> ShipResult:
        for rel, src in files.items():
            assert os.path.isfile(src), src
            assert not os.path.realpath(src).startswith(os.path.realpath(self.repo) + os.sep), \
                "a record ship's source must be a copy outside main"
        self.shipped.append(dict(files))
        return ShipResult(commit="feedface1", files=sorted(files))


def deps(w: World, runner: FakeRunner, shipper: FakeShipper) -> C.Deps:
    env = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/tmp"),
           # the supervisor's OWN environment carries the poison a CPU step exports (CB-02): the GO must not see it
           "CUDA_VISIBLE_DEVICES": "", "GEN3AI_TEST_ALLOW_GPU": "1"}
    gate = lambda: GateResult(True, True, 1.0, (), "PASS: fake")  # noqa: E731
    return C.Deps(runner=runner, base_env=env, poll_s=0.05,
                  launcher=lambda s: X35Launcher(s, runner, env, gate=gate, niceness=lambda: 0),
                  shipper=lambda s: shipper,  # type: ignore[arg-type, return-value]
                  gpu_idle=lambda: (True, "idle"),
                  lease=lambda: "LEASED: lease owner 'Training Oct', token …",
                  lease_token=lambda: (True, "token valid: fake"),
                  # the unfixed kit (F1) unless a test says otherwise: the ship stays synchronous
                  kit_k9b_fixed=lambda s: (False, "test: the kit's k9b-batch still writes into main's stage"),
                  sleep=lambda x: __import__("time").sleep(min(x, 0.05)), stop_wait_s=10,
                  # a real launcher may be live on this box (production): the GO's second-GO refusal must not see it
                  live_runs=lambda: [])


def runner_for(w: World, **planner: Any) -> FakeRunner:
    return FakeRunner().on(*k9b_handler()).on(*planner_handler(w.models, **planner))


def status(w: World) -> str:
    return open(os.path.join(w.job, "status.txt")).read()


def phases(ch: C.Chain) -> Dict[str, str]:
    return {k: v["phase"] for k, v in ch.state["items"].items()}


def world_then_wait(tmp_path: Any) -> Iterator[World]:
    """The ``w`` fixture's body (``chain_review_test`` shares it)."""
    world = make_world(str(tmp_path))
    yield world
    # a test may leave its fake launch running (a HOLD never stops a run). pytest re-uses a passed test's temp dir
    # name, so the NEXT test's GO would read it as a live GO wrapper of its own kit (finding 7's refusal): wait it out
    assert _wait_world_procs_gone(world.root, 60), f"a fake launch of {world.root} outlived its test by 60 s"


@pytest.fixture
def w(tmp_path) -> Iterator[World]:
    yield from world_then_wait(tmp_path)


def _wait_world_procs_gone(root: str, timeout_s: float) -> bool:
    import time as _t
    from utils import procfs
    t0 = _t.monotonic()
    while _t.monotonic() - t0 < timeout_s:
        if not any(root in procfs.read_text(pid, "cmdline").replace("\0", " ") for pid in procfs.snapshot()
                   if pid != os.getpid()):
            return True
        _t.sleep(0.1)
    return False


def two_arm_spec(w: World, window_step: str = "true", on_fail: str = "continue") -> ChainSpec:
    win = {"kind": "window", "name": "between", "worktree": w.repo, "require_commit": w.pin_a, "budget_min": 5,
           "on_fail": on_fail, "preempt_marker": "NO_GO_BETWEEN",
           "steps": [{"name": "s1", "argv": [window_step], "gpu": False, "mem_gb": 1, "timeout_s": 30}]}
    return w.spec([w.arm("a1", 1, w.pin_a), win, w.arm("a2", 2, w.pin_b)])


def test_cross_pin_chain_hands_each_arm_its_own_pin_and_a_visible_card(w):
    """CB-05 (one ambient pin per chain) + CB-02 (an empty CUDA_VISIBLE_DEVICES leaked into the GO, toy_x47 3602)."""
    shipper = FakeShipper(w.repo)
    ch = C.Chain(two_arm_spec(w), deps(w, runner_for(w), shipper))
    assert ch.run() == C.EXIT_OK, status(w)
    assert w.calls() == [f"DRY a1 1 {w.pin_a} CVD=unset", f"GO a1 1 {w.pin_a} CVD=unset",
                         f"DRY a2 2 {w.pin_b} CVD=unset", f"GO a2 2 {w.pin_b} CVD=unset"]
    assert set(phases(ch).values()) == {"DONE"}
    assert len(shipper.shipped) == 2 and all(len(f) == 2 for f in shipper.shipped)
    assert "chain COMPLETE" in status(w)
    # main is never dirty: the K9(b) record went to the scratch stage only (CB-11)
    assert subprocess.run(["git", "-C", w.repo, "status", "--porcelain"], capture_output=True,
                          text=True).stdout == ""


def test_a_stopped_run_is_never_k9b_read_thinned_or_chained(w):
    """CB-08: rb_x35_d1wide_s3501 ended launcher_rc=0 after a SIGTERM and was nearly treated as finished."""
    w.knobs("rb_a1_s1", FAKE_LAST=4500, FAKE_FINAL=0)
    runner = runner_for(w)
    ch = C.Chain(two_arm_spec(w), deps(w, runner, FakeShipper(w.repo)))
    assert ch.run() == C.EXIT_HOLD
    assert ch.state["items"]["arm:rb_a1_s1"]["phase"] == "ENDED"
    assert "stopped, not finished" in ch.state["items"]["arm:rb_a1_s1"]["hold"]["reason"]
    assert not any("k9b" in c[0] or any(a.endswith("archive_grooming_dryrun.py") for a in c[0]) for c in runner.calls)
    assert [c for c in w.calls() if c.startswith("GO a2")] == []


def test_a_launcher_rc_other_than_zero_holds(w):
    w.knobs("rb_a1_s1", FAKE_RC=1)
    ch = C.Chain(two_arm_spec(w), deps(w, runner_for(w), FakeShipper(w.repo)))
    assert ch.run() == C.EXIT_HOLD
    assert "launcher_rc=1" in ch.state["items"]["arm:rb_a1_s1"]["hold"]["reason"]


class _Died(BaseException):
    """The supervisor process dying (not an Exception: nothing in the chain may catch it)."""


def test_a_resumed_supervisor_sees_an_end_written_while_nobody_watched(w):
    """CB-16: a re-armed waiter counted lines from its own arming and could miss an END in the swap gap."""
    sh = FakeShipper(w.repo)
    d = deps(w, runner_for(w), sh)
    ch = C.Chain(w.spec([w.arm("a1", 1, w.pin_a)]), d)

    def die(_it: Any) -> None:
        raise _Died()
    ch._arm_launched = die  # type: ignore[method-assign]
    with pytest.raises(_Died):
        ch.run()
    assert phases(ch)["arm:rb_a1_s1"] == "LAUNCHED"
    wrapper = ProcRef.from_json(ch.state["items"]["arm:rb_a1_s1"]["data"]["wrapper"])
    # nobody watches while the launch wrapper finishes the run and writes END
    assert wait_gone(wrapper, 30, poll_s=0.1)
    assert "END rb_a1_s1 launcher_rc=0" in status(w)
    ch2 = C.Chain(w.spec([w.arm("a1", 1, w.pin_a)]), d)
    assert ch2.run() == C.EXIT_OK, status(w)
    assert phases(ch2)["arm:rb_a1_s1"] == "DONE"


def test_a_supervisor_that_died_mid_go_never_issues_a_second_go(w):
    """The GOING phase: died after the spawn, before LAUNCHED was saved ⇒ the launch is ADOPTED, not repeated."""
    d = deps(w, runner_for(w), FakeShipper(w.repo))
    spec = w.spec([w.arm("a1", 1, w.pin_a)])
    ch = C.Chain(spec, d)
    real_go = ch.launcher.go

    def go_then_die(it: Any) -> Any:
        real_go(it)
        raise _Died()
    ch.launcher.go = go_then_die  # type: ignore[method-assign]
    with pytest.raises(_Died):
        ch.run()
    assert phases(ch)["arm:rb_a1_s1"] == "GOING"
    ch2 = C.Chain(spec, d)
    assert ch2.run() == C.EXIT_OK, status(w)
    assert [c for c in w.calls() if c.startswith("GO")] == [f"GO a1 1 {w.pin_a} CVD=unset"]


def test_mid_go_death_without_evidence_holds_and_clearing_resets_to_pending(w):
    d = deps(w, runner_for(w), FakeShipper(w.repo))
    spec = w.spec([w.arm("a1", 1, w.pin_a)])
    ch = C.Chain(spec, d)

    def die(_it: Any) -> Any:
        raise _Died()
    ch.launcher.go = die  # type: ignore[method-assign]
    with pytest.raises(_Died):
        ch.run()
    ch2 = C.Chain(spec, d)
    assert ch2.run() == C.EXIT_HOLD
    assert "never a blind re-GO" in ch2.state["items"]["arm:rb_a1_s1"]["hold"]["reason"]
    assert [c for c in w.calls() if c.startswith("GO")] == []
    with pytest.raises(C.ChainError):
        ch2.clear_hold("arm:rb_a1_s1", by="", why="x")
    ch2.clear_hold("arm:rb_a1_s1", by="orchestrator", why="verified nothing launched")
    assert phases(ch2)["arm:rb_a1_s1"] == "PENDING"
    assert C.Chain(spec, d).run() == C.EXIT_OK
    assert len([c for c in w.calls() if c.startswith("GO")]) == 1


def test_one_supervisor_per_chain(w):
    """The 10-26 duplicate supervisor: a second ``run`` on a held chain exits 4 and touches nothing."""
    spec = w.spec([w.arm("a1", 1, w.pin_a)])
    ch = C.Chain(spec, deps(w, runner_for(w), FakeShipper(w.repo)))
    fd = os.open(spec.state_path + ".lock", os.O_RDWR | os.O_CREAT)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        assert ch.run() == C.EXIT_BUSY
    finally:
        os.close(fd)
    assert w.calls() == []


def test_a_started_item_cannot_change_under_the_chain(w):
    d = deps(w, runner_for(w), FakeShipper(w.repo))
    C.Chain(w.spec([w.arm("a1", 1, w.pin_a)]), d).run()
    with pytest.raises(C.ChainError, match="CHANGED"):
        C.Chain(w.spec([w.arm("a1", 1, w.pin_b)]), d)
    # appending a new item to a started chain is fine
    C.Chain(w.spec([w.arm("a1", 1, w.pin_a), w.arm("a2", 2, w.pin_b)]), d)


def test_a_failed_dry_run_never_goes(w):
    w.knobs("rb_a1_s1", FAKE_DRY_FAIL=1)
    ch = C.Chain(w.spec([w.arm("a1", 1, w.pin_a)]), deps(w, runner_for(w), FakeShipper(w.repo)))
    assert ch.run() == C.EXIT_HOLD
    assert phases(ch)["arm:rb_a1_s1"] == "PENDING"
    assert [c for c in w.calls() if c.startswith("GO")] == []


def test_a_launch_without_its_pool_seed_pass_holds_before_the_arm_end(w):
    w.knobs("rb_a1_s1", FAKE_NO_PASS=1)
    runner = runner_for(w)
    ch = C.Chain(w.spec([w.arm("a1", 1, w.pin_a)]), deps(w, runner, FakeShipper(w.repo)))
    assert ch.run() == C.EXIT_HOLD
    assert phases(ch)["arm:rb_a1_s1"] == "LAUNCHED"
    assert not any("k9b" in c[0] for c in runner.calls)


def test_a_void_banner_holds_the_arm(w):
    """D3's VOID guard: a fork whose lever did not load trains the parent's recipe under the arm's name."""
    w.knobs("rb_a1_s1", FAKE_BANNER="[TEAM-MIX] FATAL: the fork does NOT inherit its parent's --team-mix")
    arm = w.arm("a1", 1, w.pin_a, banner={"expect": "[TEAM-MIX] d3_setmut_v1@ALL",
                                          "void": ["TEAM-MIX] FATAL", "does NOT inherit"], "within_min": 1})
    ch = C.Chain(w.spec([arm]), deps(w, runner_for(w), FakeShipper(w.repo)))
    assert ch.run() == C.EXIT_HOLD
    assert ch.state["items"]["arm:rb_a1_s1"]["hold"]["reason"].startswith("VOID")


def test_the_banner_in_the_launcher_stdout_passes(w):
    """CB-06: the [TEAM-MIX] banner is in the LAUNCHER's stdout log, not launcher_child.log."""
    w.knobs("rb_a1_s1", FAKE_BANNER="[TEAM-MIX] d3_setmut_v1@ALL: 50% of episodes")
    arm = w.arm("a1", 1, w.pin_a, banner={"expect": "[TEAM-MIX] d3_setmut_v1@ALL", "void": ["TEAM-MIX] FATAL"],
                                          "within_min": 1})
    ch = C.Chain(w.spec([arm]), deps(w, runner_for(w), FakeShipper(w.repo)))
    assert ch.run() == C.EXIT_OK, status(w)


def test_a_window_on_fail_hold_stops_the_chain_and_continue_hands_off(w):
    ch = C.Chain(two_arm_spec(w, window_step="false", on_fail="hold"), deps(w, runner_for(w), FakeShipper(w.repo)))
    assert ch.run() == C.EXIT_HOLD
    assert [c for c in w.calls() if c.startswith("GO a2")] == []


def test_a_failed_window_with_continue_hands_off_to_the_next_arm(w):
    """X59-S: a failed pre-flight trains no student, and the N1-P chain resumes."""
    ch = C.Chain(two_arm_spec(w, window_step="false", on_fail="continue"), deps(w, runner_for(w), FakeShipper(w.repo)))
    assert ch.run() == C.EXIT_OK
    assert phases(ch)["window:between"] == "DONE_FAILED"
    assert f"GO a2 2 {w.pin_b} CVD=unset" in w.calls()


def test_the_dose_record_shipped_is_the_runs_own(w):
    sh = FakeShipper(w.repo)
    C.Chain(w.spec([w.arm("a1", 1, w.pin_a)]), deps(w, runner_for(w), sh)).run()
    files = sh.shipped[0]
    dose = json.loads(open(files[os.path.join(w.stage_rel, "dose", "rb_a1_s1.json")]).read())
    assert (dose["run"], dose["pin"], dose["first_step"], dose["last_step"], dose["segments"]) == \
        ("rb_a1_s1", w.pin_a, 1000, 5000, 1)
    k9b = json.loads(open(files[os.path.join(w.stage_rel, "k9b", "rb_a1_s1.json")]).read())
    assert k9b["verdict"] == "PASS"


def test_an_abandoned_held_arm_hands_off_past_itself(w):
    """A VOID arm the orchestrator stopped can never finish: abandoning it is the only way past, and only when held."""
    w.knobs("rb_a1_s1", FAKE_RC=1)
    d = deps(w, runner_for(w), FakeShipper(w.repo))
    spec = w.spec([w.arm("a1", 1, w.pin_a), w.arm("a2", 2, w.pin_b)])
    ch = C.Chain(spec, d)
    with pytest.raises(C.ChainError, match="not HELD"):
        ch.abandon("arm:rb_a1_s1", by="orchestrator", why="x")
    assert ch.run() == C.EXIT_HOLD
    ch.abandon("arm:rb_a1_s1", by="orchestrator", why="VOID: stopped by the orchestrator")
    assert C.Chain(spec, d).run() == C.EXIT_OK
    assert phases(C.Chain(spec, d)) == {"arm:rb_a1_s1": "DONE_FAILED", "arm:rb_a2_s2": "DONE"}


def test_the_cli_plans_and_reports_status(w, capsys):
    from main.ops.arm_chain import cli
    p = os.path.join(w.root, "chain.json")
    with open(p, "w") as f:
        json.dump(w.chain_dict([w.arm("a1", 1, w.pin_a)]), f)
    assert cli.main(["plan", p, "--check-pins"]) == 0
    assert "PLAN OK" in capsys.readouterr().out
    assert cli.main(["status", p]) == 0
    assert "arm:rb_a1_s1" in capsys.readouterr().out
    bad = w.chain_dict([w.arm("a1", 1, "0" * 40)])
    with open(p, "w") as f:
        json.dump(bad, f)
    assert cli.main(["plan", p, "--check-pins"]) == 1


def test_an_earlier_failed_gos_start_line_does_not_poison_the_dose(w):
    """Review 10-26: a START with no END from an earlier GO made the dose HOLD for ever, after the thin."""
    with open(os.path.join(w.job, "status.txt"), "a") as f:
        f.write("[2026-10-26T01:00:00-07:00] a1 s1: START rb_a1_s1: an earlier GO that died before its run dir\n")
    ch = C.Chain(w.spec([w.arm("a1", 1, w.pin_a)]), deps(w, runner_for(w), FakeShipper(w.repo)))
    assert ch.run() == C.EXIT_OK, status(w)


def test_an_unexpected_exception_becomes_a_hold_line_not_a_silent_death(w):
    ch = C.Chain(w.spec([w.arm("a1", 1, w.pin_a)]), deps(w, runner_for(w), FakeShipper(w.repo)))

    def boom(_it: Any) -> None:
        raise OSError("disk vanished")
    ch._arm_ended = boom  # type: ignore[method-assign]
    assert ch.run() == C.EXIT_HOLD
    assert "HOLD: arm:rb_a1_s1 at ENDED: UNEXPECTED OSError: disk vanished" in status(w)


def test_a_busy_gpu_before_the_go_holds(w):
    d = deps(w, runner_for(w), FakeShipper(w.repo))
    d.gpu_idle = lambda: (False, "compute process(es) 4242")
    clock = {"t": 0.0}
    d.monotonic = lambda: clock["t"]
    d.sleep = lambda s: clock.__setitem__("t", clock["t"] + s)
    ch = C.Chain(w.spec([w.arm("a1", 1, w.pin_a)]), d)
    assert ch.run() == C.EXIT_HOLD
    assert "not idle before the GO" in ch.state["items"]["arm:rb_a1_s1"]["hold"]["reason"]
    assert [c for c in w.calls() if c.startswith("GO")] == []


def test_a_restart_on_a_stale_dry_run_rechecks_instead_of_going(w):
    d = deps(w, runner_for(w), FakeShipper(w.repo))
    spec = w.spec([w.arm("a1", 1, w.pin_a)])
    ch = C.Chain(spec, d)
    ch._arm_step(spec.arms()[0])                     # PENDING -> PRECHECKED (one dry run)
    assert phases(ch)["arm:rb_a1_s1"] == "PRECHECKED"
    later = d.now() + C.PRECHECK_MAX_AGE_S + 60
    d.now = lambda: later
    ch2 = C.Chain(spec, d)
    ch2._arm_step(spec.arms()[0])
    assert phases(ch2)["arm:rb_a1_s1"] == "PENDING"
    assert [c for c in w.calls() if c.startswith("GO")] == []


def test_a_supervisor_death_mid_window_holds_instead_of_rerunning_it(w):
    d = deps(w, runner_for(w), FakeShipper(w.repo))
    spec = two_arm_spec(w)
    ch = C.Chain(spec, d)
    ch.state["items"]["arm:rb_a1_s1"]["phase"] = "DONE"
    ch.state["items"]["window:between"]["phase"] = "STARTED"
    ch._save()
    ch2 = C.Chain(spec, d)
    assert ch2.run() == C.EXIT_HOLD
    assert "stopped while window between ran" in ch2.state["items"]["window:between"]["hold"]["reason"]
    ch2.clear_hold("window:between", by="orchestrator", why="the step is gone (gpu_lock --status)")
    assert phases(ch2)["window:between"] == "PENDING"


def test_abandon_refuses_while_the_launch_is_alive(w):
    """Review 10-26: abandoning a held arm whose run still lives would let the next arm GO beside it."""
    w.knobs("rb_a1_s1", FAKE_RUN_S=4, FAKE_BANNER="[TEAM-MIX] FATAL: no lever")
    arm = w.arm("a1", 1, w.pin_a, banner={"expect": "[TEAM-MIX] ok", "void": ["TEAM-MIX] FATAL"], "within_min": 1})
    d = deps(w, runner_for(w), FakeShipper(w.repo))
    spec = w.spec([arm, w.arm("a2", 2, w.pin_b)])
    ch = C.Chain(spec, d)
    assert ch.run() == C.EXIT_HOLD
    with pytest.raises(C.ChainError, match="still ALIVE"):
        ch.abandon("arm:rb_a1_s1", by="orchestrator", why="VOID")
    assert wait_gone(ProcRef.from_json(ch.state["items"]["arm:rb_a1_s1"]["data"]["wrapper"]), 30, poll_s=0.1)
    ch.abandon("arm:rb_a1_s1", by="orchestrator", why="VOID; stopped")
    assert phases(ch)["arm:rb_a1_s1"] == "DONE_FAILED"


def test_clearing_a_mid_go_hold_takes_the_orchestrators_edit(w):
    """Review 10-26: the reset kept the old snapshot, so a corrected pin made the chain refuse itself."""
    d = deps(w, runner_for(w), FakeShipper(w.repo))
    ch = C.Chain(w.spec([w.arm("a1", 1, w.pin_a)]), d)

    def die(_it: Any) -> Any:
        raise _Died()
    ch.launcher.go = die  # type: ignore[method-assign]
    with pytest.raises(_Died):
        ch.run()
    ch2 = C.Chain(w.spec([w.arm("a1", 1, w.pin_a)]), d)
    assert ch2.run() == C.EXIT_HOLD
    ch2.clear_hold("arm:rb_a1_s1", by="orchestrator", why="wrong pin; corrected")
    ch3 = C.Chain(w.spec([w.arm("a1", 1, w.pin_b)]), d)
    assert ch3.run() == C.EXIT_OK, status(w)
    assert [c for c in w.calls() if c.startswith("GO")] == [f"GO a1 1 {w.pin_b} CVD=unset"]


# -- registered actions (operator review 10-27) -------------------------------------------------------------

def _recording_kill(killed: List[Any]):
    from utils import procfs

    def kill(pid: int, sig: int) -> None:
        cmd = procfs.cmdline(pid)
        assert "rb_a1_s1" in cmd, f"a test signals only its own fake launcher, not {cmd!r}"
        killed.append((pid, sig, cmd))
        os.kill(pid, sig)
    return kill


def _void_arm(w: World, on_void: str) -> Dict[str, Any]:
    return w.arm("a1", 1, w.pin_a, on_void=on_void,
                 banner={"expect": "[TEAM-MIX] ok", "void": ["TEAM-MIX] FATAL"], "within_min": 1})


def test_a_void_arm_registered_stop_is_sigtermed_through_the_identity_gate_recorded_and_held(w):
    import signal as S
    w.knobs("rb_a1_s1", FAKE_RUN_S=30, FAKE_BANNER="[TEAM-MIX] FATAL: the lever did not load")
    killed: List[Any] = []
    d = deps(w, runner_for(w), FakeShipper(w.repo))
    d.kill = _recording_kill(killed)
    ch = C.Chain(w.spec([_void_arm(w, "stop"), w.arm("a2", 2, w.pin_b)]), d)
    assert ch.run() == C.EXIT_HOLD
    assert [(k[1], "main.launcher --run-name rb_a1_s1" in k[2]) for k in killed] == [(S.SIGTERM, True)]
    r = ch.state["items"]["arm:rb_a1_s1"]
    assert r["hold"]["reason"].startswith("VOID") and "STOPPED by the registered on_void stop" in r["hold"]["reason"]
    assert r["data"]["stopped"]["signalled"] is True and r["data"]["stopped"]["gone"] is True
    assert "arm:rb_a1_s1: STOP (on_void stop)" in status(w)
    assert [c for c in w.calls() if c.startswith("GO a2")] == []


def test_a_void_arm_registered_page_is_held_and_never_signalled(w):
    w.knobs("rb_a1_s1", FAKE_RUN_S=3, FAKE_BANNER="[TEAM-MIX] FATAL: the lever did not load")
    killed: List[Any] = []
    d = deps(w, runner_for(w), FakeShipper(w.repo))
    d.kill = _recording_kill(killed)
    ch = C.Chain(w.spec([_void_arm(w, "page")]), d)
    assert ch.run() == C.EXIT_HOLD
    assert killed == [] and "on_void page" in ch.state["items"]["arm:rb_a1_s1"]["hold"]["reason"]
    assert "stopped" not in ch.state["items"]["arm:rb_a1_s1"]["data"]


@pytest.mark.parametrize("pidfile", ["absent", "not-the-launcher"])
def test_a_stop_signals_nothing_without_this_runs_live_launcher(w, pidfile):
    """The identity gate: no live process in the pid file, or one whose cmdline is not THIS run's launcher (here the
    test process itself), is never signalled; the HOLD says so."""
    w.knobs("rb_a1_s1", FAKE_RUN_S=3, FAKE_BANNER="[TEAM-MIX] FATAL: the lever did not load")
    killed: List[Any] = []
    d = deps(w, runner_for(w), FakeShipper(w.repo))
    d.kill = _recording_kill(killed)
    ch = C.Chain(w.spec([_void_arm(w, "stop")]), d)
    fake_pidfile = os.path.join(w.root, "other.pid")
    if pidfile == "not-the-launcher":
        with open(fake_pidfile, "w") as f:
            f.write(f"{os.getpid()}\n")
    ch.launcher.launcher_pidfile = lambda it: fake_pidfile  # type: ignore[method-assign]
    assert ch.run() == C.EXIT_HOLD
    reason = ch.state["items"]["arm:rb_a1_s1"]["hold"]["reason"]
    assert killed == [] and "NOT signalled" in reason
    if pidfile == "not-the-launcher":
        assert "identity MISMATCH" in reason and ch.state["items"]["arm:rb_a1_s1"]["data"]["stopped"]["signalled"] is False


def test_a_failed_throughput_gate_registered_stop_is_sigtermed(w):
    """The fake run writes 10 steps/s checkpoints: a 1,000 steps/s bar fails, and the registered stop applies."""
    import signal as S
    w.knobs("rb_a1_s1", FAKE_RUN_S=30)
    killed: List[Any] = []
    d = deps(w, runner_for(w), FakeShipper(w.repo))
    d.kill = _recording_kill(killed)
    arm = w.arm("a1", 1, w.pin_a, on_throughput_fail="stop", throughput={"min_steps_per_s": 1000, "within_min": 1})
    ch = C.Chain(w.spec([arm]), d)
    assert ch.run() == C.EXIT_HOLD
    assert [k[1] for k in killed] == [S.SIGTERM]
    assert "STOPPED by the registered on_throughput_fail stop" in ch.state["items"]["arm:rb_a1_s1"]["hold"]["reason"]


def test_the_stop_identity_never_matches_another_run():
    import re
    ere = re.compile(C.launcher_ere("rb_x35_c0_s350"))
    assert ere.search("python -m main.launcher --pin-commit abc --run-name rb_x35_c0_s350 --device cuda")
    assert ere.search("python -m main.launcher --run-name=rb_x35_c0_s350")
    assert not ere.search("python -m main.launcher --run-name rb_x35_c0_s3501 --device cuda")
    assert not ere.search("bash launch_x35.sh c0 350 --run-name rb_x35_c0_s350")


def test_the_throughput_gate_counts_from_the_first_trainer_line_not_the_go(w):
    """Operator review 10-27: a healthy cold start's first trainer line came ~23 min after the GO. GO 50 min ago, the
    first line seen now, the bar reached 30 min later: PASS (counted from the GO it would have failed at 60)."""
    clock = {"t": 0.0}
    wall0 = 2_000_000_000.0
    d = deps(w, runner_for(w), FakeShipper(w.repo))
    d.now, d.monotonic = (lambda: wall0 + clock["t"]), (lambda: clock["t"])
    run_dir = os.path.join(w.models, "rb_a1_s1")
    os.makedirs(run_dir)
    with open(os.path.join(run_dir, "launcher_child.log"), "w") as f:
        f.write("===== child attached 2026-10-27 01:00:00 (pid 1) =====\ntrainer: building envs\n")

    def sleep(s: float) -> None:
        clock["t"] += s
        if clock["t"] >= 30 * 60 and not os.path.isdir(os.path.join(run_dir, "checkpoints")):
            os.makedirs(os.path.join(run_dir, "checkpoints"))
            for step, dt_s in ((2000, -2.0), (5000, 0.0)):
                p = os.path.join(run_dir, "checkpoints", f"checkpoint_{step}_steps.zip")
                open(p, "w").close()
                os.utime(p, (wall0 + clock["t"] + dt_s, wall0 + clock["t"] + dt_s))
    d.sleep = sleep
    arm = w.arm("a1", 1, w.pin_a, throughput={"min_steps_per_s": 1000, "within_min": 60})
    spec = w.spec([arm])
    ch = C.Chain(spec, d)
    r = ch.state["items"]["arm:rb_a1_s1"]
    r["phase"] = "BANNER_OK"
    r["data"].update(go_epoch=wall0 - 50 * 60, fork_step=1000)
    ch._arm_step(spec.arms()[0])
    r = ch.state["items"]["arm:rb_a1_s1"]
    assert r["phase"] == "RUNNING" and r["data"]["first_trainer_epoch"] == wall0
    assert "counts from the first trainer line" in status(w) and "50 min after the GO" in status(w)


# -- the record ship: asynchronous once the kit's k9b-batch writes outside main (operator review 10-27) ------

def _git_status(w: World) -> str:
    return subprocess.run(["git", "-C", w.repo, "status", "--porcelain"], capture_output=True, text=True).stdout


def _async_spawner(w: World, ok: bool, spawned: List[str]):
    """A real detached worker that writes its result only once the NEXT arm's GO is in calls.log (bounded 30 s): a
    chain that waited on the ship would never GO a2, and the worker would write a timeout instead."""
    from main.ops.arm_chain.runner import SubprocessRunner

    def spawn(spec: ChainSpec, it: Any) -> ProcRef:
        spawned.append(it.key)
        res = C.ship_result_path(spec, it)
        calls = os.path.join(w.job, "fake", "calls.log")
        body = '{"ok": true, "commit": "c0ffee123"}' if ok else '{"ok": false, "reason": "push failed twice"}'
        script = (f"for i in $(seq 600); do grep -q '^GO a2' {calls} 2>/dev/null && break; sleep 0.05; done; "
                  f"grep -q '^GO a2' {calls} && echo '{body}' > {res}.tmp || echo '{{\"ok\": false, \"reason\": "
                  f"\"the chain waited on the ship\"}}' > {res}.tmp; mv {res}.tmp {res}")
        os.makedirs(os.path.dirname(res), exist_ok=True)
        return SubprocessRunner().spawn(["bash", "-c", script], cwd=w.root, env=dict(os.environ),
                                        out_path=res + ".worker.txt")
    return spawn


def test_an_async_ship_lets_the_next_arm_go_while_the_records_are_in_flight(w):
    sh, spawned = FakeShipper(w.repo), []
    d = deps(w, runner_for(w), sh)
    d.kit_k9b_fixed = lambda s: (True, "test: the kit writes outside main")
    d.ship_spawner = _async_spawner(w, True, spawned)
    ch = C.Chain(w.spec([w.arm("a1", 1, w.pin_a), w.arm("a2", 2, w.pin_b)]), d)
    assert ch.run() == C.EXIT_OK, status(w)
    assert spawned == ["arm:rb_a1_s1", "arm:rb_a2_s2"] and sh.shipped == []
    assert [o["outcome"]["ok"] for o in ch.state["ships"].values()] == [True, True]
    assert "records ship ASYNC" in status(w) and "chain COMPLETE" in status(w)
    req = json.loads(open(C.ship_request_path(ch.spec, ch.spec.arms()[0])).read())
    assert sorted(req["files"]) == [os.path.join(w.stage_rel, "dose", "rb_a1_s1.json"),
                                    os.path.join(w.stage_rel, "k9b", "rb_a1_s1.json")]


def test_a_failed_async_ship_pages_ship_hold_without_idling_the_gpu(w):
    """The point of the async ship: a ship HOLD no longer stops the next GO; it pages, and the chain's end says so."""
    spawned: List[str] = []
    d = deps(w, runner_for(w), FakeShipper(w.repo))
    d.kit_k9b_fixed = lambda s: (True, "test")
    d.ship_spawner = _async_spawner(w, False, spawned)
    ch = C.Chain(w.spec([w.arm("a1", 1, w.pin_a), w.arm("a2", 2, w.pin_b)]), d)
    assert ch.run() == C.EXIT_HOLD
    assert f"GO a2 2 {w.pin_b} CVD=unset" in w.calls()
    assert set(phases(ch).values()) == {"DONE"}
    assert "SHIP-HOLD: every item is DONE but records are NOT all shipped" in status(w)
    assert "chain COMPLETE" not in status(w)
    # the by-hand retry (``ship-records``) lands them; the next ``run`` takes it and completes the chain
    for it in ch.spec.arms():
        assert C.ship_worker(ch.spec, it.key, FakeShipper(w.repo)) == C.EXIT_OK
    assert C.Chain(ch.spec, d).run() == C.EXIT_OK and "chain COMPLETE" in status(w)


def test_a_ship_worker_that_dies_without_a_result_pages(w):
    from main.ops.arm_chain.runner import SubprocessRunner
    d = deps(w, runner_for(w), FakeShipper(w.repo))
    d.kit_k9b_fixed = lambda s: (True, "test")
    d.ship_spawner = lambda spec, it: SubprocessRunner().spawn(["true"], cwd=w.root, env=dict(os.environ),
                                                               out_path=os.path.join(w.root, "worker.txt"))
    ch = C.Chain(w.spec([w.arm("a1", 1, w.pin_a)]), d)
    assert ch.run() == C.EXIT_HOLD
    assert "SHIP-HOLD: arm:rb_a1_s1: the ship worker died without a result" in status(w)


def test_an_unfixed_kit_keeps_the_ship_synchronous(w):
    """Until the kit's k9b-batch writes outside main (F1), the async path refuses: the records land before the next
    dry run can run k9b-batch."""
    sh, spawned = FakeShipper(w.repo), []
    d = deps(w, runner_for(w), sh)
    d.ship_spawner = _async_spawner(w, True, spawned)
    ch = C.Chain(w.spec([w.arm("a1", 1, w.pin_a)]), d)
    assert ch.run() == C.EXIT_OK
    assert spawned == [] and len(sh.shipped) == 1
    assert "the records ship is SYNCHRONOUS (test: the kit's k9b-batch still writes into main's stage)" in status(w)


def test_a_dirty_main_holds_before_the_dry_run(w):
    with open(os.path.join(w.repo, "stray.txt"), "w") as f:
        f.write("x")
    ch = C.Chain(w.spec([w.arm("a1", 1, w.pin_a)]), deps(w, runner_for(w), FakeShipper(w.repo)))
    assert ch.run() == C.EXIT_HOLD
    assert "main is DIRTY before the dry run" in ch.state["items"]["arm:rb_a1_s1"]["hold"]["reason"]
    assert w.calls() == []


def test_a_dry_run_that_dirties_main_holds_and_never_goes(w):
    """F1's shape: the unfixed kit's k9b-batch wrote a missing record into main's stage during the dry run."""
    w.knobs("rb_a1_s1", FAKE_DIRTY_MAIN=1)
    ch = C.Chain(w.spec([w.arm("a1", 1, w.pin_a)]), deps(w, runner_for(w), FakeShipper(w.repo)))
    assert ch.run() == C.EXIT_HOLD
    assert "main is DIRTY after the dry run" in ch.state["items"]["arm:rb_a1_s1"]["hold"]["reason"]
    assert [c for c in w.calls() if c.startswith("GO")] == [] and "written_by_k9b_batch" in _git_status(w)


class _FailingShipper(FakeShipper):
    def ship(self, files: Mapping[str, str], message: str, tag: str) -> ShipResult:
        from main.ops.arm_chain.launch import Hold
        raise Hold("static tier failed (rc 1) — not shipped")


def test_the_ship_worker_writes_its_result_and_one_line_and_is_the_retry(w):
    sh = FakeShipper(w.repo)
    d = deps(w, runner_for(w), sh)
    spec = w.spec([w.arm("a1", 1, w.pin_a)])
    C.Chain(spec, d).run()                                   # a synchronous chain leaves the request file behind
    it = spec.arms()[0]
    assert C.ship_worker(spec, it.key, _FailingShipper(w.repo)) == C.EXIT_HOLD
    assert json.loads(open(C.ship_result_path(spec, it)).read())["ok"] is False
    assert "SHIP-HOLD: arm:rb_a1_s1: static tier failed" in status(w) and "ship-records" in status(w)
    assert C.ship_worker(spec, it.key, sh) == C.EXIT_OK
    assert json.loads(open(C.ship_result_path(spec, it)).read())["commit"] == "feedface1"
    assert "arm:rb_a1_s1: records shipped as feedface1 (async)" in status(w)


def test_the_kit_check_refuses_until_the_launch_script_and_main_code_write_outside_main(w):
    """``ship.kit_k9b_out_of_main``: the launch script's k9b-batch must pass --out-dir AND main's batch module must
    refuse a write inside main; either missing keeps the ship synchronous."""
    import sys
    from main.ops.arm_chain.ship import kit_k9b_out_of_main
    spec = w.spec([w.arm("a1", 1, w.pin_a)])
    sh = os.path.join(w.repo, "kit", "launch_x35.sh")
    body = open(sh).read()
    with open(sh, "a") as f:
        f.write('\n  kit k9b-batch --stage "$STAGE" --models-dir "$MODELS" > "$out.k9b.txt" 2>&1\n')
    ok, why = kit_k9b_out_of_main(spec, python=sys.executable)
    assert ok is False and "no --out-dir" in why
    with open(sh, "w") as f:
        f.write(body + '\n  kit k9b-batch --stage "$STAGE" --models-dir "$MODELS" --out-dir "$K9B_OUT" > x 2>&1\n')
    pkg = os.path.join(w.repo, "src", "main", "x35")
    os.makedirs(pkg)
    for p in (os.path.join(w.repo, "src", "main", "__init__.py"), os.path.join(pkg, "__init__.py")):
        open(p, "w").close()
    with open(os.path.join(pkg, "batch.py"), "w") as f:
        f.write("def k9b_batch(stage, models_dir, *, reader=None):\n    pass\n")
    ok, why = kit_k9b_out_of_main(spec, python=sys.executable)
    assert ok is False and "takes no out_dir" in why
    with open(os.path.join(pkg, "batch.py"), "w") as f:
        f.write("import os\nclass BatchError(Exception): pass\n"
                "def k9b_batch(stage, models_dir, *, out_dir=None, reader=None):\n    pass\n"
                "def refuse_inside_main_checkout(path, *, roots=None):\n"
                "    if any(os.path.realpath(path).startswith(r + os.sep) for r in roots):\n"
                "        raise BatchError(path)\n")
    ok, why = kit_k9b_out_of_main(spec, python=sys.executable)
    assert ok is True, why


def test_the_kit_check_passes_on_this_checkouts_real_kit(tmp_path):
    """F12: the check keys on the kit fix's names as landed (7f9ff6236); a rename in the kit fails HERE, not by a
    silent fall-back to the synchronous ship in production."""
    import sys
    from main.ops.arm_chain.ship import kit_k9b_out_of_main
    from main.ops.arm_chain.spec import parse
    from utils.paths import repo_path
    stage = "designs/research_state/measurements/x35_breakers_stage_2026-10-17"
    d = {"schema": "arm_chain_v1", "name": "probe", "job_dir": str(tmp_path), "repo": str(repo_path()),
         "python": sys.executable,
         "backend": {"kind": "x35", "launch_script": f"{stage}/launch_x35.sh", "stage": stage, "lease_owner": "x",
                     "lease_token_file": str(tmp_path / "token")},
         "records": {"stage": stage},
         "items": [{"kind": "arm", "arm": "n1p", "seed": 1, "pin": "a" * 40, "run": "r", "go_by": "x", "thin": "planner",
                    "disk_need_gib": 1, "on_void": "page", "on_throughput_fail": "page"}]}
    ok, why = kit_k9b_out_of_main(parse(d))
    assert ok is True, why
