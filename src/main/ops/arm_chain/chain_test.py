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
from typing import Any, Dict, List, Mapping

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
                  lease=lambda: "LEASED: lease owner 'Training Oct', token …")


def runner_for(w: World, **planner: Any) -> FakeRunner:
    return FakeRunner().on(*k9b_handler()).on(*planner_handler(w.models, **planner))


def status(w: World) -> str:
    return open(os.path.join(w.job, "status.txt")).read()


def phases(ch: C.Chain) -> Dict[str, str]:
    return {k: v["phase"] for k, v in ch.state["items"].items()}


@pytest.fixture
def w(tmp_path) -> World:
    return make_world(str(tmp_path))


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
