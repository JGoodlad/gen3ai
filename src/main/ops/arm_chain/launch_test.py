"""Pre-launch and GO: the kit's dry run is re-verified independently, at the ARM's pin, with the card visible.

``integration``: the kit is the bash stand-in from ``testing_fakes`` (a real subprocess in a temp checkout).
"""
from __future__ import annotations

import os
from typing import Any, List

import pytest

from main.ops.arm_chain.cpugate import GateResult
from main.ops.arm_chain.launch import Hold, X35Launcher, dry_transcript_path, verify_dry_transcript
from main.ops.arm_chain.testing_fakes import FakeRunner, World, make_world

pytestmark = pytest.mark.integration

PIN = "5f8f6cfa37de8e283ba553c0d415680cf51bff18"
PARENT = "/m/models/rb_x26_s2601"
# trimmed from a real launcher --dry-run transcript (rb_x35_n1p_s3501, 2026-10-26)
REAL = f"""
── launcher --dry-run — resolving only, nothing will be created ────────────
  role        : FORK of {PARENT}
  run dir     : /m/models/rb_x35_n1p_s3501   [would be created]
  pin         : {PIN}  (source: pin_commit)
  steps       : --steps 300,000,170 vs checkpoint at 280,000,170 (sidecar) → +20,000,000 steps
  cuda devices : ✓ CUDA_VISIBLE_DEVICES unset
  ✓ DRY RUN — this command would launch. Nothing was created or modified.
"""


def test_the_real_transcript_shape_verifies():
    assert verify_dry_transcript(REAL, PIN, PARENT, "+20,000,000 steps") == []


@pytest.mark.parametrize("mutate, why", [
    (lambda t: t.replace("FORK of", "RESTART of"), "role"),
    (lambda t: t.replace(PARENT, "/m/models/rb_other"), "forks"),
    (lambda t: t.replace(PIN, "f4b25238def57900c0305bb2a3a5a8f9edaef285"), "pin"),
    (lambda t: t.replace("✓ CUDA_VISIBLE_DEVICES unset", "✗ CUDA_VISIBLE_DEVICES is EMPTY"), "cuda devices"),
    (lambda t: t.replace("this command would launch", "REFUSED"), "would launch"),
    (lambda t: t.replace("+20,000,000", "+5,000,000"), "step arithmetic"),
])
def test_each_departure_from_the_meant_launch_is_named(mutate: Any, why: str):
    """A dry run that would launch SOMETHING is not the arm (a RESTART, another parent, another pin, no card)."""
    bad = verify_dry_transcript(mutate(REAL), PIN, PARENT, "+20,000,000 steps")
    assert bad and any(why in b for b in bad), bad


def _launcher(w: World, runner: FakeRunner, gate: Any = None, polls: int = 2) -> X35Launcher:
    spec = w.spec([w.arm("a1", 1, w.pin_a, verify=[["true"]])])
    env = {"PATH": os.environ["PATH"], "HOME": os.environ.get("HOME", "/tmp"), "CUDA_VISIBLE_DEVICES": ""}
    return X35Launcher(spec, runner, env, gate=gate or (lambda: GateResult(True, True, 1.0, (), "PASS")),
                       sleep=lambda s: None, gate_polls=polls, niceness=lambda: 0)


@pytest.fixture
def w(tmp_path) -> World:
    return make_world(str(tmp_path))


def test_prelaunch_passes_at_the_arms_own_pin_with_the_card_visible(w):
    """CB-02: the CPU steps' empty CUDA_VISIBLE_DEVICES must never reach the dry run or the GO."""
    lau = _launcher(w, FakeRunner())
    d = lau.prelaunch(lau.spec.arms()[0])
    assert d.run_name == "rb_a1_s1" and d.steps == 5000
    assert w.calls() == [f"DRY a1 1 {w.pin_a} CVD=unset"]


def test_a_describe_naming_another_run_holds(w):
    """H-06: the describe is read at its deterministic path and must name THIS run."""
    os.makedirs(os.path.join(w.job, "fake"), exist_ok=True)
    open(os.path.join(w.job, "fake", "a1.1.run"), "w").write("rb_someone_else_s1")
    lau = _launcher(w, FakeRunner())
    with pytest.raises(Hold, match="names run 'rb_someone_else_s1'"):
        lau.prelaunch(lau.spec.arms()[0])


def test_a_stale_transcript_is_never_read_as_this_launch(w):
    lau = _launcher(w, FakeRunner())
    it = lau.spec.arms()[0]
    lau.prelaunch(it)
    os.utime(dry_transcript_path(lau.spec, it), (1, 1))
    lau._now = lambda: 10_000_000_000.0   # this dry run "starts" after every file on disk
    with pytest.raises(Hold, match="not written by THIS dry run"):
        lau.prelaunch(it)


def test_a_verify_command_failing_at_the_go_never_spawns(w):
    """AM-X52: the practised team files are verified before the dry run AND again right before the GO."""
    runner = FakeRunner()
    state = {"n": 0}

    def verify(argv: List[str], out: str) -> int:
        state["n"] += 1
        open(out, "w").write("verify\n")
        return 0 if state["n"] == 1 else 1
    runner.on(lambda a: list(a) == ["true"], verify)
    lau = _launcher(w, runner)
    it = lau.spec.arms()[0]
    lau.prelaunch(it)
    with pytest.raises(Hold, match="at the GO"):
        lau.go(it)
    assert [c for c in w.calls() if c.startswith("GO")] == []


def test_a_go_never_relaunches_an_existing_run(w):
    lau = _launcher(w, FakeRunner())
    it = lau.spec.arms()[0]
    os.makedirs(os.path.join(w.models, it.run))
    with pytest.raises(Hold, match="already exists"):
        lau.go(it)


def test_the_cpu_gate_holds_after_its_polls_and_never_dry_runs(w):
    lau = _launcher(w, FakeRunner(), gate=lambda: GateResult(False, True, 0.2, (), "FAIL: busy"), polls=3)
    with pytest.raises(Hold, match="never PASSED in 3 polls"):
        lau.prelaunch(lau.spec.arms()[0])
    assert w.calls() == []


def test_the_go_env_is_an_allow_list(w):
    runner = FakeRunner()
    lau = _launcher(w, runner)
    lau.base_env["GEN3AI_TEST_ALLOW_GPU"] = "1"
    lau.base_env["PYTHONPATH"] = "/some/worktree/src"
    lau.prelaunch(lau.spec.arms()[0])
    kit = [env for argv, env in runner.calls if argv[0].endswith("launch_x35.sh")]
    assert kit and "CUDA_VISIBLE_DEVICES" not in kit[0] and "GEN3AI_TEST_ALLOW_GPU" not in kit[0]
    assert kit[0]["PYTHONPATH"] == os.path.join(w.repo, "src") and kit[0]["X35_JOB_DIR"] == w.job


def test_plan_describe_refuses_an_arm_the_kit_does_not_know(w):
    """CB-01 (10-20 06:46): ``x47`` reached the GO; the kit knows only ``toy_x47``. ``plan --describe`` asks first."""
    def refuse(argv: Any, out: str) -> int:
        open(out, "w").write("[x35 kit] REFUSED: arm toy_x47 reads the toy call record, got side 'production'\n")
        return 2
    lau = _launcher(w, FakeRunner().on(lambda a: "describe" in a, refuse))
    with pytest.raises(Hold, match="the kit refuses arm 'a1'"):
        lau.describe_offline(lau.spec.arms()[0])


def test_plan_describe_refuses_a_run_name_the_kit_does_not_give(w):
    def other(argv: Any, out: str) -> int:
        open(out, "w").write('warning on stderr\n{"run_name": "rb_other_s1", "parent_run_dir": "/p", '
                             '"delta_text": "+1 steps", "argv": ["--steps", "5"]}\n')
        return 0
    lau = _launcher(w, FakeRunner().on(lambda a: "describe" in a, other))
    with pytest.raises(Hold, match="names arm a1 s1's run 'rb_other_s1'"):
        lau.describe_offline(lau.spec.arms()[0])


def test_a_niced_supervisor_never_goes(w):
    """A GO inherits the supervisor's niceness and RLIMIT_NICE forbids lowering it: a nice-19 trainer for life."""
    lau = _launcher(w, FakeRunner())
    lau._niceness = lambda: 19
    with pytest.raises(Hold, match="nice 19"):
        lau.go(lau.spec.arms()[0])
    assert w.calls() == []


def test_the_go_env_pins_the_chains_interpreter_and_drops_blocking(w):
    runner = FakeRunner()
    lau = _launcher(w, runner)
    lau.base_env.update({"GEN3AI_PYTHON": "/elsewhere/python", "GEN3AI_BLOCKING": "x"})
    lau.prelaunch(lau.spec.arms()[0])
    kit = [env for argv, env in runner.calls if argv[0].endswith("launch_x35.sh")][0]
    assert kit["GEN3AI_PYTHON"] == lau.spec.python and "GEN3AI_BLOCKING" not in kit
