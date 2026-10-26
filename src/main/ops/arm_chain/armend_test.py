"""The arm end, step by step: completeness, K9(b) into a scratch stage, the thin, the dose record, the END wait.

``integration``: the K9(b) guard runs git in a temp checkout; the END-wait tests spawn their own ``sleep``.
"""
from __future__ import annotations

import os
import subprocess
from typing import Any, List

import pytest

from main.ops.arm_chain import armend as A
from main.ops.arm_chain.launch import Describe, Hold
from main.ops.arm_chain.procwatch import ProcRef, ref_of
from main.ops.arm_chain.statuslog import StatusLog, end_body, launch_tag
from main.ops.arm_chain.testing_fakes import FakeRunner, World, k9b_handler, make_world, planner_handler

pytestmark = pytest.mark.integration


def _run_dir(models: str, run: str, steps: List[int], final: bool = True) -> str:
    rd = os.path.join(models, run)
    os.makedirs(os.path.join(rd, "checkpoints"), exist_ok=True)
    for s in steps:
        open(os.path.join(rd, "checkpoints", f"checkpoint_{s}_steps.zip"), "w").close()
    if final:
        open(os.path.join(rd, "final_model.zip"), "w").close()
    return rd


def _summary(last: int) -> A.LauncherSummary:
    return A.LauncherSummary(last_step=last, fps=1400, restarts=0, pins=["a" * 40])


# --- completeness (CB-08) -------------------------------------------------------------------------------------------

def test_the_d1wide_s3501_shape_is_stopped_not_finished(tmp_path):
    """rc 0 after a SIGTERM: last step short of --steps, no final_model.zip."""
    rd = _run_dir(str(tmp_path), "r", [287_000_000, 288_000_000], final=False)
    with pytest.raises(Hold, match="stopped, not finished"):
        A.completeness(rd, 300_000_170, 0, _summary(288_101_376))


@pytest.mark.parametrize("below", [29, 160])
def test_a_finished_run_whose_end_checkpoint_lands_just_below_steps_is_complete(tmp_path, below: int):
    """The end-step checkpoint lands 29-160 steps under --steps; a strict >= on it would HOLD every good run."""
    rd = _run_dir(str(tmp_path), "r", [299_000_000, 300_000_170 - below])
    assert "complete" in A.completeness(rd, 300_000_170, 0, _summary(300_000_170))


def test_a_missing_final_model_alone_holds(tmp_path):
    rd = _run_dir(str(tmp_path), "r", [300_000_000], final=False)
    with pytest.raises(Hold, match="final_model.zip ABSENT"):
        A.completeness(rd, 300_000_000, 0, _summary(300_000_000))


def test_parse_launcher_log_reads_the_last_summary():
    s = A.parse_launcher_log("x --pin-commit " + "b" * 40 + "\n  Last step   : 1,000\n  FPS : 9\n"
                             "  Last step   : 2,000\n  FPS         : 1,400\n  Restarts    : 1\n")
    assert (s.last_step, s.fps, s.restarts, s.pins) == (2000, 1400, 1, ["b" * 40])


# --- K9(b) into a scratch stage (CB-11) ------------------------------------------------------------------------------

@pytest.fixture
def w(tmp_path) -> World:
    return make_world(str(tmp_path))


def test_k9b_writes_only_the_scratch_stage_and_main_stays_clean(w):
    spec = w.spec([w.arm("a1", 1, w.pin_a)])
    rec = A.k9b_scratch(spec, spec.arms()[0], FakeRunner().on(*k9b_handler()), {"PATH": os.environ["PATH"]})
    assert rec.startswith(spec.work_dir) and os.path.isfile(rec)
    assert subprocess.run(["git", "-C", w.repo, "status", "--porcelain"], capture_output=True, text=True).stdout == ""
    assert not os.path.exists(os.path.join(w.repo, w.stage_rel, "k9b", "rb_a1_s1.json"))


def test_a_k9b_read_that_writes_into_mains_stage_holds(w):
    """arm_end v3-v12 passed main's stage to ``kit k9b --stage``: main was dirty after every arm end."""
    spec = w.spec([w.arm("a1", 1, w.pin_a)])
    runner = FakeRunner().on(*k9b_handler(into_main_stage=os.path.join(w.repo, w.stage_rel)))
    with pytest.raises(Hold, match="changed MAIN's records stage"):
        A.k9b_scratch(spec, spec.arms()[0], runner, {"PATH": os.environ["PATH"]})


def test_k9b_reads_the_verdict_from_the_record_not_from_grep(w):
    """H-07: ``grep -q PASS`` on the whole output; a VOID record with rc 0 must still HOLD."""
    spec = w.spec([w.arm("a1", 1, w.pin_a)])
    with pytest.raises(Hold, match="verdict 'VOID'"):
        A.k9b_scratch(spec, spec.arms()[0], FakeRunner().on(*k9b_handler(verdict="VOID")), {"PATH": "/usr/bin"})


def test_k9b_refuses_a_dirty_main_stage_before_reading(w):
    open(os.path.join(w.repo, w.stage_rel, "stray.json"), "w").write("{}")
    spec = w.spec([w.arm("a1", 1, w.pin_a)])
    with pytest.raises(Hold, match="dirty BEFORE"):
        A.k9b_scratch(spec, spec.arms()[0], FakeRunner().on(*k9b_handler()), {"PATH": "/usr/bin"})


# --- the thin (CB-09, CB-10, CB-21, H-05) ----------------------------------------------------------------------------

def _thin_world(w: World, **arm: Any) -> Any:
    spec = w.spec([w.arm("a1", 1, w.pin_a, **arm)])
    _run_dir(w.models, "rb_a1_s1", [2000, 3000, 4000, 4999])
    return spec, spec.arms()[0]


def test_a_keep_all_arm_is_never_thinned(w):
    spec, it = _thin_world(w, thin="none", thin_reason="AM-R0 reads every 2M point")
    runner = FakeRunner().on(*planner_handler(w.models))
    assert A.thin(spec, it, runner, {}, 5000).startswith("NOT thinned (AM-R0")
    assert runner.calls == []
    assert len(os.listdir(os.path.join(w.models, "rb_a1_s1", "checkpoints"))) == 4


def test_the_open_run_dir_refusal_is_retried_and_others_are_not(w):
    spec, it = _thin_world(w)
    runner = FakeRunner().on(*planner_handler(w.models, refuse_open_times=2))
    assert "thinned" in A.thin(spec, it, runner, {}, 5000, sleep=lambda s: None)
    assert os.listdir(os.path.join(w.models, "rb_a1_s1", "checkpoints")) == ["checkpoint_4999_steps.zip"]
    spec2, it2 = spec, it
    runner2 = FakeRunner().on(*planner_handler(w.models, other_refusal=True))
    with pytest.raises(Hold, match="thin dry run rc=3"):
        A.thin(spec2, it2, runner2, {}, 5000, sleep=lambda s: None)
    assert len(runner2.calls) == 1


def test_a_plan_that_drops_the_end_step_checkpoint_holds(w):
    spec, it = _thin_world(w)
    pred, fn = planner_handler(w.models)

    def drop_newest(argv: Any, out: str) -> int:
        open(out, "w").write("checkpoint zips kept 1\n    checkpoint_2000_steps.zip   first\n")
        return 0
    with pytest.raises(Hold, match="does NOT keep the end-step checkpoint_4999"):
        A.thin(spec, it, FakeRunner().on(pred, drop_newest), {}, 5000)


def test_a_checkpoint_a_later_window_reads_blocks_the_thin(w):
    """CB-21: the 10-25 segment-1 thin deleted the u1000 neighbours a GPU replay hard-coded."""
    ck = os.path.join(w.models, "rb_a1_s1", "checkpoints", "checkpoint_3000_steps.zip")
    win = {"kind": "window", "name": "replay", "worktree": w.repo, "require_commit": w.pin_a, "budget_min": 5,
           "on_fail": "hold", "preempt_marker": "NO_GO",
           "steps": [{"name": "r", "argv": ["python3", "replay.py", f"--ckpt=3M={ck}"], "gpu": True, "mem_gb": 8,
                      "timeout_s": 60}]}
    spec = w.spec([w.arm("a1", 1, w.pin_a), win])
    _run_dir(w.models, "rb_a1_s1", [2000, 3000, 4000, 4999])
    assert A.later_evidence(spec, spec.arms()[0]) == ["checkpoint_3000_steps.zip"]
    with pytest.raises(Hold, match="a later chain item reads"):
        A.thin(spec, spec.arms()[0], FakeRunner().on(*planner_handler(w.models)), {}, 5000)
    keep = FakeRunner().on(*planner_handler(w.models, keep_extra=["checkpoint_3000_steps.zip"]))
    assert "thinned" in A.thin(spec, spec.arms()[0], keep, {}, 5000)
    assert os.path.exists(ck)


# --- the dose record (H-06, H-16) ------------------------------------------------------------------------------------

def _desc() -> Describe:
    return Describe(run_name="rb_x_s1", parent_run_dir="/p", delta_text="+4,000 steps", steps=5000,
                    argv=["--model", "/p/checkpoints/checkpoint_1000_steps.zip", "--steps", "5000"])


def test_dose_sums_every_segment_and_refuses_a_cross_pin_log(w):
    it = w.spec([w.arm("x", 1, w.pin_a, run="rb_x_s1")]).arms()[0]
    t = launch_tag("x", 1)
    lines = [f"[2026-10-26T01:00:00-07:00] {t}: START rb_x_s1: argv",
             f"[2026-10-26T01:30:00-07:00] {t}: END rb_x_s1 launcher_rc=0",
             f"[2026-10-26T01:40:00-07:00] {t}: START rb_x_s10: another run whose name extends this one",
             f"[2026-10-26T02:00:00-07:00] {t}: START rb_x_s1: resumed",
             f"[2026-10-26T03:00:00-07:00] {t}: END rb_x_s1 launcher_rc=0"]
    log = f"--pin-commit {w.pin_a}\n  Last step   : 5,000\n  FPS         : 1,400\n  Restarts    : 0\n"
    d = A.dose_record(it, lines, _desc(), log, 0)
    assert (d["wall_min"], d["segments"], d["first_step"], d["last_step"]) == (90.0, 2, 1000, 5000)
    with pytest.raises(Hold, match="cross-pin"):
        A.dose_record(it, lines, _desc(), log.replace(w.pin_a, w.pin_b), 0)


def test_dose_refuses_a_run_with_no_segment(w):
    it = w.spec([w.arm("x", 1, w.pin_a, run="rb_x_s1")]).arms()[0]
    with pytest.raises(Hold, match="no START"):
        A.dose_record(it, ["[2026-10-26T01:00:00-07:00] x s1: END rb_x_s1 launcher_rc=0"], _desc(), "", 0)


# --- the END wait (CB-17, CB-26) -------------------------------------------------------------------------------------

def _sleep_named(name: str, secs: float) -> subprocess.Popen:
    return subprocess.Popen(["bash", "-c", f"exec -a '{name}' sleep {secs}"])


def test_a_dead_wrapper_with_a_live_launcher_keeps_the_wait_going(tmp_path):
    """CB-17: the END writer died with its parent; the run was still live and must not read as ended."""
    log = StatusLog(str(tmp_path / "status.txt"), "ARMCHAIN[t]")
    end = log.arm("END", launch_tag("a", 1), end_body("rb_a_s1"))
    lp = _sleep_named("python -m main.launcher --run-name rb_a_s1", 30)
    try:
        lref = ref_of(lp.pid)
        assert lref is not None
        (tmp_path / "rb_a_s1.pid").write_text(str(lp.pid))
        ticks = {"n": 0}

        def sleep(s: float) -> None:
            ticks["n"] += 1
            if ticks["n"] == 3:   # the launcher's own END arrives later, written by someone still alive
                with open(log.path, "a") as f:
                    f.write("[2026-10-26T01:00:00-07:00] a s1: END rb_a_s1 launcher_rc=0\n")
        dead_wrapper = ProcRef(999_999_999, 1)
        assert A.wait_end(log, end, dead_wrapper, str(tmp_path / "rb_a_s1.pid"), 60, sleep=sleep) == 0
    finally:
        lp.kill()
        lp.wait()


def test_both_gone_without_an_end_line_holds(tmp_path):
    log = StatusLog(str(tmp_path / "status.txt"), "ARMCHAIN[t]")
    end = log.arm("END", launch_tag("a", 1), end_body("rb_a_s1"))
    with pytest.raises(Hold, match="rc is unreadable"):
        A.wait_end(log, end, ProcRef(999_999_999, 1), str(tmp_path / "none.pid"), 60, sleep=lambda s: None)


def test_wait_end_never_uses_kill_zero():
    """CB-26 (acf2193fa): ``kill -0`` read a live supervisor as dead in a sandboxed shell."""
    src = open(A.__file__).read() + open(os.path.join(os.path.dirname(A.__file__), "procwatch.py")).read()
    assert "os.kill(" not in src.replace("kill: Callable[[int, int], None] = os.kill", "")
