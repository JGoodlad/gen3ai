"""Regression tests for the post-GO checks (postspawn.py): pool-seed PASS, banner VOID guard, throughput gate."""
import os

import pytest

from main.ops.arm_chain import postspawn as ps
from main.ops.arm_chain import statuslog as sl
from main.ops.arm_chain.launch import Hold
from main.ops.arm_chain.procwatch import ProcRef, ref_of
from main.ops.arm_chain.spec import Banner, Throughput


class FakeClock(ps.Clock):
    """sleep() advances the fake monotonic; nothing ever really waits."""

    def __init__(self):
        self.now = 0.0
        self.sleeps = []
        super().__init__(sleep=self._sleep, monotonic=lambda: self.now)

    def _sleep(self, s):
        self.sleeps.append(s)
        self.now += s


BANNER = Banner(expect="[TEAM-MIX] d3_setmut_v1@ALL", void=("[TEAM-MIX] FATAL", "does NOT inherit"), within_min=5)


# -- banner_guard -------------------------------------------------------------------------------------------

def test_banner_found_when_only_in_launcher_stdout_log(tmp_path):
    """CB-06: d3_redirect watched the wrong log; the banner lives in the launcher's stdout, launcher_child.log is empty."""
    child = tmp_path / "launcher_child.log"
    child.write_text("unrelated\n")
    stdout = tmp_path / "launcher_stdout.log"
    stdout.write_text("x\n[TEAM-MIX] d3_setmut_v1@ALL ok\n")
    line = ps.banner_guard([str(child), str(stdout)], BANNER, clock=FakeClock())
    assert line == "[TEAM-MIX] d3_setmut_v1@ALL ok"


def test_banner_only_in_the_log_we_did_not_watch_would_hold(tmp_path):
    """CB-06 converse: watching only launcher_child.log (the wrong log) never sees the banner, so it HOLDs."""
    child = tmp_path / "launcher_child.log"
    child.write_text("unrelated\n")
    (tmp_path / "launcher_stdout.log").write_text("[TEAM-MIX] d3_setmut_v1@ALL\n")
    with pytest.raises(Hold, match="^VOID"):
        ps.banner_guard([str(child)], BANNER, clock=FakeClock())


def test_void_line_wins_over_present_banner(tmp_path):
    """A fork that logs both the banner and '[TEAM-MIX] FATAL' trains the wrong recipe: VOID wins."""
    log = tmp_path / "c.log"
    log.write_text("[TEAM-MIX] d3_setmut_v1@ALL\n[TEAM-MIX] FATAL no mix\n")
    with pytest.raises(Hold) as e:
        ps.banner_guard([str(log)], BANNER, clock=FakeClock())
    assert str(e.value).startswith("VOID") and "FATAL" in str(e.value)


def test_no_banner_by_deadline_holds(tmp_path):
    """No banner by within_min is a HOLD, after waiting about that long on the fake clock."""
    log = tmp_path / "c.log"
    log.write_text("nothing here\n")
    clock = FakeClock()
    with pytest.raises(Hold, match="VOID"):
        ps.banner_guard([str(log)], BANNER, poll_s=10, clock=clock)
    assert 300 < clock.now <= 320


def test_no_log_within_wait_for_log_holds(tmp_path):
    """No child log at all within wait_for_log_s: the banner cannot be checked, so HOLD (never pass)."""
    clock = FakeClock()
    with pytest.raises(Hold, match="no child log"):
        ps.banner_guard([str(tmp_path / "a.log"), str(tmp_path / "b.log")], BANNER, wait_for_log_s=100, poll_s=10,
                        clock=clock)
    assert 100 < clock.now <= 110


def test_log_appearing_late_then_banner_passes(tmp_path):
    """The log may appear after some polls (a slow cold start); the guard then reads it."""
    log = tmp_path / "late.log"
    clock = FakeClock()
    orig = clock._sleep

    def sleep(s):
        orig(s)
        if clock.now >= 30:
            log.write_text("[TEAM-MIX] d3_setmut_v1@ALL\n")
    clock.sleep = sleep
    assert ps.banner_guard([str(log)], BANNER, poll_s=10, clock=clock).startswith("[TEAM-MIX]")


# -- throughput_gate ----------------------------------------------------------------------------------------

def mk_ckpts(run_dir, items):
    d = os.path.join(run_dir, "checkpoints")
    os.makedirs(d, exist_ok=True)
    for step, mtime in items:
        p = os.path.join(d, f"checkpoint_{step}_steps.zip")
        open(p, "w").close()
        os.utime(p, (mtime, mtime))


T0 = 1_700_000_000.0


def test_checkpoints_at_or_below_fork_step_are_ignored(tmp_path):
    """The parent's checkpoints (step <= fork step) are not this run's rate evidence."""
    mk_ckpts(str(tmp_path), [(1000, T0), (5000, T0 + 0.001)])      # a 4M steps/s "pair", all <= fork step
    tp = Throughput(min_steps_per_s=1000, within_min=1)
    with pytest.raises(Hold, match="fewer than two new checkpoints"):
        ps.throughput_gate(str(tmp_path), 5000, tp, poll_s=10, clock=FakeClock())
    assert ps.new_checkpoints(str(tmp_path), 1000) == [(5000, T0 + 0.001)]


def test_1500_steps_per_s_passes_a_1000_bar(tmp_path):
    """Two new checkpoints 2 s apart and 3,000 steps apart are 1,500 steps/s: PASS."""
    mk_ckpts(str(tmp_path), [(1000, T0 - 99), (2000, T0), (5000, T0 + 2)])
    out = ps.throughput_gate(str(tmp_path), 1000, Throughput(1000, 60), poll_s=10, clock=FakeClock())
    assert "1,500 steps/s" in out and ">= 1,000" in out


def test_500_steps_per_s_holds_at_deadline_naming_best_rate(tmp_path):
    """500 steps/s against a 1,000 bar HOLDs at the deadline, naming the best rate."""
    mk_ckpts(str(tmp_path), [(2000, T0), (3000, T0 + 2)])
    clock = FakeClock()
    with pytest.raises(Hold) as e:
        ps.throughput_gate(str(tmp_path), 1000, Throughput(1000, 2), poll_s=10, clock=clock)
    assert "best 500 steps/s" in str(e.value) and "1,000" in str(e.value)
    assert 120 < clock.now <= 130


def test_already_s_shortens_the_window(tmp_path):
    """already_s (time since the GO) counts against within_min, so the HOLD comes sooner."""
    mk_ckpts(str(tmp_path), [(2000, T0), (3000, T0 + 2)])
    full, short = FakeClock(), FakeClock()
    tp = Throughput(1000, 2)
    for clock, already in ((full, 0.0), (short, 100.0)):
        with pytest.raises(Hold):
            ps.throughput_gate(str(tmp_path), 1000, tp, poll_s=10, clock=clock, already_s=already)
    assert short.now < full.now and short.now <= 30


def test_marginal_rates_skip_non_increasing_mtimes():
    """A pair with equal mtimes has no rate (no division by zero)."""
    assert ps.marginal_rates([(1, 5.0), (2, 5.0), (12, 10.0)]) == [2.0]


# -- wait_spawned -------------------------------------------------------------------------------------------

def triggers(tmp_path):
    log = sl.StatusLog(str(tmp_path / "status.txt"), "arm_chain", clock=lambda: "2026-10-26T12:00:00+00:00")
    tag = sl.launch_tag("n1p", 3501)
    run = "rb_x35_n1p_s3501"
    passed = log.arm("pass", tag, f"post-spawn pool-seed check PASS for {run}.*")
    held = log.arm("hold", tag, "HOLD: (.*)")
    wrapper = sl.StatusLog(log.path, tag, clock=lambda: "2026-10-26T12:00:01+00:00")
    return log, passed, held, wrapper, run


def test_pass_line_under_the_right_tag_returns(tmp_path):
    """The wrapper's PASS line for THIS run returns it (even if the wrapper has exited since writing it)."""
    log, passed, held, wrapper, run = triggers(tmp_path)
    wrapper.append(f"post-spawn pool-seed check PASS for {run} (seeds 3)")
    got = ps.wait_spawned(log, passed, held, ProcRef(999999999, 1), 60, clock=FakeClock())
    assert got.endswith(f"PASS for {run} (seeds 3)")


def test_pass_line_under_another_tag_does_not_return(tmp_path):
    """Another arm's PASS line is not this arm's launch having taken."""
    log, passed, held, wrapper, run = triggers(tmp_path)
    other = sl.StatusLog(log.path, sl.launch_tag("d3", 3503), clock=lambda: "t")
    other.append(f"post-spawn pool-seed check PASS for {run}")
    me = ref_of(os.getpid())
    with pytest.raises(Hold, match="no post-spawn pool-seed PASS within 1 min"):
        ps.wait_spawned(log, passed, held, me, 60, poll_s=10, clock=FakeClock())


def test_wrapper_hold_line_raises_hold(tmp_path):
    """A wrapper 'HOLD: …' line is a HOLD for the chain, carrying the reason."""
    log, passed, held, wrapper, run = triggers(tmp_path)
    wrapper.append("HOLD: pool seed mismatch")
    with pytest.raises(Hold, match="HELD: pool seed mismatch"):
        ps.wait_spawned(log, passed, held, ref_of(os.getpid()), 60, clock=FakeClock())


def test_dead_wrapper_without_a_line_raises_hold(tmp_path):
    """The wrapper gone with neither a PASS nor a HOLD line is a HOLD, not an endless wait."""
    log, passed, held, wrapper, run = triggers(tmp_path)
    clock = FakeClock()
    with pytest.raises(Hold, match="is gone without"):
        ps.wait_spawned(log, passed, held, ProcRef(999999999, 1), 3600, clock=clock)
    assert clock.now == 0


# -- the registered-action exception types and the first-trainer-line clock (operator review 10-27) ----------

def test_a_void_line_and_a_missing_banner_raise_void_for_the_registered_action(tmp_path):
    """The chain applies ``on_void`` only to a VOID: both VOID shapes raise :class:`Void` (a Hold)."""
    log = tmp_path / "launcher.log"
    log.write_text("[TEAM-MIX] FATAL: no lever\n")
    with pytest.raises(ps.Void):
        ps.banner_guard([str(log)], BANNER, clock=FakeClock())
    log.write_text("nothing yet\n")
    with pytest.raises(ps.Void, match="no '\\[TEAM-MIX\\] d3_setmut_v1@ALL' banner"):
        ps.banner_guard([str(log)], BANNER, poll_s=60, clock=FakeClock())


def test_a_failed_gate_raises_throughput_fail(tmp_path):
    mk_ckpts(str(tmp_path), [(2000, T0), (3000, T0 + 2)])
    with pytest.raises(ps.ThroughputFail):
        ps.throughput_gate(str(tmp_path), 1000, Throughput(1000, 1), poll_s=10, clock=FakeClock())


def test_the_launchers_child_attached_header_is_not_a_trainer_line(tmp_path):
    p = tmp_path / "launcher_child.log"
    assert ps.first_trainer_line(str(p)) is None
    p.write_text("\n===== child attached 2026-10-27 01:00:00 (pid 42) =====\n\n")
    assert ps.first_trainer_line(str(p)) is None
    p.write_text(p.read_text() + "[rust_rollout] building 256 envs\n")
    assert ps.first_trainer_line(str(p)) == "[rust_rollout] building 256 envs"


def test_the_gate_counts_from_the_first_trainer_line_not_the_go(tmp_path):
    """The 10-27 cold start: the first trainer line came 23 min after the GO and the bar was reached 48 min after
    THAT line (71 min after the GO). Counted from the first line (60 min) it PASSES; counted from the GO it failed."""
    run = str(tmp_path / "run")
    child = os.path.join(run, "launcher_child.log")
    os.makedirs(run)
    open(child, "w").write("===== child attached 2026-10-27 01:00:00 (pid 42) =====\n")
    clock = FakeClock()
    wall = {"go": 1_000_000.0}
    now = lambda: wall["go"] + clock.now  # noqa: E731

    def tick(s):
        clock._sleep(s)
        if clock.now >= 23 * 60 and "trainer" not in open(child).read():
            open(child, "a").write("trainer: starting\n")
        if clock.now >= 71 * 60 and not ps.new_checkpoints(run, 1000):
            mk_ckpts(run, [(2000, now() - 2), (5000, now())])
    clock.sleep = tick
    epoch, what = ps.wait_first_trainer_line(child, run, 1000, deadline_s=90 * 60, now=now, poll_s=30, clock=clock)
    assert "first trainer line" in what and 23 * 60 <= epoch - wall["go"] <= 23 * 60 + 30
    out = ps.throughput_gate(run, 1000, Throughput(1000, 60), poll_s=60, clock=clock, already_s=now() - epoch)
    assert "1,500 steps/s" in out
    # the reverted rule (the deadline from the GO) fails the same healthy cold start
    clock.now, wall["go"] = 23 * 60, wall["go"] + 10_000
    for f in os.listdir(os.path.join(run, "checkpoints")):
        os.unlink(os.path.join(run, "checkpoints", f))
    with pytest.raises(ps.ThroughputFail):
        ps.throughput_gate(run, 1000, Throughput(1000, 60), poll_s=60, clock=clock, already_s=23 * 60)


def test_no_trainer_line_by_its_bound_is_a_throughput_fail(tmp_path):
    run = str(tmp_path / "run")
    os.makedirs(run)
    with pytest.raises(ps.ThroughputFail, match="never started"):
        ps.wait_first_trainer_line(os.path.join(run, "launcher_child.log"), run, 1000, deadline_s=600, poll_s=30,
                                   clock=FakeClock())


def test_a_first_own_checkpoint_starts_the_clock_at_its_mtime(tmp_path):
    mk_ckpts(str(tmp_path), [(1000, T0 - 50), (2000, T0)])
    epoch, what = ps.wait_first_trainer_line(str(tmp_path / "none.log"), str(tmp_path), 1000, deadline_s=1,
                                             clock=FakeClock())
    assert epoch == T0 and "first own checkpoint" in what
