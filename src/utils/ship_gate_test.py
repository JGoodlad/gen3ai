"""The ship gate's SHAPE: it runs the routine gate's whole set, and only the shape follows the box."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pytest

from utils import core_split
from utils import fake_proc_testing as FP
from utils import heavy_slots as HS
from utils import ship_gate as SG

_ONLINE, _TRAINER, _AGENTS = FP.ONLINE, FP.TRAINER, FP.AGENTS


_REAL_PRECHECK = SG.static_precheck


@pytest.fixture(autouse=True)
def _no_lease(monkeypatch: pytest.MonkeyPatch) -> None:
    """The box's real GPU lease must not decide a unit test's shape."""
    monkeypatch.setattr(HS, "lease_reason", lambda: None)


@pytest.fixture(autouse=True)
def _static_precheck_green(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests are about the gate's own argv and exit; the static pre-check has its own tests below (``_REAL_PRECHECK``)."""
    monkeypatch.setattr(SG, "static_precheck", lambda shape, python, env: (0, 0.0))


def test_quiet_box_runs_the_quiet_policy_unpinned_at_normal_priority() -> None:
    shape = SG.resolve_shape({}, online=_ONLINE)
    assert (shape.cpus, shape.nice, shape.workers) == (None, 0, SG.QUIET_WORKERS)
    argv = SG.build_argv(shape, "py")
    assert argv[0].endswith("gate_lock.sh") and "nice" not in argv and "taskset" not in argv


def test_a_live_run_takes_rule_5b_on_every_cpu_the_trainer_does_not_hold() -> None:
    shape = SG.resolve_shape({"/m/rb_x": 42}, online=_ONLINE, trainer_cpus=_TRAINER)
    assert shape.cpus == _AGENTS and shape.nice == 19
    assert shape.workers == int(6 * SG.LIVE_WORKERS_PER_CPU)
    argv = SG.build_argv(shape, "py")
    assert argv[:5] == ["nice", "-n", "19", "taskset", "-c"] and argv[5] == core_split.DEFAULT.agent_list
    assert "rb_x" in shape.describe()


def test_a_trainer_narrowed_below_the_split_frees_nothing_for_the_gate() -> None:
    """The core-split A/B narrows the live tree for 8 s at a time (``main.ops.core_split_ab``). A gate that starts
    inside a narrow arm must not take the cores the driver hands back: the gate reads ``core_split.agent_ids_beside``
    (the trainer's affinity UNION the active trainer list). Fails on a revert to the bare complement."""
    narrow = core_split.narrowed(_TRAINER, 2)
    shape = SG.resolve_shape({"/m/rb_x": 42}, online=_ONLINE, trainer_cpus=narrow)
    assert shape.cpus == _AGENTS and not set(shape.cpus) & set(_TRAINER)
    wide = SG.resolve_shape({"/m/rb_x": 42}, online=_ONLINE, trainer_cpus=list(range(14)))
    assert wide.cpus == [14, 15]                      # a trainer pinned WIDER than the split is still respected


# --- agent work is not a training run (2026-11-09; the rule itself is core_split's: live_runs_test.py) ---------------
def test_an_agents_cpu_smoke_on_the_agent_cores_is_not_a_training_run_for_the_shape(tmp_path: Path) -> None:
    """THE INCIDENT (2026-11-09, FINDING 1 of 1504c1639): an agent's CPU-only training smoke on the agent cores was
    read as a second trainer, the union of the two affinities was every CPU, and the gate REFUSED ("pinned to EVERY
    cpu") for the smoke's 25 minutes. The core-split owner calls that run agent work; the gate agrees. Fails on
    revert: the first line raises SystemExit."""
    proc = FP.incident_box(tmp_path)
    shape = SG.resolve_shape(online=_ONLINE, proc=proc, affinity=FP.affinities())
    assert shape.live_runs == {FP.REAL: FP.REAL_PID} and shape.cpus == _AGENTS and shape.nice == 19
    assert "rb_x" in shape.describe() and "agent work, not a training run: smoke_b1" in shape.describe()


def test_a_smoke_alone_leaves_the_quiet_shape(tmp_path: Path) -> None:
    shape = SG.resolve_shape(online=_ONLINE, proc=FP.incident_box(tmp_path, real=False), affinity=FP.affinities())
    assert (shape.live_runs, shape.cpus, shape.nice, shape.workers) == ({}, None, 0, SG.QUIET_WORKERS)
    assert "smoke_b1" in shape.describe()


@pytest.mark.parametrize("cuda,agent_work", [("", True), ("-1", True), ("0", False), (None, False)])
def test_a_run_started_with_cuda_hidden_shapes_nothing_wherever_it_runs(
        tmp_path: Path, cuda: Optional[str], agent_work: bool) -> None:
    """Rule 5: an agent without a lease runs with CUDA hidden, so such a run is never the GPU's trainer, pinned or
    not. With CUDA visible (or unset) an unpinned run is a training run and the every-CPU refusal stands."""
    proc = FP.incident_box(tmp_path, smoke_cuda=cuda)
    everywhere = FP.affinities(smoke=_ONLINE)
    if agent_work:
        assert SG.resolve_shape(online=_ONLINE, proc=proc, affinity=everywhere).cpus == _AGENTS
    else:
        with pytest.raises(SystemExit, match="no agent cores"):
            SG.resolve_shape(online=_ONLINE, proc=proc, affinity=everywhere)


def test_a_trainer_pinned_to_every_cpu_is_refused_not_overlapped() -> None:
    with pytest.raises(SystemExit, match="no agent cores"):
        SG.resolve_shape({"/m/rb_x": 42}, online=_ONLINE, trainer_cpus=_ONLINE)


@pytest.mark.parametrize("live", [False, True])
def test_it_runs_the_WHOLE_routine_set_never_a_selection(live: bool) -> None:
    """The owner's rule (2026-09-30): the gate runs everything. The argv names every collected root
    (pytest.ini's testpaths) and the routine gate's marker expression verbatim, under the gate lock and the memory cap."""
    shape = (SG.resolve_shape({"/m/r": 1}, online=_ONLINE, trainer_cpus=_TRAINER) if live
             else SG.resolve_shape({}, online=_ONLINE))
    argv = SG.build_argv(shape, "py", ["--durations=20"])
    i = argv.index("pytest")
    roots = SG.collected_roots()
    assert "src/" in roots and "tools/" in roots
    assert argv[i + 1:i + 3 + len(roots)] == [*roots, "-m", "not slow and not e2e"]
    assert SG.MARKERS == "not slow and not e2e"
    assert argv[-1] == "--durations=20"
    lock, cap = argv.index(next(a for a in argv if a.endswith("gate_lock.sh"))), argv.index(str(SG.mem_cap_gb(shape.workers)))
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


def test_the_gate_runs_with_the_8s_bar_on(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = {}
    monkeypatch.setattr(SG, "resolve_shape", lambda: SG.Shape({}, None, 0, 6))
    monkeypatch.setattr(SG.subprocess, "call", lambda cmd, cwd, env: seen.update(env=env) or 0)
    monkeypatch.delenv("GEN3AI_SHIP_BAR", raising=False)
    assert SG.main(["run"]) == 0
    assert seen["env"]["GEN3AI_SHIP_BAR"] == "1" and seen["env"]["GEN3AI_SHIP_BAR_REPORT"] == SG.BAR_REPORT


def test_the_session_warms_the_rust_binaries_before_its_first_test(monkeypatch: pytest.MonkeyPatch) -> None:
    """A cargo build paid inside a test body is that test's cost to the 8 s bar (2026-10-19: a 2 s test read
    22 s). The gate switches the root conftest's warm-up on (`$GEN3AI_SHIP_WARM`); it runs inside the gate's
    own lock / slot / niceness, so the gate argv is unchanged."""
    seen = {}
    monkeypatch.setattr(SG, "resolve_shape", lambda: SG.Shape({}, None, 0, 6))
    monkeypatch.setattr(SG.subprocess, "call", lambda cmd, cwd, env: seen.update(env=env) or 0)
    monkeypatch.delenv("GEN3AI_SHIP_WARM", raising=False)
    assert SG.main(["run"]) == 0
    assert seen["env"]["GEN3AI_SHIP_WARM"] == "1"


def test_the_warm_up_names_a_failed_build_and_counts_it() -> None:
    from utils import ship_gate_warm as W

    def boom():
        raise RuntimeError("no cargo")
    out = []
    assert W.warm([("ok", lambda: None), ("bad", boom)], emit=out.append) == 1
    assert "ok: ready" in out[0] and "bad: NOT BUILT" in out[1] and "no cargo" in out[1]
def test_a_held_gpu_lease_with_no_run_live_takes_the_rule_5b_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tonight an unpinned -n 6 gate nearly ran across a GPU battery's learner-timing cells (no run live, lease
    held). The shape follows `heavy_slots.wanted` - one predicate. Fails on a revert of the shared predicate."""
    monkeypatch.setattr(HS, "lease_reason", lambda: "the GPU is LEASED (lease owner 'battery')")
    shape = SG.resolve_shape({}, online=_ONLINE)
    assert shape.nice == 19 and shape.cpus == _AGENTS
    assert shape.workers == int(6 * SG.LIVE_WORKERS_PER_CPU)
    argv = SG.build_argv(shape, "py")
    assert argv[:5] == ["nice", "-n", "19", "taskset", "-c"] and argv[5] == core_split.DEFAULT.agent_list
    assert "LEASED" in shape.describe() and "battery" in shape.describe()
    # a live run still wins and keeps the complement-of-the-trainer rule
    live = SG.resolve_shape({"/m/rb_x": 42}, online=_ONLINE, trainer_cpus=_TRAINER)
    assert live.leased == "" and live.cpus == _AGENTS
    # the shape and the slots read ONE predicate: no lease and no run is quiet on both
    monkeypatch.setattr(HS, "lease_reason", lambda: None)
    assert HS.wanted(runs={}) is None and SG.resolve_shape({}, online=_ONLINE).cpus is None


def test_max_run_s_goes_to_mem_cap_the_innermost_wrapper_so_no_queue_wait_counts(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """2026-11-09: `timeout N ship_gate.sh run` fired while the gate still queued and lost its place. The gate's own bound is
    mem_cap's `--max-run-s` (clock starts with pytest), placed INSIDE gate_lock.sh; with no flag the argv is unchanged."""
    shape = SG.Shape({}, None, 0, 6)
    plain = SG.build_argv(shape, "py")
    assert "--max-run-s" not in plain
    argv = SG.build_argv(shape, "py", max_run_s=2700.0)
    cap = next(i for i, a in enumerate(argv) if a.endswith("mem_cap.sh"))
    assert argv[cap + 1:cap + 4] == ["--max-run-s", "2700", str(SG.mem_cap_gb(6))]
    assert argv[cap - 1].endswith("gate_lock.sh")
    assert argv[:cap + 1] + argv[cap + 3:] == plain
    seen: dict = {}
    monkeypatch.setattr(SG, "resolve_shape", lambda: shape)
    monkeypatch.setattr(SG.subprocess, "call", lambda cmd, cwd, env: seen.update(cmd=cmd) or SG.mem_cap.EXIT_RUN_TIMEOUT)
    assert SG.main(["run", "--max-run-s", "2700", "--", "-x"]) == 89
    assert seen["cmd"] == SG.build_argv(shape, SG.os.environ.get("GEN3AI_PYTHON") or SG.sys.executable, ["-x"],
                                        max_run_s=2700.0, ignore=SG.static_gate_files())
    assert "STOPPED by --max-run-s 2700" in capsys.readouterr().err
    monkeypatch.setattr(SG.subprocess, "call", lambda *a, **k: pytest.fail("a refused or dry invocation ran the gate"))
    assert SG.main(["--dry-run", "--max-run-s", "60"]) == 0 and "--max-run-s 60" in capsys.readouterr().err
    for bad in (["run", "--max-run-s", "0"], ["run", "--max-run-s", "soon"], ["run", "--max-run-s"],
                ["--max-run-s", "60", "run"]):
        assert SG.main(bad) == 2, bad


# --- the pre-queue bootstrap preflight (recurring 2026-11-16: a never-bootstrapped worktree queued 20-40 min, then failed) ---

def _checkout(root: Path, *, bootstrapped: bool, worktree: bool) -> Path:
    """A fake repo root: a linked worktree's ``.git`` is a file, the main checkout's a directory."""
    if worktree:
        (root / ".git").write_text("gitdir: /elsewhere/.git/worktrees/x\n")
    else:
        (root / ".git").mkdir()
    if bootstrapped:
        from utils import checkout_deps
        (root / checkout_deps.REQUIRED_DEPS[0]).parent.mkdir(parents=True)
        (root / checkout_deps.REQUIRED_DEPS[0]).write_text("//\n")
        (root / checkout_deps.REQUIRED_DEPS[1]).mkdir(parents=True)
    return root


def _spy_nothing_queues(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(SG, "resolve_shape", lambda: pytest.fail("the shape was resolved before the preflight"))
    monkeypatch.setattr(SG.subprocess, "call", lambda *a, **k: pytest.fail("the gate lock / heavy slot was reached"))


def test_an_unbootstrapped_worktree_is_refused_in_seconds_before_any_queue(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    import time
    monkeypatch.delenv("GEN3AI_SKIP_DEPS_GUARD", raising=False)
    monkeypatch.setattr(SG, "repo_path", lambda *a: _checkout(tmp_path, bootstrapped=False, worktree=True).joinpath(*a))
    _spy_nothing_queues(monkeypatch)
    t0 = time.monotonic()
    assert SG.main(["run"]) == SG.EXIT_UNBOOTSTRAPPED == 4
    assert time.monotonic() - t0 < 2.0
    err = capsys.readouterr().err
    assert "REFUSING before queueing" in err
    assert "scripts/bootstrap.sh --skip-env --no-rust --no-check" in err          # the exact worktree command
    assert "dist/sim/index.js" in err and "node_modules" in err                    # both probes, by name


def test_an_unbootstrapped_main_checkout_is_told_the_plain_bootstrap(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    monkeypatch.delenv("GEN3AI_SKIP_DEPS_GUARD", raising=False)
    monkeypatch.setattr(SG, "repo_path", lambda *a: _checkout(tmp_path, bootstrapped=False, worktree=False).joinpath(*a))
    _spy_nothing_queues(monkeypatch)
    assert SG.main(["run"]) == 4
    err = capsys.readouterr().err
    assert "./scripts/bootstrap.sh" in err and "--skip-env" not in err


def test_a_bootstrapped_checkout_passes_the_preflight_and_reaches_the_gate(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEN3AI_SKIP_DEPS_GUARD", raising=False)
    root = _checkout(tmp_path, bootstrapped=True, worktree=True)
    monkeypatch.setattr(SG, "repo_path", lambda *a: root.joinpath(*a))
    assert SG.preflight() == ""
    seen = {}
    monkeypatch.setattr(SG, "resolve_shape", lambda: SG.Shape({}, None, 0, 6))
    monkeypatch.setattr(SG.subprocess, "call", lambda cmd, cwd, env: seen.update(cmd=cmd) or 0)
    assert SG.main(["run"]) == 0 and seen["cmd"]


def test_the_preflight_honours_the_deps_guard_opt_out_and_a_dry_run_never_checks(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(SG, "repo_path", lambda *a: _checkout(tmp_path, bootstrapped=False, worktree=True).joinpath(*a))
    monkeypatch.delenv("GEN3AI_SKIP_DEPS_GUARD", raising=False)
    monkeypatch.setattr(SG, "resolve_shape", lambda: SG.Shape({}, None, 0, 6))
    assert SG.main(["--dry-run"]) == 0                      # prints the plan; checks nothing
    assert SG.preflight() != ""
    monkeypatch.setenv("GEN3AI_SKIP_DEPS_GUARD", "1")
    assert SG.preflight() == ""


# --- the gate's memory cap is MEASURED and scales with -n (2026-11-17) ---------------------------------------------------
#
# The gate declared 24 GB and peaked at 6.6-10.2: the heavy slots admit by the SUM OF DECLARED CAPS, so 14 GB of promise nobody
# used kept the next job waiting. The numbers below are the measured peaks of ``ship_gate``'s module docs.

#: the largest peak of 64 gates at -n 3 (2026-11-09..17), read from events.jsonl's ``capped`` rows
MEASURED_MAX_AT_3_GB = 10.18


def test_the_gates_cap_is_the_measured_peak_with_a_margin_not_a_habit() -> None:
    """Fails on a revert to the fixed 24 (`16` is the cap at the live shape) and on a cap that no longer covers the measured
    peak with the stated margin (a gate that outgrows its cap is OOM-killed, so the margin is what keeps it green)."""
    assert SG.mem_cap_gb(3) == 16
    assert SG.mem_cap_gb(3) >= SG.CAP_MARGIN * MEASURED_MAX_AT_3_GB
    assert SG.mem_cap_gb(6) == 21 and SG.mem_cap_gb(6) >= SG.CAP_MARGIN * 13.57     # the quiet shape's largest no-slot job
    assert all(SG.mem_cap_gb(n) < 24 for n in (1, 2, 3, 6)), "the 24 GB habit is back"
    caps = [SG.mem_cap_gb(n) for n in range(1, 13)]
    assert caps == sorted(caps) and len(set(caps[:6])) > 1, "the cap must grow with the workers that hold the memory"
    assert SG.mem_cap_gb(0) == SG.mem_cap_gb(1) >= 1                              # a nonsense count never gives a zero cap


def test_the_argv_declares_the_cap_of_the_workers_it_really_runs() -> None:
    """``-- -n 6`` after the shape's ``-n 3`` is the LAST ``-n``, the one pytest takes: the cap follows it. Fails if the cap
    is read from the shape alone (a 6-worker gate under the 3-worker cap is OOM-killed at the margin the 3-worker peak bought)."""
    from utils import mem_cap as M
    live = SG.Shape({"/m/rb_x": 1}, [5, 6, 7], 19, 3)

    def declared(extra: list) -> Optional[float]:
        argv = SG.build_argv(live, "py", extra)
        return M.declared_cap_gb(argv[next(i for i, a in enumerate(argv) if a.endswith("gate_lock.sh")) + 1:])

    assert declared([]) == 16.0
    assert declared(["-n", "6"]) == 21.0 and declared(["-n6"]) == 21.0 and declared(["--numprocesses=6"]) == 21.0
    assert declared(["-n", "2", "-n", "6"]) == 21.0, "pytest takes the last -n"
    assert declared(["--durations=20"]) == 16.0
    assert SG.effective_workers(3, ["-n", "auto"]) == len(os.sched_getaffinity(0))


# --- the static pre-check (2026-11-17: 6 of 16 gates on the night of 11-16/17 failed after a ~15-min exclusive turn) ---------

class _Box:
    """A fake box: every ``subprocess.call`` is recorded, the pre-check's (no gate lock in its argv) returns ``static_rc``, the
    gate's returns 0; a gate lock directory in ``tmp_path`` is the fake lock / slot state, which only a gate takes."""

    def __init__(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, static_rc: int) -> None:
        from utils import gate_lock as GL
        self.calls: list = []
        self.lock_dir = tmp_path / "gate_slots"
        monkeypatch.setenv(GL.DIR_ENV, str(self.lock_dir))
        monkeypatch.setattr(SG, "static_precheck", _REAL_PRECHECK)
        monkeypatch.setattr(SG, "resolve_shape", lambda: SG.Shape({}, None, 0, 6))
        monkeypatch.setattr(SG.subprocess, "call", self.call)
        self.static_rc = static_rc

    def call(self, cmd, cwd, env):
        self.calls.append((list(cmd), dict(env)))
        if any(a.endswith("gate_lock.sh") for a in cmd):
            self.lock_dir.mkdir(parents=True, exist_ok=True)          # what taking the gate lock leaves behind
            (self.lock_dir / "slot0.lock").write_text("")
            return 0
        return self.static_rc

    def gate_calls(self) -> list:
        return [c for c in self.calls if any(a.endswith("gate_lock.sh") for a in c[0])]


def test_a_failing_static_tier_exits_before_the_gate_ever_queues(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    """THE INCIDENT (night of 11-16/17): a static gate red cost a ~15-min exclusive gate-lock turn. Now it costs the pre-check:
    the exit is the pre-check's, and neither the gate lock nor a heavy slot is ever asked for (fake lock state stays empty).
    Fails on a revert: the gate argv would be called and the lock dir written."""
    box = _Box(tmp_path, monkeypatch, static_rc=1)
    assert SG.main(["run"]) == 1
    assert len(box.calls) == 1 and not box.gate_calls() and not box.lock_dir.exists()
    assert "REFUSING before queueing: a static gate FAILED" in capsys.readouterr().err


def test_a_static_precheck_that_overran_is_incomplete_not_a_red_and_still_never_queues(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    box = _Box(tmp_path, monkeypatch, static_rc=SG.mem_cap.EXIT_SHORT_OVERRUN)
    assert SG.main(["run"]) == 88 and not box.gate_calls()
    assert "OVERRAN its short-lane bound" in capsys.readouterr().err


def test_a_passing_static_tier_proceeds_to_the_gate_which_skips_exactly_what_it_ran(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Green pre-check: the gate runs as before (lock, cap, marker expression), with the pre-check's files ``--ignore``d (run once)
    and its seconds in the environment for the gate's event line. Fails on a revert of either the order or the dedupe."""
    from utils import gate_lock as GL
    from utils import static_gates
    box = _Box(tmp_path, monkeypatch, static_rc=0)
    assert SG.main(["run"]) == 0
    (pre, _), (gate, genv) = box.calls
    assert not any(a.endswith("gate_lock.sh") for a in pre) and any(a.endswith("mem_cap.sh") for a in pre)
    files = static_gates.static_gate_files()
    assert files and [a for a in pre if a.endswith("_test.py")] == files        # the static tier's ONE inventory
    assert pre[pre.index("--max-s") + 1] == f"{SG.STATIC_MAX_S:g}" and pre[pre.index("-n") + 1] == "2"
    marker = lambda argv: argv[len(argv) - 1 - argv[::-1].index("-m") + 1]   # the LAST "-m" (python -m pytest has one before)
    assert marker(pre[:pre.index("-q")]) == marker(gate[:gate.index("-q")]) == SG.MARKERS  # the same -m: skipped == ran
    assert [a[len("--ignore="):] for a in gate if a.startswith("--ignore=")] == files
    assert float(genv[GL.PRECHECK_ENV]) >= 0.0


def test_the_precheck_follows_rule_5b_beside_a_live_run_and_a_dry_run_runs_nothing(
        monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture) -> None:
    live = SG.resolve_shape({"/m/rb_x": 42}, online=_ONLINE, trainer_cpus=_TRAINER)
    argv = SG.static_argv(live, "py", ["src/a_test.py"])
    assert argv[:5] == ["nice", "-n", "19", "taskset", "-c"] and argv[5] == core_split.DEFAULT.agent_list
    assert SG.static_argv(SG.Shape({}, None, 0, 6), "py", ["src/a_test.py"])[0].endswith("mem_cap.sh")
    monkeypatch.setattr(SG, "resolve_shape", lambda: SG.Shape({}, None, 0, 6))
    monkeypatch.setattr(SG, "static_precheck", lambda *a: pytest.fail("a dry run ran the pre-check"))
    assert SG.main(["--dry-run"]) == 0 and "static pre-check, before any queue" in capsys.readouterr().err
