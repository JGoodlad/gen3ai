"""The ship gate's SHAPE: it runs the routine gate's whole set, and only the shape follows the box."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import pytest

from utils import core_split
from utils import heavy_slots as HS
from utils import ship_gate as SG

_ONLINE = list(range(16))
_TRAINER = list(core_split.DEFAULT.trainer_ids)
_AGENTS = list(core_split.DEFAULT.agent_ids)


@pytest.fixture(autouse=True)
def _no_lease(monkeypatch: pytest.MonkeyPatch) -> None:
    """The box's real GPU lease must not decide a unit test's shape."""
    monkeypatch.setattr(HS, "lease_reason", lambda: None)


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


# --- agent work is not a training run (2026-11-09) -------------------------------------------------------------------
_TRAINER_SCRIPT = "train_rl_agent" + ".py"
_SMOKE, _REAL = "/m/smoke_b1", "/m/rb_x"


def _fake_proc(root: Path, table: Dict[int, Tuple[int, List[str], Optional[str]]]) -> str:
    """A fake process table: ``{pid: (ppid, argv, CUDA_VISIBLE_DEVICES or None)}`` as ``/proc`` would show it."""
    for pid, (ppid, argv, cuda) in table.items():
        d = root / str(pid)
        d.mkdir()
        (d / "cmdline").write_bytes("\0".join(argv).encode() + b"\0")
        (d / "stat").write_text(f"{pid} (python3) S {ppid} {pid} {pid} " + " ".join(["0"] * 30) + "\n")
        env = ["LANG=C"] + ([] if cuda is None else [f"CUDA_VISIBLE_DEVICES={cuda}"])
        (d / "environ").write_bytes("\0".join(env).encode() + b"\0")
    return str(root)


def _box(root: Path, *, smoke_cuda: Optional[str] = "0", real: bool = True) -> str:
    """The incident's box: an agent's training smoke (wrapper 100 > launcher 110 > trainer 111 > worker 112) and,
    with ``real``, the pinned GPU run (launcher 200 > trainer 201)."""
    table: Dict[int, Tuple[int, List[str], Optional[str]]] = {
        100: (1, ["python3", "-m", "main.b1.resume_check", "smoke"], smoke_cuda),
        110: (100, ["python3", "-m", "main.launcher", "--debug", "--steps", "10000"], smoke_cuda),
        111: (110, ["python3", f"/wt/src/main/{_TRAINER_SCRIPT}", "--run-dir", _SMOKE], smoke_cuda),
        112: (111, ["python3", "-c", "from multiprocessing.spawn import spawn_main"], smoke_cuda)}
    if real:
        table[200] = (1, ["python3", "-m", "main.launcher", "--steps", "15000000", "--device", "cuda"], "0")
        table[201] = (200, ["python3", f"/repo/src/main/{_TRAINER_SCRIPT}", "--run-dir", _REAL], "0")
    return _fake_proc(root, table)


def _aff(smoke: Sequence[int] = tuple(_AGENTS), **over: Optional[Sequence[int]]):
    table: Dict[int, Optional[List[int]]] = {100: list(_ONLINE), 110: list(smoke), 111: list(smoke), 112: list(smoke),
                                             200: list(_TRAINER), 201: list(_TRAINER)}
    table.update({int(k[1:]): (None if v is None else list(v)) for k, v in over.items()})
    return lambda pid: table[pid]


def test_an_agents_cpu_smoke_on_the_agent_cores_is_not_a_training_run_for_the_shape(tmp_path: Path) -> None:
    """THE INCIDENT (2026-11-09, FINDING 1 of 1504c1639): an agent's CPU-only training smoke on the agent cores was
    read as a second trainer, the union of the two affinities was every CPU, and the gate REFUSED ("pinned to EVERY
    cpu") for the smoke's 25 minutes. The core-split owner calls that run agent work; the gate agrees. Fails on
    revert: the first line raises SystemExit."""
    proc = _box(tmp_path)
    shape = SG.resolve_shape(online=_ONLINE, proc=proc, affinity=_aff())
    assert shape.live_runs == {_REAL: 201} and shape.cpus == _AGENTS and shape.nice == 19
    assert "rb_x" in shape.describe() and "agent work, not a training run: smoke_b1" in shape.describe()
    assert SG.split_runs({_SMOKE: 111, _REAL: 201}, proc=proc, affinity=_aff()) == ({_REAL: 201}, {_SMOKE: 111})


def test_a_smoke_alone_leaves_the_quiet_shape(tmp_path: Path) -> None:
    shape = SG.resolve_shape(online=_ONLINE, proc=_box(tmp_path, real=False), affinity=_aff())
    assert (shape.live_runs, shape.cpus, shape.nice, shape.workers) == ({}, None, 0, SG.QUIET_WORKERS)
    assert "smoke_b1" in shape.describe()


@pytest.mark.parametrize("over,why", [
    ({"p112": _ONLINE}, "a worker of the smoke runs on the trainer's cores"),
    ({"p110": _ONLINE}, "its launcher is unpinned: the next child inherits every CPU"),
    ({"p111": None}, "the trainer's affinity cannot be read"),
    ({"p112": None}, "a LIVE descendant's affinity cannot be read"),
    ({"p111": [], "p110": [], "p112": []}, "an empty affinity proves nothing"),
])
def test_a_run_counts_as_training_unless_its_WHOLE_tree_sits_inside_the_agent_list(
        tmp_path: Path, over: Dict[str, Optional[Sequence[int]]], why: str) -> None:
    proc = _box(tmp_path, real=False)
    training, agent = SG.split_runs({_SMOKE: 111}, proc=proc, affinity=_aff(**over))
    assert (training, agent) == ({_SMOKE: 111}, {}), why
    (tmp_path / "112" / "stat").unlink()                        # the descendant EXITED after it was listed: no evidence
    gone = SG.split_runs({_SMOKE: 111}, proc=proc, affinity=_aff(p112=None))
    assert gone == ({}, {_SMOKE: 111})


def test_the_agent_list_is_the_ACTIVE_splits_read_from_the_owner(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One rule, the owner's (`core_split.is_agent_work`): a smoke on the 4+4 agent cores is agent work only while
    4+4 is the active split. Fails if the gate spells its own list or its own subset test."""
    proc, four = _box(tmp_path, real=False), core_split.parse_split("4+4")
    wide = _aff(smoke=four.agent_ids)
    assert SG.split_runs({_SMOKE: 111}, proc=proc, affinity=wide) == ({_SMOKE: 111}, {})
    monkeypatch.setattr(core_split, "active", lambda path=None: four)
    assert SG.split_runs({_SMOKE: 111}, proc=proc, affinity=wide) == ({}, {_SMOKE: 111})
    seen = []
    monkeypatch.setattr(core_split, "is_agent_work", lambda aff, *s: seen.append(aff) or False)
    assert SG.split_runs({_SMOKE: 111}, proc=proc, affinity=wide) == ({_SMOKE: 111}, {}) and seen


@pytest.mark.parametrize("cuda,agent_work", [("", True), ("-1", True), ("0", False), (None, False)])
def test_a_run_started_with_cuda_hidden_is_agent_work_wherever_it_runs(
        tmp_path: Path, cuda: Optional[str], agent_work: bool) -> None:
    """Rule 5: an agent without a lease runs with CUDA hidden, so such a run is never the GPU's trainer, pinned or
    not. With CUDA visible (or unset) an unpinned run is a training run and the every-CPU refusal stands."""
    proc = _box(tmp_path, smoke_cuda=cuda)
    everywhere = _aff(smoke=_ONLINE)
    training, agent = SG.split_runs({_SMOKE: 111, _REAL: 201}, proc=proc, affinity=everywhere)
    assert (_SMOKE in agent, _SMOKE in training) == (agent_work, not agent_work) and _REAL in training
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
    """The owner's rule (2026-09-30): the gate runs everything. The argv names all of src/ and the
    routine gate's marker expression verbatim, under the gate lock and the memory cap."""
    shape = (SG.resolve_shape({"/m/r": 1}, online=_ONLINE, trainer_cpus=_TRAINER) if live
             else SG.resolve_shape({}, online=_ONLINE))
    argv = SG.build_argv(shape, "py", ["--durations=20"])
    i = argv.index("pytest")
    assert argv[i + 1:i + 4] == ["src/", "-m", "not slow and not e2e"]
    assert SG.MARKERS == "not slow and not e2e"
    assert argv[-1] == "--durations=20"
    lock, cap = argv.index(next(a for a in argv if a.endswith("gate_lock.sh"))), argv.index(str(SG.MEM_CAP_GB))
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
    assert argv[cap + 1:cap + 4] == ["--max-run-s", "2700", str(SG.MEM_CAP_GB)]
    assert argv[cap - 1].endswith("gate_lock.sh")
    assert argv[:cap + 1] + argv[cap + 3:] == plain
    seen: dict = {}
    monkeypatch.setattr(SG, "resolve_shape", lambda: shape)
    monkeypatch.setattr(SG.subprocess, "call", lambda cmd, cwd, env: seen.update(cmd=cmd) or SG.mem_cap.EXIT_RUN_TIMEOUT)
    assert SG.main(["run", "--max-run-s", "2700", "--", "-x"]) == 89
    assert seen["cmd"] == SG.build_argv(shape, SG.os.environ.get("GEN3AI_PYTHON") or SG.sys.executable, ["-x"],
                                        max_run_s=2700.0)
    assert "STOPPED by --max-run-s 2700" in capsys.readouterr().err
    monkeypatch.setattr(SG.subprocess, "call", lambda *a, **k: pytest.fail("a refused or dry invocation ran the gate"))
    assert SG.main(["--dry-run", "--max-run-s", "60"]) == 0 and "--max-run-s 60" in capsys.readouterr().err
    for bad in (["run", "--max-run-s", "0"], ["run", "--max-run-s", "soon"], ["run", "--max-run-s"],
                ["--max-run-s", "60", "run"]):
        assert SG.main(bad) == 2, bad
