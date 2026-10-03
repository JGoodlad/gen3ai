"""K3 — the HERMETIC per-run compile cache (`agents.model.compile_cache`,
gen3_hermetic_compile_cache_v1). Every test here fails on a revert of the rule it names."""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from agents.model import compile_cache as CC
from utils.paths import src_path

_KEYS = (CC.ENV_ROOT, CC.ENV_INDUCTOR, CC.ENV_TRITON)


@pytest.fixture
def isolated_env():
    """Snapshot + restore the three variables (the session's own declared cache must survive)."""
    saved = {k: os.environ.get(k) for k in _KEYS}
    saved_private = dict(CC._PRIVATE)
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    CC._PRIVATE.clear()
    CC._PRIVATE.update(saved_private)


def _stamp(**over):
    s = {"schema": CC.SCHEMA, "code_sha": "a" * 40, "code_dirty": False, "torch": "2.5.1+cu121",
         "python": "/env/bin/python3", "config_row_sha": "c" * 64}
    s.update(over)
    return s


@pytest.fixture
def clean_stamp(monkeypatch):
    """Pin the stamp to a CLEAN identity (the worktree running the test may be dirty)."""
    cur = _stamp()
    monkeypatch.setattr(CC, "cache_stamp", lambda: dict(cur))
    return cur


def _poison(root: Path) -> Path:
    f = root / "inductor" / "fxgraph" / "stale_artifact"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("compiled by someone else")
    return f


# --------------------------------------------------------------------------- the run's cache
def test_fresh_launch_creates_an_EMPTY_cache_inside_the_run_and_exports_it(tmp_path, isolated_env,
                                                                         clean_stamp):
    d = CC.prepare_run_cache(str(tmp_path / "run"), same_run_resume=False, emit=lambda _m: None)
    root = tmp_path / "run" / CC.CACHE_SUBDIR
    assert d.action == "created" and Path(d.root) == root
    assert os.environ[CC.ENV_INDUCTOR] == str(root / "inductor")
    assert os.environ[CC.ENV_TRITON] == str(root / "triton")
    assert os.environ[CC.ENV_ROOT] == str(root)
    assert sorted(p.name for p in root.iterdir()) == ["inductor", "stamp.json", "triton"]
    assert not any((root / "inductor").iterdir()) and not any((root / "triton").iterdir())
    assert json.loads((root / "stamp.json").read_text())["stamp"] == clean_stamp


def test_a_fresh_launch_or_fork_WIPES_a_cache_already_in_the_dir(tmp_path, isolated_env, clean_stamp):
    run = tmp_path / "run"
    CC.prepare_run_cache(str(run), same_run_resume=False, emit=lambda _m: None)
    stale = _poison(run / CC.CACHE_SUBDIR)
    d = CC.prepare_run_cache(str(run), same_run_resume=False, emit=lambda _m: None)
    assert d.action == "wiped" and d.bytes_before > 0
    assert not stale.exists(), "a fresh launch must never inherit a compile artifact"


def test_a_same_run_restart_with_a_matching_stamp_REUSES(tmp_path, isolated_env, clean_stamp):
    run = tmp_path / "run"
    CC.prepare_run_cache(str(run), same_run_resume=False, emit=lambda _m: None)
    kept = _poison(run / CC.CACHE_SUBDIR)
    d = CC.prepare_run_cache(str(run), same_run_resume=True, emit=lambda _m: None)
    assert d.action == "reused", d.reason
    assert kept.exists(), "the run's own restart must keep its warm cache"
    rec = json.loads((run / CC.CACHE_SUBDIR / "stamp.json").read_text())
    assert rec["reuses"] == 1


@pytest.mark.parametrize("field,value", [
    ("code_sha", "b" * 40), ("torch", "2.8.0+cu126"), ("config_row_sha", "d" * 64),
    ("python", "/other_env/bin/python3"), ("code_dirty", True), ("schema", "old_schema"),
])
def test_a_stamp_mismatch_WIPES_on_a_same_run_restart(tmp_path, isolated_env, monkeypatch, field, value):
    run = tmp_path / "run"
    monkeypatch.setattr(CC, "cache_stamp", lambda: _stamp())
    CC.prepare_run_cache(str(run), same_run_resume=False, emit=lambda _m: None)
    stale = _poison(run / CC.CACHE_SUBDIR)
    monkeypatch.setattr(CC, "cache_stamp", lambda: _stamp(**{field: value}))
    d = CC.prepare_run_cache(str(run), same_run_resume=True, emit=lambda _m: None)
    assert d.action == "wiped" and field in d.reason, d.reason
    assert not stale.exists()


def test_a_DIRTY_or_unnamed_tree_never_reuses_even_with_an_identical_stamp(tmp_path, isolated_env,
                                                                          monkeypatch):
    for ident in ({"code_dirty": True}, {"code_dirty": None}, {"code_sha": "unknown"}):
        run = tmp_path / f"run_{len(ident)}_{list(ident.values())[0]}"
        monkeypatch.setattr(CC, "cache_stamp", lambda ident=ident: _stamp(**ident))
        CC.prepare_run_cache(str(run), same_run_resume=False, emit=lambda _m: None)
        stale = _poison(run / CC.CACHE_SUBDIR)
        d = CC.prepare_run_cache(str(run), same_run_resume=True, emit=lambda _m: None)
        assert d.action == "wiped", (ident, d.reason)
        assert not stale.exists()


def test_a_missing_or_corrupt_stamp_WIPES(tmp_path, isolated_env, clean_stamp):
    run = tmp_path / "run"
    CC.prepare_run_cache(str(run), same_run_resume=False, emit=lambda _m: None)
    (run / CC.CACHE_SUBDIR / "stamp.json").write_text("{not json")
    stale = _poison(run / CC.CACHE_SUBDIR)
    d = CC.prepare_run_cache(str(run), same_run_resume=True, emit=lambda _m: None)
    assert d.action == "wiped" and "no readable stamp" in d.reason
    assert not stale.exists()


def test_the_stamp_names_code_torch_and_the_compile_config_row():
    from agents.model.compile_control import config_row_hash
    s = CC.cache_stamp()
    assert {"code_sha", "code_dirty", "torch", "config_row_sha"} <= set(s)
    assert s["config_row_sha"] == config_row_hash()
    import torch
    assert s["torch"] == torch.__version__
    # the row hash MOVES with the row (the K1b lesson: a cache key can omit a setting)
    from agents.model import compile_control as C
    ver = torch.__version__
    before = config_row_hash(ver)
    row = C._COMPILE_CONFIG.get(ver)
    if row is None:
        pytest.fail(f"no compile-config row for the installed torch {ver}")
    C._COMPILE_CONFIG[ver] = dict(row, **{"torch._functorch.config.donated_buffer": True})
    try:
        assert config_row_hash(ver) != before
    finally:
        C._COMPILE_CONFIG[ver] = row


def test_a_compile_that_already_ran_against_the_SHARED_cache_is_refused(tmp_path, isolated_env):
    os.environ[CC.ENV_INDUCTOR] = CC.torch_default_cache_dir()
    with pytest.raises(CC.CompileCacheError, match="before the run declared its own"):
        CC.prepare_run_cache(str(tmp_path / "run"), same_run_resume=False, emit=lambda _m: None)


# --------------------------------------------------------------------------- everyone else
def test_an_undeclared_process_gets_a_private_cache_deleted_at_exit(tmp_path):
    """A real child process with none of the three variables: it compiles into a fresh private dir,
    never torch's shared default, and the dir is gone after it exits."""
    env = {k: v for k, v in os.environ.items() if k not in _KEYS}
    env["PYTHONPATH"] = str(src_path()) + os.pathsep + env.get("PYTHONPATH", "")
    code = textwrap.dedent("""
        import os, json
        from agents.model.compile_cache import ensure_hermetic_cache, _is_shared
        root = ensure_hermetic_cache("test")
        print(json.dumps({"root": root, "ind": os.environ["TORCHINDUCTOR_CACHE_DIR"],
                          "tri": os.environ["TRITON_CACHE_DIR"],
                          "shared": _is_shared(os.environ["TORCHINDUCTOR_CACHE_DIR"]),
                          "exists": os.path.isdir(root)}))
    """)
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    got = json.loads(r.stdout.strip().splitlines()[-1])
    assert got["exists"] and not got["shared"]
    assert got["ind"].startswith(got["root"]) and got["tri"].startswith(got["root"])
    assert os.path.basename(got["root"]).startswith("gen3ai_compile_")
    assert os.path.dirname(got["root"]) == CC.scratch_root(), "a temp cache must live on the real disk"
    assert not os.path.exists(got["root"]), "the private cache must be deleted when its creator exits"


def test_temp_caches_live_on_the_real_disk_never_tmpfs(tmp_path, monkeypatch):
    """`/tmp` here is tmpfs (RAM + a 1M-inode cap, exhausted 2026-09-30): the scratch root is
    `$GEN3AI_SCRATCH` or `~/.cache/gen3ai/tmp`, and a tmpfs root is REFUSED."""
    monkeypatch.delenv(CC.ENV_SCRATCH, raising=False)
    default = CC.scratch_root()
    assert default == os.path.join(os.path.expanduser("~"), ".cache", "gen3ai", "tmp")
    assert CC.fs_type(default) not in ("tmpfs", "ramfs"), CC.fs_type(default)
    monkeypatch.setenv(CC.ENV_SCRATCH, str(tmp_path / "scratch"))
    if CC.fs_type(str(tmp_path)) in ("tmpfs", "ramfs"):
        with pytest.raises(CC.CompileCacheError, match="tmpfs|ramfs"):
            CC.scratch_root()
    else:
        assert CC.scratch_root() == str(tmp_path / "scratch")
    shm = "/dev/shm"
    if os.path.isdir(shm) and CC.fs_type(shm) == "tmpfs":
        monkeypatch.setenv(CC.ENV_SCRATCH, os.path.join(shm, f"k3_probe_{os.getpid()}"))
        try:
            with pytest.raises(CC.CompileCacheError, match="tmpfs"):
                CC.scratch_root()
        finally:
            import shutil
            shutil.rmtree(os.path.join(shm, f"k3_probe_{os.getpid()}"), ignore_errors=True)


@pytest.mark.parametrize("fail", [False, True], ids=["passing", "FAILING"])
def test_a_pytest_session_removes_its_cache_even_when_it_fails(tmp_path, fail):
    """A child pytest session (the real root conftest) declares a fresh cache and removes it at
    teardown — also when its test FAILS. Reverting the conftest teardown leaves the dir behind."""
    from utils.paths import repo_root
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src_path()) + os.pathsep + env.get("PYTHONPATH", "")
    if fail:
        env["GEN3AI_K3_PROBE_FAIL"] = "1"
    node = "src/agents/model/compile_cache_test.py::test_THIS_pytest_process_compiles_into_a_fresh_private_dir"
    r = subprocess.run([sys.executable, "-m", "pytest", node, "-q", "-s", "-p", "no:cacheprovider",
                        "-p", "no:xdist"], cwd=str(repo_root()), env=env, capture_output=True, text=True,
                       timeout=300)
    assert (r.returncode != 0) == fail, r.stdout[-2000:]
    roots = [ln.split("=", 1)[1].strip() for ln in r.stdout.splitlines() if ln.startswith("K3ROOT=")]
    assert roots and os.path.basename(roots[0]).startswith("gen3ai_pytest_compile_"), r.stdout[-2000:]
    assert not os.path.exists(roots[0]), f"the session's compile cache SURVIVED its session: {roots[0]}"


def test_a_SIGKILLED_owner_s_cache_is_swept_by_the_next_declaration(tmp_path):
    """SIGKILL skips atexit: the dir survives its owner — until the next process declares a cache,
    which sweeps every K3 temp cache whose creator PID is dead (and never a live one)."""
    env = {k: v for k, v in os.environ.items() if k not in _KEYS}
    env["PYTHONPATH"] = str(src_path()) + os.pathsep + env.get("PYTHONPATH", "")
    code = ("import os, sys, time\n"
            "from agents.model.compile_cache import ensure_hermetic_cache\n"
            "print(ensure_hermetic_cache('victim'), flush=True)\n"
            "time.sleep(120)\n")
    p = subprocess.Popen([sys.executable, "-c", code], env=env, stdout=subprocess.PIPE, text=True)
    try:
        root = p.stdout.readline().strip()
        assert os.path.isdir(root)
        assert root not in CC.sweep_stale_temp_caches(), "a LIVE owner's cache must never be swept"
        assert os.path.isdir(root)
        p.kill()
        p.wait(timeout=30)
        assert os.path.isdir(root), "precondition: SIGKILL skipped the owner's own cleanup"
        assert root in CC.sweep_stale_temp_caches()
        assert not os.path.exists(root)
    finally:
        if p.poll() is None:
            p.kill()


@pytest.mark.parametrize("shared", ["torch_default", "/tmp/gen3ai_inductor_cache"])
def test_a_SHARED_value_is_never_honoured(isolated_env, shared):
    os.environ.pop(CC.ENV_ROOT, None)
    os.environ[CC.ENV_INDUCTOR] = CC.torch_default_cache_dir() if shared == "torch_default" else shared
    try:
        root = CC.ensure_hermetic_cache("test")
        assert not CC._is_shared(os.environ[CC.ENV_INDUCTOR])
        assert os.environ[CC.ENV_INDUCTOR].startswith(root)
    finally:
        CC._cleanup_private()


def test_a_declared_cache_is_used_as_is(tmp_path, isolated_env):
    os.environ[CC.ENV_ROOT] = str(tmp_path)
    os.environ[CC.ENV_INDUCTOR] = str(tmp_path / "inductor")
    assert CC.ensure_hermetic_cache("test") == str(tmp_path)
    assert os.environ[CC.ENV_INDUCTOR] == str(tmp_path / "inductor")


def test_THIS_pytest_process_compiles_into_a_fresh_private_dir():
    """conftest.py declares one per test process; the names it inlines are this module's. (Also the
    probe `test_a_pytest_session_removes_its_cache_even_when_it_fails` runs in a child session.)"""
    print(f"K3ROOT={os.environ.get(CC.ENV_ROOT)}")
    assert not os.environ.get("GEN3AI_K3_PROBE_FAIL"), "deliberate failure (the teardown probe)"
    from utils.paths import repo_path
    conf = Path(repo_path("conftest.py")).read_text()
    # ONE scratch helper: the conftest LOADS src/utils/scratch.py (by path) and has no copy of its own
    assert '"src", "utils", "scratch.py"' in conf and "def _fs_type" not in conf
    assert '".cache", "gen3ai", "tmp"' not in conf, "a second copy of the scratch-root rule"
    assert ('_COMPILE_CACHE_ENV = ("GEN3AI_COMPILE_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", '
            '"TRITON_CACHE_DIR")') in conf
    assert (CC.ENV_ROOT, CC.ENV_INDUCTOR, CC.ENV_TRITON) == (
        "GEN3AI_COMPILE_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")
    ind = os.environ.get(CC.ENV_INDUCTOR, "")
    assert ind and not CC._is_shared(ind)
    assert os.path.basename(os.environ[CC.ENV_ROOT]).startswith(f"gen3ai_pytest_compile_{os.getpid()}_")
    assert os.path.dirname(os.environ[CC.ENV_ROOT]) == CC.scratch_root(), "the test cache must be on disk"
    assert CC.fs_type(os.environ[CC.ENV_ROOT]) not in ("tmpfs", "ramfs")


def test_a_real_compile_lands_in_the_run_cache(tmp_path):
    """End to end in a child: declare a run's cache, `torch.compile` a function on CPU, and Inductor's
    own `cache_dir()` is the run's — with its artifacts inside `<run>/compile_cache/inductor`."""
    env = {k: v for k, v in os.environ.items() if k not in _KEYS}
    env["PYTHONPATH"] = str(src_path()) + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = ""
    run = tmp_path / "run"
    code = textwrap.dedent(f"""
        import json, os
        from agents.model import compile_cache as CC
        CC.cache_stamp = lambda: {{"schema": CC.SCHEMA, "code_sha": "x", "code_dirty": False,
                                  "torch": "t", "python": "p", "config_row_sha": "c"}}
        CC.prepare_run_cache({str(run)!r}, same_run_resume=False, emit=lambda m: None)
        import torch
        f = torch.compile(lambda x: (x.sin() * 2 + 1).relu())
        f(torch.randn(8, 8))
        try:
            from torch._inductor.runtime.cache_dir_utils import cache_dir
        except ImportError:
            from torch._inductor.runtime.runtime_utils import cache_dir
        print(json.dumps({{"cache_dir": cache_dir()}}))
    """)
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    got = json.loads(r.stdout.strip().splitlines()[-1])
    ind = run / CC.CACHE_SUBDIR / "inductor"
    assert got["cache_dir"] == str(ind)
    assert CC.dir_bytes(str(ind)) > 0, "the compile wrote nothing into the run's cache"


# --------------------------------------------------------------------------- the wiring
def _module_source(rel: str) -> str:
    return Path(src_path(rel)).read_text()


def test_every_production_compile_site_declares_the_hermetic_cache():
    """A module under src/agents, src/main or src/utils that calls `torch.compile` /
    `aoti_compile_and_package` must route through the hermetic cache in the same module (directly, or
    via `compile_opponents._inductor_cache_dir` / `compile_cache.t2_aot_dir`). A new compile site that
    forgets would compile into whatever torch defaults to — the shared dir this retired."""
    markers = ("ensure_hermetic_cache", "_inductor_cache_dir", "t2_aot_dir")
    offenders = []
    for top in ("agents", "main", "utils"):
        for p in Path(src_path(top)).rglob("*.py"):
            if p.name.endswith("_test.py") or p.name == "compile_cache.py":
                continue
            src = p.read_text(errors="replace")
            if "compile(" not in src:
                continue
            tree = ast.parse(src)
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute)
                     and ((n.func.attr == "compile" and isinstance(n.func.value, ast.Name)
                           and n.func.value.id in ("torch", "th"))
                          or n.func.attr == "aoti_compile_and_package")]
            called = {(n.func.id if isinstance(n.func, ast.Name) else getattr(n.func, "attr", None))
                      for n in ast.walk(tree) if isinstance(n, ast.Call)}
            if calls and not (called & set(markers)):     # a CALL, never a bare import
                offenders.append(str(p.relative_to(src_path())))
    assert not offenders, f"compile sites with no hermetic cache declaration: {offenders}"


def test_the_trainer_declares_the_cache_before_anything_compiles_or_spawns():
    """In `train_rl_agent.main`, `_declare_compile_cache(` follows `os.makedirs(model_dir` and precedes
    every compile / spawn site that would otherwise inherit or create a cache first."""
    src = _module_source("main/train_rl_agent.py")
    body = src[src.index("async def main():"):]
    at = body.index("_declare_compile_cache(args, model_dir)")
    assert body.index("os.makedirs(model_dir, exist_ok=True)") < at
    for later in ("build_rust_vec_env(", "build_and_train("):
        if later in body:
            assert at < body.index(later), f"{later} runs before the run's compile cache is declared"


def test_the_same_run_predicate_is_the_fork_lr_one():
    """Reuse is licensed only by a checkpoint THIS run wrote — the one predicate the launcher and the
    fork-LR guard already share; a warm-start init inside the run dir is NOT one."""
    src = _module_source("main/train/lifecycle.py")
    fn = src[src.index("def _declare_compile_cache"):src.index("def _maybe_compile_trainer")]
    assert "is_same_run_checkpoint(" in fn and "same_run_resume=" in fn
    from main.train.fork_lr import is_same_run_checkpoint
    assert is_same_run_checkpoint("/m/run/checkpoints/a.zip", "/m/run")
    assert not is_same_run_checkpoint("/m/run/warmstart/warmstart_consensus.zip", "/m/run")
    assert not is_same_run_checkpoint("/m/other/checkpoints/a.zip", "/m/run")


def test_the_retired_shared_dirs_are_named_nowhere_as_a_value():
    """No string LITERAL under src/ may be `/tmp/gen3ai_inductor_cache` (the retired opponents' dir) or
    a `torchinductor_` path, except `compile_cache`'s own retired-dir marker (history in comments is
    fine; a value is a use)."""
    hits = []
    for p in Path(src_path()).rglob("*.py"):
        if p.name in ("compile_cache.py", "compile_cache_test.py", "compile_extractor_test.py"):
            continue
        text = p.read_text(errors="replace")
        if "gen3ai_inductor_cache" not in text and "torchinductor_" not in text:
            continue
        for n in ast.walk(ast.parse(text)):
            if isinstance(n, ast.Constant) and isinstance(n.value, str) and (
                    "/tmp/gen3ai_inductor_cache" in n.value or "/tmp/torchinductor_" in n.value):
                hits.append(f"{p.relative_to(src_path())}:{n.lineno}")
    assert not hits, hits


def test_the_benchmark_writes_only_the_explicit_json_path(tmp_path, monkeypatch):
    """`compile_cache_benchmark` writes NOTHING unless `--json` names a file, and then exactly that
    file — never a repo-relative path that could land in another checkout (2026-09-30: a result was
    filed into the MAIN checkout by a shell `cd` that ran in a backgrounded subshell)."""
    from agents.model import compile_cache_benchmark as B
    monkeypatch.setattr(B, "run", lambda parts, rounds, device, timeout_s, **kw: {"rows": [], "parts": list(parts)})
    monkeypatch.chdir(tmp_path)
    assert B.main(["run", "--device", "cpu", "--parts", "opponent"]) == 0
    assert list(tmp_path.iterdir()) == [], "no --json must write no file"
    out = tmp_path / "sub" / "r.json"
    out.parent.mkdir()
    assert B.main(["run", "--device", "cpu", "--parts", "opponent", "--json", str(out)]) == 0
    assert json.loads(out.read_text())["parts"] == ["opponent"]
    assert sorted(p.name for p in tmp_path.rglob("*") if p.is_file()) == ["r.json"]
