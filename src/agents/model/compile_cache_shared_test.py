"""The SHARED compile cache (`agents.model.compile_cache_shared`, gen3_shared_compile_cache_v1; owner
2026-10-10: "Shared cache is fine as long as we check the git hash and the workspace is clean").

Every test fails on a revert of the rule it names. The checkouts are REAL git repos in `tmp_path`
(a tracked file, a `src/` file, a submodule at `deps/pokemon-showdown`), the shared root is the test's
own, and the root conftest keeps sharing OFF for every other test (`GEN3AI_SHARED_COMPILE_CACHE=0`).
"""
from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.model import compile_cache as CC
from agents.model import compile_cache_shared as SH
from utils.paths import src_path

_ENV = (CC.ENV_ROOT, CC.ENV_INDUCTOR, CC.ENV_TRITON, SH.ENV_T2_AOT)
_PLATFORM = {"python_version": "3.11.test", "triton": "3.4.0", "cuda": "12.6",
             "gpu": ["NVIDIA Test GPU, 8.9, 555.00"], "cuda_visible_devices": None}
_RUN = {"arch": {"token_encoding": "legacy", "trunk_rounds": 2},
        "compile_args": {"compile_trainer": True, "device": "cuda", "batch_size": 4096}}


# --------------------------------------------------------------------------- helpers
def _git(cwd, *args):
    return subprocess.check_output(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "-c", "commit.gpgsign=false",
         "-c", "core.hooksPath=/dev/null", "-c", "protocol.file.allow=always", *args],
        cwd=str(cwd), text=True, stderr=subprocess.STDOUT)


def _commit(repo, msg="c"):
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD").strip()


def _make_repo(base: Path, name="checkout") -> Path:
    """A checkout shaped like ours: `src/`, `designs/`, a `.gitignore` for `__pycache__`, and a git
    SUBMODULE at `deps/pokemon-showdown`."""
    sub = base / f"{name}_showdown_upstream"
    sub.mkdir()
    _git(sub, "init", "-q")
    (sub / "sim.js").write_text("// the JS simulator\n")
    _commit(sub)
    repo = base / name
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "src" / "agents").mkdir(parents=True)
    (repo / "src" / "agents" / "model.py").write_text("X = 1\n")
    (repo / "designs").mkdir()
    (repo / "designs" / "doc.md").write_text("# doc\n")
    (repo / ".gitignore").write_text("__pycache__/\ntmp/\n")
    _git(repo, "submodule", "add", "-q", str(sub), "deps/pokemon-showdown")
    _commit(repo, "init")
    return repo


@pytest.fixture
def repo(tmp_path):
    return _make_repo(tmp_path)


@pytest.fixture
def shared_on(tmp_path, monkeypatch):
    """Sharing ON against this test's own root; the cache variables and leases restored after."""
    saved = {k: os.environ.get(k) for k in _ENV}
    monkeypatch.setenv(SH.ENV_OPT, "1")
    monkeypatch.setenv(SH.ENV_DIR, str(tmp_path / "shared"))
    monkeypatch.delenv("LAUNCHER_GIT_HASH", raising=False)
    monkeypatch.setattr(SH, "platform_fields", lambda: dict(_PLATFORM))
    yield tmp_path / "shared"
    for path in list(SH._LEASES):
        SH.release(path)
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _stamp_for(repo, **over):
    s = {"schema": CC.SCHEMA, "code_sha": _git(repo, "rev-parse", "HEAD").strip(),
         "code_dirty": False, "torch": "2.8.0+cu126", "python": "/env/bin/python3",
         "config_row_sha": "c" * 64}
    s.update(over)
    return s


@pytest.fixture
def checkout(repo, monkeypatch):
    """Point the cache at `repo` as the code's checkout (production: `utils.paths.repo_root()`);
    the K3 stamp follows the repo's HEAD unless a test overrides a field."""
    over: dict = {}
    monkeypatch.setattr(SH, "code_root", lambda: str(repo))
    monkeypatch.setattr(CC, "cache_stamp", lambda: _stamp_for(repo, **over))
    return SimpleNamespace(repo=repo, over=over)


def _prepare(run_dir, run=_RUN, same_run_resume=False):
    lines: list = []
    d = CC.prepare_run_cache(str(run_dir), same_run_resume=same_run_resume, emit=lines.append, run=run)
    return d, lines


def _entries(root: Path):
    return sorted(p.name for p in root.iterdir() if re.fullmatch(r"[0-9a-f]{32}", p.name)) \
        if root.exists() else []


# --------------------------------------------------------------------------- the clean rule
def test_the_clean_rule_on_a_real_checkout(repo):
    assert SH.workspace_state(str(repo)).clean is True
    (repo / "tmp").mkdir()
    (repo / "tmp" / "scratch.py").write_text("print(1)\n")          # ignored
    (repo / "notes.txt").write_text("untracked OUTSIDE src/\n")     # untracked, outside src/
    (repo / "src" / "agents" / "__pycache__").mkdir()
    (repo / "src" / "agents" / "__pycache__" / "model.cpython-311.pyc").write_bytes(b"\0")
    st = SH.workspace_state(str(repo))
    assert st.clean is True, st.detail
    # a tracked edit ANYWHERE (not only src/) is dirty
    (repo / "designs" / "doc.md").write_text("# edited\n")
    st = SH.workspace_state(str(repo))
    assert st.clean is False and "designs/doc.md" in st.detail
    _git(repo, "checkout", "--", "designs/doc.md")
    # a STAGED new file is dirty
    (repo / "designs" / "new.md").write_text("x\n")
    _git(repo, "add", "designs/new.md")
    assert SH.workspace_state(str(repo)).clean is False
    _git(repo, "reset", "-q", "designs/new.md")
    os.remove(repo / "designs" / "new.md")
    # an untracked file under src/ is dirty
    (repo / "src" / "agents" / "stray.py").write_text("Y = 2\n")
    st = SH.workspace_state(str(repo))
    assert st.clean is False and "src/agents/stray.py" in st.detail
    os.remove(repo / "src" / "agents" / "stray.py")
    assert SH.workspace_state(str(repo)).clean is True


def test_an_IGNORED_importable_file_under_src_is_DIRT(repo):
    """Owner 2026-10-10: the shared cache is fine "as long as we're confident it comes from a hash with no
    changes". `git status` never reports an IGNORED file, so a gitignored `src/foo.py` (importable, able to change
    behaviour) passed as clean before `untracked_importables`. Fails on revert of that scan."""
    with open(repo / ".gitignore", "a") as f:
        f.write("src/foo.py\nsrc/legacy.pyc\nsrc/ext.so\nsrc/*.json\nsrc/rust_env/target/\n")
    _commit(repo, "ignore rules")
    assert SH.workspace_state(str(repo)).clean is True
    for planted in ("foo.py", "legacy.pyc", "ext.so"):                  # each one alone makes the tree dirty
        (repo / "src" / planted).write_bytes(b"X = 1\n")
        st = SH.workspace_state(str(repo))
        assert st.clean is False and f"src/{planted}" in st.detail, (planted, st.detail)
        os.remove(repo / "src" / planted)
    # NOT dirt: a non-importable ignored file, a __pycache__ byte-code cache (CPython validates it against the
    # tracked source), and a Cargo build dir (rebuilt from tracked crate sources; no torch-compile input)
    (repo / "src" / "table.json").write_text("{}\n")
    (repo / "src" / "agents" / "__pycache__").mkdir()
    (repo / "src" / "agents" / "__pycache__" / "model.cpython-311.pyc").write_bytes(b"\0")
    (repo / "src" / "rust_env" / "target" / "release").mkdir(parents=True)
    (repo / "src" / "rust_env" / "target" / "release" / "libpokesim.so").write_bytes(b"\0")
    st = SH.workspace_state(str(repo))
    assert st.clean is True, st.detail
    assert SH.untracked_importables(str(repo)) == []


def test_the_launchers_symlinked_submodule_is_not_dirt(repo, tmp_path):
    """The launcher's pinned worktree replaces `deps/pokemon-showdown` with a SYMLINK to the launching
    checkout's submodule (`main/launcher/worktree.py`). No torch compile reads it; it must not make
    every pinned run unshareable."""
    import shutil
    ps = repo / "deps" / "pokemon-showdown"
    shutil.rmtree(ps)
    os.symlink(str(tmp_path / "checkout_showdown_upstream"), str(ps))
    # precondition: git DOES see it — a plain `status --porcelain` even ERRORS on this layout
    plain = subprocess.run(["git", "status", "--porcelain"], cwd=str(repo), capture_output=True, text=True)
    assert plain.returncode != 0 and "symbolic link" in plain.stderr, (plain.returncode, plain.stderr)
    assert "deps/pokemon-showdown" in _git(repo, "diff-files", "--name-status")
    st = SH.workspace_state(str(repo))
    assert st.clean is True, st.detail


def test_a_DIRTY_tree_never_reads_or_writes_the_shared_cache(tmp_path, shared_on, checkout):
    repo = checkout.repo
    d0, _ = _prepare(tmp_path / "run_clean")
    assert d0.action == "shared-new"
    (entry,) = _entries(shared_on)
    used = (shared_on / entry / SH.LAST_USED_FILE).stat().st_mtime_ns
    before = sorted(str(p) for p in shared_on.rglob("*"))
    time.sleep(0.02)
    for dirt in ("tracked", "untracked_src"):
        if dirt == "tracked":
            (repo / "src" / "agents" / "model.py").write_text("X = 2\n")
        else:
            _git(repo, "checkout", "--", "src/agents/model.py")
            (repo / "src" / "agents" / "new_module.py").write_text("Z = 3\n")
        run = tmp_path / f"run_{dirt}"
        d, lines = _prepare(run)
        assert d.action == "created" and Path(d.root) == run / CC.CACHE_SUBDIR, (dirt, d)
        assert os.environ[CC.ENV_INDUCTOR] == str(run / CC.CACHE_SUBDIR / "inductor")
        assert not str(os.environ[CC.ENV_ROOT]).startswith(str(shared_on))
        notices = [ln for ln in lines if "NOT SHARED" in ln]
        assert len(notices) == 1 and "not clean" in notices[0], lines
        assert sorted(str(p) for p in shared_on.rglob("*")) == before, "a dirty tree WROTE the shared root"
        assert (shared_on / entry / SH.LAST_USED_FILE).stat().st_mtime_ns == used, \
            "a dirty tree OPENED a shared entry"


def test_a_clean_second_run_with_the_same_key_HITS(tmp_path, shared_on, checkout):
    d1, _ = _prepare(tmp_path / "a")
    marker = Path(d1.root) / "inductor" / "fxgraph" / "artifact"
    marker.parent.mkdir(parents=True)
    marker.write_text("compiled by run a")
    d2, lines = _prepare(tmp_path / "b")
    assert d2.action == "shared-hit" and d2.root == d1.root, lines
    assert os.environ[CC.ENV_INDUCTOR] == str(Path(d1.root) / "inductor")
    assert os.environ[CC.ENV_TRITON] == str(Path(d1.root) / "triton")
    assert marker.exists()
    rec = json.loads((tmp_path / "b" / CC.CACHE_SUBDIR / "stamp.json").read_text())
    assert rec["mode"] == "shared" and rec["entry"] == d1.root
    # T2's AOT packages go to the RUN's dir, never the shared entry
    aot = CC.t2_aot_dir()
    assert aot.startswith(str(tmp_path / "b" / CC.CACHE_SUBDIR / "t2_aot"))


# --------------------------------------------------------------------------- the key
def test_a_different_COMMIT_never_hits_another_commits_entry(tmp_path, shared_on, checkout):
    d1, _ = _prepare(tmp_path / "a")
    (checkout.repo / "src" / "agents" / "model.py").write_text("X = 99\n")
    _commit(checkout.repo, "next")
    d2, _ = _prepare(tmp_path / "b")
    assert d2.action == "shared-new" and d2.root != d1.root
    fields = json.loads((Path(d2.root) / SH.STAMP_FILE).read_text())["fields"]
    assert fields["code_sha"] == _git(checkout.repo, "rev-parse", "HEAD").strip()


@pytest.mark.parametrize("field,value", [("torch", "2.9.0+cu128"), ("config_row_sha", "d" * 64),
                                         ("python", "/other_env/bin/python3")])
def test_a_K3_stamp_field_mismatch_is_a_MISS(tmp_path, shared_on, checkout, field, value):
    d1, _ = _prepare(tmp_path / "a")
    checkout.over[field] = value
    d2, _ = _prepare(tmp_path / "b")
    assert d2.action == "shared-new" and d2.root != d1.root, (field, d2)


@pytest.mark.parametrize("change", ["triton", "gpu", "torch_env", "arch", "compile_flag"])
def test_a_platform_env_or_run_surface_change_is_a_MISS(tmp_path, shared_on, checkout, monkeypatch,
                                                        change):
    d1, _ = _prepare(tmp_path / "a")
    run = json.loads(json.dumps(_RUN))
    if change == "triton":
        monkeypatch.setattr(SH, "platform_fields", lambda: dict(_PLATFORM, triton="3.5.0"))
    elif change == "gpu":
        monkeypatch.setattr(SH, "platform_fields", lambda: dict(_PLATFORM, gpu=["Other GPU, 9.0, 560"]))
    elif change == "torch_env":
        monkeypatch.setenv("TORCHINDUCTOR_MAX_AUTOTUNE", "1")
    elif change == "arch":
        run["arch"]["trunk_rounds"] = 3
    else:
        run["compile_args"]["compile_trainer"] = False
    d2, _ = _prepare(tmp_path / "b", run=run)
    assert d2.action == "shared-new" and d2.root != d1.root, change


def test_an_entry_whose_stamp_is_not_this_key_is_NEVER_served(tmp_path, shared_on, checkout):
    """The last line of defence: the entry's stamp holds the FULL fields and is compared on every
    open. A stamp that differs (a key-derivation bug, a hand edit) is a MISS → the per-run cache."""
    d1, _ = _prepare(tmp_path / "a")
    for path in list(SH._LEASES):
        SH.release(path)
    stamp = Path(d1.root) / SH.STAMP_FILE
    rec = json.loads(stamp.read_text())
    rec["fields"]["k3_stamp"]["torch"] = "2.5.1+cu121"
    stamp.write_text(json.dumps(rec))
    d2, lines = _prepare(tmp_path / "b")
    assert d2.action == "created" and Path(d2.root) == tmp_path / "b" / CC.CACHE_SUBDIR
    assert any("NOT SHARED" in ln and "never served" in ln for ln in lines), lines


def test_the_opt_out_and_an_unreadable_gpu_fall_back(tmp_path, shared_on, checkout, monkeypatch):
    monkeypatch.setenv(SH.ENV_OPT, "0")
    d, lines = _prepare(tmp_path / "a")
    assert d.action == "created" and any("opted out" in ln for ln in lines)
    monkeypatch.setenv(SH.ENV_OPT, "maybe")
    assert SH.enabled()[0] is False
    monkeypatch.setenv(SH.ENV_OPT, "1")
    monkeypatch.setattr(SH, "platform_fields", lambda: dict(_PLATFORM, gpu=None))
    d, lines = _prepare(tmp_path / "b")
    assert d.action == "created" and any("GPU identity" in ln for ln in lines)
    assert _entries(shared_on) == []


def test_a_PINNED_childs_key_uses_the_PIN_commit(tmp_path, shared_on, monkeypatch):
    """The key's commit is the HEAD of the checkout the code was IMPORTED from — a pinned child's
    own worktree — and must agree with the launcher's `$LAUNCHER_GIT_HASH`."""
    pin = _make_repo(tmp_path, "pin")
    pin_sha = _git(pin, "rev-parse", "HEAD").strip()
    main = _make_repo(tmp_path, "main_checkout")
    (main / "src" / "agents" / "model.py").write_text("X = 7\n")
    main_sha = _commit(main, "main moved on")
    assert pin_sha != main_sha
    monkeypatch.chdir(main)                         # the child stands in main (no `cwd=` at spawn)
    p = SH.plan(_stamp_for(pin), _RUN, root=str(pin))
    assert p.use and p.fields["code_sha"] == pin_sha and p.fields["k3_stamp"]["code_sha"] == pin_sha
    monkeypatch.setenv("LAUNCHER_GIT_HASH", pin_sha)
    assert SH.plan(_stamp_for(pin), _RUN, root=str(pin)).use
    monkeypatch.setenv("LAUNCHER_GIT_HASH", main_sha)
    p = SH.plan(_stamp_for(pin), _RUN, root=str(pin))
    assert not p.use and "LAUNCHER_GIT_HASH" in p.reason


def test_the_code_root_is_the_IMPORTED_checkout_not_the_cwd(tmp_path):
    """A real child standing in ANOTHER repo (the launcher spawns a pinned child with no `cwd=`):
    the commit the key reads is the one this code was imported from."""
    other = _make_repo(tmp_path, "elsewhere")
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src_path()) + os.pathsep + env.get("PYTHONPATH", "")
    code = ("import json\nfrom agents.model import compile_cache_shared as SH\n"
            "r = SH.code_root()\nprint(json.dumps({'root': r, 'sha': SH.head_sha(r)}))\n")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(other), env=env, capture_output=True,
                       text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    got = json.loads(r.stdout.strip().splitlines()[-1])
    from utils.paths import repo_root
    assert got["root"] == str(repo_root())
    assert got["sha"] == _git(repo_root(), "rev-parse", "HEAD").strip()
    assert got["sha"] != _git(other, "rev-parse", "HEAD").strip()


def test_every_compile_flag_is_keyed_or_declared_irrelevant():
    """A trainer flag whose name says it shapes a compile must be in the key (`COMPILE_ARG_KEYS`)
    or declared irrelevant — a new one forces the decision."""
    from main.train.parser import build_parser
    dests = set(vars(build_parser().parse_args([])))
    pat = re.compile(r"compil|precision|checkpoint|cudagraph|inductor|dynamo|determin|bf16|dtype|"
                     r"device|batch_size|t2_|backend|bucket|lanes")
    flagged = {d for d in dests if pat.search(d)}
    assert flagged <= set(SH.COMPILE_ARG_KEYS) | set(SH.NOT_COMPILE_ARG_KEYS), \
        sorted(flagged - set(SH.COMPILE_ARG_KEYS) - set(SH.NOT_COMPILE_ARG_KEYS))
    assert set(SH.COMPILE_ARG_KEYS) <= dests, sorted(set(SH.COMPILE_ARG_KEYS) - dests)
    class Resolved(SimpleNamespace):                 # resolve_config fills every toggle before this
        def __getattr__(self, name):
            return 0.0 if name.endswith("_coef") else None
    fields = SH.run_fields(Resolved(compile_trainer=True, device="cuda"))
    assert fields["compile_args"]["compile_trainer"] is True
    assert set(fields["compile_args"]) == set(SH.COMPILE_ARG_KEYS) - set(SH.COMPILE_ARG_KEYS_WHEN_SET) \
        and fields["arch"]
    assert set(SH.COMPILE_ARG_KEYS_WHEN_SET) <= set(SH.COMPILE_ARG_KEYS)


def test_a_when_set_compile_key_keys_only_when_set():
    """X45 (`--t2-grouped-opponents`): a compile key added after entries existed enters the key only when SET, so a
    run that leaves it off keeps the key it had before the flag existed — and one that sets it gets its own."""
    class Resolved(SimpleNamespace):
        def __getattr__(self, name):
            return 0.0 if name.endswith("_coef") else None
    for k in SH.COMPILE_ARG_KEYS_WHEN_SET:
        off = SH.run_fields(Resolved(compile_trainer=True, device="cuda", **{k: False}))
        unset = SH.run_fields(Resolved(compile_trainer=True, device="cuda"))
        on = SH.run_fields(Resolved(compile_trainer=True, device="cuda", **{k: True}))
        assert k not in off["compile_args"] and SH.key_of(off) == SH.key_of(unset)
        assert on["compile_args"][k] is True and SH.key_of(on) != SH.key_of(off)


def test_the_real_platform_fields_name_the_versions(monkeypatch):
    import torch
    monkeypatch.setattr(SH, "_gpu_identity", lambda: ["stub"])   # no NVML read from a test
    p = SH.platform_fields()
    assert p["python_version"] == sys.version and p["cuda"] == torch.version.cuda
    assert p["gpu"] == ["stub"] and "cuda_visible_devices" in p


# --------------------------------------------------------------------------- atomicity
def test_an_entry_is_published_atomically_under_a_concurrent_writer(tmp_path, monkeypatch):
    """A concurrent creator publishes the same key WHILE this one is writing its stamp. The entry is
    never visible with an incomplete stamp, the loser leaves nothing behind, and both read one
    complete stamp."""
    root = tmp_path / "shared"
    root.mkdir()
    key, fields = "a" * 32, {"schema": SH.SCHEMA, "x": 1}
    real = SH._write_atomic
    state = {"competitor": False}

    def racing_write(path, text):
        # while ANY stamp is being written its entry must not yet be visible at the key's path
        assert not (root / key).exists() or state["competitor"], "entry visible before its stamp"
        if not state["competitor"]:
            state["competitor"] = True
            assert SH._publish(str(root), key, dict(fields, by="competitor")) is True
        real(path, text)

    monkeypatch.setattr(SH, "_write_atomic", racing_write)
    ours = SH._publish(str(root), key, fields)
    assert ours is False, "the loser of the rename must say so"
    rec = json.loads((root / key / SH.STAMP_FILE).read_text())
    assert rec["fields"]["by"] == "competitor"
    assert [p.name for p in root.iterdir()] == [key], "the loser left its .new-* dir behind"
    assert sorted(p.name for p in (root / key).iterdir()) == sorted(
        [SH.LOCK_FILE, SH.STAMP_FILE, SH.LAST_USED_FILE, "inductor", "triton"])


def test_concurrent_processes_create_one_entry_and_all_lease_it(tmp_path):
    """Eight real processes acquire the same key at once: exactly one publishes, all hold a lease on
    the same entry, no `.new-*` survives."""
    root = tmp_path / "shared"
    root.mkdir()
    go = tmp_path / "go"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src_path()) + os.pathsep + env.get("PYTHONPATH", "")
    code = textwrap.dedent(f"""
        import json, os, time
        from agents.model import compile_cache_shared as SH
        while not os.path.exists({str(go)!r}):
            time.sleep(0.005)
        e = SH.acquire({str(root)!r}, "b" * 32, {{"schema": SH.SCHEMA, "x": 1}})
        print(json.dumps({{"created": e.created, "path": e.path}}), flush=True)
    """)
    procs = [subprocess.Popen([sys.executable, "-c", code], env=env, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True) for _ in range(8)]
    time.sleep(1.0)
    go.write_text("go")
    outs = []
    for p in procs:
        o, e = p.communicate(timeout=120)
        assert p.returncode == 0, e[-2000:]
        outs.append(json.loads(o.strip().splitlines()[-1]))
    assert sum(o["created"] for o in outs) == 1, outs
    assert {o["path"] for o in outs} == {str(root / ("b" * 32))}
    assert [p.name for p in root.iterdir()] == ["b" * 32]


def test_torchs_own_cache_writes_are_atomic_renames():
    """The writers INSIDE an entry are torch's and triton's (the module docstring's audit): pinned so
    an upgrade that drops the temp-file-plus-rename pattern fails here instead of corrupting a shared
    entry under a concurrent writer."""
    import inspect

    from torch._inductor import codecache
    src = inspect.getsource(codecache.write_atomic)
    assert "tmp_path" in src and ".rename(" in src
    from torch._functorch._aot_autograd import autograd_cache
    assert "write_atomic" in inspect.getsource(autograd_cache)
    try:
        from triton.runtime import cache as tcache
    except ImportError:
        pytest.skip("no triton in this interpreter (a CPU torch)")
    put = inspect.getsource(tcache.FileCacheManager.put)
    assert "os.replace(" in put and "tmp.pid_" in put


# --------------------------------------------------------------------------- leases + pruning
def _fake_entry(root: Path, key: str, mb: float, used: float) -> Path:
    e = root / key
    (e / "inductor").mkdir(parents=True)
    (e / "triton").mkdir()
    (e / SH.LOCK_FILE).touch()
    (e / SH.STAMP_FILE).write_text(json.dumps({"fields": {"k": key}}))
    (e / "inductor" / "blob").write_bytes(b"\0" * int(mb * (1 << 20)))
    lu = e / SH.LAST_USED_FILE
    lu.touch()
    os.utime(lu, (used, used))
    return e


def _hold_lease_in_child(entry: Path):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src_path()) + os.pathsep + env.get("PYTHONPATH", "")
    code = ("import fcntl, os, sys, time\n"
            f"fd = os.open({str(entry / SH.LOCK_FILE)!r}, os.O_RDWR)\n"
            "fcntl.flock(fd, fcntl.LOCK_SH)\nprint('held', flush=True)\ntime.sleep(300)\n")
    p = subprocess.Popen([sys.executable, "-c", code], env=env, stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "held"
    return p


def test_pruning_keeps_the_cap_LRU_first_and_never_a_leased_entry(tmp_path):
    root = tmp_path / "shared"
    root.mkdir()
    now = time.time()
    keys = [f"{i:x}" * 32 for i in range(1, 6)]               # five 1 MiB entries, oldest first
    entries = [_fake_entry(root, k[:32], 1.0, now - 1000 + 100 * i) for i, k in enumerate(keys)]
    (root / f"{SH.TRASH_PREFIX}leftover").mkdir()
    (root / f"{SH.NEW_PREFIX}{'f' * 32}-999999999-x").mkdir()   # a dead creator's half-built entry
    child = _hold_lease_in_child(entries[0])                     # the OLDEST is in use by a live process
    try:
        rep = SH.prune(str(root), int(2.5 * (1 << 20)), keep=(entries[4].name,))
    finally:
        child.kill()
        child.wait(timeout=30)
    left = sorted(p.name for p in root.iterdir() if not p.name.startswith("."))
    assert entries[0].name in left, "a LEASED entry was pruned"
    assert entries[4].name in left, "the caller's own entry was pruned"
    assert entries[1].name not in left and entries[2].name not in left, "LRU order not honoured"
    assert rep.in_use_skipped == [str(entries[0])]
    # LRU: [1], [2], [3] go; the leased oldest [0] and the kept newest [4] stay (2 MiB <= the cap)
    assert entries[3].name not in left and rep.total_after <= int(2.5 * (1 << 20))
    assert not any(p.name.startswith((SH.TRASH_PREFIX, SH.NEW_PREFIX)) for p in root.iterdir())
    # once the lease is gone, the next prune reaches the cap
    rep2 = SH.prune(str(root), int(1.5 * (1 << 20)), keep=(entries[4].name,))
    assert rep2.removed == [str(entries[0])] and rep2.total_after <= int(1.5 * (1 << 20))


def test_a_child_that_inherits_a_shared_entry_takes_its_own_lease(tmp_path):
    root = tmp_path / "shared"
    root.mkdir()
    e = SH.acquire(str(root), "c" * 32, {"schema": SH.SCHEMA})
    SH.release(e.path)                                       # the "trainer" is gone
    env = {k: v for k, v in os.environ.items() if k not in _ENV}
    env["PYTHONPATH"] = str(src_path()) + os.pathsep + env.get("PYTHONPATH", "")
    env.update({CC.ENV_ROOT: e.path, CC.ENV_INDUCTOR: os.path.join(e.path, "inductor"),
                CC.ENV_TRITON: os.path.join(e.path, "triton")})
    code = ("import time\nfrom agents.model.compile_cache import ensure_hermetic_cache\n"
            "print(ensure_hermetic_cache('child'), flush=True)\ntime.sleep(300)\n")
    p = subprocess.Popen([sys.executable, "-c", code], env=env, stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout.readline().strip() == e.path
        rep = SH.prune(str(root), 0)
        assert os.path.isdir(e.path) and rep.in_use_skipped == [e.path], "a child's entry was pruned"
    finally:
        p.kill()
        p.wait(timeout=30)
    assert SH.prune(str(root), 0).removed == [e.path]


# --------------------------------------------------------------------------- the gates run on a hit
@pytest.fixture
def a_shared_HIT(tmp_path, shared_on, checkout):
    _prepare(tmp_path / "first")
    d, _ = _prepare(tmp_path / "second")
    assert d.action == "shared-hit"
    return d


def test_the_learner_startup_gate_and_the_canary_are_ARMED_on_a_hit(a_shared_HIT, monkeypatch):
    from agents.model import compile_canary, compile_control, compile_regions
    from agents.model.compile_trainer import arm_compile_sentinel
    calls: list = []

    class Ctl:
        canary = None

        def install(self):
            return self

        def reset(self):
            return "reset"

        def prewarm(self, _calls):
            calls.append("prewarm")
            return "prewarm"

        def lock(self, _why):
            calls.append("lock")

        def attach(self, _model):
            calls.append("attach")

    ctl = Ctl()
    monkeypatch.setattr(compile_control, "control", lambda emit=None: ctl)
    monkeypatch.setattr(compile_control, "set_strict_errors", lambda: None)
    monkeypatch.setattr(compile_regions, "install", lambda model, emit=None: ["R1"])
    monkeypatch.setattr(compile_regions, "gate_regions",
                        lambda model, batch_size, say: calls.append(("gate", batch_size)))
    monkeypatch.setattr(compile_regions, "prewarm_calls", lambda model, batch_size: [])
    monkeypatch.setattr(compile_regions, "assert_inventory", lambda model: "inventory")
    monkeypatch.setattr(compile_canary.CompileCanary, "run",
                        lambda self: calls.append(("canary", self.updates)) or {"canary": 1.0})
    # `modules` : the startup gate now releases its stashed autograd graphs (gen3_gate_releases_memory_v1,
    # `release_gate_memory` walks `policy.modules()`); a fake policy with no modules has none to release
    model = SimpleNamespace(policy=SimpleNamespace(features_extractor=object(), modules=lambda: []),
                            _micro_static=object())
    arm_compile_sentinel(model, batch_size=64)
    assert ("gate", 64) in calls, "the R1 startup parity gate did not run on a cache HIT"
    assert calls.index(("gate", 64)) < calls.index("lock")
    assert isinstance(ctl.canary, compile_canary.CompileCanary) and "attach" in calls
    for _ in range(compile_canary.CANARY_FIRST):
        ctl.canary.after_update()
    assert ("canary", compile_canary.CANARY_FIRST) in calls, "the update-10 canary did not run on a HIT"


def test_T2s_startup_parity_gate_runs_every_slot_x_bucket_on_a_hit(a_shared_HIT):
    import torch

    from agents.inference.service import InferenceService, ServiceSpec, SlotGroupSpec
    from agents.inference.service.fixtures import perturbed_fresh_policy
    from utils.torch_state_guard import torch_globals
    with torch_globals(num_threads=2):
        pol = perturbed_fresh_policy(0)
        svc = InferenceService(ServiceSpec(groups=(SlotGroupSpec("pool", 2, pol),), device="cpu",
                                           backend="eager", buckets=(2, 8))).startup()
    wheres = [r.where for r in svc.startup_reports]
    for slot in (0, 1):
        for b, rows in ((2, 2), (2, 1), (8, 8), (8, 7)):
            assert f"eager group=pool slot={slot} bucket={b} rows={rows}" in wheres
    assert torch.get_num_threads() >= 1


def test_no_gate_module_reads_the_cache_decision():
    """Static twin of the two tests above: nothing that runs a compiled-learner gate may consult
    the shared cache (a hit only skips compile time)."""
    mods = ("agents/model/compile_regions.py", "agents/model/compile_canary.py",
            "agents/model/compile_trainer.py", "agents/model/compile_control.py",
            "agents/inference/service/service.py", "agents/inference/service/parity.py")
    for rel in mods:
        tree = ast.parse(Path(src_path(rel)).read_text())
        names = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module} | \
            {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
        assert not any("compile_cache_shared" in (m or "") for m in names), rel
        assert "shared-hit" not in Path(src_path(rel)).read_text(), rel


# --------------------------------------------------------------------------- a real compile, shared
def test_two_real_processes_share_one_compile(tmp_path):
    """End to end on CPU: a first process compiles into a shared entry; a second one, with the same
    key, is served from it (Inductor's FX-graph cache HIT) instead of recompiling."""
    env = {k: v for k, v in os.environ.items() if k not in _ENV}
    env["PYTHONPATH"] = str(src_path()) + os.pathsep + env.get("PYTHONPATH", "")
    env["CUDA_VISIBLE_DEVICES"] = ""
    env[SH.ENV_OPT] = "1"
    env[SH.ENV_DIR] = str(tmp_path / "shared")
    env.pop("LAUNCHER_GIT_HASH", None)
    repo = _make_repo(tmp_path)

    def child(run: str) -> dict:
        code = textwrap.dedent(f"""
            import json
            from agents.model import compile_cache as CC, compile_cache_shared as SH
            SH.code_root = lambda: {str(repo)!r}
            SH.platform_fields = lambda: {_PLATFORM!r}
            sha = SH.head_sha({str(repo)!r})
            CC.cache_stamp = lambda: {{"schema": CC.SCHEMA, "code_sha": sha, "code_dirty": False,
                                      "torch": "t", "python": "p", "config_row_sha": "c"}}
            d = CC.prepare_run_cache({str(tmp_path / run)!r}, same_run_resume=False,
                                     emit=lambda m: None, run={_RUN!r})
            import torch
            from torch._dynamo.utils import counters
            f = torch.compile(lambda x: (x.cos() * 3 - 1).relu())
            f(torch.randn(8, 8))
            print(json.dumps({{"action": d.action, "root": d.root,
                              "hit": counters["inductor"]["fxgraph_cache_hit"],
                              "miss": counters["inductor"]["fxgraph_cache_miss"]}}))
        """)
        r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True,
                           timeout=600)
        assert r.returncode == 0, r.stderr[-3000:]
        return json.loads(r.stdout.strip().splitlines()[-1])

    a = child("a")
    b = child("b")
    assert a["action"] == "shared-new" and a["miss"] >= 1, a
    assert b["action"] == "shared-hit" and b["root"] == a["root"], b
    assert b["hit"] >= 1 and b["miss"] == 0, b
