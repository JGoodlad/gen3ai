"""Concurrent pytest sessions do not share a temp tree (``utils.pytest_session_tmp``, the root ``conftest.py``).

The box runs several gates at once and a slow tier is hours. Each session used to take the next ``pytest-N`` of one
shared ``pytest-of-<user>``, and what kept another session's cleanup off a live one's tree was a ``.lock`` younger than
three days -- a clock, not a liveness read (FINDINGS of ``5fea300b8`` / ``b0441623e``: six slow tests errored when "a
concurrent pytest rotated the shared tmp root under them"). Now a session's tree is named for its owner, and the only
thing that removes it from outside is the sweep of a DEAD owner's.

The child-session cases run the REAL root conftest in real subprocesses (``pytest_session_tmp_probe_child.py``) with
``$GEN3AI_SCRATCH`` on a fresh on-disk dir, several at once, and fail on revert of the conftest half (the layout and
the dead-owner-only sweep) and of the liveness half (``/proc``, not ``kill``). The unit cases pin the module's rules
with injected liveness, so none depends on a pid the box happens to hold.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import pytest

from utils import procfs
from utils import pytest_session_tmp as T
from utils.contention import scale_timeout
from utils.paths import repo_root

_PROBE = "src/utils/pytest_session_tmp_probe_child.py"
DAY = 24 * 3600.0


# ----------------------------------------------------------------------------------------------------- unit rules
def _tree(root: Path, name: str, age_h: float = 0.0, now: float | None = None) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "f.txt").write_text("x")
    t = (time.time() if now is None else now) - age_h * 3600.0
    os.utime(d, (t, t))
    return d


def test_a_session_tree_is_named_for_the_process_that_owns_it(tmp_path):
    base = Path(T.session_basetemp(str(tmp_path)))
    pid, start = T.owner_token()
    assert base.parent == tmp_path / f"pytest-of-{T.user_name()}"
    assert base.name == f"session-{pid}-{start}" and T.parse_session_name(base.name) == (pid, start)
    assert start == procfs.read_stat(pid).starttime
    other = Path(T.session_basetemp(str(tmp_path), pid=os.getppid()))
    assert other != base and T.parse_session_name(other.name)[0] == os.getppid()


@pytest.mark.parametrize("name", ["pytest-3", "pytest-current", "session-12", "session-12-x", "session-a-1",
                                  "xsession-12-34", "session-12-34-5", ""])
def test_only_a_session_name_parses(name):
    assert T.parse_session_name(name) is None


def test_the_owner_is_alive_only_while_the_same_process_runs():
    pid, start = T.owner_token()
    assert T.owner_alive(pid, start)
    assert not T.owner_alive(pid, start + 1), "a REUSED pid (same number, another start time) is not the owner"
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    assert not T.owner_alive(dead.pid, 1)


def test_the_sweep_removes_only_dead_owners_and_never_a_live_one_at_any_age(tmp_path):
    root = tmp_path / f"pytest-of-{T.user_name()}"
    now = time.time()
    live_old = _tree(root, "session-100-1", age_h=500, now=now)          # alive, and older than any retention
    dead_old = _tree(root, "session-200-1", age_h=48, now=now)
    dead_new = _tree(root, "session-300-1", age_h=1, now=now)
    legacy = _tree(root, "pytest-7", age_h=500, now=now)                 # not a session's: tmp_sweep.py's
    junk = _tree(root, "session-junk", age_h=500, now=now)
    alive = lambda pid, start: pid == 100  # noqa: E731

    removed = T.sweep_dead_sessions(str(tmp_path), keep=1, now=now, alive=alive)

    assert removed == [str(dead_old)]
    assert live_old.exists() and dead_new.exists() and legacy.exists() and junk.exists()


def test_the_sweep_keeps_the_newest_dead_trees_up_to_the_retention_count_and_only_while_young(tmp_path):
    root = tmp_path / f"pytest-of-{T.user_name()}"
    now = time.time()
    trees = {n: _tree(root, f"session-{n}-1", age_h=h, now=now) for n, h in
             {1: 0.5, 2: 2.0, 3: 5.0, 4: 30.0}.items()}
    never = lambda pid, start: False  # noqa: E731

    T.sweep_dead_sessions(str(tmp_path), keep=2, retain_hours=24.0, now=now, alive=never)

    assert [n for n, d in trees.items() if d.exists()] == [1, 2], "the newest two, young enough, are kept"
    T.sweep_dead_sessions(str(tmp_path), keep=2, retain_hours=1.0, now=now, alive=never)
    assert [n for n, d in trees.items() if d.exists()] == [1], "retention is also by age"
    T.sweep_dead_sessions(str(tmp_path), keep=0, now=now, alive=never)
    assert not any(d.exists() for d in trees.values())


@pytest.mark.parametrize("policy,status,removed", [("failed", 0, True), ("failed", 1, False), ("failed", 2, False),
                                                   ("all", 0, False), ("none", 0, True), ("none", 1, True)])
def test_a_session_removes_its_own_tree_by_the_retention_policy(tmp_path, policy, status, removed):
    d = _tree(tmp_path, "session-1-1")
    assert T.finish_session(str(d), status, policy) is removed
    assert d.exists() is (not removed)
    assert T.finish_session(str(tmp_path / "never-made"), 0, "failed") is False
    assert T.finish_session(None, 0, "failed") is False


def test_the_compile_cache_sweep_removes_only_dead_creators_dirs(tmp_path):
    keep = [tmp_path / "gen3ai_pytest_compile_100_a", tmp_path / "gen3ai_compile_100_b", tmp_path / "other_300_c",
            tmp_path / "gen3ai_pytest_compile_nopid_d"]
    drop = [tmp_path / "gen3ai_pytest_compile_200_a", tmp_path / "gen3ai_compile_200_b"]
    for d in keep + drop:
        d.mkdir()
    got = T.sweep_dead_compile_caches(str(tmp_path), alive=lambda pid: pid == 100)
    assert sorted(got) == sorted(str(d) for d in drop)
    assert all(d.exists() for d in keep) and not any(d.exists() for d in drop)


# ----------------------------------------------------------------------------------------- real concurrent sessions
def _env(scratch: Path, out: Path, mode: str, extra_path: Path | None = None) -> Dict[str, str]:
    env = dict(os.environ, GEN3AI_SCRATCH=str(scratch), PROBE_OUT=str(out), PROBE_MODE=mode,
               PROBE_WAIT_S=str(scale_timeout(120)), GEN3AI_SKIP_SLOW_STATUS_RECORD="1")
    for k in ("TMPDIR", "PYTEST_XDIST_WORKER", "PYTEST_ADDOPTS", "PYTEST_DEBUG_TEMPROOT"):
        env.pop(k, None)                        # the child sets its own temp root; ours must not leak in
    if extra_path is not None:
        env["PYTHONPATH"] = os.pathsep.join([str(extra_path), env.get("PYTHONPATH", "")]).rstrip(os.pathsep)
    return env


def _argv() -> List[str]:
    return [sys.executable, "-m", "pytest", _PROBE, "-q", "-p", "no:cacheprovider", "-p", "no:xdist"]


def _start(scratch: Path, out: Path, mode: str, extra_path: Path | None = None) -> subprocess.Popen:
    return subprocess.Popen(_argv(), cwd=repo_root(), env=_env(scratch, out, mode, extra_path),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def _run(scratch: Path, out: Path, mode: str, extra_path: Path | None = None) -> dict:
    r = subprocess.run(_argv(), cwd=repo_root(), env=_env(scratch, out, mode, extra_path), capture_output=True,
                       text=True, timeout=scale_timeout(180), check=False)
    if mode != "fail":
        assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    else:
        assert r.returncode != 0, r.stdout[-3000:]
    return json.loads(out.read_text())


def _wait_for(path: Path, proc: subprocess.Popen) -> None:
    deadline = time.monotonic() + scale_timeout(120)
    while not path.exists():
        assert proc.poll() is None, f"the holding session ended early: {proc.stdout.read()[-3000:]}"
        assert time.monotonic() < deadline, f"{path} never appeared"
        time.sleep(0.1)


def _age(tree: Path, days: float) -> None:
    """Make a live session's tree LOOK abandoned by every clock a cleanup could read: every mtime in it (its
    ``.lock``, if it has one, included) is ``days`` old."""
    t = time.time() - days * DAY
    for dirpath, dirnames, filenames in os.walk(tree, topdown=False):
        for n in filenames + dirnames:
            os.utime(os.path.join(dirpath, n), (t, t), follow_symlinks=False)
    os.utime(tree, (t, t))


def _finish(proc: subprocess.Popen, out: Path) -> dict:
    stdout, _ = proc.communicate(timeout=scale_timeout(180))
    return json.loads(out.read_text()) | {"rc": proc.returncode, "stdout": stdout}


@pytest.mark.integration
def test_concurrent_sessions_own_separate_trees_and_no_session_removes_a_live_ones(tmp_path):
    """Three sessions at once: A holds a tree (aged ten days, so only its live owner protects it), B passes and
    C fails, both starting and exiting beside it. (The retention COUNT is the unit case above; each real session
    costs ~2 s of the ship gate's 8 s bar.)"""
    scratch, out = tmp_path / "scratch", tmp_path / "out"
    out.mkdir()
    holder = _start(scratch, out / "a.json", "hold")
    try:
        _wait_for(out / "a.json.up", holder)
        a = json.loads((out / "a.json.up").read_text())
        a_base, a_file = Path(a["basetemp"]), Path(a["tmp_path"]) / "keep.txt"
        assert a_base.parent == scratch / f"pytest-of-{T.user_name()}", a
        assert a_base.name == T.session_name(holder.pid, procfs.read_stat(holder.pid).starttime), (a, holder.pid)
        # A looks dead to every age / lock clock; only its owner being ALIVE can protect it from here on.
        _age(a_base, days=10)

        passing_proc = _start(scratch, out / "b.json", "quick")        # both start, sweep and exit beside A
        failing_proc = _start(scratch, out / "c.json", "fail")
        passing, failing = _finish(passing_proc, out / "b.json"), _finish(failing_proc, out / "c.json")

        assert passing["rc"] == 0, passing["stdout"][-3000:]
        assert failing["rc"] != 0, failing["stdout"][-3000:]
        trees = {Path(r["basetemp"]) for r in (passing, failing)} | {a_base}
        assert len(trees) == 3, "two concurrent sessions shared a basetemp"
        for r in (passing, failing):
            assert Path(r["basetemp"]).parent == a_base.parent
            assert Path(r["basetemp"]).name.startswith(f"session-{r['pid']}-"), r
        assert not Path(passing["basetemp"]).exists(), "a PASSING session removes its own tree"
        assert Path(failing["basetemp"]).exists(), "a FAILED session's tree is kept for the post-mortem"
        assert a_file.exists(), "B's or C's start / exit removed the live session A's tree"
    finally:
        (out / "a.json.go").write_text("go")
    stdout, _ = holder.communicate(timeout=scale_timeout(120))
    assert holder.returncode == 0, stdout[-3000:]
    assert json.loads((out / "a.json").read_text())["alive_at_end"] is True
    assert not a_base.exists(), "A passed: its own tree goes at its own end"


@pytest.mark.integration
def test_a_sandboxed_shell_that_reads_every_other_pid_as_dead_deletes_no_live_sessions_dirs(tmp_path):
    """Inside another agent's sandboxed shell ``kill(pid, 0)`` answers ESRCH for a LIVE process (acf2193f). The
    child's ``os.kill`` is made to do exactly that; the sweeps must read ``/proc``, so the live owner's compile cache
    and session tree survive and a really dead one's does not."""
    scratch, out, site = tmp_path / "scratch", tmp_path / "out", tmp_path / "site"
    out.mkdir()
    site.mkdir()
    (site / "sitecustomize.py").write_text(
        "import os\n_real = os.kill\n\n"
        "def _kill(pid, sig):\n"
        "    if sig == 0 and pid != os.getpid():\n"
        "        raise ProcessLookupError(3, 'No such process')\n"
        "    return _real(pid, sig)\n\n"
        "os.kill = _kill\n")
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    live = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)"])
    try:
        start = procfs.read_stat(live.pid).starttime
        scratch.mkdir()
        sessions = scratch / f"pytest-of-{T.user_name()}"
        live_cache = scratch / f"gen3ai_pytest_compile_{live.pid}_x"
        dead_cache = scratch / f"gen3ai_pytest_compile_{gone.pid}_x"
        live_tree = sessions / T.session_name(live.pid, start)
        dead_tree = sessions / T.session_name(gone.pid, 1)
        for d in (live_cache, dead_cache, live_tree, dead_tree):
            d.mkdir(parents=True)
            (d / "f").write_text("x")
        _age(live_tree, days=10)
        _age(dead_tree, days=10)

        _run(scratch, out / "s.json", "quick", extra_path=site)

        assert live_cache.exists(), "a live session's compile cache was swept (liveness read through kill)"
        assert live_tree.exists(), "a live owner's session tree was swept (liveness read through kill)"
        assert not dead_cache.exists() and not dead_tree.exists(), "a dead owner's dirs must still be swept"
    finally:
        live.kill()
        live.wait()
