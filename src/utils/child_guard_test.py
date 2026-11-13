"""``utils.child_guard`` against REAL processes: a child a test spawns dies with the test, and so does everything under
it, whatever kills the test (the 2026-11-13 orphan: ``timeout`` killed pytest, not the launcher's trainer).

The core test is the one the incident asks for: an INTERMEDIATE process (standing in for pytest) starts a guarded
tree (a shell standing in for the launcher, with a long sleeper standing in for the trainer), the intermediate is
SIGKILLed, and the whole tree must be gone within a bound. It fails on revert: with a plain ``Popen`` the sleeper lives
its full 300 s. No trainer is needed.

``integration``: spawns and signals its own processes, ~10 s. Every pid it makes is cleaned by explicit pid / pgid.
"""
from __future__ import annotations

import ast
import gc
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pytest

from utils import child_guard, procfs
from utils.child_guard import guarded_popen, guarded_run, kill_session
from utils.contention import scale_timeout
from utils.paths import src_root

pytestmark = pytest.mark.integration

#: The "pytest" of these tests: it starts a guarded launcher-shaped tree (a shell that starts a sleeper and waits),
#: records the pids and then idles. ``ignore-term`` makes the tree ignore SIGTERM (the launcher's SIGTERM path can
#: wait a trainer's whole save), so only the SIGKILL after the grace ends it.
_INTERMEDIATE = r"""
import json, os, sys, time
sys.path.insert(0, sys.argv[1])
from utils.child_guard import guarded_popen
out, mode, grace = sys.argv[2], sys.argv[3], float(sys.argv[4])
trap = "trap '' TERM; " if mode == "ignore-term" else ""
child = guarded_popen(["sh", "-c", trap + "sleep 300 & echo $! > " + out + ".gc; wait"], grace_s=grace)
json.dump({"intermediate": os.getpid(), "child": child.pid}, open(out, "w"))
time.sleep(300)
"""

_alive = procfs.pid_alive      # one atomic /proc read; a zombie or a vanished pid is not alive


def _wait_for(pred: Callable[[], object], what: str, timeout: float = 20.0) -> object:
    end = time.monotonic() + scale_timeout(timeout)
    while time.monotonic() < end:
        v = pred()
        if v:
            return v
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {what}")


def _pid_file(path: Path) -> Optional[int]:
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _start_intermediate(tmp_path: Path, mode: str, grace: float):
    """The intermediate, once its guarded tree exists: ``(intermediate Popen, child pid, grandchild pid)``."""
    out = tmp_path / "pids.json"
    inter = subprocess.Popen([sys.executable, "-I", "-c", _INTERMEDIATE, str(src_root()), str(out), mode, str(grace)])
    try:
        _wait_for(lambda: out.exists() and out.read_text().strip().endswith("}"), "the intermediate's pid file")
        child = int(re.search(r'"child": (\d+)', out.read_text()).group(1))      # type: ignore[union-attr]
        gc_pid = _wait_for(lambda: _pid_file(Path(str(out) + ".gc")), "the grandchild's pid")
    except BaseException:
        inter.kill()
        inter.wait()
        raise
    assert isinstance(gc_pid, int)
    return inter, child, gc_pid


def _kill_leftovers(*pids: int) -> None:
    """A failed assertion must not leave a sleeper: kill by EXPLICIT pid (each is a process this test started)."""
    for pid in pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


@pytest.mark.parametrize("sig", [signal.SIGKILL, signal.SIGTERM], ids=["SIGKILL", "SIGTERM"])
def test_the_whole_tree_dies_when_its_parent_is_killed(tmp_path, sig):
    """The incident: the parent dies by a signal that runs no ``finally`` and no ``atexit`` (SIGKILL, or the default
    SIGTERM of a ``timeout``), and the child AND the grandchild it started must go, not just the direct child."""
    inter, child, grandchild = _start_intermediate(tmp_path, "plain", 5.0)
    try:
        assert _alive(child) and _alive(grandchild)
        assert procfs.read_stat(child).session == child, "the guarded child must lead its own session"
        assert procfs.read_stat(grandchild).session == child, "the grandchild must stay in the child's session"
        inter.send_signal(sig)
        inter.wait()
        _wait_for(lambda: not _alive(child) and not _alive(grandchild),
                  f"the guarded tree ({child}, {grandchild}) to die after the parent's {sig.name}")
    finally:
        _kill_leftovers(grandchild, child)


def test_a_tree_that_ignores_sigterm_is_sigkilled_after_the_grace(tmp_path):
    """The launcher's own SIGTERM path waits a trainer's save; a test's tree has no use for it: after ``grace_s`` the
    SIGKILL goes out, so a tree that ignores SIGTERM still cannot outlive the parent."""
    inter, child, grandchild = _start_intermediate(tmp_path, "ignore-term", 1.0)
    try:
        inter.kill()
        inter.wait()
        t0 = time.monotonic()
        _wait_for(lambda: not _alive(child) and not _alive(grandchild), "the SIGTERM-ignoring tree to be SIGKILLed")
        assert time.monotonic() - t0 > 0.5, "it died before the grace: the tree did not ignore SIGTERM, the test is vacuous"
    finally:
        _kill_leftovers(grandchild, child)


def test_a_run_that_times_out_kills_the_whole_tree_not_only_the_direct_child(tmp_path, monkeypatch):
    """``subprocess.run`` kills the direct child on a timeout; a launcher's trainer would live on. The timeout is
    SIMULATED at the moment the tree exists (a real one races the shell's start-up)."""
    gc_file = tmp_path / "gc.pid"
    real = subprocess.Popen.communicate
    seen: Dict[str, int] = {}

    def communicate(self, input=None, timeout=None):
        if not seen:
            seen["child"] = self.pid
            _wait_for(lambda: _pid_file(gc_file), "the grandchild's pid")
            raise subprocess.TimeoutExpired(self.args, 0.1)
        return real(self, input, timeout)

    monkeypatch.setattr(subprocess.Popen, "communicate", communicate)
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            guarded_run(["sh", "-c", f"sleep 300 & echo $! > {gc_file}; wait"], timeout=1.0, grace_s=2.0)
        grandchild = _pid_file(gc_file)
        assert grandchild is not None and not _alive(grandchild), "the grandchild outlived the run's timeout"
        assert not _alive(seen["child"])
    finally:
        _kill_leftovers(*(p for p in (_pid_file(gc_file), seen.get("child")) if p))


def test_a_run_that_finishes_returns_like_subprocess_run_and_leaves_no_watcher(tmp_path):
    done = guarded_run(["sh", "-c", "echo out; echo err >&2; exit 3"], capture_output=True, text=True)
    assert (done.returncode, done.stdout, done.stderr) == (3, "out\n", "err\n")
    with pytest.raises(subprocess.CalledProcessError) as bad:
        guarded_run(["sh", "-c", "exit 4"], check=True)
    assert bad.value.returncode == 4
    watchers = list(child_guard._WATCHERS[-2:])          # the two runs above, whatever earlier tests left behind
    assert len(watchers) == 2
    gc.collect()
    _wait_for(lambda: all(w.poll() is not None for w in watchers), "every watcher to exit once its child's tree is gone")


def test_dropping_the_popen_kills_what_the_child_left_behind(tmp_path):
    """The child exits 0 and leaves a sleeper in its session (a launcher that exited and left its trainer): the
    watcher keeps the session, and the leash closes with the ``Popen``, so the leftover goes with it."""
    gc_file = tmp_path / "gc.pid"
    p = guarded_popen(["sh", "-c", f"sleep 300 & echo $! > {gc_file}; exit 0"])
    assert p.wait() == 0
    left = _pid_file(gc_file)
    assert left is not None
    try:
        assert _alive(left), "the sleeper must outlive its parent shell for this test to mean anything"
        del p
        gc.collect()
        _wait_for(lambda: not _alive(left), "the leftover to be killed when the Popen was dropped")
    finally:
        _kill_leftovers(left)


def test_kill_session_reaches_every_group_of_the_session_and_refuses_its_own():
    p = guarded_popen(["sh", "-c", "sleep 300 & sleep 300 & wait"])
    try:
        _wait_for(lambda: len(child_guard.session_members(p.pid)) == 3, "the shell and its two sleepers")
        assert kill_session(p.pid, grace_s=3.0) == []
        p.wait()
        assert child_guard.session_members(p.pid) == []
    finally:
        _kill_leftovers(*child_guard.session_members(p.pid))
    me = os.getsid(0)
    assert kill_session(me, grace_s=0.0) == [] and _alive(os.getpid()), "the caller's own session is never killed"
    assert kill_session(1, grace_s=0.0) == [] and kill_session(0, grace_s=0.0) == []


@pytest.mark.parametrize("kw", ["start_new_session", "process_group", "preexec_fn"])
def test_the_guard_owns_the_childs_session(kw):
    with pytest.raises(TypeError, match="guarded_popen owns"):
        guarded_popen(["true"], **{kw: True})


# ------------------------------------------------------------------------------------ no raw trainer spawn left
_CALLS = {"Popen", "run", "call", "check_call", "check_output"}
#: What names the trainer or the launcher as a process to RUN (a script path, the split runpy name tests use so that
#: no argv carries the trainer's literal name, or the ``-m`` target): not an ``import main.launcher`` inside a probe.
_ENTRY_RE = re.compile(r"""train_rl_agent\.py|train_rl['"]\s*['"]_agent|['"]main\.train_rl_agent['"]|['"]main\.launcher['"]""")


def _assigned(scope: ast.AST) -> Dict[str, List[ast.AST]]:
    """``name -> the values assigned to it`` directly in ``scope`` (not in a nested function)."""
    out: Dict[str, List[ast.AST]] = {}
    stack = list(ast.iter_child_nodes(scope))
    while stack:
        n = stack.pop()
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            continue
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Name):
                    out.setdefault(t.id, []).append(n.value)
        stack.extend(ast.iter_child_nodes(n))
    return out


def _text(node: ast.AST, source: str, names: Dict[str, List[ast.AST]], depth: int = 3, seen: Optional[set] = None) -> str:
    """The source of ``node`` plus, to ``depth`` hops, the source of every name it uses (``argv = [...]; run(argv)``)."""
    seen = set() if seen is None else seen
    out = ast.get_source_segment(source, node) or ""
    if depth:
        for name in {x.id for x in ast.walk(node) if isinstance(x, ast.Name)} - seen:
            seen.add(name)
            for value in names.get(name, ()):
                out += "\n" + _text(value, source, names, depth - 1, seen)
    return out


def raw_trainer_spawns(source: str) -> List[int]:
    """The line of every plain ``subprocess.Popen/run/...`` whose argv (a name resolved within its function, then the
    module) names the trainer or the launcher as the process to run."""
    tree = ast.parse(source)
    module_names = _assigned(tree)
    found: List[int] = []

    def visit(scope: ast.AST, names: Dict[str, List[ast.AST]]) -> None:
        for node in ast.iter_child_nodes(scope):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                visit(node, {**module_names, **_assigned(node)})
                continue
            if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _CALLS
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "subprocess" and node.args
                    and _ENTRY_RE.search(_text(node.args[0], source, names))):
                found.append(node.lineno)
            visit(node, names)

    visit(tree, module_names)
    return sorted(found)


def test_the_scan_names_a_raw_trainer_spawn_and_only_that():
    assert raw_trainer_spawns('''
import subprocess, sys
_RUNNER = "import runpy; runpy.run_module('main.train_rl' '_agent', run_name='__main__')"
def a():
    subprocess.Popen([sys.executable, "-c", _RUNNER, "--debug"])
def b(tmp):
    argv = [sys.executable, str(tmp / "train_rl_agent.py"), "--debug"]
    return subprocess.run(argv, capture_output=True)
def c():
    subprocess.run([sys.executable, "-m", "main.launcher", "--dry-run"])
''') == [5, 8, 10]
    assert raw_trainer_spawns('''
import subprocess, sys
from utils.child_guard import guarded_popen
def a():
    guarded_popen([sys.executable, "-m", "main.launcher"])
def b():
    subprocess.run(["git", "status"])
def c():
    subprocess.run([sys.executable, "-c", "import main.launcher.child as c; print(c.__file__)"])
''') == []


def test_no_test_side_spawn_of_the_trainer_or_the_launcher_bypasses_the_guard():
    """Every test module, and ``main.b1.resume_check`` (a smoke driver a test runs), starts the trainer or the launcher
    through ``utils.child_guard``, never a plain ``subprocess`` call: one that a ``timeout`` or a kill of pytest
    leaves behind is the 2026-11-13 orphan. (A new site that needs the plain call is a new reason to read the module
    docstring, not an entry for a list.)"""
    root = src_root()
    files = sorted(root.rglob("*_test.py")) + [root / "main" / "b1" / "resume_check.py"]
    bad = [f"{f.relative_to(root)}:{line}" for f in files for line in raw_trainer_spawns(f.read_text())]
    assert bad == [], "plain subprocess call(s) that run the trainer or the launcher; use utils.child_guard: " + ", ".join(bad)
