"""``utils.gpu_lock`` against REAL holder processes — never the real GPU lock (every test points
``$GEN3AI_GPU_LOCK`` / ``path`` at a temp file).

The two load-bearing pins (each FAILS on revert of the re-entrancy / verification it guards):
* a nested take under a real holder PARENT does not block;
* a NON-ancestor holder still blocks — even when the waiter carries a forged marker naming it.

``integration``: spawns and waits on its own processes. ~5 s.
"""
from __future__ import annotations

import fcntl
import os
import subprocess
import sys
import textwrap
import time

import pytest

from utils import gpu_lock as G
from utils import procfs as P
from utils.paths import src_root

pytestmark = pytest.mark.integration


def _env(lock) -> dict:
    return {**os.environ, G.GPU_LOCK_ENV: str(lock),
            "PYTHONPATH": os.pathsep.join([str(src_root()), os.environ.get("PYTHONPATH", "")])}


def _py(code: str) -> list:
    return [sys.executable, "-c", textwrap.dedent(code)]


def _wait_for(pred, what: str, timeout: float = 15.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


def test_a_nested_take_under_a_real_holder_parent_does_not_block(tmp_path):
    lock = tmp_path / "gpu.lock"
    inner = _py("""
        import os
        from utils.gpu_lock import gpu_lock
        with gpu_lock(timeout_s=10) as h:
            print(h, os.getppid(), os.environ["GEN3AI_GPU_LOCK_HELD"])
    """)
    t0 = time.monotonic()
    r = subprocess.run([sys.executable, "-m", "utils.gpu_lock", "--", *inner], env=_env(lock),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    holder, parent, marker = r.stdout.split()
    assert holder == parent == marker, "the inner take must reuse its PARENT's hold, not take its own"
    assert time.monotonic() - t0 < 10, "the nested take waited — it re-acquired instead of re-entering"


def _holder(lock) -> subprocess.Popen:
    """A SIBLING holder: the helper CLI holding the lock around a long sleep."""
    h = subprocess.Popen([sys.executable, "-m", "utils.gpu_lock", "--", "sleep", "60"], env=_env(lock))
    key = None

    def held() -> bool:
        nonlocal key
        key = key or P.file_key(str(lock))
        return key is not None and P.holders(key) == [h.pid]

    _wait_for(held, "the sibling holder to take the lock")
    return h


def _reap(h: subprocess.Popen) -> None:
    procs = P.snapshot()
    for q in P.descendants(h.pid, P.children_map(procs)):
        try:
            os.kill(q, 9)
        except ProcessLookupError:
            pass
    h.kill()
    h.wait(timeout=10)


@pytest.mark.parametrize("forged", [False, True], ids=["no_marker", "forged_marker"])
def test_a_non_ancestor_holder_still_blocks(tmp_path, forged):
    lock = tmp_path / "gpu.lock"
    h = _holder(lock)
    try:
        env = _env(lock)
        if forged:
            env[G.HELD_ENV] = str(h.pid)          # a real holder — but NOT this waiter's ancestor
        w = subprocess.Popen(_py("""
            from utils.gpu_lock import gpu_lock
            with gpu_lock(wait=True, timeout_s=3, poll_s=0.1, report_every_s=0.5):
                print("ACQUIRED")
        """), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        key = P.file_key(str(lock))
        assert key is not None
        # the wait is a KERNEL-blocked flock, so /proc/locks (and the idle-waiter watchdog) sees it
        _wait_for(lambda: w.pid in P.waiters(key), "the waiter to be listed as blocked in /proc/locks")
        out, err = w.communicate(timeout=60)
        assert "ACQUIRED" not in out and w.returncode != 0, (out, err)
        assert "TimeoutError" in err and f"held by pid {h.pid}" in err, err
        assert "utils.gpu_lock" in err, "the periodic report names the holder's command line"
        assert ("STALE" in err) == forged, err
        assert P.holders(key) == [h.pid], "an abandoned wait must not leave the waiter holding anything"
    finally:
        _reap(h)


def test_a_bare_flock_ancestor_is_a_typed_self_deadlock_at_once(tmp_path):
    lock = tmp_path / "gpu.lock"
    lock.touch()
    t0 = time.monotonic()
    r = subprocess.run(["flock", str(lock), sys.executable, "-m", "utils.gpu_lock", "--report-every-s", "600",
                        "--", "true"], env=_env(lock), capture_output=True, text=True, timeout=60)
    assert r.returncode == 3, (r.returncode, r.stderr)
    assert "SELF-DEADLOCK" in r.stderr and "ANCESTOR" in r.stderr, r.stderr
    assert time.monotonic() - t0 < 10, "raised only after the report interval — must be at first contention"


def test_the_typed_error_names_both_pids(tmp_path):
    lock = tmp_path / "gpu.lock"
    lock.touch()
    r = subprocess.run(["flock", str(lock), *_py("""
        import os
        from utils.gpu_lock import gpu_lock, GpuLockSelfDeadlock
        try:
            with gpu_lock(timeout_s=10):
                print("ACQUIRED")
        except GpuLockSelfDeadlock as e:
            print(e.holder_pid, e.waiter_pid, os.getppid(), os.getpid())
    """)], env=_env(lock), capture_output=True, text=True, timeout=60)
    holder, waiter, ppid, me = r.stdout.split()
    assert (holder, waiter) == (ppid, me), r.stdout   # flock(1) forks the command: the holder is its parent


def test_a_stale_marker_naming_a_non_holding_ancestor_is_ignored(tmp_path, monkeypatch):
    lock = tmp_path / "gpu.lock"
    monkeypatch.setenv(G.HELD_ENV, str(os.getppid()))          # an ancestor, but it holds nothing
    logs: list = []
    with G.gpu_lock(lock, timeout_s=5, log=logs.append) as h:
        assert h == os.getpid()
        key = P.file_key(str(lock))
        assert key is not None and P.holders(key) == [os.getpid()]
        assert os.environ[G.HELD_ENV] == str(os.getpid())
    assert any("STALE" in m for m in logs), logs
    assert os.environ[G.HELD_ENV] == str(os.getppid()), "the caller's environment is restored on exit"


def test_in_process_nesting_is_reentrant_and_releases(tmp_path, monkeypatch):
    monkeypatch.delenv(G.HELD_ENV, raising=False)
    lock = tmp_path / "gpu.lock"
    with G.gpu_lock(lock, timeout_s=5) as a:
        with G.gpu_lock(lock, timeout_s=5) as b:
            assert a == b == os.getpid()
        assert G.verified_holder(lock) == os.getpid(), "the inner exit must not release the outer hold"
    assert G.HELD_ENV not in os.environ
    fd = os.open(str(lock), os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)            # free again
    finally:
        os.close(fd)


def test_the_cli_returns_the_commands_status_and_refuses_no_command(tmp_path):
    lock = tmp_path / "gpu.lock"
    r = subprocess.run([sys.executable, "-m", "utils.gpu_lock", "--", "bash", "-c", "exit 7"], env=_env(lock),
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 7, r.stderr
    for argv in ([], ["--"]):
        r = subprocess.run([sys.executable, "-m", "utils.gpu_lock", *argv], env=_env(lock),
                           capture_output=True, text=True, timeout=60)
        assert r.returncode == 2, (argv, r)

