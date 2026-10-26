"""Regression tests for /proc identity, bounded waits and the identity-gated signal (procwatch.py)."""
import inspect
import re
import signal
import subprocess
import time

import pytest

from main.ops.arm_chain import procwatch as pw
from utils import procfs

pytestmark = pytest.mark.integration


def sleeper(secs="30"):
    return subprocess.Popen(["sleep", secs], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL)


def reap(p):
    """Kill by explicit PID and reap; safe to call twice."""
    if p.poll() is None:
        p.kill()
    p.wait(timeout=10)


def test_alive_on_live_child():
    """Baseline: one /proc read identifies a live child by (pid, start ticks)."""
    p = sleeper()
    try:
        ref = pw.ref_of(p.pid)
        assert ref is not None and ref.pid == p.pid
        assert pw.alive(ref)
    finally:
        reap(p)


def test_wrong_start_ticks_reads_not_alive():
    """CB-23: the right pid with the wrong start ticks is a RE-USED pid, never the old process alive."""
    p = sleeper()
    try:
        ref = pw.ref_of(p.pid)
        assert pw.alive(pw.ProcRef(ref.pid, ref.start_ticks + 1)) is False
        assert pw.alive(ref) is True
    finally:
        reap(p)


def test_zombie_reads_not_alive():
    """A killed-but-unreaped child (state Z) is gone: a bare pid-exists check read it as alive."""
    p = sleeper()
    try:
        ref = pw.ref_of(p.pid)
        p.kill()                                    # do NOT wait: it stays a zombie
        deadline = time.monotonic() + 5
        st = procfs.read_stat(p.pid)
        while st is not None and st.state != "Z" and time.monotonic() < deadline:
            time.sleep(0.01)
            st = procfs.read_stat(p.pid)
        assert st is not None and st.state == "Z"
        assert pw.alive(ref) is False
        assert pw.ref_of(p.pid) is None
    finally:
        reap(p)


def test_wait_gone_bounded_false_then_true():
    """Every wait is bounded: False at a short timeout on a live process, True once it is killed."""
    p = sleeper()
    try:
        ref = pw.ref_of(p.pid)
        assert pw.wait_gone(ref, timeout_s=0.15, poll_s=0.02) is False
        p.kill()
        p.wait(timeout=10)
        assert pw.wait_gone(ref, timeout_s=5, poll_s=0.02) is True
    finally:
        reap(p)


def test_signal_if_refuses_cmdline_mismatch():
    """CB-27 (kill_if): a matching start ticks but a different cmdline is NOT signalled."""
    p = sleeper()
    try:
        ref = pw.ref_of(p.pid)
        sent, why = pw.signal_if(ref, r"definitely_not_this_cmdline_xyz", signal.SIGKILL)
        assert sent is False and "MISMATCH" in why
        assert p.poll() is None and pw.alive(ref)
    finally:
        reap(p)


def test_signal_if_refuses_gone_process():
    """CB-27: a process that already exited (or a re-used pid) is NOT signalled."""
    p = sleeper()
    ref = pw.ref_of(p.pid)
    reap(p)
    sent, why = pw.signal_if(ref, "sleep", signal.SIGKILL)
    assert sent is False and "gone" in why
    sent, why = pw.signal_if(pw.ProcRef(999999999, 1), "sleep", signal.SIGKILL)
    assert sent is False


def test_signal_if_sends_on_identity_match():
    """CB-27: start ticks and cmdline both match, so the signal is sent and the child dies of it."""
    p = sleeper()
    try:
        ref = pw.ref_of(p.pid)
        sent, why = pw.signal_if(ref, r"^sleep 30$", signal.SIGKILL)
        assert sent is True and "identity OK" in why
        assert p.wait(timeout=10) == -signal.SIGKILL
    finally:
        reap(p)


def test_ref_from_pidfile(tmp_path):
    """A pid file that names a live process yields its identity; a stale or garbage one yields None."""
    p = sleeper()
    try:
        f = tmp_path / "pid"
        f.write_text(f"{p.pid}\n")
        assert pw.ref_from_pidfile(str(f)) == pw.ref_of(p.pid)
        f.write_text("not-a-pid")
        assert pw.ref_from_pidfile(str(f)) is None
        assert pw.ref_from_pidfile(str(tmp_path / "absent")) is None
    finally:
        reap(p)
    (tmp_path / "pid").write_text(f"{p.pid}\n")
    assert pw.ref_from_pidfile(str(tmp_path / "pid")) is None


def test_module_never_probes_liveness_with_kill_zero():
    """CB-26: kill -0 read a LIVE supervisor as dead in a sandboxed agent shell; the module reads /proc only."""
    src = inspect.getsource(pw)
    code = "\n".join(re.sub(r"#.*", "", ln) for ln in src.splitlines())
    assert not re.search(r"os\.kill\s*\(", code)
    assert not re.search(r"\bkill\s*\(\s*[^,()]+,\s*0\s*\)", code)
    assert not re.search(r"\bsignal\s*\(\s*0\s*\)", code)
