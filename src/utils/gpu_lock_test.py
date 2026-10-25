"""``utils.gpu_lock`` against REAL holder processes — never the real GPU lock (every test points
``$GEN3AI_GPU_LOCK`` / ``path`` at a temp file).

The two load-bearing pins (each FAILS on revert of the re-entrancy / verification it guards):
* a nested take under a real holder PARENT does not block;
* a NON-ancestor holder still blocks — even when the waiter carries a forged marker naming it.

Since 2026-10-25 the command runs as the LEADER of a process group of its own (backlog: "a preempt must target the
right PROCESS GROUP"); the tests at the bottom pin each part of that, each against a revert: the group, its record
and ``--status`` line, a signal to the wrapper reaching a GRANDCHILD, a preempt by the recorded pgid (and nothing
else), the leftover warning, and the terminal behaviour (reads, ^C, ^Z) on a REAL pty with a shell-like parent.

``integration``: spawns and waits on its own processes. ~15 s.
"""
from __future__ import annotations

import fcntl
import os
import pty
import re
import select
import signal
import subprocess
import sys
import textwrap
import time

import pytest

from utils import gpu_lease as L
from utils import gpu_lock as G
from utils import procfs as P
from utils.paths import src_root

pytestmark = pytest.mark.integration


def _env(lock) -> dict:
    return {**os.environ, G.GPU_LOCK_ENV: str(lock),
            "PYTHONPATH": os.pathsep.join([str(src_root()), os.environ.get("PYTHONPATH", "")])}


def _py(code: str) -> list:
    return [sys.executable, "-c", textwrap.dedent(code)]


def _wait_for(pred, what: str, timeout: float = 15.0):
    """Poll ``pred`` until it is truthy; return that value. A timeout FAILS the test."""
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = pred()
        if v:
            return v
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



# ───────────────────────────────── the command's own PROCESS GROUP (2026-10-25) ─────────────────────────────────

def _alive(pid: int) -> bool:
    p = P.read_stat(pid)
    return p is not None and p.state not in ("Z", "X")


def _killpg(pgid: int) -> None:
    if pgid in (0, 1, os.getpgrp(), os.getsid(0)):          # a mutated implementation must never be able to kill the runner
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


_SLEEPER = "sleep 60 & echo $! > {f}; wait"


def test_the_command_leads_a_process_group_of_its_own_and_the_wrapper_logs_it(tmp_path):
    lock = tmp_path / "gpu.lock"
    r = subprocess.run([sys.executable, "-m", "utils.gpu_lock", "--", *_py(
        "import os; print(os.getpid(), os.getpgrp(), os.getppid())")], env=_env(lock),
        capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    pid, pgrp, wrapper = (int(t) for t in r.stdout.split())
    assert pgrp == pid, "the command must be the LEADER of its own process group"
    assert pgrp != os.getpgrp(), "...not a member of the launching tool's group (a preempt by that group hits the caller)"
    assert f"as process group {pid} (wrapper pid {wrapper})" in r.stderr, r.stderr
    assert f"kill -TERM -- -{pid}" in r.stderr and "never the wrapper's pid" in r.stderr


def test_status_names_the_job_group_and_a_preempt_by_that_group_stops_the_job_and_nothing_else(tmp_path):
    """The record (``<lock>.jobs/``) and ``--status`` carry the pgid; ``killpg(pgid)`` reaches the grandchild too, spares
    a bystander in the caller's own group, and leaves the wrapper to report 128+9; the record goes with the job."""
    lock, f = tmp_path / "gpu.lock", tmp_path / "gc.pid"
    bystander = subprocess.Popen(["sleep", "60"])                          # the TEST's group: a caller-group kill would hit it
    w = subprocess.Popen([sys.executable, "-m", "utils.gpu_lock", "--", "bash", "-c", _SLEEPER.format(f=f)],
                         env=_env(lock), stderr=subprocess.DEVNULL)
    pgid = None
    try:
        jobs = _wait_for(lambda: L.live_jobs(lock), "the job record")
        pgid = jobs[0]["pgid"]
        assert jobs[0]["wrapper_pid"] == w.pid and "sleep 60" in jobs[0]["cmd"]
        _wait_for(lambda: f.exists() and f.read_text().strip(), "the grandchild")
        grandchild = int(f.read_text())
        assert P.read_stat(grandchild).pgrp == pgid, "the grandchild stayed in the job's group"
        st = subprocess.run([sys.executable, "-m", "utils.gpu_lock", "--status"], env=_env(lock),
                            capture_output=True, text=True, timeout=60).stdout
        assert f"process group {pgid} (gpu_lock wrapper pid {w.pid}" in st and f"kill -TERM -- -{pgid}" in st, st
        assert "groups inside it" not in st, "bash and its sleep share the job's group: nothing is nested"
        os.killpg(pgid, signal.SIGKILL)                                    # THE PREEMPT
        assert w.wait(timeout=30) == 128 + signal.SIGKILL, "the wrapper survived and reported the job's death"
        _wait_for(lambda: not _alive(grandchild), "the grandchild to die with the group")
        assert _alive(bystander.pid), "a preempt by the job's pgid must not touch the caller's own group"
        assert L.live_jobs(lock) == [] and not list(L.jobs_dir(lock).glob("*.json")), "the record goes with the job"
    finally:
        if pgid:
            _killpg(pgid)
        bystander.kill()
        bystander.wait(timeout=10)
        w.kill()
        w.wait(timeout=10)


def test_status_lists_the_groups_nested_inside_a_job_which_a_sigkill_of_its_pgid_misses(tmp_path):
    """`timeout`, a nested gpu_lock.sh or a job-control shell inside the job make a group of their own: `kill -KILL --
    -<pgid>` leaves them running, so `--status` names them next to the pgid."""
    lock, f = tmp_path / "gpu.lock", tmp_path / "nested.pid"
    w = subprocess.Popen([sys.executable, "-m", "utils.gpu_lock", "--", "bash", "-c",
                          "set -m; " + _SLEEPER.format(f=f)], env=_env(lock), stderr=subprocess.DEVNULL)
    pgid = None
    try:
        pgid = _wait_for(lambda: L.live_jobs(lock), "the job record")[0]["pgid"]
        nested = int(_wait_for(lambda: f.exists() and f.read_text().strip(), "the nested sleep"))
        assert P.read_stat(nested).pgrp == nested != pgid, "`set -m` gave the sleep a group of its own"
        st = subprocess.run([sys.executable, "-m", "utils.gpu_lock", "--status"], env=_env(lock),
                            capture_output=True, text=True, timeout=60).stdout
        assert f"which a SIGKILL of -{pgid} misses: -{nested}" in st, st
    finally:
        for g in (pgid, locals().get("nested")):
            if g:
                _killpg(g)
        w.kill()
        w.wait(timeout=10)


def test_a_signal_to_the_wrapper_is_forwarded_to_the_whole_group_not_the_one_child(tmp_path):
    """Before: SIGTERM to the wrapper reached ONE child (bash); the sleep it started kept running, orphaned."""
    lock, f = tmp_path / "gpu.lock", tmp_path / "gc.pid"
    w = subprocess.Popen([sys.executable, "-m", "utils.gpu_lock", "--", "bash", "-c", _SLEEPER.format(f=f)],
                         env=_env(lock), stderr=subprocess.DEVNULL)
    grandchild = None
    try:
        _wait_for(lambda: f.exists() and f.read_text().strip(), "the grandchild")
        grandchild = int(f.read_text())
        w.send_signal(signal.SIGTERM)
        assert w.wait(timeout=30) == 128 + signal.SIGTERM
        _wait_for(lambda: not _alive(grandchild), "the GRANDCHILD to receive the forwarded SIGTERM", timeout=10)
    finally:
        if grandchild is not None and _alive(grandchild):
            os.kill(grandchild, signal.SIGKILL)                           # our own sleep, by explicit pid
        w.kill()
        w.wait(timeout=10)


def test_a_command_that_leaves_processes_in_its_group_is_said_so(tmp_path):
    lock, f = tmp_path / "gpu.lock", tmp_path / "left.pid"
    pgid = None
    try:
        r = subprocess.run([sys.executable, "-m", "utils.gpu_lock", "--", "bash", "-c", f"sleep 60 >/dev/null 2>&1 & echo $! > {f}"],
                           env=_env(lock), capture_output=True, text=True, timeout=60)
        pgid = int(re.search(r"as process group (\d+)", r.stderr).group(1))
        assert r.returncode == 0, r.stderr
        assert f"process group {pgid} still has 1 live process(es) (pids {f.read_text().strip()})" in r.stderr, r.stderr
        assert "the lock is released" in r.stderr
    finally:
        if pgid:
            _killpg(pgid)


# ── the terminal: a command on a tty behaves as a shell job (reads, ^C, ^Z) ─────────────────────────

_SHELL = r"""
import fcntl, os, signal, subprocess, sys, termios
rep, ctl = os.fdopen(int(sys.argv[1]), "w", buffering=1), os.fdopen(int(sys.argv[2]), "r")
cmd = sys.argv[3:]
fcntl.ioctl(0, termios.TIOCSCTTY, 0)                 # this session leader owns the pty: it plays the interactive shell
signal.signal(signal.SIGTTOU, signal.SIG_IGN)
def foreground_group():                              # what a job-control shell does for a foreground job
    os.setpgid(0, 0)
    os.tcsetpgrp(0, os.getpid())
    signal.signal(signal.SIGTTOU, signal.SIG_DFL)
job = subprocess.Popen(cmd, preexec_fn=foreground_group)
rep.write("started %d\n" % job.pid)
while True:
    _, st = os.waitpid(job.pid, os.WUNTRACED)
    if os.WIFSTOPPED(st):
        rep.write("stopped fg=%s\n" % ("wrapper" if os.tcgetpgrp(0) == job.pid else "other"))
        ctl.readline()                               # the test says when to `fg`
        os.tcsetpgrp(0, job.pid)
        os.killpg(job.pid, signal.SIGCONT)
        rep.write("continued\n")
        continue
    rep.write("exit %d\n" % os.waitstatus_to_exitcode(st))
    break
rep.write("tty-owner %s\n" % ("wrapper" if os.tcgetpgrp(0) == job.pid else "other"))
"""


class _Tty:
    """A pty with a shell-like parent (``_SHELL``) running ``gpu_lock -- <cmd>`` as its foreground job."""

    def __init__(self, lock, *cmd: str) -> None:
        self.master, slave = pty.openpty()
        rr, rw = os.pipe()
        cr, cw = os.pipe()
        self.lock, self.out, self.reports = lock, "", ""
        self.report_fd, self.ctl_fd = rr, cw
        self.shell = subprocess.Popen(
            [sys.executable, "-c", _SHELL, str(rw), str(cr), sys.executable, "-m", "utils.gpu_lock", "--", *cmd],
            env=_env(lock), stdin=slave, stdout=slave, stderr=slave, pass_fds=(rw, cr), start_new_session=True)
        os.close(slave)
        os.close(rw)
        os.close(cr)

    def _pump(self, until, what: str, timeout: float = 30.0) -> None:
        end = time.monotonic() + timeout
        while not until():
            left = end - time.monotonic()
            if left <= 0:
                raise AssertionError(f"timed out waiting for {what}\n  tty: {self.out!r}\n  reports: {self.reports!r}")
            ready, _, _ = select.select([self.master, self.report_fd], [], [], min(left, 0.2))
            for fd in ready:
                try:
                    data = os.read(fd, 4096).decode(errors="replace")
                except OSError:
                    data = ""
                if fd == self.master:
                    self.out += data
                else:
                    self.reports += data

    def wait_tty(self, text: str) -> None:
        self._pump(lambda: text in self.out, f"{text!r} on the terminal")

    def wait_report(self, text: str) -> None:
        self._pump(lambda: text in self.reports, f"{text!r} from the shell")

    def type(self, data: bytes) -> None:
        os.write(self.master, data)

    def fg(self) -> None:
        os.write(self.ctl_fd, b"go\n")

    def job_pgid(self) -> int:
        return _wait_for(lambda: L.live_jobs(self.lock), "the job record")[0]["pgid"]

    def close(self) -> None:
        for g in [j["pgid"] for j in L.live_jobs(self.lock)] + [self.shell.pid]:
            _killpg(g)
        wrapper = re.search(r"started (\d+)", self.reports)
        if wrapper:
            _killpg(int(wrapper.group(1)))
        self.shell.kill()
        self.shell.wait(timeout=10)
        for fd in (self.master, self.report_fd, self.ctl_fd):
            os.close(fd)


def test_a_command_on_a_terminal_reads_stdin_and_owns_the_terminal_like_a_shell_job(tmp_path):
    """The command's group takes the terminal over: its read does not stop on SIGTTIN, and the terminal is handed
    back to the wrapper's group when it ends. (Without the hand-over the read stops; without the hand-back the
    shell's terminal stays owned by a dead group.)"""
    t = _Tty(tmp_path / "gpu.lock", *_py("""
        import os
        print("rea" + "dy", os.getpgrp(), os.tcgetpgrp(0), flush=True)      # not spelled in the wrapper's log line
        print("got=" + input(), flush=True)
    """))
    try:
        t.wait_tty("ready")
        t.type(b"hello\n")
        t.wait_tty("got=hello")
        t.wait_report("tty-owner")
        m = re.search(r"ready (\d+) (\d+)", t.out)
        job_pgrp, fg_pgrp = int(m.group(1)), int(m.group(2))
        wrapper = int(re.search(r"started (\d+)", t.reports).group(1))
        assert job_pgrp == fg_pgrp, "the command's group is the terminal's FOREGROUND group while it runs"
        assert job_pgrp != wrapper, "...and it is a group of its own, not the wrapper's"
        assert "exit 0" in t.reports and "tty-owner wrapper" in t.reports, t.reports
    finally:
        t.close()


def test_control_c_on_a_terminal_reaches_the_command_alone_and_ends_the_job(tmp_path):
    t = _Tty(tmp_path / "gpu.lock", "bash", "-c", "echo READY$((1+1)); sleep 60")
    try:
        t.wait_tty("READY2")
        t.job_pgid()
        t.type(b"\x03")
        t.wait_report("tty-owner")
        assert f"exit {128 + signal.SIGINT}" in t.reports, t.reports
        assert "tty-owner wrapper" in t.reports
        assert "Traceback" not in t.out, "the wrapper must not die of the terminal's SIGINT"
    finally:
        t.close()


def test_control_z_on_a_terminal_stops_the_wrapper_too_and_fg_continues_the_job(tmp_path):
    """The shell sees ITS job (the wrapper) stopped, holds the terminal, and a `fg` hands it to the command again.
    (A wrapper that did not stop would leave the shell waiting on a stopped job it cannot see.)"""
    t = _Tty(tmp_path / "gpu.lock", "bash", "-c", "echo READY$((1+1)); sleep 60")
    try:
        t.wait_tty("READY2")
        pgid = t.job_pgid()
        t.type(b"\x1a")
        t.wait_report("stopped fg=wrapper")                                # the wrapper stopped AND took the terminal back
        _wait_for(lambda: P.read_stat(pgid).state == "T", "the job to be stopped")
        t.fg()
        t.wait_report("continued")
        _wait_for(lambda: P.read_stat(pgid).state != "T", "the job to continue")
        t.type(b"\x03")                                                   # ^C only works if the job owns the terminal again
        t.wait_report("tty-owner")
        assert f"exit {128 + signal.SIGINT}" in t.reports and "tty-owner wrapper" in t.reports, t.reports
    finally:
        t.close()


def test_a_stop_for_the_terminal_before_the_hand_over_is_continued_not_left_stopped(tmp_path):
    """A command that reads or writes the terminal in the instant before the hand-over is stopped by SIGTTIN; the wrapper
    sees that stop only after it handed the terminal over, so it continues the group (a plain wait would hang forever)."""
    t = _Tty(tmp_path / "gpu.lock", "bash", "-c", "kill -TTIN $$; echo resumed-after-ttin")
    try:
        t.wait_tty("resumed-after-ttin")
        t.wait_report("tty-owner")
        assert "exit 0" in t.reports and "stopped" not in t.reports, "the stop is the wrapper's to handle, not the shell's"
    finally:
        t.close()
