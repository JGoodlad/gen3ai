"""``utils.procfs`` on SYNTHETIC input: the ``/proc/locks`` grammar and the tree arithmetic.
(The real-process behaviour is pinned by ``main/ops/idle_waiter_watchdog_test.py`` and
``utils/gpu_lock_test.py``.)"""
from __future__ import annotations

import os
import subprocess
import time

from utils import procfs as P

_LOCKS = """\
1: FLOCK  ADVISORY  WRITE 9006 00:58:86 0 EOF
1: -> FLOCK  ADVISORY  WRITE 9100 00:58:86 0 EOF
2: POSIX  ADVISORY  READ 7568 103:06:4829 0 0
3: OFDLCK ADVISORY  WRITE -1 00:1d:4830 0 EOF
garbage line
"""


def test_proc_locks_grammar(tmp_path):
    f = tmp_path / "locks"
    f.write_text(_LOCKS)
    locks = P.read_locks(str(f))
    assert [(e.kind, e.mode, e.pid, e.file, e.blocked) for e in locks] == [
        ("FLOCK", "WRITE", 9006, "00:58:86", False),
        ("FLOCK", "WRITE", 9100, "00:58:86", True),
        ("POSIX", "READ", 7568, "103:06:4829", False),
        ("OFDLCK", "WRITE", -1, "00:1d:4830", False),
    ]
    assert P.holders("00:58:86", locks) == [9006]
    assert P.waiters("00:58:86", locks) == [9100]
    assert P.holders("00:1d:4830", locks) == [], "an OFD lock's pid (-1) names no holder"
    assert locks[2].inode == 4829


def test_file_key_matches_the_kernel_format(tmp_path):
    f = tmp_path / "x"
    f.touch()
    st = os.stat(f)
    assert P.file_key(str(f)) == f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}"
    assert P.file_key(str(tmp_path / "absent")) is None


def _p(pid, ppid, cpu):
    return P.Proc(pid=pid, comm="x", state="S", ppid=ppid, utime=cpu, stime=0, cutime=0, cstime=0, starttime=pid)


def test_ancestry_and_subtree_cpu():
    procs = {1: _p(1, 0, 1), 2: _p(2, 1, 10), 3: _p(3, 2, 100), 4: _p(4, 2, 1000), 5: _p(5, 1, 5)}
    kids = P.children_map(procs)
    assert P.ancestors(4, procs) == [2, 1]
    assert sorted(P.descendants(2, kids)) == [3, 4]
    assert P.subtree_cpu(procs, kids) == {1: 1116, 2: 1110, 3: 100, 4: 1000, 5: 5}


def test_this_process_reads_back():
    me = P.read_stat(os.getpid())
    assert me is not None and me.ppid == os.getppid() and me.key == f"{os.getpid()}:{me.starttime}"
    assert os.getppid() in P.ancestors(os.getpid())
    assert me.pgrp == os.getpgrp() and me.session == os.getsid(0), "the group and session ids are parsed"


def test_pid_alive_is_one_atomic_read_and_every_failure_mode_is_not_alive(tmp_path, monkeypatch):
    assert P.pid_alive(os.getpid()) is True
    assert P.pid_alive(2 ** 22 + 12345) is False, "an absent pid (FileNotFoundError) is not alive"
    child = subprocess.Popen(["true"])
    try:
        end = time.monotonic() + 30
        while P.pid_alive(child.pid) and time.monotonic() < end:     # exited, NOT reaped: a zombie
            time.sleep(0.01)
        st = P.read_stat(child.pid)
        assert st is not None and st.state == "Z", "precondition: the unreaped exited child is a zombie"
        assert P.pid_alive(child.pid) is False, "a zombie is not alive"
    finally:
        child.wait()
    assert P.pid_alive(child.pid) is False, "reaped: gone"

    def _vanishes_mid_read(*a, **k):                                  # the process exits between open and read
        raise ProcessLookupError(3, "No such process")
    monkeypatch.setattr("builtins.open", _vanishes_mid_read)
    assert P.pid_alive(os.getpid()) is False
    monkeypatch.undo()
    (tmp_path / "7").mkdir()
    (tmp_path / "7" / "stat").write_text("7 (junk")                   # unparseable
    assert P.pid_alive(7, proc=str(tmp_path)) is False
    (tmp_path / "7" / "stat").write_text("7 (a) X 1 1 1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 5 0")
    assert P.pid_alive(7, proc=str(tmp_path)) is False, "state X (dead) is not alive"


def test_holds_lock_reads_the_process_own_fdinfo_never_the_listing(tmp_path, monkeypatch):
    """``/proc/locks`` can skip a live entry (``read_locks``); a known pid is decided from its fdinfo. Synthetic
    tree: the lock line must be a HELD lock on that file taken by that pid; a real flock is read back too."""
    d = tmp_path / "77" / "fdinfo"
    d.mkdir(parents=True)
    (d / "3").write_text("pos:\t0\nflags:\t02100002\nmnt_id:\t29\nino:\t86\n"
                         "lock:\t1: FLOCK  ADVISORY  WRITE 77 00:58:86 0 EOF\n")
    (d / "4").write_text("pos:\t0\nflags:\t02\nmnt_id:\t29\nino:\t90\n"
                         "lock:\t1: FLOCK  ADVISORY  WRITE 55 00:58:90 0 EOF\n")      # inherited: another pid took it
    (d / "5").write_text("pos:\t0\nflags:\t02\nmnt_id:\t29\nino:\t91\n")
    proc = str(tmp_path)
    assert P.holds_lock(77, "00:58:86", proc)
    assert not P.holds_lock(77, "00:58:90", proc), "a lock another pid took through a shared descriptor"
    assert not P.holds_lock(77, "00:58:91", proc) and not P.holds_lock(78, "00:58:86", proc)

    import fcntl
    f = tmp_path / "real.lock"
    fd = os.open(f, os.O_RDWR | os.O_CREAT)
    key = P.file_key(str(f))
    assert key is not None and not P.holds_lock(os.getpid(), key)
    fcntl.flock(fd, fcntl.LOCK_EX)
    monkeypatch.setattr(P, "read_locks", lambda *a, **k: [])           # the listing skipping it changes nothing
    assert P.holds_lock(os.getpid(), key) and P.holders(key) == []
    os.close(fd)
    assert not P.holds_lock(os.getpid(), key)
