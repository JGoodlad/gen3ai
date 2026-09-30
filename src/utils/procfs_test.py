"""``utils.procfs`` on SYNTHETIC input: the ``/proc/locks`` grammar and the tree arithmetic.
(The real-process behaviour is pinned by ``main/ops/idle_waiter_watchdog_test.py`` and
``utils/gpu_lock_test.py``.)"""
from __future__ import annotations

import os

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
