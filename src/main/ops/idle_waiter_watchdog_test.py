"""``scripts/ops/idle_waiter_watchdog.py`` against REAL processes.

Each test builds a small process tree under a FAKE Claude session — a symlink named ``claude`` to
bash, so the kernel's comm for it reads ``claude`` exactly as a real session's does — and scans it
twice, a few seconds apart, with the age / interval bars lowered and the idle bar raised to 5% so a
fork-per-second loop cannot read as busy on a loaded box. The scans are SCOPED to this test process's
subtree, so the live box's own waiters never enter an assertion.

``integration``: the tests spawn and kill their own processes (by explicit PID). ~10 s in all.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

import pytest

from utils import procfs as P
from utils.contention import scale_timeout
from utils.paths import repo_path

pytestmark = pytest.mark.integration

_SCRIPT = repo_path("scripts", "ops", "idle_waiter_watchdog.py")
_spec = importlib.util.spec_from_file_location("idle_waiter_watchdog", _SCRIPT)
assert _spec is not None and _spec.loader is not None
W = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = W          # @dataclass resolves its module through sys.modules
_spec.loader.exec_module(W)

GAP_S = 3.0
KW = dict(min_age_s=0.0, min_interval_s=1.0, idle_frac=0.05)


def _tree(root: int) -> Dict[int, str]:
    procs = P.snapshot()
    return {q: P.cmdline(q) for q in [root, *P.descendants(root, P.children_map(procs))]}


def _wait_for(pred: Callable[[], bool], what: str, timeout: float = 15.0) -> None:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return
        time.sleep(0.1)
    raise AssertionError(f"timed out waiting for {what}")


def _reap(p: subprocess.Popen) -> None:
    """Kill the tree under ``p`` BY EXPLICIT PID (descendants first), then ``p``."""
    procs = P.snapshot()
    for q in reversed(P.descendants(p.pid, P.children_map(procs))):
        try:
            os.kill(q, signal.SIGKILL)
        except ProcessLookupError:
            pass
    p.kill()
    p.wait(timeout=10)


@pytest.fixture
def session(tmp_path):
    """Spawn ``script`` under a fake ``claude`` session; every tree is reaped at teardown."""
    link = tmp_path / "claude"
    link.symlink_to(shutil.which("bash") or "/bin/bash")
    spawned: List[subprocess.Popen] = []

    def spawn(script: str) -> subprocess.Popen:
        p = subprocess.Popen([str(link), "-c", script + "\nwait"], stdin=subprocess.DEVNULL,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        spawned.append(p)
        _wait_for(lambda: P.read_stat(p.pid) is not None and P.read_stat(p.pid).comm == "claude",
                  "the fake session's comm to read 'claude'")
        return p

    yield spawn
    for p in spawned:
        _reap(p)


def _pid_of(root: int, needle: str, comm: str = "", exact: bool = False) -> int:
    """The descendant of ``root`` whose command line contains ``needle`` — waited for, since the
    fake session forks its children asynchronously."""
    found: List[int] = []

    def look() -> bool:
        for q, c in _tree(root).items():
            st = P.read_stat(q)
            if q != root and (c == needle if exact else needle in c) and st is not None and (not comm or st.comm == comm):
                found.append(q)
                return True
        return False

    _wait_for(look, f"a process under {root} matching {needle!r}")
    return found[0]


def _has_sleep_child(pid: int) -> bool:
    procs = P.snapshot()
    return any(procs[c].comm == "sleep" for c in P.children_map(procs).get(pid, ()) if c in procs)


def _two_scans(**kw):
    first = W.Scan(None, scope_pid=os.getpid(), **{**KW, **kw})
    f1 = first.run()
    time.sleep(GAP_S)
    second = W.Scan(first.next_state(), scope_pid=os.getpid(), **{**KW, **kw})
    return f1, second.run()


def _flagged(findings, pid: int):
    return [f for f in findings if f.pid == pid]


def test_idle_loops_are_flagged_and_healthy_ones_are_not(session, tmp_path):
    never = tmp_path / "never_written.txt"
    tag = f"iww_gone_{os.getpid()}_{time.monotonic_ns()}"
    root = session(
        f"{sys.executable} -c 'while True: pass' &\n"
        "SP=$!\n"
        'bash -c "while kill -0 $SP 2>/dev/null; do sleep 1; done" &\n'                 # waits on a BUSY pid
        f"bash -c 'while pgrep -f {tag} >/dev/null; do sleep 1; done' &\n"             # self-match: target gone
        f"bash -c 'until [ -s {never} ]; do sleep 1; done' &\n"                         # waits on a file nobody writes
        "bash -c 'sleep 120; echo done' &\n")                                           # ONE deliberate long sleep
    busy_waiter = _pid_of(root.pid, "kill -0")
    selfmatch = _pid_of(root.pid, f"pgrep -f {tag}")
    filewait = _pid_of(root.pid, str(never))
    long_sleep = _pid_of(root.pid, "sleep 120; echo done")
    spinner = _pid_of(root.pid, "while True: pass")
    for w in (busy_waiter, selfmatch, filewait, long_sleep):
        _wait_for(lambda w=w: _has_sleep_child(w), f"pid {w}'s sleep child")

    first, second = _two_scans()
    assert first == [], "a first run has no baseline and must judge nothing"
    assert len(_flagged(second, selfmatch)) == 1, second
    assert "self-match" in _flagged(second, selfmatch)[0].reason
    assert len(_flagged(second, filewait)) == 1, second
    assert str(never) in _flagged(second, filewait)[0].reason
    assert _flagged(second, filewait)[0].session == root.pid
    for healthy in (busy_waiter, long_sleep, spinner, root.pid, os.getpid()):
        assert not _flagged(second, healthy), (healthy, second)


def _child_comm_is(pid: int, comm: str) -> bool:
    procs = P.snapshot()
    kids = [c for c in P.children_map(procs).get(pid, ()) if c in procs]
    return [procs[c].comm for c in kids] == [comm] and procs[kids[0]].state == "S"


def test_a_poll_loop_caught_mid_condition_is_still_a_waiter(session, tmp_path):
    """The race behind the 3-in-8 flake of the test above: a scan that lands while the loop runs
    its CONDITION (there, ``pgrep``) sees no ``sleep`` child. Here the condition blocks on a FIFO
    nobody opens, so BOTH scans land mid-condition every time — deterministic, not a timing race."""
    fifo = tmp_path / "nobody_writes.fifo"
    os.mkfifo(fifo)
    root = session(f"bash -c 'until cat {fifo} >/dev/null; do sleep 1; done' &")
    loop = _pid_of(root.pid, f"until cat {fifo}", comm="bash")
    _wait_for(lambda: _child_comm_is(loop, "cat"), f"pid {loop}'s cat child to block on the fifo")

    first, second = _two_scans()
    assert first == []
    assert _child_comm_is(loop, "cat"), "the premise: no sleep child at either scan"
    hit = _flagged(second, loop)
    assert len(hit) == 1, second
    assert hit[0].reason.startswith("sleep-loop") and str(fifo) in hit[0].reason


def test_a_script_loop_seen_sleeping_last_run_is_still_a_waiter(session, tmp_path):
    """A loop in a script FILE has no loop text on its command line; mid-condition it is known
    only by the previous run's record of its sleep child."""
    fifo = tmp_path / "nobody_writes.fifo"
    os.mkfifo(fifo)
    script = tmp_path / "poll.sh"
    script.write_text(f"until cat {fifo} >/dev/null; do sleep 1; done\n")
    script.chmod(0o755)                          # a program being run, not a file waited on
    old = time.time() - 3600
    os.utime(fifo, (old, old))                   # untouched since before the synthetic baseline
    root = session(f"bash {script} &")
    loop = _pid_of(root.pid, f"bash {script}", comm="bash", exact=True)
    _wait_for(lambda: _child_comm_is(loop, "cat"), f"pid {loop}'s cat child to block on the fifo")
    s0 = W.Scan(None, scope_pid=os.getpid(), **KW)
    key = s0.procs[loop].key
    assert s0.waiter_kind(loop) == "", "the premise: nothing in THIS snapshot marks it a loop"
    # the previous run: 60 s ago, the same subtree CPU (idle), and a sleep child it no longer has
    stale = {"version": 1, "t": time.time() - 60, "cpu": dict(s0.next_state()["cpu"]), "sleep": {key: "1:1"}}
    hit = _flagged(W.Scan(stale, scope_pid=os.getpid(), **KW).run(), loop)
    assert len(hit) == 1 and hit[0].reason.startswith("sleep-loop"), hit
    # ...and with no such record it is not a waiter at all (the shape is unknowable from one snapshot)
    assert not _flagged(W.Scan({**stale, "sleep": {}}, scope_pid=os.getpid(), **KW).run(), loop)
    # ...nor with only the "" marker (a loop known WITHOUT an observed sleep): the memory lasts one
    # run unless a sleep is seen again, so a shell that once slept is not a waiter forever
    assert not _flagged(W.Scan({**stale, "sleep": {key: ""}}, scope_pid=os.getpid(), **KW).run(), loop)


def test_self_deadlock_is_reported_on_the_first_run(session, tmp_path):
    lock = tmp_path / "held_by_parent.lock"
    lock.touch()
    root = session(f"flock {lock} bash -c 'flock {lock} true' &")
    outer = _pid_of(root.pid, f"flock {lock} bash")
    inner = _pid_of(root.pid, f"flock {lock} true", comm="flock", exact=True)
    key = P.file_key(str(lock))
    assert key is not None
    _wait_for(lambda: inner in P.waiters(key), "the inner flock to block")
    assert P.holders(key) == [outer]

    findings = W.Scan(None, scope_pid=os.getpid(), **KW).run()
    dead = [f for f in findings if f.kind == "SELF-DEADLOCK"]
    assert [f.pid for f in dead] == [inner], findings
    assert f"pid {outer}" in dead[0].reason and "OWN ancestor" in dead[0].reason
    assert str(lock) in dead[0].reason
    # a second run: the holder's idle subtree is the SAME stall, reported once (as the deadlock)
    s1 = W.Scan(None, scope_pid=os.getpid(), **KW)
    s1.run()
    time.sleep(1.5)
    again = W.Scan(s1.next_state(), scope_pid=os.getpid(), **KW).run()
    assert [(f.kind, f.pid) for f in again] == [("SELF-DEADLOCK", inner)], again

    # the CLI: one line, exit 1 — and nothing about itself
    state = tmp_path / "state.json"
    r = subprocess.run([sys.executable, str(_SCRIPT), "--state", str(state), "--scope-pid", str(os.getpid())],
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 1, r
    lines = r.stdout.splitlines()
    assert len(lines) == 1 and "SELF-DEADLOCK" in lines[0] and f"pid={inner}" in lines[0], r.stdout
    assert state.is_file()


def _holder(lock: Path, busy: bool) -> subprocess.Popen:
    body = "while True: pass" if busy else "time.sleep(600)"
    p = subprocess.Popen([sys.executable, "-c", f"import fcntl, time\nf = open({str(lock)!r}, 'w')\n"
                          f"fcntl.flock(f, fcntl.LOCK_EX)\nprint('held', flush=True)\n{body}"],
                         stdout=subprocess.PIPE, text=True)
    assert p.stdout is not None and p.stdout.readline().strip() == "held"
    return p


@pytest.mark.parametrize("busy", [True, False], ids=["busy_holder", "idle_holder"])
def test_a_lock_waiter_is_judged_by_its_non_ancestor_holder(session, tmp_path, busy):
    lock = tmp_path / "other.lock"
    holder = _holder(lock, busy)
    try:
        root = session(f"flock {lock} true &")
        waiter = _pid_of(root.pid, f"flock {lock} true", comm="flock", exact=True)
        key = P.file_key(str(lock))
        assert key is not None
        _wait_for(lambda: waiter in P.waiters(key), "the flock to block")
        first, second = _two_scans()
        assert not [f for f in first + second if f.kind == "SELF-DEADLOCK"], "a sibling holder is not a deadlock"
        if busy:
            assert not _flagged(second, waiter), second
        else:
            assert len(_flagged(second, waiter)) == 1, second
            assert f"held by pid {holder.pid}" in _flagged(second, waiter)[0].reason
    finally:
        holder.kill()
        holder.wait(timeout=10)


def test_a_reused_pid_is_a_new_process(session):
    root = session("bash -c 'while true; do sleep 1; done' &")
    loop = _pid_of(root.pid, "while true")
    _wait_for(lambda: _has_sleep_child(loop), "the loop's sleep child")
    s0 = W.Scan(None, scope_pid=os.getpid(), **KW)
    start = s0.procs[loop].starttime
    stale = {"version": 1, "t": time.time() - 60, "cpu": {f"{loop}:{start + 1}": 0}, "sleep": {}}
    s = W.Scan(stale, scope_pid=os.getpid(), **KW)
    assert s.advanced(loop) is None, "a (pid, starttime) mismatch must read as NO baseline"
    assert not _flagged(s.run(), loop)


def test_quiet_when_nothing_is_waiting(tmp_path):
    state = tmp_path / "s.json"
    for _ in range(2):
        r = subprocess.run([sys.executable, str(_SCRIPT), "--state", str(state), "--scope-pid", str(os.getpid()),
                            "--min-age-s", "0", "--min-interval-s", "0"], capture_output=True, text=True, timeout=60)
        assert (r.returncode, r.stdout) == (0, ""), r


def test_one_scan_of_the_whole_box_is_cheap(tmp_path):
    t0 = time.perf_counter()
    W.Scan(None).run()
    # 1 s on an idle box (measured ~0.02 s over ~400 processes), stretched by measured contention
    assert time.perf_counter() - t0 < scale_timeout(1.0)


def test_duplicate_waiters_are_counted_on_the_first_run_and_never_killed(session, tmp_path):
    """2026-09-30, Lane K: FIVE identical `until grep -q … <log>` loops in one session — each Bash-tool
    wait backgrounded at its 600 s timeout kept looping, each retry added one — and none was IDLE,
    because the target was legitimately waiting. Revert `duplicate_waiters` and this fails.

    The three loops differ ONLY in the Bash tool's per-call cwd-file suffix (never reached: the loop
    never ends), so normalisation is under test too; a fourth, different loop is not a duplicate."""
    never = tmp_path / "never_written.log"
    fifo = tmp_path / "nobody_writes.fifo"
    os.mkfifo(fifo)
    loop = f"until grep -q DONE {never} 2>/dev/null; do sleep 1; done"
    root = session(
        "".join(f"bash -c '{loop} && pwd -P >| /tmp/claude-{h}-cwd' &\n" for h in ("a1b2", "c3d4", "e5f6"))
        + f"bash -c 'until [ -s {never}.other ]; do sleep 1; done' &\n"
        # ONE loop whose $(...) condition forks a subshell with the SAME command line: not a duplicate
        + f"bash -c 'until [ -n \"$(cat {fifo}; true)\" ]; do sleep 1; done' &\n")
    _wait_for(lambda: len([q for q, c in _tree(root.pid).items() if c.startswith(f"bash -c {loop}")]) == 3,
              "the three identical loops")
    dups = sorted(q for q, c in _tree(root.pid).items() if c.startswith(f"bash -c {loop}"))
    odd = _pid_of(root.pid, f"{never}.other", comm="bash")
    forking = _pid_of(root.pid, f"cat {fifo}; true", comm="bash")
    _wait_for(lambda: any(P.cmdline(c) == P.cmdline(forking)
                          for c in P.children_map(P.snapshot()).get(forking, ())),
              "the $(...) subshell that shares its parent's command line")

    findings = W.Scan(None, scope_pid=os.getpid(), dup_min_age_s=0.0, **KW).run()
    dup = [f for f in findings if f.kind == "DUPLICATE-WAITER"]
    assert len(dup) == 1, findings
    assert dup[0].pid in dups and dup[0].session == root.pid
    assert dup[0].reason.startswith("3 identical waiters")
    for q in dups:
        assert str(q) in dup[0].reason, (q, dup[0].reason)
    assert str(odd) not in dup[0].reason and str(forking) not in dup[0].reason
    for q in (*dups, odd, forking):
        assert P.read_stat(q) is not None, f"pid {q} is gone — the watchdog must never kill"

    # below the age bar nothing is flagged (a wait being replaced must not flap)
    assert not [f for f in W.Scan(None, scope_pid=os.getpid(), dup_min_age_s=3600.0, **KW).run()
                if f.kind == "DUPLICATE-WAITER"]
    # the CLI prints it on a first run and exits 1
    r = subprocess.run([sys.executable, str(_SCRIPT), "--state", str(tmp_path / "s.json"), "--scope-pid",
                        str(os.getpid()), "--dup-min-age-s", "0"], capture_output=True, text=True, timeout=60)
    assert r.returncode == 1 and "DUPLICATE-WAITER" in r.stdout, r


def test_normalise_cmd_erases_only_the_per_call_cwd_file():
    a = W.normalise_cmd("bash -c  until x; do sleep 1; done && pwd -P >| /tmp/claude-dd38-cwd")
    b = W.normalise_cmd("bash -c until x; do sleep 1; done && pwd -P >| /tmp/claude-0f1e-cwd ")
    assert a == b
    assert W.normalise_cmd("sleep 1") != W.normalise_cmd("sleep 2")
