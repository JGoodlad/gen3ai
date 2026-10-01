"""Unit tests for the orphan watchdog (the --debug smoke-run zombie guard).

`start_orphan_watchdog` exits the process when its parent dies and it gets
reparented. The exit path is `os._exit`, which can't be observed in-process,
so the orphan scenario is driven through real subprocesses: a short-lived
"parent" spawns a "child" that arms the watchdog, then the parent exits — the
child is reparented and must self-exit. No Showdown server / bridge needed.

Two things the first version got wrong, both fixed here (2026-09-30):

* **A RACE made the orphan test VACUOUS.** The parent exited the instant it had spawned the child, so
  the child usually captured its `original_ppid` AFTER the reparent — the subreaper's pid — saw no
  change, and never fired. The test still passed: the child also held the test's captured stdout pipe,
  so `subprocess.run` waited out the child's whole 15 s natural lifetime and the poll then found it
  gone. Now the parent waits until the child has ARMED the watchdog (a ready file written after
  `start_orphan_watchdog` returns), the child's stdout goes to a log instead of our pipe, and the test
  asserts the watchdog's own exit line is in that log — a child that merely died passes nothing.
* **It cost ~30 s of routine-gate time for ~1 s of signal.** The no-false-fire window is now ~7 polls
  (2.1 s) instead of 15 s; the orphan case ends as soon as the watchdog fires.
"""
import os
import sys
import time
import subprocess

from utils.contention import scale_timeout

# Re-exec helper modes. When this file is run as a subprocess with one of these
# argv[1] values, it acts as the parent/child rather than running pytest.
_PARENT_MODE = "__orphan_parent__"
_CHILD_MODE = "__orphan_child__"
_POLL = 0.3
_NO_FIRE_WINDOW_S = 7 * _POLL       # parent alive: the child must outlive ~7 of its own polls
_ORPHAN_LIFETIME_S = 60.0           # orphaned: far past the deadline, so only the watchdog ends it
_FIRED = "Exiting orphaned run."    # start_orphan_watchdog's own exit line


def _run_helper(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, os.path.abspath(__file__), *argv],
        capture_output=True, text=True, timeout=scale_timeout(60),
    )


def test_orphan_watchdog_exits_when_reparented(tmp_path):
    """Parent dies AFTER the child armed the watchdog → the reparented child self-exits, and the exit
    is the watchdog's (its log line), not a crash or its natural end."""
    log = tmp_path / "grandchild.log"
    out = _run_helper(_PARENT_MODE, str(tmp_path / "armed"), str(log))
    gcpid = None
    for ln in out.stdout.splitlines():
        if ln.startswith("GCPID="):
            gcpid = int(ln.split("=", 1)[1])
    assert gcpid is not None, f"helper did not report a grandchild pid: {out.stdout!r} {out.stderr!r}"

    deadline = time.monotonic() + scale_timeout(8.0)
    while os.path.exists(f"/proc/{gcpid}") and time.monotonic() < deadline:
        time.sleep(0.1)
    if os.path.exists(f"/proc/{gcpid}"):
        try:
            os.kill(gcpid, 9)  # don't leak the stuck process if we failed
        except ProcessLookupError:
            pass
        raise AssertionError(f"orphaned grandchild {gcpid} still alive past the deadline")
    text = log.read_text()
    assert _FIRED in text, f"the grandchild exited, but not through the watchdog: {text!r}"


def test_orphan_watchdog_no_false_fire_while_parent_alive():
    """Parent stays alive (waits on child) → child must NOT self-exit."""
    out = _run_helper(_CHILD_MODE, str(_NO_FIRE_WINDOW_S))
    assert out.returncode == 0, f"child exited nonzero while parent alive: rc={out.returncode} {out.stdout!r}"
    assert "CHILD-ALIVE-OK" in out.stdout, f"child did not complete normally: {out.stdout!r}"


# --- subprocess entry points -------------------------------------------------

def _child_main(lifetime_s: float, armed_path: str = ""):
    # Anchored at THIS file, not the cwd: the helper is re-executed as a bare subprocess and
    # the caller's working directory is not ours to assume. It cannot use `utils.paths` — this
    # line is what puts `utils` on the path in the first place. src/agents/training/… -> src/.
    _SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, _SRC)
    from agents.training.watchdog import start_orphan_watchdog
    start_orphan_watchdog(label="test", poll_seconds=_POLL)
    if armed_path:
        open(armed_path, "w").close()   # ONLY now may the parent exit: original_ppid is captured
    end = time.monotonic() + lifetime_s
    while time.monotonic() < end:
        time.sleep(_POLL)
    print("CHILD-ALIVE-OK", flush=True)  # only reached if the watchdog never fired


def _parent_main(armed_path: str, log_path: str):
    # Spawn the grandchild with its OWN stdout (never our caller's pipe, which would make the caller
    # wait out the grandchild's lifetime), wait until it has armed the watchdog, report its pid, exit.
    with open(log_path, "w") as log:
        # -u: the watchdog leaves through os._exit, which never flushes a block-buffered file stdout
        gc = subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__), _CHILD_MODE,
                               str(_ORPHAN_LIFETIME_S), armed_path],
                              stdout=log, stderr=subprocess.STDOUT)
    deadline = time.monotonic() + 50.0
    while not os.path.exists(armed_path):
        if gc.poll() is not None or time.monotonic() > deadline:
            print(f"GRANDCHILD-NEVER-ARMED rc={gc.poll()}", flush=True)
            os._exit(1)
        time.sleep(0.02)
    print(f"GCPID={gc.pid}", flush=True)
    os._exit(0)  # exit immediately without reaping -> grandchild is orphaned


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == _CHILD_MODE:
        _child_main(float(sys.argv[2]), sys.argv[3] if len(sys.argv) > 3 else "")
    elif mode == _PARENT_MODE:
        _parent_main(sys.argv[2], sys.argv[3])
    else:
        print("run via pytest, not directly", file=sys.stderr)
        sys.exit(2)
