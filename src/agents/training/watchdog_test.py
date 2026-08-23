"""Unit tests for the orphan watchdog (the --debug smoke-run zombie guard).

`start_orphan_watchdog` exits the process when its parent dies and it gets
reparented. The exit path is `os._exit`, which can't be observed in-process,
so the orphan scenario is driven through real subprocesses: a short-lived
"parent" spawns a "child" that arms the watchdog, then the parent exits — the
child is reparented and must self-exit. No Showdown server / bridge needed.
"""
import os
import sys
import time
import subprocess

# Re-exec helper modes. When this file is run as a subprocess with one of these
# argv[1] values, it acts as the parent/child rather than running pytest.
_PARENT_MODE = "__orphan_parent__"
_CHILD_MODE = "__orphan_child__"
_POLL = 0.3
_CHILD_NATURAL_LIFETIME_S = 15.0  # far longer than any watchdog/no-fire window


def _run_helper(mode: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, os.path.abspath(__file__), mode],
        capture_output=True, text=True, timeout=30,
    )


def test_orphan_watchdog_exits_when_reparented():
    """Parent dies → reparented child self-exits within a few poll cycles."""
    out = _run_helper(_PARENT_MODE)
    gcpid = None
    for ln in out.stdout.splitlines():
        if ln.startswith("GCPID="):
            gcpid = int(ln.split("=", 1)[1])
    assert gcpid is not None, f"helper did not report a grandchild pid: {out.stdout!r}"

    # Parent has now exited → grandchild is orphaned. It must die well before its
    # natural lifetime. Poll /proc for liveness (Linux).
    deadline = 8.0
    waited = 0.0
    while waited < deadline:
        time.sleep(0.5)
        waited += 0.5
        if not os.path.exists(f"/proc/{gcpid}"):
            return  # PASS: watchdog fired
    try:
        os.kill(gcpid, 9)  # don't leak the stuck process if we failed
    except ProcessLookupError:
        pass
    raise AssertionError(f"orphaned grandchild {gcpid} still alive after {deadline}s")


def test_orphan_watchdog_no_false_fire_while_parent_alive():
    """Parent stays alive (waits on child) → child must NOT self-exit."""
    out = _run_helper(_CHILD_MODE)
    assert out.returncode == 0, f"child exited nonzero while parent alive: rc={out.returncode}"
    assert "CHILD-ALIVE-OK" in out.stdout, f"child did not complete normally: {out.stdout!r}"


# --- subprocess entry points -------------------------------------------------

def _child_main():
    # Anchored at THIS file, not the cwd: the helper is re-executed as a bare subprocess and
    # the caller's working directory is not ours to assume. It cannot use `utils.paths` — this
    # line is what puts `utils` on the path in the first place. src/agents/training/… -> src/.
    _SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, _SRC)
    from agents.training.watchdog import start_orphan_watchdog
    start_orphan_watchdog(label="test", poll_seconds=_POLL)
    end = _CHILD_NATURAL_LIFETIME_S
    waited = 0.0
    while waited < end:
        time.sleep(_POLL)
        waited += _POLL
    print("CHILD-ALIVE-OK", flush=True)  # only reached if the watchdog never fired


def _parent_main():
    # Spawn the grandchild detached enough to outlive us, report its pid, exit.
    gc = subprocess.Popen([sys.executable, os.path.abspath(__file__), _CHILD_MODE])
    print(f"GCPID={gc.pid}", flush=True)
    os._exit(0)  # exit immediately without reaping -> grandchild is orphaned


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == _CHILD_MODE:
        _child_main()
    elif mode == _PARENT_MODE:
        _parent_main()
    else:
        print("run via pytest, not directly", file=sys.stderr)
        sys.exit(2)
