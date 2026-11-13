import os
import time
import threading


def start_orphan_watchdog(label="debug", poll_seconds=2.0, shutdown_event=None):
    """Exit if our parent process dies and we get reparented (orphaned).

    A `--debug` smoke run is launched as a child of the shell/agent that started
    it. If that parent exits, the kernel reparents us (our PPID changes — to the
    nearest subreaper or PID 1) and nothing else will ever stop us. A smoke run
    that is also hung (e.g. waiting on a 9XXX Showdown server that went away)
    then lingers forever as a zombie holding a GB+ of RAM. This daemon thread
    notices the reparent and exits.

    Detection is by PPID *change* (capture the launching parent's PID up front,
    compare each poll) rather than `== 1`, so it works under PID-namespace
    subreapers where init is not the reaper. (The SubprocVecEnv worker watchdog
    that sat beside it went with the Python env core's workers, deletion pass U4.)

    Pass a threading.Event as shutdown_event and set it before a graceful exit
    so the thread stops cleanly instead of polling through teardown.
    """
    original_ppid = os.getppid()

    def _watch():
        while True:
            if shutdown_event is not None:
                if shutdown_event.wait(timeout=poll_seconds):
                    return
            else:
                time.sleep(poll_seconds)
            current_ppid = os.getppid()
            if current_ppid != original_ppid:
                # The message is best effort and the exit is NOT: a launcher-managed child's stdout is a pipe to its
                # launcher (`main.launcher.child`, PYTHONUNBUFFERED=1), so when the launcher is the parent that died
                # the pipe has no reader and this `print` raises BrokenPipeError. Unguarded, that killed this thread
                # before the `os._exit` and the orphan lived on (measured 2026-11-13: alive 12 s after its parent died).
                try:
                    print(f"\n🛑 [{label}] Parent process {original_ppid} died "
                          f"(reparented to {current_ppid}). Exiting orphaned run.")
                except Exception:  # noqa: BLE001
                    pass
                finally:
                    os._exit(1)

    threading.Thread(target=_watch, daemon=True).start()
