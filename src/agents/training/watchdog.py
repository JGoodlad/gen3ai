import os
import time
import threading


def start_subprocess_watchdog(vec_env, label="env", shutdown_event=None):
    """Kill the main process immediately if any SubprocVecEnv worker dies unexpectedly.

    SubprocVecEnv workers that crash leave the main process hanging on a pipe
    recv forever. This daemon thread detects the death and calls os._exit(1).

    Pass a threading.Event as shutdown_event and set it before a graceful exit
    to prevent the watchdog from firing during planned shutdown.
    """
    processes = getattr(vec_env, "processes", None)
    if not processes:
        return

    def _watch():
        while True:
            for p in processes:
                if not p.is_alive() and p.exitcode not in (0, None):
                    if shutdown_event is not None and shutdown_event.is_set():
                        return
                    print(f"\n🛑 [{label}] Worker PID {p.pid} died (exitcode={p.exitcode}). Exiting.")
                    os._exit(1)
            time.sleep(1)

    threading.Thread(target=_watch, daemon=True).start()
