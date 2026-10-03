"""The train-env worker watchdog must not turn a clean finish into a crash.

2026-09-27..29, every run end (N0, L95, C_fix, K2, K3): after the final save (and, then, the final eval), a
SubprocVecEnv worker died in teardown with exitcode -15, the watchdog called `os._exit(1)`, and the
launcher logged "🛑 Child crashed (exit 1) … crash #1" and auto-restarted a run whose steps were all
done. Two changes close it, and each has a test that fails on revert:
  * the watchdog checks its shutdown event BEFORE the workers (a -15 in the same poll no longer wins);
  * `model_build` sets that event as soon as `model.learn` returns, on BOTH the resume and fresh paths.
"""
import re
import threading
import time

from agents.training import watchdog
from utils.paths import src_path


class _DeadWorker:
    pid = 4242
    exitcode = -15

    def is_alive(self):
        return False


class _Env:
    processes = [_DeadWorker()]


def _arm(monkeypatch, event):
    fired = threading.Event()

    def _fake_exit(code):
        fired.set()
        event.set()          # let the (otherwise infinite) watch loop end

    monkeypatch.setattr(watchdog.os, "_exit", _fake_exit)
    watchdog.start_subprocess_watchdog(_Env(), label="train_env", shutdown_event=event)
    return fired


def test_a_worker_dead_after_shutdown_is_not_a_crash(monkeypatch):
    ev = threading.Event()
    ev.set()                 # training is over, THEN teardown kills a worker
    fired = _arm(monkeypatch, ev)
    time.sleep(1.5)
    assert not fired.is_set(), "watchdog os._exit'd on a teardown worker death after shutdown"


def test_a_worker_dead_during_training_still_kills_the_run(monkeypatch):
    ev = threading.Event()   # control: the guard is still live while training runs
    fired = _arm(monkeypatch, ev)
    assert fired.wait(3.0), "watchdog no longer fires on a worker that dies mid-training"


def test_model_build_stands_the_watchdog_down_between_learn_and_the_final_save():
    src = src_path("main", "train", "model_build.py").read_text()
    learns = [m.end() for m in re.finditer(r"model\.learn\(", src)]
    assert len(learns) == 2, f"expected the resume and the fresh learn() sites, found {len(learns)}"
    for start in learns:
        save = src.index('final_path = os.path.join(model_dir, "final_model")', start)
        assert "_shutdown_event.set()" in src[start:save], (
            "model_build must set _shutdown_event after model.learn returns and before the final "
            "save/eval, or a teardown worker death reads as a crash")
