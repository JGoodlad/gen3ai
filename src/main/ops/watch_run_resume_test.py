"""`scripts/ops/watch_run.sh` is RESUME-AWARE: it survives a launcher crash-restart.

2026-10-08 (the static-token screen): the watcher exited on its FIRST failure line, including a crash
the LAUNCHER restarts from the checkpoint by itself (`Child crashed ... Auto-restart #1`), so the run
went unwatched from the crash-resume on. The training agent hand-built a resume-aware copy. Two facts
shape the repo watcher:

  * the child log RESTARTS at a crash-resume, so a crash COUNT can only come from the LAUNCHER log;
  * the launcher decides what is final (`will NOT restart`, `giving up`, `Training complete`).

Every test drives the real script on a SYNTHETIC launcher log + child log pair that the test appends
to over time (crash -> resume -> keep watching -> final). Each fails on a revert of the resume-aware
half: the old script wrote `FAILURE: error text in child log` and exited at the first traceback.
"""
import os
import subprocess
import time

import pytest

from utils.paths import repo_path

SCRIPT = repo_path("scripts", "ops", "watch_run.sh")

TRACEBACK = 'Traceback (most recent call last):\n  File "x.py", line 1, in <module>\nValueError: boom\n'
CRASH = "[03:04:13] 🛑 Child crashed (exit 1) — saved crashes/restart_err_x.txt · crash #{n}\n"
RESUME = "[03:04:13] ♻️  Auto-restart #{n} after crash from checkpoint_14000046_steps.zip\n"
COMPLETE = "[03:18:45] ✅ Training complete — all steps done\n"


def _progress(step):
    return f"| time/ |  |\n|    total_timesteps | {step} |\n|    ep_len_mean | 12.5 |\n"


class Harness:
    def __init__(self, tmp_path, *, launcher_log=True, child_text=None, launcher_text="", extra=(),
                 pid_file=None):
        self.models = tmp_path / "models"
        self.run = self.models / "r"
        (self.run / "checkpoints").mkdir(parents=True, exist_ok=True)
        self.child = self.run / "launcher_child.log"
        self.child.write_text(child_text if child_text is not None else _progress(1000))
        self.launcher = tmp_path / "launcher.log"
        if launcher_log:
            self.launcher.write_text(launcher_text)
        self.status_path = tmp_path / "status.txt"
        argv = ["bash", str(SCRIPT), "r", "--status-file", str(self.status_path), "--interval", "1",
                "--launcher-log", str(self.launcher), *extra]
        argv += ["--pid-file", str(pid_file)] if pid_file else ["--no-pid-check"]
        env = dict(os.environ, GEN3AI_MODELS_DIR=str(self.models))
        self.proc = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True)

    def status(self):
        return self.status_path.read_text() if self.status_path.exists() else ""

    def wait_for(self, needle, timeout=25):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if needle in self.status():
                return
            if self.proc.poll() is not None and needle not in self.status():
                break
            time.sleep(0.2)
        raise AssertionError(f"never saw {needle!r} in:\n{self.status()}")

    def alive_for(self, seconds):
        """The watcher is still running after `seconds` of further ticks."""
        time.sleep(seconds)
        return self.proc.poll() is None

    def append(self, path, text):
        with open(path, "a") as fh:
            fh.write(text)

    def finish(self, timeout=25):
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            raise AssertionError(f"the watcher never exited:\n{self.status()}") from None
        return self.status()

    def close(self):
        if self.proc.poll() is None:
            self.proc.kill()
        self.proc.wait()


@pytest.fixture
def harness(tmp_path):
    made = []

    def make(**kw):
        h = Harness(tmp_path, **kw)
        made.append(h)
        return h
    yield make
    for h in made:
        h.close()


def test_crash_then_resume_keeps_watching_until_the_final(harness):
    h = harness()
    h.wait_for("ok step=1000")
    assert "crashes=0" in h.status()

    # the child crashes: its traceback lands in the child log, the launcher logs the crash
    h.append(h.child, TRACEBACK)
    h.append(h.launcher, CRASH.format(n=1))
    h.wait_for("CRASH:")
    h.wait_for("ERROR: error text in child log")
    assert h.alive_for(3), f"the watcher exited at a crash the launcher restarts:\n{h.status()}"
    assert "FAILURE" not in h.status()

    # the launcher resumes: the CHILD LOG RESTARTS (the old traceback is gone), the step counter too
    h.child.write_text(_progress(2000))
    h.append(h.launcher, RESUME.format(n=1))
    h.wait_for("RESUMED:")
    h.wait_for("ok step=2000")
    assert "crashes=1" in h.status().splitlines()[-1] or "crashes=1" in h.status()
    assert h.alive_for(2), f"the watcher exited after the resume:\n{h.status()}"
    assert "FAILURE" not in h.status()

    # the run truly ends
    h.append(h.launcher, COMPLETE)
    status = h.finish()
    assert "DONE:" in status and "1 crash(es) auto-resumed" in status
    assert "FAILURE" not in status
    assert status.rstrip().endswith("WATCHER EXIT")
    assert h.proc.returncode == 0


def test_each_crash_is_an_event_and_the_count_comes_from_the_launcher_log(harness):
    h = harness()
    h.wait_for("ok step=1000")
    for n in (1, 2):
        h.append(h.launcher, CRASH.format(n=n) + RESUME.format(n=n))
        h.wait_for(f"crash #{n}")
        h.wait_for(f"Auto-restart #{n}")
    h.wait_for("crashes=2")
    crash_events = [ln for ln in h.status().splitlines() if "] CRASH:" in ln]
    assert len(crash_events) == 2, h.status()
    assert h.proc.poll() is None


def test_a_crash_already_in_the_launcher_log_at_start_is_reported_as_history(harness):
    h = harness(launcher_text=CRASH.format(n=1) + RESUME.format(n=1), child_text=_progress(3000))
    h.wait_for("already in the launcher log when the watcher started")
    h.wait_for("crashes=1")
    assert h.proc.poll() is None and "FAILURE" not in h.status()


def test_a_traceback_already_in_the_child_log_at_start_does_not_end_a_resumed_watch(harness):
    h = harness(child_text=TRACEBACK + _progress(3000))
    h.wait_for("already in the child log at watcher start")
    h.wait_for("ok step=3000")
    assert h.alive_for(2) and "FAILURE" not in h.status()


@pytest.mark.parametrize("giveup", [
    "[03:04:13] 🛑 Fatal config error — will NOT restart — saved crashes/x.txt\n",
    "[03:04:13] 🛑 3 rapid crashes in a row (< 5m each) — giving up\n",
    "[03:04:13] 🛑 No checkpoint found under the run dir — cannot restart\n",
])
def test_a_launcher_that_will_not_restart_ends_the_watch_as_a_failure(harness, giveup):
    h = harness()
    h.wait_for("ok step=1000")
    h.append(h.child, TRACEBACK)
    h.append(h.launcher, CRASH.format(n=1) + giveup)
    status = h.finish()
    assert "FAILURE: the launcher will not restart" in status
    assert "DONE:" not in status and status.rstrip().endswith("WATCHER EXIT")


def test_a_final_model_written_since_the_start_ends_the_watch_as_done(harness):
    h = harness()
    h.wait_for("ok step=1000")
    (h.run / "final_model.zip").write_bytes(b"")
    status = h.finish()
    assert "DONE: final_model.zip written since the watcher started" in status
    assert "FAILURE" not in status


def test_a_final_model_from_before_the_watcher_is_not_a_finish(harness, tmp_path):
    # a resumed/forked run dir that already holds an OLD final_model.zip must keep being watched
    run = tmp_path / "models" / "r"
    (run / "checkpoints").mkdir(parents=True)
    old = run / "final_model.zip"
    old.write_bytes(b"")
    long_ago = time.time() - 3600
    os.utime(old, (long_ago, long_ago))
    h = harness()
    h.wait_for("ok step=1000")
    assert h.alive_for(2) and "DONE" not in h.status()


def test_the_launcher_pid_going_away_is_done_after_a_completion_line_and_a_failure_without(
        harness, tmp_path):
    # with the completion line: DONE (the launcher exits right after it, so the pid vanishes)
    launcher = subprocess.Popen(["sleep", "60"])
    pidf = tmp_path / "launcher.pid"
    pidf.write_text(str(launcher.pid))
    h = harness(pid_file=pidf, launcher_text=CRASH.format(n=1) + RESUME.format(n=1))
    try:
        h.wait_for("ok step=1000")
        h.append(h.launcher, COMPLETE)
        launcher.kill()
        launcher.wait()
        status = h.finish()
        assert "DONE:" in status and "FAILURE" not in status
    finally:
        launcher.kill()
        launcher.wait()


def test_the_launcher_pid_going_away_with_no_completion_line_is_still_a_failure(harness, tmp_path):
    launcher = subprocess.Popen(["sleep", "60"])
    pidf = tmp_path / "launcher2.pid"
    pidf.write_text(str(launcher.pid))
    h = harness(pid_file=pidf)
    try:
        h.wait_for("ok step=1000")
        launcher.kill()
        launcher.wait()
        status = h.finish()
        assert "FAILURE: launcher pid" in status and "GONE" in status and "DONE:" not in status
    finally:
        launcher.kill()
        launcher.wait()


def test_without_a_readable_launcher_log_the_first_child_error_still_ends_the_watch(harness):
    """The old contract holds when there is no launcher log to be the authority: an ABSENT file is
    not a clean one, so the watcher cannot know the launcher will resume and fails loudly."""
    h = harness(launcher_log=False, child_text=TRACEBACK)
    status = h.finish()
    assert "FAILURE: error text in child log" in status
    assert "ABSENT" in status


def test_the_sync_to_main_invalidation_still_ends_the_watch(harness):
    h = harness()
    h.wait_for("ok step=1000")
    h.append(h.launcher, "launcher --sync-to-main\n")
    status = h.finish()
    assert "FAILURE: --sync-to-main in the launcher log" in status
