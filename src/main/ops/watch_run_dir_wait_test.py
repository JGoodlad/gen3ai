"""`scripts/ops/watch_run.sh` WAITS for a run dir that a live launch has not created yet.

2026-09-28: the T32 arm's watcher was started beside its launch, found no `models/<run>/` (the
launcher's child creates it seconds later) and REFUSED, so the arm ran with no layer-1 watcher. With a
`--pid-file` and a bare run NAME the watcher now waits (`--dir-wait-seconds`, default 900) while the
launcher pid is alive; a directory that never appears is still a refusal.
"""
import os
import subprocess
import time

from utils.paths import repo_path

SCRIPT = repo_path("scripts", "ops", "watch_run.sh")


def _start_watcher(tmp_path, run, pid, *extra):
    pidf = tmp_path / "launcher.pid"
    pidf.write_text(str(pid))
    status = tmp_path / "status.txt"
    env = dict(os.environ, GEN3AI_MODELS_DIR=str(tmp_path / "models"))
    proc = subprocess.Popen(
        ["bash", str(SCRIPT), run, "--pid-file", str(pidf), "--status-file", str(status),
         "--interval", "1", *extra],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return proc, status


def test_waits_for_the_run_dir_while_the_launcher_lives(tmp_path):
    (tmp_path / "models").mkdir()
    launcher = subprocess.Popen(["sleep", "60"])
    try:
        proc, status = _start_watcher(tmp_path, "late_run", launcher.pid, "--dir-wait-seconds", "30")
        time.sleep(3)
        assert proc.poll() is None, f"watcher exited before the dir appeared: {proc.communicate()}"
        (tmp_path / "models" / "late_run").mkdir()
        deadline = time.time() + 20
        while time.time() < deadline and not (status.exists() and "WATCHER START" in status.read_text()):
            time.sleep(0.5)
        assert status.exists() and "WATCHER START" in status.read_text(), \
            "watcher never started once the run dir appeared"
    finally:
        launcher.kill()
        launcher.wait()
    proc.wait(timeout=30)            # the launcher is gone → FAILURE … GONE → WATCHER EXIT
    assert "GONE" in status.read_text()


def test_still_refuses_when_the_launcher_dies_and_the_dir_never_appears(tmp_path):
    (tmp_path / "models").mkdir()
    launcher = subprocess.Popen(["true"])
    launcher.wait()          # REAPED: an unreaped exited child is a zombie, and `kill -0` succeeds on it
    proc, _ = _start_watcher(tmp_path, "never_run", launcher.pid, "--dir-wait-seconds", "60")
    out, err = proc.communicate(timeout=20)
    assert proc.returncode == 2
    assert "REFUSING: no run 'never_run'" in err
