"""`scripts/ops/watch_run.sh` reads launcher liveness from /proc (pid_alive), never `kill -0`.

`kill -0` reads a live pid as dead in a sandboxed shell (acf2193f, 7f9ff623) and reads a ZOMBIE as alive. The zombie
is the deterministic discriminator: each test fails on a revert to `kill -0`.
"""
import os
import subprocess
import sys
import time

from utils.paths import repo_path

SCRIPT = repo_path("scripts", "ops", "watch_run.sh")
COMMON = repo_path("scripts", "ops", "_common.sh")


def _alive(pid):
    r = subprocess.run(["bash", "-c", f'. "{COMMON}"; pid_alive {pid}'], capture_output=True)
    return r.returncode == 0


def _zombie():
    """A child that has exited and not been waited on."""
    p = subprocess.Popen([sys.executable, "-c", "pass"])
    for _ in range(200):
        with open(f"/proc/{p.pid}/stat") as f:
            if f.read().rsplit(") ", 1)[1].split()[0] == "Z":
                return p
        time.sleep(0.05)
    raise AssertionError("child never became a zombie")


def test_pid_alive_live_reaped_and_zombie():
    live = subprocess.Popen(["sleep", "30"])
    try:
        assert _alive(live.pid)
    finally:
        live.kill(); live.wait()
    assert not _alive(live.pid), "a reaped pid is gone"
    z = _zombie()
    try:
        assert not _alive(z.pid), "a zombie is gone (kill -0 would say alive)"
    finally:
        z.wait()


def _watch(tmp_path, pid):
    models = tmp_path / "models"
    (models / "r" / "checkpoints").mkdir(parents=True)
    (models / "r" / "launcher_child.log").write_text("|    total_timesteps | 1000 |\n")
    pidf = tmp_path / "pid"
    pidf.write_text(str(pid))
    st = tmp_path / "status.txt"
    env = dict(os.environ, GEN3AI_MODELS_DIR=str(models))
    p = subprocess.Popen(["bash", str(SCRIPT), "r", "--status-file", str(st), "--interval", "0.25",
                          "--pid-file", str(pidf)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p, st


def test_watcher_reports_a_zombie_launcher_gone(tmp_path):
    z = _zombie()
    p, st = _watch(tmp_path, z.pid)
    try:
        out, _ = p.communicate(timeout=60)
    finally:
        if p.poll() is None:
            p.kill(); p.wait()
        z.wait()
    assert "GONE" in out + (st.read_text() if st.exists() else "")


def test_watcher_keeps_watching_a_live_launcher(tmp_path):
    live = subprocess.Popen(["sleep", "30"])
    p, st = _watch(tmp_path, live.pid)
    try:
        time.sleep(2)
        assert p.poll() is None, "the watcher exited on a live launcher"
        assert "GONE" not in (st.read_text() if st.exists() else "")
    finally:
        p.kill(); p.wait(); live.kill(); live.wait()
