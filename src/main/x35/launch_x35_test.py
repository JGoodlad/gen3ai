"""``launch_x35.sh``'s liveness helpers and its K9(b) pre-check wiring (F1 / F2 of ``designs/ops/arm_chain.md``, d4387711).

F2: ``kill -0`` reads a live pid as dead in a sandboxed shell (acf2193f, CB-26); the script reads ``/proc`` instead,
with pid + start ticks as the identity. The helper block is extracted between its markers and run in a real bash.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest

from utils.paths import repo_path

SCRIPT = repo_path("designs", "research_state", "measurements", "x35_breakers_stage_2026-10-17", "launch_x35.sh")
TEXT = SCRIPT.read_text()


def _helpers() -> str:
    m = re.search(r"^# ---- liveness helpers \(begin.*?\n(.*?)^# ---- liveness helpers \(end\)", TEXT, re.S | re.M)
    assert m, "the liveness helper block markers are gone"
    return m.group(1)


def sh(code: str, **env: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", "set -u\n" + _helpers() + code], capture_output=True, text=True,
                          env=dict(os.environ, **env), timeout=30)


@pytest.fixture
def sleeper():
    p = subprocess.Popen(["sleep", "300"])
    try:
        yield p
    finally:
        p.kill()
        p.wait()


def test_the_script_never_uses_kill_zero():
    code = "\n".join(ln for ln in TEXT.splitlines() if not ln.lstrip().startswith("#"))
    assert not re.search(r"kill\s+-0|kill\s+-s\s+0|kill\s+-n\s+0", code)
    assert len(re.findall(r"launcher_alive", code)) >= 4          # the three former sites + the guarded SIGTERM


def test_a_live_pid_is_alive_and_start_ticks_are_the_identity(sleeper):
    r = sh(f'pid_alive {sleeper.pid} && echo alive; t=$(pid_start {sleeper.pid}); echo "$t"')
    assert r.stdout.split()[0] == "alive" and r.stdout.split()[1].isdigit(), r
    ticks = int(Path(f"/proc/{sleeper.pid}/stat").read_text().rsplit(") ", 1)[1].split()[19])
    assert int(r.stdout.split()[1]) == ticks                       # stat field 22


def test_launcher_alive_needs_the_same_start_ticks(sleeper):
    ticks = sh(f"pid_start {sleeper.pid}").stdout.strip()
    ok = sh(f'LP={sleeper.pid}; LP_START={ticks}; launcher_alive && echo yes')
    assert ok.stdout.strip() == "yes", ok
    for start in (str(int(ticks) + 1), ""):                        # a REUSED pid has other start ticks; no ticks = dead
        r = sh(f'LP={sleeper.pid}; LP_START={start}; launcher_alive && echo yes')
        assert r.returncode != 0 and not r.stdout, (start, r)


def test_a_dead_or_unreaped_pid_is_not_alive():
    p = subprocess.Popen(["true"])
    deadline = time.time() + 10
    while time.time() < deadline and not Path(f"/proc/{p.pid}/stat").read_text().rsplit(") ", 1)[1].startswith("Z"):
        time.sleep(0.01)
    assert sh(f"pid_alive {p.pid}").returncode != 0                # a zombie is gone (kill -0 says alive)
    p.wait()
    assert sh(f"pid_alive {p.pid}").returncode != 0                # reaped: no /proc entry
    assert sh("pid_alive 999999999").returncode != 0


def test_a_comm_with_parentheses_and_spaces_does_not_shift_the_fields():
    code = ("import ctypes, time; ctypes.CDLL(None).prctl(15, b'a) 1 (b) Z 9'); time.sleep(300)")   # PR_SET_NAME
    p = subprocess.Popen([sys.executable, "-c", code])
    try:
        deadline = time.time() + 10
        while time.time() < deadline and "a)" not in Path(f"/proc/{p.pid}/stat").read_text():
            time.sleep(0.02)
        stat = Path(f"/proc/{p.pid}/stat").read_text()
        assert "(a) 1 (b) Z 9)" in stat, stat
        ticks = int(stat.rsplit(") ", 1)[1].split()[19])
        assert sh(f"pid_start {p.pid}").stdout.strip() == str(ticks)
    finally:
        p.kill()
        p.wait()


def test_precheck_11_passes_k9b_batch_a_scratch_out_dir_and_never_the_stage_for_writing():
    m = re.search(r"kit k9b-batch (.*?)\\?\n", TEXT)
    assert m and '--stage "$STAGE"' in m.group(1) and '--out-dir "$K9B_OUT"' in m.group(1)
    assert re.search(r'^K9B_OUT=\$\{X35_K9B_OUT:-\$J/k9b_scratch\}', TEXT, re.M)       # under the job dir: no checkout


def test_the_script_still_parses_and_a_bad_call_keeps_its_exit_codes(tmp_path):
    assert subprocess.run(["bash", "-n", str(SCRIPT)]).returncode == 0
    env = dict(os.environ, X35_JOB_DIR=str(tmp_path / "job"), X35_STAGE=str(tmp_path), CUDA_VISIBLE_DEVICES="")
    usage = subprocess.run([str(SCRIPT)], capture_output=True, text=True, env=env, timeout=30)
    assert usage.returncode == 2
    flag = subprocess.run([str(SCRIPT), "c0", "3501", "--pin", "x", "--bogus"], capture_output=True, text=True, env=env, timeout=30)
    assert flag.returncode == 2 and "unknown flag" in flag.stderr
    hold = subprocess.run([str(SCRIPT), "c0", "3501", "--pin", "0" * 40], capture_output=True, text=True, env=env, timeout=30)
    assert hold.returncode == 7 and "HOLD" in hold.stderr
