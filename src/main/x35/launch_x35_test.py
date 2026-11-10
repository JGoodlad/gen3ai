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

from utils import core_split
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


def test_the_launcher_is_spawned_in_its_own_session():
    """11-06: a ``timeout`` around GO signalled GO's whole process group and SIGTERMed X60-T s3501's launcher."""
    spawn = [ln for ln in TEXT.splitlines() if '"$PY" -m main.launcher' in ln and "--dry-run" not in ln
             and not ln.lstrip().startswith("#")]
    assert spawn, "the launcher spawn line is gone"
    assert all('exec setsid taskset -c "$TRAINER_CPUS"' in ln for ln in spawn), spawn


def test_the_trainer_pin_is_the_core_splits_read_at_launch(tmp_path):
    """The kit spells no CPU list: it reads ``utils.core_split trainer-cpus --launch`` (the read that makes a queued
    split active) after every pre-check and before the spawn, and HOLDs on an empty answer. Run for real against the
    sealed default state, the read prints the default split's trainer list."""
    code = "\n".join(ln for ln in TEXT.splitlines() if not ln.lstrip().startswith("#"))
    assert not re.search(r"taskset -c [0-9]", code)
    read = code.index('-m utils.core_split trainer-cpus --launch 2>>')
    assert code.index("prechecks go") < read < code.index('exec setsid taskset -c "$TRAINER_CPUS"')
    assert re.search(r'&& \[ -n "\$TRAINER_CPUS" \] \|\| hold ', code)
    r = subprocess.run([sys.executable, "-m", "utils.core_split", "trainer-cpus", "--launch"], capture_output=True,
                       text=True, timeout=60, env=dict(os.environ, GEN3AI_CORE_SPLIT_STATE=str(tmp_path / "s.json")))
    assert r.returncode == 0 and r.stdout.strip() == core_split.DEFAULT.trainer_list and not (tmp_path / "s.json").exists()
    # ...and every pre-check pass reads it first WITHOUT writing: an unreadable state, or a queued split this launch
    # would not get, HOLDs before the once-only batch records are written, and again at the launch read itself
    dry = code.index('-m utils.core_split trainer-cpus --launch --dry-run')
    assert dry < code.index("\nprechecks go") and dry < code.index("kit pin-check --pin \"$PIN\" --side \"$SIDE\" --arm \"$ARM\" --stage \"$STAGE\" --models-dir \"$MODELS\" --write")
    assert code.count(f'grep -q "{core_split.STAYS_QUEUED}"') == 2 and code.count("|| hold \"the queued core split") == 2
    # the launch line names the note, or says none was queued (an empty "(core split: )" read as a broken read, 11-10)
    assert '(core split: ${CSNOTE:-no queued split})' in code
    state = tmp_path / "q" / "core_split.json"
    env = dict(os.environ, GEN3AI_CORE_SPLIT_STATE=str(state))
    assert subprocess.run([sys.executable, "-m", "utils.core_split", "set", "4+4", "--next-launch"], env=env,
                          capture_output=True, timeout=60).returncode == 0
    before = sorted(x.name for x in state.parent.iterdir()), state.read_bytes()
    r = subprocess.run([sys.executable, "-m", "utils.core_split", "trainer-cpus", "--launch", "--dry-run"], env=env,
                       capture_output=True, text=True, timeout=60)
    assert r.returncode == 0 and r.stdout.strip() in (core_split.parse_split("4+4").trainer_list, core_split.DEFAULT.trainer_list)
    assert (core_split.STAYS_QUEUED in r.stderr) == (r.stdout.strip() == core_split.DEFAULT.trainer_list), "the kit greps stderr"
    assert before == (sorted(x.name for x in state.parent.iterdir()), state.read_bytes()), "a dry run writes nothing"


def test_a_group_signal_to_go_does_not_reach_a_setsid_child(tmp_path):
    """The spawn pattern itself: under ``timeout`` (which signals its process group), a ``( exec setsid … ) &`` child
    keeps its pid as ``$!``, sits in its own session, and survives; a plain ``( exec … ) &`` child dies with the group."""
    pids = {}
    for mode in ("setsid", "plain"):
        pidf = tmp_path / f"{mode}.pid"
        prefix = "setsid " if mode == "setsid" else ""
        subprocess.run(["timeout", "-s", "TERM", "2", "bash", "-c",
                        f"( exec {prefix}sleep 60 ) & echo $! > {pidf}; wait"], timeout=30)
        pids[mode] = int(pidf.read_text())
    time.sleep(0.5)
    try:
        stat = Path(f"/proc/{pids['setsid']}/stat").read_text()
        fields = stat[stat.rindex(")") + 2:].split()
        assert int(fields[3]) == pids["setsid"], "the setsid child is not its own session leader"   # field 6: session
        assert not Path(f"/proc/{pids['plain']}").exists() or \
            Path(f"/proc/{pids['plain']}/stat").read_text().split()[2] == "Z", "the plain child survived the group signal"
    finally:
        for p in pids.values():
            try:
                os.kill(p, 9)
            except ProcessLookupError:
                pass
