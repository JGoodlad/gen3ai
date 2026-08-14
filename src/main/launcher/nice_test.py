"""Unit tests for the launcher's --nice deprioritisation.

A training run holds ~940 processes on a 16-core box. At nice 0 it competes on equal
terms with interactive work sharing the machine: measured 2026-08-13, an interactive
client in the same cgroup as the run waited 2.1 s in the run queue per 1 s of CPU it
got, and every training process sat at nice 0.

The contract has three parts, and the third is the one that actually matters over a
multi-day run: niceness is applied to the LAUNCHER, and children inherit it across
fork/exec — so the training child, its SubprocVecEnv workers and every eval worker are
covered without any per-spawn wiring, including the processes spawned by the periodic
3-hourly restarts and by crash auto-restarts.

``os.nice`` mutates the calling process irreversibly for an unprivileged user (it can
only ever raise), so every test that actually applies a level runs in a subprocess —
otherwise the first one would renice the pytest session itself and poison the rest.
"""

import os
import subprocess
import sys

from main.launcher.checkpoint import _strip_launcher_args
from main.launcher.run import DEFAULT_NICE, _apply_nice

_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _in_subprocess(body: str) -> str:
    """Run `body` in a fresh interpreter that can import main.launcher; return stdout."""
    env = dict(os.environ)
    env["PYTHONPATH"] = _SRC + os.pathsep + env.get("PYTHONPATH", "")
    proc = subprocess.run(
        [sys.executable, "-c", body],
        capture_output=True, text=True, env=env, timeout=60,
    )
    assert proc.returncode == 0, f"subprocess failed: {proc.stderr}"
    return proc.stdout.strip()


# --- disabled / no-op cases (safe to run in-process: they must not call os.nice) ---

def test_zero_disables():
    before = os.nice(0)
    assert _apply_nice(0) is None
    assert os.nice(0) == before, "a disabled --nice must not touch the process"


def test_negative_is_a_noop():
    before = os.nice(0)
    assert _apply_nice(-5) is None
    assert os.nice(0) == before, "a negative target needs CAP_SYS_NICE — must no-op, not raise"


# --- the applying cases ---

def test_raises_to_target():
    out = _in_subprocess(
        "from main.launcher.run import _apply_nice;"
        "print(_apply_nice(10))"
    )
    assert out == "10"


def test_never_lowers_priority_below_current():
    """Already niced further than the target ⇒ leave it alone (and never raise)."""
    out = _in_subprocess(
        "import os;"
        "os.nice(12);"
        "from main.launcher.run import _apply_nice;"
        "print(_apply_nice(5), os.nice(0))"
    )
    assert out == "12 12", "must report and keep the existing, higher niceness"


def test_children_inherit_the_nice_level():
    """The whole mechanism: set once on the launcher, inherited by every descendant.

    If this breaks, the training child and its ~940 workers silently run at nice 0
    again and nothing else in the suite would notice.
    """
    out = _in_subprocess(
        "import os, subprocess, sys;"
        "from main.launcher.run import _apply_nice;"
        "_apply_nice(10);"
        "print(subprocess.run([sys.executable, '-c', 'import os; print(os.nice(0))'],"
        "                     capture_output=True, text=True).stdout.strip())"
    )
    assert out == "10", "a grandchild of the launcher must inherit the nice level"


# --- the flag must not leak into train_rl_agent.py ---

def test_strip_space_form():
    out = _strip_launcher_args(["--nice", "10", "--steps", "100"])
    assert out == ["--steps", "100"]


def test_strip_equals_form():
    out = _strip_launcher_args(["--nice=10", "--steps", "100"])
    assert out == ["--steps", "100"]


def test_strip_leaves_other_flags_alone():
    """Guard the prefix match — a future --nice-something must not be eaten."""
    out = _strip_launcher_args(["--n-envs", "48", "--nice", "5", "--device", "cuda"])
    assert out == ["--n-envs", "48", "--device", "cuda"]


def test_default_is_on_by_default():
    """The default must be non-zero: an unattended run should deprioritise itself
    without anyone remembering to pass a flag."""
    assert DEFAULT_NICE == 10


def test_default_is_wired_into_the_parser():
    """Guard the flag→default wiring, not just the constant."""
    out = _in_subprocess(
        "import sys, io, contextlib;"
        "sys.argv = ['launcher', '--help'];"
        "from main.launcher.run import main;"
        "buf = io.StringIO();"
        "\ntry:\n"
        "    with contextlib.redirect_stdout(buf):\n"
        "        main()\n"
        "except SystemExit:\n"
        "    pass\n"
        "print('--nice' in buf.getvalue())"
    )
    assert out == "True"
