"""Unit tests for the launcher child's interpreter resolution.

``child.py`` used to hardcode one box's conda interpreter
(``/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3``) with no flag and no env
override, so a fresh clone died with ``FileNotFoundError`` on its first launcher run and
the only fix was editing the source. The rule now is:

    $GEN3AI_PYTHON  →  sys.executable

``sys.executable`` is the *correct* default rather than a guess: the launcher is already
running under the environment the run wants, so the child inherits it on any machine
under any env name. On this box the resolved value is still the conda interpreter
whenever the launcher was started with it, so the change is behaviour-identical here.

The absolute-literal test is the durable one — it fails if anyone re-introduces a
machine-specific path anywhere in the launcher package, not just at the old line.
"""

import os
import re
import subprocess
import sys
import time

from main.launcher import child
from main.launcher.child import PYTHON_ENV_VAR, resolve_child_python

_LAUNCHER_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.dirname(os.path.dirname(_LAUNCHER_DIR))


# --- the resolution rule ---

def test_default_is_the_launchers_own_interpreter(monkeypatch):
    monkeypatch.delenv(PYTHON_ENV_VAR, raising=False)
    assert resolve_child_python() == sys.executable


def test_env_override_wins(monkeypatch):
    monkeypatch.setenv(PYTHON_ENV_VAR, "/opt/some/other/python3")
    assert resolve_child_python() == "/opt/some/other/python3"


def test_blank_override_falls_back_rather_than_spawning_the_empty_string(monkeypatch):
    """An empty/whitespace GEN3AI_PYTHON must not become argv[0] — that is a spawn
    failure at the worst possible moment (a 3 a.m. crash restart), and an exported-but-
    unset var is a routine shell state."""
    for blank in ("", "   ", "\t"):
        monkeypatch.setenv(PYTHON_ENV_VAR, blank)
        assert resolve_child_python() == sys.executable


def test_resolution_is_not_cached_at_import(monkeypatch):
    """A periodic/crash restart must see a changed override — the launcher process can
    outlive a dozen children, so an import-time constant would pin the first value."""
    monkeypatch.setenv(PYTHON_ENV_VAR, "/first/python3")
    assert resolve_child_python() == "/first/python3"
    monkeypatch.setenv(PYTHON_ENV_VAR, "/second/python3")
    assert resolve_child_python() == "/second/python3"


# --- the durable guard ---

def test_no_machine_specific_interpreter_literal_in_the_launcher_package():
    """No launcher module may name an absolute interpreter/home path. This is what
    actually keeps a fresh clone launchable; the unit tests above only pin the helper."""
    bad = re.compile(r"/home/\w+|/(?:opt|usr/local)/miniconda|/miniconda\d*/envs/")
    offenders = []
    for name in sorted(os.listdir(_LAUNCHER_DIR)):
        if not name.endswith(".py") or name == os.path.basename(__file__):
            continue
        text = open(os.path.join(_LAUNCHER_DIR, name), encoding="utf-8").read()
        for i, line in enumerate(text.splitlines(), 1):
            if bad.search(line):
                offenders.append(f"{name}:{i}: {line.strip()}")
    assert not offenders, (
        "launcher modules must not hardcode a machine-specific path — use "
        f"resolve_child_python() / ${PYTHON_ENV_VAR}:\n  " + "\n  ".join(offenders)
    )


# --- end to end: the resolved interpreter is the one actually spawned ---

def test_the_spawned_child_really_runs_under_the_resolved_interpreter(tmp_path):
    """Drive the real ``_launch_child`` and read the child's own ``sys.executable`` back.

    Substitutes a throwaway script for ``train_rl_agent.py`` — the contract under test is
    the argv[0] of the spawn, not what training does with it."""
    from main.launcher.state import LauncherState

    probe = tmp_path / "probe.py"
    probe.write_text("import sys; print(sys.executable, flush=True)\n")

    state = LauncherState(interval_hours=0.0)
    state.run_dir = str(tmp_path)
    proc = child._launch_child([], child._build_child_env(), state, str(probe), _SRC)
    assert proc.wait(timeout=60) == 0
    # The reader thread streams stdout into the state scrollback.
    for _ in range(200):
        if any(sys.executable in line for line in state.snapshot().log_lines):
            break
        time.sleep(0.05)
    logged = "\n".join(state.snapshot().log_lines)
    assert sys.executable in logged, f"child did not run under {sys.executable}:\n{logged}"


def test_env_override_reaches_the_spawn(tmp_path, monkeypatch):
    """A bogus override must fail the SPAWN — proof the value is really argv[0] and not
    decoration. (A wrong interpreter is loud, which is the point of the override.)"""
    monkeypatch.setenv(PYTHON_ENV_VAR, str(tmp_path / "definitely-not-a-python"))
    from main.launcher.state import LauncherState

    probe = tmp_path / "probe.py"
    probe.write_text("print('unreachable')\n")
    state = LauncherState(interval_hours=0.0)
    state.run_dir = str(tmp_path)
    try:
        child._launch_child([], child._build_child_env(), state, str(probe), _SRC)
    except (FileNotFoundError, PermissionError, OSError):
        return
    raise AssertionError("a nonexistent $%s was silently ignored" % PYTHON_ENV_VAR)


def test_a_child_spawned_this_way_can_import_the_repo(tmp_path):
    """The interpreter change must not disturb the PYTHONPATH pin (Finding B): the child
    still imports ``main.launcher`` from the src dir it was handed."""
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONPATH"] = _SRC
    out = subprocess.run(
        [resolve_child_python(), "-c",
         "import main.launcher.child as c; print(c.__file__)"],
        capture_output=True, text=True, env=env, timeout=120,
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip().startswith(_SRC), out.stdout
