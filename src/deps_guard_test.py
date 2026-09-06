"""The fresh-worktree guard: ONE message about the checkout, not ~15 fake regressions.

A linked git worktree materializes the `deps/pokemon-showdown` submodule PATH but not its contents,
and the build artifacts inside it are gitignored — so a worktree that has not been bootstrapped runs
the suite with a bridge that cannot load. Every battle-backed test then dies inside Node on
`Cannot find module '.../deps/pokemon-showdown/dist/sim/...'`, and pytest presents that as failures
in `utils/bridge`, `agents/battle` and the obs-golden linchpin — which is indistinguishable, at a
glance, from a real regression in the code under test. Three agents each lost a cycle to it on
2026-09-06. The root `conftest.py` now refuses the session instead.

WHY THIS TEST RUNS THE REAL CONFTEST IN A SUBPROCESS. The guard's whole job is to change what a
pytest SESSION does, and a session's startup cannot be observed from inside a test that is already
running in one — by the time this module is imported, `pytest_sessionstart` has long since returned
(and returned CLEAN, since this checkout is bootstrapped, or the suite would not be running). So the
only honest way to test it is to start a second pytest against a tree that is missing the file.

It copies the ACTUAL `conftest.py` rather than restating the guard, so a change to the message, the
probe paths, or the opt-out is exercised here rather than drifting away from a copy. That is safe
because the file's top-level imports are `os` alone — the torch/contention reads are all deferred
into functions — so it stands up in a bare temp directory with nothing on the path.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from utils.paths import repo_root

_CONFTEST = repo_root() / "conftest.py"
_PROBE = Path("deps") / "pokemon-showdown" / "dist" / "sim" / "index.js"
_NODE_MODULES = Path("deps") / "pokemon-showdown" / "node_modules"


def _tree(tmp_path: Path, *, present: bool) -> Path:
    """A minimal pytest tree carrying the real root conftest and one trivial test."""
    shutil.copy(_CONFTEST, tmp_path / "conftest.py")
    (tmp_path / "canary_test.py").write_text("def test_canary():\n    assert True\n")
    if present:
        (tmp_path / _PROBE).parent.mkdir(parents=True)
        (tmp_path / _PROBE).write_text("// stand-in for the compiled Showdown entry point\n")
        (tmp_path / _NODE_MODULES).mkdir(parents=True)
    return tmp_path


def _run_pytest(cwd: Path, **env_overrides) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "GEN3AI_SKIP_DEPS_GUARD"}
    env.update(env_overrides)
    return subprocess.run(
        # `-p no:cacheprovider` keeps the temp tree free of a .pytest_cache; `-p no:randomly` is not
        # needed (one test), and rootdir is the temp dir because that is where cwd is.
        [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", str(cwd)],
        cwd=str(cwd), env=env, capture_output=True, text=True, timeout=300,
    )


def test_a_missing_submodule_fails_the_session_with_one_message(tmp_path):
    """The whole point: ONE actionable line about the checkout, and ZERO test failures."""
    res = _run_pytest(_tree(tmp_path, present=False))
    out = res.stdout + res.stderr

    # pytest.UsageError => exit code 4 (USAGE_ERROR), which is distinct from 1 (tests failed).
    # That distinction is the guard's contract: "your checkout is wrong" must not be reported
    # through the same channel as "your code is wrong".
    assert res.returncode == 4, f"expected UsageError exit 4, got {res.returncode}\n{out}"

    assert "deps/pokemon-showdown is not usable" in out, out
    assert "./scripts/bootstrap.sh" in out, "the message must name the one-command fix\n" + out
    assert "git submodule update --init" in out, "and the manual steps\n" + out
    assert "GEN3AI_SKIP_DEPS_GUARD=1" in out, "and the opt-out\n" + out
    # Both artifacts are named, because they are restored by DIFFERENT steps.
    assert "dist/sim/index.js" in out and "node_modules" in out, out

    # The canary must never have run: the session is refused before collection, so the reader is
    # not left correlating one usage error against a wall of failures.
    assert "canary" not in out, f"the session should not have collected anything\n{out}"
    assert " failed" not in out, f"the guard must REPLACE test failures, not add to them\n{out}"


def test_a_bootstrapped_tree_runs_normally(tmp_path):
    """The guard is invisible once the checkout is finished — the control arm."""
    res = _run_pytest(_tree(tmp_path, present=True))
    out = res.stdout + res.stderr
    assert res.returncode == 0, out
    assert "1 passed" in out, out
    assert "deps/pokemon-showdown is not usable" not in out, out


def test_the_opt_out_runs_the_suite_anyway(tmp_path):
    """A pure-unit CI with no submodule at all opts out explicitly rather than being blocked."""
    res = _run_pytest(_tree(tmp_path, present=False), GEN3AI_SKIP_DEPS_GUARD="1")
    out = res.stdout + res.stderr
    assert res.returncode == 0, out
    assert "1 passed" in out, out
    assert "deps/pokemon-showdown is not usable" not in out, out


def test_this_checkout_satisfies_the_guard():
    """A bootstrapped worktree is the precondition for every bridge-backed test in the tree, so if
    this one fails the rest of the suite's battle-backed results are not worth reading. It cannot
    fail while the guard is armed (the session would have been refused) — it is the assertion that
    keeps the fact stated where a reader looks, and it fails honestly under the opt-out."""
    for rel in (_PROBE, _NODE_MODULES):
        assert (repo_root() / rel).exists(), (
            f"{rel} missing — run ./scripts/bootstrap.sh (see the root CLAUDE.md, Git Worktree Setup)")
