"""THE UNCOLLECTED-TESTS GATE: every test file in the repo sits where the routine gate collects it.

Sits at the `src/` root beside the other whole-tree gates, because its subject is the whole tree.

## Why

`pytest.ini` names the collected roots (`testpaths`) and the filename patterns (`python_files`); the ship gate runs
exactly those roots (`utils.ship_gate.collected_roots`). A test file outside them, or one whose name no pattern
matches, is never run by anything and nobody sees it not run. Found 2026-11-17: the 7 tests that protect BOUNDARY-1's
banked runs from thinning lived under `designs/` (not a root) for weeks; `tools/`'s 77 tests were not in the gate's
argv; `tests/test_teams_validity.py` matched no pattern and no root. A protection test that never runs is not a protection.

## What it asserts

Every tracked file whose name looks like a test (`test_*.py`, `*_test.py`, `*_tests.py`) both MATCHES a `python_files`
pattern and lies under a `testpaths` root. The allowlist is EMPTY: move the file under a root, rename a script that is
not a test (`*_check.py`, `*_probe.py`), or add the root to `pytest.ini` (it is then gated too).

Static, ~0.1 s. No opt-out.
"""
from __future__ import annotations

import configparser
import fnmatch
import os
import subprocess
from pathlib import Path
from typing import Iterable, List, Sequence

import pytest

from utils.paths import repo_root

pytestmark = pytest.mark.static

#: What LOOKS like a test file to a human (and to pytest's defaults). Wider than `python_files` on purpose: a name that
#: looks like a test and matches no `python_files` pattern is exactly the file that never runs.
TESTISH = ("test_*.py", "*_test.py", "*_tests.py")
_SKIP_DIRS = {"__pycache__", "node_modules", "target", ".git", ".mypy_cache", ".pytest_cache", "deps"}


def pytest_config(root: Path) -> tuple[List[str], List[str]]:
    """``(testpaths, python_files)`` from ``pytest.ini``."""
    cp = configparser.ConfigParser(interpolation=None)
    cp.read(root / "pytest.ini")
    return cp.get("pytest", "testpaths", fallback="").split(), cp.get("pytest", "python_files", fallback="").split()


def tracked_files(root: Path) -> List[str]:
    """Repo-relative paths git tracks (the submodule's files are not listed); a tarball with no git is walked."""
    try:
        out = subprocess.run(["git", "-C", str(root), "ls-files"], capture_output=True, text=True, check=True,
                             timeout=60).stdout
        return out.splitlines()
    except (OSError, subprocess.SubprocessError):
        found: List[str] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            found += [(Path(dirpath) / f).relative_to(root).as_posix() for f in filenames]
        return found


def uncollected(files: Iterable[str], testpaths: Sequence[str], python_files: Sequence[str]) -> List[str]:
    """Each test-looking file that is outside every root, or that no `python_files` pattern matches, with the reason."""
    roots = tuple(t.rstrip("/") + "/" for t in testpaths)
    bad: List[str] = []
    for rel in sorted(files):
        name = rel.rsplit("/", 1)[-1]
        if not rel.endswith(".py") or not any(fnmatch.fnmatch(name, p) for p in TESTISH):
            continue
        if not rel.startswith(roots):
            bad.append(f"{rel}: outside every collected root {list(testpaths)}")
        elif not any(fnmatch.fnmatch(name, p) for p in python_files):
            bad.append(f"{rel}: its name matches no python_files pattern {list(python_files)}")
    return bad


def test_every_test_file_is_in_a_collected_root() -> None:
    root = repo_root()
    testpaths, python_files = pytest_config(root)
    assert testpaths and python_files, "pytest.ini names no testpaths / python_files"
    bad = uncollected(tracked_files(root), testpaths, python_files)
    assert not bad, ("test files the routine gate never runs (move each under a collected root, or rename it if it is "
                     "not a test; the allowlist is empty):\n  " + "\n  ".join(bad))


def test_the_ship_gate_collects_exactly_the_testpaths() -> None:
    """The roots this gate checks against are the roots the ship gate runs: one list, so they cannot drift."""
    from utils import ship_gate

    testpaths, _ = pytest_config(repo_root())
    assert ship_gate.collected_roots() == [t.rstrip("/") + "/" for t in testpaths]


def test_the_scan_has_teeth() -> None:
    """A test under `designs/`, a `test_x.py` no pattern matches, and a `*_tests.py` are each found; a script is not."""
    files = ["src/ok_test.py", "src/sub/ok_integration_test.py", "tools/ok_test.py", "designs/x/record_test.py",
             "tests/test_teams.py", "src/test_nomatch.py", "src/many_tests.py", "designs/x/flip_probe.py",
             "src/not_python_test.txt", "conftest.py"]
    got = uncollected(files, ["src", "tools"], ["*_test.py", "*_integration_test.py"])
    assert [g.split(":")[0] for g in got] == ["designs/x/record_test.py", "src/many_tests.py", "src/test_nomatch.py",
                                              "tests/test_teams.py"]
    assert "outside every collected root" in got[0] and "matches no python_files" in got[1]
    assert uncollected(files, ["src", "tools", "designs/x", "tests"], ["*_test.py", "test_*.py", "*_tests.py"]) == []
