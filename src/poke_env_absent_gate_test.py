"""There is NO `poke_env` package in our tree, none installed in our interpreter, and neither env file installs one.

Sits at the `src/` root beside `poke_env_import_gate_test.py`, because its subject is the whole tree.

## Why

Until T27 P6 (2026-10-08) this repo VENDORED a fork of poke-env at `src/poke_env/`, and its gate
(`poke_env_fork_gate_test.py`, retired with it) guarded the one hazard a vendored copy carries: an INSTALLED
upstream `poke-env` shadowing the fork by `sys.path` order, silently. P6 deleted the fork — the training, eval,
live-play and offline stacks are Rust end to end — so the guarantee is now the simpler one stated above:

* no directory named `poke_env` holding an `__init__.py`, and no `poke_env.py`, anywhere under `src/`, `tools/` or
  `scripts/` (a re-vendored copy, or one restored from history, fails here);
* `import poke_env` does not resolve in THIS interpreter (`gen3ai_torch28`): with no package there is nothing for a
  stray import to reach, and the shrink-only import gate (`poke_env_import_gate_test.py`) plus the run-time blocker
  (`utils/poke_env_blocker.py`) keep our code from trying;
* neither `environment.yml` nor `environment_torch28.yml` installs `poke-env` (a requirement LINE, not the prose
  that explains its absence).

The ONE place poke-env still runs is the Metamon peer script (`src/main/anchors/peer_scripts/metamon_side.py`), which
imports UPSTREAM poke-env inside Metamon's OWN interpreter (a separate process with an empty `PYTHONPATH`); it is the
import gate's one permanent allowlist entry and is untouched here.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from utils.paths import repo_root

pytestmark = pytest.mark.static

#: The trees our code lives in (the Showdown submodule, `models/` and caches are not ours).
SCANNED = ("src", "tools", "scripts")
_SKIP_DIRS = {"__pycache__", "node_modules", "target", ".git", ".mypy_cache", ".pytest_cache"}


def poke_env_packages(root: Path) -> list[Path]:
    """Every `poke_env` package directory (one holding `__init__.py`) and every `poke_env.py` under ``SCANNED``."""
    found: list[Path] = []
    for top in SCANNED:
        base = root / top
        if not base.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
            here = Path(dirpath)
            if here.name == "poke_env" and "__init__.py" in filenames:
                found.append(here.relative_to(root))
            if "poke_env.py" in filenames:
                found.append((here / "poke_env.py").relative_to(root))
    return sorted(found)


def test_no_poke_env_package_in_our_tree() -> None:
    found = poke_env_packages(repo_root())
    assert not found, (f"a `poke_env` package is back in our tree: {[str(p) for p in found]} — the vendored fork was "
                       "deleted in T27 P6 (one Rust stack); fix the importer instead of restoring it")


def test_the_scan_has_teeth(tmp_path: Path) -> None:
    """A re-vendored package and a stray single-file module are both found; a non-package dir is not."""
    (tmp_path / "src" / "poke_env").mkdir(parents=True)
    (tmp_path / "src" / "poke_env" / "__init__.py").write_text("")
    (tmp_path / "tools" / "x").mkdir(parents=True)
    (tmp_path / "tools" / "x" / "poke_env.py").write_text("")
    (tmp_path / "scripts" / "poke_env").mkdir(parents=True)          # data dir, no __init__: not a package
    assert poke_env_packages(tmp_path) == [Path("src/poke_env"), Path("tools/x/poke_env.py")]


_INSTALLED_SPEC = ("import importlib.machinery, sys\n"
                   "paths = [p for p in sys.path if p.rstrip('/').endswith(('site-packages', 'dist-packages'))]\n"
                   "assert paths, sys.path\n"
                   "s = importlib.machinery.PathFinder.find_spec('poke_env', paths)\n"
                   "print(s.origin or list(s.submodule_search_locations) if s else '')\n")


def installed_poke_env(extra_path: str = "") -> str:
    """Where `poke_env` resolves among this interpreter's INSTALLED (site-packages) entries; '' when nowhere."""
    env = {"PATH": "/usr/bin:/bin", **({"PYTHONPATH": extra_path} if extra_path else {})}
    r = subprocess.run([sys.executable, "-c", _INSTALLED_SPEC], env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
    return r.stdout.strip()


def test_poke_env_is_not_installed_in_our_interpreter() -> None:
    """No `poke_env` in this interpreter's site-packages. Only the INSTALLED entries are searched: the editable
    install's `.pth` names the MAIN checkout's `src/`, which is not the tree under test (a worktree's gate would
    otherwise read main's state), and our own tree is the scan above."""
    where = installed_poke_env()
    assert where == "", (f"`poke_env` is installed at {where} in {sys.executable} — `pip uninstall poke-env` "
                         "(it is deliberately absent from both env files)")


def test_the_installed_check_has_teeth(tmp_path: Path) -> None:
    """A decoy `site-packages` on the path holding a `poke_env` package is found."""
    decoy = tmp_path / "site-packages"
    (decoy / "poke_env").mkdir(parents=True)
    (decoy / "poke_env" / "__init__.py").write_text("")
    assert installed_poke_env(str(decoy)) == str(decoy / "poke_env" / "__init__.py")


@pytest.mark.parametrize("env_file", ["environment.yml", "environment_torch28.yml"])
def test_environment_files_do_not_install_poke_env(env_file: str) -> None:
    """Text-scanned (no `yaml` dependency): a requirement LINE, never the prose that explains the removal."""
    path = repo_root() / env_file
    assert path.is_file(), f"{env_file} not found at {path}"
    offenders = [ln.strip() for ln in path.read_text().splitlines()
                 if ln.strip().startswith("- ") and ln.strip()[2:].lstrip().startswith(("poke-env", "poke_env"))]
    assert not offenders, f"{env_file} installs poke-env: {offenders} — it is deliberately absent (T27)"
