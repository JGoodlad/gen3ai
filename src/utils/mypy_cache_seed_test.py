"""``utils.mypy_cache_seed`` on a REAL mypy over a tiny stdlib-only package (a few seconds cold).

Load-bearing pins, each FAILS on revert of what it guards:
* a seed taken where ``a.f`` returns ``int``, applied to a checkout at ANOTHER PATH where it returns
  ``str``, still reports the error in the UNCHANGED dependent ``b.py`` — a seed can never hide an error;
* a checkout that already holds a cache is never re-seeded (its own entries are never overwritten);
* a corrupt source shard discards the whole seed (a cold run, not a red gate from a torn file);
* ``GEN3AI_MYPY_CACHE_SEED=0`` turns it off.

``integration``: runs mypy as a subprocess.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from utils import mypy_cache_seed as S

pytestmark = pytest.mark.integration


def _pkg(base: Path, ret: str) -> Path:
    (base / "pkg").mkdir(parents=True, exist_ok=True)
    (base / "pkg" / "__init__.py").write_text("")
    val = "1" if ret == "int" else "'x'"
    (base / "pkg" / "a.py").write_text(f"def f() -> {ret}:\n    return {val}\n")
    (base / "pkg" / "b.py").write_text("from pkg.a import f\n\n\ndef g() -> int:\n    return f() + 1\n")
    return base


def _mypy(base: Path, cache: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "mypy", "--no-color-output", "--cache-dir", str(cache), "pkg"],
                          cwd=str(base), capture_output=True, text=True, timeout=600)


@pytest.fixture(scope="module")
def warm(tmp_path_factory):
    a = _pkg(tmp_path_factory.mktemp("seed_src") / "A", "int")
    r = _mypy(a, a / "cache")
    assert r.returncode == 0, r.stdout + r.stderr
    assert S.has_cache(a / "cache"), sorted(p.name for p in (a / "cache").rglob("*"))
    return a / "cache"


def test_a_seed_from_another_path_cannot_hide_an_interface_error(warm, tmp_path):
    c = _pkg(tmp_path / "C", "str")
    used = S.seed(c / "cache", [warm])
    assert used == warm and S.has_cache(c / "cache")
    r = _mypy(c, c / "cache")
    assert r.returncode == 1 and "pkg/b.py:5" in r.stdout and "Unsupported operand" in r.stdout, r.stdout + r.stderr


def test_a_seed_from_another_path_passes_the_same_source(warm, tmp_path):
    b = _pkg(tmp_path / "B", "int")
    assert S.seed(b / "cache", [warm]) == warm
    r = _mypy(b, b / "cache")
    assert r.returncode == 0, r.stdout + r.stderr


def test_a_warm_cache_is_never_reseeded(warm, tmp_path):
    own = tmp_path / "own"
    assert S.seed(own, [warm]) == warm
    marker = next(own.glob("*/cache.*.db"))
    before = marker.stat().st_mtime_ns
    assert S.seed(own, [warm]) is None
    assert marker.stat().st_mtime_ns == before


def test_a_corrupt_shard_discards_the_whole_seed(warm, tmp_path):
    import shutil

    bad = tmp_path / "bad"
    shutil.copytree(warm, bad)
    next(bad.glob("*/cache.*.db")).write_bytes(b"this is not a sqlite database" * 100)
    notes = []
    assert S.seed(tmp_path / "own", [bad], log=notes.append) is None
    assert not (tmp_path / "own").exists() and not list(tmp_path.glob(".own.seed.*"))
    assert notes and "NOT seeded" in notes[0], notes


def test_seeding_can_be_turned_off_and_skips_a_symlink_and_own(warm, tmp_path, monkeypatch):
    link = tmp_path / "link"
    link.symlink_to(warm)
    assert S.pick_source(tmp_path / "own", [link]) is None
    assert S.pick_source(warm, [warm]) is None
    monkeypatch.setenv(S.SEED_ENV, "0")
    assert S.seed(tmp_path / "own", [warm]) is None
