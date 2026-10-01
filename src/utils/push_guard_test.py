"""The push guard (`gen3_push_guard_v1`) in a throwaway repo: an HONEST rebase passes, and the
soft-reset accident of 2026-10-01 (`aebae9a1`: a squash onto a MOVED origin/main that carried the
worktree's stale copies of other commits' files) is REFUSED, naming the stale files. Each case fails on
revert of the rule."""
from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from utils.push_guard import check, main


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                          text=True).stdout


@pytest.fixture
def repo(tmp_path):
    """origin (bare) + a clone; main has a.txt; the clone's branch `unit` is created from main, then
    main gains a commit by someone else (other.txt, and a change to a.txt)."""
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    seed = tmp_path / "seed"
    subprocess.run(["git", "clone", "-q", str(origin), str(seed)], check=True)
    for k, v in (("user.email", "t@t"), ("user.name", "t")):
        _git(seed, "config", k, v)
    (seed / "a.txt").write_text("a1\n")
    _git(seed, "add", "-A")
    _git(seed, "commit", "-q", "-m", "a")
    _git(seed, "push", "-q", "origin", "HEAD:main")
    work = tmp_path / "work"
    subprocess.run(["git", "clone", "-q", str(origin), str(work)], check=True)
    for k, v in (("user.email", "t@t"), ("user.name", "t")):
        _git(work, "config", k, v)
    _git(work, "checkout", "-q", "-b", "unit", "origin/main")
    (work / "mine.txt").write_text("mine\n")
    _git(work, "add", "-A")
    _git(work, "commit", "-q", "-m", "wip mine")
    # someone else lands on main meanwhile
    (seed / "other.txt").write_text("other\n")
    (seed / "a.txt").write_text("a2\n")
    _git(seed, "add", "-A")
    _git(seed, "commit", "-q", "-m", "other")
    _git(seed, "push", "-q", "origin", "HEAD:main")
    _git(work, "fetch", "-q", "origin")
    return work


def test_an_honest_rebase_passes(repo):
    _git(repo, "rebase", "-q", "origin/main")
    assert check(str(repo)) == []
    assert main(["--repo", str(repo)]) == 0


def test_the_soft_reset_accident_is_REFUSED_naming_the_stale_files(repo, capsys):
    _git(repo, "reset", "-q", "--soft", "origin/main")          # parent = the NEW main, tree = stale
    _git(repo, "commit", "-q", "-m", "squashed unit")
    assert check(str(repo)) == ["a.txt", "other.txt"]            # would revert a2 and delete other.txt
    assert main(["--repo", str(repo)]) == 1
    assert "REFUSED" in capsys.readouterr().err


def test_land_sh_runs_the_guard_before_the_push():
    from utils.paths import repo_path
    src = Path(repo_path("scripts", "land.sh")).read_text()
    assert src.index("utils.push_guard") < src.index('git push -q origin "$BRANCH":main')
