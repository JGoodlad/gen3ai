"""The push guard (`gen3_push_guard_v1`) in a throwaway repo: an HONEST rebase passes, and the
soft-reset accident of 2026-10-01 (`aebae9a1`: a squash onto a MOVED origin/main that carried the
worktree's stale copies of other commits' files) is REFUSED, naming the stale files. Each case fails on
revert of the rule.

`gen3_push_guard_v2` adds the one exemption: a net-zero RESTORE of a version this branch itself authored and
already landed (the perf agent's false positive) passes, while restoring a file whose main-side content came
from SOMEONE ELSE (or from someone else's later edit on top of ours) is still refused."""
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


# -- the exemption: undoing the branch's OWN landed change ----------------------------------------

def _identify(repo: Path) -> None:
    for k, v in (("user.email", "t@t"), ("user.name", "t")):
        _git(repo, "config", k, v)


def _other_agent_commits(origin: Path, tmp_path: Path, **files: str) -> None:
    """Another agent lands a commit on origin/main touching `files` (name=content)."""
    other = tmp_path / "other"
    subprocess.run(["git", "clone", "-q", str(origin), str(other)], check=True)
    _identify(other)
    for name, content in files.items():
        (other / name).write_text(content)
    _git(other, "add", "-A")
    _git(other, "commit", "-q", "-m", "someone else")
    _git(other, "push", "-q", "origin", "HEAD:main")


@pytest.fixture
def landed(tmp_path):
    """origin + a clone whose branch `unit` was created from main (a.txt, b.txt); its first commit C1 changed
    b.txt and LANDED (pushed to origin/main as it stood). Returns (work clone, origin path, tmp_path)."""
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "-q", "--bare", "-b", "main", str(origin)], check=True)
    seed = tmp_path / "seed"
    subprocess.run(["git", "clone", "-q", str(origin), str(seed)], check=True)
    _identify(seed)
    (seed / "a.txt").write_text("a1\n")
    (seed / "b.txt").write_text("b1\n")
    _git(seed, "add", "-A")
    _git(seed, "commit", "-q", "-m", "base")
    _git(seed, "push", "-q", "origin", "HEAD:main")
    work = tmp_path / "work"
    subprocess.run(["git", "clone", "-q", str(origin), str(work)], check=True)
    _identify(work)
    _git(work, "checkout", "-q", "-b", "unit", "origin/main")
    (work / "b.txt").write_text("b-mine\n")
    _git(work, "add", "-A")
    _git(work, "commit", "-q", "-m", "C1 - my change to b.txt")
    _git(work, "push", "-q", "origin", "HEAD:main")                 # C1 LANDS
    _git(work, "fetch", "-q", "origin")
    return work, origin, tmp_path


def _restore_b(work: Path) -> None:
    """C2: b.txt back to its pre-branch bytes - NET-ZERO against the creation base."""
    (work / "b.txt").write_text("b1\n")
    _git(work, "add", "-A")
    _git(work, "commit", "-q", "-m", "C2 - undo C1")


def test_a_net_zero_restore_of_the_branchs_own_landed_change_passes(landed):
    """The false positive: C1 landed, C2 puts b.txt back. b.txt is in diff(origin/main, HEAD) but not in
    diff(creation base, HEAD), so the old rule called it a stale copy; main's b.txt is the branch's OWN C1."""
    work, _origin, _tmp = landed
    _restore_b(work)
    base = _git(work, "reflog", "show", "--format=%H", "unit").split()[-1]
    assert _git(work, "diff", "--name-only", base, "HEAD").split() == [], "b.txt is net-zero vs the creation base"
    assert _git(work, "diff", "--name-only", "origin/main", "HEAD").split() == ["b.txt"]
    assert check(str(work)) == []
    assert main(["--repo", str(work)]) == 0


def test_a_net_zero_restore_still_passes_after_an_honest_rebase_over_unrelated_main_work(landed):
    work, origin, tmp_path = landed
    _other_agent_commits(origin, tmp_path, **{"a.txt": "a2\n"})
    _git(work, "fetch", "-q", "origin")
    _restore_b(work)
    _git(work, "rebase", "-q", "origin/main")
    assert check(str(work)) == []


def test_restoring_over_SOMEONE_ELSES_later_change_to_the_same_file_is_REFUSED(landed, capsys):
    """C1 landed, then another agent edited the SAME file on main. A net-zero restore now reverts THEIR work:
    main's b.txt matches none of the versions this branch authored."""
    work, origin, tmp_path = landed
    _other_agent_commits(origin, tmp_path, **{"b.txt": "b-theirs-after-mine\n"})
    _git(work, "fetch", "-q", "origin")
    _git(work, "rebase", "-q", "origin/main")
    _restore_b(work)
    assert check(str(work)) == ["b.txt"]
    assert main(["--repo", str(work)]) == 1
    assert "REFUSED" in capsys.readouterr().err


def test_the_soft_reset_accident_after_a_landed_commit_is_still_REFUSED(landed):
    """The 2026-10-01 shape with an earlier landed commit in the history: main moved (another agent's a.txt),
    the agent soft-resets onto it and re-commits its OLD tree - stale a.txt, which is not any version this
    branch ever committed."""
    work, origin, tmp_path = landed
    _other_agent_commits(origin, tmp_path, **{"a.txt": "a2\n"})
    _git(work, "fetch", "-q", "origin")
    (work / "mine.txt").write_text("mine\n")
    _git(work, "add", "-A")
    _git(work, "commit", "-q", "-m", "wip mine")
    _git(work, "reset", "-q", "--soft", "origin/main")              # parent = the NEW main, tree = stale
    _git(work, "commit", "-q", "-m", "squashed unit")
    assert check(str(work)) == ["a.txt"]


def test_only_commit_tips_count_as_authored(landed):
    """A reset onto a moved ref is a reflog tip too, but it authored nothing: it is not in the list."""
    from utils.push_guard import own_commit_tips

    work, origin, tmp_path = landed
    _other_agent_commits(origin, tmp_path, **{"a.txt": "a2\n"})
    _git(work, "fetch", "-q", "origin")
    authored = own_commit_tips(str(work), "unit")
    _git(work, "reset", "-q", "--hard", "origin/main")
    assert own_commit_tips(str(work), "unit") == authored
    assert own_commit_tips(str(work), "no-such-branch-xyzzy") == []
