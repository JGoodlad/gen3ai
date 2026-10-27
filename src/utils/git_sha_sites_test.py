"""Every site that resolves a RECORDED commit sha goes through ``utils.git_sha`` (the history rewrite must not break a
resume, a ``checkargs`` verdict, a halt clear or a pinned-argv check).

Each test feeds a sha that names no object anywhere (``fake_old``) through a fake map to a real commit of a tiny repo:
it passes only when the site asks the map, so it fails on revert. With an empty map every site is the old git call
(the existing tests of each site); the last tests pin that.
"""
from __future__ import annotations

import pytest

from main import checkargs
from main.launcher import pinned_argv as pa
from main.launcher import worktree as wt
from main.live import halt
from utils import git_sha as G
from utils import paths
from utils.git_sha_testkit import fake_old, use_map
from utils.history_rewrite.synthetic_repo_testkit import git

OLD1, OLD2 = fake_old("a"), fake_old("b")


@pytest.fixture()
def repo(tmp_path):
    """c1 (a_test.py) <- c2 (b_test.py); HEAD = c2. Returns ``{path, c1, c2}``."""
    r = tmp_path / "tiny"
    r.mkdir()
    git(r, "init", "-q")
    out = {"path": str(r)}
    for n, name in ((1, "a_test.py"), (2, "b_test.py")):
        (r / name).write_text("def test_x(): pass\n")
        git(r, "add", "-A")
        git(r, "commit", "-q", "-m", f"c{n}")
        out[f"c{n}"] = git(r, "rev-parse", "HEAD")
    return out


@pytest.fixture()
def mapped(repo, tmp_path, monkeypatch):
    use_map(monkeypatch, tmp_path, {OLD1: repo["c1"], OLD2: repo["c2"]})
    return repo


# ---------------------------------------------------------------------------------------------- the helpers
def test_commit_of_and_current_answer_an_old_sha_with_the_new_one(mapped):
    assert G.commit_of(OLD1, repo=mapped["path"]) == mapped["c1"]
    assert G.commit_of(OLD1[:9], repo=mapped["path"]) == mapped["c1"]
    assert G.current(OLD1[:9], repo=mapped["path"]) == mapped["c1"]
    assert G.commit_of("HEAD", repo=mapped["path"]) == mapped["c2"]                # a ref is git's alone
    assert G.commit_of(mapped["c1"][:8], repo=mapped["path"]) == mapped["c1"]
    assert G.commit_of(fake_old("c"), repo=mapped["path"]) is None                 # in neither
    assert G.current(mapped["c1"][:8], repo=mapped["path"]) == mapped["c1"][:8]    # live and unmapped: unchanged text


def test_is_ancestor_reads_recorded_shas_and_is_none_when_git_cannot_say(mapped):
    p = mapped["path"]
    assert G.is_ancestor(OLD1, mapped["c2"], repo=p) is True
    assert G.is_ancestor(OLD2, OLD1, repo=p) is False
    assert G.is_ancestor(OLD1, "HEAD", repo=p) is True
    assert G.is_ancestor(fake_old("c"), "HEAD", repo=p) is None


def test_with_an_empty_map_the_helpers_are_plain_git(repo, tmp_path, monkeypatch):
    use_map(monkeypatch, tmp_path, {})
    assert G.commit_of(repo["c1"][:10], repo=repo["path"]) == repo["c1"]
    assert G.commit_of(OLD1, repo=repo["path"]) is None
    assert G.current(OLD1, repo=repo["path"]) == OLD1 and G.current("HEAD", repo=repo["path"]) == "HEAD"
    assert G.is_ancestor(repo["c1"], repo["c2"], repo=repo["path"]) is True


# ---------------------------------------------------------------------------------------------- the launcher
def test_worktree_rev_parse_and_resolve_commit_read_an_old_pin(mapped):
    assert wt._rev_parse(OLD1, repo_root=mapped["path"]) == mapped["c1"]
    sha, subject = wt.resolve_commit(OLD1[:12], repo_root=mapped["path"])
    assert (sha, subject) == (mapped["c1"], "c1")
    assert wt._same_commit(OLD1, mapped["c1"][:8], repo_root=mapped["path"])
    assert not wt._same_commit(OLD1, mapped["c2"], repo_root=mapped["path"])


def test_a_resume_pins_to_the_new_sha_of_the_checkpoints_old_git_hash(mapped, monkeypatch):
    """The sidecar's `git_hash` is immutable: a resume of an old run must pin the commit it became."""
    monkeypatch.setattr(wt, "_read_checkpoint_git_hash", lambda model_path: OLD1)
    d = wt.resolve_pin(model_path="models/r/checkpoints/c.zip", run_dir=None, pin_commit=None, sync_to_main=False,
                       repo_root=mapped["path"])
    assert (d.sha, d.source) == (mapped["c1"], "checkpoint")
    # a same-run restart whose --pin-commit is the NEW sha of the checkpoint's OLD hash is the same pin (no refusal)
    monkeypatch.setattr(wt, "is_same_run_checkpoint", lambda m, r: True)
    d = wt.resolve_pin(model_path="models/r/checkpoints/c.zip", run_dir="models/r", pin_commit=mapped["c1"],
                       sync_to_main=False, repo_root=mapped["path"])
    assert d.sha == mapped["c1"]


def test_the_run_worktree_is_added_at_the_new_sha(mapped, monkeypatch):
    import subprocess
    seen, real = [], subprocess.run

    class _R:
        returncode, stderr = 1, "stop here"

    def fake(argv, **kw):          # `wt.subprocess` is the shared module: intercept only the worktree add
        return (seen.append(argv) or _R()) if argv[1:3] == ["worktree", "add"] else real(argv, **kw)

    monkeypatch.setattr(wt, "get_repo_root", lambda: mapped["path"])
    monkeypatch.setattr(subprocess, "run", fake)
    with pytest.raises(RuntimeError, match="stop here"):
        wt._create_run_worktree(OLD1)
    assert seen[0][:4] == ["git", "worktree", "add", "--detach"] and seen[0][-1] == mapped["c1"]


# ---------------------------------------------------------------------------------------------- checkargs / halt
def test_pin_predates_commit_reads_an_old_pin(mapped, monkeypatch):
    monkeypatch.setattr(paths, "repo_root", lambda: mapped["path"])
    assert checkargs.pin_predates_commit(OLD1, mapped["c2"]) is True
    assert checkargs.pin_predates_commit(OLD2, OLD1) is False
    assert checkargs.pin_predates_commit(OLD1, OLD2) is True                       # both recorded
    assert checkargs.pin_predates_commit(fake_old("c"), mapped["c2"]) is None      # git cannot say


def test_halt_fixed_by_an_old_sha_is_the_new_commit(mapped):
    fix = halt.check_fix_commit(OLD2[:10], cwd=mapped["path"])
    assert fix["commit"] == mapped["c2"] and fix["tests"] == ["b_test.py"]
    with pytest.raises(halt.ClearRefused, match="not a commit"):
        halt.check_fix_commit(fake_old("c"), cwd=mapped["path"])


# ---------------------------------------------------------------------------------------------- the pinned parser
def test_pinned_argv_reads_an_old_pin(mapped, monkeypatch):
    p = mapped["path"]
    monkeypatch.setattr(pa, "_CACHE", {})
    assert pa.differs_from_head(OLD2, p) is False and pa.differs_from_head(OLD1, p) is True
    assert pa._tree_paths(OLD1, p) == []                                           # resolved, none of the paths exist
    with pytest.raises(pa._Unavailable, match="does not name a commit"):
        pa._tree_paths(fake_old("c"), p)
    with pytest.raises(pa._Unavailable, match=f"commit {mapped['c1'][:8]} carries none"):
        pa.materialise(OLD1, p)
