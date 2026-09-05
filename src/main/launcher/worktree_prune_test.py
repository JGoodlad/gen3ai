"""The startup prune must only remove a DEAD launcher's worktree.

🚨 THE INCIDENT THIS FILE EXISTS FOR (2026-09-05). `_prune_stale_launcher_worktrees` used to
force-remove every `launcher-*` worktree at startup with no liveness check. A one-second
validation command — `python -m main.launcher --pin-commit deadbeef --steps 1`, which never
even got as far as creating its own worktree — deleted the isolated worktree of a LIVE
production run. The run kept going on its open file descriptors, looked healthy for hours, and
died at its next 3 h periodic restart when the launcher re-exec'd the child out of a directory
that no longer existed (exit 2, no final_model).

The rule now: a worktree is removed ONLY when its owner is provably gone. Ambiguity keeps it.

Run: python -m pytest src/main/launcher/worktree_prune_test.py -q
(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""

import json
import os
import subprocess
import time

import pytest

import main.launcher.worktree as wt


# ---------------------------------------------------------------------------------------
# A real (tiny) git repo — the prune IS `git worktree list/remove`, so a fake tests nothing.
# Same shape as pin_commit_test's fixture (its `repo` helper), reused rather than imported so
# neither file's fixture can be changed out from under the other.
# ---------------------------------------------------------------------------------------

def _git(repo, *args):
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True,
    ).stdout.strip()


@pytest.fixture
def repo(tmp_path):
    """A 1-commit repo; yields its root path."""
    root = tmp_path / "repo"
    root.mkdir()
    _git(root, "init", "-q", "-b", "main")
    _git(root, "config", "user.email", "t@t")
    _git(root, "config", "user.name", "t")
    # `_create_run_worktree` symlinks deps/pokemon-showdown in, so deps/ must exist.
    (root / "deps").mkdir()
    (root / "deps" / ".keep").write_text("")
    (root / "marker.txt").write_text("first")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "commit 1")
    return str(root)


def _make_worktree(repo_root, monkeypatch):
    """Create a launcher worktree in `repo_root` and return (path, cleanup)."""
    monkeypatch.setattr(wt, "get_repo_root", lambda *a, **k: repo_root)
    sha = _git(repo_root, "rev-parse", "HEAD")
    _train, src_dir, cleanup = wt._create_run_worktree(sha, run_dir="models/pretend_run")
    return os.path.dirname(src_dir), cleanup


def _listed(repo_root):
    out = _git(repo_root, "worktree", "list", "--porcelain")
    return [ln[len("worktree "):] for ln in out.splitlines() if ln.startswith("worktree ")]


# ---------------------------------------------------------------------------------------
# (a) a worktree owned by a LIVE process survives — the incident's exact case
# ---------------------------------------------------------------------------------------

def test_a_a_live_owners_worktree_is_never_pruned(repo, monkeypatch):
    path, cleanup = _make_worktree(repo, monkeypatch)
    try:
        owner = json.load(open(wt._owner_path(path)))
        assert owner["pid"] == os.getpid(), "the creator must claim the worktree"
        assert owner["run_dir"] == "models/pretend_run"
        assert owner["proc_starttime"], "the pid-reuse guard needs /proc starttime recorded"

        lines = []
        wt._prune_stale_launcher_worktrees(repo, report=lines.append)

        assert os.path.isdir(path), "a LIVE owner's worktree was deleted — the 2026-09-05 bug"
        assert path in _listed(repo)
        assert any("KEPT" in ln and str(os.getpid()) in ln for ln in lines), (
            f"the skip must be announced and name the owning pid; got {lines}")
    finally:
        cleanup()


# ---------------------------------------------------------------------------------------
# (b) a worktree owned by a DEAD pid IS removed — the prune still does its job
# ---------------------------------------------------------------------------------------

def _a_dead_pid() -> int:
    """A pid that has certainly exited: spawn `true`, reap it, reuse its number."""
    p = subprocess.Popen(["true"])
    p.wait()
    return p.pid


def test_b_a_dead_owners_worktree_is_removed(repo, monkeypatch):
    path, cleanup = _make_worktree(repo, monkeypatch)
    try:
        dead = _a_dead_pid()
        with open(wt._owner_path(path), "w") as f:
            json.dump({"pid": dead, "proc_starttime": "12345"}, f)

        lines = []
        wt._prune_stale_launcher_worktrees(repo, report=lines.append)

        assert not os.path.isdir(path), "a dead owner's worktree must be collected"
        assert path not in _listed(repo)
        assert not os.path.exists(wt._owner_path(path)), "the owner file goes with it"
        assert any("pruned" in ln for ln in lines)
    finally:
        cleanup()


def test_b_a_reused_pid_is_not_mistaken_for_the_owner(repo, monkeypatch):
    """A LIVE pid whose /proc starttime differs is a DIFFERENT process — prune it."""
    path, cleanup = _make_worktree(repo, monkeypatch)
    try:
        with open(wt._owner_path(path), "w") as f:
            json.dump({"pid": os.getpid(), "proc_starttime": "0"}, f)  # cannot be ours
        wt._prune_stale_launcher_worktrees(repo, report=lambda _l: None)
        assert not os.path.isdir(path)
    finally:
        cleanup()


def test_b_an_unverifiable_owner_is_KEPT(repo, monkeypatch):
    """No recorded starttime ⇒ the reuse check cannot run ⇒ fail SAFE, keep the worktree."""
    path, cleanup = _make_worktree(repo, monkeypatch)
    try:
        with open(wt._owner_path(path), "w") as f:
            json.dump({"pid": os.getpid()}, f)          # no proc_starttime
        wt._prune_stale_launcher_worktrees(repo, report=lambda _l: None)
        assert os.path.isdir(path), "an unverifiable but LIVE pid must not lose its worktree"
    finally:
        cleanup()


# ---------------------------------------------------------------------------------------
# (c) no owner file (a pre-fix worktree): FRESH survives, OLD is collected
# ---------------------------------------------------------------------------------------

def test_c_a_legacy_worktree_with_a_fresh_mtime_survives(repo, monkeypatch):
    path, cleanup = _make_worktree(repo, monkeypatch)
    try:
        os.remove(wt._owner_path(path))                  # pre-fix worktree
        lines = []
        wt._prune_stale_launcher_worktrees(repo, report=lines.append)
        assert os.path.isdir(path), (
            "a pre-fix worktree may belong to a live run started before this fix landed")
        assert any("KEPT" in ln for ln in lines)
    finally:
        cleanup()


def test_c_a_legacy_worktree_older_than_a_day_is_collected(repo, monkeypatch):
    path, cleanup = _make_worktree(repo, monkeypatch)
    try:
        os.remove(wt._owner_path(path))
        old = time.time() - (wt._LEGACY_ORPHAN_MAX_AGE_S + 3600)
        os.utime(path, (old, old))
        wt._prune_stale_launcher_worktrees(repo, report=lambda _l: None)
        assert not os.path.isdir(path), "an abandoned pre-fix worktree must still be collected"
    finally:
        cleanup()


# ---------------------------------------------------------------------------------------
# (d) the owner file must not dirty the worktree's `git status`
# ---------------------------------------------------------------------------------------

def test_d_an_owned_worktree_has_a_clean_git_status(repo, monkeypatch):
    """The record lives BESIDE the tree (`<worktree>.owner.json`), never inside it.

    Inside, it would be untracked in a checkout of a commit whose .gitignore predates it —
    and the child runs git in there.
    """
    path, cleanup = _make_worktree(repo, monkeypatch)
    try:
        assert os.path.exists(wt._owner_path(path)), "the claim must actually be written"
        assert not wt._owner_path(path).startswith(path + os.sep), (
            "the owner file must live OUTSIDE the worktree")
        with_claim = _git(path, "status", "--porcelain")
        assert ".owner.json" not in with_claim, (
            f"the claim must be invisible to git; status said:\n{with_claim}")
        # The strong form: removing the claim must not change git's opinion at all. (What DOES
        # show here is the deps/pokemon-showdown symlink `_create_run_worktree` plants — a
        # real submodule path in the real repo, merely untracked in this toy one.)
        os.remove(wt._owner_path(path))
        without_claim = _git(path, "status", "--porcelain")
        assert with_claim == without_claim, (
            f"the claim changed the worktree's git status:\n{with_claim!r}\nvs\n{without_claim!r}")
    finally:
        cleanup()


# ---------------------------------------------------------------------------------------
# housekeeping: a non-launcher worktree is never touched, and cleanup removes the claim
# ---------------------------------------------------------------------------------------

def test_a_non_launcher_worktree_is_ignored(repo, tmp_path, monkeypatch):
    other = str(tmp_path / "someone-elses-worktree")
    _git(repo, "worktree", "add", "--detach", other, "HEAD")
    try:
        wt._prune_stale_launcher_worktrees(repo, report=lambda _l: None)
        assert os.path.isdir(other), "only launcher-* worktrees are in scope"
    finally:
        subprocess.run(["git", "worktree", "remove", "--force", other],
                       cwd=repo, capture_output=True)


def test_cleanup_removes_both_the_worktree_and_its_claim(repo, monkeypatch):
    path, cleanup = _make_worktree(repo, monkeypatch)
    cleanup()
    assert not os.path.isdir(path)
    assert not os.path.exists(wt._owner_path(path)), "a stale claim would outlive its worktree"
