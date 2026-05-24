import os
import subprocess


def get_git_hash(short: bool = False) -> str:
    """Return the current git HEAD hash, or 'unknown' if git is unavailable."""
    args = ["git", "rev-parse", "HEAD"] if not short else ["git", "rev-parse", "--short", "HEAD"]
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def get_repo_root() -> str:
    """Return the absolute path of the current git working tree root.

    When called from a git worktree this returns the worktree root, not the
    main repo root.  Use ``get_main_repo_root()`` when you need the directory
    that contains ``models/`` and ``tensorboard/``.
    """
    return subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True, stderr=subprocess.DEVNULL
    ).strip()


def get_main_repo_root() -> str:
    """Return the root of the *main* working tree, even when running inside a worktree.

    Git stores a shared ``--git-common-dir`` (e.g. ``/repo/.git``) whose parent
    is always the main repo root.  This is the directory that contains
    ``models/``, ``tensorboard/``, etc.
    """
    git_common = subprocess.check_output(
        ["git", "rev-parse", "--git-common-dir"], text=True, stderr=subprocess.DEVNULL
    ).strip()
    return os.path.dirname(os.path.abspath(git_common))
