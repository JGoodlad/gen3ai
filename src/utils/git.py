"""Git-derived facts: the HEAD hash and the two working-tree roots.

**These answer a git question, and git needs a cwd.** Both root helpers run ``git rev-parse`` in
the *caller's* current directory unless you pass ``cwd=``, so a process that has chdir'd
elsewhere gets a different (or no) answer. For "where is the checkout this CODE came from" —
which is what almost every caller actually means, and which must also work in a checkout with no
``.git`` at all — use ``utils.paths`` instead; it derives the same roots from ``__file__`` with
no subprocess. ``utils/paths.py`` documents the three-way split.
"""
import os
import subprocess
from typing import Optional


def get_git_hash(short: bool = False) -> str:
    """Return the current git HEAD hash, or 'unknown' if git is unavailable."""
    args = ["git", "rev-parse", "HEAD"] if not short else ["git", "rev-parse", "--short", "HEAD"]
    try:
        return subprocess.check_output(args, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def get_repo_root(cwd: Optional[str] = None) -> str:
    """Return the absolute path of the current git working tree root.

    When called from a git worktree this returns the worktree root, not the
    main repo root.  Use ``get_main_repo_root()`` when you need the directory
    that contains ``models/`` and ``tensorboard/``.

    ``cwd`` pins which directory git is asked about; the default is the process's, which means
    the answer depends on where the caller happens to be standing.
    """
    return subprocess.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True, stderr=subprocess.DEVNULL, cwd=cwd
    ).strip()


def get_main_repo_root(cwd: Optional[str] = None) -> str:
    """Return the root of the *main* working tree, even when running inside a worktree.

    Git stores a shared ``--git-common-dir`` (e.g. ``/repo/.git``) whose parent
    is always the main repo root.  This is the directory that contains
    ``models/``, ``tensorboard/``, etc.

    ``cwd`` pins which directory git is asked about (``utils.paths.main_models_dir`` passes its
    own checkout, so the run archive resolves the same way from any working directory).
    """
    git_common = subprocess.check_output(
        ["git", "rev-parse", "--git-common-dir"], text=True, stderr=subprocess.DEVNULL, cwd=cwd
    ).strip()
    # `--git-common-dir` is RELATIVE to the queried directory, so resolve it against `cwd`
    # rather than the process's own — otherwise pinning `cwd` would silently change the answer.
    if not os.path.isabs(git_common):
        git_common = os.path.join(cwd or os.getcwd(), git_common)
    return os.path.dirname(os.path.abspath(git_common))
