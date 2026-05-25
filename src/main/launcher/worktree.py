"""Git worktree management for launcher process isolation."""

import json
import os
import subprocess
import tempfile

from utils.git import get_git_hash, get_repo_root

_WORKTREE_PREFIX = "launcher-"


def _git_hash() -> str:
    return get_git_hash(short=True)


def _read_checkpoint_git_hash(model_path: str) -> "str | None":
    """Read git_hash from the per-checkpoint sidecar or run-level metadata.json.

    Prefers the sidecar (checkpoint_XXXX.json) because it's written at save time
    by the child inside the pinned worktree. The run-level metadata.json can be
    overwritten by later saves and may not reflect the checkpoint's actual hash.
    """
    abs_path = os.path.abspath(model_path)
    # Per-checkpoint sidecar: replace .zip extension (or append .json)
    sidecar = abs_path[:-4] + ".json" if abs_path.endswith(".zip") else abs_path + ".json"
    if os.path.exists(sidecar):
        try:
            with open(sidecar) as f:
                h = json.load(f).get("git_hash")
            if h:
                return h
        except Exception:
            pass

    # Fall back to run-level metadata.json
    run_dir = os.path.dirname(abs_path)
    meta = os.path.join(run_dir, "metadata.json")
    if os.path.exists(meta):
        try:
            with open(meta) as f:
                return json.load(f).get("git_hash")
        except Exception:
            return None
    return None


def _read_checkpoint_lr(model_path: str) -> "float | None":
    """Read current_lr from the metadata.json saved alongside a checkpoint."""
    candidates = [
        os.path.dirname(os.path.abspath(model_path)),
        os.path.abspath(model_path),
    ]
    for d in candidates:
        meta = os.path.join(d, "metadata.json")
        if os.path.exists(meta):
            try:
                with open(meta) as f:
                    return json.load(f).get("current_lr")
            except Exception:
                return None
    return None


def _prune_stale_launcher_worktrees(repo_root: str) -> None:
    """Remove any stale launcher-* worktrees left by crashed sessions."""
    result = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        capture_output=True, text=True, cwd=repo_root,
    )
    if result.returncode != 0:
        return
    for line in result.stdout.splitlines():
        if line.startswith("worktree "):
            path = line[len("worktree "):]
            if os.path.basename(path).startswith(_WORKTREE_PREFIX):
                subprocess.run(
                    ["git", "worktree", "remove", "--force", path],
                    capture_output=True, cwd=repo_root,
                )


def _create_run_worktree(git_hash: str) -> "tuple[str, str, callable]":
    """Create a detached git worktree pinned to git_hash.

    Returns (train_script_path, src_dir_path, cleanup_fn).
    Raises RuntimeError if git worktree add fails.
    """
    repo_root = get_repo_root()
    tmp = tempfile.mkdtemp(prefix=f"{_WORKTREE_PREFIX}{git_hash[:8]}-")
    os.rmdir(tmp)  # git worktree add requires the target not to exist
    result = subprocess.run(
        ["git", "worktree", "add", "--detach", tmp, git_hash],
        capture_output=True, text=True, cwd=repo_root,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git worktree add failed:\n{result.stderr.strip()}")

    # Replace the empty submodule placeholder with a symlink to the main repo's
    # fully-initialized pokemon-showdown checkout.  Node's require() needs the
    # whole directory (including package.json), not just dist/ + node_modules/.
    ps_main = os.path.join(repo_root, "deps", "pokemon-showdown")
    ps_wt = os.path.join(tmp, "deps", "pokemon-showdown")
    if os.path.isdir(ps_wt) and not os.path.islink(ps_wt):
        os.rmdir(ps_wt)  # git leaves an empty placeholder directory
    if not os.path.lexists(ps_wt):
        os.symlink(ps_main, ps_wt)

    train_script = os.path.join(tmp, "src", "main", "train_rl_agent.py")
    src_dir = os.path.join(tmp, "src")

    def cleanup() -> None:
        subprocess.run(
            ["git", "worktree", "remove", "--force", tmp],
            capture_output=True, cwd=repo_root,
        )

    return train_script, src_dir, cleanup
