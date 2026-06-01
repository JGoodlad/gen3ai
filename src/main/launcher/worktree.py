"""Git worktree management for launcher process isolation."""

import json
import os
import subprocess
import tempfile

from utils.git import get_git_hash, get_repo_root

_WORKTREE_PREFIX = "launcher-"


def _git_hash() -> str:
    return get_git_hash(short=True)


def _load_json_dict(path: str) -> "dict | None":
    """Load a JSON object from path, or None if missing/unreadable/not-an-object."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _read_checkpoint_field(model_path: str, *, key: str, toplevel_key: str):
    """Read a per-checkpoint field, most checkpoint-specific source first.

    Resolution order:
      1. the per-checkpoint sidecar (``checkpoint_XXXX.json`` beside the .zip),
         written at save time by the child inside the pinned worktree;
      2. this checkpoint's entry in metadata.json's ``snapshot_history``;
      3. the run-level top-level value — the LAST resort, because metadata.json's
         top-level fields are overwritten by every later save and so reflect the
         most recent checkpoint, not necessarily the (possibly older) one being
         resumed.

    The sidecar and ``snapshot_history`` entries share one schema (see
    ``_build_snapshot_entry`` in ``agents.model.snapshot``), so ``key`` reads both.
    Only the run-level top-level field may use a different name (``toplevel_key``,
    e.g. ``current_lr`` vs the per-checkpoint ``lr``).
    """
    abs_path = os.path.abspath(model_path)

    # 1. Per-checkpoint sidecar: replace .zip extension (or append .json).
    sidecar = abs_path[:-4] + ".json" if abs_path.endswith(".zip") else abs_path + ".json"
    side = _load_json_dict(sidecar)
    if side is not None and side.get(key) is not None:
        return side[key]

    # 2./3. metadata.json: this checkpoint's history entry, then the top-level value.
    meta = _load_json_dict(os.path.join(os.path.dirname(abs_path), "metadata.json"))
    if meta is None:
        return None
    name = os.path.basename(abs_path)
    if not name.endswith(".zip"):
        name += ".zip"
    entry = meta.get("snapshot_history", {}).get(name)
    if isinstance(entry, dict) and entry.get(key) is not None:
        return entry[key]
    return meta.get(toplevel_key)


def _read_checkpoint_git_hash(model_path: str) -> "str | None":
    """git_hash of the resumed checkpoint (sidecar → snapshot_history → top-level)."""
    return _read_checkpoint_field(model_path, key="git_hash", toplevel_key="git_hash")


def _read_checkpoint_lr(model_path: str) -> "float | None":
    """Current LR of the resumed checkpoint (sidecar → snapshot_history → top-level).

    The sidecar/history store this under ``lr``; the run-level fallback uses the
    legacy top-level ``current_lr`` key written by ``save_model_snapshot``.
    """
    return _read_checkpoint_field(model_path, key="lr", toplevel_key="current_lr")


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
