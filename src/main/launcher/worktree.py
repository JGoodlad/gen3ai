"""Git worktree management for launcher process isolation — and the PIN DECISION.

The pin is *which commit the isolated worktree is checked out at*, and there are four
sources for it (``resolve_pin``): an explicit ``--pin-commit``, the resumed checkpoint's
recorded ``git_hash``, HEAD under ``--sync-to-main``, and HEAD for a fresh run. Keeping
that decision in one pure function is what lets a batch of arms pin to ONE commit instead
of to whatever HEAD happened to be at each arm's own launch.
"""

import json
import os
import subprocess
import tempfile
from dataclasses import dataclass

from main.exit_codes import TrainExitCode
from main.launcher.checkpoint import run_dir_for_checkpoint
from main.train.fork_lr import is_same_run_checkpoint
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
    # metadata.json is run-LEVEL (at the run root), but the checkpoint may sit one level
    # down in checkpoints/ — derive the run dir so the lookup doesn't miss it.
    meta = _load_json_dict(os.path.join(run_dir_for_checkpoint(abs_path), "metadata.json"))
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


# --------------------------------------------------------------------------------------
# The pin decision
# --------------------------------------------------------------------------------------


class PinRefused(Exception):
    """The launcher cannot pin a worktree, and must not guess.

    Carries the exit code to leave with: ``FATAL_CONFIG`` for a pin the caller NAMED and we
    could not honour (an unresolvable ``--pin-commit``, a restart that would move the pin),
    since re-running the identical command would fail identically.
    """

    def __init__(self, message: str, exit_code: int = int(TrainExitCode.FATAL_CONFIG)):
        super().__init__(message)
        self.exit_code = exit_code


@dataclass
class PinDecision:
    """Where the worktree gets checked out, and WHY."""

    sha: str            # the commit-ish handed to `git worktree add` (full sha for --pin-commit)
    source: str         # "pin_commit" | "checkpoint" | "sync_to_main" | "head"
    subject: str = ""   # the commit's subject line, when we resolved it ourselves


def _rev_parse(spec: str, *, repo_root: "str | None" = None) -> "str | None":
    """Full sha for ``spec`` if it names a commit in this repo, else None."""
    if not spec:
        return None
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--verify", "--quiet", f"{spec}^{{commit}}"],
            capture_output=True, text=True, cwd=repo_root or get_repo_root(),
        )
    except Exception:
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _commit_subject(sha: str, *, repo_root: "str | None" = None) -> str:
    try:
        out = subprocess.run(
            ["git", "log", "-1", "--format=%s", sha],
            capture_output=True, text=True, cwd=repo_root or get_repo_root(),
        )
    except Exception:
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def resolve_commit(spec: str, *, repo_root: "str | None" = None) -> "tuple[str, str]":
    """Resolve a full sha / unambiguous prefix / ref to ``(full_sha, subject)``.

    Raises :class:`PinRefused` when it does not name a commit in this repository — never
    a silent fall-back to HEAD, which is exactly the failure ``--pin-commit`` exists to
    prevent.
    """
    root = repo_root or get_repo_root()
    sha = _rev_parse(spec, repo_root=root)
    if not sha:
        raise PinRefused(
            f"--pin-commit {spec!r} does not resolve to a commit in {root}.\n"
            f"Check the sha (a fetch may be needed for a commit that is not local yet), or "
            f"drop --pin-commit to pin the checkpoint's own hash / HEAD."
        )
    return sha, _commit_subject(sha, repo_root=root)


def _same_commit(a: str, b: str, *, repo_root: "str | None" = None) -> bool:
    """Do two commit-ish strings name the same commit? Prefix-tolerant (short hashes)."""
    ra = _rev_parse(a, repo_root=repo_root) or a
    rb = _rev_parse(b, repo_root=repo_root) or b
    return ra == rb or ra.startswith(rb) or rb.startswith(ra)


def resolve_pin(
    *,
    model_path: "str | None",
    run_dir: "str | None",
    pin_commit: "str | None",
    sync_to_main: bool,
    repo_root: "str | None" = None,
) -> PinDecision:
    """Which commit this session's isolated worktree is pinned to.

    Precedence: an explicit ``--pin-commit`` beats everything — a genuine FORK's parent
    hash, and HEAD on a fresh run. The one case it does NOT beat is a same-run RESTART: a
    launcher periodic restart re-invokes the identical argv into the identical run dir, so
    honouring a ``--pin-commit`` that differs from the checkpoint's recorded ``git_hash``
    there would silently move a live run onto other code. That is REFUSED, naming both.
    (``is_same_run_checkpoint`` is IMPORTED from ``main.train.fork_lr``, never re-derived —
    the trainer's LR pin and the pool seeding key on the same predicate.)
    """
    if pin_commit:
        sha, subject = resolve_commit(pin_commit, repo_root=repo_root)
        if model_path and run_dir and is_same_run_checkpoint(model_path, run_dir):
            recorded = _read_checkpoint_git_hash(model_path)
            if recorded and not _same_commit(recorded, sha, repo_root=repo_root):
                raise PinRefused(
                    f"--pin-commit {pin_commit} resolves to {sha}, but this is a RESTART of the "
                    f"run at {run_dir} whose checkpoint {os.path.basename(model_path)} records "
                    f"git_hash {recorded}.\n"
                    f"A periodic/crash restart must never move a run's pin. Drop --pin-commit "
                    f"(the checkpoint's hash is used automatically), pass --pin-commit {recorded} "
                    f"to say the same thing explicitly, or --sync-to-main to move it deliberately."
                )
        return PinDecision(sha, "pin_commit", subject)

    if model_path and not sync_to_main:
        recorded = _read_checkpoint_git_hash(model_path)
        if not recorded:
            raise PinRefused(
                f"--model given but no git_hash found in metadata.json for {model_path!r}.\n"
                f"Use --no-pin to skip worktree isolation.",
                exit_code=1,
            )
        return PinDecision(recorded, "checkpoint")

    return PinDecision(
        get_git_hash(),  # full hash for worktree add
        "sync_to_main" if (model_path and sync_to_main) else "head",
    )


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
