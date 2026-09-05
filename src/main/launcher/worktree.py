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
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Callable, Optional

from main.exit_codes import TrainExitCode
from main.launcher.checkpoint import run_dir_for_checkpoint
from main.train.fork_lr import is_same_run_checkpoint
from utils.git import get_git_hash, get_repo_root

_WORKTREE_PREFIX = "launcher-"

#: Suffix of the OWNERSHIP file written BESIDE each launcher worktree (never inside it).
#: See :func:`_owner_path` for why it lives outside the tree.
_OWNER_SUFFIX = ".owner.json"

#: A pre-fix worktree carries no owner file. It is only pruned once this old, so a LIVE
#: pre-fix run keeps its worktree while an abandoned one still gets collected eventually.
_LEGACY_ORPHAN_MAX_AGE_S = 24 * 60 * 60


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


# --------------------------------------------------------------------------------------
# Worktree OWNERSHIP - who is using this launcher-* worktree, and are they still alive?
# --------------------------------------------------------------------------------------
#
# 2026-09-05 INCIDENT. `_prune_stale_launcher_worktrees` used to force-remove EVERY
# `launcher-*` worktree at startup with no liveness check at all, on the assumption that a
# launcher-* worktree can only be the debris of a crashed session. A one-second validation
# command (`python -m main.launcher --pin-commit deadbeef --steps 1`) therefore deleted the
# isolated worktree of a LIVE production run. The run kept going on its already-open file
# descriptors and looked healthy for hours; it died at its next 3 h periodic restart, when
# the launcher re-exec'd the child from a directory that no longer existed (exit 2, no
# final_model).
#
# So a worktree is now pruned ONLY when its owner is provably gone. Every ambiguity - no
# /proc, an unreadable owner file, a pid we cannot interrogate - resolves to KEEP: leaving a
# stale directory in /tmp costs disk, deleting a live one costs a run.


def _owner_path(worktree_path: str) -> str:
    """Where a worktree's owner record lives: ``<worktree>.owner.json``, BESIDE the tree.

    Deliberately OUTSIDE the checkout. A file inside the worktree would show up in that
    worktree's ``git status`` as untracked (the pinned tree is a real checkout of a real
    commit, and its `.gitignore` is whatever THAT commit says - an ignore rule added today
    does not exist in a worktree pinned to a commit from last month), and the child's own
    tooling runs `git` in there. A sibling file is invisible to git by construction, and it
    sits in the same ``tempfile.mkdtemp`` parent so the same /tmp cleanup collects it.
    """
    return worktree_path.rstrip("/") + _OWNER_SUFFIX


def _proc_starttime(pid: int) -> "str | None":
    """Field 22 of ``/proc/<pid>/stat`` - the process start time in clock ticks since boot.

    This is the PID-REUSE guard: a pid alone is not an identity (Linux wraps pids), but
    (pid, starttime) is unique for the life of a boot. ``None`` when /proc is unavailable or
    unreadable, which callers must read as "cannot verify", never as "a different process".
    """
    try:
        with open(f"/proc/{pid}/stat") as f:
            raw = f.read()
    except OSError:
        return None
    # The comm field is parenthesised and may itself contain spaces/parens - split after the
    # LAST ')' so the field offsets below are stable for any process name.
    close = raw.rfind(")")
    if close == -1:
        return None
    fields = raw[close + 2:].split()
    # After comm, field numbering restarts at 3 (state), so starttime (22) is index 19.
    return fields[19] if len(fields) > 19 else None


def _write_owner_file(worktree_path: str, run_dir: "str | None" = None) -> None:
    """Record THIS process as the owner of ``worktree_path``. Best-effort, never raises."""
    pid = os.getpid()
    record = {
        "pid": pid,
        "proc_starttime": _proc_starttime(pid),
        "created_at": time.time(),
        "worktree": worktree_path,
        "run_dir": run_dir,
        "argv": list(sys.argv),
    }
    try:
        with open(_owner_path(worktree_path), "w") as f:
            json.dump(record, f, indent=2)
    except OSError:
        pass  # a worktree with no owner file degrades to the LEGACY (age-gated) rule


def _remove_owner_file(worktree_path: str) -> None:
    try:
        os.remove(_owner_path(worktree_path))
    except OSError:
        pass


def _owner_is_alive(owner: dict) -> bool:
    """Is the process that created this worktree still running?

    FAIL-SAFE: every uncertainty returns True (keep the worktree). Only a pid that is
    provably gone - or provably a DIFFERENT process that reused the number - returns False.
    """
    pid = owner.get("pid")
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True          # alive, just owned by another user
    except OSError:
        return True          # cannot tell - keep it
    recorded = owner.get("proc_starttime")
    if recorded is None:
        return True          # written without a starttime - cannot check reuse, keep it
    actual = _proc_starttime(pid)
    if actual is None:
        return True          # /proc unreadable - cannot check reuse, keep it
    return str(actual) == str(recorded)


def _classify_worktree(path: str) -> "tuple[str, str]":
    """``(verdict, reason)`` for one ``launcher-*`` worktree - ``"keep"`` or ``"remove"``."""
    if not os.path.isdir(path):
        return "remove", "its directory no longer exists"
    owner = _load_json_dict(_owner_path(path))
    if owner is None:
        # Pre-fix worktree, or an owner file cleaned out from under it. Age is the only
        # signal left, so give it a full day before assuming it was abandoned.
        try:
            age = time.time() - os.path.getmtime(path)
        except OSError:
            return "keep", "no owner file and its mtime is unreadable"
        if age > _LEGACY_ORPHAN_MAX_AGE_S:
            return "remove", f"no owner file and {age / 3600:.1f} h old (legacy orphan)"
        return "keep", f"no owner file but only {age / 3600:.1f} h old - may be a live pre-fix run"
    if _owner_is_alive(owner):
        return "keep", f"owned by LIVE pid {owner.get('pid')}"
    return "remove", f"owner pid {owner.get('pid')} is gone"


def _prune_stale_launcher_worktrees(
    repo_root: str, report: Optional[Callable[[str], None]] = None
) -> None:
    """Remove launcher-* worktrees whose OWNER IS DEAD. A live owner's tree is never touched.

    ``report`` receives one line per worktree KEPT (naming the owning pid) and one per
    removal, so a startup that leaves debris behind says so instead of looking like a no-op.
    Defaults to ``print``; the launcher passes ``state.add_event``.
    """
    say = report if report is not None else print
    result = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        capture_output=True, text=True, cwd=repo_root,
    )
    if result.returncode != 0:
        return
    for line in result.stdout.splitlines():
        if not line.startswith("worktree "):
            continue
        path = line[len("worktree "):]
        if not os.path.basename(path).startswith(_WORKTREE_PREFIX):
            continue
        verdict, reason = _classify_worktree(path)
        if verdict == "keep":
            say(f"[worktree] KEPT {os.path.basename(path)} - {reason}")
            continue
        subprocess.run(
            ["git", "worktree", "remove", "--force", path],
            capture_output=True, cwd=repo_root,
        )
        _remove_owner_file(path)
        say(f"[worktree] pruned {os.path.basename(path)} - {reason}")


def _create_run_worktree(
    git_hash: str, run_dir: "str | None" = None
) -> "tuple[str, str, callable]":
    """Create a detached git worktree pinned to git_hash, and CLAIM it for this process.

    Returns (train_script_path, src_dir_path, cleanup_fn).
    Raises RuntimeError if git worktree add fails.

    The claim is a ``<worktree>.owner.json`` sibling naming this pid + its /proc start time;
    it is what stops another launcher's startup prune from deleting a LIVE run's worktree
    (see the 2026-09-05 incident note above). ``cleanup_fn`` removes both.
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
    _write_owner_file(tmp, run_dir)

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
        _remove_owner_file(tmp)

    return train_script, src_dir, cleanup
