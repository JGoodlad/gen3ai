"""Resolve a run ARGUMENT to a run DIRECTORY — the one place ``main.ops`` answers that question.

Every instrument in this package took a hardcoded ``models/<arm>`` string in the session copies
it was promoted from. That is wrong twice over: it is correct on exactly one box, and
``models/`` is **not committed and exists only in the MAIN checkout** — a linked worktree has
none, so a literal is not even correct on the right box from the wrong directory.

``utils.paths.main_models_dir()`` answers "where is the run archive" via git's shared common
dir, and returns ``None`` when there is no archive. **Every caller must turn that ``None`` into
a refusal**, which is what :func:`resolve_run_dir` does: it raises ``SystemExit(2)`` with a
message naming what is missing and the escape hatch, rather than returning a path that then
globs to nothing and reads as "no data".
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

from utils.paths import MODELS_DIR_ENV_VAR, main_models_dir


def refuse(*lines: str) -> "None":
    """Print a REFUSAL and exit 2. Nothing is read and nothing is concluded."""
    for line in lines:
        print(line)
    raise SystemExit(2)


def resolve_run_dir(arg: str) -> Path:
    """A run NAME (looked up in the archive) or a run DIRECTORY -> the directory.

    A bare name with no path separator is looked up under :func:`main_models_dir`. Anything
    holding a separator, or starting with a dot, is taken as a path and must already exist. A
    miss of either kind is a refusal, never a silent empty read.
    """
    if not arg:
        refuse("REFUSING: no run given. Pass a run NAME (looked up in models/) or a run DIRECTORY.")
    candidate = Path(arg)
    if candidate.is_dir():
        return candidate.resolve()
    if os.sep in arg or arg.startswith("."):
        refuse(f"REFUSING: no such run directory: {arg}")
    models = main_models_dir()
    if models is None:
        refuse(
            "REFUSING: no run archive — the main checkout has no models/ directory.",
            "  models/ is NOT committed; it exists only on a machine that has trained, and only",
            "  in the MAIN checkout (a linked worktree has none). Set "
            f"${MODELS_DIR_ENV_VAR} if the archive",
            "  lives elsewhere. Nothing is read and nothing is concluded.",
        )
    assert models is not None  # refuse() never returns; this is for the type checker
    run_dir = models / arg
    if not run_dir.is_dir():
        refuse(f"REFUSING: no run {arg!r} under {models}")
    return run_dir


def interpreter() -> str:
    """The interpreter a subprocess should be spawned with.

    ``$GEN3AI_PYTHON`` if it names an executable — the launcher's own knob for pinning a child's
    interpreter — else ``sys.executable``, which is by construction the one running this code.
    Never a literal path: that was correct on one box and silently wrong everywhere else.
    """
    pinned = os.environ.get("GEN3AI_PYTHON")
    if pinned and os.access(pinned, os.X_OK):
        return pinned
    return sys.executable


def event_dirs(run_dir: Path) -> list:
    """The TensorBoard event directories under ``<run>/tb`` — its subdirectories, else itself.

    A run whose ``tb/`` is absent yields an empty list, and every caller reports that as a
    MISSING SCALAR rather than an empty series that reads like a zero.
    """
    tb = Path(run_dir) / "tb"
    if not tb.is_dir():
        return []
    subs = sorted(d for d in tb.iterdir() if d.is_dir())
    return [str(d) for d in subs] or [str(tb)]
