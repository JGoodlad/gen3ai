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


def scalar_events(run_dir: Path, tags, *, include_inherited: bool = False) -> dict:
    """``{tag: [ScalarEvent, …]}`` (step-sorted) for each tag present — the ARM's OWN points only.

    🚨 A FORK's ``tb/`` holds its parent's curve at steps ``<= fork_step`` (``gen3_tb_inherit_v1``); a
    window, a ``--last N`` or a whole-series max read across it measures the PARENT under this arm's
    name (ledger ``36f8f7eb``). The boundary is the lineage record's ``fork_step``
    (``agents.training.tb_inherit.own_step_floor``); an unconfirmable boundary is a REFUSAL.
    ``include_inherited=True`` keeps the prefix — for a reader whose question IS the whole history.
    Every live-tier TensorBoard reader goes through here; a tag with no point is absent from the dict.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from agents.training.tb_inherit import ArmBoundaryError, own_step_floor

    floor = None
    if not include_inherited:
        try:
            floor = own_step_floor(str(run_dir))
        except ArmBoundaryError as exc:
            refuse(f"REFUSING: {exc}.", "  Nothing is read: a slice at a guessed boundary would mix the "
                   "parent's points into this arm's.")
    out: dict = {}
    want = list(tags)
    for d in event_dirs(Path(run_dir)):
        ea = EventAccumulator(d, size_guidance={"scalars": 0})
        ea.Reload()
        avail = set(ea.Tags().get("scalars", []))
        for t in want:
            if t in avail:
                out.setdefault(t, []).extend(e for e in ea.Scalars(t) if floor is None or e.step > floor)
    return {t: sorted(v, key=lambda e: (e.step, e.wall_time)) for t, v in out.items() if v}


def floor_line(run_dir: Path) -> str:
    """The line a live-tier TB reader prints first: whether, and where, an inherited prefix was cut."""
    from agents.training.tb_inherit import ArmBoundaryError, floor_note

    try:
        note = floor_note(str(run_dir))
    except ArmBoundaryError as exc:
        refuse(f"REFUSING: {exc}.")
        raise AssertionError("unreachable")  # pragma: no cover
    return note or "not a fork (no lineage fork_step): every TensorBoard point is this run's own"
