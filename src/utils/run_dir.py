"""A run dir is created ONCE, at its run's startup; nothing a live run writes ever re-creates it.

A run dir deleted or moved aside under a live run must stay gone: a writer that ``mkdir -p``\\ s its target (a
checkpoint, ``metadata.json``, an eval result) pointed at ``models/`` would otherwise resurrect it at its old name as a
half-run (2026-11-14). So every run-dir writer goes through these:

* :func:`require_run_dir` -- the run root must exist, else :class:`RunDirGoneError`;
* :func:`ensure_subdir` -- a subdirectory UNDER an existing run root, created one level at a time with ``os.mkdir``
  (never ``makedirs``), so a root that vanishes mid-call raises instead of coming back;
* ``utils.atomic_io.atomic_write(root=)`` -- the atomic writer with that guard built in.

:class:`RunDirGoneError` maps to ``TrainExitCode.FATAL_CONFIG`` (3) by NAME (``main.exit_codes._FATAL_BY_NAME``, so
this module imports nothing from ``main``): the launcher does not restart into a run dir that is not there.
"""
from __future__ import annotations

import contextlib
import os
from typing import Iterator


class RunDirGoneError(RuntimeError):
    """A run-dir writer found its run dir GONE (deleted, or moved aside, while the run was live) and REFUSED rather
    than re-create it. Exit ``FATAL_CONFIG`` (3), mapped by name; the interrupted save exits with it too, never
    INTERRUPTED (15), which would claim a save."""


def require_run_dir(run_dir: "str | os.PathLike[str]") -> None:
    """Raise :class:`RunDirGoneError` unless ``run_dir`` exists as a directory."""
    if not os.path.isdir(run_dir):
        raise RunDirGoneError(
            f"the run dir {os.path.abspath(run_dir)} no longer exists — a write will not re-create it (a deleted "
            "or moved-aside run must not be resurrected under its old name). Nothing was written; the run's "
            "files are wherever its dir now is.")


def ensure_subdir(run_dir: "str | os.PathLike[str]", name: "str | os.PathLike[str]") -> str:
    """``<run_dir>/<name>``, created if missing, one level at a time below the EXISTING ``run_dir``; returns it.

    ``name`` is a path inside ``run_dir``: relative to it (it may have several components), or absolute. Raises
    :class:`RunDirGoneError` when ``run_dir`` is gone, including when it vanishes between levels (``os.mkdir``
    of a level whose parent is gone raises, where ``makedirs`` would have re-created the parent)."""
    root = os.fspath(run_dir)
    rel = os.fspath(name)
    if os.path.isabs(rel):
        rel = os.path.relpath(rel, os.path.abspath(root))
    rel = os.path.normpath(rel)
    if os.path.isabs(rel) or rel == os.pardir or rel.startswith(os.pardir + os.sep):
        raise ValueError(f"ensure_subdir: {name!r} is not a relative path inside the run dir {root}")
    require_run_dir(root)
    path = root
    for part in ([] if rel == os.curdir else rel.split(os.sep)):
        path = os.path.join(path, part)
        try:
            os.mkdir(path)
        except FileExistsError:
            if not os.path.isdir(path):
                raise
        except FileNotFoundError as e:
            require_run_dir(root)        # the root went away under us: the typed refusal
            raise RunDirGoneError(f"a level above {path} vanished while it was being created") from e
    return path


@contextlib.contextmanager
def fatal_if_gone(where: str) -> Iterator[None]:
    """Run a run-dir writer's body; a :class:`RunDirGoneError` inside it prints ``[RunDir] FATAL`` (a launcher FATAL
    signature, so the reason surfaces) and is re-raised, ending the run FATAL_CONFIG (3) through the trainer's
    fail-fast handler, the way a periodic checkpoint into a gone run dir does."""
    try:
        yield
    except RunDirGoneError as e:
        print(f"\n[RunDir] FATAL ({where}): {e}", flush=True)
        raise
