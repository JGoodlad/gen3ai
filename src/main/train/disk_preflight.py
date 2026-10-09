"""The disk-space PREFLIGHT's glue (`utils.disk_guard` holds the arithmetic and the verdict).

ONE function builds the :class:`~utils.disk_guard.RunPlan` from a parsed namespace, and the three surfaces
that decide "may this run start" call it — the trainer's startup (`train_rl_agent.main`, a refusal exits
`FATAL_CONFIG`), the launcher's real launch (`launcher.run._prepare_session`, before the pin / worktree /
run dir exist, so it guards a PINNED child whose own trainer predates the guard) and the launcher's
`--dry-run` (the same verdict, printed). Three copies of the question would drift.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Optional

from utils import disk_guard


def _resume_steps(model_path: Optional[str]) -> int:
    if not model_path:
        return 0
    mp = model_path
    if os.path.isdir(mp):
        mp = disk_guard._run_checkpoint_zip(mp) or ""
    if not mp:
        return 0
    try:
        from agents.training.lineage import checkpoint_num_timesteps
        v = checkpoint_num_timesteps(mp)
    except Exception:                              # noqa: BLE001 — an estimate never kills a launch
        v = None
    return int(v) if isinstance(v, (int, float)) else 0


def _archive_dir() -> Optional[str]:
    try:
        from utils.paths import main_models_dir
        d = main_models_dir()
        return str(d) if d is not None else None
    except Exception:                              # noqa: BLE001
        return None


def plan_for_namespace(ns: Any, run_dir: str, *, model_path: Optional[str] = None,
                       resume_steps: Optional[int] = None) -> disk_guard.RunPlan:
    """The plan for a run writing into ``run_dir`` with the flag values ``ns`` carries (a trainer namespace,
    after inheritance on a resume). ``resume_steps`` None = read the checkpoint."""
    from main.train.fork_lr import is_same_run_checkpoint
    model = model_path if model_path is not None else getattr(ns, "model", None)
    fork = bool(model) and not is_same_run_checkpoint(model, run_dir)
    return disk_guard.plan_from_values(
        run_dir=run_dir,
        model_path=model,
        steps=int(getattr(ns, "steps", 0) or 0),
        resume_steps=_resume_steps(model) if resume_steps is None else int(resume_steps),
        checkpoint_every_steps=getattr(ns, "checkpoint_every_steps", None),
        eval_freq=getattr(ns, "eval_freq", None),
        fork=fork,
        self_play=bool(getattr(ns, "self_play", True)),
        device=str(getattr(ns, "device", "auto") or "auto"),
        archive_dir=_archive_dir(),
    )


def verdict_for_namespace(ns: Any, run_dir: str, *, model_path: Optional[str] = None,
                          resume_steps: Optional[int] = None,
                          free_fn: Optional[Callable[[str], int]] = None,
                          allow: Optional[bool] = None) -> disk_guard.DiskVerdict:
    """The verdict every surface reads: ``--debug`` is exempt, ``--allow-low-disk`` tolerates (``allow``
    overrides the namespace's flag: the launcher consumes it from a pinned child's argv)."""
    debug = bool(getattr(ns, "debug", False))
    allow = bool(getattr(ns, "allow_low_disk", False)) if allow is None else bool(allow)
    plan = None if debug else plan_for_namespace(ns, run_dir, model_path=model_path,
                                                 resume_steps=resume_steps)
    return disk_guard.check_for_run(plan, debug=debug, allow=allow, free_fn=free_fn)
