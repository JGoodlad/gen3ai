"""The launcher's half of the disk-space guard (`utils.disk_guard`, `main.train.disk_preflight`).

The launcher asks BEFORE anything exists on disk (no worktree, no run dir, no child): the same
:func:`~main.train.disk_preflight.verdict_for_namespace` the trainer asks at its own startup and
`--dry-run` prints. Asking here matters most for a PINNED child — its trainer is the pinned commit's, which
predates this guard, so the launcher's check is the only one it gets (the 2026-10-09 screen chain ran
pinned).

For the same reason ``--allow-low-disk`` is CONSUMED here when the pinned commit's parser does not know the
flag (``flags_only_in_current_tree``): it is dropped from the argv the pinned child receives, instead of
failing the pinned-parser check or crashing the child at startup.
"""
from __future__ import annotations

from typing import Any, List, Optional, Tuple

from utils.disk_guard import OPT_OUT_FLAG, DiskVerdict


def strip_opt_out(argv: List[str]) -> List[str]:
    """``argv`` without ``--allow-low-disk`` (a store_true flag: no value to remove)."""
    return [a for a in argv if a != OPT_OUT_FLAG]


def consume_if_absent_at_pin(argv: List[str], report: Any) -> Tuple[List[str], bool]:
    """``(argv', stripped)``: ``--allow-low-disk`` is removed from the child's argv iff the PINNED parser
    (``report``, an authoritative `ParseReport`) does not declare it. Called with the report of a check
    made on the full argv; the caller re-checks the stripped one."""
    if OPT_OUT_FLAG not in argv or report is None:
        return argv, False
    from main.launcher.pinned_argv import flags_only_in_current_tree
    if getattr(report, "authoritative", False) and OPT_OUT_FLAG in flags_only_in_current_tree(report, argv):
        return strip_opt_out(argv), True
    return argv, False


def verdict_for_launch(child_args: List[str], run_dir: str, *, ns: Optional[Any] = None,
                       free_fn=None, allow: Optional[bool] = None) -> DiskVerdict:
    """The disk verdict for a launch of ``child_args`` writing into ``run_dir``. ``ns`` is the resolved
    namespace when the caller has it (`--dry-run`); otherwise it is resolved the way `--dry-run` does
    (`main.checkargs`, the parent's recorded config overlaid on the argv)."""
    from main.launcher.checkpoint import _find_model_arg
    from main.train.disk_preflight import verdict_for_namespace
    model_path = _find_model_arg(child_args)
    if ns is None:
        try:
            from main.launcher.dry_run import _effective_namespace
            ns = (_effective_namespace(child_args).get("resolution") or {}).get("ns")
        except Exception:                           # noqa: BLE001 — an unresolvable argv is reported by its own path
            ns = None
    if ns is None:                                  # an argv that does not parse: its own path reports it
        import argparse
        from main.launcher.checkpoint import _peek_arg
        ns = argparse.Namespace(
            steps=_peek_arg(child_args, "--steps", type_=int) or 0,
            debug="--debug" in child_args, allow_low_disk=OPT_OUT_FLAG in child_args,
            checkpoint_every_steps=_peek_arg(child_args, "--checkpoint-every-steps", type_=int),
            eval_freq=_peek_arg(child_args, "--eval-freq", type_=int),
            device=_peek_arg(child_args, "--device") or "auto")
    return verdict_for_namespace(ns, run_dir, model_path=model_path, free_fn=free_fn,
                                 allow=(OPT_OUT_FLAG in child_args) if allow is None else allow)
