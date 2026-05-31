"""Training launcher package.

Re-exports the full public API so that `from main.launcher import X` keeps working
after the flat module was split into submodules.
"""

import sys

from main.launcher.checkpoint import (
    find_latest_checkpoint,
    _apply_default_showdown_port,
    _find_model_arg,
    _insert_or_replace_model_arg,
    _insert_or_replace_run_dir_arg,
    _peek_arg,
    _set_arg,
    _strip_launcher_args,
)
from main.launcher.child import _TRAIN_SCRIPT, _SRC_DIR, _read_metrics_pipe
from main.launcher.input import _PollFlags, _dispatch_command, _setup_raw_input
from main.launcher.run import run
from main.launcher.state import LauncherSnapshot, LauncherState
from main.launcher.ui import LauncherUI
from main.launcher.worktree import _git_hash, _read_checkpoint_git_hash, get_git_hash


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Launcher: wraps train_rl_agent.py with periodic full-process restart.",
        add_help=False,
    )
    parser.add_argument(
        "--restart-interval-hours",
        type=float,
        default=3.0,
        help="Restart the training process every N hours (0 = run once)",
    )
    parser.add_argument(
        "--restart-grace-minutes",
        type=float,
        default=20.0,
        help=(
            "Fallback window after a scheduled restart's deadline. The child normally "
            "stops itself at the next rollout boundary; the launcher only force-kills if "
            "the child overruns the deadline by this many minutes (hung / very long rollout)."
        ),
    )
    parser.add_argument(
        "--no-pin",
        action="store_true",
        default=False,
        help="Skip worktree pinning — child runs from the current working tree (old behaviour)",
    )
    pin_group = parser.add_mutually_exclusive_group()
    pin_group.add_argument(
        "--sync-to-main",
        action="store_true",
        default=False,
        help=(
            "When resuming from a checkpoint, pin the isolated worktree to the current HEAD "
            "instead of the checkpoint's original git hash. Useful for picking up UI or tooling "
            "fixes without discarding the checkpoint."
        ),
    )
    pin_group.add_argument(
        "--pin-to-hash",
        metavar="HASH",
        default=None,
        help=(
            "Pin the isolated worktree to a specific git hash instead of the checkpoint's "
            "recorded hash or current HEAD. Useful for resuming a run against a known-good commit."
        ),
    )
    parser.add_argument("-h", "--help", action="store_true")

    known, _ = parser.parse_known_args()

    if known.help:
        parser.print_help()
        print("\nAll other arguments are forwarded to train_rl_agent.py.")
        sys.exit(0)

    if known.pin_to_hash and known.no_pin:
        parser.error("--pin-to-hash and --no-pin are mutually exclusive")

    child_args = _strip_launcher_args(sys.argv[1:])
    # Launcher sessions are long-lived: default them to the dedicated training server
    # (8001) so dev-server churn on 8000 can't drop every worker's connection mid-run.
    # An explicit --showdown-port still wins.
    child_args = _apply_default_showdown_port(child_args)
    try:
        run(
            child_args,
            interval_hours=known.restart_interval_hours,
            pin=not known.no_pin,
            sync_to_main=known.sync_to_main,
            pin_hash_override=known.pin_to_hash,
            grace_minutes=known.restart_grace_minutes,
        )
    except KeyboardInterrupt:
        sys.exit(0)
