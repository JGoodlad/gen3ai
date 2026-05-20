"""Training launcher package.

Re-exports the full public API so that `from main.launcher import X` keeps working
after the flat module was split into submodules.
"""

import sys

from main.launcher.checkpoint import (
    find_latest_checkpoint,
    _find_model_arg,
    _insert_or_replace_model_arg,
    _insert_or_replace_run_dir_arg,
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
        "--no-pin",
        action="store_true",
        default=False,
        help="Skip worktree pinning — child runs from the current working tree (old behaviour)",
    )
    parser.add_argument(
        "--sync-to-main",
        action="store_true",
        default=False,
        help=(
            "When resuming from a checkpoint, pin the isolated worktree to the current HEAD "
            "instead of the checkpoint's original git hash. Useful for picking up UI or tooling "
            "fixes without discarding the checkpoint."
        ),
    )
    parser.add_argument("-h", "--help", action="store_true")

    known, _ = parser.parse_known_args()

    if known.help:
        parser.print_help()
        print("\nAll other arguments are forwarded to train_rl_agent.py.")
        sys.exit(0)

    child_args = _strip_launcher_args(sys.argv[1:])
    try:
        run(
            child_args,
            interval_hours=known.restart_interval_hours,
            pin=not known.no_pin,
            sync_to_main=known.sync_to_main,
        )
    except KeyboardInterrupt:
        sys.exit(0)
