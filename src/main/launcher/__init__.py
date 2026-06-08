"""Training launcher package (Textual UI).

`python -m main.launcher …` runs the launcher; `python -m main.launcher.tui …` is a
back-compat alias for the same entry point. Re-exports the public API so
`from main.launcher import X` keeps working after the flat module was split into submodules.
"""

from main.launcher.checkpoint import (
    find_latest_checkpoint,
    _apply_default_showdown_port,
    _find_model_arg,
    _insert_or_replace_model_arg,
    _insert_or_replace_run_dir_arg,
    _peek_arg,
    _resolve_fresh_run_dir,
    _set_arg,
    _strip_launcher_args,
)
from main.launcher.child import _TRAIN_SCRIPT, _SRC_DIR, _read_metrics_pipe
from main.launcher.input import _PollFlags, _dispatch_command
from main.launcher.run import main, run
from main.launcher.state import LauncherSnapshot, LauncherState
from main.launcher.worktree import _git_hash, _read_checkpoint_git_hash, get_git_hash

__all__ = [
    "main",
    "run",
    "LauncherState",
    "LauncherSnapshot",
    "find_latest_checkpoint",
]
