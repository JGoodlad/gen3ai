"""The launcher's SUBMODULE PREFLIGHT: does the checkout it launches from have a usable ``deps/pokemon-showdown``?

``_create_run_worktree`` replaces the pinned worktree's empty submodule placeholder with a LINK to the LAUNCHING
checkout's ``deps/pokemon-showdown`` (``worktree.showdown_link_source``), and an un-pinned (``--no-pin``) child reads
the launching tree's own. Either way, a launch from a checkout whose submodule was never initialised or built gives the
child an empty placeholder, and it dies minutes later in team validation after the model and the inference service are
already up (``designs/research_state/measurements/gpu_checks_endstate_2026-10-09/`` F-GE-4: the first P attempt of the
end-state GPU check died this way).

So the launcher asks BEFORE anything exists on disk - no worktree, no run dir, no child - the same way it asks the disk
and desktop-GPU questions: :func:`verdict_for_launch` is what ``_prepare_session`` enforces (``FATAL_CONFIG``, the fix
named) and ``--dry-run`` prints. It is the LAUNCHER's check and never advisory: a pinned child's trainer may predate any
check of its own, and the link is the launcher's doing. What counts as "usable" is ``utils.showdown_deps.NEEDS`` - the
files team validation reads.
"""
from __future__ import annotations

import os
from typing import Callable, Optional

from main.launcher import worktree
from main.launcher.child import _SRC_DIR
from utils import showdown_deps
from utils.showdown_deps import DepsVerdict


def source_for_launch(pin: bool, repo_root: Optional[str] = None) -> str:
    """The ``deps/pokemon-showdown`` directory the child will end up reading: the one the pinned worktree is LINKED
    to (``worktree.showdown_link_source(repo_root)``, ``repo_root`` being the launching checkout the caller already
    resolved), or - under ``--no-pin`` - the launching tree's own."""
    if pin:
        return worktree.showdown_link_source(repo_root)
    return showdown_deps.source_dir(os.path.dirname(_SRC_DIR))


def verdict_for_launch(*, pin: bool, repo_root: Optional[str] = None,
                       which: Optional[Callable[[str], Optional[str]]] = None) -> DepsVerdict:
    """The preflight verdict for a launch with (``pin``) or without (``--no-pin``) the isolated worktree."""
    root = source_for_launch(pin, repo_root)
    return showdown_deps.check(root) if which is None else showdown_deps.check(root, which=which)
