"""THE definition of "this checkout is bootstrapped enough to run the suite", and the one message that names the fix.

Two callers, ONE definition (no second list to drift):

* the root ``conftest.py``'s ``pytest_sessionstart`` refuses a pytest session on an unfinished checkout (the static tier
  included) with one ``UsageError`` instead of ~15 Node "Cannot find module" failures;
* ``utils.ship_gate`` runs the SAME check BEFORE it takes the gate lock or a heavy slot. Without that, a never-bootstrapped
  worktree queued for 20-40 min and was only refused when pytest finally started (recurring 2026-11-16).

The probe is what ``scripts/bootstrap.sh`` step 5 tests (``dist/sim/index.js`` is the file the Node children load;
``node_modules/`` is what that compiled code resolves its own imports through). Pure stdlib file-existence: no network, no
subprocess, milliseconds. It never RUNS bootstrap (that can build Rust); it names the command.

``GEN3AI_SKIP_DEPS_GUARD=1`` opts out of both (a pure-unit CI with no submodule). Pinned by ``src/deps_guard_test.py`` (the
pytest session) and ``src/utils/ship_gate_test.py`` (the pre-queue refusal).
"""
from __future__ import annotations

import os
from typing import List

#: Repo-relative probe paths. Each is restored by a different bootstrap step, so each is reported by name.
REQUIRED_DEPS = (
    os.path.join("deps", "pokemon-showdown", "dist", "sim", "index.js"),
    os.path.join("deps", "pokemon-showdown", "node_modules"),
)
#: The cheap, env-untouching, Rust-free bootstrap a fresh linked worktree needs (it symlinks ``dist`` / ``node_modules``
#: from the main checkout; ~40 s, most of it the submodule checkout).
WORKTREE_FIX = "scripts/bootstrap.sh --skip-env --no-rust --no-check"
OPT_OUT = "GEN3AI_SKIP_DEPS_GUARD"


def missing_deps(root: str) -> List[str]:
    """The probe paths absent under ``root`` (empty = bootstrapped)."""
    return [rel for rel in REQUIRED_DEPS if not os.path.exists(os.path.join(root, rel))]


def is_linked_worktree(root: str) -> bool:
    """A linked worktree's ``.git`` is a FILE; the main checkout's is a directory."""
    return os.path.isfile(os.path.join(root, ".git"))


def refusal_message(missing: List[str], root: str) -> str:
    fix = WORKTREE_FIX if is_linked_worktree(root) else "./scripts/bootstrap.sh"
    return f"""\
deps/pokemon-showdown is not usable in this checkout.

MISSING: {", ".join(missing)}

This is a CHECKOUT problem, not a test failure. Left alone it surfaces as ~15 failures across
unrelated subsystems, each dying inside Node on "Cannot find module
'.../deps/pokemon-showdown/dist/sim/...'". Nothing is wrong with the code.

FIX (idempotent, run from this checkout; it never touches the shared conda env):

    {fix}

or the two manual steps (docs/DEVELOPING.md):

    git submodule update --init
    for n in dist node_modules; do \\
      [ -e "deps/pokemon-showdown/$n" ] || \\
        ln -s "<MAIN CHECKOUT>/deps/pokemon-showdown/$n" "deps/pokemon-showdown/$n"; done

Keep the [ -e ] guard, and run it only from a fresh WORKTREE: from the main checkout `dist/` already
exists, so `ln -s` drops the link INSIDE it as dist/dist -> its own parent and every websocket path
dies with ELOOP.

Set {OPT_OUT}=1 to run anyway (a pure-unit CI with no submodule)."""


def refusal(root: str) -> str:
    """The refusal text for ``root``, or ``""`` when it is bootstrapped or the opt-out is set."""
    if os.environ.get(OPT_OUT):
        return ""
    missing = missing_deps(root)
    return refusal_message(missing, root) if missing else ""
