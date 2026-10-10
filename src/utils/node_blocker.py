"""Make NODE impossible in this interpreter (and its children) and record every attempt (owner 2026-10-10:
"delete node").

**Why.** The offline search / replay driver, ``reconstruction``, the prober's search-shaped probes and its
packed-team decode used to run Node (``search_driver.js`` / ``replay_driver.js`` children, a ``node -e`` dump
of Showdown's alias table) — by DEFAULT. Those paths are Rust / ``data/`` now, and a grep cannot say whether a
RUN-TIME path still reaches Node: a default argument, a fallback or a helper three calls down can. This blocker
answers it the way ``utils.poke_env_blocker`` answers it for poke-env: install it first, then RUN the path.

Two layers, because Node can be reached two ways:

* **in this interpreter** — a ``sys.addaudithook`` hook REFUSES (raises :class:`NodeBlocked`) and records:
  every process spawn whose program is ``node`` / ``nodejs`` (``subprocess``, ``os.exec*``, ``os.posix_spawn``,
  ``os.system``), whatever its path; and every ``open`` of a file under ``deps/pokemon-showdown`` — the
  submodule is HIDDEN, so a path that reads Showdown's source or ``dist/`` directly (no process at all) fails
  too. A swallowed exception still lands in :data:`ATTEMPTS` (and in ``$GEN3AI_NODE_BLOCK_LOG``).
* **in any child** (a Rust binary, a shell) — :func:`shim_env` puts a ``node`` TRIPWIRE first on ``PATH``: an
  executable that appends to the log and exits 97, so a child that spawns ``node`` by name fails AND is recorded.

An audit hook cannot be removed: install it only in a FRESH interpreter dedicated to the check.

    python -m utils.node_blocker <module> [args…]     # run <module> as __main__ under the blocker
"""
from __future__ import annotations

import os
import runpy
import stat
import sys
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

#: When set, every attempt is APPENDED here (``pid<TAB>what<TAB>frames``) — the record that survives a child.
LOG_ENV = "GEN3AI_NODE_BLOCK_LOG"

#: The program names that ARE Node.
NODE_NAMES = ("node", "nodejs")

#: A path containing this is the Showdown submodule (the worktree's own, or main's through the dist symlink).
SUBMODULE_MARK = os.sep + os.path.join("deps", "pokemon-showdown")

#: The exit status of the PATH tripwire (distinct from node's own codes and from 127 "not found").
SHIM_EXIT = 97

#: ``(what was attempted, the calling frames as ``file:line`` strings, innermost last)`` per attempt.
ATTEMPTS: List[Tuple[str, List[str]]] = []


class NodeBlocked(RuntimeError):
    """Node (or the Showdown submodule) was reached although the interpreter was told it must not be."""


def _is_node(prog) -> bool:
    if isinstance(prog, (bytes, bytearray)):
        prog = os.fsdecode(prog)
    if not isinstance(prog, (str, os.PathLike)):
        return False
    return os.path.basename(os.fspath(prog)) in NODE_NAMES


def _argv0(args) -> object:
    """The program a spawn runs — looking THROUGH ``sh -c "<cmd>"`` (``shell=True`` reaches the hook as
    ``['/bin/sh', '-c', cmd]``), so ``subprocess.run("node x.js", shell=True)`` is a Node spawn too."""
    if isinstance(args, (str, bytes, os.PathLike)):
        parts = os.fsdecode(args).split()
        return parts[0] if parts else ""
    if isinstance(args, (list, tuple)) and args:
        first = args[0]
        if (isinstance(first, (str, bytes, os.PathLike))
                and os.path.basename(os.fsdecode(first)) in ("sh", "bash", "dash")
                and len(args) >= 3 and args[1] == "-c"):
            return _argv0(args[2])
        return first
    return ""


def _record(what: str) -> None:
    frames = [f"{f.filename}:{f.lineno}" for f in traceback.extract_stack()[:-2] if "node_blocker" not in f.filename]
    ATTEMPTS.append((what, frames))
    log = os.environ.get(LOG_ENV)
    if log:
        try:
            with open(log, "a") as fh:     # the hook ignores the log itself (not under the submodule)
                fh.write(f"{os.getpid()}\t{what}\t{' <- '.join(reversed(frames[-4:]))}\n")
        except OSError:
            pass
    raise NodeBlocked(f"{what} while Node is blocked — from {frames[-1] if frames else '?'}")


_IN_HOOK = False


def _hook(event: str, args: tuple) -> None:
    global _IN_HOOK
    if _IN_HOOK:
        return
    _IN_HOOK = True
    try:
        if event == "subprocess.Popen":
            executable, argv = args[0], args[1]
            if _is_node(executable) or _is_node(_argv0(argv)):
                _record(f"spawn {argv!r}")
        elif event in ("os.exec", "os.posix_spawn"):
            path, argv = args[0], args[1]
            if _is_node(path) or _is_node(_argv0(argv)):
                _record(f"{event} {argv!r}")
        elif event == "os.system":
            if _is_node(_argv0(args[0])):
                _record(f"os.system {args[0]!r}")
        elif event == "open":
            path = args[0]
            if isinstance(path, (str, bytes, os.PathLike)) and SUBMODULE_MARK in os.fsdecode(path):
                _record(f"open {os.fsdecode(path)!r} (the Showdown submodule is hidden)")
    finally:
        _IN_HOOK = False


_INSTALLED = False


def install() -> None:
    """Add the audit hook (idempotent; it cannot be removed — use a fresh interpreter)."""
    global _INSTALLED
    if not _INSTALLED:
        sys.addaudithook(_hook)
        _INSTALLED = True


def report() -> str:
    """The attempts as text (empty when there were none)."""
    return "\n".join(f"{what}\n    " + "\n    ".join(frames[-6:]) for what, frames in ATTEMPTS)


def shim_env(shim_dir: Path, log: Path, env: Optional[dict] = None) -> dict:
    """``env`` (default ``os.environ``) with a ``node`` TRIPWIRE first on ``PATH`` and the attempt log set.

    The tripwire (``<shim_dir>/node`` and ``nodejs``) appends ``<pid> node <args>`` to ``log`` and exits
    :data:`SHIM_EXIT` — so a CHILD process (which the audit hook cannot see) that runs ``node`` fails and is
    recorded."""
    shim_dir = Path(shim_dir)
    shim_dir.mkdir(parents=True, exist_ok=True)
    for name in NODE_NAMES:
        p = shim_dir / name
        p.write_text(f'#!/bin/sh\necho "$$\t{name} $*\tPATH tripwire" >> "${LOG_ENV}"\nexit {SHIM_EXIT}\n')
        p.chmod(p.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    out = dict(os.environ if env is None else env)
    out["PATH"] = f"{shim_dir}{os.pathsep}{out.get('PATH', '')}"
    out[LOG_ENV] = str(log)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    """``python -m utils.node_blocker <module> [args…]``: run ``<module>`` as ``__main__`` under the blocker; exit
    non-zero if ANY attempt was made (even one the module swallowed)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print(__doc__)
        return 2
    install()
    sys.argv = argv
    code = 0
    try:
        runpy.run_module(argv[0], run_name="__main__", alter_sys=True)
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 1)
    if ATTEMPTS:
        print(f"[node_blocker] {len(ATTEMPTS)} Node attempt(s):\n{report()}", file=sys.stderr)
        return code or 1
    return code


if __name__ == "__main__":
    sys.exit(main())
