"""Make ``import poke_env`` IMPOSSIBLE in this interpreter and record every attempt (P1 of the retirement, ``T27``).

**Why.** The import gate (``poke_env_import_gate_test.py``) counts files that NAME ``poke_env``; it cannot say
whether a given entry point's import CLOSURE (or its RUN-TIME path) reaches one. A ``sys.modules`` look after an
import answers the first half only — on 2026-10-06 ``main.h2h`` / ``main.plateau`` loaded 0 poke-env modules at
import and 36 at run time (the P2 report, survey finding A-F7). This blocker answers both: install it first, then
import / RUN the entry point; every ``import poke_env[.x]`` raises :class:`PokeEnvBlocked` (an ``ImportError``)
AND is appended to :data:`ATTEMPTS` with the import stack, so a ``try: … except ImportError`` that swallows the
raise still shows up in the record.

**Use.** ``install()`` once, early, in a FRESH interpreter (a pytest session has long since imported the fork)::

    from utils import poke_env_blocker
    poke_env_blocker.install()
    import main.h2h                     # raises PokeEnvBlocked at the first poke_env import, naming the importer
    assert not poke_env_blocker.ATTEMPTS

``python -m utils.poke_env_blocker <module> [args…]`` runs ``<module>`` as ``__main__`` under the blocker and exits
non-zero if ANY attempt was made — the form the entry-point test and a manual trainer smoke use.
"""
from __future__ import annotations

import importlib.abc
import importlib.machinery
import os
import runpy
import sys
import traceback
from typing import List, Optional, Tuple

PACKAGE = "poke_env"

#: When set, every attempt is APPENDED to this file (``pid<TAB>module<TAB>importing frames``) — the only record that
#: survives a CHILD interpreter (a subprocess the entry point spawns gets its own ``ATTEMPTS``).
LOG_ENV = "GEN3AI_POKE_ENV_BLOCK_LOG"

#: ``(module name asked for, the importing frames as ``file:line`` strings, innermost last)`` per attempt.
ATTEMPTS: List[Tuple[str, List[str]]] = []


class PokeEnvBlocked(ImportError):
    """``poke_env`` was imported although the interpreter was told it must not be."""


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname: str, path=None, target=None) -> Optional[importlib.machinery.ModuleSpec]:
        if fullname == PACKAGE or fullname.startswith(PACKAGE + "."):
            frames = [f"{f.filename}:{f.lineno}" for f in traceback.extract_stack()[:-1]
                      if "importlib" not in f.filename and "poke_env_blocker" not in f.filename]
            ATTEMPTS.append((fullname, frames))
            log = os.environ.get(LOG_ENV)
            if log:
                try:
                    with open(log, "a") as f:
                        f.write(f"{os.getpid()}\t{fullname}\t{' <- '.join(reversed(frames[-4:]))}\n")
                except OSError:
                    pass
            raise PokeEnvBlocked(
                f"import of {fullname!r} while poke_env is blocked (P1 of the poke-env retirement) — imported from "
                f"{frames[-1] if frames else '?'}")
        return None


_INSTALLED: Optional[_Blocker] = None


def install() -> None:
    """Put the blocker first on ``sys.meta_path`` and evict any ``poke_env`` module already loaded (idempotent)."""
    global _INSTALLED
    for name in [k for k in sys.modules if k == PACKAGE or k.startswith(PACKAGE + ".")]:
        del sys.modules[name]
    if _INSTALLED is None:
        _INSTALLED = _Blocker()
        sys.meta_path.insert(0, _INSTALLED)


def report() -> str:
    """The attempts as text (empty when there were none)."""
    return "\n".join(f"{name}\n    " + "\n    ".join(frames[-6:]) for name, frames in ATTEMPTS)


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: python -m utils.poke_env_blocker <module> [args…]", file=sys.stderr)
        return 2
    module, rest = argv[0], argv[1:]
    install()
    sys.argv = [module, *rest]
    code = 0
    try:
        runpy.run_module(module, run_name="__main__", alter_sys=True)
    except SystemExit as e:                      # the entry point's own exit status is reported, then the verdict
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    if ATTEMPTS:
        print(f"[poke_env_blocker] {len(ATTEMPTS)} attempt(s) to import poke_env:\n{report()}", file=sys.stderr)
        return 97
    return code


if __name__ == "__main__":
    sys.exit(main())
