"""The head-to-head tool's base error (its own module so ``main.h2h.arch`` and ``main.h2h.play`` share it)."""
from __future__ import annotations


class H2HError(RuntimeError):
    """A head-to-head read that cannot be run honestly (incompatible players, a pinned-team run, a refused
    resume, a game log that does not score)."""
