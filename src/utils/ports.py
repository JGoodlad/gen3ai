"""THE ports rule for every tool that starts a server (``designs/tools/convergence_2026-10-26.md``).

* :data:`RESERVED_PORTS` -- 8000 / 8001, with the reason each refusal prints (root rule 1). Every in-repo refusal reads
  this one dict; the Metamon peer (``main/anchors/peer_scripts/metamon_side.py``) runs under another interpreter and
  never imports this tree, so it keeps a literal copy that ``ports_test`` pins equal to this one.
* :func:`bindable` -- can a server bind the port right now (bound and released at once)?
* :func:`first_bindable` / :func:`claim` -- pick the first free port of a range; :func:`claim` also takes an exclusive
  ``flock`` on ``<lock_dir>/<port>.lock`` held for the unit's life, so two supervisors never pick one port, which
  bind-and-release alone cannot promise (D1 look 1's 9540 collision).

The probe's ``host`` and ``reuse_addr`` are the CALLER's, unchanged from each tool's own copy before convergence: the
anchors server and the team-strength lane probe ``127.0.0.1`` with ``SO_REUSEADDR`` (as their server sets it, so a
previous unit's TIME_WAIT connections do not read as busy); the Kakuna queue probes ``0.0.0.0`` without it.

STDLIB ONLY: it sits under the play path (``main.reads.playpath``) and the websocket front end.
"""
from __future__ import annotations

import os
import socket
from typing import IO, Container, Optional, Tuple

from utils.filelock import try_exclusive

#: Ports no tool may bind, start a server on or dial, with the reason the refusal states.
RESERVED_PORTS = {8000: "the shared DEV server",
                  8001: "the reserved TRAINING server (the trainer itself is in-process and connects to none)"}


def bindable(port: int, *, host: str = "127.0.0.1", reuse_addr: bool = True) -> bool:
    """Can a TCP server bind ``(host, port)`` now? Bound and released at once (a small race window remains; a
    :func:`claim` closes it between our own supervisors)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if reuse_addr:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((host, port))
        except OSError:
            return False
    return True


def first_bindable(lo: int, hi: int, *, host: str = "127.0.0.1", reuse_addr: bool = True,
                   skip: Container[int] = ()) -> Optional[int]:
    """The first port in ``[lo, hi]`` not in ``skip`` that :func:`bindable` passes, or ``None``."""
    for port in range(lo, hi + 1):
        if port in skip:
            continue
        if bindable(port, host=host, reuse_addr=reuse_addr):
            return port
    return None


def claim(lo: int, hi: int, lock_dir: "str | os.PathLike[str]", *, host: str = "127.0.0.1",
          reuse_addr: bool = True) -> Optional[Tuple[int, IO[str]]]:
    """``(port, held lock file)``: the first port in ``[lo, hi]`` no OTHER claimant holds (an exclusive flock on
    ``<lock_dir>/<port>.lock``) and :func:`bindable` passes; ``None`` when there is none. The caller keeps the file
    open until its unit ends; closing it releases the claim."""
    os.makedirs(lock_dir, exist_ok=True)
    for port in range(lo, hi + 1):
        fh = try_exclusive(os.path.join(lock_dir, f"{port}.lock"), "w")
        if fh is None:
            continue
        if not bindable(port, host=host, reuse_addr=reuse_addr):
            fh.close()
            continue
        return port, fh
    return None
