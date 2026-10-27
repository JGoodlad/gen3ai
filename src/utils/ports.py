"""THE ports rule for every tool that starts a server (``designs/tools/convergence_2026-10-26.md``).

* :data:`RESERVED_PORTS` -- 8000 / 8001, with the reason each refusal prints (root rule 1). Every in-repo refusal reads
  this one dict; the Metamon peer (``main/anchors/peer_scripts/metamon_side.py``) runs under another interpreter and
  never imports this tree, so it keeps a literal copy that ``ports_test`` pins equal to this one.
* :func:`bindable` -- can a server bind the port right now (bound and released at once)?
* :func:`first_bindable` / :func:`claim` -- pick the first free port of a range; :func:`claim` also takes an exclusive
  ``flock`` on ``<lock_dir>/<port>.lock`` held for the unit's life, so two supervisors never pick one port, which
  bind-and-release alone cannot promise (D1 look 1's 9540 collision).
* :data:`PORT_LOCK_DIR` -- THE lock directory every claimant uses (the Kakuna lanes, a standalone ``main.anchors`` read,
  a test session): claimants whose ranges overlap exclude each other only if they flock the same files.
* :func:`claimed` / :data:`TEST_PORT_RANGE` -- the same claim as a ``with`` block, for a test that starts a server on a
  port it picked (the root conftest's ``claimed_port`` fixture is its function-scoped form).

The probe's ``host`` and ``reuse_addr`` are the CALLER's, unchanged from each tool's own copy before convergence: the
anchors server and the team-strength lane probe ``127.0.0.1`` with ``SO_REUSEADDR`` (as their server sets it, so a
previous unit's TIME_WAIT connections do not read as busy); the Kakuna queue probes ``0.0.0.0`` without it.

STDLIB ONLY: it sits under the play path (``main.reads.playpath``) and the websocket front end.
"""
from __future__ import annotations

import contextlib
import os
import socket
from pathlib import Path
from typing import IO, Container, Iterator, Optional, Tuple

from utils.filelock import try_exclusive

#: Every claimant's lock directory (one ``<port>.lock`` flock per port, held for the claimant's unit): the Kakuna lanes
#: (``main.x35.kakuna.PORT_LOCK_DIR``), a standalone ``main.anchors`` read (``main.anchors.server.pick_port``) and the
#: test sessions' ``claimed_port`` fixture all claim here, so none of them takes a port another holds. The directory
#: keeps the name the live Kakuna supervisors already hold their locks in: a rename would hide their claims.
PORT_LOCK_DIR = Path.home() / ".claude" / "jobs" / "x35_kakuna_ports"

#: Where the test sessions claim their server ports: clear of the Kakuna lanes' 9540-9599 and a standalone anchors
#: read's default 9500-9599, so a test never waits on a live read's ports even before the shared lock dir says so.
TEST_PORT_RANGE = (9600, 9699)

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
          reuse_addr: bool = True, skip: Container[int] = ()) -> Optional[Tuple[int, IO[str]]]:
    """``(port, held lock file)``: the first port in ``[lo, hi]`` not in ``skip`` that no OTHER claimant holds (an
    exclusive flock on ``<lock_dir>/<port>.lock``) and :func:`bindable` passes; ``None`` when there is none. A port in
    ``skip`` is passed over before its lock file is touched (the reserved ports get no lock file). The caller keeps
    the file open until its unit ends; closing it releases the claim."""
    os.makedirs(lock_dir, exist_ok=True)
    for port in range(lo, hi + 1):
        if port in skip:
            continue
        fh = try_exclusive(os.path.join(lock_dir, f"{port}.lock"), "w")
        if fh is None:
            continue
        if not bindable(port, host=host, reuse_addr=reuse_addr):
            fh.close()
            continue
        return port, fh
    return None


@contextlib.contextmanager
def claimed(lo: int = TEST_PORT_RANGE[0], hi: int = TEST_PORT_RANGE[1],
            lock_dir: "str | os.PathLike[str]" = PORT_LOCK_DIR, *, host: str = "127.0.0.1",
            reuse_addr: bool = True) -> Iterator[int]:
    """A port in ``[lo, hi]`` CLAIMED (:func:`claim`, the reserved ports skipped) for the ``with`` block: held while
    the server the block starts binds it and serves, released when the block ends. ``RuntimeError`` when none is free
    and unclaimed. A test that picked a port by bind-and-release shared it with any other session that did the same."""
    got = claim(lo, hi, lock_dir, host=host, reuse_addr=reuse_addr, skip=RESERVED_PORTS)
    if got is None:
        raise RuntimeError(f"no free, unclaimed port in {lo}-{hi} (claims in {lock_dir})")
    port, fh = got
    try:
        yield port
    finally:
        fh.close()
