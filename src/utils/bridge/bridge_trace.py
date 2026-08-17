"""Opt-in raw wire trace for the local bridge — the tool for diagnosing a SILENT stall.

When a bridge battle wedges, the two candidate stories look identical from outside: the CHILD
stopped emitting, or PYTHON stopped answering. Stack dumps can't tell them apart (both sides are
just waiting), and the failure only reproduces under a real policy at scale, so a unit test won't
reach it. This records the actual frames, in order, with timestamps — so the tail of the log names
whoever fell silent FIRST.

OFF unless ``POKESIM_BRIDGE_TRACE`` is set to a directory; then each process writes
``<dir>/bridge_<pid>.trace`` with one line per frame:

    <monotonic-ms> OUT <command>
    <monotonic-ms> IN  <side> <first protocol line of the chunk>

Deliberately cheap and lossy-safe: line-buffered appends, no locks (each env worker is a separate
process with its own file), payloads truncated. It is a DIAGNOSTIC, not a protocol record — the
byte-exact record is the bridge's own golden corpus.
"""

from __future__ import annotations

import os
import time

_TRACE_ENV = "POKESIM_BRIDGE_TRACE"
_MAX_PAYLOAD = 240

_fh = None
_init = False


def _handle():
    """Lazily open this process's trace file; None when tracing is off."""
    global _fh, _init
    if _init:
        return _fh
    _init = True
    d = os.environ.get(_TRACE_ENV)
    if not d:
        return None
    try:
        os.makedirs(d, exist_ok=True)
        _fh = open(os.path.join(d, f"bridge_{os.getpid()}.trace"), "a", buffering=1)
    except OSError:
        _fh = None
    return _fh


def enabled() -> bool:
    return _handle() is not None


def _emit(kind: str, payload: str) -> None:
    fh = _handle()
    if fh is None:
        return
    try:
        fh.write(f"{time.monotonic() * 1000.0:.1f} {kind} {payload[:_MAX_PAYLOAD]}\n")
    except (OSError, ValueError):  # closed / full — a diagnostic must never break the run
        pass


def out(command: str) -> None:
    """A command Python wrote to the child's stdin (START / CHOOSE / FORCELOSE / END)."""
    _emit("OUT", command.replace("\n", "\\n"))


def inbound(side: str, chunk: str) -> None:
    """A frame the child emitted. Logs the side tag + the chunk's first protocol line."""
    first = next((l for l in chunk.splitlines() if l.strip()), "")
    _emit("IN ", f"{side} {first}")


def note(text: str) -> None:
    """A free-form marker (e.g. a battle boundary) to anchor the timeline."""
    _emit("NOTE", text.replace("\n", " "))
