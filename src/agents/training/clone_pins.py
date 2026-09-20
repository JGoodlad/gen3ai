"""``clone_pins`` — which objects a per-arm CLONE must SHARE instead of copying.

One definition, two clone mechanisms. :mod:`agents.training.obs_materializer`'s
``_PlayerSnapshot`` pickles the whole replay player per decision and rebuilds it per arm;
:class:`agents.training.view_successor.ViewSuccessorFactory` deep-copies the ``EpisodeTracker``
per arm on the one-sided VIEW road. Both must pin the same set, and a second copy of the rules
would be a second chance for the two roads to clone differently — which is exactly the class of
difference a byte-identity gate is least able to explain.

The pin walk is run ONCE per fork point; the resulting ``{id: obj}`` map stays valid for every
clone taken from it, because a pinned object is the SAME object in the source graph each time.
"""

from __future__ import annotations

import io
import logging
import pickle
import types
from collections import deque
from types import MappingProxyType
from typing import Optional

import numpy as np


#: Objects that must be SHARED by a clone rather than copied. Two reasons, both real:
#:  * ``logging.Logger`` — deep-copying one drags in ``Logger.manager.loggerDict``, i.e.
#:    EVERY logger in the process (torch's included), whose handlers hold
#:    ``_thread.RLock``; that is unpicklable, and a logger is shared infrastructure, not
#:    per-battle state.
#:  * ``MappingProxyType`` — unpicklable too, and copying one would be WRONG anyway: it
#:    is a read-only view (``LegalActions.last_request``), so a copy is at best a
#:    type-changing no-op.
_SHARED_TYPES: tuple = (logging.Logger, logging.Manager, logging.Handler, MappingProxyType)

def _immutable_record_types() -> tuple:
    """Additionally shared: APPEND-ONLY IMMUTABLE RECORDS.

    A branch APPENDS to the event log and the tracker history; it never rewrites an existing
    entry. ``BattleEvent`` is a ``frozen=True`` dataclass documented as "one immutable,
    ordered record"; ``BattleContext`` is a per-turn snapshot built once in
    ``EpisodeTracker.record()``. Sharing them instead of copying them is therefore exactly
    equivalent, AND it is the difference between a clone that costs O(turns) of deep
    structure and one that costs O(live objects): measured at turn 12, pinning the 16
    history contexts took the tracker clone from 2.96 ms to 1.06 ms.

    ⚠️ This is a CONTRACT, not an inference. If either type ever gains in-place mutation,
    counterfactual arms after the first would read history the previous arm corrupted —
    which is precisely what ``obs_materializer_branch_integration_test.py`` exists to catch,
    and why that test compares EVERY arm rather than sampling one.

    Imported lazily so this module keeps its narrow import graph.
    """
    from agents.battle.battle_event import BattleEvent
    from agents.training.battle_snapshot import BattleContext
    return (BattleEvent, BattleContext)


def _shared_singleton_types() -> tuple:
    """Also shared: PROCESS SINGLETONS that declare themselves one via ``__deepcopy__``.

    ``GenData.__deepcopy__`` returns ``self``, so the deepcopy path shares the gen-3 dex whether
    or not it is pinned. :meth:`_PlayerSnapshot._freeze` uses PICKLE, which honours no such hook
    and would deep-copy the whole dex into every arm instead. Pinning makes the two clone
    mechanisms provably agree rather than agreeing by accident of what is reachable today
    (measured 2026-08-23: ``GenData`` is NOT reachable from a live battle graph — the frozen blob
    is ~44 kB — so this is a latent-hazard guard, not a live fix).
    """
    from poke_env.data.gen_data import GenData
    return (GenData,)


def _pin_shared(obj, pins: dict, seen: "Optional[set]" = None) -> dict:
    """Walk ``obj``'s object graph and record ``{id: obj}`` for every member of
    :data:`_SHARED_TYPES`, so a later ``deepcopy(..., memo=pins)`` returns them
    unchanged. Run ONCE per snapshot; the resulting pin set stays valid for every restore
    (a restore copies from the snapshot, whose pinned objects are the same objects)."""
    if seen is None:
        seen = set()
    shared = _SHARED_TYPES + _immutable_record_types() + _shared_singleton_types()
    stack = [obj]
    while stack:
        o = stack.pop()
        i = id(o)
        if i in seen:
            continue
        seen.add(i)
        if isinstance(o, shared):
            pins[i] = o
            continue
        if isinstance(o, (str, bytes, bytearray, int, float, complex, bool,
                          type(None), np.ndarray, np.generic)):
            continue
        if isinstance(o, type) or isinstance(o, types.ModuleType):
            continue
        if isinstance(o, dict):
            stack.extend(o.keys())
            stack.extend(o.values())
            continue
        if isinstance(o, (list, tuple, set, frozenset, deque)):
            stack.extend(o)
            continue
        d = getattr(o, "__dict__", None)
        if isinstance(d, dict):
            stack.extend(d.values())
        for cls in type(o).__mro__:
            for slot in getattr(cls, "__slots__", ()) or ():
                try:
                    stack.append(getattr(o, slot))
                except AttributeError:
                    pass
    return pins


class _PinnedPickler(pickle.Pickler):
    """A pickler that writes PINNED objects by reference instead of copying them.

    The exact pickle-side counterpart of ``deepcopy(obj, memo=pins)``: the same
    ``{id: obj}`` set that a memo pre-seeds is here turned into ``persistent_id`` indices, so a
    shared logger / mapping proxy / immutable record comes back as itself."""

    def __init__(self, buf, pin_idx: dict):
        super().__init__(buf, protocol=pickle.HIGHEST_PROTOCOL)
        self._pin_idx = pin_idx

    def persistent_id(self, obj):
        return self._pin_idx.get(id(obj))


class _PinnedUnpickler(pickle.Unpickler):
    """The read side: resolve a persistent id back to the very object that was pinned."""

    def __init__(self, buf, pin_list: list):
        super().__init__(buf)
        self._pin_list = pin_list

    def persistent_load(self, pid):
        return self._pin_list[pid]


def _pickle_pinned(obj, pin_idx: dict) -> bytes:
    buf = io.BytesIO()
    _PinnedPickler(buf, pin_idx).dump(obj)
    return buf.getvalue()


def _unpickle_pinned(blob: bytes, pin_list: list):
    return _PinnedUnpickler(io.BytesIO(blob), pin_list).load()
