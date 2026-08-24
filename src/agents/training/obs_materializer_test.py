"""Unit tests for the branch snapshot's FAST CLONE — the pickle-per-restore path.

The bit-identity of the arms themselves is pinned by
``obs_materializer_branch_integration_test.py`` on real bridge battles. What is pinned HERE is
the object-graph contract that makes that equivalence hold, in a form that runs in milliseconds
and fails for a readable reason:

* a pinned object comes back as the SAME object (``persistent_id`` ≡ ``memo=pins``);
* an object reachable from two of the three structures is DUPLICATED, not aliased — the reason
  :meth:`_PlayerSnapshot._freeze` writes three separate blobs and never one;
* restores are independent of each other and of the snapshot;
* a graph that will not pickle falls back to ``deepcopy`` instead of raising, and says so once.

Written against the 2026-08-23 measurement that ``restore`` was 57% of the materializer's
per-arm cost (3.69 of 6.45 ms) and ~25% of the search-dividend probe's whole wall clock.
"""
from __future__ import annotations

import copy
import logging
import threading

import pytest

from agents.training import obs_materializer as om


class _Node:
    """A tiny mutable graph node — stands in for the poke-env battle graph."""

    def __init__(self, name, kid=None):
        self.name = name
        self.kid = kid
        self.hits = []


class _Unpicklable:
    """Deep-copyable but NOT picklable — the shape a real one has (a lock, a socket, a live
    logger handler): ``deepcopy`` is satisfied by a hook, ``pickle`` refuses outright."""

    def __deepcopy__(self, memo):
        return _Unpicklable()

    def __reduce__(self):
        raise TypeError("cannot pickle _Unpicklable")


class _Queue:
    def __init__(self, n=0):
        self._n = n

    def qsize(self):
        return self._n

    def get_nowait(self):
        self._n -= 1

    def put_nowait(self, _):
        self._n += 1


class _FakePlayer:
    def __init__(self, battles, trackers, stall_loggers):
        self._battles = battles
        self._trackers = trackers
        self._stall_loggers = stall_loggers
        self._materialized = []
        self._actions_exhausted = False
        self._stopped = False
        self.action_choices = {0: "move tackle"}
        self._battle_count_queue = _Queue(0)
        self._trying_again = threading.Event()


def _player(*, unpicklable: bool = False):
    """A player whose three structures share one node and one logger, as the real one does."""
    logger = logging.getLogger("obs_materializer_test")
    shared = _Node("shared-by-battles-and-trackers")
    battles = {"tag": _Node("battle", kid=shared)}
    battles["tag"].log = logger
    trackers = {"tag": _Node("tracker", kid=shared)}
    stall = {"tag": _Node("stall")}
    if unpicklable:
        battles["tag"].oops = _Unpicklable()
    return _FakePlayer(battles, trackers, stall), logger, shared


def test_the_fast_path_is_taken_on_an_ordinary_graph():
    player, _logger, _shared = _player()
    snap = om._PlayerSnapshot(player)
    assert snap._blobs is not None and len(snap._blobs) == 3
    assert all(isinstance(b, bytes) and b for b in snap._blobs)


def test_a_pinned_object_comes_back_as_ITSELF():
    """``persistent_id`` must reproduce what ``deepcopy(memo=pins)`` does: share, never copy.

    A copied ``logging.Logger`` would drag in every logger in the process — the reason
    ``_SHARED_TYPES`` exists at all."""
    player, logger, _shared = _player()
    snap = om._PlayerSnapshot(player)
    snap.restore(player)
    assert player._battles["tag"].log is logger


def test_an_object_reachable_from_TWO_structures_is_duplicated_not_aliased():
    """Three blobs, three memos — exactly three ``deepcopy`` calls.

    One combined blob would alias the shared node across ``battles`` and ``trackers``, which is a
    DIFFERENT object graph: a mutation through one arm's battle would then be visible through its
    tracker. Deepcopy does not do that, so neither may the fast path."""
    player, _logger, _shared = _player()
    snap = om._PlayerSnapshot(player)
    snap.restore(player)
    assert player._battles["tag"].kid is not player._trackers["tag"].kid
    # and the same is true of the deepcopy path it must match
    snap._blobs = None
    snap.restore(player)
    assert player._battles["tag"].kid is not player._trackers["tag"].kid


def test_restores_are_independent_of_each_other_and_of_the_snapshot():
    player, _logger, _shared = _player()
    snap = om._PlayerSnapshot(player)
    snap.restore(player)
    player._battles["tag"].hits.append("arm-0")
    first = player._battles["tag"]
    snap.restore(player)
    assert player._battles["tag"].hits == []
    assert player._battles["tag"] is not first
    assert snap.battles["tag"].hits == []


def test_the_fast_path_and_the_deepcopy_path_agree_field_for_field():
    player, _logger, _shared = _player()
    snap = om._PlayerSnapshot(player)
    snap.restore(player)
    fast = (player._battles["tag"].name, player._battles["tag"].kid.name,
            player._trackers["tag"].name, player._stall_loggers["tag"].name)
    snap._blobs = None
    snap.restore(player)
    slow = (player._battles["tag"].name, player._battles["tag"].kid.name,
            player._trackers["tag"].name, player._stall_loggers["tag"].name)
    assert fast == slow


def test_an_unpicklable_graph_FALLS_BACK_instead_of_raising(capsys):
    """A ~9x slowdown that nothing mentions is the failure mode this project eats most often, so
    the fall-back is announced — once."""
    om._FREEZE_WARNED = False
    player, _logger, _shared = _player(unpicklable=True)
    snap = om._PlayerSnapshot(player)
    assert snap._blobs is None
    snap.restore(player)                    # still works, via deepcopy
    assert player._battles["tag"].name == "battle"
    assert "fell back to deepcopy" in capsys.readouterr().err
    # second snapshot: still no blobs, but no second warning
    om._PlayerSnapshot(player)
    assert "fell back to deepcopy" not in capsys.readouterr().err


def test_GenData_is_pinned_so_the_two_clone_paths_cannot_disagree():
    """``GenData.__deepcopy__`` returns ``self``; pickle honours no such hook. Unpinned, the fast
    path would deep-copy the whole gen-3 dex into every arm — a silent semantic AND cost defect."""
    from poke_env.data.gen_data import GenData

    dex = GenData.from_gen(3)
    assert copy.deepcopy(dex) is dex, "the singleton contract this pin mirrors"
    player, _logger, _shared = _player()
    player._battles["tag"].dex = dex
    snap = om._PlayerSnapshot(player)
    assert id(dex) in snap._pins
    snap.restore(player)
    assert player._battles["tag"].dex is dex


@pytest.mark.parametrize("unpicklable", [False, True])
def test_restore_reconciles_the_battle_count_queue_either_way(unpicklable):
    """The queue bookkeeping is outside the blob and must not regress with the fast path."""
    om._FREEZE_WARNED = True
    player, _logger, _shared = _player(unpicklable=unpicklable)
    player._battle_count_queue = _Queue(2)
    snap = om._PlayerSnapshot(player)
    player._battle_count_queue = _Queue(0)
    snap.restore(player)
    assert player._battle_count_queue.qsize() == 2
