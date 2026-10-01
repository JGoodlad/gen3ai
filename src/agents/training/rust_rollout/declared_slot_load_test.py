"""A pool refresh is a DECLARED LOAD into a T2 slot, never a new allocation (`gen3_declared_slot_load_v1`).

Sizing arm A (2026-10-01): the quiescent CUDA floor stepped up ~33 MiB at every self-play promotion —
the rust core's pool loaded each snapshot ONTO the card and kept it in its LRU beside the slot T2 had
copied it into. Now the pool loads on the CPU, a pool weight source on the card is a typed refusal, and
a slot load that leaves memory allocated is one too. Each test fails on revert of the part it names."""
from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch

from agents.training.learner_lifecycle import LazyAcquisitionError
from agents.training.rust_rollout import build as B


def test_the_rust_core_pool_loads_its_snapshots_on_the_CPU():
    from main.train import rust_env_setup
    src = inspect.getsource(rust_env_setup)
    i = src.index("sources.pool = SnapshotPool(")
    assert 'device="cpu"' in src[i:i + 200], src[i:i + 200]


def _sources(device):
    pol = torch.nn.Linear(2, 2, device=device)
    pol.eval = lambda: pol
    entry = SimpleNamespace(step=7)
    pool = SimpleNamespace(_entries=[entry], load_model=lambda e: SimpleNamespace(policy=pol))
    return B.OpponentSources(pool=pool)


def test_a_pool_weight_source_on_the_card_is_REFUSED():
    assert isinstance(_sources("cpu").policy_for("pool:7"), torch.nn.Module)
    with pytest.raises(LazyAcquisitionError, match="must stay on the CPU"):
        _sources("meta").policy_for("pool:7")             # any non-CPU device


def test_a_slot_load_that_leaves_memory_allocated_is_REFUSED():
    loads = []
    svc = SimpleNamespace(load=lambda slot, pol, mid: loads.append((slot, mid)))
    state = {"bytes": 1000}

    def grows(device):
        return state["bytes"]

    def policy():
        state["bytes"] += 33 << 20                           # the load acquires 33 MiB and keeps it
        return object()
    with pytest.raises(LazyAcquisitionError, match="NEWLY allocated"):
        B.checked_slot_load(svc, 3, policy, "pool:7", "cuda:0", allocated=grows)
    state["bytes"] = 1000
    assert B.checked_slot_load(svc, 3, lambda: object(), "pool:8", "cuda:0", allocated=grows) == 0
    assert loads == [(3, "pool:7"), (3, "pool:8")]
