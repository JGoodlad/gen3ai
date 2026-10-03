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


def test_the_rust_core_loads_its_stable_and_exploiter_opponents_on_the_CPU():
    """P10 follow-up F1: a stable / exploiter opponent is a weight source T2 copies into its slot; loaded on
    the card it stayed there for the run beside that slot. Every `load_foreign_opponent` in `rust_env_setup`
    (the stable loop, the exploiter) names `device="cpu"`."""
    import ast

    from main.train import rust_env_setup
    tree = ast.parse(inspect.getsource(rust_env_setup))
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and getattr(n.func, "id", getattr(n.func, "attr", "")) == "load_foreign_opponent"]
    assert len(calls) == 2, "the stable loop and the exploiter — a new load site needs this test's attention"
    for c in calls:
        dev = {k.arg: k.value for k in c.keywords}.get("device")
        assert isinstance(dev, ast.Constant) and dev.value == "cpu", ast.dump(c)


@pytest.mark.parametrize("model_id", ["stable:a", "exploiter:target"])
def test_a_stable_or_exploiter_weight_source_on_the_card_is_REFUSED(model_id):
    """The pool's refusal now covers EVERY source (`OpponentSources._require_cpu`): a stable / exploiter
    policy on the card is a duplicate of its slot for the whole run."""
    cpu = torch.nn.Linear(2, 2, device="cpu")
    assert B.OpponentSources(stable={"a": cpu}, exploiter=cpu).policy_for(model_id) is cpu
    card = torch.nn.Linear(2, 2, device="meta")                   # any non-CPU device
    with pytest.raises(LazyAcquisitionError, match="must stay on the CPU"):
        B.OpponentSources(stable={"a": card}, exploiter=card).policy_for(model_id)


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
