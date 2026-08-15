"""The tier-ordering contract, asserted (`gen3_tiered_pipeline_v1`).

Three things are checked, and the third is what makes the first two worth having:

1. the production forward, and a forward with EVERY optional phase on, satisfy the contract;
2. every `nn.Module` child of the extractor is either tiered or explicitly untiered, so a new
   phase cannot escape the contract by simply not being listed;
3. **the checker actually fails when the invariant is broken** — one planted out-of-order call
   and one planted cross-forward leak, each asserted to be reported. A contract test that cannot
   fail is worse than no test, so the falsification cases are first-class here rather than a
   comment claiming the check is sound.

The honest boundary is documented on `tier_contract`: this is a check on DATA FLOW (a later
tier's tensor reaching an earlier tier, and the order in which tiers are entered), not on
MEANING (a T0 leg *recomputing* something intent-like from raw tokens is invisible to it).
"""
from __future__ import annotations

import inspect
import json
import os

import gymnasium as gym
import numpy as np
import pytest
import torch

from agents.model.damage_tables import sanitize_historical_move_floor
from agents.model.features_extractor import Gen3FeaturesExtractor
from agents.model.tier_contract import (
    TIER_NAMES,
    TIER_OF,
    UNTIERED_CHILDREN,
    assert_tier_contract,
    trace_tiers,
)
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
_PRODUCTION_CONFIG = os.path.join(_REPO, "designs", "production_config.json")

# Every optional phase the contract tiers, switched on together — so the trace exercises all four
# tiers rather than only the ones production happens to build.
_ALL_ON = dict(
    opp_belief_slots=True, opp_belief_cls_k=2,
    spread_belief=True, opp_intent=True, species_prior_fusion=True,
    win_prob_mode="read_only", pubval_mode="read_only",
    value_dist_mode="read_only", value_dist_bins=51,
    value_dist_vmin=-30.0, value_dist_vmax=30.0,
    value_threat_inject=True,
)


def _build(**overrides):
    """A REAL extractor on the production config (optionally overridden), plus a dummy obs."""
    with open(_PRODUCTION_CONFIG) as fh:
        cfg = json.load(fh)
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    sig = set(inspect.signature(Gen3FeaturesExtractor.__init__).parameters)
    kwargs = {k: v for k, v in cfg.items() if k in sig}
    sanitize_historical_move_floor(kwargs)
    kwargs.update(overrides)
    space = gym.spaces.Box(0.0, 1.0, shape=(layout["total_dim"],), dtype=np.float32)
    torch.manual_seed(0)
    fe = Gen3FeaturesExtractor(space, layout=layout, mappings=mappings, **kwargs).eval()
    if hasattr(fe, "disable_observation_debugger"):
        fe.disable_observation_debugger()
    g = torch.Generator().manual_seed(7)
    obs = {"observation": torch.rand((3, layout["total_dim"]), generator=g)}
    return fe, obs


@pytest.fixture(scope="module")
def production():
    return _build()


@pytest.fixture(scope="module")
def all_on():
    return _build(**_ALL_ON)


# --------------------------------------------------------------------- the contract holds

def test_production_forward_satisfies_the_tier_contract(production):
    fe, obs = production
    trace = assert_tier_contract(fe, obs)
    assert trace.order, "the tracer saw no tier-declared entry point at all"


def test_full_stack_forward_satisfies_the_tier_contract(all_on):
    fe, obs = all_on
    assert_tier_contract(fe, obs)


def test_the_trace_is_not_vacuous_all_four_tiers_run_in_order(all_on):
    """With every optional phase on, the forward must visit T0 -> T1 -> T2 -> T3, in that order.

    Without this, a contract that only ever observed one tier would pass trivially.
    """
    fe, obs = all_on
    trace = trace_tiers(fe, obs)
    tiers = [t for _n, t in trace.order]
    firsts = sorted({t: tiers.index(t) for t in set(tiers)}.items())
    assert [t for t, _i in firsts] == [0, 1, 2, 3], (
        f"expected all four tiers, saw {sorted(set(tiers))}")
    assert [i for _t, i in firsts] == sorted(i for _t, i in firsts), (
        "the tiers were first entered out of order")
    assert tiers == sorted(tiers), "the observed tier sequence is not non-decreasing"


@pytest.mark.parametrize("fixture_name", ["production", "all_on"])
def test_every_child_module_declares_a_tier(fixture_name, request):
    """A new phase module must pick a tier (or be explicitly listed as untiered) to exist."""
    fe, _obs = request.getfixturevalue(fixture_name)
    children = set(dict(fe.named_children()))
    undeclared = sorted(children - UNTIERED_CHILDREN - set(TIER_OF))
    assert not undeclared, (
        f"these extractor children own no tier: {undeclared}. Add them to TIER_OF (choosing a "
        f"tier from {sorted(TIER_NAMES.values())}) or to UNTIERED_CHILDREN in tier_contract.py.")


def test_move_belief_is_instrumented_through_its_named_methods(production):
    """`MoveBelief` is never `__call__`ed — it is reached via `move_logits`/`reinject_moves`.

    Without the named-entry-point map it would be invisible to both checks, so the contract
    would silently stop guarding the very phase step 3 made unconditional.
    """
    fe, obs = production
    trace = trace_tiers(fe, obs)
    names = {n for n, _t in trace.order}
    assert "move_belief.move_logits" in names
    assert "move_belief.reinject_moves" in names


# --------------------------------------------------------------------- the contract can FAIL

def test_a_planted_out_of_order_call_is_detected(production):
    """Re-introducing a POST-transformer T0 call site must FAIL the order check.

    This is the exact regression the prefuse-unconditional change is exposed to: a `move_belief`
    (or any T0) call after the trunk. Planted here by invoking a T0 module from a pre-hook on the
    T3 assembler.
    """
    fe, obs = production
    held = {}
    h_ctx = fe.unpack.register_forward_hook(lambda _m, _a, out: held.__setitem__("ctx", out))

    def plant(_m, args, kwargs):
        fe.pokemon_encoder(held["ctx"], fe.embeddings)     # a T0 module, entered at T3
        return None

    h_plant = fe.assembler.register_forward_pre_hook(plant, with_kwargs=True)
    try:
        trace = trace_tiers(fe, obs)
    finally:
        h_plant.remove()
        h_ctx.remove()

    order_hits = [v for v in trace.violations if v.kind == "order"]
    assert order_hits, f"the planted out-of-order T0 call went undetected: {trace.violations}"
    assert any(v.module == "pokemon_encoder" for v in order_hits), order_hits
    # and the same extractor is clean once the plant is removed
    assert_tier_contract(fe, obs)


def test_a_planted_cross_forward_leak_is_detected(production):
    """A T0 leg reading a LATER tier's tensor stashed on the previous forward must FAIL.

    This is the pure provenance case — within one monotone forward a later tier's output does not
    exist yet, so the stale stash is the failure mode provenance uniquely catches. It is not
    hypothetical: the edge-cell builder used to guard `last_spread_belief` for exactly this
    reason. The leak is planted DETACHED and as a VIEW, which is what a real one would look like
    and what tensor-identity tracking would miss.
    """
    fe, obs = production
    stash = {}

    def capture(_m, _a, out):
        stash["late"] = out[0]                              # a T2 cls_pool output

    def leak(_m, args, kwargs):
        if "late" in stash:
            return args + (stash["late"][:, :4].detach(),), kwargs   # detached VIEW, T2 -> T0
        return None

    original_forward = fe.pokemon_encoder.forward
    fe.pokemon_encoder.forward = lambda *a, **k: original_forward(*a[:2], **k)
    h_cap = fe.cls_pool.register_forward_hook(capture)
    h_leak = fe.pokemon_encoder.register_forward_pre_hook(leak, with_kwargs=True)
    try:
        trace = trace_tiers(fe, obs, forwards=2)
    finally:
        h_leak.remove()
        h_cap.remove()
        del fe.pokemon_encoder.forward

    prov = [v for v in trace.violations if v.kind == "provenance"]
    assert prov, f"the planted cross-forward T2->T0 leak went undetected: {trace.violations}"
    assert any(v.module == "pokemon_encoder" and v.tier == 0 for v in prov), prov
    assert not [v for v in trace.violations if v.kind == "order"], (
        "the leak plant should be a PURE provenance failure — an order hit here would mean the "
        "two checks are not separable")
    assert_tier_contract(fe, obs)                            # clean again once un-planted


def test_assert_tier_contract_raises_with_the_observed_order(production):
    """The failure MESSAGE must carry the order it saw — a bare AssertionError is unactionable."""
    fe, obs = production
    held = {}
    h_ctx = fe.unpack.register_forward_hook(lambda _m, _a, out: held.__setitem__("ctx", out))
    def plant(_m, args, kwargs):
        fe.pokemon_encoder(held["ctx"], fe.embeddings)
        return None

    h_plant = fe.assembler.register_forward_pre_hook(plant, with_kwargs=True)
    try:
        with pytest.raises(AssertionError, match="tier-ordering contract violated"):
            assert_tier_contract(fe, obs)
    finally:
        h_plant.remove()
        h_ctx.remove()
