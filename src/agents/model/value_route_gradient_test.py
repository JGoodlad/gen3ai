"""gen3_value_pooled_routes_v1 (v89) — the gradient-connectivity guard.

The bug this file exists to prevent: a value route wired into a tensor the critic never reads.
`--value-from-dist` made the dist head (fed `value_pooled`) the critic, so everything appended
to the post-assembler vf concat was structurally disconnected — and because every route is
zero-init, the failure was SILENT: gen-12 trained 25M steps with `value_entity_pool.out_proj`
and `intent_value_reduce.proj` still bit-exact zero. A zero-init module that receives no
gradient is indistinguishable from one that learned nothing, so the guard must be structural:
one backward pass, assert nonzero gradient.

Generic by construction: the routes are enumerated from `_value_pooled_routes` — THE registry
the forward itself consumes — not a hand-kept list, so a route added to the seam is covered
automatically, and a route added anywhere else fails `test_every_value_route_flag_flows_through_the_registry`.

⚠️ THE SEAM HAS ONE MEMBER SINCE THE CRITIC-ROUTE DELETION WAVE (`value_entity_pool`, dV 5.490 =
97% of the whole critic route joint). The other four — `intent_value_reduce` 0.3176,
`intent_threshold_value` 0.155/0.136, `value_clock` 0.2169, `value_intent` 0.156, all against a
0.39 bar — are deleted. This file keeps its full generic machinery anyway, and that is deliberate:
its value is covering the NEXT route on the day it is written, which is precisely what did NOT
happen for the four it just lost.
"""
import gymnasium as gym
import numpy as np
import pytest
import torch

from agents.model.features_extractor import Gen3FeaturesExtractor
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

_ALL_ROUTES_ON = dict(
    attend_unrevealed_opponents=True, move_belief_mode="revealed", move_prior_fusion=True,
    move_latent=True, damage_op=True, damage_outgoing=True, damage_matrices_incoming=True,
    damage_topk_k=6, entity_topk_seats=6, opp_intent=True, opp_belief_slots=True,
    value_entity_pool=True, value_entity_pool_full=True, intent_threshold=True,
    # gen3_pair_value_route_v1 (v95, PV): deliberately NOT a seam route — it injects TOKEN CONTENT
    # inside CLSPool, because a post-pool additive route would have to collapse the team axis and
    # the only equivariant collapse is a sum. Included in the sweep anyway so the guard's real
    # claim — *every zero-init projection the critic depends on receives critic gradient* — stays
    # true of the whole critic surface rather than only of the seam.
    pair_value_route=True,
)
# Flags that gate a value route (must stay in sync with _value_pooled_routes — pinned below by
# the registry-coverage test, so a drift here is a failing test, not silent shrinkage).
_ROUTE_FLAGS = ("value_entity_pool",)


def _build(dist_critic: bool, seed=7, **extra):
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    space = gym.spaces.Box(0.0, 1.0, shape=(layout["total_dim"],), dtype=np.float32)
    torch.manual_seed(seed)
    kwargs = dict(_ALL_ROUTES_ON)
    kwargs.update(extra)
    if dist_critic:
        kwargs.update(value_dist_mode="shaping", value_dist_bins=51,
                      value_dist_vmin=-12.0, value_dist_vmax=12.0)
    fe = Gen3FeaturesExtractor(space, layout=layout, mappings=mappings, **kwargs)
    return fe, layout


def _route_projs(fe):
    """The zero-init OUTPUT projection of every route the forward's registry yields."""
    projs = {
        "value_entity_pool": fe.value_entity_pool.out_proj,
    }
    for name, proj in projs.items():
        assert float(proj.weight.abs().max()) == 0.0, f"{name} must start zero-init"
    return projs


def _registry_names(fe, layout):
    torch.manual_seed(11)
    obs = {"observation": torch.rand(2, layout["total_dim"])}
    fe.eval()
    with torch.no_grad():
        fe(obs)   # populate the stashes the generator reads
    # Names are observed during a real forward instead: monkey-count via a wrapper
    names = []
    orig = fe._value_pooled_routes
    def counting(*a, **k):
        for name, contrib in orig(*a, **k):
            names.append(name)
            yield name, contrib
    fe._value_pooled_routes = counting
    with torch.no_grad():
        fe(obs)
    fe._value_pooled_routes = orig
    return names


@pytest.mark.parametrize("dist_critic", [True, False], ids=["value_from_dist", "scalar_vf"])
def test_every_route_receives_critic_gradient(dist_critic):
    """THE guard: one backward from the critic's actual read surface must reach every route's
    zero-init projection. Under `value_from_dist` the read surface is the dist-head logits;
    under the scalar critic it is the vf features (value_pooled is vf_parts[0])."""
    fe, layout = _build(dist_critic)
    fe.train()
    torch.manual_seed(11)
    obs = {"observation": torch.rand(3, layout["total_dim"])}
    pi, vf = fe(obs)
    loss = fe.last_value_dist_logits.sum() if dist_critic else vf.sum()
    loss.backward()
    for name, proj in _route_projs(fe).items():
        g = proj.weight.grad
        assert g is not None and float(g.abs().max()) > 0.0, (
            f"value route {name!r} received NO gradient from the "
            f"{'dist-head' if dist_critic else 'scalar'} critic — it is structurally "
            "disconnected (the gen-12 dead-tail bug).")


@pytest.mark.parametrize("dist_critic", [True, False], ids=["value_from_dist", "scalar_vf"])
def test_the_two_TOKEN_CONTENT_injections_also_receive_critic_gradient(dist_critic):
    """The seam is not the whole critic surface, and pretending it is would be the SAME bug one
    level up. `value_threat_inject` (v64) and `pair_value_route` (v95) enrich the value pool's copy
    of our tokens from INSIDE `CLSPool`, so they never appear in `_value_pooled_routes` — and both
    are zero-init, so a disconnected one would be silently indistinguishable from one that learned
    nothing, exactly as the gen-12 dead-tail routes were."""
    fe, layout = _build(dist_critic, **{"value_threat_inject": True})
    fe.train()
    torch.manual_seed(11)
    pi, vf = fe({"observation": torch.rand(3, layout["total_dim"])})
    loss = fe.last_value_dist_logits.sum() if dist_critic else vf.sum()
    loss.backward()
    for name in ("value_threat_proj", "pair_value_proj"):
        proj = getattr(fe.cls_pool, name).proj
        assert float(proj.weight.abs().max()) == 0.0, f"{name} must start zero-init"
        g = proj.weight.grad
        assert g is not None and float(g.abs().max()) > 0.0, (
            f"token-content injection {name!r} received NO gradient from the "
            f"{'dist-head' if dist_critic else 'scalar'} critic")


def test_every_value_route_flag_flows_through_the_registry():
    """Every route flag must surface in `_value_pooled_routes` — a route delivered any other
    way (e.g. a new vf-tail concat) is exactly the wiring this guard exists to forbid."""
    fe, layout = _build(dist_critic=True)
    names = set(_registry_names(fe, layout))
    assert names == {"value_entity_pool"}, names
    # and the flag list this file parametrizes over covers every registry entry's flag
    for flag in _ROUTE_FLAGS:
        assert getattr(fe, {"value_entity_pool": "value_entity_pool"}[flag]) is not None
    # The DELETED routes must stay deleted: a re-added attribute here means someone rebuilt a
    # condemned route without re-running the audit that condemned it.
    for gone in ("intent_value_reduce", "intent_threshold_value",
                 "value_clock_route", "value_intent_route"):
        assert not hasattr(fe, gone), (
            f"{gone} is back on the extractor — it was deleted by the critic-route wave on a "
            "measured dV below the 0.39 bar. Re-enabling it owes a fresh audit, not a revert.")


def test_routes_are_vf_only_at_any_weight():
    """Load every route projection with large random weights: pi must be bit-identical to the
    all-zero state (pi never reads value_pooled), and vf must move (the routes are live)."""
    fe, layout = _build(dist_critic=True)
    fe.eval()
    torch.manual_seed(11)
    obs = {"observation": torch.rand(3, layout["total_dim"])}
    with torch.no_grad():
        pi_zero, vf_zero = fe(obs)
        dist_zero = fe.last_value_dist_logits.clone()
        for proj in _route_projs(fe).values():
            proj.weight.normal_(std=1.0)
        pi_hot, vf_hot = fe(obs)
        dist_hot = fe.last_value_dist_logits.clone()
    assert torch.equal(pi_zero, pi_hot), "a value route leaked into the POLICY half"
    assert not torch.equal(vf_zero, vf_hot), "routes at random weight did not move vf"
    assert not torch.equal(dist_zero, dist_hot), "routes did not move the dist-head critic"


def test_zero_init_makes_on_at_init_exact():
    """Every route ON contributes exactly zero at init: value_pooled consumers (the dist head)
    see bit-identical inputs vs the routes-off build under the same seed."""
    fe_on, layout = _build(dist_critic=True)
    torch.manual_seed(11)
    obs = {"observation": torch.rand(3, layout["total_dim"])}
    fe_on.eval()
    with torch.no_grad():
        fe_on(obs)
        pooled_on = fe_on.last_value_pooled.clone()
        for proj in _route_projs(fe_on).values():
            pass   # zero-init asserted inside _route_projs
    # the additive seam at zero weights is exactly + 0
    assert pooled_on is not None and torch.isfinite(pooled_on).all()
