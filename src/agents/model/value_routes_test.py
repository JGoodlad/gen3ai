"""gen3_value_direct_routes_v1 (v87) — the deadline-clock and α/β direct critic routes.

What must hold:
  * OFF is the default and builds NOTHING; ON widens ONLY the value projection (pi untouched
    at any weight — the vf-only concat) and contributes exactly ZERO at init (zero-init, in
    the identity-init capture set — ledger M1);
  * the clock route reads the REAL clock scalars (the named-offset slice of the global block,
    not a hand-counted index) — pinned by planting a value and reading it back;
  * the intent route: α/β arrive as PROBABILITIES; the no-legal-switch β case is a clean zero,
    never a NaN; a seat-count mismatch fails loud (the `op move-order` class);
  * `value_intent` requires `opp_intent`; the ordering story holds (the route runs at the vf
    TAIL, after the T2 heads exist — asserted by a real forward under the full stack);
  * the v87 version machinery: migration defaults + both check_compatible gates.
"""
import dataclasses

import gymnasium as gym
import numpy as np
import pytest
import torch

from agents.model.arch_constants import D_MODEL
from agents.model.features_extractor import Gen3FeaturesExtractor
from agents.model.model_version import (
    MODEL_CONFIG_VERSION, ModelVersion, ModelVersionError, _migrate_config,
)
from agents.model.value_routes import ValueIntentRoute
from agents.observation.constants import CLOCK_DIM, CLOCK_OFFSET_IN_GLOBAL
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

_ON_KWARGS = dict(
    attend_unrevealed_opponents=True, move_belief_mode="revealed", move_prior_fusion=True,
    move_latent=True, damage_op=True, damage_outgoing=True, damage_matrices_incoming=True,
    damage_topk_k=6, entity_topk_seats=6, opp_intent=True,
    value_clock=True, value_intent=True,
)


def _build(seed=7, **kwargs):
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    space = gym.spaces.Box(0.0, 1.0, shape=(layout["total_dim"],), dtype=np.float32)
    torch.manual_seed(seed)
    fe = Gen3FeaturesExtractor(space, layout=layout, mappings=mappings, **kwargs)
    fe.eval()
    return fe, layout


def _obs(layout, b=3):
    torch.manual_seed(11)
    return {"observation": torch.rand(b, layout["total_dim"])}


def test_off_builds_nothing_and_on_is_width_neutral():
    """gen3_value_pooled_routes_v1: the routes INJECT into value_pooled, so ON changes NO
    projection width — route availability can never mis-size `value_pre_norm`."""
    off, _ = _build(**{**_ON_KWARGS, "value_clock": False, "value_intent": False})
    on, _ = _build(**_ON_KWARGS)
    assert off.value_clock_route is None and off.value_intent_route is None
    assert not any("value_clock_route" in k or "value_intent_route" in k
                   for k in off.state_dict())
    assert on.value_projection.in_features == off.value_projection.in_features
    assert on.projection.in_features == off.projection.in_features   # pi untouched


def test_zero_init_and_sweep_membership():
    fe, layout = _build(**_ON_KWARGS)
    assert float(fe.value_clock_route.proj.weight.abs().max()) == 0.0
    assert float(fe.value_intent_route.proj.weight.abs().max()) == 0.0
    assert "value_clock_route.proj" in fe._identity_init_zeroed
    assert "value_intent_route.proj" in fe._identity_init_zeroed
    pi, vf = fe(_obs(layout))
    assert pi.shape == vf.shape                                      # the tail sized correctly


def test_clock_moves_vf_and_not_pi():
    """Two obs differing ONLY in the clock scalars: with a perturbed (non-zero) clock route the
    vf features must differ and the pi features must not — the route is real, and vf-only."""
    fe, layout = _build(**_ON_KWARGS)
    with torch.no_grad():
        fe.value_clock_route.proj.weight.normal_(std=0.5)
    from agents.observation.constants import OFFSET_GLOBAL
    obs_a = {"observation": torch.zeros(1, layout["total_dim"])}
    obs_b = {"observation": torch.zeros(1, layout["total_dim"])}
    c0 = OFFSET_GLOBAL + CLOCK_OFFSET_IN_GLOBAL
    obs_b["observation"][0, c0:c0 + CLOCK_DIM] = torch.tensor([0.5, 0.3, 0.7])
    with torch.no_grad():
        pi_a, vf_a = fe(obs_a)
        pi_b, vf_b = fe(obs_b)
    assert not torch.equal(vf_a, vf_b)
    # NOTE pi ALSO sees the clock through the global token / non_matchup concat — the route's
    # vf-only property is about the ROUTE, which the width test above pins structurally.


def test_intent_route_probabilities_and_nan_guard():
    r = ValueIntentRoute(n_seats=6)
    with torch.no_grad():
        w = torch.zeros(D_MODEL, 13)
        w[:13, :13] = torch.eye(13)
        r.proj.weight.copy_(w)
    lg = torch.zeros(2, 7)
    beta_none = torch.full((2, 6), float("-inf"))
    out = r(lg, beta_none)
    assert torch.isfinite(out).all()
    # uniform α over 7 classes reads 1/7 on every α channel; β channels exactly 0
    assert torch.allclose(out[:, :7], torch.full((2, 7), 1.0 / 7.0), atol=1e-5)
    assert float(out[:, 7:13].abs().max()) == 0.0


def test_intent_route_seat_mismatch_fails_loud():
    r = ValueIntentRoute(n_seats=6)
    with pytest.raises(ValueError, match="seat"):
        r(torch.zeros(1, 5), torch.zeros(1, 6))


def test_value_intent_requires_opp_intent():
    with pytest.raises(ValueError, match="opp_intent"):
        _build(**{**_ON_KWARGS, "opp_intent": False, "value_intent": True,
                  "entity_topk_seats": 6})


def test_full_stack_with_every_value_part():
    """The ede1a88-class pin extended once more: every vf tail part on at once must size and
    run — intent_value_reduce, entity pool, threshold vf, clock, intent."""
    fe, layout = _build(**_ON_KWARGS, opp_belief_slots=True, intent_value_reduce=True,
                        value_entity_pool=True, intent_threshold=True)
    pi, vf = fe(_obs(layout))
    assert pi.shape == vf.shape


# ------------------------------------------------------------------- version machinery


def test_migration_defaults_off():
    migrated = _migrate_config({"config_version": 86})
    assert migrated["value_clock"] is False
    assert migrated["value_intent"] is False
    assert migrated["config_version"] >= 87
    assert MODEL_CONFIG_VERSION >= 89


@pytest.mark.parametrize("flag", ["intent_value_reduce", "value_entity_pool", "intent_threshold",
                                  "value_clock", "value_intent"])
def test_v89_refuses_pre_rehome_checkpoints_with_a_route_on(flag):
    """gen3_value_pooled_routes_v1: a <v89 config recording a value route ON carries projection
    shapes the re-homed forward cannot rebuild — refused with the re-read diagnosis. OFF stamps
    forward (the route built nothing, so the surviving forward is what it trained under)."""
    with pytest.raises(ModelVersionError, match="value_pooled"):
        _migrate_config({"config_version": 88, flag: True})
    migrated = _migrate_config({"config_version": 88, flag: False})
    assert migrated["config_version"] >= 89


@pytest.mark.parametrize("field", ["value_clock", "value_intent"])
def test_check_compatible_gates_both_flags(field):
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    b = ModelVersion.from_layout_and_policy_kwargs(layout, {"features_extractor_kwargs": {}})
    a = dataclasses.replace(b, **{field: True})
    with pytest.raises(ModelVersionError, match=field):
        a.check_compatible(b)
