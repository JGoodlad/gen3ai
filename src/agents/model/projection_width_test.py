"""gen3_static_widths_v1 — the DISCOVERY FORWARD, preserved as a test.

`Gen3FeaturesExtractor.__init__` used to MEASURE its projection-input widths by running a
dummy `forward_internal` with `_intent_reduce_discovering` zero-fill branches threaded
through the runtime forward. That mechanism was the parent of a shipped bug class
(ede5a88: an early `return` in a discovery branch hid every width appended below it and
built the critic 128 dims short). Since v89 (`gen3_value_pooled_routes_v1`) every value
route injects additively into `value_pooled`, so no width is emergent — the widths are
now STATIC ARITHMETIC in `compute_projection_widths`, and THIS FILE is the old mechanism
inverted into the new mechanism's verifier: for a sweep of flag configurations it builds
the real extractor, runs a REAL forward, and asserts the measured concat widths equal the
arithmetic. A wrong width for any combo fails here, in the suite, instead of at a
production launch.

Sweep coverage: production (designs/production_config.json), all-routes-on (the
`value_route_gradient_test._ALL_ROUTES_ON` shape), minimal (bare flags), and targeted
combos toggling each width-relevant flag independently — the hidden-opp belief pool
(`opp_belief_cls_k`, at two different k), the seed window (`damage_op`), and the
width-NEUTRAL families (opp_belief_slots, value_entity_pool[_full], the intent_* cells,
value_threat_inject, history_events) that must move nothing.
"""

import json

import gymnasium as gym
import numpy as np
import pytest
import torch

from agents.model.arch_constants import D_MODEL
from agents.model.features_extractor import Gen3FeaturesExtractor, compute_projection_widths
from agents.model.value_route_gradient_test import _ALL_ROUTES_ON
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from utils.git import get_repo_root

_MAPPINGS = load_mappings()
_LAYOUT = Gen3ObservationEncoder(_MAPPINGS).get_layout()
_SPACE = gym.spaces.Box(0.0, 1.0, shape=(_LAYOUT["total_dim"],), dtype=np.float32)


def _production_kwargs():
    """The live production shape, mapped config->kwargs by the same recipe the trainer uses
    (`extractor_arch.ARCH_ARG_KEYS` + the frozen tier + the two slot/intent toggles)."""
    import agents.model.extractor_arch as EA
    cfg = json.load(open(get_repo_root() + "/designs/production_config.json"))
    kwargs = {k: cfg[k] for k in EA.ARCH_ARG_KEYS if k in cfg}
    kwargs.update({k: cfg.get(k, v) for k, v in EA.FROZEN_ARCH_KWARGS.items()})
    kwargs.update({k: cfg[k] for k in ("opp_belief_slots", "opp_intent") if k in cfg})
    return kwargs


_DAMAGE_OP_MIN = dict(attend_unrevealed_opponents=True, move_belief_mode="revealed",
                      damage_op=True)

# name -> extractor kwargs. Every width-relevant flag is toggled independently at least
# once, and each width-NEUTRAL family appears at least once (its case asserts the widths
# of its base config, so a family that silently grew a concat fails its own case).
_COMBOS = {
    "minimal": {},
    "all_routes_on": dict(_ALL_ROUTES_ON),
    # -- the hidden-opp belief pool: the one remaining width-moving flag (k * D_MODEL on
    #    the POLICY head only, since the wave deleted its vf half); two different k so a
    #    hardcoded k=6 cannot pass.
    "belief_pool_k6": dict(attend_unrevealed_opponents=True, opp_belief_cls_k=6),
    "belief_pool_k3": dict(attend_unrevealed_opponents=True, opp_belief_cls_k=3),
    # -- the op: width-neutral on BOTH heads since the seed window's deletion.
    "damage_op_min": dict(_DAMAGE_OP_MIN),
    "damage_op_plus_pool": dict(_DAMAGE_OP_MIN, opp_belief_cls_k=6),
    # -- width-neutral families (each must move NEITHER width off its base).
    "belief_slots_only": dict(attend_unrevealed_opponents=True, opp_belief_slots=True),
    "value_entity_pool_no_op": dict(value_entity_pool=True, value_entity_pool_full=True),
    "value_threat_inject": dict(_DAMAGE_OP_MIN, value_threat_inject=True),
    "history_events": dict(history_events=True),
    # -- the full intent-cell stack (move cell + threshold + conditional widen the POINTER
    #    stash, never pi/vf) on top of the all-routes base.
    "intent_cells_full": dict(_ALL_ROUTES_ON, intent_move_cell=True, intent_conditional=True,
                              damage_outgoing=True, damage_matrices_outgoing=True),
}


def _build(kwargs):
    torch.manual_seed(3)
    fe = Gen3FeaturesExtractor(_SPACE, layout=_LAYOUT, mappings=_MAPPINGS, **kwargs)
    fe.eval()
    return fe


def _assert_widths(fe, kwargs):
    """The verifier: measured REAL-forward widths == the static arithmetic == the built modules."""
    exp_pi, exp_vf = compute_projection_widths(
        _LAYOUT, opp_belief_cls_k=kwargs.get("opp_belief_cls_k", 0))
    g = torch.Generator().manual_seed(11)
    obs = {"observation": torch.rand(3, _LAYOUT["total_dim"], generator=g)}
    with torch.no_grad():
        pi, vf = fe.forward_internal(obs)
    assert pi.shape[1] == exp_pi, (
        f"pi concat width {pi.shape[1]} != computed {exp_pi} (delta {pi.shape[1] - exp_pi}) — "
        f"compute_projection_widths has drifted from ProjectionAssembler.forward")
    assert vf.shape[1] == exp_vf, (
        f"vf concat width {vf.shape[1]} != computed {exp_vf} (delta {vf.shape[1] - exp_vf}) — "
        f"compute_projection_widths has drifted from the vf concat (a value part appended "
        f"outside the additive value_pooled routes?)")
    # The built projections must be sized from the same arithmetic.
    assert fe.projection_input_dim == exp_pi
    assert fe.value_projection_input_dim == exp_vf
    assert fe.pre_proj_norm.normalized_shape[0] == exp_pi
    assert fe.projection.in_features == exp_pi
    assert fe.value_pre_norm.normalized_shape[0] == exp_vf
    assert fe.value_projection.in_features == exp_vf
    # And the full forward (concat -> norm -> projection) must actually run.
    with torch.no_grad():
        pi_f, vf_f = fe(obs)
    assert pi_f.shape[1] == fe.projection_dim and vf_f.shape[1] == fe.projection_dim


@pytest.mark.parametrize("name", sorted(_COMBOS))
def test_measured_widths_equal_static_arithmetic(name):
    kwargs = _COMBOS[name]
    fe = _build(kwargs)
    _assert_widths(fe, kwargs)


def test_production_config_widths():
    """The combo that matters most: the literal production flag set."""
    kwargs = _production_kwargs()
    fe = _build(kwargs)
    _assert_widths(fe, kwargs)


def test_arithmetic_deltas_without_building():
    """The pure-helper contract, no model build: k moves the POLICY head by k*D_MODEL and
    leaves the critic alone.

    `vf` is a CONSTANT `D_MODEL` since the critic-route deletion wave — no flag moves it, because
    `vf_combined IS value_pooled` and every surviving critic route injects additively. That is a
    stronger statement than "the arithmetic is right": there is no longer a vf concat for a route
    to be appended to outside the seam, so the ede5a88 mis-sizing class has nothing to act on.
    """
    base_pi, base_vf = compute_projection_widths(_LAYOUT)
    assert base_vf == D_MODEL
    for k in (1, 3, 6):
        pi, vf = compute_projection_widths(_LAYOUT, opp_belief_cls_k=k)
        assert pi == base_pi + k * D_MODEL
        assert vf == D_MODEL, "the hidden-opp belief's vf half was deleted — vf must not move"
