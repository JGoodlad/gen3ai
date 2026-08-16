"""gen3_op_lean_forward_v1 (v86) — design_op_tensors step 3 + the believed-lean d3 physics.

What must hold:
  * OFF is the default and is byte-identical (the production sha probe pins it externally);
  * `drop_renders` ON: the flat block is exactly the lean width, every surviving offset is
    unchanged (renders always appended LAST), so pi/vf at init are BIT-IDENTICAL to renders-on —
    the step-3 "gone by construction" claim as an executable fact;
  * the render views are None, the consumer stashes (top-K index, pair cells, out_pko) survive,
    and the full intent stack still runs on them;
  * the boom cell's pko source (the PRE-gain stash) equals the old post-gain flat view at
    gain-init (gain starts at 1.0 on the pko channel);
  * `believed_lean` ON: requires spread_belief; a believed spread that differs from the de-timid
    fiction produces different d3 cells (and the same spread reproduces the same cells);
  * the v86 version machinery: migration defaults + both check_compatible gates.
"""
import dataclasses

import gymnasium as gym
import numpy as np
import pytest
import torch

from agents.model.damage_op import _DMG_OMX_IDX_PKO, _DMG_OMX_CELL, _DMG_OUT_N_MOVES
from agents.model.features_extractor import Gen3FeaturesExtractor, TEAM_SIZE
from agents.model.model_version import (
    MODEL_CONFIG_VERSION, ModelVersion, ModelVersionError, _migrate_config,
)
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

_ON_KWARGS = dict(
    attend_unrevealed_opponents=True, move_belief_mode="revealed", move_prior_fusion=True,
    move_latent=True, damage_op=True, damage_outgoing=True, damage_matrices_incoming=True,
    damage_matrices_outgoing=True, damage_topk_k=6, entity_topk_seats=6, opp_intent=True,
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


def test_drop_renders_shrinks_the_block_and_keeps_every_offset():
    on, layout = _build(**_ON_KWARGS, op_drop_renders=True)
    off, _ = _build(**_ON_KWARGS)
    lean = on.damage_op.out_dim
    assert lean < off.damage_op.out_dim
    assert on.damage_op.out_gain.shape == (lean,)
    # the surviving prefix of the gain init is IDENTICAL — offsets unchanged by construction
    assert torch.equal(on.damage_op.out_gain, off.damage_op.out_gain[:lean])


def test_drop_renders_is_bit_identical_at_init():
    """The step-3 claim as an executable fact: the renders had no forward consumer, so removing
    them changes NOTHING the heads see — pi/vf bitwise equal between the two modes at init
    (same seed; out_gain is built from constants, so the RNG streams align)."""
    on, layout = _build(**_ON_KWARGS, op_drop_renders=True)
    off, _ = _build(**_ON_KWARGS)
    obs = _obs(layout)
    with torch.no_grad():
        pi_on, vf_on = on(obs)
        pi_off, vf_off = off(obs)
    assert torch.equal(pi_on, pi_off)
    assert torch.equal(vf_on, vf_off)


def test_drop_renders_nulls_the_views_and_keeps_the_stashes():
    fe, layout = _build(**_ON_KWARGS, op_drop_renders=True)
    with torch.no_grad():
        fe(_obs(layout))
    t = fe.damage_op.last_tensors
    assert t.outgoing_matrix is None and t.incoming_matrix is None and t.oax_cells is None
    assert t.incoming_rows is not None and t.out_per_move is not None
    assert fe.damage_op.last_topk_idx is not None
    assert fe.damage_op.last_out_pko is not None
    assert fe.damage_op.last_out_pko.shape == (3, _DMG_OUT_N_MOVES, TEAM_SIZE)


def test_full_intent_stack_runs_lean():
    """v77+v84+v85 all consume STASHES, so the lean block must feed them unchanged."""
    fe, layout = _build(**_ON_KWARGS, op_drop_renders=True, intent_move_cell=True,
                        intent_threshold=True, intent_conditional=True,
                        opp_belief_slots=True, intent_value_reduce=True,
                        value_entity_pool=True)
    pi, vf = fe(_obs(layout))
    assert pi.shape == vf.shape


def test_boom_pko_stash_equals_the_old_flat_view_at_init():
    """The v85 boom cell moved from the post-gain flat view to the pre-gain stash; at gain-init
    (pko channel gain = 1.0) the two are equal, so the source switch is init-neutral."""
    fe, layout = _build(**_ON_KWARGS)                       # renders ON: both sources exist
    with torch.no_grad():
        fe(_obs(layout))
    op = fe.damage_op
    flat = op.last_tensors.outgoing_matrix[:, :_DMG_OUT_N_MOVES * TEAM_SIZE * _DMG_OMX_CELL]
    flat_pko = flat.reshape(-1, _DMG_OUT_N_MOVES, TEAM_SIZE, _DMG_OMX_CELL)[..., _DMG_OMX_IDX_PKO]
    assert torch.allclose(op.last_out_pko, flat_pko, atol=1e-7)


def test_believed_lean_requires_spread_belief():
    with pytest.raises(ValueError, match="spread_belief"):
        _build(**_ON_KWARGS, op_believed_lean=True)


def test_believed_lean_changes_the_lean_physics():
    """A believed spread far from the de-timid fiction must move the lean rolls; passing None
    reproduces the legacy pricing exactly."""
    from agents.model import damage_op_test as DT
    op, layout = DT._op_and_layout()
    # a REAL bulky defender (Swampert) so the rolls land under the chip cap — a base-0 dummy
    # defender saturates both arms at the 1.5 cap and hides the pricing difference.
    ctx = DT._fake_ctx(op, attacker_num=248, attacker_t1=DT._T2I["ROCK"],
                       attacker_t2=DT._T2I["DARK"],
                       defenders=[(260, DT._T2I["WATER"], DT._T2I["GROUND"])] + [(0, 0, 0)] * 5,
                       hp_probs_active=[0.0] * 16)
    logits = DT._believe_active(op, "earthquake")
    legacy = op._incoming_rolls(ctx, logits)
    legacy2 = op._incoming_rolls(ctx, logits, spread_belief=None)
    assert torch.equal(legacy[0], legacy2[0])
    sb = torch.full((1, TEAM_SIZE, 5), 120.0)              # a weak believed attacker
    believed = op._incoming_rolls(ctx, logits, spread_belief=sb)
    assert not torch.equal(legacy[0], believed[0])
    # compare on the REAL defender's row — the base-0 dummy rows saturate the 1.5 cap in
    # both arms and would mask the difference under a whole-tensor max
    assert float(believed[0][0, 0].max()) < float(legacy[0][0, 0].max())


# ------------------------------------------------------------------- version machinery


def test_migration_defaults_off():
    migrated = _migrate_config({"config_version": 85})
    assert migrated["op_drop_renders"] is False
    assert migrated["op_believed_lean"] is False
    assert migrated["config_version"] >= 86
    assert MODEL_CONFIG_VERSION >= 86


@pytest.mark.parametrize("field", ["op_drop_renders", "op_believed_lean"])
def test_check_compatible_gates_both_flags(field):
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    b = ModelVersion.from_layout_and_policy_kwargs(layout, {"features_extractor_kwargs": {}})
    a = dataclasses.replace(b, **{field: True})
    with pytest.raises(ModelVersionError, match=field):
        a.check_compatible(b)
