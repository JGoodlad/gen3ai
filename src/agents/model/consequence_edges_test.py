"""Unit tests for the C1 CONSEQUENCE edges (gen3_edge_bias_trunk_v1 — the first hypothetical-world
damage family).

The claim: `DamageOperator.pairwise_boost` re-runs the validated `_outgoing_matrix` kernel under
the post-setup-move stat stages (`MOVE_SELF_BOOSTS`) and emits per-(E3 setup seat, opp mon) DELTA
cells — Swords Dance raises the best physical line, Agility raises P(outspeed) and NOTHING else,
non-setup slots are exactly zero, and `boost_delta=None` leaves every existing caller byte-identical.
"""
import inspect

import gymnasium as gym
import numpy as np
import pytest
import torch

from agents import gen3_data
from agents.model.damage_tables import build_self_boost_tables
from agents.model.features_extractor import (
    EdgeBias, Gen3FeaturesExtractor, TEAM_SIZE, _EDGE_C1_CELL,
)
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

_mappings = load_mappings()
_layout = Gen3ObservationEncoder(_mappings).get_layout()
_SIG = set(inspect.signature(Gen3FeaturesExtractor.__init__).parameters)

_C1_TOGGLES = dict(attend_unrevealed_opponents=True, move_belief_mode="both",
                   move_belief_prefuse=True, move_belief_single_compute=True,
                   damage_op=True, damage_outgoing=True, move_latent=True,
                   damage_op_prefuse=True, move_prior_fusion=True,
                   entity_topk_seats=5, edge_bias_families="c1")

_SD = gen3_data.moves.get("swordsdance").num
_AGILITY = gen3_data.moves.get("agility").num
_BODYSLAM = gen3_data.moves.get("bodyslam").num


def _make(**kw):
    space = gym.spaces.Box(0.0, 1.0, shape=(_layout["total_dim"],), dtype=np.float32)
    return Gen3FeaturesExtractor(space, layout=_layout, mappings=_mappings,
                                 **{k: v for k, v in kw.items() if k in _SIG})


def _obs(batch=2, seed=71):
    return {"observation": torch.rand(batch, _layout["total_dim"],
                                      generator=torch.Generator().manual_seed(seed))}


def _boost_ctx(fe, seed=71):
    """A real ctx forced into the canonical C1 scenario: slot 0 = Body Slam (physical, legal),
    slot 1 = the setup move under test, every opp slot revealed."""
    obs = _obs(seed=seed)
    with torch.no_grad():
        fe(obs)
    ctx = fe.unpack(obs)
    ctx.our_active_req_move_ids[:, :] = 0
    ctx.our_active_req_move_ids[:, 0] = _BODYSLAM
    ctx.our_active_req_move_type_ids[:, 0] = fe.damage_op.MOVE_TYPE_IDX[_BODYSLAM]
    ctx.our_active_req_move_legal[:, :] = 0.0
    ctx.our_active_req_move_legal[:, 0] = 1.0
    ctx.opp_believed_mask[:, :] = False
    return ctx


def test_self_boost_table_rows():
    t = build_self_boost_tables(_layout["max_moves"])["MOVE_SELF_BOOSTS"]
    assert t[_SD].tolist() == [2.0, 0.0, 0.0, 0.0, 0.0]
    assert t[_AGILITY].tolist() == [0.0, 0.0, 0.0, 0.0, 2.0]
    dd = gen3_data.moves.get("dragondance").num
    assert t[dd].tolist() == [1.0, 0.0, 0.0, 0.0, 1.0]
    assert t[_BODYSLAM].abs().sum() == 0.0, "a non-setup move must be an all-zero row"
    # The pure-setup gates: Belly Drum / Curse are deliberately NOT rows (unpriced, never wrong).
    assert t[gen3_data.moves.get("bellydrum").num].abs().sum() == 0.0
    assert t[gen3_data.moves.get("curse").num].abs().sum() == 0.0


def test_boost_delta_none_is_byte_identical():
    fe = _make(**_C1_TOGGLES).eval()
    obs = _obs()
    with torch.no_grad():
        fe(obs)
    ctx = fe.unpack(obs)
    with torch.no_grad():
        a = fe.damage_op.pairwise_outgoing(ctx)
        b = fe.damage_op.pairwise_outgoing(ctx, boost_delta=torch.zeros(2, 5))
        c = fe.damage_op.pairwise_outgoing(ctx)
    assert torch.equal(a, c), "the default path must be deterministic"
    assert torch.equal(a, b), "a zero delta must reproduce the current world exactly"


def test_swords_dance_raises_the_physical_line():
    fe = _make(**_C1_TOGGLES).eval()
    ctx = _boost_ctx(fe)
    ctx.our_active_req_move_ids[:, 1] = _SD
    with torch.no_grad():
        cells = fe.damage_op.pairwise_boost(ctx)
    assert cells.shape == (2, 4, TEAM_SIZE, _EDGE_C1_CELL)
    # Non-setup slots (0, 2, 3) are exactly zero everywhere.
    assert float(cells[:, [0, 2, 3]].abs().sum()) == 0.0
    # The SD seat: is_boost == the shared gate, and the best-physical delta is positive wherever
    # the base kernel priced a live target (Body Slam is physical, +2 atk ⇒ strictly more damage).
    with torch.no_grad():
        base = fe.damage_op.pairwise_outgoing(ctx)
    live = base[..., 1].amax(dim=1) > 1e-6                       # [B,6] priced targets
    assert bool(live.any()), "fixture must price at least one live target"
    assert bool((cells[:, 1, :, 1][live] > 0).all()), "SD must raise the best physical line"
    assert float(cells[:, 1, :, 3].abs().sum()) == 0.0, "SD gives no outspeed delta"


def test_agility_moves_only_the_speed_channel():
    fe = _make(**_C1_TOGGLES).eval()
    ctx = _boost_ctx(fe, seed=73)
    ctx.our_active_req_move_ids[:, 1] = _AGILITY
    with torch.no_grad():
        cells = fe.damage_op.pairwise_boost(ctx)
    assert float(cells[:, 1, :, 1].abs().sum()) == 0.0, "spe does not enter the damage formula"
    assert float(cells[:, 1, :, 2].abs().sum()) == 0.0
    d_spd = cells[:, 1, :, 3]
    assert float(d_spd.min()) >= 0.0, "+2 spe can never lower P(outspeed)"
    assert float(d_spd.max()) > 0.0, "some gated matchup must gain outspeed probability"


def test_c1_gate_and_integration():
    with pytest.raises(ValueError, match="edge_bias_families d1/s1/c1"):
        _make(**dict(_C1_TOGGLES, damage_outgoing=False))
    fe = _make(**dict(_C1_TOGGLES,
                      edge_bias_families="d1,d2,d3,d4,s1,s3,v,t,x,g,c4,c1")).eval()
    assert fe.edge_bias.c1_map is not None
    assert fe.edge_bias.c1_map.weight.abs().sum() == 0.0, "zero-init map (identity at init)"
    with torch.no_grad():
        pi, vf = fe(_obs(seed=75))
    assert torch.isfinite(pi).all() and torch.isfinite(vf).all()


def test_c1_on_is_bitwise_identical_at_init():
    obs = _obs(seed=77)
    torch.manual_seed(0)
    off = _make(**dict(_C1_TOGGLES, edge_bias_families="off")).eval()
    torch.manual_seed(0)
    on = _make(**_C1_TOGGLES).eval()
    on.load_state_dict(off.state_dict(), strict=False)   # align shared weights; c1_map is zero-init
    with torch.no_grad():
        pi_off, vf_off = off(obs)
        pi_on, vf_on = on(obs)
    assert torch.equal(pi_off, pi_on) and torch.equal(vf_off, vf_on)
