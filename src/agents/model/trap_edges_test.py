"""Unit tests for the T-family TRAPPING edges (gen3_edge_bias_trunk_v1 — Stage 2).

The claim: `DamageOperator.pairwise_trap` prices P(cannot switch) both directions from the three
gen3 trap abilities (Shadow Tag / Arena Trap / Magnet Pull), our side exact, the opp side
revealed-exact else the Smogon species prior, with the Levitate fold in the grounded check.
"""
import inspect

import gymnasium as gym
import numpy as np
import pytest
import torch

from agents import gen3_data
from agents.model.damage_tables import build_trap_tables
from agents.model.features_extractor import EdgeBias, Gen3FeaturesExtractor, TEAM_SIZE, _EDGE_T_CELL
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

_mappings = load_mappings()
_layout = Gen3ObservationEncoder(_mappings).get_layout()
_SIG = set(inspect.signature(Gen3FeaturesExtractor.__init__).parameters)

_T_TOGGLES = dict(attend_unrevealed_opponents=True, move_belief_mode="both",
                  move_belief_prefuse=True, move_belief_single_compute=True,
                  damage_op=True, damage_outgoing=True, move_latent=True,
                  damage_op_prefuse=True, move_prior_fusion=True,
                  entity_topk_seats=5, edge_bias_families="t")


def _make(**kw):
    space = gym.spaces.Box(0.0, 1.0, shape=(_layout["total_dim"],), dtype=np.float32)
    return Gen3FeaturesExtractor(space, layout=_layout, mappings=_mappings,
                                 **{k: v for k, v in kw.items() if k in _SIG})


def _obs(batch=2, seed=31):
    return {"observation": torch.rand(batch, _layout["total_dim"],
                                      generator=torch.Generator().manual_seed(seed))}


def test_trap_tables_resolve_the_three_abilities():
    t = build_trap_tables(_layout["max_species"], _layout["max_abilities"])
    assert int((t["ABILITY_TRAP"].sum(0) > 0).sum()) == 3, "each trap column must have its ability"
    assert float(t["ABILITY_IS_LEVITATE"].sum()) == 1.0
    # A canonical carrier: Dugtrio's prior should put real mass on arenatrap (column 1).
    dug = gen3_data.species.get("dugtrio")
    assert float(t["SPECIES_TRAP_PRIOR"][dug.num, 1]) > 0.5


def test_shadow_tag_traps_every_revealed_alive_victim():
    """Force OUR mon 0's ability to Shadow Tag on a REAL ctx: its trap prob vs opp j must equal
    exactly the gate (revealed_j · both_alive) — Shadow Tag has no victim condition."""
    fe = _make(**_T_TOGGLES).eval()
    obs = _obs()
    ctx = fe.unpack(obs)
    st_num = gen3_data.abilities.get("shadowtag").num
    ctx.ability1_ids[:, 0] = st_num
    with torch.no_grad():
        cells = fe.damage_op.pairwise_trap(ctx)
    assert cells.shape == (2, TEAM_SIZE, TEAM_SIZE, _EDGE_T_CELL)
    revealed = (1.0 - ctx.opp_believed_mask.float())
    alive_i = (ctx.hp_and_active[:, :TEAM_SIZE, 0] > 0).float()
    alive_j = (ctx.hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, 0] > 0).float()
    expected = revealed * alive_j * alive_i[:, 0:1]
    assert torch.allclose(cells[:, 0, :, 0], expected, atol=1e-6)
    assert float(cells.min()) >= 0.0 and float(cells.max()) <= 1.0


def test_magnet_pull_only_traps_steel():
    fe = _make(**_T_TOGGLES).eval()
    ctx = fe.unpack(_obs(seed=33))
    mp_num = gen3_data.abilities.get("magnetpull").num
    ctx.ability1_ids[:, 1] = mp_num
    steel_idx = None
    from agents.model.damage_tables import _T2I
    steel_idx = _T2I["STEEL"]
    with torch.no_grad():
        cells = fe.damage_op.pairwise_trap(ctx)
    is_steel_j = ((ctx.type1_ids[:, TEAM_SIZE:2 * TEAM_SIZE] == steel_idx)
                  | (ctx.type2_ids[:, TEAM_SIZE:2 * TEAM_SIZE] == steel_idx))
    # Any non-steel victim must read exactly 0 from our magnet-pull mon (row 1, channel 0).
    non_steel = ~is_steel_j
    assert float(cells[:, 1, :, 0][non_steel].abs().sum()) == 0.0


def test_entry_edge_kernel_invariants():
    """X family: a Flying victim takes NO spikes chip; forcing OUR mon to carry Pursuit drives the
    opp-side exposure prob to 1; cells live in [0,1] (eff normalized)."""
    fe = _make(**_T_TOGGLES).eval()
    obs = _obs(seed=41)
    with torch.no_grad():
        fe(obs)
    ctx = fe.unpack(obs)
    from agents.model.damage_tables import _pursuit_num, _T2I
    ctx.all_move_ids[:, 0, 0] = _pursuit_num()              # our mon 0 carries Pursuit
    ctx.type1_ids[:, 1] = _T2I["FLYING"]                    # our mon 1 is Flying
    ctx.type2_ids[:, 1] = _T2I["FLYING"]
    with torch.no_grad():
        our_c, opp_c = fe.damage_op.pairwise_entry(ctx, fe.last_move_belief_logits)
    assert our_c.shape == (2, TEAM_SIZE, 4) and opp_c.shape == (2, TEAM_SIZE, 4)
    assert float(our_c[:, 1, 0].abs().sum()) == 0.0, "Flying mon takes no spikes chip"
    alive0 = (ctx.hp_and_active[:, 0, 0] > 0).float()
    # Wherever our Pursuit carrier is alive, every revealed+alive opp cell reads exposure 1.0.
    for b in range(2):
        if alive0[b] > 0:
            live = (opp_c[b, :, 1] > 0)
            assert bool((opp_c[b, live, 1] == 1.0).all())
    assert float(our_c.min()) >= 0.0


def test_family_integration_and_gate():
    with pytest.raises(ValueError, match="edge_bias_families t"):
        _make(edge_bias_families="t")                       # no damage_op
    fe = _make(**dict(_T_TOGGLES, edge_bias_families="d1,d2,d3,d4,s1,s3,v,t,x")).eval()
    with torch.no_grad():
        pi, vf = fe(_obs(seed=35))
    assert torch.isfinite(pi).all() and torch.isfinite(vf).all()
    assert fe.edge_bias.t_map is not None
