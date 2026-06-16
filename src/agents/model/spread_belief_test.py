"""Unit tests for gen3_unified_spread_belief_v1 — the SpreadBelief module, the op's consumption of the
believed opp stats (replacing the hand-coded constants), and the --unified-obs disable-redundant masks.
"""
import gymnasium as gym
import numpy as np
import torch

from agents.model import damage_tables as dt
from agents.model.features_extractor import (
    Gen3FeaturesExtractor, SpreadBelief, DamageOperator, TEAM_SIZE,
    _SB_ATK, _SB_DEF, _SB_SPA, _SB_SPD, _SB_SPE,
)
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.observation.types import TypeEncoder
from agents import gen3_data

_T2I = TypeEncoder.TYPE_TO_IDX
_mp = load_mappings()
_layout = Gen3ObservationEncoder(_mp).get_layout()


def _model(**kw):
    obs_space = gym.spaces.Box(0.0, 1.0, shape=(_layout["total_dim"],), dtype=np.float32)
    return Gen3FeaturesExtractor(obs_space, layout=_layout, mappings=_mp, **kw)


# ------------------------------------------------------------------- the prior buffer
def test_spread_prior_sensible_per_species():
    """The usage prior captures real spreads: a max-speed sweeper reads high speed, a special wall reads
    high SpD / ~0 Atk investment, and a multi-set mon has a wide std."""
    p = dt.build_opp_spread_prior(600)
    si = {c: i for i, c in enumerate(dt.SPREAD_STAT_COLS)}
    aero = gen3_data.species.get("aerodactyl").num
    blissey = gen3_data.species.get("blissey").num
    assert p[aero, si["spe"], 0] > 300                      # Aerodactyl invests max speed
    assert p[blissey, si["atk"], 0] < 100                   # Blissey runs no attack
    assert p[blissey, si["spd"], 0] > 250                   # ... and is a special wall
    # std is floored at 1 (never 0) for any REAL species (unpopulated num-space gaps stay all-zero).
    tar = gen3_data.species.get("tyranitar").num
    assert (p[tar, :, 1] >= 1.0).all() and (p[aero, :, 1] >= 1.0).all()
    assert p.shape == (600, dt.N_SPREAD_STATS, 2)


def test_spread_belief_cold_start_equals_prior():
    """Zero-init head ⇒ the believed stats at cold start == the usage prior mean (the clean A/B baseline)."""
    sb = SpreadBelief(_layout["max_species"])
    opp_tokens = torch.zeros(2, TEAM_SIZE, 128)
    species = torch.full((2, TEAM_SIZE), gen3_data.species.get("tyranitar").num, dtype=torch.long)
    _, believed = sb(opp_tokens, torch.ones(2, TEAM_SIZE, dtype=torch.bool), species)
    prior_mean = sb.spread_prior[species][..., 0]           # [2,6,5]
    assert torch.allclose(believed, prior_mean.clamp(min=1.0), atol=1e-4)


def test_off_builds_no_module_and_state_dict_clean():
    off, on = _model(), _model(spread_belief=True)
    assert off.spread_belief is None and on.spread_belief is not None
    assert not any("spread_belief" in k for k in off.state_dict())
    assert any("spread_belief" in k for k in on.state_dict())
    assert on.projection_input_dim == off.projection_input_dim    # enriches the token, not the projection


# ------------------------------------------------------------------- op consumption
from agents.model import damage_op_test as _DT  # reuse its proven _fake_ctx / _make_layout


def _op_and_ctx():
    op = DamageOperator(_DT._make_layout())
    ctx = _DT._fake_ctx(op, attacker_num=248, attacker_t1=_T2I["ROCK"], attacker_t2=_T2I["DARK"],
                        defenders=[(260, _T2I["WATER"], _T2I["GROUND"])] + [(0, 0, 0)] * 5,
                        hp_probs_active=[0.0] * 16)
    lg = torch.full((1, TEAM_SIZE, op.MOVE_BP.shape[0]), -10.0)
    lg[:, :, gen3_data.moves.get("earthquake").num] = 10.0   # believe Earthquake
    return op, ctx, lg


def test_op_consumes_believed_attack_raises_incoming_damage():
    """A higher believed opp ATTACK → higher believed incoming physical damage to our mons. Proves the op
    reads the spread belief in place of its fixed offense constant."""
    op, ctx, lg = _op_and_ctx()
    low = torch.full((1, TEAM_SIZE, 5), 150.0)               # low believed stats
    high = low.clone(); high[:, 0, _SB_ATK] = 500.0          # high believed atk for the opp active (slot 0)
    out_low = op(ctx, lg, low)[:, :TEAM_SIZE * op.per_mon].reshape(1, TEAM_SIZE, op.per_mon)
    out_high = op(ctx, lg, high)[:, :TEAM_SIZE * op.per_mon].reshape(1, TEAM_SIZE, op.per_mon)
    # phys high-roll (idx 1) on our active (slot 0) rises with the believed attack.
    assert out_high[0, 0, 1] > out_low[0, 0, 1] + 1e-4


def test_op_signature_back_compat_none():
    """spread_belief defaults to None → the op uses its legacy constants (byte-identical to pre-v25)."""
    op, ctx, lg = _op_and_ctx()
    a = op(ctx, lg)                       # no spread_belief arg
    b = op(ctx, lg, None)                 # explicit None
    assert torch.equal(a, b)


# ------------------------------------------------------------------- the --unified-obs masks
def test_masks_zero_only_their_region():
    """Each obs-ablation mask zeros its region of the model's view and nothing else; OFF leaves the obs
    untouched. We compare the unpacked non_matchup_rest with/without each mask."""
    torch.manual_seed(0)
    x = {"observation": torch.rand(2, _layout["total_dim"])}
    base = _model().unpack(x).non_matchup_rest
    # incoming-damage mask
    inc = _model(mask_incoming_damage_obs=True).unpack(x).non_matchup_rest
    # move-effects mask
    eff = _model(mask_move_effects_obs=True).unpack(x).non_matchup_rest
    assert not torch.equal(base, inc) and not torch.equal(base, eff)
    # the two masks zero DISJOINT regions (their difference-from-base supports don't overlap)
    inc_changed = (base != inc).any(0)
    eff_changed = (base != eff).any(0)
    assert not (inc_changed & eff_changed).any()
    # a masked region is all-zero in the masked view
    assert (eff[:, eff_changed] == 0).all()
