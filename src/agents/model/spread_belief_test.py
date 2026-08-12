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
    _, believed, _, _ = sb(opp_tokens, torch.ones(2, TEAM_SIZE, dtype=torch.bool), species)
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

# ============================ gen3_nature_ev_belief_v1 (v40): nature/EV generative head ============================
from agents.model.features_extractor import _EV_DELTA_SCALE, _DMG_IDX_PHYS_PKO
from agents.training.instrumented_ppo import InstrumentedMaskablePPO as _PPO

_COLS = dt.SPREAD_STAT_COLS


def _derive(sid, nature_name, evs5):
    sd = gen3_data.species.get(sid); v = gen3_data.natures.raw()[nature_name]
    return [gen3_data.priors.gen3_stat(int(sd.base_stats[c]), evs5[j], float(v[c])) for j, c in enumerate(_COLS)]


def _base5(sid):
    sd = gen3_data.species.get(sid); return [int(sd.base_stats[c]) for c in _COLS]


# ---- Step 1: data buffers + inversion ----
def test_nature_buffers_sensible():
    nm = dt.build_nature_mult()
    assert nm.shape == (dt.N_NATURES, dt.N_SPREAD_STATS)
    assert torch.allclose(nm[0], torch.tensor([1.1, 1.0, 0.9, 1.0, 1.0]))    # adamant: atk+ spa-
    assert torch.allclose(nm[24], torch.tensor([0.9, 1.0, 1.0, 1.0, 1.1]))   # timid: spe+ atk-
    npri = dt.build_species_nature_prior(600)
    assert torch.allclose(npri[1].exp().sum(), torch.tensor(1.0), atol=1e-4)  # rows are a distribution
    assert torch.isfinite(npri).all()                                        # the uniform floor → no log(0)
    ev = dt.build_species_ev_prior(600)
    assert ev.shape == (600, 5) and float(ev.min()) >= 0.0 and float(ev.max()) <= 252.5  # weighted mean ≤252 (+fp)


def test_invert_nature_evs_round_trips_real_spreads():
    """The inverter recovers a (nature, EVs) decomposition that EXACTLY reproduces the derived stats."""
    cases = [("tyranitar", "adamant", [252, 0, 0, 0, 252]), ("salamence", "naive", [0, 0, 252, 0, 252]),
             ("blissey", "calm", [0, 252, 0, 252, 0]), ("skarmory", "impish", [0, 252, 0, 0, 4]),
             ("jolteon", "timid", [0, 0, 252, 4, 252])]
    for sid, nat, evs in cases:
        d = _derive(sid, nat, evs)
        res = dt.invert_nature_evs(d, _base5(sid), species_id=sid)
        assert res is not None
        num, iev = res
        inv_name = next(k for k, v in gen3_data.natures.raw().items() if int(v["num"]) == num)
        assert _derive(sid, inv_name, iev) == d                              # reproduces the derived stats


# ---- Step 2: the generative head ----
def test_nature_head_off_byte_identical_state_dict():
    """--spread-belief (additive) keeps the OLD SpreadBelief params; --spread-belief-nature swaps them."""
    add = SpreadBelief(_layout["max_species"], nature=False)
    nat = SpreadBelief(_layout["max_species"], nature=True)
    assert "stat_head.weight" in dict(add.named_parameters())               # additive head (baseline)
    assert "stat_head.weight" not in dict(nat.named_parameters())
    assert {"nature_head.weight", "ev_head.weight"} <= set(dict(nat.named_parameters()))


def test_nature_head_cold_start_equals_generative_prior():
    """Zero-init heads ⇒ believed == the prior-derived stat ((2·base+31+E[ev]/4+5)·E[mult]) at cold start."""
    sb = SpreadBelief(_layout["max_species"], nature=True)
    sid = gen3_data.species.get("tyranitar").num
    species = torch.full((2, TEAM_SIZE), sid, dtype=torch.long)
    _, believed, nat_logits, ev = sb(torch.zeros(2, TEAM_SIZE, 128),
                                     torch.ones(2, TEAM_SIZE, dtype=torch.bool), species)
    e_mult = torch.softmax(sb.nature_logprior[species], -1) @ sb.nature_mult
    expect = ((2.0 * sb.base_nonhp[species] + 31.0 + sb.ev_prior[species] / 4.0 + 5.0) * e_mult).clamp(min=1.0)
    assert torch.allclose(believed, expect, atol=1e-3)
    assert nat_logits.shape == (2, TEAM_SIZE, dt.N_NATURES) and ev.shape == (2, TEAM_SIZE, 5)

