"""Unit tests for gen3_bidir_threat_trunk_v1 (v36): the bidirectional in-trunk threat field.

  #1 threat_refine_outgoing  — discrete_outgoing kernel + zero-init outgoing_proj onto OPP tokens
  #2 threat_unrevealed_outgoing — expected-LATENT defender (E[mult] via SPECIES_EXP_MULT, P(KO) nulled)
  #3 threat_prob_outspeed    — uncertainty-aware P(outspeed) (÷ believed speed std)

The kernel correctness is exercised on hand-built ctxs (controllable revealed/unrevealed slots +
belief); identity-at-init / gradient-flow / OFF-byte-identical on the full Gen3FeaturesExtractor.
"""
import types

import numpy as np
import gymnasium as gym
import torch

from agents import gen3_data
from agents.model.features_extractor import (
    Gen3FeaturesExtractor, DamageOperator, TEAM_SIZE, _DMG_OUT_REFINE, _DMG_SPEED_SCALE,
)
from agents.observation.constants import POKEMON_SPREAD_OFFSET, POKEMON_FULL_DIM, ACTIVE_CONTEXT_DIM
from agents.observation.types import TypeEncoder
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

_T2I = TypeEncoder.TYPE_TO_IDX


def _op():
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    return DamageOperator(layout, prob_outspeed=False), layout


def _op_prob():
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    return DamageOperator(layout, prob_outspeed=True), layout


def _num(species_id):
    return gen3_data.species.get(species_id).num


def _ctx_out(*, our_species, our_t1, our_t2, our_moves, our_move_types, move_mask,
             opp_defenders, believed, our_alive=True, B=1):
    """ctx for discrete_outgoing. our active = slot 0; opp 6 = slots TEAM_SIZE.. .
    opp_defenders: list of 6 (species_num, t1, t2); believed: list of 6 bool (True = UNREVEALED).
    The opp ACTIVE is slot TEAM_SIZE+0 (its active flag set). Spread = IV31/EV0/neutral, full HP."""
    n = 2 * TEAM_SIZE
    species = torch.zeros(B, n, dtype=torch.long)
    t1 = torch.zeros(B, n, dtype=torch.long)
    t2 = torch.zeros(B, n, dtype=torch.long)
    species[:, 0] = our_species
    t1[:, 0] = our_t1
    t2[:, 0] = our_t2
    for i, (num, a, b) in enumerate(opp_defenders):
        species[:, TEAM_SIZE + i] = num
        t1[:, TEAM_SIZE + i] = a
        t2[:, TEAM_SIZE + i] = b
    hp = torch.zeros(B, n, POKEMON_FULL_DIM)
    hp[:, 0, 0] = 1.0 if our_alive else 0.0                 # our active HP
    hp[:, TEAM_SIZE, -1] = 1.0                               # opp active flag (slot 0)
    # revealed opp slots full HP; unrevealed slots left 0 (the kernel force-revives them)
    for i, bel in enumerate(believed):
        hp[:, TEAM_SIZE + i, 0] = 0.0 if bel else 1.0
    sp = torch.zeros(B, n, POKEMON_FULL_DIM)
    spv = sp[:, :, POKEMON_SPREAD_OFFSET:POKEMON_SPREAD_OFFSET + 18]
    spv[..., 0:6] = 1.0      # IV 31
    spv[..., 13:18] = 1.0    # neutral nature
    amid = torch.zeros(B, n, 4, dtype=torch.long)
    amty = torch.zeros(B, n, 4, dtype=torch.long)
    for k, (mid, mty) in enumerate(zip(our_moves, our_move_types)):
        amid[:, 0, k] = mid
        amty[:, 0, k] = mty
    return types.SimpleNamespace(
        batch_size=B, device=torch.device("cpu"),
        our_active_idx=torch.zeros(B, dtype=torch.long),
        opp_active_local=torch.zeros(B, dtype=torch.long),
        species_ids=species, type1_ids=t1, type2_ids=t2, ability1_ids=torch.zeros(B, n, dtype=torch.long),
        item_ids=torch.zeros(B, n, dtype=torch.long),
        hp_and_active=hp, pokemon_part=sp,
        all_move_ids=amid, all_move_type_ids=amty,
        move_mask=torch.tensor([list(move_mask)] * B, dtype=torch.float32),
        screen_feature=torch.zeros(B, 8),
        our_ctx_raw=torch.zeros(B, ACTIVE_CONTEXT_DIM), opp_ctx_raw=torch.zeros(B, ACTIVE_CONTEXT_DIM),
        weather_feature=torch.zeros(B, 7),
        opp_believed_mask=torch.tensor([[bool(x) for x in believed]] * B),
    )


def _eq():
    return gen3_data.moves.get("earthquake")   # Ground, physical, BP 100


# --------------------------------------------------------------------- #1 discrete_outgoing
def test_discrete_outgoing_shape_and_no_opp_gate():
    op, _ = _op()
    eq = _eq()
    ctx = _ctx_out(our_species=_num("tyranitar"), our_t1=_T2I["ROCK"], our_t2=_T2I["DARK"],
                   our_moves=[eq.num, 0, 0, 0], our_move_types=[_T2I["GROUND"], 0, 0, 0],
                   move_mask=[1, 0, 0, 0],
                   opp_defenders=[(_num("magneton"), _T2I["ELECTRIC"], _T2I["STEEL"])] + [(0, 0, 0)] * 5,
                   believed=[False] * 6)
    out = op.discrete_outgoing(ctx)
    assert out.shape == (1, TEAM_SIZE, _DMG_OUT_REFINE)
    # EQ (Ground) vs Magneton (Electric/Steel: Steel 2× Ground, ×2) → real physical damage on slot 0.
    assert out[0, 0, 0].item() > 0.0
    # our active fainted → whole block zero
    ctx_dead = _ctx_out(our_species=_num("tyranitar"), our_t1=_T2I["ROCK"], our_t2=_T2I["DARK"],
                        our_moves=[eq.num, 0, 0, 0], our_move_types=[_T2I["GROUND"], 0, 0, 0],
                        move_mask=[1, 0, 0, 0],
                        opp_defenders=[(_num("magneton"), _T2I["ELECTRIC"], _T2I["STEEL"])] + [(0, 0, 0)] * 5,
                        believed=[False] * 6, our_alive=False)
    assert op.discrete_outgoing(ctx_dead).abs().sum().item() == 0.0


def test_discrete_outgoing_revealed_immunity():
    """EQ (Ground) into a REVEALED Flying defender = 0 (type immunity through the real chart)."""
    op, _ = _op()
    eq = _eq()
    ctx = _ctx_out(our_species=_num("tyranitar"), our_t1=_T2I["ROCK"], our_t2=_T2I["DARK"],
                   our_moves=[eq.num, 0, 0, 0], our_move_types=[_T2I["GROUND"], 0, 0, 0],
                   move_mask=[1, 0, 0, 0],
                   opp_defenders=[(_num("skarmory"), _T2I["STEEL"], _T2I["FLYING"])] + [(0, 0, 0)] * 5,
                   believed=[False] * 6)
    out = op.discrete_outgoing(ctx)
    assert out[0, 0, 0].item() == 0.0          # Flying immune to Ground


def test_discrete_outgoing_unrevealed_zeroed_without_belief():
    """species_probs=None → UNREVEALED slots contribute nothing (the legacy revealed-gating)."""
    op, _ = _op()
    eq = _eq()
    ctx = _ctx_out(our_species=_num("tyranitar"), our_t1=_T2I["ROCK"], our_t2=_T2I["DARK"],
                   our_moves=[eq.num, 0, 0, 0], our_move_types=[_T2I["GROUND"], 0, 0, 0],
                   move_mask=[1, 0, 0, 0],
                   opp_defenders=[(_num("magneton"), _T2I["ELECTRIC"], _T2I["STEEL"])]
                                 + [(0, 0, 0)] + [(0, 0, 0)] * 4,
                   believed=[False, True, True, True, True, True])
    out = op.discrete_outgoing(ctx, species_probs=None)
    assert out[0, 0, 0].item() > 0.0           # revealed active still computes
    assert out[0, 1:, :].abs().sum().item() == 0.0   # unrevealed slots zero


def _one_hot_species(layout, slot_species, B=1):
    """species_probs [B,6,n_species] one-hot per slot on the given species nums (None = uniform-ish 0)."""
    n_sp = layout["max_species"]
    p = torch.zeros(B, TEAM_SIZE, n_sp)
    for i, num in enumerate(slot_species):
        if num is not None:
            p[:, i, num] = 1.0
    return p


def test_discrete_outgoing_expected_latent_levitate():
    """#2: an UNREVEALED slot believed to be Gengar (Levitate) reads EQ damage 0 — the expected mult folds
    Levitate's Ground immunity (SPECIES_EXP_MULT[gengar, GROUND] == 0), even though species_ids is the
    sentinel and the real chart can't see it."""
    op, layout = _op()
    eq = _eq()
    ctx = _ctx_out(our_species=_num("tyranitar"), our_t1=_T2I["ROCK"], our_t2=_T2I["DARK"],
                   our_moves=[eq.num, 0, 0, 0], our_move_types=[_T2I["GROUND"], 0, 0, 0],
                   move_mask=[1, 0, 0, 0],
                   opp_defenders=[(_num("magneton"), _T2I["ELECTRIC"], _T2I["STEEL"])] + [(0, 0, 0)] * 5,
                   believed=[False, True, True, True, True, True])
    sp = _one_hot_species(layout, [None, _num("gengar")] + [None] * 4)
    out = op.discrete_outgoing(ctx, species_probs=sp)
    assert out[0, 1, 0].item() == 0.0          # believed-Gengar slot: Levitate ⇒ EQ 0
    # control: believed Swampert (Water/Ground, neutral to Ground) → EQ lands (>0)
    sp2 = _one_hot_species(layout, [None, _num("swampert")] + [None] * 4)
    out2 = op.discrete_outgoing(ctx, species_probs=sp2)
    assert out2[0, 1, 0].item() > 0.0


def test_discrete_outgoing_pko_nulled_for_unrevealed():
    """#2 owner rule: P(KO) is NULLED for UNREVEALED defenders — the expected MAGNITUDE survives but the
    pko channel is exactly 0, even for a believed frail mon a STAB move would OHKO."""
    op, layout = _op()
    eq = _eq()
    ctx = _ctx_out(our_species=_num("tyranitar"), our_t1=_T2I["ROCK"], our_t2=_T2I["DARK"],
                   our_moves=[eq.num, 0, 0, 0], our_move_types=[_T2I["GROUND"], 0, 0, 0],
                   move_mask=[1, 0, 0, 0],
                   opp_defenders=[(_num("magneton"), _T2I["ELECTRIC"], _T2I["STEEL"])] + [(0, 0, 0)] * 5,
                   believed=[False, True, True, True, True, True])
    sp = _one_hot_species(layout, [None, _num("magneton")] + [None] * 4)   # 2× weak to Ground
    out = op.discrete_outgoing(ctx, species_probs=sp)
    assert out[0, 1, 0].item() > 0.0           # phys_high > 0 (magnitude survives)
    assert out[0, 1, 2].item() == 0.0          # phys_pko NULLED for the unrevealed slot
    assert out[0, 1, 3].item() == 0.0          # spec_pko NULLED too


# --------------------------------------------------------------------- #3 probabilistic outspeed
def test_prob_outspeed_bounds_and_differs():
    op_fixed, _ = _op()
    op_prob, _ = _op_prob()
    our = torch.tensor([200.0, 100.0, 300.0])
    opp = torch.tensor([180.0, 180.0, 180.0])
    std = torch.tensor([40.0, 40.0, 40.0])
    p_fixed = op_fixed._p_outspeed(our, opp, std)            # ignores std (prob_outspeed off)
    p_prob = op_prob._p_outspeed(our, opp, std)
    assert torch.all((p_prob >= 0.0) & (p_prob <= 1.0))
    # fixed path == legacy sigmoid over the gap / fixed scale
    assert torch.allclose(p_fixed, torch.sigmoid((our - opp) / _DMG_SPEED_SCALE))
    # uncertainty-aware path differs (wider std → softer toward 0.5 for the same gap)
    assert not torch.allclose(p_prob, p_fixed)
    # monotonic in our speed, and ~0.5 when speeds tie
    assert p_prob[2] > p_prob[0] > p_prob[1]
    tie = op_prob._p_outspeed(torch.tensor([180.0]), torch.tensor([180.0]), torch.tensor([40.0]))
    assert abs(tie.item() - 0.5) < 1e-5


# --------------------------------------------------------------------- full-extractor: safety + grad
def _make_model(**kwargs):
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    obs_space = gym.spaces.Box(0.0, 1.0, shape=(layout["total_dim"],), dtype=np.float32)
    return Gen3FeaturesExtractor(obs_space, layout=layout, mappings=mappings, **kwargs), layout


_REFINE_BASE = dict(attend_unrevealed_opponents=True, opp_belief_slots=True,
                    move_belief_mode="both", move_prior_fusion=True, damage_op=True, move_latent=True,
                    damage_refine_rounds=2)


def test_threat_identity_at_init():
    """Identity-at-init: the zero-init outgoing_proj makes the outgoing residual EXACTLY 0, so the threat-ON
    forward equals the SAME model with the outgoing branch bypassed (outgoing_proj=None → refine_cb's
    else-path). Compared on one model (not two seeded builds — the extra Linear shifts the RNG stream)."""
    threat, layout = _make_model(**_REFINE_BASE, threat_refine_outgoing=True,
                                 threat_unrevealed_outgoing=True, threat_prob_outspeed=True)
    threat.eval()
    obs = {"observation": torch.rand(4, layout["total_dim"])}
    with torch.no_grad():
        pi_on, vf_on = threat(obs)
        saved = threat.outgoing_proj
        threat.outgoing_proj = None             # bypass the outgoing branch (residual was 0 anyway)
        pi_off, vf_off = threat(obs)
        threat.outgoing_proj = saved
    assert torch.equal(pi_on, pi_off) and torch.equal(vf_on, vf_off)


def test_threat_outgoing_proj_zero_init():
    threat, _ = _make_model(**_REFINE_BASE, threat_refine_outgoing=True)
    assert threat.outgoing_proj is not None
    assert threat.outgoing_proj.weight.abs().sum().item() == 0.0
    assert threat.outgoing_proj.bias.abs().sum().item() == 0.0


def test_threat_grad_flows_to_outgoing_proj_and_species_belief():
    """The outgoing residual is differentiable: a backward through outgoing_proj(discrete_outgoing) reaches
    BOTH outgoing_proj's weight AND P(species) (so the policy/value loss sharpens the species belief — the
    whole point of the expected-latent read). Uses a hand-built ctx with a REAL move (BP>0) so out_feats is
    non-zero (a fully-random obs gives BP-0 move ids → no signal)."""
    op, layout = _op()
    proj = torch.nn.Linear(_DMG_OUT_REFINE, 128)        # stand-in for the extractor's outgoing_proj (non-zero init)
    eq = _eq()
    ctx = _ctx_out(our_species=_num("tyranitar"), our_t1=_T2I["ROCK"], our_t2=_T2I["DARK"],
                   our_moves=[eq.num, 0, 0, 0], our_move_types=[_T2I["GROUND"], 0, 0, 0],
                   move_mask=[1, 0, 0, 0],
                   opp_defenders=[(_num("magneton"), _T2I["ELECTRIC"], _T2I["STEEL"])] + [(0, 0, 0)] * 5,
                   believed=[False, True, True, True, True, True])
    sp = _one_hot_species(layout, [None, _num("magneton")] + [None] * 4).clone().requires_grad_(True)
    out_feats = op.discrete_outgoing(ctx, species_probs=sp)     # [1,6,4]
    proj(out_feats).sum().backward()
    assert proj.weight.grad is not None and proj.weight.grad.abs().sum().item() > 0.0
    assert sp.grad is not None and sp.grad.abs().sum().item() > 0.0   # gradient reaches P(species)


def test_threat_requires_refine_rounds():
    import pytest
    with pytest.raises(ValueError):
        _make_model(attend_unrevealed_opponents=True, damage_op=True, move_belief_mode="both",
                    move_prior_fusion=True, move_latent=True, damage_refine_rounds=0,
                    threat_refine_outgoing=True)
