"""Integrated tests for the UNIFIED opponent-move system (Stage B): the move-belief prior fusion +
the enriched DamageOperator (parity features + believed-status threat) + the obs-ablation toggle.

The headline requirement (the user's): the system must produce finite, sane output for EVERY
combination of how-much-is-known — a hidden mon with zero moves, a revealed mon with zero revealed
attacks (→ the species move prior carries it), partial reveals, full reveals, and no opponent active —
and all combinations in between. These tests pin that, plus the enriched features (crit-delta,
P(outspeed), provenance, the noisy-OR effect/status threat) and the ablation toggle.
"""
import types

import numpy as np
import gymnasium as gym
import torch
import pytest

from agents.model.features_extractor import (
    Gen3FeaturesExtractor, DamageOperator, MoveBelief,
    _DMG_PER_MON, _DMG_EFFECT, _DMG_INCOMING_SEC, _DMG_CB, TEAM_SIZE,
)
from agents.model import damage_tables as dt
from agents.observation.constants import (
    POKEMON_SPREAD_OFFSET, POKEMON_FULL_DIM, OFFSET_REACTIVE,
)
from agents.observation.types import TypeEncoder
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents import gen3_data

_T2I = TypeEncoder.TYPE_TO_IDX
_mappings = load_mappings()
_layout = Gen3ObservationEncoder(_mappings).get_layout()
_M = _layout["max_moves"]
_S = _layout["max_species"]
_EMB = _layout["move_embedding_dim"]
_OUT = TEAM_SIZE * _DMG_PER_MON + _DMG_EFFECT + _DMG_INCOMING_SEC + _DMG_CB
# Effect-scalar column order (== damage_tables.MOVE_EFFECT_COLS): recovery, status, phaze, boost, hazard, protect
_EFF = {name: i for i, name in enumerate(dt.MOVE_EFFECT_COLS)}


def _ctx(*, opp_species=0, opp_t1=0, opp_t2=0, defenders=None, opp_active=True,
         def_abilities=None, our_reflect=False, our_light_screen=False, B=1):
    """Hand-built ctx exercising the op's reads. `defenders` = list of (species_num, t1, t2); default
    one full-HP Swampert + pads. Our mons get IV31/EV0/neutral spread + full HP. `def_abilities` = list
    of defender ability nums (0 = none); screens are our-side bits."""
    if defenders is None:
        defenders = [(260, _T2I["WATER"], _T2I["GROUND"])] + [(0, 0, 0)] * 5
    n = 2 * TEAM_SIZE
    species = torch.zeros(B, n, dtype=torch.long)
    t1 = torch.zeros(B, n, dtype=torch.long); t2 = torch.zeros(B, n, dtype=torch.long)
    for i, (num, a, b) in enumerate(defenders):
        species[:, i] = num; t1[:, i] = a; t2[:, i] = b
    species[:, TEAM_SIZE] = opp_species; t1[:, TEAM_SIZE] = opp_t1; t2[:, TEAM_SIZE] = opp_t2
    ability1 = torch.zeros(B, n, dtype=torch.long)
    for i, a in enumerate(def_abilities or []):
        ability1[:, i] = a
    hp_and_active = torch.zeros(B, n, POKEMON_FULL_DIM)
    hp_and_active[:, :TEAM_SIZE, 0] = 1.0
    if opp_active:
        hp_and_active[:, TEAM_SIZE, -1] = 1.0
    pokemon_part = torch.zeros(B, n, POKEMON_FULL_DIM)
    sp = pokemon_part[:, :, POKEMON_SPREAD_OFFSET:POKEMON_SPREAD_OFFSET + 18]
    sp[..., 0:6] = 1.0; sp[..., 13:18] = 1.0
    screen = torch.zeros(B, 8)
    if our_reflect:
        screen[:, 0] = 1.0
    if our_light_screen:
        screen[:, 2] = 1.0
    return types.SimpleNamespace(
        batch_size=B, device=torch.device("cpu"),
        opp_active_local=torch.zeros(B, dtype=torch.long),
        our_active_idx=torch.zeros(B, dtype=torch.long),
        species_ids=species, type1_ids=t1, type2_ids=t2, ability1_ids=ability1,
        item_ids=torch.zeros(B, 2 * TEAM_SIZE, dtype=torch.long),   # no Choice Band by default (gen3_unified_choice_band_v1)
        all_move_ids=torch.zeros(B, n, 4, dtype=torch.long), screen_feature=screen,
        our_ctx_raw=torch.zeros(B, 14), opp_ctx_raw=torch.zeros(B, 14),  # no boosts
        weather_feature=torch.zeros(B, 7),                              # no weather
        hp_and_active=hp_and_active, pokemon_part=pokemon_part, hp_probs=torch.zeros(B, n, 16),
    )


def _op():
    return DamageOperator(_layout)


def _belief_logits(present_moves=(), B=1, base=-10.0, hi=10.0):
    """Move-belief logits with `present_moves` (ids) set high, rest low — emulates a sharp belief."""
    lg = torch.full((B, TEAM_SIZE, _M), base)
    for mid in present_moves:
        lg[:, :, gen3_data.moves.get(mid).num] = hi
    return lg


# --------------------------------------------------------------------------- enriched op shape/bounds
def test_enriched_output_shape_and_bounds():
    op = _op()
    out = op(_ctx(opp_species=248, opp_t1=_T2I["ROCK"], opp_t2=_T2I["DARK"], B=2),
             _belief_logits(("earthquake", "rockslide"), B=2))
    assert out.shape == (2, _OUT) and op.out_dim == _OUT
    permon = out[:, :TEAM_SIZE * _DMG_PER_MON].reshape(2, TEAM_SIZE, _DMG_PER_MON)
    eff = out[:, TEAM_SIZE * _DMG_PER_MON:]
    # [phys_low,high,crit,pko,acc, spec_low,high,crit,pko,acc, outspeed, prov]
    rolls = permon[..., [0, 1, 5, 6]]      # low/high rolls — fraction of max HP, clamped to 1.5
    crit = permon[..., [2, 7]]             # crit roll — clamped to 3.0
    pko = permon[..., [3, 8]]
    acc = permon[..., [4, 9]]              # per-channel dominant-threat accuracy
    outspeed = permon[..., 10]; prov = permon[..., 11]
    assert (rolls >= 0).all() and (rolls <= 1.5001).all()
    assert (crit >= 0).all() and (crit <= 3.0001).all()
    assert (0 <= pko).all() and (pko <= 1.0001).all()
    assert (0 <= acc).all() and (acc <= 1.0001).all()
    assert (0 <= outspeed).all() and (outspeed <= 1).all()
    assert (0 <= prov).all() and (prov <= 1.0001).all()
    assert (0 <= eff).all() and (eff <= 1.0001).all()
    assert torch.isfinite(out).all()
    # rolls are ordered within a channel: low <= high (low = 0.85·high, both clamped).
    assert (permon[..., 0] <= permon[..., 1] + 1e-6).all() and (permon[..., 5] <= permon[..., 6] + 1e-6).all()


# --------------------------------------------------------------------------- believed-status threat
def test_effect_max_surfaces_believed_status():
    """Believing the opp active has Toxic raises the STATUS effect scalar; Recover raises RECOVERY;
    a pure-attacker belief leaves both near the floor — the status axis the damage-only block lacked.
    (The effect scalar is `max_m w_m·flag`, NOT a full-axis noisy-OR — see the saturation test below.)"""
    op = _op()
    ctx = _ctx(opp_species=242, opp_t1=_T2I["NORMAL"], opp_t2=0)   # Blissey-ish
    eff_toxic = op(ctx, _belief_logits(("toxic",)))[0, TEAM_SIZE * _DMG_PER_MON:]
    eff_recover = op(ctx, _belief_logits(("recover",)))[0, TEAM_SIZE * _DMG_PER_MON:]
    eff_attack = op(ctx, _belief_logits(("earthquake",)))[0, TEAM_SIZE * _DMG_PER_MON:]
    assert eff_toxic[_EFF["status"]] > 0.9                 # believed Toxic → high status threat
    assert eff_recover[_EFF["recovery"]] > 0.9             # believed Recover → high recovery threat
    assert eff_attack[_EFF["status"]] < 0.2 and eff_attack[_EFF["recovery"]] < 0.2


def test_effect_scalars_not_saturated_under_realistic_fusion_belief():
    """REGRESSION for the review's M1: a full-axis noisy-OR saturated the effect scalars to ~1 from the
    floor alone. With the max aggregation + the realistic prior-fusion belief, a PURE ATTACKER
    (Aerodactyl — its prior moveset is almost all attacks) reads LOW status/recovery/boost threat."""
    op = _op()
    mb = MoveBelief(_M, _EMB, prior_fusion=True, n_species=_S)
    # Aerodactyl (num 142, Rock/Flying), no revealed moves → belief = its move prior (attacking).
    ctx = _ctx(opp_species=142, opp_t1=_T2I["ROCK"], opp_t2=_T2I["FLYING"])
    out = _fused_then_op(op, mb, ctx, revealed_moves_active=())
    eff = out[0, TEAM_SIZE * _DMG_PER_MON:]
    for k in ("status", "recovery", "boost", "phaze"):
        assert eff[_EFF[k]] < 0.5, f"{k} effect scalar saturated under fusion: {eff[_EFF[k]].item():.3f}"


def test_provenance_revealed_vs_guess():
    """Provenance ≈ the belief weight of the dominant KO threat: ~1 for a believed (high-logit) move,
    low when the threat is only a usage-prior guess (low logit)."""
    op = _op()
    ctx = _ctx(opp_species=248, opp_t1=_T2I["ROCK"], opp_t2=_T2I["DARK"])  # vs a Water/Ground defender
    # high-confidence Earthquake (4× vs Water/Ground) → provenance ~1 on that defender.
    permon_hi = op(ctx, _belief_logits(("earthquake",)))[:, :TEAM_SIZE * _DMG_PER_MON].reshape(1, TEAM_SIZE, _DMG_PER_MON)
    # all-floor belief (no high logit) → the dominant threat is a guess → low provenance.
    permon_lo = op(ctx, _belief_logits(()))[:, :TEAM_SIZE * _DMG_PER_MON].reshape(1, TEAM_SIZE, _DMG_PER_MON)
    assert permon_hi[0, 0, 11].item() > 0.9               # provenance is now slot 11
    assert permon_lo[0, 0, 11].item() < 0.2


# --------------------------------------------------------------------------- ALL known/unknown combos
def _fused_then_op(op, mb, ctx, revealed_moves_active=()):  # integrated fusion→op
    opp_tok = torch.zeros(ctx.batch_size, TEAM_SIZE, 128)
    emb = torch.nn.Embedding(_M, _EMB)
    opp_species = ctx.species_ids[:, TEAM_SIZE:]
    opp_move_ids = ctx.all_move_ids[:, TEAM_SIZE:, :].clone()
    for j, mid in enumerate(revealed_moves_active):
        opp_move_ids[:, 0, j] = gen3_data.moves.get(mid).num     # reveal on the active (slot 0 of opp)
    ctx.all_move_ids = torch.cat([ctx.all_move_ids[:, :TEAM_SIZE, :], opp_move_ids], dim=1)
    _, logits = mb(opp_tok, torch.ones(ctx.batch_size, TEAM_SIZE, dtype=torch.bool), emb,
                   opp_species_ids=opp_species, opp_move_ids=opp_move_ids)
    return op(ctx, logits)


@pytest.mark.parametrize("desc,opp_species,revealed,opp_active", [
    ("hidden mon, zero moves (species 0 → floor)",          0,    (),                       True),
    ("revealed mon, zero revealed attacks (→ prior)",        227,  (),                       True),  # Skarmory
    ("revealed mon, partial reveal (1 move pinned)",         227,  ("drillpeck",),           True),
    ("revealed mon, fully revealed (4 moves pinned)",        227,  ("drillpeck", "spikes", "roar", "rest"), True),
    ("no opponent active (block must be zero)",              227,  (),                       False),
])
def test_all_known_unknown_combinations_finite(desc, opp_species, revealed, opp_active):
    """The unified fusion→op must give finite, sane output for every reveal state — and zero when
    there is no opp active. This is the core robustness the design must guarantee."""
    op = _op()
    mb = MoveBelief(_M, _EMB, prior_fusion=True, n_species=_S)
    ot = (_T2I["STEEL"], _T2I["FLYING"]) if opp_species == 227 else (0, 0)
    ctx = _ctx(opp_species=opp_species, opp_t1=ot[0], opp_t2=ot[1], opp_active=opp_active)
    out = _fused_then_op(op, mb, ctx, revealed_moves_active=revealed)
    assert out.shape == (1, _OUT)
    assert torch.isfinite(out).all(), desc
    if not opp_active:
        assert (out == 0).all(), desc                       # no-active gate
    else:
        assert out.abs().sum() > 0, desc                    # some threat signal exists


def test_revealed_no_moves_uses_species_prior():
    """The key case: we KNOW it's a Skarmory but it hasn't attacked → the belief (hence the op's threat)
    is driven by Skarmory's move PRIOR, not zero. Drill Peck (its prior Flying STAB) should register a
    threat vs a Fighting-type defender, even with zero revealed moves."""
    op = _op()
    mb = MoveBelief(_M, _EMB, prior_fusion=True, n_species=_S)
    # defender weak to Flying (e.g. a Fighting type — Heracross 214, Bug/Fighting, 4× Flying)
    ctx = _ctx(opp_species=227, opp_t1=_T2I["STEEL"], opp_t2=_T2I["FLYING"],
               defenders=[(214, _T2I["BUG"], _T2I["FIGHTING"])] + [(0, 0, 0)] * 5)
    out = _fused_then_op(op, mb, ctx, revealed_moves_active=())   # ZERO revealed moves
    permon = out[:, :TEAM_SIZE * _DMG_PER_MON].reshape(1, TEAM_SIZE, _DMG_PER_MON)
    phys_high = permon[0, 0, 1].item()                            # Drill Peck is physical (high-roll magnitude)
    assert phys_high > 0.05, f"species prior gave no threat with zero revealed moves: {phys_high}"
def test_defender_ability_immunity():
    """A Levitate defender reads ~0 incoming Ground (the believed Earthquake is nullified); an identical
    defender without Levitate reads a real Ground threat. Our abilities are revealed → known, not believed."""
    op = _op()
    levi = gen3_data.abilities.get("levitate").num
    ttar = (248, _T2I["ROCK"], _T2I["DARK"])              # Tyranitar: 2× weak to Ground
    ctx = _ctx(defenders=[ttar, ttar] + [(0, 0, 0)] * 4, def_abilities=[levi, 0],
               opp_species=248, opp_t1=_T2I["GROUND"], opp_t2=0)
    permon = op(ctx, _belief_logits(("earthquake",)))[:, :TEAM_SIZE * _DMG_PER_MON].reshape(1, TEAM_SIZE, _DMG_PER_MON)
    # Levitate nullifies the believed Earthquake (eff=0). The residual ~sigmoid(-10)≈4.5e-5 is the
    # belief-floor weight on the ~399 OTHER physical candidates (this test's base=-10), not an immunity
    # leak — the >0.1 on the non-Levitate slot below proves the channel works.
    assert permon[0, 0, 1].item() < 1e-4                  # Levitate → ~0 incoming Ground (phys high-roll)
    assert permon[0, 1, 1].item() > 0.1                   # no Levitate → real Ground threat
    assert permon[0, 0, 3].item() < 1e-4                  # and ~0 phys P(KO) (slot 3)


def test_thick_fat_halves_fire():
    op = _op()
    tf = gen3_data.abilities.get("thickfat").num
    dfn = (143, _T2I["NORMAL"], 0)                        # Snorlax (Normal): Fire neutral
    ctx = _ctx(defenders=[dfn, dfn] + [(0, 0, 0)] * 4, def_abilities=[tf, 0],
               opp_species=146, opp_t1=_T2I["FIRE"], opp_t2=0)   # a Fire attacker
    permon = op(ctx, _belief_logits(("flamethrower",)))[:, :TEAM_SIZE * _DMG_PER_MON].reshape(1, TEAM_SIZE, _DMG_PER_MON)
    # Thick Fat halves Fire → spec high-roll is ~half the no-ability defender's (both < clamp).
    assert permon[0, 0, 6].item() == pytest.approx(0.5 * permon[0, 1, 6].item(), rel=0.03)


# --------------------------------------------------------------------------- our-side screens
def test_our_reflect_halves_physical_chip():
    op = _op()
    dfn = [(242, _T2I["NORMAL"], 0)] + [(0, 0, 0)] * 5    # Blissey: huge HP → chip well under the 1.5 clamp
    atk = dict(opp_species=248, opp_t1=_T2I["GROUND"], opp_t2=0)
    base = op(_ctx(defenders=dfn, **atk), _belief_logits(("earthquake",)))
    refl = op(_ctx(defenders=dfn, our_reflect=True, **atk), _belief_logits(("earthquake",)))
    b = base[:, :TEAM_SIZE * _DMG_PER_MON].reshape(1, TEAM_SIZE, _DMG_PER_MON)
    r = refl[:, :TEAM_SIZE * _DMG_PER_MON].reshape(1, TEAM_SIZE, _DMG_PER_MON)
    assert b[0, 0, 1].item() > 0.0
    assert r[0, 0, 1].item() == pytest.approx(0.5 * b[0, 0, 1].item(), rel=0.02)   # Reflect halves physical (high-roll)
