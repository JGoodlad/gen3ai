"""The 2026-10-07 damage-op GIGO fix (gen3_op_ability_status_gigo_v1): every test here FAILS on revert.

* gen3_op_ability_known_v1 — an opponent's ability is revealed only by its `known` flag; an unrevealed slot's
  id1 is the species' top-1 PRIOR and the op takes the species' Smogon marginal (damage, status, secondary).
* gen3_op_status_rules_v1 — status landing folds their Safeguard (outgoing), our Safeguard, incoming Sleep
  Clause, Freeze Clause and our Substitute (incoming), and Yawn's delayed sleep (both directions).
Every rule is verified at `deps/pokemon-showdown` (cited in `status_rules.py` / `damage_tables.py`).
"""
from __future__ import annotations

import pytest
import torch

from agents import gen3_data
from agents.model.damage_op_layout import (_COND_SLP_IDX, _COND_FRZ_IDX, _SUBSTITUTE_CTX_IDX,
                                           _YAWN_CTX_IDX)
from agents.model.damage_op_test import _T2I, _fake_ctx, _fake_ctx_out, _status_land
from agents.model.extractor_ctx import POKEMON_ABILITY_KNOWN_OFFSET
from agents.model.features_extractor import DamageOperator
from agents.observation.constants import POKEMON_CONDITION_OFFSET, POKEMON_SLEEP_BELIEF_OFFSET, TEAM_SIZE
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings


@pytest.fixture(scope="module")
def op():
    return DamageOperator(Gen3ObservationEncoder(load_mappings()).get_layout(), outgoing=True)


def _n(mid: str) -> int:
    return int(gen3_data.moves.get(mid).num)


def _a(aid: str) -> int:
    return int(gen3_data.abilities.get(aid).num)


# ------------------------------------------------------------------ the ability's `known` flag, DAMAGE side
def test_unrevealed_flash_fire_is_a_prior_not_a_certain_immunity(op):
    """Houndoom (Fire/Dark): Smogon Flash Fire 0.78 / Early Bird 0.22. The REAL observation of an unrevealed
    Houndoom carries Flash Fire (its top-1) in id1 with known = 0: our Flamethrower is then an expected
    0.5 × (1 − 0.78) hit, NOT a certain 0 (the shipped read). Revealed, it is exactly 0."""
    hi = 1                                                      # move 0's [low, high, crit, ko]: high

    def high(known):
        ctx = _fake_ctx_out(our_species=6, our_t1=_T2I["FIRE"], our_t2=_T2I["FLYING"],
                            our_moves=[_n("flamethrower"), 0, 0, 0], our_move_types=[_T2I["FIRE"], 0, 0, 0],
                            opp_species=229, opp_t1=_T2I["FIRE"], opp_t2=_T2I["DARK"], move_mask=[1, 0, 0, 0],
                            opp_ability=_a("flashfire"), opp_known=known)
        return op._outgoing_block(ctx)[0, hi].item()

    assert high(known=1.0) == 0.0
    unrevealed = high(known=0.0)
    p_ff = float(gen3_data.priors.ability("houndoom")["flashfire"])
    assert 0.0 < unrevealed
    no_ability = _fake_ctx_out(our_species=6, our_t1=_T2I["FIRE"], our_t2=_T2I["FLYING"],
                               our_moves=[_n("flamethrower"), 0, 0, 0], our_move_types=[_T2I["FIRE"], 0, 0, 0],
                               opp_species=229, opp_t1=_T2I["FIRE"], opp_t2=_T2I["DARK"], move_mask=[1, 0, 0, 0])
    no_ability.pokemon_part[:, TEAM_SIZE, POKEMON_ABILITY_KNOWN_OFFSET] = 1.0      # revealed, no-effect ability
    assert unrevealed == pytest.approx((1.0 - p_ff) * op._outgoing_block(no_ability)[0, hi].item(), rel=1e-4)


def test_the_view_zeroes_unrevealed_ids_and_mixes_the_prior(op):
    ctx = _fake_ctx_out(our_species=6, our_t1=0, our_t2=0, our_moves=[0, 0, 0, 0], our_move_types=[0, 0, 0, 0],
                        opp_species=94, opp_t1=_T2I["GHOST"], opp_t2=_T2I["POISON"], move_mask=[0, 0, 0, 0],
                        opp_ability=_a("levitate"), opp_known=0.0)                    # Gengar: Levitate 1.0
    ids, known, sp = op.opp_ability_view(ctx)
    assert int(ids[0, 0]) == 0 and float(known[0, 0]) == 0.0 and int(sp[0, 0]) == 94
    gnd = _T2I["GROUND"]
    assert float(op.opp_ability_damage_mult(ctx)[0, 0, gnd]) == pytest.approx(0.0, abs=1e-6)   # prior: 1.0
    ctx2 = _fake_ctx_out(our_species=6, our_t1=0, our_t2=0, our_moves=[0, 0, 0, 0], our_move_types=[0, 0, 0, 0],
                         opp_species=229, opp_t1=_T2I["FIRE"], opp_t2=_T2I["DARK"], move_mask=[0, 0, 0, 0],
                         opp_ability=_a("flashfire"), opp_known=0.0)
    p_ff = float(gen3_data.priors.ability("houndoom")["flashfire"])
    assert float(op.opp_ability_damage_mult(ctx2)[0, 0, _T2I["FIRE"]]) == pytest.approx(1.0 - p_ff, abs=1e-5)


# ----------------------------------------------------------------------------------- OUTGOING status rules
def test_their_safeguard_blocks_every_major_status_and_yawn_not_leech_seed(op):
    moves = ["toxic", "thunderwave", "yawn", "leechseed"]
    p, known = _status_land(op, our_moves=moves, opp_t1=_T2I["NORMAL"], opp_t2=0)
    assert p.tolist() == pytest.approx([0.85, 1.0, 1.0, 0.9], abs=1e-4)
    p, known = _status_land(op, our_moves=moves, opp_t1=_T2I["NORMAL"], opp_t2=0, opp_safeguard=True)
    assert p.tolist() == pytest.approx([0.0, 0.0, 0.0, 0.9], abs=1e-4)
    assert known[:3].tolist() == [1.0, 1.0, 1.0]                    # a public side condition: certain


def test_yawn_is_a_delayed_sleep_with_its_own_fail_rules(op):
    """Yawn (`data/moves.ts` yawn): fails on a statused target, on a drowsy one, behind a Substitute, under
    Sleep Clause, and its sleep is blocked by Insomnia / Vital Spirit. The shipped op priced it as no status."""
    y = ["yawn", "yawn", "yawn", "yawn"]
    assert _status_land(op, our_moves=y, opp_t1=_T2I["NORMAL"], opp_t2=0)[0][0].item() == pytest.approx(1.0)
    assert _status_land(op, our_moves=y, opp_t1=_T2I["NORMAL"], opp_t2=0, opp_drowsy=True)[0][0].item() == 0.0
    assert _status_land(op, our_moves=y, opp_t1=_T2I["NORMAL"], opp_t2=0,
                        opp_active_status_idx=2)[0][0].item() == 0.0                    # paralysed
    assert _status_land(op, our_moves=y, opp_t1=_T2I["NORMAL"], opp_t2=0, opp_substitute=True)[0][0].item() == 0.0
    assert _status_land(op, our_moves=y, opp_t1=_T2I["NORMAL"], opp_t2=0, bench_sleep=True)[0][0].item() == 0.0
    assert _status_land(op, our_moves=y, opp_t1=_T2I["NORMAL"], opp_t2=0, bench_sleep=True,
                        bench_sleep_is_rest=True)[0][0].item() == pytest.approx(1.0)
    assert _status_land(op, our_moves=y, opp_t1=_T2I["NORMAL"], opp_t2=0, opp_species=248,
                        opp_ability=_a("insomnia"))[0][0].item() == 0.0


# ----------------------------------------------------------------------------------- INCOMING status rules
def _incoming(op, move: str, *, high=0.3, setup=None):
    """P(their `move` (the single top-K seat) applies a status to each of our six) — slot 0 our active."""
    ctx = _fake_ctx(op, attacker_num=143, attacker_t1=_T2I["NORMAL"], attacker_t2=0,
                    defenders=[(143, _T2I["NORMAL"], 0)] * TEAM_SIZE, hp_probs_active=[1.0 / 16] * 16)
    ctx.hp_and_active[:, :TEAM_SIZE, 0] = 1.0
    if setup is not None:
        setup(ctx)
    topk = torch.tensor([[_n(move)]])
    high_topk = torch.full((1, TEAM_SIZE, 1), float(high))
    return op._incoming_status_lands(ctx, topk, high_topk)[0, :, 0]


def test_our_safeguard_blocks_their_status_and_secondaries(op):
    def sg(ctx):
        ctx.screen_feature[:, 4] = 1.0                            # our Safeguard
    assert _incoming(op, "thunderwave").tolist() == pytest.approx([1.0] * 6)
    assert _incoming(op, "thunderwave", setup=sg).tolist() == [0.0] * 6
    assert _incoming(op, "bodyslam").max().item() == pytest.approx(0.3, abs=1e-4)
    assert _incoming(op, "bodyslam", setup=sg).max().item() == 0.0


def test_our_substitute_blocks_their_status_on_our_active_only(op):
    def sub(ctx):
        ctx.our_ctx_raw[:, _SUBSTITUTE_CTX_IDX] = 1.0
    for mv in ("toxic", "leechseed"):
        p = _incoming(op, mv, setup=sub)
        assert p[0].item() == 0.0 and p[1].item() > 0.8, mv


def test_incoming_sleep_clause_and_freeze_clause(op):
    def asleep(rest):
        def f(ctx):
            ctx.pokemon_part[:, 3, POKEMON_CONDITION_OFFSET + _COND_SLP_IDX] = 1.0
            ctx.pokemon_part[:, 3, POKEMON_SLEEP_BELIEF_OFFSET] = 1.0 if rest else 0.0
        return f

    assert _incoming(op, "spore")[0].item() == pytest.approx(1.0)
    assert _incoming(op, "spore", setup=asleep(rest=False))[0].item() == 0.0     # our live non-Rest sleeper
    assert _incoming(op, "spore", setup=asleep(rest=True))[0].item() == pytest.approx(1.0)
    assert _incoming(op, "icebeam")[0].item() == pytest.approx(0.1, abs=1e-4)

    def frozen(ctx):
        ctx.pokemon_part[:, 4, POKEMON_CONDITION_OFFSET + _COND_FRZ_IDX] = 1.0
    assert _incoming(op, "icebeam", setup=frozen)[0].item() == 0.0              # Freeze Clause


def test_their_yawn_on_our_drowsy_active_fails(op):
    def drowsy(ctx):
        ctx.our_ctx_raw[:, _YAWN_CTX_IDX] = 1.0
    assert _incoming(op, "yawn").tolist() == pytest.approx([1.0] * 6)
    p = _incoming(op, "yawn", setup=drowsy)
    assert p[0].item() == 0.0 and p[1].item() == pytest.approx(1.0)
