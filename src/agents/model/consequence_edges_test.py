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
    assert cells.shape == (2, 4, TEAM_SIZE, 4)   # the OUTGOING kernel half of _EDGE_C1_CELL (6)
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


def test_curse_non_ghost_prices_atk_up_spe_down():
    """CurseLax semantics: a non-Ghost user's Curse raises the best physical line AND reads a
    NEGATIVE outspeed delta (−1 spe is part of the priced consequence)."""
    fe = _make(**_C1_TOGGLES).eval()
    ctx = _boost_ctx(fe, seed=81)
    curse = gen3_data.moves.get("curse").num
    ctx.our_active_req_move_ids[:, 1] = curse
    from agents.model.damage_tables import _T2I
    ghost = _T2I["GHOST"]
    ctx.type1_ids[torch.arange(2), ctx.our_active_idx] = _T2I["NORMAL"]   # Snorlax-shaped user
    ctx.type2_ids[torch.arange(2), ctx.our_active_idx] = _T2I["NORMAL"]
    with torch.no_grad():
        cells = fe.damage_op.pairwise_boost(ctx)
        base = fe.damage_op.pairwise_outgoing(ctx)
    assert bool((cells[:, 1, :, 0] > 0).any()), "non-Ghost Curse must fire is_boost"
    live = base[..., 1].amax(dim=1) > 1e-6
    assert bool((cells[:, 1, :, 1][live] > 0).all()), "+1 atk must raise the best physical line"
    d_spd = cells[:, 1, :, 3]
    assert float(d_spd.max()) <= 0.0, "-1 spe can never RAISE P(outspeed)"
    assert float(d_spd.min()) < 0.0, "some gated matchup must LOSE outspeed probability"
    # The GHOST branch: same slot, Ghost-typed user → the whole row must be unpriced.
    ctx.type1_ids[torch.arange(2), ctx.our_active_idx] = ghost
    with torch.no_grad():
        cells_g = fe.damage_op.pairwise_boost(ctx)
    assert float(cells_g[:, 1].abs().sum()) == 0.0, "a Ghost user's Curse stays a zero row"


def _force_active_speed_scenario(fe, ctx):
    """Our active = neutral-spread AERODACTYL (spe 130, SpD 75) at slot 0; opp slot 0 = revealed
    alive SNORLAX (spe 30, SpD 110). The two stats separate HARD in both directions: correct
    physics reads P(outspeed) ≈ 1; the SpD-as-speed bug read ≈ 0 (186 vs 256). No spread belief
    → the neutral fallback (the arm where BOTH sides of the 2026-08-06 GIGO were wrong)."""
    from agents.observation.pokemon import POKEMON_SPREAD_OFFSET
    aero = gen3_data.species.get("aerodactyl")
    lax = gen3_data.species.get("snorlax")
    B = ctx.batch_size
    ctx.our_active_idx[:] = 0
    ctx.species_ids[:, 0] = aero.num
    ctx.species_ids[:, TEAM_SIZE] = lax.num
    sp = POKEMON_SPREAD_OFFSET
    ctx.pokemon_part[:, 0, sp:sp + 18] = 0.0
    ctx.pokemon_part[:, 0, sp:sp + 6] = 1.0          # IV 31
    ctx.pokemon_part[:, 0, sp + 12] = 1.0            # spread_known
    ctx.pokemon_part[:, 0, sp + 13:sp + 18] = 1.0    # neutral nature
    from agents.model.damage_op import POKEMON_CONDITION_OFFSET
    ctx.pokemon_part[:, 0, POKEMON_CONDITION_OFFSET:POKEMON_CONDITION_OFFSET + 7] = 0.0
    ctx.pokemon_part[:, TEAM_SIZE, POKEMON_CONDITION_OFFSET:POKEMON_CONDITION_OFFSET + 7] = 0.0
    ctx.hp_and_active[:, 0, 0] = 1.0
    ctx.hp_and_active[:, TEAM_SIZE, 0] = 1.0
    ctx.opp_believed_mask[:, 0] = False
    ctx.our_ctx_raw[:, 0:14] = 0.0                   # no stat stages


def test_speed_reads_the_speed_stat_not_spd():
    """REGRESSION (the 2026-08-06 SpD-as-speed GIGO): pairwise_speed and pairwise_boost read
    BASE_STATS/iv/ev index 4 (Special Defense) + nat[3] as "speed", so neutral Aerodactyl (spe
    130) read SLOWER than Snorlax's fallback — both trained generations' V edge priced bulk as
    speed. Correct physics: P(Aero outspeeds Lax) ≈ 1, and Agility's d_outspeed SATURATES to
    ~0 (already faster). The buggy indices read ≈0 and a LARGE Agility gain — this test fails
    hard if either kernel's index reverts."""
    fe = _make(**_C1_TOGGLES).eval()
    obs = _obs(seed=91)
    with torch.no_grad():
        fe(obs)
    ctx = fe.unpack(obs)
    _force_active_speed_scenario(fe, ctx)
    with torch.no_grad():
        v = fe.damage_op.pairwise_speed(ctx)                       # NO spread belief: fallback arm
    assert float(v[:, 0, 0, 0].min()) > 0.9, \
        "neutral Aerodactyl must read ~certain to outspeed Snorlax (the V kernel)"
    ctx.our_active_req_move_ids[:, :] = 0
    ctx.our_active_req_move_ids[:, 1] = _AGILITY
    with torch.no_grad():
        cells = fe.damage_op.pairwise_boost(ctx)                   # C1 shares the recipe
    assert float(cells[:, 1, 0, 3].abs().max()) < 0.05, \
        "Agility on an already-faster mon must read a SATURATED ~0 outspeed delta"


def _constructed_logits(fe, ctx, move_id):
    """Belief logits making ONE move the near-certain candidate for every opp slot."""
    M = fe.damage_op.MOVE_BP.shape[0]
    logits = torch.full((ctx.batch_size, TEAM_SIZE, M), -30.0)
    logits[:, :, move_id] = 10.0
    return logits


def _c1b_ctx(fe, seed):
    """A gated C1b scenario: our active = a REAL-spread neutral Snorlax (a random-obs spread is
    near-zero → the divisor is tiny and damage saturates the chip cap in EVERY world, cancelling
    the delta), NORMAL-typed, neutral ability, alive; every opp revealed+alive."""
    from agents.model.damage_tables import _T2I
    from agents.observation.pokemon import POKEMON_SPREAD_OFFSET
    ctx = _boost_ctx(fe, seed=seed)
    ar = torch.arange(2)
    ctx.species_ids[ar, ctx.our_active_idx] = gen3_data.species.get("snorlax").num
    sp = POKEMON_SPREAD_OFFSET
    act = ctx.our_active_idx
    ctx.pokemon_part[ar, act, sp:sp + 18] = 0.0
    ctx.pokemon_part[ar, act, sp:sp + 6] = 1.0              # IV 31
    ctx.pokemon_part[ar, act, sp + 12] = 1.0                # spread_known
    ctx.pokemon_part[ar, act, sp + 13:sp + 18] = 1.0        # neutral nature
    ctx.type1_ids[ar, act] = _T2I["NORMAL"]
    ctx.type2_ids[ar, act] = _T2I["NORMAL"]
    ctx.ability1_ids[ar, act] = 0                           # neutral row (no Levitate surprises)
    ctx.our_ctx_raw[:, 0:14] = 0.0                          # no live stat stages
    ctx.hp_and_active[ar, act, 0] = 1.0
    ctx.hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, 0] = 1.0
    return ctx


def test_incoming_deltas_respect_the_boosted_channel():
    """C1b channel physics: Iron Defense shrinks the worst PHYSICAL believed hit and ignores a
    special one; Amnesia the exact reverse; Swords Dance reads ~0 BY PHYSICS (its hypothetical
    world leaves def/spd unchanged); non-Ghost Curse's +1 Def half now prices."""
    fe = _make(**_C1_TOGGLES).eval()
    ctx = _c1b_ctx(fe, seed=95)
    eq = gen3_data.moves.get("earthquake").num
    surf = gen3_data.moves.get("surf").num
    ctx.our_active_req_move_ids[:, 1] = gen3_data.moves.get("irondefense").num
    ctx.our_active_req_move_ids[:, 2] = gen3_data.moves.get("amnesia").num
    ctx.our_active_req_move_ids[:, 3] = _SD
    with torch.no_grad():
        vs_phys = fe.damage_op.pairwise_boost_incoming(ctx, _constructed_logits(fe, ctx, eq))
        vs_spec = fe.damage_op.pairwise_boost_incoming(ctx, _constructed_logits(fe, ctx, surf))
    assert vs_phys.shape == (2, 4, TEAM_SIZE, 2)
    assert float(vs_phys[:, 1, :, 0].max()) < -1e-4, "Iron Defense must shrink the physical hit"
    assert float(vs_phys[:, 2, :, 0].abs().max()) < 1e-6, "Amnesia ignores a physical hit"
    assert float(vs_spec[:, 2, :, 0].max()) < -1e-4, "Amnesia must shrink the special hit"
    assert float(vs_spec[:, 1, :, 0].abs().max()) < 1e-6, "Iron Defense ignores a special hit"
    assert float(vs_phys[:, 3].abs().max()) < 1e-6, "SD's world leaves def/spd unchanged"
    ctx.our_active_req_move_ids[:, 1] = gen3_data.moves.get("curse").num
    with torch.no_grad():
        vs_curse = fe.damage_op.pairwise_boost_incoming(ctx, _constructed_logits(fe, ctx, eq))
    assert float(vs_curse[:, 1, :, 0].max()) < -1e-4, "Curse's +1 Def half must price (C1b)"


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
