"""Unit tests for the differentiable GPU damage operator (`DamageOperator`, `damage_tables`).

Pins: the lookup buffers (TypeEncoder axis, immunity exact-0, FAIRY skipped, HP num-237 collision,
type-split), the typed-Hidden-Power expansion (16 candidates with DISTINCT type effectiveness,
weighted P(present)·P(type)), the per-channel soft-max aggregation, finiteness (zero obs + a
random-logits/HP fuzz — the degenerate-weights NaN case), the gradient flowing back into the
move-belief head (the "sharpens the belief" property), the extractor wiring (off → no module /
baseline projection dims; on → +6·feats on both heads), the no-opp-active gate, and the
move-belief dependency guard.
"""
import types

import numpy as np
import gymnasium as gym
import torch
import pytest

from agents.model.features_extractor import (
    Gen3FeaturesExtractor, DamageOperator, _DMG_PER_MON, _DMG_EFFECT, _DMG_INCOMING_SEC,
    _DMG_OUT_N_MOVES, _DMG_OUT_PER_MOVE, _DMG_STATUS, _DMG_STATUS_N_MOVES, _DMG_CB, _COND_SLP_IDX,
    _SUBSTITUTE_CTX_IDX, TEAM_SIZE,
)
from agents.model import damage_tables as dt
from agents.observation.constants import (
    POKEMON_SPREAD_OFFSET, POKEMON_FULL_DIM, POKEMON_CONDITION_OFFSET, POKEMON_SLEEP_BELIEF_OFFSET,
    ACTIVE_CONTEXT_DIM)
from agents.observation.types import TypeEncoder
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

_T2I = TypeEncoder.TYPE_TO_IDX


# --------------------------------------------------------------------------- shared builders
def _make_model(**kwargs):
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    obs_space = gym.spaces.Box(0.0, 1.0, shape=(layout["total_dim"],), dtype=np.float32)
    model = Gen3FeaturesExtractor(obs_space, layout=layout, mappings=mappings, **kwargs)
    return model, layout


def _op_and_layout():
    mappings = load_mappings()
    layout = Gen3ObservationEncoder(mappings).get_layout()
    return DamageOperator(layout), layout


def _make_layout():
    return Gen3ObservationEncoder(load_mappings()).get_layout()


def _hp_slot(type_name: str) -> int:
    """Index into the 16-slot HP order for a given type (via the op's HP_TYPE_IDX buffer)."""
    op, _ = _op_and_layout()
    target = _T2I[type_name]
    return int((op.HP_TYPE_IDX == target).nonzero()[0].item())


def _fake_ctx(op, *, attacker_num, attacker_t1, attacker_t2,
              defenders, hp_probs_active, opp_active_local=0, B=1):
    """Hand-built ctx (SimpleNamespace) exercising the op's exact reads. `defenders` is a list of
    (species_num, type1, type2) for our 6 slots; spread set to IV 31 / EV 0 / neutral nature, full HP.
    The opp active sits at slot TEAM_SIZE+opp_active_local with its active flag set."""
    n = 2 * TEAM_SIZE
    species = torch.zeros(B, n, dtype=torch.long)
    t1 = torch.zeros(B, n, dtype=torch.long)
    t2 = torch.zeros(B, n, dtype=torch.long)
    for i, (num, a, b) in enumerate(defenders):
        species[:, i] = num
        t1[:, i] = a
        t2[:, i] = b
    species[:, TEAM_SIZE + opp_active_local] = attacker_num
    t1[:, TEAM_SIZE + opp_active_local] = attacker_t1
    t2[:, TEAM_SIZE + opp_active_local] = attacker_t2

    hp_and_active = torch.zeros(B, n, POKEMON_FULL_DIM)
    hp_and_active[:, :TEAM_SIZE, 0] = 1.0                      # our mons full HP (alive)
    hp_and_active[:, TEAM_SIZE + opp_active_local, -1] = 1.0   # opp active flag

    pokemon_part = torch.zeros(B, n, POKEMON_FULL_DIM)
    sp = pokemon_part[:, :, POKEMON_SPREAD_OFFSET:POKEMON_SPREAD_OFFSET + 18]
    sp[..., 0:6] = 1.0      # IV 31/31
    sp[..., 6:12] = 0.0     # EV 0
    sp[..., 13:18] = 1.0    # neutral nature

    hp_probs = torch.zeros(B, n, 16)
    hp_probs[:, TEAM_SIZE + opp_active_local] = torch.tensor(hp_probs_active, dtype=torch.float32)

    return types.SimpleNamespace(
        batch_size=B, device=torch.device("cpu"),
        opp_active_local=torch.full((B,), opp_active_local, dtype=torch.long),
        species_ids=species, type1_ids=t1, type2_ids=t2,
        ability1_ids=torch.zeros(B, n, dtype=torch.long),       # no ability (mult 1.0) by default
        item_ids=torch.zeros(B, n, dtype=torch.long),           # no item (no Choice Band) by default
        screen_feature=torch.zeros(B, 8),                       # no screens by default
        our_ctx_raw=torch.zeros(B, ACTIVE_CONTEXT_DIM), opp_ctx_raw=torch.zeros(B, ACTIVE_CONTEXT_DIM),  # boosts++volatiles
        our_active_idx=torch.zeros(B, dtype=torch.long), weather_feature=torch.zeros(B, 7),  # no weather
        hp_and_active=hp_and_active, pokemon_part=pokemon_part, hp_probs=hp_probs,
    )


def _logits_hp_only(n_moves, B=1):
    """Belief logits that put ~all mass on Hidden Power (num 237) and ~none elsewhere, so only the
    typed-HP candidates contribute — isolates the HP-expansion path."""
    lg = torch.full((B, TEAM_SIZE, n_moves), -10.0)
    lg[:, :, dt.HIDDEN_POWER_NUM] = 10.0
    return lg


# --------------------------------------------------------------------------- lookup buffers
def test_buffers_axis_and_immunity():
    b = dt.build_damage_buffers(400, 400, 100)
    C = b["CHART"]
    # Immunities are EXACT 0 (fall out of the effectiveness product, no branch).
    assert C[_T2I["FLYING"], _T2I["GROUND"]].item() == 0.0     # Ground vs Flying
    assert C[_T2I["GHOST"], _T2I["NORMAL"]].item() == 0.0      # Normal vs Ghost
    assert C[_T2I["DARK"], _T2I["PSYCHIC"]].item() == 0.0      # Psychic vs Dark
    # 4× stacks through the product.
    assert C[_T2I["WATER"], _T2I["ELECTRIC"]].item() * C[_T2I["FLYING"], _T2I["ELECTRIC"]].item() == 4.0
    # Unknown (idx 0) row/col stays neutral.
    assert C[0].unique().tolist() == [1.0] and C[:, 0].unique().tolist() == [1.0]


def test_buffers_hp_collision_and_type_split():
    b = dt.build_damage_buffers(400, 400, 100)
    assert b["MOVE_BP"][dt.HIDDEN_POWER_NUM].item() == 0.0     # HP collision: bare slot left 0
    # gen3 type-split: Ground physical, Water special.
    assert b["TYPE_IS_PHYS"][_T2I["GROUND"]].item() == 1.0
    assert b["TYPE_IS_PHYS"][_T2I["WATER"]].item() == 0.0
    # No FAIRY on the gen3 axis.
    assert "FAIRY" not in _T2I
    # HP candidate buffers cover 16 types, is_phys per type.
    assert b["HP_TYPE_IDX"].shape == (16,) and b["HP_IS_PHYS"].shape == (16,)
    assert b["HP_IS_PHYS"][_hp_slot("GROUND")].item() == 1.0
    assert b["HP_IS_PHYS"][_hp_slot("GRASS")].item() == 0.0


def test_move_buffers_known_values():
    b = dt.build_damage_buffers(400, 400, 100)
    from agents import gen3_data
    eq = gen3_data.moves.get("earthquake")
    assert b["MOVE_BP"][eq.num].item() == 100.0
    assert b["MOVE_TYPE_IDX"][eq.num].item() == _T2I["GROUND"]
    assert b["MOVE_PHYS"][eq.num].item() == 1.0
    tb = gen3_data.moves.get("thunderbolt")
    assert b["MOVE_PHYS"][tb.num].item() == 0.0      # Electric → special


# --------------------------------------------------------------------------- typed-HP expansion
def test_typed_hp_distinct_effectiveness():
    """HP Grass vs a Water/Ground mon (4×) reads far higher than the SAME mon vs HP Ice (neutral) —
    typed HPs get DISTINCT type effectiveness, the whole point of expanding num-237 into 16 candidates."""
    op, layout = _op_and_layout()
    swampert = (260, _T2I["WATER"], _T2I["GROUND"])            # 4× weak to Grass; Ice neutral (½×2)
    defenders = [swampert] + [(0, 0, 0)] * 5
    attacker = dict(attacker_num=248, attacker_t1=_T2I["NORMAL"], attacker_t2=0)  # Normal → no Grass STAB

    grass = [0.0] * 16; grass[_hp_slot("GRASS")] = 1.0
    ice = [0.0] * 16; ice[_hp_slot("ICE")] = 1.0
    lg = _logits_hp_only(layout["max_moves"])

    g = op(_fake_ctx(op, defenders=defenders, hp_probs_active=grass, **attacker), lg)[:, :TEAM_SIZE * _DMG_PER_MON].reshape(1, TEAM_SIZE, _DMG_PER_MON)
    i = op(_fake_ctx(op, defenders=defenders, hp_probs_active=ice, **attacker), lg)[:, :TEAM_SIZE * _DMG_PER_MON].reshape(1, TEAM_SIZE, _DMG_PER_MON)
    # feature order per mon: [phys_low,high,crit,pko,acc, spec_low,high,crit,pko,acc, outspeed, prov].
    # HP Grass/Ice are SPECIAL → read the spec high-roll (index 6).
    grass_spec = g[0, 0, 6].item()
    ice_spec = i[0, 0, 6].item()
    assert grass_spec > 2.5 * ice_spec > 0.0, (grass_spec, ice_spec)
    # HP is special → the physical channel stays ~0 (no believed physical candidate).
    assert g[0, 0, 1].item() < 1e-3


def test_typed_hp_channel_and_effectiveness_ranking():
    """Grass HP: a Water/Ground defender (4×) outreads a Fire defender (½×) despite Fire's lower SpD —
    effectiveness dominates, and the threat lands on the SPECIAL channel."""
    op, layout = _op_and_layout()
    defenders = [
        (260, _T2I["WATER"], _T2I["GROUND"]),   # Swampert — 4× Grass
        (126, _T2I["FIRE"], 0),                 # Magmar — ½× Grass (resists)
    ] + [(0, 0, 0)] * 4
    grass = [0.0] * 16; grass[_hp_slot("GRASS")] = 1.0
    lg = _logits_hp_only(layout["max_moves"])
    out = op(_fake_ctx(op, defenders=defenders, hp_probs_active=grass,
                       attacker_num=248, attacker_t1=_T2I["NORMAL"], attacker_t2=0),
             lg)[:, :TEAM_SIZE * _DMG_PER_MON].reshape(1, TEAM_SIZE, _DMG_PER_MON)
    assert out[0, 0, 6].item() > out[0, 1, 6].item() > 0.0   # spec high-roll: Water/Ground > Fire
    assert out[0, 0, 1].item() < 1e-3 and out[0, 1, 1].item() < 1e-3   # phys high-roll ~0 (HP is special)


def test_immune_defender_reads_zero():
    """A believed Ground move deals EXACTLY 0 to a Flying defender (the chart 0 falls through the
    product), while a Ground-weak defender reads large — immunity holds at the candidate level. The
    immune channel collapses to the sigmoid(−20) belief tail on other moves (negligible, <1e-6)."""
    op, layout = _op_and_layout()
    from agents import gen3_data
    eq_num = gen3_data.moves.get("earthquake").num
    defenders = [
        (227, _T2I["STEEL"], _T2I["FLYING"]),    # Skarmory — Ground-IMMUNE (Flying)
        (248, _T2I["ROCK"], _T2I["DARK"]),       # Tyranitar — 2× weak to Ground
    ] + [(0, 0, 0)] * 4
    lg = torch.full((1, TEAM_SIZE, layout["max_moves"]), -20.0)
    lg[:, :, eq_num] = 20.0                                                # believe ONLY Earthquake
    out = op(_fake_ctx(op, defenders=defenders, hp_probs_active=[0.0] * 16,
                       attacker_num=248, attacker_t1=_T2I["GROUND"], attacker_t2=0),
             lg)[:, :TEAM_SIZE * _DMG_PER_MON].reshape(1, TEAM_SIZE, _DMG_PER_MON)
    assert out[0, 0, 1].item() < 1e-6                # immune: only the −20 belief tail remains (phys high-roll)
    assert out[0, 1, 1].item() > 0.1                 # Ground-weak: a real physical threat
    assert out[0, 1, 1].item() > 1e6 * out[0, 0, 1].item()


# --------------------------------------------------------------------------- extractor wiring
def test_off_path_projection_dims_unchanged_by_damage_op():
    base, _ = _make_model(attend_unrevealed_opponents=True, move_belief_mode="revealed")
    on, _ = _make_model(attend_unrevealed_opponents=True, move_belief_mode="revealed", damage_op=True)
    assert base.damage_op is None and on.damage_op is not None
    grow = TEAM_SIZE * _DMG_PER_MON + _DMG_EFFECT + _DMG_INCOMING_SEC + _DMG_CB
    assert on.projection_input_dim - base.projection_input_dim == grow
    assert on.value_projection_input_dim - base.value_projection_input_dim == grow


def test_dependency_guard_requires_revealed_or_both():
    for bad in ("off", "unrevealed"):
        with pytest.raises(ValueError, match="damage_op"):
            _make_model(attend_unrevealed_opponents=True, move_belief_mode=bad, damage_op=True)
    # revealed and both are allowed.
    _make_model(attend_unrevealed_opponents=True, move_belief_mode="revealed", damage_op=True)


def test_finite_on_zero_obs_and_block_is_zero():
    """Zero obs → no opp active → the damage block is deterministically ZERO (the dummy-forward path),
    and the full forward is finite."""
    model, layout = _make_model(attend_unrevealed_opponents=True, move_belief_mode="revealed", damage_op=True)
    model.eval()
    with torch.no_grad():
        pi, vf = model.forward({"observation": torch.zeros(3, layout["total_dim"])})
        assert torch.isfinite(pi).all() and torch.isfinite(vf).all()
        ctx = model.unpack({"observation": torch.zeros(3, layout["total_dim"])})
        block = model.damage_op(ctx, model.last_move_belief_logits)
    assert block.shape == (3, TEAM_SIZE * _DMG_PER_MON + _DMG_EFFECT + _DMG_INCOMING_SEC + _DMG_CB)
    assert (block == 0).all()


def test_finite_under_random_fuzz():
    """Random obs + random belief logits (the degenerate-weights case the dummy can't hit) stay finite —
    the soft-max aggregation + clamped denominators have no /0."""
    model, layout = _make_model(attend_unrevealed_opponents=True, move_belief_mode="revealed", damage_op=True)
    model.eval()
    torch.manual_seed(0)
    for _ in range(25):
        with torch.no_grad():
            pi, vf = model.forward({"observation": torch.rand(4, layout["total_dim"])})
        assert torch.isfinite(pi).all() and torch.isfinite(vf).all()


def test_grad_flows_to_move_belief_head():
    """The damage block is differentiable in the move-belief logits → a gradient on it reaches
    move_belief.move_head (the 'the op sharpens the belief toward real KO threats' property)."""
    model, layout = _make_model(attend_unrevealed_opponents=True, move_belief_mode="revealed", damage_op=True)
    model.train()
    obs = {"observation": torch.rand(4, layout["total_dim"])}
    model.forward(obs)
    ctx = model.unpack(obs)
    block = model.damage_op(ctx, model.last_move_belief_logits)
    model.zero_grad()
    block.sum().backward()
    g = model.move_belief.move_head.weight.grad
    assert g is not None and g.abs().sum() > 0


def test_no_opp_active_gates_block_to_zero():
    """With no opponent active flag set, the whole block (and its gradient) is zeroed."""
    op, layout = _op_and_layout()
    ctx = _fake_ctx(op, defenders=[(260, _T2I["WATER"], _T2I["GROUND"])] + [(0, 0, 0)] * 5,
                    hp_probs_active=[0.0] * 16, attacker_num=248,
                    attacker_t1=_T2I["NORMAL"], attacker_t2=0)
    ctx.hp_and_active[:, TEAM_SIZE:, -1] = 0.0      # clear ALL opp active flags
    out = op(ctx, _logits_hp_only(layout["max_moves"]))
    assert (out == 0).all()


def test_accuracy_scalar_and_pko_fold():
    """The per-channel accuracy scalar reports the dominant believed move's base hit rate, and pko folds
    it (pko = acc·P(KO|hit) ≤ acc — the exact realized KO probability). Fire Blast (85%, special) →
    spec_acc≈0.85; a 100%-accurate move → ≈1.0; and pko never exceeds acc."""
    from agents import gen3_data
    op, layout = _op_and_layout()
    defenders = [(0, _T2I["GRASS"], 0)] + [(0, 0, 0)] * 5     # a frail Grass mon Fire Blast OHKOs on hit
    atk = dict(attacker_num=146, attacker_t1=_T2I["FIRE"], attacker_t2=0)   # Moltres (Fire) → STAB

    def _believe(move):
        lg = torch.full((1, TEAM_SIZE, layout["max_moves"]), -20.0)
        lg[:, :, gen3_data.moves.get(move).num] = 20.0
        return lg

    def _run(move):
        return op(_fake_ctx(op, defenders=defenders, hp_probs_active=[0.0] * 16, **atk),
                  _believe(move))[:, :TEAM_SIZE * _DMG_PER_MON].reshape(1, TEAM_SIZE, _DMG_PER_MON)

    fb, surf = _run("fireblast"), _run("surf")
    SPEC_ACC, SPEC_PKO = 9, 8                                # the new spec-channel slots
    assert fb[0, 0, SPEC_ACC].item() == pytest.approx(0.85, abs=0.01)    # Fire Blast 85%
    assert surf[0, 0, SPEC_ACC].item() == pytest.approx(1.0, abs=0.01)   # Surf 100%
    assert fb[0, 0, SPEC_PKO].item() <= fb[0, 0, SPEC_ACC].item() + 1e-6  # pko = acc·ko_hit ≤ acc
    # an 85%-accurate sure-KO reads pko ≈ 0.85 (not 1.0 — accuracy folded); the 100% move reads higher.
    assert surf[0, 0, SPEC_PKO].item() > fb[0, 0, SPEC_PKO].item()


def test_three_roll_relationship():
    """The shared kernel's three rolls are the gen3 roll band: low = 0.85·high, and crit = ×2·high
    (gen3 crit ignores screens → 2× the pre-screen damage; with no screen that is exactly 2× high)."""
    op, _ = _op_and_layout()
    B, n = 1, 1
    atk, spa = torch.tensor([300.0]), torch.tensor([200.0])
    at1 = at2 = torch.zeros(B, dtype=torch.long)
    def_stat, spd_stat = torch.tensor([[200.0]]), torch.tensor([[200.0]])
    maxhp, cur_hp = torch.tensor([[300.0]]), torch.tensor([[300.0]])
    t1d = t2d = torch.zeros(B, n, dtype=torch.long)
    ability1 = torch.zeros(B, n, dtype=torch.long)
    reflect = light = torch.zeros(B, 1)
    bp, mty, phys = torch.tensor([100.0]), torch.tensor([5], dtype=torch.long), torch.tensor([1.0])
    acc = torch.tensor([1.0])                                    # 100%-accurate → pko undiscounted
    fixed = torch.zeros(1)                                       # not a fixed-damage move
    weather = torch.ones(B, 1)                                   # no weather
    high, low, crit, pko, _hcb, _kcb = op._damage_rolls(atk, spa, at1, at2, def_stat, spd_stat, maxhp, cur_hp,
                                            t1d, t2d, ability1, reflect, light, bp, mty, phys, acc, fixed, weather)
    h = high[0, 0, 0].item()
    assert 0.0 < h < 1.5                                          # unclamped (relationship is clean)
    assert low[0, 0, 0].item() == pytest.approx(0.85 * h, rel=1e-5)
    assert crit[0, 0, 0].item() == pytest.approx(2.0 * h, rel=1e-5)
    assert pko[0, 0, 0].item() == 0.0                             # 118 dmg vs 300 HP → no KO


def _fake_ctx_out(*, our_species, our_t1, our_t2, our_moves, our_move_types,
                  opp_species, opp_t1, opp_t2, move_mask, opp_ability=0, B=1):
    """Hand-built ctx for the OUTGOING block: our active in slot 0 (4 moves in request order), opp active
    at slot TEAM_SIZE. Spread = IV31/EV0/neutral; full HP both sides."""
    n = 2 * TEAM_SIZE
    species = torch.zeros(B, n, dtype=torch.long)
    t1 = torch.zeros(B, n, dtype=torch.long); t2 = torch.zeros(B, n, dtype=torch.long)
    ability = torch.zeros(B, n, dtype=torch.long)
    species[:, 0] = our_species; t1[:, 0] = our_t1; t2[:, 0] = our_t2
    species[:, TEAM_SIZE] = opp_species; t1[:, TEAM_SIZE] = opp_t1; t2[:, TEAM_SIZE] = opp_t2
    ability[:, TEAM_SIZE] = opp_ability
    hp_and_active = torch.zeros(B, n, POKEMON_FULL_DIM)
    hp_and_active[:, 0, 0] = 1.0                       # our active full HP
    hp_and_active[:, TEAM_SIZE, 0] = 1.0               # opp active full HP
    hp_and_active[:, TEAM_SIZE, -1] = 1.0              # opp active flag
    pokemon_part = torch.zeros(B, n, POKEMON_FULL_DIM)
    sp = pokemon_part[:, :, POKEMON_SPREAD_OFFSET:POKEMON_SPREAD_OFFSET + 18]
    sp[..., 0:6] = 1.0      # IV 31
    sp[..., 13:18] = 1.0    # neutral nature
    all_move_ids = torch.zeros(B, n, 4, dtype=torch.long)
    all_move_type_ids = torch.zeros(B, n, 4, dtype=torch.long)
    for k, (mid, mty) in enumerate(zip(our_moves, our_move_types)):
        all_move_ids[:, 0, k] = mid
        all_move_type_ids[:, 0, k] = mty
    return types.SimpleNamespace(
        batch_size=B, device=torch.device("cpu"),
        our_active_idx=torch.zeros(B, dtype=torch.long),
        opp_active_local=torch.zeros(B, dtype=torch.long),
        species_ids=species, type1_ids=t1, type2_ids=t2, ability1_ids=ability,
        item_ids=torch.zeros(B, n, dtype=torch.long),           # no item (no Choice Band) by default
        hp_and_active=hp_and_active, pokemon_part=pokemon_part,
        all_move_ids=all_move_ids, all_move_type_ids=all_move_type_ids,
        move_mask=torch.tensor([list(move_mask)] * B, dtype=torch.float32),
        screen_feature=torch.zeros(B, 8),
        our_ctx_raw=torch.zeros(B, ACTIVE_CONTEXT_DIM), opp_ctx_raw=torch.zeros(B, ACTIVE_CONTEXT_DIM),  # boosts++volatiles
        weather_feature=torch.zeros(B, 7),                              # no weather
    )


def test_outgoing_per_move_discriminates_equal_effectiveness():
    """The EQ-vs-equal-effectiveness fix: Earthquake (Ground, 100 BP) and Brick Break (Fighting, 75 BP)
    are BOTH 2× super-effective vs a Rock defender — the obs type-multiplier can't break the tie, but the
    per-move OUTGOING damage must (EQ > Brick Break by base power). Attacker is Normal → no STAB for either,
    isolating BP. Move slot k occupies [k*4 : k*4+4] = [low, high, crit, pko] (request order = action 6+k)."""
    from agents import gen3_data
    mappings = load_mappings(); layout = Gen3ObservationEncoder(mappings).get_layout()
    op = DamageOperator(layout, outgoing=True)
    eq, bb = gen3_data.moves.get("earthquake"), gen3_data.moves.get("brickbreak")
    ctx = _fake_ctx_out(our_species=143, our_t1=_T2I["NORMAL"], our_t2=0,           # Snorlax (Normal)
                        our_moves=[eq.num, bb.num, 0, 0],
                        our_move_types=[_T2I["GROUND"], _T2I["FIGHTING"], 0, 0],
                        opp_species=248, opp_t1=_T2I["ROCK"], opp_t2=0,             # mono-Rock (both 2×)
                        move_mask=[1, 1, 0, 0])
    out = op._outgoing_block(ctx)                                                    # [1, 17]
    eq_high, bb_high = out[0, 1].item(), out[0, 5].item()       # move 0 high, move 1 high
    assert eq_high > bb_high > 0.0, (eq_high, bb_high)           # same 2× eff, EQ wins on base power
    assert out[0, 8:12].abs().sum().item() == 0.0               # empty move slots 2/3 → zero


def test_outgoing_legality_mask_and_immunity():
    """A move the action mask forbids (Choice-lock / Disable / no-PP) is zeroed; a Ground move into a
    Flying (immune) defender is zeroed."""
    from agents import gen3_data
    mappings = load_mappings(); layout = Gen3ObservationEncoder(mappings).get_layout()
    op = DamageOperator(layout, outgoing=True)
    eq, bb = gen3_data.moves.get("earthquake"), gen3_data.moves.get("brickbreak")
    # Brick Break (slot 1) made ILLEGAL by the action mask → its block is zero, EQ (slot 0) still computes.
    ctx = _fake_ctx_out(our_species=143, our_t1=_T2I["NORMAL"], our_t2=0,
                        our_moves=[eq.num, bb.num, 0, 0],
                        our_move_types=[_T2I["GROUND"], _T2I["FIGHTING"], 0, 0],
                        opp_species=248, opp_t1=_T2I["ROCK"], opp_t2=0, move_mask=[1, 0, 0, 0])
    out = op._outgoing_block(ctx)
    assert out[0, 0:4].abs().sum().item() > 0.0                  # EQ legal → computed
    assert out[0, 4:8].abs().sum().item() == 0.0                # Brick Break illegal → zeroed
    # Earthquake (Ground) into Skarmory (Steel/Flying) → immune → zero.
    imm = _fake_ctx_out(our_species=143, our_t1=_T2I["NORMAL"], our_t2=0,
                        our_moves=[eq.num, 0, 0, 0], our_move_types=[_T2I["GROUND"], 0, 0, 0],
                        opp_species=227, opp_t1=_T2I["STEEL"], opp_t2=_T2I["FLYING"], move_mask=[1, 0, 0, 0])
    assert op._outgoing_block(imm)[0, 0:4].abs().sum().item() == 0.0


def test_op_is_leak_free_of_privileged_keys():
    """No-leak gate: the unified op (incoming + outgoing) reads ONLY public obs (via ctx) + the model's own
    predicted belief — never a training-only privileged label. Its output is bit-identical whether or not
    belief_species / belief_moves / known_moves / belief_target_slots are present in the obs dict."""
    # spread_belief=True so the SpreadBelief leg (its reinjection + the op's consumption of last_spread_belief)
    # is exercised by the clean-vs-poisoned bit-identity check too (v25).
    model, layout = _make_model(attend_unrevealed_opponents=True, move_belief_mode="revealed",
                                move_prior_fusion=True, damage_op=True, damage_outgoing=True,
                                spread_belief=True)
    model.eval()
    torch.manual_seed(0)
    obs_t = torch.rand(4, layout["total_dim"])
    with torch.no_grad():
        ctx = model.unpack({"observation": obs_t}); model.forward({"observation": obs_t})
        clean = model.damage_op(ctx, model.last_move_belief_logits).clone()
        poisoned = {"observation": obs_t,
                    "belief_species": torch.rand(4, TEAM_SIZE, layout["max_species"]),
                    "belief_moves": torch.rand(4, TEAM_SIZE, layout["max_moves"]),
                    "known_moves": torch.rand(4, TEAM_SIZE, layout["max_moves"]),
                    "belief_target_slots": torch.rand(4, TEAM_SIZE, 107)}
        ctx2 = model.unpack(poisoned); model.forward(poisoned)
        poisoned_out = model.damage_op(ctx2, model.last_move_belief_logits)
    assert torch.equal(clean, poisoned_out)


def test_decode_damage_block_for_prober():
    """The prober decode exposes the full operator output from the PRE-gain stash: per-mon incoming
    in TEAM-SLOT order (incoming[i] = our team slot i; the active is whichever slot holds the active
    flag, the bench slots are the safe-switch reads), the opp effect scalars, and the outgoing
    per-move block. The single source of truth the TUI mirrors."""
    from agents.model.features_extractor import decode_damage_block
    model, layout = _make_model(attend_unrevealed_opponents=True, move_belief_mode="revealed",
                                move_prior_fusion=True, damage_op=True, damage_outgoing=True)
    model.eval()
    with torch.no_grad():
        model.forward({"observation": torch.rand(2, layout["total_dim"])})
    view = decode_damage_block(model.damage_op.last_raw_block[0], outgoing=True)
    assert len(view["incoming"]) == TEAM_SIZE                       # our 6 team slots (team-slot order)
    assert set(view["incoming"][0]["phys"]) == {"low", "high", "crit", "pko", "acc"}
    assert set(view["effect"]) == {"recovery", "status", "phaze", "boost", "hazard", "protect"}
    assert view["outgoing"] is not None and len(view["outgoing"]["moves"]) == 4
    assert set(view["outgoing"]["moves"][0]) == {"low", "high", "crit", "pko"}
    # an incoming-only model decodes outgoing → None.
    m2, l2 = _make_model(attend_unrevealed_opponents=True, move_belief_mode="revealed",
                         move_prior_fusion=True, damage_op=True)
    m2.eval()
    with torch.no_grad():
        m2.forward({"observation": torch.rand(1, l2["total_dim"])})
    v2 = decode_damage_block(m2.damage_op.last_raw_block[0], outgoing=False)
    assert v2["outgoing"] is None and len(v2["incoming"]) == TEAM_SIZE


# ---------------------------------------------------- gen3_unified_move_system_v1: secondary effects
_SEC_IDX = {c: i for i, c in enumerate(dt.SECONDARY_COLS)}
_INC_SEC_BASE = TEAM_SIZE * _DMG_PER_MON + _DMG_EFFECT          # incoming per-status secondary sub-block
_OUT_SEC_BASE = _DMG_OUT_N_MOVES * _DMG_OUT_PER_MOVE + 1        # outgoing per-move secondary (after p_outspeed)


def _believe_active(op, *names):
    """Belief logits putting ~all mass on `names` (move ids) at the opp active slot."""
    from agents import gen3_data
    lg = torch.full((1, TEAM_SIZE, op.MOVE_BP.shape[0]), -12.0)
    for nm in names:
        lg[:, :, gen3_data.moves.get(nm).num] = 12.0
    return lg


def test_incoming_secondary_status_serene_grace_and_accuracy_fold():
    """The opp active's DAMAGING-move secondaries surface per-status: Body Slam → 30% para; Serene Grace
    DOUBLES it to 60%; Zap Cannon's 100% para × 50% accuracy folds to 50% (accuracy-in-the-op, like pko)."""
    from agents import gen3_data
    op = DamageOperator(_make_layout())
    ctx = _fake_ctx(op, attacker_num=242, attacker_t1=_T2I["NORMAL"], attacker_t2=0,
                    defenders=[(260, _T2I["WATER"], _T2I["GROUND"])] + [(0, 0, 0)] * 5,
                    hp_probs_active=[0.0] * 16)
    par = _INC_SEC_BASE + _SEC_IDX["par"]
    out = op(ctx, _believe_active(op, "bodyslam"))
    assert abs(out[0, par].item() - 0.30) < 1e-4
    ctx.ability1_ids[:, TEAM_SIZE] = gen3_data.abilities.get("serenegrace").num
    out_sg = op(ctx, _believe_active(op, "bodyslam"))
    assert abs(out_sg[0, par].item() - 0.60) < 1e-4                # Serene Grace ×2
    ctx.ability1_ids[:, TEAM_SIZE] = 0
    out_zc = op(ctx, _believe_active(op, "zapcannon"))
    assert abs(out_zc[0, par].item() - 0.50) < 1e-4               # 100% × 50% acc folded
    # flinch is exposed RAW (no speed coupling): Rock Slide 30% × 90% acc = 0.27, regardless of speed.
    out_rs = op(ctx, _believe_active(op, "rockslide"))
    assert abs(out_rs[0, _INC_SEC_BASE + _SEC_IDX["flinch"]].item() - 0.27) < 1e-4


def test_outgoing_secondary_serene_grace_and_shield_dust():
    """Per OUR move: Thunderbolt → 10% para; OUR Serene Grace doubles to 20%; the OPP's Shield Dust
    negates it to 0. Move 0's secondary lives at [_OUT_SEC_BASE : +N_SECONDARY]."""
    from agents import gen3_data
    layout = _make_layout()
    op = DamageOperator(layout, outgoing=True)
    tb = gen3_data.moves.get("thunderbolt").num
    par = _OUT_SEC_BASE + _SEC_IDX["par"]

    def _out(our_ab=0, opp_ab=0):
        ctx = _fake_ctx_out(our_species=135, our_t1=_T2I["ELECTRIC"], our_t2=0,
                            our_moves=[tb, 0, 0, 0], our_move_types=[_T2I["ELECTRIC"], 0, 0, 0],
                            opp_species=260, opp_t1=_T2I["WATER"], opp_t2=_T2I["GROUND"], move_mask=[1, 0, 0, 0])
        ctx.ability1_ids[:, 0] = our_ab
        ctx.ability1_ids[:, TEAM_SIZE] = opp_ab
        return op._outgoing_block(ctx)[0, par].item()

    sg, sd = gen3_data.abilities.get("serenegrace").num, gen3_data.abilities.get("shielddust").num
    assert abs(_out() - 0.10) < 1e-4
    assert abs(_out(our_ab=sg) - 0.20) < 1e-4                     # our Serene Grace ×2
    assert abs(_out(opp_ab=sd) - 0.0) < 1e-4                      # opp Shield Dust negates


def test_decode_includes_secondary_blocks():
    """The prober decode exposes both new sub-blocks: per-status `incoming_secondary` + per-move
    `outgoing.secondary` (keyed by SECONDARY_COLS)."""
    from agents.model.features_extractor import decode_damage_block
    model, layout = _make_model(attend_unrevealed_opponents=True, move_belief_mode="revealed",
                                move_prior_fusion=True, damage_op=True, damage_outgoing=True)
    model.eval()
    with torch.no_grad():
        model.forward({"observation": torch.rand(2, layout["total_dim"])})
    view = decode_damage_block(model.damage_op.last_raw_block[0], outgoing=True)
    assert set(view["incoming_secondary"]) == set(dt.SECONDARY_COLS)
    assert len(view["outgoing"]["secondary"]) == _DMG_OUT_N_MOVES
    assert set(view["outgoing"]["secondary"][0]) == set(dt.SECONDARY_COLS)


def test_gen3_formula_matches_incoming_damage_kernel():
    """The op's SMOOTH (un-floored) gen3 core matches the live floored `incoming_damage.gen3_damage_max`
    within the floor rounding (validates the 42/50/+2/STAB/eff constants are the gen3 formula)."""
    from agents.observation.incoming_damage import gen3_damage_max
    for bp, atk, dfn, stab, eff in [(100, 318, 226, True, 1.0), (95, 350, 200, False, 2.0),
                                    (120, 405, 230, True, 0.5)]:
        floored = gen3_damage_max(bp, atk, dfn, stab=stab, type_eff=eff)
        smooth = (42.0 * bp * atk / dfn / 50.0 + 2.0) * (1.5 if stab else 1.0) * eff
        assert abs(smooth - floored) / max(1.0, floored) < 0.05, (bp, atk, dfn, smooth, floored)


# ---------------------------------------------------- gen3_unified_op_physics_v1: fixed-damage
def _believe_op(op, name):
    from agents import gen3_data
    lg = torch.full((1, TEAM_SIZE, op.MOVE_BP.shape[0]), -12.0)
    lg[:, :, gen3_data.moves.get(name).num] = 12.0
    return lg


def test_fixed_damage_constant_and_ghost_immune():
    """Seismic Toss (Fighting, fixed 100) reads ~100/maxHP on the PHYSICAL channel vs a Normal defender,
    and EXACTLY 0 vs a Ghost (Fighting is immune) — your named Seismic-Toss-into-Ghosts edge. The BP-0
    formula would otherwise read ~0."""
    op = DamageOperator(_make_layout())
    from agents.model.features_extractor import _DMG_IDX_PHYS_HIGH
    # defender slot 0 = Snorlax (Normal); the op reads our team's defenders.
    ctx_normal = _fake_ctx(op, attacker_num=248, attacker_t1=_T2I["ROCK"], attacker_t2=0,
                           defenders=[(143, _T2I["NORMAL"], 0)] + [(0, 0, 0)] * 5, hp_probs_active=[0.0] * 16)
    ctx_ghost = _fake_ctx(op, attacker_num=248, attacker_t1=_T2I["ROCK"], attacker_t2=0,
                          defenders=[(94, _T2I["GHOST"], _T2I["POISON"])] + [(0, 0, 0)] * 5, hp_probs_active=[0.0] * 16)
    lg = _believe_op(op, "seismictoss")
    pm_n = op(ctx_normal, lg)[:, :TEAM_SIZE * _DMG_PER_MON].reshape(1, TEAM_SIZE, _DMG_PER_MON)
    pm_g = op(ctx_ghost, lg)[:, :TEAM_SIZE * _DMG_PER_MON].reshape(1, TEAM_SIZE, _DMG_PER_MON)
    assert pm_n[0, 0, _DMG_IDX_PHYS_HIGH].item() > 0.1          # ~100/maxHP, on the physical channel
    # Ghost is immune to Fighting Seismic Toss → its contribution is 0; the ~1e-5 residual is the belief
    # floor mass on OTHER physical moves, far below the Normal-defender reading.
    assert pm_g[0, 0, _DMG_IDX_PHYS_HIGH].item() < 0.01


def test_op_physics_boosts_burn_weather_para():
    """gen3_unified_op_physics_v1 parity: a +2 opp Atk doubles incoming physical damage; burn halves it;
    rain ×1.5 a Water move and sun ×0.5; opp paralysis raises p_outspeed (we now outspeed)."""
    from agents.model.features_extractor import (
        _DMG_IDX_PHYS_HIGH, _DMG_IDX_SPEC_HIGH, _DMG_IDX_OUTSPEED, _COND_BRN_IDX, _COND_PAR_IDX)
    from agents.observation.constants import POKEMON_CONDITION_OFFSET
    from agents import gen3_data
    op = DamageOperator(_make_layout())
    cond = POKEMON_CONDITION_OFFSET

    def inc(move, idx, **flags):
        ctx = _fake_ctx(op, attacker_num=248, attacker_t1=_T2I["ROCK"], attacker_t2=_T2I["GROUND"],
                        defenders=[(143, _T2I["NORMAL"], 0)] + [(0, 0, 0)] * 5, hp_probs_active=[0.0] * 16)
        if flags.get("atk2"):    ctx.opp_ctx_raw[:, 0] = 2.0 / 6.0                             # opp +2 atk
        if flags.get("burn"):    ctx.pokemon_part[:, TEAM_SIZE, cond + _COND_BRN_IDX] = 1.0    # opp burned
        if flags.get("rain"):    ctx.weather_feature[:, 2] = 1.0
        if flags.get("sun"):     ctx.weather_feature[:, 1] = 1.0
        if flags.get("opp_par"): ctx.pokemon_part[:, TEAM_SIZE, cond + _COND_PAR_IDX] = 1.0
        lg = torch.full((1, TEAM_SIZE, op.MOVE_BP.shape[0]), -10.0)
        lg[:, :, gen3_data.moves.get(move).num] = 10.0
        return op(ctx, lg)[:, :TEAM_SIZE * _DMG_PER_MON].reshape(1, TEAM_SIZE, _DMG_PER_MON)[0, 0, idx].item()

    eq0 = inc("earthquake", _DMG_IDX_PHYS_HIGH)
    assert inc("earthquake", _DMG_IDX_PHYS_HIGH, atk2=True) == pytest.approx(2.0 * eq0, rel=0.02)   # +2 doubles
    assert inc("earthquake", _DMG_IDX_PHYS_HIGH, burn=True) == pytest.approx(0.5 * eq0, rel=0.02)   # burn halves
    s0 = inc("surf", _DMG_IDX_SPEC_HIGH)
    assert inc("surf", _DMG_IDX_SPEC_HIGH, rain=True) == pytest.approx(1.5 * s0, rel=0.02)          # rain ×1.5
    assert inc("surf", _DMG_IDX_SPEC_HIGH, sun=True) == pytest.approx(0.5 * s0, rel=0.02)           # sun ×0.5
    assert inc("earthquake", _DMG_IDX_OUTSPEED, opp_par=True) > inc("earthquake", _DMG_IDX_OUTSPEED) + 0.3


def test_op_modifiers_match_cpu_reference_which_is_showdown_fuzz_validated():
    """gen3_unified_op_physics_v1 VALIDATION: the op's modifier physics (boost / weather / burn /
    paralysis / fixed-damage) MATCHES the CPU `incoming_damage` reference — which is itself bridge-fuzz-
    validated against the REAL Showdown sim (incoming_damage_fuzz_test). So tying the op to it transitively
    validates the op's new physics against Showdown, deterministically + fast."""
    from agents.observation import incoming_damage as cpu
    from agents.model.features_extractor import DamageOperator, _DMG_PARA_SPEED
    from agents.enums import PokemonType
    from agents import gen3_data
    # 1. boost multiplier: op == CPU for EVERY gen3 stage.
    for stage in range(-6, 7):
        assert DamageOperator._boost_mult(torch.tensor([float(stage)])).item() == pytest.approx(
            cpu.boost_mult(stage), rel=1e-6), stage
    # 2. paralysis speed factor identical.
    assert _DMG_PARA_SPEED == cpu._PARA_SPEED
    # 3. fixed-damage buffer == the CPU FIXED_DAMAGE dict (Seismic Toss/Night Shade/Dragon Rage/Sonic Boom).
    fd = dt.build_move_fixed_damage(400)
    for mid, dmg in cpu.FIXED_DAMAGE.items():
        assert fd[gen3_data.moves.get(mid).num].item() == float(dmg), mid
    # 4. burn halves PHYSICAL: CPU gen3_damage_max(burned=True) == ×0.5 of unburned.
    base = cpu.gen3_damage_max(100, 300, 200, stab=False, type_eff=1.0, burned=False)
    burned = cpu.gen3_damage_max(100, 300, 200, stab=False, type_eff=1.0, burned=True)
    assert burned == pytest.approx(0.5 * base, rel=0.02)
    # 5. weather: op _weather_mult == CPU weather_damage_mult for Water/Fire in rain/sun.
    one = torch.ones(1, 1)
    def op_w(weather_idx, is_water, is_fire):
        wf = torch.zeros(1, 7); wf[0, weather_idx] = 1.0
        return DamageOperator._weather_mult(wf, torch.tensor([[float(is_water)]]),
                                            torch.tensor([[float(is_fire)]]))[0, 0].item()
    # weather idx: 1=sun, 2=rain (global_env one-hot).
    assert op_w(2, 1, 0) == pytest.approx(cpu.weather_damage_mult(PokemonType.WATER, "raindance"))   # rain Water 1.5
    assert op_w(2, 0, 1) == pytest.approx(cpu.weather_damage_mult(PokemonType.FIRE, "raindance"))    # rain Fire 0.5
    assert op_w(1, 0, 1) == pytest.approx(cpu.weather_damage_mult(PokemonType.FIRE, "sunnyday"))     # sun Fire 1.5
    assert op_w(1, 1, 0) == pytest.approx(cpu.weather_damage_mult(PokemonType.WATER, "sunnyday"))    # sun Water 0.5


# --------------------------------------------------------------------------- status-landing block
def _status_land(op, *, our_moves, opp_t1, opp_t2, opp_species=248, opp_ability=0,
                 move_mask=(1, 1, 1, 1), opp_active_status_idx=None,
                 bench_sleep=None, bench_sleep_is_rest=False, opp_substitute=False):
    """Run op._status_landing for our 4 status moves vs the opp active. Returns (p_land[4], known[4]).
    Default opp_species 248 = Tyranitar (Sand Stream → NO status-blocking ability prior → a clean
    "no-ability-immunity" baseline; pass opp_species=143 Snorlax to exercise the Immunity prior).
    opp_active_status_idx: set the opp ACTIVE's condition bit (1=BRN..6=TOX) → already-statused.
    bench_sleep: put a BENCH opp mon (slot TEAM_SIZE+1) to sleep; bench_sleep_is_rest flags its source.
    opp_substitute: set the opp active's Substitute volatile (blocks ALL status moves incl. Leech Seed)."""
    from agents import gen3_data
    ctx = _fake_ctx_out(our_species=143, our_t1=_T2I["NORMAL"], our_t2=0,
                        our_moves=[gen3_data.moves.get(m).num for m in our_moves],
                        our_move_types=[0, 0, 0, 0],
                        opp_species=opp_species, opp_t1=opp_t1, opp_t2=opp_t2,
                        move_mask=list(move_mask), opp_ability=opp_ability)
    if opp_active_status_idx is not None:
        ctx.pokemon_part[:, TEAM_SIZE, POKEMON_CONDITION_OFFSET + opp_active_status_idx] = 1.0
    if bench_sleep:
        ctx.pokemon_part[:, TEAM_SIZE + 1, POKEMON_CONDITION_OFFSET + _COND_SLP_IDX] = 1.0
        ctx.pokemon_part[:, TEAM_SIZE + 1, POKEMON_SLEEP_BELIEF_OFFSET] = 1.0 if bench_sleep_is_rest else 0.0
    if opp_substitute:
        ctx.opp_ctx_raw[:, _SUBSTITUTE_CTX_IDX] = 1.0
    out = op._status_landing(ctx)[0]
    assert out.numel() == _DMG_STATUS
    return out[:_DMG_STATUS_N_MOVES], out[_DMG_STATUS_N_MOVES:]


def test_status_landing_type_and_ability_immunity():
    """Toxic/Will-O-Wisp/Thunder Wave/Leech Seed land per the gen3 rules, incl. the NEW Leech Seed Grass
    immunity. Mirrors gen3_mechanics (the single source). Snorlax (unrevealed) reads the Immunity prior."""
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    op = DamageOperator(layout, outgoing=True)
    # Toxic(85% acc) / Will-O-Wisp(75%) / Thunder Wave(100%) / Leech Seed(90%) vs a NORMAL mon (no immunity).
    p, known = _status_land(op, our_moves=["toxic", "willowisp", "thunderwave", "leechseed"],
                            opp_t1=_T2I["NORMAL"], opp_t2=0)
    assert p[0].item() == pytest.approx(0.85, abs=1e-4)    # Toxic
    assert p[1].item() == pytest.approx(0.75, abs=1e-4)    # Will-O-Wisp
    assert p[2].item() == pytest.approx(1.00, abs=1e-4)    # Thunder Wave
    assert p[3].item() == pytest.approx(0.90, abs=1e-4)    # Leech Seed
    assert known.sum().item() == 0.0                       # unrevealed ability, no block → all prior-estimates

    # Type immunity: Toxic vs Steel = 0, Will-O-Wisp vs Fire = 0, Thunder Wave vs Ground = 0, LEECH SEED vs GRASS = 0.
    p, known = _status_land(op, our_moves=["toxic", "willowisp", "thunderwave", "leechseed"],
                            opp_t1=_T2I["STEEL"], opp_t2=_T2I["GROUND"])  # Steel+Ground blocks Toxic & T-Wave
    assert p[0].item() == 0.0 and p[2].item() == 0.0       # Toxic vs Steel, T-Wave vs Ground
    assert known[0].item() == 1.0 and known[2].item() == 1.0  # type-certain blocks are KNOWN
    p, _ = _status_land(op, our_moves=["willowisp", "leechseed", "toxic", "toxic"],
                        opp_t1=_T2I["FIRE"], opp_t2=_T2I["GRASS"])
    assert p[0].item() == 0.0                              # Will-O-Wisp vs Fire
    assert p[1].item() == 0.0                              # LEECH SEED vs GRASS (the new rule)

    # Ability immunity: Toxic vs an UNREVEALED Snorlax (Immunity 0.86) → 0.85·(1−0.86) ≈ 0.119, then REVEALED.
    p, known = _status_land(op, our_moves=["toxic", "toxic", "toxic", "toxic"],
                            opp_t1=_T2I["NORMAL"], opp_t2=0, opp_species=143)  # Snorlax
    assert p[0].item() == pytest.approx(0.85 * (1 - 0.86), abs=0.01)
    assert known[0].item() == 0.0                          # a prior estimate → not known
    imm_num = __import__("agents", fromlist=["gen3_data"]).gen3_data.abilities.get("immunity").num
    p, known = _status_land(op, our_moves=["toxic", "toxic", "toxic", "toxic"],
                            opp_t1=_T2I["NORMAL"], opp_t2=0, opp_species=143, opp_ability=imm_num)
    assert p[0].item() == 0.0 and known[0].item() == 1.0   # revealed Immunity → certain 0


def test_status_landing_already_statused_and_sleep_clause():
    """A major status can't double-apply; Sleep Clause blocks a 2nd inflicted sleep — but a Rest self-sleep
    does NOT consume our cap (the user's rule)."""
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    op = DamageOperator(layout, outgoing=True)
    # Opp active already BURNED → Toxic (major) blocked, but Leech Seed (not a major status) still lands.
    p, known = _status_land(op, our_moves=["toxic", "leechseed", "toxic", "toxic"],
                            opp_t1=_T2I["NORMAL"], opp_t2=0, opp_active_status_idx=1)  # 1 = BRN
    assert p[0].item() == 0.0 and known[0].item() == 1.0   # Toxic blocked (already statused), certain
    assert p[1].item() == pytest.approx(0.90, abs=1e-4)    # Leech Seed unaffected by status

    # Sleep Clause: a BENCH opp mon asleep by OUR move (non-Rest) → Spore can't land a 2nd sleep.
    p, known = _status_land(op, our_moves=["spore", "toxic", "spore", "spore"],
                            opp_t1=_T2I["NORMAL"], opp_t2=0, bench_sleep=True, bench_sleep_is_rest=False)
    assert p[0].item() == 0.0 and known[0].item() == 1.0   # Spore blocked by Sleep Clause
    assert p[1].item() == pytest.approx(0.85, abs=1e-4)    # Toxic (non-sleep) unaffected by Sleep Clause

    # Rest self-sleep does NOT consume our cap → Spore still lands.
    p, _ = _status_land(op, our_moves=["spore", "spore", "spore", "spore"],
                        opp_t1=_T2I["NORMAL"], opp_t2=0, bench_sleep=True, bench_sleep_is_rest=True)
    assert p[0].item() == pytest.approx(1.00, abs=1e-4)    # Rest sleep doesn't trigger Sleep Clause


def test_status_landing_matches_gen3_mechanics_reference():
    """The op's status-landing rules MIRROR the Showdown-fuzz-validated gen3_mechanics (single source):
    spot-check the type-immunity + ability-block tables agree for the OU-relevant cases."""
    from agents import gen3_mechanics as gm
    from agents import gen3_data
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    op = DamageOperator(layout, outgoing=True)
    # every move in STATUS_MOVE_IMMUNITY → the op's per-move type-immune row matches the rule
    for mid, immune_types in gm.STATUS_MOVE_IMMUNITY.items():
        num = gen3_data.moves.get(mid).num
        for pt in immune_types:
            assert op.MOVE_STATUS_TYPE_IMMUNE[num, _T2I[pt.name]].item() == 1.0, (mid, pt)
    # every ability in ABILITY_STATUS_IMMUNITY → the op's revealed-block table marks its category
    for aid, statuses in gm.ABILITY_STATUS_IMMUNITY.items():
        anum = gen3_data.abilities.get(aid).num
        for s in statuses:
            c = dt._STATUS_CAT[s]
            assert op.ABILITY_STATUS_BLOCK[anum, c].item() == 1.0, (aid, s)


def test_status_landing_substitute_blocks_all_status():
    """A Substitute blocks EVERY status move in gen3 — major status AND Leech Seed — and the block is
    CERTAIN (the Sub is a public volatile → known=1)."""
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    op = DamageOperator(layout, outgoing=True)
    p, known = _status_land(op, our_moves=["toxic", "spore", "leechseed", "thunderwave"],
                            opp_t1=_T2I["NORMAL"], opp_t2=0, opp_substitute=True)
    assert p.abs().sum().item() == 0.0                    # all four status moves blocked by the Sub
    assert known.tolist() == [1.0, 1.0, 1.0, 1.0]         # a Sub is public → every block is certain
    # Sanity: WITHOUT the Sub the same moves DO land (so the Sub is what zeroed them).
    p2, _ = _status_land(op, our_moves=["toxic", "spore", "leechseed", "thunderwave"],
                         opp_t1=_T2I["NORMAL"], opp_t2=0, opp_substitute=False)
    assert p2.abs().sum().item() > 0.0


def test_status_landing_not_gated_by_legality():
    """Parity with the masked CPU `status_will_land` (reactive.py request-slot fill): the per-move value is
    the move's INTRINSIC landing probability — the action mask handles legality separately — so a
    Disabled/Choice-locked/no-PP status move still reports p_land>0 (NOT zeroed like the outgoing DAMAGE
    block). The policy never picks the masked action; the feature stays action-aligned and value-stable."""
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    op = DamageOperator(layout, outgoing=True)
    p, _ = _status_land(op, our_moves=["toxic", "spore", "thunderwave", "willowisp"],
                        opp_t1=_T2I["NORMAL"], opp_t2=0, move_mask=(0, 0, 0, 0))  # all "illegal"
    assert p.abs().sum().item() > 0.0                     # still computed (intrinsic landing prob)
    assert p[1].item() == pytest.approx(1.0, abs=1e-4)    # Spore 100% acc, Tyranitar — lands regardless of mask


def test_status_landing_known_bit_on_revealed_nonblocking_ability():
    """`known`=1 once the opp ability is REVEALED even if it does NOT block the status (the value rests on
    confirmed info — a future reveal can't move it), matching the gen3_mechanics status_will_land_known
    routing. Tyranitar's Sand Stream is revealed but blocks no status → Toxic lands at full accuracy, known."""
    from agents import gen3_data
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    op = DamageOperator(layout, outgoing=True)
    sand = gen3_data.abilities.get("sandstream").num
    p, known = _status_land(op, our_moves=["toxic", "thunderwave", "spore", "willowisp"],
                            opp_t1=_T2I["NORMAL"], opp_t2=0, opp_species=248, opp_ability=sand)
    assert p[0].item() == pytest.approx(0.85, abs=1e-4)   # Toxic lands (Sand Stream blocks nothing)
    assert known.tolist() == [1.0, 1.0, 1.0, 1.0]         # ability revealed → all four certain


# --------------------------------------------------------------------------- Choice Band (gen3_unified_choice_band_v1)
def test_choice_band_outgoing_x15_physical_only():
    """Our OWN Choice Band (item known) ×1.5 the PHYSICAL outgoing damage, deterministically; a special move
    is unaffected (CB boosts Atk, not SpA). Move k's high roll sits at outgoing index k*4+1."""
    from agents import gen3_data
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    op = DamageOperator(layout, outgoing=True)
    eq, surf = gen3_data.moves.get("earthquake"), gen3_data.moves.get("surf")     # physical / special
    base = dict(our_species=143, our_t1=_T2I["NORMAL"], our_t2=0,                 # Snorlax (no STAB on either)
                our_moves=[eq.num, surf.num, 0, 0],
                our_move_types=[_T2I["GROUND"], _T2I["WATER"], 0, 0],
                opp_species=143, opp_t1=_T2I["NORMAL"], opp_t2=0, move_mask=[1, 1, 0, 0])  # bulky neutral → no clamp
    ctx, ctx_cb = _fake_ctx_out(**base), _fake_ctx_out(**base)
    ctx_cb.item_ids[:, 0] = dt.CHOICE_BAND_ITEM_NUM                               # our active holds Choice Band
    out, out_cb = op._outgoing_block(ctx), op._outgoing_block(ctx_cb)
    assert out[0, 1].item() > 0.0
    # ×1.5 at the STAT level → slightly under 1.5× the damage (the +2 floor in core=k*A+2 isn't boosted).
    assert out[0, 1].item() < out_cb[0, 1].item() == pytest.approx(1.5 * out[0, 1].item(), rel=2e-2)
    assert out_cb[0, 5].item() == pytest.approx(out[0, 5].item(), rel=1e-5)         # Surf (spec) unchanged


def test_choice_band_incoming_conditional_tail_and_pcb():
    """Incoming: the op exposes the CB-CONDITIONAL physical tail (phys_high_cb ≈ 1.5× modal phys_high, P(OHKO|CB)
    ≥ modal pko) + a shared p_cb, DECORRELATED so the head weights them. p_cb = the species usage prior for an
    unrevealed item, collapsing to 1.0 (revealed CB) / 0.0 (revealed other)."""
    from agents import gen3_data
    from agents.model.features_extractor import decode_damage_block
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    op = DamageOperator(layout, outgoing=False)
    de = gen3_data.moves.get("doubleedge")                                        # a physical threat
    defenders = [(143, _T2I["NORMAL"], 0)] + [(0, 0, 0)] * 5                       # our Snorlax active (bulky)
    lg = torch.full((1, TEAM_SIZE, layout["max_moves"]), -10.0); lg[:, :, de.num] = 10.0

    ctx = _fake_ctx(op, attacker_num=142, attacker_t1=_T2I["ROCK"], attacker_t2=_T2I["FLYING"],  # Aerodactyl
                    defenders=defenders, hp_probs_active=[0.0] * 16)
    op(ctx, lg)
    dec = decode_damage_block(op.last_raw_block[0], outgoing=False)
    cb, modal = dec["choice_band"], dec["incoming"][0]["phys"]
    assert modal["high"] < cb["phys_high_cb"][0] == pytest.approx(1.5 * modal["high"], rel=2e-2)  # CB-cond high ≈1.5×
    assert cb["phys_pko_cb"][0] >= modal["pko"]                                    # CB only raises KO odds
    assert cb["p_cb"] == pytest.approx(0.761, abs=0.03)                            # Aerodactyl prior (unrevealed)

    # revealed Choice Band → p_cb collapses to 1.0; revealed Leftovers → 0.0.
    ctx.item_ids[:, TEAM_SIZE] = dt.CHOICE_BAND_ITEM_NUM
    op(ctx, lg)
    assert decode_damage_block(op.last_raw_block[0], outgoing=False)["choice_band"]["p_cb"] == pytest.approx(1.0)
    ctx.item_ids[:, TEAM_SIZE] = 234   # leftovers
    op(ctx, lg)
    assert decode_damage_block(op.last_raw_block[0], outgoing=False)["choice_band"]["p_cb"] == pytest.approx(0.0)


def test_choice_band_prior_buffer_values():
    """The SPECIES_CB_PRIOR buffer matches the Smogon item usage prior (non-persistent, zero new params)."""
    from agents import gen3_data
    op = DamageOperator(Gen3ObservationEncoder(load_mappings()).get_layout())
    aero = gen3_data.species.get("aerodactyl").num
    assert op.SPECIES_CB_PRIOR[aero].item() == pytest.approx(0.761, abs=0.02)
    assert "SPECIES_CB_PRIOR" not in dict(op.state_dict())          # non-persistent buffer


def test_choice_band_incoming_fixed_damage_invariant():
    """A fixed-damage move (Seismic Toss = 100) is CB-INVARIANT: the CB-conditional physical high equals the
    modal physical high (CB scales Atk, fixed damage ignores Atk). Pins the override-vs-CB-scale ordering."""
    from agents import gen3_data
    from agents.model.features_extractor import decode_damage_block
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    op = DamageOperator(layout, outgoing=False)
    st = gen3_data.moves.get("seismictoss")                                       # Fighting (phys), fixed 100
    defenders = [(143, _T2I["NORMAL"], 0)] + [(0, 0, 0)] * 5                       # our Snorlax (not Ghost → hit)
    lg = torch.full((1, TEAM_SIZE, layout["max_moves"]), -10.0); lg[:, :, st.num] = 10.0
    ctx = _fake_ctx(op, attacker_num=68, attacker_t1=_T2I["FIGHTING"], attacker_t2=0,  # Machamp
                    defenders=defenders, hp_probs_active=[0.0] * 16)
    op(ctx, lg)
    dec = decode_damage_block(op.last_raw_block[0], outgoing=False)
    assert dec["incoming"][0]["phys"]["high"] > 0.0                               # Seismic Toss lands
    assert dec["choice_band"]["phys_high_cb"][0] == pytest.approx(dec["incoming"][0]["phys"]["high"], rel=1e-5)


def test_choice_band_incoming_special_channel_excluded():
    """The CB-conditional tail is PHYSICAL-only — a special believed move (Surf) contributes ~0 to phys_high_cb
    (CB boosts Atk, not SpA), even though it has a large special modal threat."""
    from agents import gen3_data
    from agents.model.features_extractor import decode_damage_block
    layout = Gen3ObservationEncoder(load_mappings()).get_layout()
    op = DamageOperator(layout, outgoing=False)
    surf = gen3_data.moves.get("surf")                                            # Water → special
    defenders = [(143, _T2I["NORMAL"], 0)] + [(0, 0, 0)] * 5
    lg = torch.full((1, TEAM_SIZE, layout["max_moves"]), -10.0); lg[:, :, surf.num] = 10.0
    ctx = _fake_ctx(op, attacker_num=130, attacker_t1=_T2I["WATER"], attacker_t2=_T2I["FLYING"],  # Gyarados
                    defenders=defenders, hp_probs_active=[0.0] * 16)
    op(ctx, lg)
    dec = decode_damage_block(op.last_raw_block[0], outgoing=False)
    assert dec["incoming"][0]["spec"]["high"] > 0.1                               # Surf IS a real special threat
    # the CB physical tail is negligible — only the ~sigmoid(-10) belief floor on other physical moves, far
    # below the special threat (CB boosts Atk, not the believed Surf).
    assert dec["choice_band"]["phys_high_cb"][0] < 0.01
    assert dec["choice_band"]["phys_high_cb"][0] < dec["incoming"][0]["spec"]["high"]
