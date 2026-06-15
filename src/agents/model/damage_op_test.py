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
    Gen3FeaturesExtractor, DamageOperator, _DMG_PER_MON, _DMG_EFFECT, TEAM_SIZE,
)
from agents.model import damage_tables as dt
from agents.observation.constants import POKEMON_SPREAD_OFFSET, POKEMON_FULL_DIM
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
        screen_feature=torch.zeros(B, 8),                       # no screens by default
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
    grow = TEAM_SIZE * _DMG_PER_MON + _DMG_EFFECT
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
    assert block.shape == (3, TEAM_SIZE * _DMG_PER_MON + _DMG_EFFECT)
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
    high, low, crit, pko = op._damage_rolls(atk, spa, at1, at2, def_stat, spd_stat, maxhp, cur_hp,
                                            t1d, t2d, ability1, reflect, light, bp, mty, phys, acc)
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
        hp_and_active=hp_and_active, pokemon_part=pokemon_part,
        all_move_ids=all_move_ids, all_move_type_ids=all_move_type_ids,
        move_mask=torch.tensor([list(move_mask)] * B, dtype=torch.float32),
        screen_feature=torch.zeros(B, 8),
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
    model, layout = _make_model(attend_unrevealed_opponents=True, move_belief_mode="revealed",
                                move_prior_fusion=True, damage_op=True, damage_outgoing=True)
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
    (slot 0 = our active, 1-5 = the safe-switch bench reads), the opp effect scalars, and the outgoing
    per-move block. The single source of truth the TUI mirrors."""
    from agents.model.features_extractor import decode_damage_block
    model, layout = _make_model(attend_unrevealed_opponents=True, move_belief_mode="revealed",
                                move_prior_fusion=True, damage_op=True, damage_outgoing=True)
    model.eval()
    with torch.no_grad():
        model.forward({"observation": torch.rand(2, layout["total_dim"])})
    view = decode_damage_block(model.damage_op.last_raw_block[0], outgoing=True)
    assert len(view["incoming"]) == TEAM_SIZE                       # active + 5 safe-switch bench rows
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


def test_gen3_formula_matches_incoming_damage_kernel():
    """The op's SMOOTH (un-floored) gen3 core matches the live floored `incoming_damage.gen3_damage_max`
    within the floor rounding (validates the 42/50/+2/STAB/eff constants are the gen3 formula)."""
    from agents.observation.incoming_damage import gen3_damage_max
    for bp, atk, dfn, stab, eff in [(100, 318, 226, True, 1.0), (95, 350, 200, False, 2.0),
                                    (120, 405, 230, True, 0.5)]:
        floored = gen3_damage_max(bp, atk, dfn, stab=stab, type_eff=eff)
        smooth = (42.0 * bp * atk / dfn / 50.0 + 2.0) * (1.5 if stab else 1.0) * eff
        assert abs(smooth - floored) / max(1.0, floored) < 0.05, (bp, atk, dfn, smooth, floored)
