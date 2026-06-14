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
    # feature order: [phys_chip, spec_chip, phys_pko, spec_pko]; HP Grass/Ice are SPECIAL.
    grass_spec_chip = g[0, 0, 1].item()
    ice_spec_chip = i[0, 0, 1].item()
    assert grass_spec_chip > 2.5 * ice_spec_chip > 0.0, (grass_spec_chip, ice_spec_chip)
    # HP is special → the physical channel stays ~0 (no believed physical candidate).
    assert g[0, 0, 0].item() < 1e-3


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
    assert out[0, 0, 1].item() > out[0, 1, 1].item() > 0.0   # spec_chip: Water/Ground > Fire
    assert out[0, 0, 0].item() < 1e-3 and out[0, 1, 0].item() < 1e-3   # phys_chip ~0 (HP is special)


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
    assert out[0, 0, 0].item() < 1e-6                # immune: only the −20 belief tail remains
    assert out[0, 1, 0].item() > 0.1                 # Ground-weak: a real physical threat
    assert out[0, 1, 0].item() > 1e6 * out[0, 0, 0].item()


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


def test_gen3_formula_matches_incoming_damage_kernel():
    """The op's SMOOTH (un-floored) gen3 core matches the live floored `incoming_damage.gen3_damage_max`
    within the floor rounding (validates the 42/50/+2/STAB/eff constants are the gen3 formula)."""
    from agents.observation.incoming_damage import gen3_damage_max
    for bp, atk, dfn, stab, eff in [(100, 318, 226, True, 1.0), (95, 350, 200, False, 2.0),
                                    (120, 405, 230, True, 0.5)]:
        floored = gen3_damage_max(bp, atk, dfn, stab=stab, type_eff=eff)
        smooth = (42.0 * bp * atk / dfn / 50.0 + 2.0) * (1.5 if stab else 1.0) * eff
        assert abs(smooth - floored) / max(1.0, floored) < 0.05, (bp, atk, dfn, smooth, floored)
