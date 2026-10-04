"""gen3_nonformula_damage_v1 — the DamageOperator prices every move whose damage is NOT the BP formula
(X5 Tier 0 finding F8: OUR Seismic Toss / Night Shade / Dragon Rage / Sonic Boom read UNUSABLE in every
outgoing block, and Return / Super Fang / Endeavor / Flail / … read 0 on both sides).

Pins, each against the gen-3 mechanic verified in `deps/pokemon-showdown` (citations in
`damage_tables.DAMAGE_MODELS`):
  * OUTGOING (all three blocks): Seismic Toss / Night Shade deal exactly the level, 100 HP; Dragon Rage
    40; Super Fang half the CURRENT HP; Endeavor `target − attacker`; an OHKO the whole current bar at
    its 30% accuracy; Return priced (102 BP, physical); type immunity still zeroes them; an
    `unmodelled` row reads 0 by declaration.
  * INCOMING: Super Fang and Return, believed by the opponent, are priced on the physical channel.
  * The HP-dependent BP kinds: the gen-3 Flail table (incl. the exact-quotient boundary) and Eruption ∝ HP.
  * The THROWING guard: a damaging BP-0 move with no declared model RAISES at table build.
"""
import pytest
import torch

from agents import gen3_data
from agents.gen3_data.moves import MoveData
from agents.model import damage_tables as dt
from agents.model.features_extractor import DamageOperator, TEAM_SIZE, _DMG_PER_MON
from agents.model.damage_op_layout import _DMG_CHIP_CAP, _DMG_IDX_PHYS_HIGH, _DMG_OAX_PER_MOVE
from agents.model.damage_op_test import _fake_ctx, _fake_ctx_out, _make_layout, _believe_op
from agents.observation.types import TypeEncoder

_T2I = TypeEncoder.TYPE_TO_IDX
_BLISSEY, _SNORLAX, _SKARMORY, _GENGAR = 242, 143, 227, 94
_N = {m: gen3_data.moves.get(m).num for m in (
    "seismictoss", "nightshade", "dragonrage", "superfang", "endeavor", "fissure", "return",
    "bodyslam", "counter", "flail", "eruption")}
_TY = {m: _T2I[gen3_data.moves.get(m).type.name] for m in _N}


@pytest.fixture(scope="module")
def op():
    return DamageOperator(_make_layout(), outgoing=True)


def _neutral_maxhp(species: int) -> float:
    """The op's estimate of an OPPONENT's max HP (neutral 0 EV, IV 31)."""
    return 2.0 * gen3_data.species.get(
        next(s for s in gen3_data.species.base_form_ids() if gen3_data.species.get(s).num == species)
    ).base_stats["hp"] + 31.0 + 110.0


def _ctx(move: str, opp_species: int, opp_t1: str, opp_t2: str = "", *, our_species: int = _BLISSEY,
         opp_hp: float = 1.0, our_hp: float = 1.0):
    ctx = _fake_ctx_out(our_species=our_species, our_t1=_T2I["NORMAL"], our_t2=0,
                        our_moves=[_N[move], 0, 0, 0], our_move_types=[_TY[move], 0, 0, 0],
                        opp_species=opp_species, opp_t1=_T2I[opp_t1], opp_t2=_T2I[opp_t2] if opp_t2 else 0,
                        move_mask=[1, 0, 0, 0])
    ctx.hp_and_active[:, TEAM_SIZE, 0] = opp_hp
    ctx.hp_and_active[:, 0, 0] = our_hp
    ctx.opp_believed_mask = torch.zeros(1, TEAM_SIZE, dtype=torch.bool)   # the opp active is revealed
    return ctx


def _out_hp(op, ctx, opp_species: int):
    """(high, low, crit, pko) of OUR move slot 0 into the opp active, high/low/crit in HP — read from
    ALL THREE outgoing blocks, which must agree."""
    maxhp = _neutral_maxhp(opp_species)
    blk = op._outgoing_block(ctx)[0, 0:4]                                  # [low, high, crit, pko]
    omx = op._outgoing_matrix(ctx)[0, 0:5]                                 # move 0 × opp slot 0
    oax = op._outgoing_attacker_matrix(ctx)[0, 0:_DMG_OAX_PER_MOVE]        # our slot 0 × move 0
    for other in (omx[:4], oax):
        assert torch.allclose(other, blk, atol=1e-5), (blk, other)
    return blk[1].item() * maxhp, blk[0].item() * maxhp, blk[2].item() * maxhp, blk[3].item()


@pytest.mark.parametrize("move,opp,t1,t2", [
    ("seismictoss", _SNORLAX, "NORMAL", ""),           # Fighting vs Normal is 2x — irrelevant to fixed damage
    ("nightshade", _SKARMORY, "STEEL", "FLYING"),      # Ghost vs Steel is 0.5x — irrelevant too
])
def test_outgoing_level_damage_is_exactly_100(op, move, opp, t1, t2):
    high, low, crit, pko = _out_hp(op, _ctx(move, opp, t1, t2), opp)
    assert high == pytest.approx(100.0, abs=1e-3)
    assert low == pytest.approx(100.0, abs=1e-3) and crit == pytest.approx(100.0, abs=1e-3)   # no roll, no crit
    assert pko == 0.0                                                         # 100 < a full Snorlax / Skarmory


def test_outgoing_dragon_rage_is_40(op):
    high, _l, _c, _k = _out_hp(op, _ctx("dragonrage", _SNORLAX, "NORMAL"), _SNORLAX)
    assert high == pytest.approx(40.0, abs=1e-3)


def test_outgoing_fixed_damage_ko_and_immunity(op):
    maxhp = _neutral_maxhp(_SNORLAX)
    _h, _l, _c, pko = _out_hp(op, _ctx("seismictoss", _SNORLAX, "NORMAL", opp_hp=90.0 / maxhp), _SNORLAX)
    assert pko == pytest.approx(1.0)                                          # 100 covers the last 90 HP, acc 100
    gh = _out_hp(op, _ctx("seismictoss", _GENGAR, "GHOST", "POISON"), _GENGAR)
    assert gh == (0.0, 0.0, 0.0, 0.0)                                         # Fighting into a Ghost
    ns = _out_hp(op, _ctx("nightshade", _SNORLAX, "NORMAL"), _SNORLAX)
    assert ns == (0.0, 0.0, 0.0, 0.0)                                         # Ghost into a Normal


def test_outgoing_super_fang_is_half_current_hp(op):
    maxhp = _neutral_maxhp(_SNORLAX)
    high, _l, _c, pko = _out_hp(op, _ctx("superfang", _SNORLAX, "NORMAL", opp_hp=0.6), _SNORLAX)
    assert high == pytest.approx(0.5 * 0.6 * maxhp, abs=1e-2)
    assert pko == 0.0


def test_outgoing_endeavor_is_target_minus_attacker(op):
    our_maxhp = 2.0 * 255 + 31.0 + 110.0                                      # Blissey, IV 31 / EV 0
    high, _l, _c, pko = _out_hp(op, _ctx("endeavor", _SNORLAX, "NORMAL", our_hp=0.25), _SNORLAX)
    assert high == pytest.approx(_neutral_maxhp(_SNORLAX) - 0.25 * our_maxhp, abs=1e-2)
    assert pko == 0.0
    hi_full, *_ = _out_hp(op, _ctx("endeavor", _SNORLAX, "NORMAL", our_hp=1.0), _SNORLAX)
    assert hi_full == 0.0                                                     # 651 HP Blissey ≥ Snorlax: it fails


def test_outgoing_ohko_takes_the_current_bar_at_its_accuracy(op):
    maxhp = _neutral_maxhp(_SNORLAX)
    high, _l, _c, pko = _out_hp(op, _ctx("fissure", _SNORLAX, "NORMAL", opp_hp=0.5), _SNORLAX)
    assert high == pytest.approx(0.5 * maxhp, abs=1e-2)
    assert pko == pytest.approx(0.30)                                         # gen-3 OHKO accuracy 30
    assert _out_hp(op, _ctx("fissure", _SKARMORY, "STEEL", "FLYING"), _SKARMORY) == (0.0, 0.0, 0.0, 0.0)


def test_outgoing_return_is_priced_physical_102(op):
    ret, *_ = _out_hp(op, _ctx("return", _SKARMORY, "STEEL", "FLYING", our_species=_SNORLAX), _SKARMORY)
    bs, *_ = _out_hp(op, _ctx("bodyslam", _SKARMORY, "STEEL", "FLYING", our_species=_SNORLAX), _SKARMORY)
    assert ret > bs > 0.0                                                     # 102 BP > Body Slam's 85
    assert float(op.MOVE_BP[_N["return"]]) == 102.0
    assert float(op.MOVE_PHYS[_N["return"]]) == 1.0                           # Normal = the PHYSICAL channel


def test_outgoing_unmodelled_reads_zero_by_declaration(op):
    assert dt.DAMAGE_MODELS["counter"][0] == "unmodelled"
    assert _out_hp(op, _ctx("counter", _SNORLAX, "NORMAL"), _SNORLAX) == (0.0, 0.0, 0.0, 0.0)


def _incoming_phys_high(op, move: str, our_hp: float = 1.0) -> float:
    ctx = _fake_ctx(op, attacker_num=128, attacker_t1=_T2I["NORMAL"], attacker_t2=0,       # Tauros
                    defenders=[(_SNORLAX, _T2I["NORMAL"], 0)] + [(0, 0, 0)] * 5, hp_probs_active=[0.0] * 16)
    ctx.hp_and_active[:, 0, 0] = our_hp
    pm = op(ctx, _believe_op(op, move))[:, :TEAM_SIZE * _DMG_PER_MON].reshape(1, TEAM_SIZE, _DMG_PER_MON)
    return pm[0, 0, _DMG_IDX_PHYS_HIGH].item() * _DMG_CHIP_CAP          # the per-mon block is cap-normalised


def test_incoming_super_fang_and_return_are_priced(op):
    assert _incoming_phys_high(op, "superfang", our_hp=0.5) == pytest.approx(0.25, abs=1e-3)
    assert _incoming_phys_high(op, "superfang", our_hp=1.0) == pytest.approx(0.5, abs=1e-3)
    assert _incoming_phys_high(op, "return") > 0.1


def test_flail_table_and_eruption_scale():
    from agents.model.damage_kinds import effective_bp, flail_bp
    f = torch.tensor([1.0, 0.5, 0.3, 0.15, 0.08, 0.02, 0.0])
    assert flail_bp(f).tolist() == [20.0, 40.0, 80.0, 100.0, 150.0, 200.0, 200.0]
    # exact-quotient boundary: 33 / 48 HP → ratio exactly 33 → the 20-BP band (Showdown integer floor)
    assert flail_bp(torch.tensor([33.0]) / torch.tensor([48.0])).item() == 20.0
    one, zero = torch.ones(1), torch.zeros(1)
    assert effective_bp(torch.tensor([150.0]), one, zero, torch.tensor([0.5])).item() == 75.0
    assert effective_bp(torch.tensor([20.0]), zero, one, torch.tensor([0.08])).item() == 150.0


def test_guard_passes_on_the_real_dex():
    dt.check_damage_models()
    # every damaging BP-0 move of the dex is declared — the class enumeration is COMPLETE
    undeclared = [m for m in gen3_data.moves.raw()
                  if gen3_data.moves.get(m).is_dex_damaging and gen3_data.moves.get(m).base_power == 0
                  and m not in dt.DAMAGE_MODELS]
    assert undeclared == []


def test_guard_raises_on_a_dropped_row():
    models = {k: v for k, v in dt.DAMAGE_MODELS.items() if k != "seismictoss"}
    with pytest.raises(ValueError, match="seismictoss"):
        dt.check_damage_models(models)


def test_guard_raises_on_a_planted_bp0_damaging_move(monkeypatch):
    planted = MoveData(id="plantedtoss", num=999, base_power=0, type=gen3_data.moves.get("seismictoss").type,
                       category=gen3_data.moves.get("seismictoss").category, accuracy=100, never_miss=False,
                       has_secondary=False, has_recoil=False, dex_category="Physical")
    real_raw, real_get = gen3_data.moves.raw, gen3_data.moves.get
    monkeypatch.setattr(gen3_data.moves, "raw", lambda: {**real_raw(), "plantedtoss": {}})
    monkeypatch.setattr(gen3_data.moves, "get", lambda m: planted if m == "plantedtoss" else real_get(m))
    with pytest.raises(ValueError, match="plantedtoss"):
        dt.build_nonformula_tables(400)


def test_guard_refuses_data_without_the_dex_category():
    md = MoveData(id="x", num=1, base_power=0, type=gen3_data.moves.get("tackle").type,
                  category=gen3_data.moves.get("tackle").category, accuracy=100, never_miss=False,
                  has_secondary=False, has_recoil=False)
    with pytest.raises(ValueError, match="no dex category"):
        _ = md.is_dex_damaging
