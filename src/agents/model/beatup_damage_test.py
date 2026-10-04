"""gen3_beatup_exact_v1 — the DamageOperator models gen-3 Beat Up EXACTLY (it was priced as one 10-BP Dark
special hit that ran the user's SpA against the target's SpD: ~1% of a Blissey, where the real move is ~80%).

The mechanic, verified in `deps/pokemon-showdown/data/mods/gen3/moves.ts` `beatup` (and cross-checked against
the real sim by `beatup_sim_parity_test.py` + `measurements/beatup_golden_2026-10-03/`):

  * TYPELESS ('???') and SPECIAL — no STAB, no effectiveness (Ghost / Steel / Dark / Psychic targets read 1x),
    Light Screen halves it, Reflect and a burned user do not;
  * one hit per ally in `side.pokemon.filter(a => !a.fainted && !a.status)` (the USER counts iff healthy);
  * hit i = the gen-3 formula at BP 10 with A = ally i's species BASE Atk and D = the TARGET's species BASE Def
    (`event.modifier = 1`: no stat stage, item or ability reaches either).

So, with the op's constants (`0.925` mean roll, smooth formula): per hit `(42/50)*10*A_i/D + 2`, summed.
Every expected value below is computed BY HAND from the stated base stats, never from the op.
"""
import pytest
import torch

from agents import gen3_data
from agents.model import damage_kinds as dk
from agents.model import damage_tables as dt
from agents.model.damage_op_layout import (_DMG_CHIP_CAP, _DMG_IDX_SPEC_HIGH, _DMG_OAX_PER_MOVE,
                                           _DMG_PER_MON)
from agents.model.damage_op_test import _believe_op, _fake_ctx, _fake_ctx_out, _make_layout
from agents.model.features_extractor import DamageOperator, TEAM_SIZE
from agents.observation.constants import POKEMON_CONDITION_OFFSET
from agents.observation.types import TypeEncoder

_T2I = TypeEncoder.TYPE_TO_IDX
_BEATUP = gen3_data.moves.get("beatup").num
_STATUS_COL = {"brn": 1, "par": 2, "slp": 3, "frz": 4, "psn": 5, "tox": 6}


def _num(species: str) -> int:
    return gen3_data.species.get(species).num


def _base(species: str, stat: str) -> int:
    return int(gen3_data.species.get(species).base_stats[stat])


# The six-mon party the tests use. Atks are PINNED below so a data change fails loudly, not silently.
_PARTY = ["umbreon", "tyranitar", "salamence", "snorlax", "metagross", "gengar"]
_PARTY_ATK = [65, 134, 135, 110, 135, 65]
_BLISSEY_MAXHP = 2.0 * 255 + 31.0 + 110.0                  # the op's neutral 0-EV estimate: 651


def test_party_base_atk_is_pinned():
    assert [_base(s, "atk") for s in _PARTY] == _PARTY_ATK
    assert _base("blissey", "def") == 10 and _base("blissey", "spd") == 135   # the anti-Blissey move's whole point


def _expect_mean(atks, target_def: float) -> float:
    """Expected damage in HP at the op's 0.925 mean roll: sum_i [ (42/50)*10*A_i/D + 2 ] * 0.925."""
    return 0.925 * sum((42.0 / 50.0) * 10.0 * a / target_def + 2.0 for a in atks)


@pytest.fixture(scope="module")
def op():
    return DamageOperator(_make_layout(), outgoing=True)


def _out_ctx(target: str, *, t1: str, t2: str = "", party=None, hp=None, status=None, user: str = "umbreon"):
    """OUR party in slots 0..5 (slot 0 = the Beat Up user), a REVEALED opp active `target` at slot 6."""
    party = list(party or _PARTY)
    ctx = _fake_ctx_out(our_species=_num(user), our_t1=_T2I["DARK"], our_t2=0,
                        our_moves=[_BEATUP, 0, 0, 0], our_move_types=[_T2I["DARK"], 0, 0, 0],   # the obs: DEX type
                        opp_species=_num(target), opp_t1=_T2I[t1], opp_t2=_T2I[t2] if t2 else 0,
                        move_mask=[1, 0, 0, 0])
    for i, sp in enumerate(party):
        ctx.species_ids[:, i] = _num(sp)
        ctx.hp_and_active[:, i, 0] = 1.0 if hp is None or i not in hp else hp[i]
    for i, st in (status or {}).items():
        ctx.pokemon_part[:, i, POKEMON_CONDITION_OFFSET + _STATUS_COL[st]] = 1.0
    ctx.opp_believed_mask = torch.zeros(1, TEAM_SIZE, dtype=torch.bool)     # the opp active is revealed
    return ctx


def _blocks(op, ctx):
    """[low, high, crit, pko] of move slot 0 into the opp active from ALL THREE outgoing kernels — they must
    agree — and the matrix cell's type multiplier."""
    blk = op._outgoing_block(ctx)[0, 0:4]
    omx = op._outgoing_matrix(ctx)[0, 0:5]                                 # move 0 x opp slot 0 [low,high,crit,pko,mult]
    oax = op._outgoing_attacker_matrix(ctx)[0, 0:_DMG_OAX_PER_MOVE]        # our slot 0 x move 0
    for other in (omx[:4], oax):
        assert torch.allclose(other, blk, atol=1e-5), (blk, other)
    return blk, omx[4].item()


def _hp(op, ctx, target: str):
    blk, mult = _blocks(op, ctx)
    maxhp = 2.0 * _base(target, "hp") + 31.0 + 110.0
    return blk[1].item() * maxhp, blk[0].item() * maxhp, blk[2].item() * maxhp, mult


# ------------------------------------------------------------------------------------------- the table
def test_table_row_flags_and_typeless_type(op):
    kind, value, cite = dt.DAMAGE_MODELS["beatup"]
    assert kind == "beatup_party" and value == 10.0 and "onFoeModifySpD" in cite
    assert float(op.MOVE_BEATUP[_BEATUP]) == 1.0 and float(op.MOVE_BEATUP.sum()) == 1.0
    assert float(op.MOVE_BP[_BEATUP]) == 10.0                              # the per-hit BP, kept
    assert int(op.MOVE_TYPE_IDX[_BEATUP]) == _T2I["???"]                   # typeless in battle
    assert float(op.MOVE_PHYS[_BEATUP]) == 0.0                             # category Special
    assert float(op.MOVE_NONFORMULA[_BEATUP]) == 0.0                       # still the formula, not a replaced roll


def test_check_damage_models_refuses_a_bp_that_contradicts_the_dex():
    models = dict(dt.DAMAGE_MODELS)
    models["beatup"] = ("beatup_party", 40.0, "x")
    with pytest.raises(ValueError, match="declares 40.0 BP per hit, the dex has 10"):
        dt.check_damage_models(models)


# ------------------------------------------------------------------------ OUTGOING: the Blissey example
def test_outgoing_into_blissey_is_the_hand_computed_sum(op):
    """6 healthy allies, base Atk [65,134,135,110,135,65] (sum 644), Blissey base Def 10 -> 511.49 HP at the
    mean roll (78.6% of its 651), 0.85x at the low roll, 2x at a crit. The old one-hit SpA-vs-SpD price was 8.7."""
    high, low, crit, mult = _hp(op, _out_ctx("blissey", t1="NORMAL"), "blissey")
    want = _expect_mean(_PARTY_ATK, 10.0)
    assert want == pytest.approx(511.488, abs=1e-3)
    assert high == pytest.approx(want, rel=1e-4)
    assert low == pytest.approx(0.85 * want, rel=1e-4) and crit == pytest.approx(2.0 * want, rel=1e-4)
    assert mult == 1.0
    assert high > 50 * 8.7                                                  # the approximation it replaces


def test_the_dealt_damage_does_not_depend_on_the_users_own_attack_stats(op):
    """A Beat Up user's own SpA / Atk are irrelevant: the lead swapped for a 5-Atk 300-SpA mon changes only the
    party sum (the lead's BASE Atk), by exactly its base-Atk difference."""
    base = _hp(op, _out_ctx("blissey", t1="NORMAL"), "blissey")[0]
    swapped = ["alakazam"] + _PARTY[1:]                                       # base Atk 50, base SpA 135
    got = _hp(op, _out_ctx("blissey", t1="NORMAL", party=swapped, user="alakazam"), "blissey")[0]
    assert got == pytest.approx(_expect_mean([_base("alakazam", "atk")] + _PARTY_ATK[1:], 10.0), rel=1e-4)
    assert got < base


# ------------------------------------------------------------------------------ who is an eligible ally
def test_fainted_and_statused_allies_are_excluded(op):
    ok = lambda skip: [a for i, a in enumerate(_PARTY_ATK) if i not in skip]
    # ally 3 fainted (HP 0): 5 hits
    h, *_ = _hp(op, _out_ctx("blissey", t1="NORMAL", hp={3: 0.0}), "blissey")
    assert h == pytest.approx(_expect_mean(ok({3}), 10.0), rel=1e-4)
    # every major status excludes (ally 2 each in turn), alone
    for st in _STATUS_COL:
        h, *_ = _hp(op, _out_ctx("blissey", t1="NORMAL", status={2: st}), "blissey")
        assert h == pytest.approx(_expect_mean(ok({2}), 10.0), rel=1e-4), st
    # fainted + two statused: 3 hits
    h, *_ = _hp(op, _out_ctx("blissey", t1="NORMAL", hp={5: 0.0}, status={1: "slp", 4: "par"}), "blissey")
    assert h == pytest.approx(_expect_mean(ok({1, 4, 5}), 10.0), rel=1e-4)


def test_a_statused_user_does_not_count_itself_and_no_allies_means_no_damage(op):
    h, *_ = _hp(op, _out_ctx("blissey", t1="NORMAL", status={0: "brn"}), "blissey")      # the USER is burned
    assert h == pytest.approx(_expect_mean(_PARTY_ATK[1:], 10.0), rel=1e-4)
    none = _out_ctx("blissey", t1="NORMAL", status={i: "psn" for i in range(6)})
    assert _hp(op, none, "blissey")[:3] == (0.0, 0.0, 0.0)                    # multihit 0: the move deals nothing
    assert op._outgoing_block(none)[0, 3].item() == 0.0                       # and cannot KO


# ----------------------------------------------------------------------------------------- typelessness
@pytest.mark.parametrize("target,t1,t2", [
    ("gengar", "GHOST", "POISON"),        # Dark 2x under the dex type
    ("skarmory", "STEEL", "FLYING"),      # Dark 0.5x under the dex type
    ("tyranitar", "ROCK", "DARK"),        # Dark 0.5x again, and a DARK defender
    ("alakazam", "PSYCHIC", ""),          # Dark 2x
    ("snorlax", "NORMAL", ""),
])
def test_typeless_no_effectiveness_and_no_stab_into_any_type(op, target, t1, t2):
    high, low, crit, mult = _hp(op, _out_ctx(target, t1=t1, t2=t2), target)
    assert mult == 1.0
    assert high == pytest.approx(_expect_mean(_PARTY_ATK, float(_base(target, "def"))), rel=1e-4)
    assert low == pytest.approx(0.85 * high, rel=1e-4)       # the user is a DARK mon: STAB would have been 1.5x


# -------------------------------------------------- what reaches the damage and what does not (all by sim)
def test_light_screen_and_stat_stages(op):
    base = _hp(op, _out_ctx("blissey", t1="NORMAL"), "blissey")
    # LIGHT SCREEN on the target's side halves it (special); a crit ignores the screen.
    ls = _out_ctx("blissey", t1="NORMAL")
    ls.screen_feature[:, 3] = 1.0
    h, lo, cr, _m = _hp(op, ls, "blissey")
    want = _expect_mean(_PARTY_ATK, 10.0)
    assert h == pytest.approx(0.5 * want, rel=1e-4) and lo == pytest.approx(0.5 * 0.85 * want, rel=1e-4)
    assert cr == pytest.approx(2.0 * want, rel=1e-4)                        # a crit ignores the screen
    # REFLECT does not.
    rf = _out_ctx("blissey", t1="NORMAL")
    rf.screen_feature[:, 1] = 1.0
    assert _hp(op, rf, "blissey")[0] == pytest.approx(base[0], rel=1e-6)
    # the user's +6 Atk / +6 SpA and the TARGET's +6 Def / +6 SpD stages change nothing (`event.modifier = 1`)
    boosted = _out_ctx("blissey", t1="NORMAL")
    boosted.our_ctx_raw[:, 0] = 1.0       # +6 Atk  (positive half of the first boost pair)
    boosted.our_ctx_raw[:, 4] = 1.0       # +6 SpA
    boosted.opp_ctx_raw[:, 2] = 1.0       # the target's +6 Def
    boosted.opp_ctx_raw[:, 6] = 1.0       # the target's +6 SpD
    assert _hp(op, boosted, "blissey")[0] == pytest.approx(base[0], rel=1e-6)


# ---------------------------------------------------------------------- an UNREVEALED defender (O2 only)
def test_unrevealed_defender_prices_the_expected_damage_exactly(op):
    """A hidden target's Def is unknown: the matrix's cell must be E[damage] under the Species-Clause usage
    prior — computed here as the explicit sum over species, not through the op's harmonic-mean shortcut."""
    ctx = _out_ctx("blissey", t1="NORMAL")
    ctx.opp_believed_mask = torch.ones(1, TEAM_SIZE, dtype=torch.bool)       # every opp slot hidden
    p = op.unrevealed_species_probs(ctx)[0]                                  # [S] the prior the op uses
    want_hp, e_maxhp = 0.0, 0.0
    for s in torch.nonzero(p > 0).flatten().tolist():
        d = float(op.BASE_STATS[s, 2])
        want_hp += float(p[s]) * _expect_mean(_PARTY_ATK, d)
        e_maxhp += float(p[s]) * (2.0 * float(op.BASE_STATS[s, 0]) + 31.0 + 110.0)
    cell = op._outgoing_matrix(ctx)[0, 0:5]                                   # move 0 x opp slot 0 (hidden)
    assert cell[1].item() * e_maxhp == pytest.approx(want_hp, rel=2e-3)
    assert cell[4].item() == pytest.approx(1.0, abs=1e-6)                     # typeless: the E[mult] column is 1
    assert cell[3].item() == 0.0                                              # P(KO) stays NULLED at a hidden slot


# ------------------------------------------------------------------------------- the opponent's party
def _incoming_ctx(op, *, opp_party, opp_hp=None, opp_status=None, hidden=(), our_def="blissey"):
    """The opp active (slot 6) is `opp_party[0]`; its bench follows. Our slot 0 is the defender. Revealed
    opp slots are full HP; `hidden` slots are unrevealed (species 0, HP encoded 0)."""
    ctx = _fake_ctx(op, attacker_num=_num(opp_party[0]), attacker_t1=_T2I["DARK"], attacker_t2=0,
                    defenders=[(_num(our_def), _T2I["NORMAL"], 0)] + [(0, 0, 0)] * 5,
                    hp_probs_active=[0.0] * 16)
    believed = torch.zeros(1, TEAM_SIZE, dtype=torch.bool)
    for j, sp in enumerate(opp_party):
        slot = TEAM_SIZE + j
        if j in hidden:
            believed[0, j] = True
            ctx.species_ids[:, slot] = 0
            ctx.hp_and_active[:, slot, 0] = 0.0
            continue
        ctx.species_ids[:, slot] = _num(sp)
        ctx.hp_and_active[:, slot, 0] = 1.0 if opp_hp is None or j not in opp_hp else opp_hp[j]
    for j, st in (opp_status or {}).items():
        ctx.pokemon_part[:, TEAM_SIZE + j, POKEMON_CONDITION_OFFSET + _STATUS_COL[st]] = 1.0
    ctx.opp_believed_mask = believed
    return ctx


def _incoming_spec_high(op, ctx) -> float:
    pm = op(ctx, _believe_op(op, "beatup"))[:, :TEAM_SIZE * _DMG_PER_MON].reshape(1, TEAM_SIZE, _DMG_PER_MON)
    return pm[0, 0, _DMG_IDX_SPEC_HIGH].item() * _DMG_CHIP_CAP * _BLISSEY_MAXHP


def test_incoming_agrees_with_outgoing_on_the_same_configuration(op):
    """THEIR Beat Up into OUR Blissey, the same six-mon party on their side, the same healthy / statused
    pattern: the incoming block reads exactly what the outgoing block reads for ours."""
    out = _hp(op, _out_ctx("blissey", t1="NORMAL"), "blissey")[0]
    inc = _incoming_spec_high(op, _incoming_ctx(op, opp_party=_PARTY))
    assert inc == pytest.approx(out, rel=1e-4)
    # an ineligible pattern: ally 3 fainted, ally 2 asleep, the user (slot 0) paralysed
    skip = {0, 2, 3}
    out2 = _hp(op, _out_ctx("blissey", t1="NORMAL", hp={3: 0.0}, status={2: "slp", 0: "par"}), "blissey")[0]
    inc2 = _incoming_spec_high(op, _incoming_ctx(op, opp_party=_PARTY, opp_hp={3: 0.0},
                                                 opp_status={2: "slp", 0: "par"}))
    want = _expect_mean([a for i, a in enumerate(_PARTY_ATK) if i not in skip], 10.0)
    assert out2 == pytest.approx(want, rel=1e-4) and inc2 == pytest.approx(want, rel=1e-4)


def test_incoming_is_typeless_and_ignores_our_stat_stages_and_the_believed_spread(op):
    ctx = _incoming_ctx(op, opp_party=_PARTY, our_def="gengar")             # a Ghost: Dark would be 2x
    ctx.type1_ids[:, 0], ctx.type2_ids[:, 0] = _T2I["GHOST"], _T2I["POISON"]
    gengar_maxhp = 2.0 * _base("gengar", "hp") + 31.0 + 110.0
    pm = op(ctx, _believe_op(op, "beatup"))[:, :TEAM_SIZE * _DMG_PER_MON].reshape(1, TEAM_SIZE, _DMG_PER_MON)
    got = pm[0, 0, _DMG_IDX_SPEC_HIGH].item() * _DMG_CHIP_CAP * gengar_maxhp
    assert got == pytest.approx(_expect_mean(_PARTY_ATK, float(_base("gengar", "def"))), rel=1e-4)
    ctx.our_ctx_raw[:, 2] = 1.0        # our active's +6 Def ... and +6 SpD (Beat Up reads neither)
    ctx.our_ctx_raw[:, 6] = 1.0
    pm2 = op(ctx, _believe_op(op, "beatup"))[:, :TEAM_SIZE * _DMG_PER_MON].reshape(1, TEAM_SIZE, _DMG_PER_MON)
    assert pm2[0, 0, _DMG_IDX_SPEC_HIGH].item() == pytest.approx(pm[0, 0, _DMG_IDX_SPEC_HIGH].item(), rel=1e-6)


def test_light_screen_halves_their_beat_up_reflect_does_not(op):
    want = _want_incoming_hp()
    ls = _incoming_ctx(op, opp_party=_PARTY)
    ls.screen_feature[:, 2] = 1.0                                           # OUR side's Light Screen
    assert _incoming_spec_high(op, ls) == pytest.approx(0.5 * want, rel=1e-4)
    rf = _incoming_ctx(op, opp_party=_PARTY)
    rf.screen_feature[:, 0] = 1.0                                           # OUR side's Reflect
    assert _incoming_spec_high(op, rf) == pytest.approx(want, rel=1e-4)


def test_hidden_opp_slots_are_certainly_eligible_at_their_expected_base_atk(op):
    """The convention: a revealed mon counts iff alive + unstatused; an UNREVEALED slot counts WITH CERTAINTY
    (it never entered the battle) at E[base Atk] under the Species-Clause usage prior. Its HP encodes 0 — the
    discriminating case is a revealed FAINTED mon (excluded) next to a hidden slot (included)."""
    ctx = _incoming_ctx(op, opp_party=_PARTY, hidden=(3, 4, 5), opp_hp={2: 0.0})   # slots 3-5 hidden, 2 fainted
    S, N = dk.beatup_party_opp(op, ctx)
    p = op.unrevealed_species_probs(ctx)[0]
    e_atk = sum(float(p[s]) * float(op.BASE_STATS[s, 1]) for s in torch.nonzero(p > 0).flatten().tolist())
    assert 40.0 < e_atk < 140.0                                              # a real base-Atk mean, not a sentinel 0
    assert N.item() == 2 + 3                                                 # slots 0,1 revealed-eligible + 3 hidden
    assert S.item() == pytest.approx(_PARTY_ATK[0] + _PARTY_ATK[1] + 3 * e_atk, rel=1e-5)
    # and the op's incoming Beat Up reads the same total
    got = _incoming_spec_high(op, ctx)
    want = 0.925 * ((42.0 / 50.0) * 10.0 * S.item() / 10.0 + 2.0 * N.item())
    assert got == pytest.approx(want, rel=1e-4)


def test_hidden_attackers_follow_the_t0_species_belief_when_one_is_handed_in(op):
    """The extractor hands every pricing site the SAME T0 species belief (`t0_species_probs`); Beat Up's hidden
    allies read it, team-level `[B,S]` or per-slot `[B,6,S]`, instead of the static prior."""
    ctx = _incoming_ctx(op, opp_party=_PARTY, hidden=(3, 4, 5))              # slots 0-2 revealed + healthy
    n_species = op.SPECIES_USAGE_PRIOR.shape[0]
    team = torch.zeros(1, n_species)
    team[0, _num("blissey")] = 1.0                                           # every hidden mon IS a Blissey (Atk 10)
    S, N = dk.beatup_party_opp(op, ctx, team)
    assert N.item() == 6 and S.item() == pytest.approx(sum(_PARTY_ATK[:3]) + 3 * 10)
    per_slot = torch.zeros(1, TEAM_SIZE, n_species)
    for slot, sp in ((3, "blissey"), (4, "tyranitar"), (5, "salamence")):
        per_slot[0, slot, _num(sp)] = 1.0
    S2, N2 = dk.beatup_party_opp(op, ctx, per_slot)
    assert N2.item() == 6 and S2.item() == pytest.approx(sum(_PARTY_ATK[:3]) + 10 + 134 + 135)
    # and the op's forward prices exactly that belief
    pm = op(ctx, _believe_op(op, "beatup"), species_probs=team)[:, :TEAM_SIZE * _DMG_PER_MON]
    got = pm.reshape(1, TEAM_SIZE, _DMG_PER_MON)[0, 0, _DMG_IDX_SPEC_HIGH].item() * _DMG_CHIP_CAP * _BLISSEY_MAXHP
    assert got == pytest.approx(_expect_mean(_PARTY_ATK[:3] + [10, 10, 10], 10.0), rel=1e-4)


# ----------------------------------------------- the pairwise / refine kernels carry it too (d3 c1b c2 c3 d4)
def _beatup_logits(op, B=1):
    return _believe_op(op, "beatup")


def _their_party_ctx(op, defender: str = "blissey"):
    """Opp's six revealed healthy (the attacker side) vs OUR Blissey active in slot 0."""
    return _incoming_ctx(op, opp_party=_PARTY, our_def=defender)


def _want_incoming_hp(defender: str = "blissey", atks=None) -> float:
    return _expect_mean(atks or _PARTY_ATK, float(_base(defender, "def")))


def test_d3_refine_kernel(op):
    ctx = _their_party_ctx(op)
    high, ko, eff, phys_k, w_topk, alive, has_opp = op._incoming_rolls(ctx, _beatup_logits(op))
    j = int(torch.argmax(w_topk[0]))
    assert eff[0, 0, j].item() == 1.0 and phys_k[0, j].item() == 0.0
    assert high[0, 0, j].item() * _BLISSEY_MAXHP == pytest.approx(_want_incoming_hp(), rel=1e-4)


def test_d4_bench_incoming_kernel(op):
    ctx = _their_party_ctx(op)
    cells = op.pairwise_bench_incoming(ctx, _beatup_logits(op).expand(1, TEAM_SIZE, -1).contiguous(), k_bench=3)
    # cells [B, our i, opp j, (phys_high, spec_high, phys_pko, spec_pko)] — j = 1..5 are the revealed bench
    got = cells[0, 0, 1:, 1] * _BLISSEY_MAXHP
    assert got.tolist() == pytest.approx([_want_incoming_hp()] * 5, rel=1e-3)


def test_c3_recovery_kernel_prices_the_ko_ramp_from_the_summed_damage(op):
    """d_in_pko = -P(KO) before the heal when the heal fully removes the threat; P(KO) = clamp((dmg - hp) /
    (0.15 dmg)) is the op's ramp, so it pins the SUMMED damage."""
    dmg = _want_incoming_hp()                                                # 511.49 HP
    hp_frac = (dmg * (1.0 - 0.075)) / _BLISSEY_MAXHP                          # the ramp's midpoint: P(KO) = 0.5
    ctx = _their_party_ctx(op)
    ctx.hp_and_active[:, 0, 0] = hp_frac
    ctx.our_active_req_move_ids = torch.tensor([[gen3_data.moves.get("softboiled").num, 0, 0, 0]])
    cells = op.pairwise_recovery(ctx, _beatup_logits(op).expand(1, TEAM_SIZE, -1).contiguous(), k_cand=3)
    # cells [B,4 our slot,6 opp j,(is_rec, d_in_pko, rest)] ; j = 0 is the opp active (revealed + alive)
    assert cells[0, 0, 0, 0].item() == 1.0
    p_ko_before = -cells[0, 0, 0, 1].item()
    ramp = (dmg - hp_frac * _BLISSEY_MAXHP) / (0.15 * dmg)
    assert p_ko_before == pytest.approx(ramp, rel=2e-3) and p_ko_before == pytest.approx(0.5, abs=1e-3)


def test_c2_status_consequence_kernel_sleep_leg(op):
    """Landing Spore removes their whole believed threat: d_in_all = -(the Beat Up high roll)."""
    ctx = _their_party_ctx(op)
    ctx.our_active_req_move_ids = torch.tensor([[gen3_data.moves.get("spore").num, 0, 0, 0]])
    cells = op.pairwise_status_consequence(ctx, _beatup_logits(op).expand(1, TEAM_SIZE, -1).contiguous(), k_cand=3)
    # cells [B,4,6,(is_status, land, d_outspeed, d_in_phys, d_sched, d_in_all, e_slp_free)]
    d_in_all = cells[0, 0, 0, 5].item()
    assert -d_in_all * _BLISSEY_MAXHP == pytest.approx(_want_incoming_hp(), rel=1e-3)


def test_c1b_a_defensive_setup_move_does_not_move_beat_up(op):
    """Beat Up's Def is the species BASE Def (`event.modifier = 1`): +2 Def (Iron Defense) AND +2 SpD (Amnesia)
    change it by EXACTLY 0 — the old special-hit price shrank under +SpD — where controls (Body Slam under
    +Def, Surf under +SpD) shrink."""
    logits = _beatup_logits(op).expand(1, TEAM_SIZE, -1).contiguous()
    for setup in ("irondefense", "amnesia"):
        ctx = _their_party_ctx(op, "snorlax")
        ctx.our_active_req_move_ids = torch.tensor([[gen3_data.moves.get(setup).num, 0, 0, 0]])
        cells = op.pairwise_boost_incoming(ctx, logits, k_cand=3)             # [B,4,6,(d_in_high, d_in_pko)]
        assert cells[0, 0, 0, 0].item() == 0.0 and cells[0, 0, 0, 1].item() == 0.0, setup
    for setup, control in (("irondefense", "bodyslam"), ("amnesia", "surf")):
        ctx = _their_party_ctx(op, "snorlax")
        ctx.our_active_req_move_ids = torch.tensor([[gen3_data.moves.get(setup).num, 0, 0, 0]])
        ctrl = op.pairwise_boost_incoming(ctx, _believe_op(op, control).expand(1, TEAM_SIZE, -1).contiguous(),
                                          k_cand=3)
        assert ctrl[0, 0, 0, 0].item() < 0.0, (setup, control)                # the control DOES move


def test_the_status_operands_kernel_prices_the_same_sum():
    """`pointer_intent_status_operands` runs `_damage_rolls` on a [B,K] seat axis vs OUR active only."""
    op2 = DamageOperator(_make_layout(), outgoing=True, matrices_incoming=True, topk_k=3)
    from agents.model.features_extractor import MOVE_LATENT_DIM
    ctx = _their_party_ctx(op2)
    ctx.all_move_ids = torch.zeros(1, 2 * TEAM_SIZE, 4, dtype=torch.long)        # the matrix's meaningful-K gate
    ctx.our_active_req_move_ids = torch.tensor([[gen3_data.moves.get("spore").num, 0, 0, 0]])
    logits = _believe_op(op2, "beatup")
    op2(ctx, logits, move_latent_all=torch.zeros(op2.MOVE_BP.shape[0], MOVE_LATENT_DIM))
    _base_cells, _d_burn, d_slp, _ib, _is = op2.pointer_intent_status_operands(ctx, logits, k_cand=3)
    k = int(torch.argmax(op2.last_topk_w[0]))
    assert int(op2.last_topk_idx[0, k]) == _BEATUP
    assert -d_slp[0, k].item() * _BLISSEY_MAXHP == pytest.approx(_want_incoming_hp(), rel=1e-3)
