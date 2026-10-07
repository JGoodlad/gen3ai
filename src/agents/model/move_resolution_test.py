"""gen3_move_resolution_v1 (v141, `--move-resolution`) — the move-resolution family, mechanic by mechanic.

Every gen-3 rule the family applies gets a CONSTRUCTED state here and an exact expected value, so reverting the
rule fails its test (`designs/endstate/design_arch_audit.md` §9; each rule's Showdown source line is cited in
`move_resolution_rules.py`). The states are hand-built `MoveResolutionOps` over the REAL move / flag tables, so a
test reads one fact in isolation: the family's arithmetic, not the operator's physics (which is the op's own
tests' subject). The end-to-end gathers from a real extractor forward are in `move_resolution_extractor_test.py`.
"""
from __future__ import annotations

from typing import Any, Dict, Sequence

import pytest
import torch

from agents import gen3_data
from agents.model.move_resolution import MoveResolutionOps, move_facts, switch_facts
from agents.model.move_resolution_rules import (MOVE_RESOLUTION_MOVE_COORDS, MOVE_RESOLUTION_MOVE_IDX as MI,
                                                MOVE_RESOLUTION_SWITCH_COORDS, MOVE_RESOLUTION_SWITCH_IDX as SI,
                                                P_FULL_PARA, P_THAW)
from agents.model.move_resolution_tables import (C_FRZ, C_PAR, C_SLP, NAMED_ABILITIES, S_REFLECT_OURS, S_SG_OPP,
                                                 S_SG_OURS, build_move_tables)
from agents.observation.gen3_effects import VOLATILE_SLOTS
from agents.observation.types import TypeEncoder

N_MOVES = 400
TABLES = build_move_tables(N_MOVES)
KIND_T, FLAG_T, SEAT_T = TABLES["KIND"], TABLES["FLAG"], TABLES["SEAT_KIND"]
VOL = {n: i for i, n in enumerate(VOLATILE_SLOTS)}
T = TypeEncoder.TYPE_TO_IDX
K = 3


def num(mid: str) -> int:
    return int(gen3_data.moves.get(mid).num)


def _move(mid: str) -> Dict[str, Any]:
    md = gen3_data.moves.get(mid)
    raw = gen3_data.moves.raw()[mid]
    acc = raw.get("accuracy")
    return {"num": int(md.num), "type": T[str(raw["type"]).upper()], "prio": float(md.priority),
            "acc": 1.0 if acc is True else float(acc) / 100.0,
            "dmg": float(int(raw.get("basePower") or 0) > 0 or mid in ("seismictoss", "nightshade")),
            "phys": float(str(raw.get("category")) == "Physical")}


def ops(moves: Sequence[str] = ("bodyslam",), seats: Sequence[str] = ("growl", "growl", "growl"),
        alpha: Sequence[float] = (0.3, 0.3, 0.3), a_switch: float = 0.1, **over: Any) -> MoveResolutionOps:
    """A neutral one-row state: our move(s) in the request slots, their K=3 believed seats, nobody hurt."""
    ms = [_move(m) for m in moves] + [None] * (4 - len(moves))
    ss = [_move(s) for s in seats]

    def col(key: str, default: float = 0.0) -> torch.Tensor:
        return torch.tensor([[m[key] if m else default for m in ms]])

    req = torch.tensor([[m["num"] if m else 0 for m in ms]])
    inflicts = torch.tensor([[float(m is not None and (gen3_data.moves.get(n).status_inflicted is not None
                                                       or n == "leechseed")) for m, n in
                              zip(ms, list(moves) + [""] * (4 - len(moves)))]])
    cat_of = {"par": 1, "brn": 2, "frz": 3, "slp": 4, "psn": 5, "tox": 5}
    st_cat = torch.tensor([[(6 if n == "leechseed" else cat_of.get(str(gen3_data.moves.get(n).status_inflicted
                                                                          or "").lower(), 0))
                            for n in moves] + [0] * (4 - len(moves))], dtype=torch.long)
    sb = torch.zeros(1, 4, 5)
    order = ("atk", "def", "spa", "spd", "spe")
    for i, n in enumerate(moves):
        for stat, st in (gen3_data.moves.get(n).self_boosts or ()):
            if stat in order:
                sb[0, i, order.index(stat)] = float(st)
    snums = torch.tensor([[s["num"] for s in ss]])
    base = dict(
        gate=torch.ones(1, 1), req_ids=req, req_type=col("type").long(), is_dmg=col("dmg"), acc=col("acc", 1.0),
        prio=col("prio"), inflicts=inflicts, p_land=inflicts * col("acc", 1.0), st_type_imm=torch.zeros(1, 4, 19),
        st_cat=st_cat, st_blocked=inflicts.clone(), self_boost=sb,
        sec_flinch=torch.zeros(1, 4),
        our_stage=torch.zeros(1, 5), our_hp=torch.tensor([1.0]), our_cond=torch.zeros(1, 6, 7),
        our_alive=torch.ones(1, 6), our_rest=torch.zeros(1, 6), our_active=torch.tensor([0]),
        our_p_wake=torch.tensor([0.0]), our_vol=torch.zeros(1, len(VOLATILE_SLOTS)),
        our_protect_odds=torch.tensor([1.0]), our_named_abl=torch.zeros(1, len(NAMED_ABILITIES)),
        our_is_ghost=torch.zeros(1, 6), our_hp_all=torch.ones(1, 6), beatup_n=torch.tensor([6.0]),
        our_wish=torch.tensor([0.0]),
        opp_active=torch.tensor([0]), opp_vol=torch.zeros(1, len(VOLATILE_SLOTS)), opp_hp=torch.tensor([1.0]),
        opp_protect_odds=torch.tensor([1.0]), opp_last_move=torch.tensor([num("tackle")]),
        opp_p_wake=torch.tensor([0.0]),
        opp_alive_total=torch.tensor([6.0]), opp_cond=torch.zeros(1, 6, 7), opp_rest=torch.zeros(1, 6),
        opp_alive=torch.ones(1, 6), imm_dmg=torch.zeros(1, 6, 19), chart0=torch.zeros(1, 6, 19),
        p_type=torch.zeros(1, 6, 19), abl_block=torch.zeros(1, 6, 7),
        opp_named_abl=torch.zeros(1, 6, len(NAMED_ABILITIES)), screens=torch.zeros(1, 8),
        weather=torch.zeros(1, 7), spikes=torch.zeros(1, 2),
        alpha=torch.tensor([list(alpha)]), a_switch=torch.tensor([[a_switch]]),
        beta=torch.tensor([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]), seat_nums=snums,
        seat_prio=torch.tensor([[s["prio"] for s in ss]]), seat_kind=SEAT_T[snums], seat_flag=FLAG_T[snums],
        seat_phys=torch.tensor([[s["phys"] for s in ss]]), seat_flinch=torch.zeros(1, K),
        pair_in=torch.zeros(1, 6, K, 14), pair_gate=torch.ones(1, 6, 1), pair_type_mult=torch.ones(1, 6, K),
        p_out=torch.tensor([[0.5]]), out_cells=torch.zeros(1, 4, 6, 5), c2_base=torch.zeros(1, 4, 4),
        d_burn_k=torch.zeros(1, K), d_slp_k=torch.zeros(1, K), is_brn=torch.zeros(1, 4), is_slp=torch.zeros(1, 4))
    base.update(over)
    return MoveResolutionOps(**base)


def mf(o: MoveResolutionOps, coord: str = "p_resolve", slot: int = 0) -> float:
    return float(move_facts(o, KIND_T, FLAG_T)[0, slot, MI[coord]])


def sf(o: MoveResolutionOps, coord: str, j: int = 0) -> float:
    return float(switch_facts(o)[0, j, SI[coord]])


def with_vol(side: str, name: str, value: float = 1.0) -> Dict[str, torch.Tensor]:
    v = torch.zeros(1, len(VOLATILE_SLOTS))
    v[0, VOL[name]] = value
    return {f"{side}_vol": v}


def cond(slot: int, col_: int, *, n: int = 6) -> torch.Tensor:
    c = torch.zeros(1, n, 7)
    c[0, slot, col_] = 1.0
    return c


def hit_grid(high: float = 0.3, ko: float = 0.0, acc: float = 1.0, seat: int = 0, j: int = 0) -> torch.Tensor:
    p = torch.zeros(1, 6, K, 14)
    p[0, j, seat, 1] = high
    p[0, j, seat, 0] = 0.85 * high
    p[0, j, seat, 3] = ko
    p[0, j, seat, 4] = acc
    return p


# ------------------------------------------------------------------------------------- the contract
def test_the_coordinate_tables_are_the_single_spelling_of_the_widths():
    from agents.model.arch_constants import _MOVE_RESOLUTION_MOVE_RAW, _MOVE_RESOLUTION_SWITCH_RAW
    assert len(MOVE_RESOLUTION_MOVE_COORDS) == _MOVE_RESOLUTION_MOVE_RAW
    assert len(MOVE_RESOLUTION_SWITCH_COORDS) == _MOVE_RESOLUTION_SWITCH_RAW
    assert move_facts(ops(), KIND_T, FLAG_T).shape == (1, 4, _MOVE_RESOLUTION_MOVE_RAW)
    assert switch_facts(ops()).shape == (1, 6, _MOVE_RESOLUTION_SWITCH_RAW)


def test_the_dropped_judgments_are_not_coordinates():
    """The owner's ruling: tempo_cost, wasted_ko, neutralization and the hand thresholds are GONE."""
    names = set(MOVE_RESOLUTION_MOVE_COORDS) | set(MOVE_RESOLUTION_SWITCH_COORDS)
    for judgment in ("tempo_cost", "wasted_ko", "neutralization", "spin_value_lost", "spin_denied",
                     "p_sub_broken", "p_fp_broken", "e_pko_acc"):
        assert judgment not in names


def test_a_clean_attack_resolves_with_certainty():
    assert mf(ops(("bodyslam",))) == pytest.approx(1.0)
    assert mf(ops(("tackle",))) == pytest.approx(0.95)                 # gen-3 Tackle is 95 % accurate


def test_an_empty_request_slot_reads_zero():
    assert move_facts(ops(("tackle",)), KIND_T, FLAG_T)[0, 1].abs().sum() == 0


# ------------------------------------------------------------------------- the owner's named facts
def test_yawn_fails_on_a_statused_or_already_drowsy_target():
    assert mf(ops(("yawn",))) == pytest.approx(1.0)
    assert mf(ops(("yawn",), opp_cond=cond(0, C_PAR), a_switch=0.0)) == 0.0
    assert mf(ops(("yawn",), **with_vol("opp", "yawn"), a_switch=0.0)) == 0.0
    # a switching target is a FRESH arrival: Yawn lands on it
    assert mf(ops(("yawn",), opp_cond=cond(0, C_PAR)), "p_lands_switch") == pytest.approx(1.0)


def test_sleep_clause_blocks_yawn_but_not_for_a_rest_or_a_fainted_sleeper():
    sleeper = cond(3, C_SLP)
    assert mf(ops(("yawn",), opp_cond=sleeper)) == 0.0
    rest = torch.zeros(1, 6)
    rest[0, 3] = 1.0
    assert mf(ops(("yawn",), opp_cond=sleeper, opp_rest=rest)) == pytest.approx(1.0)
    dead = torch.ones(1, 6)
    dead[0, 3] = 0.0
    assert mf(ops(("yawn",), opp_cond=sleeper, opp_alive=dead)) == pytest.approx(1.0)


def test_a_substitute_stops_status_but_not_the_bypass_set_or_damage():
    sub = with_vol("opp", "substitute")
    assert mf(ops(("confuseray",), **sub), "p_lands_stay") == 0.0
    assert mf(ops(("taunt",), **sub)) == pytest.approx(1.0)          # bypasssub
    assert mf(ops(("bodyslam",), **sub)) == pytest.approx(1.0)       # the hit lands on the sub


def test_rapid_spin_fails_on_a_ghost_active_and_a_ghost_arrival():
    imm = torch.zeros(1, 6, 19)
    imm[0, 0, T["NORMAL"]] = 1.0
    assert mf(ops(("rapidspin",), imm_dmg=imm), "p_lands_stay") == 0.0
    imm2 = torch.zeros(1, 6, 19)
    imm2[0, 1, T["NORMAL"]] = 1.0                                      # β's arrival is a Ghost
    assert mf(ops(("rapidspin",), imm_dmg=imm2), "p_lands_switch") == 0.0
    assert mf(ops(("rapidspin",)), "p_lands_switch") == pytest.approx(1.0)


def test_counter_needs_a_physical_hit_on_the_user_before_it():
    phys = ops(("counter",), seats=("earthquake", "growl", "growl"), alpha=(0.6, 0.2, 0.1),
               pair_in=hit_grid(seat=0))
    assert mf(phys) == pytest.approx(0.6)
    spec = ops(("counter",), seats=("surf", "growl", "growl"), alpha=(0.6, 0.2, 0.1), pair_in=hit_grid(seat=0))
    assert mf(spec) == 0.0
    assert mf(ops(("mirrorcoat",), seats=("surf", "growl", "growl"), alpha=(0.6, 0.2, 0.1),
                  pair_in=hit_grid(seat=0))) == pytest.approx(0.6)
    # our Substitute takes the hit -> nothing to return
    assert mf(ops(("counter",), seats=("earthquake", "growl", "growl"), alpha=(0.6, 0.2, 0.1),
                  pair_in=hit_grid(seat=0), **with_vol("our", "substitute"))) == 0.0
    # a miss returns nothing: the hit must LAND
    assert mf(ops(("counter",), seats=("earthquake", "growl", "growl"), alpha=(0.6, 0.2, 0.1),
                  pair_in=hit_grid(seat=0, acc=0.5))) == pytest.approx(0.3)
    # a Ghost target is immune to Counter (Fighting)
    imm = torch.zeros(1, 6, 19)
    imm[0, 0, T["FIGHTING"]] = 1.0
    assert mf(ops(("counter",), seats=("earthquake", "growl", "growl"), alpha=(0.6, 0.2, 0.1),
                  pair_in=hit_grid(seat=0), imm_dmg=imm)) == 0.0


def test_hidden_power_counts_for_counter_never_for_mirror_coat():
    seats = ("hiddenpowerfire", "growl", "growl")       # a SPECIAL type in gen 3
    kw = dict(seats=seats, alpha=(0.6, 0.2, 0.1), pair_in=hit_grid(seat=0))
    assert mf(ops(("counter",), **kw)) == pytest.approx(0.6)
    assert mf(ops(("mirrorcoat",), **kw)) == 0.0


def test_beat_up_fails_with_no_healthy_party_member_and_is_typeless():
    assert mf(ops(("beatup",))) == pytest.approx(1.0)
    assert mf(ops(("beatup",), beatup_n=torch.tensor([0.0]))) == 0.0


def test_focus_punch_fails_on_a_hit_but_not_on_one_into_our_substitute_or_a_status_move():
    hit = dict(seats=("tackle", "growl", "growl"), alpha=(0.5, 0.2, 0.2), pair_in=hit_grid(seat=0))
    assert mf(ops(("focuspunch",), **hit)) == pytest.approx(0.5)      # a=0.1 switch + 0.2+0.2 status seats
    assert mf(ops(("focuspunch",), **hit, **with_vol("our", "substitute"))) == pytest.approx(1.0)
    assert mf(ops(("focuspunch",), seats=("thunderwave", "growl", "growl"))) == pytest.approx(1.0)


def test_glare_checks_ghost_immunity_in_gen3():
    chart0 = torch.zeros(1, 6, 19)
    chart0[0, 0, T["NORMAL"]] = 1.0
    assert mf(ops(("glare",), chart0=chart0), "p_lands_stay") == 0.0
    assert mf(ops(("glare",))) == pytest.approx(gen3_data.moves.raw()["glare"]["accuracy"] / 100.0)


def test_an_unrevealed_immunity_ability_discounts_a_damaging_move():
    imm = torch.zeros(1, 6, 19)
    imm[0, 0, T["GROUND"]] = 0.7                       # P(Levitate) from the species prior
    assert mf(ops(("earthquake",), imm_dmg=imm), "p_lands_stay") == pytest.approx(0.3, abs=1e-6)


def test_taunt_encore_disable_rules():
    assert mf(ops(("taunt",), **with_vol("opp", "taunt")), "p_lands_stay") == 0.0
    assert mf(ops(("disable",), **with_vol("opp", "disable")), "p_lands_stay") == 0.0
    assert mf(ops(("encore",), **with_vol("opp", "encore")), "p_lands_stay") == 0.0
    # a switching opponent's arrival has no last move: Encore / Disable find nothing to lock
    assert mf(ops(("encore",)), "p_lands_switch") == 0.0
    # no last move: Encore / Disable fail — unless they move FIRST this turn (then they have one)
    slow = ops(("encore",), opp_last_move=torch.tensor([0]), p_out=torch.tensor([[1.0]]))
    assert mf(slow) == 0.0
    fast = ops(("encore",), opp_last_move=torch.tensor([0]), p_out=torch.tensor([[0.0]]), a_switch=0.0,
               alpha=(0.4, 0.3, 0.3))
    assert mf(fast) == pytest.approx(1.0)
    assert mf(ops(("encore",), opp_last_move=torch.tensor([num("transform")])), "p_lands_stay") == 0.0


def test_their_faster_taunt_stops_our_status_move_only():
    kw = dict(seats=("taunt", "growl", "growl"), alpha=(0.5, 0.2, 0.2), p_out=torch.tensor([[0.0]]))
    assert mf(ops(("swordsdance",), **kw)) == pytest.approx(0.5)
    assert mf(ops(("bodyslam",), **kw)) == pytest.approx(1.0)


def test_their_protect_blocks_a_protectable_move_by_its_odds():
    kw = dict(seats=("protect", "growl", "growl"), alpha=(0.5, 0.2, 0.2))
    assert mf(ops(("bodyslam",), **kw)) == pytest.approx(0.5)
    assert mf(ops(("bodyslam",), opp_protect_odds=torch.tensor([0.5]), **kw)) == pytest.approx(0.75)
    assert mf(ops(("swordsdance",), **kw)) == pytest.approx(1.0)       # a self move is not protectable


def test_their_magic_coat_bounces_only_a_reflectable_move():
    kw = dict(seats=("magiccoat", "growl", "growl"), alpha=(0.5, 0.2, 0.2))
    assert mf(ops(("toxic",), **kw)) == pytest.approx((0.4 + 0.1) * 0.85)   # the 0.5 Magic Coat mass bounces
    assert mf(ops(("taunt",), **kw)) == pytest.approx(1.0)              # Taunt carries no reflectable flag


def test_our_magic_coat_resolves_iff_they_click_a_reflectable_move():
    kw = dict(seats=("toxic", "tackle", "growl"), alpha=(0.4, 0.3, 0.2))
    assert mf(ops(("magiccoat",), **kw)) == pytest.approx(0.4 + 0.2)


# ------------------------------------------------------------------------- Destiny Bond (owner)
def test_destiny_bond_feature_is_p_the_opponent_kos_us_with_no_threshold():
    pin = hit_grid(seat=0, ko=0.8) + hit_grid(seat=1, ko=0.25)
    o = ops(("destinybond",), seats=("earthquake", "surf", "growl"), alpha=(0.5, 0.2, 0.2), pair_in=pin)
    assert mf(o, "dbond_p_ko") == pytest.approx(0.5 * 0.8 + 0.2 * 0.25)
    assert mf(o, "p_ko_us") == pytest.approx(0.5 * 0.8 + 0.2 * 0.25)
    # the speed-free owner feature is unchanged by who moves first
    assert mf(ops(("destinybond",), seats=("earthquake", "surf", "growl"), alpha=(0.5, 0.2, 0.2), pair_in=pin,
                  p_out=torch.tensor([[0.0]])), "dbond_p_ko") == pytest.approx(0.45)


def test_destiny_bond_triggers_only_if_the_bond_is_up_before_the_ko():
    pin = hit_grid(seat=0, ko=1.0)
    kw = dict(seats=("earthquake", "growl", "growl"), alpha=(0.5, 0.2, 0.2), pair_in=pin)
    assert mf(ops(("destinybond",), p_out=torch.tensor([[1.0]]), **kw)) == pytest.approx(0.5)
    assert mf(ops(("destinybond",), p_out=torch.tensor([[0.0]]), **kw)) == 0.0
    # a bond from LAST turn persists until we move
    assert mf(ops(("destinybond",), p_out=torch.tensor([[0.0]]), **kw, **with_vol("our", "destinybond"))) \
        == pytest.approx(0.5)


# --------------------------------------------------------------------------- intent weighting
def test_a_faster_ko_seat_removes_its_alpha_mass():
    kw = dict(seats=("earthquake", "growl", "growl"), pair_in=hit_grid(seat=0, ko=1.0))
    assert mf(ops(("bodyslam",), alpha=(0.5, 0.2, 0.2), p_out=torch.tensor([[0.0]]), **kw)) == pytest.approx(0.5)
    assert mf(ops(("bodyslam",), alpha=(0.5, 0.2, 0.2), p_out=torch.tensor([[0.0]]), **kw), "p_ko_first") \
        == pytest.approx(0.5)
    assert mf(ops(("bodyslam",), alpha=(0.5, 0.2, 0.2), p_out=torch.tensor([[1.0]]), **kw)) == pytest.approx(1.0)
    # priority beats speed: Quick Attack goes first even when slower
    assert mf(ops(("quickattack",), alpha=(0.5, 0.2, 0.2), p_out=torch.tensor([[0.0]]), **kw)) == pytest.approx(1.0)


def test_the_resolution_is_linear_in_alpha():
    kw = dict(seats=("protect", "growl", "growl"))
    a = mf(ops(("bodyslam",), alpha=(0.2, 0.3, 0.3), **kw))
    b = mf(ops(("bodyslam",), alpha=(0.6, 0.1, 0.1), **kw))
    assert a == pytest.approx(0.8) and b == pytest.approx(0.4)


def test_the_switch_branch_is_weighted_by_beta():
    imm = torch.zeros(1, 6, 19)
    imm[0, 1, T["GROUND"]] = 1.0
    beta = torch.tensor([[0.0, 0.25, 0.75, 0.0, 0.0, 0.0]])
    o = ops(("earthquake",), imm_dmg=imm, beta=beta, a_switch=0.5, alpha=(0.2, 0.2, 0.1))
    assert mf(o, "p_lands_switch") == pytest.approx(0.75)
    assert mf(o) == pytest.approx(0.5 + 0.5 * 0.75)


# ---------------------------------------------------------------------------- our own state
def test_p_act_reads_paralysis_freeze_and_sleep():
    assert mf(ops(("tackle",), our_cond=cond(0, C_PAR)), "p_act") == pytest.approx(1 - P_FULL_PARA)
    assert mf(ops(("tackle",), our_cond=cond(0, C_FRZ)), "p_act") == pytest.approx(P_THAW)
    assert mf(ops(("flamewheel",), our_cond=cond(0, C_FRZ)), "p_act") == pytest.approx(1.0)
    asleep = dict(our_cond=cond(0, C_SLP), our_p_wake=torch.tensor([0.25]))
    assert mf(ops(("tackle",), **asleep), "p_act") == pytest.approx(0.25)
    assert mf(ops(("sleeptalk",), **asleep), "p_act") == pytest.approx(0.75)
    assert mf(ops(("sleeptalk",)), "p_act") == 0.0


def test_self_move_rules():
    hp = lambda x: {"our_hp": torch.tensor([x])}  # noqa: E731
    assert mf(ops(("substitute",), **hp(0.25))) == 0.0
    assert mf(ops(("substitute",), **hp(0.26))) == pytest.approx(1.0)
    assert mf(ops(("substitute",), **hp(0.9), **with_vol("our", "substitute"))) == 0.0
    assert mf(ops(("recover",), **hp(1.0))) == 0.0
    assert mf(ops(("recover",), **hp(0.5))) == pytest.approx(1.0)
    assert mf(ops(("rest",), **hp(0.5), our_cond=cond(0, C_SLP))) == 0.0
    assert mf(ops(("bellydrum",), **hp(0.5))) == 0.0
    assert mf(ops(("bellydrum",), **hp(0.6))) == pytest.approx(1.0)
    stages = torch.zeros(1, 5)
    stages[0, 0] = 6.0
    assert mf(ops(("swordsdance",), our_stage=stages)) == 0.0
    assert mf(ops(("dragondance",), our_stage=stages)) == pytest.approx(1.0)   # Speed can still rise
    sp = torch.tensor([[0.0, 1.0]])
    assert mf(ops(("spikes",), spikes=sp)) == 0.0
    assert mf(ops(("spikes",), spikes=torch.tensor([[0.0, 2 / 3]]))) == pytest.approx(1.0)
    sc = torch.zeros(1, 8)
    sc[0, S_REFLECT_OURS] = 1.0
    assert mf(ops(("reflect",), screens=sc)) == 0.0
    w = torch.zeros(1, 7)
    w[0, 2] = 1.0
    assert mf(ops(("raindance",), weather=w)) == 0.0
    assert mf(ops(("sunnyday",), weather=w)) == pytest.approx(1.0)


def test_leech_seed_fails_on_a_seeded_target():
    assert mf(ops(("leechseed",))) == pytest.approx(0.9)
    assert mf(ops(("leechseed",), **with_vol("opp", "leechseed")), "p_lands_stay") == 0.0


def test_safeguard_on_their_side_stops_our_major_status():
    sc = torch.zeros(1, 8)
    sc[0, S_SG_OPP] = 1.0
    assert mf(ops(("thunderwave",), screens=sc)) == 0.0
    assert mf(ops(("leechseed",), screens=sc)) == pytest.approx(0.9)       # Safeguard does not stop Leech Seed


# ------------------------------------------------------------- the incoming status corrections
def _status_grid(col: int, p: float = 0.6) -> torch.Tensor:
    g = torch.zeros(1, 6, K, 14)
    g[0, :, 0, 6 + col] = p
    return g


def test_our_safeguard_zeroes_every_incoming_status():
    sc = torch.zeros(1, 8)
    sc[0, S_SG_OURS] = 1.0
    o = ops(("tackle",), seats=("thunderwave", "growl", "growl"), pair_in=_status_grid(0), screens=sc)
    assert mf(o, "in_p_par") == 0.0 and sf(o, "p_par", 2) == 0.0
    o2 = ops(("tackle",), seats=("thunderwave", "growl", "growl"), pair_in=_status_grid(0))
    assert mf(o2, "in_p_par") == pytest.approx(0.3 * 0.6)


def test_our_substitute_zeroes_only_our_active_row():
    o = ops(("tackle",), seats=("thunderwave", "growl", "growl"), pair_in=_status_grid(0),
            **with_vol("our", "substitute"))
    assert sf(o, "p_par", 0) == 0.0
    assert sf(o, "p_par", 2) == pytest.approx(0.3 * 0.6)


def test_incoming_sleep_and_freeze_clauses():
    sleeper = cond(4, C_SLP)
    o = ops(("tackle",), seats=("spore", "growl", "growl"), pair_in=_status_grid(3), our_cond=sleeper)
    assert sf(o, "p_slp", 2) == 0.0
    rest = torch.zeros(1, 6)
    rest[0, 4] = 1.0
    o = ops(("tackle",), seats=("spore", "growl", "growl"), pair_in=_status_grid(3), our_cond=sleeper, our_rest=rest)
    assert sf(o, "p_slp", 2) == pytest.approx(0.18)
    o = ops(("tackle",), seats=("icebeam", "growl", "growl"), pair_in=_status_grid(2), our_cond=cond(5, C_FRZ))
    assert sf(o, "p_frz", 2) == 0.0


# -------------------------------------------------------------------------------- the switch cell
def test_e_pko_counts_accuracy_once():
    """The shipped `conditional_threat` multiplies ko_ramp by acc, but the op's ko_ramp is ALREADY acc · P(KO|hit)."""
    o = ops(("tackle",), seats=("blizzard", "growl", "growl"), alpha=(0.5, 0.2, 0.2),
            pair_in=hit_grid(seat=0, ko=0.7, acc=0.7, j=3))
    assert sf(o, "e_pko", 3) == pytest.approx(0.5 * 0.7)


def test_spin_denied_keeps_the_fact_and_drops_the_stake():
    ghost = torch.zeros(1, 6)
    ghost[0, 2] = 1.0
    o = ops(("tackle",), seats=("rapidspin", "growl", "growl"), alpha=(0.4, 0.2, 0.2), our_is_ghost=ghost)
    assert sf(o, "p_spin_denied", 2) == pytest.approx(0.4)
    assert sf(o, "p_spin_denied", 1) == 0.0


def test_a_pursuit_ko_on_the_departing_mon_means_the_switch_does_not_resolve():
    o = ops(("tackle",), seats=("pursuit", "growl", "growl"), alpha=(0.5, 0.2, 0.2),
            pair_in=hit_grid(seat=0, high=0.4), our_hp=torch.tensor([0.5]))
    assert sf(o, "p_switch_resolves", 2) == pytest.approx(0.5)     # 2 × 0.4 ≥ 0.5 on every roll
    o = ops(("tackle",), seats=("pursuit", "growl", "growl"), alpha=(0.5, 0.2, 0.2),
            pair_in=hit_grid(seat=0, high=0.2), our_hp=torch.tensor([0.5]))
    assert sf(o, "p_switch_resolves", 2) == pytest.approx(1.0)


def test_gate_zeroes_everything():
    o = ops(("tackle",), gate=torch.zeros(1, 1))
    assert move_facts(o, KIND_T, FLAG_T).abs().sum() == 0


# ------------------------------------------------- the states a FASTER seat changes before our move
def test_a_heal_at_full_hp_fails_unless_a_faster_hit_makes_room():
    kw = dict(seats=("bodyslam", "growl", "growl"), alpha=(0.5, 0.2, 0.2), pair_in=hit_grid(seat=0))
    assert mf(ops(("recover",), p_out=torch.tensor([[1.0]]), **kw)) == 0.0
    assert mf(ops(("recover",), p_out=torch.tensor([[0.0]]), **kw)) == pytest.approx(0.5)
    # a hit on our Substitute leaves our HP full
    assert mf(ops(("recover",), p_out=torch.tensor([[0.0]]), **kw, **with_vol("our", "substitute"))) == 0.0


def test_protect_fails_when_no_action_follows_it():
    """`onPrepareHit: !!this.queue.willAct()` — Protect / Detect / Endure fail as the LAST action of the turn."""
    assert mf(ops(("protect",), a_switch=0.0)) == pytest.approx(1.0)
    assert mf(ops(("protect",)), "p_lands_switch") == 0.0                 # into a switch: nothing follows
    kw = dict(seats=("protect", "growl", "growl"), alpha=(1.0, 0.0, 0.0), a_switch=0.0)
    assert mf(ops(("protect",), p_out=torch.tensor([[0.0]]), **kw)) == 0.0   # their faster Protect: ours is last
    assert mf(ops(("protect",), p_out=torch.tensor([[1.0]]), **kw)) == pytest.approx(1.0)


def test_a_boost_at_plus_six_resolves_after_a_faster_haze_or_an_intimidate_arrival():
    stages = torch.zeros(1, 5)
    stages[0, 0] = 6.0
    kw = dict(seats=("haze", "growl", "growl"), alpha=(0.5, 0.2, 0.2), our_stage=stages, a_switch=0.0)
    assert mf(ops(("swordsdance",), p_out=torch.tensor([[1.0]]), **kw)) == 0.0
    assert mf(ops(("swordsdance",), p_out=torch.tensor([[0.0]]), **kw)) == pytest.approx(0.5)
    from agents.model.move_resolution_tables import NAMED_ABILITIES as NA
    named = torch.zeros(1, 6, len(NA))
    named[0, 1, NA.index("intimidate")] = 1.0
    assert mf(ops(("swordsdance",), our_stage=stages), "p_lands_switch") == 0.0
    assert mf(ops(("swordsdance",), our_stage=stages, opp_named_abl=named), "p_lands_switch") == pytest.approx(1.0)


def test_a_departing_natural_cure_sleeper_lifts_the_switch_branch_sleep_clause():
    from agents.model.move_resolution_tables import NAMED_ABILITIES as NA
    asleep = cond(0, C_SLP)
    assert mf(ops(("hypnosis",), opp_cond=asleep), "p_lands_switch") == 0.0
    named = torch.zeros(1, 6, len(NA))
    named[0, 0, NA.index("naturalcure")] = 1.0
    assert mf(ops(("hypnosis",), opp_cond=asleep, opp_named_abl=named), "p_lands_switch") == pytest.approx(0.6)


def test_wish_fails_while_one_is_pending():
    assert mf(ops(("wish",))) == pytest.approx(1.0)
    assert mf(ops(("wish",), our_wish=torch.tensor([0.5]))) == 0.0


def test_their_faster_self_cure_lifts_already_statused():
    kw = dict(seats=("refresh", "growl", "growl"), alpha=(0.5, 0.2, 0.2), opp_cond=cond(0, C_PAR), a_switch=0.0)
    assert mf(ops(("toxic",), p_out=torch.tensor([[1.0]]), p_land=torch.zeros(1, 4), **kw)) == 0.0
    assert mf(ops(("toxic",), p_out=torch.tensor([[0.0]]), p_land=torch.zeros(1, 4), **kw)) \
        == pytest.approx(0.5 * 0.85)


def test_a_faster_hit_may_break_our_substitute_so_a_new_one_goes_up():
    kw = dict(seats=("bodyslam", "growl", "growl"), alpha=(0.5, 0.2, 0.2), pair_in=hit_grid(seat=0),
              our_hp=torch.tensor([0.9]), a_switch=0.0, **with_vol("our", "substitute"))
    assert mf(ops(("substitute",), p_out=torch.tensor([[1.0]]), **kw)) == 0.0
    assert mf(ops(("substitute",), p_out=torch.tensor([[0.0]]), **kw)) == pytest.approx(0.5)


def test_refresh_resolves_after_a_faster_status_lands_on_us():
    g = torch.zeros(1, 6, K, 14)
    g[0, 0, 0, 6] = 1.0                                   # their seat 0 paralyses our active
    kw = dict(seats=("thunderwave", "growl", "growl"), alpha=(0.5, 0.2, 0.2), pair_in=g, a_switch=0.0)
    assert mf(ops(("refresh",), p_out=torch.tensor([[1.0]]), **kw)) == 0.0
    # the faster Thunder Wave also fully paralyses us 1 time in 4 before we can Refresh
    assert mf(ops(("refresh",), p_out=torch.tensor([[0.0]]), **kw)) == pytest.approx(0.5 * (1 - P_FULL_PARA))


def test_sleep_talk_works_after_a_faster_sleep_move():
    g = torch.zeros(1, 6, K, 14)
    g[0, 0, 0, 9] = 1.0                                   # their seat 0 puts our active to sleep
    kw = dict(seats=("spore", "growl", "growl"), alpha=(0.5, 0.2, 0.2), pair_in=g, a_switch=0.0)
    assert mf(ops(("sleeptalk",), p_out=torch.tensor([[1.0]]), **kw)) == 0.0
    assert mf(ops(("sleeptalk",), p_out=torch.tensor([[0.0]]), **kw)) == pytest.approx(0.5)


def test_their_faster_thaw_or_wake_lifts_already_statused():
    kw = dict(seats=("bodyslam", "growl", "growl"), alpha=(0.5, 0.2, 0.2), a_switch=0.0, p_land=torch.zeros(1, 4))
    frozen = dict(opp_cond=cond(0, C_FRZ), **kw)
    assert mf(ops(("thunderwave",), p_out=torch.tensor([[1.0]]), **frozen)) == 0.0
    assert mf(ops(("thunderwave",), p_out=torch.tensor([[0.0]]), **frozen)) == pytest.approx(0.9 * P_THAW)  # every seat moves first
    asleep = dict(opp_cond=cond(0, C_SLP), opp_p_wake=torch.tensor([0.4]), **kw)
    assert mf(ops(("thunderwave",), p_out=torch.tensor([[0.0]]), **asleep)) == pytest.approx(0.9 * 0.4)
