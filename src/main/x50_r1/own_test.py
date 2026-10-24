"""X50 rung 1b step 0: the own-side tracker and request synthesiser on hand-written public lines (pure; no binary).

Each test pins ONE rule of ``main.x50_r1.own`` and fails when that rule is reverted."""
from __future__ import annotations

import json

import pytest

from main.x50_r1 import own as O

TEAM = ("Zapdos||leftovers|pressure|thunderbolt,hiddenpowerice,toxic,rest|Modest|252,,,252,4,||,30,30,,,|||]"
        "Snorlax||choiceband|thickfat|bodyslam,earthquake,selfdestruct,curse|Adamant|252,252,,,4,|||||]"
        "Gengar||leftovers|levitate|thunderbolt,icepunch,hiddenpowerfire,explosion|Timid|,,,252,4,252||,30,,30,,30|||]"
        "Ursaring||leftovers|guts|return,earthquake,focuspunch,swordsdance|Adamant|252,252,,,4,||||||255")


def _trk(seat: str = "p1") -> O.OwnTracker:
    return O.OwnTracker(seat, "me", O.parse_team(TEAM, {"Snorlax": "M"}))


def _req(t: O.OwnTracker, force: bool = False, locked=None) -> dict:
    return json.loads(t.request(force, locked)[0][len("|request|"):])


def _feed(t: O.OwnTracker, *lines: str) -> list:
    return [x for ln in lines for x in t.lines(ln)]


def test_percent_to_hp_lands_in_the_displays_preimage_and_its_middle():
    for mx in (1, 100, 251, 273, 298, 401, 714):
        for hp in range(0, mx + 1):
            pct = O._shown(hp, mx)
            got = O.percent_to_hp(pct, mx)
            assert O._shown(got, mx) == pct, (mx, hp)
    # 76 % of 273: the preimage is 205..207; the middle is 206 (not pct x max rounded, 207)
    assert O.percent_to_hp(76, 273) == 206


def test_stats_are_the_gen3_formula():
    z = O.parse_team(TEAM)[0]
    assert z.stats == {"hp": 384, "atk": 193, "def": 205, "spa": 383, "spd": 217, "spe": 236}
    assert O.gen3_stats("shedinja", "Adamant", [0] * 6, [31] * 6)["hp"] == 1


def test_own_hp_is_rewritten_exact_and_the_foes_is_not():
    t = _trk()
    out = _feed(t, "|switch|p1a: Zapdos|Zapdos|100/100", "|switch|p2a: Foe|Tyranitar, M|100/100",
                "|-damage|p1a: Zapdos|50/100 par", "|-damage|p2a: Foe|40/100")
    assert out[0] == "|switch|p1a: Zapdos|Zapdos|384/384"
    assert "|-damage|p1a: Zapdos|190/384 par" in out
    assert "|-damage|p2a: Foe|40/100" in out
    assert _req(t)["side"]["pokemon"][0]["condition"] == "190/384 par"


def test_an_own_pressure_switch_in_owes_its_silent_announce_after_the_switch_batch():
    t = _trk()
    out = _feed(t, "|switch|p1a: Zapdos|Zapdos|100/100", "|switch|p2a: Foe|Tyranitar, M|100/100", "|turn|1")
    assert out == ["|switch|p1a: Zapdos|Zapdos|384/384", "|switch|p2a: Foe|Tyranitar, M|100/100",
                   "|-ability|p1a: Zapdos|Pressure|[silent]", "|turn|1"]


def test_the_roster_is_whole_and_a_switch_swaps_the_incoming_mon_into_slot_one():
    t = _trk()
    _feed(t, "|switch|p1a: Zapdos|Zapdos|100/100", "|switch|p2a: Foe|Tyranitar, M|100/100")
    assert [m["ident"] for m in _req(t)["side"]["pokemon"]] == ["p1: Zapdos", "p1: Snorlax", "p1: Gengar",
                                                                "p1: Ursaring"]
    _feed(t, "|switch|p1a: Gengar|Gengar|100/100")
    r = _req(t)
    assert [m["ident"] for m in r["side"]["pokemon"]] == ["p1: Gengar", "p1: Snorlax", "p1: Zapdos", "p1: Ursaring"]
    assert r["side"]["pokemon"][0]["active"] and not r["side"]["pokemon"][2]["active"]
    assert r["side"]["pokemon"][1]["details"] == "Snorlax, M"


def test_pp_pressure_and_a_non_ghost_curse():
    t = _trk()
    _feed(t, "|switch|p1a: Snorlax|Snorlax, M|100/100", "|switch|p2a: Foe|Moltres|100/100",
          "|move|p1a: Snorlax|Body Slam|p2a: Foe", "|move|p1a: Snorlax|Curse|p1a: Snorlax")
    pp = {m["id"]: m["pp"] for m in _req(t)["active"][0]["moves"]}
    # Moltres holds Pressure by the Smogon majority (gen 3 never announces it): Body Slam pays 2; a non-Ghost's
    # Curse targets itself and pays 1
    assert pp["bodyslam"] == 22 and pp["curse"] == 15


def test_a_called_move_pays_no_pp_and_pursuits_switch_hit_is_its_own():
    t = _trk()
    _feed(t, "|switch|p1a: Snorlax|Snorlax, M|100/100", "|switch|p2a: Foe|Tyranitar, M|100/100",
          "|move|p1a: Snorlax|Earthquake|p2a: Foe|[from] Sleep Talk")
    pp = {m["id"]: m["pp"] for m in _req(t)["active"][0]["moves"]}
    assert pp["earthquake"] == 16
    pts = [(0, "turn")]
    lines = ["|move|p1a: Snorlax|Sleep Talk|p1a: Snorlax", "|move|p1a: Snorlax|Earthquake|p2a: Foe|[from] Sleep Talk"]
    assert O.viewer_actions(lines, "p1", pts) == [O.OwnAction("move", "sleeptalk")]
    lines = ["|-activate|p2a: Foe|move: Pursuit", "|move|p1a: Snorlax|Pursuit|p2a: Foe|[from] Pursuit"]
    assert O.viewer_actions(lines, "p1", pts) == [O.OwnAction("move", "pursuit")]


def test_a_choice_band_locks_the_first_move_until_the_mon_leaves():
    t = _trk()
    _feed(t, "|switch|p1a: Snorlax|Snorlax, M|100/100", "|switch|p2a: Foe|Tyranitar, M|100/100",
          "|move|p1a: Snorlax|Earthquake|p2a: Foe")
    dis = {m["id"]: m["disabled"] for m in _req(t)["active"][0]["moves"]}
    assert dis == {"bodyslam": True, "earthquake": False, "selfdestruct": True, "curse": True}
    _feed(t, "|switch|p1a: Zapdos|Zapdos|100/100", "|switch|p1a: Snorlax|Snorlax, M|100/100")
    assert not any(m["disabled"] for m in _req(t)["active"][0]["moves"])


def test_encore_disable_taunt_and_zero_pp_close_moves_and_nothing_left_is_struggle():
    t = _trk()
    _feed(t, "|switch|p1a: Zapdos|Zapdos|100/100", "|switch|p2a: Foe|Tyranitar, M|100/100",
          "|move|p1a: Zapdos|Toxic|p2a: Foe", "|-start|p1a: Zapdos|Encore")
    assert [m["disabled"] for m in _req(t)["active"][0]["moves"]] == [True, True, False, True]
    _feed(t, "|-end|p1a: Zapdos|Encore", "|-start|p1a: Zapdos|move: Taunt")
    assert [m["disabled"] for m in _req(t)["active"][0]["moves"]] == [False, False, True, True]
    _feed(t, "|-end|p1a: Zapdos|move: Taunt", "|-start|p1a: Zapdos|Disable|Thunderbolt")
    assert [m["disabled"] for m in _req(t)["active"][0]["moves"]] == [True, False, False, False]
    t.mons["Zapdos"].pp = [0, 0, 0, 0]
    r, acts = t.request(False)
    moves = json.loads(r[len("|request|"):])["active"][0]["moves"]
    assert moves == [{"move": "Struggle", "id": "struggle", "target": "randomNormal", "disabled": False}]


def test_a_locked_turn_lists_that_move_alone_and_traps():
    t = _trk()
    _feed(t, "|switch|p1a: Snorlax|Snorlax, M|100/100", "|switch|p2a: Foe|Tyranitar, M|100/100")
    a = _req(t, locked="bodyslam")["active"][0]
    assert a == {"moves": [{"move": "Body Slam", "id": "bodyslam"}], "trapped": True}
    assert _req(t, locked="recharge")["active"][0]["moves"] == [{"move": "Recharge", "id": "recharge"}]
    lines = ["|move|p1a: Snorlax|Outrage|p2a: Foe|[from] lockedmove"]
    assert O.viewer_actions(lines, "p1", [(0, "turn")]) == [O.OwnAction("move", "outrage", locked="outrage")]
    assert O.viewer_actions(["|cant|p1a: Snorlax|recharge"], "p1", [(0, "turn")]) == [
        O.OwnAction("none", locked="recharge")]


def test_return_writes_its_power_into_the_request():
    t = _trk()
    _feed(t, "|switch|p1a: Ursaring|Ursaring, F|100/100")
    r = _req(t)
    assert r["active"][0]["moves"][0]["move"] == "Return 102"
    assert r["side"]["pokemon"][0]["moves"][0] == "return102"


def test_a_charge_locks_the_next_turn_unless_it_released_in_the_same_window():
    pts = [(0, "turn"), (3, "turn"), (6, "turn")]
    lines = ["|move|p1a: X|Solar Beam||[still]", "|-prepare|p1a: X|Solar Beam", "|upkeep",
             "|move|p1a: X|Solar Beam|p2a: Y|[from] lockedmove", "|upkeep", "|turn|3"]
    assert O.charge_locks(lines, "p1", pts) == [None, "solarbeam", None]
    sun = ["|move|p1a: X|Solar Beam||[still]", "|-prepare|p1a: X|Solar Beam", "|-anim|p1a: X|Solar Beam|p2a: Y",
           "|move|p1a: X|Toxic|p2a: Y", "|upkeep", "|turn|3"]
    assert O.charge_locks(sun, "p1", pts) == [None, None, None]
    fainted = ["|move|p1a: X|Solar Beam||[still]", "|-prepare|p1a: X|Solar Beam", "|faint|p1a: X"]
    assert O.charge_locks(fainted, "p1", [(0, "turn"), (3, "turn")]) == [None, None]


def test_viewer_actions_cant_is_hidden_and_a_forced_decision_reads_its_replacement():
    pts = [(0, "turn"), (2, "forced")]
    lines = ["|cant|p1a: X|par", "|faint|p1a: X", "|switch|p1a: W|W|100/100"]
    assert O.viewer_actions(lines, "p1", pts) == [O.OwnAction("none"), O.OwnAction("switch", "W")]


def test_action_index_maps_by_token_and_refuses_what_the_request_closes():
    tok = {0: "switch Zapdos", 6: "move thunderbolt", 7: "move hiddenpowerice", 8: "move toxic"}
    mask = [1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
    moves = ["thunderbolt", "hiddenpowerice", "toxic", "rest"]
    assert O.action_index(O.OwnAction("move", "hiddenpower"), tok, mask, moves) == 7
    assert O.action_index(O.OwnAction("switch", "Zapdos"), tok, mask, moves) == 0
    assert O.action_index(O.OwnAction("move", "toxic"), tok, mask, moves) == -2       # closed by the mask
    assert O.action_index(O.OwnAction("move", "surf"), tok, mask, moves) == -2        # not in the set
    assert O.action_index(O.OwnAction("none"), tok, mask, moves) == -1


def test_an_unknown_own_mon_is_a_refusal_but_a_side_token_is_not_a_mon():
    t = _trk()
    _feed(t, "|-sidestart|p1: me|Spikes")
    with pytest.raises(O.OwnError):
        _feed(t, "|switch|p1a: Mewtwo|Mewtwo|100/100")
