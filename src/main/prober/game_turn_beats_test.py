"""`engine.turn_beats` — a turn read as ORDERED BEATS (designs/prober/battle_viewer_ux_2026-10-09.md §5).

Pure: synthetic gen-3 protocol in the exact framing the sim emits (`deps/pokemon-showdown/sim/battle.ts`:
the residual action opens with a blank line and closes with `upkeep`; in gen 3 a fainted mon is
replaced straight after the action that KO'd it, before the next move). Each test pins one reading
the owner found hard ("the switch, faint etc. is so hard to understand") and fails on its revert.
"""

from __future__ import annotations

from main.prober.engine.turn_events import fold_turns

_HEAD = ("|player|p1|trainee||", "|player|p2|foe||", "|teamsize|p1|6", "|teamsize|p2|6", "|start",
         "|switch|p1a: Bob|Metagross|343/343", "|switch|p2a: Milotic|Milotic, F|100/100")


def _turn(log, n=1, side="p1"):
    turns = fold_turns(_HEAD + tuple(log), trainee_side=side)
    return next(t for t in turns if t["turn"] == n)


def _shape(t):
    return [(b["phase"], b.get("side")) for b in t["beats"]]


def test_explosion_double_ko_then_both_replacements_are_REPLACE_beats_not_switches():
    """Gen 3: an Explosion that KOs both sides → each side sends a replacement in straight away.
    Those are FORCED, and must never read like the voluntary switch at the start of the turn."""
    t = _turn(("|turn|1", "|", "|switch|p2a: Gross|Metagross|100/100",
               "|move|p1a: Bob|Explosion|p2a: Gross", "|-resisted|p2a: Gross", "|-crit|p2a: Gross",
               "|-damage|p2a: Gross|0 fnt", "|faint|p1a: Bob", "|faint|p2a: Gross", "|",
               "|switch|p1a: Zard|Charizard, M|297/297", "|switch|p2a: Lax|Snorlax, M|100/100", "|", "|upkeep"))
    assert _shape(t) == [("switch", "opp"), ("move", "we"), ("replace", "we"), ("replace", "opp")]
    move = t["beats"][1]
    assert move["first"] and move["order"] == 1 and move["species"] == "Metagross" and move["move"] == "Explosion"
    dmg, f1, f2 = move["effects"]
    assert dmg["kind"] == "damage" and dmg["species"] == "Metagross" and dmg["side"] == "opp"
    assert dmg["tags"] == ["not very effective", "critical hit"], "the qualifiers left the hit they qualify"
    assert dmg["hp_before"] == 100.0 and dmg["hp_after"] == 0.0 and dmg["cause"] == "Explosion"
    assert (f1["side"], f1["cause"]) == ("we", "its own Explosion")
    assert (f2["side"], f2["cause"]) == ("opp", "our Metagross's Explosion")
    assert t["beats"][2]["species"] == "Charizard" and t["beats"][2]["from_species"] == "Metagross"
    s = t["summary"]
    assert s["we"]["action"]["label"] == "Explosion" and s["we"]["fainted"] == ["Metagross"]
    assert s["we"]["sent_in"] == ["Charizard"] and s["opp"]["action"]["label"] == "Metagross"


def test_a_mid_turn_replacement_sits_between_two_moves_and_the_order_counts_on():
    """Gen 3's replacement comes BEFORE the next move (it takes that move): move 1, replace, move 2."""
    t = _turn(("|turn|1", "|", "|move|p2a: Milotic|Explosion|p1a: Bob", "|-damage|p1a: Bob|0 fnt",
               "|faint|p2a: Milotic", "|faint|p1a: Bob", "|",
               "|switch|p1a: Zard|Charizard, M|297/297", "|switch|p2a: Lax|Snorlax, M|100/100",
               "|move|p1a: Zard|Flamethrower|p2a: Lax", "|-damage|p2a: Lax|80/100", "|", "|upkeep"))
    assert _shape(t) == [("move", "opp"), ("replace", "we"), ("replace", "opp"), ("move", "we")]
    assert [b.get("order") for b in t["beats"]] == [1, None, None, 2]


def test_end_of_turn_residuals_are_their_own_beat_and_a_residual_faint_is_replaced_after_upkeep():
    t = _turn(("|turn|1", "|", "|move|p1a: Bob|Meteor Mash|p2a: Milotic", "|-damage|p2a: Milotic|4/100",
               "|", "|-weather|Sandstorm|[upkeep]", "|-damage|p2a: Milotic|0 fnt|[from] Sandstorm",
               "|-heal|p1a: Bob|343/343|[from] item: Leftovers", "|upkeep", "|faint|p2a: Milotic", "|",
               "|switch|p2a: Lax|Snorlax, M|100/100"))
    assert _shape(t) == [("move", "we"), ("residual", None), ("replace", "opp")]
    res = t["beats"][1]["effects"]
    assert [e["kind"] for e in res] == ["weather", "damage", "heal", "faint"]
    assert res[-1]["cause"] == "Sandstorm", "a residual KO lost its cause"
    assert res[2]["source"] == "Leftovers"


def test_cant_move_is_an_action_in_the_order():
    t = _turn(("|turn|1", "|", "|cant|p2a: Milotic|par", "|move|p1a: Bob|Earthquake|p2a: Milotic",
               "|-damage|p2a: Milotic|50/100", "|", "|upkeep"))
    assert [(b["kind"], b.get("order")) for b in t["beats"]] == [("cant", 1), ("move", 2)]
    assert "fully paralyzed" in t["beats"][0]["text"]
    assert t["summary"]["opp"]["action"]["label"] == "can't move"


def test_a_mon_is_named_by_its_species_never_its_nickname():
    """`Bob` is our Metagross: no sentence may say "Bob" (a nickname is not something a reader knows;
    the HEAD smoke's French nicknames read 'our Leuphorie switched in for Tyranocif')."""
    t = _turn(("|turn|1", "|", "|move|p1a: Bob|Meteor Mash|p2a: Milotic", "|-damage|p2a: Milotic|60/100",
               "|move|p2a: Milotic|Surf|p1a: Bob", "|-damage|p1a: Bob|200/343", "|", "|upkeep"))
    texts = [e["text"] for e in t["events"]] + [b.get("text") or "" for b in t["beats"]]
    assert not any("Bob" in x for x in texts), texts
    assert all(e.get("species") in (None, "Metagross", "Milotic") for e in t["events"])


def test_still_is_not_a_charge():
    """`[still]` only suppresses the animation — a failed Protect is not "charging"."""
    t = _turn(("|turn|1", "|", "|move|p2a: Milotic|Protect||[still]", "|-fail|p2a: Milotic", "|", "|upkeep"))
    assert "charging" not in t["events"][0]["text"]
    assert [e["kind"] for e in t["beats"][0]["effects"]] == ["fail"]


def test_a_baton_pass_switch_stays_in_the_baton_pass_beat():
    t = _turn(("|turn|1", "|", "|move|p2a: Milotic|Baton Pass|p2a: Milotic",
               "|switch|p2a: Lax|Snorlax, M|100/100", "|move|p1a: Bob|Earthquake|p2a: Lax",
               "|-damage|p2a: Lax|70/100", "|", "|upkeep"))
    assert _shape(t) == [("move", "opp"), ("move", "we")]
    assert t["beats"][0]["effects"][0]["kind"] == "switch"


def test_a_multi_hit_move_is_one_line_with_its_total_and_hit_count():
    t = _turn(("|turn|1", "|", "|move|p1a: Bob|Bullet Seed|p2a: Milotic", "|-damage|p2a: Milotic|90/100",
               "|-damage|p2a: Milotic|81/100", "|-damage|p2a: Milotic|70/100", "|-hitcount|p2a: Milotic|3",
               "|", "|upkeep"))
    (dmg,) = t["beats"][0]["effects"]
    assert (dmg["hp_before"], dmg["hp_after"], dmg["pct"], dmg["hits"]) == (100.0, 70.0, -30.0, 3)


def test_the_end_of_the_battle_names_who_won_from_the_trainees_seat():
    turns = fold_turns(_HEAD + ("|turn|1", "|", "|move|p1a: Bob|Meteor Mash|p2a: Milotic",
                                "|-damage|p2a: Milotic|0 fnt", "|faint|p2a: Milotic", "|", "|win|trainee"),
                       trainee_side="p1")
    end = turns[-1]["beats"][-1]
    assert end["phase"] == "end" and end["effects"][0]["text"] == "we won the battle"
    assert fold_turns(_HEAD + ("|turn|1", "|", "|win|trainee"), trainee_side="p2")[-1]["beats"][-1][
        "effects"][0]["text"] == "they won the battle"


def test_the_board_records_what_the_protocol_revealed_about_items_and_abilities():
    turns = fold_turns(_HEAD + ("|-weather|Sandstorm|[from] ability: Sand Stream|[of] p2a: Milotic",
                                "|turn|1", "|", "|-heal|p1a: Bob|343/343|[from] item: Leftovers",
                                "|-enditem|p2a: Milotic|Lum Berry|[eat]", "|", "|upkeep"), trainee_side="p1")
    b = turns[1]["board"]
    assert b["we"]["mons"][0]["item"] == "Leftovers"
    opp = b["opp"]["mons"][0]
    assert opp["ability"] == "Sand Stream" and opp["item"] == "Lum Berry" and opp["item_gone"] is True
