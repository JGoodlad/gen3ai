"""The ONE source of what play revealed, on hand-written protocol (each rule, and the conservative refusals). The
same parser against the Rust reading on real battles is the log-consistency audit (``audit_integration_test.py``)."""
from agents.info_levers import revealed as R

SW_HERA = "|switch|p2a: Hera|Heracross, M|100/100"
SW_SKARM = "|switch|p2a: Skarm|Skarmory, F|100/100"
SW_OURS = "|switch|p1a: Zap|Zapdos|100/100"


def _r(*lines, viewer=0):
    return R.from_lines(list(lines), viewer)


def test_own_moves_and_the_moves_that_are_not_its_own():
    r = _r(SW_OURS, SW_HERA, "|move|p2a: Hera|Megahorn|p1a: Zap", "|-singleturn|p2a: Hera|move: Focus Punch",
           "|move|p2a: Hera|Rest|p2a: Hera|[from] Sleep Talk", "|move|p2a: Hera|Earthquake|p1a: Zap|[from] Assist",
           "|move|p2a: Hera|Struggle|p1a: Zap", "|move|p1a: Zap|Thunderbolt|p2a: Hera",
           "|-start|p2a: Hera|move: Leech Seed", "|-activate|p2a: Hera|move: Pursuit",
           SW_SKARM, "|cant|p2a: Skarm|move: Taunt|Spikes", "|cant|p2a: Skarm|slp")
    assert r.moves() == {"heracross": {"megahorn", "focuspunch", "rest"}, "skarmory": {"spikes"}}
    assert set(r.mons) == {"heracross", "skarmory"} and r.seen() == {"heracross", "skarmory"}


def test_a_transformed_or_mimicking_mon_reveals_none_of_the_borrowed_moves():
    r = _r(SW_OURS, "|switch|p2a: Ditto|Ditto|100/100", "|move|p2a: Ditto|Transform|p1a: Zap",
           "|-transform|p2a: Ditto|p1a: Zap", "|move|p2a: Ditto|Thunderbolt|p1a: Zap",
           SW_HERA, "|move|p2a: Hera|Mimic|p1a: Zap", "|-start|p2a: Hera|Mimic|Thunderbolt",
           "|move|p2a: Hera|Thunderbolt|p1a: Zap",
           "|switch|p2a: Ditto|Ditto|100/100", "|move|p2a: Ditto|Transform|p1a: Zap")
    assert r.moves() == {"ditto": {"transform"}, "heracross": {"mimic"}}


def test_items_named_before_any_transfer_are_the_original():
    r = _r(SW_OURS, SW_SKARM, "|-heal|p2a: Skarm|88/100|[from] item: Leftovers",
           "|-enditem|p2a: Skarm|Sitrus Berry|[eat]",                       # first fact wins
           SW_HERA, "|-boost|p2a: Hera|spe|1|[from] item: Salac Berry", "|-enditem|p2a: Hera|Salac Berry|[eat]")
    assert r.mons["skarmory"].item == "leftovers" and r.mons["heracross"].item == "salacberry"


def test_knock_off_thief_and_a_berry_reveal_the_victims_item():
    r = _r(SW_OURS, SW_SKARM, "|-enditem|p2a: Skarm|Leftovers|[from] move: Knock Off|[of] p1a: Zap",
           SW_HERA, "|-enditem|p2a: Hera|Choice Band|[silent]|[from] move: Thief|[of] p1a: Zap",
           "|-item|p1a: Zap|Choice Band|[from] move: Thief|[of] p2a: Hera")
    assert r.mons["skarmory"].item == "leftovers" and r.mons["heracross"].item == "choiceband"


def test_trick_reveals_the_opponents_pre_trick_item_either_way_round_and_nothing_after():
    # the opponent tricks us: we receive its item
    r = _r(SW_OURS, SW_HERA, "|-activate|p2a: Hera|move: Trick|[of] p1a: Zap",
           "|-item|p1a: Zap|Choice Scarf|[from] move: Trick", "|-item|p2a: Hera|Leftovers|[from] move: Trick",
           "|-heal|p2a: Hera|90/100|[from] item: Leftovers")
    assert r.mons["heracross"].item == "choicescarf"
    # we trick it: the opponent's -item line is what it RECEIVED; ours names its item
    r = _r(SW_OURS, SW_HERA, "|-activate|p1a: Zap|move: Trick|[of] p2a: Hera",
           "|-item|p2a: Hera|Choice Band|[from] move: Trick", "|-item|p1a: Zap|Lum Berry|[from] move: Trick",
           "|-heal|p2a: Hera|90/100|[from] item: Leftovers")
    assert r.mons["heracross"].item == "lumberry"
    # a thief that steals OUR item: its later item lines are not its original
    r = _r(SW_OURS, SW_HERA, "|-enditem|p1a: Zap|Leftovers|[silent]|[from] move: Thief|[of] p2a: Hera",
           "|-item|p2a: Hera|Leftovers|[from] move: Thief|[of] p1a: Zap",
           "|-heal|p2a: Hera|90/100|[from] item: Leftovers")
    assert r.mons["heracross"].item is None and r.mons["heracross"].item_moved


def test_abilities_and_the_owner_of_a_from_ability_tag():
    r = _r(SW_OURS, "|switch|p2a: Mence|Salamence, M|100/100", "|-ability|p2a: Mence|Intimidate|boost",
           "|switch|p2a: Umb|Umbreon, F|100/100",
           "|-status|p1a: Zap|par|[from] ability: Synchronize|[of] p2a: Umb",
           "|switch|p2a: Ttar|Tyranitar, M|100/100", "|-weather|Sandstorm|[from] ability: Sand Stream|[of] p2a: Ttar",
           "|switch|p2a: Lan|Lanturn, F|100/100", "|-heal|p2a: Lan|100/100|[from] ability: Volt Absorb|[of] p1a: Zap",
           "|switch|p2a: Bliss|Blissey, F|100/100", "|-curestatus|p2a: Bliss|par|[from] ability: Natural Cure",
           "|switch|p2a: Shark|Sharpedo, M|100/100",
           "|-damage|p1a: Zap|300/322|[from] ability: Rough Skin|[of] p2a: Shark",
           "|switch|p2a: Slak|Slaking, M|100/100", "|cant|p2a: Slak|ability: Truant")
    got = {k: v.ability for k, v in r.mons.items()}
    assert got == {"salamence": "intimidate", "umbreon": "synchronize", "tyranitar": "sandstream",
                   "lanturn": "voltabsorb", "blissey": "naturalcure", "sharpedo": "roughskin", "slaking": "truant"}
    assert "zapdos" not in r.mons, "our own mons are never in the reveals"


def test_trace_reveals_the_tracers_original_and_the_traced_mons_ability():
    r = _r("|switch|p1a: P2|Porygon2|100/100", "|switch|p2a: Zard|Charizard, M|100/100",
           "|-ability|p1a: P2|Blaze|Trace|[from] ability: Trace|[of] p2a: Zard")
    assert r.mons["charizard"].ability == "blaze"
    r = _r("|switch|p1a: Zard|Charizard, M|100/100", "|switch|p2a: P2|Porygon2|100/100",
           "|-ability|p2a: P2|Blaze|Trace|[from] ability: Trace|[of] p1a: Zard",
           "|-damage|p1a: Zard|50/100|[from] ability: Blaze|[of] p2a: P2")
    assert r.mons["porygon2"].ability == "trace"


def test_an_ambiguous_owner_reveals_nothing_and_is_counted():
    r = _r(SW_OURS, SW_HERA, "|-whatever|p1a: Zap|x|[from] ability: Mystery|[of] p2a: Hera",
           "|-item|p2a: Hera|Leftovers|[from] something")
    assert r.mons["heracross"].ability is None and r.mons["heracross"].item is None
    assert r.ambiguous == {"-whatever [from] ability [of]": 1, "-item [from] something": 1}


def test_viewer_two_reads_p1_as_the_opponent_and_decisions_index_the_tracker_entries():
    lines = ["|switch|p1a: Hera|Heracross, M|100/100", "|move|p1a: Hera|Megahorn|p2a: X"]
    assert R.from_lines(lines, 1).moves() == {"heracross": {"megahorn"}}
    res = {"chunks": [[1, lines[:1]], [0, ["|x|"]], [1, lines[1:]]],
           "trackers": [[], [{"after": 0, "obs": {}}, {"after": 2, "obs": {}}, {"terminal": 1.0}]]}
    assert R.at_decision(res, 1, 0).moves() == {"heracross": set()}
    assert R.at_decision(res, 1, 1).moves() == {"heracross": {"megahorn"}}
