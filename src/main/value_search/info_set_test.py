"""``main.value_search.info_set``: what the leak-honest builder reads off I alone (segments, the key, the opponent's
intents, the counters' re-roll targets, the hidden-HP hits), on synthetic lines (no engine)."""
from __future__ import annotations

from main.value_search import info_set as IS

REQ_MOVE = '|request|{"active":[{"moves":[]}],"side":{"name":"me","id":"p1","pokemon":[{"ident":"p1: Snorlax",' \
           '"details":"Snorlax, M","condition":"500/500","active":true}]}}'
REQ_WAIT = '|request|{"wait":true,"side":{"name":"me","id":"p1","pokemon":[{"ident":"p1: Snorlax",' \
           '"details":"Snorlax, M","condition":"500/500","active":true}]}}'


def info(lines, cmds=(), team="Snorlax|||||||||||") -> IS.InfoSet:
    return IS.InfoSet(format_id="gen3ou", names=("me", "foe"), side="p1", team=team, lines=tuple(lines),
                      cmds=tuple(cmds))


START = ["|teamsize|p1|6", "|teamsize|p2|6", "|start", "|switch|p1a: Snorlax|Snorlax, M|500/500",
         "|switch|p2a: Gengar|Gengar, F|100/100", "|turn|1", REQ_MOVE]


def test_segments_split_after_each_request():
    i = info(START + ["|move|p2a: Gengar|Hypnosis|p1a: Snorlax", "|-status|p1a: Snorlax|slp", "|upkeep", "|turn|2",
                      REQ_MOVE])
    segs = i.segments()
    assert len(segs) == 2 and segs[0][-1] == REQ_MOVE and segs[1][-1] == REQ_MOVE
    assert IS.team_size(i, "p2") == 6


def test_key_is_a_function_of_I_alone_and_moves_with_it():
    a = info(START, ["move bodyslam"])
    b = info(list(START), ["move bodyslam"])
    assert a.key() == b.key()
    assert info(START, ["move rest"]).key() != a.key()
    assert info(START[:-1] + ["|-hint|x", START[-1]]).key() != info(START).key()


def test_intents_read_the_first_own_action_and_stand_in_otherwise():
    t1 = ["|move|p1a: Snorlax|Body Slam|p2a: Gengar", "|-immune|p2a: Gengar",
          "|move|p2a: Gengar|Hidden Power|p1a: Snorlax", "|-damage|p1a: Snorlax|400/500", "|upkeep", "|turn|2",
          REQ_MOVE]
    t2 = ["|switch|p2a: Snorlax2|Snorlax, F|100/100", "|move|p1a: Snorlax|Body Slam|p2a: Snorlax2",
          "|upkeep", "|turn|3", REQ_MOVE]
    t3 = ["|cant|p2a: Snorlax2|slp", "|upkeep", "|turn|4", REQ_MOVE]
    t4 = ["|move|p2a: Snorlax2|Sleep Talk|p2a: Snorlax2", "|move|p2a: Snorlax2|Earthquake|p1a: Snorlax|[from]Sleep Talk",
          "|upkeep", "|turn|5", REQ_MOVE]
    got = IS.intents(info(START + t1 + t2 + t3 + t4))
    assert got == [None, "move hiddenpower", "switch Snorlax2", None, "move sleeptalk"]


def test_intent_looks_past_a_mid_turn_request_but_not_into_a_new_choice():
    # our Baton Pass splits the turn: the opponent's turn choice shows after our switch
    bp = ["|move|p1a: Snorlax|Baton Pass|p1a: Snorlax", REQ_MOVE]
    cont = ["|switch|p1a: Jolteon|Jolteon, M|300/300", "|move|p2a: Gengar|Thunderbolt|p1a: Jolteon",
            "|upkeep", "|turn|2", REQ_MOVE]
    assert IS.intents(info(START + bp + cont))[1] == "move thunderbolt"
    # the opponent KO'd before moving, then a forced switch: the switch is a NEW choice, not the turn's
    ko = ["|move|p1a: Snorlax|Body Slam|p2a: Gengar", "|-damage|p2a: Gengar|0 fnt", "|faint|p2a: Gengar", REQ_WAIT]
    sw = ["|switch|p2a: Zapdos|Zapdos|100/100", "|upkeep", "|turn|2", REQ_MOVE]
    assert IS.intents(info(START + ko + sw)) == [None, None, "switch Zapdos"]


def test_backtrack_names_running_counters_onsets():
    t1 = ["|move|p2a: Gengar|Hypnosis|p1a: Snorlax", "|-status|p1a: Snorlax|slp", "|upkeep", "|turn|2", REQ_MOVE]
    t2 = ["|cant|p1a: Snorlax|slp", "|upkeep", "|turn|3", REQ_MOVE]
    t3 = ["|-curestatus|p1a: Snorlax|slp|[msg]", "|upkeep", "|turn|4", REQ_MOVE]
    t4 = ["|upkeep", "|turn|5", REQ_MOVE]
    bt = IS.backtrack(info(START + t1 + t2 + t3 + t4))
    assert bt == [[], [], [1], [1], []]
    rest = ["|move|p1a: Snorlax|Rest|p1a: Snorlax", "|-status|p1a: Snorlax|slp|[from] move: Rest", "|upkeep",
            "|turn|2", REQ_MOVE]
    assert IS.backtrack(info(START + rest + t2)) == [[], [], []]       # Rest's sleep is fixed: no die


def test_hp_last_lists_the_opponents_rolled_hits_latest_first():
    t1 = ["|move|p1a: Snorlax|Body Slam|p2a: Gengar", "|-damage|p2a: Gengar|60/100",
          "|-heal|p2a: Gengar|66/100|[from] item: Leftovers", "|upkeep", "|turn|2", REQ_MOVE]
    t2 = ["|move|p1a: Snorlax|Body Slam|p2a: Gengar", "|-damage|p2a: Gengar|20/100",
          "|-damage|p2a: Gengar|14/100|[from] Sandstorm", "|upkeep", "|turn|3", REQ_MOVE]
    t3 = ["|upkeep", "|turn|4", REQ_MOVE]
    hl = IS.hp_last(info(START + t1 + t2 + t3))
    assert hl == [[], [], [["p2a: Gengar", [1]]], [["p2a: Gengar", [2, 1]]]]


def test_own_genders_names_every_gender_from_our_request():
    i = info(START, team="Snorlax|||||||||||")
    assert IS.own_genders(i).split("|")[7] == "M"
    assert IS.own_genders(info(START, team="Snorlax|||||||F||||")).split("|")[7] == "F"


def test_hp_last_forgets_hits_once_the_exact_hp_is_known_again():
    hit = ["|move|p1a: Snorlax|Body Slam|p2a: Gengar", "|-damage|p2a: Gengar|60/100", "|upkeep", "|turn|2", REQ_MOVE]
    full = ["|-heal|p2a: Gengar|100/100|[from] move: Recover", "|upkeep", "|turn|3", REQ_MOVE]
    t3 = ["|upkeep", "|turn|4", REQ_MOVE]
    hl = IS.hp_last(info(START + hit + full + t3))
    assert hl[2] == [["p2a: Gengar", [1]]] and hl[3] == []


def test_prefix_cuts_I_at_an_earlier_decision_and_its_key_is_that_I():
    t1 = ["|move|p2a: Gengar|Hypnosis|p1a: Snorlax", "|-status|p1a: Snorlax|slp", "|upkeep", "|turn|2", REQ_MOVE]
    early, late = info(START, []), info(START + t1, ["move rest"])
    cut = IS.prefix(late, len(START), 0)
    assert cut == early and cut.key() == early.key()
    for bad in ((0, 0), (len(late.lines) + 1, 0), (len(START), 2)):
        try:
            IS.prefix(late, *bad)
        except ValueError:
            continue
        raise AssertionError(f"{bad}: a cut outside I is refused")


def test_turn_of_and_checkpoints_follow_the_live_searchers_rule():
    assert IS.turn_of(START) == 1 and IS.turn_of(START[:3]) == 0
    pts = [IS.Point(lines=5, cmds=0, n_legal=2, turn=9, row="a"), IS.Point(lines=9, cmds=1, n_legal=1, turn=10, row="b"),
           IS.Point(lines=14, cmds=2, n_legal=2, turn=10, row="c"), IS.Point(lines=20, cmds=3, n_legal=5, turn=11, row="d")]
    assert [p.row for p in IS.checkpoints(pts, 10)] == ["c", "d"]       # turn >= 10 AND >= 2 legal actions
    assert [p.row for p in IS.checkpoints(pts, 1)] == ["a", "c", "d"]
