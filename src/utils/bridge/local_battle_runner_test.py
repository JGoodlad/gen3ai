"""Unit tests for the LocalBattleRunner framing (no Node, no subprocess).

The runner fabricates the room header the sim does not emit (`>battle-…` +
`|init|battle`). These tests pin that framing so `_create_battle` fires exactly
once per side and every later chunk routes to the existing battle.
"""

from utils.bridge.local_battle_runner import _LocalBattleRunner


def test_first_chunk_per_side_gets_init():
    inited = {"p1": False, "p2": False}
    tag = "battle-gen3ou-1"

    first = _LocalBattleRunner._frame(tag, "p1", "|gametype|singles\n|start", inited)
    lines = first.split("\n")
    assert lines[0] == ">battle-gen3ou-1"  # room header -> _handle_message current_room
    assert lines[1] == "|init|battle"       # triggers _create_battle
    assert lines[2] == "|gametype|singles"
    assert inited["p1"] is True


def test_subsequent_chunks_have_no_init():
    inited = {"p1": True, "p2": False}
    tag = "battle-gen3ou-1"

    chunk = _LocalBattleRunner._frame(tag, "p1", "|turn|2", inited)
    assert chunk == ">battle-gen3ou-1\n|turn|2"
    assert "|init|battle" not in chunk


def test_sides_init_independently():
    inited = {"p1": False, "p2": False}
    tag = "battle-gen3ou-3"

    p1_first = _LocalBattleRunner._frame(tag, "p1", "|start", inited)
    p2_first = _LocalBattleRunner._frame(tag, "p2", "|start", inited)
    assert "|init|battle" in p1_first
    assert "|init|battle" in p2_first  # p2's first chunk still gets its own init
    assert inited == {"p1": True, "p2": True}


def test_tag_round_trips_to_create_battle_contract():
    # _create_battle does ">battle-gen3ou-1".split("-") -> [">battle","gen3ou","1"],
    # checks [1] == format, and rebuilds the tag via "-".join(...)[1:].
    inited = {"p1": False, "p2": False}
    framed = _LocalBattleRunner._frame("battle-gen3ou-1", "p1", "|start", inited)
    room_line = framed.split("\n")[0]
    parts = room_line.split("-")
    assert parts[1] == "gen3ou"
    assert "-".join(parts)[1:] == "battle-gen3ou-1"
