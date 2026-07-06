import time

from poke_env.spectator.spectated_battle import SpectatedBattle


def _make_battle() -> SpectatedBattle:
    return SpectatedBattle("battle-gen3ou-123456")


def test_battle_tag():
    b = _make_battle()
    assert b.battle_tag == "battle-gen3ou-123456"


def test_initial_state():
    b = _make_battle()
    assert not b.finished
    assert b.winner is None
    assert b.log_text == ""


def test_add_lines_accumulates():
    b = _make_battle()
    b.add_lines([["", "turn", "1"]])
    b.add_lines([["", "move", "p1a: Snorlax", "Body Slam", "p2a: Gengar"]])
    lines = b.log_text.splitlines()
    assert lines[0] == "|turn|1"
    assert lines[1] == "|move|p1a: Snorlax|Body Slam|p2a: Gengar"


def test_add_lines_skips_empty():
    b = _make_battle()
    b.add_lines([[], [""], ["", "turn", "1"]])
    assert b.log_text == "|turn|1"


def test_finish_with_winner():
    b = _make_battle()
    b.add_lines([["", "turn", "1"]])
    b.finish("Alice")
    assert b.finished
    assert b.winner == "Alice"
    assert b.log_text.endswith("|win|Alice")


def test_finish_tie():
    b = _make_battle()
    b.finish(None)
    assert b.finished
    assert b.winner is None
    assert b.log_text == "|tie"


def test_finish_idempotent():
    b = _make_battle()
    b.finish("Alice")
    b.finish("Bob")  # second call must be ignored
    assert b.winner == "Alice"
    assert b.log_text.count("|win|") == 1


def test_add_lines_after_finish_ignored():
    b = _make_battle()
    b.finish("Alice")
    b.add_lines([["", "move", "p1a: Snorlax", "Body Slam", "p2a: Gengar"]])
    # The move line must not appear — finish came first
    assert "|move|" not in b.log_text


def test_last_activity_initialized():
    b = _make_battle()
    assert b.last_activity == b.joined_at


def test_last_activity_bumps_on_add_lines():
    b = _make_battle()
    before = b.last_activity
    time.sleep(0.01)
    b.add_lines([["", "turn", "1"]])
    assert b.last_activity > before


def test_last_activity_bumps_even_when_lines_skipped():
    # An all-empty batch still means the server sent us something — the room is live.
    b = _make_battle()
    before = b.last_activity
    time.sleep(0.01)
    b.add_lines([[], [""]])
    assert b.last_activity > before


def test_log_text_format():
    b = _make_battle()
    b.add_lines([
        ["", "player", "p1", "Alice", "60", "1200"],
        ["", "player", "p2", "Bob", "113", "1300"],
        ["", "turn", "1"],
    ])
    b.finish("Alice")
    lines = b.log_text.splitlines()
    assert lines[0] == "|player|p1|Alice|60|1200"
    assert lines[1] == "|player|p2|Bob|113|1300"
    assert lines[2] == "|turn|1"
    assert lines[3] == "|win|Alice"
