"""Synthesizing a reconstruction record for a battle still in flight."""

from __future__ import annotations

import json

import pytest

from main.search_dividend.record import (LiveRecordBuilder, active_builder, mint_seed,
                                         observed_our_lines, set_active_builder,
                                         uninstall_choice_tap)


@pytest.fixture(autouse=True)
def _clear_tap():
    """Leave the tree UNPATCHED. A class patch survives the whole pytest process, so without this
    every bridge test collected after this file would run through the tap."""
    set_active_builder(None)
    yield
    set_active_builder(None)
    uninstall_choice_tap()


def _builder(**kw):
    b = LiveRecordBuilder(battle_format="gen3ou", seed="sodium,ab12", **kw)
    b.set_player("p1", "Trainee", "OUR_TEAM")
    b.set_player("p2", "Bot", "THEIR_TEAM")
    return b


def test_a_record_needs_both_player_payloads():
    b = LiveRecordBuilder(battle_format="gen3ou", seed="sodium,ab12")
    assert not b.ready()
    with pytest.raises(ValueError, match="both >player"):
        b.build()
    b.set_player("p1", "A", "T1")
    assert not b.ready()
    b.set_player("p2", "B", "T2")
    assert b.ready()


def test_the_built_record_is_what_the_replay_driver_READS():
    """`replay_kernels.js` reads exactly two things: `>start`/`>player` out of `input_log`
    (`writeStart`) and the `[side, payload]` pairs in `commands` (`buildToTurn`). Everything else
    on the dataclass is metadata, which is why a live record can be synthesized at all."""
    b = _builder()
    b.add_command("p1", "switch Magneton")
    b.add_command("p2", "move earthquake")
    rec = b.build()

    assert rec.start_options() == {"formatid": "gen3ou", "seed": "sodium,ab12"}
    assert rec.players() == {"p1": {"name": "Trainee", "team": "OUR_TEAM"},
                             "p2": {"name": "Bot", "team": "THEIR_TEAM"}}
    assert rec.commands == (("p1", "switch Magneton"), ("p2", "move earthquake"))
    assert rec.packed_team("p2") == "THEIR_TEAM"
    assert rec.side_of("Trainee") == "p1"
    assert [ln.split(" ")[0] for ln in rec.input_log] == [">start", ">player", ">player"]


def test_the_command_payloads_match_a_REAL_records_shape():
    """A real bridge record stores `['p1', 'switch Magneton']` / `['p2', 'move earthquake']` —
    the `/choose ` prefix already stripped. The tap records the same bytes, so a synthesized
    record and a `__RECON__` one are interchangeable to the driver."""
    b = _builder()
    b.add_command("p2", "move hiddenpowerice")
    payload = b.build().commands[0][1]
    assert not payload.startswith("/")
    assert payload.split(" ")[0] in ("move", "switch", "team", "default", "pass")


def test_the_record_grows_as_the_battle_plays():
    b = _builder()
    assert b.n_commands == 0 and b.build().commands == ()
    b.add_command("p1", "move surf")
    assert b.n_commands == 1
    assert len(b.build().commands) == 1, "build() must be a snapshot of RIGHT NOW"


def test_our_lines_read_the_live_sink_rather_than_a_second_copy():
    """The runner APPENDS to `chunk_sink` as the battle plays, so reading it is always current.
    A captured copy is one more thing to forget to update — and the gate would then compare the
    world against a stale prefix."""
    sink: list = []
    b = _builder(chunk_sink=sink, our_side="p1")
    assert b.our_lines == []
    sink.append(("p1", "|switch|p1a: X|Salamence, F|100/100\n|turn|1"))
    sink.append(("p2", "|switch|p2a: Y|Skarmory, M|100/100"))
    assert b.our_lines == ["|switch|p1a: X|Salamence, F|100/100", "|turn|1"]


def test_observed_our_lines_filters_by_side():
    sink = [("p1", "A\nB"), ("p2", "C"), ("p1", "D")]
    assert observed_our_lines(sink, "p1") == ["A", "B", "D"]
    assert observed_our_lines(sink, "p2") == ["C"]


def test_a_minted_seed_is_explicit_and_parseable():
    """The seed MUST be pinned: a seedless START makes the child mint one and report it only at
    `__RECON__` time — i.e. after the battle we are trying to search inside."""
    s = mint_seed()
    assert s.startswith("sodium,") and len(s) == len("sodium,") + 32
    assert int(s.split(",")[1], 16) >= 0
    assert mint_seed() != mint_seed()


# -- the tap ------------------------------------------------------------------


def test_only_one_builder_may_be_active_at_a_time():
    """Two overlapping battles would interleave their commands into ONE record, and the result
    replays a battle that never happened while looking perfectly well-formed."""
    b1, b2 = _builder(), _builder()
    set_active_builder(b1)
    assert active_builder() is b1
    with pytest.raises(RuntimeError, match="one battle at a time"):
        set_active_builder(b2)
    set_active_builder(None)
    set_active_builder(b2)          # released, so the next battle may claim it
    assert active_builder() is b2


def test_the_tap_records_the_side_the_client_holds():
    """The tap is installed on `BattleStreamClient._write_choice` — the last point before the
    bytes reach the child — so it sees poke-env's OWN fallbacks (`/choose default` from a None
    predict, a redecide exhaustion) that a `choose_move` return value never shows."""
    import asyncio

    from main.search_dividend.record import install_choice_tap
    from utils.bridge.battle_stream_client import BattleStreamClient

    install_choice_tap()
    b = _builder()
    set_active_builder(b)

    class _Fake:
        _side = "p2"

        async def _write_raw(self, room, command):
            self.sent = command

    fake = _Fake()
    asyncio.run(BattleStreamClient._write_choice(fake, "room", "move earthquake"))
    assert b.build().commands == (("p2", "move earthquake"),)
    assert fake.sent == "CHOOSE p2 move earthquake", "the tap must not swallow the write"


def test_the_tap_is_inert_with_no_active_builder():
    import asyncio

    from main.search_dividend.record import install_choice_tap
    from utils.bridge.battle_stream_client import BattleStreamClient

    install_choice_tap()

    class _Fake:
        _side = "p1"

        async def _write_raw(self, room, command):
            self.sent = command

    fake = _Fake()
    asyncio.run(BattleStreamClient._write_choice(fake, "room", "move surf"))
    assert fake.sent == "CHOOSE p1 move surf"


def test_installing_the_tap_twice_does_not_double_record():
    import asyncio

    from main.search_dividend.record import install_choice_tap
    from utils.bridge.battle_stream_client import BattleStreamClient

    install_choice_tap()
    install_choice_tap()
    b = _builder()
    set_active_builder(b)

    class _Fake:
        _side = "p1"

        async def _write_raw(self, room, command):
            pass

    asyncio.run(BattleStreamClient._write_choice(_Fake(), "room", "move surf"))
    assert b.n_commands == 1, "a doubly-wrapped tap would log every choice twice"


def test_the_start_line_is_valid_json_the_sim_can_parse():
    rec = _builder().build()
    start = next(ln for ln in rec.input_log if ln.startswith(">start "))
    parsed = json.loads(start[len(">start "):])
    assert parsed["formatid"] == "gen3ou"
    assert parsed["seed"].startswith("sodium,")


def test_the_tap_rejects_a_ROLLOUT_battles_commands_by_room():
    """The playoff arm plays whole battles from inside a live decision, and every one of their
    choices passes through the same patched ``_write_choice``. Letting them into this builder would
    produce a record that replays a battle nobody played — and it would look perfectly well-formed
    until the interleaved commands shifted the whole line.

    The filter is on the ROOM and not an on/off switch, because in the mirror the UNSEARCHED side
    commits its real live choice on ``POKE_LOOP`` during exactly the window a rollout occupies: a
    blanket suspend would drop that command and desynchronize the record the other way. Both halves
    are asserted here.
    """
    import asyncio

    from main.search_dividend.record import install_choice_tap
    from utils.bridge.battle_stream_client import BattleStreamClient

    install_choice_tap()
    b = _builder()
    b.battle_tag = "battle-gen3ou-7"
    set_active_builder(b)

    class _Fake:
        def __init__(self, side):
            self._side = side
            self.sent = None

        async def _write_raw(self, room, command):
            self.sent = command

    live_us, live_them, rollout = _Fake("p1"), _Fake("p2"), _Fake("p1")
    asyncio.run(BattleStreamClient._write_choice(live_us, "battle-gen3ou-7", "move surf"))
    # a paired rollout's own battle, in flight at the same time
    asyncio.run(BattleStreamClient._write_choice(rollout, "battle-gen3ou-8", "move explosion"))
    # the mirror's unsearched side, committing its LIVE choice inside the rollout's window
    asyncio.run(BattleStreamClient._write_choice(live_them, "battle-gen3ou-7", "switch 3"))

    assert b.build().commands == (("p1", "move surf"), ("p2", "switch 3"))
    # the write itself is never swallowed for anyone — the tap only observes
    assert rollout.sent == "CHOOSE p1 move explosion"


def test_an_untagged_builder_still_accepts_its_first_commands():
    """Before our first decision the battle TAG is not known (``run_local_battles`` mints it), and
    the opponent may commit turn 1 first. Rejecting an unknown room would silently lose that
    command — and a rollout can only ever start from INSIDE a decision, i.e. after the tag has been
    adopted, so accepting here cannot let a rollout in."""
    b = _builder()
    assert b.battle_tag is None
    assert b.accepts_room("battle-gen3ou-1")
    b.battle_tag = "battle-gen3ou-1"
    assert b.accepts_room("battle-gen3ou-1")
    assert not b.accepts_room("battle-gen3ou-2")
