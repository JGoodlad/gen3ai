"""The live client's FRAMING and RULES, on a fake reader (no binary, no socket).

What is pinned: the declared non-battle lines never reach the reader; every other line does (an unknown keyword is
the READER's to refuse, never skipped here); lines before our side is known are held and fed with the first write;
a frame becomes one ``/choose <token>|<rqid>`` with the token noted on the reader FIRST; the stall forfeit; and every
T28 trigger raises :class:`LiveParseHalt` carrying the battle id, the offending lines and the received stream.
"""
from typing import List, Optional

import numpy as np
import pytest

from main.live.client import ALLOWED_COMMANDS, LiveBattle, LiveClient, ClientConfig, ClientError
from main.live.halt import LiveParseHalt
from main.live.reader import ROOM_SKIP, Frame, ReaderRefusal, is_room_line, keyword, split_room_message


class FakeReader:
    def __init__(self, frames=None, refuse_on: Optional[str] = None):
        self.opened = None
        self.fed: List[List[str]] = []
        self.chosen: List[str] = []
        self.frames = list(frames or [])
        self.refuse_on = refuse_on
        self.turn = 0
        self.events: List[str] = []

    def open(self, side, name, team):
        self.opened = (side, name, team)

    def feed(self, lines):
        self.fed.append(list(lines))
        self.events.append("feed")
        if self.refuse_on and any(self.refuse_on in ln for ln in lines):
            raise ReaderRefusal(f"line: unknown keyword {self.refuse_on!r}", "refusal")
        if any(ln.startswith("|request|") for ln in lines) and self.frames:
            f = self.frames.pop(0)
            self.turn = f.turn
            return f
        return None

    def choose(self, token):
        self.chosen.append(token)
        self.events.append("choose")


def frame(turn=3, rqid=7, mask=(1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0), tokens=None):
    tokens = tokens if tokens is not None else {0: "move 1", 1: "move 2"}
    return Frame(side="p1", row=np.zeros(4, np.float32), mask=np.asarray(mask, np.int8), tokens=tokens,
                 turn=turn, line=10, rqid=rqid, n=0)


class Pick:
    def __init__(self, idx=1):
        self.idx = idx

    def choose(self, frame):
        return self.idx


def battle(reader, policy=None, limit=250):
    sent: List[str] = []
    b = LiveBattle("battle-gen3ou-9", our_name="P4 A", packed_team="TEAM", reader=reader,
                   policy=policy or Pick(), forfeit_turn_limit=limit, send=sent.append)
    return b, sent


START = ["|init|battle", "|title|P4 A vs. P4 B", "|j|☆P4 A", "", "|t:|1", "|gametype|singles"]


def test_declared_room_lines_are_skipped_and_everything_else_is_fed():
    r = FakeReader()
    b, _ = battle(r)
    b.on_lines(START)
    b.on_lines(["|player|p1|P4 A|102|", "|c|☆P4 B|hello", "|raw|<b>x</b>", "|inactive|Battle timer is ON",
                "|player|p2|P4 B|266|", "|-weird|p2a: X"])
    assert r.opened == ("p1", "P4 A", "TEAM")
    fed = [ln for w in r.fed for ln in w]
    # the held pre-side lines come first, in order; "" (framing) and the room lines never reach the reader
    assert fed == ["|t:|1", "|gametype|singles", "|player|p1|P4 A|102|", "|player|p2|P4 B|266|", "|-weird|p2a: X"]
    assert all(not is_room_line(ln) for ln in fed)


def test_the_room_skip_list_is_the_declared_one():
    assert {"c", "j", "l", "raw", "html", "uhtml", "inactive", "init", "title"} <= ROOM_SKIP
    # battle protocol is never on it — skipping one would be a silent skip (T28)
    assert not {"request", "move", "switch", "-damage", "turn", "player", "win", "error", "t:", "", "-message"} & ROOM_SKIP
    assert keyword("|") == "" and keyword("|t:|1") == "t:" and keyword("plain") is None


def test_a_frame_becomes_a_choice_with_the_token_noted_first():
    r = FakeReader([frame(rqid=12)])
    b, sent = battle(r, Pick(1))
    b.on_lines(["|player|p1|P4 A|1|"])
    b.on_lines(['|request|{"rqid":12}'])
    assert r.chosen == ["move 2"]
    assert r.events[-1] == "choose"
    assert sent == ["battle-gen3ou-9|/choose move 2|12"]
    assert b.decisions == 1


def test_the_stall_forfeit_fires_at_the_trainers_turn_limit():
    r = FakeReader([frame(turn=250)])
    b, sent = battle(r, limit=250)
    b.on_lines(["|player|p1|P4 A|1|", '|request|{"rqid":1}'])
    assert sent == ["battle-gen3ou-9|/forfeit"] and r.chosen == [] and b.forfeited


@pytest.mark.parametrize("case", ["reader", "invalid_choice", "illegal", "no_tokens", "policy_raises"])
def test_every_T28_trigger_raises_LiveParseHalt_with_the_evidence(case):
    lines = ["|player|p1|P4 A|1|", '|request|{"rqid":1}']
    if case == "reader":
        r, pol = FakeReader(refuse_on="|zzz|"), Pick()
        lines = ["|player|p1|P4 A|1|", "|zzz|p2a: X"]
    elif case == "invalid_choice":
        r, pol = FakeReader(), Pick()
        lines = ["|error|[Invalid choice] Can't move: Foo's move 9 is invalid"]
    elif case == "illegal":
        r, pol = FakeReader([frame()]), Pick(5)
    elif case == "no_tokens":
        r, pol = FakeReader([frame(tokens={}, mask=(0,) * 11)]), Pick(0)
    else:
        class Boom:
            def choose(self, f):
                raise RuntimeError("encoder shape")
        r, pol = FakeReader([frame()]), Boom()
    b, sent = battle(r, pol)
    with pytest.raises(LiveParseHalt) as ei:
        b.on_lines(lines)
    e = ei.value
    assert e.battle_id == "battle-gen3ou-9" and e.offending_lines and e.stream[-len(lines):] == lines
    assert not any("/choose" in s for s in sent)


def test_the_end_stops_reading_and_leaves():
    r = FakeReader()
    b, sent = battle(r)
    b.on_lines(["|player|p1|P4 A|1|", "|", "|win|P4 A"])
    b.on_lines(["|raw|rating", "|zzz|after the end"])  # nothing after the end is read
    assert b.ended and b.result().won is True
    assert sent == ["|/leave battle-gen3ou-9"]
    assert all("|zzz|" not in ln for w in r.fed for ln in w)


def test_split_room_message():
    assert split_room_message(">battle-x\n|t:|1\n") == ("battle-x", ["|t:|1", ""])
    assert split_room_message("|challstr|4|ab") == (None, ["|challstr|4|ab"])


def test_the_outgoing_command_set_has_no_chat_no_ladder():
    assert "/search" not in ALLOWED_COMMANDS and "/msg" not in ALLOWED_COMMANDS and "/pm" not in ALLOWED_COMMANDS
    c = LiveClient(ClientConfig(uri="ws://x", username="a"), policy=Pick(), team_fn=lambda: None)
    for bad in ("|/search gen3ou", "battle-x|hello there", "|/msg someone, hi", "|/reject someone"):
        with pytest.raises(ClientError, match="declared command set"):
            c.send_raw(bad)
