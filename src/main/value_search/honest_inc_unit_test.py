"""``main.value_search.honest_inc``'s pure pieces (no engine): the carry's state round trip, the replay boundary, the
preset's spelling. The builder itself, on the engine: ``honest_inc_test.py`` (slow)."""
from __future__ import annotations

import json

import pytest

from main.anchors import search_presets as SP
from main.value_search import honest_inc as HI


def test_a_state_survives_the_carry_byte_for_byte():
    st = ("leftovers", "naturalcure", "calm", (252, 0, 4, 0, 252, 0), "fire/2")
    assert HI._state_of(json.loads(json.dumps(HI._state_json(st)))) == st
    st2 = ("choiceband", "intimidate", "adamant", (4, 252, 0, 0, 0, 252), None)
    got = HI.coords_of([["Salamence", HI._state_json(st2), ["dragonclaw", "earthquake"]]])
    assert got == {"Salamence": (st2, ["dragonclaw", "earthquake"])}


class _Sc:
    def __init__(self, header):
        self.header = json.dumps(header)


def _hdr(segs, intents, backtrack=None, hp=None):
    n = len(segs)
    return {"segments": segs, "intents": intents, "backtrack": backtrack or [[]] * n, "hp_last": hp or [[]] * n}


def test_the_boundary_is_the_first_segment_whose_header_moved():
    a = _hdr([["s0"], ["s1"], ["s2"]], [None, "move a", None])
    b = _hdr([["s0"], ["s1"], ["s2"], ["s3"]], [None, "move a", None, "move b"])
    assert HI.boundary(_Sc(a), _Sc(b)) == 3                       # nothing moved: the whole earlier I replays
    c = _hdr([["s0"], ["s1"], ["s2"], ["s3"]], [None, "move a", "move c", None])
    assert HI.boundary(_Sc(a), _Sc(c)) == 2                       # a stand-in became an intent
    d = _hdr([["s0"], ["s1"], ["s2"], ["s3"]], [None, "move a", None, None], backtrack=[[], [], [1], []])
    assert HI.boundary(_Sc(a), _Sc(d)) == 2
    with pytest.raises(RuntimeError):
        HI.boundary(_Sc(a), _Sc(_hdr([["s0"], ["x"], ["s2"], ["s3"]], [None] * 4)))


def test_the_preset_is_r0s4_honest_but_its_dice():
    inc, hon = SP.parse("r0s4_honest_inc@10")[0], SP.parse("r0s4_honest")[0]
    assert inc.dice == "fresh_inc" and hon.dice == "fresh"
    assert {k: v for k, v in inc.as_dict().items() if k not in ("name", "dice")} == \
           {k: v for k, v in hon.as_dict().items() if k not in ("name", "dice")}
    assert SP.canonical("r0s4_honest_inc") == "r0s4_honest_inc@10"


def test_a_packed_carry_unpacks_byte_for_byte():
    c = json.dumps({"schema": HI.CARRY_SCHEMA, "key": "k", "worlds": [{"team": "Snorlax|||" * 40}]})
    assert HI.unpack(HI.pack(c)) == c and HI.pack(c) == HI.pack(c)


def test_a_stuck_battle_backs_off_its_full_attempts():
    assert [k for k in range(70) if HI.full_attempt(k)] == [0, 4, 8, 16, 32, 64]


def test_the_warm_chain_waits_after_a_failure_unless_it_is_needed():
    assert [k for k in range(20) if HI.chain_due(k, False)] == [0, 4, 8, 16]
    assert all(HI.chain_due(k, True) for k in range(20))
