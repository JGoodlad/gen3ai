"""`main.prober.core_walk` without the binary: the core_events script, the replay-log dispatch, the
result parser's refusals and the stall rule. The walk itself — and its equality with the live
Python recorder and the poke-env materializer, until P6 slice 6d-1 retired that oracle — is now
`core_trace_integration_test.py` (`sim`, core-only).
Milliseconds; unmarked.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from main.prober.core_walk import (PLAYER_IGNORED, CoreWalkError, StreamDecision, from_result, replay_log,
                                   script)
from utils.bridge.reconstruction import ReconstructionRecord


def _record(commands) -> ReconstructionRecord:
    return ReconstructionRecord(
        format_id="gen3ou", prng_seed="1,2,3,4",
        input_log=('>start {"formatid":"gen3ou","seed":"1,2,3,4"}',
                   '>player p1 {"name":"a","team":"TA"}', '>player p2 {"name":"b","team":"TB"}'),
        commands=tuple(tuple(c) for c in commands), battle_tag="core-1-x-g0", trainee_username="a")


def test_the_script_replays_the_raw_start_seed_and_every_command_in_order():
    s = script(_record([["p1", "move earthquake"], ["p2", "switch Jolteon"], ["forcelose", "p1"]]))
    start = json.loads(s[0][len("START "):])
    assert start["seed"] == "1,2,3,4" and start["formatid"] == "gen3ou"
    assert start["p1"] == {"name": "a", "team": "TA"} and start["p2"] == {"name": "b", "team": "TB"}
    assert s[1:] == ["CHOOSE p1 move earthquake", "CHOOSE p2 switch Jolteon", "FORCELOSE p1", "END"]
    with pytest.raises(CoreWalkError, match="unknown reconstruction command"):
        script(_record([["p3", "move x"]]))


def test_the_replay_log_is_the_players_dispatch():
    lines = ["|init|battle", "|t:|17", "|player|p1|a||", "", "|request|{\"x\":1}", "|", "|move|p1a: X|Tackle|p2a: Y",
             "|error|[Unavailable choice] Can't switch", "|bigerror|x", "|expire|", "|uhtmlchange|x", "plain text",
             "|win|a", "|win|a"]
    assert set(PLAYER_IGNORED) == {"t:", "expire", "uhtmlchange"}
    assert replay_log(lines) == ("|init|battle", "|player|p1|a||", "|", "|move|p1a: X|Tackle|p2a: Y", "|win|a")
    # the room framing opens the log when the engine stream does not carry it; a tie is logged once
    assert replay_log(["|gametype|singles", "|tie", "|tie"]) == ("|init|battle", "|gametype|singles", "|tie")


def test_a_result_without_the_walk_fields_is_refused():
    res = {"ok": True, "chunks": [], "ended": True, "winner": "p1",
           "trackers": [[{"after": 3, "reward": 0.0, "trackers": {}, "window": []}], []]}
    with pytest.raises(CoreWalkError, match="not run with --walk"):
        from_result(res, "p1")


def test_the_legal_choices_follow_the_mask():
    d = StreamDecision(k=0, turn=1, mask=np.array([0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.int8),
                       tokens={1: "switch B", 6: "move tackle", 7: "move growl"}, obs=None)
    assert d.choices == {1: "switch B", 6: "move tackle"}


def test_only_the_trainees_stall_forfeit_may_add_one_decision():
    from main.prober.core_trace import _stall_extra

    def walk(turns):
        return SimpleNamespace(decisions=[SimpleNamespace(view=SimpleNamespace(turn=t)) for t in turns])

    forfeit = [("p2", "move x"), ("forcelose", "p1")]
    assert _stall_extra(walk([1, 2, 250]), 2, 250, "p1", forfeit)
    assert not _stall_extra(walk([1, 2, 249]), 2, 250, "p1", forfeit)          # below the threshold
    assert not _stall_extra(walk([1, 2, 250]), 2, 250, "p1", [("forcelose", "p2")])  # the other side
    assert not _stall_extra(walk([1, 2, 250, 251]), 2, 250, "p1", forfeit)       # two extra decisions
