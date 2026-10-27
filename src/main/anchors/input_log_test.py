"""``main.anchors.input_log`` (the S-U gate test's kept states): the window parse, what is recorded and what is not,
and that a record is the live searched player's own cut and keying (``search_side.honest_log``, ``LiveSearcher.key``),
so the offline search runs at exactly the decision a searched game would have searched. Stubs, no game."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from main.anchors import input_log as IL
from main.anchors.search_side import KEY_SCHEMA, deal_key, honest_log

TEAM1 = "Blissey||leftovers|naturalcure|seismictoss,icebeam|Bold|,,252,244,,12||,0,,,,|||"
TEAM2 = "Skarmory|||keeneye|spikes,protect|Careful|248,,,,248,12||,0,,,,|||"


def _front():
    start = {"formatid": "gen3ou", "seed": [1, 2, 3, 4], "p1": {"name": "Us", "team": TEAM1},
             "p2": {"name": "Them", "team": TEAM2}}
    us, them = object(), object()
    battle = SimpleNamespace(tag="battle-gen3ou-1", index=0, start_doc=start, conns={"p1": us, "p2": them},
                             req_n={"p1": 3, "p2": 3},
                             choice_log=[(1, "p1", "CHOOSE p1 move seismictoss"), (1, "p2", "CHOOSE p2 move spikes"),
                                         (2, "p2", "CHOOSE p2 move protect"), (2, "p1", "CHOOSE p1 move icebeam"),
                                         (3, "p2", "CHOOSE p2 move spikes")])
    return SimpleNamespace(battles={battle.tag: battle}), battle, us


def _d(turn: int, n: int = 5):
    return {"turn": turn, "n": n, "tokens": {"0": "move seismictoss", "1": "move icebeam"},
            "frame": {"b64": "QUJD", "dtype": "<f4", "shape": [1]}}


def test_the_window_is_FROM_TO_inclusive_and_canonical():
    assert IL.parse_window("10-30") == (10, 30) and IL.canonical(" 010-30 ") == "10-30"
    for bad in ("10", "30-10", "0-5", "a-b", "10-30-40"):
        with pytest.raises(IL.InputLogError):
            IL.parse_window(bad)


def test_a_record_is_the_live_searchers_cut_and_key_and_nothing_outside_the_window(tmp_path):
    front, battle, us = _front()
    rec = IL.InputLogRecorder("10-30", tmp_path / "input_logs.jsonl")
    rec.record(front, battle.tag, _d(9), [1, 1, 0], us, 0)          # before the window
    rec.record(front, battle.tag, _d(31), [1, 1, 0], us, 0)         # after it
    rec.record(front, battle.tag, _d(12), [1, 0, 0], us, 0)         # one legal action
    rec.record(front, battle.tag, _d(12, 7), [1, 1, 0], us, 1)      # recorded
    rec.record(front, battle.tag, _d(30, 9), [0, 1, 1], us, 2)      # the window's last turn: recorded
    rows = [json.loads(x) for x in (tmp_path / "input_logs.jsonl").read_text().splitlines()]
    assert [(r["turn"], r["n"]) for r in rows] == [(12, 7), (30, 9)]
    assert rec.summary() == {"window": "10-30", "decisions": 5, "in_window": 3, "recorded": 2, "one_legal": 1,
                             "path": str(tmp_path / "input_logs.jsonl")}
    r = rows[0]
    log, at, update = honest_log(battle, "p1")
    assert r["log"] == log and r["at"] == at == 4 and r["update"] == update == 3
    assert not any("p2 move spikes" in c and i >= 4 for i, c in enumerate(r["log"]["cmds"])), \
        "the opponent's choice of OUR update is never in the log"
    deal = deal_key(battle.start_doc, "p1")
    assert r["deal"] == deal and r["key"] == f"{KEY_SCHEMA}|{deal}|t12|n7"
    assert r["schema"] == IL.SCHEMA and r["side"] == "p1" and r["pi_argmax"] == 1 and r["n_legal"] == 2
    assert r["frame_sha256"] == IL.frame_sha256("QUJD") and r["tokens"] == {"0": "move seismictoss", "1": "move icebeam"}


def test_a_connection_that_is_not_a_side_is_a_named_failure(tmp_path):
    from main.anchors.search_side import LiveSearchError

    front, battle, _us = _front()
    rec = IL.InputLogRecorder("10-30", tmp_path / "x.jsonl")
    with pytest.raises(LiveSearchError, match="not a side"):
        rec.record(front, battle.tag, _d(12), [1, 1], object(), 0)
