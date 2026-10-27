"""The S-U gate test's kept input logs on a REAL in-process battle: a RAW core slot with the recorder plays a fixture
battle; every decision it kept must then pass the offline scorer's root guard (``sign_gate.GateScorer.root``: the
kept log's replay reproduces the decision's tokens and the live row's sha256), and the scorer runs the search and, on
a deviation, the self-referee on one of them. A tampered row hash or token is refused by name. A stub network plays
both sides and is the forward (``search_side_integration_test``'s); the search runs its tiny honest preset."""
from __future__ import annotations

import asyncio
import json

import pytest

from main.anchors import core_side
from main.anchors import search_side as SS
from main.anchors.input_log import InputLogRecorder
from main.anchors.search_side_integration_test import TEAM_A, TEAM_B, TINY, _Greedy, _stub, _Team
from main.anchors.session import OurSideState
from main.value_search import sign_gate as SG

pytestmark = [pytest.mark.sim, pytest.mark.integration, pytest.mark.slow]


def test_every_kept_decision_replays_to_the_live_row_and_scores(tmp_path, monkeypatch):
    from utils.bridge.ws_frontend import ShowdownFrontEnd
    from utils.rust_env import ffi as F
    from utils.rust_env.build import ensure_built

    lib = F.load(ensure_built("selfcheck")[1], nan_poison=True)
    out = tmp_path / "input_logs.jsonl"

    async def play() -> int:
        front = ShowdownFrontEnd(impl="rust", seed_base=11, validate_teams=False, battle_index_base=3)
        st_a, st_b = OurSideState(), OurSideState()
        rec = InputLogRecorder("2-12", out)
        a = core_side.make_slot(front, username="Alpha", policy=_Greedy(), team_source=_Team(TEAM_A), state=st_a,
                                forfeit_limit=13, recorder=rec)
        b = core_side.make_slot(front, username="Beta", policy=_Greedy(), team_source=_Team(TEAM_B), state=st_b,
                                forfeit_limit=250)
        try:
            await a.login()
            await b.login()
            tb = asyncio.create_task(b.accept_series("Alpha", 1))
            await asyncio.sleep(0)
            await asyncio.wait_for(a.challenge_series("Beta", 1, 120.0), timeout=900.0)
            await asyncio.wait_for(tb, timeout=120.0)
            a.check()
            b.check()
        finally:
            await front.close()
        return rec.summary()["recorded"]

    n = asyncio.run(play())
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert n == len(rows) >= 4, "the fixture battle kept too few decisions to test"
    searcher = SS.LiveSearcher(None, _Greedy(), "r0s4@2", tmp_path / "unused.jsonl", engine_lib=lib, forward=_stub,
                               preset=TINY)
    sc = SG.GateScorer("unused", searcher=searcher)
    monkeypatch.setattr(SG, "REFEREE_SLICE", (16, 18))       # 2 referee seeds: the test bounds the cost
    try:
        for r in rows:
            sc.root(r)                                         # every kept decision replays to the live row
        bad = dict(rows[0], frame_sha256="0" * 64)
        with pytest.raises(SG.GateError, match="live row"):
            sc.root(bad)
        bad = dict(rows[0], tokens={"0": "move nothing"})
        with pytest.raises(SG.GateError, match="kept decision"):
            sc.root(bad)
        meta = {"panel": "home72", "cell": "c", "unit_seed": 0, "select_hash": "h"}
        scored = [sc.score({**r, **meta}, flips=(i == 0)) for i, r in enumerate(rows[:4])]
    finally:
        sc.close()
    assert all(s["search"]["searched"] or s["search"]["why"] == "excluded" for s in scored)
    assert "speed_flip" in scored[0] and scored[0]["speed_flip"]["ok"]
    for s in scored:
        dev = s["search"].get("deviation")
        assert ("referee" in s) == bool(dev)
        if dev:
            assert set(s["referee"]["out"]) == {str(s["pi_argmax"]), str(s["search"]["played"])}
            assert all(len(v) == 2 for v in s["referee"]["out"].values())
