"""V-RANK's shadow arms on a REAL in-process battle (``v_rank.Scorer``): a raw core slot keeps its decisions, and on
each the scorer's search must hand back one leaf value per world for EVERY legal action whose mean is the search's
own Q̂ (the tap on ``engine.leaves`` reads what the search read, no second evaluation), the same Q̂ a stand-alone
``GateScorer`` search returns on that record, and ``oracle_leaf8`` an outcome string per legal action. A stub network
plays both sides and is the forward (``search_side_integration_test``'s); the search runs its tiny honest preset."""
from __future__ import annotations

import asyncio
import json

import numpy as np
import pytest

from main.anchors import core_side
from main.anchors import search_side as SS
from main.anchors.input_log import InputLogRecorder
from main.anchors.search_side_integration_test import TEAM_A, TEAM_B, TINY, _Greedy, _stub, _Team
from main.anchors.session import OurSideState
from main.value_search import sign_gate as SG
from main.value_search import v_rank as V

pytestmark = [pytest.mark.sim, pytest.mark.integration, pytest.mark.slow]


def test_the_scorer_values_every_legal_action_per_world_and_referees_them(tmp_path, monkeypatch):
    from utils.bridge.ws_frontend import ShowdownFrontEnd
    from utils.rust_env import ffi as F
    from utils.rust_env.build import ensure_built

    lib = F.load(ensure_built("selfcheck")[1], nan_poison=True)
    out = tmp_path / "input_logs.jsonl"

    async def play() -> int:
        front = ShowdownFrontEnd(impl="rust", seed_base=11, validate_teams=False, battle_index_base=3)
        rec = InputLogRecorder("2-12", out)
        a = core_side.make_slot(front, username="Alpha", policy=_Greedy(), team_source=_Team(TEAM_A),
                                state=OurSideState(), forfeit_limit=13, recorder=rec)
        b = core_side.make_slot(front, username="Beta", policy=_Greedy(), team_source=_Team(TEAM_B),
                                state=OurSideState(), forfeit_limit=250)
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
    for i, r in enumerate(rows):
        r.update(panel="home72", cell="c", unit_seed=0, select_hash=V.select_hash(r["deal"], r["turn"], r["n"]),
                 stratum="A", team="team_a.txt")
    rows = [r for r in rows if r["n_legal"] >= 2][:3]
    assert len(rows) == 3
    searcher = SS.LiveSearcher(None, _Greedy(), "r0s4@2", tmp_path / "unused.jsonl", engine_lib=lib, forward=_stub,
                               preset=TINY)
    sc = V.Scorer("unused", "r0s4@2", gate=SG.GateScorer("unused", searcher=searcher))
    monkeypatch.setattr(V, "ORACLE_SEEDS", 2)                 # 2 referee seeds an action: the test bounds the cost
    try:
        scored = [sc.score(r) for r in rows]
        again = [searcher._search(r["log"], int(r["at"]), r["side"],
                                  *_root(sc, r), r["key"], int(r["pi_argmax"])) for r in rows]
    finally:
        sc.close()
    searched = [s for s in scored if s["search"]["searched"]]
    assert searched, [s["search"].get("why") for s in scored]
    for s, twin in zip(scored, again):
        assert s["schema"] == V.STATE_SCHEMA and s["key"] and s["stratum"] == "A" and s["team"] == "team_a.txt"
        if not s["search"]["searched"]:
            assert "worlds" not in s and "referee" not in s
            continue
        q = s["search"]["q"]
        legal = sorted(q, key=int)
        assert len(legal) >= 2 and set(s["worlds"]) == set(legal), "every legal action is valued, in every world"
        n_w = {len(v) for v in s["worlds"].values()}
        assert len(n_w) == 1 and n_w.pop() >= 2
        for a in legal:
            assert abs(float(np.mean(s["worlds"][a])) - float(q[a])) < 1e-5     # the tap is the search's own leaves
        assert twin["q"] == q, "the tap changed the search"
        ref = s["referee"]
        assert ref["preset"] == V.ORACLE and set(ref["out"]) == set(legal) == set(ref["q_self"])
        assert all(len(v) == 2 and set(v) <= set("+0-") for v in ref["out"].values())


def _root(sc, rec):
    from agents.battle.core_obs import wrap_row

    _truth, d = sc.g.root(rec)
    return np.array(wrap_row(d["frame"]), dtype=np.float32), np.asarray(rec["mask"], dtype=np.int8), d
