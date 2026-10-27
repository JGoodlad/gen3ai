"""The live searched player on a REAL in-process battle (S-U build B1): two core slots of the Rust front end play a
fixture battle, and at every decision of ours from turn 2 the HONEST input log the front end recorded must replay, in
the playout core and the replay core, to the live decision: the same tokens, turn and decision index, and the same
observation row byte for byte (``LiveSearcher._check_root``). At the first such decision the search itself runs on that
live log, twice, plus once on the log with an update's two choices swapped: one result each time (deterministic per
key, and blind to the order two sides' choices of one update reached the child).

A stub network (a fixed projection of the row: logits, and V on the win scale) plays both sides and is the search's
forward, so the slot's greedy choice is the forward's own top choice. The search runs a tiny honest preset (2 worlds,
2 playouts an action, 1 turn deep) to bound the cost; ``r0s4`` is only a bigger instance of the same code.
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List

import numpy as np
import pytest

from main.anchors import core_side
from main.anchors import search_side as SS
from main.anchors.search_presets import SearchPreset
from main.anchors.session import OurSideState

pytestmark = [pytest.mark.sim, pytest.mark.integration]

#: two gen3ou teams a front-end capture carried (``x52_read``): the sim and Showdown's validator accept both
TEAM_A = ("Tyranitar||choiceband|sandstream|focuspunch,earthquake,hiddenpowerbug,rockslide|Adamant|76,216,36,,,180|M|"
          ",,,,30,30|||]Skarmory||leftovers|keeneye|toxic,protect,spikes,whirlwind|Impish|248,,116,,136,8|M||||]"
          "Blissey||leftovers|naturalcure|seismictoss,icebeam,wish,softboiled|Bold|4,,252,252,,|F||||]Flygon||choiceband|"
          "levitate|earthquake,rockslide,hiddenpowerghost,fireblast|Jolly|4,252,,,,252|M|,30,,,30,|||]Zapdos||leftovers|"
          "pressure|roar,rest,sleeptalk,thunderbolt|Calm|252,,24,,204,28|||||]Gengar||leftovers|levitate|thunderbolt,"
          "icepunch,willowisp,taunt|Naive|96,96,96,40,,180|M||||")
TEAM_B = ("Swampert||leftovers|torrent|rest,curse,earthquake,icebeam|Relaxed|240,,136,,84,48|||||]Magneton||magnet|"
          "magnetpull|substitute,toxic,thunderbolt,hiddenpowerfire|Modest|4,,,252,,252||,30,,30,,30|||]Celebi||leftovers|"
          "naturalcure|leechseed,calmmind,gigadrain,psychic|Modest|252,,,220,,36||,0,,,,|||]Snorlax||leftovers|immunity|"
          "bodyslam,earthquake,shadowball,selfdestruct|Adamant|32,136,152,,172,16|||||]Metagross||leftovers|clearbody|"
          "toxic,meteormash,earthquake,explosion|Adamant|252,176,20,,28,32|||||]Charizard||leftovers|blaze|dragonclaw,"
          "flamethrower,hiddenpowergrass,brickbreak|Mild|,4,,252,,252||,30,,30,,|||")
TINY = SearchPreset(name="tiny", n_worlds=2, per_action=2, depth=1, rule="pikl", pikl_lambda=0.015, reveal="off",
                    dice="seed", max_tries=30)


def _stub(rows: np.ndarray, masks: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(7)
    proj = rng.standard_normal((rows.shape[1], 12)).astype(np.float32) * 0.05
    out = rows @ proj
    out[:, 11] = 1.0 / (1.0 + np.exp(-out[:, 11].astype(np.float64)))     # V on the win scale, varying with the state
    return out.astype(np.float32)


class _Greedy:
    """The slot's acting policy: the stub's masked argmax (ties to the lower index), as ``CorePolicy`` greedy."""

    stochastic = False

    def decide(self, row, mask):
        lg = np.where(np.asarray(mask).astype(bool), _stub(np.asarray(row, np.float32)[None],
                                                           np.asarray(mask)[None])[0, :11], -np.inf)
        a = int(np.argmax(lg))
        return a, a


class _Team:
    def __init__(self, packed: str) -> None:
        self.packed = packed

    def yield_team(self) -> str:
        return self.packed


class _Probe(SS.LiveSearcher):
    """Checks the root at EVERY decision from turn 2; searches (three ways) at the first one."""

    def __init__(self, *a: Any, **k: Any) -> None:
        super().__init__(*a, **k)
        self.checked = 0
        self.searches: List[Dict[str, Any]] = []

    def _search(self, log, at, side, row, mask, d, key, argmax):
        self._check_root(log, at, side, d, key)
        self.checked += 1
        if self.searches:
            return {"searched": False, "why": "probe", "played": argmax}
        real = SS.LiveSearcher._search
        out = [real(self, log, at, side, row, mask, d, key, argmax) for _ in range(2)]
        swapped = dict(log, cmds=list(log["cmds"]))
        swapped["cmds"][0], swapped["cmds"][1] = swapped["cmds"][1], swapped["cmds"][0]   # update 1: p2's first
        assert swapped["cmds"] != log["cmds"]
        out.append(real(self, swapped, at, side, row, mask, d, key, argmax))
        self.searches = out
        return out[0]


def _same(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    keys = ("searched", "played", "deviation", "q", "log_pi", "root_v", "spent", "world_tries", "species_hit")
    return {k: a.get(k) for k in keys} == {k: b.get(k) for k in keys}


def test_the_honest_log_of_a_live_battle_replays_to_every_decision_and_the_search_is_deterministic(tmp_path):
    from utils.bridge.ws_frontend import ShowdownFrontEnd
    from utils.rust_env import ffi as F
    from utils.rust_env.build import ensure_built

    lib = F.load(ensure_built("selfcheck")[1], nan_poison=True)

    async def play() -> _Probe:
        front = ShowdownFrontEnd(impl="rust", seed_base=11, validate_teams=False, battle_index_base=3)
        st_a, st_b = OurSideState(), OurSideState()
        probe = _Probe(None, _Greedy(), "r0s4@2", tmp_path / "search_decisions.jsonl", engine_lib=lib,
                       forward=_stub, preset=TINY)
        a = core_side.make_slot(front, username="Alpha", policy=_Greedy(), team_source=_Team(TEAM_A), state=st_a,
                                forfeit_limit=9, searcher=probe)
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
            assert len(st_a.records) == 1
            battle = front.battles.get(st_a.records[0].battle_tag) or front.captured_battles[-1]
            assert battle.index == 3                                  # the series' battle 3: its seed index
        finally:
            await front.close()
            probe.close()
        return probe

    probe = asyncio.run(play())
    assert probe.checked >= 4, "the fixture battle never reached enough decisions from turn 2"
    a, b, swapped = probe.searches
    assert a["searched"] is True, a
    assert len(set(a["q"].values())) > 1, a          # the leaf values differ: the comparison below has teeth
    assert _same(a, b), (a, b)
    assert _same(a, swapped), (a, swapped)
    recs = [json.loads(ln) for ln in (tmp_path / "search_decisions.jsonl").read_text().splitlines()]
    assert recs[0]["turn"] < 2 and not recs[0]["searched"]
    assert any(r.get("searched") for r in recs)
    assert all(r["played"] == r["pi_argmax"] for r in recs if not r.get("searched"))
