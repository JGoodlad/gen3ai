"""The LEAK-HONEST builder end to end (:mod:`main.value_search.honest_worlds`) on a banked U decision (the leak test's
fixture): J worlds, each replayed on its own fresh-dice schedule through every ``core_events`` check to exactly I's
lines and the live root row (the strict gate, inside the build), each opened by the playout tool at its root with the
live decision's tokens; the build is deterministic and keyed by I alone; the scheduled world log replays the same in
both replays (``core_events`` START ``reseed`` and ``rust_env``'s ``Log.reseed``)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from main.value_search import honest_inv as INV
from main.value_search import info_set as IS
from main.value_search.honest_worlds import HonestSampler

FIXTURE = Path(__file__).with_name("honest_inv_fixtures") / "cases.json.gz"

pytestmark = [pytest.mark.slow, pytest.mark.sim]


@pytest.fixture(scope="module")
def built():
    fx = INV.load_fixture(FIXTURE)
    d = INV.Decision(**next(c for c in fx["cases"] if c["channel"] == "L3")["d"])
    info, row = IS.extract(d.log, d.at, d.side)
    smp = HonestSampler()
    return d, info, row, smp.sample_info(info, 8, row), HonestSampler().sample_info(info, 8, row)


def test_eight_worlds_each_reproducing_I(built):
    d, info, row, w, _again = built
    assert len(w.logs) == 8 and len(set(map(len, (lg["cmds"] for lg in w.logs)))) == 1
    for lg in w.logs:
        got, r = IS.extract(lg, w.at, d.side)
        # a world ships I's lines and commands and the live row; its copy of our team names every gender (C12)
        assert got.lines == info.lines and got.cmds == info.cmds and r == row
        assert got.team == IS.own_genders(info)
        assert lg["seed"] != d.log["seed"] and lg["reseed"], "every world is on its own fresh dice"


def test_the_build_is_deterministic_and_keyed_by_I(built):
    _d, info, _row, w, again = built
    assert json.dumps(w.logs, sort_keys=True) == json.dumps(again.logs, sort_keys=True)
    assert w.leak["key"] == info.key()


def test_the_playout_tool_opens_every_world_at_the_live_decision(built):
    from main.value_search.playout import LeafEngine

    d, _info, _row, w, _again = built
    eng = LeafEngine(INV.stub_forward(), n_cores=1)
    try:
        for lg in w.logs:
            root = eng.root(lg, w.at, d.side)
            assert {str(k): str(v) for k, v in root["tokens"].items()} == d.tokens
    finally:
        eng.close()
