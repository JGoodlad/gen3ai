"""The turn-timer model (:mod:`main.reads.timer_model`): its constants pinned to Showdown's own source (a Showdown bump
that moves one fails here), its replay pinned to the prototype that produced the cost read's ``timer_replay.json``
(on real banked games and on random ones), and the safety net's semantics (the bank-aware deadline never forfeits;
the cut step's overrun delays the next decision).

The fixture ``timer_fixtures/banked_rows.json.gz`` is 90 of S′-HONEST's banked searched games' clock rows (decision
logs only; ``measurements/searched_cost_2026-11-09/inputs/make_timer_fixture.py``) with the prototype's own timer
block on them. The model reproduced the prototype's ``timer_replay.json`` on all 5,100 banked games, every policy
and panel (80 of 80 rows), when it was built."""
from __future__ import annotations

import gzip
import importlib.util
import json
import re

import numpy as np
import pytest

from main.reads import timer_model as TM
from utils.paths import repo_path

ROOM_BATTLE = repo_path("deps", "pokemon-showdown", "server", "room-battle.ts")
PROTO = repo_path("designs", "research_state", "measurements", "searched_cost_2026-11-09", "inputs", "timer_replay.py")
FIXTURE = repo_path("src", "main", "reads", "timer_fixtures", "banked_rows.json.gz")


@pytest.fixture(scope="module")
def src() -> str:
    return ROOM_BATTLE.read_text()


@pytest.fixture(scope="module")
def proto():
    spec = importlib.util.spec_from_file_location("timer_replay_proto", PROTO)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _const(src: str, name: str) -> int:
    m = re.search(rf"^const {name} = (\d+);", src, re.M)
    assert m, f"room-battle.ts no longer declares {name}"
    return int(m.group(1))


def test_the_constants_are_room_battle_ts_literals(src):
    assert TM.TICK_S == _const(src, "TICK_TIME")
    assert TM.STARTING_S == _const(src, "STARTING_TIME")
    assert TM.GRACE_S == _const(src, "STARTING_GRACE_TIME")
    assert TM.MAX_TURN_S == _const(src, "MAX_TURN_TIME")
    # singles: addPerTurn is the second arm of `hasLongTurns ? 25 : 10`, hasLongTurns = not singles
    assert "const hasLongTurns = format.gameType !== 'singles';" in src
    m = re.search(r"addPerTurn: hasLongTurns \? (\d+) : (\d+),", src)
    assert m and TM.ADD_PER_TURN_S == int(m.group(2))
    # the ladder (non-challenge, no timer rule) accelerates; time-out forfeits
    assert "accelerate: !timerSettings && !isChallenge," in src and "timeoutAutoChoose: false," in src
    m = re.search(r"if \(this\.turn > (\d+) && addPerTurn > TICK_TIME\) \{\s*addPerTurn -= TICK_TIME;", src)
    assert m and TM.ACCEL_TURN == int(m.group(1))
    m = re.search(r"if \(this\.turn > (\d+) && Math\.floor\(this\.battle\.requestCount / 2\) % 2\) \{\s*addPerTurn = 0;",
                  src)
    assert m and TM.ACCEL_TURN_2 == int(m.group(1))
    # the bank: starting + grace, a new turn capped at `starting`, a mid-turn request min(addPerTurn, TICK) uncapped,
    # one request's limit min(bank, maxPerTurn), charged in TICK steps
    assert "player.secondsLeft = this.settings.starting + this.settings.grace;" in src
    assert "player.secondsLeft = Math.min(player.secondsLeft + addPerTurn, this.settings.starting);" in src
    assert "const addPerMidTurnRequest = Math.min(this.settings.addPerTurn, TICK_TIME);" in src
    assert "player.turnSecondsLeft = Math.min(player.secondsLeft, maxTurnTime);" in src
    assert "player.secondsLeft -= TICK_TIME;" in src
    assert "starting: isChallenge ? STARTING_TIME_CHALLENGE : STARTING_TIME," in src
    assert "maxPerTurn: isChallenge ? MAX_TURN_TIME_CHALLENGE : MAX_TURN_TIME," in src


def test_the_bank_added_per_turn_follows_accelerate():
    assert [TM.add_for_turn(t) for t in (1, 100, 101, 200, 201, 202)] == [10, 10, 5, 5, 0, 5]


@pytest.fixture(scope="module")
def banked():
    with gzip.open(FIXTURE, "rt") as f:
        return json.load(f)


def _games(doc):
    return [(g["panel"], [{"turn": t, "searched": bool(s), "excluded": bool(x), "wall": w} for t, s, x, w in g["rows"]])
            for g in doc["games"]]


def test_the_replay_reproduces_the_prototype_on_banked_games(banked):
    rep = TM.replay(_games(banked))
    want = banked["prototype_timer"]
    assert set(rep["policies"]) == set(want)
    for name, by in want.items():
        for panel, sh in by.items():
            got = rep["policies"][name][panel]
            assert {k: got[k] for k in sh} == sh, (name, panel)
    # the fixture holds forfeits as played and none under the bank-aware net (the design's headline, in small)
    assert want["as_played"]["U"]["forfeit_share"] > 0 and want["bank_r10+bg"]["U"]["forfeit_share"] == 0


def _random_games(seed: int, n: int):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        turn, rows = 1, []
        for _k in range(int(rng.integers(5, 260))):
            turn += int(rng.random() < 0.85)
            path = turn >= 10 and rng.random() < 0.9
            excl = path and rng.random() < 0.08
            wall = float(rng.choice([rng.exponential(6.0), rng.exponential(120.0)], p=[0.97, 0.03])) if path else 0.0
            rows.append({"turn": turn, "searched": path and not excl, "excluded": excl, "wall": round(wall, 3)})
        out.append(rows)
    return out


@pytest.mark.parametrize("name", sorted(TM.POLICIES))
@pytest.mark.parametrize("bg", (False, True))
def test_simulate_equals_the_prototype_on_random_games(proto, name, bg):
    kw = dict(TM.POLICIES[name])
    for rows in _random_games(20261109, 60):
        assert TM.simulate(rows, background=bg, **kw) == proto.simulate(rows, background=bg, **kw)


def test_the_bank_aware_net_never_forfeits_and_cuts_only_on_the_tail():
    # a 400-s decision on turn 12: as played it forfeits; the net cuts it at min(bank, 150) - 10 - 1 and survives
    rows = [{"turn": t, "searched": t >= 10, "excluded": False, "wall": 400.0 if t == 12 else 3.0} for t in range(1, 40)]
    assert TM.simulate(rows)["forfeit"] == 1
    got = TM.simulate(rows, reserve=TM.RESERVE_S, background=False)
    assert got == {"forfeit": 0, "cut": 1, "cut_searched": 1, "path": 30, "battle_cut": 1}
    # the cut step running on: its overrun (400 - 139) delays the next decisions, which are cut until it is spent
    bg = TM.simulate(rows, reserve=TM.RESERVE_S, background=True)
    assert bg["forfeit"] == 0 and bg["cut"] > got["cut"]


def test_the_clock_records_its_deadline_and_need():
    c = TM.Clock(reserve=10.0, background=True)
    s = c.step(10, True, 300.0)
    assert (s.bank_s, s.limit_s, s.deadline_s, s.need_s, s.cut, s.forfeit) == (210.0, 150.0, 139.0, 300.0, True, False)
    assert s.resp_s == 140.0 and c.bank == 210.0 - 140.0 and c.backlog == 300.0 - 139.0
    s2 = c.step(11, True, 2.0)                                    # the backlog + 2 s: still past the deadline
    assert s2.need_s == pytest.approx(163.0) and s2.cut
    rec = c.step(11, False).record()                              # a decision off the search path: overhead only
    assert rec["cut"] is False and rec["deadline_s"] is None and rec["need_s"] == 0.0


def test_a_policy_name_resolves_and_an_unknown_one_is_refused():
    assert TM.policy_kwargs("bank_r10+bg") == {"reserve": 10.0, "background": True}
    assert TM.policy_kwargs(TM.NET_POLICY)["background"] is True
    with pytest.raises(ValueError):
        TM.policy_kwargs("bank_r20")
