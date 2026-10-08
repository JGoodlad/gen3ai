"""The in-process CORE SLOT (P3 of the poke-env retirement): our side of an anchor read deciding on
``sim_bridge``'s core frame, speaking the client protocol into the front end by function call.

No battle, no model, no peer: a recording fake front end and a stub policy, so each protocol rule
is pinned on its own — the choice token and the spliced ``rqid`` reach ``/choose``; the trainer's
forfeit fires at the limit; a ``|win|`` books one record; and every protocol surprise (a live
request with no frame, a frame nobody consumed, a frame out of order, an unwritten NaN cell, a
choice the mask forbids, an ``[Invalid choice]``, a popup) is a NAMED ``core_slot_error``.
The real thing — against a real Metamon on the in-process front end — is
``anchors_integration_test.py``'s routine smoke, run with poke-env BLOCKED.
"""
from __future__ import annotations

import asyncio
import base64
import json
from typing import Any, List

import numpy as np
import pytest

from agents.battle.core_obs import obs_dim
from main.anchors import core_side
from main.anchors.session import OurSideState, SeriesFailure

TAG = "battle-gen3ou-7"


class _Front:
    def __init__(self) -> None:
        self.lines: List[str] = []
        self._g = 0

    def next_guest(self) -> int:
        self._g += 1
        return self._g

    async def client_line(self, conn: Any, line: str) -> None:
        self.lines.append(line)
        if line.startswith("|/trn "):
            name = line[len("|/trn "):].split(",")[0]
            await conn.send(f"|updateuser| {name}|1|1|{{}}")

    async def drop(self, conn: Any) -> None:
        self.lines.append("<drop>")


class _Policy:
    stochastic = False

    def __init__(self, idx: int = 2) -> None:
        self.idx = idx
        self.rows: List[np.ndarray] = []

    def decide(self, row, mask):
        self.rows.append(np.array(row))
        return self.idx, self.idx


class _Teams:
    def yield_team(self) -> str:
        return "packed"


def _frame(n: int, *, turn: int = 1, mask=None, nan: bool = False) -> str:
    row = np.zeros(obs_dim(), dtype="<f4")
    if nan:
        row[5] = np.nan
    mask = mask or [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    return json.dumps({"frame": {"dtype": "<f4", "shape": [obs_dim()],
                                 "b64": base64.b64encode(row.tobytes()).decode()},
                       "mask": mask, "tokens": {"2": "move earthquake", "3": "move protect"},
                       "turn": turn, "line": 9, "rqid": None, "n": n})


def _slot(policy=None, limit: int = 250):
    front = _Front()
    state = OurSideState()
    slot = core_side.make_slot(front, username="Gen3AIAnchor1", policy=policy or _Policy(),
                               team_source=_Teams(), state=state, forfeit_limit=limit)
    return front, state, slot


def _run(coro):
    return asyncio.run(coro)


async def _start(slot) -> None:
    await slot.send(f">{TAG}\n|init|battle\n|title|Gen3AIAnchor1 vs. Meta1")


def test_the_slot_asks_for_the_core_row_and_logs_in_through_the_protocol() -> None:
    front, _state, slot = _slot()
    assert slot.core_obs is True
    _run(slot.login())
    assert front.lines == ["|/trn Gen3AIAnchor1,0,"] and slot.logged_in.is_set()


def test_a_request_with_its_frame_sends_the_TOKEN_and_the_RQID() -> None:
    front, state, slot = _slot()

    async def go():
        await _start(slot)
        await slot.on_obs(TAG, "p1", _frame(0))
        await slot.send(f'>{TAG}\n|request|{{"active":[],"rqid":4}}')
    _run(go())
    assert front.lines[-1] == f"{TAG}|/choose move earthquake|4"
    assert slot.failure is None and state.stochastic_kwargs == [False]


def test_the_trainers_forfeit_fires_at_the_limit_and_is_not_a_decision() -> None:
    front, state, slot = _slot(limit=30)

    async def go():
        await _start(slot)
        await slot.on_obs(TAG, "p1", _frame(0, turn=30))
        await slot.send(f'>{TAG}\n|turn|30\n|request|{{"active":[],"rqid":9}}')
        await slot.send(f">{TAG}\n|win|Meta1")
    _run(go())
    assert front.lines[-1] == f"{TAG}|/forfeit"
    rec = state.records[-1]
    assert rec.won is False and rec.turns == 30 and rec.hit_forfeit_limit and rec.n_decisions == 0


def test_a_win_books_ONE_record_with_our_argmax_counts() -> None:
    _front, state, slot = _slot()

    async def go():
        await _start(slot)
        await slot.on_obs(TAG, "p1", _frame(0))
        await slot.send(f'>{TAG}\n|turn|1\n|request|{{"rqid":1}}')
        await slot.send(f">{TAG}\n|win|Gen3AIAnchor1")
        await slot.send(f">{TAG}\n|win|Gen3AIAnchor1")      # a repeat is not a second game
    _run(go())
    assert len(state.records) == 1
    rec = state.records[0]
    assert rec.won is True and rec.n_decisions == 1 and rec.our_argmax_matches == 1
    assert rec.n_defaults == 0 and rec.n_redecides == 0


def test_a_wait_request_is_not_a_decision() -> None:
    front, _state, slot = _slot()

    async def go():
        await _start(slot)
        await slot.send(f'>{TAG}\n|request|{{"wait":true,"rqid":2}}')
    _run(go())
    assert not any("/choose" in ln for ln in front.lines) and slot.failure is None


@pytest.mark.parametrize("case", ["no_frame", "unconsumed", "out_of_order", "nan", "masked",
                                  "invalid_choice", "popup", "unknown_battle"])
def test_every_protocol_surprise_is_a_NAMED_failure(case: str) -> None:
    policy = _Policy(idx=7) if case == "masked" else _Policy()
    _front, state, slot = _slot(policy=policy)

    async def go():
        if case != "unknown_battle":
            await _start(slot)
        if case == "no_frame":
            await slot.send(f'>{TAG}\n|request|{{"rqid":1}}')
        elif case == "unconsumed":
            await slot.on_obs(TAG, "p1", _frame(0))
            await slot.on_obs(TAG, "p1", _frame(1))
        elif case == "out_of_order":
            await slot.on_obs(TAG, "p1", _frame(3))
        elif case == "unknown_battle":
            await slot.on_obs(TAG, "p1", _frame(0))
        elif case in ("nan", "masked"):
            await slot.on_obs(TAG, "p1", _frame(0, nan=(case == "nan")))
            await slot.send(f'>{TAG}\n|request|{{"rqid":1}}')
        elif case == "invalid_choice":
            await slot.send(f">{TAG}\n|error|[Invalid choice] Can't move: nope")
        elif case == "popup":
            await slot.send("|popup|Your team was rejected for the following reasons:")
    _run(go())
    assert isinstance(slot.failure, SeriesFailure) and slot.failure.cause == "core_slot_error"
    assert state.error and state.error.startswith("core_slot_error")
    with pytest.raises(SeriesFailure):
        slot.check()


def test_accept_series_waits_for_THIS_opponent_and_draws_before_the_challenge() -> None:
    front, _state, slot = _slot()

    async def go():
        await slot.login()
        task = asyncio.create_task(slot.accept_series("MetaSmallRL2", 1))
        await slot.send("|pm| Stranger| Gen3AIAnchor1|/challenge gen3ou|gen3ou|||")
        await slot.send("|pm| MetaSmallRL2| Gen3AIAnchor1|/challenge gen3ou|gen3ou|||")
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        await _start(slot)
        await slot.send(f">{TAG}\n|win|MetaSmallRL2")
        await asyncio.wait_for(task, timeout=5)
    _run(go())
    assert front.lines[1:] == ["|/utm packed", "|/accept MetaSmallRL2"]


def test_our_side_regime_reads_plays_temperature_rule() -> None:
    class P:
        regime = "greedy"
        our_temperature = None
    assert core_side.our_stochastic(P()) == (False, 1.0)
    P.our_temperature = 0.7
    assert core_side.our_stochastic(P()) == (True, 0.7)
    P.our_temperature, P.regime = None, "t1"
    assert core_side.our_stochastic(P()) == (True, 1.0)


def test_the_core_policy_is_RLPlayers_arithmetic() -> None:
    """``CorePolicy.decide`` = ``RLPlayer._predict_best_action``: masked logits
    ``logits + (mask - 1) * 1e9``, greedy = argmax — an illegal max-logit action is never chosen."""
    import torch

    class Dist:
        def __init__(self, logits):
            self.distribution = type("D", (), {"logits": logits})()

    class Pol:
        def get_distribution(self, obs):
            assert obs["observation"].dtype == torch.float32 and obs["action_mask"].dtype == torch.int8
            return Dist(torch.tensor([[0.0, 9.0, 1.0, 2.0, 0, 0, 0, 0, 0, 0, 0]]))

    model = type("M", (), {"device": "cpu", "policy": Pol()})()
    mask = np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
    row = np.zeros(obs_dim(), dtype=np.float32)
    assert core_side.CorePolicy(model, stochastic=False, temperature=1.0).decide(row, mask) == (3, 3)
    idx, arg = core_side.CorePolicy(model, stochastic=True, temperature=1.0).decide(row, mask)
    assert arg == 3 and idx in (2, 3)
