"""Unit guards for SingleAgentWrapper's opponent-poll settle (the self-play stale-decision
race fix). These are the DETERMINISTIC regression guard for the *prevention* layer — they
replace the need for a production "disable the settle" env-var: if the settle call is ever
dropped from step(), test_step_settles_before_opponent_reads goes red.
"""

from unittest.mock import MagicMock

import numpy as np

from poke_env.environment.single_agent_wrapper import (
    SingleAgentWrapper,
    _battle_decision_signature,
)


def _mk_move(mid):
    m = MagicMock()
    m.id = mid
    return m


def _mk_mon(species):
    p = MagicMock()
    p.species = species
    return p


def test_battle_decision_signature_captures_action_axes():
    b = MagicMock()
    b.force_switch = False
    b.maybe_trapped = False
    b.available_moves = [_mk_move("earthquake"), _mk_move("rockslide")]
    b.available_switches = [_mk_mon("skarmory")]
    assert _battle_decision_signature(b) == (
        False, False, ("earthquake", "rockslide"), ("skarmory",)
    )
    # The faint→force-switch transition (the move-face) flips the signature → "not settled".
    b.force_switch = True
    b.available_moves = []
    assert _battle_decision_signature(b) == (True, False, (), ("skarmory",))


def test_battle_decision_signature_handles_none_lists():
    b = MagicMock()
    b.force_switch = False
    b.maybe_trapped = False
    b.available_moves = None
    b.available_switches = None
    assert _battle_decision_signature(b) == (False, False, (), ())


def test_settle_noops_without_ps_client():
    # Bridge / no-connection path: no ps_client → no loop → settle returns without raising.
    w = SingleAgentWrapper.__new__(SingleAgentWrapper)
    w.env = MagicMock()
    w.env.battle2 = MagicMock()
    w.env.agent2 = MagicMock(spec=[])  # no ps_client attribute at all
    w._settle_opponent_battle()  # must not raise


def test_step_settles_before_opponent_reads():
    """The settle MUST run in step() before the opponent decides — that IS the race fix.
    A regression that drops the call (or moves it after the poll) fails here."""
    w = SingleAgentWrapper.__new__(SingleAgentWrapper)
    order = []
    w._settle_opponent_battle = MagicMock(side_effect=lambda: order.append("settle"))

    w.env = MagicMock()
    b2 = MagicMock()
    b2.wait = False
    b2.teampreview = False
    w.env.battle2 = b2
    w.env._fake = False
    w.env._strict = False
    w.env.agent1.username = "a1"
    w.env.agent2.username = "a2"
    w.env.order_to_action.return_value = np.array([0])
    w.env.step.return_value = (
        {"a1": {}}, {"a1": 0.0}, {"a1": False}, {"a1": False}, {"a1": {}},
    )
    w.opponent = MagicMock()
    # Return a plain non-awaitable order; record that the opponent was polled.
    w.opponent.choose_move = MagicMock(side_effect=lambda b: order.append("poll") or object())

    w.step(np.array([6]))

    assert order and order[0] == "settle", (
        f"settle must run before the opponent is polled; call order was {order}"
    )
    assert "poll" in order  # the opponent was actually polled (we exercised the real branch)
