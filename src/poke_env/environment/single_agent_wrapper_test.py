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
from poke_env.player.battle_order import DefaultBattleOrder


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


# ---------------------------------------------------------------------------
# Opponent order_to_action stale-race fallback — the window PAST choose_move's
# re-decide: the order can go stale between choose_move and the env-level serialize
# (battle finished / flipped to wait under it). The opponent must default, not crash
# the worker. Guards against regressing the restart_err_*_fa1fe3 crash.
# ---------------------------------------------------------------------------

def _mk_wrapper_for_opp_poll(order_to_action_side_effect):
    """A SingleAgentWrapper wired to drive the opponent-poll branch of step()
    deterministically. ``order_to_action_side_effect`` (a list) controls what
    env.order_to_action does on each call: an exception instance is raised, a value returned."""
    w = SingleAgentWrapper.__new__(SingleAgentWrapper)
    w._settle_opponent_battle = MagicMock()
    w.env = MagicMock()
    b2 = MagicMock()
    b2.wait = False
    b2.teampreview = False
    w.env.battle2 = b2
    w.env._fake = False
    w.env._strict = True
    w.env.agent1.username = "a1"
    w.env.agent2.username = "a2"
    w.env.order_to_action = MagicMock(side_effect=order_to_action_side_effect)
    w.env.step.return_value = (
        {"a1": {}}, {"a1": 0.0}, {"a1": False}, {"a1": False}, {"a1": {}},
    )
    w.opponent = MagicMock()
    w.opponent.choose_move = MagicMock(return_value=object())  # a non-awaitable order
    return w


def test_opp_order_race_falls_back_to_default():
    """The opponent's order goes stale between choose_move and the env-level serialize. The
    wrapper must fall back to the default order, NOT propagate the ValueError (which would kill
    the SubprocVecEnv worker → launcher restart)."""
    sentinel_action = np.array([0])
    w = _mk_wrapper_for_opp_poll(
        [ValueError("order /choose switch Skarmory not in valid orders ['/choose default']!"),
         sentinel_action]
    )
    w.opponent._n_redecides = 0

    w.step(np.array([6]))  # must NOT raise

    assert w.env.order_to_action.call_count == 2, "expected the opponent order + the default fallback"
    fallback_order = w.env.order_to_action.call_args_list[1].args[0]
    assert isinstance(fallback_order, DefaultBattleOrder), "fallback must serialize a default order"
    assert w.opponent._n_redecides == 1, "the resolved race must be counted"


def test_opp_order_race_with_bot_opponent_without_counter():
    """A non-RL opponent (no ``_n_redecides``) hits the same race; the wrapper must still default
    without an AttributeError on the missing counter."""
    w = _mk_wrapper_for_opp_poll([ValueError("stale"), np.array([0])])
    w.opponent = MagicMock(spec=["choose_move"])   # no _n_redecides attribute
    w.opponent.choose_move = MagicMock(return_value=object())

    w.step(np.array([6]))  # must NOT raise

    assert w.env.order_to_action.call_count == 2


def test_opp_clean_order_does_not_fall_back():
    """The common path: a valid opponent order is serialized once, with no default fallback."""
    w = _mk_wrapper_for_opp_poll([np.array([3])])
    w.opponent._n_redecides = 0

    w.step(np.array([6]))

    assert w.env.order_to_action.call_count == 1, "a clean order must not trigger the fallback"
    assert w.opponent._n_redecides == 0
