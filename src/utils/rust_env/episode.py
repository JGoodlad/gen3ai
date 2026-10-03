"""The Rust env's EPISODE declaration — the terminal reward and the stall forfeit (M5 Lane D).

The core's end-of-episode semantics (``src/rust_env/src/episode.rs``) equal ``Gen3Env``'s:

* the REWARD is the terminal alone (``Gen3RewardManager.process_turn_reward``; the shaped path was
  deleted, ``e3ef16db``): ``victory_value`` on a p1 win; on any other end ``0.0`` under the win
  INDICATOR, else ``draw_penalty`` when the end turn ``>= timeout_turn_cap`` (a cap forfeit) and
  ``-victory_value`` otherwise (a decisive loss, or a pre-cap tie);
* ``terminated`` / ``truncated`` are ``PokeEnv.calc_term_trunc`` on p1's reading;
* the STALL FORFEIT is ``Gen3Env.action_to_order``: at a p1 decision whose turn is ``>=
  StallConfig().threshold`` p1 forfeits INSTEAD of acting, and nothing else is fed that op.

This module is the host's half: the ``terminal`` spec object, built from a ``RewardConfig``, and the
production values it defaults to (pinned by ``episode_test.py`` against ``production_config()``, so
the default cannot drift from the surface training runs).
"""
from __future__ import annotations

from typing import Dict, Mapping, Union

#: The ``terminal`` object's keys (``episode::Terminal::from_json``): every one required.
TERMINAL_KEYS = ("victory_value", "terminal_indicator", "draw_penalty", "timeout_turn_cap")

#: The PRODUCTION terminal: the win indicator, victory 1.0 (``draw_penalty`` 0, the
#: only value ``combination_checks`` admits under the indicator) and the reward's timeout cap
#: ``reward_weights._TIMEOUT_TURN_CAP`` (== ``StallConfig().threshold`` == ``MAX_TURNS``).
PRODUCTION_TERMINAL: Dict[str, Union[float, bool, int]] = {
    "victory_value": 1.0, "terminal_indicator": True, "draw_penalty": 0.0, "timeout_turn_cap": 250,
}


def terminal_json(t: Mapping) -> dict:
    """Validate a terminal mapping (exactly ``TERMINAL_KEYS``) and return it JSON-ready."""
    if set(t) != set(TERMINAL_KEYS):
        raise ValueError(f"terminal: keys {sorted(t)} != {sorted(TERMINAL_KEYS)}")
    return {"victory_value": float(t["victory_value"]), "terminal_indicator": bool(t["terminal_indicator"]),
            "draw_penalty": float(t["draw_penalty"]), "timeout_turn_cap": int(t["timeout_turn_cap"])}


def terminal_from_reward_config(rc) -> dict:
    """The ``terminal`` object for a ``RewardConfig`` — what a training host passes."""
    from agents.training.reward_weights import _TIMEOUT_TURN_CAP

    return terminal_json({"victory_value": rc.victory_value, "terminal_indicator": rc.terminal_indicator,
                          "draw_penalty": rc.draw_penalty, "timeout_turn_cap": _TIMEOUT_TURN_CAP})


def stall_threshold() -> int:
    """The spec's ``turn_limit`` in production: ``StallConfig().threshold`` (read, never a literal)."""
    from agents.training.stall import StallConfig

    return int(StallConfig().threshold)
