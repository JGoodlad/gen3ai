"""gen3ai event-sourced battle layer.

A thin, additive layer over poke-env's state tracker, exposing two clean, disjoint
read surfaces — each a single source of truth for its concern:

* :class:`Gen3Battle` + :class:`TurnView` — history ("what happened, in order"), from
  the revealed-order protocol event log.
* :class:`LiveView` — the current board ("what is true now"), with no past-turn state.

See ``designs/ai_v4/design_event_sourced_battle.md``.
"""

from agents.battle.battle_event import (
    EVENT_KIND,
    MESSAGE_POLICY,
    OPP,
    OURS,
    BattleEvent,
    EventKind,
    Policy,
    UnknownMessageType,
    UnsupportedMessageType,
    classify,
)
from agents.battle.gen3_battle import Gen3Battle
from agents.battle.live_view import LivePokemon, LiveSide, LiveView
from agents.battle.turn_view import TurnView

__all__ = [
    "EVENT_KIND",
    "MESSAGE_POLICY",
    "OPP",
    "OURS",
    "BattleEvent",
    "EventKind",
    "Policy",
    "UnknownMessageType",
    "UnsupportedMessageType",
    "classify",
    "Gen3Battle",
    "TurnView",
    "LiveView",
    "LiveSide",
    "LivePokemon",
]
