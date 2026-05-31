"""gen3ai event-sourced battle layer.

A thin, additive layer over poke-env's state tracker, exposing two clean, disjoint
read surfaces — each a single source of truth for its concern:

* :class:`Gen3Battle` + :class:`TurnView` — history ("what happened, in order"), from
  the revealed-order protocol event log.
* :class:`LiveView` — the current board ("what is true now"), with no past-turn state.

See ``designs/ai_v4/design_event_sourced_battle.md``.

Re-exports are **lazy** (PEP 562 ``__getattr__``): importing a single submodule
(e.g. ``from agents.battle.gen3_battle import Gen3Battle``) must not force-import the
whole package, which would create an import cycle (turn_view/live_view ↔ battle_event)
when consumers like ``agents.inference.player`` pull in just ``Gen3Battle`` at module
load. Top-level access (``from agents.battle import TurnView``) still works.
"""

from typing import TYPE_CHECKING

# attribute name -> submodule it lives in
_EXPORTS = {
    "EVENT_KIND": "battle_event",
    "MESSAGE_POLICY": "battle_event",
    "OPP": "battle_event",
    "OURS": "battle_event",
    "BattleEvent": "battle_event",
    "EventKind": "battle_event",
    "Policy": "battle_event",
    "UnknownMessageType": "battle_event",
    "UnsupportedMessageType": "battle_event",
    "classify": "battle_event",
    "Gen3Battle": "gen3_battle",
    "TurnView": "turn_view",
    "LiveView": "live_view",
    "LiveSide": "live_view",
    "LivePokemon": "live_view",
    "LiveMove": "live_view",
    "LiveWeather": "live_view",
    "LegalActions": "live_view",
    "LegalMove": "live_view",
    "LegalSwitch": "live_view",
    "StrictBattleView": "strict_view",
}

__all__ = list(_EXPORTS)


def __getattr__(name: str):
    """Lazily resolve a re-exported symbol from its submodule on first access."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    mod = importlib.import_module(f"{__name__}.{module}")
    return getattr(mod, name)


if TYPE_CHECKING:  # for static analysers / IDEs only — not executed at runtime
    from agents.battle.battle_event import (  # noqa: F401
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
    from agents.battle.gen3_battle import Gen3Battle  # noqa: F401
    from agents.battle.live_view import (  # noqa: F401
        LegalActions,
        LegalMove,
        LegalSwitch,
        LiveMove,
        LivePokemon,
        LiveSide,
        LiveView,
        LiveWeather,
    )
    from agents.battle.strict_view import StrictBattleView  # noqa: F401
    from agents.battle.turn_view import TurnView  # noqa: F401
