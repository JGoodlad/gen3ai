"""gen3ai battle read-models — the frozen data classes the Rust core's reading is copied into.

The Python event-sourced battle layer (``Gen3Battle``, ``BattleEvent``, ``TurnView``, ``StrictBattleView``,
the offline feed) is DELETED (T27 P6 slice 6d-2): every row, view and legality surface is the Rust core's.
What remains:

* :mod:`agents.battle.live_view` — ``LiveView`` / ``LiveSide`` / ``LivePokemon`` / ``LiveMove`` /
  ``LiveWeather`` and ``LegalActions`` / ``LegalMove`` / ``LegalSwitch`` (data classes only);
* :mod:`agents.battle.core_view` — builds them from the core's ``present()`` / ``legal_actions()`` JSON;
* :mod:`agents.battle.core_obs` / :mod:`agents.battle.core_replay` — the core's observation frames and
  the recorded-corpus replay through ``core_events``;
* :mod:`agents.battle.faint_causes` — the faint-cause vocabulary the model and the archive decoder read.

Re-exports are lazy (PEP 562 ``__getattr__``) so importing one submodule does not load the others.
"""

from typing import TYPE_CHECKING

# attribute name -> submodule it lives in
_EXPORTS = {
    "LiveView": "live_view",
    "LiveSide": "live_view",
    "LivePokemon": "live_view",
    "LiveMove": "live_view",
    "LiveWeather": "live_view",
    "LegalActions": "live_view",
    "LegalMove": "live_view",
    "LegalSwitch": "live_view",
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
