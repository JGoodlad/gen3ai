"""poke_env.player module init — LAZY (PEP 562), for the same reason as the package root.

`poke_env.battle.battle` imports `poke_env.player.battle_order` (a genuine cross-subtree
dependency: orders serialize against Move/Pokemon), and importing ANY `poke_env.player.*`
module first executes THIS init. Eagerly it pulled `concurrency` (starting the global asyncio
loop thread) plus `baselines` → `battle` (a circular import once the package root went lazy).
Lazy, `poke_env.player.battle_order` costs exactly battle_order, the battle subtree stays
thread-free, and the loop thread starts when something imports `player.player` / `ps_client` /
`utils` — the modules that actually use it. Public surface unchanged.
"""

_LAZY = {
    "POKE_LOOP": "poke_env.concurrency",
    "MaxBasePowerPlayer": "poke_env.player.baselines",
    "RandomPlayer": "poke_env.player.baselines",
    "SimpleHeuristicsPlayer": "poke_env.player.baselines",
    "BattleOrder": "poke_env.player.battle_order",
    "DefaultBattleOrder": "poke_env.player.battle_order",
    "DoubleBattleOrder": "poke_env.player.battle_order",
    "ForfeitBattleOrder": "poke_env.player.battle_order",
    "PassBattleOrder": "poke_env.player.battle_order",
    "SingleBattleOrder": "poke_env.player.battle_order",
    "Player": "poke_env.player.player",
    "background_cross_evaluate": "poke_env.player.utils",
    "background_evaluate_player": "poke_env.player.utils",
    "cross_evaluate": "poke_env.player.utils",
    "evaluate_player": "poke_env.player.utils",
    "PSClient": "poke_env.ps_client",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    import importlib

    target = _LAZY.get(name)
    if target is None:
        # `import poke_env.player; poke_env.player.<submodule>.X` used to work because the eager init imported
        # everything; resolve a submodule asked for as an attribute the same lazy way.
        try:
            value = importlib.import_module(f"poke_env.player.{name}")
        except ImportError:
            raise AttributeError(f"module 'poke_env.player' has no attribute {name!r}") from None
        globals()[name] = value
        return value

    value = getattr(importlib.import_module(target), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
