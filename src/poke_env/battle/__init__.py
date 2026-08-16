"""poke_env.battle module init — LAZY (PEP 562), like the package root and `player`.

The eager form was the amplifier of an order-dependent circular import: importing ANY
`poke_env.battle.*` submodule first executed this init, whose `from poke_env.battle.battle
import Battle` reaches `poke_env.player.battle_order` — so an entry that began IN
`player.battle_order` (whose own first import is `poke_env.battle.move`) re-entered itself
mid-initialization and died on a genuine ImportError. Lazy, `poke_env.battle.move` costs
exactly `move` and the cycle is unreachable from every entry order. It also keeps this subtree
thread-free for the forkserver preload (`gen3_forkserver_preload_v1`). Public surface unchanged.
"""

_LAZY = {
    "AbstractBattle": "poke_env.battle.abstract_battle",
    "Battle": "poke_env.battle.battle",
    "DoubleBattle": "poke_env.battle.double_battle",
    "Effect": "poke_env.battle.effect",
    "Field": "poke_env.battle.field",
    "Move": "poke_env.battle.move",
    "MoveSet": "poke_env.battle.move",
    "SPECIAL_MOVES": "poke_env.battle.move",
    "MoveCategory": "poke_env.battle.move_category",
    "Pokemon": "poke_env.battle.pokemon",
    "PokemonGender": "poke_env.battle.pokemon_gender",
    "PokemonType": "poke_env.battle.pokemon_type",
    "STACKABLE_CONDITIONS": "poke_env.battle.side_condition",
    "SideCondition": "poke_env.battle.side_condition",
    "Status": "poke_env.battle.status",
    "Target": "poke_env.battle.target",
    "Weather": "poke_env.battle.weather",
    "Z_CRYSTAL": "poke_env.battle.z_crystal",
}

__all__ = sorted(_LAZY)


def __getattr__(name: str):
    import importlib

    target = _LAZY.get(name)
    if target is None:
        try:
            value = importlib.import_module(f"poke_env.battle.{name}")
        except ImportError:
            raise AttributeError(f"module 'poke_env.battle' has no attribute {name!r}") from None
        globals()[name] = value
        return value

    value = getattr(importlib.import_module(target), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
