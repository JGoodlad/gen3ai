"""Accepted poke-env value-enums — the single re-export seam for our code.

These four enums are *spec-defined value vocabularies* (a fixed set of Gen-3 types,
status conditions, move categories, and weathers). We borrow their members **as keys
only** — to index a lookup table, build a one-hot, or compare identity. We never call
methods on them (no ``PokemonType.damage_multiplier``, no behaviour), so re-exporting
them through this module gives every consumer one import to point at while keeping the
"poke-env is an implementation detail behind the strict battle-API" standard intact.

Why a seam at all, and why exactly these four:
  * They are pure data enums with no temporal/stateful semantics, unlike the ``Battle`` /
    ``Pokemon`` objects (which flow exclusively through ``LiveView`` / ``TurnView`` /
    ``LegalActions`` — see ``agents/battle/strict_view.py``). There is nothing to misread.
  * Routing them here means the static Phase-4 guard (``enums_lock_test.py`` /
    ``no_raw_battle_read_test.py``) can forbid ``from poke_env … import {PokemonType,
    Status, MoveCategory, Weather}`` everywhere under ``agents/{observation,action,
    training,inference}`` *except* this module — a new direct poke-env enum import then
    fails CI.

``Effect`` is **intentionally excluded.** poke-env's ``Effect`` enum is the temporal,
overwrite-on-every-line volatile vocabulary that has caused real bugs; it is replaced by
our own source-derived ``agents/observation/gen3_effects.py``. Do not add it here.

The members are the same objects poke-env uses internally, so this is a pure import-path
indirection — ``agents.enums.PokemonType is poke_env.battle.pokemon_type.PokemonType``.
"""

from poke_env.battle.move_category import MoveCategory
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.status import Status
from poke_env.battle.weather import Weather

__all__ = ["PokemonType", "Status", "MoveCategory", "Weather"]
