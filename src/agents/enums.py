"""The four Gen-3 value-enums — OWNED here, poke-env-free, the single seam for our code.

These four enums are *spec-defined value vocabularies* (a fixed set of Gen-3 types, status
conditions, move categories, and weathers). Our code uses their members **as keys only** — to index a
lookup table, build a one-hot, or compare identity. It never calls methods on them (no
``PokemonType.damage_multiplier``, no behaviour), so routing them through this module gives every
consumer one import to point at.

**Who defines them (P1 of the poke-env retirement, ``T27``).** The DEFINITIONS live HERE, and this module
imports nothing outside the standard library. Before P1 the arrow ran the other way (this module re-exported the
vendored fork's classes), which put 36 poke-env modules on the trainer's import path although training never
called poke-env; from P1 to P6 the fork re-exported THESE classes; P6 deleted the fork.

The classes carry the members, ``auto()`` values, ``__str__`` and (for ``PokemonType`` / ``Weather``) the helper
methods the fork's Python battle layer called (``damage_multiplier``, ``from_name``, ``from_showdown_message``),
copied verbatim. Production code never calls them; ``damage_multiplier`` is still the ORACLE
``gen3_mechanics_test`` holds the fast type-chart path to. Do not add behaviour here, and do not add ``Effect`` —
poke-env's ``Effect`` enum was the temporal, overwrite-on-every-line volatile vocabulary that caused real bugs; it
is replaced by our own source-derived ``agents/observation/gen3_effects.py`` (pinned by ``enums_test.py``).
"""
from __future__ import annotations

import logging
from enum import Enum, auto, unique
from typing import Dict, Optional

__all__ = ["PokemonType", "Status", "MoveCategory", "Weather"]


@unique
class PokemonType(Enum):
    """A Pokemon type

    This enumeration represents pokemon types. Each type is an instance of this class,
    whose name corresponds to the upper case spelling of its english name (ie. FIRE).
    """

    BUG = auto()
    DARK = auto()
    DRAGON = auto()
    ELECTRIC = auto()
    FAIRY = auto()
    FIGHTING = auto()
    FIRE = auto()
    FLYING = auto()
    GHOST = auto()
    GRASS = auto()
    GROUND = auto()
    ICE = auto()
    NORMAL = auto()
    POISON = auto()
    PSYCHIC = auto()
    ROCK = auto()
    STEEL = auto()
    WATER = auto()
    THREE_QUESTION_MARKS = auto()
    STELLAR = auto()

    def __str__(self) -> str:
        return f"{self.name} (pokemon type) object"

    def damage_multiplier(
        self,
        type_1: PokemonType,
        type_2: Optional[PokemonType] = None,
        *,
        type_chart: Dict[str, Dict[str, float]],
    ) -> float:
        """Computes the damage multiplier from this type on a pokemon with types `type_1`
        and, optionally, `type_2`.

        :param type_1: The first type of the target.
        :type type_1: PokemonType
        :param type_2: The second type of the target. Defaults to None.
        :type type_2: PokemonType, optional
        :return: The damage multiplier from this type on a pokemon with types `type_1`
            and, optionally, `type_2`.
        :rtype: float
        """
        if self in {
            PokemonType.THREE_QUESTION_MARKS,
            PokemonType.STELLAR,
        } or type_1 in {PokemonType.THREE_QUESTION_MARKS, PokemonType.STELLAR}:
            return 1

        damage_multiplier = type_chart[type_1.name][self.name]
        if type_2 is not None:
            return damage_multiplier * type_chart[type_2.name][self.name]
        return damage_multiplier

    @staticmethod
    def from_name(name: str) -> PokemonType:
        """Returns a pokemon type based on its name.

        :param name: The name of the pokemon type.
        :type name: str
        :return: The corresponding type object.
        :rtype: PokemonType
        """
        if name == "???":
            return PokemonType.THREE_QUESTION_MARKS
        return PokemonType[name.upper()]


@unique
class Status(Enum):
    """Enumeration, represent a status a pokemon can be afflicted with."""

    BRN = auto()
    FNT = auto()
    FRZ = auto()
    PAR = auto()
    PSN = auto()
    SLP = auto()
    TOX = auto()

    def __str__(self) -> str:
        return f"{self.name} (status) object"


@unique
class MoveCategory(Enum):
    """Enumeration, represent a move category."""

    PHYSICAL = auto()
    SPECIAL = auto()
    STATUS = auto()

    def __str__(self) -> str:
        return f"{self.name} (move category) object"


class Weather(Enum):
    """Enumeration, represent a non null weather in a battle."""

    UNKNOWN = auto()
    DESOLATELAND = auto()
    DELTASTREAM = auto()
    HAIL = auto()
    PRIMORDIALSEA = auto()
    RAINDANCE = auto()
    SANDSTORM = auto()
    SNOWSCAPE = SNOW = auto()
    SUNNYDAY = auto()

    def __str__(self) -> str:
        return f"{self.name} (weather) object"

    @staticmethod
    def from_showdown_message(message: str):
        """Returns the Weather object corresponding to the message.

        :param message: The message to convert.
        :type message: str
        :return: The corresponding Weather object.
        :rtype: Weather
        """
        message = message.replace("move: ", "")
        message = message.replace(" ", "_")
        message = message.replace("-", "_")

        try:
            return Weather[message.upper()]
        except KeyError:
            logging.getLogger("poke-env").warning(
                "Unexpected weather '%s' received. Weather.UNKNOWN will be used "
                "instead. If this is unexpected, please open an issue at "
                "https://github.com/hsahovic/poke-env/issues/ along with this error "
                "message and a description of your program.",
                message,
            )
            return Weather.UNKNOWN
