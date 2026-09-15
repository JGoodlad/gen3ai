"""This module defines the Teambuilder abstract class, which represents objects yielding
Pokemon Showdown teams in the context of communicating with Pokemon Showdown.
"""

from abc import ABC, abstractmethod
from typing import List

from poke_env.teambuilder.teambuilder_pokemon import TeambuilderPokemon


class Teambuilder(ABC):
    """Teambuilder objects allow the generation of teams by Player instances.

    They must implement the yield_team method, which must return a valid
    packed-formatted showdown team every time it is called.

    This format is a custom format decribed in Pokemon's showdown protocol
    documentation:
    https://github.com/smogon/pokemon-showdown/blob/master/PROTOCOL.md#team-format

    This class also implements a helper function to convert teams from the classical
    showdown team text format into the packed-format.
    """

    @abstractmethod
    def yield_team(self) -> str:
        """Returns a packed-format team."""

    @staticmethod
    def parse_showdown_team(team: str) -> List[TeambuilderPokemon]:
        """Converts a showdown-formatted team string into a list of TeambuilderPokemon
        objects.

        This method can be used when using teams built in the showdown teambuilder.

        :param team: The showdown-format team to convert.
        :type team: str
        :return: The formatted team.
        :rtype: list of TeambuilderPokemon
        """
        mons = []

        for ps_mon in team.split("\n\n"):
            # 🚨 `not ps_mon.strip()`, NOT `ps_mon == ""`. A Showdown paste that ends with a
            # blank line (every Metamon `competitive` gen3ou file ends "\n\n\n") splits into a
            # final chunk of "\n" — which is not "", so the old check let it through and
            # `TeambuilderPokemon.from_showdown` turned it into an EMPTY 7th Pokemon. Showdown
            # then answers the packed team with
            #   |popup|... - You may only bring up to 6 Pokemon (your team has 7).
            # and the rejected challenge NEVER BECOMES A BATTLE: the symptom is a HANG, not an
            # error, and `validate_teams_locally` cannot see it because it validates the PASTE
            # and the defect is created by the PACK. Upstream poke-env 0.8.3.3 parses the same
            # file correctly, so this was OUR fork's defect. Measured 2026-09-14,
            # designs/research_state/measurements/metamon_matched_regime_2026-09-14/ hazard H-A.
            if not ps_mon.strip():
                continue
            mons.append(TeambuilderPokemon.from_showdown(ps_mon))

        return mons

    @staticmethod
    def parse_packed_team(team: str) -> List[TeambuilderPokemon]:
        """Converts a packed-format team string into a list of TeambuilderPokemon
        objects.

        :param team: The packed-format team to convert.
        :type team: str
        :return: The formatted team.
        :rtype: list of TeambuilderPokemon
        """
        packed_mons = team.split("]")

        mons = []

        for packed_mon in packed_mons:
            if packed_mon == "":
                continue

            mons.append(TeambuilderPokemon.from_packed(packed_mon))

        return mons

    @staticmethod
    def join_team(team: List[TeambuilderPokemon]) -> str:
        """Converts a list of TeambuilderPokemon objects into the corresponding packed
        showdown team format.

        :param team: The list of TeambuilderPokemon objects that form the team.
        :type team: list of TeambuilderPokemon
        :return: The formatted team string.
        :rtype: str"""
        return "]".join([mon.packed for mon in team])
