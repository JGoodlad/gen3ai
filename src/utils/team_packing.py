"""Team PACKING — the Showdown paste <-> packed-team machinery, owned here and poke-env-free (P1 of the poke-env
retirement, ``T27``).

``TeambuilderPokemon`` (one mon: parse a Showdown paste or a packed string, write the packed form),
``Teambuilder`` (the abstract per-battle team source, with ``parse_showdown_team`` / ``parse_packed_team`` /
``join_team``), ``ConstantTeambuilder`` and the ``STATS_TO_IDX`` stat-name table they read. ``Gen3Teambuilder``
(``utils.teambuilder``), the team sources (``utils.team_sources``), the archetype / species-prior labellers and the
trainer's whole team path subclass or call these; they used to import them from ``poke_env.teambuilder``, which put
the whole poke-env package on the import path of every run that draws a team.

The vendored fork's ``poke_env/teambuilder/{teambuilder,teambuilder_pokemon,constant_teambuilder}.py`` (and
``poke_env.stats.STATS_TO_IDX``) now RE-EXPORT these objects, so the fork's ``Player`` still accepts them
(``isinstance(team, Teambuilder)`` holds: it is the same class). The code below is the fork's, verbatim — including the
2026-09-14 trailing-blank-line fix in ``parse_showdown_team`` that ``src/poke_env_teambuilder_gate_test.py`` pins.
"""
from abc import ABC, abstractmethod
from typing import List, Optional

from utils.showdown_id import to_id_str

__all__ = ["STATS_TO_IDX", "TeambuilderPokemon", "Teambuilder", "ConstantTeambuilder"]

STATS_TO_IDX = {
    "hp": 0,
    "atk": 1,
    "def": 2,
    "spa": 3,
    "spd": 4,
    "spe": 5,
    "satk": 3,
    "sdef": 4,
}


class TeambuilderPokemon:
    HP_TO_IVS = {
        "bug": [31, 31, 31, 30, 31, 30],
        "dark": [31, 31, 31, 31, 31, 31],
        "dragon": [30, 31, 31, 31, 31, 31],
        "electric": [31, 31, 31, 31, 30, 31],
        "fighting": [31, 31, 30, 30, 30, 30],
        "fire": [31, 30, 31, 30, 31, 30],
        "flying": [31, 31, 31, 30, 30, 30],
        "ghost": [31, 30, 31, 31, 31, 30],
        "grass": [30, 31, 31, 31, 30, 31],
        "ground": [31, 31, 31, 31, 30, 30],
        "ice": [31, 30, 30, 31, 31, 31],
        "poison": [31, 31, 30, 31, 30, 30],
        "psychic": [30, 31, 31, 30, 31, 31],
        "rock": [31, 31, 30, 30, 31, 30],
        "steel": [31, 31, 31, 31, 31, 30],
        "water": [31, 31, 31, 30, 30, 31],
    }
    evs: List[int]
    ivs: List[int]
    moves: List[str]

    def __init__(
        self,
        nickname: Optional[str] = None,
        species: Optional[str] = None,
        item: Optional[str] = None,
        ability: Optional[str] = None,
        moves: Optional[List[str]] = None,
        nature: Optional[str] = None,
        evs: Optional[List[int]] = None,
        gender: Optional[str] = None,
        ivs: Optional[List[int]] = None,
        shiny: Optional[bool] = None,
        level: Optional[int] = None,
        happiness: Optional[int] = None,
        hiddenpowertype: Optional[str] = None,
        gmax: Optional[bool] = None,
        tera_type: Optional[str] = None,
    ):
        self.nickname = nickname
        self.species = species
        self.item = item
        self.ability = ability
        self.nature = nature
        self.gender = gender
        self.shiny = shiny
        self.level = level
        self.happiness = happiness
        self.hiddenpowertype = hiddenpowertype
        self.gmax = gmax
        self.tera_type = tera_type
        self.evs = evs if evs is not None else [0] * 6
        self.ivs = ivs if ivs is not None else [31] * 6

        if moves is None:
            self.moves = []
        else:
            self.moves = moves

    def __repr__(self) -> str:
        return self.packed

    def __str__(self) -> str:
        return self.packed

    @property
    def packed_evs(self) -> str:
        f_evs = ",".join([str(el) if el != 0 else "" for el in self.evs])
        if f_evs == "," * 5:
            return ""
        return f_evs

    @property
    def packed_ivs(self) -> str:
        f_ivs = ",".join([str(iv) if iv != 31 else "" for iv in self.ivs])
        if f_ivs == "," * 5:
            return ""
        return f_ivs

    @property
    def packed_moves(self) -> str:
        return ",".join([to_id_str(move) for move in self.moves])

    @property
    def packed_endstring(self) -> str:
        f_str = f",{self.hiddenpowertype or ''},"

        if self.gmax:
            return f_str + ",G"
        elif self.tera_type:
            return f_str + f",,,{self.tera_type}"

        if self.hiddenpowertype:
            return f_str

        return ""

    @property
    def packed(self) -> str:
        self._prepare_for_formatting()
        return "%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s|%s%s" % (
            self.nickname or "",
            to_id_str(self.species) if self.species else "",
            to_id_str(self.item) if self.item else "",
            to_id_str(self.ability) if self.ability else "",
            self.packed_moves or "",
            self.nature or "",
            self.packed_evs or "",
            self.gender or "",
            self.packed_ivs or "",
            "S" if self.shiny else "",
            self.level or "",
            self.happiness or "",
            self.packed_endstring,
        )

    def _prepare_for_formatting(self):
        for move in self.moves:
            move = to_id_str(move)
            if (
                move.startswith("hiddenpower")
                and len(move) > 11
                and all([iv == 31 for iv in self.ivs])
            ):
                self.ivs = list(self.HP_TO_IVS[move[11:]])

    @staticmethod
    def from_packed(packed_mon: str) -> "TeambuilderPokemon":
        """Converts a packed-format pokemon string into a TeambuilderPokemon object.

        :param packed_mon: The packed-format pokemon string to convert.
        :type packed_mon: str
        :return: The converted TeambuilderPokemon object.
        :rtype: TeambuilderPokemon
        """
        (
            raw_nickname,
            raw_species,
            raw_item,
            raw_ability,
            raw_moves,
            raw_nature,
            raw_evs,
            raw_gender,
            raw_ivs,
            raw_shiny,
            raw_level,
            endstring,
        ) = packed_mon.split("|")

        gmax = False
        tera_type = None
        hiddenpowertype = None
        happiness = None

        if endstring:
            split_endstring = endstring.split(",")

            if split_endstring[0]:
                happiness = int(split_endstring[0])

            if len(split_endstring) == 1:
                pass
            elif split_endstring[-1] == "G":
                gmax = True
            elif split_endstring[-1] != "":
                tera_type = split_endstring[-1]
            elif len(split_endstring) >= 3:
                hiddenpowertype = split_endstring[-2]

        nickname = raw_nickname or None
        species = raw_species or None
        item = raw_item or None
        ability = raw_ability or None
        nature = raw_nature or None
        gender = raw_gender or None

        if raw_moves:
            moves = raw_moves.split(",")
        else:
            moves = None

        if raw_evs:
            evs = [int(ev) if ev else 0 for ev in raw_evs.split(",")]
        else:
            evs = None

        if raw_ivs:
            ivs = [int(iv) if iv else 31 for iv in raw_ivs.split(",")]
        else:
            ivs = None

        if raw_shiny:
            assert raw_shiny == "S"
            shiny = True
        else:
            shiny = False

        if raw_level:
            level = int(raw_level)
        else:
            level = None

        return TeambuilderPokemon(
            nickname=nickname,
            species=species,
            item=item,
            ability=ability,
            moves=moves,
            nature=nature,
            evs=evs,
            gender=gender,
            ivs=ivs,
            shiny=shiny,
            level=level,
            happiness=happiness,
            hiddenpowertype=hiddenpowertype,
            gmax=gmax,
            tera_type=tera_type,
        )

    @staticmethod
    def from_showdown(showdown_mon: str) -> "TeambuilderPokemon":
        """Converts a showdown-format pokemon string into a TeambuilderPokemon object.

        :param showdown_mon: The showdown-format pokemon string to convert.
        :type showdown_mon: str
        :return: The converted TeambuilderPokemon object.
        :rtype: TeambuilderPokemon
        """
        mon = TeambuilderPokemon()

        for line in showdown_mon.split("\n"):
            line = line.strip()
            if not line:
                continue
            elif line.startswith("Ability"):
                ability = line.replace("Ability: ", "")
                mon.ability = ability.strip()
            elif line.startswith("Level: "):
                level = line.replace("Level: ", "")
                mon.level = int(level.strip())
            elif line.startswith("Happiness: "):
                happiness = line.replace("Happiness: ", "")
                mon.happiness = int(happiness.strip())
            elif line.startswith("EVs: "):
                evs = line.replace("EVs: ", "")
                split_evs = evs.split(" / ")
                for ev in split_evs:
                    n, stat = ev.split(" ")[:2]
                    idx = STATS_TO_IDX[stat.lower()]
                    mon.evs[idx] = int(n)
            elif line.startswith("IVs: "):
                ivs = line.replace("IVs: ", "")
                ivs_split = ivs.split(" / ")
                for iv in ivs_split:
                    n, stat = iv.split(" ")[:2]
                    idx = STATS_TO_IDX[stat.lower()]
                    mon.ivs[idx] = int(n)
            elif line.startswith("- "):
                line = line.replace("- ", "").strip()
                mon.moves.append(line)
            elif line.startswith("Shiny"):
                mon.shiny = line.strip().endswith("Yes")
            elif line.startswith("Gigantamax"):
                mon.gmax = line.strip().endswith("Yes")
            elif line.strip().endswith(" Nature"):
                nature = line.strip().replace(" Nature", "")
                mon.nature = nature
            elif line.startswith("Hidden Power: "):
                hp_type = line.replace("Hidden Power: ", "").strip()
                mon.hiddenpowertype = hp_type
            elif line.startswith("Tera Type: "):
                tera_type = line.replace("Tera Type: ", "").strip()
                mon.tera_type = tera_type
            else:
                if "@" in line:
                    mon_info, item = line.split(" @ ")
                    mon.item = item.strip()
                else:
                    mon_info = line
                split_mon_info = mon_info.split(" ")
                if split_mon_info[-1] == "(M)":
                    mon.gender = "M"
                    split_mon_info.pop()
                if split_mon_info[-1] == "(F)":
                    mon.gender = "F"
                    split_mon_info.pop()
                if split_mon_info[-1].endswith(")"):
                    for i, info in enumerate(split_mon_info):
                        if info[0] == "(":
                            mon.species = " ".join(split_mon_info[i:])[1:-1]
                            split_mon_info = split_mon_info[:i]
                            break
                mon.nickname = " ".join(split_mon_info)

        return mon


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


class ConstantTeambuilder(Teambuilder):
    def __init__(self, team: str):
        if "|" in team:
            self._mons = self.parse_packed_team(team)
        else:
            self._mons = self.parse_showdown_team(team)

        self.packed_team = self.join_team(self._mons)

    def yield_team(self) -> str:
        return self.packed_team

    @property
    def team(self) -> List[TeambuilderPokemon]:
        return self._mons
