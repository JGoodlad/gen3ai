"""The Python packed-team seams tolerate the gender DRAW PAD (``<g>*``, ``utils.team_packing.GENDER_DRAW_PAD``).

The search world builder writes the marker into an opponent world team; every Python reader that parses such a team
must keep the rest of the set intact and must not lose the marker on a re-pack. The engine side is
``src/rust_sim/tests/gender_draw_pad_test.rs``.
"""

from __future__ import annotations

from agents.gen3_data import team_legality
from utils.team_packing import GENDER_DRAW_PAD, Teambuilder, TeambuilderPokemon

PADDED = "Tauros||leftovers|intimidate|bodyslam,earthquake,rockslide,return|Jolly|,252,,,4,252|M*||||"
PLAIN = "Gengar||leftovers|levitate|thunderbolt,icepunch,firepunch,explosion|Timid|,,,252,4,252|||||"


def test_the_marker_is_one_named_character():
    assert GENDER_DRAW_PAD == "*"
    assert PADDED.split("|")[7] == "M" + GENDER_DRAW_PAD


def test_teambuilder_keeps_the_marker_through_a_repack():
    mon = TeambuilderPokemon.from_packed(PADDED)
    assert mon.gender == "M" + GENDER_DRAW_PAD
    assert mon.packed.split("|")[7] == "M" + GENDER_DRAW_PAD
    team = Teambuilder.parse_packed_team(PADDED + "]" + PLAIN)
    assert [(m.nickname, m.gender) for m in team] == [("Tauros", "M" + GENDER_DRAW_PAD), ("Gengar", None)]
    assert team[0].moves == ["bodyslam", "earthquake", "rockslide", "return"]
    assert TeambuilderPokemon.from_packed(PLAIN).gender is None


def test_team_legality_parses_a_padded_team_like_its_unmarked_twin():
    unmarked = PADDED.replace("|M*|", "|M|")
    assert team_legality.parse_packed(PADDED + "]" + PLAIN) == team_legality.parse_packed(unmarked + "]" + PLAIN)
    assert [s.species for s in team_legality.parse_packed(PADDED)] == ["tauros"]
