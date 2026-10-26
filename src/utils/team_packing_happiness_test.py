"""gen3_team_pack_happiness_v1 (2026-10-26 GIGO audit C-2): our paste -> packed path must carry happiness exactly as
Showdown's `Teams.import` + `Teams.pack` do. It dropped an explicit `Happiness: 0` (`self.happiness or ""`) and lacked
the importer's rule "a set running Frustration with no Happiness line gets 0" (sim/teams.ts), so the core unpacked
255 and Frustration hit at base power 1 (max((255 - 255) * 10 / 25, 1)) on 13 mons of the pool and the corpora.
The fail-closed guard over whole teams is the pack oracle (`utils.bridge.team_validator.check_pack_identity`, run by
every `Gen3Teambuilder` construction); these pin the two rules alone."""
from utils.team_packing import TeambuilderPokemon

_FRUSTRATION = """Snorlax @ Leftovers
Ability: Thick Fat
EVs: 252 HP / 4 Atk / 252 SpD
Careful Nature
- Frustration
- Curse
- Rest
- Sleep Talk"""


def _misc_happiness(packed: str) -> str:
    return packed.split("|")[11]


def test_frustration_without_a_happiness_line_packs_happiness_zero():
    mon = TeambuilderPokemon.from_showdown(_FRUSTRATION)
    assert mon.happiness == 0
    assert _misc_happiness(mon.packed) == "0"


def test_an_explicit_happiness_zero_is_packed_not_dropped():
    mon = TeambuilderPokemon.from_showdown(_FRUSTRATION.replace("Careful Nature", "Happiness: 0\nCareful Nature")
                                           .replace("- Frustration", "- Body Slam"))
    assert mon.happiness == 0 and _misc_happiness(mon.packed) == "0"


def test_default_and_stated_happiness_pack_as_showdown_packs_them():
    plain = TeambuilderPokemon.from_showdown(_FRUSTRATION.replace("- Frustration", "- Return"))
    assert plain.happiness is None and _misc_happiness(plain.packed) == ""          # 255 by default: omitted
    stated = TeambuilderPokemon.from_showdown(_FRUSTRATION.replace("Careful Nature", "Happiness: 255\nCareful Nature"))
    assert _misc_happiness(stated.packed) == ""                                     # Showdown omits 255 too
    mid = TeambuilderPokemon.from_showdown(_FRUSTRATION.replace("Careful Nature", "Happiness: 70\nCareful Nature"))
    assert _misc_happiness(mid.packed) == "70"
    # an explicit Happiness beside Frustration is kept (Showdown applies the 0 rule only when none is stated)
    assert mid.happiness == 70
