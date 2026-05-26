"""Pin the raw_id patch in Pokemon._update_from_teambuilder.

Without raw_id, "hiddenpowergrass" collapses to bare "hiddenpower" → Normal/0bp,
silently losing the type info that the team file declared. This test fails
loudly if the patch ever regresses.
"""
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.pokemon_type import PokemonType
from poke_env.teambuilder.teambuilder import Teambuilder


def _build_celebi_with_hp_grass() -> Pokemon:
    paste = """Celebi @ Leftovers
Ability: Natural Cure
EVs: 252 HP / 40 SpA / 32 Spe
Bold Nature
- Leech Seed
- Recover
- Perish Song
- Hidden Power [Grass]
"""
    parsed = Teambuilder.parse_showdown_team(paste)
    mon = Pokemon(gen=3)
    mon._update_from_teambuilder(parsed[0])
    return mon


def test_hidden_power_move_id_keeps_type_suffix():
    """The move in mon.moves must be keyed by the typed variant id, not bare hp."""
    mon = _build_celebi_with_hp_grass()
    assert "hiddenpowergrass" in mon.moves, (
        f"expected 'hiddenpowergrass' in moves, got {list(mon.moves.keys())}"
    )
    assert "hiddenpower" not in mon.moves, (
        "bare 'hiddenpower' must not coexist — would conflict in lookups"
    )


def test_hidden_power_move_type_is_grass():
    """move.type must return Grass — the whole point of the raw_id patch.
    If this returns NORMAL, the encoder downstream will mis-type the move."""
    mon = _build_celebi_with_hp_grass()
    move = mon.moves["hiddenpowergrass"]
    assert move.type == PokemonType.GRASS, (
        f"expected Grass, got {move.type}. raw_id is not being forwarded."
    )


def test_hidden_power_move_base_power_is_70():
    """Typed HP variant has basePower=70 in the move data, vs 0 for bare HP."""
    mon = _build_celebi_with_hp_grass()
    move = mon.moves["hiddenpowergrass"]
    assert move.base_power == 70, (
        f"expected 70 bp, got {move.base_power}. raw_id is not being forwarded."
    )


def test_hidden_power_different_types_resolve_distinctly():
    """Two different HP types in a team must resolve to two different move.type values."""
    paste = """Salamence @ Leftovers
Ability: Intimidate
EVs: 4 Atk / 252 SpA / 252 Spe
Rash Nature
- Dragon Claw
- Brick Break
- Hidden Power [Fire]
- Fire Blast

Celebi @ Leftovers
Ability: Natural Cure
EVs: 252 HP / 40 SpA / 32 Spe
Bold Nature
- Leech Seed
- Recover
- Perish Song
- Hidden Power [Grass]
"""
    parsed = Teambuilder.parse_showdown_team(paste)
    salamence = Pokemon(gen=3); salamence._update_from_teambuilder(parsed[0])
    celebi = Pokemon(gen=3); celebi._update_from_teambuilder(parsed[1])

    assert salamence.moves["hiddenpowerfire"].type == PokemonType.FIRE
    assert celebi.moves["hiddenpowergrass"].type == PokemonType.GRASS
