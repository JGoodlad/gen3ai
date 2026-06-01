import pytest
from poke_env.battle.move_category import MoveCategory
from poke_env.battle.pokemon_type import PokemonType

from agents.gen3_data import moves as movedex
from agents.gen3_data.moves import MoveData


def test_known_move_fields():
    tb = movedex.move_data("thunderbolt")
    assert tb.id == "thunderbolt"
    assert tb.num == 85
    assert tb.base_power == 95
    assert tb.type is PokemonType.ELECTRIC
    assert tb.category is MoveCategory.SPECIAL
    assert tb.accuracy == 100
    assert tb.has_secondary is True
    assert tb.has_recoil is False
    assert tb.is_damaging is True


def test_physical_move_category_from_type():
    # Gen3 category is type-based: Ground is a physical type.
    eq = movedex.move_data("earthquake")
    assert eq.type is PokemonType.GROUND
    assert eq.category is MoveCategory.PHYSICAL
    assert eq.is_damaging is True


def test_status_move_not_damaging():
    toxic = movedex.move_data("toxic")
    assert toxic.category is MoveCategory.STATUS
    assert toxic.base_power == 0
    assert toxic.is_damaging is False


def test_curse_typeless_loads():
    # Curse stores type "???" in the data → THREE_QUESTION_MARKS; 0-power → STATUS.
    curse = movedex.move_data("curse")
    assert curse.type is PokemonType.THREE_QUESTION_MARKS
    assert curse.category is MoveCategory.STATUS
    assert curse.is_damaging is False


def test_never_miss_flag_carried():
    # swift bypasses accuracy/evasion in the data → never_miss True.
    assert movedex.move_data("swift").never_miss is True
    assert movedex.move_data("thunderbolt").never_miss is False


def test_get_unknown_returns_none():
    assert movedex.get("notarealmove") is None
    assert movedex.get(None) is None


def test_move_data_raises_on_unknown():
    with pytest.raises(KeyError):
        movedex.move_data("notarealmove")


def test_is_damaging_helper():
    assert movedex.is_damaging("surf") is True
    assert movedex.is_damaging("toxic") is False
    assert movedex.is_damaging("notarealmove") is False  # unknown → not assumed damaging
    assert movedex.is_damaging(None) is False


def test_borrowed_enums_are_the_spec_enums():
    # Discipline check: the dex files data under poke-env's value-enums as keys —
    # type/category ARE PokemonType/MoveCategory members, so callers can key their
    # own tables by them (we never call methods on them).
    eq = movedex.move_data("earthquake")
    assert isinstance(eq.type, PokemonType)
    assert isinstance(eq.category, MoveCategory)


def test_all_moves_load_with_valid_enums():
    # Every JSON entry builds cleanly into the borrowed enums — proves there's no
    # transcription gap between the data's type/category names and the enums.
    dex = movedex._dex()
    assert len(dex) >= 300
    for md in dex.values():
        assert isinstance(md, MoveData)
        assert isinstance(md.type, PokemonType)
        assert isinstance(md.category, MoveCategory)
        assert 0 <= md.accuracy <= 100


def test_frozen():
    md = movedex.move_data("tackle")
    with pytest.raises(Exception):
        md.base_power = 999  # frozen dataclass
