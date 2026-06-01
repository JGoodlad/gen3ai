import pytest

from agents.gen3_data import abilities
from agents.gen3_data.abilities import AbilityData


def test_known_ability_fields():
    lev = abilities.ability_data("levitate")
    assert isinstance(lev, AbilityData)
    assert lev.id == "levitate"
    assert lev.name == "Levitate"
    assert lev.num == 26


def test_get_unknown_returns_none():
    assert abilities.get("notanability") is None
    assert abilities.get(None) is None


def test_ability_data_raises_on_unknown():
    with pytest.raises(KeyError):
        abilities.ability_data("notanability")


def test_only_gen3_abilities():
    # Air Lock (#76) is the last gen-3 ability; nothing later should be present.
    assert all(ab.num <= 76 for ab in abilities._dex().values())
