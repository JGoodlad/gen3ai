import pytest

from agents.gen3_data import species
from agents.gen3_data.species import SpeciesData


def test_known_species_fields():
    tt = species.species_data("tyranitar")
    assert isinstance(tt, SpeciesData)
    assert tt.id == "tyranitar"
    assert tt.num == 248
    assert tt.name == "Tyranitar"
    assert tt.base_stats == {"atk": 134, "def": 110, "hp": 100, "spa": 95, "spd": 100, "spe": 61}


def test_get_unknown_returns_none():
    assert species.get("notamon") is None
    assert species.get(None) is None


def test_species_data_raises_on_unknown():
    with pytest.raises(KeyError):
        species.species_data("notamon")


def test_dex_covers_all_gen3():
    # 386 base-form species (Bulbasaur #1 .. Deoxys #386).
    assert len(species.raw()) == 386
    assert len(species._dex()) == 386


def test_raw_matches_dex():
    raw = species.raw()
    for sid, sd in species._dex().items():
        assert sd.num == raw[sid]["num"]
