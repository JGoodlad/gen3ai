import pytest
import numpy as np
from unittest.mock import MagicMock
from .abilities import AbilitiesEncoder


_BASE_MAPPING = {
    "intimidate": {"num": 1},
    "levitate":   {"num": 2},
    "thickfat":   {"num": 3},
    "immunity":   {"num": 4},
    "wonderguard": {"num": 5},
}


def _mon(ability=None, species=None):
    m = MagicMock()
    m.ability = ability
    m.species = species
    return m


def test_abilities_encoder_dimension():
    encoder = AbilitiesEncoder(_BASE_MAPPING)
    assert encoder.dimension == 3


def test_abilities_encoder_revealed_writes_ability1_and_known():
    encoder = AbilitiesEncoder(_BASE_MAPPING)
    vec = encoder.encode(_mon(ability="Intimidate", species="snorlax"), None)
    assert vec[0] == 1.0    # ability1 = Intimidate num
    assert vec[1] == 0.0    # ability2 zeroed when revealed
    assert vec[2] == 1.0    # known


def test_abilities_encoder_none_returns_all_zeros():
    encoder = AbilitiesEncoder(_BASE_MAPPING)
    vec = encoder.encode(None, None)
    assert np.all(vec == 0.0)


def test_abilities_encoder_no_ability_no_species_priors_all_zero():
    encoder = AbilitiesEncoder(_BASE_MAPPING)
    vec = encoder.encode(_mon(ability=None, species="unknownmon"), None)
    assert vec[0] == 0.0
    assert vec[1] == 0.0
    assert vec[2] == 0.0    # not known


def test_abilities_encoder_unknown_placeholder_falls_back_to_species_prior():
    """poke-env reports 'unknownability' for opp mons whose ability hasn't fired;
    the encoder should still emit the dex priors for the species."""
    encoder = AbilitiesEncoder(
        _BASE_MAPPING,
        species_to_abilities={"snorlax": ["immunity", "thickfat"]},
    )
    vec = encoder.encode(_mon(ability="unknown_ability", species="snorlax"), None)
    assert vec[0] == 4.0   # immunity
    assert vec[1] == 3.0   # thickfat
    assert vec[2] == 0.0   # not known


def test_abilities_encoder_opp_unrevealed_uses_dex_priors():
    encoder = AbilitiesEncoder(
        _BASE_MAPPING,
        species_to_abilities={"snorlax": ["immunity", "thickfat"]},
    )
    vec = encoder.encode(_mon(ability=None, species="snorlax"), None)
    assert vec[0] == 4.0   # immunity (slot 0)
    assert vec[1] == 3.0   # thickfat (slot 1)
    assert vec[2] == 0.0   # not known


def test_abilities_encoder_single_ability_species_zeroes_slot2():
    """Shedinja has only Wonder Guard in Gen 3. ability2 must stay zero so the
    embedding lookup hits the sentinel rather than a stale second ability."""
    encoder = AbilitiesEncoder(
        _BASE_MAPPING,
        species_to_abilities={"shedinja": ["wonderguard", None]},
    )
    vec = encoder.encode(_mon(ability=None, species="shedinja"), None)
    assert vec[0] == 5.0   # wonderguard
    assert vec[1] == 0.0   # no second ability
    assert vec[2] == 0.0


def test_abilities_encoder_unknown_ability_name_raises():
    encoder = AbilitiesEncoder(_BASE_MAPPING)
    with pytest.raises(ValueError, match="Unrecognized ability"):
        encoder.encode(_mon(ability="MadeUpAbility", species="snorlax"), None)
