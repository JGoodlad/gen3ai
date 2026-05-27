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
    "voltabsorb": {"num": 6},
    "illuminate": {"num": 7},
}


def _mon(ability=None, species=None):
    m = MagicMock()
    m.ability = ability
    m.species = species
    return m


# ---------------------------------------------------------------------------
# Dimension and basic shape
# ---------------------------------------------------------------------------

def test_abilities_encoder_dimension():
    encoder = AbilitiesEncoder(_BASE_MAPPING)
    assert encoder.dimension == 4   # [ability1_id, ability2_id, dominance, known]


def test_abilities_encoder_layout():
    encoder = AbilitiesEncoder(_BASE_MAPPING)
    layout = encoder.get_layout()
    assert layout["id1"]["offset"] == 0
    assert layout["id2"]["offset"] == 1
    assert layout["dominance"]["offset"] == 2
    assert layout["known"]["offset"] == 3


def test_abilities_encoder_none_returns_all_zeros():
    encoder = AbilitiesEncoder(_BASE_MAPPING)
    vec = encoder.encode(None, None)
    assert np.all(vec == 0.0)


# ---------------------------------------------------------------------------
# Revealed path (own team always; opp once an ability fires)
# ---------------------------------------------------------------------------

def test_revealed_ability_emits_id_dominance_known():
    encoder = AbilitiesEncoder(_BASE_MAPPING)
    vec = encoder.encode(_mon(ability="Intimidate", species="snorlax"), None)
    assert vec[0] == 1.0    # ability1 = Intimidate
    assert vec[1] == 0.0    # ability2 cleared
    assert vec[2] == 1.0    # dominance forced to 1.0 once revealed
    assert vec[3] == 1.0    # known


def test_revealed_overrides_species_priors():
    """Once an opp ability is revealed it must override the Smogon priors:
    Snorlax priors say 86% Immunity, but if Showdown reports Thick Fat, the
    encoder should emit Thick Fat with dominance=1, not the prior split."""
    encoder = AbilitiesEncoder(
        _BASE_MAPPING,
        species_to_ability_priors={"snorlax": {"immunity": 0.86, "thickfat": 0.14}},
    )
    vec = encoder.encode(_mon(ability="Thick Fat", species="snorlax"), None)
    assert vec[0] == 3.0    # Thick Fat
    assert vec[1] == 0.0
    assert vec[2] == 1.0
    assert vec[3] == 1.0


# ---------------------------------------------------------------------------
# Unrevealed path — Smogon priors
# ---------------------------------------------------------------------------

def test_unrevealed_uses_top2_sorted_by_probability():
    encoder = AbilitiesEncoder(
        _BASE_MAPPING,
        species_to_ability_priors={"snorlax": {"immunity": 0.86, "thickfat": 0.14}},
    )
    vec = encoder.encode(_mon(ability=None, species="snorlax"), None)
    assert vec[0] == 4.0                              # Immunity (top by prob)
    assert vec[1] == 3.0                              # Thick Fat (second)
    assert vec[2] == pytest.approx(0.86, abs=1e-5)    # dominance = prob(top1)
    assert vec[3] == 0.0                              # not known


def test_unrevealed_single_ability_species():
    """Single-ability species → ability2=0 and dominance=1.0."""
    encoder = AbilitiesEncoder(
        _BASE_MAPPING,
        species_to_ability_priors={"shedinja": {"wonderguard": 1.0}},
    )
    vec = encoder.encode(_mon(ability=None, species="shedinja"), None)
    assert vec[0] == 5.0    # Wonder Guard
    assert vec[1] == 0.0    # no second ability
    assert vec[2] == 1.0    # dominance
    assert vec[3] == 0.0


def test_unrevealed_unknown_species_emits_zeros():
    encoder = AbilitiesEncoder(_BASE_MAPPING)
    vec = encoder.encode(_mon(ability=None, species="madeupmon"), None)
    assert np.all(vec == 0.0)


def test_unknown_placeholder_falls_back_to_priors():
    """poke-env reports 'unknownability' for opp mons whose ability hasn't fired."""
    encoder = AbilitiesEncoder(
        _BASE_MAPPING,
        species_to_ability_priors={"lanturn": {"voltabsorb": 0.997, "illuminate": 0.003}},
    )
    vec = encoder.encode(_mon(ability="unknown_ability", species="lanturn"), None)
    assert vec[0] == 6.0                              # Volt Absorb
    assert vec[1] == 7.0                              # Illuminate
    assert vec[2] == pytest.approx(0.997, abs=1e-3)   # dominance
    assert vec[3] == 0.0


def test_priors_sorted_by_probability_descending():
    """Ranking must be by Smogon usage, not alphabetic. Even when the alphabetically
    earlier ability has the smaller share, it should appear in slot 2."""
    encoder = AbilitiesEncoder(
        _BASE_MAPPING,
        # 'illuminate' alphabetically < 'voltabsorb' but voltabsorb is dominant
        species_to_ability_priors={"lanturn": {"illuminate": 0.003, "voltabsorb": 0.997}},
    )
    vec = encoder.encode(_mon(ability=None, species="lanturn"), None)
    assert vec[0] == 6.0    # Volt Absorb in slot 1
    assert vec[1] == 7.0    # Illuminate in slot 2
    assert vec[2] == pytest.approx(0.997, abs=1e-3)


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------

def test_unknown_ability_name_raises():
    encoder = AbilitiesEncoder(_BASE_MAPPING)
    with pytest.raises(ValueError, match="Unrecognized ability"):
        encoder.encode(_mon(ability="MadeUpAbility", species="snorlax"), None)
