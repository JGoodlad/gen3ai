from agents.gen3_data import natures
from agents.gen3_data.natures import NatureData


def test_known_nature_multipliers():
    adamant = natures.get("adamant")
    assert isinstance(adamant, NatureData)
    assert adamant.multipliers == {"atk": 1.1, "def": 1.0, "spa": 0.9, "spd": 1.0, "spe": 1.0}


def test_neutral_nature_all_ones():
    bashful = natures.get("bashful")
    assert set(bashful.multipliers.values()) == {1.0}


def test_get_unknown_returns_none():
    assert natures.get("notanature") is None
    assert natures.get(None) is None


def test_twenty_five_natures():
    mult = natures.multipliers()
    assert len(mult) == 25
    # Spread-encoder shape: every nature carries exactly the 5 battle stats (HP is never modified).
    for name, m in mult.items():
        assert set(m) == {"atk", "def", "spa", "spd", "spe"}
