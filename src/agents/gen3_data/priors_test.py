from agents.gen3_data import priors


def test_ability_prior_snorlax():
    # Dex-anchored Smogon usage: ~86% Immunity / ~14% Thick Fat (see compute_priors.py).
    p = priors.ability("snorlax")
    assert abs(p["immunity"] - 0.86) < 0.02
    assert abs(p["thickfat"] - 0.14) < 0.02
    assert abs(sum(p.values()) - 1.0) < 1e-6


def test_hidden_power_prior_normalized():
    p = priors.hidden_power("zapdos")
    assert p  # non-empty
    assert abs(sum(p.values()) - 1.0) < 1e-6


def test_unknown_species_empty_dict():
    assert priors.ability("notamon") == {}
    assert priors.hidden_power("notamon") == {}
