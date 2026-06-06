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
    assert priors.moves("notamon") == {}
    assert priors.items("notamon") == {}
    assert priors.spreads("notamon") == []
    assert priors.stat_distribution("notamon", "spe") == ()


def test_move_priors_tyranitar():
    p = priors.moves("tyranitar")
    assert p["rockslide"] > 0.3 and p["earthquake"] > 0.3   # the STAB staples
    assert all(0.0 < v <= 1.0 for v in p.values())          # per-move probabilities
    assert sum(p.values()) > 2.0                             # ~4 moves/set → sums well above 1


def test_item_priors_choiceband_signal():
    tt = priors.items("tyranitar")
    assert abs(sum(tt.values()) - 1.0) < 1e-6
    assert 0.0 < tt.get("choiceband", 0) < 0.2              # TTar runs some CB (the inferred item)
    assert priors.items("suicune").get("choiceband", 0) < 0.01  # walls ~never CB (tiny usage noise ok)


def test_stat_distribution_speed_gen3_formula():
    # Tyranitar base spe 61; Jolly 252 → 2*61+31+63+5=221, *11//10=243; Adamant 252 (neutral spe)=221.
    dist = dict(priors.stat_distribution("tyranitar", "spe"))
    assert 243 in dist and 221 in dist
    assert abs(sum(dist.values()) - 1.0) < 1e-6
    # monotone: a faster spread (Jolly 252) is a real, distinct point in the belief
    assert max(dist) >= 243


def test_stat_distribution_attack_present():
    d = priors.stat_distribution("tyranitar", "atk")
    assert d and abs(sum(w for _, w in d) - 1.0) < 1e-6
    assert max(v for v, _ in d) > 300   # max-Atk TTar (base 134) is a hard hitter
