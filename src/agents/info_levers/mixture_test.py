"""The shared mixture of a lever (``agents.info_levers.mixture``): targets, the presence mixture, and the
conditional-Poisson draw against its exact inclusion probabilities."""
import numpy as np

from agents.info_levers import mixture as MX

W = {"a": 0.5, "b": 0.3, "c": 0.15, "d": 0.05}


def test_categorical_targets():
    rng = np.random.default_rng(0)
    assert MX.categorical_target("truth", "c", W, rng) == "c"
    assert MX.categorical_target("map", "c", W, rng) == "a"
    assert MX.categorical_target("map", "c", W, rng, exclude=["a"]) == "b"
    draws = {MX.categorical_target("placebo", "a", W, rng) for _ in range(200)}
    assert draws == {"b", "c", "d"}, "a placebo is any plausible value but the truth"
    assert MX.mix_draw(rng, 1.0, "d", W) == "d"
    hits = sum(MX.mix_draw(rng, 0.5, "d", W) == "d" for _ in range(4000))
    assert 0.5 * 4000 < hits < 0.56 * 4000          # p + (1 - p)·π(d) = 0.525


def test_set_targets_never_repeat_and_keep_at_most_one_of_a_family():
    rng = np.random.default_rng(1)
    w = {"hp1": 0.9, "hp2": 0.8, "x": 0.5, "y": 0.4, "z": 0.1}
    fam = (lambda m: m.startswith("hp"))
    assert MX.set_target("map", ["x"], w, 3, rng, at_most_one=fam) == ["hp1", "x", "y"]
    for _ in range(50):
        t = MX.set_target("placebo", ["x", "y"], w, 2, rng, at_most_one=fam)
        assert len(t) == 2 and not ({"x", "y"} & set(t)) and sum(map(fam, t)) <= 1
    assert MX.set_target("truth", ["x", "q"], w, 2, rng, exclude=["q"]) == ["x"]


def test_mix_presence_is_the_prior_at_zero_and_the_target_at_one():
    pres0 = MX.mix_presence(W, ["d"], 0.0, 2)
    assert {k: round(v, 12) for k, v in pres0.items()} == {"a": 1.0, "b": 0.6, "c": 0.3, "d": 0.1}
    pres1 = MX.mix_presence(W, ["d", "new"], 1.0, 2)
    assert pres1["d"] == 1.0 and pres1["new"] == 1.0 and pres1["a"] == 0.0


def test_cp_sample_matches_its_exact_inclusion_probabilities():
    rng = np.random.default_rng(7)
    keys = ["a", "b", "c", "d", "e"]
    pis = [0.9, 0.6, 0.3, 0.15, 0.05]
    incl = MX.cp_inclusion(pis, 2)
    assert abs(sum(incl) - 2.0) < 1e-9
    n = 20000
    counts = dict.fromkeys(keys, 0)
    for _ in range(n):
        got = MX.cp_sample(rng, keys, pis, 2)
        assert got is not None and len(set(got)) == 2
        for k in got:
            counts[k] += 1
    for k, p in zip(keys, incl):
        sd = (p * (1 - p) / n) ** 0.5
        assert abs(counts[k] / n - p) < 5 * sd + 1e-9, (k, counts[k] / n, p)
    # a presence of 1 is drawn surely, 0 never
    assert set(MX.cp_sample(rng, ["s", "t", "u"], [1.0, 0.5, 0.0], 2)) == {"s", "t"}
    assert MX.cp_inclusion([1.0, 0.5, 0.0], 2) == [1.0, 1.0, 0.0]
