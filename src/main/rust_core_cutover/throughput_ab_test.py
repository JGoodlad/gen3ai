"""The throughput A/B's parse and its pre-registered verdict (the delta's own CI inside the bar)."""
from main.rust_core_cutover import throughput_ab as T


def test_parse_reads_the_benchmark_medians():
    text = ("    full decision cycle          :   1.535 ms   (p90 1.837)\n"
            "    └ our controllable CPU       :   0.925 ms   (60% of the cycle; p90 1.112)\n")
    assert T.parse(text) == {"cycle_ms": 1.535, "ours_ms": 0.925}
    assert T.parse("nothing") == {"cycle_ms": None, "ours_ms": None}


def _pairs(deltas):
    return [{"python": {"cycle_ms": 1.0, "ours_ms": 1.0}, "core": {"cycle_ms": 1.0 + d, "ours_ms": 0.8}}
            for d in deltas]


def test_non_regression_needs_the_whole_ci_under_the_bar():
    assert T.verdict(_pairs([0.0, 0.01, -0.01, 0.005, 0.0, -0.005]))["non_regression"]
    # a mean under the bar with a CI that straddles it is NOT a pass
    v = T.verdict(_pairs([0.0, 0.06, -0.02, 0.05, 0.0, -0.03]))
    assert v["cycle_rel_mean"] < T.BAR and not v["non_regression"]
    assert not T.verdict(_pairs([0.08] * 6))["non_regression"]
