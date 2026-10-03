"""``main.h2h.stats`` — pooling an edge's batches, the pair-clustered interval and the regime refusal."""
from __future__ import annotations

import math

import pytest

from agents.training import mirrored_pairs as MP
from main.h2h import stats as ST


def row(counts, w=None, l=None, d=0, a="a", b="b", regime="r1", wall=10.0, purpose="audit"):
    n = sum(counts)
    half = sum(i * c for i, c in enumerate(counts))
    w = (half - d) // 2 if w is None else w
    l = 2 * n - w - d if l is None else l
    return {"player": {"id": f"run_{a}@1", "sha256": a * 64}, "opponent": {"id": f"run_{b}@1", "sha256": b * 64},
            "regime": {"regime_id": regime}, "purpose": purpose, "counts": {"w": w, "l": l, "d": d},
            "pairs": {"counts": list(counts), "n_pairs": n, "voided": 0},
            "compute": {"wall_s": wall, "near_tie_games": 1, "near_tie_decisions": 2, "device": "cpu", "backend": "eager"}}


def test_pair_se_matches_the_hand_computation_and_is_unbiased():
    counts = [10, 0, 60, 0, 30]                                     # 100 pairs
    mu = (0 * 10 + 2 * 60 + 4 * 30) / 4 / 100
    var = (10 * (0 - mu) ** 2 + 60 * (0.5 - mu) ** 2 + 30 * (1 - mu) ** 2) / 99
    assert ST.pair_se(counts) == pytest.approx(math.sqrt(var / 100), rel=1e-12)
    assert ST.pair_sd(counts) == pytest.approx(math.sqrt(var), rel=1e-12)
    assert ST.pair_se([5, 0, 0, 0, 0]) == 0.0 and ST.pair_se([1, 0, 0, 0, 0]) is None


def test_the_pair_interval_is_wider_than_the_per_game_binomial_when_pairs_are_correlated():
    counts = [40, 0, 20, 0, 40]                                     # pairs are all-or-nothing: games are NOT independent
    s = ST.edge_summary([row(counts)])
    lo, hi = s["score_ci95"]
    p = s["score"]
    binom = 1.96 * math.sqrt(p * (1 - p) / (2 * 100))               # what the per-game binomial would claim
    assert (hi - lo) / 2 > 1.2 * binom                # measured ratio 1.27 (sqrt 2 if no pair were a split)
    assert (hi - lo) / 2 == pytest.approx(MP.pair_score_ci(counts)[2] - p)


def test_edge_summary_pools_batches_exactly_and_reports_throughput():
    r1, r2 = row([1, 0, 8, 0, 1], wall=4.0), row([0, 1, 7, 1, 1], wall=6.0)
    s = ST.edge_summary([r1, r2])
    assert s["pairs"] == 20 and s["pair_counts"] == [1, 1, 15, 1, 2] and s["batches"] == 2
    assert s["games"] == 40 and s["w"] + s["l"] + s["d"] == 40
    assert s["wall_s"] == 10.0 and s["games_per_s"] == pytest.approx(4.0)
    assert s["near_tie_games"] == 2 and s["near_tie_decisions"] == 4
    assert s["score"] == pytest.approx(MP.pair_score([1, 1, 15, 1, 2]))
    assert s["win_rate"] == pytest.approx(s["w"] / 40)


def test_a_draw_makes_score_and_win_rate_differ_by_half_the_draw_rate():
    s = ST.edge_summary([row([0, 0, 4, 0, 0], w=3, l=3, d=2)])
    assert s["score"] == 0.5 and s["win_rate"] == pytest.approx(3 / 8)


def test_rows_of_two_regimes_or_two_edges_are_never_pooled():
    with pytest.raises(ST.MixedRegimeError, match="regime"):
        ST.edge_summary([row([0, 0, 2, 0, 0]), row([0, 0, 2, 0, 0], regime="r2")])
    with pytest.raises(ST.MixedRegimeError, match="edge"):
        ST.edge_summary([row([0, 0, 2, 0, 0]), row([0, 0, 2, 0, 0], b="c")])
    with pytest.raises(ST.MixedRegimeError, match="2 regimes"):
        ST.group_edges([row([0, 0, 2, 0, 0]), row([0, 0, 2, 0, 0], regime="r2")])
    g = ST.group_edges([row([0, 0, 2, 0, 0]), row([0, 0, 2, 0, 0], regime="r2")], regime_id="r2")
    assert len(g) == 1 and len(next(iter(g.values()))) == 1


def test_group_edges_keeps_the_two_directions_apart_and_format_is_readable():
    g = ST.group_edges([row([0, 0, 2, 0, 0], a="a", b="b"), row([0, 0, 2, 0, 0], a="b", b="a")])
    assert len(g) == 2
    line = ST.format_edge(ST.edge_summary([row([1, 0, 8, 0, 1])]))
    assert "score 50.00%" in line and "pentanomial [1, 0, 8, 0, 1]" in line
