"""The acting rules and the policy-improvement target (pure numpy)."""
import math

import numpy as np
import pytest

from main.value_search import rules


def _logits(vals):
    lg = np.full(11, -np.inf)
    for a, v in vals.items():
        lg[a] = v
    return lg


def test_policy_order_breaks_ties_to_the_lower_index():
    lg = _logits({2: 1.0, 7: 1.0, 9: 3.0})
    assert rules.policy_order(lg, [2, 7, 9]) == [9, 2, 7]


def test_rescale_and_sigma():
    assert rules.rescale({1: 0.2, 2: 0.6}) == {1: 0.0, 2: 1.0}
    assert rules.rescale({1: 0.5, 2: 0.5}) == {1: 0.0, 2: 0.0}
    assert rules.sigma(1.0, 8) == pytest.approx((50 + 8) * 0.1)


def test_pick_ties_go_to_the_policy_order():
    assert rules.pick({3: 0.5, 6: 0.5 + 1e-9}, [6, 3]) == 6
    assert rules.pick({3: 0.5, 6: 0.5 + 1e-9}, [3, 6]) == 3
    assert rules.pick({3: 0.5, 6: 0.6}, [3, 6]) == 6


def test_gumbel_score_trades_prior_against_value():
    lg = _logits({0: 3.0, 1: 0.0})
    # a full win-scale gap (q̃ 1 vs 0) is worth (50 + N)·0.1 logits: 5.8 at N = 8 > the 3-logit prior gap
    assert rules.pick(rules.gumbel_scores(lg, {0: 0.4, 1: 0.6}, 8), [0, 1]) == 1
    # at N = 0 it is worth 5.0, still more than 3
    assert rules.pick(rules.gumbel_scores(lg, {0: 0.4, 1: 0.6}, 0), [0, 1]) == 1
    lg2 = _logits({0: 7.0, 1: 0.0})
    assert rules.pick(rules.gumbel_scores(lg2, {0: 0.4, 1: 0.6}, 8), [0, 1]) == 0


def test_improved_policy_is_the_prior_when_every_q_is_equal():
    lg = _logits({0: 1.0, 4: 0.2, 8: -0.5})
    legal = [0, 4, 8]
    pt = rules.improved_policy(lg, legal, {0: 0.5, 4: 0.5, 8: 0.5}, {0: 4, 4: 4, 8: 4}, 0.5)
    assert pt.sum() == pytest.approx(1.0)
    assert np.allclose(pt, rules.softmax_masked(lg, legal))
    assert pt[1] == 0.0


def test_improved_policy_moves_mass_toward_the_better_action():
    lg = _logits({0: 1.0, 4: 0.0})
    pt = rules.improved_policy(lg, [0, 4], {0: 0.3, 4: 0.7}, {0: 8, 4: 8}, 0.5)
    prior = rules.softmax_masked(lg, [0, 4])
    assert pt[4] > prior[4]
    assert pt[4] > 0.9


def test_completed_q_fills_unvisited_with_the_mixed_value():
    lg = _logits({0: 0.0, 1: 0.0, 2: 0.0})
    cq = rules.completed_q(lg, [0, 1, 2], {0: 0.8, 1: 0.2}, {0: 2, 1: 2}, 0.5)
    v_mix = (0.5 + 4 * 0.5) / 5
    assert cq[0] == 0.8 and cq[1] == 0.2 and cq[2] == pytest.approx(v_mix)
    assert rules.completed_q(lg, [0, 1, 2], {}, {}, 0.37) == {0: 0.37, 1: 0.37, 2: 0.37}


def _run(sh, samples):
    while not sh.done:
        req = sh.next_round()
        for a, n in req.items():
            k = len(sh.samples[a])
            sh.observe(a, samples[a][k:k + n])
    return sh


def test_sequential_halving_allocation_and_budget():
    lg = _logits({a: -a for a in range(7)})
    legal = list(range(7))
    rng = np.random.default_rng(0)
    samples = {a: rng.random(16).tolist() for a in legal}
    sh = _run(rules.SequentialHalving(logits=lg, legal=legal, budget=28, m=4, cap=16), samples)
    # R = 2 rounds: 4 candidates × ⌊28/8⌋ = 3, then 2 × ⌊28/4⌋ = 7
    assert sh.spent == 4 * 3 + 2 * 7
    assert sorted(a for a, n in sh.visits().items() if n) == [0, 1, 2, 3]
    assert sum(n == 10 for n in sh.visits().values()) == 2
    assert sh.pick() in (0, 1, 2, 3)
    assert not sh.capped


def test_sequential_halving_picks_the_dominant_action_and_respects_the_cap():
    lg = _logits({a: 0.0 for a in range(5)})
    legal = list(range(5))
    samples = {a: [1.0 if a == 3 else 0.0] * 16 for a in legal}
    sh = _run(rules.SequentialHalving(logits=lg, legal=legal, budget=400, m=None, cap=16), samples)
    assert sh.pick() == 3
    assert sh.capped and max(sh.visits().values()) == 16


def test_sequential_halving_minimum_one_sample_per_survivor():
    lg = _logits({a: -a for a in range(8)})
    sh = rules.SequentialHalving(logits=lg, legal=list(range(8)), budget=8, m=None, cap=16)
    req = sh.next_round()
    assert req == {a: 1 for a in range(8)}       # ⌊8 / (3·8)⌋ = 0 → 1 each
    assert sh.rounds == math.ceil(math.log2(8))
