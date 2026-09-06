"""Unit tests for the DECLARED priority sampler (`cf_producer_sampler.py`).

The arithmetic that decides WHICH decisions get a tight-MC label: the normalized
entropy, the conviction-region surprise term, the declared weighting, and the
move-round filter that says a forced switch is not labelable by this path.

These moved out of `cf_producer_test.py` with the functions they cover (2026-09-06, the file-size
ratchet's third cut). They still reach every subject through `cf_producer`'s re-exports — as `P.<name>`,
unchanged — which is what proves the extraction changed nothing a caller can see, and the
extraction-parity golden that pins it stays in `cf_producer_test.py` beside the fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest

# `P` is the HUB, deliberately: these tests assert the names still resolve there.
from agents.training import cf_producer as P


class TestPriorityScoring:
    def test_entropy_is_normalized_by_the_support_size(self):
        """A 2-way coin-flip must outrank a 9-way near-certainty, which raw entropy inverts."""
        coin = P.normalized_entropy([0.5, 0.5])
        wide = P.normalized_entropy([0.92] + [0.01] * 8)
        assert coin == pytest.approx(1.0)
        assert wide < coin, "un-normalized entropy would rank the 9-way state higher"

    def test_a_degenerate_decision_scores_zero_entropy(self):
        assert P.normalized_entropy([1.0]) == 0.0
        assert P.normalized_entropy([]) == 0.0
        assert P.normalized_entropy([1.0, 0.0, 0.0]) == 0.0

    def test_entropy_is_bounded_to_the_unit_interval(self):
        for k in (2, 3, 4, 11):
            assert P.normalized_entropy([1.0 / k] * k) == pytest.approx(1.0)

    def test_critic_surprise_is_the_conviction_region(self):
        # Sure of a win, and lost the battle: the "0.827 class" G0 measured at +0.23.
        assert P.critic_surprise(0.9, 0.0) == pytest.approx(0.9)
        # Sure of a win, and won: nothing to learn.
        assert P.critic_surprise(0.9, 1.0) == pytest.approx(0.1)

    def test_a_tie_is_half_not_a_loss(self):
        """A turn-cap draw is uninformative about conviction, not evidence the head was wrong."""
        assert P.critic_surprise(0.5, 0.5) == 0.0
        assert P.critic_surprise(1.0, 0.5) == pytest.approx(0.5)

    def test_no_win_prob_head_yields_no_surprise_term_rather_than_a_confident_zero(self):
        assert P.critic_surprise(None, 0.0) == 0.0
        assert P.critic_surprise(float("nan"), 0.0) == 0.0

    def test_surprise_dominates_the_declared_weighting(self):
        """The weights are a DECLARATION; this pins their ordering so a silent edit fails a test."""
        assert P.PRIORITY_WEIGHTS["critic_surprise"] > P.PRIORITY_WEIGHTS["policy_entropy"]
        # Max entropy cannot outrank a moderate surprise.
        assert P.priority_score(0.4, 0.0) > P.priority_score(0.0, 1.0)
        assert P.priority_score(0.5, 0.5) == pytest.approx(0.5 + 0.35 * 0.5)

    def test_sampler_version_is_stamped_and_stable(self):
        assert P.SAMPLER_VERSION == "cf_producer_priority_v1"


class TestMoveRoundFilter:
    def test_a_forced_switch_round_is_not_labelable(self):
        """The counterfactual divergence anchors at a start-of-turn MOVE round; a mask offering
        only switches is a mid-turn forced switch, which has no valid recorded answer to script."""
        switches_only = np.zeros(11, dtype=np.int8)
        switches_only[[1, 2, 3]] = 1
        assert not P.is_move_round(switches_only)

    def test_a_move_round_is_labelable(self):
        m = np.zeros(11, dtype=np.int8)
        m[[1, 6, 7]] = 1
        assert P.is_move_round(m)

    def test_struggle_counts_as_a_move_round(self):
        m = np.zeros(11, dtype=np.int8)
        m[10] = 1
        assert P.is_move_round(m)
