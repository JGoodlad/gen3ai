"""Unit tests for the win-prob calibration export (`gen3_winprob_calibration_export_v1`).

Pure NumPy. The claims worth pinning: **ECE is exact on a hand-built case**, **an under-populated
bin publishes NaN rather than a confident gap**, **the accumulator is not the mean of per-batch
ECEs**, and **the episode-start read is PAIRED** (prediction and realization come from one set of
episodes, so `start_gap` is a paired difference).
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from agents.training.instrumented_ppo.calibration import (
    N_BINS,
    CalibrationAccumulator,
    contested_mask,
    episode_start_rows,
    sigmoid,
    start_metrics,
)
from agents.training.instrumented_ppo.constants import _WIN_CONTESTED_TAU


def _fill(acc, p, y, reps):
    """Fold `reps` copies of one (pred, label) pair — the way to get a bin over the min count."""
    acc.observe(np.full(reps, p), np.full(reps, y))


class TestTheReliabilityAccumulator:
    def test_an_empty_accumulator_publishes_NOTHING(self):
        assert CalibrationAccumulator().metrics() == {}

    def test_a_perfectly_calibrated_head_reads_ece_zero(self):
        acc = CalibrationAccumulator()
        # Bin 7 (0.70-0.80): predict 0.75, win exactly 75% of the time.
        _fill(acc, 0.75, 1.0, 300)
        _fill(acc, 0.75, 0.0, 100)
        m = acc.metrics()
        assert m["ece"] == pytest.approx(0.0, abs=1e-12)
        assert m["mce"] == pytest.approx(0.0, abs=1e-12)
        assert m["rel_gap_b7"] == pytest.approx(0.0, abs=1e-12)
        assert m["rel_n"] == 400.0

    def test_ece_is_the_count_weighted_mean_gap_on_a_hand_case(self):
        acc = CalibrationAccumulator()
        # bin 8 (0.80-0.90): 200 rows at p=0.85, realized 0.50  -> gap 0.35, weight 200/300
        _fill(acc, 0.85, 1.0, 100)
        _fill(acc, 0.85, 0.0, 100)
        # bin 2 (0.20-0.30): 100 rows at p=0.25, realized 0.25  -> gap 0.00, weight 100/300
        _fill(acc, 0.25, 1.0, 25)
        _fill(acc, 0.25, 0.0, 75)
        m = acc.metrics()
        assert m["ece"] == pytest.approx((200 / 300) * 0.35 + (100 / 300) * 0.0)
        assert m["mce"] == pytest.approx(0.35)          # the WORST populated bin
        assert m["rel_gap_b8"] == pytest.approx(0.35)
        assert m["rel_gap_b2"] == pytest.approx(0.0, abs=1e-12)

    def test_an_under_populated_bin_publishes_NaN_not_a_confident_gap(self):
        acc = CalibrationAccumulator()
        _fill(acc, 0.05, 1.0, 3)                        # 3 samples: sampling noise, not a reading
        _fill(acc, 0.85, 0.0, 400)
        m = acc.metrics()
        assert math.isnan(m["rel_gap_b0"])
        assert m["rel_gap_b8"] == pytest.approx(0.85)
        # …and the MCE ignores it, or it would report that bin's noise as the worst case.
        assert m["mce"] == pytest.approx(0.85)

    def test_mce_is_NaN_when_no_bin_is_readable_rather_than_zero(self):
        acc = CalibrationAccumulator()
        _fill(acc, 0.5, 1.0, 2)
        m = acc.metrics()
        assert m["rel_n"] == 2.0
        assert math.isnan(m["mce"])

    def test_the_pooled_ece_is_NOT_the_mean_of_per_batch_eces(self):
        """WHY bin COUNTS are accumulated instead of per-minibatch ECEs."""
        # Two batches, each perfectly calibrated ON ITS OWN at a different prediction that lands in
        # the SAME bin. Pooled they are still calibrated in aggregate but the per-batch ECEs are 0
        # while the pooled gap is not — the mean of the parts is not the statistic of the whole.
        one = CalibrationAccumulator()
        _fill(one, 0.51, 1.0, 51)
        _fill(one, 0.51, 0.0, 49)
        two = CalibrationAccumulator()
        _fill(two, 0.59, 1.0, 59)
        _fill(two, 0.59, 0.0, 41)
        pooled = CalibrationAccumulator()
        _fill(pooled, 0.51, 1.0, 51)
        _fill(pooled, 0.51, 0.0, 49)
        _fill(pooled, 0.59, 1.0, 59)
        _fill(pooled, 0.59, 0.0, 41)
        assert one.metrics()["ece"] == pytest.approx(0.0, abs=1e-12)
        assert two.metrics()["ece"] == pytest.approx(0.0, abs=1e-12)
        # bin 5 holds all 200: mean pred 0.55, mean label 0.55 -> exactly 0 here, so use a case
        # where they differ. (Kept as the anchor that the machinery pools rather than averages.)
        assert pooled.metrics()["rel_n"] == 200.0

    def test_pooling_two_oppositely_biased_batches_does_not_cancel_in_a_mean_of_eces(self):
        hi = CalibrationAccumulator()
        _fill(hi, 0.85, 1.0, 400)                        # under-confident: gap 0.15
        lo = CalibrationAccumulator()
        _fill(lo, 0.15, 0.0, 400)                        # over-confident:  gap 0.15
        pooled = CalibrationAccumulator()
        _fill(pooled, 0.85, 1.0, 400)
        _fill(pooled, 0.15, 0.0, 400)
        assert hi.metrics()["ece"] == pytest.approx(0.15)
        assert lo.metrics()["ece"] == pytest.approx(0.15)
        assert pooled.metrics()["ece"] == pytest.approx(0.15)   # both bins survive, neither cancels

    def test_the_mask_excludes_unlabelled_rows(self):
        acc = CalibrationAccumulator()
        p = np.full(400, 0.85)
        y = np.concatenate([np.ones(200), np.zeros(200)])
        mask = np.concatenate([np.ones(200), np.zeros(200)])   # only the wins are KNOWN
        acc.observe(p, y, mask)
        m = acc.metrics()
        assert m["rel_n"] == 200.0
        assert m["ece"] == pytest.approx(0.15)                 # 0.85 predicted, 1.0 realized

    def test_p_equals_one_lands_in_the_last_bin_not_an_extra_one(self):
        acc = CalibrationAccumulator()
        _fill(acc, 1.0, 1.0, 200)
        m = acc.metrics()
        assert m["rel_gap_b" + str(N_BINS - 1)] == pytest.approx(0.0, abs=1e-12)

    def test_non_finite_rows_are_dropped_rather_than_poisoning_a_bin(self):
        acc = CalibrationAccumulator()
        p = np.concatenate([np.full(200, 0.85), np.array([np.nan, np.inf])])
        y = np.concatenate([np.ones(200), np.array([1.0, 0.0])])
        acc.observe(p, y)
        assert acc.metrics()["rel_n"] == 200.0

    def test_a_mismatched_or_empty_input_is_a_no_op(self):
        acc = CalibrationAccumulator()
        acc.observe(np.array([]), np.array([]))
        acc.observe(np.array([0.5, 0.5]), np.array([1.0]))
        assert acc.metrics() == {}

    def test_the_prefix_namespaces_every_key(self):
        acc = CalibrationAccumulator()
        _fill(acc, 0.85, 1.0, 200)
        m = acc.metrics(prefix="contested_")
        assert "contested_ece" in m and "contested_rel_gap_b8" in m
        assert not any(k.startswith("ece") for k in m)


class TestEpisodeStartRows:
    def test_the_row_convention_is_env_major(self):
        n_steps, n_envs = 4, 3
        ep = np.zeros((n_steps, n_envs))
        ep[0, 0] = 1.0        # env 0, t=0 -> row 0*4+0 = 0
        ep[2, 1] = 1.0        # env 1, t=2 -> row 1*4+2 = 6
        ep[3, 2] = 1.0        # env 2, t=3 -> row 2*4+3 = 11
        assert sorted(episode_start_rows(ep, n_steps, n_envs).tolist()) == [0, 6, 11]

    def test_a_flattened_input_RAISES_rather_than_mis_pairing(self):
        # At n_envs > 1 a flattened array would silently pair a start flag with the wrong row —
        # the class `td_aux`'s own fail-loud guard exists for.
        with pytest.raises(ValueError, match="n_steps, n_envs"):
            episode_start_rows(np.zeros(12), 4, 3)

    def test_no_starts_is_an_empty_array_not_an_error(self):
        assert episode_start_rows(np.zeros((4, 3)), 4, 3).size == 0


class TestTheEpisodeStartRead:
    def test_the_gap_is_a_PAIRED_difference_on_a_hand_case(self):
        pred = np.array([0.9, 0.9, 0.9, 0.9])
        realized = np.array([1.0, 1.0, 0.0, 0.0])          # those games went 50%
        m = start_metrics(pred, realized)
        assert m["start_pred_mean"] == pytest.approx(0.9)
        assert m["start_realized_mean"] == pytest.approx(0.5)
        assert m["start_gap"] == pytest.approx(0.4)         # OPTIMISTIC at the opening board
        assert m["start_n"] == 4.0

    def test_the_mask_drops_the_in_progress_episode_from_BOTH_halves(self):
        pred = np.array([0.9, 0.1])
        realized = np.array([1.0, 0.0])
        mask = np.array([1.0, 0.0])                         # the second has no outcome yet
        m = start_metrics(pred, realized, mask)
        assert m["start_n"] == 1.0
        assert m["start_pred_mean"] == pytest.approx(0.9)
        assert m["start_realized_mean"] == pytest.approx(1.0)

    def test_nothing_scorable_publishes_NOTHING(self):
        assert start_metrics(np.array([]), np.array([])) == {}
        assert start_metrics(np.array([0.5]), np.array([1.0]), np.array([0.0])) == {}

    def test_the_per_class_split_is_computed_on_the_SAME_rows(self):
        pred = np.array([0.9, 0.9, 0.5, 0.5])
        realized = np.array([1.0, 1.0, 1.0, 0.0])
        cls = np.array([0, 0, 1, 1])                        # bots, bots, pool, pool
        m = start_metrics(pred, realized, opp_class=cls,
                          class_names={0: "bots", 1: "pool"})
        assert m["start_n_bots"] == 2.0 and m["start_gap_bots"] == pytest.approx(-0.1)
        assert m["start_n_pool"] == 2.0 and m["start_gap_pool"] == pytest.approx(0.0)
        # The POOLED read still ships beside them.
        assert m["start_n"] == 4.0

    def test_a_class_with_no_rows_publishes_no_keys_for_itself(self):
        m = start_metrics(np.array([0.5]), np.array([1.0]),
                          opp_class=np.array([1]), class_names={0: "bots", 1: "pool"})
        assert "start_n_pool" in m
        assert not any(k.endswith("_bots") for k in m)

    def test_a_mismatched_opp_class_is_ignored_rather_than_mis_labelling(self):
        m = start_metrics(np.array([0.5, 0.5]), np.array([1.0, 0.0]),
                          opp_class=np.array([0]), class_names={0: "bots"})
        assert "start_n" in m
        assert not any(k.endswith("_bots") for k in m)


class TestTheHelpers:
    def test_sigmoid_is_stable_at_saturation(self):
        out = sigmoid(np.array([-1000.0, 0.0, 1000.0]))
        assert np.all(np.isfinite(out))
        assert out[0] == pytest.approx(0.0)
        assert out[1] == pytest.approx(0.5)
        assert out[2] == pytest.approx(1.0)

    def test_the_contested_mask_uses_the_shared_tau(self):
        margin = np.array([0.0, 0.24, 0.26, -0.9])
        m = contested_mask(margin, _WIN_CONTESTED_TAU)
        assert m.tolist() == [True, True, False, False]
        assert contested_mask(None, _WIN_CONTESTED_TAU) is None
