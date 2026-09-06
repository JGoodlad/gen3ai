"""Unit tests for the `reward/` export math (`gen3_reward_term_export_v1`).

Pure — no torch, no SB3, no filesystem. The claims worth pinning are the two the owner's brief
names: **the per-term shares partition the reward stream's movement**, and **the tracked terms sum
to the total** (which is what `reward/untracked_abs_mean` reports when they do not).
"""
from __future__ import annotations

import math

import pytest

from agents.training.reward_manager import (
    RewardBreakdown,
    RewardClass,
    RewardConfig,
    Gen3RewardManager,
    reward_class_composition,
)
from agents.training.reward_term_stats import (
    CLASS_NAMES,
    REFUND_FIELD,
    RewardTermAccumulator,
    merge_drained,
    reward_term_metrics,
    term_class_map,
    tracked_terms,
)


def _bd(**fields) -> RewardBreakdown:
    return RewardBreakdown(**fields)


class TestTheClassVocabulary:
    def test_class_names_match_the_reward_registrys_own_enum(self):
        # The module deliberately does NOT import RewardClass (it stays free of the reward
        # manager's heavy import graph), so the two lists are pinned against each other here.
        assert set(CLASS_NAMES) == {c.value for c in RewardClass}

    def test_the_refund_field_is_a_real_breakdown_field_and_not_a_registry_term(self):
        assert REFUND_FIELD in RewardBreakdown.field_names()
        assert REFUND_FIELD not in RewardBreakdown._REGISTRY


class TestTheTrackedSet:
    def test_the_production_composition_tracks_terminal_pbrs_bias_and_the_refund(self):
        comp = reward_class_composition(RewardConfig())
        terms = tracked_terms(comp)
        assert terms[0] == "win_loss"                      # terminal first
        assert terms[-1] == REFUND_FIELD                   # the mechanism last
        assert set(comp["pbrs_terms"]).issubset(terms)
        assert set(comp["bias_terms"]).issubset(terms)
        assert len(terms) == len(set(terms))               # no duplicates

    def test_every_tracked_term_is_a_real_breakdown_field(self):
        for cfg in (RewardConfig(), RewardConfig(all_shaping_pbrs=False)):
            for name in tracked_terms(reward_class_composition(cfg)):
                assert name in RewardBreakdown.field_names(), name

    def test_the_class_map_labels_every_tracked_term(self):
        comp = reward_class_composition(RewardConfig())
        cmap = term_class_map(comp)
        for name in tracked_terms(comp):
            assert cmap[name] in (*CLASS_NAMES, "refund"), name


class TestTheAccumulator:
    def test_it_sums_signed_and_absolute_separately(self):
        acc = RewardTermAccumulator(["win_loss", "pbrs_material"])
        acc.observe(_bd(pbrs_material=+2.0), total=+2.0)
        acc.observe(_bd(pbrs_material=-2.0), total=-2.0)
        d = acc.drain()
        assert d["n"] == 2
        # A telescoping potential's SIGNED sum cancels; its |·| sum does not — which is the whole
        # reason the share is |·|-weighted.
        assert d["sum"]["pbrs_material"] == pytest.approx(0.0)
        assert d["abs"]["pbrs_material"] == pytest.approx(4.0)

    def test_drain_zeroes_the_window_so_two_drains_cannot_double_count(self):
        acc = RewardTermAccumulator(["win_loss"])
        acc.observe(_bd(win_loss=30.0), total=30.0)
        first = acc.drain()
        second = acc.drain()
        assert first["n"] == 1 and first["sum"]["win_loss"] == pytest.approx(30.0)
        assert second["n"] == 0 and second["sum"]["win_loss"] == pytest.approx(0.0)

    def test_an_untracked_term_lands_in_the_residual_rather_than_vanishing(self):
        # `spikes` is a real BIAS field the accumulator was not told to track. The GIGO guard is
        # that its magnitude shows up in the residual, not that it is silently dropped.
        acc = RewardTermAccumulator(["win_loss"])
        bd = _bd(win_loss=30.0, spikes=0.5)
        acc.observe(bd, total=bd.total)
        d = acc.drain()
        assert d["residual_abs_sum"] == pytest.approx(0.5)

    def test_a_fully_tracked_breakdown_has_a_residual_of_exactly_zero(self):
        acc = RewardTermAccumulator(RewardBreakdown.field_names())
        bd = _bd(win_loss=-30.0, pbrs_material=1.25, spikes=0.5, bias_refund=-0.75)
        acc.observe(bd, total=bd.total)
        assert acc.drain()["residual_abs_sum"] == pytest.approx(0.0, abs=1e-12)


class TestMerging:
    def test_workers_sum(self):
        a = RewardTermAccumulator(["win_loss"])
        b = RewardTermAccumulator(["win_loss"])
        a.observe(_bd(win_loss=30.0), total=30.0)
        b.observe(_bd(win_loss=-30.0), total=-30.0)
        b.observe(_bd(win_loss=30.0), total=30.0)
        m = merge_drained([a.drain(), b.drain()])
        assert m["n"] == 3
        assert m["sum"]["win_loss"] == pytest.approx(30.0)
        assert m["abs"]["win_loss"] == pytest.approx(90.0)

    def test_none_workers_are_skipped_and_an_all_none_input_is_an_empty_window(self):
        assert merge_drained([None, None])["n"] == 0
        m = merge_drained([None, RewardTermAccumulator(["win_loss"]).drain()])
        assert m["n"] == 0


class TestTheMetrics:
    def test_an_empty_window_publishes_NOTHING(self):
        # A rollout that scored no decision must leave a GAP in the curves, not a confident zero.
        assert reward_term_metrics({"n": 0}, {}) == {}

    def test_the_abs_shares_sum_to_one(self):
        """THE OWNER'S CHECK: the per-term shares partition the reward stream's movement."""
        acc = RewardTermAccumulator(["win_loss", "pbrs_material", "no_progress_tax"])
        acc.observe(_bd(win_loss=30.0, pbrs_material=-1.5, no_progress_tax=-0.25),
                    total=30.0 - 1.5 - 0.25)
        acc.observe(_bd(pbrs_material=+0.75), total=0.75)
        m = reward_term_metrics(merge_drained([acc.drain()]),
                                {"win_loss": "terminal", "pbrs_material": "pbrs",
                                 "no_progress_tax": "bias"})
        shares = [v for k, v in m.items() if k.endswith("_abs_share")
                  and not k.startswith("class_")]
        assert len(shares) == 3
        assert sum(shares) == pytest.approx(1.0)
        # …and so do the CLASS rollups, over the same denominator.
        class_shares = [v for k, v in m.items() if k.startswith("class_")
                        and k.endswith("_abs_share")]
        assert sum(class_shares) == pytest.approx(1.0)

    def test_the_hand_computed_values(self):
        acc = RewardTermAccumulator(["win_loss", "pbrs_material"])
        acc.observe(_bd(win_loss=30.0, pbrs_material=-1.0), total=29.0)
        acc.observe(_bd(pbrs_material=+3.0), total=3.0)
        m = reward_term_metrics(merge_drained([acc.drain()]),
                                {"win_loss": "terminal", "pbrs_material": "pbrs"})
        assert m["n_decisions"] == 2.0
        assert m["total_mean"] == pytest.approx(16.0)          # (29 + 3) / 2
        assert m["total_abs_mean"] == pytest.approx(16.0)
        assert m["win_loss_mean"] == pytest.approx(15.0)       # 30 / 2
        assert m["pbrs_material_mean"] == pytest.approx(1.0)   # (-1 + 3) / 2
        assert m["pbrs_material_abs_mean"] == pytest.approx(2.0)
        # denominator = |30| + |−1| + |3| = 34
        assert m["win_loss_abs_share"] == pytest.approx(30.0 / 34.0)
        assert m["pbrs_material_abs_share"] == pytest.approx(4.0 / 34.0)
        assert m["class_terminal_abs_share"] == pytest.approx(30.0 / 34.0)
        assert m["class_pbrs_abs_share"] == pytest.approx(4.0 / 34.0)

    def test_a_signed_share_would_report_a_healthy_potential_as_inert(self):
        """WHY the share is |·|-weighted, expressed as the counterfactual it avoids."""
        acc = RewardTermAccumulator(["pbrs_material"])
        for v in (+1.0, -1.0, +2.0, -2.0):                 # perfect telescoping
            acc.observe(_bd(pbrs_material=v), total=v)
        m = reward_term_metrics(merge_drained([acc.drain()]), {"pbrs_material": "pbrs"})
        assert m["pbrs_material_mean"] == pytest.approx(0.0)     # the signed read: "inert"
        assert m["pbrs_material_abs_share"] == pytest.approx(1.0)  # the |·| read: "carries it all"

    def test_an_all_zero_window_publishes_NaN_shares_rather_than_zero(self):
        acc = RewardTermAccumulator(["win_loss"])
        acc.observe(_bd(), total=0.0)
        m = reward_term_metrics(merge_drained([acc.drain()]), {"win_loss": "terminal"})
        assert m["n_decisions"] == 1.0
        assert math.isnan(m["win_loss_abs_share"])
        assert math.isnan(m["class_terminal_abs_share"])

    def test_an_unmapped_term_rolls_up_as_other_rather_than_disappearing(self):
        acc = RewardTermAccumulator(["win_loss"])
        acc.observe(_bd(win_loss=1.0), total=1.0)
        m = reward_term_metrics(merge_drained([acc.drain()]), {})
        assert m["class_other_abs_share"] == pytest.approx(1.0)


class TestTheManagerSeam:
    def test_a_real_manager_tracks_its_own_composition_and_drains_to_None_twice_safely(self):
        mgr = Gen3RewardManager(config=RewardConfig())
        assert mgr._term_stats is not None
        assert set(mgr._term_stats.terms) == set(
            tracked_terms(reward_class_composition(mgr.config)))
        drained = mgr.drain_reward_terms()
        assert drained is not None and drained["n"] == 0

    def test_the_shadow_twin_never_accumulates(self):
        # The verify twin must stay observationally identical to the production manager, and an
        # accumulator on it would double-count every decision into a shared metric.
        mgr = Gen3RewardManager(config=RewardConfig(), _shadow=True)
        assert mgr._term_stats is None
        assert mgr.drain_reward_terms() is None

    def test_the_accumulator_survives_an_episode_reset(self):
        # The window is a ROLLOUT window, drained by the callback — not an episode window.
        mgr = Gen3RewardManager(config=RewardConfig())
        mgr._term_stats.observe(_bd(win_loss=30.0), total=30.0)
        mgr.reset()
        assert mgr.drain_reward_terms()["n"] == 1
