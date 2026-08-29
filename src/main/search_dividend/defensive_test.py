"""Tests for DEFENSIVE PAIRED SEARCH (``--root-strategy defensive``).

The module under test is pure arithmetic plus one seam, so almost everything here runs without a
sim, a session or a model. Three tests are deliberately NOT of that shape and each pins a claim
that only holds at a boundary:

* :func:`test_the_leaf_seam_raises_when_the_scorer_returns_the_value_head` drives
  ``SearchEngine._score_batch`` against a checkpoint with no win-prob head. It is the guard on
  probe G's one build-deciding number, and it FAILS if anyone reverts the leaf to scalar V.
* :func:`test_off_is_byte_identical_to_racing_and_grid` asserts the OFF paths are untouched — the
  same discipline ``racing`` shipped under, because an experiment that perturbs its own control is
  not an experiment.
* the ``_apply_defensive`` tests drive the real engine method over a scripted
  :class:`~main.search_dividend.search.DecisionResult`, because the futility rule reads
  ``widths.racing_resolved`` and a re-implementation of that read in the test would pin nothing.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from main.search_dividend import defensive as dfn
from main.search_dividend.budget import FALLBACK_REASONS, RealizedWidths
from main.search_dividend.racing import RacingConfig
from main.search_dividend.search import ROOT_STRATEGIES, DecisionResult, SearchConfig, SearchEngine

# ---------------------------------------------------------------------------
# the config
# ---------------------------------------------------------------------------


def test_defensive_is_a_registered_root_strategy_and_grid_is_still_the_default():
    assert "defensive" in ROOT_STRATEGIES
    assert SearchConfig().root_strategy == "grid"


def test_the_config_refuses_a_leaf_it_does_not_know():
    with pytest.raises(ValueError, match="unknown defensive leaf"):
        dfn.DefensiveConfig(leaf="value_head")


@pytest.mark.parametrize("margin", [-0.01, 0.51, 1.0])
def test_the_config_refuses_a_margin_outside_the_only_range_the_rule_means_anything_on(margin):
    with pytest.raises(ValueError, match="wp_margin"):
        dfn.DefensiveConfig(wp_margin=margin)


def test_the_config_refuses_a_negative_confirm_budget():
    with pytest.raises(ValueError, match="confirm_rollouts"):
        dfn.DefensiveConfig(confirm_rollouts=-1)


def test_the_default_operating_point_is_probe_h_s():
    cfg = dfn.DefensiveConfig()
    assert cfg.wp_margin == 0.15
    assert cfg.leaf == "winprob"
    assert cfg.confirm_rollouts == 0          # the first cell runs ONE new mechanism
    assert cfg.contested_deadline_s is None   # ...and iteration 2 adds exactly one more


# ---------------------------------------------------------------------------
# the CONTESTED DEADLINE (iteration 2's one change) — see `DefensiveConfig`
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", [0.0, -1.0, -0.001])
def test_the_config_refuses_a_non_positive_contested_deadline(bad):
    """0 is not "off" — off is ``None``. A zero deadline expires before round 1 and turns every
    contested decision into a counted fallback, which reads like a strategy that refused rather
    than a clock that was never granted."""
    with pytest.raises(ValueError, match="contested_deadline_s"):
        dfn.DefensiveConfig(contested_deadline_s=bad)


@pytest.mark.parametrize("budget", [0.5, 1.0, 3.0, 8.0])
def test_an_unset_contested_deadline_is_the_first_cells_behaviour_exactly(budget):
    """THE REVERT-CATCHER for the iteration-1 baseline: with the knob unset, the clock a contested
    decision gets is ``--budget`` and nothing else, at every budget the battery ships."""
    assert dfn.DefensiveConfig().deadline_for(budget) == budget
    cfg = SearchConfig(root_strategy="defensive", budget_s=budget)
    assert cfg.contested_budget_s() == budget


def test_a_set_contested_deadline_replaces_the_budget_for_a_contested_decision():
    cfg = SearchConfig(root_strategy="defensive", budget_s=1.0,
                       defensive=dfn.DefensiveConfig(contested_deadline_s=3.0))
    assert cfg.contested_budget_s() == 3.0
    # ...and the notional per-decision budget is UNCHANGED, because that is what the gate hands
    # back on a forced decision and what the banked total is quoted against.
    assert cfg.budget_s == 1.0


@pytest.mark.parametrize("strategy", ["grid", "racing"])
def test_a_contested_deadline_is_inert_off_the_defensive_strategy(strategy):
    cfg = SearchConfig(root_strategy=strategy, budget_s=1.0,
                       defensive=dfn.DefensiveConfig(contested_deadline_s=30.0))
    assert cfg.contested_budget_s() == 1.0


def _capture_search_clock(cfg: SearchConfig, win_prob: float) -> dict:
    """Drive the REAL ``choose`` far enough to read the clock the search was handed.

    ``_run`` is the one method every root strategy funnels through — the racer consults exactly
    this ``Deadline`` in its ``deadline.fits(batch_cost)`` guard — so capturing its arguments is
    the seam, not a re-derivation of the arithmetic above.
    """
    eng = _engine(cfg, _FakeModel(3))
    seen: dict = {}

    def _spy(record, side, turn, our_history, our_tokens, observed_our_lines, pub,
             policy_action, opp_true_packed, plan, widths, deadline):
        seen["deadline_budget_s"] = deadline.budget_s
        seen["plan"] = plan
        return DecisionResult(policy_action, "no_candidates", widths,
                              policy_action=policy_action)

    eng._run = _spy                                     # type: ignore[assignment]
    eng.choose(record=None, side="p1", turn=5, our_history=[],
               our_tokens={0: "move 1", 1: "move 2", 6: "switch 2"},
               observed_our_lines=[], pub=None, policy_action=0, root_win_prob=win_prob)
    return seen


def test_the_contested_deadline_actually_reaches_the_racer():
    """THE SEAM. A knob that parses, validates and is never consulted is the failure this pins."""
    base = SearchConfig(arm="honest", root_strategy="defensive", budget_s=1.0)
    assert _capture_search_clock(base, 0.52)["deadline_budget_s"] == 1.0
    bumped = replace(base, defensive=dfn.DefensiveConfig(contested_deadline_s=3.0))
    assert _capture_search_clock(bumped, 0.52)["deadline_budget_s"] == 3.0


def test_the_width_plan_is_sized_to_the_SAME_clock_the_deadline_enforces():
    """One number, not two. A plan sized to a budget the clock does not honour over-runs it; a
    plan sized below the clock leaves the extra seconds structurally unreachable — which is the
    precise defect the first cell measured on the FORCED side of the gate."""
    bumped = SearchConfig(arm="honest", root_strategy="defensive", budget_s=1.0,
                          defensive=dfn.DefensiveConfig(contested_deadline_s=3.0))
    seen = _capture_search_clock(bumped, 0.52)
    from main.search_dividend.budget import allocate
    eng = _engine(bumped, _FakeModel(3))
    assert seen["plan"] == allocate(3.0, 3, eng._cost, bumped.resolved_caps())


def test_a_FORCED_decision_still_banks_the_uniform_notional_not_the_contested_deadline():
    """The bank is what a UNIFORM search would have burned on the decisions the gate declined —
    that is the quantity a time manager redistributes. Charging a forced decision the contested
    deadline it never received would inflate the bank by the very knob that spends it."""
    reason, w = _gate(SearchConfig(root_strategy="defensive", budget_s=1.0,
                                   defensive=dfn.DefensiveConfig(contested_deadline_s=3.0)),
                      6, 0.93)
    assert reason == "defensive_forced"
    assert w.defensive_banked_s == 1.0


# ---------------------------------------------------------------------------
# the GATE (probe H)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_legal", [0, 1])
def test_a_decision_with_no_real_choice_is_forced_without_consulting_the_critic(n_legal):
    # `win_prob=None` would RAISE on the win-prob clause, so reaching FORCED here proves the
    # n_legal clause is checked first and independently — which is the point of it being separate.
    assert dfn.gate(n_legal, None) == dfn.GATE_N_LEGAL


@pytest.mark.parametrize("wp", [0.0, 0.05, 0.34, 0.35, 0.65, 0.66, 0.95, 1.0])
def test_a_decided_position_is_forced(wp):
    assert dfn.gate(4, wp) == dfn.GATE_WP_EXTREME


@pytest.mark.parametrize("wp", [0.36, 0.45, 0.5, 0.55, 0.64])
def test_a_contested_position_is_searched(wp):
    assert dfn.gate(4, wp) == dfn.GATE_SEARCH


def test_the_threshold_is_inclusive_at_exactly_the_margin():
    # `>=`, not `>`. Stated as a test because the boundary is where a swept operating point lives
    # and an off-by-one-comparison changes the forced mass by whatever sits exactly on it.
    assert dfn.gate(4, 0.5 + dfn.DEFAULT_WP_MARGIN) == dfn.GATE_WP_EXTREME
    assert dfn.gate(4, 0.5 - dfn.DEFAULT_WP_MARGIN) == dfn.GATE_WP_EXTREME
    assert dfn.gate(4, 0.5 + dfn.DEFAULT_WP_MARGIN - 1e-9) == dfn.GATE_SEARCH


def test_the_margin_is_configurable_and_moves_the_frontier_the_right_way():
    tight, loose = dfn.DefensiveConfig(wp_margin=0.05), dfn.DefensiveConfig(wp_margin=0.30)
    assert dfn.gate(4, 0.60, tight) == dfn.GATE_WP_EXTREME     # forced at 0.05
    assert dfn.gate(4, 0.60, loose) == dfn.GATE_SEARCH         # searched at 0.30


def test_a_missing_win_prob_is_refused_rather_than_imputed_to_one_half():
    # The most expensive possible reading of an absent number would be "maximally contested",
    # which routes EVERY decision into the searched class on a headless checkpoint.
    with pytest.raises(ValueError, match="needs a win probability"):
        dfn.gate(6, None)


# ---------------------------------------------------------------------------
# the LEAF (probe G) — the seam, and the assertion on it
# ---------------------------------------------------------------------------


def test_the_defensive_score_mode_is_named_explicitly_never_auto():
    cfg = SearchConfig(root_strategy="defensive", score="auto")
    assert cfg.effective_score() == "win_prob"


def test_the_value_leaf_is_selectable_but_is_not_the_default():
    cfg = SearchConfig(root_strategy="defensive", defensive=dfn.DefensiveConfig(leaf="value"))
    assert cfg.effective_score() == "value"
    assert SearchConfig(root_strategy="defensive").effective_score() == "win_prob"


@pytest.mark.parametrize("strategy", ["grid", "racing"])
@pytest.mark.parametrize("score", ["auto", "value", "win_prob"])
def test_every_other_strategy_passes_its_own_score_flag_through_unchanged(strategy, score):
    assert SearchConfig(root_strategy=strategy, score=score).effective_score() == score


def test_check_leaf_is_a_no_op_off_the_defensive_strategy():
    dfn.check_leaf("value", None)          # must not raise


def test_check_leaf_accepts_the_leaf_it_asked_for():
    dfn.check_leaf("win_prob", dfn.DefensiveConfig())
    dfn.check_leaf("value", dfn.DefensiveConfig(leaf="value"))


def test_check_leaf_raises_when_the_scorer_silently_substituted_the_value_head():
    with pytest.raises(dfn.DefensiveLeafError, match="win_prob"):
        dfn.check_leaf("value", dfn.DefensiveConfig())


class _FakeExtractor:
    """Stands in for the real one. ``last_win_prob_logits`` of ``None`` is exactly what a run
    trained with ``--win-prob-mode none`` publishes."""

    def __init__(self, wp):
        self.last_win_prob_logits = wp


class _FakePolicy:
    def __init__(self, n, wp):
        self.features_extractor = _FakeExtractor(wp)
        self._n = n

    def predict_values(self, inp):
        import torch

        return torch.zeros((int(inp["observation"].shape[0]), 1))


class _FakeModel:
    def __init__(self, n, wp=None):
        self.policy = _FakePolicy(n, wp)
        self.device = "cpu"


def _engine(cfg: SearchConfig, model) -> SearchEngine:
    return SearchEngine(model=model, mappings=None, cfg=cfg)


def test_the_leaf_seam_raises_when_the_scorer_returns_the_value_head():
    """THE guard on probe G's build-deciding number.

    A checkpoint with no win-prob head makes ``batch_scores`` return the SCALAR value readout with
    no error at all — the search then runs to completion and reports a full set of healthy
    counters while ranking on the arm that does NOT beat the played action (+0.0135
    [-0.0007, +0.0280] against the win-prob head's +0.0219 [+0.0089, +0.0364]). Reverting the
    defensive leaf to scalar V, or dropping the check, fails HERE.
    """
    eng = _engine(SearchConfig(root_strategy="defensive"), _FakeModel(3, wp=None))
    obs = np.zeros((3, 4), dtype=np.float32)
    mask = np.ones((3, 11), dtype=np.float32)
    with pytest.raises(dfn.DefensiveLeafError):
        eng._score_batch(obs, mask)


def test_the_same_headless_checkpoint_is_fine_on_grid_and_on_racing():
    """The seam must not become a new way for the CONTROL arms to fail."""
    obs = np.zeros((3, 4), dtype=np.float32)
    mask = np.ones((3, 11), dtype=np.float32)
    for strategy in ("grid", "racing"):
        eng = _engine(SearchConfig(root_strategy=strategy), _FakeModel(3, wp=None))
        scores, mode = eng._score_batch(obs, mask)
        assert mode == "value" and scores.shape == (3,)


def test_the_seam_passes_the_win_prob_head_through_when_it_exists():
    import torch

    eng = _engine(SearchConfig(root_strategy="defensive"),
                  _FakeModel(3, wp=torch.zeros((3, 1))))
    scores, mode = eng._score_batch(np.zeros((3, 4), dtype=np.float32),
                                    np.ones((3, 11), dtype=np.float32))
    assert mode == "win_prob"
    assert np.allclose(scores, 0.5)          # sigmoid(0)


# ---------------------------------------------------------------------------
# the FUTILITY rule (probe I) — an overrule requires SEPARATION
# ---------------------------------------------------------------------------


def test_a_race_that_never_separated_is_futility_whatever_its_leader_says():
    assert dfn.verdict(False, race_action=7, policy_action=2) == dfn.VERDICT_FUTILITY
    assert dfn.resolve_action(dfn.VERDICT_FUTILITY, 7, 2) == 2


def test_a_race_that_separated_on_the_policys_own_action_is_KEPT_not_an_overrule():
    assert dfn.verdict(True, 2, 2) == dfn.VERDICT_KEPT
    assert dfn.resolve_action(dfn.VERDICT_KEPT, 2, 2) == 2


def test_only_a_separated_race_on_a_different_action_overrules():
    assert dfn.verdict(True, 7, 2) == dfn.VERDICT_OVERRULED
    assert dfn.resolve_action(dfn.VERDICT_OVERRULED, 7, 2) == 7


def _result(action, policy_action, resolved, fallback=None) -> DecisionResult:
    w = RealizedWidths(planned={}, n_our_actions=5)
    w.racing_rounds = 6
    w.racing_resolved = bool(resolved)
    return DecisionResult(action, fallback, w, scores={action: 0.9, policy_action: 0.4},
                          policy_action=policy_action)


def test_apply_defensive_keeps_the_policy_action_on_a_race_that_did_not_separate():
    eng = _engine(SearchConfig(root_strategy="defensive"), _FakeModel(3))
    out = eng._apply_defensive(_result(7, 2, resolved=False), 2)
    assert out.action == 2 and out.changed is False
    assert out.widths.defensive_verdict == dfn.VERDICT_FUTILITY
    # ...and it is a search VERDICT, not a fallback: it stays inside `n_searched`, which is what
    # makes `change_rate` over `searched` read as the overrule rate among RACED decisions.
    assert out.fallback is None
    assert out.diagnostics["defensive"]["separated"] is False


def test_apply_defensive_overrules_only_on_separation():
    eng = _engine(SearchConfig(root_strategy="defensive"), _FakeModel(3))
    out = eng._apply_defensive(_result(7, 2, resolved=True), 2)
    assert out.action == 7 and out.changed is True
    assert out.widths.defensive_verdict == dfn.VERDICT_OVERRULED


def test_apply_defensive_passes_a_counted_search_FAILURE_through_untouched():
    """A driver problem must not be re-labelled as a design choice."""
    eng = _engine(SearchConfig(root_strategy="defensive"), _FakeModel(3))
    res = _result(2, 2, resolved=False, fallback="root_failed")
    out = eng._apply_defensive(res, 2)
    assert out is res and out.fallback == "root_failed"
    assert out.widths.defensive_verdict == ""


# ---------------------------------------------------------------------------
# the ENGINE gate — the counted fallbacks and the banked clock
# ---------------------------------------------------------------------------


def _gate(cfg: SearchConfig, n_legal, wp):
    eng = _engine(cfg, _FakeModel(3))
    w = RealizedWidths(planned={}, n_our_actions=n_legal)
    return eng._gate(w, n_legal, wp), w


@pytest.mark.parametrize("strategy", ["grid", "racing"])
def test_the_gate_does_nothing_at_all_off_the_defensive_strategy(strategy):
    reason, w = _gate(SearchConfig(root_strategy=strategy, budget_s=1.0), 6, 0.99)
    assert reason is None
    assert w.defensive_verdict == "" and w.defensive_banked_s == 0.0
    assert w.defensive_root_win_prob == -1.0


def test_a_gated_decision_banks_the_WHOLE_budget_and_is_a_counted_fallback():
    reason, w = _gate(SearchConfig(root_strategy="defensive", budget_s=1.0), 6, 0.93)
    assert reason == "defensive_forced" and reason in FALLBACK_REASONS
    assert w.defensive_verdict == dfn.VERDICT_FORCED
    assert w.defensive_gate_reason == dfn.GATE_WP_EXTREME
    # The gate runs BEFORE the clock starts, so the whole budget is genuinely unspent.
    assert w.defensive_banked_s == 1.0
    assert w.defensive_root_win_prob == pytest.approx(0.93)


def test_a_contested_decision_is_let_through_with_the_win_prob_recorded():
    reason, w = _gate(SearchConfig(root_strategy="defensive", budget_s=1.0), 6, 0.52)
    assert reason is None
    assert w.defensive_verdict == "" and w.defensive_root_win_prob == pytest.approx(0.52)


def test_a_checkpoint_with_no_win_prob_head_is_refused_loudly_and_counted():
    reason, w = _gate(SearchConfig(root_strategy="defensive", budget_s=2.0), 6, None)
    assert reason == "defensive_no_win_prob" and reason in FALLBACK_REASONS
    assert w.defensive_no_win_prob is True
    assert w.defensive_root_win_prob == -1.0     # NEGATIVE, never 0.5
    assert w.defensive_banked_s == 2.0


def test_the_absent_win_prob_sentinel_cannot_be_read_as_a_contested_position():
    """-1.0 rather than None keeps the row dtype stable; NEGATIVE rather than 0.5 keeps a missing
    measurement from being pooled with the most contested decisions in the cell."""
    w = RealizedWidths(planned={})
    assert w.defensive_root_win_prob < 0.0
    assert abs(w.defensive_root_win_prob - 0.5) > dfn.DEFAULT_WP_MARGIN


# ---------------------------------------------------------------------------
# the FOLD
# ---------------------------------------------------------------------------


def _dec(**w) -> dict:
    return {"widths": {**w}}


def test_the_fold_is_all_zeros_for_a_grid_or_racing_cell():
    out = dfn.fold_defensive([_dec(), _dec(racing_rounds=4), {"note": "policy_default"}])
    assert out["n_defensive"] == 0 and out["defensive_banked_s"] == 0.0
    assert set(out) == set(dfn.fold_defensive(()))


def test_the_fold_counts_every_branch_apart():
    rows = [
        _dec(defensive_verdict=dfn.VERDICT_FORCED, defensive_gate_reason=dfn.GATE_WP_EXTREME,
             defensive_banked_s=1.0),
        _dec(defensive_verdict=dfn.VERDICT_FORCED, defensive_gate_reason=dfn.GATE_N_LEGAL,
             defensive_banked_s=1.0),
        _dec(defensive_verdict=dfn.VERDICT_FUTILITY, defensive_banked_s=0.3),
        _dec(defensive_verdict=dfn.VERDICT_KEPT, defensive_banked_s=0.1),
        _dec(defensive_verdict=dfn.VERDICT_OVERRULED, defensive_banked_s=0.05),
        _dec(defensive_no_win_prob=True, defensive_banked_s=1.0),
    ]
    out = dfn.fold_defensive(rows)
    assert out["n_defensive"] == 6
    assert out["n_defensive_forced"] == 2
    assert out["n_defensive_forced_wp"] == 1 and out["n_defensive_forced_n_legal"] == 1
    assert out["n_defensive_raced"] == 3
    assert out["n_defensive_futility"] == 1
    assert out["n_defensive_separated"] == 2
    assert out["n_defensive_kept"] == 1 and out["n_defensive_overruled"] == 1
    assert out["n_defensive_no_win_prob"] == 1
    assert out["defensive_banked_s"] == pytest.approx(3.45)


def test_forced_and_futility_are_never_summed_into_one_counter():
    """They are opposite findings — 'the position was decided' vs 'the actions were
    indistinguishable' — and a cell where one is large and the other zero is a different
    instrument from a balanced one."""
    forced = dfn.fold_defensive([_dec(defensive_verdict=dfn.VERDICT_FORCED,
                                      defensive_gate_reason=dfn.GATE_WP_EXTREME)])
    futile = dfn.fold_defensive([_dec(defensive_verdict=dfn.VERDICT_FUTILITY)])
    assert forced["n_defensive_forced"] == 1 and forced["n_defensive_futility"] == 0
    assert futile["n_defensive_futility"] == 1 and futile["n_defensive_forced"] == 0
    assert forced["n_defensive_raced"] == 0 and futile["n_defensive_raced"] == 1


def test_the_fold_splits_a_futility_the_CLOCK_ended_from_a_genuine_non_separation():
    """The first cell could not tell these apart — all 3,301 of its futility stops were also
    ``deadline_truncated``, an exact identity — so "the race ran out of clock" and "the actions
    are genuinely indistinguishable" arrived as one number. A time manager that merely bought more
    rounds would then look identical to one that learned something."""
    rows = [
        _dec(defensive_verdict=dfn.VERDICT_FUTILITY, deadline_truncated=True),
        _dec(defensive_verdict=dfn.VERDICT_FUTILITY, deadline_truncated=True),
        _dec(defensive_verdict=dfn.VERDICT_FUTILITY, deadline_truncated=False),
        # A truncated race that STILL separated is not futility and must not be counted as one.
        _dec(defensive_verdict=dfn.VERDICT_KEPT, deadline_truncated=True),
    ]
    out = dfn.fold_defensive(rows)
    assert out["n_defensive_futility"] == 3
    assert out["n_defensive_futility_deadline"] == 2
    b = dfn.defensive_block(out)
    assert b["futility_deadline"] == 2 and b["futility_genuine"] == 1
    assert b["futility_deadline_frac"] == pytest.approx(2 / 3, abs=1e-4)


def test_the_futility_split_folds_to_zero_on_a_cell_that_never_raced():
    out = dfn.fold_defensive([_dec(defensive_verdict=dfn.VERDICT_FORCED,
                                   defensive_gate_reason=dfn.GATE_WP_EXTREME)])
    assert out["n_defensive_futility_deadline"] == 0
    b = dfn.defensive_block(out)
    assert b["futility_genuine"] == 0 and b["futility_deadline_frac"] is None


def test_the_fold_splits_a_confirm_that_acted_from_one_that_declined():
    out = dfn.fold_defensive([
        _dec(defensive_verdict=dfn.VERDICT_OVERRULED, defensive_confirm_stage="played"),
        _dec(defensive_verdict=dfn.VERDICT_KEPT, defensive_confirm_stage="inconclusive"),
        _dec(defensive_verdict=dfn.VERDICT_KEPT, defensive_confirm_stage="error"),
    ])
    assert out["n_defensive_confirmed"] == 1
    assert out["n_defensive_confirm_declined"] == 2


def test_the_block_is_None_on_a_cell_that_never_ran_defensively():
    assert dfn.defensive_block(dfn.fold_defensive(())) is None


def test_the_block_quotes_every_rate_against_the_decisions_the_strategy_handled():
    rows = ([_dec(defensive_verdict=dfn.VERDICT_FORCED,
                  defensive_gate_reason=dfn.GATE_WP_EXTREME, defensive_banked_s=1.0)] * 8
            + [_dec(defensive_verdict=dfn.VERDICT_FUTILITY, defensive_banked_s=0.5)]
            + [_dec(defensive_verdict=dfn.VERDICT_OVERRULED, defensive_banked_s=0.0)])
    b = dfn.defensive_block(dfn.fold_defensive(rows))
    assert b["decisions"] == 10
    assert b["forced_rate"] == 0.8
    assert b["race_rate"] == 0.2
    assert b["overrule_rate"] == 0.1              # over ALL decisions
    assert b["overrule_rate_raced"] == 0.5        # over the raced ones
    assert b["separation_rate"] == 0.5 and b["futility_rate"] == 0.5
    assert b["banked_s"] == pytest.approx(8.5)
    assert b["banked_s_per_decision"] == pytest.approx(0.85)


# ---------------------------------------------------------------------------
# OFF is byte-identical
# ---------------------------------------------------------------------------


def test_off_is_byte_identical_to_racing_and_grid():
    """The defensive config exists on every SearchConfig and must change NOTHING unless selected.

    Same discipline `--root-strategy racing` shipped under: an experiment that also perturbs its
    own control is not one.
    """
    base = SearchConfig(arm="honest", budget_s=1.0, score="auto")
    with_cfg = replace(base, defensive=dfn.DefensiveConfig(wp_margin=0.4, leaf="value",
                                                           confirm_rollouts=9))
    assert with_cfg.effective_score() == base.effective_score() == "auto"
    assert with_cfg.defensive_cfg() is None and base.defensive_cfg() is None
    assert with_cfg.resolved_caps() == base.resolved_caps()
    assert with_cfg.effective_max_depth() == base.effective_max_depth()


def test_a_defensive_config_inherits_the_racing_parameters_rather_than_copying_them():
    """One elimination threshold, not two that can drift. The floor that `seq` enforces is the
    binding parameter (probe I: raising it 3 -> 5 moved the never-separate rate 16.7% -> 52.2% and
    the agreement ceiling 0.933 -> 1.000), so a second copy of it would be a second experiment."""
    cfg = SearchConfig(root_strategy="defensive", racing=RacingConfig(min_samples=3))
    assert cfg.racing.rule == "seq"
    assert cfg.racing.effective_min_samples() == 5
    assert not hasattr(cfg.defensive, "min_samples")


# ---------------------------------------------------------------------------
# reproducibility
# ---------------------------------------------------------------------------


def test_the_gate_and_the_verdict_are_pure_functions_of_their_inputs():
    """No RNG, no clock, no global state — a decision replayed on the same numbers replays."""
    for _ in range(3):
        assert dfn.gate(6, 0.5123, dfn.DefensiveConfig()) == dfn.GATE_SEARCH
        assert dfn.gate(6, 0.8, dfn.DefensiveConfig()) == dfn.GATE_WP_EXTREME
        assert dfn.verdict(True, 7, 2) == dfn.VERDICT_OVERRULED


def test_a_fixed_seed_engine_gates_identically_across_two_constructions():
    a = _gate(SearchConfig(root_strategy="defensive", budget_s=1.0, seed=11), 6, 0.7)
    b = _gate(SearchConfig(root_strategy="defensive", budget_s=1.0, seed=11), 6, 0.7)
    assert a[0] == b[0] and a[1].as_dict() == b[1].as_dict()
