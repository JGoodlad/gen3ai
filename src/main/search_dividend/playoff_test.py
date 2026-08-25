"""The TOP-2 PLAYOFF's decision rule, tested where it can be tested — above the sim.

Every assertion here is about ARITHMETIC or about WHICH rollouts get requested, which is the half
that decides an action and the half a battery run cannot inspect after the fact. The rollout itself
is injected (``rollout_fn``), so these run in milliseconds and never spawn a bridge child; the live
path is exercised by the cell.

Four properties are pinned deliberately, because each one is a way this instrument could lie:

* the SCREEN-DECISIVE short circuit spends no rollouts, and only when the screen agrees with the
  policy — an override candidate always goes to a playoff however wide the screen's margin;
* an INCONCLUSIVE playoff returns the POLICY's action, never the screen's top1 (returning the
  screen's pick would smuggle the convicted estimator back in through the tie-break);
* the pairing uses ONE seed per draw across both candidates — with the seeds free, the paired
  difference would also contain "the two lines drew different dice", which is exactly the shared
  noise the pairing exists to cancel;
* a CAPPED rollout scores 0.5, not the winner the sim's forfeit ordering happened to name.
"""

from __future__ import annotations

import math
from typing import List

import pytest

from main.search_dividend import playoff as P
from main.search_dividend.budget import FALLBACK_REASONS, Deadline


class _Clock:
    """A hand-cranked monotonic clock, so a deadline test never sleeps."""

    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def _runner(rollout_fn, **cfg_kw) -> P.PlayoffRunner:
    cfg = P.PlayoffConfig(**cfg_kw)
    return P.PlayoffRunner(model=None, mappings=None, battle_format="gen3ou", cfg=cfg,
                           rollout_fn=rollout_fn)


def _tokens(*idxs) -> dict:
    return {int(i): f"move slot{i}" for i in idxs}


# ---------------------------------------------------------------------------
# the pure rule
# ---------------------------------------------------------------------------


def test_top_two_ranks_by_score_and_breaks_ties_on_index():
    assert P.top_two({3: 0.4, 7: 0.9, 1: 0.6}) == (7, 1, pytest.approx(0.3))
    # A deterministic pair matters: two processes handed the same screen must nominate the same
    # two candidates, or a "rerun" is not one.
    assert P.top_two({5: 0.5, 2: 0.5, 9: 0.1})[:2] == (2, 5)


def test_top_two_needs_two_scored_actions():
    assert P.top_two({}) is None
    assert P.top_two({4: 1.0}) is None


def test_screen_is_decisive_only_when_it_agrees_with_the_policy():
    # agrees + wide  -> settled, no rollout worth spending
    assert P.screen_is_decisive(4, 0.10, policy_action=4, margin_threshold=0.023)
    # agrees + narrow -> exactly the blurry comparison the playoff exists for
    assert not P.screen_is_decisive(4, 0.005, policy_action=4, margin_threshold=0.023)
    # DISAGREES -> a playoff runs however wide the screen is. A wide margin on a biased estimator
    # is confidence, not evidence, and this is the override case the week's harm came from.
    assert not P.screen_is_decisive(4, 0.99, policy_action=9, margin_threshold=0.023)


def test_paired_stats_floors_the_se_so_a_degenerate_sample_cannot_certify():
    mean, se, n = P.paired_stats([1.0, 1.0, 1.0, 1.0])
    assert (mean, n) == (1.0, 4)
    # sd == 0 over four identical draws; the floor keeps it from reading as infinite confidence.
    assert se == pytest.approx(P.SE_FLOOR_UNIT / 4)
    assert P.paired_stats([])[1] == math.inf
    assert P.paired_stats([0.5])[1] == math.inf


def test_is_conclusive_needs_both_the_sample_size_and_the_margin():
    # two identical draws: |mean| is huge relative to a floored SE, but n < MIN_PAIRS refuses.
    m, se, n = P.paired_stats([1.0, 1.0])
    assert not P.is_conclusive(m, se, n)
    # enough pairs, but the difference sits inside its own noise
    m, se, n = P.paired_stats([1.0, -1.0, 1.0, -1.0, 0.0, 0.0])
    assert not P.is_conclusive(m, se, n)
    # enough pairs and a difference that clears 2 SE
    m, se, n = P.paired_stats([1.0, 1.0, 1.0, 0.5, 1.0, 1.0])
    assert P.is_conclusive(m, se, n)


def test_decide_returns_the_policy_action_on_an_inconclusive_sweep():
    m, se, n = P.paired_stats([1.0, -1.0, 1.0, -1.0])
    action, stage = P.decide(a1=3, a2=8, mean=m, se=se, n=n, policy_action=5)
    assert stage == P.STAGE_INCONCLUSIVE
    # NOT 3. Falling back to the screen's own top1 would re-admit the estimator the rollouts were
    # called in to replace — the single most important line in this module.
    assert action == 5


def test_decide_plays_the_rollout_winner_when_it_clears_the_bar():
    m, se, n = P.paired_stats([1.0, 1.0, 1.0, 1.0, 0.5, 1.0])
    assert P.decide(a1=3, a2=8, mean=m, se=se, n=n, policy_action=5) == (3, P.STAGE_PLAYED)
    m, se, n = P.paired_stats([-1.0, -1.0, -1.0, -1.0, -0.5, -1.0])
    assert P.decide(a1=3, a2=8, mean=m, se=se, n=n, policy_action=5) == (8, P.STAGE_PLAYED)


# ---------------------------------------------------------------------------
# the capped-rollout rule (gen3_cf_draw_at_cap_v1, f8eec73)
# ---------------------------------------------------------------------------


def test_a_capped_rollout_scores_a_half_whatever_the_winner_says():
    # At the 250-turn stall cap BOTH sides forfeit and p1's FORCELOSE is processed first, so
    # `outcome` there is a fact about seat order. Scoring it as a loss is a systematic bias.
    assert P.rollout_score({"outcome": "loss", "capped": True}) == 0.5
    assert P.rollout_score({"outcome": "win", "capped": True}) == 0.5
    assert P.rollout_score({"outcome": "win", "capped": False}) == 1.0
    assert P.rollout_score({"outcome": "loss", "capped": False}) == 0.0
    assert P.rollout_score({"outcome": "tie", "capped": False}) == 0.5
    # a transport pathology is not a confident loss
    assert P.rollout_score({"outcome": "unfinished", "capped": False}) == 0.5


# ---------------------------------------------------------------------------
# the runner
# ---------------------------------------------------------------------------


def test_screen_decisive_short_circuit_spends_no_rollout():
    calls: List[dict] = []

    def never(**kw):
        calls.append(kw)
        raise AssertionError("a settled screen must not spend a rollout")

    r = _runner(never)
    out = r.adjudicate(scores={4: 0.9, 2: 0.1}, policy_action=4, our_tokens=_tokens(2, 4),
                       record=None, turn=7, deadline=Deadline(10.0), rng=__import__("random").Random(0))
    assert out.stage == P.STAGE_SCREEN_DECISIVE
    assert out.action == 4
    assert out.r == 0 and not calls
    assert out.fallback is None                    # a DECISION, not a fallback


def test_the_pair_shares_one_sim_seed_and_one_torch_seed_per_draw():
    """CRN across the two candidates — the property the whole pairing rests on."""
    seen: List[tuple] = []

    def rollout(*, record, turn, choice, sim_seed, torch_seed):
        seen.append((choice, sim_seed, torch_seed))
        return {"outcome": "win" if choice.endswith("3") else "loss", "capped": False}

    r = _runner(rollout, rollouts=5, rollout_cost_s=0.0)
    out = r.adjudicate(scores={3: 0.50, 8: 0.49}, policy_action=8, our_tokens=_tokens(3, 8),
                       record=None, turn=11, deadline=Deadline(10.0),
                       rng=__import__("random").Random(7))
    assert out.r == 5 and len(seen) == 10
    for i in range(5):
        a, b = seen[2 * i], seen[2 * i + 1]
        assert a[0] != b[0], "the pair must roll the two DIFFERENT candidates"
        assert a[1] == b[1], "same post-divergence dice across the pair"
        assert a[2] == b[2], "same policy-sampling seed across the pair"
    # ... and DIFFERENT across draws, or R>1 would be one line played R times.
    assert len({s for (_c, s, _t) in seen}) == 5


def test_a_clean_sweep_overrides_the_policy_and_a_split_one_does_not():
    def clean(*, record, turn, choice, sim_seed, torch_seed):
        return {"outcome": "win" if choice.endswith("3") else "loss", "capped": False}

    def split(*, record, turn, choice, sim_seed, torch_seed):
        # every pair ties -> mean 0, and a mean of zero can never clear 2 SE
        return {"outcome": "tie", "capped": False}

    rng = __import__("random").Random(1)
    won = _runner(clean, rollouts=6, rollout_cost_s=0.0).adjudicate(
        scores={3: 0.50, 8: 0.49}, policy_action=8, our_tokens=_tokens(3, 8),
        record=None, turn=11, deadline=Deadline(10.0), rng=rng)
    assert (won.action, won.stage, won.fallback) == (3, P.STAGE_PLAYED, None)

    tied = _runner(split, rollouts=6, rollout_cost_s=0.0).adjudicate(
        scores={3: 0.50, 8: 0.49}, policy_action=8, our_tokens=_tokens(3, 8),
        record=None, turn=11, deadline=Deadline(10.0), rng=rng)
    assert tied.action == 8                       # the POLICY's action
    assert tied.stage == P.STAGE_INCONCLUSIVE
    assert tied.fallback == P.FALLBACK_INCONCLUSIVE


def test_the_deadline_caps_the_realized_r_and_the_row_says_so():
    """R is a CAP; what the clock bought is the measurement. A cell that silently ran at R=3 under
    an ``R=12`` label would make every reading of its inconclusive rate wrong."""
    clock = _Clock()

    def slow(*, record, turn, choice, sim_seed, torch_seed):
        clock.t += 1.0
        return {"outcome": "win" if choice.endswith("3") else "loss", "capped": False}

    r = _runner(slow, rollouts=12, rollout_cost_s=1.0)
    out = r.adjudicate(scores={3: 0.50, 8: 0.49}, policy_action=8, our_tokens=_tokens(3, 8),
                       record=None, turn=11, deadline=Deadline(9.0, clock=clock),
                       rng=__import__("random").Random(3))
    assert 0 < out.r < 12
    assert out.stage == P.STAGE_PLAYED           # 4 clean pairs still clear the bar


def test_a_pair_that_raises_is_dropped_WHOLE_and_counted():
    """Never a half-pair. Keeping one arm of a failed pair would leave the two candidates measured
    under different dice, which is the pairing silently switching itself off."""
    calls = {"n": 0}

    def flaky(*, record, turn, choice, sim_seed, torch_seed):
        calls["n"] += 1
        if calls["n"] == 3:                       # the FIRST arm of the second pair
            raise RuntimeError("bridge child died")
        return {"outcome": "win" if choice.endswith("3") else "loss", "capped": False}

    out = _runner(flaky, rollouts=6, rollout_cost_s=0.0).adjudicate(
        scores={3: 0.50, 8: 0.49}, policy_action=8, our_tokens=_tokens(3, 8),
        record=None, turn=11, deadline=Deadline(10.0), rng=__import__("random").Random(5))
    assert out.r == 5 and out.failed == 1
    # 6 attempts x 2 arms, MINUS the partner of the arm that raised: the surviving half is never
    # kept, so the failure costs a whole pair and leaves no orphan under mismatched dice.
    assert calls["n"] == 11
    assert "bridge child died" in (out.error or "")


def test_no_budget_and_missing_token_both_return_the_policy_action():
    r = _runner(lambda **kw: {"outcome": "win"}, rollouts=1, rollout_cost_s=0.0)
    only_one = r.adjudicate(scores={4: 0.9}, policy_action=4, our_tokens=_tokens(4),
                            record=None, turn=3, deadline=Deadline(10.0),
                            rng=__import__("random").Random(0))
    assert (only_one.action, only_one.stage) == (4, P.STAGE_NO_BUDGET)
    # The screen scored an action the live mapper produced no sim token for: decline rather than
    # invent a token to script the rollout with.
    no_tok = r.adjudicate(scores={3: 0.9, 8: 0.1}, policy_action=8, our_tokens=_tokens(8),
                          record=None, turn=3, deadline=Deadline(10.0),
                          rng=__import__("random").Random(0))
    assert (no_tok.action, no_tok.stage) == (8, P.STAGE_NO_BUDGET)


def test_every_playoff_fallback_is_a_declared_reason():
    """A fallback the report cannot name is a search that degraded invisibly — the one failure
    this whole package is built to make impossible."""
    for stage in (P.STAGE_INCONCLUSIVE, P.STAGE_NO_BUDGET, P.STAGE_ERROR):
        reason = P.PlayoffResult(0, stage).fallback
        assert reason in FALLBACK_REASONS, stage
    for stage in (P.STAGE_SCREEN_DECISIVE, P.STAGE_PLAYED):
        assert P.PlayoffResult(0, stage).fallback is None


# ---------------------------------------------------------------------------
# the fold
# ---------------------------------------------------------------------------


def test_fold_playoff_is_additive_and_pools_sums():
    rows = [
        {"playoff": {"stage": P.STAGE_SCREEN_DECISIVE, "r": 0}},
        {"playoff": {"stage": P.STAGE_PLAYED, "r": 12, "wall_s": 4.0, "capped": 1}},
        {"playoff": {"stage": P.STAGE_INCONCLUSIVE, "r": 8, "wall_s": 3.0, "failed": 2}},
        {"playoff": {"stage": P.STAGE_NO_BUDGET, "r": 0}},
        {"fallback": "no_search"},                 # a decision from ANOTHER arm
    ]
    out = P.fold_playoff(rows)
    assert out["n_screen_decisive"] == 1 and out["n_playoff"] == 1
    assert out["n_playoff_inconclusive"] == 1 and out["n_playoff_no_budget"] == 1
    assert out["n_playoff_ran"] == 2 and out["playoff_r_total"] == 20
    assert out["playoff_wall_s"] == pytest.approx(7.0)
    assert out["n_playoff_capped"] == 1 and out["n_playoff_failed"] == 2
    # A decision list with no playoff block at all folds to zeros — one schema, extended.
    assert P.fold_playoff([{"fallback": "no_search"}])["n_playoff"] == 0
