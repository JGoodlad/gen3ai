"""Gates for the opponent-intent METRIC fixes (`gen3_opp_class_v1`).

Each test pins a property that would be silently wrong if the metric were computed carelessly —
not a shape check. The motivating problem: `α`'s accuracy is read off an opponent pool spanning a
RANDOM bot, several heuristics and frozen selves, where "predict their move" is a different problem
per opponent. Accuracy cannot express that; a proper scoring rule can.
"""
import math

import pytest
import torch

from agents.model.opp_intent import (
    INTENT_IGNORE,
    OPP_CLASS_NAMES,
    info_gain_nats,
    intent_losses,
)


# ------------------------------------------------------------------ the scoring rule itself

def test_a_perfect_predictor_gains_the_full_entropy():
    """Gain == H(marginal) when the model is certain and right, since CE == 0."""
    tgt = torch.tensor([0, 1, 0, 1])
    logits = torch.full((4, 2), -30.0)
    logits[torch.arange(4), tgt] = 30.0
    assert info_gain_nats(logits, tgt) == pytest.approx(math.log(2), abs=1e-3)


def test_predicting_the_base_rate_gains_nothing():
    """THE load-bearing case. A model that has learned only the marginal must score ~0, not 'good'.

    This is what makes the number readable against an unpredictable opponent: uniform truth plus a
    uniform prediction is a perfect score under accuracy's rival framings and 0 nats here.
    """
    tgt = torch.tensor([0, 1, 0, 1, 0, 1])
    logits = torch.zeros(6, 2)                       # uniform => CE == H(marginal)
    assert info_gain_nats(logits, tgt) == pytest.approx(0.0, abs=1e-6)


def test_worse_than_the_base_rate_goes_NEGATIVE():
    """Negative is a real reading, not a clamp bug — it is the state alpha's move axis may be in."""
    tgt = torch.tensor([0, 0, 0, 1])
    logits = torch.zeros(4, 2)
    logits[:, 1] = 3.0                               # confidently backs the RARE class
    assert info_gain_nats(logits, tgt) < 0.0


def test_an_unpredictable_opponent_scores_about_zero_however_confident_the_truth_looks():
    """The random-opponent case, stated as a test.

    Targets are drawn uniformly and the model predicts uniformly — the Bayes-optimal answer. Under
    accuracy this scores 1/n and reads as failure; the gain must read ~0 = 'nothing was learnable'.
    """
    g = torch.Generator().manual_seed(0)
    tgt = torch.randint(0, 4, (4096,), generator=g)
    logits = torch.zeros(4096, 4)
    assert abs(info_gain_nats(logits, tgt)) < 0.02


def test_gain_is_invariant_to_a_constant_logit_shift():
    """Softmax-invariant reparameterisation must not move a scoring rule."""
    tgt = torch.tensor([0, 1, 2, 1])
    logits = torch.randn(4, 3, generator=torch.Generator().manual_seed(1))
    assert info_gain_nats(logits, tgt) == pytest.approx(
        info_gain_nats(logits + 7.5, tgt), abs=1e-5)


# ------------------------------------------------------------------ stratification

def _alpha_case(n=64, k=3):
    torch.manual_seed(0)
    logits = torch.randn(n, k + 1)
    target = torch.randint(0, k + 1, (n,))
    return logits, target


def test_stratified_metrics_appear_per_present_class_only():
    logits, target = _alpha_case()
    opp_class = torch.zeros(logits.shape[0], dtype=torch.long)
    opp_class[32:] = 1                                   # only bot + pool present
    _, m = intent_losses(logits, target, None, None, opp_class=opp_class)
    assert "opp_intent/alpha_info_gain_nats_bot" in m
    assert "opp_intent/alpha_info_gain_nats_pool" in m
    assert "opp_intent/alpha_info_gain_nats_stable" not in m
    assert "opp_intent/alpha_info_gain_nats_exploiter" not in m


def test_stratified_counts_sum_to_the_pooled_supervised_count():
    """A row must land in exactly one bucket — no double-counting, no silent drops."""
    logits, target = _alpha_case(n=96)
    opp_class = torch.randint(0, 4, (96,))
    _, m = intent_losses(logits, target, None, None, opp_class=opp_class)
    per = sum(v for k, v in m.items() if k.startswith("opp_intent/alpha_n_supervised_"))
    assert per == pytest.approx(m["opp_intent/alpha_n_supervised"])


def test_stratification_actually_separates_a_planted_difference():
    """The gate that matters: an EASY class and a HARD class must not report the same number.

    A stratification that silently pooled (wrong index, wrong mask) would still emit both keys with
    both values equal — which is exactly the failure a presence check cannot see.
    """
    n = 200
    target = torch.cat([torch.zeros(n, dtype=torch.long),          # easy: always class 0
                        torch.randint(0, 4, (n,))])                # hard: uniform
    logits = torch.zeros(2 * n, 4)
    logits[:n, 0] = 8.0                                            # predicts the easy half exactly
    opp_class = torch.cat([torch.zeros(n, dtype=torch.long),
                           torch.ones(n, dtype=torch.long)])
    _, m = intent_losses(logits, target, None, None, opp_class=opp_class)
    easy = m["opp_intent/alpha_acc_bot"]
    hard = m["opp_intent/alpha_acc_pool"]
    assert easy > 0.95 and hard < 0.5, (easy, hard)
    # And the pooled number sits between them — i.e. it is the average that hides both.
    assert hard < m["opp_intent/alpha_acc"] < easy


def test_ignored_rows_are_excluded_from_every_stratum():
    logits, target = _alpha_case(n=40)
    target[:20] = INTENT_IGNORE
    opp_class = torch.zeros(40, dtype=torch.long)
    _, m = intent_losses(logits, target, None, None, opp_class=opp_class)
    assert m["opp_intent/alpha_n_supervised_bot"] == pytest.approx(20.0)


def test_opp_class_is_optional_and_changes_nothing_when_absent():
    """Off must be exactly the old behaviour — no stratified keys, identical pooled values."""
    logits, target = _alpha_case()
    _, without = intent_losses(logits, target, None, None)
    _, with_cls = intent_losses(logits, target, None, None,
                                opp_class=torch.zeros(logits.shape[0], dtype=torch.long))
    # Check for the CLASS SUFFIXES specifically. (This assertion used to test for "__" and became
    # vacuous the moment the separator changed to "_" — a test that cannot fail is not a test.)
    assert not any(k.endswith(tuple("_" + n for n in OPP_CLASS_NAMES.values())) for k in without)
    for k, v in without.items():
        assert with_cls[k] == pytest.approx(v), k


def test_the_loss_is_untouched_by_the_metric_change():
    """Metrics may not perturb training. Same inputs, same scalar, with and without stratification."""
    logits, target = _alpha_case()
    loss_a, _ = intent_losses(logits, target, None, None)
    loss_b, _ = intent_losses(logits, target, None, None,
                              opp_class=torch.randint(0, 4, (logits.shape[0],)))
    assert torch.equal(loss_a, loss_b)


def test_class_names_match_the_wrapper_constants():
    """The table is duplicated to keep `model/` from importing `training/` — pin them together."""
    from agents.training.wrappers import MaskableAgentWrapper as W
    assert OPP_CLASS_NAMES == {W.OPP_CLASS_BOT: "bot", W.OPP_CLASS_POOL: "pool",
                               W.OPP_CLASS_STABLE: "stable", W.OPP_CLASS_EXPLOITER: "exploiter"}
