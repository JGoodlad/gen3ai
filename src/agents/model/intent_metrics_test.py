"""Gates for the opponent-intent METRICS' scoring rule and class table (`gen3_opp_class_v1`).

Each test pins a property that would be silently wrong if the metric were computed carelessly —
not a shape check. The motivating problem: the intent head's accuracy is read off an opponent pool
spanning a RANDOM bot, several heuristics and frozen selves, where "predict their move" is a
different problem per opponent. Accuracy cannot express that; a proper scoring rule can — the
INFORMATION GAIN over the batch's empirical marginal (`flat_info_gain_nats`, computed by the
static `info_gain_nats_static` the intent fold traces).
"""
import math

import pytest
import torch

from agents.model.opp_intent import OPP_CLASS_NAMES
from agents.training.instrumented_ppo.intent_fold import info_gain_nats_static


def info_gain_nats(logits: torch.Tensor, target: torch.Tensor) -> float:
    """The gain over EVERY row given (the fold's mask all-True), as a float."""
    mask = torch.ones(target.shape[0], dtype=torch.bool)
    return float(info_gain_nats_static(logits, target, mask, torch.float32))


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


def test_class_names_match_the_declared_constants():
    """The table is duplicated to keep `model/` from importing `training/` — pin them together."""
    from agents.training import opponent_classes as oc
    assert OPP_CLASS_NAMES == {oc.OPP_CLASS_BOT: "bot", oc.OPP_CLASS_POOL: "pool",
                               oc.OPP_CLASS_STABLE: "stable", oc.OPP_CLASS_EXPLOITER: "exploiter"}
