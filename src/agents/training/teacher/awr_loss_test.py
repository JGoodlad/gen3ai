"""Pure unit tests for the search-teacher AWR loss (`InstrumentedMaskablePPO._searchteacher_loss`)."""

import numpy as np
import torch as th

from agents.training.instrumented_ppo import InstrumentedMaskablePPO

_LOSS = InstrumentedMaskablePPO._searchteacher_loss


def test_none_guard_on_empty():
    assert _LOSS(None, None, None, None) is None
    empty = th.zeros((0, 11))
    assert _LOSS(empty, th.ones((0, 11)), th.zeros((0,), dtype=th.int64), th.zeros((0,))) is None


def test_agree_rate_when_policy_already_predicts_astar():
    # logits put all mass on the target action → agree_rate 1, CE ~0.
    logits = th.full((4, 11), -10.0)
    a = th.tensor([6, 2, 8, 0])
    logits[th.arange(4), a] = 10.0
    out = _LOSS(logits, th.ones((4, 11)), a, th.ones(4) * 0.5)
    assert out is not None
    loss, m = out
    assert m["agree_rate"] == 1.0
    assert m["ce"] < 0.01 and float(loss) < 0.01


def test_disagree_gives_positive_loss():
    logits = th.zeros((3, 11))                 # uniform
    a = th.tensor([6, 2, 8])
    loss, m = _LOSS(logits, th.ones((3, 11)), a, th.ones(3) * 0.5)
    assert float(loss) > 0.0 and m["ce"] > 0.0


def test_advantage_weighting_scales_high_margin_more():
    # Same wrong prediction, but one sample has a bigger confirmed advantage → bigger AWR weight.
    logits = th.zeros((2, 11))
    a = th.tensor([6, 6])
    _, m_lo = _LOSS(logits.clone(), th.ones((2, 11)), a, th.tensor([0.1, 0.1]), beta_awr=1.0)
    _, m_hi = _LOSS(logits.clone(), th.ones((2, 11)), a, th.tensor([0.1, 0.9]), beta_awr=1.0)
    assert m_hi["mean_w"] > m_lo["mean_w"]               # higher adv → higher weight
    # w = exp(adv/beta), clamped
    assert abs(m_lo["mean_w"] - float(np.exp(0.1))) < 1e-4


def test_illegal_actions_get_no_mass():
    # An illegal action's logit is huge, but the mask zeros it → argmax must ignore it.
    logits = th.zeros((1, 11)); logits[0, 3] = 100.0     # action 3 illegal but high
    mask = th.ones((1, 11)); mask[0, 3] = 0
    a = th.tensor([6])
    _, m = _LOSS(logits, mask, a, th.ones(1) * 0.5)
    assert m["agree_rate"] == 0.0                         # argmax is among LEGAL, not the masked 3


def test_weight_clip():
    logits = th.zeros((1, 11)); a = th.tensor([6])
    _, m = _LOSS(logits, th.ones((1, 11)), a, th.tensor([100.0]), beta_awr=1.0, w_clip=20.0)
    assert m["mean_w"] == 20.0                            # exp(100) clamped to w_clip


def test_gradient_flows_to_logits():
    logits = th.zeros((2, 11), requires_grad=True)
    a = th.tensor([6, 2])
    loss, _ = _LOSS(logits, th.ones((2, 11)), a, th.ones(2) * 0.5)
    loss.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
