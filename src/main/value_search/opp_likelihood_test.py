"""The opponent-choice likelihood's numerics (``opp_likelihood.log_probs``): a masked softmax in float64, a forced
decision worth exactly 0, the familiarity control never above the chosen action's log-probability's ceiling."""
from __future__ import annotations

import numpy as np
import pytest

from main.value_search import opp_likelihood as OL


def test_a_forced_decision_contributes_exactly_zero():
    logits = np.array([[5.0, -3.0] + [0.0] * 10], dtype=np.float32)
    mask = np.zeros((1, 11), dtype=np.uint8)
    mask[0, 1] = 1
    lp, cf = OL.log_probs(logits, mask, [1])
    assert lp[0] == 0.0 and cf[0] == 0.0


def test_the_masked_softmax_ignores_illegal_logits():
    logits = np.zeros((1, 12), dtype=np.float32)
    logits[0, 0] = 100.0                                 # illegal: must not take any mass
    mask = np.zeros((1, 11), dtype=np.uint8)
    mask[0, [3, 4]] = 1
    lp, cf = OL.log_probs(logits, mask, [3])
    assert lp[0] == pytest.approx(np.log(0.5), abs=1e-12) and cf[0] == pytest.approx(np.log(0.5), abs=1e-12)


def test_the_chosen_and_top_log_probabilities():
    logits = np.array([[0.0, 1.0, 2.0] + [0.0] * 9], dtype=np.float32)
    mask = np.zeros((1, 11), dtype=np.uint8)
    mask[0, :3] = 1
    lp, cf = OL.log_probs(logits, mask, [1])
    z = np.log(np.exp(0.0) + np.exp(1.0) + np.exp(2.0))
    assert lp[0] == pytest.approx(1.0 - z, abs=1e-12) and cf[0] == pytest.approx(2.0 - z, abs=1e-12)
    assert lp[0] <= cf[0] <= 0.0


def test_a_masked_out_choice_is_a_defect():
    mask = np.zeros((1, 11), dtype=np.uint8)
    mask[0, 2] = 1
    with pytest.raises(OL.OppLikelihoodError):
        OL.log_probs(np.zeros((1, 12), dtype=np.float32), mask, [5])
