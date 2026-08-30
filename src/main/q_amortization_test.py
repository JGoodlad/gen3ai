"""The AMORTIZATION-RESIDUAL probe (`gen3_q_winprob_head_v1`, E5 step 5).

The real comparison needs a run at the current architecture with bridge-eval traces and a Q-head
checkpoint, none of which exists yet — so what is gated here is the part that CAN be: the
harness's arithmetic, and the init-state sanity it ships. A probe whose statistics are wrong is
worse than no probe, because its numbers get quoted.

The one that matters most is `spearman` returning **None** on a constant row. An untrained Q head
emits a constant row by construction, and reporting that as rho = 0.0 would put "the head has
learned nothing" and "the head has learned something uncorrelated with the sweep" in the same
bucket — which is precisely the distinction this probe exists to make.
"""
from __future__ import annotations

import pytest

from main.q_amortization import compare_rows, main, self_check, spearman


def test_the_shipped_init_state_sanity_passes():
    """The gated half of the deliverable: zero-init ⇒ P(win|s,a) = 0.5 everywhere ⇒ a total tie.
    Run as a function AND through the CLI, because `--self-check` is the documented entry point and
    an exit code nobody executes is not a check."""
    assert self_check() == 0
    assert main(["--self-check"]) == 0


def test_spearman_is_UNDEFINED_on_a_constant_row_not_zero():
    assert spearman([0.5] * 5, [1, 2, 3, 4, 5]) is None
    assert spearman([1, 2, 3, 4, 5], [0.5] * 5) is None
    assert spearman([1, 2], [2, 1]) is None                    # too few points to be meaningful
    assert spearman([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert spearman([1, 2, 3], [3, 2, 1]) == pytest.approx(-1.0)


def test_spearman_shares_ranks_across_TIES():
    """A head that ties two actions must not be silently ordered by array position — that would
    manufacture agreement (or disagreement) out of a tie-break the head never expressed."""
    # [1, 1, 2] ranks as [1.5, 1.5, 3]. Two rows that tie the SAME pair agree perfectly…
    assert spearman([1, 1, 2], [5, 5, 9]) == pytest.approx(1.0)
    # …and a row that ties a DIFFERENT pair does not. A positional tie-break would rank both rows
    # [1, 2, 3] and report perfect agreement between two rows that in fact express different
    # orderings — manufacturing agreement out of a tie-break neither head made.
    assert spearman([1, 1, 2], [9, 5, 5]) == pytest.approx(-0.5)


def test_compare_rows_scores_only_the_SHARED_actions():
    """The Q head scores all eleven slots (the extractor forward reads no mask); the sweep covers
    only the legal actions whose re-roll produced a scorable successor. Scoring the head on a slot
    the sweep could not evaluate would be comparing it against nothing."""
    q = {0: 0.1, 1: 0.9, 2: 0.5, 7: 0.99}
    truth = {0: 0.2, 1: 0.8, 2: 0.4}                    # 7 was terminal / illegal — no ground truth
    got = compare_rows(q, truth)
    assert got["n_actions"] == 3 and got["actions"] == [0, 1, 2]
    assert got["top1_agree"] == 1                       # action 7 must NOT win the head's argmax
    assert got["rho"] == pytest.approx(1.0)
    assert got["residual"] == pytest.approx((0.1 + 0.1 + 0.1) / 3, abs=1e-6)


def test_compare_rows_reports_a_disagreement_as_one():
    got = compare_rows({0: 0.9, 1: 0.1, 2: 0.5}, {0: 0.1, 1: 0.9, 2: 0.5})
    assert got["top1_agree"] == 0
    assert got["rho"] is not None and got["rho"] < 0


def test_too_few_shared_actions_is_a_NON_measurement():
    """One shared action cannot rank anything. It must come back as undefined rather than as a
    perfect (or zero) correlation — a decision with a single scorable alternative is a decision the
    probe has nothing to say about."""
    got = compare_rows({3: 0.4}, {3: 0.6})
    assert got["n_actions"] == 1
    assert got["rho"] is None and got["top1_agree"] is None and got["residual"] is None
