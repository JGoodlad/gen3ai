"""Pure unit tests for `cf_audit_twin` — the twin-head paired read and the shadow critic.

Moved here with the functions when they left `cf_audit.py` (2026-09-06). Two of the tests assert
`cf_audit.bias_map`'s COMPOSITION rather than this module's arithmetic — that the twin and shadow
blocks appear only when the heads were actually read — and they stay beside the `_twin_labels`
fixture they are written against, because splitting a fixture is how two copies of it start
drifting. `_dec` is imported from `cf_audit_test` for the same reason.
"""

from __future__ import annotations

import numpy as np
import pytest

from agents.training.cf_audit_test import _dec


# --------------------------------------------------------- TWIN HEADS + SHADOW (v99)
# gen3_cf_twin_heads_v1. The amended R1 primary is a WITHIN-RUN paired difference, so the tests
# here are about the two properties that make a paired read trustworthy: the SIGN convention (these
# are ERROR scores, so lower is better and a reader gets this backwards), and the refusal to report
# a comparison it cannot make.

def _twin_labels(n=60, b_err=0.30, c_err=0.05, a_err=0.30, shadow=None, live=None):
    """Rows where head C is deliberately CLOSER to the MC truth than A and B.

    A and B are given the SAME error on purpose: the coverage arm and the control should look alike
    unless the extra states bought something, so a test fixture that made them differ would let a
    broken contrast pass on the fixture's own asymmetry.
    """
    rows = []
    for i in range(n):
        mc = 0.1 + 0.8 * ((i * 7) % 10) / 10.0
        sign = 1.0 if i % 2 else -1.0
        r = {"battle": f"/t/b{i % 6}", "inv": i, "turn": 10 + i % 20, "opponent": "heuristic",
             "opp_class": "bot", "outcome": "loss" if i % 3 else "win", "mc": mc, "n": 8,
             "value": 1.0,
             "win_prob": float(np.clip(mc + sign * a_err, 0.0, 1.0)),
             "twin_b_pred": float(np.clip(mc + sign * b_err, 0.0, 1.0)),
             "twin_c_pred": float(np.clip(mc + sign * c_err, 0.0, 1.0))}
        if shadow is not None:
            r["shadow_value"] = shadow(i)
        if live is not None:
            r["live_v"] = live(i)
        rows.append(r)
    return rows


def test_the_paired_read_scores_the_better_labelled_head_as_BETTER():
    """SIGN CONVENTION, pinned: these are ERROR scores, so a NEGATIVE difference means the
    first-named head is better. Getting this backwards is the single easiest way to read the arm's
    result as its own opposite."""
    from agents.training.cf_audit import paired_head_read
    out = paired_head_read(_twin_labels(), draws=200)
    assert out["heads_present"] == ["A", "B", "C"]
    c_minus_b = next(c for c in out["contrasts"] if c["contrast"] == "C_minus_B")
    assert c_minus_b["brier"] < 0, "the tighter-labelled head scored as WORSE — sign flipped?"
    lo, hi = c_minus_b["brier_ci"]
    assert lo is not None and hi < 0, "the CI does not exclude zero on a fixture built to"
    assert "PRECISION" in c_minus_b["isolates"]


def test_the_coverage_contrast_reads_NULL_when_B_and_A_are_alike():
    """B−A must be ~0 when the coverage arm bought nothing. A fixture in which A and B have the
    SAME error is the honest null, and a contrast that reported an effect here would be reporting
    its own arithmetic."""
    from agents.training.cf_audit import paired_head_read
    out = paired_head_read(_twin_labels(a_err=0.3, b_err=0.3), draws=200)
    b_minus_a = next(c for c in out["contrasts"] if c["contrast"] == "B_minus_A")
    assert abs(b_minus_a["brier"]) < 1e-9
    assert b_minus_a["mean_abs_pred_diff"] < 1e-9


def test_a_near_zero_contrast_with_no_prediction_divergence_is_flagged_by_the_pred_diff():
    """The reading that a flat contrast is AMBIGUOUS. `mean_abs_pred_diff` is the field that tells
    "the labels did not separate the heads" (a coverage/dosage fact) apart from "they separated and
    it bought nothing" (the pre-registered kill)."""
    from agents.training.cf_audit import paired_head_read
    same = paired_head_read(_twin_labels(b_err=0.3, c_err=0.3), draws=100)
    moved = paired_head_read(_twin_labels(b_err=0.3, c_err=0.05), draws=100)
    cb_same = next(c for c in same["contrasts"] if c["contrast"] == "C_minus_B")
    cb_moved = next(c for c in moved["contrasts"] if c["contrast"] == "C_minus_B")
    assert cb_same["mean_abs_pred_diff"] < 1e-9
    assert cb_moved["mean_abs_pred_diff"] > 0.1


def test_the_paired_read_REFUSES_a_comparison_it_cannot_make():
    """A checkpoint without the twin heads must produce no contrasts at all — not zeros. "This run
    has no twin heads" and "the twins agree exactly" are opposite findings."""
    from agents.training.cf_audit import paired_head_read
    bare = [{k: v for k, v in r.items() if not k.startswith("twin_")}
            for r in _twin_labels(n=20)]
    out = paired_head_read(bare, draws=50)
    assert out["contrasts"] == [] and out["heads_present"] == ["A"]


def test_the_bias_map_carries_the_twin_blocks_ONLY_when_the_heads_were_read():
    from agents.training.cf_audit import bias_map
    labels = _twin_labels()
    frame = [_dec(win_prob=r["win_prob"], outcome=r["outcome"], battle=r["battle"], turn=r["turn"])
             for r in labels]
    design = {"turn_tercile_edges": [12.0, 18.0], "sampler_version": "test", "seed": 0}
    with_heads = bias_map(labels, frame, n_rollouts=8, design=design, accounting={})
    assert with_heads["twin_paired"]["contrasts"]
    assert set(with_heads["twin_resolution"]["by_head"]) >= {"A", "C"}
    # The weighting caveat must travel WITH the numbers, not live only in a docstring: absolute
    # levels here are not comparable with the bias map's population-weighted headline.
    assert "UNWEIGHTED" in with_heads["twin_resolution"]["weighting"]

    bare = [{k: v for k, v in r.items() if not k.startswith("twin_")} for r in labels]
    without = bias_map(bare, frame, n_rollouts=8, design=design, accounting={})
    assert "twin_paired" not in without and "twin_resolution" not in without


def test_the_shadow_block_reports_a_SIGNED_divergence_against_the_live_critic():
    """THE staged-promotion meter. A shadow sitting systematically BELOW the live critic says the
    live critic is optimistic about the states the factory samples — so the SIGN is the reading and
    an absolute value would destroy it."""
    from agents.training.cf_audit import shadow_read
    out = shadow_read(_twin_labels(shadow=lambda i: 1.0, live=lambda i: 3.0), draws=200)
    assert out["shadow_vs_live_v"] == pytest.approx(-2.0)
    assert out["shadow_vs_live_v_abs"] == pytest.approx(2.0)
    lo, hi = out["shadow_vs_live_v_ci"]
    assert lo is not None and hi < 0


def test_the_shadow_block_is_ABSENT_without_a_shadow_head():
    from agents.training.cf_audit import bias_map, shadow_read
    assert shadow_read(_twin_labels()) is None
    labels = _twin_labels()
    frame = [_dec(win_prob=r["win_prob"], outcome=r["outcome"], battle=r["battle"], turn=r["turn"])
             for r in labels]
    out = bias_map(labels, frame, n_rollouts=8,
                   design={"turn_tercile_edges": [12.0, 18.0], "sampler_version": "t", "seed": 0},
                   accounting={})
    assert "shadow" not in out


def test_attach_twin_heads_is_a_NO_OP_when_the_checkpoint_has_no_heads(capsys):
    """Same best-effort contract as `attach_evidential`: a model that will not load (79 of 79
    archived runs, at any given HEAD) costs the audit these columns and NOTHING ELSE."""
    from agents.training.cf_audit import attach_twin_heads

    class _NoModel:
        def probe_model(self, _p):
            raise RuntimeError("ArchDriftError")

    labels = _twin_labels(n=4)
    assert attach_twin_heads(_NoModel(), labels, {}) == 0
    assert "columns omitted" in capsys.readouterr().out
    assert attach_twin_heads(object(), labels, {}) == 0      # no probe_model at all
    assert attach_twin_heads(_NoModel(), [], {}) == 0        # no labels

