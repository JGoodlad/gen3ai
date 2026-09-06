"""THE `--vf-coef` SCALE READOUT under `--critic winprob` (`gen3_winprob_critic_mode_v1`).

`--vf-coef` means a different thing under each critic and NOTHING in a metric name says so: under
`shaped` it weights an MSE on a PopArt-normalised shaped return (O(100) on a ±30 scale); under
`winprob` it weights the win-prob head's BCE against a Bernoulli outcome (ln 2 ≈ 0.693 at init,
falling). The historical default 0.5 was tuned against the first and carries no information about
the second — so the first arm's operator has to READ the ratio rather than inherit a number, and
this prints it on the first rollout that can be read.

Both halves are pinned here because they fail differently: the arithmetic can be wrong in silence,
and the latch can be wrong in a way that only shows up on the SECOND rollout or after a restart.
"""
from __future__ import annotations

import math

import pytest

from agents.training.instrumented_ppo.calibration import (LN2, announce_vf_coef_scale,
                                                          vf_coef_scale_line)


class _FakeModel:
    def __init__(self, vf_coef=0.5):
        self.vf_coef = vf_coef


# ---------------------------------------------------------------------------------------------
# the arithmetic
# ---------------------------------------------------------------------------------------------

def test_the_line_reports_the_TERM_and_the_RATIO_not_just_the_coefficient():
    line = vf_coef_scale_line(0.5, 0.6931471805599453, 0.02)
    assert "0.3466" in line, "the value TERM (coef x BCE) must be stated"
    assert "17.3x" in line, "the ratio against |policy loss| is the actionable number"
    assert "0.0200" in line


def test_ln2_is_the_documented_anchor_and_is_the_real_constant():
    assert LN2 == pytest.approx(math.log(2))
    assert "0.693" in vf_coef_scale_line(0.5, 0.5, 0.1), "the ln 2 anchor must be quoted"


def test_a_near_zero_policy_loss_reports_UNAVAILABLE_rather_than_an_infinite_ratio():
    """A clipped surrogate is SIGNED and sits near zero on a well-fit rollout. Dividing by it
    would print a huge ratio, which reads as 'the value term dominates' — the opposite of what a
    near-zero policy loss means."""
    line = vf_coef_scale_line(0.5, 0.69, 0.0)
    assert "UNAVAILABLE" in line and "x." not in line.split("->")[1][:12]


def test_the_ratio_is_taken_on_MAGNITUDE_so_a_negative_policy_loss_reads_the_same():
    a = vf_coef_scale_line(0.5, 0.69, -0.02)
    b = vf_coef_scale_line(0.5, 0.69, 0.02)
    assert a.split("->")[1] == b.split("->")[1]


def test_the_line_names_the_hazard_and_where_to_read_about_it():
    line = vf_coef_scale_line(0.5, 0.69, 0.02)
    assert "BCE" in line and "MSE" in line, "both currencies must be named"
    assert "design_winprob_only_critic.md" in line


# ---------------------------------------------------------------------------------------------
# the latch
# ---------------------------------------------------------------------------------------------

def test_it_prints_once_and_only_once(capsys):
    m = _FakeModel()
    announce_vf_coef_scale(m, [0.69, 0.70], [0.02])
    announce_vf_coef_scale(m, [0.60], [0.03])
    out = capsys.readouterr().out
    assert out.count("first rollout scale") == 1


def test_an_unreadable_rollout_prints_NOTHING_and_does_NOT_latch(capsys):
    """A `train()` with no scorable win-prob label must not invent a number — and must not burn
    the one announcement either, or the run loses the reading entirely."""
    m = _FakeModel()
    announce_vf_coef_scale(m, None, [0.02])
    announce_vf_coef_scale(m, [], [0.02])
    announce_vf_coef_scale(m, [0.69], [])
    assert capsys.readouterr().out == ""
    assert not getattr(m, "_vf_scale_announced", False)
    announce_vf_coef_scale(m, [0.69], [0.02])
    assert "first rollout scale" in capsys.readouterr().out


def test_both_inputs_are_the_per_minibatch_LISTS_train_accumulates(capsys):
    """`win_prob_metrics['loss']` and `pg_losses` are lists, averaged by `train()` for its own
    `record` calls; this must average them the SAME way so the printed pair is exactly what
    `win_prob/loss` and `train/policy_gradient_loss` publish for that rollout."""
    announce_vf_coef_scale(_FakeModel(), [0.60, 0.80], [0.01, 0.03])
    out = capsys.readouterr().out
    assert "BCE 0.7000" in out and "0.0200" in out


def test_the_latch_is_EXCLUDED_from_the_checkpoint_so_a_restart_re_prints_it():
    """A launcher restart re-prints the startup `[CRITIC]` banner this line belongs beside, and
    the run's `--vf-coef` may have been changed between them — a latch riding the checkpoint would
    silence the reading for the rest of the run's life after its first three hours."""
    from agents.training.instrumented_ppo_test import _build_tiny_ppo

    model, _ = _build_tiny_ppo(n_steps=2, n_envs=1)
    # The BOUND method on a real model, not the source — an exclusion list that is right in the
    # source and wrong after `super()` composes it would pass a source scan and fail here.
    assert "_vf_scale_announced" in model._excluded_save_params()


def test_train_calls_it_only_on_the_winprob_critic():
    """A source pin: the call must sit inside `train()`'s `critic_winprob` block. On `shaped` the
    coefficient still multiplies the MSE it was tuned for, so the whole warning would be false."""
    import inspect

    from agents.training.instrumented_ppo import ppo as ppo_mod

    src = inspect.getsource(ppo_mod.InstrumentedMaskablePPO.train)
    i_call = src.index("announce_vf_coef_scale(")
    guard = src.rindex("if critic_winprob:", 0, i_call)
    assert "\n\n" not in src[guard:i_call], (
        "the announcement must be inside the `if critic_winprob:` block it follows")
