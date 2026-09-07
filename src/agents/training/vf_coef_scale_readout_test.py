"""THE `--vf-coef` SCALE READOUT under `--critic winprob` (`gen3_winprob_critic_mode_v1`).

`--vf-coef` means a different thing under each critic and NOTHING in a metric name says so: under
`shaped` it weights an MSE on a PopArt-normalised shaped return (O(100) on a ±30 scale); under
`winprob` it weights the win-prob head's BCE against a Bernoulli outcome (ln 2 ≈ 0.693 at init,
falling). The historical default 0.5 was tuned against the first and carries no information about
the second — so the first arm's operator has to READ the balance rather than inherit a number.

🚨 **THE FIRST SHIPPED VERSION READ THE WRONG QUANTITY, and this file is mostly about that.** It
divided the value TERM by `|policy loss|`, and that denominator is degenerate BY CONSTRUCTION: on
epoch 1 of a rollout PPO's clipped surrogate has `ratio == 1` and sits at its stationary point, so
`|policy loss| ≈ 0`. The live arm `ai_v12_01_winprob_critic` printed **165x** on rollout 1 from a
policy loss of 0.0004. The gradient form on that same run is UNREADABLE at rollout 1
(`grad/policy_norm_shared` was exactly 0.0), reads **91x** at rollout 2, and converges to **4.6x**
by rollout 17. What competes on the shared trunk is the GRADIENT, and it is a quantity the run
already computes.

Three halves are pinned here because they fail differently: the arithmetic can be wrong in silence,
the DEGENERACY GUARD can be wrong in a way that only shows up on the rollout it should have
skipped, and the latch can be wrong in a way that only shows up on the SECOND rollout or after a
restart.
"""
from __future__ import annotations

import math

import pytest

from agents.training.instrumented_ppo.calibration import (LN2, MIN_POLICY_GRAD_NORM,
                                                          announce_vf_coef_scale,
                                                          vf_coef_scale_line)


class _FakeModel:
    def __init__(self, vf_coef=0.5):
        self.vf_coef = vf_coef


def _gb(policy_norm, value_norm):
    """A `grad_balance_metrics`-shaped dict, with the two keys this reads."""
    return {"grad/policy_norm_shared": policy_norm,
            "grad/value_norm_shared": value_norm,
            "grad/policy_share": 0.1,
            "grad/value_share": 0.4}


# ---------------------------------------------------------------------------------------------
# the arithmetic — a ratio of GRADIENTS, plus the raw BCE on its own
# ---------------------------------------------------------------------------------------------

def test_the_line_reports_the_raw_BCE_and_the_GRADIENT_norm_ratio():
    # The live arm's rollout-2 reading: policy 0.0048881, value 0.44544 -> 91.1x.
    line = vf_coef_scale_line(0.5, 0.1330, 0.0048881, 0.44544)
    assert "BCE 0.1330" in line, "the raw BCE must be stated on its own"
    assert "91.1x" in line, "the gradient-norm ratio is the actionable number"
    assert "0.4454" in line and "0.004888" in line, "both norms must be shown"


def test_the_ratio_is_exactly_the_grad_balance_paths_own_number():
    """`grad/value_policy_logratio` is `log10(n_vf / n_pi)`, so the printed ratio must be
    `10 ** logratio` — the SAME number the per-rollout series the operator is told to confirm
    against publishes. Two spellings of one quantity is how two published values start
    disagreeing."""
    n_pi, n_vf = 0.0048881, 0.44544
    logratio = math.log10(n_vf / n_pi)                 # what grad_balance_metrics records
    assert 10.0 ** logratio == pytest.approx(n_vf / n_pi)
    assert f"{10.0 ** logratio:.3g}x" in vf_coef_scale_line(0.5, 0.1330, n_pi, n_vf)


def test_ln2_is_the_documented_anchor_and_is_the_real_constant():
    assert LN2 == pytest.approx(math.log(2))
    assert "0.693" in vf_coef_scale_line(0.5, 0.5, 0.01, 0.1), "the ln 2 anchor must be quoted"


def test_the_line_carries_a_one_line_READING_RULE_in_both_directions():
    line = vf_coef_scale_line(0.5, 0.69, 0.01, 0.5)
    assert "cut --vf-coef" in line, "the >>1 direction must name the action"
    assert "raise" in line, "the <<1 direction must name the action"


def test_the_line_names_the_hazard_and_where_to_read_about_it():
    line = vf_coef_scale_line(0.5, 0.69, 0.01, 0.5)
    assert "BCE" in line and "MSE" in line, "both currencies must be named"
    assert "GRADIENT" in line, "it must say the ratio is of gradients, not of losses"
    assert "design_winprob_only_critic.md" in line


def test_a_non_positive_policy_norm_reports_UNAVAILABLE_rather_than_an_infinite_ratio():
    """The pure function stays total. `announce_vf_coef_scale` refuses this case up front, so the
    guard here only keeps a direct caller from printing `inf` — which would read as 'the value
    gradient dominates', a statement about a degenerate minibatch rather than about --vf-coef."""
    line = vf_coef_scale_line(0.5, 0.69, 0.0, 7.53)
    assert "UNAVAILABLE" in line and "inf" not in line


# ---------------------------------------------------------------------------------------------
# the DEGENERACY GUARD — the defect this rewrite exists for
# ---------------------------------------------------------------------------------------------

def test_a_DEGENERATE_first_update_prints_NOTHING_and_does_NOT_latch(capsys):
    """The live arm's rollout 1: `grad/policy_norm_shared` exactly 0.0 against a value norm of
    7.53. The old loss-ratio form printed a confident 165x there. The gradient form must decline
    to answer, and must not burn the one announcement doing so."""
    m = _FakeModel()
    announce_vf_coef_scale(m, [0.1330], _gb(0.0, 7.5331))
    assert capsys.readouterr().out == ""
    assert not getattr(m, "_vf_scale_announced", False)


def test_the_FIRST_NON_DEGENERATE_update_is_the_one_that_prints(capsys):
    """Rollout 1 refuses, rollout 2 (policy 0.0048881 / value 0.44544) answers — and the number it
    prints is rollout 2's, never a carried-over one."""
    m = _FakeModel()
    announce_vf_coef_scale(m, [0.1330], _gb(0.0, 7.5331))
    assert capsys.readouterr().out == ""
    announce_vf_coef_scale(m, [0.1519], _gb(0.0048881, 0.44544))
    out = capsys.readouterr().out
    assert "91.1x" in out and "BCE 0.1519" in out


def test_the_floor_is_on_the_NORM_and_a_norm_just_under_it_is_refused(capsys):
    """The threshold is on the policy gradient NORM, not on the epoch — see the constant's own
    docstring. A norm below the float32 accumulation floor is noise, not a pull."""
    m = _FakeModel()
    announce_vf_coef_scale(m, [0.69], _gb(MIN_POLICY_GRAD_NORM / 2, 0.5))
    assert capsys.readouterr().out == ""
    assert not getattr(m, "_vf_scale_announced", False)
    announce_vf_coef_scale(m, [0.69], _gb(MIN_POLICY_GRAD_NORM * 10, 0.5))
    assert "scale readout" in capsys.readouterr().out


def test_the_floor_sits_well_below_the_live_arms_first_readable_norm():
    """`ai_v12_01_winprob_critic` read 4.9e-3 at rollout 2 — ~3.7 decades of clearance. A floor
    that could refuse a real reading would silence the banner for the life of a run."""
    assert 0.0 < MIN_POLICY_GRAD_NORM < 0.0048881 / 1000


def test_an_unreadable_rollout_prints_NOTHING_and_does_NOT_latch(capsys):
    """A `train()` with no scorable win-prob label, or one where the grad probe did not run at all
    (a non-Gen3 extractor yields `{}`), must not invent a number — and must not burn the one
    announcement either, or the run loses the reading entirely."""
    m = _FakeModel()
    announce_vf_coef_scale(m, None, _gb(0.01, 0.5))
    announce_vf_coef_scale(m, [], _gb(0.01, 0.5))
    announce_vf_coef_scale(m, [0.69], {})
    announce_vf_coef_scale(m, [0.69], None)
    announce_vf_coef_scale(m, [0.69], _gb(0.01, 0.0))          # no value gradient either
    announce_vf_coef_scale(m, [0.69], _gb(float("nan"), 0.5))
    assert capsys.readouterr().out == ""
    assert not getattr(m, "_vf_scale_announced", False)
    announce_vf_coef_scale(m, [0.69], _gb(0.01, 0.5))
    assert "scale readout" in capsys.readouterr().out


# ---------------------------------------------------------------------------------------------
# the latch
# ---------------------------------------------------------------------------------------------

def test_it_prints_once_and_only_once(capsys):
    m = _FakeModel()
    announce_vf_coef_scale(m, [0.69, 0.70], _gb(0.01, 0.5))
    announce_vf_coef_scale(m, [0.60], _gb(0.02, 0.6))
    out = capsys.readouterr().out
    assert out.count("scale readout") == 1


def test_the_BCE_is_the_per_minibatch_LIST_train_accumulates(capsys):
    """`win_prob_metrics['loss']` is a list, averaged by `train()` for its own `record` call; this
    must average it the SAME way so the printed BCE is exactly what `win_prob/loss` publishes for
    that rollout."""
    announce_vf_coef_scale(_FakeModel(), [0.60, 0.80], _gb(0.01, 0.5))
    assert "BCE 0.7000" in capsys.readouterr().out


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
    from agents.training.instrumented_ppo import ppo as ppo_mod

    # The whole TRAIN STEP: the announcement is recorded in `metrics_export._record_head_metrics`,
    # and this predicate is about the fold's guard, not about which module the line sits in.
    src = ppo_mod.train_step_source()
    i_call = src.index("announce_vf_coef_scale(")
    guard = src.rindex("if critic_winprob:", 0, i_call)
    assert "\n\n" not in src[guard:i_call], (
        "the announcement must be inside the `if critic_winprob:` block it follows")


def test_train_hands_it_the_GRAD_BALANCE_dict_and_runs_no_second_backward():
    """A source pin on the seam: the norms must be READ from the probe `train()` already ran, never
    recomputed. A second `autograd.grad` here would cost a rollout's worth of graph AND could
    disagree with the `grad/value_policy_logratio` series the line tells the operator to confirm
    against."""
    from agents.training.instrumented_ppo import ppo as ppo_mod

    # The whole TRAIN STEP: the announcement is recorded in `metrics_export._record_head_metrics`,
    # and this predicate is about the fold's guard, not about which module the line sits in.
    src = ppo_mod.train_step_source()
    i_call = src.index("announce_vf_coef_scale(")
    # Slice to the MATCHING close paren — the first `)` closes `get("loss")`, not the call.
    depth, end = 0, i_call
    for end in range(i_call, len(src)):
        if src[end] == "(":
            depth += 1
        elif src[end] == ")":
            depth -= 1
            if depth == 0:
                break
    call = src[i_call:end + 1]
    assert "grad_balance" in call, "it must be handed the existing grad-balance dict"
    assert "pg_losses" not in call, "the |policy loss| denominator is the defect being removed"
