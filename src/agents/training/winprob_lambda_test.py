"""Gates for `--win-prob-lambda` / `--win-prob-lambda-truncated` (`gen3_winprob_lambda_v1`, v116).

THE LEVER. Under `--critic winprob` the value loss is a BCE against ONE terminal bit copied to
every state of the episode. The head refit
(`designs/research_state/measurements/winprob_head_refit_2026-09-09/`) located the win-prob
critic's conditional miscalibration in that TARGET rather than in the head: only 10.2 % / 14.4 % of
the label's variance lies BETWEEN (cycle, opponent) cells, so a learner minimising a proper scoring
rule shrinks the weak axes toward the marginal and the turn-1 value barely separates opponents
(spread ratio ~0.1 at turn 1 against ~0.5-0.8 over all states). Mid- and late-game values DO
separate them, so a λ-return moves that information backward within the episode along a channel
with far less noise than the terminal draw. This file is that flag's gate.

WHAT EACH TEST IS FOR — every failure mode of a target rewrite is silent, so each is made
unrepresentable rather than merely unlikely:

  * OFF must be BIT-identical, not approximately equal, or every arm ever run without the flag is
    a different experiment from the one recorded. λ = 1.0 skips the recursion WHOLE, which is also
    what keeps the truncation convention unchanged at the default.
  * The recursion must equal its hand-computed value on a toy episode, and the outcome's weight
    must be exactly λ**d — the number `lambda_bootstrap_frac` is a summary of.
  * A TRUNCATED episode must take the declared branch, and the rows it unmasks must be COUNTED,
    or an effect from "more rows" is indistinguishable from an effect from "different targets".
  * An episode that ended with NO recorded outcome must stay masked under both branches — its
    rows anchor at `y = 0`, and unmasking them would train the head against a fabricated label.
  * The BCE must accept a SOFT target and pull the logit toward it (it is a proper scoring rule
    for the target's expectation; an implementation that quietly rounded would still "work").
  * The pair must REFUSE without the win-prob critic rather than no-op: under `--critic shaped`
    `rollout_buffer.values` is a PopArt-normalised shaped return, not a probability.
  * The values must be RECORDED and re-read on a flagless resume, or a launcher restart converts
    the arm back into its own control under the same run name (the v100 defect).
"""
import inspect

import numpy as np
import pytest
import torch

from agents.model.model_version.migrations import _migrate_config
from agents.training.instrumented_ppo import InstrumentedMaskablePPO
from agents.training.instrumented_ppo.value_terms import ValueTerms
from agents.training.win_prob_callback import (
    LAMBDA_OFF, LAMBDA_TRUNCATED_MODES, WinProbLabelCallback, lambda_metrics,
    lambda_return_targets,
)


# ──────────────────────────────────────────────────────────────────────────────────────────────
# A synthetic rollout: one env column, four states, the terminal at row 3.
# ──────────────────────────────────────────────────────────────────────────────────────────────

def _col(values, y, mask, starts):
    """Column-shaped [n_steps, 1] float arrays from four flat lists."""
    f = lambda a: np.asarray(a, dtype=np.float64).reshape(-1, 1)
    return f(values), f(y), f(mask), f(starts)


def test_the_recursion_matches_the_hand_computed_lambda_return():
    # V = [.4 .5 .6 .7], y = 1, terminal at row 3, λ = 0.5.
    #   G3 = y                       = 1.0
    #   G2 = .5*V3 + .5*G3 = .35+.5  = 0.85
    #   G1 = .5*V2 + .5*G2 = .30+.425= 0.725
    #   G0 = .5*V1 + .5*G1 = .25+.3625 = 0.6125
    v, y, m, es = _col([.4, .5, .6, .7], [1, 1, 1, 1], [1, 1, 1, 1], [1, 0, 0, 0])
    g, w, nm, nu = lambda_return_targets(v, y, m, es, np.array([0.9]), np.array([1.0]), 0.5)
    assert np.allclose(g.ravel(), [0.6125, 0.725, 0.85, 1.0])
    assert nu == 0 and np.all(nm >= 0.5)


def test_the_outcome_keeps_weight_lambda_to_the_power_of_the_distance():
    """`lambda_bootstrap_frac` is a summary of this vector, so the vector itself is pinned."""
    lam = 0.5
    v, y, m, es = _col([.4, .5, .6, .7], [1, 1, 1, 1], [1, 1, 1, 1], [1, 0, 0, 0])
    _, w, _, _ = lambda_return_targets(v, y, m, es, np.array([0.9]), np.array([1.0]), lam)
    assert np.allclose(w.ravel(), [lam ** 3, lam ** 2, lam ** 1, lam ** 0])


@pytest.mark.parametrize("lam", [0.0, 0.3, 0.9])
def test_a_terminal_state_ALWAYS_gets_the_outcome_exactly(lam):
    v, y, m, es = _col([.4, .5, .6, .7], [1, 1, 1, 1], [1, 1, 1, 1], [1, 0, 0, 0])
    g, w, _, _ = lambda_return_targets(v, y, m, es, np.array([0.9]), np.array([1.0]), lam)
    assert g[3, 0] == 1.0 and w[3, 0] == 1.0


def test_lambda_zero_is_a_pure_one_step_bootstrap():
    v, y, m, es = _col([.4, .5, .6, .7], [0, 0, 0, 0], [1, 1, 1, 1], [1, 0, 0, 0])
    g, w, _, _ = lambda_return_targets(v, y, m, es, np.array([0.9]), np.array([1.0]), 0.0)
    assert np.allclose(g.ravel(), [0.5, 0.6, 0.7, 0.0])       # V(s[t+1]); the terminal keeps y
    assert np.allclose(w.ravel(), [0.0, 0.0, 0.0, 1.0])


def test_a_TRUNCATED_episode_bootstraps_to_V_of_the_post_rollout_state_and_is_UNMASKED():
    # Nothing ended inside the buffer: mask is 0 everywhere today. V(s_T) = 0.9.
    #   G3 = V(s_T) = 0.9 ; G2 = .5*.7+.5*.9 = .8 ; G1 = .5*.6+.5*.8 = .7 ; G0 = .5*.5+.5*.7 = .6
    v, y, m, es = _col([.4, .5, .6, .7], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0])
    g, w, nm, nu = lambda_return_targets(v, y, m, es, np.array([0.9]), np.array([0.0]), 0.5,
                                         "bootstrap")
    assert np.allclose(g.ravel(), [0.6, 0.7, 0.8, 0.9])
    assert np.all(nm >= 0.5) and nu == 4
    assert np.allclose(w, 0.0), "a bootstrapped row carries NO weight on any outcome"


def test_the_mask_branch_leaves_a_TRUNCATED_episode_EXACTLY_as_it_is_today():
    v, y, m, es = _col([.4, .5, .6, .7], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0])
    g, w, nm, nu = lambda_return_targets(v, y, m, es, np.array([0.9]), np.array([0.0]), 0.5,
                                         "mask")
    assert np.allclose(g, 0.0) and np.allclose(nm, 0.0) and nu == 0


def test_an_episode_that_ENDED_WITHOUT_AN_OUTCOME_is_never_unmasked():
    """The fabricated-label guard. Rows 0-1 are a COMPLETE episode with no recorded outcome (mask
    0, y 0); rows 2-3 are the trailing in-progress one. Only the trailing pair may be bootstrapped
    — unmasking the first would train the head toward a loss that never happened."""
    v, y, m, es = _col([.4, .5, .6, .7], [0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 1, 0])
    g, w, nm, nu = lambda_return_targets(v, y, m, es, np.array([0.9]), np.array([0.0]), 0.5,
                                         "bootstrap")
    assert nm.ravel().tolist() == [0.0, 0.0, 1.0, 1.0]
    assert nu == 2
    assert np.allclose(g.ravel()[:2], 0.0), "an unscored row keeps the placeholder, not a number"


def test_two_episodes_in_one_column_do_not_leak_across_the_boundary():
    # rows 0-1 = a WON episode, rows 2-3 = a LOST one. Each anchors on its own outcome.
    v, y, m, es = _col([.4, .5, .6, .7], [1, 1, 0, 0], [1, 1, 1, 1], [1, 0, 1, 0])
    g, _, _, _ = lambda_return_targets(v, y, m, es, np.array([0.9]), np.array([1.0]), 0.5)
    assert g[1, 0] == 1.0 and g[3, 0] == 0.0
    assert g[0, 0] == 0.5 * 0.5 + 0.5 * 1.0          # (1-λ)V(s1) + λ·y_win
    assert g[2, 0] == 0.5 * 0.7 + 0.5 * 0.0          # (1-λ)V(s3) + λ·y_loss


def test_the_recursion_REFUSES_an_unknown_truncation_mode():
    v, y, m, es = _col([.4, .5, .6, .7], [1, 1, 1, 1], [1, 1, 1, 1], [1, 0, 0, 0])
    with pytest.raises(ValueError):
        lambda_return_targets(v, y, m, es, np.array([0.9]), np.array([1.0]), 0.5, "keep")
    assert LAMBDA_TRUNCATED_MODES == ("bootstrap", "mask")


# ──────────────────────────────────────────────────────────────────────────────────────────────
# The CALLBACK: OFF is bit-identical, ON rewrites the two obs keys in place.
# ──────────────────────────────────────────────────────────────────────────────────────────────

class _Policy:
    _critic_mode = "winprob"

    def __init__(self, boot):
        self._boot = boot

    def predict_values(self, obs):
        return torch.as_tensor(self._boot, dtype=torch.float32).reshape(-1, 1)


class _Buf:
    def __init__(self, values, es, y, mask):
        self.values = np.asarray(values, dtype=np.float32)
        self.episode_starts = np.asarray(es, dtype=np.float32)
        self.observations = {
            "win_target": np.asarray(y, dtype=np.float32)[..., None],
            "win_mask": np.asarray(mask, dtype=np.float32)[..., None],
        }


class _Model:
    def __init__(self, buf, boot, **kw):
        self.rollout_buffer = buf
        self.n_steps, self.n_envs = buf.values.shape
        self.policy = _Policy(boot)
        self.device = "cpu"
        self._last_obs = {"obs": np.zeros((self.n_envs, 1), dtype=np.float32)}
        self._last_episode_starts = np.zeros(self.n_envs, dtype=np.float32)
        self._win_terminal_scratch = np.full((self.n_steps, self.n_envs), np.nan, np.float32)
        self._win_prob_lambda_metrics = None
        for k, v in kw.items():
            setattr(self, k, v)


def _run_callback(**kw):
    """One rollout: a 4-state column terminating at row 3 with a WIN. Returns (model, wt, wm)."""
    buf = _Buf([[.4], [.5], [.6], [.7]], [[1], [0], [0], [0]], [[0], [0], [0], [0]],
               [[0], [0], [0], [0]])
    model = _Model(buf, [0.9], **kw)
    model._last_episode_starts = np.array([1.0], dtype=np.float32)   # row 3 ENDED the episode
    cb = WinProbLabelCallback()
    cb.model = model
    model._win_terminal_scratch[3, 0] = 1.0                          # the win
    cb._on_rollout_end()
    return model, buf.observations["win_target"], buf.observations["win_mask"]


def test_an_UNFLAGGED_run_leaves_the_terminal_bit_target_BIT_identical():
    _, wt_off, wm_off = _run_callback()                              # no attribute at all
    _, wt_one, wm_one = _run_callback(win_prob_lambda=1.0)
    assert wt_off.tobytes() == wt_one.tobytes()
    assert wm_off.tobytes() == wm_one.tobytes()
    assert np.allclose(wt_off.ravel(), 1.0), "every state carries the episode outcome"


def test_an_UNFLAGGED_run_publishes_NO_lambda_family():
    model, _, _ = _run_callback()
    assert model._win_prob_lambda_metrics is None
    model2, _, _ = _run_callback(win_prob_lambda=1.0)
    assert model2._win_prob_lambda_metrics is None, \
        "an absent win_prob/lambda_* family must mean λ = 1.0 and nothing else"


def test_the_loss_at_lambda_ONE_is_BIT_identical_to_the_unflagged_loss():
    logits = torch.tensor([[0.3], [-0.2], [1.1], [0.05]])
    _, wt_off, wm_off = _run_callback()
    _, wt_one, wm_one = _run_callback(win_prob_lambda=1.0)
    t = lambda a: torch.as_tensor(a.reshape(-1, 1))
    a, _ = ValueTerms._win_prob_loss(logits, t(wt_off), t(wm_off))
    b, _ = ValueTerms._win_prob_loss(logits, t(wt_one), t(wm_one))
    assert torch.equal(a, b) and a.item() == b.item()


def test_the_callback_REWRITES_the_targets_in_place_below_one():
    model, wt, wm = _run_callback(win_prob_lambda=0.5)
    assert np.allclose(wt.ravel(), [0.6125, 0.725, 0.85, 1.0])
    assert np.all(wm >= 0.5)
    assert model._win_prob_lambda_metrics["lambda"] == 0.5


def test_the_callback_is_INERT_under_a_critic_that_is_not_winprob():
    """Belt-and-braces behind the combination check: under `shaped` the buffer's values are a
    PopArt-normalised shaped return, and blending them into a BCE target is a category error."""
    buf = _Buf([[.4], [.5], [.6], [.7]], [[1], [0], [0], [0]], [[0], [0], [0], [0]],
               [[0], [0], [0], [0]])
    model = _Model(buf, [0.9], win_prob_lambda=0.5)
    model.policy._critic_mode = "shaped"
    model._last_episode_starts = np.array([1.0], dtype=np.float32)
    cb = WinProbLabelCallback()
    cb.model = model
    model._win_terminal_scratch[3, 0] = 1.0
    cb._on_rollout_end()
    assert np.allclose(buf.observations["win_target"].ravel(), 1.0)
    assert model._win_prob_lambda_metrics is None


def test_a_MISSING_last_obs_falls_back_to_V_of_the_last_row_and_SAYS_SO():
    buf = _Buf([[.4], [.5], [.6], [.7]], [[1], [0], [0], [0]], [[0], [0], [0], [0]],
               [[0], [0], [0], [0]])
    model = _Model(buf, [0.9], win_prob_lambda=0.5)
    model._last_obs = None                                   # no bootstrap forward available
    cb = WinProbLabelCallback()
    cb.model = model
    cb._on_rollout_end()                                     # nothing terminated -> all truncated
    assert model._win_prob_lambda_metrics["lambda_bootstrap_fallback"] == 1.0
    assert buf.observations["win_target"][3, 0] == pytest.approx(0.7)   # V(s[last]), not V(s_T)


def test_the_metric_names_a_read_depends_on_are_all_emitted():
    model, _, _ = _run_callback(win_prob_lambda=0.5)
    m = model._win_prob_lambda_metrics
    for k in ("lambda", "lambda_rows", "lambda_unmasked", "lambda_bootstrap_frac",
              "lambda_weight_mean", "lambda_target_shift", "lambda_loss",
              "lambda_loss_terminal", "lambda_truncated_bootstrap",
              "lambda_bootstrap_fallback"):
        assert k in m, k


def test_the_two_losses_are_scored_on_the_SAME_predictions_and_differ_only_by_the_target():
    """`lambda_loss` vs `lambda_loss_terminal` is the read that says the target moved. Both use the
    RECORDED pre-update V, so their difference cannot be a step of learning."""
    v, y, m, es = _col([.4, .5, .6, .7], [1, 1, 1, 1], [1, 1, 1, 1], [1, 0, 0, 0])
    g, w, nm, nu = lambda_return_targets(v, y, m, es, np.array([0.9]), np.array([1.0]), 0.5)
    out = lambda_metrics(v, y, m, g, w, nm, nu, 0.5, "bootstrap")
    hand = float(np.mean(-(1.0 * np.log(np.clip(v, 1e-6, 1 - 1e-6)))))
    assert out["lambda_loss_terminal"] == pytest.approx(hand)
    assert out["lambda_loss"] != out["lambda_loss_terminal"]
    assert out["lambda_target_shift"] == pytest.approx(float(np.mean(np.abs(g - y))))
    assert out["lambda_bootstrap_frac"] == pytest.approx(0.5)   # λ**d < 0.5 at d = 2 and d = 3


# ──────────────────────────────────────────────────────────────────────────────────────────────
# The SOFT target: a BCE against a probability is a proper score for its expectation.
# ──────────────────────────────────────────────────────────────────────────────────────────────

def test_the_BCE_accepts_a_SOFT_target_and_is_MINIMISED_where_the_prediction_equals_it():
    soft = torch.full((256, 1), 0.7)
    mask = torch.ones_like(soft)
    losses = {}
    for p in (0.5, 0.7, 0.9):
        logit = torch.logit(torch.full((256, 1), p))
        losses[p], _ = ValueTerms._win_prob_loss(logit, soft, mask)
    assert losses[0.7].item() < losses[0.5].item()
    assert losses[0.7].item() < losses[0.9].item()


def test_the_gradient_pulls_the_logit_TOWARD_the_soft_target():
    soft = torch.full((32, 1), 0.7)
    mask = torch.ones_like(soft)
    for p, want_up in ((0.3, True), (0.95, False)):
        logit = torch.logit(torch.full((32, 1), p)).requires_grad_(True)
        loss, _ = ValueTerms._win_prob_loss(logit, soft, mask)
        loss.backward()
        # descent direction is −grad; positive means "raise the logit toward 0.7"
        assert (float(-logit.grad.mean()) > 0) is want_up


# ──────────────────────────────────────────────────────────────────────────────────────────────
# The FLAGS: refusal, range, default, provenance.
# ──────────────────────────────────────────────────────────────────────────────────────────────

def _args(**kw):
    import argparse
    d = dict(critic="winprob", win_prob_lambda=1.0, win_prob_lambda_truncated="bootstrap")
    d.update(kw)
    return argparse.Namespace(**d)


def test_the_combination_check_refuses_a_lambda_under_the_shaped_critic():
    from main.train.combination_checks import COMBINATION_CHECKS, failing_checks
    assert "winprob_lambda_needs_the_winprob_critic" in {c.name for c in COMBINATION_CHECKS}, (
        "the refusal must live in combination_checks so BOTH `resolve_config` and "
        "`python -m main.checkargs` report it")
    bad = [c.name for c in failing_checks(_args(critic="shaped", win_prob_lambda=0.9))]
    assert "winprob_lambda_needs_the_winprob_critic" in bad


def test_the_check_is_silent_under_the_winprob_critic_and_at_the_default():
    from main.train.combination_checks import failing_checks
    for a in (_args(win_prob_lambda=0.9, win_prob_mode="shaping"),
              _args(critic="shaped", win_prob_lambda=1.0),
              _args(critic=None, win_prob_lambda=None)):
        assert "winprob_lambda_needs_the_winprob_critic" not in [
            c.name for c in failing_checks(a)]


def test_the_range_check_refuses_outside_zero_to_one():
    import main.train.config as cfg
    src = inspect.getsource(cfg)
    assert "not (0.0 <= args.win_prob_lambda <= 1.0)" in src
    assert "--win-prob-lambda must be in [0, 1]" in src


def test_the_argparse_defaults_are_None_so_the_resolve_lines_are_reachable():
    """A non-None default makes `_resolve` dead code while a test that only checks the LINE's
    presence keeps passing — the 2026-08-22 five-flag catch."""
    from main.train.parser import build_parser
    args = build_parser().parse_args([])
    assert args.win_prob_lambda is None and args.win_prob_lambda_truncated is None
    assert build_parser().parse_args(["--win-prob-lambda", "0.9"]).win_prob_lambda == 0.9
    assert build_parser().parse_args(["--win_prob_lambda", "0"]).win_prob_lambda == 0.0
    assert build_parser().parse_args(
        ["--win-prob-lambda-truncated", "mask"]).win_prob_lambda_truncated == "mask"


def test_the_truncation_flag_REFUSES_a_value_outside_the_declared_set():
    import argparse

    from main.train.parser import build_parser
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--win-prob-lambda-truncated", "keep"])
    assert isinstance(argparse.ArgumentParser(), argparse.ArgumentParser)


def test_a_flagless_resume_INHERITS_the_recorded_values():
    import main.train.config as cfg
    src = inspect.getsource(cfg)
    assert '_resolve("win_prob_lambda", 1.0)' in src
    assert '_resolve("win_prob_lambda_truncated", "bootstrap")' in src


def test_they_are_dataclass_FIELDS_with_the_pre_flag_defaults():
    from agents.model.model_version import ModelVersion as MV
    assert MV.__dataclass_fields__["win_prob_lambda"].default == LAMBDA_OFF == 1.0
    assert MV.__dataclass_fields__["win_prob_lambda_truncated"].default == "bootstrap"


def test_the_ModelVersion_CONSTRUCTOR_accepts_BOTH_kwargs_and_FORWARDS_them():
    """`from_layout_and_policy_kwargs` spells its keywords out one by one, so a field plus both
    `model_build` call sites still raises `TypeError` at launch. Accepted-but-not-forwarded is
    strictly worse: silently dropped rather than loud."""
    from agents.model.model_version.construct import ModelVersionConstruction
    fn = ModelVersionConstruction.from_layout_and_policy_kwargs
    sig = inspect.signature(fn)
    assert sig.parameters["win_prob_lambda"].default == 1.0
    assert sig.parameters["win_prob_lambda_truncated"].default == "bootstrap"
    src = inspect.getsource(fn)
    assert "win_prob_lambda=float(win_prob_lambda)" in src
    assert 'win_prob_lambda_truncated=str(win_prob_lambda_truncated or "bootstrap")' in src


def test_a_pre_v116_config_migrates_to_OFF():
    """1.0 is a RECORD, not a guess: λ = 1.0 IS the terminal-outcome target every prior run used."""
    from agents.model.model_version.constants import MODEL_CONFIG_VERSION
    out = _migrate_config({"config_version": 115})
    assert out["win_prob_lambda"] == 1.0 and out["win_prob_lambda_truncated"] == "bootstrap"
    assert out["config_version"] == MODEL_CONFIG_VERSION


def test_it_is_NOT_gated_by_check_compatible():
    """A training-only loss TARGET inside `check_compatible` would FATAL a run while loading its
    own frozen pool / eval / distill opponents, whose forward is identical regardless."""
    from agents.model.model_version import compat
    src = inspect.getsource(compat)
    assert "win_prob_lambda" not in src


def test_the_writer_records_both_and_the_hparams_default_to_off():
    import main.train.lifecycle as lifecycle
    import main.train.run_io as run_io
    assert '"win_prob_lambda": float(' in inspect.getsource(run_io)
    assert '"win_prob_lambda_truncated": str(' in inspect.getsource(run_io)
    assert "win_prob_lambda=float(" in inspect.getsource(lifecycle)
    assert InstrumentedMaskablePPO.win_prob_lambda == 1.0
    assert InstrumentedMaskablePPO.win_prob_lambda_truncated == "bootstrap"


def test_BOTH_build_paths_pass_the_pair_through():
    """`model_build` carries the fresh and the resume construction separately; a knob added to one
    and not the other trains with the class default on the other and nothing reports it."""
    from main.train import model_build
    src = inspect.getsource(model_build)
    assert src.count("win_prob_lambda=args.win_prob_lambda") == 2
    assert src.count("win_prob_lambda_truncated=args.win_prob_lambda_truncated") == 2
    assert '("win_prob_lambda",               _PLAIN)' in src
    assert '("win_prob_lambda_truncated",     _PLAIN)' in src


def test_train_folds_the_family_under_the_win_prob_prefix():
    src = inspect.getsource(InstrumentedMaskablePPO.train)
    assert '_win_prob_lambda_metrics' in src
    assert 'win_prob_metrics.setdefault(_lk, []).append(float(_lv))' in src


def test_the_lever_is_declared_in_the_generated_coefficient_table():
    from agents.model.arch_tables import _COEF_MODULE
    assert _COEF_MODULE["win_prob_lambda"] == "win_head"
