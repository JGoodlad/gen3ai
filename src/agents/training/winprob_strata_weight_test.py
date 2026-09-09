"""Gates for `--win-prob-strata-weight` (`gen3_winprob_strata_weight_v1`, config v115).

THE LEVER. The win-prob critic's head barely separates opponents, and the head refit
(`designs/research_state/measurements/winprob_head_refit_2026-09-09/`) located the fault in the
TARGET rather than the head: only 10.2 % / 14.4 % of the terminal 0/1 label's variance lies BETWEEN
(cycle, opponent) cells, so a head minimising BCE buys its resolution from the board and its own
team, which is cheaper. §11 names the un-built lever with the best effect-to-cost ratio —
opponent-stratified weighting of the win-prob loss, which raises the between-cell SHARE of the
objective directly, with no new labels and no rollout cost. This file is that flag's gate.

WHAT EACH TEST IS FOR. The failure modes of a per-sample loss weight are all SILENT, so each one is
made unrepresentable rather than merely unlikely:

  * OFF must be BIT-identical, not approximately equal — otherwise every arm ever run without the
    flag is a different experiment from the one recorded.
  * At `s = 1` the CLASS TOTALS must actually balance, or the flag is a placebo with a scalar.
  * The mean weight must be exactly 1, or `s` silently rescales the value gradient and the arm
    confounds "re-priced the mix" with "raised the learning rate on the critic".
  * The CAP must bind where it is documented to bind, because at the production mix it DOES bind
    and a reader comparing `strata_share_*` against 50/50 must know why it is 44/56.
  * The flag must REFUSE without the win-prob critic rather than no-op: under `--critic shaped`
    the BCE it reweights is an auxiliary diagnostic, not the value loss.
  * The value must be RECORDED and re-read on a flagless resume, or a launcher restart converts
    the arm back into its own control under the same run name (the v100 defect).
"""
import inspect

import numpy as np
import pytest
import torch

from agents.model.model_version.migrations import _migrate_config
from agents.model.opp_intent import OPP_CLASS_NAMES
from agents.training.instrumented_ppo import InstrumentedMaskablePPO
from agents.training.instrumented_ppo.constants import _STRATA_WEIGHT_CAP
from agents.training.instrumented_ppo.value_terms import ValueTerms

_N_CLASSES = len(OPP_CLASS_NAMES)


def _weights(c, m, s):
    """`(w, metrics)` with the ON-path contract unpacked — the helper returns a PAIR whenever the
    flag is on, and `w is None` is 'no weighting applies', not 'nothing to say'."""
    out = ValueTerms._win_prob_strata_weights(c, m, s)
    assert out is not None, "the flag is on; the helper must publish metrics either way"
    return out


def _buffer(counts, known=None):
    """A synthetic rollout buffer's (opp_class, win_mask) columns with the given per-class counts.

    `known` optionally marks a suffix of each class's rows UNLABELED (mask 0) — the trailing
    in-progress episode the callback never fills — so the tests can assert the weights are computed
    over the KNOWN rows and not over the buffer's shape.
    """
    cls, msk = [], []
    known = list(counts) if known is None else list(known) + list(counts)[len(known):]
    for code, n in enumerate(counts):
        k = known[code]
        cls += [code] * n
        msk += [1.0] * k + [0.0] * (n - k)
    return (np.asarray(cls, dtype=np.int64).reshape(-1, 1),
            np.asarray(msk, dtype=np.float32).reshape(-1, 1))


# ── (1) OFF is BIT-identical ────────────────────────────────────────────────────────────────────
def test_weight_zero_returns_no_vector_at_all():
    """`s = 0` must not produce a vector of ones — the caller has to take the ORIGINAL expression.

    A ones-vector would multiply through and (float32 being exact at 1.0) probably come out equal,
    but "probably" is not the contract: OFF is the baseline every archived arm was run under.
    """
    c, m = _buffer([100, 900])
    assert ValueTerms._win_prob_strata_weights(c, m, 0.0) is None
    assert ValueTerms._win_prob_strata_weights(c, m, -0.5) is None


def test_the_loss_at_weight_zero_is_BIT_identical_to_the_unweighted_path():
    torch.manual_seed(0)
    logits = torch.randn(64, 1, requires_grad=True)
    target = (torch.rand(64, 1) > 0.5).float()
    mask = torch.ones(64, 1)
    base, _ = ValueTerms._win_prob_loss(logits, target, mask)
    off, _ = ValueTerms._win_prob_loss(logits, target, mask, None, None, None)
    assert off.item() == base.item()                      # bit-identical, not approx
    assert torch.equal(off, base)


def test_a_missing_opp_class_column_takes_the_unweighted_path():
    """Half the pair present is not half the feature — it is the unweighted loss, unchanged."""
    torch.manual_seed(1)
    logits, target, mask = torch.randn(32, 1), torch.zeros(32, 1), torch.ones(32, 1)
    base, _ = ValueTerms._win_prob_loss(logits, target, mask)
    w = torch.full((_N_CLASSES,), 3.0)
    only_w, m1 = ValueTerms._win_prob_loss(logits, target, mask, None, w, None)
    assert torch.equal(only_w, base) and "loss_unweighted" not in m1


# ── (2) at s = 1 the CLASS TOTALS balance ───────────────────────────────────────────────────────
def test_at_one_every_present_class_contributes_the_same_share():
    """The definition of the lever. Uncapped mix (min freq 0.2 > 1/8), so parity is exact."""
    counts = [200, 600, 200, 0]
    c, m = _buffer(counts)
    w, met = _weights(c, m, 1.0)
    n = float(sum(counts))
    for code, k in enumerate(counts):
        if k:
            assert k * float(w[code]) == pytest.approx(n / 3.0, rel=1e-6)
            assert met[f"strata_share_{OPP_CLASS_NAMES[code]}"] == pytest.approx(1 / 3.0, rel=1e-6)
    assert met["strata_capped"] == 0.0
    assert met["strata_n_classes"] == 3.0
    assert met["strata_w_entropy"] == pytest.approx(1.0, abs=1e-6)   # perfectly balanced


def test_the_exponent_INTERPOLATES_rather_than_switching():
    """`s` between 0 and 1 must move the share monotonically toward parity, not jump."""
    c, m = _buffer([200, 600, 200, 0])
    shares = []
    for s in (0.25, 0.5, 0.75, 1.0):
        _, met = _weights(c, m, s)
        shares.append(met["strata_share_bot"])
    assert shares == sorted(shares)                       # 0.20 -> 1/3, monotone
    assert shares[0] > 0.20 and shares[-1] == pytest.approx(1 / 3.0, rel=1e-6)


def test_a_single_present_class_WEIGHS_NOTHING_but_still_REPORTS():
    """🚨 The case this build's first smoke landed in and could not see.

    Under `--debug`, and on any run before the self-play pool seeds, every opponent is a bot — one
    class, so every weight would be 1 and the correct action is to weigh nothing. With the metrics
    suppressed too, TB showed an EMPTY `strata_*` family, which is indistinguishable from the flag
    never being passed, the critic not being `winprob`, and the plumbing being broken. So the
    inactive case reports `strata_active = 0` and says how many classes it saw.
    """
    c, m = _buffer([0, 500, 0, 0])
    w, met = _weights(c, m, 1.0)
    assert w is None                                   # nothing is weighted -> bit-identical loss
    assert met["strata_active"] == 0.0
    assert met["strata_n_classes"] == 1.0
    assert met["strata_rows"] == 500.0
    assert met["strata_weight"] == 1.0


def test_an_UNFILLED_rollout_reports_inactive_rather_than_vanishing():
    """`win_mask` all zero = the labels were never back-filled. Same rule: report, never vanish."""
    c, m = _buffer([100, 900], known=[0, 0])
    w, met = _weights(c, m, 1.0)
    assert w is None and met["strata_active"] == 0.0 and met["strata_rows"] == 0.0


def test_a_MISSING_opp_class_column_still_reports():
    w, met = _weights(None, None, 1.0)
    assert w is None and met["strata_active"] == 0.0


def test_rows_with_NO_LABEL_are_excluded_from_the_frequencies():
    """The weights describe the population the loss is computed over, which is the KNOWN rows."""
    c, m = _buffer([400, 600, 0, 0], known=[100, 600])    # 400 bot rows, only 100 labeled
    _, met = _weights(c, m, 1.0)
    assert met["strata_rows"] == 700.0
    assert met["strata_frac_bot"] == pytest.approx(100 / 700)


# ── (3) the mean weight is exactly 1 ────────────────────────────────────────────────────────────
@pytest.mark.parametrize("counts", [[100, 900, 0, 0], [50, 800, 100, 50], [1, 999, 0, 0],
                                    [300, 300, 300, 100]])
@pytest.mark.parametrize("s", [0.3, 0.6, 1.0])
def test_the_mean_weight_over_the_buffer_is_one(counts, s):
    """If it were not, `s` would silently scale the value gradient and confound the arm."""
    c, m = _buffer(counts)
    w, _ = _weights(c, m, s)
    n = float(sum(counts))
    mean_w = sum(k * float(w[code]) for code, k in enumerate(counts)) / n
    assert mean_w == pytest.approx(1.0, rel=1e-6)


def test_the_minibatch_read_back_confirms_the_mean_weight_LIVE():
    """`strata_row_w_mean` is what a reader checks in TB; it must be the same quantity."""
    counts = [100, 900, 0, 0]
    c, m = _buffer(counts)
    w, _ = _weights(c, m, 1.0)
    logits = torch.zeros(sum(counts), 1)
    # The per-row BCE must VARY with the class or a reweighting cannot move the mean, and a test
    # that cannot distinguish "the weights were applied" from "the weights were ones" is vacuous.
    # Bots are LOST here and the pool is WON, so the two classes carry different per-row losses.
    cls_t = torch.as_tensor(c)
    target = (cls_t != 0).float()
    _, met = ValueTerms._win_prob_loss(
        logits + 1.0, target, torch.ones_like(logits), None, w, cls_t)
    assert met["strata_row_w_mean"] == pytest.approx(1.0, rel=1e-5)
    # and the weighted loss is NOT the unweighted one, or the lever did nothing
    assert met["loss"] != met["loss_unweighted"]


# ── (4) the CAP ─────────────────────────────────────────────────────────────────────────────────
def test_the_cap_binds_below_one_over_cap_and_is_reported():
    """A class at 1/1000 of the buffer would ask for 1000x; the cap is the whole variance bound."""
    c, m = _buffer([1, 999, 0, 0])
    w, met = _weights(c, m, 1.0)
    assert met["strata_capped"] == 1.0
    # raw was capped at `_STRATA_WEIGHT_CAP` BEFORE renormalisation, so the post-norm weight is
    # cap/Z — strictly below the cap, and far below the 1000x the uncapped rule would have asked.
    assert float(w[0]) < _STRATA_WEIGHT_CAP
    # The RATIO is cap / raw_pool, and raw_pool = (999/1000) ** -1 — so it is `cap * 0.999`, not
    # `cap`. Renormalisation is by a scalar and cannot change a ratio; the cap alone can.
    assert float(w[0]) / float(w[1]) == pytest.approx(_STRATA_WEIGHT_CAP * 0.999, rel=1e-6)


def test_the_cap_BINDS_AT_THE_PRODUCTION_MIX_and_the_split_is_44_56():
    """The documented arithmetic, pinned so the doc cannot drift from the code.

    ~10 % bots / ~90 % self-play is the measured post-promotion episode mix. At `s = 1` the bot
    stratum asks for 10x, the cap holds it at 8x, and the objective splits 44/56 rather than 50/50.
    That is a deliberate trade (a 4.4x re-pricing with a bounded per-row weight, instead of 5x with
    an unbounded one) and a reader comparing `strata_share_*` against parity has to be able to find
    it stated somewhere that a test enforces.
    """
    c, m = _buffer([100, 900, 0, 0])
    _, met = _weights(c, m, 1.0)
    assert met["strata_capped"] == 1.0
    assert met["strata_share_bot"] == pytest.approx(0.444, abs=0.002)
    assert met["strata_share_pool"] == pytest.approx(0.556, abs=0.002)
    assert met["strata_w_bot"] == pytest.approx(4.444, abs=0.01)


def test_no_class_can_ever_exceed_the_cap_relative_to_the_smallest():
    for counts in ([1, 9999, 0, 0], [2, 100, 3000, 4], [1, 1, 1, 100000]):
        c, m = _buffer(counts)
        w, _ = _weights(c, m, 1.0)
        live = [float(w[i]) for i, k in enumerate(counts) if k]
        assert max(live) / min(live) <= _STRATA_WEIGHT_CAP * (1 + 1e-9)


# ── (5) it REFUSES without the win-prob critic, never no-ops ────────────────────────────────────
def _ns(**kw):
    import argparse
    return argparse.Namespace(**kw)


def test_the_combination_check_refuses_a_strata_weight_under_the_shaped_critic():
    from main.train.combination_checks import COMBINATION_CHECKS, failing_checks
    names = {c.name for c in COMBINATION_CHECKS}
    assert "winprob_strata_needs_the_winprob_critic" in names, (
        "the refusal must live in combination_checks so BOTH `resolve_config` and "
        "`python -m main.checkargs` report it — a rule only one surface knows is the defect that "
        "module's docstring exists to end.")
    bad = [c.name for c in failing_checks(
        _ns(critic="shaped", win_prob_strata_weight=1.0, _explicit_flags={"win_prob_strata_weight"}))]
    assert "winprob_strata_needs_the_winprob_critic" in bad


def test_the_check_is_silent_under_the_winprob_critic_and_when_the_flag_is_off():
    from main.train.combination_checks import failing_checks
    ok = [c.name for c in failing_checks(
        _ns(critic="winprob", win_prob_strata_weight=1.0, win_prob_mode="shaping"))]
    assert "winprob_strata_needs_the_winprob_critic" not in ok
    off = [c.name for c in failing_checks(_ns(critic="shaped", win_prob_strata_weight=0.0))]
    assert "winprob_strata_needs_the_winprob_critic" not in off
    unset = [c.name for c in failing_checks(_ns(critic=None, win_prob_strata_weight=None))]
    assert "winprob_strata_needs_the_winprob_critic" not in unset


def test_the_range_check_refuses_outside_zero_to_one():
    import main.train.config as cfg
    src = inspect.getsource(cfg)
    assert "not (0.0 <= args.win_prob_strata_weight <= 1.0)" in src
    assert "--win-prob-strata-weight must be in [0, 1]" in src


# ── (6) RECORDED and resume-INHERITED ───────────────────────────────────────────────────────────
def test_the_argparse_default_is_None_so_the_resolve_line_is_reachable():
    """Surface 1 is TWO claims. A non-None default makes `_resolve` dead code while the test that
    only checks the line's PRESENCE keeps passing — the 2026-08-22 five-flag catch."""
    from main.train.parser import build_parser
    args = build_parser().parse_args([])
    assert args.win_prob_strata_weight is None
    assert build_parser().parse_args(["--win-prob-strata-weight", "0.5"]).win_prob_strata_weight == 0.5
    assert build_parser().parse_args(["--win_prob_strata_weight", "1"]).win_prob_strata_weight == 1.0


def test_a_flagless_resume_INHERITS_the_recorded_value():
    import main.train.config as cfg
    assert '_resolve("win_prob_strata_weight", 0.0)' in inspect.getsource(cfg)


def test_it_round_trips_through_model_config_json():
    from agents.model.model_version import ModelVersion as MV
    fields = {f for f in MV.__dataclass_fields__}
    assert "win_prob_strata_weight" in fields
    assert MV.__dataclass_fields__["win_prob_strata_weight"].default == 0.0


def test_a_pre_v115_config_migrates_to_OFF():
    """0.0 is a RECORD, not a guess: no run before this flag could have weighted its BCE."""
    from agents.model.model_version.constants import MODEL_CONFIG_VERSION
    out = _migrate_config({"config_version": 114})
    assert out["win_prob_strata_weight"] == 0.0
    assert out["config_version"] == MODEL_CONFIG_VERSION


def test_the_writer_records_it_and_the_snapshot_rebuilds_it():
    import main.train.run_io as run_io
    import main.train.lifecycle as lifecycle
    import main.train.model_build as model_build
    assert '"win_prob_strata_weight": float(' in inspect.getsource(run_io)
    assert "win_prob_strata_weight=float(" in inspect.getsource(lifecycle)
    src = inspect.getsource(model_build)
    assert src.count("win_prob_strata_weight=args.win_prob_strata_weight") == 2, (
        "both the RESUME and the FRESH ModelVersion construction must carry it, or a resume "
        "records a different config from the one it is running.")


def test_the_ModelVersion_CONSTRUCTOR_accepts_the_kwarg():
    """The surface a dataclass FIELD does not buy, and the one this build actually tripped on.

    `from_layout_and_policy_kwargs` spells its keywords out one by one, so adding the field and
    both `model_build` call sites still raises `TypeError: unexpected keyword argument` at launch —
    caught here by the `--debug` smoke, ~40 s in, which is exactly the class of failure
    `flag_registry_test` exists to move offline for the toggles it covers. This is that check for
    the training-coefficient family, which that registry deliberately does not declare.
    """
    from agents.model.model_version.construct import ModelVersionConstruction
    sig = inspect.signature(ModelVersionConstruction.from_layout_and_policy_kwargs)
    assert "win_prob_strata_weight" in sig.parameters
    assert sig.parameters["win_prob_strata_weight"].default == 0.0
    assert "win_prob_strata_weight=float(win_prob_strata_weight)" in inspect.getsource(
        ModelVersionConstruction.from_layout_and_policy_kwargs), (
        "the parameter is accepted but never forwarded — it would be silently dropped, which is "
        "strictly worse than the TypeError it replaces.")


def test_it_is_NOT_gated_by_check_compatible():
    """A training-only loss weight inside `check_compatible` would FATAL a run while loading its
    own frozen pool / eval / distill opponents, whose forward is identical regardless."""
    from agents.model.model_version import compat
    assert "win_prob_strata_weight" not in inspect.getsource(compat)


# ── the PPO wiring itself ───────────────────────────────────────────────────────────────────────
def test_the_ppo_hparam_defaults_to_off():
    assert InstrumentedMaskablePPO.win_prob_strata_weight == 0.0


def test_train_computes_the_weights_ONCE_PER_BUFFER_and_gates_on_the_winprob_critic():
    src = inspect.getsource(InstrumentedMaskablePPO.train)
    assert "_win_prob_strata_weights(" in src
    assert "critic_winprob and float(getattr(self, \"win_prob_strata_weight\", 0.0)) > 0.0" in src
    # computed from the BUFFER, not from `rollout_data` — a per-minibatch recomputation would make
    # the class balance a sampling-noise term and break the mean-weight-1 normalisation.
    assert "self.rollout_buffer.observations" in src.split("_win_prob_strata_weights(")[0][-800:]


def test_the_counterfactual_callers_stay_UNWEIGHTED():
    """`cf_terms` scores FOREIGN recorded states whose opponent mix is the label factory's, not the
    rollout's — reweighting them by the rollout's frequencies would be a category error."""
    import agents.training.cf_terms as cf_terms
    src = inspect.getsource(cf_terms)
    assert "_win_prob_loss(head(pooled.detach()), target, mask, margin)" in src
    assert "strata" not in src


def test_the_metric_names_a_read_depends_on_are_all_emitted():
    """A TB read is a contract: `designs/training/critic_and_value_losses.md` names these keys."""
    c, m = _buffer([100, 700, 150, 50])
    w, met = _weights(c, m, 1.0)
    assert met["strata_active"] == 1.0
    for k in ("strata_weight", "strata_active", "strata_rows", "strata_n_classes", "strata_capped",
              "strata_w_min", "strata_w_max", "strata_w_entropy"):
        assert k in met, k
    for name in OPP_CLASS_NAMES.values():
        for pre in ("strata_w_", "strata_frac_", "strata_share_"):
            assert pre + name in met, pre + name
    logits = torch.zeros(len(m), 1)
    _, lm = ValueTerms._win_prob_loss(logits, torch.zeros_like(logits), torch.ones_like(logits),
                                      None, w, torch.as_tensor(c))
    assert "loss_unweighted" in lm and "strata_row_w_mean" in lm


def test_the_weighted_loss_actually_changes_the_GRADIENT():
    """The end-to-end claim, and the ONE that says the flag does its job: the share of the value
    gradient the minority class carries moves from its episode proportion toward parity.

    A weight that never reaches `.grad` is the vacuity this repo's stub gate exists for, one layer
    up — so this is asserted on the gradient itself, not on the metrics dict.
    """
    counts = [100, 900, 0, 0]                              # the measured ~10/90 production mix
    c, m = _buffer(counts)
    w, _ = _weights(c, m, 1.0)
    cls_t, msk_t = torch.as_tensor(c), torch.as_tensor(m)
    target = torch.where(cls_t == 0, torch.zeros_like(msk_t), torch.ones_like(msk_t))
    pulls = []
    for weights in (None, w):
        logit = torch.zeros(len(m), 1, requires_grad=True)
        loss, _ = ValueTerms._win_prob_loss(
            logit, target, msk_t, None, weights, cls_t if weights is not None else None)
        loss.backward()
        g = logit.grad.reshape(-1).abs()
        bot = float(g[cls_t.reshape(-1) == 0].sum())
        pulls.append(bot / (bot + float(g[cls_t.reshape(-1) == 1].sum())))
    unweighted, weighted = pulls
    assert unweighted == pytest.approx(0.10, abs=1e-6)     # episode proportion, exactly
    # 0.444 and not 0.5 because the 8x cap binds at this mix — see the 44/56 test above. The point
    # is the 4.4x re-pricing of the between-class signal, and it must be MEASURED here rather than
    # assumed from the weight vector.
    assert weighted == pytest.approx(0.444, abs=0.002)
