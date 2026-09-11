"""Gates for `--win-prob-rollout-weight` (`gen3_winprob_rollout_weight_v1`, v119, critic arm 10).

THE LEVER, and why each gate exists. `--win-prob-rollout-target` buys new bits (R continuations
from a sampled state, `wins / R` as its target) but it buys them at a price: the fraction that
costs 1x the run's own simulation budget is ~0.0012, so the rollout-labelled rows are ~0.12 % of
the win-prob BCE's mass. **A treatment carrying 0.12 % of an objective cannot move the head by
ARITHMETIC, whatever the labels say** — the read at a feasible fraction is a read of nothing. This
flag multiplies the ANCHORED rows' per-row BCE by a constant and renormalises the whole vector to
mean 1 over the scored rows, which raises the treatment's share of the objective at FIXED
simulation cost. At fraction 0.0012 and weight 64 the anchors go from 0.12 % to ~7.1 % of the mass.

Each gate below stands for a way this could be wrong in silence:

  * OFF (1.0) must be BIT-identical — `torch.equal` on the loss TENSOR, not `approx` — or every
    arm run without the flag is a different experiment from the one recorded.
  * Weight `k` must multiply EXACTLY the anchored rows. A weight that leaked onto a neighbour
    would re-price a row nobody paid for, and nothing downstream would show it.
  * The mean weight over the scored rows must stay 1, so the loss SCALE does not move with the
    fraction: a lever that re-prices a mix must not also rescale the value gradient, or the two
    effects are inseparable in the read.
  * It must COMPOSE with `--win-prob-strata-weight` by MULTIPLYING. An overwrite would silently
    disable whichever of the two flags was applied second.
  * 🚨 Under `--win-prob-lambda < 1` it must weight ONLY the anchors, never the rows that
    bootstrap toward them: those rows' targets are a MIXTURE of the anchor, the network's own
    later values and the copied bit, so weighting them would dose arm 8's channel under arm 10's
    flag. The propagated influence is MEASURED (`rollout_influence_lambda`), not dosed.
  * It must REFUSE without a fraction — with no anchors the weight is a vector of ones and the run
    is the unflagged one under a flagged name.
  * The value must be RECORDED and re-read on a flagless resume, or a launcher restart keeps
    paying for every continuation while delivering ~1/50th of the registered dose.
"""
import inspect

import numpy as np
import pytest
import torch

from agents.model.model_version.migrations import _migrate_config
from agents.training.instrumented_ppo.value_terms import ValueTerms
from agents.training.win_prob_callback import lambda_return_targets
from agents.training.win_prob_rollout import (ROLLOUT_WEIGHT_KEY, ROLLOUT_WEIGHT_OFF,
                                              anchor_row_weights, weight_metrics, weighted_mass)
from agents.training.winprob_rollout_test import _run


# ──────────────────────────────────────────────────────────────────────────────────────────────
# THE ARITHMETIC — the whole reason the flag exists
# ──────────────────────────────────────────────────────────────────────────────────────────────

def test_the_weight_vector_has_MEAN_ONE_over_the_scored_rows():
    """The loss SCALE must not move with the fraction. `_win_prob_strata_weights` keeps exactly
    this convention (mean 1 over the known rows, denominator `n_known`), and the two weights
    multiply — so a second convention here would make their product meaningless."""
    anchors = np.zeros((20, 4), dtype=bool)
    anchors[3, 1] = anchors[11, 2] = True
    scored = np.ones((20, 4))
    w = anchor_row_weights(anchors, scored, 64.0)
    assert float((w * scored).sum() / scored.sum()) == pytest.approx(1.0, abs=1e-6)


def test_an_UNSCORED_row_is_in_neither_side_of_the_normaliser():
    """A row the BCE does not score contributes nothing at any weight, so it must not dilute the
    mean the scored rows were normalised to — otherwise the delivered dose would move with the
    truncated-episode fraction, which has nothing to do with this lever."""
    anchors = np.zeros((10, 2), dtype=bool)
    anchors[4, 0] = True
    scored = np.ones((10, 2))
    scored[:, 1] = 0.0                                   # env 1's whole episode is unscored
    w = anchor_row_weights(anchors, scored, 16.0)
    assert float((w * scored).sum() / scored.sum()) == pytest.approx(1.0, abs=1e-6)


def test_the_measured_mass_EQUALS_the_closed_form_a_cost_calculation_quotes():
    """`weighted_mass` is what a design doc and a pre-launch calculation quote; the metric is what
    the run reports. A drift between them is a bug, so both are computed and compared."""
    anchors = np.zeros((50, 10), dtype=bool)
    anchors[np.arange(6), 0] = True                      # 6 anchors of 500 scored rows
    scored = np.ones((50, 10))
    w = anchor_row_weights(anchors, scored, 64.0)
    m = weight_metrics(anchor_mask=anchors, new_mask=scored, row_w=w, weight=64.0)
    assert m["rollout_mass_weighted"] == pytest.approx(weighted_mass(6 / 500.0, 64.0))


def test_THE_HEADLINE_ARITHMETIC_at_the_registered_operating_point():
    """The number the arm was registered on: at the 1x-budget fraction (~0.0012) the anchors carry
    0.12 % of the BCE; weight 64 puts them at ~7.1 %. If this moves, the arm's registered dose bar
    (5-15 % of the mass) moves with it and the read has to be re-registered."""
    assert weighted_mass(0.0012, ROLLOUT_WEIGHT_OFF) == pytest.approx(0.0012, rel=1e-6)
    assert weighted_mass(0.0012, 64.0) == pytest.approx(0.0714, abs=0.0005)


def test_OFF_returns_None_rather_than_a_vector_of_ones():
    """A vector of ones would be arithmetically identical but not BIT-identical through the loss,
    and it would cost a gather per minibatch to change nothing."""
    anchors = np.zeros((5, 2), dtype=bool)
    anchors[1, 0] = True
    assert anchor_row_weights(anchors, np.ones((5, 2)), 1.0) is None
    assert anchor_row_weights(anchors, np.ones((5, 2)), 0.5) is None
    assert anchor_row_weights(np.zeros((5, 2), bool), np.ones((5, 2)), 64.0) is None, \
        "no anchors ⇒ nothing to weigh"


# ──────────────────────────────────────────────────────────────────────────────────────────────
# THE LOSS
# ──────────────────────────────────────────────────────────────────────────────────────────────

def _loss_inputs(n=8):
    g = torch.Generator().manual_seed(7)
    logits = torch.randn(n, 1, generator=g)
    target = (torch.rand(n, 1, generator=g) > 0.5).float()
    mask = torch.ones(n, 1)
    return logits, target, mask


def test_the_DEFAULT_is_BIT_identical_on_the_loss_TENSOR():
    """`torch.equal`, not `approx`: an OFF run must be the same experiment as the pre-flag tree,
    and a 1e-8 difference compounds over 10M steps into a different run."""
    logits, target, mask = _loss_inputs()
    base, _ = ValueTerms._win_prob_loss(logits, target, mask)
    off, _ = ValueTerms._win_prob_loss(logits, target, mask, None, None, None, None)
    assert torch.equal(base, off)


def test_weight_k_multiplies_EXACTLY_the_anchored_rows():
    """Hand-computed against the masked-mean expression: the weighted loss must be
    `sum(per * row_w) / n_known` with `row_w` the normalised two-value vector and NOTHING else —
    in particular the denominator stays `n_known`, not the weighted count."""
    logits, target, mask = _loss_inputs(8)
    anchors = np.zeros((8, 1), dtype=bool)
    anchors[2, 0] = anchors[5, 0] = True
    row_w = anchor_row_weights(anchors, np.ones((8, 1)), 16.0)
    w = torch.as_tensor(row_w.reshape(8, 1))
    got, _ = ValueTerms._win_prob_loss(logits, target, mask, None, None, None, w)
    per = torch.nn.functional.binary_cross_entropy_with_logits(
        logits.reshape(-1), target.reshape(-1), reduction="none")
    want = (per * w.reshape(-1)).sum() / 8.0
    assert float(got) == pytest.approx(float(want), rel=1e-6)
    # ...and the two anchored rows are the only ones whose contribution moved.
    ratio = (w.reshape(-1) / w.reshape(-1)[0]).numpy()
    assert ratio[2] == pytest.approx(16.0) and ratio[5] == pytest.approx(16.0)
    assert np.allclose(np.delete(ratio, [2, 5]), 1.0)


def test_it_MULTIPLIES_with_the_strata_weight_rather_than_replacing_it():
    """The two flags price different axes of the same mix. An overwrite would silently disable
    whichever was applied second — and the run would still log both families."""
    logits, target, mask = _loss_inputs(8)
    opp = torch.tensor([[0], [0], [1], [1], [0], [1], [0], [1]])
    strata = torch.tensor([0.5, 2.0, 1.0, 1.0])
    anchors = np.zeros((8, 1), dtype=bool)
    anchors[3, 0] = True
    row_w = torch.as_tensor(anchor_row_weights(anchors, np.ones((8, 1)), 8.0).reshape(8, 1))
    both, m = ValueTerms._win_prob_loss(logits, target, mask, None, strata, opp, row_w)
    per = torch.nn.functional.binary_cross_entropy_with_logits(
        logits.reshape(-1), target.reshape(-1), reduction="none")
    composed = strata[opp.reshape(-1)] * row_w.reshape(-1)
    assert float(both) == pytest.approx(float((per * composed).sum() / 8.0), rel=1e-6)
    # Neither factor alone reproduces it — the gate against an overwrite in either direction.
    only_s, _ = ValueTerms._win_prob_loss(logits, target, mask, None, strata, opp, None)
    only_r, _ = ValueTerms._win_prob_loss(logits, target, mask, None, None, None, row_w)
    assert not torch.equal(both, only_s) and not torch.equal(both, only_r)
    assert m["row_w_mean"] == pytest.approx(float(composed.mean()), rel=1e-6)


def test_the_unweighted_loss_is_published_beside_the_weighted_one():
    """A reader must be able to separate 'the weights moved the loss' from 'the head got better'
    without a second run."""
    logits, target, mask = _loss_inputs(8)
    anchors = np.zeros((8, 1), dtype=bool)
    anchors[1, 0] = True
    row_w = torch.as_tensor(anchor_row_weights(anchors, np.ones((8, 1)), 32.0).reshape(8, 1))
    _, m = ValueTerms._win_prob_loss(logits, target, mask, None, None, None, row_w)
    base, _ = ValueTerms._win_prob_loss(logits, target, mask)
    assert m["loss_unweighted"] == pytest.approx(float(base), rel=1e-6)
    assert "row_w_mean" in m and "strata_row_w_mean" not in m, \
        "the strata series must not appear on a run with no strata weight"


# ──────────────────────────────────────────────────────────────────────────────────────────────
# λ COMPOSITION — weight the ANCHORS, measure the propagation
# ──────────────────────────────────────────────────────────────────────────────────────────────

def test_the_anchor_SHARE_decays_by_lambda_per_step_and_stops_at_an_episode_boundary():
    """`share[t] = λ · share[t+1]`, 1.0 at the anchor, 0.0 on a row whose target IS the terminal
    bit. It is the exact λ^k weight the anchor's information enters an earlier row's target at, and
    it is computed by the SAME backward pass as the target so the two cannot disagree."""
    n = 6
    values = np.full((n, 1), 0.5)
    y = np.ones((n, 1))
    mask = np.ones((n, 1))
    starts = np.zeros((n, 1))
    starts[0, 0] = 1.0
    anchors = np.zeros((n, 1), dtype=bool)
    anchors[4, 0] = True
    share = np.zeros((n, 1))
    lambda_return_targets(values, y, mask, starts, np.zeros(1), np.ones(1), 0.9, "bootstrap",
                          anchor_mask=anchors, anchor_value=np.full((n, 1), 0.25),
                          anchor_share_out=share)
    assert share[4, 0] == pytest.approx(1.0)
    assert share[3, 0] == pytest.approx(0.9)
    assert share[2, 0] == pytest.approx(0.81)
    assert share[5, 0] == pytest.approx(0.0), "row 5 ENDS the episode: its target is the bit"


def test_the_share_is_NOT_computed_when_nobody_asked_and_the_targets_are_unchanged():
    """The out-parameter exists so the recursion stays the one place that defines 'ends'. Passing
    it must change nothing about what the recursion returns."""
    n, kw = 5, dict(anchor_mask=np.zeros((5, 1), bool), anchor_value=np.zeros((5, 1)))
    args = (np.full((n, 1), 0.4), np.ones((n, 1)), np.ones((n, 1)), np.zeros((n, 1)),
            np.zeros(1), np.ones(1), 0.8, "bootstrap")
    a = lambda_return_targets(*args, **kw)
    out = np.zeros((n, 1))
    b = lambda_return_targets(*args, **kw, anchor_share_out=out)
    for x, y_ in zip(a, b):
        assert np.array_equal(np.asarray(x), np.asarray(y_))


def test_ONLY_the_anchor_rows_are_WEIGHTED_while_the_propagated_rows_are_MEASURED():
    """🚨 The composition rule. Under λ < 1 the rows before an anchor blend toward it, but their
    target is a MIXTURE of the anchor, the network's own later values and the copied bit — dosing
    them would dose arm 8's channel under arm 10's flag, and would make the delivered dose a
    function of the episode-length distribution. So the weight vector is flat off the anchors, and
    the λ^k reach shows up in `rollout_influence_lambda` instead."""
    n = 8
    anchors = np.zeros((n, 1), dtype=bool)
    anchors[5, 0] = True
    scored = np.ones((n, 1))
    w = anchor_row_weights(anchors, scored, 20.0)
    off_anchor = np.delete(w.reshape(-1), 5)
    assert np.allclose(off_anchor, off_anchor[0]), "a propagated row must not be up-weighted"
    share = np.zeros((n, 1))
    share[5, 0] = 1.0
    for t in range(4, -1, -1):
        share[t, 0] = 0.9 * share[t + 1, 0]
    m = weight_metrics(anchor_mask=anchors, new_mask=scored, row_w=w, weight=20.0,
                       anchor_share=share)
    assert m["rollout_influence_lambda"] > m["rollout_mass_weighted"], \
        "λ propagation must show up in the INFLUENCE meter"
    # ...and it is the weighted λ^k sum, not a row COUNT: the row one step before the anchor
    # counts 0.9, not 1.0.
    denom = float((w * scored).sum())
    want = float((w.reshape(-1) * np.where(anchors.reshape(-1), 1.0, share.reshape(-1))).sum())
    assert m["rollout_influence_lambda"] == pytest.approx(want / denom)


def test_with_lambda_OFF_the_influence_meter_EQUALS_the_weighted_mass():
    """There is no propagation to measure, and the tag is published anyway: an absent
    `rollout_influence_lambda` must mean 'the weight is off' and nothing else."""
    anchors = np.zeros((6, 1), dtype=bool)
    anchors[2, 0] = True
    w = anchor_row_weights(anchors, np.ones((6, 1)), 12.0)
    m = weight_metrics(anchor_mask=anchors, new_mask=np.ones((6, 1)), row_w=w, weight=12.0,
                       anchor_share=None)
    assert m["rollout_influence_lambda"] == pytest.approx(m["rollout_mass_weighted"])


# ──────────────────────────────────────────────────────────────────────────────────────────────
# THE CALLBACK, end to end
# ──────────────────────────────────────────────────────────────────────────────────────────────

def test_the_callback_WRITES_the_key_and_PUBLISHES_the_three_dose_meters(monkeypatch, tmp_path):
    model, _wt, wm, _c = _run(monkeypatch, tmp_path, win_prob_rollout_target=1.0,
                              win_prob_rollout_weight=64.0)
    row_w = model.rollout_buffer.observations[ROLLOUT_WEIGHT_KEY][:, :, 0]
    scored = wm[:, :, 0] >= 0.5
    assert float((row_w * scored).sum() / scored.sum()) == pytest.approx(1.0, abs=1e-5)
    assert row_w.max() > 1.0, "some row must have been up-weighted"
    m = model._win_prob_rollout_metrics
    for k in ("rollout_weight", "rollout_mass_weighted", "rollout_influence_lambda"):
        assert k in m, k
    assert m["rollout_weight"] == 64.0
    assert m["rollout_mass_weighted"] > m["rollout_mass"], \
        "the whole point: the weighted share exceeds the row share"


def test_an_UNFLAGGED_run_leaves_the_buffer_and_the_metrics_untouched(monkeypatch, tmp_path):
    model, _wt, _wm, _c = _run(monkeypatch, tmp_path, win_prob_rollout_target=1.0)
    assert ROLLOUT_WEIGHT_KEY not in model.rollout_buffer.observations
    assert "rollout_weight" not in model._win_prob_rollout_metrics


def test_the_flag_ON_with_the_key_ABSENT_is_a_SKIP_not_a_crash(monkeypatch, tmp_path):
    """A config mismatch (an env that did not declare the key) must not kill a run mid-rollout —
    the same rule `_on_rollout_end` already keeps for `win_target`."""
    model, _wt, _wm, _c = _run(monkeypatch, tmp_path, win_prob_rollout_target=1.0,
                               win_prob_rollout_weight=64.0)
    del model.rollout_buffer.observations[ROLLOUT_WEIGHT_KEY]
    model._win_terminal_scratch[-1, :] = 1.0
    model._win_prob_rollout_metrics = None
    model._win_prob_lambda_metrics = None
    from agents.training.win_prob_callback import WinProbLabelCallback
    cb = WinProbLabelCallback(records_dir=str(tmp_path))
    cb.model = model
    cb._on_rollout_end()                                  # must not raise
    assert "rollout_weight" not in (model._win_prob_rollout_metrics or {})


def test_the_env_declares_the_key_under_EXACTLY_the_predicate_the_loss_reads_it_under():
    """Three files have to agree on when this key exists — the env that emits it, the factory that
    turns the flags into the emit switch, and `train()` that reads it. A disagreement is either a
    KeyError mid-run or a silently unweighted arm."""
    import main.train.env_factory as ef
    import agents.training.instrumented_ppo.ppo as ppo_mod
    fac = inspect.getsource(ef.create_training_env_random)
    assert "emit_win_row_weight=(" in fac
    assert 'win_prob_rollout_weight", 1.0) or 1.0) > 1.0' in fac
    assert 'win_prob_rollout_target", 0.0) or 0.0) > 0.0' in fac
    tr = inspect.getsource(ppo_mod.InstrumentedMaskablePPO.train)
    assert "rollout_weight_on = (" in tr
    assert 'rollout_data.observations.get("win_row_w") if rollout_weight_on else None' in tr


# ──────────────────────────────────────────────────────────────────────────────────────────────
# THE RECORD: the refusal, model_config + the migration
# ──────────────────────────────────────────────────────────────────────────────────────────────

def test_the_weight_REFUSES_without_a_fraction_to_weigh():
    """With no anchors the weight is a vector of ones: the run is the unflagged one, under a
    flagged name, in a ledger line that says otherwise."""
    import argparse

    from main.train.combination_checks import COMBINATION_CHECKS
    check = next(c for c in COMBINATION_CHECKS
                 if c.name == "winprob_rollout_weight_needs_the_rollout_target")

    def _ns(**kw):
        return argparse.Namespace(**kw)

    assert check.predicate(_ns(win_prob_rollout_weight=64.0, win_prob_rollout_target=0.0))
    # ...and an UNSET fraction is the same refusal: `checkargs` hands a namespace of Nones, and
    # reading a None as "the flag is on" is the false positive `_val` exists to prevent.
    assert check.predicate(_ns(win_prob_rollout_weight=64.0, win_prob_rollout_target=None))
    assert not check.predicate(_ns(win_prob_rollout_weight=64.0, win_prob_rollout_target=0.0012))
    assert not check.predicate(_ns(win_prob_rollout_weight=1.0, win_prob_rollout_target=0.0))
    assert not check.predicate(_ns(win_prob_rollout_weight=None, win_prob_rollout_target=None))


def test_a_weight_BELOW_one_is_refused_by_the_range_check():
    """Refused rather than clamped: below 1 is a coherent-looking instruction for the OPPOSITE of
    this lever — the rows that cost thousands of continuations counting for LESS than the copied
    bits they were bought to replace."""
    import main.train.config as cfg
    src = inspect.getsource(cfg.resolve_config)
    assert "args.win_prob_rollout_weight < 1.0" in src
    assert "--win-prob-rollout-weight must be >= 1.0" in src


def test_BOTH_ModelVersion_sites_carry_the_weight_at_config_119():
    """The FIELD (what a flagless resume reads back) and the CONSTRUCTION (what a save writes) are
    two files; a field declared in one but not threaded through the other records a default
    forever, silently."""
    import dataclasses

    from agents.model.model_version import MODEL_CONFIG_VERSION, ModelVersionFields
    from agents.model.model_version.construct import ModelVersionConstruction

    assert MODEL_CONFIG_VERSION >= 119
    defaults = {f.name: f.default for f in dataclasses.fields(ModelVersionFields)}
    assert defaults["win_prob_rollout_weight"] == ROLLOUT_WEIGHT_OFF
    sig = inspect.signature(ModelVersionConstruction.from_layout_and_policy_kwargs)
    src = inspect.getsource(ModelVersionConstruction.from_layout_and_policy_kwargs)
    assert "win_prob_rollout_weight" in sig.parameters
    assert "win_prob_rollout_weight=" in src


def test_a_pre_v119_config_MIGRATES_to_an_unweighted_objective():
    """1.0 is a RECORD, not a chosen default: every scored row of every pre-v119 run weighed the
    same, because there was no flag that could make one weigh more."""
    out = _migrate_config({"config_version": 118})
    assert out["win_prob_rollout_weight"] == 1.0
    assert out["config_version"] >= 119


def test_the_weight_is_INHERITED_on_a_flagless_resume():
    import main.train.config as cfg
    src = inspect.getsource(cfg.resolve_config)
    assert '_resolve("win_prob_rollout_weight"' in src, (
        "a launcher restart that dropped the weight would keep paying for every continuation "
        "while delivering ~1/50th of the registered dose — the arm's cost with none of its "
        "treatment, and nothing in the cost profile would show it")


def test_it_is_RECORDED_in_the_run_hparams_and_the_roundtrip_smoke():
    import main.train.lifecycle as lc
    import main.train.run_io as rio
    assert "win_prob_rollout_weight" in inspect.getsource(rio._model_hparams)
    assert "win_prob_rollout_weight" in inspect.getsource(lc._run_roundtrip_test)


def test_the_weight_is_declared_in_the_coefficient_module_table():
    from agents.model.arch_tables import _COEF_MODULE
    assert _COEF_MODULE["win_prob_rollout_weight"] == "win_head"


def test_the_sidecar_does_NOT_treat_the_weight_as_a_target_identity_field():
    """🚨 The rule for QUANTITY_FIELDS is 'does it change what the `target` COLUMN HOLDS?', not 'is
    it a new flag'. This one changes how much a row COUNTS IN THE LOSS and no row's target, so
    adding it would refuse to pool two files that are genuinely poolable — the exact false alarm
    SCHEMA_EQUIVALENCE exists to name."""
    from agents.training.value_sidecar import QUANTITY_FIELDS, TARGET_IDENTITY_FIELDS
    assert "win_prob_rollout_weight" not in QUANTITY_FIELDS
    assert "win_prob_rollout_weight" not in TARGET_IDENTITY_FIELDS


def test_the_OFF_constant_is_one():
    assert ROLLOUT_WEIGHT_OFF == 1.0 and ROLLOUT_WEIGHT_KEY == "win_row_w"
