"""Gates for the opponent-intent METRIC fixes (`gen3_opp_class_v1`).

Each test pins a property that would be silently wrong if the metric were computed carelessly —
not a shape check. The motivating problem: `α`'s accuracy is read off an opponent pool spanning a
RANDOM bot, several heuristics and frozen selves, where "predict their move" is a different problem
per opponent. Accuracy cannot express that; a proper scoring rule can.
"""
import math

import pytest
import torch

from agents.model.opp_intent import (
    INTENT_IGNORE,
    OPP_CLASS_NAMES,
    info_gain_nats,
    intent_losses,
    switch_coverage_metrics,
)


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


# ------------------------------------------------------------------ stratification

def _alpha_case(n=64, k=3):
    torch.manual_seed(0)
    logits = torch.randn(n, k + 1)
    target = torch.randint(0, k + 1, (n,))
    return logits, target


def test_stratified_metrics_appear_per_present_class_only():
    logits, target = _alpha_case()
    opp_class = torch.zeros(logits.shape[0], dtype=torch.long)
    opp_class[32:] = 1                                   # only bot + pool present
    _, m = intent_losses(logits, target, None, None, opp_class=opp_class)
    assert "opp_intent/alpha_info_gain_nats_bot" in m
    assert "opp_intent/alpha_info_gain_nats_pool" in m
    assert "opp_intent/alpha_info_gain_nats_stable" not in m
    assert "opp_intent/alpha_info_gain_nats_exploiter" not in m


def test_stratified_counts_sum_to_the_pooled_supervised_count():
    """A row must land in exactly one bucket — no double-counting, no silent drops."""
    logits, target = _alpha_case(n=96)
    opp_class = torch.randint(0, 4, (96,))
    _, m = intent_losses(logits, target, None, None, opp_class=opp_class)
    per = sum(v for k, v in m.items() if k.startswith("opp_intent/alpha_n_supervised_"))
    assert per == pytest.approx(m["opp_intent/alpha_n_supervised"])


def test_stratification_actually_separates_a_planted_difference():
    """The gate that matters: an EASY class and a HARD class must not report the same number.

    A stratification that silently pooled (wrong index, wrong mask) would still emit both keys with
    both values equal — which is exactly the failure a presence check cannot see.
    """
    n = 200
    target = torch.cat([torch.zeros(n, dtype=torch.long),          # easy: always class 0
                        torch.randint(0, 4, (n,))])                # hard: uniform
    logits = torch.zeros(2 * n, 4)
    logits[:n, 0] = 8.0                                            # predicts the easy half exactly
    opp_class = torch.cat([torch.zeros(n, dtype=torch.long),
                           torch.ones(n, dtype=torch.long)])
    _, m = intent_losses(logits, target, None, None, opp_class=opp_class)
    easy = m["opp_intent/alpha_acc_bot"]
    hard = m["opp_intent/alpha_acc_pool"]
    assert easy > 0.95 and hard < 0.5, (easy, hard)
    # And the pooled number sits between them — i.e. it is the average that hides both.
    assert hard < m["opp_intent/alpha_acc"] < easy


def test_ignored_rows_are_excluded_from_every_stratum():
    logits, target = _alpha_case(n=40)
    target[:20] = INTENT_IGNORE
    opp_class = torch.zeros(40, dtype=torch.long)
    _, m = intent_losses(logits, target, None, None, opp_class=opp_class)
    assert m["opp_intent/alpha_n_supervised_bot"] == pytest.approx(20.0)


def test_opp_class_is_optional_and_changes_nothing_when_absent():
    """Off must be exactly the old behaviour — no stratified keys, identical pooled values."""
    logits, target = _alpha_case()
    _, without = intent_losses(logits, target, None, None)
    _, with_cls = intent_losses(logits, target, None, None,
                                opp_class=torch.zeros(logits.shape[0], dtype=torch.long))
    # Check for the CLASS SUFFIXES specifically. (This assertion used to test for "__" and became
    # vacuous the moment the separator changed to "_" — a test that cannot fail is not a test.)
    assert not any(k.endswith(tuple("_" + n for n in OPP_CLASS_NAMES.values())) for k in without)
    for k, v in without.items():
        assert with_cls[k] == pytest.approx(v), k


def test_the_loss_is_untouched_by_the_metric_change():
    """Metrics may not perturb training. Same inputs, same scalar, with and without stratification."""
    logits, target = _alpha_case()
    loss_a, _ = intent_losses(logits, target, None, None)
    loss_b, _ = intent_losses(logits, target, None, None,
                              opp_class=torch.randint(0, 4, (logits.shape[0],)))
    assert torch.equal(loss_a, loss_b)


def test_class_names_match_the_wrapper_constants():
    """The table is duplicated to keep `model/` from importing `training/` — pin them together."""
    from agents.training.wrappers import MaskableAgentWrapper as W
    assert OPP_CLASS_NAMES == {W.OPP_CLASS_BOT: "bot", W.OPP_CLASS_POOL: "pool",
                               W.OPP_CLASS_STABLE: "stable", W.OPP_CLASS_EXPLOITER: "exploiter"}


# ------------------------------------------------- the AXIS metrics carry the split too
# The split originally covered acc / info-gain / n only, which left the four axis metrics — the
# ones a reader uses to LOCATE a deficit — pooled over an opponent mix that is 100% bot early and
# ~7% bot later (measured, gen-11). Bot rows score differently, so a pooled axis metric drifts with
# the MIX and that drift is indistinguishable from the head improving.

#: Every pooled `α` metric that must now have a per-class twin. `loss`/`mask_rate` are deliberately
#: absent: the loss is one scalar for the batch, and the mask rate's denominator is all rows.
_ALPHA_AXIS = (
    "alpha_acc", "alpha_switch_recall", "alpha_switch_precision",
    "alpha_move_kind_recall", "alpha_move_kind_precision", "alpha_pred_switch_rate",
    "alpha_move_recall_top1", "alpha_move_recall_top2", "alpha_move_baseline_argmax_w",
    "alpha_switch_rate", "alpha_info_gain_nats", "alpha_info_gain_nats_move",
)


def test_every_pooled_alpha_axis_metric_has_a_per_class_twin():
    """The actual ask. A pooled axis metric with no stratified counterpart cannot be read."""
    n = 200
    torch.manual_seed(3)
    logits = torch.randn(n, 4)
    target = torch.randint(0, 4, (n,))
    opp_class = torch.zeros(n, dtype=torch.long)
    opp_class[n // 2:] = 1
    _, m = intent_losses(logits, target, None, None, opp_class=opp_class)
    missing = [f"{key}_{cls}" for key in _ALPHA_AXIS for cls in ("bot", "pool")
               if f"opp_intent/{key}" in m and f"opp_intent/{key}_{cls}" not in m]
    assert not missing, f"pooled-only, cannot be read per opponent: {missing}"


def test_a_planted_axis_difference_survives_the_split_but_is_hidden_when_pooled():
    """The gate that matters, on an AXIS metric rather than on accuracy.

    Bot rows: every switch predicted. Pool rows: no switch predicted. Switch recall must read 1.0
    and 0.0 per class — and the pooled number must sit between, which is the hiding it does.
    """
    n = 50
    target = torch.full((2 * n,), 3, dtype=torch.long)      # k=3 => class 3 is SWITCH, always
    logits = torch.zeros(2 * n, 4)
    logits[:n, 3] = 9.0                                     # bot half: says SWITCH
    logits[n:, 0] = 9.0                                     # pool half: says seat 0
    opp_class = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])
    _, m = intent_losses(logits, target, None, None, opp_class=opp_class)
    assert m["opp_intent/alpha_switch_recall_bot"] == pytest.approx(1.0)
    assert m["opp_intent/alpha_switch_recall_pool"] == pytest.approx(0.0)
    assert m["opp_intent/alpha_switch_recall"] == pytest.approx(0.5)


def test_beta_is_stratified_too():
    """Beta needs it MORE: its supervised rows are only the switches, so one class can dominate
    the subset even while being a minority of decisions."""
    n = 40
    target = torch.zeros(2 * n, dtype=torch.long)           # always slot 0
    logits = torch.zeros(2 * n, 6)
    logits[:n, 0] = 9.0                                     # bot half: right
    logits[n:, 4] = 9.0                                     # pool half: wrong
    opp_class = torch.cat([torch.zeros(n, dtype=torch.long), torch.ones(n, dtype=torch.long)])
    _, m = intent_losses(None, None, logits, target, opp_class=opp_class)
    assert m["opp_intent/beta_recall_top1_bot"] == pytest.approx(1.0)
    assert m["opp_intent/beta_recall_top1_pool"] == pytest.approx(0.0)
    assert m["opp_intent/beta_recall_top1"] == pytest.approx(0.5)
    assert m["opp_intent/beta_n_supervised_bot"] == pytest.approx(float(n))


def test_the_axis_metrics_still_mean_what_they_meant():
    """A hand-computed pin, so the extraction into `_alpha_subset_metrics` cannot have quietly
    changed a denominator. Every value below is derived on paper from the 5 rows, not recorded
    from a previous run — a golden captured from the code it guards proves only self-consistency.

    k=2 (seats 0,1 + SWITCH=2).  tgt = [SW, SW, 0, 1, 0],  pred = [SW, 0, 0, 0, 1]
    """
    logits = torch.tensor([[0., 0., 5.],      # pred SWITCH, tgt SWITCH   -> hit
                           [5., 0., 0.],      # pred seat0,  tgt SWITCH   -> missed switch
                           [5., 0., 0.],      # pred seat0,  tgt seat0    -> hit
                           [5., 0., 0.],      # pred seat0,  tgt seat1    -> wrong move
                           [0., 5., 0.]])     # pred seat1,  tgt seat0    -> wrong move
    target = torch.tensor([2, 2, 0, 1, 0])
    _, m = intent_losses(logits, target, None, None)
    g = lambda k: m[f"opp_intent/{k}"]                                      # noqa: E731
    assert g("alpha_acc") == pytest.approx(2 / 5)                # rows 0,2 correct
    assert g("alpha_switch_recall") == pytest.approx(1 / 2)      # of 2 switches, caught 1
    assert g("alpha_switch_precision") == pytest.approx(1.0)     # said switch once, was right
    assert g("alpha_move_kind_recall") == pytest.approx(1.0)     # all 3 moves called a move
    assert g("alpha_move_kind_precision") == pytest.approx(3 / 4)  # said move 4x, 3 truly moves
    assert g("alpha_pred_switch_rate") == pytest.approx(1 / 5)
    assert g("alpha_switch_rate") == pytest.approx(2 / 5)
    assert g("alpha_move_recall_top1") == pytest.approx(1 / 3)   # of 3 move rows, 1 right seat
    assert g("alpha_move_recall_top2") == pytest.approx(1.0)     # 2 seats, top-2 covers both
    assert g("alpha_move_baseline_argmax_w") == pytest.approx(2 / 3)  # 2 of 3 move tgts are seat 0


# ------------------------------------------------- the switch-coverage matrix
# Previously a closure inside `InstrumentedPPO.train()`, and therefore covered by NOTHING. It was
# lifted to module scope so these can exist: a metric with no test is a metric that can silently
# read zero, which is the exact failure `beta_wanted_content` and `beta_believed_targets` were
# added to catch in the first place.

def _cov_case():
    """8 rows: 5 voluntary switches (kind==1), 3 non-switches.
    Of the 5 switches: 2 revealed (need=False), 3 needed the belief, of which 2 were resolved."""
    kind = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0])
    need = torch.tensor([False, False, True, True, True, False, False, False])
    content = torch.tensor([-1, -1, 4, 2, -1, -1, -1, -1])      # rows 2,3 resolved; row 4 missed
    return kind, need, content


def test_the_three_buckets_partition_the_switches():
    """They are fractions of voluntary switches, so they must sum to exactly 1."""
    m = switch_coverage_metrics(*_cov_case())
    assert m["opp_intent/beta_switch_n"] == pytest.approx(5.0)
    assert m["opp_intent/beta_switch_to_revealed"] == pytest.approx(2 / 5)
    assert m["opp_intent/beta_switch_to_hidden_found"] == pytest.approx(2 / 5)
    assert m["opp_intent/beta_switch_to_hidden_missed"] == pytest.approx(1 / 5)
    assert sum(m[f"opp_intent/beta_switch_to_{b}"]
               for b in ("revealed", "hidden_found", "hidden_missed")) == pytest.approx(1.0)


def test_belief_miss_rate_is_over_the_rows_that_ASKED_not_over_all_switches():
    """3 switches needed the belief, 1 was missed -> 1/3, NOT 1/5. Using the wrong denominator
    understates the belief's failure by the share of switches it was never asked about."""
    m = switch_coverage_metrics(*_cov_case())
    assert m["opp_intent/beta_belief_miss_rate"] == pytest.approx(1 / 3)


def test_no_switches_emits_nothing_rather_than_a_zero():
    """A batch with no voluntary switch has no coverage to report. Emitting 0.0 would average into
    the dashboard as 'every switch went to a revealed mon', which is a claim, not an absence."""
    kind = torch.zeros(6, dtype=torch.long)
    assert switch_coverage_metrics(kind, None, None) == {}


def test_belief_head_off_reports_every_switch_as_revealed():
    """need/content are None when the belief head is off. That is the TRUTH (no row was
    content-addressed), not a fallback — and it must not divide by zero."""
    kind = torch.tensor([1, 1, 0])
    m = switch_coverage_metrics(kind, None, None)
    assert m["opp_intent/beta_switch_to_revealed"] == pytest.approx(1.0)
    assert m["opp_intent/beta_switch_to_hidden_found"] == pytest.approx(0.0)
    assert "opp_intent/beta_belief_miss_rate" not in m       # nothing asked => no rate to report


def test_the_matrix_is_stratified_and_the_slice_uses_only_its_own_rows():
    """The row mask must select before every counter, not after — a mask applied to the numerator
    only would leave `revealed` reading >1 or negative."""
    kind, need, content = _cov_case()
    rows = torch.tensor([True, True, False, False, False, True, True, True])   # the 2 revealed
    m = switch_coverage_metrics(kind, need, content, rows, "_bot")
    assert m["opp_intent/beta_switch_n_bot"] == pytest.approx(2.0)
    assert m["opp_intent/beta_switch_to_revealed_bot"] == pytest.approx(1.0)
    assert m["opp_intent/beta_switch_to_hidden_missed_bot"] == pytest.approx(0.0)
    assert "opp_intent/beta_belief_miss_rate_bot" not in m
    # and the complementary slice sees exactly the other three
    other = switch_coverage_metrics(kind, need, content, ~rows, "_pool")
    assert other["opp_intent/beta_switch_n_pool"] == pytest.approx(3.0)
    assert other["opp_intent/beta_belief_miss_rate_pool"] == pytest.approx(1 / 3)


def test_every_bucket_stays_a_fraction_under_any_random_slice():
    """A property sweep: whatever the mask, the three buckets are in [0,1] and sum to 1."""
    kind, need, content = _cov_case()
    torch.manual_seed(11)
    for _ in range(50):
        rows = torch.rand(8) < 0.6
        m = switch_coverage_metrics(kind, need, content, rows)
        if not m:
            continue
        vals = [m[f"opp_intent/beta_switch_to_{b}"]
                for b in ("revealed", "hidden_found", "hidden_missed")]
        assert all(-1e-9 <= v <= 1 + 1e-9 for v in vals), vals
        assert sum(vals) == pytest.approx(1.0)
