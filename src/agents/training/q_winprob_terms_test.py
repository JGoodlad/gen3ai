"""The PER-ACTION win-prob FOLD (v107, `gen3_q_winprob_head_v1`) — the loss math and the label seam.

Three things are pinned here, and each one is a way the term could be wrong while every published
number looked healthy:

1. **The masked likelihood is a strict GENERALIZATION of the scalar one.** At full coverage it must
   equal `cf_terms.cf_binomial_nll` EXACTLY. Without that, "the Q head's loss is the same
   likelihood, restricted" is an analogy rather than a fact, and the two coefficients stop being
   comparable.
2. **An UNLABELLED cell contributes nothing — not a zero target.** A zero-filled absent label is
   indistinguishable from a confident "this action loses", which is the most dangerous silent
   target this schema could produce. It has to be checked as an invariance (the loss must not move
   when an unlabelled cell's logit swings), not by reading the code.
3. **The wire format's ACTION INDEX survives the trip to the tensor.** The buffer parses a list of
   self-describing objects precisely so a per-action label cannot land in a neighbour's column, and
   the scatter that builds the [B, A] matrices is where that guarantee is either kept or lost.
"""
from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from agents.action.constants import ACTION_SPACE_SIZE
from agents.training.cf_label_buffer import CfLabelBuffer, batch_tensors
from agents.training.cf_terms import cf_binomial_nll
from agents.training.q_winprob_terms import q_masked_binomial_nll


# ── the loss math ─────────────────────────────────────────────────────────────

def test_full_coverage_reduces_EXACTLY_to_the_scalar_binomial_nll():
    """The generalization claim, as an equality rather than a resemblance. Flattening a fully
    labelled [B, A] grid is the same set of observations the scalar term folds over one column."""
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(6, ACTION_SPACE_SIZE, generator=g) * 3.0
    labels = torch.rand(6, ACTION_SPACE_SIZE, generator=g)
    n = torch.randint(1, 17, (6, ACTION_SPACE_SIZE), generator=g).float()
    mask = torch.ones_like(labels)
    got = q_masked_binomial_nll(logits, labels, n, mask)
    want = cf_binomial_nll(logits.flatten(), labels.flatten(), n.flatten())
    assert torch.allclose(got, want, atol=1e-6), f"{float(got)} != {float(want)}"


def test_an_UNLABELLED_cell_contributes_nothing_at_any_logit():
    """The masked-loss invariance. Swing an unlabelled cell's logit across ±50 and the loss must
    not move by a float — a masked-out target that still pulled would teach the head that every
    unlabelled action loses, silently and everywhere."""
    logits = torch.zeros(2, ACTION_SPACE_SIZE)
    labels = torch.zeros(2, ACTION_SPACE_SIZE)
    n = torch.zeros(2, ACTION_SPACE_SIZE)
    mask = torch.zeros(2, ACTION_SPACE_SIZE)
    labels[:, 3], n[:, 3], mask[:, 3] = 0.75, 8.0, 1.0
    base = float(q_masked_binomial_nll(logits, labels, n, mask))
    for value in (-50.0, -1.0, 1.0, 50.0):
        moved = logits.clone()
        moved[:, 7] = value                       # an UNLABELLED column
        assert float(q_masked_binomial_nll(moved, labels, n, mask)) == base


def test_a_masked_out_cell_takes_NO_GRADIENT():
    """The same invariance one level down: not merely 'the value does not move' but 'the gradient
    is exactly zero there'. A term that moved a masked column by 1e-12 per step would still, over
    a multi-day run, train the head on nothing."""
    logits = torch.zeros(2, ACTION_SPACE_SIZE, requires_grad=True)
    labels = torch.zeros(2, ACTION_SPACE_SIZE)
    n = torch.zeros(2, ACTION_SPACE_SIZE)
    mask = torch.zeros(2, ACTION_SPACE_SIZE)
    labels[:, 3], n[:, 3], mask[:, 3] = 0.75, 8.0, 1.0
    q_masked_binomial_nll(logits, labels, n, mask).backward()
    grad = logits.grad
    assert bool(grad[:, 3].abs().sum() > 0), "the LABELLED cell took no gradient"
    off = [c for c in range(ACTION_SPACE_SIZE) if c != 3]
    assert float(grad[:, off].abs().sum()) == 0.0


def test_evidence_weighting_is_the_LIKELIHOOD_not_an_emphasis_choice():
    """A 16-rollout label must pull exactly 4x a 4-rollout one — that is `n` times the
    per-observation cross-entropy, i.e. what the data's likelihood IS. The normalizer is Σ(mask·n),
    so equal-evidence cells give equal per-cell loss regardless of how many cells are labelled."""
    logits = torch.zeros(1, ACTION_SPACE_SIZE, requires_grad=True)
    labels = torch.zeros(1, ACTION_SPACE_SIZE)
    n = torch.zeros(1, ACTION_SPACE_SIZE)
    mask = torch.zeros(1, ACTION_SPACE_SIZE)
    # Two cells, SAME label and SAME logit, differing ONLY in evidence.
    labels[0, 0], n[0, 0], mask[0, 0] = 1.0, 4.0, 1.0
    labels[0, 5], n[0, 5], mask[0, 5] = 1.0, 16.0, 1.0
    q_masked_binomial_nll(logits, labels, n, mask).backward()
    g = logits.grad[0]
    assert float(g[5]) == pytest.approx(4.0 * float(g[0]), rel=1e-6), \
        "an R=16 label must pull exactly 4x an R=4 one — that IS the likelihood of the data"

    # Per-ROLLOUT normalization ⇒ the MEAN is independent of both R and label DENSITY, which is
    # what keeps the coefficient's meaning fixed across producers and across starving minibatches.
    one = torch.zeros(1, ACTION_SPACE_SIZE)
    lab = torch.zeros(1, ACTION_SPACE_SIZE)
    nn_ = torch.zeros(1, ACTION_SPACE_SIZE)
    mk = torch.zeros(1, ACTION_SPACE_SIZE)
    lab[0, 0], nn_[0, 0], mk[0, 0] = 1.0, 4.0, 1.0
    four = float(q_masked_binomial_nll(one, lab, nn_, mk))
    nn_[0, 0] = 16.0
    assert float(q_masked_binomial_nll(one, lab, nn_, mk)) == pytest.approx(four, abs=1e-6)
    lab[0, 5], nn_[0, 5], mk[0, 5] = 1.0, 16.0, 1.0
    assert float(q_masked_binomial_nll(one, lab, nn_, mk)) == pytest.approx(four, abs=1e-6)


# ── the label seam ────────────────────────────────────────────────────────────

def _row(tmp_path, obs, **extra):
    row = {"schema": 1, "kind": "mc_winprob", "battle": "b", "decision_idx": 0,
           "label": 0.5, "n_rollouts": 4, "policy_step": 0, "opponent": "x",
           "obs_inline": __import__("base64").b64encode(obs.tobytes()).decode(), **extra}
    (tmp_path / "labels_0.jsonl").write_text(json.dumps(row) + "\n")
    return CfLabelBuffer(tmp_path, obs_dim=obs.size)


def test_the_wire_formats_ACTION_INDEX_lands_in_its_own_column(tmp_path):
    """The order-mismatch guard, end to end: JSONL → `CfLabel.q_labels` → the [B, A] scatter. Two
    labels at NON-adjacent, out-of-order indices, so any positional zip would be visibly wrong."""
    obs = np.arange(9, dtype=np.float32)
    buf = _row(tmp_path, obs, q_labels=[{"action": 9, "label": 0.25, "n_rollouts": 8},
                                        {"action": 2, "label": 0.75, "n_rollouts": 4}])
    assert buf.poll(0) == 1
    batch = batch_tensors(buf.sample(1), device="cpu")
    assert batch.q_mask.shape == (1, ACTION_SPACE_SIZE)
    assert [int(i) for i in batch.q_mask[0].nonzero().flatten()] == [2, 9]
    assert float(batch.q_label[0, 2]) == pytest.approx(0.75)
    assert float(batch.q_label[0, 9]) == pytest.approx(0.25)
    assert float(batch.q_n[0, 2]) == 4.0 and float(batch.q_n[0, 9]) == 8.0
    assert float(batch.q_label[0, 0]) == 0.0 and float(batch.q_mask[0, 0]) == 0.0


@pytest.mark.parametrize("bad,reason", [
    ([{"action": 11, "label": 0.5, "n_rollouts": 1}], "q_labels_action_range"),
    ([{"action": -1, "label": 0.5, "n_rollouts": 1}], "q_labels_action_range"),
    ([{"action": 1, "label": 1.5, "n_rollouts": 1}], "q_labels_label_range"),
    ([{"action": 1, "label": float("nan"), "n_rollouts": 1}], "q_labels_label_range"),
    ([{"label": 0.5}], "q_labels_malformed"),
    ([[1, 0.5, 4]], "q_labels_entry_not_an_object"),
    ("nope", "q_labels_not_a_list"),
])
def test_a_malformed_per_action_entry_is_a_FIELD_skip_not_a_row_skip(tmp_path, bad, reason):
    """The row's three OTHER label streams are perfectly good; a producer bug in one must not cost
    the trainer the rest. `skipped_total` stays 0 — it is the GIGO meter for whole ROWS, and
    conflating a field rejection with it would make that meter climb at the ingestion rate."""
    obs = np.arange(9, dtype=np.float32)
    buf = _row(tmp_path, obs, q_labels=bad)
    assert buf.poll(0) == 1
    assert buf.skipped_total == 0 and buf.field_skipped_total >= 1
    assert buf.skip_reasons.get(reason, 0) >= 1
    assert buf.sample(1)[0].q_labels == ()


def test_duplicate_actions_collapse_keep_LAST(tmp_path):
    """Two entries for one action are the producer contradicting itself. Summing or averaging them
    would invent evidence no rollout supports; keep-last mirrors the buffer's own keep-newest."""
    obs = np.arange(9, dtype=np.float32)
    buf = _row(tmp_path, obs, q_labels=[{"action": 3, "label": 0.1, "n_rollouts": 2},
                                        {"action": 3, "label": 0.9, "n_rollouts": 6}])
    buf.poll(0)
    assert buf.sample(1)[0].q_labels == ((3, 0.9, 6),)


def test_the_on_policy_fallback_needs_BOTH_halves(tmp_path):
    """`taken_action` alone is an index with no outcome; `outcome_label` alone is an outcome with no
    action. Either without the other is NOT a label at index 0 — which is what a naive mask would
    make it, and is a confident wrong target on whatever action happens to sit there."""
    obs = np.arange(9, dtype=np.float32)
    for extra in ({"taken_action": 4}, {"outcome_label": 1.0}):
        buf = _row(tmp_path, obs, **extra)
        buf.poll(0)
        assert float(batch_tensors(buf.sample(1), device="cpu").taken_mask[0]) == 0.0
    buf = _row(tmp_path, obs, taken_action=4, outcome_label=1.0)
    buf.poll(0)
    batch = batch_tensors(buf.sample(1), device="cpu")
    assert float(batch.taken_mask[0]) == 1.0 and int(batch.taken_action[0]) == 4


def test_an_older_producers_row_supervises_the_Q_head_on_NOTHING(tmp_path):
    """Backward compatibility in the direction that matters: a row with no `q_labels` and no
    `taken_action` is ingested normally and simply carries an all-zero mask — the additive-optional
    contract the schema version deliberately does not move for."""
    obs = np.arange(9, dtype=np.float32)
    buf = _row(tmp_path, obs)
    assert buf.poll(0) == 1 and buf.field_skipped_total == 0
    batch = batch_tensors(buf.sample(1), device="cpu")
    assert float(batch.q_mask.sum()) == 0.0 and float(batch.taken_mask.sum()) == 0.0
    assert float(batch.label[0]) == pytest.approx(0.5)          # the per-state stream is untouched


# ── the fold's read seam ──────────────────────────────────────────────────────

class _Stand:
    """The smallest thing the two folds actually need: a features_extractor exposing the head and
    the pointer stash, plus the two coefficients. Deliberately NOT a real PPO — the point of these
    tests is the SEAM (which stash is read, what happens when it is absent or stale), and a real
    policy would hide a wrong read behind a plausible number."""

    class _Fe:
        pass

    class _Policy:
        pass

    def __init__(self, batch, *, pooled_dim=16, n_rows=None, stale=False):
        from agents.model.q_winprob_head import QWinProbHead
        torch.manual_seed(5)
        self.q_winprob_coef = 1.0
        self.q_winprob_onpolicy_coef = 1.0
        B = batch.obs.shape[0]
        self.policy = self._Policy()
        fe = self.policy.features_extractor = self._Fe()
        fe.q_winprob_head = QWinProbHead(move_token_dim=pooled_dim, d_model=pooled_dim,
                                         ctx_dim=pooled_dim, move_cell_dim=2, switch_cell_dim=2)
        # Break the zero-init: a head whose every logit is 0 makes every metric trivially equal and
        # would let a wrong read pass.
        torch.nn.init.normal_(fe.q_winprob_head.q_score.weight, std=0.3)
        pb = B + 1 if stale else B
        fe.last_pointer_inputs = _PointerStub(pb, pooled_dim)
        self._pooled = torch.randn(B, pooled_dim, requires_grad=True)
        self.n_rows = B if n_rows is None else n_rows
        self.batch = batch

    def ctx(self):
        from agents.training.cf_terms import CfForward
        return CfForward(batch=self.batch, value_pooled=self._pooled,
                         n_rows=self.n_rows, vf_features=None)


class _PointerStub:
    def __init__(self, B, d):
        self.move_tokens = torch.randn(B, 4, d, requires_grad=True)
        self.move_valid = torch.ones(B, 4)
        self.team_tokens = torch.randn(B, 6, d, requires_grad=True)
        self.move_cells = torch.randn(B, 4, 2)
        self.switch_cells = torch.randn(B, 6, 2)


def _batch(tmp_path, **extra):
    obs = np.arange(9, dtype=np.float32)
    buf = _row(tmp_path, obs, **extra)
    buf.poll(0)
    return batch_tensors(buf.sample(1), device="cpu")


def test_the_fold_reads_the_pointer_stash_and_trains_ONLY_the_head(tmp_path):
    """Two claims that fail silently in opposite directions. If the fold read the wrong tensors the
    loss would still be a number; if it let a gradient through, the readout would be shaping the
    policy while every doc said it could not."""
    from agents.training.q_winprob_terms import q_winprob_term

    batch = _batch(tmp_path, q_labels=[{"action": 2, "label": 0.9, "n_rollouts": 8},
                                       {"action": 6, "label": 0.1, "n_rollouts": 8}])
    model = _Stand(batch)
    term, metrics = q_winprob_term(model, model.ctx())
    assert term is not None
    assert metrics["labels_per_row"] == pytest.approx(2.0)
    assert metrics["label_coverage"] == pytest.approx(1.0)
    assert metrics["n_rollouts_mean"] == pytest.approx(8.0)
    # The spread pair — the column that distinguishes "amortized the VALUE" from "amortized the
    # SEARCH". The labels differ by 0.8, so `label_spread` must reflect that, not 0.
    assert metrics["label_spread"] == pytest.approx(0.8, abs=1e-6)
    assert "pred_spread" in metrics

    term.backward()
    fe = model.policy.features_extractor
    assert any(p.grad is not None and bool(p.grad.abs().sum() > 0)
               for p in fe.q_winprob_head.parameters()), "the head trained on nothing"
    assert model._pooled.grad is None, "the fold leaked a gradient into the trunk summary"
    assert fe.last_pointer_inputs.team_tokens.grad is None, \
        "the fold leaked a gradient into the pointer tokens"


def test_no_per_action_label_is_a_PUBLISHED_zero_coverage_not_silence(tmp_path):
    """The starvation case must not look like a healthy head with nothing to say — the oldest
    failure mode in this tree (the search teacher's silent starvation) and the reason
    `cf_shadow_term` publishes the same shape."""
    from agents.training.q_winprob_terms import q_winprob_term

    model = _Stand(_batch(tmp_path))            # an older producer's row: no q_labels at all
    term, metrics = q_winprob_term(model, model.ctx())
    assert term is None
    assert metrics["label_coverage"] == 0.0 and metrics["labels_per_row"] == 0.0
    assert "loss" not in metrics, "a starved fold must publish NO loss — 0.0 reads as a perfect fit"


def test_the_on_policy_fallback_supervises_EXACTLY_the_taken_column(tmp_path):
    """Its whole hazard is teaching the head where the policy already goes, so the one thing that
    must be exactly right is WHICH column it touches."""
    from agents.training.q_winprob_terms import q_winprob_onpolicy_term

    batch = _batch(tmp_path, taken_action=8, outcome_label=1.0)
    model = _Stand(batch)
    term, metrics = q_winprob_onpolicy_term(model, model.ctx())
    assert term is not None and metrics["onpolicy_coverage"] == pytest.approx(1.0)
    term.backward()
    grads = model.policy.features_extractor.last_pointer_inputs.move_tokens.grad
    assert grads is None                        # head-only, same as the counterfactual fold

    # The label the head was pulled toward is the RECORDED outcome (1.0), so its prediction error
    # is signed the one way that could not happen if column 8 were not the supervised one.
    assert metrics["onpolicy_label_mean"] == pytest.approx(1.0)
    assert metrics["onpolicy_bias"] < 0.0 or metrics["onpolicy_pred_mean"] < 1.0


def test_a_STALE_pointer_stash_raises_instead_of_supervising_the_wrong_board(tmp_path):
    """Structurally impossible (one forward writes both stashes), which is precisely why it must
    fail LOUD rather than degrade: a silent mismatch would train the head on one batch's actions
    and another batch's board — the `_critic_value` stale-stash precedent."""
    from agents.training.q_winprob_terms import q_winprob_term

    batch = _batch(tmp_path, q_labels=[{"action": 1, "label": 0.5, "n_rollouts": 2}])
    model = _Stand(batch, stale=True)
    with pytest.raises(RuntimeError, match="stale pointer stash"):
        q_winprob_term(model, model.ctx())


def test_no_head_built_means_no_term_and_no_crash(tmp_path):
    """`--q-winprob-mode none` with a live coefficient is refused at the CLI, but the fold must
    still be a no-op rather than an AttributeError — the belt-and-braces `cf_winprob_term` keeps
    for the same reason."""
    from agents.training.q_winprob_terms import q_winprob_onpolicy_term, q_winprob_term

    model = _Stand(_batch(tmp_path, q_labels=[{"action": 1, "label": 0.5, "n_rollouts": 2}]))
    model.policy.features_extractor.q_winprob_head = None
    assert q_winprob_term(model, model.ctx()) == (None, {})
    assert q_winprob_onpolicy_term(model, model.ctx()) == (None, {})
    assert q_winprob_term(model, None) == (None, {})


def test_the_coverage_counters_separate_RUNNING_from_running_at_one_action(tmp_path):
    """`q_labels_per_row` is the number that distinguishes a live counterfactual factory from an
    on-policy trickle, and it is the first thing an operator is told to read. One row with two
    labelled actions ⇒ coverage 1.0, per-row 2.0."""
    obs = np.arange(9, dtype=np.float32)
    buf = _row(tmp_path, obs, q_labels=[{"action": 0, "label": 0.2, "n_rollouts": 4},
                                        {"action": 7, "label": 0.8, "n_rollouts": 4}])
    buf.poll(0)
    stats = buf.stats(0)
    assert stats["cf/q_label_coverage"] == pytest.approx(1.0)
    assert stats["cf/q_labels_per_row"] == pytest.approx(2.0)
