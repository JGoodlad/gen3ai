"""`tie_margins` (`gen3_behaviour_tie_exclusion_v1`): each declared MARGIN rule on known tensors (RELATIVE
margins, exact ties 0, structural zeros, a gated argmax, a top-k cutoff that is one of its own values),
and the recorder on the REAL production forward — every row gets a finite margin, an UNDECLARED site is
refused, and a site whose tensor is not row-major is refused rather than mis-attributed."""
from __future__ import annotations

import numpy as np
import pytest
import torch as th

from agents.model import selection_sites as SS
from agents.training.rust_rollout import tie_margins as T


def _m(rule, name, *args, **kw):
    g = T.site_margin(rule, name, args, kw)
    return None if g is None else g.reshape(g.shape[0], -1).amin(1).numpy()


def test_topk_margin_is_the_relative_gap_between_the_kth_and_the_next_and_an_exact_tie_is_zero():
    x = th.tensor([[0.9, 0.5, 0.5 * (1 - 3e-6), 0.1], [0.9, 0.8, 0.2, 0.1], [0.7, 0.3, 0.3, 0.0]], dtype=th.float64)
    g = _m(SS.Rule("topk"), "topk", x, 2, -1)
    assert g[0] == pytest.approx(3e-6, rel=1e-2) and g[1] == pytest.approx(0.75, rel=1e-6) and g[2] == 0.0
    assert _m(SS.Rule("topk"), "topk", x, 4, -1) is None          # keeps every candidate: nothing to cross


def test_a_gated_argmax_ignores_the_slots_its_gate_masks_and_judges_the_rest():
    x = th.tensor([[[0.0, 0.0, 0.0], [0.4, 0.4 * (1 - 1e-7), 0.1]],         # slot 0 masked (max 0), slot 1 near-tie
                   [[0.0, 0.0, 0.0], [0.4, 0.2, 0.1]]], dtype=th.float64)
    g = _m(SS.Rule("argmax", gate=1e-6), "argmax", x, -1)
    assert g[0] == pytest.approx(1e-7, rel=1e-2) and g[1] == pytest.approx(0.5)
    assert _m(SS.Rule("argmax"), "argmax", x, -1)[1] == 0.0          # ungated: the all-zero slot is an exact tie


def test_a_threshold_is_relative_and_a_structural_zero_is_not_a_tie_only_when_declared():
    x = th.tensor([[1e-6 * (1 + 2e-7), 0.5], [0.0, 0.3]], dtype=th.float64)
    g = _m(SS.Rule("threshold"), "__gt__", x, 1e-6)
    assert g[0] == pytest.approx(2e-7, rel=1e-2) and g[1] == pytest.approx(1.0, rel=1e-4)
    cur = th.tensor([[0.0, 120.0], [100.0, 0.0]])
    fixed = th.tensor([[0.0, 100.0], [100.0, 0.0]])
    assert _m(SS.Rule("threshold"), "__ge__", fixed, cur).tolist() == [0.0, 0.0]
    assert _m(SS.Rule("threshold", zero_exact=True), "__ge__", fixed, cur).tolist() == [pytest.approx(20 / 120), 0.0]


def test_a_threshold_at_its_own_kth_value_skips_that_value_unless_it_is_shared():
    w = th.tensor([[0.9, 0.6, 0.6 * (1 - 1e-6), 0.1], [0.9, 0.6, 0.6, 0.1], [0.9, 0.6, 0.3, 0.1]], dtype=th.float64)
    thr = th.tensor([[0.6], [0.6], [0.6]], dtype=th.float64)
    g = _m(SS.Rule("threshold_self"), "__ge__", w, thr)
    assert g[0] == pytest.approx(1e-6, rel=1e-2) and g[1] == 0.0 and g[2] == pytest.approx(1 / 3)      # 0.9 is the nearest other value


@pytest.fixture(scope="module")
def production():
    from stable_baselines3.common.utils import obs_as_tensor

    from agents.training import learner_golden as L

    th.manual_seed(0)
    model = L.build_learner()
    L.load_buffer_into(model)
    rb = model.rollout_buffer
    obs = obs_as_tensor({k: v.reshape((-1,) + v.shape[2:]) for k, v in rb.observations.items()}, "cpu")
    return model, obs, rb.actions.reshape(-1), rb.action_masks.reshape(rb.actions.size, -1)


def test_the_recorder_gives_every_production_row_a_finite_margin_and_leaves_the_forward_unchanged(production):
    model, obs, acts, masks = production
    model.policy.set_training_mode(True)
    with th.no_grad():
        _v, ref, _e = model.policy.evaluate_actions(obs, th.as_tensor(acts).long(), action_masks=th.as_tensor(masks))
    rec = T.TieMargins(int(acts.size))
    with th.no_grad(), rec:
        _v, got, _e = model.policy.evaluate_actions(obs, th.as_tensor(acts).long(), action_masks=th.as_tensor(masks))
    rec.check()
    assert th.equal(ref, got), "the recorder must not change what the forward computes"
    assert np.isfinite(rec.margin).all() and all(rec.site)
    assert any("pointer_head" in s for s in rec.sites_seen) and any("damage_op.py" in s for s in rec.sites_seen)


def test_an_undeclared_discrete_op_in_the_forward_is_refused(production, monkeypatch):
    model, obs, acts, masks = production
    key = ("damage_op", "wfc.argmax(dim=-1, keepdim=True)")
    SS.line_map.cache_clear()
    monkeypatch.delitem(SS.MARGIN, key)
    try:
        with pytest.raises(T.TieMarginError, match=r"damage_op\.py:\d+ argmax"):
            T.selection_gaps(model.policy, obs, acts, masks, "cpu")
    finally:
        monkeypatch.undo()
        SS.line_map.cache_clear()
    g, _s = T.selection_gaps(model.policy, obs, acts, masks, "cpu")          # declared again: clean
    assert np.isfinite(g).all()


def test_a_margin_site_whose_tensor_is_not_row_major_is_refused(production):
    model, obs, acts, masks = production
    rec = T.TieMargins(int(acts.size) + 1)                               # rows != the forward's batch
    with pytest.raises(T.TieMarginError, match="not row-major"):
        with th.no_grad(), rec:
            model.policy.evaluate_actions(obs, th.as_tensor(acts).long(), action_masks=th.as_tensor(masks))
