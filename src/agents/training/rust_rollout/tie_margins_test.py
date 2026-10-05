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


# ---------------------------------------------- gen3_behaviour_tie_identity_v1: an argmax's PAYLOAD identity
_PAYLOAD_RULE = SS.Rule("argmax", gate=1e-6, payload=("p",))


def _pm(x, payload):
    out = x.argmax(dim=-1, keepdim=True)
    g = T.site_margin(_PAYLOAD_RULE, "argmax", (x,), {"dim": -1, "keepdim": True}, payload, out)
    return g.reshape(g.shape[0], -1).amin(1).numpy()


def test_an_argmax_tie_between_payload_identical_candidates_is_not_a_tie():
    """Two moves EXACTLY tied at the max that gather the same accuracy select the same value: the margin is
    the gap to the nearest candidate whose payload DIFFERS (here 0.4 vs 0.1), never 0. Without the
    payload (the rule before) the same row is an exact tie."""
    x = th.tensor([[[0.4, 0.1, 0.4, 0.05]]], dtype=th.float64)
    acc = th.tensor([[[1.0, 0.9, 1.0, 0.8]]], dtype=th.float64)
    assert _pm(x, (acc,))[0] == pytest.approx(0.75)
    assert _pm(x, None)[0] == 0.0
    # a near tie (1e-7 apart) with an identical payload is not a tie either
    xn = th.tensor([[[0.4, 0.1, 0.4 * (1 - 1e-7), 0.05]]], dtype=th.float64)
    assert _pm(xn, (acc,))[0] == pytest.approx(0.75)


def test_a_near_tie_between_distinct_candidates_IS_a_tie_and_every_payload_must_agree():
    xn = th.tensor([[[0.4, 0.1, 0.4 * (1 - 1e-7), 0.05]]], dtype=th.float64)
    acc = th.tensor([[[1.0, 0.9, 0.95, 0.8]]], dtype=th.float64)
    g = _pm(xn, (acc,))[0]
    assert g == pytest.approx(1e-7, rel=1e-3) and g < 2e-4
    same = th.tensor([[[1.0, 0.9, 1.0, 0.8]]], dtype=th.float64)
    w = th.tensor([[0.5, 0.2, 0.25, 0.1]], dtype=th.float64)       # a per-ROW payload, broadcast over the slot
    assert _pm(xn, (same, w[:, None, :]))[0] == pytest.approx(1e-7, rel=1e-3)
    nan = th.tensor([[[float("nan"), 0.9, float("nan"), 0.8]]], dtype=th.float64)
    assert _pm(xn, (nan,))[0] == 0.0       # NaN never equals itself — not even the selected one: a tie


def test_a_stale_payload_declaration_is_refused_not_read_as_no_payload():
    class _F:
        f_locals = {"q": th.zeros(2, 3)}

    x = th.zeros(2, 6, 3)
    with pytest.raises(T.TieMarginError, match="declares payload 'p'.*stale"):
        T.payload_tensors(_PAYLOAD_RULE, _F(), x, "damage_op.py:1 argmax")
    _F.f_locals = {"p": th.zeros(2, 4)}
    with pytest.raises(T.TieMarginError, match="does not align"):
        T.payload_tensors(_PAYLOAD_RULE, _F(), x, "damage_op.py:1 argmax")
    _F.f_locals = {"p": th.ones(2, 3)}
    (p,) = T.payload_tensors(_PAYLOAD_RULE, _F(), x, "damage_op.py:1 argmax")
    assert p.shape == (2, 1, 3)


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
