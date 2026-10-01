"""The ride-along step's meters and its fail-closed rule (`gen3_ridealong_heads_v1`)."""
from __future__ import annotations

import types

import numpy as np
import torch as th

from agents.training.instrumented_ppo.ridealong_terms import (RideAlongAccumulator,
                                                              RideAlongTerms, error_by_decile,
                                                              rank_auroc, spearman,
                                                              uncertainty_meters)


def test_rank_auroc_known_values_and_ties():
    assert rank_auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
    assert rank_auroc(np.array([0.9, 0.8, 0.2, 0.1]), np.array([0, 0, 1, 1])) == 0.0
    assert rank_auroc(np.ones(6), np.array([0, 1, 0, 1, 0, 1])) == 0.5       # all tied
    assert rank_auroc(np.arange(4.0), np.zeros(4)) is None                     # one class empty


def test_spearman_and_deciles():
    x = np.arange(100.0)
    assert abs(spearman(x, x ** 3) - 1.0) < 1e-12
    assert spearman(np.ones(10), x[:10]) is None
    dec = error_by_decile(x, x)
    assert len(dec) == 10 and dec[0] < dec[-1]
    m = uncertainty_meters("ens", x, np.where(x > 50, 0.9, 0.1))
    assert m["ens_auroc_err"] == 1.0 and m["ens_err_top_decile"] > m["ens_err_bottom_decile"]


class _Learner(RideAlongTerms):
    def __init__(self, policy):
        self.policy = policy
        self.opp_intent_coef = 0.05
        self.logged = {}
        self.logger = types.SimpleNamespace(record=lambda k, v: self.logged.__setitem__(k, v))


def _forward(pol, rows, masks):
    actions = th.tensor([int(np.flatnonzero(m)[0]) for m in masks])
    obs = {"observation": th.tensor(rows), "action_mask": th.tensor(masks.astype(np.float32))}
    values, _, _ = pol.evaluate_actions(obs, actions, action_masks=masks)
    n = len(rows)
    rd = types.SimpleNamespace(
        observations={**obs, "win_target": (th.arange(n) % 2).float()[:, None],
                      "win_mask": th.ones(n, 1), "opp_action_kind": th.ones(n, 1),
                      "opp_action_num": th.zeros(n, 1), "opp_class": th.zeros(n, 1)},
        advantages=th.linspace(-0.2, 0.2, n))
    return rd, values, actions


def test_a_step_trains_the_heads_and_a_NONFINITE_step_disables_them_without_touching_anything():
    from agents.model.compile_parity_fixture import load_parity_rows
    from agents.model.ridealong_heads_test import _policy

    pol, layout = _policy()
    rows, masks = load_parity_rows(layout["total_dim"])
    rows, masks = rows[:8].copy(), masks[:8]
    lr = _Learner(pol)
    lr._ridealong_acquire()            # the startup acquisition (there is no lazy build)
    core0 = {k: v.clone() for k, v in pol.state_dict().items() if not k.startswith("ridealong.")}
    ra0 = {k: v.clone() for k, v in pol.ridealong.state_dict().items()}
    acc = RideAlongAccumulator()
    lr._ridealong_update(*_forward(pol, rows, masks), epoch=0, acc=acc)
    moved = {k for k, v in pol.ridealong.state_dict().items() if not th.equal(v, ra0[k])}
    assert any(k.startswith("ensemble.members") for k in moved)
    assert all(p.grad is None for p in pol.ridealong.parameters()), "grads must be back to None"
    assert acc.metrics().get("ens_loss") is not None
    # Poison the observation: the RND error goes non-finite. No step is taken, the heads disable
    # themselves, and nothing — theirs or PPO's — is touched.
    ra1 = {k: v.clone() for k, v in pol.ridealong.state_dict().items()}
    rd, values, actions = _forward(pol, rows, masks)       # a clean forward (ids must embed)…
    bad = rd.observations["observation"].clone()
    bad[0, 0] = float("nan")                               # …then a poisoned row for the heads
    rd.observations["observation"] = bad
    lr._ridealong_update(rd, values, actions, epoch=0, acc=RideAlongAccumulator())
    assert lr._ridealong_disabled
    # the observation variants share base's poisoned normaliser: each stopped ALONE; feat did not
    assert lr._ridealong_variant_disabled == {"fast", "decay", "small"}
    for k, v in pol.ridealong.state_dict().items():
        if (k.startswith("rnd.obs_") or k.startswith("rnd.err_")
                or (k.startswith("rnd_variants.") and (".obs_" in k or ".err_" in k))):
            continue        # the running statistics saw the batch; the WEIGHTS must not move
        if k.startswith("rnd_variants.decay.predictor."):
            continue        # the per-UPDATE pull toward init ran at this update's start (no step)
        if k.startswith("rnd_variants.feat."):
            continue        # feat reads value_pooled, which is FINITE here: it legitimately stepped
        assert th.equal(v, ra1[k]), f"a non-finite step moved {k}"
    for k, v in pol.state_dict().items():
        if not k.startswith("ridealong."):
            assert th.equal(v, core0[k]), k
    lr._record_ridealong_metrics(RideAlongAccumulator())
    assert lr.logged["ridealong/disabled"] == 1.0
    # …and once disabled, a step is a no-op.
    ra2 = {k: v.clone() for k, v in pol.ridealong.state_dict().items()}
    lr._ridealong_update(*_forward(pol, rows, masks), epoch=0, acc=RideAlongAccumulator())
    assert all(th.allclose(v.double(), ra2[k].double(), rtol=0, atol=0, equal_nan=True)
               for k, v in pol.ridealong.state_dict().items())


def test_a_NONFINITE_VARIANT_disables_ONLY_that_variant():
    """gen3_ridealong_rnd_variants_v1: each variant fails closed ALONE — the four heads and the other
    variants keep stepping, and the dead one is logged."""
    from agents.model.compile_parity_fixture import load_parity_rows
    from agents.model.ridealong_heads_test import _policy

    pol, layout = _policy()
    rows, masks = load_parity_rows(layout["total_dim"])
    rows, masks = rows[:8].copy(), masks[:8]
    lr = _Learner(pol)
    lr._ridealong_acquire()
    with th.no_grad():
        pol.ridealong.rnd_variants["small"].predictor[0].bias[0] = float("inf")
    before = {k: v.clone() for k, v in pol.ridealong.state_dict().items()}
    acc = RideAlongAccumulator()
    lr._ridealong_update(*_forward(pol, rows, masks), epoch=0, acc=acc)
    assert lr._ridealong_variant_disabled == {"small"}
    assert not getattr(lr, "_ridealong_disabled", False)
    after = pol.ridealong.state_dict()
    for pre in ("rnd.predictor.", "ensemble.members.", "rnd_variants.fast.predictor.",
                "rnd_variants.decay.predictor.", "rnd_variants.feat.predictor."):
        assert any(not th.equal(after[k], before[k]) for k in after if k.startswith(pre)), pre
    small_w = [k for k in after if k.startswith("rnd_variants.small.predictor.")]
    assert all(th.equal(after[k], before[k]) for k in small_w), "a non-finite variant was stepped"
    lr._record_ridealong_metrics(acc)
    assert lr.logged["ridealong/rndv_small_disabled"] == 1.0
    assert lr.logged["ridealong/rndv_fast_disabled"] == 0.0
    assert lr.logged["ridealong/disabled"] == 0.0
    # the dead variant is no longer read: no later series of it
    acc2 = RideAlongAccumulator()
    lr._ridealong_update(*_forward(pol, rows, masks), epoch=0, acc=acc2)
    m = acc2.metrics()
    assert "rndv_small_err_mean" not in m and "rndv_fast_err_mean" in m


def test_the_SATURATION_and_IDENTIFICATION_series():
    acc = RideAlongAccumulator()
    acc.col("rndv_fast_err_raw", th.tensor([1.0, 2.0, 3.0, 4.0, 5.0]))
    acc.col("rnd_err_raw", th.full((5,), 2.0))
    acc.observe_identification({"rnd_err": th.ones(4), "rndv_fast_err": th.ones(4)},
                               {"rnd": th.full((4,), 3.0), "rndv_fast": th.ones(4)})
    m = acc.metrics()
    assert m["rndv_fast_err_median"] == 3.0
    assert m["rndv_fast_err_iqr"] == 2.0 and abs(m["rndv_fast_err_rel_spread"] - 2.0 / 3.0) < 1e-12
    assert m["rnd_err_iqr"] == 0.0 and m["rnd_err_rel_spread"] == 0.0     # a collapsed spread
    assert m["rnd_ident_ratio"] == 3.0 and m["rndv_fast_ident_ratio"] == 1.0
