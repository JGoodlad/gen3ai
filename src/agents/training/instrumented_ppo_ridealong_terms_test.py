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
    for k, v in pol.ridealong.state_dict().items():
        if k.startswith("rnd.obs_") or k.startswith("rnd.err_"):
            continue        # the running statistics saw the batch; the WEIGHTS must not move
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
