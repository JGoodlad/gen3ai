"""critic_route_audit — the instrument's own gates.

The audit exists to run ONCE per generation on a trained checkpoint, so its failure mode is
silent staleness: a hook that stops matching its argument measures nothing while producing a
plausible report. These tests pin (a) every arm FIRES on a live-route policy, (b) the zero-init
routes read exactly zero at init (threat's W_inj, the pointer-side KL), (c) the fail-loud path
raises when a hook never matches.
"""
import numpy as np
import pytest
import torch

pytest.importorskip("sb3_contrib")

from agents.model.critic_route_audit import audit, _Arms
from agents.model.identity_init_test import _build_real_policy


@pytest.fixture(scope="module")
def model_and_enc():
    return _build_real_policy(opp_belief_cls_k=6, attend_unrevealed_opponents=True,
                              value_threat_inject=True, value_entity_pool=True)


def _states(enc, n=16):
    rng = np.random.default_rng(0)
    return (rng.random((n, enc.dimension), dtype=np.float32),
            np.ones((n, 11), dtype=bool))


def test_every_arm_fires_and_reports(model_and_enc):
    model, enc = model_and_enc
    obs, masks = _states(enc)
    rep = audit(model.policy, obs, masks, batch=8)
    assert set(rep) == {"seed", "threat", "hidden_opp_both", "hidden_opp_pi",
                       "hidden_opp_vf", "entity_pool", "all_off"}
    for arm, row in rep.items():
        assert set(row) == {"kl_mean", "kl_p95", "flip_rate", "dv_mean"}


def test_zero_init_routes_read_zero_at_init(model_and_enc):
    """At init: threat's W_inj is zero (M1-guarded) so its arm is a strict no-op, and the
    pointer scorers are zero-init so NO arm can move the policy distribution yet — any
    nonzero here means an identity-at-init contract broke upstream."""
    model, enc = model_and_enc
    obs, masks = _states(enc)
    rep = audit(model.policy, obs, masks, batch=8)
    assert rep["threat"]["dv_mean"] == 0.0
    # the v80 entity pool's out projection is zero-init, so its arm reads a strict no-op cold
    assert rep["entity_pool"]["dv_mean"] == 0.0
    for arm, row in rep.items():
        assert row["kl_mean"] == 0.0 and row["flip_rate"] == 0.0, (arm, row)


def test_live_routes_move_the_critic(model_and_enc):
    model, enc = model_and_enc
    obs, masks = _states(enc)
    rep = audit(model.policy, obs, masks, batch=8)
    assert rep["seed"]["dv_mean"] > 0.0, "the seed window feeds vf even at init"
    assert rep["hidden_opp_vf"]["dv_mean"] > 0.0
    assert rep["hidden_opp_both"]["dv_mean"] > 0.0


def test_a_dead_hook_fails_loud():
    """The staleness guard: a marker that never fired must RAISE with the arm named, never
    return a plausible zero report."""
    from agents.model.critic_route_audit import _assert_fired

    class _Cold:
        fired = False

    with pytest.raises(RuntimeError, match="seed.*never matched"):
        _assert_fired("seed", [_Cold()])
    _assert_fired("ok", [{"fired": True}])   # dict-style marker, fired: no raise
