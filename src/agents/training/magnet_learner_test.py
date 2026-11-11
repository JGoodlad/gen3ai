"""The trailing magnet in the LEARNER (`gen3_magnet_kl_v1`, CE-3d / T56): `micro_step`'s term and `train()`'s meter,
on the K9 learner and its committed buffer (`learner_golden`), fp32 trunk.

* **coefficient 0 is the update it was**: R1 gets its nine arguments, the loss has no magnet term, a column in the
  buffer is inert, and one whole `train()` still reproduces the recorded fp32 golden bit for bit;
* **on**, the term is β · KL(π ‖ π_mag) over the legal actions: exactly 0 against the policy's own log-probabilities,
  and against the masked uniform it IS an entropy bonus of coefficient β (value and gradient);
* **on with no column** is a typed error, never a silently unregularised update;
* `train()` records ``train/magnet_kl`` (finite), and the magnet moves the update.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from agents.model import compile_regions as cr
from agents.training import learner_golden as LG
from agents.training import magnet as M
from agents.training.instrumented_ppo import ppo as PPO
from agents.training.instrumented_ppo.micro_step import PPO_LISTS, micro_step

BATCH = 16
BETA = 0.05


@pytest.fixture
def learner():
    model = LG.build_learner(trunk_precision="fp32")
    LG.load_buffer_into(model)
    return model


def _args(model, slice_=0):
    return cr._r1_args(model, cr.r1_batch(model, BATCH, slice_))


def _run(model, args, column=None, grad=False):
    model.policy.set_training_mode(True)
    extra = () if column is None else (column,)
    if not grad:
        with th.no_grad():
            return micro_step(*args[:9], *extra), None
    model.policy.zero_grad(set_to_none=True)
    out = micro_step(*args[:9], *extra)
    out.loss.backward()
    g = th.cat([p.grad.reshape(-1) for p in model.policy.parameters() if p.grad is not None]).clone()
    model.policy.zero_grad(set_to_none=True)
    return out, g


def _own(out):
    return th.where(out.masks_bool, out.logp, out.logp.new_zeros(())).detach()


def test_OFF_is_the_nine_argument_call_with_no_term_and_a_column_is_inert(learner):
    assert learner.magnet_kl_coef == 0.0 and getattr(learner, "_magnet", None) is None       # the default
    args = _args(learner)
    assert len(args) == 9 and args[8].magnet_coef == 0.0
    assert PPO._magnet_args(args[8], SimpleNamespace(magnet_logp=th.zeros(BATCH, 9))) == ()
    off, _ = _run(learner, args)
    assert "magnet" not in off.terms and "magnet_kl/" not in off.metrics
    junk, _ = _run(learner, args, column=th.full_like(off.logp, 3.0))        # a column, but β = 0: never read
    assert th.equal(junk.loss, off.loss) and "magnet" not in junk.terms
    assert "magnet_kl" in PPO_LISTS


def test_OFF_one_whole_update_is_the_recorded_golden_even_with_a_column_in_the_buffer(learner):
    """The K9 fp32 golden was recorded before the magnet existed: with β = 0 the update reproduces its bytes, and a
    column left in the buffer (a fork that turned the lever off mid-rollout cannot leave one, but the gather would
    serve it) changes nothing."""
    rb = learner.rollout_buffer

    def plant():
        rb.magnet_logp = np.full(rb.action_masks.shape, -2.0, dtype=np.float32)
    now = LG.compute(learner, before_train=plant)
    assert LG.diff(LG.entry_fp32(), now) == []
    assert M.TAG_KL not in learner.logger.name_to_value


def test_ON_the_KL_to_the_policys_own_log_probs_is_exactly_zero_and_the_loss_is_the_OFF_loss(learner):
    off, _ = _run(learner, _args(learner))
    learner.magnet_kl_coef = BETA
    args = _args(learner)
    assert len(args) == 10 and args[8].magnet_coef == BETA
    # the gate fixture's column is the masked uniform (the region's signature, not a magnet)
    assert th.equal(args[9], M.masked_uniform_logp(args[3]))
    on, _ = _run(learner, args, column=_own(off))
    assert float(on.metrics["magnet_kl/"][0]) == 0.0
    assert th.equal(on.loss, off.loss)                                       # β · 0 added: the same bytes
    assert on.term_groups["magnet"] == "entropy" and float(on.terms["magnet"]) == 0.0


def test_ON_against_the_masked_uniform_the_term_IS_an_entropy_bonus_of_coefficient_beta(learner):
    """KL(π ‖ uniform over the legal actions) = log n_legal − H(π): the value is the OFF loss plus β · mean of that,
    and the gradient is the one a run with ``ent_coef + β`` and no magnet takes. Fails on the wrong KL direction (the
    forward KL is not an entropy), on a KL summed over illegal actions, and on a coefficient that is not applied."""
    args0 = _args(learner)
    st0 = args0[8]
    off, g_off = _run(learner, args0, grad=True)
    learner.magnet_kl_coef = BETA
    args = _args(learner)
    uni = args[9]
    on, g_on = _run(learner, args, column=uni, grad=True)
    n_legal = args[3].to(th.bool).sum(-1).float()
    want_kl = float((th.log(n_legal) - off.entropy.detach()).mean())
    kl = float(on.metrics["magnet_kl/"][0])
    assert want_kl > 5e-3 and abs(kl - want_kl) < 1e-6, (kl, want_kl)   # a near-uniform fresh policy: KL ~ 0.02
    assert abs(float(on.loss.detach()) - (float(off.loss.detach()) + BETA * want_kl)) < 1e-6
    # the gradient: OFF at ent_coef + β
    ent = (*args0[:8], st0._replace(ent_coef=st0.ent_coef + BETA))
    _, g_ent = _run(learner, ent, grad=True)
    moved = float((g_on - g_off).norm() / g_off.norm())
    same = float((g_on - g_ent).norm() / g_off.norm())
    assert moved > 1e-3, moved                       # precondition: the magnet's gradient is not rounding
    assert same < 1e-2 * moved, (same, moved)


def test_ON_with_no_column_is_a_typed_error_never_a_silently_unregularised_update(learner):
    from main.exit_codes import TrainExitCode, exit_code_for
    learner.magnet_kl_coef = BETA
    args = _args(learner)
    with pytest.raises(ValueError, match="magnet column"):
        _run(learner, args)                                                  # R1 itself
    with pytest.raises(M.MagnetError, match="no magnet column"):
        PPO._magnet_args(args[8], SimpleNamespace(magnet_logp=None))
    LG.load_buffer_into(learner)                                             # a buffer no collector filled
    with pytest.raises(M.MagnetError, match="no magnet column") as e:
        learner.train()
    assert exit_code_for(e.value) == int(TrainExitCode.FATAL_CONFIG)


def test_ON_a_whole_update_records_the_KL_and_is_not_the_OFF_update(learner):
    learner.magnet_kl_coef = BETA
    rb = learner.rollout_buffer

    def plant():
        rb.magnet_logp = M.masked_uniform_logp(th.as_tensor(rb.action_masks)).numpy().astype(np.float32)
    now = LG.compute(learner, before_train=plant)
    kl = learner.logger.name_to_value[M.TAG_KL]
    assert np.isfinite(kl) and kl > 5e-3, kl
    recorded = LG.entry_fp32()
    assert now["init_params_sha256"] == recorded["init_params_sha256"]       # the same learner going in
    assert now["post_params_sha256"] != recorded["post_params_sha256"]       # a different update coming out
    assert rb.magnet_logp is not None                                        # train() reads the column, never drops it
