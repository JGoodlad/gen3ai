"""The in-run parity canary (K6, `gen3_compile_canary_v1`; persistence `gen3_compile_canary_v2`): it
passes on a healthy compiled learner, FAILS (typed) on a compiled region that drifted from eager —
including a backward-only drift that only the gradient check sees — runs on its cadence only, and
changes nothing about training (RNG, `.grad`, training mode). CPU, the production-surface learner
(the K9 golden's) with K8's declared regions installed under dynamo's `eager` backend standing in for
Inductor; a fault is PLANTED in the installed region."""
from __future__ import annotations

import pytest
import torch

from agents.model.compile_canary import CompileCanary, CompileCanaryError
from agents.model.compile_trainer import CompileTrainerError


# ------------------------------------------------------- the regions learner + planted faults
@pytest.fixture
def regions_learner(tmp_path):
    """The production-surface learner with K8's regions installed (CPU, dynamo `eager` backend, so
    compiled == eager unless a test plants a fault), its run dir a tmp dir."""
    from agents.model import compile_control as cc
    from agents.model import compile_regions as cr
    from agents.training import learner_golden as LG
    cc._reset_control_for_tests()
    torch._dynamo.reset()
    m = LG.build_learner()
    LG.load_buffer_into(m)
    m.behaviour_dump_dir = str(tmp_path)
    cc.control().install()
    cr.install(m, backend="eager")
    try:
        yield m
    finally:
        cr.uninstall(m)
        cc._reset_control_for_tests()
        torch._dynamo.config.error_on_recompile = False
        torch._dynamo.reset()


def _plant_r0(model, armed):
    """Replace the installed R0 with one that is off by +0.05 log-prob while ``armed['on']`` (and,
    with ``armed['once']``, only on its next call): a planted compiled-graph fault."""
    from agents.model.policy import _ROLLOUT_REGIONS
    real = _ROLLOUT_REGIONS[model.policy]

    def faulty(pol, obs, masks):
        values, logp = real(pol, obs, masks)
        if armed["on"]:
            if armed.get("once"):
                armed["on"] = False
            return values, logp + 0.05
        return values, logp
    _ROLLOUT_REGIONS[model.policy] = faulty


def _verdicts(tmp):
    import json
    from agents.model.compile_canary import VERDICTS_FILE
    p = tmp / VERDICTS_FILE
    return [json.loads(x)["verdict"] for x in p.read_text().splitlines()] if p.exists() else []


def test_a_healthy_compiled_learner_passes_with_the_gradient_checked(regions_learner):
    can = CompileCanary(regions_learner, n_envs=4, batch_size=16, every=1, grad_every=1)
    out = can.after_update()
    assert out["compile/canary_ok"] == 1.0 and out["compile/canary_grad_checked"] == 1.0
    assert out["compile/canary_grad_cosine"] > 0.9999
    assert out["compile/canary_max_abs_legal_logprob"] < 1e-5


def _plant_r1_backward(model):
    """Replace the installed R1 with one whose FORWARD (the loss) is exact but whose BACKWARD is x3:
    a backward-only compiled-graph fault, invisible to the decision readout."""
    real = model._compiled_micro_step

    class _ScaleGrad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.view_as(x)

        @staticmethod
        def backward(ctx, g):
            return g * 3.0

    def bad(*a):
        out = real(*a)
        return out._replace(loss=_ScaleGrad.apply(out.loss))
    model._compiled_micro_step = bad


def test_a_backward_only_drift_is_caught_on_a_gradient_canary(regions_learner):
    """A region whose forward matches but whose gradient is wrong: only the train-graph check sees it.
    Fails if the gradient canary stops comparing R1's compiled gradient with eager's."""
    _plant_r1_backward(regions_learner)
    decision_only = CompileCanary(regions_learner, n_envs=4, batch_size=16, every=1, grad_every=10,
                                  saver=lambda p: None)
    assert decision_only.after_update()["compile/canary_ok"] == 1.0   # the forward agrees
    with_grad = CompileCanary(regions_learner, n_envs=4, batch_size=16, every=1, grad_every=1,
                              saver=lambda p: None)
    with pytest.raises(CompileCanaryError):
        with_grad.after_update()


def test_the_cadence_runs_every_N_updates_and_writes_nothing_between(regions_learner):
    can = CompileCanary(regions_learner, n_envs=4, batch_size=16, every=3, grad_every=2)
    got = [can.after_update() for _ in range(6)]
    assert [bool(g) for g in got] == [False, False, True, False, False, True]
    assert got[2]["compile/canary_grad_checked"] == 0.0 and got[5]["compile/canary_grad_checked"] == 1.0


def test_the_canary_changes_nothing_about_training(regions_learner):
    model = regions_learner
    model.policy.set_training_mode(True)
    before = [p.detach().clone() for p in model.policy.parameters()]
    torch.manual_seed(7)
    expect = torch.rand(4)
    torch.manual_seed(7)
    CompileCanary(model, n_envs=4, batch_size=16, every=1, grad_every=1).after_update()
    assert torch.equal(torch.rand(4), expect)                 # RNG-neutral
    assert model.policy.training                              # mode restored
    assert all(p.grad is None for p in model.policy.parameters())
    assert all(torch.equal(a, b) for a, b in zip(before, model.policy.parameters()))


def test_a_PERSISTENT_fault_is_CONFIRMED_and_FATAL_in_the_SAME_update_with_a_checkpoint(regions_learner, tmp_path):
    saved = []
    _plant_r0(regions_learner, {"on": True})
    can = CompileCanary(regions_learner, n_envs=4, batch_size=16, every=1, saver=saved.append)
    with pytest.raises(CompileCanaryError, match="CONFIRMED") as ei:
        can.after_update()
    assert isinstance(ei.value, CompileTrainerError)          # -> FATAL_CONFIG, not restarted
    assert saved and saved[0].endswith("final_model_canary_fatal")
    assert "SAFE ROLLBACK POINT" in str(ei.value)
    assert _verdicts(tmp_path) == ["fatal"]


def test_a_one_shot_BLIP_warns_counts_and_training_continues(regions_learner, tmp_path):
    _plant_r0(regions_learner, {"on": True, "once": True})
    can = CompileCanary(regions_learner, n_envs=4, batch_size=16, every=1, saver=lambda p: None)
    out = can.after_update()                                   # no raise: it did not reproduce
    assert out["compile/canary_ok"] == 0.0
    assert out["compile/canary_unconfirmed_disagreements"] == 1.0
    out = can.after_update()                                   # healthy again: a pass resets the streak
    assert out["compile/canary_ok"] == 1.0 and out["compile/canary_unconfirmed_disagreements"] == 1.0
    assert _verdicts(tmp_path) == ["unconfirmed", "pass"]


def test_TWO_CONSECUTIVE_unconfirmed_disagreements_are_FATAL(regions_learner, tmp_path):
    armed = {"on": True, "once": True}
    _plant_r0(regions_learner, armed)
    can = CompileCanary(regions_learner, n_envs=4, batch_size=16, every=1, saver=lambda p: None)
    can.after_update()                                         # unconfirmed #1
    armed["on"] = True                                         # the next scheduled canary blips again
    with pytest.raises(CompileCanaryError, match="two CONSECUTIVE"):
        can.after_update()
    assert _verdicts(tmp_path) == ["unconfirmed", "fatal"]


def test_the_rollback_point_is_the_newest_checkpoint_at_or_before_the_last_PASSING_canary(tmp_path):
    import json
    from agents.model.compile_canary import VERDICTS_FILE, rollback_point
    assert rollback_point(str(tmp_path)).startswith("none: no canary of this run has PASSED")
    rows = [{"verdict": "pass", "num_timesteps": 1000}, {"verdict": "pass", "num_timesteps": 3000},
            {"verdict": "unconfirmed", "num_timesteps": 4000}]
    (tmp_path / VERDICTS_FILE).write_text("".join(json.dumps(r) + "\n" for r in rows))
    ck = tmp_path / "checkpoints"
    ck.mkdir()
    for n in (500, 2500, 3500):
        (ck / f"checkpoint_{n}_steps.zip").write_text("x")
    got = rollback_point(str(tmp_path))
    assert got.startswith(str(ck / "checkpoint_2500_steps.zip")) and "3,000" in got


def test_the_production_cadence_runs_FIRST_at_update_10_then_every_100(regions_learner):
    """No run of any length goes unchecked: an 82-update run (sizing arm A) never reached 100."""
    from agents.model.compile_canary import CANARY_EVERY, CANARY_FIRST
    assert (CANARY_FIRST, CANARY_EVERY) == (10, 100)
    can = CompileCanary(regions_learner, n_envs=4, batch_size=16)
    ran = []
    can.run = lambda *, grad: ran.append(can.updates) or {"compile/canary_ok": 1.0}
    for _ in range(205):
        can.after_update()
    assert ran == [10, 100, 200]
