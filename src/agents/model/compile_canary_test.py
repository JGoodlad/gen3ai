"""The in-run parity canary (K6, `gen3_compile_canary_v1`; persistence `gen3_compile_canary_v2`): it
passes on a healthy compiled learner, FAILS (typed) on a compiled region that drifted from eager —
including a backward-only drift that only the gradient check sees, on EVERY canary (there is no
decision-only canary: R1, the micro-step, is the only compiled region, P10-E) — runs on its cadence only,
and changes nothing about training (RNG, `.grad`, training mode). CPU, the production-surface learner
(the K9 golden's) with K8's declared region installed under dynamo's `eager` backend standing in for
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


class _ScaleGrad(torch.autograd.Function):
    """Identity forward, x3 backward: a backward-only compiled-graph fault."""
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return g * 3.0


def _plant_r1(model, armed):
    """Replace the installed R1 with one whose FORWARD (the loss) is exact but whose BACKWARD is x3 while
    ``armed['on']`` (and, with ``armed['once']``, only on its next call): a planted compiled-graph fault,
    invisible to the loss and to the gradient's cosine — only the per-parameter rule sees it."""
    real = model._compiled_micro_step

    def faulty(*a):
        out = real(*a)
        if armed["on"]:
            if armed.get("once"):
                armed["on"] = False
            return out._replace(loss=_ScaleGrad.apply(out.loss))
        return out
    model._compiled_micro_step = faulty


def _verdicts(tmp):
    import json
    from agents.model.compile_canary import VERDICTS_FILE
    p = tmp / VERDICTS_FILE
    return [json.loads(x)["verdict"] for x in p.read_text().splitlines()] if p.exists() else []


def test_a_healthy_compiled_learner_passes_with_the_gradient_checked(regions_learner):
    can = CompileCanary(regions_learner, batch_size=16, every=1)
    out = can.after_update()
    assert out["compile/canary_ok"] == 1.0
    assert out["compile/canary_grad_cosine"] > 0.9999


def test_a_backward_only_drift_is_caught_on_EVERY_canary_including_the_first(regions_learner):
    """A region whose forward matches but whose gradient is wrong: only the train-graph check sees it.
    There is no decision-only canary to pass it (the old cadence ran the gradient every K-th canary and
    a K > 1 canary checked nothing once R0 was gone): the FIRST scheduled canary of a custom cadence
    raises. Fails if the gradient check stops being part of every canary."""
    _plant_r1(regions_learner, {"on": True})
    can = CompileCanary(regions_learner, batch_size=16, every=3, saver=lambda p: None)
    assert can.after_update() == {} and can.after_update() == {}      # not due yet
    with pytest.raises(CompileCanaryError):
        can.after_update()                                            # the first scheduled canary


def test_the_cadence_runs_every_N_updates_and_writes_nothing_between(regions_learner):
    can = CompileCanary(regions_learner, batch_size=16, every=3)
    got = [can.after_update() for _ in range(6)]
    assert [bool(g) for g in got] == [False, False, True, False, False, True]
    assert all(got[i]["compile/canary_ok"] == 1.0 and got[i]["compile/canary_grad_cosine"] > 0.9999
               for i in (2, 5))


def test_the_canary_changes_nothing_about_training(regions_learner):
    model = regions_learner
    model.policy.set_training_mode(True)
    before = [p.detach().clone() for p in model.policy.parameters()]
    torch.manual_seed(7)
    expect = torch.rand(4)
    torch.manual_seed(7)
    CompileCanary(model, batch_size=16, every=1).after_update()
    assert torch.equal(torch.rand(4), expect)                 # RNG-neutral
    assert model.policy.training                              # mode restored
    assert all(p.grad is None for p in model.policy.parameters())
    assert all(torch.equal(a, b) for a, b in zip(before, model.policy.parameters()))


def test_a_PERSISTENT_fault_is_CONFIRMED_and_FATAL_in_the_SAME_update_with_a_checkpoint(regions_learner, tmp_path):
    saved = []
    _plant_r1(regions_learner, {"on": True})
    can = CompileCanary(regions_learner, batch_size=16, every=1, saver=saved.append)
    with pytest.raises(CompileCanaryError, match="CONFIRMED") as ei:
        can.after_update()
    assert isinstance(ei.value, CompileTrainerError)          # -> FATAL_CONFIG, not restarted
    assert saved and saved[0].endswith("final_model_canary_fatal")
    assert "SAFE ROLLBACK POINT" in str(ei.value)
    assert _verdicts(tmp_path) == ["fatal"]


def test_a_one_shot_BLIP_warns_counts_and_training_continues(regions_learner, tmp_path):
    _plant_r1(regions_learner, {"on": True, "once": True})
    can = CompileCanary(regions_learner, batch_size=16, every=1, saver=lambda p: None)
    out = can.after_update()                                   # no raise: it did not reproduce
    assert out["compile/canary_ok"] == 0.0
    assert out["compile/canary_unconfirmed_disagreements"] == 1.0
    out = can.after_update()                                   # healthy again: a pass resets the streak
    assert out["compile/canary_ok"] == 1.0 and out["compile/canary_unconfirmed_disagreements"] == 1.0
    assert _verdicts(tmp_path) == ["unconfirmed", "pass"]


def test_TWO_CONSECUTIVE_unconfirmed_disagreements_are_FATAL(regions_learner, tmp_path):
    armed = {"on": True, "once": True}
    _plant_r1(regions_learner, armed)
    can = CompileCanary(regions_learner, batch_size=16, every=1, saver=lambda p: None)
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
    can = CompileCanary(regions_learner, batch_size=16)
    ran = []
    can.run = lambda: ran.append(can.updates) or {"compile/canary_ok": 1.0}
    for _ in range(205):
        can.after_update()
    assert ran == [10, 100, 200]


# ------------------------------------------------- the WATCH-LIST record (gen3_canary_watch_v1, 2026-10-15)
def _records(tmp):
    import json
    from agents.model.compile_canary import VERDICTS_FILE
    return [json.loads(x) for x in (tmp / VERDICTS_FILE).read_text().splitlines()]


def test_a_PASS_record_carries_the_watch_lists_per_parameter_numbers(regions_learner, tmp_path):
    """The `rb_toy_ent_s4001` FATAL named the pointer head at update 700, but a pass logged only each check's WORST
    parameter, so the pointer head's ratio at the six passing canaries before it is unrecoverable. Every record now
    carries the watch-list (`WATCH_PREFIXES`): under the bf16 rule (the golden learner's production recipe) the
    ratio, the compiled distance, eager bf16's yardstick and the floor of every watched parameter. Fails if a pass
    stops writing them."""
    from agents.model.compile_canary import WATCH_PREFIXES
    from agents.model.trunk_precision import trunk_precision_of
    assert trunk_precision_of(regions_learner.policy) == "bf16"          # PRECONDITION: the bf16 rule judges
    assert WATCH_PREFIXES == ("pointer_head.",)
    CompileCanary(regions_learner, batch_size=16, every=1).after_update()
    (rec,) = _records(tmp_path)
    w = rec["watch"]["scheduled"]
    assert rec["verdict"] == "pass" and w["rule"] == "bf16"
    head = {n for n, _ in regions_learner.policy.named_parameters() if n.startswith("pointer_head.")}
    assert set(w["params"]) == head
    judged = {n: v for n, v in w["params"].items() if "ratio" in v}
    assert judged, w["params"]
    for v in judged.values():
        assert set(v) == {"ratio", "compiled", "eager", "floor"} and v["ratio"] <= 12.0
    assert w["worst"]["ratio"] >= max(v["ratio"] for v in judged.values())
    assert {"drift_loss", "drift_grad", "envelope_loss", "envelope_grad"} <= set(w["global"])


def test_a_FATAL_record_carries_the_watch_of_the_scheduled_check_and_of_BOTH_confirmations(regions_learner, tmp_path):
    _plant_r1(regions_learner, {"on": True})
    can = CompileCanary(regions_learner, batch_size=16, every=1, saver=lambda p: None)
    with pytest.raises(CompileCanaryError, match="CONFIRMED"):
        can.after_update()
    (rec,) = _records(tmp_path)
    assert rec["verdict"] == "fatal"
    assert set(rec["watch"]) == {"scheduled", "confirm_slice0", "confirm_slice1"}
    for w in rec["watch"].values():
        # the planted x3 backward: compiled is ~2.0 from the reference on every judged pointer-head parameter
        ratios = [v["ratio"] for v in w["params"].values() if "ratio" in v]
        assert ratios and max(ratios) > 12.0, w


def test_the_watch_is_filled_even_when_a_GLOBAL_check_refuses_first():
    """The bf16 rule checks the global gradient before the per-parameter one; the watch is filled before either."""
    from agents.model import compile_regions as cr
    names = ["pointer_head.move_proj.weight", "trunk.w"]
    sizes = torch.tensor([2, 2])
    ref = {"loss": torch.tensor([1.0]), "grad": torch.tensor([1.0, 1.0, 1.0, 1.0]), "grad_sizes": sizes}
    eager = dict(ref, grad=torch.tensor([1.02, 1.0, 1.0, 1.01]))
    comp = dict(ref, grad=torch.tensor([1.5, 1.0, 1.0, 1.01]))
    w = cr.GradWatch(("pointer_head.",))
    with pytest.raises(CompileTrainerError, match="grad_rel_l2"):
        cr._r1_verdict_bf16(eager, comp, ref, names, "trained", watch=w)
    p = w.record()["params"]["pointer_head.move_proj.weight"]
    assert p["compiled"] == pytest.approx(0.5 / 2 ** 0.5, rel=1e-3)
    assert p["eager"] == pytest.approx(0.02 / 2 ** 0.5, rel=1e-3)
    assert p["ratio"] == pytest.approx(25.0, rel=1e-3)
    assert set(w.record()["params"]) == {"pointer_head.move_proj.weight"}


def test_the_fp32_rule_fills_the_watch_with_its_own_numbers():
    from agents.model import compile_regions as cr
    names = ["pointer_head.move_score.weight", "trunk.w"]
    sizes = torch.tensor([2, 2])
    eager = {"loss": torch.tensor([1.0]), "grad": torch.tensor([1.0, 2.0, 3.0, 4.0]), "grad_sizes": sizes}
    comp = dict(eager, grad=torch.tensor([1.0, 2.0 + 1e-6, 3.0, 4.0]))
    w = cr.GradWatch(("pointer_head.",))
    cr._r1_verdict(eager, comp, names, "trained", watch=w)
    rec = w.record()
    assert rec["rule"] == "fp32"
    p = rec["params"]["pointer_head.move_score.weight"]
    assert p["bar"] == pytest.approx(cr.R1_PARAM_BAR["trained"], rel=1e-3)
    assert 0 < p["compiled_vs_eager"] < 1e-5
