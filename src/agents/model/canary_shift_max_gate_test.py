"""`--canary-shift-max` (T37, `gen3_canary_shift_max_v1`): the JUDGEMENT half — the compile gate's and the canary's clip margin
is built from the model's own shift, and every R1 rule line and every canary record says which.

The plumbing (parse, resolve, inherit, record) and the bit-identity of training are `main/train/canary_shift_max_test.py`.
Here, on CPU (the production-surface learner, R1 under dynamo's `eager` backend standing in for Inductor):

* `resolve_shift_max` is the default for None and RAISES-ONLY above it (typed refusals);
* the margin follows the model: `M = R1_BAR_K x shift` and the relocation uses it;
* every R1 rule line of the startup gate prints `shift_max <v> [default | --canary-shift-max]` and the margin, and the
  relocation reads the margin off the model;
* the margin rule's refusal names the flag as the way to raise it, and a RAISED value turns an arm the default margin
  refuses (a healthy-but-large shift: the X26 RED read) into a pass — the whole point of the flag;
* every canary record (pass, unconfirmed, fatal) carries `canary_shift_max` {value, source, margin}, and the watch's
  `clip_margin` carries the shift beside the margin.
"""
from __future__ import annotations

import json
import math

import pytest
import torch

from agents.model import compile_canary as cc_
from agents.model import compile_control as cc
from agents.model import compile_regions as cr
from agents.model import compile_trainer as ct
from agents.model.compile_canary import CompileCanary, CompileCanaryError

DEFAULT_M = cr.R1_CLIP_MARGIN


# ── the pure resolver ───────────────────────────────────────────────────────────────────────────
def test_the_resolver_is_the_default_for_None_and_only_RAISES_above_it():
    assert cr.resolve_shift_max(None) == cr.R1_LOGPROB_SHIFT_MAX == 0.18
    assert cr.resolve_shift_max(0.18) == 0.18
    assert cr.resolve_shift_max("0.5") == 0.5
    assert cr.resolve_shift_max(3) == 3.0
    for bad in (0.179, 0.0, -1.0, float("nan"), float("inf"), float("-inf"), "x", object()):
        with pytest.raises(cr.CanaryShiftMaxError):
            cr.resolve_shift_max(bad)


def test_the_model_helpers_read_the_models_attribute_and_name_the_source():
    class M:                                   # a duck-typed double, a reader, a pre-v158 model
        pass
    m = M()
    assert (cr.shift_max_of(m), cr.shift_max_source(m)) == (0.18, "default")
    assert cr.clip_margin_of(m) == pytest.approx(DEFAULT_M) == pytest.approx(0.72)
    m._canary_shift_max = None
    assert cr.shift_max_source(m) == "default"
    m._canary_shift_max = 0.5
    assert (cr.shift_max_of(m), cr.shift_max_source(m)) == (0.5, "--canary-shift-max")
    assert cr.clip_margin_of(m) == pytest.approx(cr.R1_BAR_K * 0.5) == pytest.approx(2.0)
    m._canary_shift_max = 0.1                  # a model that somehow carries a LOWER bar is refused, not clamped
    with pytest.raises(cr.CanaryShiftMaxError, match="BELOW"):
        cr.clip_margin_of(m)


# ── the margin rule: a RAISED shift turns the RED read's refusal into a pass ────────────────────
CLIP = 0.15


def _rows():
    lp = torch.tensor([-1.0, -0.5, -2.0, -1.5, -0.7], dtype=torch.float64)
    log_ratio = torch.tensor([math.log(1 + CLIP), math.log(1 - CLIP), -2.0, 3.0, math.log(1 + CLIP)],
                             dtype=torch.float64)
    adv = torch.tensor([1.3, -0.8, 0.5, 0.9, 0.0], dtype=torch.float64)
    return lp, (lp - log_ratio).float(), adv


def test_a_RAISED_shift_turns_the_defaults_refusal_of_a_healthy_large_shift_into_a_pass():
    """The X26 RED read, in miniature: the compiled arm moves a log-prob by 1.2 x the default margin — past the 4x
    premise, the run stopped by the gate. Typing a raised shift (here 0.5, so M = 2.0) moves the relocation 2M further
    inside and the same arm passes with no clip-branch switch. Fails if the margin ignores the shift."""
    lp, old, adv = _rows()
    arm = {"compiled": {"log_prob": (lp + 1.2 * DEFAULT_M).float()}}
    default = cr.clip_relocation(lp, old, adv, CLIP, cr.R1_BAR_K * 0.18, "default")
    with pytest.raises(ct.CompileTrainerError) as e:
        cr._clip_margin_rule(default, arm)
    msg = str(e.value)
    assert "--canary-shift-max <v> on a RESUME" in msg and "[default]" in msg, msg           # names how to raise it
    raised = cr.clip_relocation(lp, old, adv, CLIP, cr.R1_BAR_K * 0.5, "--canary-shift-max")
    w = cr.GradWatch(("pointer_head.",))
    line = cr._clip_margin_rule(raised, arm, w)
    assert "0 clip-branch switches" in line and "shift_max 0.5 [--canary-shift-max]" in line, line
    rec = w.record()["clip_margin"]
    assert rec["shift_max"] == pytest.approx(0.5) and rec["shift_max_source"] == "--canary-shift-max"
    assert rec["margin"] == pytest.approx(2.0)


def test_the_refusal_of_a_RAISED_margin_names_the_effective_value():
    lp, old, adv = _rows()
    fix = cr.clip_relocation(lp, old, adv, CLIP, cr.R1_BAR_K * 0.5, "--canary-shift-max")
    with pytest.raises(ct.CompileTrainerError, match=r"shift 0\.5 \[--canary-shift-max\]"):
        cr._clip_margin_rule(fix, {"compiled": {"log_prob": (lp + 1.2 * fix.margin).float()}})


# ── the gate on real rows: every R1 rule line carries the value ─────────────────────────────────
@pytest.fixture
def learner():
    from agents.training import learner_golden as LG
    cc._reset_control_for_tests()
    torch._dynamo.reset()
    model = LG.build_learner()
    LG.load_buffer_into(model)
    try:
        yield model
    finally:
        cr.uninstall(model)
        cc._reset_control_for_tests()
        torch._dynamo.config.error_on_recompile = False
        torch._dynamo.reset()


def _gate(model):
    cc.control().install()
    cr.install(model, backend="eager")
    return [r for r in cr.gate_regions(model, batch_size=16, say=lambda _m: None) if r.startswith("R1 ")]


def test_the_relocation_uses_the_models_margin_and_a_larger_one_moves_a_SUPERSET_of_rows(learner):
    """`relocate_clip_boundary` (every rung's first step) reads the margin off the model. Fails on revert of that read:
    the record's margin / source stay the default's. A larger margin relocates every row a smaller one does (|d| <= M)."""
    args = cr._r1_args(learner, cr.r1_batch(learner, 16, 0))
    _, fix0 = cr.relocate_clip_boundary(learner, args)
    assert (fix0.margin, fix0.shift_source) == (pytest.approx(0.72), "default")
    assert "shift_max 0.18 [default]" in fix0.line() and "M 0.720 = 4 x" in fix0.line()
    learner._canary_shift_max = 0.5
    _, fix1 = cr.relocate_clip_boundary(learner, args)
    assert (fix1.margin, fix1.shift_source) == (pytest.approx(2.0), "--canary-shift-max")
    assert "shift_max 0.5 [--canary-shift-max]" in fix1.line() and "M 2.000 = 4 x" in fix1.line()
    assert bool((fix1.moved | ~fix0.moved).all())              # moved(default) is a subset of moved(raised)
    assert int(fix1.moved.sum()) >= int(fix0.moved.sum())


@pytest.mark.slow  # ship gate (2026-10-19): 26 s beside a live run
def test_every_R1_rule_line_of_the_startup_gate_prints_the_raised_shift_its_source_and_the_margin(learner):
    """The gate end to end (both fresh/perturbed or trained rungs): each R1 line carries the value, and the raised margin
    still passes with no clip-branch switch."""
    learner._canary_shift_max = 0.5
    lines = _gate(learner)
    assert lines, "PRECONDITION: the gate judged R1"
    assert all("shift_max 0.5 [--canary-shift-max]" in r and "M 2.000 = 4 x" in r and "0 clip-branch switches" in r
               for r in lines), lines


# ── the canary's records ────────────────────────────────────────────────────────────────────────
@pytest.fixture
def regions_learner(tmp_path):
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


def _records(tmp):
    return [json.loads(x) for x in (tmp / cc_.VERDICTS_FILE).read_text().splitlines()]


@pytest.mark.slow  # ship gate (2026-10-19): 25 s beside a live run
@pytest.mark.parametrize("flag, value, source, margin", [(None, 0.18, "default", 0.72),
                                                         (0.3, 0.3, "--canary-shift-max", 1.2)])
def test_a_PASS_record_carries_the_shift_its_source_and_the_margin(regions_learner, tmp_path, flag, value, source, margin):
    regions_learner._canary_shift_max = flag
    CompileCanary(regions_learner, batch_size=16, every=1).after_update()
    (rec,) = _records(tmp_path)
    assert rec["verdict"] == "pass"
    assert rec["canary_shift_max"] == {"value": pytest.approx(value), "source": source, "margin": pytest.approx(margin)}
    cm = rec["watch"]["scheduled"]["clip_margin"]
    assert cm["shift_max"] == pytest.approx(value, rel=1e-3) and cm["shift_max_source"] == source
    assert cm["margin"] == pytest.approx(margin, rel=1e-3)


@pytest.mark.slow  # ship gate (2026-10-19): 28 s beside a live run
def test_a_FATAL_record_carries_it_too(regions_learner, tmp_path):
    from agents.model.compile_canary_test import _plant_r1
    regions_learner._canary_shift_max = 0.3
    _plant_r1(regions_learner, {"on": True})
    can = CompileCanary(regions_learner, batch_size=16, every=1, saver=lambda p: None)
    with pytest.raises(CompileCanaryError, match="CONFIRMED"):
        can.after_update()
    (rec,) = _records(tmp_path)
    assert rec["verdict"] == "fatal"
    assert rec["canary_shift_max"]["source"] == "--canary-shift-max" and rec["canary_shift_max"]["value"] == 0.3
