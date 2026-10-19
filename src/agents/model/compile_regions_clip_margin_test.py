"""R1's FIXTURE OFF THE PPO CLIP BOUNDARY (`gen3_r1_clip_margin_v1`, T36; standing rule 8).

The startup gate and the canary judge R1 on the K9 golden's committed rows, whose old log-probs are a near-uniform
policy's. On trained weights a row can sit within rounding of the clip boundary that SWITCHES its policy gradient, and
then the per-parameter statistic is a 0/1 lottery (the 2026-10-15 `rb_toy_ent_s4001` canary FATAL;
`designs/training/compile_flags.md` "The bf16 canary FATAL on trained weights"). `compile_regions.clip_relocation`
moves every such row 2 x `R1_CLIP_MARGIN` inside its ACTIVE side before judging, and `_clip_margin_rule` refuses an arm
that moves a log-prob past the margin. Here, on CPU:

* `clip_active` IS the surrogate's gradient switch (held to autograd of `micro_step`'s own expression);
* a row exactly AT its switching boundary is moved, deterministically (bit-equal on a re-run), to 2M inside;
* the relocation's own threshold is safe on BOTH sides: after it, no shift <= M switches any row;
* rows far from their boundary, and A = 0 rows, keep the golden's old log-prob bit-exactly;
* the margin rule passes inside the margin and refuses beyond it by the arm's name;
* the margin is its derivation;
* the gate relocates on EVERY rung's own weights and says so on its rule line (fails on revert);
* a planted 10 % backward miscompile on a pointer-head parameter is still refused at the fp32 rule (teeth intact).

The FATAL's own weights (the dithered null 18 % -> 0) are `compile_regions_clip_margin_fatal_test.py` (slow; needs the
run archive)."""
from __future__ import annotations

import math

import pytest
import torch

from agents.model import compile_control as cc
from agents.model import compile_regions as cr
from agents.model import compile_trainer as ct
from agents.model import trunk_precision as TP

CLIP = 0.15
M = cr.R1_CLIP_MARGIN


def _surrogate_grad(lp: torch.Tensor, old: torch.Tensor, adv: torch.Tensor, clip: float) -> torch.Tensor:
    """d/d(log pi) of `micro_step`'s policy loss, per row (its own expression, verbatim)."""
    lp = lp.clone().double().requires_grad_(True)
    ratio = torch.exp(lp - old.double())
    l1 = adv.double() * ratio
    l2 = adv.double() * torch.clamp(ratio, 1 - clip, 1 + clip)
    (-torch.min(l1, l2).sum()).backward()
    assert lp.grad is not None
    return lp.grad


def test_clip_active_IS_the_surrogates_gradient_switch():
    g = torch.Generator().manual_seed(11)
    n = 4000
    lp = -torch.rand(n, generator=g, dtype=torch.float64) * 3
    log_ratio = (torch.rand(n, generator=g, dtype=torch.float64) - 0.5) * 3          # rho 0.22 .. 4.5
    adv = torch.randn(n, generator=g, dtype=torch.float64)
    old = lp - log_ratio
    # rule 8: no row within rounding of a bound, and no A = 0 row, decides this check
    near = torch.minimum((log_ratio - math.log(1 + CLIP)).abs(), (log_ratio - math.log(1 - CLIP)).abs()) < 1e-6
    keep = ~near & (adv.abs() > 1e-6)
    lp, old, adv = lp[keep], old[keep], adv[keep]
    grad_on = _surrogate_grad(lp, old, adv, CLIP).ne(0)
    assert torch.equal(cr.clip_active(lp, old, adv, CLIP), grad_on)
    assert 0 < int(grad_on.sum()) < int(keep.sum())                    # both branches exercised


def _rows():
    """Four rows: AT the A > 0 switching bound, AT the A < 0 one, far on the active side, far on the inactive side,
    plus an A = 0 row at a bound."""
    lp = torch.tensor([-1.0, -0.5, -2.0, -1.5, -0.7], dtype=torch.float64)
    log_ratio = torch.tensor([math.log(1 + CLIP), math.log(1 - CLIP), -2.0, 3.0, math.log(1 + CLIP)],
                             dtype=torch.float64)
    adv = torch.tensor([1.3, -0.8, 0.5, 0.9, 0.0], dtype=torch.float64)
    return lp, (lp - log_ratio).float(), adv


def test_a_row_exactly_AT_its_switching_boundary_is_moved_2M_inside_DETERMINISTICALLY():
    lp, old, adv = _rows()
    fix = cr.clip_relocation(lp, old, adv, CLIP)
    again = cr.clip_relocation(lp, old, adv, CLIP)
    assert torch.equal(fix.old_log_prob, again.old_log_prob) and torch.equal(fix.moved, again.moved)
    assert fix.moved.tolist() == [True, True, False, False, False]
    lr = lp - fix.old_log_prob.double()
    assert float(lr[0]) == pytest.approx(math.log(1 + CLIP) - 2 * M, abs=1e-6)      # A > 0: below 1 + clip
    assert float(lr[1]) == pytest.approx(math.log(1 - CLIP) + 2 * M, abs=1e-6)      # A < 0: above 1 - clip
    assert cr.clip_active(lp, fix.old_log_prob, adv, CLIP)[:2].tolist() == [True, True]
    assert torch.equal(fix.old_log_prob[2:], old[2:])                  # far rows and the A = 0 row: untouched
    assert fix.old_log_prob.dtype == old.dtype
    assert "2/5 boundary rows relocated" in fix.line()


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_the_relocation_threshold_is_SAFE_ON_BOTH_SIDES(sign):
    """A row at distance exactly M is moved (to 2M); one just beyond stays (at > M). Either way, after the relocation
    no arm shift of at most M (in either direction) switches ANY row's clip branch — the threshold is not a verdict
    boundary."""
    a = 1.0 if sign > 0 else -1.0
    logb = math.log(1 + CLIP) if a > 0 else math.log(1 - CLIP)
    d = torch.tensor([0.0, 0.3 * M, M, M + 1e-3, 1.5 * M, 3 * M], dtype=torch.float64)
    dists = torch.cat([d, -d])                                         # both sides of the bound
    lp = torch.full_like(dists, -1.2)
    old = (lp - (logb + dists)).float()
    adv = torch.full_like(dists, a)
    fix = cr.clip_relocation(lp, old, adv, CLIP)
    moved = fix.moved.tolist()
    for i, dv in enumerate(dists.abs().tolist()):
        if abs(dv - M) > 1e-4:                     # rule 8: a row AT the threshold may go either way (both are safe)
            assert moved[i] == (dv < M), (i, dv)
    after = (lp - fix.old_log_prob.double() - logb).abs()
    assert float(after.min()) > M * (1 - 1e-5)
    act = cr.clip_active(lp, fix.old_log_prob, adv, CLIP)
    for shift in (0.999 * M, -0.999 * M):
        assert torch.equal(cr.clip_active(lp + shift, fix.old_log_prob, adv, CLIP), act)
    assert fix.nearest_unmoved > M


def test_the_margin_rule_PASSES_inside_the_margin_and_REFUSES_an_arm_beyond_it_by_name():
    lp, old, adv = _rows()
    fix = cr.clip_relocation(lp, old, adv, CLIP)
    ok = {"eager": {"log_prob": (lp + 0.5 * M).float()}, "compiled": {"log_prob": (lp - 0.5 * M).float()}}
    w = cr.GradWatch(("pointer_head.",))
    line = cr._clip_margin_rule(fix, ok, w)
    assert "0 clip-branch switches" in line and "eager" in line and "compiled" in line
    rec = w.record()["clip_margin"]
    assert rec["relocated_rows"] == 2 and rec["rows"] == 5
    assert rec["max_abs_dlogp"]["compiled"] == pytest.approx(0.5 * M, rel=1e-3)
    bad = dict(ok, compiled={"log_prob": (lp + 1.2 * M).float()})
    with pytest.raises(ct.CompileTrainerError, match=r"the compiled arm moved a fixture row's log-prob"):
        cr._clip_margin_rule(fix, bad)


def test_the_margin_IS_its_derivation():
    assert cr.R1_CLIP_MARGIN == cr.R1_BAR_K * cr.R1_LOGPROB_SHIFT_MAX
    assert cr.R1_LOGPROB_SHIFT_MAX == 0.18 and cr.R1_CLIP_MARGIN == pytest.approx(0.72)


# ------------------------------------------------------------------------------------- the gate on real rows
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


def _install(model):
    cc.control().install()
    cr.install(model, backend="eager")


@pytest.fixture
def fresh_learner(monkeypatch):
    """The K9 golden learner at a FRESH launch's weights (no seeded perturbation: the zero-init pointer head), so the
    gate runs two rungs on two weight states — the live weights and its own seeded perturbation."""
    from agents.model import parity_probe as PP
    monkeypatch.setattr(PP, "PERTURB_SCALE", 0.0)
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


@pytest.mark.slow  # ship gate (2026-10-19): 34 s beside a live run
@pytest.mark.parametrize("precision", ["bf16", "fp32"])
def test_the_gate_RELOCATES_on_every_rungs_own_weights_and_says_so(fresh_learner, monkeypatch, precision):
    """Fails on revert: every R1 rung's line names its relocation and its margin check, and the relocation runs once
    per rung (the fresh launch's live and perturbed rungs), each on that rung's weights."""
    learner = fresh_learner
    TP.set_trunk_precision(learner.policy, precision)
    _install(learner)
    seen = []
    real = cr.relocate_clip_boundary

    def spy(model, args, margin=cr.R1_CLIP_MARGIN):
        seen.append(float(next(model.policy.parameters()).detach().double().sum()))
        return real(model, args, margin)
    monkeypatch.setattr(cr, "relocate_clip_boundary", spy)
    rules = cr.gate_regions(learner, batch_size=16, say=lambda _m: None)
    r1 = [r for r in rules if r.startswith("R1 ")]
    assert cr.weights_regime(learner) == "fresh" and len(r1) == 2, rules       # PRECONDITION: live + perturbed
    assert all("clip margin:" in r and "0 clip-branch switches" in r for r in r1), r1
    assert len(seen) == 2 and seen[0] != seen[1]                       # the perturbed rung relocated on ITS weights


@pytest.mark.slow  # ship gate (2026-10-19): 28 s beside a live run
def test_a_planted_10pct_POINTER_HEAD_miscompile_is_still_refused_at_the_fp32_rule(learner, monkeypatch):
    """Teeth intact on the relocated rows: the fp32 rule refuses a 10 % backward error on the pointer head's move
    projection (the parameter the s4001 FATAL named)."""
    from agents.model import compile_regions_resolution_test as RT
    TP.set_trunk_precision(learner.policy, "fp32")
    _install(learner)
    cr.gate_regions(learner, batch_size=16, say=lambda _m: None)       # honest: passes
    with monkeypatch.context() as mp:
        RT.plant(mp, "pointer_head.move_proj.weight", 0.10)
        with pytest.raises(ct.CompileTrainerError, match=r"DISAGREES with eager.*pointer_head\.move_proj\.weight"):
            cr.gate_regions(learner, batch_size=16, say=lambda _m: None)
