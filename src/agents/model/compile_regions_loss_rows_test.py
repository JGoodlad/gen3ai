"""The bf16 rule's LOSS comparisons read the golden's OWN rows (`gen3_r1_loss_on_golden_rows_v1`, 2026-10-17;
`designs/training/compile_flags.md` "bf16 drift on a sharpening policy").

The clip relocation (T36) moves an A < 0 fixture row to rho = 0.85 e^(2M) so its GRADIENT cannot switch branch; the
LOSS is continuous in rho and needs no relocation, but the move multiplies that row's weight in the loss (rho |A|) by
~4x, and the drift / envelope bars were derived on the golden's own rows before the relocation existed. On X26 one such
row carried 88-100 % of the startup gate's loss drift from 116M on. So:

* `golden_row_loss` IS the loss a forward on the golden's own rows gives (K9 learner, bf16 and fp32 trunk; one
  micro-batch and the accumulation rung's two) — the reconstruction is exact, not an approximation;
* the gate hands the bf16 rule those losses (a spy on `_r1_verdict_bf16`), and the rule line names them;
* a relocation-AMPLIFIED healthy shift on one row passes on the golden's rows and is refused on the relocated ones (the
  X26 case, on hand-built arms: fails on revert of the reading);
* THE TEETH are the derived ones: a broad log-prob shift, a non-policy loss fault in the eager arm (DRIFT) and one in
  the compiled arm (ENVELOPE) are still refused.
"""
from __future__ import annotations

import math

import pytest
import torch

from agents.model import compile_control as cc
from agents.model import compile_regions as cr
from agents.model import compile_trainer as ct
from agents.model import trunk_precision as TP
from agents.training.instrumented_ppo.micro_step import micro_step

BATCH = 16


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


def _relocated(model, slice_=0, margin=2.0):
    """R1's args on golden slice ``slice_``, relocated at a WIDE margin (so many rows move), and the record."""
    args = cr._r1_args(model, cr.r1_batch(model, BATCH, slice_))
    new, fix = cr.relocate_clip_boundary(model, args, margin=margin)
    return args, new, fix


def _arm(model, args):
    model.policy.set_training_mode(True)
    with torch.no_grad():
        out = micro_step(*args)
    return {"loss": out.loss.detach().float().reshape(1), "log_prob": out.log_prob.detach().float().reshape(-1)}


@pytest.mark.parametrize("precision", ["bf16", "fp32"])
def test_golden_row_loss_IS_the_loss_of_a_forward_on_the_golden_rows(learner, precision):
    TP.set_trunk_precision(learner.policy, precision)
    raw_args, args, fix = _relocated(learner)
    moved = int(fix.moved.sum())
    assert moved > 0, "precondition: the wide margin relocates rows"
    on_reloc, on_raw = _arm(learner, args), _arm(learner, raw_args)
    got = cr.golden_row_loss(on_reloc, fix, args[-1])
    want = float(on_raw["loss"])
    # precondition: the relocation moves the loss far more than the reconstruction's float rounding
    assert abs(float(on_reloc["loss"]) - want) > 1e-3, (float(on_reloc["loss"]), want)
    assert abs(got - want) <= 1e-5 * max(1.0, abs(want)), (got, want, moved)


def test_the_ACCUMULATION_rung_sums_its_two_micro_batches_on_their_own_rows(learner):
    TP.set_trunk_precision(learner.policy, "bf16")
    parts = [_relocated(learner, s) for s in (0, 1)]
    fixes = [f for _, _, f in parts]
    fix = cr.ClipRelocation(torch.cat([f.old_log_prob for f in fixes]), torch.cat([f.moved for f in fixes]),
                            torch.cat([f.ref_log_prob for f in fixes]), torch.cat([f.advantages for f in fixes]),
                            fixes[0].clip_range, fixes[0].margin, 0.0, "default",
                            torch.cat([f.raw_old_log_prob for f in fixes]))
    arms = [_arm(learner, a) for _, a, _ in parts]
    acc = {"loss": torch.stack([a["loss"] for a in arms]).sum().reshape(1),
           "log_prob": torch.cat([a["log_prob"] for a in arms])}
    got = cr.golden_row_losses({"x": acc}, fix, [a for _, a, _ in parts])["x"]
    want = sum(float(_arm(learner, r)["loss"]) for r, _, _ in parts)
    assert abs(got - want) <= 1e-5 * max(1.0, abs(want)), (got, want)


@pytest.mark.slow  # ship gate (2026-10-19): 71 s beside a live run
def test_the_gate_hands_the_bf16_rule_the_GOLDEN_row_losses(learner, monkeypatch):
    monkeypatch.setattr(cr, "BF16_DRIFT_BARS", {"loss": 1.0, "grad_rel_l2": 5.0})
    monkeypatch.setattr(cr, "BF16_ENVELOPE_K", {"loss": 4.0, "grad_rel_l2": 4.0, "param": 4.0})
    TP.set_trunk_precision(learner.policy, "bf16")
    learner._canary_shift_max = 0.5                     # M = 2.0: rows move, so the two readings differ
    cc.control().install()
    cr.install(learner, backend="eager")
    seen = []
    real = cr._r1_verdict_bf16

    def spy(eager, comp, ref, names, regime, **kw):
        seen.append(kw.get("losses"))
        return real(eager, comp, ref, names, regime, **kw)
    monkeypatch.setattr(cr, "_r1_verdict_bf16", spy)
    rules = cr.gate_regions(learner, batch_size=BATCH, say=lambda _m: None)
    assert seen and all(x is not None and len(x) == 3 for x in seen), seen
    assert any("loss on the golden's own rows; relocated rows: drift" in r for r in rules), rules


# ------------------------------------------------------------------------------- the rule on hand-built arms
CLIP = 0.15


def _pg(lp, old, adv):
    rho = torch.exp(lp - old)
    return float(-torch.min(adv * rho, adv * rho.clamp(1 - CLIP, 1 + CLIP)).mean())


def _case(delta_rows, rest=1.0, eager_extra=0.0, comp_extra=0.0):
    """Eight rows; row 0 has the fixture's largest |A| (A < 0) at raw rho 0.9, within the margin of its switching
    bound 0.85, so the relocation moves it to rho 0.85 e^(2 x 0.72) = 3.59 — X26's row 9. Arms: the fp32 reference,
    eager bf16 = reference + ``delta_rows`` on the log-probs (+ ``eager_extra`` on a non-policy term), compiled =
    eager (+ ``comp_extra``). Losses = the policy term on the RELOCATED rows + ``rest`` (as the arms report them)."""
    adv = torch.tensor([-3.2, 0.9, -0.4, 0.6, 1.1, -0.7, 0.8, 0.9], dtype=torch.float64)
    ref_lp = torch.tensor([-1.0, -1.3, -0.5, -2.0, -0.9, -1.6, -0.4, -1.2], dtype=torch.float64)
    raw_old = ref_lp - torch.log(torch.tensor([0.9, 0.5, 2.5, 0.3, 0.4, 3.0, 0.45, 0.35], dtype=torch.float64))
    fix = cr.clip_relocation(ref_lp, raw_old, adv, CLIP, cr.R1_CLIP_MARGIN)
    assert bool(fix.moved[0]) and int(fix.moved.sum()) == 1, fix.moved           # only row 0 relocates
    old = fix.old_log_prob.double()
    e_lp = ref_lp + delta_rows
    g = torch.linspace(1.0, 2.0, 30)
    sizes = torch.tensor([10, 10, 10])

    def arm(lp, extra, gscale):
        return {"loss": torch.tensor([_pg(lp, old, adv) + rest + extra]), "log_prob": lp.float(),
                "grad": g * gscale, "grad_sizes": sizes}
    ref, eager, comp = arm(ref_lp, 0.0, 1.0), arm(e_lp, eager_extra, 1.001), arm(e_lp, eager_extra + comp_extra, 1.001)
    st = type("St", (), {"policy_grad_coef": 1.0, "fork_pg_mask": False})()
    gl = cr.golden_row_losses({"ref": ref, "eager": eager, "comp": comp}, fix, [(None, st)])
    return ref, eager, comp, (gl["ref"], gl["eager"], gl["comp"]), fix


def _verdict(ref, eager, comp, losses):
    return cr._r1_verdict_bf16(eager, comp, ref, ["a", "b", "c"], "trained", losses=losses)


def test_a_relocation_AMPLIFIED_one_row_shift_passes_on_the_golden_rows_and_is_refused_on_the_relocated_ones():
    d = torch.zeros(8, dtype=torch.float64)
    d[0] = 0.08                                     # a healthy single-row shift (X26's row 9: up to 0.09 at 118M)
    ref, eager, comp, losses, _ = _case(d, rest=3.0)
    bar = cr.BF16_DRIFT_BARS["loss"]
    reloc = cr._rel(float(eager["loss"]), float(ref["loss"]))
    golden = cr._rel(losses[1], losses[0])
    # preconditions, each far from the bar (standing rule 8): the relocated reading is over it, the golden one under
    assert reloc > 1.5 * bar and golden < 0.67 * bar, (reloc, golden, bar)
    line = _verdict(ref, eager, comp, losses)
    assert "loss on the golden's own rows" in line, line
    with pytest.raises(ct.CompileTrainerError, match="DRIFTED"):
        _verdict(ref, eager, comp, None)            # the pre-2026-10-17 reading


def test_THE_TEETH_a_broad_log_prob_shift_is_still_a_DRIFT_refusal():
    ref, eager, comp, losses, _ = _case(torch.full((8,), 0.2, dtype=torch.float64))
    assert cr._rel(losses[1], losses[0]) > 1.5 * cr.BF16_DRIFT_BARS["loss"]
    with pytest.raises(ct.CompileTrainerError, match="DRIFTED.*loss on the golden's own rows"):
        _verdict(ref, eager, comp, losses)


def test_THE_TEETH_a_non_policy_loss_fault_in_the_eager_arm_is_still_a_DRIFT_refusal():
    ref, eager, comp, losses, _ = _case(torch.zeros(8, dtype=torch.float64), eager_extra=0.05)
    with pytest.raises(ct.CompileTrainerError, match="DRIFTED"):
        _verdict(ref, eager, comp, losses)


def test_THE_TEETH_a_compiled_only_loss_fault_is_still_an_ENVELOPE_refusal():
    ref, eager, comp, losses, _ = _case(torch.zeros(8, dtype=torch.float64), comp_extra=0.05)
    floor_bar = cr.BF16_ENVELOPE_K["loss"] * cr.BF16_LOSS_FLOOR
    assert cr._rel(losses[2], losses[0]) > 1.5 * floor_bar and math.isfinite(losses[2])
    with pytest.raises(ct.CompileTrainerError, match="COMPILED bf16 micro-step is farther"):
        _verdict(ref, eager, comp, losses)


def test_a_record_without_raw_rows_is_REFUSED_by_name():
    ref, eager, comp, _, fix = _case(torch.zeros(8, dtype=torch.float64))
    st = type("St", (), {"policy_grad_coef": 1.0, "fork_pg_mask": False})()
    with pytest.raises(ct.CompileTrainerError, match="no raw old log-probs"):
        cr.golden_row_loss(eager, fix._replace(raw_old_log_prob=None), st)
