"""Region R1's UNMOVED parameters, by the INIT RECORD (`gen3_r1_unmoved_init_v1`, 2026-10-04).

THE GAP. `gen3_r1_unmoved_param_v1` (8b8fbac0) judged an unmoved parameter at the FRESH bar only when
every element was exactly 0.0. A parameter training never moved but that starts at a NON-ZERO init was
still judged at the model's TRAINED bar. In `rb_x5ab_oracle_sp_s1001`'s FATAL checkpoint, seven
parameters were bit-identical to the fresh build at seed 1001: the belief head's ortho-init
`moves_head.weight`, its LayerNorm weight, `belief_slots.unknown_slot_emb`, and the zero ones
(`designs/research_state/measurements/oracle_canary_2026-10-04/unmoved_init_species_fatal.json`).

THE RULE. A parameter is UNMOVED iff every element is bit-identical to its value at the run's fresh
build (its sha256 equals the init record's: `compile_regions.record_param_init`, written by the
trainer's fresh construction and carried in every checkpoint), or every element is exactly 0.0. A
checkpoint with no record (saved before this rule) gets the zero rule alone.

The fixture stands in for the oracle run. The golden learner's weights are taken as the "fresh build"
(the record is taken there), then every parameter EXCEPT the belief moves head is moved by name-keyed
seeded noise (training). `moves_head.weight` is ortho-init and non-zero, so only the record can call
it unmoved. Faults are PLANTED on R1's compiled arm through `compile_regions._r1_arm`, as in
`compile_regions_unmoved_test`:

  * NOISE: a relative error BETWEEN the two bars on the head's gradient, present only while the head
    is still at its init;
  * MISCOMPILE: the same error on every call, whatever the weights.
"""
from __future__ import annotations

import pytest
import torch

from agents.model import compile_control as cc
from agents.model import compile_regions as cr
from agents.model import compile_trainer as ct

BATCH = 16
MW = "features_extractor.belief_head.moves_head.weight"
MB = "features_extractor.belief_head.moves_head.bias"
#: moved by the fixture's "training": a judged parameter that must be judged exactly as before
MOVED = "features_extractor.belief_head.species_head.weight"
#: between the trained bar (9.88e-3) and the fresh bar (0.1004): absorbed only by the fresh bar
EPS = 0.5 * (cr.R1_PARAM_BAR["trained"] + cr.R1_PARAM_BAR["fresh"])


def _train_all_but(model, keep) -> None:
    """'Training': move every policy parameter not in ``keep`` by name-keyed seeded noise."""
    from agents.model.parity_probe import _keyed_noise
    noise = _keyed_noise(model.policy, 20261004, 1e-3)
    with torch.no_grad():
        for n, p in model.policy.named_parameters():
            if n not in keep:
                p.add_(noise[n])


def _learner(tmp_path):
    from agents.training import learner_golden as LG
    m = LG.build_learner()
    LG.load_buffer_into(m)
    m.behaviour_dump_dir = str(tmp_path)
    cr.record_param_init(m)                       # the run's fresh build
    _train_all_but(m, {MW, MB})
    return m


@pytest.fixture
def dead_ortho_head(tmp_path):
    """The golden learner, R1 installed under dynamo's `eager` backend, trained except the belief moves
    head, which is still bit-identical to its (non-zero) init: the oracle run's state."""
    cc._reset_control_for_tests()
    torch._dynamo.reset()
    m = _learner(tmp_path)
    cc.control().install()
    cr.install(m, backend="eager")
    try:
        yield m
    finally:
        cr.uninstall(m)
        cc._reset_control_for_tests()
        torch._dynamo.config.error_on_recompile = False
        torch._dynamo.reset()


def _at_init(model, name) -> bool:
    rec = cr.init_record(model)
    p = dict(model.policy.named_parameters())[name]
    return rec is not None and cr.param_sha256(p) == rec[name]


def _plant(monkeypatch, name: str, *, only_at_init: bool) -> None:
    """Add a relative error of exactly ``EPS`` to ``name``'s gradient on R1's COMPILED arm: only while
    ``name`` is bit-identical to its init record (NOISE), or always (MISCOMPILE)."""
    real = cr._r1_arm

    def arm(model, fn, args):
        out = real(model, fn, args)
        if fn is model._compiled_micro_step and (not only_at_init or _at_init(model, name)):
            names = [n for n, _ in ct.grad_parameters(model, model.policy.features_extractor)]
            sizes = [int(x) for x in out["grad_sizes"].tolist()]
            i = names.index(name)
            start = sum(sizes[:i])
            g = out["grad"].clone()
            seg = g[start:start + sizes[i]]
            sign = torch.where(torch.arange(seg.numel()) % 2 == 0, 1.0, -1.0).to(seg)
            g[start:start + sizes[i]] = seg + EPS * float(seg.norm()) * sign / float(sign.norm())
            out = {**out, "grad": g}
        return out
    monkeypatch.setattr(cr, "_r1_arm", arm)


def _judged(model, name) -> bool:
    """PRECONDITION: the golden rows put ``name``'s gradient above the per-parameter floor (it is judged
    at all), and the model's regime is TRAINED."""
    from agents.model.compile_gate_probe import per_param_grad_errors
    from agents.training.instrumented_ppo.micro_step import micro_step
    model.policy.set_training_mode(True)
    e = cr._r1_arm(model, micro_step, cr._r1_args(model, cr.r1_batch(model, BATCH)))
    names = [n for n, _ in ct.grad_parameters(model, model.policy.features_extractor)]
    sizes = [int(x) for x in e["grad_sizes"].tolist()]
    judged = dict(per_param_grad_errors(e["grad"], e["grad"], sizes, floor_frac=ct._PARAM_GRAD_FLOOR))
    return names.index(name) in judged and cr.weights_regime(model) == "trained"


def test_unmoved_is_BIT_IDENTICAL_to_the_init_record_or_exactly_zero(tmp_path):
    """Categorical: digest equality with the record, or every element == 0.0. One ulp on one element
    moves a parameter; with no record only the zero rule is left."""
    m = _learner(tmp_path)
    head = dict(m.policy.named_parameters())[MW]
    assert bool(head.detach().any()), "PRECONDITION: the head is non-zero, so only the record can see it"
    assert cr.unmoved_parameters(m) == [MW, MB]
    with torch.no_grad():
        head[0, 0] = torch.nextafter(head[0, 0], torch.tensor(float("inf")))
    assert cr.unmoved_parameters(m) == [MB]
    with torch.no_grad():
        dict(m.policy.named_parameters())[MB].zero_()     # moved off its init, but exactly 0.0
    assert cr.unmoved_parameters(m) == [MB]
    delattr(m, cr.INIT_RECORD_ATTR)
    assert cr.init_record(m) is None and cr.unmoved_parameters(m) == [MB]


def test_the_GATE_passes_a_dead_ortho_heads_noise_and_FATALs_its_miscompile(dead_ortho_head, monkeypatch):
    """Noise between the bars on the dead non-zero head passes (fails on revert: the zero rule alone
    judged it at the trained bar). The same error on every call is FATAL through the unmoved rung."""
    assert _judged(dead_ortho_head, MW)
    _plant(monkeypatch, MW, only_at_init=True)
    rules = cr.gate_regions(dead_ortho_head, batch_size=BATCH, say=lambda _m: None)
    assert any("declared-exception param(s)" in r and MW in r for r in rules), rules
    assert any("unmoved rule: bit-identical to the init record" in r for r in rules), rules
    assert any("unmoved param(s) perturbed" in r for r in rules), rules
    monkeypatch.undo()
    _plant(monkeypatch, MW, only_at_init=False)
    with pytest.raises(ct.CompileTrainerError, match="DISAGREES with eager.*moves_head"):
        cr.gate_regions(dead_ortho_head, batch_size=BATCH, say=lambda _m: None)


def test_a_MOVED_parameter_is_judged_exactly_as_before(dead_ortho_head, monkeypatch):
    """With a record present, a parameter training moved is not unmoved and keeps the TRAINED bar on
    the live weights: the same between-the-bars error on it is FATAL."""
    assert _judged(dead_ortho_head, MOVED)
    assert MOVED not in cr.unmoved_parameters(dead_ortho_head)
    _plant(monkeypatch, MOVED, only_at_init=False)
    with pytest.raises(ct.CompileTrainerError, match="DISAGREES with eager.*species_head"):
        cr.gate_regions(dead_ortho_head, batch_size=BATCH, say=lambda _m: None)


def test_a_RESUME_reads_the_record_from_the_checkpoint_and_without_one_falls_back_to_the_zero_rule(
        dead_ortho_head, monkeypatch, tmp_path):
    """The record is plain checkpoint data: save then load returns it unchanged, and the loaded model
    finds the same unmoved set. A checkpoint with no record (saved before the rule) gets the zero rule
    alone, so the dead non-zero head is judged at the trained bar again (8b8fbac0's behaviour)."""
    m = dead_ortho_head
    assert cr.INIT_RECORD_ATTR not in m._excluded_save_params()
    path = str(tmp_path / "ckpt")
    m.save(path)
    loaded = type(m).load(path + ".zip", env=None, device="cpu")
    assert cr.init_record(loaded) == cr.init_record(m)
    assert cr.unmoved_parameters(loaded) == [MW, MB]
    assert _judged(m, MW)
    delattr(m, cr.INIT_RECORD_ATTR)
    assert cr.unmoved_parameters(m) == []
    _plant(monkeypatch, MW, only_at_init=False)       # no record: nothing marks the head unmoved
    with pytest.raises(ct.CompileTrainerError, match="DISAGREES with eager.*moves_head"):
        cr.gate_regions(m, batch_size=BATCH, say=lambda _m: None)


def test_the_CANARY_passes_a_dead_ortho_heads_noise_and_FATALs_its_miscompile(dead_ortho_head, monkeypatch):
    from agents.model.compile_canary import CompileCanary, CompileCanaryError
    assert _judged(dead_ortho_head, MW)
    _plant(monkeypatch, MW, only_at_init=True)
    out = CompileCanary(dead_ortho_head, batch_size=BATCH, every=1, saver=lambda p: None).after_update()
    assert out["compile/canary_ok"] == 1.0 and out["compile/canary_unmoved_params"] == 2.0
    monkeypatch.undo()
    _plant(monkeypatch, MW, only_at_init=False)
    with pytest.raises(CompileCanaryError, match="CONFIRMED"):
        CompileCanary(dead_ortho_head, batch_size=BATCH, every=1, saver=lambda p: None).after_update()
