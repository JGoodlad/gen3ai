"""THE GATE RELEASES ITS MEMORY (`gen3_gate_releases_memory_v1`, 2026-10-15) and the canary's memory record
(`gen3_canary_memory_v1`).

The 2026-10-15 `rb_xs_s2701` launch (X26's argv at `71351ff5`) was refused by the update-fit check: the startup
gate's T36 relocation forward (an fp32 grad-mode forward at the micro size, no backward) left its outputs STASHED on
the policy, and through them its whole autograd graph (4,486 MiB on the X26 surface), alive under the compiled arm
that followed — and the allocator kept the gate's peak cached into the update-fit reading. These tests pin the
mechanism on CPU (the relocation keeps NOTHING for a backward and its numbers are unchanged; the gate and the canary
leave no module-level stash holding a graph; the dry canary records nothing) and the canary's first-check test hook.
The card-level numbers are GPU reads (`designs/training/compile_flags.md` "The gate releases its memory").
"""
from __future__ import annotations

import inspect

import pytest
import torch

from agents.model import compile_control as cc
from agents.model import compile_regions as cr
from agents.model import compile_trainer as ct


def _graph_stashes(policy):
    held = [(type(m).__name__, k) for m in policy.modules() for k, v in vars(m).items()
            if isinstance(v, torch.Tensor) and v.grad_fn is not None]
    if getattr(policy, "_last_pi_distribution", None) is not None:
        held.append(("policy", "_last_pi_distribution"))
    return held


@pytest.fixture
def learner(tmp_path):
    from agents.training import learner_golden as LG
    cc._reset_control_for_tests()
    torch._dynamo.reset()
    model = LG.build_learner()
    LG.load_buffer_into(model)
    model.behaviour_dump_dir = str(tmp_path)
    try:
        yield model
    finally:
        cr.uninstall(model)
        cc._reset_control_for_tests()
        torch._dynamo.config.error_on_recompile = False
        torch._dynamo.reset()


def _args(model, rows=16):
    model.policy.set_training_mode(True)
    return cr._r1_args(model, cr.r1_batch(model, rows))


class _CountSaved:
    """An OUTER `saved_tensors_hooks` pair that counts the bytes autograd keeps for a backward: a forward that installs
    its own (inner) hooks hides its saves from it — which is exactly what the relocation must do."""

    def __init__(self):
        self.n, self.bytes = 0, 0

    def pack(self, t):
        self.n += 1
        self.bytes += t.numel() * t.element_size()
        return t

    @staticmethod
    def unpack(t):
        return t


def test_the_relocation_forward_KEEPS_NOTHING_for_a_backward(learner):
    """THE FIX (gen3_gate_releases_memory_v1): the relocation's grad-mode fp32 forward is never backpropagated, so
    autograd keeps no saved tensor for it — on the GPU its graph was 4,486 MiB held (through the policy's stashes)
    under the compiled arm that followed. The PRECONDITION: the same forward without the hooks saves activations.
    Fails on revert."""
    from agents.model.trunk_precision import fp32_scope
    from agents.training.instrumented_ppo.micro_step import micro_step
    args = _args(learner)
    plain = _CountSaved()
    with fp32_scope(learner.policy), torch.autograd.graph.saved_tensors_hooks(plain.pack, plain.unpack):
        out = micro_step(*args)
    del out
    assert plain.n > 100 and plain.bytes > 0, "PRECONDITION: a grad-mode R1 forward saves its activations"
    seen = _CountSaved()
    with torch.autograd.graph.saved_tensors_hooks(seen.pack, seen.unpack):
        cr.relocate_clip_boundary(learner, args)
    assert (seen.n, seen.bytes) == (0, 0)


def test_the_relocations_record_is_bit_identical_to_a_plain_forwards(learner):
    """Dropping the saved tensors changes no value: the relocation's reference log-probs and moved rows equal a plain
    grad-mode fp32 forward's, bit for bit."""
    from agents.model.trunk_precision import fp32_scope
    from agents.training.instrumented_ppo.micro_step import micro_step
    args = _args(learner)
    with fp32_scope(learner.policy):
        out = micro_step(*args)
    _a, fix = cr.relocate_clip_boundary(learner, args)
    assert torch.equal(fix.ref_log_prob, out.log_prob.detach().double().cpu().reshape(-1))
    assert torch.equal(fix.advantages, out.advantages.detach().double().cpu().reshape(-1))


def test_the_release_changes_no_number(learner, monkeypatch):
    """The relocation's record and the arms' arithmetic are bit-identical with and without the release: a stash keeps
    its value (detached) and the next forward overwrites it."""
    from agents.training.instrumented_ppo.micro_step import micro_step
    args = _args(learner)
    a1, f1 = cr.relocate_clip_boundary(learner, args)
    e1 = cr._r1_arm(learner, micro_step, a1)
    monkeypatch.setattr(cr, "release_gate_memory", lambda model: 0)
    a2, f2 = cr.relocate_clip_boundary(learner, args)
    e2 = cr._r1_arm(learner, micro_step, a2)
    assert torch.equal(f1.old_log_prob, f2.old_log_prob) and torch.equal(f1.ref_log_prob, f2.ref_log_prob)
    assert torch.equal(f1.moved, f2.moved)
    for k in ("loss", "grad", "log_prob"):
        assert torch.equal(e1[k], e2[k]), k


def test_the_gate_leaves_no_graph_behind_and_says_where_its_memory_went(learner, monkeypatch):
    """The whole gate (both rungs of a fresh launch) ends with no stash holding a graph. Its memory line is a CUDA
    read (a no-op here: no row, no line)."""
    cc.control().install()
    cr.install(learner, backend="eager")
    said = []
    cr.gate_regions(learner, batch_size=16, say=said.append)
    assert not _graph_stashes(learner.policy)
    assert not any("gate memory" in s for s in said)            # CPU: nothing to read


def test_the_gate_memory_record_reads_retention_and_the_high_water_without_resetting_the_peak(monkeypatch):
    """`GateMemory`'s line from planted allocator reads (no card needed): allocated AFTER each step and the
    HIGH-WATER so far, and it never resets the allocator's peak counters (the CUDA ledger reads them across the
    gate)."""
    import types
    reads = iter([(1_000, 1_400), (3_800, 4_200), (3_900, 8_100), (1_300, 8_100), (1_200, 9_000)])
    cur = {}

    def nxt():
        cur["a"], cur["h"] = next(reads)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *a: nxt())
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda *a: cur["a"] * cr.MiB)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a: cur["h"] * cr.MiB)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda *a: 9_790 * cr.MiB)

    def no_reset(*a):
        raise AssertionError("GateMemory must never reset the allocator's peak counters")
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", no_reset)
    monkeypatch.setattr(ct, "resolve_device", lambda fe: torch.device("cuda"))
    model = types.SimpleNamespace(policy=types.SimpleNamespace(features_extractor=None))
    mem = cr.GateMemory(model)
    with cr.gate_memory(mem):
        for step in ("relocation forward", "compiled arm", "eager arm", "fp32 reference arm"):
            cr._mem(step)
    cr._mem("outside")                                         # not recorded: the context is closed
    assert [r[0] for r in mem.rows] == ["relocation forward", "compiled arm", "eager arm", "fp32 reference arm"]
    assert mem.peak_above_start() == 8_000.0
    line = mem.line()
    assert "relocation forward 3,800/4,200" in line and "transient above its start 8,000 MiB" in line


def test_the_sentinel_hands_the_cached_blocks_back_AFTER_the_lock():
    """gen3_gate_releases_memory_v1: `arm_compile_sentinel` releases the stashes and empties the allocator's cache
    once the gate, the prewarm and the lock are done — before the update-fit check measures. Fails on revert."""
    src = inspect.getsource(ct.arm_compile_sentinel)
    i_lock = src.index("ctl.lock(")
    assert i_lock < src.index("_cr.release_gate_memory(model)") < src.index("torch.cuda.empty_cache()")
    assert src.index("torch.cuda.empty_cache()") < src.index("ctl.canary = CompileCanary(")


# ------------------------------------------------------------------------------------------ the canary
@pytest.fixture
def regions_learner(learner):
    cc.control().install()
    cr.install(learner, backend="eager")
    return learner


def test_the_canary_check_leaves_no_graph_and_measure_changes_nothing(regions_learner, tmp_path):
    """`CompileCanary.measure` (the update-fit check's dry canary) runs the scheduled check's arms once and records
    nothing: no verdict file, no count, the weights and `.grad` as found, no stash holding a graph."""
    from agents.model.compile_canary import VERDICTS_FILE, CompileCanary
    from agents.training import learner_golden as LG
    canary = CompileCanary(regions_learner, batch_size=16, every=5)
    h0 = LG.params_sha256(regions_learner)
    assert canary.measure() is None
    assert (canary.updates, canary.runs, canary.unconfirmed) == (0, 0, 0)
    assert not (tmp_path / VERDICTS_FILE).exists()
    assert LG.params_sha256(regions_learner) == h0
    assert all(p.grad is None for p in regions_learner.policy.parameters())
    assert not _graph_stashes(regions_learner.policy)


def test_the_canary_first_check_test_hook(regions_learner, monkeypatch):
    """`$GEN3AI_CANARY_FIRST` moves only the FIRST scheduled check (a GPU smoke reads the canary at update 2); unset,
    the production cadence's first stays `CANARY_FIRST`; a bad value is refused, typed."""
    from agents.model import compile_canary as CC
    monkeypatch.delenv(CC.FIRST_ENV, raising=False)
    assert CC.CompileCanary(regions_learner, batch_size=16).first == CC.CANARY_FIRST
    monkeypatch.setenv(CC.FIRST_ENV, "2")
    c = CC.CompileCanary(regions_learner, batch_size=16)
    assert (c.first, c.every) == (2, CC.CANARY_EVERY)
    for bad in ("0", "x"):
        monkeypatch.setenv(CC.FIRST_ENV, bad)
        with pytest.raises(ct.CompileTrainerError, match=CC.FIRST_ENV):
            CC.CompileCanary(regions_learner, batch_size=16)
