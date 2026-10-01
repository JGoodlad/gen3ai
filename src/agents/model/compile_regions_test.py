"""K8 — the DECLARED REGIONS, as the routine compile INVENTORY test (`gen3_declared_regions_v1`).

The production-surface learner (the K9 golden's), CPU, torch 2.8, dynamo's `eager` backend (graphs,
breaks, guards and recompiles are dynamo's — independent of the backend; the CUDA Inductor compile is
the GPU tier's and the startup gate's): install R0 + R1 `fullgraph=True`, gate them against eager,
prewarm the declared signatures, LOCK, then run two REAL updates (every fold step, the eager tail, the
probes) and a rollout forward. The inventory must EQUAL the declaration table:

  * graphs == declared regions x signatures (`compile_regions.declared_signature_count`);
  * 0 undeclared breaks — guaranteed by `fullgraph=True` (a break is a compile error), and the teeth
    test below proves a break inside R1 is the typed STARTUP FATAL naming dynamo's reason;
  * 0 compiles / rejections after the lock, one cache entry per region code object.

torch 2.5.1 is legacy (regions are not supported there; `--compile-trainer` keeps the extractor-only
compile — a documented, declared exception): those tests are skipped on it, and one pins the legacy
routing."""
from __future__ import annotations

import pytest
import torch

from agents.model import compile_control as cc
from agents.model import compile_regions as cr
from agents.model import compile_trainer as ct

_28 = pytest.mark.skipif(not cr.regions_supported(), reason="declared regions are a torch 2.8 feature")

N_ENVS, BATCH = 4, 16


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


def _graphs() -> int:
    from torch._dynamo.utils import counters
    return int(counters["stats"]["unique_graphs"])


def test_the_declaration_table_is_well_formed():
    names = [r.name for r in cr.REGIONS]
    assert len(names) == len(set(names))
    for r in cr.REGIONS:
        assert r.why, r.name
        assert bool(r.signatures) == r.compiled, r.name        # compiled <=> it declares signatures
    assert {r.name for r in cr.compiled_regions()} == {"R0_rollout_forward", "R1_learner_micro_step"}
    assert cr.declared_signature_count(48) == 2
    assert cr.declared_signature_count(1) == 1                  # batch 1 is never compiled


@_28
def test_the_compiled_inventory_EQUALS_the_declaration_table(learner):
    model = learner
    ctl = cc.control()
    ctl.install()
    g0 = _graphs()
    assert cr.install(model, backend="eager") == ["R0_rollout_forward", "R1_learner_micro_step"]
    rules = cr.gate_regions(model, n_envs=N_ENVS, batch_size=BATCH, say=lambda _m: None)
    assert any(r.startswith("R1 ") for r in rules) and any(r.startswith("R0 ") for r in rules)
    ctl.prewarm(cr.prewarm_calls(model, n_envs=N_ENVS, batch_size=BATCH))
    ctl.lock("test: the end of startup")
    assert _graphs() - g0 == cr.declared_signature_count(N_ENVS)
    ent = cc.cache_entries_by_code()
    assert {k.split(" ")[0] for k in ent} == {"micro_step", "_rollout_core"}, ent
    assert all(v == 1 for v in ent.values()), ent
    from agents.training import learner_golden as LG
    for _ in range(2):                                      # two REAL updates: the fold, the tail, the probes
        LG.load_buffer_into(model)
        with ctl.guard("update end"):
            model.train()
    obs = ct._prewarm_obs(model, N_ENVS)
    _, mask = ct._parity_obs(int(obs["observation"].shape[-1]), N_ENVS, torch.device("cpu"))
    model.policy.set_training_mode(False)
    with ctl.guard("rollout"), torch.no_grad():
        model.policy(obs, action_masks=mask)               # the rollout forward through R0
        model.policy(ct._prewarm_obs(model, 1), action_masks=mask[:1])   # batch 1: eager, compiles nothing
    ctl.check("end")
    assert ctl.compiles_after_lock == 0 and ctl.rejected_after_lock == 0, ctl.after_lock_frames
    assert _graphs() - g0 == cr.declared_signature_count(N_ENVS)


@_28
def test_a_graph_break_INSIDE_a_region_is_the_typed_startup_FATAL(learner, monkeypatch):
    """`fullgraph=True` is the mechanism: a host read inside R1 must refuse at startup, naming
    dynamo's reason — never a silent split into two graphs. Fails if the region is compiled without
    `fullgraph=True` (the break would be absorbed) or the error escapes untyped (a restartable crash)."""
    from agents.training.instrumented_ppo import micro_step as ms
    real = ms.value_loss_from_se

    def with_a_host_read(se, w):
        if float(se.mean()) > 1e30:                        # a data-dependent Python branch
            return se.sum()
        return real(se, w)
    monkeypatch.setattr(ms, "value_loss_from_se", with_a_host_read)
    cc.control().install()
    cr.install(learner, backend="eager")
    with pytest.raises(ct.CompileTrainerError, match="DECLARED REGION does not compile as one graph"):
        cr.gate_regions(learner, n_envs=N_ENVS, batch_size=BATCH, say=lambda _m: None)


@_28
def test_an_UNDECLARED_signature_after_the_lock_names_the_failing_guard(learner):
    ctl = cc.control()
    ctl.install()
    cr.install(learner, backend="eager")
    ctl.prewarm(cr.prewarm_calls(learner, n_envs=N_ENVS, batch_size=BATCH))
    ctl.lock("test")
    args = cr._r1_args(learner, cr.r1_batch(learner, BATCH * 2))     # a batch the table never declared
    with pytest.raises(cc.CompileSentinelError, match="UNDECLARED SIGNATURE") as ei:
        with ctl.guard("update end"):
            learner._compiled_micro_step._gen3_compiled(*args)       # the compiled region itself
    assert "size mismatch" in str(ei.value) or "expected 16" in str(ei.value), str(ei.value)[-800:]


@_28
def test_a_RAGGED_micro_batch_takes_the_declared_eager_route(learner):
    """The last micro-batch of an epoch over a rollout that does not divide evenly (fork rows, an
    uneven sizing) has another row count: through R1's dispatcher it runs the SAME function eager —
    no compile after the lock, the same loss as the eager function. Fails if the dispatcher hands it
    to the compiled region (an undeclared signature -> the sentinel's FATAL)."""
    from agents.training.instrumented_ppo.micro_step import micro_step
    ctl = cc.control()
    ctl.install()
    cr.install(learner, backend="eager")
    ctl.prewarm(cr.prewarm_calls(learner, n_envs=N_ENVS, batch_size=BATCH))
    ctl.lock("test")
    args = cr._r1_args(learner, cr.r1_batch(learner, BATCH - 5))
    with ctl.guard("update end"):
        out = learner._compiled_micro_step(*args)
    ctl.check("end")
    assert ctl.compiles_after_lock == 0 and ctl.rejected_after_lock == 0
    assert torch.equal(out.loss.detach(), micro_step(*args).loss.detach())


def test_torch_2_5_1_keeps_the_legacy_extractor_compile():
    """The declared, documented exception: on 2.5.1 (`forward_guard`'s weakref lookup breaks
    `fullgraph=True` there) the region path is not taken."""
    assert cr.regions_supported("2.8.0+cu126")
    assert not cr.regions_supported("2.5.1+cu121")
