"""PERFORMANCE-SHAPE tests — deterministic, functional pins of the properties that make a PPO update fast.

Owner, 2026-10-02: a wall-clock gate is the wrong tool ("noisy, needs an idle box and goes stale" — the
retired `compiled_perf_guard_test` died with its TF32 baseline). Speed is checked by (a) tests of the
STRUCTURE that produces it, asserted as counts and graph facts, never as a time, in every routine gate,
and (b) the on-demand update benchmark at milestones (`designs/ops/testing.md` "Performance-shape tests").
This file holds the ones the compile / lifecycle suites did not already pin. Each carries a TEETH test:
the regression is planted through the same seam production reads and the pin must see it.

  1. every micro-batch of a REAL update runs R1 through its COMPILED route, none eager — the counters
     the run logs as `lifecycle/compiled_region_calls` / `eager_fallback_calls`. A dispatcher bypass
     that runs the eager body uncounted (the 2.9x regression the benchmark read on 2026-10-02) leaves
     every run-level FATAL silent and is caught only here.
  2. a real update makes a BOUNDED number of host scalar reads (`.item()`, `float(tensor)`), each a
     device sync. K8's regions took them from 66,784 to 9,942 per production update (-10 s); a read
     added to the per-micro-batch path is x480 there.
  3. attention lowers to the FUSED SDPA kernel, never Inductor's MATH decomposition (bmm + safe-softmax +
     bmm; forcing MATH inside R1 measured +5.2% of its fwd+bwd, `measurements/k6_k8/r1_noise/`). Two
     cells: the CUDA cell asks the card's own dispatcher about the trunk layer (`_scaled_dot_product_
     efficient_attention`; GPU tier), and the CPU cell pins the ATTENTION PROFILE of R1's AOT forward graph
     (routine tier). On CPU a gradient-requiring bias is not eligible for the fused CPU kernel, so the
     trunk's two calls are MATH there BY CONSTRUCTION and the two fused ops are other modules' — the profile
     is a structural snapshot, and a change to it (a global flag, a hand-rolled attention, a forced MATH
     context) is a reviewed edit, not a silent one.

Already pinned elsewhere (the audit, `designs/ops/testing.md`): inventory == declaration and 0
compiles after the lock (`compile_regions_test`, `compile_control_test`), the no-silent-eager FATALs
(`compile_regions_test`), no CUDA stream / optimizer / parameter built after startup
(`learner_lifecycle_test`, `learner_lifecycle_gate_test`), the device-resident micro-batch
(`instrumented_ppo_device_batches_test`)."""
from __future__ import annotations

import contextlib
import math
from typing import Any, Dict, Iterator, List

import numpy as np
import pytest
import torch
from torch._functorch.aot_autograd import aot_module_simplified
from torch.nn.attention import SDPBackend, sdpa_kernel

from agents.model import compile_control as cc
from agents.model import compile_regions as cr
from agents.model import region_calls as RC
from agents.model.extractor_compiles_test import _cuda_skip_reason
from agents.model.team_transformer import BiasedEncoderLayer

N_ENVS, BATCH = 4, 16


@pytest.fixture
def learner() -> Iterator[Any]:
    """The production-surface learner (the K9 golden's), CPU, R1 installed under dynamo's `eager`
    backend, prewarmed and LOCKED — `compile_regions_test`'s own setup minus the parity gate (these pins
    read counts, not numbers)."""
    from agents.training import learner_golden as LG
    cc._reset_control_for_tests()
    torch._dynamo.reset()
    model = LG.build_learner()
    LG.load_buffer_into(model)
    ctl = cc.control()
    ctl.install()
    cr.install(model, backend="eager")
    ctl.prewarm(cr.prewarm_calls(model, batch_size=BATCH))
    ctl.lock("test: the end of startup")
    try:
        yield model
    finally:
        cr.uninstall(model)
        cc._reset_control_for_tests()
        torch._dynamo.config.error_on_recompile = False
        torch._dynamo.reset()


def _real_update(model: Any, meter: Dict[str, float]) -> Dict[str, int]:
    """ONE real `train()` on the golden buffer, seeds fixed (the permutation and every data-dependent
    branch are then deterministic); returns the update's region route counts and fills ``meter``."""
    from agents.training import learner_benchmark as lb
    from agents.training import learner_golden as LG
    LG.load_buffer_into(model)
    np.random.seed(LG.UPDATE_SEED)
    torch.manual_seed(LG.UPDATE_SEED)
    RC.take()                                              # a clean per-update window
    with lb._scalar_read_meter(meter):
        model.train()
    return RC.take()


def _micro_batches(model: Any) -> int:
    rb = model.rollout_buffer
    rows = int(rb.buffer_size) * int(rb.n_envs)
    return math.ceil(rows / int(model.batch_size)) * int(model.n_epochs)


# ---------------------------------------------------------------- 1. every micro-batch runs R1 compiled
def test_a_real_update_runs_every_micro_batch_through_R1_compiled_and_none_eager(learner, monkeypatch):
    """The golden's 64 rows / micro-batch 16 / 2 epochs = 8 micro-batches, ALL through R1's compiled
    route, none through the declared eager route (the rows divide evenly: no ragged tail).

    TEETH, in the same test (one fixture build): the regression the benchmark read at 2.9x (116 s against
    40 s, compiled share 0.000) — R1 routed AROUND its dispatcher, running the eager body uncounted — leaves
    every run-level FATAL silent (no counter, no inventory change after the lock) and must show here."""
    from agents.training.instrumented_ppo.micro_step import micro_step
    routes = _real_update(learner, {"s": 0.0, "n": 0})
    assert _micro_batches(learner) == 8
    assert routes == {"R1_compiled": 8}, routes
    with monkeypatch.context() as m:
        m.setattr(learner, "_compiled_micro_step", lambda *a, **k: micro_step(*a, **k))
        bypassed = _real_update(learner, {"s": 0.0, "n": 0})
    assert bypassed.get("R1_compiled", 0) == 0 and bypassed != routes, bypassed


# ---------------------------------------------------------------- 2. a bounded host-scalar-read budget
#: Host scalar reads (`Tensor.item` + `float(tensor)`: each blocks the host on the device) in ONE real
#: update of the golden learner — 8 micro-batches, the diagnostic (first) update, every probe on —
#: through the LOCKED regions. Measured 2026-10-02: 1,050, identical on three consecutive updates. An
#: EXACT pin on purpose: a count is deterministic, so any change is a reviewed one-line edit — a lower
#: number is a win to write down, a higher one is a host sync somebody added (x480 micro-batches in a
#: production update). Production, diag-skipped: 9,942 reads per update (K8 acceptance), was 66,784.
#: 1,050 → 1,060 at the X5 VERSION BREAK (2026-10-07): the golden learner became X5's surface. MEASURED: 988 of
#: the 1,060 are `optimizer.step()`'s per-parameter step-count reads (`ppo.py`'s two step sites, 494 each), which
#: scale with the parameter TENSOR count X5 changes; no read sits in the micro-step (the TEETH below still add
#: exactly one per micro-batch). The blob learner's per-site split was not re-measured (blob is deleted).
#: 1,060 → 1,056 at the version break's part 2 (2026-10-07): the flat pointer's scorer bias (F16b) was the one
#: deleted parameter tensor that received a gradient (MEASURED: re-attaching it restores 1,060); the deleted value
#: tower (F1) held no gradient, so it was never stepped and moved no read.
#: 1,056 → 984 (−72) at the BATCHED HOST READS (gen3_batched_host_reads_v1, T25 item 1, 2026-10-08): the grad-balance
#: probe's 16 norm / dot reads, the edge (34) and cell (8) liveness norms, the per-term noise sampler's 8 `sq_norm`s,
#: the step's 4 `float(norm)`s, the noise base's `float` and `train/loss`'s `.item()` became BATCHED transfers
#: (`.cpu()` of a stack — which this meter does not count; `src/agents/training/host_sync_guard_test.py` counts
#: TRANSFERS, residency-aware). What is left is ALL `optimizer.step()`'s: the per-parameter `step` reads of the
#: non-fused AdamW (MEASURED: 984 − 72 removed = the 984 the X5 part-2 note above attributes to the step sites), and
#: those `step` counters are HOST tensors — not device syncs on CUDA.
GOLDEN_UPDATE_HOST_SCALAR_READS = 984


def test_a_real_update_makes_a_bounded_number_of_host_scalar_reads(learner, monkeypatch):
    meter = {"s": 0.0, "n": 0}
    _real_update(learner, meter)
    assert meter["n"] > 0, "the meter read nothing: it is not counting the reads it exists to count"
    assert meter["n"] == GOLDEN_UPDATE_HOST_SCALAR_READS, (
        f"{meter['n']} host scalar reads in one golden update, the pin is {GOLDEN_UPDATE_HOST_SCALAR_READS}: "
        f"each is a device sync x {_micro_batches(learner)} micro-batches here (x480 in production). A HIGHER "
        f"number is a sync added to the per-micro-batch path; a lower one is a win — update the pin and say why")
    # TEETH (same fixture): ONE host read added to the micro-step moves the count by one per micro-batch
    real = learner._compiled_micro_step

    def noisy(*a: Any, **k: Any) -> Any:
        out = real(*a, **k)
        torch.zeros(()).item()
        return out
    with monkeypatch.context() as m:
        m.setattr(learner, "_compiled_micro_step", noisy)
        planted = {"s": 0.0, "n": 0}
        _real_update(learner, planted)
    assert planted["n"] == GOLDEN_UPDATE_HOST_SCALAR_READS + _micro_batches(learner), planted["n"]


# ---------------------------------------------------------------- 3. attention lowers to the FUSED op
_FUSED = "_scaled_dot_product"          # _flash_attention_for_cpu / _efficient_attention / _cudnn_attention / _flash_attention
_MATH_SOFTMAX = "aten._safe_softmax.default"


def _r1_forward_ops(model: Any, ctx: Any) -> List[str]:
    """The call_function targets of R1's AOT FORWARD graph, traced (not run) under ``ctx``."""
    cap: List[List[str]] = []

    def backend(gm: Any, example: Any) -> Any:
        def fw(g: Any, _e: Any) -> Any:
            cap.append([str(n.target) for n in g.graph.nodes if n.op == "call_function"])
            return g.forward
        return aot_module_simplified(gm, example, fw_compiler=fw)

    torch._dynamo.reset()
    cc._reset_control_for_tests()
    cc.control().install()
    cr.install(model, backend=backend)
    try:
        args = cr._r1_args(model, cr.r1_batch(model, BATCH))
        with ctx:
            model._compiled_micro_step._gen3_compiled(*args)
    finally:
        cr.uninstall(model)
    assert cap, "the R1 forward graph was never captured"
    return cap[0]


@pytest.fixture
def plain_learner() -> Iterator[Any]:
    from agents.training import learner_golden as LG
    cc._reset_control_for_tests()
    torch._dynamo.reset()
    model = LG.build_learner()
    LG.load_buffer_into(model)
    try:
        yield model
    finally:
        cc._reset_control_for_tests()
        torch._dynamo.reset()


#: R1's AOT forward graph on the CPU golden: attention ops by kind. Measured 2026-10-02: 2 fused SDPA ops
#: (`_scaled_dot_product_flash_attention_for_cpu`) and 2 MATH-path SDPA ops (`_safe_softmax`: the trunk's two
#: `BiasedEncoderLayer` calls, whose bias requires grad). Under `sdpa_kernel(MATH)` it reads 0 / 4; with the
#: trunk's attention hand-rolled, 2 / 0; with the fused kernels disabled by a global flag, 0 / 4.
R1_CPU_ATTENTION_PROFILE = {"fused": 2, "math": 2}


@pytest.fixture(scope="module")
def r1_cpu_forward_ops() -> Iterator[List[str]]:
    """R1's AOT forward graph traced ONCE per module (a full dynamo + AOT trace of the golden is ~30 s, and the
    tier budget is per test call, `designs/ops/testing.md`); the pin below only reads it."""
    from agents.training import learner_golden as LG
    cc._reset_control_for_tests()
    torch._dynamo.reset()
    model = LG.build_learner()
    LG.load_buffer_into(model)
    try:
        yield _r1_forward_ops(model, contextlib.nullcontext())
    finally:
        cc._reset_control_for_tests()
        torch._dynamo.reset()


def test_R1s_attention_profile_on_cpu_is_pinned(r1_cpu_forward_ops):
    ops = r1_cpu_forward_ops
    got = {"fused": sum(_FUSED in o for o in ops), "math": sum(o == _MATH_SOFTMAX for o in ops)}
    assert got == R1_CPU_ATTENTION_PROFILE, (
        f"R1's attention lowering changed: {got}, pinned {R1_CPU_ATTENTION_PROFILE}. A lost fused kernel or a new "
        f"MATH path costs time (forcing MATH inside R1 measured +5.2% of its fwd+bwd on the card); a global "
        f"`enable_*_sdp(False)`, a hand-rolled attention and an `sdpa_kernel` context all move this — if the "
        f"change is intended, update the pin and say why")


def test_TEETH_forcing_the_MATH_backend_removes_the_fused_op_from_the_trunk_layer():
    """Plant: `sdpa_kernel(MATH)` around the attention (what the r1_noise probe did; +5.2% of R1 on the
    card). Traced at the layer, on the same aten-level reader: the fused op is replaced by bmm +
    `_safe_softmax` + bmm (the R1-level trace is the pin above; this is its cheap twin)."""
    from torch.fx.experimental.proxy_tensor import make_fx

    from agents.model.arch_constants import D_MODEL
    torch.manual_seed(0)
    layer = BiasedEncoderLayer()
    x, bias = torch.randn(4, 20, D_MODEL), torch.zeros(4, layer.n_heads, 20, 20)

    def ops(ctx: Any) -> List[str]:
        with ctx:
            g = make_fx(lambda a, b: layer(a, b))(x, bias)
        return [str(nd.target) for nd in g.graph.nodes if nd.op == "call_function"]

    default, forced = ops(contextlib.nullcontext()), ops(sdpa_kernel(SDPBackend.MATH))
    assert sum(_FUSED in o for o in default) == 1 and _MATH_SOFTMAX not in default, default
    assert sum(_FUSED in o for o in forced) == 0 and _MATH_SOFTMAX in forced, forced


def _trunk_token_count(model: Any) -> int:
    """The token count the trunk layers see in one eager R1 forward on the golden (its real surface)."""
    from agents.training.instrumented_ppo.micro_step import micro_step
    seen: List[int] = []
    hooks = [m.register_forward_hook(lambda _m, inp, _o: seen.append(int(inp[0].shape[1])))
             for m in model.policy.modules() if isinstance(m, BiasedEncoderLayer)]
    try:
        micro_step(*cr._r1_args(model, cr.r1_batch(model, BATCH)))
    finally:
        for h in hooks:
            h.remove()
    return seen[0]


@pytest.mark.slow
@pytest.mark.skipif(_cuda_skip_reason() is not None, reason=_cuda_skip_reason() or "")
def test_on_cuda_the_trunk_layer_dispatches_a_fused_sdpa_kernel_not_math(plain_learner):
    """CUDA cell (the card's own dispatcher; runs under `GEN3AI_TEST_ALLOW_GPU=1` on an idle box): the
    trunk layer at its production width and head count, float32, an additive float bias, at the golden's
    real token count, traced at the aten level on the card — the fused op, never bmm + `_safe_softmax`;
    and the same trace under `sdpa_kernel(MATH)` shows the decomposition (so the check can fail)."""
    from torch.fx.experimental.proxy_tensor import make_fx

    from agents.model.arch_constants import D_MODEL
    n = _trunk_token_count(plain_learner)
    torch.manual_seed(0)
    layer = BiasedEncoderLayer().cuda()
    x = torch.randn(64, n, D_MODEL, device="cuda", requires_grad=True)
    bias = torch.zeros(64, layer.n_heads, n, n, device="cuda", requires_grad=True)   # production's bias is learned

    def ops(ctx: Any) -> List[str]:
        with ctx:
            g = make_fx(lambda a, b: layer(a, b))(x, bias)
        return [str(nd.target) for nd in g.graph.nodes if nd.op == "call_function"]

    default, forced = ops(contextlib.nullcontext()), ops(sdpa_kernel(SDPBackend.MATH))
    assert any("_scaled_dot_product_efficient_attention" in o for o in default) and _MATH_SOFTMAX not in default, default
    assert not any(_FUSED in o for o in forced) and _MATH_SOFTMAX in forced, forced
