"""A FULL real `train()` on a learner compiled as its DECLARED REGIONS (R0 + R1), as the first update of a
process, with `torch._functorch.config.donated_buffer` handed in at TORCH'S DEFAULT (True) — the protection
deletion pass K1 removed with the extractor-only compile's `compiled_train_probes_test` and left uncovered
(`designs/ops/deletion_pass_manifest.md` §6 finding 10; `gen3_donated_buffer_off_v1`, Lane K1b).

WHY IT EXISTS. Torch >= 2.6 defaults `donated_buffer` to True: AOTAutograd then lets a compiled BACKWARD
reuse the forward's saved activations IN PLACE, legal only when every backward through that graph is
single-use. The learner's are not — the first update's read-only probes (`grad_balance._flat_grads`, the
per-term noise-scale probe) call `autograd.grad(..., retain_graph=True)` on the compiled graph — and the
sentinel's prewarm puts a plain `.backward()` through R1's train graph FIRST, which bakes donated buffers
into it. `compile_control.install()` pins the option False (`_COMPILE_CONFIG`). Nothing else exercises a
real first `train()` through the compiled regions with the donating default forced, so a dropped pin (or a
moved prewarm, or an Inductor cache key that stopped carrying the adapter's tag) would crash a LAUNCH, two
minutes in, after every startup gate had passed.

THE SHAPE. The real `arm_compile_sentinel` pipeline (install, gate, reset, prewarm, lock, canary) on the
K9 golden's production-surface learner (the CPU cell skips the numeric gate: CPU Inductor's kernels differ
from eager beyond the fp32 bars — loss rel 1.5e-3 on this learner, measured 2026-10-02 — and the gate is a
CUDA fact; everything after it, the prewarm that bakes the donation and the first update, runs), then one `train()` with every probe forced on
(`diagnostics_every = 1`, the noise EMAs primed past warm-up so the per-term probe's emit gate passes and
its tags prove it ran). It FORCES `donated_buffer = True` before the pipeline and asserts the adapter
pinned it False. Reverting the pin fails it on both devices: the first update's probe raises "one of the
variables needed for gradient computation has been modified by an inplace operation" (measured 2026-10-02,
CPU Inductor). CPU runs Inductor's C++ backend (about 3.5 minutes of compile, hence `slow`); the CUDA cell
is the production device and the real `--compile-trainer` gate.
"""
from __future__ import annotations

import math

import pytest
import torch

from agents.model import compile_control as cc
from agents.model.extractor_compiles_test import _cuda_skip_reason

pytestmark = pytest.mark.slow

_skip_cuda = pytest.mark.skipif(_cuda_skip_reason() is not None, reason=_cuda_skip_reason() or "")

GATED_PREFIXES = ("grad/", "rank/", "edge/", "cell/")
PER_TERM_TAGS = ("train/noise_scale_ratio_policy", "train/noise_scale_policy",
                 "train/noise_scale_share_policy", "train/noise_per_term_ms")


def _compiled_first_update(device: str, gate: bool = True):
    import torch._functorch.config as fcfg

    from agents.model import compile_regions as cr

    from agents.model.compile_trainer import arm_compile_sentinel
    from agents.training import learner_golden as LG
    from agents.training.instrumented_ppo.noise_scale_terms import NOISE_TERM_GROUPS

    cc._reset_control_for_tests()
    torch._dynamo.reset()
    prev = fcfg.donated_buffer
    prev_det = torch.backends.cudnn.deterministic
    fcfg.donated_buffer = True                       # the torch >= 2.6 default — hostile to a retain_graph probe
    model = None
    try:
        model = LG.build_learner()
        LG.load_buffer_into(model)
        if device == "cuda":
            dev = torch.device("cuda")
            model.policy.to(dev)
            model.device = dev
            model.rollout_buffer.device = dev
        if not gate:
            real_gate = cr.gate_regions
            cr.gate_regions = lambda *a, **k: []        # `arm_compile_sentinel` reads it at call time
        try:
            arm_compile_sentinel(model, n_envs=int(model.n_envs), batch_size=int(model.batch_size))
        finally:
            if not gate:
                cr.gate_regions = real_gate
        assert getattr(model, "_compiled_micro_step", None) is not None, \
            "R1 is not installed — the test is vacuous"
        assert fcfg.donated_buffer is False, (
            "compile_control did not pin torch._functorch.config.donated_buffer=False at install — a "
            "compiled backward will then donate its saved activations and every retain_graph probe on it "
            "raises (torch >= 2.6)")
        # The first update of a process: every gated probe runs (K2's cadence). Prime the noise EMAs
        # past warm-up so the per-term probe's EMIT gate passes and its tags prove it ran.
        model.diagnostics_every = 1
        model._noise_ema_s, model._noise_ema_g2, model._noise_ema_n = 1e6, 1e6, 500
        model._noise_ema_terms = {g: [1e6, 1e6, 500] for g in NOISE_TERM_GROUPS}
        model.logger.name_to_value.clear()
        before = {k: v.detach().clone() for k, v in model.policy.state_dict().items()}
        model.train()
        return before, model.policy.state_dict(), dict(model.logger.name_to_value)
    finally:
        if model is not None:
            cr.uninstall(model)
        cc._reset_control_for_tests()
        torch._dynamo.config.error_on_recompile = False
        torch._dynamo.reset()
        fcfg.donated_buffer = prev
        torch.backends.cudnn.deterministic = prev_det


def _assert_full_update(before, after, logged):
    for prefix in GATED_PREFIXES:
        assert any(k.startswith(prefix) for k in logged), f"no {prefix}* tag: a probe did not run"
    for tag in PER_TERM_TAGS:
        assert tag in logged, f"{tag}: the per-term noise probe did not run"
    for k, v in logged.items():
        if k.startswith("grad/") and isinstance(v, float):
            assert math.isfinite(v), (k, v)
    assert math.isfinite(logged["train/loss"])
    assert any(not torch.equal(before[k].cpu(), after[k].cpu()) for k in before), "the update moved nothing"


def test_a_first_update_through_the_compiled_regions_survives_the_donating_default_on_cpu(
        restore_torch_globals):
    """CPU, Inductor's C++ backend: AOTAutograd's donated-buffer logic is device-agnostic, so this
    reproduces the torch 2.8 crash without a GPU. Fails (the probe's retain_graph backward) if the
    adapter's `donated_buffer=False` pin is reverted."""
    torch.set_num_threads(2)
    _assert_full_update(*_compiled_first_update("cpu", gate=False))


@_skip_cuda
def test_a_first_update_through_the_compiled_regions_survives_the_donating_default_on_cuda(
        restore_torch_globals):
    """The production device and the real `--compile-trainer` gate (real-obs parity fixture). Skips with a
    named reason on CPU / a busy card; runs under `GEN3AI_TEST_ALLOW_GPU=1` on an idle box."""
    _assert_full_update(*_compiled_first_update("cuda"))
