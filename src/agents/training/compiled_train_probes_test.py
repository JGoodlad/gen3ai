"""A FULL real `train()` on a COMPILED learner, with every diagnostic probe of a process's first
update ON — the test Lane K1 did not have (gen3_donated_buffer_off_v1, Lane K1b, 2026-09-29).

WHY IT EXISTS. K1 moved the learner to torch 2.8 and verified parity on compiled forwards and on a
single-loss gradient; neither ever put TWO backwards through one compiled graph. The first real
update does: the read-only probes (`grad_balance._flat_grads`, the per-term noise-scale probe) call
`autograd.grad(..., retain_graph=True)` on the compiled graph before `loss.backward()`. Torch >= 2.6
defaults `torch._functorch.config.donated_buffer` to True, and the compile sentinel's prewarm puts a
plain `.backward()` through each train graph FIRST — which bakes donated buffers into the compiled
backward — so on 2.8 the first update's grad-balance probe raised "This backward function was
compiled with non-empty donated buffers which requires create_graph=False and retain_graph=False"
(the K1 learner-bench A/B, both workers). `compile_control` now pins it False per torch version.

THE SHAPE. The same production pipeline a `--compile-trainer` launch runs, on the production policy
surface (`main.fresh_checkpoint`'s kwargs): compile (the real gate on CUDA; the gate's `control()`
phase + a direct compile on CPU, since `compile_trainer_extractor` refuses a CPU device by design),
the post-gate reset, the PRODUCTION prewarm (its plain backward is the trigger), the attach, then one
real `train()` as the first update of the process — K2's cadence always runs every probe there.

It FORCES `donated_buffer = True` before the pipeline starts — the torch >= 2.6 default, and a
hostile value on 2.5.1 — and first compiles the same production signatures under that default,
outside the adapter, as any other torch-2.8 process would. Reverting `donated_buffer=False` fails it:
on 2.8 the probe raises the donated-buffer error (measured, CPU and CUDA), and on both torches the
post-install config assertion fails. The cache-key-tag half is NOT reproduced at this toy's size —
reverting only the tag passes here (measured 2026-09-29: the repeated-backward difference is 2.2e-7
with and without it); that half is proven by the real-graph A/B (default cache: "modified by an
inplace operation"; fresh cache: completes) and pinned by `compile_control_test`'s cache-key contract.
The repeated-`retain_graph`-backward check stays as a standing invariant: an overwritten saved
activation shows there even where no version counter moves.

Critic: `shaped`, for the reason `diagnostics_cadence_test._real_gen3_ppo` gives (the toy has no
win-prob outcome labels); the retain-graph mechanism does not depend on which critic term is measured.
"""
from __future__ import annotations

import math

import pytest
import torch

from agents.model import compile_control as cc
from agents.model.extractor_compiles_test import _cuda_skip_reason
from agents.training.diagnostics_cadence_test import PER_TERM_TAGS, _real_gen3_ppo

pytestmark = pytest.mark.slow

REPEAT_TOL_CUDA = 1e-5


def _compiled_first_update(device: str):
    import torch._functorch.config as fcfg

    from agents.model.compile_trainer import (arm_compile_sentinel, compile_trainer_extractor,
                                              production_prewarm_calls)
    from agents.training.instrumented_ppo.noise_scale_terms import NOISE_TERM_GROUPS

    cc._reset_control_for_tests()
    torch._dynamo.reset()
    prev = fcfg.donated_buffer
    # SB3's seeding (`set_random_seed(using_cuda=True)` inside `_real_gen3_ppo`) sets the process
    # global `cudnn.deterministic` on CUDA — handed back below (the root conftest's torch
    # global-state guard failed the CUDA cell's teardown on it, 2026-10-01; torch-independent).
    prev_det = torch.backends.cudnn.deterministic
    fcfg.donated_buffer = True          # the torch >= 2.6 default; hostile on 2.5.1 too
    try:
        model = _real_gen3_ppo(device=device)
        fe = model.policy.features_extractor
        # The production signatures compiled first under the DEFAULT (donating) config, outside
        # `compile_control`, as any other torch-2.8 process does (the Inductor cache key is blind to
        # donation — the 2026-09-29 A/B's second crash; see the module docstring for what this toy
        # does and does not reproduce of it).
        fe.forward = torch.compile(fe.forward)
        for _label, fn in production_prewarm_calls(model, n_envs=model.n_envs,
                                                   batch_size=model.batch_size):
            fn()
        del fe.forward
        torch._dynamo.reset()
        if device == "cuda":
            compile_trainer_extractor(model, True, batch=16)
        else:
            ctl = cc.control()
            with ctl.gate():
                fe.forward = ctl.wrap_compiled(torch.compile(fe.forward))
        assert "forward" in vars(fe), "the learner extractor is not compiled — the test is vacuous"
        assert fcfg.donated_buffer is False, (
            "compile_control did not pin torch._functorch.config.donated_buffer=False at install — "
            "a compiled backward will then donate its saved activations and every retain_graph "
            "probe on it raises (torch >= 2.6)")
        # PRODUCTION's prewarm set. No batch-1 signature exists on any critic: batch 1 always runs
        # the eager forward (`compile_trainer.EAGER_BATCHES`, gen3_batch1_eager_v1).
        arm_compile_sentinel(model, n_envs=model.n_envs, batch_size=model.batch_size)
        # The first update of a process: every gated probe runs (K2). Prime the noise EMAs past
        # warm-up so the per-term probe's EMIT gate passes and its tags prove it ran.
        model.diagnostics_every = 1
        # Primed LARGE: on the perturbed real-row toy (`_real_gen3_ppo`, gen3_fresh_parity_probe_v1)
        # one sample's g2 can sign-flip by more than a small prime absorbs, and the emit gate then
        # (correctly) withholds a tag this test reads as proof that the probe ran.
        model._noise_ema_s, model._noise_ema_g2, model._noise_ema_n = 1e6, 1e6, 500
        model._noise_ema_terms = {g: [1e6, 1e6, 500] for g in NOISE_TERM_GROUPS}
        model.logger.name_to_value.clear()
        # REPEATABILITY of a retain-graph backward through the compiled TRAIN graph (the prewarmed
        # signature): a backward that reuses its saved activations in place — a donating kernel —
        # makes the second read differ, or raise, even where no ATen op bumps a version counter.
        _assert_retain_graph_backward_repeats(model)
        before = {k: v.detach().clone() for k, v in model.policy.state_dict().items()}
        model.train()
        logged = dict(model.logger.name_to_value)
        after = model.policy.state_dict()
        return before, after, logged
    finally:
        cc._reset_control_for_tests()
        torch._dynamo.config.error_on_recompile = False
        torch._dynamo.reset()
        fcfg.donated_buffer = prev
        torch.backends.cudnn.deterministic = prev_det


def _assert_retain_graph_backward_repeats(model) -> None:
    from agents.model.compile_trainer import _prewarm_obs
    policy = model.policy
    policy.set_training_mode(True)
    try:
        pi, vf = policy.extract_features(_prewarm_obs(model, model.batch_size))
        loss = pi.float().square().mean() + vf.float().square().mean()
        params = [p for p in policy.features_extractor.parameters() if p.requires_grad]
        g1 = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        g2 = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        g3 = torch.autograd.grad(loss, params, allow_unused=True)
    finally:
        policy.zero_grad(set_to_none=True)
    worst = 0.0
    for a, b, c in zip(g1, g2, g3):
        if a is None:
            continue
        scale = float(a.norm()) + 1e-12
        worst = max(worst, float((a - b).norm()) / scale, float((a - c).norm()) / scale)
    print(f"[K1b] retain-graph backward repeat: worst relative gradient difference {worst:.3e}",
          flush=True)
    # Bitwise on CPU. On CUDA the scatter-add backward's atomics reorder float sums, so repeats
    # differ at float rounding (measured 2026-09-29: 2.1e-7), hence REPEAT_TOL_CUDA.
    tol = 0.0 if next(policy.parameters()).device.type == "cpu" else REPEAT_TOL_CUDA
    assert worst <= tol, (
        f"a retain_graph backward through the compiled train graph is NOT repeatable (worst "
        f"relative gradient difference {worst:.3e} > {tol:.1e}) — its saved activations were "
        "overwritten by the first backward (a donating backward, e.g. served from the Inductor "
        "cache without the adapter's cache-key tag)")


def _assert_full_update(before, after, logged):
    for prefix in ("grad/", "rank/", "edge/", "cell/"):
        assert any(k.startswith(prefix) for k in logged), f"no {prefix}* tag: a probe did not run"
    for tag in PER_TERM_TAGS:
        assert tag in logged, f"{tag}: the per-term noise probe did not run"
    for k, v in logged.items():
        if k.startswith("grad/") and isinstance(v, float):
            assert math.isfinite(v), (k, v)
    assert math.isfinite(logged["train/loss"])
    assert any(not torch.equal(before[k], after[k]) for k in before), "the update moved nothing"


def test_a_compiled_first_update_with_every_probe_runs_on_cpu(restore_torch_globals):
    """The CPU variant: torch.compile's C++ backend lowers this backward under production's
    `belief_grad_mode=label_only` (`extractor_compiles_test`), and the donated-buffer logic lives in
    AOTAutograd, which is device-agnostic — so this reproduces the torch 2.8 crash without a GPU."""
    torch.set_num_threads(2)
    _assert_full_update(*_compiled_first_update("cpu"))


@pytest.mark.skipif(_cuda_skip_reason() is not None, reason=_cuda_skip_reason() or "")
def test_a_compiled_first_update_with_every_probe_runs_on_cuda():
    """The production device and the real `--compile-trainer` gate (real-obs parity fixture)."""
    _assert_full_update(*_compiled_first_update("cuda"))
