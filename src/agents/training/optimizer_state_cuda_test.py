"""Optimizer state DECLARED at startup == torch's LAZY first-step init, on CUDA (K6 + the ride-along
lifecycle; GPU tier: `slow`, `GEN3AI_TEST_ALLOW_GPU=1` under `scripts/ops/gpu_lock.sh`).

The CPU tests (`learner_lifecycle_test`, `ridealong_update_test`) pin both startup helpers bit-identical
to the lazy init, but a CUDA Adam takes different kernels — `foreach` (the policy's AdamW: the default on
CUDA) and `fused` (the ride-along heads' Adam) — whose step counter lives on the device and whose
arithmetic is a different code path. Here: two identical small nets on CUDA, one optimizer left to
torch's lazy init, the other declared at startup by each helper, then THREE steps on identical
gradients: every parameter and every state tensor must be bit-identical. Fails if a helper leaves a
moment non-zero, mis-resets the step counter, or (AdamW) lets the zero-gradient step's decoupled weight
decay survive."""
from __future__ import annotations

import pytest
import torch

from agents.model.extractor_compiles_test import _cuda_skip_reason

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(_cuda_skip_reason() is not None, reason=_cuda_skip_reason() or "")]


def _net():
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(16, 32), torch.nn.Tanh(), torch.nn.Linear(32, 4)).cuda()


def _steps(net, opt, n=3):
    g = torch.Generator(device="cuda").manual_seed(7)
    for _ in range(n):
        x = torch.randn(64, 16, device="cuda", generator=g)
        opt.zero_grad(set_to_none=True)
        net(x).square().mean().backward()
        opt.step()


def _same(net_a, opt_a, net_b, opt_b):
    for pa, pb in zip(net_a.parameters(), net_b.parameters()):
        assert torch.equal(pa, pb)
        sa, sb = opt_a.state[pa], opt_b.state[pb]
        assert sorted(sa) == sorted(sb)
        for k in sa:
            assert torch.equal(torch.as_tensor(sa[k]).cpu(), torch.as_tensor(sb[k]).cpu()), k


@pytest.mark.parametrize("kind", ["adamw_foreach", "adamw_fused", "adam_fused", "adam_foreach"])
def test_the_K6_declaration_is_bit_identical_to_lazy_init_on_cuda(kind):
    from agents.training.learner_lifecycle import declare_optimizer_state

    def make(net):
        if kind.startswith("adamw"):
            return torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-5, eps=1e-5,
                                     fused=kind.endswith("fused") or None,
                                     foreach=True if kind.endswith("foreach") else None)
        return torch.optim.Adam(net.parameters(), lr=1e-3, fused=kind.endswith("fused") or None,
                                foreach=True if kind.endswith("foreach") else None)
    lazy_net, decl_net = _net(), _net()
    lazy, decl = make(lazy_net), make(decl_net)
    declare_optimizer_state(decl)
    for pa, pb in zip(lazy_net.parameters(), decl_net.parameters()):
        assert torch.equal(pa, pb)              # the declaration moved no weight
    _steps(lazy_net, lazy)
    _steps(decl_net, decl)
    _same(lazy_net, lazy, decl_net, decl)


def test_the_ridealong_preallocation_is_bit_identical_to_lazy_init_on_cuda_fused():
    """The ride-along heads' own helper (`ridealong_terms.preallocate_adam_state`, fused Adam on
    CUDA) — the case its CPU test skipped."""
    from agents.training.instrumented_ppo.ridealong_terms import preallocate_adam_state
    lazy_net, pre_net = _net(), _net()
    lazy = torch.optim.Adam(lazy_net.parameters(), lr=1e-3, eps=1e-6, fused=True)
    pre = torch.optim.Adam(pre_net.parameters(), lr=1e-3, eps=1e-6, fused=True)
    preallocate_adam_state(pre)
    _steps(lazy_net, lazy)
    _steps(pre_net, pre)
    _same(lazy_net, lazy, pre_net, pre)
