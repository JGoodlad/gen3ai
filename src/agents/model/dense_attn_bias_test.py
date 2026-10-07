"""`gen3_dense_attn_bias_v1` (F-ST-8): every float attention bias reaches SDPA through `dense_attn_bias`.

The defect: under Inductor the trunk bias — built `.contiguous()`, then written by `EdgeBias`'s in-place
slice adds of the head-innermost `m.permute(0, 3, 1, 2)` — is a FLEXIBLE buffer whose layout Inductor
decides after its SDPA constraint already passed it as "aligned" (the key count n a multiple of 8). The
first layer's CUDA efficient-attention kernel then receives strides `(n·n·H, 1, n·H, H)` and raises
"(*bias): last dimension must be contiguous". `--token-encoding static` (n = 64) died on it in the T2
service's first CUDA-graph build; legacy (n = 62) takes the constraint's padded-copy branch and never did.

The CPU tests pin the MECHANISM (the forward graph carries the row-major pin on the bias every SDPA
reads — they fail on revert of either call site) and eager byte-identity. The CUDA test is the
reproduction: the same build pattern at n = 64 under Inductor FAILS with the pin removed (a precondition
that names a torch which no longer needs it) and runs, matching eager, with it."""
from __future__ import annotations

from typing import Any, List

import pytest
import torch

from agents.model import dense_attn_bias as dab
from agents.model import team_transformer as tt
from agents.model.arch_constants import D_MODEL, TRANSFORMER_N_HEADS
from agents.model.extractor_compiles_test import _cuda_skip_reason
from agents.model.team_transformer import _KEY_PAD_NEG, BiasedEncoderLayer, EdgeBias

H = TRANSFORMER_N_HEADS
_FORCE = "inductor_force_stride_order"


class _TrunkLike(torch.nn.Module):
    """The trunk's bias pipeline in miniature (the `fixed_mass` arm): the key-pad bias `.expand(...).contiguous()`,
    the per-key log-presence add, then the real `EdgeBias._write_block` writing a family's head-innermost map at (rows, cols) + its transpose,
    then two `BiasedEncoderLayer`s sharing the bias — `TeamTransformer.forward`'s shape at n tokens."""

    def __init__(self, n: int) -> None:
        super().__init__()
        self.n = n
        self.map = torch.nn.Linear(6, 2 * H)
        self.edge = EdgeBias("off")
        self.layers = torch.nn.ModuleList([BiasedEncoderLayer(), BiasedEncoderLayer()])

    def forward(self, x: torch.Tensor, pad: torch.Tensor, cells: torch.Tensor, lp: torch.Tensor) -> torch.Tensor:
        B, n, _ = x.shape
        bias = (pad[:, None, None, :].to(x.dtype) * _KEY_PAD_NEG).expand(B, H, n, n).contiguous()
        # fixed_mass's per-key log-presence add (`TeamTransformer.forward`): it makes the bias a fresh FLEXIBLE
        # buffer, and is the measured trigger (without it the `.contiguous()` buffer stays row-major, 2026-10-07)
        bias = bias + torch.where(pad, torch.zeros_like(lp), lp)[:, None, None, :]
        self.edge._write_block(bias, self.map(cells), slice(n - 8, n - 4), slice(6, 12))
        self.edge._write_block(bias, self.map(cells), slice(0, 4), slice(n - 6, n))
        for layer in self.layers:
            x = layer(x, bias=bias)
        return x


def _inputs(n: int, B: int = 8, device: str = "cpu") -> Any:
    g = torch.Generator().manual_seed(7)
    x = torch.randn(B, n, D_MODEL, generator=g)
    pad = torch.zeros(B, n, dtype=torch.bool)
    pad[:, 3] = True
    cells = torch.randn(B, 4, 6, 6, generator=g)
    lp = -torch.rand(B, n, generator=g)
    return x.to(device), pad.to(device), cells.to(device), lp.to(device)


def _forward_graphs(fn: Any, *args: Any) -> List[torch.fx.GraphModule]:
    """The AOT forward graphs `fn` compiles to (ATen level, the graph Inductor would lower)."""
    from torch._dynamo.backends.common import aot_autograd
    graphs: List[torch.fx.GraphModule] = []

    def fw(gm: torch.fx.GraphModule, _ex: Any) -> Any:
        graphs.append(gm)
        return gm.forward

    torch._dynamo.reset()
    try:
        torch.compile(fn, backend=aot_autograd(fw_compiler=fw), fullgraph=True)(*args)
    finally:
        torch._dynamo.reset()
    return graphs


_VIEWS = ("expand", "view", "unsqueeze", "slice", "alias", "clone")


def _through_views(node: Any) -> Any:
    while isinstance(node, torch.fx.Node) and any(v in str(node.target) for v in _VIEWS):
        node = node.args[0]
    return node


def _sdpa_mask_sources(gm: torch.fx.GraphModule) -> List[str]:
    """For every attention in the graph, the target of the node its bias comes from (through view-only ops):
    an SDPA node's attn_mask, or — where the CPU training trace decomposes SDPA to its math (bmm, add the
    mask, softmax) — the non-matmul operand of the add a softmax reads."""
    out = []
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        if "scaled_dot_product" in str(node.target):
            mask = node.args[3] if len(node.args) > 3 else node.kwargs.get("attn_mask")
            out.append(str(getattr(_through_views(mask), "target", mask)))
        elif "softmax" in str(node.target):
            add = _through_views(node.args[0])
            assert "add" in str(add.target), f"a softmax reads {add.target}, not logits + bias"
            srcs = [_through_views(a) for a in add.args if isinstance(a, torch.fx.Node)]
            bias = [a for a in srcs if not any(m in str(a.target) for m in ("bmm", "mm", "mul", "div"))]
            out.append(" ".join(str(b.target) for b in bias) or "none")
    return out


def test_eager_is_a_noop_on_the_contiguous_bias() -> None:
    """Eager byte-identity: the trunk's bias is already contiguous, so the pin returns the SAME tensor."""
    b = torch.randn(2, H, 64, 64)
    assert dab.dense_attn_bias(b) is b
    nc = b.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2)          # head-innermost, as Inductor laid it
    assert nc.stride(-1) != 1                                              # PRECONDITION
    out = dab.dense_attn_bias(nc)
    assert out.stride(-1) == 1 and out.is_contiguous() and torch.equal(out, nc)


@pytest.mark.parametrize("grad", [False, True], ids=["inference", "training"])
def test_every_trunk_sdpa_reads_the_row_major_pin_under_compile(grad: bool) -> None:
    """Reverting the pin in `BiasedEncoderLayer.forward` FAILS this: each layer's SDPA mask must come from
    the `inductor_force_stride_order` prim (training too — its registered identity backward)."""
    torch.manual_seed(0)
    mod = _TrunkLike(64)
    args = _inputs(64)
    with torch.set_grad_enabled(grad):
        graphs = _forward_graphs(mod, *args)
    assert graphs, "no forward graph captured"
    srcs = [s for gm in graphs for s in _sdpa_mask_sources(gm)]
    assert len(srcs) == 2, srcs                                            # PRECONDITION: both layers seen
    assert all(_FORCE in s for s in srcs), f"an SDPA reads an UNPINNED bias: {srcs}"


def test_policy_query_pool_reads_the_row_major_pin_under_compile() -> None:
    """The second float-bias SDPA site (`pools.PolicyStateQuery`) pins its key bias too."""
    from agents.model.pools import PolicyStateQuery
    torch.manual_seed(0)
    pool = PolicyStateQuery()
    tokens = torch.randn(4, 16, D_MODEL)
    pad = torch.zeros(4, 16, dtype=torch.bool)
    pad[:, 2] = True
    with torch.no_grad():
        graphs = _forward_graphs(pool, tokens, pad)
    srcs = [s for gm in graphs for s in _sdpa_mask_sources(gm)]
    assert len(srcs) == 1, srcs
    assert _FORCE in srcs[0], f"the policy-query SDPA reads an UNPINNED bias: {srcs}"


_skip_cuda = pytest.mark.skipif(_cuda_skip_reason() is not None, reason=_cuda_skip_reason() or "")


@_skip_cuda
@pytest.mark.slow
def test_cuda_inductor_n64_bias_reproduction(monkeypatch: pytest.MonkeyPatch) -> None:
    """THE reproduction (CUDA, Inductor, n = 64 — the static arm's key count): without the pin the compiled
    first layer's efficient-attention kernel raises "last dimension must be contiguous"; with it the
    compiled forward runs and matches eager. GPU tier: `GEN3AI_TEST_ALLOW_GPU=1` under a lease."""
    torch.manual_seed(0)
    mod = _TrunkLike(64).cuda()
    args = _inputs(64, device="cuda")
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        with torch.no_grad():
            want = mod(*args)
            with monkeypatch.context() as mp:
                mp.setattr(tt, "dense_attn_bias", lambda b: b)
                torch._dynamo.reset()
                with pytest.raises(RuntimeError, match="last dimension must be contiguous"):
                    torch.compile(mod, fullgraph=True)(*args)
            torch._dynamo.reset()
            got = torch.compile(mod, fullgraph=True)(*args)
        torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)
    finally:
        torch._dynamo.reset()
        torch.set_float32_matmul_precision(prev)


@_skip_cuda
@pytest.mark.slow
def test_cuda_inductor_n64_bias_training_gradients_match_eager() -> None:
    """The pin under a compiled forward + BACKWARD (the learner's R1 shape): its registered identity backward
    keeps every gradient, the edge map's included, equal to eager's."""
    torch.manual_seed(0)
    mod = _TrunkLike(64).cuda()
    args = _inputs(64, device="cuda")
    prev = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    try:
        mod(*args).square().mean().backward()
        want = {k: p.grad.clone() for k, p in mod.named_parameters()}
        mod.zero_grad(set_to_none=True)
        torch._dynamo.reset()
        torch.compile(mod, fullgraph=True)(*args).square().mean().backward()
        for k, p in mod.named_parameters():
            assert p.grad is not None, k
            torch.testing.assert_close(p.grad, want[k], rtol=1e-3, atol=1e-6, msg=k)
        assert want["map.weight"].abs().sum() > 0                          # PRECONDITION: the bias carries gradient
    finally:
        torch._dynamo.reset()
        torch.set_float32_matmul_precision(prev)
