"""Entity-generation feasibility spike (design_generation_roadmap.md §2.5) — run as a script.

Answers the two unknowns that gate Stage 1/2 of the ai_v9 entity generation, with NO training
and no real encoders:

SPIKE 1 — the TOKEN-BUDGET benchmark. The generation lives or dies at B=1 on CPU (the PFSP
frozen-opponent forward is DISPATCH-bound: ~14k aten calls at ~0.44 us each, v50 measurement —
cost tracks the NUMBER of ops, not tensor sizes, at these dims). Token count 14 -> ~35-50 grows
attention FLOPs ~(n/14)^2 but adds ZERO aten calls (same graph, bigger tensors) — so the
hypothesis to verify is that B=1 wall time grows far slower than quadratic. We build the
production trunk shape (d_model=128, 2 layers, 4 heads, FFN 256, the literal
TransformerEncoderLayer kwargs) with a per-pair-biased attention variant and measure per-forward
wall time across seat counts, B=1 (threads=1, the env-worker reality) and B=256 (learner proxy),
eager and compiled.

SPIKE 2 — the BIASED-MHA kernel proof. Stage 2 delivers computed physics as per-pair per-head
additive float biases on the attention logits. Proven here: (a) numerical correctness of the
SDPA-with-additive-mask path against a hand-rolled float64 softmax reference; (b) bias=None
reproduces the unbiased path exactly; (c) the layer compiles with torch.compile(fullgraph=True)
— zero graph breaks, else it would silently forfeit the shipped 6.5x compiled-opponent lever —
and compiled output matches eager.

Absolute ms scale with machine load; the RATIOS across seat counts and the compile verdict are
the load-stable signal. Context anchors (v50, same class of box): full production forward at
B=1 = 6.45 ms (4.62 ms under --damage-op-prefuse); the 2 attention layers = 0.27 ms of it.

Usage:
  export PYTHONPATH=$PYTHONPATH:src && python3 src/agents/model/entity_spike_benchmark.py \
      [--sizes 14,36,50,64] [--reps 300] [--batch-reps 30] [--skip-compile]
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.nn.functional as F

from agents.model.arch_constants import (
    D_MODEL,
    TRANSFORMER_FFN_DIM,
    TRANSFORMER_N_HEADS,
    TRANSFORMER_N_LAYERS,
)

# The edge-cell width fed to the bias map. Stand-in for one edge family's cell (e.g. the damage
# cell [low, high, crit, pko, type_mult] + provenance); the map cost scales linearly in this.
CELL_DIM = 8


class BiasedEncoderLayer(torch.nn.Module):
    """A TransformerEncoderLayer clone (post-LN, ReLU, dropout 0 — the production kwargs) whose
    self-attention takes an ADDITIVE per-pair per-head float bias [B, H, n, n].

    Attention is computed via F.scaled_dot_product_attention with attn_mask=bias — the additive
    float-mask path, which is exactly "logits += bias" pre-softmax. bias=None is the unbiased
    baseline (byte-identical math to the stock layer up to in/out-proj weight layout)."""

    def __init__(self, d_model: int = D_MODEL, n_heads: int = TRANSFORMER_N_HEADS,
                 ffn_dim: int = TRANSFORMER_FFN_DIM):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.in_proj = torch.nn.Linear(d_model, 3 * d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)
        self.linear1 = torch.nn.Linear(d_model, ffn_dim)
        self.linear2 = torch.nn.Linear(ffn_dim, d_model)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)

    def attention(self, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
        B, n, d = x.shape
        qkv = self.in_proj(x).reshape(B, n, 3, self.n_heads, self.head_dim)
        q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))  # each [B, H, n, hd]
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        return self.out_proj(out.transpose(1, 2).reshape(B, n, d))

    def forward(self, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        x = self.norm1(x + self.attention(x, bias))
        return self.norm2(x + self.linear2(F.relu(self.linear1(x))))


class EntityTrunk(torch.nn.Module):
    """The spike trunk: TRANSFORMER_N_LAYERS biased layers + ONE bias map (cell -> per-head
    scalars), mirroring the Stage-2 delivery: edge cells are computed once per forward and mapped
    to a [B, H, n, n] additive bias shared by every layer."""

    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList(BiasedEncoderLayer() for _ in range(TRANSFORMER_N_LAYERS))
        self.bias_map = torch.nn.Linear(CELL_DIM, TRANSFORMER_N_HEADS)

    def forward(self, x: torch.Tensor, cells: torch.Tensor | None = None) -> torch.Tensor:
        bias = None
        if cells is not None:
            bias = self.bias_map(cells).permute(0, 3, 1, 2).contiguous()  # [B,H,n,n]
        for layer in self.layers:
            x = layer(x, bias)
        return x


def _time(fn, reps: int, warmup: int = 20) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1000.0  # ms


def spike2_correctness_and_compile(skip_compile: bool) -> None:
    print("=" * 72)
    print("SPIKE 2 — biased-MHA kernel proof")
    print("=" * 72)
    torch.manual_seed(0)
    layer = BiasedEncoderLayer().eval()
    B, n, H = 3, 50, TRANSFORMER_N_HEADS
    x = torch.randn(B, n, D_MODEL)
    bias = torch.randn(B, H, n, n) * 0.5

    # (a) SDPA-with-bias vs a hand-rolled float64 softmax reference.
    with torch.no_grad():
        qkv = layer.in_proj(x).reshape(B, n, 3, H, layer.head_dim)
        q, k, v = (qkv[:, :, i].transpose(1, 2).double() for i in range(3))
        logits = q @ k.transpose(-2, -1) / (layer.head_dim ** 0.5) + bias.double()
        ref = torch.softmax(logits, dim=-1) @ v
        got = F.scaled_dot_product_attention(
            *(t.float() for t in (q, k, v)), attn_mask=bias
        ).double()
    err_ref = (got - ref).abs().max().item()
    print(f"(a) SDPA+bias vs float64 reference: max|delta| = {err_ref:.3e}")
    assert err_ref < 1e-5, "biased SDPA does not match the softmax(logits+bias) reference"

    # (b) bias=None == the unbiased path; bias=0 == bias=None numerically.
    with torch.no_grad():
        out_none = layer(x, None)
        out_zero = layer(x, torch.zeros(B, H, n, n))
    err_zero = (out_none - out_zero).abs().max().item()
    print(f"(b) bias=0 vs bias=None:            max|delta| = {err_zero:.3e}")
    assert err_zero < 1e-6, "zero bias must reproduce the unbiased path"

    if skip_compile:
        print("(c) compile check SKIPPED (--skip-compile)")
        return
    # (c) fullgraph compile — an exception here means a graph break (the failure this spike exists
    # to surface); compiled output must match eager.
    compiled = torch.compile(layer, fullgraph=True, dynamic=False)
    with torch.no_grad():
        t_compile0 = time.perf_counter()
        out_c = compiled(x, bias)
        compile_s = time.perf_counter() - t_compile0
        err_c = (out_c - layer(x, bias)).abs().max().item()
    print(f"(c) torch.compile(fullgraph=True):  COMPILED OK in {compile_s:.1f}s, "
          f"eager-vs-compiled max|delta| = {err_c:.3e}")
    assert err_c < 1e-4, "compiled forward diverges from eager"
    with torch.no_grad():
        t_eager = _time(lambda: layer(x, bias), 200)
        t_comp = _time(lambda: compiled(x, bias), 200)
    print(f"    layer timing @B={B},n={n}: eager {t_eager:.4f} ms, compiled {t_comp:.4f} ms "
          f"({t_eager / t_comp:.2f}x)")
    print("SPIKE 2: PASS\n")


def spike1_token_budget(sizes: list[int], reps: int, batch_reps: int, skip_compile: bool) -> None:
    print("=" * 72)
    print(f"SPIKE 1 — token-budget benchmark (trunk: {TRANSFORMER_N_LAYERS} layers, "
          f"d_model={D_MODEL}, {TRANSFORMER_N_HEADS} heads, ffn={TRANSFORMER_FFN_DIM}; "
          f"threads={torch.get_num_threads()})")
    print("=" * 72)
    torch.manual_seed(0)
    trunk = EntityTrunk().eval()
    base: dict[str, float] = {}
    header = (f"{'n':>4} {'B=1 plain':>10} {'B=1 bias':>10} {'B=1 comp':>10} "
              f"{'B=256 bias':>11} | ratios vs n={sizes[0]} (plain/bias/comp/B256)")
    print(header)
    for n in sizes:
        x1, c1 = torch.randn(1, n, D_MODEL), torch.randn(1, n, n, CELL_DIM)
        xb, cb = torch.randn(256, n, D_MODEL), torch.randn(256, n, n, CELL_DIM)
        with torch.no_grad():
            t_plain = _time(lambda: trunk(x1), reps)
            t_bias = _time(lambda: trunk(x1, c1), reps)
            t_b256 = _time(lambda: trunk(xb, cb), batch_reps)
            t_comp = float("nan")
            if not skip_compile:
                ctr = torch.compile(trunk, fullgraph=True, dynamic=False)
                ctr(x1, c1)  # compile outside the timer
                t_comp = _time(lambda: ctr(x1, c1), reps)
        if n == sizes[0]:
            base = {"plain": t_plain, "bias": t_bias, "comp": t_comp, "b256": t_b256}
        r = (t_plain / base["plain"], t_bias / base["bias"],
             t_comp / base["comp"], t_b256 / base["b256"])
        quad = (n / sizes[0]) ** 2
        print(f"{n:>4} {t_plain:>9.4f}m {t_bias:>9.4f}m {t_comp:>9.4f}m {t_b256:>10.3f}m | "
              f"{r[0]:.2f}x / {r[1]:.2f}x / {r[2]:.2f}x / {r[3]:.2f}x   (quadratic would be {quad:.1f}x)")
    print("\nAnchors (v50): full B=1 opponent forward 6.45 ms (4.62 prefused); "
          "trunk attention was 0.27 ms of it. Verdict = the ABSOLUTE B=1 bias-column delta vs "
          f"n={sizes[0]}, read against those anchors.\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="14,36,50,64")
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--batch-reps", type=int, default=30)
    ap.add_argument("--skip-compile", action="store_true")
    args = ap.parse_args()
    # threads=1: the env-worker reality — at --n-envs 48 every opponent forward runs on an
    # oversubscribed core (the same convention as the compile-extractor B=1 measurements).
    torch.set_num_threads(1)
    spike2_correctness_and_compile(args.skip_compile)
    spike1_token_budget([int(s) for s in args.sizes.split(",")], args.reps,
                        args.batch_reps, args.skip_compile)


if __name__ == "__main__":
    main()
