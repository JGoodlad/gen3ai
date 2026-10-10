"""TRUNK DEPTH — `--trunk-layers N` (`gen3_static_recovery_v1`, config v150; `designs/endstate/design_static_tokens.md`
§13). The trunk has `TRANSFORMER_N_LAYERS` (= 2) post-LN `BiasedEncoderLayer` rounds; `N > 2` appends `N − 2` EXTRA
rounds after them, each an `IdentityInitRound`, so the network at init is EXACTLY the 2-round one.

Why (the static diagnostic, `designs/research_state/measurements/static_diag_2026-10-09/` H3): under
`--token-encoding static` a board fact reaches a mon token only through attention, and mostly at the SECOND (last)
round, which leaves no round to COMBINE it with anything (the owner's direction allows "add a trunk round if needed").

**Why a PRE-LN residual block, not a third `BiasedEncoderLayer`.** The existing rounds are POST-LN
(`x = norm2(h + ffn(h))`, `h = norm1(x + attn(x))`). A post-LN round with its attention out-projection and FFN output
at zero is `norm2(norm1(x))`, which is NOT the identity: the input is already a LayerNorm output (gain 1, bias 0 at
init), and LayerNorm's ``eps`` makes LN∘LN move every value by ≈ ``eps/2 · (1 − 1/var)`` relative (measured in
`trunk_depth_test.py`). The pre-LN form

    x ← x + out_proj(attn(LN₁(x)))        x ← x + linear2(relu(linear1(LN₂(x))))

with ``out_proj`` and ``linear2`` zero (weight AND bias) adds EXACTLY 0 at init: the extra round is the identity, BIT
FOR BIT, and the network starts equal to the 2-round one (one-lever init). Its attention reads the trunk's ONE shared
bias (key padding, X5 log-presence, the edge families), like every round.

**Init without moving any other byte.** The four projections are `IsolatedLinear`s (SB3's orthogonal re-init skips
them) built inside `torch.random.fork_rng` from a PRIVATE seed (`TRUNK_EXTRA_INIT_SEED`), so building them draws
nothing from the global stream: every other parameter's initial bytes equal the `N = 2` build's. ``in_proj`` and
``linear1`` get the init SB3 gives the existing rounds' Linears (orthogonal, gain √2, bias 0); ``out_proj`` and
``linear2`` are zero; the two LayerNorms are torch's (gain 1, bias 0).
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch

from agents.model.arch_constants import (D_MODEL, TRANSFORMER_FFN_DIM, TRANSFORMER_N_HEADS, TRANSFORMER_N_LAYERS,
                                         TRUNK_EXTRA_INIT_SEED)
from agents.model.dense_attn_bias import dense_attn_bias

#: The legal values of `--trunk-layers`: the production 2 (= `TRANSFORMER_N_LAYERS`, nothing extra built) and up to
#: two identity-init extra rounds. Fewer than 2 is not offered: removing a trained round cannot start as an identity.
TRUNK_LAYERS_CHOICES: Tuple[int, ...] = (TRANSFORMER_N_LAYERS, TRANSFORMER_N_LAYERS + 1, TRANSFORMER_N_LAYERS + 2)


class IdentityInitRound(torch.nn.Module):
    """One EXTRA trunk round (module docstring): a pre-LN residual self-attention + FFN block whose two output
    projections start at exactly zero, so it is the identity at init."""

    def __init__(self, d_model: int = D_MODEL, n_heads: int = TRANSFORMER_N_HEADS,
                 ffn_dim: int = TRANSFORMER_FFN_DIM) -> None:
        super().__init__()
        from agents.model.hypothesis_set import IsolatedLinear   # local: hypothesis_set is a heavy import
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)
        self.in_proj = IsolatedLinear(d_model, 3 * d_model)
        self.out_proj = IsolatedLinear(d_model, d_model, zero=True)
        self.linear1 = IsolatedLinear(d_model, ffn_dim)
        self.linear2 = IsolatedLinear(ffn_dim, d_model, zero=True)
        # The init SB3's `init_weights` gives the existing rounds' Linears (orthogonal, gain √2; bias 0).
        for lin in (self.in_proj, self.linear1):
            torch.nn.init.orthogonal_(lin.weight, gain=math.sqrt(2))
            assert lin.bias is not None
            torch.nn.init.zeros_(lin.bias)

    def forward(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x [B, n, d_model]; `bias` [B, H, n, n] the trunk's shared additive attention bias (or None)."""
        B, n, d = x.shape
        qkv = self.in_proj(self.norm1(x)).reshape(B, n, 3, self.n_heads, self.head_dim)
        q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))            # each [B,H,n,hd]
        attn = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=None if bias is None else dense_attn_bias(bias))
        x = x + self.out_proj(attn.transpose(1, 2).reshape(B, n, d))
        out: torch.Tensor = x + self.linear2(torch.nn.functional.relu(self.linear1(self.norm2(x))))
        return out


def build_extra_rounds(trunk_layers: int) -> Optional[torch.nn.ModuleList]:
    """The `trunk_layers − TRANSFORMER_N_LAYERS` extra rounds, built from the PRIVATE seed (no global RNG draw), or
    None at the production depth (nothing built: byte-identical)."""
    if trunk_layers not in TRUNK_LAYERS_CHOICES:
        raise ValueError(f"trunk_layers must be one of {TRUNK_LAYERS_CHOICES}, got {trunk_layers!r}")
    n_extra = trunk_layers - TRANSFORMER_N_LAYERS
    if n_extra == 0:
        return None
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(TRUNK_EXTRA_INIT_SEED)
        return torch.nn.ModuleList(IdentityInitRound() for _ in range(n_extra))
