"""`model_capture` — the attention `/game` draws is the attention the layer USED.

`scaled_dot_product_attention` never returns its weights, so the capture recomputes
``softmax(q kᵀ/√d + bias)`` from the layer's own `in_proj`. This pins that recomputation against the
layer's real output: re-deriving the layer's output from the captured weights must reproduce
`BiasedEncoderLayer.forward` (edge biases and the key-padding addend included). A recomputation that
dropped the bias, the scale or a head would fail here."""

from __future__ import annotations

import torch

from agents.model.team_transformer import BiasedEncoderLayer
from main.prober import model_capture


def _rederive(layer, x, w):
    B, n, d = x.shape
    v = layer.in_proj(x).reshape(B, n, 3, layer.n_heads, layer.head_dim)[:, :, 2].transpose(1, 2)
    attn = (w @ v).transpose(1, 2).reshape(B, n, d)
    x1 = layer.norm1(x + layer.out_proj(attn))
    return layer.norm2(x1 + layer.linear2(torch.relu(layer.linear1(x1))))


def test_the_recomputed_weights_reproduce_the_layers_own_output():
    g = torch.Generator().manual_seed(7)
    layer = BiasedEncoderLayer(d_model=32, n_heads=4, ffn_dim=64).eval()
    x = torch.randn(3, 9, 32, generator=g)
    bias = 0.5 * torch.randn(3, 4, 9, 9, generator=g)
    bias[:, :, :, 7:] += -1e9                                 # two padded keys, as the trunk masks them
    with torch.no_grad():
        w = model_capture._attn_weights(layer, x, bias)
        assert torch.allclose(w.sum(-1), torch.ones(3, 4, 9), atol=1e-5)
        assert float(w[..., 7:].max()) < 1e-6                 # a padded key gets nothing
        assert torch.allclose(_rederive(layer, x, w), layer(x, bias), atol=1e-5)
        # …and dropping the bias would NOT reproduce it (the check is not vacuous)
        w0 = model_capture._attn_weights(layer, x, None)
        assert not torch.allclose(_rederive(layer, x, w0), layer(x, bias), atol=1e-3)


def test_the_layout_reader_refuses_counts_that_do_not_add_up():
    class _TT:
        _total_tokens = 13
        board_seats = (12, 12, 12)

    class _ES:
        topk_seats = 6
        tail_seats = True

    class _FE:
        team_transformer = _TT()
        entity_seats = _ES()
        hypothesis_builder = object()

    ok = model_capture.token_layout(_FE(), 62)
    assert ok["ok"] and ok["n_events"] == 32 and ok["board_seats"] == [12]
    assert not model_capture.token_layout(_FE(), 20)["ok"]
