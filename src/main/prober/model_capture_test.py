"""`model_capture` — the attention `/game` draws is the attention the layer USED.

`scaled_dot_product_attention` never returns its weights, so the capture recomputes
``softmax(q kᵀ/√d + bias)`` from the layer's own `in_proj`. This pins that recomputation against the
layer's real output: re-deriving the layer's output from the captured weights must reproduce
`BiasedEncoderLayer.forward` (edge biases and the key-padding addend included). A recomputation that
dropped the bias, the scale or a head would fail here."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch

from agents.model.team_transformer import BiasedEncoderLayer
from agents.model.trunk_depth import IdentityInitRound
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


# ======================================================================================= the EXTRA trunk rounds
# `--trunk-layers 3/4` append PRE-LN `IdentityInitRound`s after production's two post-LN layers. They attend over
# `norm1(x)`; a capture that hooked only `BiasedEncoderLayer` never saw them, so `/game` drew a 2-round trunk.

def _rederive_pre_ln(rnd, x, w):
    """The pre-LN round's output rebuilt from CAPTURED attention weights: ``h = x + out(w·v)``, then ``h + ffn(LN₂(h))``."""
    B, n, d = x.shape
    v = rnd.in_proj(rnd.norm1(x)).reshape(B, n, 3, rnd.n_heads, rnd.head_dim)[:, :, 2].transpose(1, 2)
    h = x + rnd.out_proj((w @ v).transpose(1, 2).reshape(B, n, d))
    return h + rnd.linear2(torch.relu(rnd.linear1(rnd.norm2(h))))


def _make_live(rnd: IdentityInitRound, seed: int, scale: float, trained_ln: bool = False) -> None:
    """Give an extra round NON-ZERO output projections (it is the identity at init, which would make every check below
    vacuous: a wrong recomputation would still 'reproduce' x); optionally a trained-looking LayerNorm (gain != 1)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for lin in (rnd.out_proj, rnd.linear2):
            lin.weight.copy_(scale * torch.randn(lin.weight.shape, generator=g))
        if trained_ln:
            for ln in (rnd.norm1, rnd.norm2):
                ln.weight.copy_(1.0 + 0.3 * torch.randn(ln.weight.shape, generator=g))
                ln.bias.copy_(0.3 * torch.randn(ln.bias.shape, generator=g))


def test_the_extra_pre_ln_round_is_rebuilt_from_its_captured_weights():
    g = torch.Generator().manual_seed(5)
    rnd = IdentityInitRound(d_model=32, n_heads=4, ffn_dim=64).eval()
    _make_live(rnd, seed=11, scale=0.2, trained_ln=True)
    x = 2.0 + 3.0 * torch.randn(3, 9, 32, generator=g)        # NOT layer-normed: norm1 matters
    bias = 0.5 * torch.randn(3, 4, 9, 9, generator=g)
    bias[:, :, :, 7:] += -1e9                                 # two padded keys
    with torch.no_grad():
        w = model_capture._attn_weights(rnd, x, bias)
        assert torch.allclose(w.sum(-1), torch.ones(3, 4, 9), atol=1e-5) and float(w[..., 7:].max()) < 1e-6
        want = rnd(x, bias)
        assert float((want - x).abs().max()) > 0.1, "the live round did not move x"
        assert torch.allclose(_rederive_pre_ln(rnd, x, w), want, atol=1e-4)
        # the pre-LN input matters: projecting x itself (the post-LN recipe) gives other weights and another output
        qkv = rnd.in_proj(x).reshape(3, 9, 3, 4, 8)
        q, k = qkv[:, :, 0].transpose(1, 2), qkv[:, :, 1].transpose(1, 2)
        w_post = torch.softmax((q @ k.transpose(-1, -2)) / (8 ** 0.5) + bias, dim=-1)
        assert not torch.allclose(w_post, w, atol=1e-3)
        assert not torch.allclose(_rederive_pre_ln(rnd, x, w_post), want, atol=1e-3)
        # …and dropping the bias would NOT reproduce it
        assert not torch.allclose(_rederive_pre_ln(rnd, x, model_capture._attn_weights(rnd, x, None)), want, atol=1e-3)


def test_capture_stacks_every_trunk_round_of_a_deeper_trunk():
    """The REAL path: a production-arch policy at `--trunk-layers 3`, a non-zero extra round, `capture()` over real
    parity rows. Three rounds are captured (two post-LN, then the pre-LN extra, in execution order) and the third is
    the extra round's own attention: recomputed from the input and bias the round actually USED (hooked), and it
    rebuilds the round's real output."""
    from agents.model.compile_parity_fixture import load_parity_rows
    from agents.model.static_recovery_test import _policy
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
    obs, mask = load_parity_rows(Gen3ObservationEncoder(load_mappings()).dimension)
    obs, mask = np.asarray(obs[:6], dtype=np.float32), np.asarray(mask[:6])
    pol = _policy(trunk_layers=3)                               # cached across tests: restore what is changed
    rnd = pol.features_extractor.team_transformer.extra_rounds[0]
    saved = {k: v.clone() for k, v in rnd.state_dict().items()}
    seen = {}

    def spy(mod, args, kwargs, out):
        seen.update(x=args[0].detach().clone(), bias=kwargs["bias"].detach().clone(), out=out.detach().clone())

    # the prober's policy carries the production Dict space (observation + action_mask); the test harness's has the
    # observation alone, which `get_distribution` would refuse a masked row against
    space = pol.observation_space
    was_training = pol.training
    pol.observation_space = gym.spaces.Dict({**space.spaces, "action_mask": gym.spaces.MultiBinary(mask.shape[1])})
    pol.eval()
    _make_live(rnd, seed=2, scale=0.05)
    h = rnd.register_forward_hook(spy, with_kwargs=True)
    try:
        cap = model_capture.capture(pol, obs, mask, batch=6)
        A = cap["attention"].astype(np.float32)
        assert cap["n_layers"] == 3 and A.shape[1] == 3, (cap["n_layers"], A.shape)
        assert np.allclose(A.sum(-1), 1.0, atol=2e-3)
        assert float(np.abs(A[:, 2] - A[:, 1]).max()) > 1e-3, "round 3 is a copy of round 2: the extra was not hooked"
        with torch.no_grad():
            w = model_capture._attn_weights(rnd, seen["x"], seen["bias"])
            assert np.allclose(w.numpy(), A[:, 2], atol=2e-3), "the captured (fp16) weights are not the round's own"
            assert float((seen["out"] - seen["x"]).abs().max()) > 1e-3, "the extra round was the identity: vacuous"
            assert torch.allclose(_rederive_pre_ln(rnd, seen["x"], torch.as_tensor(A[:, 2])), seen["out"], atol=5e-3), \
                "the captured weights do not rebuild the extra round's output"
    finally:
        h.remove()
        rnd.load_state_dict(saved)
        pol.observation_space = space
        pol.train(was_training)
