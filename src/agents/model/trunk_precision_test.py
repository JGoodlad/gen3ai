"""`trunk_precision` (T25 perf lever A, `gen3_trunk_bf16_region_v1`): the bf16 region covers exactly the trunk
rounds, returns float32, keeps every parameter and gradient float32, survives deepcopy / pickling, and at fp32
(the default and the restore) is the base class — the production forward, bit for bit. CPU only (bf16 autocast
runs on CPU)."""
from __future__ import annotations

import copy
import io
import pickle

import pytest
import torch

from agents.model import trunk_precision as TP
from agents.model.team_transformer import BiasedEncoderLayer
from agents.model.trunk_depth import IdentityInitRound


def _inputs(seed: int = 0, B: int = 3, n: int = 7, d: int = 128, H: int = 4):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, n, d, generator=g)
    bias = torch.randn(B, H, n, n, generator=g)
    bias[:, :, :, -1] = -1e9                     # a key-padded column, as the trunk's mask addend
    return x, bias


def _rounds():
    torch.manual_seed(0)
    a = BiasedEncoderLayer()
    b = IdentityInitRound()
    torch.nn.init.normal_(b.out_proj.weight, std=0.05)   # off its zero init so the round is not the identity
    torch.nn.init.normal_(b.linear2.weight, std=0.05)
    return torch.nn.ModuleList([a, b])


def _run(m: torch.nn.ModuleList, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    for r in m:
        x = r(x, bias)
    return x


def test_bf16_region_returns_fp32_close_to_fp32_and_keeps_fp32_params_and_grads():
    m = _rounds()
    x, bias = _inputs()
    ref = _run(m, x, bias)
    assert TP.set_trunk_precision(m, "bf16") == 2
    assert TP.trunk_precision_of(m) == "bf16"
    xx = x.clone().requires_grad_(True)
    out = _run(m, xx, bias)
    assert out.dtype == torch.float32
    rel = ((out - ref).norm() / ref.norm()).item()
    assert 0.0 < rel < 3e-2, rel                 # bf16 rounding: not bit-equal, but close
    out.square().sum().backward()
    for p in m.parameters():
        assert p.dtype == torch.float32
        assert p.grad is not None and p.grad.dtype == torch.float32
    assert xx.grad is not None and xx.grad.dtype == torch.float32


def test_the_region_really_runs_bf16_matmuls():
    m = _rounds()
    TP.set_trunk_precision(m, "bf16")
    seen = []
    h = m[0].linear1.register_forward_hook(lambda mod, i, o: seen.append(o.dtype))
    x, bias = _inputs()
    _run(m, x, bias)
    h.remove()
    assert seen == [torch.bfloat16]


def test_fp32_restores_the_base_class_bit_for_bit():
    m = _rounds()
    x, bias = _inputs(1)
    ref = _run(m, x, bias)
    TP.set_trunk_precision(m, "bf16")
    TP.set_trunk_precision(m, "fp32")
    assert type(m[0]) is BiasedEncoderLayer and type(m[1]) is IdentityInitRound
    assert TP.trunk_precision_of(m) == "fp32"
    assert torch.equal(_run(m, x, bias), ref)


def test_state_dict_deepcopy_and_pickle_are_unchanged_by_the_swap():
    m = _rounds()
    keys = list(m.state_dict())
    TP.set_trunk_precision(m, "bf16")
    assert list(m.state_dict()) == keys
    c = copy.deepcopy(m)
    assert type(c[0]) is TP.Bf16BiasedEncoderLayer and isinstance(c[0], BiasedEncoderLayer)
    buf = io.BytesIO()
    pickle.dump(m, buf)
    p = pickle.loads(buf.getvalue())
    x, bias = _inputs(2)
    assert torch.equal(_run(p, x, bias), _run(m, x, bias))


def test_unknown_precision_and_mixed_state_refuse():
    m = _rounds()
    with pytest.raises(ValueError):
        TP.set_trunk_precision(m, "fp16")
    TP.set_trunk_precision(m[0], "bf16")
    with pytest.raises(RuntimeError):
        TP.trunk_precision_of(m)


def test_only_trunk_rounds_change_class():
    outer = torch.nn.Sequential(torch.nn.Linear(4, 4), _rounds())
    assert TP.set_trunk_precision(outer, "bf16") == 2
    assert type(outer[0]) is torch.nn.Linear
