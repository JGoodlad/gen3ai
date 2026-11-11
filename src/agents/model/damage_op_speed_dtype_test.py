"""The speed physics' stage buffers take the dtype of the stage they hold (`damage_op_speed`, 2026-11-06).

`_opp_speed_mix` and `_p_first_active_at_stage` scatter a speed stage into a ``torch.zeros`` buffer. Built with no
dtype, the buffer was float32 whatever the forward ran in, so a float64 forward (the X60-T parity root-cause read's
fp64 reference) died there: "Index put requires the source and destination dtypes match". Revert either line and the
float64 calls below raise. In float32 nothing changes (the buffer was float32 already).

The values are integer speeds and gathered float64 masses, so float64 and float32 agree exactly on ``final`` and to an
fp32 rounding on the probabilities (no decision boundary is crossed: the stages are integers, the spreads exact).
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from agents.model.damage_op import DamageOperator
from agents.model.damage_op_layout import _BS_SPE, _COND_PAR_IDX, _NAT_SPE
from agents.model.damage_op_speed import DamageOperatorSpeed
from agents.observation.constants import POKEMON_CONDITION_OFFSET, POKEMON_SPREAD_OFFSET, TEAM_SIZE

B, V, SPECIES, PER_MON = 4, 512, 400, 122


class _Op(DamageOperatorSpeed):
    """Just the attributes the speed mixin reads (`DamageOperator` owns them in the model)."""

    _boost_stages = staticmethod(DamageOperator._boost_stages)
    quick_claw_live = False
    qc_item_num = 0

    def __init__(self, dtype: torch.dtype):
        g = torch.Generator().manual_seed(11)
        self.BASE_STATS = torch.randint(20, 160, (SPECIES, 6), generator=g).to(dtype)
        mix = torch.rand(SPECIES, V, generator=g, dtype=torch.float64) * (torch.rand(SPECIES, V, generator=g) < 0.02)
        mix[:, 300] += 1.0
        self.SPEED_MIX = (mix / mix.sum(-1, keepdim=True)).to(dtype)
        self.stash = SimpleNamespace(x5=None, item_qc_prob=None)


def _ctx(dtype: torch.dtype) -> SimpleNamespace:
    g = torch.Generator().manual_seed(5)
    part = torch.zeros(B, 2 * TEAM_SIZE, PER_MON, dtype=dtype)
    part[..., POKEMON_SPREAD_OFFSET + _BS_SPE] = 1.0                       # IV 31
    part[..., POKEMON_SPREAD_OFFSET + 6 + _BS_SPE] = 1.0                   # EV 252
    part[..., POKEMON_SPREAD_OFFSET + 13 + _NAT_SPE] = torch.tensor([0.9, 1.0, 1.1])[
        torch.randint(0, 3, (B, 2 * TEAM_SIZE), generator=g)].to(dtype)
    part[..., POKEMON_CONDITION_OFFSET + _COND_PAR_IDX] = (torch.rand(B, 2 * TEAM_SIZE, generator=g) < 0.3).to(dtype)

    def raw(stages):                                                       # speed stage s -> [max(0,s)/6, max(0,-s)/6]
        r = torch.zeros(B, 14, dtype=dtype)
        s = torch.tensor(stages, dtype=dtype)
        r[:, 8], r[:, 9] = s.clamp(min=0) / 6.0, (-s).clamp(min=0) / 6.0
        return r
    return SimpleNamespace(batch_size=B, device=torch.device("cpu"), pokemon_part=part,
                           species_ids=torch.randint(1, SPECIES, (B, 2 * TEAM_SIZE), generator=g),
                           item_ids=torch.zeros(B, 2 * TEAM_SIZE, dtype=torch.long),
                           opp_active_local=torch.tensor([0, 3, 5, 1]), our_active_idx=torch.tensor([2, 0, 4, 5]),
                           opp_ctx_raw=raw([-2, -1, 0, 3]), our_ctx_raw=raw([1, 0, -3, 2]))


def test_the_opp_speed_mixture_runs_in_float64_and_matches_float32():
    final64, cum64 = _Op(torch.float64)._opp_speed_mix(_ctx(torch.float64))
    final32, cum32 = _Op(torch.float32)._opp_speed_mix(_ctx(torch.float32))
    assert final64.dtype == torch.float64 and final32.dtype == torch.float32
    assert torch.equal(final64.float(), final32)
    assert torch.allclose(cum64, cum32, atol=1e-6)


def test_p_first_at_a_stage_runs_in_float64_and_matches_float32():
    stage = [2.0, -1.0, 0.0, -3.0]
    p64 = _Op(torch.float64)._p_first_active_at_stage(_ctx(torch.float64), None, torch.tensor(stage, dtype=torch.float64))
    p32 = _Op(torch.float32)._p_first_active_at_stage(_ctx(torch.float32), None, torch.tensor(stage, dtype=torch.float32))
    assert p64.dtype == torch.float64 and p32.dtype == torch.float32
    assert torch.allclose(p64.float(), p32, atol=1e-6)
