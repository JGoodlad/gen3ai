"""The Beat Up column index is BOUNDED by construction (`gen3_beatup_pidx_bounded_v1`, 2026-10-03).

`_damage_rolls` picks the attack / defence column of a 3-wide stack (special, physical, Beat Up). It used to
SUM the bits, ``pidx = phys + 2 * is_bu``, which reads 3 — out of range — on any candidate carrying both bits.
No real move does (Beat Up is Special in gen 3, ``MOVE_PHYS[beatup] == 0``; pinned below), yet the T2 AOT
package's Inductor kernel tripped its indirect-index assert on exactly that expression
(``service_cuda_test::test_aot_backend_one_package_per_bucket_serves_every_slot``, device-side assert
"index out of bounds: 0 <= tmp76 < 3" at 336ddd27; passes at 5876c2ea). The index is now SELECTED
(``torch.where(is_bu, 2, phys)``), so it is in {0, 1, 2} whatever the bits read.

These tests plant the impossible case — the phys bit set on Beat Up — and require the op to (a) not index
out of range and (b) still price Beat Up off its own column (party Σ base Atk vs the target's base Def): the
planted forward moves the same damage from the special channel to the physical one, nothing else. On the
reverted sum, the CPU gather raises ``index 3 is out of bounds``.
"""
from __future__ import annotations

import pytest
import torch

from agents.model.beatup_damage_test import _BEATUP, _PARTY, _incoming_ctx
from agents.model.damage_op_layout import _DMG_IDX_PHYS_HIGH, _DMG_IDX_SPEC_HIGH, _DMG_PER_MON
from agents.model.damage_op_test import _believe_op, _make_layout
from agents.model.features_extractor import DamageOperator, TEAM_SIZE


def _per_mon(op, ctx):
    pm = op(ctx, _believe_op(op, "beatup"))[:, :TEAM_SIZE * _DMG_PER_MON]
    return pm.reshape(1, TEAM_SIZE, _DMG_PER_MON)[0, 0]


def test_no_real_move_carries_both_the_phys_and_the_beat_up_bit():
    op = DamageOperator(_make_layout(), outgoing=True)
    both = (op.MOVE_PHYS > 0.5) & (op.MOVE_BEATUP > 0)
    assert int(op.MOVE_BEATUP.sum()) == 1 and float(op.MOVE_PHYS[_BEATUP]) == 0.0
    assert not bool(both.any())


def test_a_beat_up_candidate_with_the_phys_bit_still_selects_the_beat_up_column():
    ref_op = DamageOperator(_make_layout(), outgoing=True)
    ref = _per_mon(ref_op, _incoming_ctx(ref_op, opp_party=_PARTY))
    spec = float(ref[_DMG_IDX_SPEC_HIGH])
    assert spec > 0.1 and float(ref[_DMG_IDX_PHYS_HIGH]) < 1e-3 * spec   # precondition: Beat Up owns the read

    op = DamageOperator(_make_layout(), outgoing=True)
    planted = op.MOVE_PHYS.clone()
    planted[_BEATUP] = 1.0                       # the impossible row: phys AND Beat Up
    op.MOVE_PHYS.copy_(planted)
    got = _per_mon(op, _incoming_ctx(op, opp_party=_PARTY))   # the reverted sum raises here (index 3)
    # the same Beat Up damage, now on the physical channel (the column is Beat Up's, not Atk / Def)
    assert float(got[_DMG_IDX_PHYS_HIGH]) == pytest.approx(float(ref[_DMG_IDX_SPEC_HIGH]), rel=1e-5)
    assert float(got[_DMG_IDX_SPEC_HIGH]) < 1e-3 * spec


def test_the_index_expression_stays_in_range_on_every_bit_pattern():
    """The op's own expression on all four (phys, bu) patterns: {0, 1, 2}, Beat Up wins."""
    phys_all = torch.tensor([[0.0, 1.0, 0.0, 1.0]])
    is_bu = torch.tensor([[False, False, True, True]])
    pidx = torch.where(is_bu, 2, (phys_all > 0.5).long())
    assert pidx.tolist() == [[0, 1, 2, 2]]
    src = open(DamageOperator._damage_rolls.__code__.co_filename).read()
    assert "pidx = torch.where(is_bu, 2, (phys_all > 0.5).long())" in src
    assert "2 * is_bu.long()" not in src
