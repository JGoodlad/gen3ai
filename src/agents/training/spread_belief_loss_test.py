"""Unit tests for the SPREAD-belief supervision loss (gen3_unified_spread_belief_v1).

Pure + static — exercises `InstrumentedMaskablePPO._spread_belief_loss` directly (no PPO/env), the way
`move_belief_loss_test` does for the move belief. Validates the masking, the scale-normalised smooth_l1,
the gradient flowing ONLY to supervised slots, the `largest_bias` over-estimate diagnostic, and the
GIGO-guard that the label's stat order matches the DamageOperator's consumption order.
"""
import torch as th

from agents.training.instrumented_ppo import InstrumentedMaskablePPO, InstrumentedMaskablePPO as P, _SPREAD_LOSS_SCALE
from agents.observation.belief_labels import SPREAD_STAT_ORDER, N_SPREAD_STATS
from agents.model.damage_tables import SPREAD_STAT_COLS
from agents.model.features_extractor import _SB_ATK, _SB_DEF, _SB_SPA, _SB_SPD, _SB_SPE


def test_stat_order_matches_op_consumption_order():
    """GIGO GUARD: the label's stat order MUST equal the prior column order AND the op's _SB_ index
    order — otherwise the loss would supervise believed[atk] toward true[def] etc. (the exact
    order-mismatch class). Pin all three to one contract."""
    assert SPREAD_STAT_ORDER == SPREAD_STAT_COLS == ("atk", "def", "spa", "spd", "spe")
    assert (_SB_ATK, _SB_DEF, _SB_SPA, _SB_SPD, _SB_SPE) == (0, 1, 2, 3, 4)
    assert N_SPREAD_STATS == 5


def test_coef_default_is_zero_off_byte_identical():
    """OFF byte-identical guard (regression insurance for the coef gate). The class default coef is 0.0,
    so `spread_belief_on = coef > 0` is False and train() never constructs the spread term → the total
    loss is untouched (the entire non-spread test suite is the byte-identical baseline). Paired with
    `test_off_or_empty_returns_none` (absent labels → loss returns None), the two-layer gate is pinned —
    mirrors `test_value_loss_default_attr_is_zero` for value_tail_weight."""
    assert InstrumentedMaskablePPO.spread_belief_coef == 0.0


def test_off_or_empty_returns_none():
    sb = th.zeros(2, 6, 5, requires_grad=True)
    assert P._spread_belief_loss(None, th.zeros(2, 6, 5), th.ones(2, 6)) is None   # head off
    assert P._spread_belief_loss(sb, None, th.ones(2, 6)) is None                   # no label
    assert P._spread_belief_loss(sb, th.zeros(2, 6, 5), th.zeros(2, 6)) is None     # nothing scored


def test_masks_and_metrics_and_grad():
    B = 2
    sb = th.full((B, 6, 5), 200.0, requires_grad=True)
    target = th.full((B, 6, 5), 250.0)
    mask = th.zeros(B, 6); mask[:, 0] = 1.0; mask[:, 1] = 1.0   # 2 revealed slots/sample
    loss, m = P._spread_belief_loss(sb, target, mask)
    assert m["n_slots"] == 4                                   # 2 slots × 2 samples
    assert abs(m["mae"] - 50.0) < 1e-3                         # raw stat points
    assert abs(m["largest_bias"] + 50.0) < 1e-3               # 200-250 = -50 (under here; sign correct)
    loss.backward()
    g = sb.grad
    assert g[0, 0].abs().sum() > 0                             # supervised slot → gradient
    assert g[0, 2].abs().sum().item() == 0.0                   # un-masked slot → NO gradient (leak-safe)


def test_scale_normalised_smooth_l1():
    """A 100-stat-point error normalises to 1.0 → smooth_l1 at the boundary = 0.5; the coef sets the
    real weight. MAE is reported in RAW points (100)."""
    sb = th.full((1, 6, 5), 100.0)
    target = th.full((1, 6, 5), 200.0)
    mask = th.zeros(1, 6); mask[:, 0] = 1.0
    loss, m = P._spread_belief_loss(sb, target, mask)
    assert abs(float(loss) - 0.5) < 1e-3
    assert abs(m["mae"] - 100.0) < 1e-3
    assert _SPREAD_LOSS_SCALE == 100.0


def test_largest_bias_detects_overestimate_of_max_stat():
    """The 'over-estimates the largest EV' diagnostic: believed max-stat ABOVE true → positive bias."""
    sb = th.zeros(1, 6, 5)
    target = th.zeros(1, 6, 5)
    target[0, 0] = th.tensor([300.0, 100.0, 80.0, 90.0, 110.0])   # largest true stat = atk (idx 0)
    sb[0, 0] = th.tensor([360.0, 100.0, 80.0, 90.0, 110.0])       # believed atk OVER by 60
    mask = th.zeros(1, 6); mask[:, 0] = 1.0
    _, m = P._spread_belief_loss(sb, target, mask)
    assert abs(m["largest_bias"] - 60.0) < 1e-3                   # +60 = over-estimating the max stat
