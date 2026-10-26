"""Engine regressions from the 2026-10-26 prober health audit against TODAY's runs (pure, no torch).

`engine_test.py` builds its observation from a synthetic layout that still carries the matchup,
incoming-damage and turn-history blocks. The row a current run plays is 2845 dims with none of them,
so every case here pins the LIVE layout (`ObsOffsets.resolve()`) or the live shape of the offsets
(`mm_off == 0`). Each fails with its fix reverted.
"""

import numpy as np

import agents.observation.constants as C
from main.prober.engine import analyze_invocation
from main.prober.engine.decode import _saliency_from_grad, current_layout_regions
from main.prober.engine_test import FakeProbeModel, _npz, _summary
from main.prober.model import ObsOffsets

# the offsets a current run ships: every matchup / multiplier / incoming / turn-history block gone
_NO_BLOCKS = ObsOffsets(mm_off=0, om_off=0, tm_off=0, active_block_dim=5)


class _FlatModel(FakeProbeModel):
    """`FakeProbeModel` reads the multiplier cell at `mm_off`, which a current layout does not have."""

    def action_dist(self, obs, mask):
        probs = np.ones(len(mask)) * mask
        return probs / probs.sum(), np.arange(len(mask), dtype=np.float64)


def test_the_saliency_regions_tile_the_whole_current_row():
    """On a current run the saliency named ONE block, `obs[0:99]` labelled "our active pokemon
    block(99)" (it is team slot 0, not the active mon): 3.5 % of the row, the rest unnamed."""
    regions = current_layout_regions()
    total = C.OFFSET_OBS_FACTS + C.OBS_FACTS_DIM
    assert regions[0][1] == 0 and regions[-1][2] == total
    for (_, _, hi), (_, lo, _) in zip(regions, regions[1:]):
        assert hi == lo, "a gap or an overlap between two named regions"
    names = [n for n, _, _ in regions]
    assert len(set(names)) == len(names)
    assert {"event window(960)", "obs facts(84)", "pair history(180)"} <= set(names)
    assert not any("active pokemon block" in n for n in names)


def test_current_layout_saliency_reports_every_region_and_conserves_the_total():
    off = ObsOffsets.resolve()
    total = off.total_dim
    assert total == C.OFFSET_OBS_FACTS + C.OBS_FACTS_DIM
    g = np.abs(np.random.default_rng(0).normal(size=total))
    sal = _saliency_from_grad(g, off)
    assert [b.name for b in sal.blocks] == [n for n, _, _ in current_layout_regions()]
    assert abs(sum(b.total_abs for b in sal.blocks) - g.sum()) < 1e-6      # nothing unnamed
    assert abs(sal.overall_mean_abs - g.mean()) < 1e-12


def test_analyze_names_the_regions_of_the_current_row_end_to_end():
    off = ObsOffsets.resolve()
    model = _FlatModel(off)
    npz = _npz()
    npz["obs"] = np.zeros((1, off.total_dim), dtype=np.float32)
    a = analyze_invocation(model, _summary(), npz, 0)
    names = [b.name for b in a.saliency.blocks]
    assert "event window(960)" in names and "our active pokemon block(99)" not in names
    assert [b.name for b in a.value_saliency.blocks] == names


def test_a_run_with_no_multiplier_block_has_no_matchup_numbers_and_no_sweep():
    """`mm_off == 0` (the shipped layout): the old code returned multipliers [0, 0, 0, 0] with
    `applicable` true on every attack — the web rendered "0.00×" (immune) beside Rapid Spin — and a
    sweep whose four rows were the same re-run, read by the page as "the policy ignores type"."""
    a = analyze_invocation(_FlatModel(_NO_BLOCKS), _summary("thunderbolt"), _npz(), 0)
    assert a.matchups.present is False
    assert a.matchups.multipliers == () and not any(a.matchups.applicable)
    assert a.matchups.move_labels                      # the action table is still keyed on them
    assert a.sweep is None


def test_a_run_with_the_multiplier_block_still_gets_both():
    from main.prober.engine_test import _OFF
    a = analyze_invocation(FakeProbeModel(_OFF), _summary("thunderbolt"), _npz(), 0)
    assert a.matchups.present is True and len(a.matchups.multipliers) == 4
    assert a.sweep is not None and a.sweep.applicable and len(a.sweep.rows) == 4
