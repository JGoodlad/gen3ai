"""`tie_margins.SelectionGaps` records, per row, the gap of every discrete SELECTION (K9(b)'s rule (iii)):
a topk's k-th minus (k+1)-th, an argmax's / max-with-dim's top-1 minus top-2, a sort's smallest adjacent
gap; an exact tie is 0; a reduction that selects nothing (a global max, amax) is not recorded."""
from __future__ import annotations

import numpy as np
import pytest
import torch as th

from agents.training.rust_rollout.tie_margins import SelectionGaps


def _run(fn, rows):
    mode = SelectionGaps(rows)
    with mode:
        fn()
    return mode


def test_a_topk_records_the_gap_between_the_kth_and_the_next_candidate_per_row():
    x = th.tensor([[0.9, 0.5, 0.5 - 3e-6, 0.1], [0.9, 0.8, 0.2, 0.1]], dtype=th.float64)
    m = _run(lambda: x.topk(2, dim=-1), 2)
    assert m.min_gap[0] == pytest.approx(3e-6, rel=1e-3) and m.min_gap[1] == pytest.approx(0.6, rel=1e-6)
    assert m.site[0].startswith("topk@") and m.ops == 1


def test_argmax_and_max_with_a_dim_record_the_top_two_gap_and_an_exact_tie_is_zero():
    x = th.tensor([[0.3, 0.3, 0.1], [0.7, 0.2, 0.1]])
    m = _run(lambda: (x.argmax(dim=-1), th.max(x, dim=1)), 2)
    assert m.min_gap[0] == 0.0 and m.min_gap[1] == pytest.approx(0.5, rel=1e-6) and m.ops == 2


def test_a_reduction_that_selects_nothing_is_not_recorded_and_rows_fold_from_a_reshaped_batch():
    x = th.rand(3, 4)
    m = _run(lambda: (x.max(), x.amax(dim=-1), th.maximum(x, x)), 3)
    assert m.ops == 0 and np.isinf(m.min_gap).all()
    y = th.tensor([[1.0, 1.0 - 1e-7], [1.0, 0.0], [5.0, 1.0], [5.0, 4.0]], dtype=th.float64)  # [rows x 2 slots, M]
    m = _run(lambda: y.topk(1, dim=-1), 2)
    assert m.min_gap[0] == pytest.approx(1e-7, rel=1e-2) and m.min_gap[1] == pytest.approx(1.0)
