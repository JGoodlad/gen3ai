"""Tier H-C (`gen3_event_ref_edges_v1`) — the `r` reference-edge family's contract.

The family is a STRUCTURAL identity (event actor/target IS mon m), so its tests are exact:
the side-gate kills mirror-species false links, PAD rows link nothing, the requirement is
fail-loud, and a full ON forward runs with the seats as rows.
"""
import numpy as np
import pytest
import torch

pytest.importorskip("sb3_contrib")

from agents.model.features_extractor import _event_reference_cells
from agents.observation.constants import EVENT_T_MOVE, EVENT_TOKEN_DIM, EventCol as C


def _ev_row(actor=0.0, side=0.0, target=0.0, valid=1.0):
    r = torch.zeros(EVENT_TOKEN_DIM)
    r[C.TYPE] = EVENT_T_MOVE
    r[C.ACTOR_SPECIES], r[C.ACTOR_SIDE], r[C.TARGET_SPECIES] = actor, side, target
    r[C.VALID] = valid
    return r


def test_reference_cells_side_gated_exact():
    """OUR mon #248 in slot 2; the OPPONENT also runs #248 (slot 6) — a mirror. An OUR-side
    event by #248 must link ONLY our slot; its target #145 (their slot 7) only theirs."""
    species = torch.zeros(1, 12)
    species[0, 2] = 248.0          # ours: tyranitar
    species[0, 6] = 248.0          # theirs: mirror tyranitar
    species[0, 7] = 145.0          # theirs: zapdos
    ev = torch.stack([
        _ev_row(actor=248.0, side=1.0, target=145.0),          # our ttar hits their zapdos
        _ev_row(actor=248.0, side=-1.0, target=0.0),           # THEIR ttar acts
        _ev_row(actor=248.0, side=1.0, target=145.0, valid=0), # PAD: links nothing
    ]).unsqueeze(0)                                            # [1, 3, 19]
    cells = _event_reference_cells(ev, species)
    assert cells.shape == (1, 3, 12, 2)
    # event 0: actor = OUR slot 2 only (never the mirror at 6); target = their slot 7 only
    assert cells[0, 0, :, 0].nonzero().flatten().tolist() == [2]
    assert cells[0, 0, :, 1].nonzero().flatten().tolist() == [7]
    # event 1: actor = THEIR slot 6 only
    assert cells[0, 1, :, 0].nonzero().flatten().tolist() == [6]
    # PAD row: nothing
    assert float(cells[0, 2].abs().sum()) == 0.0


def test_r_requires_history_events():
    from agents.model.identity_init_test import _build_real_policy

    with pytest.raises(ValueError, match="requires --history-events"):
        _build_real_policy(edge_bias_families="r")


def test_r_full_forward_runs_and_is_zero_init():
    from agents.model.identity_init_test import _build_real_policy

    model, enc = _build_real_policy(history_events=True, edge_bias_families="r")
    fe = model.policy.features_extractor
    assert fe.edge_bias.r_map is not None
    assert float(fe.edge_bias.r_map.weight.abs().max()) == 0.0    # identity at init
    rng = np.random.default_rng(0)
    obs = {"observation": torch.as_tensor(rng.random((2, enc.dimension), dtype=np.float32)),
           "action_mask": torch.ones(2, 11)}
    with torch.no_grad():
        pi, vf = fe(obs)
    assert pi.shape == (2, 512) and torch.isfinite(pi).all() and torch.isfinite(vf).all()
