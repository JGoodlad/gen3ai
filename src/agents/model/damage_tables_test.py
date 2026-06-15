"""Unit tests for `damage_tables.build_move_prior_logits` — the legacy floor vs the version-gated
learnset + <2% rarity cap (the unified-damage candidate-bounding)."""
import math

import pytest
import torch

from agents import gen3_data
from agents.model.damage_tables import build_move_prior_logits, HIDDEN_POWER_NUM, _PRIOR_FLOOR

_N_SPECIES = 600
_N_MOVES = 600
_EPS = 1e-6
_LOGIT_EPS = math.log(_EPS / (1.0 - _EPS))
_LOGIT_FLOOR = math.log(_PRIOR_FLOOR / (1.0 - _PRIOR_FLOOR))


def _num(species_id, move_id):
    return gen3_data.species.get(species_id).num, gen3_data.moves.get(move_id).num


def test_ungated_is_legacy_floor_behavior():
    # OFF (default) must reproduce the original buffer: every cell floored to >= logit(floor),
    # the unset/never-used cells sitting exactly at logit(floor).
    logits = build_move_prior_logits(_N_SPECIES, _N_MOVES)
    assert torch.isfinite(logits).all()
    assert logits.min().item() == pytest.approx(_LOGIT_FLOOR, abs=1e-4)   # never-used cells = floor
    # A staple move sits above the floor (its real usage prior).
    snum, mnum = _num("skarmory", "spikes")
    assert logits[snum, mnum].item() > _LOGIT_FLOOR + 0.1


def test_gated_prunes_illegal_move_to_eps():
    gated = build_move_prior_logits(_N_SPECIES, _N_MOVES, learnset_gate=True)
    # Skarmory cannot learn Fire Blast → pruned to the impossible default (~logit(eps)).
    snum, fb = _num("skarmory", "fireblast")
    assert gated[snum, fb].item() == pytest.approx(_LOGIT_EPS, abs=1e-3)
    # But a legal, frequently-run move keeps its real prior (same accumulation as ungated).
    ungated = build_move_prior_logits(_N_SPECIES, _N_MOVES)
    _, spikes = _num("skarmory", "spikes")
    assert gated[snum, spikes].item() == pytest.approx(ungated[snum, spikes].item(), abs=1e-4)
    assert gated[snum, spikes].item() > _LOGIT_FLOOR


def test_gated_keeps_legal_rare_move_liftable():
    """Legality-ONLY gate: a move a species CAN learn but runs below the floor is NOT pruned — it keeps a
    liftable prior (>= floor, far above the impossible eps), so the learned head can still anticipate a
    surprise tech (the old <2% prune-to-zero is gone)."""
    victim = None
    for sid in gen3_data.species.raw():
        legal = gen3_data.learnset.get_legal_moves(sid)
        if legal is None:
            continue
        for mid, p in gen3_data.priors.moves(sid).items():
            md = gen3_data.moves.get(mid)
            if md is None or md.num == HIDDEN_POWER_NUM:
                continue
            legal_id = "hiddenpower" if mid.startswith("hiddenpower") else mid
            if legal_id in legal and 0.0 < p < _PRIOR_FLOOR:
                victim = (gen3_data.species.get(sid).num, md.num)
                break
        if victim:
            break
    assert victim is not None, "expected at least one legal sub-2% move in the priors"
    gated = build_move_prior_logits(_N_SPECIES, _N_MOVES, learnset_gate=True)
    snum, mnum = victim
    # kept at the liftable floor (its sub-floor usage rounds up to the floor base) — NOT pruned to eps.
    assert gated[snum, mnum].item() == pytest.approx(_LOGIT_FLOOR, abs=1e-3)
    assert gated[snum, mnum].item() > _LOGIT_EPS + 5.0          # decisively NOT impossible


def test_gated_hidden_power_present_for_hp_user():
    # HP presence = the SUMMED legal HP-type usages, gated once — an HP user keeps a real HP prior.
    gated = build_move_prior_logits(_N_SPECIES, _N_MOVES, learnset_gate=True)
    snum = gen3_data.species.get("zapdos").num
    assert gated[snum, HIDDEN_POWER_NUM].item() > _LOGIT_EPS + 1.0   # not pruned


def test_gated_default_cell_is_eps():
    # An out-of-prior / impossible cell defaults to logit(eps), not the legacy logit(floor).
    gated = build_move_prior_logits(_N_SPECIES, _N_MOVES, learnset_gate=True)
    assert gated.min().item() == pytest.approx(_LOGIT_EPS, abs=1e-4)
