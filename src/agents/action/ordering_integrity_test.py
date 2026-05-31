"""Integrity tests for move-ordering alignment between the model's view
(sorted-by-id move slots) and the action space (request order).

These reconstruct the state the model actually masks and assert the live check
raises when — and only when — the validity bits would land on the wrong move.
"""
import types
import numpy as np
import pytest
from unittest.mock import MagicMock

from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.move import Move

from agents.action.mask_generator import Gen3ActionMasker
from agents.action.mapper import Gen3ActionMapper
from agents.action.ordering_integrity import (
    OrderingMismatchError,
    check_move_validity_alignment,
    check_switch_ordering_alignment,
    reorder_move_bits_to_sorted,
    assert_sorted_validity_correct,
)
from agents.battle.live_view import LegalActions
from agents.action.constants import MOVE_START


def _move(move_id):
    m = MagicMock(spec=Move)
    m.id = move_id
    return m


def _pokemon(species, move_ids, active=False):
    mon = MagicMock(spec=Pokemon)
    mon.species = species
    mon.moves = {mid: _move(mid) for mid in move_ids}
    mon.active = active
    mon.fainted = False
    return mon


def _battle(active_move_ids, disabled_id=None, struggle=False):
    """Build a minimal battle whose request (action) move order is
    `active_move_ids` and whose active mon owns the same moves (sorted-order is
    derived by the masker/extractor independently)."""
    active = _pokemon("gengar", active_move_ids, active=True)
    bench = [
        _pokemon("metagross", ["meteormash"]),
        _pokemon("starmie", ["surf"]),
    ]
    team = {p.species: p for p in [active, *bench]}

    battle = MagicMock(spec=AbstractBattle)
    battle.turn = 30
    battle.team = team
    battle.active_pokemon = active
    # Bench is switchable; active is not.
    battle.available_switches = bench
    battle.available_moves = (
        [_move("struggle")] if struggle else list(active.moves.values())
    )
    # LegalActions.from_battle reads these poke-env-derived legality flags.
    battle.force_switch = False
    battle.trapped = False
    battle.maybe_trapped = False
    battle.wait = False
    # last_request drives the mask: request order + disabled flags.
    battle.last_request = {
        "active": [{
            "moves": [
                {"id": mid, "disabled": (mid == disabled_id)}
                for mid in active_move_ids
            ]
        }]
    }
    return battle


def test_raw_mask_misalignment_is_detectable():
    """Pre-fix diagnostic: applying the raw action-order mask positionally to the
    sorted move slots IS a misapplication on a disabled-move turn. Documents the
    bug `check_move_validity_alignment` was written to catch."""
    battle = _battle(
        ["thunderbolt", "willowisp", "icepunch", "taunt"],
        disabled_id="willowisp",
    )
    # get_mask returns the raw action-order mask.
    raw_action_mask = Gen3ActionMasker.get_mask(battle)
    legal = LegalActions.from_battle(battle)
    # Applying that raw mask positionally to sorted slots IS a misapplication:
    with pytest.raises(OrderingMismatchError):
        check_move_validity_alignment(battle, raw_action_mask, legal)


def test_reorder_puts_validity_on_the_right_sorted_slot():
    """The fix: reordering the action-order mask into sorted order makes each
    move's bit land on its own sorted slot.

    action order: [thunderbolt, willowisp(0), icepunch, taunt]
    sorted order: [icepunch,    taunt,        thunderbolt, willowisp]
    so sorted bits should be [icepunch=1, taunt=1, thunderbolt=1, willowisp=0].
    """
    action_ids = ["thunderbolt", "willowisp", "icepunch", "taunt"]
    raw = np.zeros(11, dtype=np.float32)
    raw[MOVE_START:MOVE_START + 4] = [1, 0, 1, 1]  # willowisp disabled
    fixed = reorder_move_bits_to_sorted(raw, action_ids)
    assert fixed[MOVE_START:MOVE_START + 4].tolist() == [1, 1, 1, 0]
    # And it validates clean against the action-order source.
    assert_sorted_validity_correct(fixed, raw, action_ids)


def test_assert_catches_a_corrupted_reorder():
    """If a reorder ever produces wrong sorted validity, the assertion raises."""
    action_ids = ["thunderbolt", "willowisp", "icepunch", "taunt"]
    raw = np.zeros(11, dtype=np.float32)
    raw[MOVE_START:MOVE_START + 4] = [1, 0, 1, 1]
    corrupt = raw.copy()  # raw (action order) is NOT the sorted order -> mismatch
    with pytest.raises(OrderingMismatchError):
        assert_sorted_validity_correct(corrupt, raw, action_ids)


def test_all_legal_reorder_is_identity_on_values():
    """All moves legal -> reorder keeps all-ones; harmless permutation."""
    action_ids = ["thunderbolt", "willowisp", "icepunch", "taunt"]
    raw = np.zeros(11, dtype=np.float32)
    raw[MOVE_START:MOVE_START + 4] = [1, 1, 1, 1]
    fixed = reorder_move_bits_to_sorted(raw, action_ids)
    assert fixed[MOVE_START:MOVE_START + 4].tolist() == [1, 1, 1, 1]
    assert_sorted_validity_correct(fixed, raw, action_ids)


def test_switch_ordering_aligned_by_default():
    """Our team uses list(battle.team.values()) everywhere -> aligned, no raise."""
    battle = _battle(["icepunch", "taunt", "thunderbolt", "willowisp"])
    legal = LegalActions.from_battle(battle)
    check_switch_ordering_alignment(battle, np.ones(11, dtype=np.int8), legal)  # explicit


def test_switch_ordering_mismatch_raises():
    """If the encoder's team order ever diverges from the mask/mapper's captured
    snapshot order, a switch action would target the wrong mon -> must raise."""
    battle = _battle(["icepunch", "taunt", "thunderbolt", "willowisp"])
    legal = LegalActions.from_battle(battle)  # capture switches at the original order
    # Simulate drift: the encoder's live team order no longer matches the snapshot.
    reordered = dict(reversed(list(battle.team.items())))
    battle.team = reordered
    with pytest.raises(OrderingMismatchError):
        check_switch_ordering_alignment(battle, np.ones(11, dtype=np.int8), legal)


def test_check_is_noop_without_move_slots():
    """Defensive: a forced-switch snapshot (no move slots) -> nothing to compare, no raise."""
    battle = MagicMock(spec=AbstractBattle)
    battle.turn = 1
    battle.active_pokemon = _pokemon("gengar", ["thunderbolt"], active=True)
    forced_switch_legal = LegalActions(
        move_slots=(), switches=(), force_switch=True, trapped=False,
        maybe_trapped=False, wait=False, struggle=False, last_request=None,
    )
    check_move_validity_alignment(battle, np.ones(11, dtype=np.int8), forced_switch_legal)


# --------------------------------------------------------------------------
# 2b — mapper resolves action index -> the move/switch that index means
# --------------------------------------------------------------------------

def test_mapper_resolves_action_index_to_request_slot():
    """Action 6+k must execute request move k (the contract the model trains on)."""
    battle = _battle(["thunderbolt", "willowisp", "icepunch", "taunt"])
    for k, expected in enumerate(["thunderbolt", "willowisp", "icepunch", "taunt"]):
        order = Gen3ActionMapper.action_to_order(MOVE_START + k, battle)
        assert order.order.id == expected


def test_mapper_raises_on_stale_state():
    """If the server move list changes after the snapshot was captured, acting is
    refused rather than sending the wrong move (fail-loud)."""
    battle = _battle(["thunderbolt", "willowisp", "icepunch", "taunt"])
    snap = LegalActions.from_battle(battle)
    ctx = types.SimpleNamespace(turn=battle.turn, legal=snap, mask=np.ones(11, dtype=np.int8))
    # Server now reports a different move order:
    battle.last_request["active"][0]["moves"] = [
        {"id": m, "disabled": False} for m in ["surf", "icebeam", "thunderbolt", "taunt"]
    ]
    with pytest.raises(RuntimeError):
        Gen3ActionMapper.assert_decision_current(ctx, battle)
