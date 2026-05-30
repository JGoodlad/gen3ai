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
    check_outcome_matches_intent,
    reorder_move_bits_to_sorted,
    assert_sorted_validity_correct,
)
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
    # get_mask returns the raw action-order mask and pins the decision context.
    raw_action_mask = Gen3ActionMasker.get_mask(battle)
    # Applying that raw mask positionally to sorted slots IS a misapplication:
    with pytest.raises(OrderingMismatchError):
        check_move_validity_alignment(battle, raw_action_mask)


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
    Gen3ActionMasker.get_mask(battle)  # pins decision context + checks
    check_switch_ordering_alignment(battle, np.ones(11, dtype=np.int8))  # explicit


def test_switch_ordering_mismatch_raises():
    """If the encoder's team order ever diverges from the mask/mapper's pinned
    order, a switch action would target the wrong mon -> must raise."""
    battle = _battle(["icepunch", "taunt", "thunderbolt", "willowisp"])
    Gen3ActionMasker.get_mask(battle)  # pins team_species in decision context
    # Simulate drift: the encoder's live team order no longer matches the pin.
    reordered = dict(reversed(list(battle.team.items())))
    battle.team = reordered
    with pytest.raises(OrderingMismatchError):
        check_switch_ordering_alignment(battle, np.ones(11, dtype=np.int8))


def test_check_is_noop_without_decision_context():
    """Defensive: no pinned decision context -> nothing to compare, no raise."""
    battle = MagicMock(spec=AbstractBattle)
    battle.turn = 1
    battle.active_pokemon = _pokemon("gengar", ["thunderbolt"], active=True)
    # no _gen3_decision_context attribute
    del battle._gen3_decision_context
    check_move_validity_alignment(battle, np.ones(11, dtype=np.int8))


# --------------------------------------------------------------------------
# 2b — mapper resolves action index -> the move/switch that index means
# --------------------------------------------------------------------------

def test_mapper_resolves_action_index_to_request_slot():
    """Action 6+k must execute request move k (the contract the model trains on)."""
    battle = _battle(["thunderbolt", "willowisp", "icepunch", "taunt"])
    Gen3ActionMasker.get_mask(battle)  # pins decision context
    for k, expected in enumerate(["thunderbolt", "willowisp", "icepunch", "taunt"]):
        order = Gen3ActionMapper.action_to_order(MOVE_START + k, battle)
        assert order.order.id == expected


def test_mapper_raises_on_stale_state():
    """If the server move list changes after the mask was latched, acting is
    refused rather than sending the wrong move (2b fail-loud)."""
    battle = _battle(["thunderbolt", "willowisp", "icepunch", "taunt"])
    Gen3ActionMasker.get_mask(battle)  # latches move_ids
    # Server now reports a different move order:
    battle.last_request["active"][0]["moves"] = [
        {"id": m, "disabled": False} for m in ["surf", "icebeam", "thunderbolt", "taunt"]
    ]
    with pytest.raises(RuntimeError):
        Gen3ActionMapper.action_to_order(MOVE_START, battle)


# --------------------------------------------------------------------------
# 2c — outcome vs intent (what fired must match what we pressed)
# --------------------------------------------------------------------------

def _delta(**kw):
    d = types.SimpleNamespace(
        our_move_id=None, our_switch_to=None, our_failed_to_move=False,
        our_damaging_event=None, phase_is_forced_switch=False, turn=33,
    )
    for k, v in kw.items():
        setattr(d, k, v)
    return d


REQ = ["thunderbolt", "willowisp", "icepunch", "taunt"]  # action order


def test_2c_matching_move_no_raise():
    check_outcome_matches_intent(REQ, _delta(our_move_id="thunderbolt"), MOVE_START + 0)
    check_outcome_matches_intent(REQ, _delta(our_move_id="taunt"), MOVE_START + 3)


def test_2c_mismatched_move_raises():
    # Pressed action 0 (thunderbolt) but the game shows we used icepunch.
    with pytest.raises(OrderingMismatchError):
        check_outcome_matches_intent(REQ, _delta(our_move_id="icepunch"), MOVE_START + 0)


def test_2c_cant_is_skipped():
    # Fully paralysed: we pressed thunderbolt, nothing fired — not a mapping bug.
    check_outcome_matches_intent(
        REQ, _delta(our_move_id=None, our_failed_to_move=True), MOVE_START + 0
    )


def test_2c_forced_switch_is_skipped():
    check_outcome_matches_intent(
        REQ, _delta(our_switch_to="metagross", phase_is_forced_switch=True), 1
    )


def test_2c_action_kind_mismatch_raises():
    # Chose a MOVE but we switched.
    with pytest.raises(OrderingMismatchError):
        check_outcome_matches_intent(REQ, _delta(our_switch_to="metagross"), MOVE_START + 0)
    # Chose a SWITCH but we used a move.
    with pytest.raises(OrderingMismatchError):
        check_outcome_matches_intent(REQ, _delta(our_move_id="thunderbolt"), 1)


def test_2c_voluntary_switch_no_raise():
    check_outcome_matches_intent(REQ, _delta(our_switch_to="metagross"), 1)


def test_2c_sleeptalk_caller_no_raise():
    # Sleep Talk (action 9 here) invokes a random move; protocol shows the called
    # move, not sleeptalk — legitimate, must not raise.
    req = ["curse", "bodyslam", "rest", "sleeptalk"]
    check_outcome_matches_intent(req, _delta(our_move_id="bodyslam"), MOVE_START + 3)


def test_2c_prefers_damaging_event_move_id():
    # protocol-truth move id from the damaging event wins over last_move
    ev = types.SimpleNamespace(move_id="thunderbolt")
    check_outcome_matches_intent(
        REQ, _delta(our_move_id="stale", our_damaging_event=ev), MOVE_START + 0
    )


def test_2c_non_strict_returns_message_instead_of_raising():
    # The live training/replay path calls with strict=False: a mismatch must NOT
    # raise (it would crash the run), it returns the message to log. This is the
    # Gengar case from the run crash — pending action 9 (hypnosis) paired with a
    # delta whose protocol move is thunderbolt across a gap=0 turn.
    req = ["thunderbolt", "explosion", "willowisp", "hypnosis"]
    msg = check_outcome_matches_intent(
        req, _delta(our_move_id="thunderbolt"), MOVE_START + 3, strict=False
    )
    assert isinstance(msg, str) and "thunderbolt" in msg and "hypnosis" in msg


def test_2c_non_strict_returns_none_when_aligned():
    # No mismatch under strict=False returns None (nothing to log).
    assert (
        check_outcome_matches_intent(
            REQ, _delta(our_move_id="thunderbolt"), MOVE_START + 0, strict=False
        )
        is None
    )


def test_2c_strict_default_still_raises():
    # Default strict=True is unchanged so the unit-test assertions stay valid.
    with pytest.raises(OrderingMismatchError):
        check_outcome_matches_intent(REQ, _delta(our_move_id="icepunch"), MOVE_START + 0)
