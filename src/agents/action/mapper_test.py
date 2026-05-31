"""
Unit tests for the Gen 3 action path: the masker, the PURE action→Choice decoder,
the serialization adapter, the staleness guard, and the reverse order→action mapping.

The decode + mask logic is driven by a ``LegalActions`` STUB (no battle object) — that
pure-function seam is the point. The serialization adapter and the reverse map are
exercised with minimal mock battles.
"""
import pytest
import numpy as np
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from agents.action.constants import (
    ACTION_SPACE_SIZE, SWITCH_START, SWITCH_END, N_SWITCH_SLOTS,
    MOVE_START, MOVE_END, N_MOVE_SLOTS, STRUGGLE,
)
from agents.action.choice import Choice, ChoiceKind
from agents.action.mask_generator import Gen3ActionMasker
from agents.action.mapper import Gen3ActionMapper
from agents.action.serialize import choice_to_order, order_to_action
from agents.battle.live_view import LegalActions, LegalMove, LegalSwitch
from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.player.battle_order import SingleBattleOrder


# ---------------------------------------------------------------------------
# Pure LegalActions stub helpers (no battle)
# ---------------------------------------------------------------------------

def _lm(move_id: str, disabled: bool = False, pp: int = 16) -> LegalMove:
    return LegalMove(id=move_id, current_pp=pp, max_pp=16, disabled=disabled, target="normal")


def _ls(species: str, slot: int) -> LegalSwitch:
    return LegalSwitch(species=species, slot=slot)


def _legal(*, move_slots=(), switches=(), struggle=False, force_switch=False,
           trapped=False, maybe_trapped=False, wait=False, last_request=None) -> LegalActions:
    return LegalActions(
        move_slots=tuple(move_slots),
        switches=tuple(switches),
        force_switch=force_switch,
        trapped=trapped,
        maybe_trapped=maybe_trapped,
        wait=wait,
        struggle=struggle,
        last_request=last_request,
    )


# ---------------------------------------------------------------------------
# Mock-battle helpers (for serialization + get_mask + order_to_action + from_battle)
# ---------------------------------------------------------------------------

def _mon(species: str):
    # spec=Pokemon so isinstance(order.order, Pokemon) holds in the reverse map /
    # the send-path conformance round-trip.
    m = MagicMock(spec=Pokemon)
    m.species = species
    return m


def _move(move_id: str):
    m = MagicMock(spec=Move)
    m.id = move_id
    return m


def _make_battle(
    team_species: list,
    available_switch_indices: list,
    move_ids: list,
    disabled_indices=(),
    struggle: bool = False,
    turn: int = 1,
    tamper_server_moves=None,
):
    """A minimal mock battle whose ``LegalActions.from_battle`` yields the requested
    legality. ``tamper_server_moves`` makes ``last_request`` report different move ids
    than ``move_ids`` (simulates a mid-decision server change)."""
    battle = MagicMock()
    battle.turn = turn

    mons = [_mon(s) for s in team_species]
    battle.team = {m.species: m for m in mons}
    battle.available_switches = [mons[i] for i in available_switch_indices]

    available_moves = [_move(mid) for mid in move_ids]
    if struggle:
        available_moves.append(_move("struggle"))
    battle.available_moves = available_moves

    server_move_ids = tamper_server_moves if tamper_server_moves is not None else move_ids
    request_moves = [
        {"id": mid, "move": mid, "disabled": (i in disabled_indices)}
        for i, mid in enumerate(server_move_ids)
    ]
    battle.last_request = {
        "active": [{"moves": request_moves}],
        "side": {"pokemon": [{"ident": f"p1: {s}"} for s in team_species]},
        "turn": turn,
    }
    # LegalActions.from_battle reads these poke-env-derived flags.
    battle.force_switch = False
    battle.trapped = False
    battle.maybe_trapped = False
    battle.wait = False

    return battle, mons, available_moves


# ===========================================================================
# Action space constants
# ===========================================================================

class TestActionSpaceConstants:

    def test_space_size(self):
        assert ACTION_SPACE_SIZE == 11

    def test_switch_range(self):
        assert SWITCH_START == 0
        assert SWITCH_END == 6
        assert N_SWITCH_SLOTS == 6

    def test_move_range(self):
        assert MOVE_START == 6
        assert MOVE_END == 10
        assert N_MOVE_SLOTS == 4

    def test_struggle_index(self):
        # The reward manager's struggle-loop tax depends on this index. Lock it.
        assert STRUGGLE == 10

    def test_ranges_are_contiguous_and_cover_space(self):
        assert SWITCH_END == MOVE_START
        assert MOVE_END == STRUGGLE
        assert STRUGGLE + 1 == ACTION_SPACE_SIZE


# ===========================================================================
# Masker — mask_from_legal (PURE, stub-driven) + get_mask (battle)
# ===========================================================================

class TestMaskFromLegal:

    def test_mask_shape_and_dtype(self):
        mask = Gen3ActionMasker.mask_from_legal(_legal(move_slots=(_lm("rockslide"),)))
        assert mask.shape == (ACTION_SPACE_SIZE,)
        assert mask.dtype == np.int8

    def test_available_switches_set_by_slot(self):
        legal = _legal(switches=(_ls("skarmory", 1), _ls("blissey", 3), _ls("starmie", 5)))
        mask = Gen3ActionMasker.mask_from_legal(legal)
        assert mask[:6].tolist() == [0, 1, 0, 1, 0, 1]

    def test_no_switches_set_when_none_available(self):
        mask = Gen3ActionMasker.mask_from_legal(_legal(move_slots=(_lm("rockslide"),)))
        assert mask[:6].tolist() == [0, 0, 0, 0, 0, 0]

    def test_non_disabled_moves_set(self):
        legal = _legal(move_slots=(_lm("rockslide"), _lm("earthquake"), _lm("fireblast"), _lm("icebeam")))
        mask = Gen3ActionMasker.mask_from_legal(legal)
        assert mask[6:10].tolist() == [1, 1, 1, 1]

    def test_disabled_moves_not_set(self):
        legal = _legal(move_slots=(
            _lm("rockslide"), _lm("earthquake", disabled=True),
            _lm("fireblast"), _lm("icebeam", disabled=True),
        ))
        mask = Gen3ActionMasker.mask_from_legal(legal)
        assert mask[6:10].tolist() == [1, 0, 1, 0]

    def test_struggle_sets_only_bit_10(self):
        # struggle=True, no move slots — the single-source contract.
        legal = _legal(move_slots=(), struggle=True)
        mask = Gen3ActionMasker.mask_from_legal(legal)
        assert mask[STRUGGLE] == 1
        assert mask[MOVE_START:MOVE_END].tolist() == [0, 0, 0, 0]

    def test_no_struggle_bit_when_absent(self):
        legal = _legal(move_slots=(_lm("rockslide"),))
        assert Gen3ActionMasker.mask_from_legal(legal)[STRUGGLE] == 0


class TestGetMaskBattle:

    def test_no_last_request_raises(self):
        battle = MagicMock()
        battle.last_request = None
        with pytest.raises(RuntimeError, match="No last_request"):
            Gen3ActionMasker.get_mask(battle)

    def test_duplicate_species_raises(self):
        mon_a, mon_b, mon_c = _mon("gengar"), _mon("gengar"), _mon("skarmory")
        battle = MagicMock()
        battle.turn = 1
        battle.team = {"p1: gengar1": mon_a, "p1: gengar2": mon_b, "p1: skarmory": mon_c}
        battle.available_switches = [mon_b]
        battle.available_moves = [_move("surf")]
        battle.force_switch = battle.trapped = battle.maybe_trapped = battle.wait = False
        battle.last_request = {"active": [{"moves": [{"id": "surf", "disabled": False}]}], "turn": 1}
        with pytest.raises(RuntimeError, match="Duplicate species"):
            Gen3ActionMasker.get_mask(battle)

    def test_get_mask_switch_and_move_bits(self):
        battle, _, _ = _make_battle(
            team_species=["tyranitar", "skarmory", "gengar"],
            available_switch_indices=[1, 2],
            move_ids=["rockslide", "earthquake"],
            disabled_indices=[1],
        )
        mask = Gen3ActionMasker.get_mask(battle)
        assert mask[:6].tolist() == [0, 1, 1, 0, 0, 0]
        assert mask[6:10].tolist() == [1, 0, 0, 0]

    def test_get_mask_uses_supplied_legal_snapshot(self):
        # When the caller passes a captured legal, the mask reflects IT, even if the
        # live battle has drifted (the snapshot is the source of truth).
        battle, _, _ = _make_battle(
            team_species=["tyranitar"], available_switch_indices=[], move_ids=["rockslide"],
        )
        snap = _legal(move_slots=(_lm("rockslide"), _lm("earthquake")))
        mask = Gen3ActionMasker.get_mask(battle, legal=snap)
        assert mask[6:10].tolist() == [1, 1, 0, 0]


# ===========================================================================
# Struggle single-source: LegalActions.from_battle never lists struggle as a slot
# ===========================================================================

class TestStruggleSingleSource:

    def test_struggle_filtered_out_of_move_slots(self):
        # The server sends a lone `struggle` entry when all PP is gone. It must
        # surface ONLY as the flag, never as a move slot (the double-enabling guard).
        battle, _, _ = _make_battle(
            team_species=["tyranitar"], available_switch_indices=[], move_ids=["struggle"],
            struggle=True,
        )
        legal = LegalActions.from_battle(battle)
        assert legal.struggle is True
        assert all(m.id != "struggle" for m in legal.move_slots)
        assert "struggle" not in legal.move_ids

    def test_struggle_turn_masks_only_bit_10(self):
        battle, _, _ = _make_battle(
            team_species=["tyranitar"], available_switch_indices=[], move_ids=["struggle"],
            struggle=True,
        )
        mask = Gen3ActionMasker.get_mask(battle)
        assert mask[MOVE_START:MOVE_END].tolist() == [0, 0, 0, 0], "no move slot for struggle"
        assert mask[STRUGGLE] == 1


# ===========================================================================
# action_to_choice — PURE decode against a LegalActions stub
# ===========================================================================

class TestActionToChoice:

    def test_switch_decodes_to_switch_choice(self):
        legal = _legal(switches=(_ls("skarmory", 1), _ls("gengar", 2)),
                       move_slots=(_lm("rockslide"),))
        c = Gen3ActionMapper.action_to_choice(1, legal)
        assert c == Choice.switch("skarmory", 1)
        assert c.kind is ChoiceKind.SWITCH

    def test_switch_not_legal_raises(self):
        legal = _legal(switches=(_ls("skarmory", 1),), move_slots=(_lm("rockslide"),))
        with pytest.raises(ValueError, match="not a legal switch"):
            Gen3ActionMapper.action_to_choice(2, legal)

    def test_move_decodes_to_move_choice(self):
        legal = _legal(move_slots=(_lm("rockslide"), _lm("earthquake")))
        assert Gen3ActionMapper.action_to_choice(6, legal) == Choice.move("rockslide", 0)
        assert Gen3ActionMapper.action_to_choice(7, legal) == Choice.move("earthquake", 1)

    def test_disabled_move_raises(self):
        legal = _legal(move_slots=(_lm("rockslide"), _lm("earthquake", disabled=True)))
        with pytest.raises(ValueError, match="disabled"):
            Gen3ActionMapper.action_to_choice(7, legal)

    def test_move_slot_out_of_range_raises(self):
        legal = _legal(move_slots=(_lm("rockslide"),))
        with pytest.raises(ValueError, match="out of range"):
            Gen3ActionMapper.action_to_choice(8, legal)

    def test_struggle_decodes_when_legal(self):
        legal = _legal(struggle=True, switches=(_ls("skarmory", 1),))
        assert Gen3ActionMapper.action_to_choice(STRUGGLE, legal) == Choice.struggle()

    def test_struggle_when_not_legal_raises(self):
        # Amendment 3a: pressing 10 when the server isn't forcing struggle is rejected.
        legal = _legal(move_slots=(_lm("rockslide"),), struggle=False)
        with pytest.raises(ValueError, match="not forcing"):
            Gen3ActionMapper.action_to_choice(STRUGGLE, legal)


# ===========================================================================
# choice_to_order — the single poke-env serialization touch
# ===========================================================================

class TestChoiceToOrder:

    def test_switch_serializes_to_correct_mon(self):
        battle, mons, _ = _make_battle(["tyranitar", "skarmory", "gengar"], [1, 2], ["rockslide"])
        order = choice_to_order(Choice.switch("skarmory", 1), battle)
        assert isinstance(order, SingleBattleOrder)
        assert order.order == mons[1]

    def test_switch_not_currently_available_raises(self):
        battle, _, _ = _make_battle(["tyranitar", "skarmory"], [], ["rockslide"])
        with pytest.raises(ValueError, match="not a currently"):
            choice_to_order(Choice.switch("skarmory", 1), battle)

    def test_move_serializes_to_correct_move(self):
        battle, _, avail = _make_battle(["tyranitar"], [], ["rockslide", "earthquake"])
        order = choice_to_order(Choice.move("earthquake", 1), battle)
        assert order.order == avail[1]

    def test_move_not_in_available_moves_raises(self):
        battle, _, _ = _make_battle(["tyranitar"], [], ["surf"])
        battle.available_moves = [_move("rockslide")]  # poke-env desync
        with pytest.raises(ValueError, match="not found in available_moves"):
            choice_to_order(Choice.move("surf", 0), battle)

    def test_hiddenpower_variant_matches(self):
        battle, _, avail = _make_battle(["jolteon"], [], ["hiddenpower"])
        avail[0].id = "hiddenpowerice"  # poke-env resolves the typed id
        order = choice_to_order(Choice.move("hiddenpower", 0), battle)
        assert order.order == avail[0]

    def test_struggle_from_available_moves(self):
        battle, _, _ = _make_battle(["tyranitar"], [], [], struggle=True)
        order = choice_to_order(Choice.struggle(), battle)
        assert order.order.id == "struggle"

    def test_struggle_fallback_creates_move_object(self):
        battle, _, _ = _make_battle(["tyranitar"], [], [])
        battle.available_moves = []
        order = choice_to_order(Choice.struggle(), battle)
        assert order.order.id == "struggle"


# ===========================================================================
# action_to_order — convenience composition (decode + serialize)
# ===========================================================================

class TestActionToOrder:

    def test_switch_returns_correct_mon(self):
        battle, mons, _ = _make_battle(["tyranitar", "skarmory", "gengar"], [1, 2], ["rockslide"])
        order = Gen3ActionMapper.action_to_order(1, battle)
        assert order.order == mons[1]

    def test_move_returns_correct_order(self):
        battle, _, avail = _make_battle(["tyranitar"], [], ["rockslide", "earthquake"])
        assert Gen3ActionMapper.action_to_order(6, battle).order == avail[0]
        assert Gen3ActionMapper.action_to_order(7, battle).order == avail[1]

    def test_uses_supplied_legal_snapshot(self):
        # The env/player pass the captured snapshot; a drifted live battle is ignored.
        battle, _, avail = _make_battle(["tyranitar"], [], ["rockslide", "earthquake"])
        snap = LegalActions.from_battle(battle)
        battle.last_request["active"][0]["moves"] = [{"id": "surf", "disabled": False}]
        order = Gen3ActionMapper.action_to_order(6, battle, legal=snap)
        assert order.order == avail[0]  # resolved via the snapshot, not the drifted request

    def test_illegal_action_with_mask_raises(self):
        battle, _, _ = _make_battle(["tyranitar"], [], ["rockslide"])
        mask = np.zeros(11, dtype=np.int8)
        mask[6] = 1
        with pytest.raises(ValueError, match="Illegal action"):
            Gen3ActionMapper.action_to_order(7, battle, mask=mask)

    def test_struggle_round_trips(self):
        battle, _, _ = _make_battle(["tyranitar"], [], [], struggle=True)
        order = Gen3ActionMapper.action_to_order(10, battle)
        assert order.order.id == "struggle"


# ===========================================================================
# Send-path conformance — offered (mask) → picked (action) → sent (order) must
# stay consistent end to end; the serialized order is round-tripped back to the
# action and a divergence throws.
# ===========================================================================

class TestSendConformance:

    def test_move_switch_struggle_round_trip_clean(self):
        # Each of these must NOT raise the conformance error (order back-maps to action).
        battle, mons, avail = _make_battle(["tyranitar", "skarmory"], [1], ["rockslide", "earthquake"])
        assert Gen3ActionMapper.action_to_order(6, battle).order == avail[0]   # move slot 0
        assert Gen3ActionMapper.action_to_order(7, battle).order == avail[1]   # move slot 1
        assert Gen3ActionMapper.action_to_order(1, battle).order == mons[1]    # switch slot 1
        b2, _, _ = _make_battle(["tyranitar"], [], [], struggle=True)
        assert Gen3ActionMapper.action_to_order(10, b2).order.id == "struggle"

    def test_hiddenpower_round_trip_does_not_false_positive(self):
        # legal.move_ids carries bare "hiddenpower"; the order resolves to the typed
        # "hiddenpowerice" — the round-trip must treat them as the same move (no throw).
        battle, _, avail = _make_battle(["jolteon"], [], ["hiddenpower"])
        avail[0].id = "hiddenpowerice"
        order = Gen3ActionMapper.action_to_order(6, battle)
        assert order.order == avail[0]

    def test_serialization_drift_throws(self):
        # Force the serializer to return the WRONG move (slot 1) for action 6 (slot 0):
        # the order back-maps to action 7 != 6 → conformance violation, fail-loud.
        battle, _, avail = _make_battle(["tyranitar"], [], ["rockslide", "earthquake"])
        wrong = MagicMock(spec=SingleBattleOrder)
        wrong.order = _move("earthquake")   # payload is slot-1's move, but we pressed slot 0
        with patch("agents.action.mapper.choice_to_order", return_value=wrong):
            with pytest.raises(RuntimeError, match="conformance violation"):
                Gen3ActionMapper.action_to_order(6, battle)


# ===========================================================================
# assert_decision_current — staleness + mid-decision guard
# ===========================================================================

def _ctx(turn, legal):
    return SimpleNamespace(turn=turn, legal=legal, mask=np.ones(11, dtype=np.int8))


class TestAssertDecisionCurrent:

    def test_missing_context_raises(self):
        battle, _, _ = _make_battle(["tyranitar"], [], ["rockslide"])
        with pytest.raises(RuntimeError, match="missing"):
            Gen3ActionMapper.assert_decision_current(None, battle)

    def test_stale_turn_raises(self):
        battle, _, _ = _make_battle(["tyranitar"], [], ["rockslide"], turn=5)
        ctx = _ctx(4, LegalActions.from_battle(battle))
        with pytest.raises(RuntimeError, match="turn 4"):
            Gen3ActionMapper.assert_decision_current(ctx, battle)

    def test_mid_decision_change_raises(self):
        battle, _, _ = _make_battle(["tyranitar"], [], ["rockslide"], turn=3)
        snap = LegalActions.from_battle(battle)
        ctx = _ctx(3, snap)
        # Server's move list shifts under us after the snapshot was captured.
        battle.last_request["active"][0]["moves"] = [{"id": "earthquake", "disabled": False}]
        with pytest.raises(RuntimeError, match="Mid-decision state change"):
            Gen3ActionMapper.assert_decision_current(ctx, battle)

    def test_consistent_decision_passes(self):
        battle, _, _ = _make_battle(["tyranitar"], [], ["rockslide"], turn=3)
        ctx = _ctx(3, LegalActions.from_battle(battle))
        Gen3ActionMapper.assert_decision_current(ctx, battle)  # no raise


# ===========================================================================
# order_to_action — reverse map (diagnostics / opponent labelling)
# ===========================================================================

class TestOrderToAction:

    def _switch_order(self, mon):
        order = MagicMock(spec=SingleBattleOrder)
        order.order = mon            # SingleBattleOrder payload: a Pokemon ⇒ switch
        return order

    def _move_order(self, move):
        order = MagicMock(spec=SingleBattleOrder)
        order.order = move           # SingleBattleOrder payload: a Move ⇒ move
        return order

    def test_switch_maps_to_correct_slot(self):
        battle, mons, _ = _make_battle(["tyranitar", "skarmory", "gengar"], [1, 2], ["rockslide"])
        assert order_to_action(self._switch_order(mons[2]), battle) == 2
        assert Gen3ActionMapper.order_to_action(self._switch_order(mons[2]), battle) == 2

    def test_unknown_switch_returns_zero(self):
        battle, _, _ = _make_battle(["tyranitar"], [], ["rockslide"])
        assert order_to_action(self._switch_order(_mon("missingno")), battle) == 0

    def test_move_maps_to_correct_slot(self):
        battle, _, _ = _make_battle(["tyranitar"], [], ["rockslide", "earthquake", "fireblast"])
        assert order_to_action(self._move_order(_move("earthquake")), battle) == 7

    def test_struggle_maps_to_10(self):
        battle, _, _ = _make_battle(["tyranitar"], [], [], struggle=True)
        assert order_to_action(self._move_order(_move("struggle")), battle) == 10

    def test_non_singles_order_returns_zero(self):
        battle, _, _ = _make_battle(["tyranitar"], [], ["rockslide"])
        assert order_to_action(MagicMock(), battle) == 0

    def test_hiddenpower_reverse_maps_correctly(self):
        battle, _, _ = _make_battle(["jolteon"], [], ["hiddenpowerice"])
        assert order_to_action(self._move_order(_move("hiddenpowerice")), battle) == 6

    def test_unknown_move_returns_zero(self):
        battle, _, _ = _make_battle(["tyranitar"], [], ["rockslide"])
        assert order_to_action(self._move_order(_move("surf")), battle) == 0
