"""
Unit tests for Gen3ActionMasker and Gen3ActionMapper.

These cover the full 11-action space, all error paths (crash-over-corruption),
the decision-context latch, and the reverse order_to_action mapping.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock

from agents.action.mask_generator import Gen3ActionMasker
from agents.action.mapper import Gen3ActionMapper
from poke_env.player.battle_order import SingleBattleOrder


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _mon(species: str):
    """Minimal Pokemon mock."""
    m = MagicMock()
    m.species = species
    return m


def _move(move_id: str, disabled: bool = False):
    """Minimal Move mock (poke-env object)."""
    m = MagicMock()
    m.id = move_id
    return m


def _request_move(move_id: str, disabled: bool = False) -> dict:
    """Minimal entry in last_request["active"][0]["moves"]."""
    return {"id": move_id, "move": move_id, "disabled": disabled}


def _make_battle(
    team_species: list[str],
    available_switch_indices: list[int],
    move_ids: list[str],
    disabled_indices: list[int] = (),
    struggle: bool = False,
    turn: int = 1,
    context_turn: int | None = None,  # None → use turn
    tamper_server_moves: list[str] | None = None,
):
    """
    Build a minimal mock battle suitable for both masker and mapper.

    team_species:            ordered list of species in battle.team
    available_switch_indices: which slots are in available_switches
    move_ids:                ordered move IDs in the request (up to 4)
    disabled_indices:        which move slots are disabled in the request
    struggle:                include struggle in available_moves
    context_turn:            turn to store in _gen3_decision_context
                             (defaults to `turn`; set different to test stale-context)
    tamper_server_moves:     if set, last_request reflects these IDs instead of move_ids
                             (simulates mid-decision server change)
    """
    battle = MagicMock()
    battle.turn = turn

    # Build team
    mons = [_mon(s) for s in team_species]
    battle.team = {m.species: m for m in mons}
    battle.available_switches = [mons[i] for i in available_switch_indices]

    # Build moves
    available_moves = [_move(mid) for mid in move_ids]
    if struggle:
        available_moves.append(_move("struggle"))
    battle.available_moves = available_moves

    # Build last_request
    server_move_ids = tamper_server_moves if tamper_server_moves is not None else move_ids
    request_moves = [
        _request_move(mid, disabled=(i in disabled_indices))
        for i, mid in enumerate(server_move_ids)
    ]
    battle.last_request = {
        "active": [{"moves": request_moves}],
        "side": {"pokemon": [{"ident": f"p1: {s}"} for s in team_species]},
        "turn": turn,
    }

    # Latch decision context (mirrors what Gen3ActionMasker.get_mask does)
    battle._gen3_decision_context = {
        "turn": context_turn if context_turn is not None else turn,
        "move_ids": list(move_ids),
        "team_species": list(team_species),
        "team_objects": mons,
    }

    return battle, mons, available_moves


# ===========================================================================
# Gen3ActionMasker tests
# ===========================================================================

class TestGen3ActionMasker:

    def test_no_last_request_raises(self):
        battle = MagicMock()
        battle.last_request = None
        with pytest.raises(RuntimeError, match="No last_request"):
            Gen3ActionMasker.get_mask(battle)

    def test_duplicate_species_raises(self):
        # poke-env keys team by "p1: speciesname" so two different mons can share
        # the same .species string (e.g. two Gengar via a team-building bug).
        # We force this by giving the dict different keys but identical species values.
        mon_a = _mon("gengar")
        mon_b = _mon("gengar")
        mon_c = _mon("skarmory")
        battle = MagicMock()
        battle.turn = 1
        battle.team = {"p1: gengar1": mon_a, "p1: gengar2": mon_b, "p1: skarmory": mon_c}
        battle.available_switches = [mon_b]
        battle.available_moves = [_move("surf")]
        battle.last_request = {
            "active": [{"moves": [{"id": "surf", "disabled": False}]}],
            "side": {"pokemon": []},
            "turn": 1,
        }
        with pytest.raises(RuntimeError, match="Duplicate species"):
            Gen3ActionMasker.get_mask(battle)

    def test_available_switches_set(self):
        battle, mons, _ = _make_battle(
            team_species=["tyranitar", "skarmory", "gengar", "blissey", "swampert", "starmie"],
            available_switch_indices=[1, 3, 5],
            move_ids=["rockslide"],
        )
        mask = Gen3ActionMasker.get_mask(battle)
        assert mask[0] == 0  # tyranitar (active, not available)
        assert mask[1] == 1  # skarmory
        assert mask[2] == 0  # gengar
        assert mask[3] == 1  # blissey
        assert mask[4] == 0  # swampert
        assert mask[5] == 1  # starmie

    def test_unavailable_switches_not_set(self):
        battle, _, _ = _make_battle(
            team_species=["tyranitar", "skarmory"],
            available_switch_indices=[],   # no switches available
            move_ids=["rockslide", "earthquake"],
        )
        mask = Gen3ActionMasker.get_mask(battle)
        assert mask[0] == 0
        assert mask[1] == 0

    def test_non_disabled_moves_set(self):
        battle, _, _ = _make_battle(
            team_species=["tyranitar"],
            available_switch_indices=[],
            move_ids=["rockslide", "earthquake", "fireblast", "icebeam"],
        )
        mask = Gen3ActionMasker.get_mask(battle)
        assert mask[6] == 1
        assert mask[7] == 1
        assert mask[8] == 1
        assert mask[9] == 1

    def test_disabled_moves_not_set(self):
        battle, _, _ = _make_battle(
            team_species=["tyranitar"],
            available_switch_indices=[],
            move_ids=["rockslide", "earthquake", "fireblast", "icebeam"],
            disabled_indices=[1, 3],
        )
        mask = Gen3ActionMasker.get_mask(battle)
        assert mask[6] == 1   # rockslide – enabled
        assert mask[7] == 0   # earthquake – disabled
        assert mask[8] == 1   # fireblast – enabled
        assert mask[9] == 0   # icebeam – disabled

    def test_struggle_sets_bit_10(self):
        battle, _, _ = _make_battle(
            team_species=["tyranitar"],
            available_switch_indices=[],
            move_ids=[],
            struggle=True,
        )
        # Struggle is signalled via available_moves, not the request moves list
        battle.available_moves = [_move("struggle")]
        mask = Gen3ActionMasker.get_mask(battle)
        assert mask[10] == 1

    def test_no_struggle_bit_when_absent(self):
        battle, _, _ = _make_battle(
            team_species=["tyranitar"],
            available_switch_indices=[],
            move_ids=["rockslide"],
        )
        mask = Gen3ActionMasker.get_mask(battle)
        assert mask[10] == 0

    def test_decision_context_latched(self):
        battle, mons, moves = _make_battle(
            team_species=["tyranitar", "skarmory"],
            available_switch_indices=[1],
            move_ids=["rockslide", "earthquake"],
        )
        # Clear existing context so we verify get_mask() writes it
        del battle._gen3_decision_context
        Gen3ActionMasker.get_mask(battle)
        ctx = battle._gen3_decision_context
        assert ctx["turn"] == 1
        assert ctx["move_ids"] == ["rockslide", "earthquake"]
        assert ctx["team_species"] == ["tyranitar", "skarmory"]
        assert ctx["team_objects"] == mons

    def test_partial_team_under_six(self):
        battle, _, _ = _make_battle(
            team_species=["tyranitar", "skarmory"],
            available_switch_indices=[1],
            move_ids=["rockslide"],
        )
        mask = Gen3ActionMasker.get_mask(battle)
        # Slots 2-5 don't exist — must stay 0
        assert mask[2] == 0
        assert mask[5] == 0


# ===========================================================================
# Gen3ActionMapper.action_to_order tests
# ===========================================================================

class TestActionToOrder:

    # --- Switch actions ---

    def test_switch_returns_correct_mon(self):
        battle, mons, _ = _make_battle(
            team_species=["tyranitar", "skarmory", "gengar"],
            available_switch_indices=[1, 2],
            move_ids=["rockslide"],
        )
        order = Gen3ActionMapper.action_to_order(1, battle)
        assert isinstance(order, SingleBattleOrder)
        assert order.order == mons[1]

    def test_switch_slot_not_in_available_switches_raises(self):
        battle, _, _ = _make_battle(
            team_species=["tyranitar", "skarmory"],
            available_switch_indices=[],   # nothing available
            move_ids=["rockslide"],
        )
        with pytest.raises(ValueError, match="invalid"):
            Gen3ActionMapper.action_to_order(1, battle)

    def test_switch_action_out_of_team_range_raises(self):
        battle, _, _ = _make_battle(
            team_species=["tyranitar"],    # only one mon
            available_switch_indices=[],
            move_ids=["rockslide"],
        )
        with pytest.raises(ValueError):
            Gen3ActionMapper.action_to_order(3, battle)

    def test_switch_duplicate_species_raises(self):
        battle, mons, _ = _make_battle(
            team_species=["tyranitar", "skarmory"],
            available_switch_indices=[1],
            move_ids=["rockslide"],
        )
        battle._gen3_decision_context["team_species"] = ["tyranitar", "tyranitar"]
        with pytest.raises(RuntimeError, match="Duplicate species"):
            Gen3ActionMapper.action_to_order(1, battle)

    # --- Move actions ---

    def test_move_returns_correct_order(self):
        battle, _, avail_moves = _make_battle(
            team_species=["tyranitar"],
            available_switch_indices=[],
            move_ids=["rockslide", "earthquake"],
        )
        order = Gen3ActionMapper.action_to_order(6, battle)   # slot 0 = rockslide
        assert isinstance(order, SingleBattleOrder)
        assert order.order == avail_moves[0]

        order2 = Gen3ActionMapper.action_to_order(7, battle)  # slot 1 = earthquake
        assert order2.order == avail_moves[1]

    def test_move_no_context_raises(self):
        battle, _, _ = _make_battle(
            team_species=["tyranitar"],
            available_switch_indices=[],
            move_ids=["rockslide"],
        )
        del battle._gen3_decision_context
        with pytest.raises(RuntimeError, match="No decision context"):
            Gen3ActionMapper.action_to_order(6, battle)

    def test_move_not_in_available_moves_raises(self):
        # Context and server both say "surf" (forensic check passes), but
        # available_moves only has "rockslide" (poke-env desync).
        battle, _, _ = _make_battle(
            team_species=["tyranitar"],
            available_switch_indices=[],
            move_ids=["surf"],
        )
        battle.available_moves = [_move("rockslide")]  # poke-env has different move
        with pytest.raises(ValueError, match="not found in available_moves"):
            Gen3ActionMapper.action_to_order(6, battle)

    def test_hiddenpower_variant_matches(self):
        # The server request stores "hiddenpower" (no type suffix) in both the
        # latched context and last_request, but poke-env resolves available_moves
        # to the fully-typed id "hiddenpowerice". The mapper must match them.
        battle, _, avail_moves = _make_battle(
            team_species=["jolteon"],
            available_switch_indices=[],
            move_ids=["hiddenpower"],   # server request uses bare id
        )
        avail_moves[0].id = "hiddenpowerice"  # poke-env resolves full type
        order = Gen3ActionMapper.action_to_order(6, battle)
        assert order.order == avail_moves[0]

    # --- Struggle ---

    def test_struggle_from_available_moves(self):
        battle, _, avail_moves = _make_battle(
            team_species=["tyranitar"],
            available_switch_indices=[],
            move_ids=[],
            struggle=True,
        )
        # Struggle move is appended by _make_battle when struggle=True
        battle._gen3_decision_context["move_ids"] = []
        order = Gen3ActionMapper.action_to_order(10, battle)
        assert order.order.id == "struggle"

    def test_struggle_fallback_creates_move_object(self):
        battle, _, _ = _make_battle(
            team_species=["tyranitar"],
            available_switch_indices=[],
            move_ids=[],
        )
        battle.available_moves = []  # nothing available — fallback path
        order = Gen3ActionMapper.action_to_order(10, battle)
        assert order.order.id == "struggle"

    # --- Mask / stale-turn validation ---

    def test_illegal_action_with_mask_raises(self):
        battle, _, _ = _make_battle(
            team_species=["tyranitar"],
            available_switch_indices=[],
            move_ids=["rockslide"],
        )
        mask = np.zeros(11, dtype=np.int8)
        mask[6] = 1
        with pytest.raises(ValueError, match="Illegal action"):
            Gen3ActionMapper.action_to_order(7, battle, mask=mask)

    def test_stale_turn_raises(self):
        battle, _, _ = _make_battle(
            team_species=["tyranitar"],
            available_switch_indices=[],
            move_ids=["rockslide"],
            turn=5,
        )
        mask = np.ones(11, dtype=np.int8)
        with pytest.raises(ValueError, match="Stale mask"):
            Gen3ActionMapper.action_to_order(6, battle, mask=mask, latched_turn=3)

    # --- Forensic integrity check ---

    def test_forensic_check_raises_on_server_change(self):
        battle, _, _ = _make_battle(
            team_species=["tyranitar"],
            available_switch_indices=[],
            move_ids=["rockslide"],
            tamper_server_moves=["earthquake"],  # server now shows different move
        )
        with pytest.raises(RuntimeError, match="Mid-decision state change"):
            Gen3ActionMapper.action_to_order(6, battle)

    def test_forensic_check_passes_when_consistent(self):
        battle, _, _ = _make_battle(
            team_species=["tyranitar"],
            available_switch_indices=[],
            move_ids=["rockslide"],
        )
        # Should not raise
        Gen3ActionMapper.action_to_order(6, battle)

    # --- Stale context ---

    def test_stale_context_discarded_move_raises(self):
        battle, _, _ = _make_battle(
            team_species=["tyranitar"],
            available_switch_indices=[],
            move_ids=["rockslide"],
            turn=5,
            context_turn=4,   # context from previous turn
        )
        # With stale context, move action has no context → should raise
        with pytest.raises(RuntimeError, match="No decision context"):
            Gen3ActionMapper.action_to_order(6, battle)


# ===========================================================================
# Gen3ActionMapper.order_to_action tests
# ===========================================================================

class TestOrderToAction:

    def _switch_order(self, mon):
        order = MagicMock(spec=SingleBattleOrder)
        order.switch = mon
        order.move = None
        order.order = mon
        return order

    def _move_order(self, move):
        order = MagicMock(spec=SingleBattleOrder)
        order.move = move
        order.switch = None
        order.order = move
        return order

    def test_switch_maps_to_correct_slot(self):
        battle, mons, _ = _make_battle(
            team_species=["tyranitar", "skarmory", "gengar"],
            available_switch_indices=[1, 2],
            move_ids=["rockslide"],
        )
        order = self._switch_order(mons[2])
        assert Gen3ActionMapper.order_to_action(order, battle) == 2

    def test_unknown_switch_returns_zero(self):
        battle, _, _ = _make_battle(
            team_species=["tyranitar"],
            available_switch_indices=[],
            move_ids=["rockslide"],
        )
        unknown_mon = _mon("missingno")
        order = self._switch_order(unknown_mon)
        assert Gen3ActionMapper.order_to_action(order, battle) == 0

    def test_move_maps_to_correct_slot(self):
        battle, _, _ = _make_battle(
            team_species=["tyranitar"],
            available_switch_indices=[],
            move_ids=["rockslide", "earthquake", "fireblast"],
        )
        eq_move = _move("earthquake")
        order = self._move_order(eq_move)
        assert Gen3ActionMapper.order_to_action(order, battle) == 7  # slot 1 → index 7

    def test_struggle_maps_to_10(self):
        battle, _, _ = _make_battle(
            team_species=["tyranitar"],
            available_switch_indices=[],
            move_ids=[],
        )
        order = self._move_order(_move("struggle"))
        assert Gen3ActionMapper.order_to_action(order, battle) == 10

    def test_non_singles_order_returns_zero(self):
        battle, _, _ = _make_battle(
            team_species=["tyranitar"],
            available_switch_indices=[],
            move_ids=["rockslide"],
        )
        bad_order = MagicMock()  # not a SingleBattleOrder
        assert Gen3ActionMapper.order_to_action(bad_order, battle) == 0

    def test_hiddenpower_reverse_maps_correctly(self):
        battle, _, _ = _make_battle(
            team_species=["jolteon"],
            available_switch_indices=[],
            move_ids=["hiddenpowerice"],
        )
        # Server context may store bare "hiddenpower"
        battle._gen3_decision_context["move_ids"] = ["hiddenpower"]
        order = self._move_order(_move("hiddenpowerice"))
        assert Gen3ActionMapper.order_to_action(order, battle) == 6

    def test_unknown_move_returns_zero(self):
        battle, _, _ = _make_battle(
            team_species=["tyranitar"],
            available_switch_indices=[],
            move_ids=["rockslide"],
        )
        order = self._move_order(_move("surf"))   # not in context
        assert Gen3ActionMapper.order_to_action(order, battle) == 0
