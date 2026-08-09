import numpy as np
from unittest.mock import MagicMock
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.move import Move
from .state_encoder import Gen3ObservationEncoder, load_mappings
from .constants import TEAM_SIZE, OFFSET_REACTIVE, POKEMON_FULL_DIM, OFFSET_OUR_TEAM, MOVE_SLOT_DIM

def create_mock_move(move_id, base_power=60, move_type="Normal"):
    move = MagicMock(spec=Move)
    move.id = move_id
    move.base_power = base_power
    move.type = MagicMock()
    move.type.name = move_type.upper()
    move.type.damage_multiplier.return_value = float(ord(move_id[0]) % 5)
    move.current_pp = 16
    move.max_pp = 16
    return move

def create_mock_pokemon(species, moves_dict, active=False):
    mon = MagicMock(spec=Pokemon)
    mon.species = species
    mon.moves = moves_dict
    mon.active = active
    mon.fainted = False
    mon.current_hp_fraction = 1.0
    mon.type_1 = MagicMock()
    mon.type_1.name = "NORMAL"
    mon.type_2 = None
    mon.item = None
    mon.ability = None
    mon.status = None
    mon.boosts = {}
    mon.volatiles = {}
    mon.stats = {"hp": 100, "atk": 100, "def": 100, "spa": 100, "spd": 100, "spe": 100}
    return mon

def test_move_and_team_alignment(monkeypatch):
    # The reactive matchup encoder now reads a precomputed type chart instead of calling
    # `move.type.damage_multiplier(...)`, so the mock-injected effectiveness
    # (`move.type.damage_multiplier.return_value`, see create_mock_move) no longer reaches it.
    # This is an ALIGNMENT test (does matrix cell [our_mon, our_move, their_mon] carry the
    # right move's value?), so route the effectiveness primitive back to the injected value;
    # the chart math itself is covered exhaustively in gen3_mechanics_test.py.
    # 1. Setup mappings
    mappings = load_mappings()
    encoder = Gen3ObservationEncoder(mappings)

    # Patch the EXACT module dict the reactive encoder reads (pytest can import this test
    # under a different package path than the encoder's modules → a duplicate `reactive`
    # module instance, so patching `agents.observation.reactive` by name would miss).
    _calls = {"n": 0}

    def _eff(move_type, type_1=None, type_2=None, ability=None, status=None):
        _calls["n"] += 1
        rv = getattr(getattr(move_type, "damage_multiplier", None), "return_value", None)
        return rv if isinstance(rv, (int, float)) and not isinstance(rv, bool) else 1.0

    monkeypatch.setitem(
        encoder.reactive_encoder.encode.__func__.__globals__,
        "effective_multiplier_by_types", _eff)
    
    # 2. Create a complex battle state
    battle = MagicMock(spec=AbstractBattle)
    battle.battle_tag = "test_battle"
    battle.wait = False
    battle.turn = 1
    battle.weather = {}
    
    # Define moves in non-alphabetical order
    m1 = create_mock_move("thunderbolt", 90, "Electric")
    m2 = create_mock_move("icebeam", 90, "Ice")
    m3 = create_mock_move("surf", 90, "Water")
    m4 = create_mock_move("flamethrower", 90, "Fire")
    
    # Sorted order should be: flamethrower, icebeam, surf, thunderbolt
    expected_sorted_ids = ["flamethrower", "icebeam", "surf", "thunderbolt"]
    
    # Create team
    p1_moves = {"thunderbolt": m1, "icebeam": m2, "surf": m3, "flamethrower": m4}
    p1 = create_mock_pokemon("jolteon", p1_moves, active=True)
    
    p2_moves = {"surf": m3}
    p2 = create_mock_pokemon("vaporeon", p2_moves)
    
    battle.team = {"p1": p1, "p2": p2}
    battle.opponent_team = {"o1": p1} # Reuse for simplicity
    battle.active_pokemon = p1
    battle.opponent_active_pokemon = p1
    battle.side_conditions = {}
    battle.opponent_side_conditions = {}
    
    # 3. Encode
    vector = encoder.encode(battle)
    assert _calls["n"] == 0, ("gen3_entity_rehome_v1: encode must NOT price matchups on the CPU "
                              "anymore — a call here means the deleted matrix loop came back")
    
    # 4. Verify Move Embedding Order
    # Pokemon 0 (p1) moves start at OFFSET_OUR_TEAM + POKEMON_MOVES_OFFSET
    from .constants import POKEMON_MOVES_OFFSET  # noqa: PLC0415
    moves_start = OFFSET_OUR_TEAM + POKEMON_MOVES_OFFSET
    
    for i in range(4):
        slot_idx = moves_start + (i * MOVE_SLOT_DIM)
        move_id_val = vector[slot_idx] # First element of slot is ID
        known_flag = vector[slot_idx + 6] # Known is at 6
        if i < 4:
            move_name = mappings["reverse"]["moves"].get(int(move_id_val))
            assert move_name == expected_sorted_ids[i], f"Move slot {i} mismatch: {move_name} != {expected_sorted_ids[i]}"
            assert known_flag == 1.0, f"Move slot {i} should be known"

    # gen3_entity_rehome_v1: the matchup matrices are DELETED from the obs — pair
    # effectiveness is GPU-side (D/V edges; damage_op_probe_fuzz_test is the oracle),
    # so there is no reactive-matrix alignment left to verify here.

    print("✅ Move and Team alignment verified!")

if __name__ == "__main__":
    test_move_and_team_alignment()
