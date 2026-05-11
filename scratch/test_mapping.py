
from poke_env.environment.singles_env import SinglesEnv
from unittest.mock import MagicMock

def test_mapping():
    battle = MagicMock()
    battle.player_username = "test_player"
    battle.battle_tag = "test_battle"
    
    # Mock 4 moves
    move1 = MagicMock(); move1.id = "move1"
    move2 = MagicMock(); move2.id = "move2"
    move3 = MagicMock(); move3.id = "move3"
    move4 = MagicMock(); move4.id = "move4"
    # Poke-env SinglesEnv expects these to be present
    battle.available_moves = [move1, move2, move3, move4]
    
    # Mock 5 switches (A is active, B-F are available)
    pokemon = []
    for name in ["A", "B", "C", "D", "E", "F"]:
        p = MagicMock()
        p.species = name
        p.active = (name == "A")
        pokemon.append(p)
        
    battle.available_switches = pokemon[1:] # B, C, D, E, F
    
    print("Testing SinglesEnv.action_to_order mapping:")
    for i in range(10):
        try:
            order = SinglesEnv.action_to_order(i, battle)
            if hasattr(order, 'species'):
                print(f"Action {i} -> Switch to {order.species}")
            elif hasattr(order, 'id'):
                print(f"Action {i} -> Move {order.id}")
            else:
                print(f"Action {i} -> {order}")
        except Exception as e:
            print(f"Action {i} -> Error: {e}")

if __name__ == "__main__":
    test_mapping()
