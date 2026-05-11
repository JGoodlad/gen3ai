import numpy as np
from unittest.mock import MagicMock
from src.agents.observation.global_env import GlobalEnvEncoder
from src.agents.observation.reactive import ReactiveEncoder
from poke_env.battle.side_condition import SideCondition

def test_spikes_extraction():
    # Mock battle object
    battle = MagicMock()
    battle.side_conditions = {SideCondition.SPIKES: 2}
    battle.opponent_side_conditions = {SideCondition.SPIKES: 1}
    battle.weather = {}
    battle.turn = 10
    battle.available_moves = []
    battle.team = {}
    battle.opponent_team = {}
    battle.active_pokemon = None
    battle.opponent_active_pokemon = None

    # Test GlobalEnvEncoder
    global_encoder = GlobalEnvEncoder()
    global_vec = global_encoder.encode(battle)
    
    # Indices 6 and 7 are our_spikes and opp_spikes
    # MAX_SPIKES = 3
    our_spikes_norm = global_vec[6]
    opp_spikes_norm = global_vec[7]
    
    print(f"GlobalEnv - Our Spikes (norm): {our_spikes_norm:.2f} (expected {2/3:.2f})")
    print(f"GlobalEnv - Opp Spikes (norm): {opp_spikes_norm:.2f} (expected {1/3:.2f})")
    
    assert np.isclose(our_spikes_norm, 2/3.0)
    assert np.isclose(opp_spikes_norm, 1/3.0)
    
    # Test ReactiveEncoder
    reactive_encoder = ReactiveEncoder()
    reactive_vec = reactive_encoder.encode(battle)
    
    # Indices 12 and 13 are our_spikes and opp_spikes
    our_spikes_reactive = reactive_vec[12]
    opp_spikes_reactive = reactive_vec[13]
    
    print(f"Reactive - Our Spikes (norm): {our_spikes_reactive:.2f} (expected {2/3:.2f})")
    print(f"Reactive - Opp Spikes (norm): {opp_spikes_reactive:.2f} (expected {1/3:.2f})")
    
    assert np.isclose(our_spikes_reactive, 2/3.0)
    assert np.isclose(opp_spikes_reactive, 1/3.0)
    
    print("Verification successful!")

if __name__ == "__main__":
    try:
        test_spikes_extraction()
    except Exception as e:
        print(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()
