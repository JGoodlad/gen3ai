import numpy as np
from unittest.mock import MagicMock
from src.agents.observation.active_context import ActiveContextEncoder
from poke_env.battle.effect import Effect

def test_volatiles_extraction():
    # Mock pokemon object
    mon = MagicMock()
    mon.boosts = {}
    mon.volatiles = {
        Effect.CONFUSION: 1, 
        Effect.TAUNT: 1,
        Effect.PERISH2: 1 # Perish Song with 2 turns left
    }
    
    # Mock battle object
    battle = MagicMock()
    
    encoder = ActiveContextEncoder()
    vec = encoder.encode(mon, battle)
    
    # Volatiles start at index 14
    # CONFUSION is index 0 in volatile_map -> 14 in vec
    # TAUNT is index 2 in volatile_map -> 16 in vec
    # PERISH is index 4 in vec
    
    print(f"Vector[14] (Confusion): {vec[14]} (expected 1.0)")
    print(f"Vector[15] (Substitute): {vec[15]} (expected 0.0)")
    print(f"Vector[16] (Taunt): {vec[16]} (expected 1.0)")
    print(f"Vector[18] (Perish Song): {vec[18]} (expected 1.0)")
    
    assert vec[14] == 1.0
    assert vec[15] == 0.0
    assert vec[16] == 1.0
    assert vec[18] == 1.0
    
    # Test describe_vector
    desc = encoder.describe_vector(vec)
    print(f"Active Volatiles: {desc['volatiles']}")
    assert "CONF" in desc["volatiles"]
    assert "TAUNT" in desc["volatiles"]
    assert "PERISH" in desc["volatiles"]
    assert "SUB" not in desc["volatiles"]
    
    print("Volatiles verification successful!")

if __name__ == "__main__":
    try:
        import sys
        import os
        sys.path.append(os.getcwd())
        test_volatiles_extraction()
    except Exception as e:
        print(f"Verification failed: {e}")
        import traceback
        traceback.print_exc()
