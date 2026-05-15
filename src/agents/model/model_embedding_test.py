import torch
import numpy as np
from agents.model.features_extractor import Gen3FeaturesExtractor
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.observation.constants import (
    POKEMON_FULL_DIM, POKEMON_MOVES_OFFSET, 
    POKEMON_SPECIES_OFFSET, POKEMON_ITEMS_OFFSET,
    POKEMON_ABILITIES_OFFSET, POKEMON_TYPES_OFFSET,
    OFFSET_REACTIVE
)
import gymnasium as gym

def test_model_full_embedding_forensics():
    """
    Forensic test that hooks into the model to verify every input path
    is correctly unpacked and aligned.
    """
    # 1. Setup Model
    mappings = load_mappings()
    encoder_obj = Gen3ObservationEncoder(mappings)
    layout = encoder_obj.get_layout()
    obs_space = gym.spaces.Box(low=0, high=1, shape=(1565,), dtype=np.float32)
    model = Gen3FeaturesExtractor(obs_space, layout=layout, mappings=mappings)
    model.eval()

    # 2. Prepare Spies (Hooks)
    captured_inputs = {}
    def hook_fn(name):
        def hook(module, input, output):
            captured_inputs[name] = input[0]
        return hook

    # Hook into Move Network and Role Encoder
    model.move_network[0].register_forward_hook(hook_fn("move_input"))
    model.role_encoder[0].register_forward_hook(hook_fn("role_input"))
    
    # 3. Create a "Golden Observation" with known values
    obs = np.zeros((1565,), dtype=np.float32)
    
    # --- Pokemon 0 (Our Active) ---
    pk0_start = 0
    # Species: Jolteon (ID 135)
    obs[pk0_start + POKEMON_SPECIES_OFFSET] = 135.0
    # Item: Leftovers (ID 100), Known=1
    obs[pk0_start + POKEMON_ITEMS_OFFSET] = 100.0
    obs[pk0_start + POKEMON_ITEMS_OFFSET + 16] = 1.0 
    # Ability: Volt Absorb (ID 10), Known=1
    obs[pk0_start + POKEMON_ABILITIES_OFFSET] = 10.0
    obs[pk0_start + POKEMON_ABILITIES_OFFSET + 8] = 1.0
    # Move 0: Surf (ID 57), Type: Water (ID 11), Known=1
    obs[pk0_start + POKEMON_MOVES_OFFSET + 0] = 57.0 # ID
    obs[pk0_start + POKEMON_MOVES_OFFSET + 6] = 1.0  # Known flag
    
    # --- Reactive Matrix ---
    obs[OFFSET_REACTIVE + 16] = 0.5 
    
    obs_tensor = torch.from_numpy(obs).unsqueeze(0)
    obs_dict = {"observation": obs_tensor}
    
    # 4. Execute and Inspect
    with torch.no_grad():
        model(obs_dict)
        
    # --- VERIFICATIONS ---
    move_in = captured_inputs["move_input"] 
    role_in = captured_inputs["role_input"] # Flattened shape [12, 237]
    
    # Check Move 0
    m0_features = move_in[0]
    assert m0_features[38] == 1.0, f"Move Known Flag mismatch: {m0_features[38]}"
    assert m0_features[51] == 0.5, f"Matchup Matrix mismatch: {m0_features[51]}"
    
    # Check Mon 0
    p0_features = role_in[0]
    # Index 54: Item Known Flag (32 + 6 + 16)
    assert p0_features[54] == 1.0, f"Item Known Flag mismatch: {p0_features[54]}"
    # Index 87: Ability Known Flag (32 + 6 + 16 + 1 + 16 + 16)
    assert p0_features[87] == 1.0, f"Ability Known Flag mismatch: {p0_features[87]}"

    print("✅ Full Model Forensic Audit PASSED!")
    print(f"  - Move Path: Known (idx 38), Matchup (idx 51)")
    print(f"  - Pokemon Path: Item Known (idx 54), Ability Known (idx 87)")

if __name__ == "__main__":
    test_model_full_embedding_forensics()
