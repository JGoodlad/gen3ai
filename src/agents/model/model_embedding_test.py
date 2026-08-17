import pytest
import torch
import numpy as np
from agents.model.features_extractor import Gen3FeaturesExtractor
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.observation.constants import (
    POKEMON_MOVES_OFFSET,
    POKEMON_SPECIES_OFFSET, POKEMON_ITEMS_OFFSET, ITEM_ID_DIM,
    POKEMON_ABILITIES_OFFSET, ABILITY_SLOT_DIM, ABILITY_DOMINANCE_DIM
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
    total_dim = layout['total_dim']
    obs_space = gym.spaces.Box(low=0, high=1, shape=(total_dim,), dtype=np.float32)
    model = Gen3FeaturesExtractor(obs_space, layout=layout, mappings=mappings)
    model.eval()

    # 2. Prepare Spies (Hooks)
    captured_inputs = {}
    def hook_fn(name):
        def hook(module, input, output):
            captured_inputs[name] = input[0]
        return hook

    # Hook into Move Network and Role Encoder (now owned by the PokemonEncoder phase)
    model.pokemon_encoder.move_network[0].register_forward_hook(hook_fn("move_input"))
    model.pokemon_encoder.role_encoder[0].register_forward_hook(hook_fn("role_input"))

    # 3. Create a "Golden Observation" with known values
    obs = np.zeros((total_dim,), dtype=np.float32)

    # --- Pokemon 0 (Our Active) ---
    pk0_start = 0
    # Species: Jolteon (ID 135)
    obs[pk0_start + POKEMON_SPECIES_OFFSET] = 135.0
    # Item: Leftovers (ID 100), Known=1
    obs[pk0_start + POKEMON_ITEMS_OFFSET] = 100.0
    obs[pk0_start + POKEMON_ITEMS_OFFSET + ITEM_ID_DIM] = 1.0
    # Ability: Volt Absorb (ID 10), Known=1, dominance=1.0 (revealed → no alternative)
    obs[pk0_start + POKEMON_ABILITIES_OFFSET] = 10.0
    obs[pk0_start + POKEMON_ABILITIES_OFFSET + ABILITY_SLOT_DIM] = 1.0   # dominance
    obs[pk0_start + POKEMON_ABILITIES_OFFSET + ABILITY_SLOT_DIM + ABILITY_DOMINANCE_DIM] = 1.0  # known
    # Move 0: Surf (ID 57), Known=1, partially depleted PP (8 current / 24 max)
    m_slot_layout = layout['pokemon']['moves']['layout']['slot_layout']
    from agents.observation.constants import MAX_PP
    _cur_pp, _max_pp = 8, 24
    obs[pk0_start + POKEMON_MOVES_OFFSET + 0] = 57.0
    obs[pk0_start + POKEMON_MOVES_OFFSET + m_slot_layout['known']['offset']]    = 1.0
    obs[pk0_start + POKEMON_MOVES_OFFSET + m_slot_layout['current_pp']['offset']] = _cur_pp / MAX_PP
    obs[pk0_start + POKEMON_MOVES_OFFSET + m_slot_layout['max_pp']['offset']]     = _max_pp / MAX_PP

    # gen3_entity_rehome_v1: the reactive matchup matrices are DELETED from the obs and the
    # move network no longer has matchup/validity input columns.

    obs_tensor = torch.from_numpy(obs).unsqueeze(0)
    obs_dict = {"observation": obs_tensor}

    # 4. Execute and Inspect
    with torch.no_grad():
        model(obs_dict)

    # --- VERIFICATIONS ---
    move_in = captured_inputs["move_input"]
    role_in = captured_inputs["role_input"]

    # Compute move_input known and matchup indices from layout
    emb_dim = layout['move_embedding_dim'] + layout['type_embedding_dim']  # 32
    remnant_dim = model.pokemon_encoder.move_remnant_dim                    # 6
    known_idx    = emb_dim + remnant_dim                                    # 38

    # PP indices: current_pp and max_pp sit in the remnants block just before the known flag
    _rem1 = m_slot_layout['type']['offset'] - m_slot_layout['power']['offset']         # 3
    _rem2 = m_slot_layout['known']['offset'] - (m_slot_layout['type']['offset'] + m_slot_layout['type']['dim'])  # 1
    current_pp_idx = emb_dim + _rem1 + _rem2        # first PP dim in remnants
    max_pp_idx     = current_pp_idx + 1

    m0_features = move_in[0]
    assert m0_features[known_idx]      == 1.0,                           f"Move Known Flag mismatch at idx {known_idx}: {m0_features[known_idx]}"
    assert float(m0_features[current_pp_idx]) == pytest.approx(_cur_pp / MAX_PP), f"current_pp mismatch at idx {current_pp_idx}: {m0_features[current_pp_idx]}"
    assert float(m0_features[max_pp_idx])     == pytest.approx(_max_pp / MAX_PP), f"max_pp mismatch at idx {max_pp_idx}: {m0_features[max_pp_idx]}"
    assert float(m0_features[current_pp_idx]) < float(m0_features[max_pp_idx]),   "current_pp should be less than max_pp for depleted move"

    # Compute role_input known-flag indices from layout
    # Compute directly from known layout offsets in role-encoder input
    # role input = species_emb(32) + stats(6) + item_emb(16) + item_known(1) + item_consumed(1)
    #             + pk_types(32) + ability1_emb(16) + ability2_emb(16) + dominance(1)
    #             + ability_known(1) + ...
    _species_emb = layout['species_embedding_dim']     # 32
    _stats       = 6
    _item_emb    = layout['item_embedding_dim']        # 16
    item_known_role_idx = _species_emb + _stats + _item_emb   # 54
    _pk_types    = layout['type_embedding_dim'] * 2    # 32 (concatenated pair)
    _ability_emb = layout['ability_embedding_dim']     # 16
    # Both ability embeddings + dominance scalar precede the known flag in role input
    ability_known_role_idx = item_known_role_idx + 1 + 1 + _pk_types + 2 * _ability_emb + 1

    p0_features = role_in[0]
    assert p0_features[item_known_role_idx]    == 1.0, f"Item Known Flag mismatch at idx {item_known_role_idx}: {p0_features[item_known_role_idx]}"
    assert p0_features[ability_known_role_idx] == 1.0, f"Ability Known Flag mismatch at idx {ability_known_role_idx}: {p0_features[ability_known_role_idx]}"

    print("✅ Full Model Forensic Audit PASSED!")
    print(f"  - Move Path: Known (idx {known_idx}), cur_pp (idx {current_pp_idx}), max_pp (idx {max_pp_idx})")
    print(f"  - Pokemon Path: Item Known (idx {item_known_role_idx}), Ability Known (idx {ability_known_role_idx})")

if __name__ == "__main__":
    test_model_full_embedding_forensics()
