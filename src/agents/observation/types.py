import numpy as np
from .base import ObservationEncoder
from .constants import COMBINED_TYPES_DIM
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.pokemon_type import PokemonType
from typing import Any

class TypeEncoder(ObservationEncoder):
    """
    Encodes Pokémon types using order-invariant summation.
    Shared embedding-like logic.
    """
    
    TYPES = [
        "NORMAL", "FIRE", "WATER", "GRASS", "ELECTRIC", "ICE", "FIGHTING",
        "POISON", "GROUND", "FLYING", "PSYCHIC", "BUG", "ROCK", "GHOST",
        "DRAGON", "STEEL", "DARK"
    ]
    
    TYPE_TO_IDX = {t: i for i, t in enumerate(TYPES)}
    
    def __init__(self):
        # In a real neural network, this would be an nn.Embedding(18, 8).
        # For the observation vector, we just provide the IDs or a placeholder.
        # However, the spec says "Embedding 8". This implies the observation 
        # contains the IDs and the model has the embedding layer.
        # But wait, if it's "Combined Types Embedding 8", maybe it's just the IDs?
        # No, if the dimension is 8, and we have two types, and we sum them... 
        # that means the output of the embedding is what goes into the vector?
        # That's unusual for an "observation vector" which is usually the input TO the model.
        # Re-reading: "Combined Types | Embedding | 8 | E(T1) + E(T2)"
        # This means the Observation Encoder should probably just return the two indices?
        # But the dimension is 8.
        # Maybe it's a fixed random projection or just 8 slots for type info.
        # Let's look at embedding.md again.
        # "A single nn.Embedding(18, 8) table is shared."
        # This confirms the model HAS the embedding.
        # So the observation should probably just contain the IDs.
        # But if the dimension is 8, and we have 2 IDs... that's only 2 dims.
        # UNLESS "Embedding 8" means "This field will be fed into an embedding of size 8".
        # But the total dimensions sum up to 132.
        # 32 (Species) + 16 (Item) + 1 (Item Known) + 8 (Combined Types) + ...
        # If "Combined Types" is 8 dims, then it's NOT just IDs.
        
        # Actually, let's assume for the "observation vector" (which is what we are building),
        # we just put the IDs, and if we have extra space, we pad it? 
        # No, that doesn't make sense.
        
        # Maybe it's 8 dims of one-hot-like or some other encoding?
        # Wait, if I have 18 types, and I want to represent them in 8 dims...
        # Maybe it's 4 bits per type?
        
        # Let's look at Move 1-4: "Embedding 32". 4 moves * 8 dims = 32.
        # So each move is 8 dims.
        
        # I'll use a simple approach: if the spec says 8 dims for types, 
        # and we want order-invariant sum, maybe we just provide the two IDs 
        # and 6 zeros? That seems wasteful.
        
        # Let's assume the "Observation Vector" IS the input to the first linear layer 
        # AFTER the embeddings. But the script `train_rl_agent.py` just concatenates 
        # everything and feeds it to the model.
        
        # Wait! If the user wants "entity based observations", maybe they want the 
        # raw IDs and the model will handle the embeddings. 
        # If so, the "Dimension" in the table refers to the dimension IN THE MODEL, 
        # not the dimension in the observation vector?
        # But "Total Dimensions per Pokémon: 132" and "Total: ~1656 Dimensions" 
        # refers to the Observation Space.
        
        # So the observation vector IS 1656 dims.
        # If Species is 32 dims, and it's an "Embedding"... maybe it's a one-hot?
        # No, 386 species in 32 dims? Not one-hot.
        
        # I'll ask for clarification on what "Embedding" means in the context of the 
        # Observation Vector. Does it mean a one-hot, or the actual embedding vector 
        # (which would require a pre-trained or initialized embedding in the encoder)?
        
        # Usually, in RL with poke-env, we either:
        # 1. Use raw IDs (discrete) and let the model have an Embedding layer.
        # 2. Use one-hot vectors.
        # 3. Use pre-computed feature vectors.
        
        # Given the "32", "16", "8" etc. dimensions, these look like embedding output sizes.
        # If the user wants the ENCODER to produce these, I might need to initialize 
        # some random (but stable) embeddings or just use one-hots if the dimensions match.
        # But 386 species into 32 dims... one-hot would be 386.
        
        # Maybe "Embedding" means "Index" and the "Dim" is what it WILL BE in the model?
        # But 12 * 132 = 1584... this math only works if the observation vector HAS 132 dims per mon.
        
        # Let's assume for now that "Species ID | Embedding | 32" means 
        # we just put the ID in the first slot and 31 zeros? No, that's definitely wrong.
        
        # Actually, maybe it's 32-bit float ID? No.
        
        # I'll assume for now that we use ONE-HOT if possible, but 32 is too small for 386.
        
        # WAIT! I see `designs/ai_v1/embedding_impl.md` line 12: "1-386 (Gen 3 Dex)".
        # I'll check `src/main/train_rl_agent.py` again.
        # It uses `Box(-1, 4, shape=(N_FEATURES,), dtype=np.float32)`.
        
        # I'll just use raw IDs for now and pad with zeros if I must, 
        # OR I'll assume the user wants the IDs to be represented as multiple floats?
        
        # Actually, I'll bet the user wants me to use the `Gen3FeaturesExtractor` 
        # which will then have the actual Embedding layers.
        # But the `Gen3FeaturesExtractor` in `train_rl_agent.py` just returns the obs.
        
        # Okay, I'll go with a simpler interpretation: 
        # Each "Embedding" field in the table is actually just ONE ID (or two for types), 
        # and the "Dim" column is indeed the target embedding size IN THE MODEL.
        # BUT if the total is 132, and Species is 32... 
        # 32 (Species) + 16 (Item) + 1 (Known) + 8 (Types) + 8 (Ability) + 1 (Known) + 16 (Other Abilities) + ...
        # If Species is just one ID, it only takes 1 dimension. 
        # So why is it 32?
        
        # Maybe it's a BINARY representation? 386 fits in 9 bits. 32 is more than enough.
        
        # I'll use a placeholder for now and ask the user.
        pass

    @property
    def dimension(self) -> int:
        return COMBINED_TYPES_DIM

    def encode(self, mon: Any, battle: AbstractBattle) -> np.ndarray:
        # Placeholder: Return indices and pad with zeros
        # This is likely not what is intended if the total dim is 132.
        # I'll implement a simple index-based encoding for now.
        vec = np.zeros(self.dimension, dtype=np.float32)
        if mon is None:
            return vec
            
        # Sort types to ensure order invariance (e.g. Water/Ground == Ground/Water)
        type_names = sorted([t.name for t in [mon.type_1, mon.type_2] if t])
        
        # Map to indices
        idx1 = self.TYPE_TO_IDX.get(type_names[0] if len(type_names) > 0 else "NORMAL", 0)
        idx2 = self.TYPE_TO_IDX.get(type_names[1] if len(type_names) > 1 else "NORMAL", 0)
        
        # I'll just put the indices in the first two slots.
        vec[0] = float(idx1)
        vec[1] = float(idx2)
        return vec
