import torch
import numpy as np
import time
from gymnasium import spaces
from typing import Dict, Any
from agents.observation.state_encoder import Gen3ObservationEncoder
from agents.observation.constants import TRACE_INTERVAL

class Gen3FeaturesExtractor(torch.nn.Module):
    """
    Custom feature extractor for Gen 3 Pokémon battles.
    Uses a dynamic layout provided by the Observation Encoder to avoid magic constants.
    Supports latent embeddings for Species, Move, Item, and Ability IDs.
    """
    def __init__(self, observation_space: spaces.Dict, layout: Dict[str, Any] = None, mappings: Dict[str, Any] = None):
        super().__init__()
        self.layout = layout
        self.mappings = mappings
        self._encoder = None # Lazy init for decoding
        
        # 1. Embedding Layers
        self.species_embedding = torch.nn.Embedding(
            layout['max_species'], 
            layout['species_embedding_dim']
        )
        self.move_embedding = torch.nn.Embedding(
            layout['max_moves'],
            layout['move_embedding_dim']
        )
        self.item_embedding = torch.nn.Embedding(
            layout['max_items'],
            layout['item_embedding_dim']
        )
        self.ability_embedding = torch.nn.Embedding(
            layout['max_abilities'],
            layout['ability_embedding_dim']
        )
        
        # 2. Calculate the size of the enriched vector
        pokemon_full_dim = layout['parts']['our_team']['reshape'][1]
        
        species_growth = layout['species_embedding_dim'] - 1
        moves_growth = 4 * (layout['move_embedding_dim'] - 1)
        item_growth = layout['item_embedding_dim'] - 1
        ability_growth = layout['ability_embedding_dim'] - 1
        
        self.enriched_mon_dim = pokemon_full_dim + species_growth + moves_growth + item_growth + ability_growth
        
        # 3. Calculate total projection input dimension
        num_pokemon = layout['parts']['our_team']['reshape'][0] + layout['parts']['opp_team']['reshape'][0]
        remaining_dim = layout['total_dim'] - (num_pokemon * pokemon_full_dim)
        
        self.projection_input_dim = (num_pokemon * self.enriched_mon_dim) + remaining_dim
        
        # 4. Projection layer
        self.projection = torch.nn.Linear(self.projection_input_dim, 512)
        self.activation = torch.nn.ReLU()
        self.features_dim = 512
        self.last_trace_time = 0
        
    def _print_deep_trace(self, x, pokemon_part, species_ids):
        if self._encoder is None and self.mappings:
            self._encoder = Gen3ObservationEncoder(self.mappings)
            
        print("\n" + "🧬" * 30)
        print(f"🧬 [DEEP TRACE - {time.strftime('%H:%M:%S')}]")
        print("=" * 60)
        
        if self._encoder:
            # Use the encoder's master description logic
            desc = self._encoder.describe_vector(x[0].cpu().numpy())
            world = desc.get('world', {})
            print(f"Turn: {world.get('turn', '???')} | Weather: {world.get('weather', 'NONE')} | Spikes: {world.get('our_spikes', 0)} (Us) / {world.get('opp_spikes', 0)} (Them)")
            
            print("\n--- OUR ACTIVE CONTEXT ---")
            ctx = desc.get('our_active', {})
            print(f"Boosts: {ctx.get('boosts', {})} | Volatiles: {ctx.get('volatiles', [])}")
            
            print("\n--- TEAM SUMMARIES ---")
            for i, mon in enumerate(desc['our_team']):
                active_str = " [Actv]" if mon.get('active') else "       "
                s = mon['stats']
                stats_str = f"{s['hp']}/{s['atk']}/{s['def']}/{s['spa']}/{s['spd']}/{s['spe']}"
                print(f"[OUR {i}] {mon['species']:12} | HP: {mon['hp']:6} | Status: {mon['status']:5}{active_str} | {stats_str}")
                print(f"  Item: {mon['item']:12} | Ably: {mon['ability']:12} | Moves: {mon.get('moves', [])}")
                
            print("-" * 30)
            for i, mon in enumerate(desc['opp_team']):
                active_str = " [Actv]" if mon.get('active') else "       "
                s = mon['stats']
                stats_str = f"{s['hp']}/{s['atk']}/{s['def']}/{s['spa']}/{s['spd']}/{s['spe']}"
                print(f"[OPP {i}] {mon['species']:12} | HP: {mon['hp']:6} | Status: {mon['status']:5}{active_str} | {stats_str}")
                print(f"  Item: {mon['item']:12} | Ably: {mon['ability']:12} | Moves: {mon.get('moves', [])}")
            
            momentum = desc.get('momentum', {})
            print(f"\n--- MOMENTUM ---")
            print(f"Fainted: {momentum.get('fainted_our', 0)} (Us) / {momentum.get('fainted_opp', 0)} (Them) | Matchups: {momentum.get('move_mults', [])}")
            
            # --- INTEGRITY CHECK ---
            warnings, is_critical = self._encoder.integrity_check(x[0].cpu().numpy())
            if warnings:
                print("\n⚠️ [INTEGRITY CHECK WARNINGS]")
                for w in warnings:
                    print(f"  - {w}")
                    
            if is_critical:
                raise ValueError(f"CRITICAL INTEGRITY FAILURE: {warnings}")
        else:
            print("Trace available but encoder/mappings missing.")
            
        print("=" * 60 + "\n")

    def forward(self, obs):
        x = obs["observation"]
        batch_size = x.shape[0]
        
        # 1. Extract parts using dynamic layout
        parts = self.layout['parts']
        ot = parts['our_team']
        our_team = x[:, ot['start']:ot['end']].reshape(batch_size, *ot['reshape'])
        opt = parts['opp_team']
        opp_team = x[:, opt['start']:opt['end']].reshape(batch_size, *opt['reshape'])
        ctx = parts['context']
        remaining_part = x[:, ctx['start']:] 
        
        pokemon_part = torch.cat([our_team, opp_team], dim=1) # [B, 12, 133]
        
        # 2. Extract IDs for embedding
        pk_layout = self.layout['pokemon']
        
        # Species ID
        species_idx = pk_layout['species']['offset']
        species_ids = pokemon_part[:, :, species_idx].long()
        
        # Move IDs
        moves_layout = pk_layout['moves']
        moves_offset = moves_layout['offset']
        move_id_tensors = []
        for i in range(4):
            slot_idx = moves_offset + moves_layout['slots'][i]['offset']
            move_id_tensors.append(pokemon_part[:, :, slot_idx].long().unsqueeze(2))
        all_move_ids = torch.cat(move_id_tensors, dim=2) # [B, 12, 4]
        
        # Item ID
        items_layout = pk_layout['items']
        item_idx = items_layout['offset'] + items_layout['id']['offset']
        item_ids = pokemon_part[:, :, item_idx].long() # [B, 12]
        
        # Ability ID
        abilities_layout = pk_layout['abilities']
        ability_idx = abilities_layout['offset'] + abilities_layout['id']['offset']
        ability_ids = pokemon_part[:, :, ability_idx].long() # [B, 12]
        
        # 3. Diagnostic Trace
        current_time = time.time()
        if current_time - self.last_trace_time > TRACE_INTERVAL:
            self.last_trace_time = current_time
            self._print_deep_trace(x, pokemon_part, species_ids)
            
        # 4. Embed Everything
        embedded_species = self.species_embedding(species_ids) # [B, 12, 32]
        embedded_moves = self.move_embedding(all_move_ids) # [B, 12, 4, 16]
        embedded_moves_flat = embedded_moves.reshape(batch_size, 12, -1) # [B, 12, 64]
        embedded_items = self.item_embedding(item_ids) # [B, 12, 16]
        embedded_abilities = self.ability_embedding(ability_ids) # [B, 12, 16]
        
        # 5. Construct the enriched Pokemon vector by stitching
        
        # Part A: Stats (between Species (0) and Items (37))
        # species_idx is 0. items_layout['offset'] is 37.
        part_a = pokemon_part[:, :, species_idx + 1 : items_layout['offset']] # [B, 12, 36]
        
        # Part B: Item remnants (Known flag)
        item_remnant_idx = items_layout['offset'] + items_layout['known']['offset']
        item_remnant = pokemon_part[:, :, item_remnant_idx : item_remnant_idx + 1] # [B, 12, 1]
        
        # Part C: Types (between Items and Abilities)
        # items_layout['offset'] + dim(17) = 54. abilities_layout['offset'] is 62.
        part_c = pokemon_part[:, :, items_layout['offset'] + 17 : abilities_layout['offset']] # [B, 12, 8]
        
        # Part D: Ability remnants (Known flag)
        ability_remnant_idx = abilities_layout['offset'] + abilities_layout['known']['offset']
        ability_remnant = pokemon_part[:, :, ability_remnant_idx : ability_remnant_idx + 1] # [B, 12, 1]
        
        # Part E: Condition (between Abilities and Moves)
        # abilities_layout['offset'] + dim(25) = 87. moves_offset is 95.
        part_e = pokemon_part[:, :, abilities_layout['offset'] + 25 : moves_offset] # [B, 12, 8]
        
        # Part F: Move remnants
        move_remnants = []
        for i in range(4):
            slot_start = moves_offset + moves_layout['slots'][i]['offset']
            slot_dim = moves_layout['slots'][i]['dim']
            move_remnants.append(pokemon_part[:, :, slot_start + 1 : slot_start + slot_dim])
        all_move_remnants = torch.cat(move_remnants, dim=2) # [B, 12, 28]
        
        # Part G: Move Known Flags
        known_idx = moves_offset + moves_layout['known']['offset']
        known_dim = moves_layout['known']['dim']
        known_flags = pokemon_part[:, :, known_idx : known_idx + known_dim] # [B, 12, 4]
        
        # Part H: HP and Active Flag
        hp_and_active = pokemon_part[:, :, -2:] # [B, 12, 2]
        
        # Final stitch
        pokemon_enriched = torch.cat([
            embedded_species,      # 32
            part_a,                # 36
            embedded_items,        # 16
            item_remnant,          # 1
            part_c,                # 8
            embedded_abilities,    # 16
            ability_remnant,       # 1
            part_e,                # 8
            embedded_moves_flat,   # 64
            all_move_remnants,     # 28
            known_flags,           # 4
            hp_and_active          # 2
        ], dim=2) # Total: 32+36+16+1+8+16+1+8+64+28+4+2 = 216? 
        # Wait, let's re-calculate. 
        # Original: 133.
        # Species: +31. Move: 4*+15 = +60. Item: +15. Ability: +15.
        # 133 + 31 + 60 + 15 + 15 = 254.
        
        pokemon_flat = pokemon_enriched.reshape(batch_size, -1)
        combined = torch.cat([pokemon_flat, remaining_part], dim=1)
        
        return self.activation(self.projection(combined))
