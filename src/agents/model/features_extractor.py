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
    Supports shared latent embeddings for Species, Moves, Items, Abilities, and Types.
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
        # Shared Type Embedding for both Pokémon and Moves
        self.type_embedding = torch.nn.Embedding(
            layout['max_types'],
            layout['type_embedding_dim']
        )
        
        # 1.5 Shared Move Processor (Step 1)
        move_input_dim = layout['move_embedding_dim'] + layout['type_embedding_dim'] + 6 + 1
        self.move_network = torch.nn.Sequential(
            torch.nn.Linear(move_input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32)
        )
        
        # 1.6 Pokémon Role Encoder (Step 2)
        # pokemon_enriched dim is 226:
        # (Species: 32, Stats: 6, Item: 16+1, Types: 16, Ability: 16+1, Condition: 8, Moves: 128, HP: 2)
        self.role_encoder = torch.nn.Sequential(
            torch.nn.Linear(226, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128) # Final Role Token Size
        )
        
        # 2. Dynamic Input Dimension Discovery (Dummy Forward)
        # We run a single fake observation through the logic to determine the exact projection dimension.
        with torch.no_grad():
            dummy_obs = torch.zeros((1, layout['total_dim']))
            dummy_dict = {"observation": dummy_obs}
            # We call forward_internal which does everything except the projection
            sample_output = self.forward_internal(dummy_dict)
            self.projection_input_dim = sample_output.shape[1]
        
        # 3. Projection layer
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
            
            # --- Layout Constants ---
            NAME_W, ACTV_W, TYPE_W = 12, 6, 18
            TAB = "    " 
            
            print("\n--- TEAM SUMMARIES ---")
            for i, mon in enumerate(desc['our_team']):
                # Primary Row
                active_str = "[actv]" if mon.get('active') else "      "
                s = mon['stats']
                stats_str = f"{s['hp']}/{s['atk']}/{s['def']}/{s['spa']}/{s['spd']}/{s['spe']}"
                type_str = mon['types'].lower()
                species_name = mon['species'].lower()
                
                line1 = f"[OUR {i}] {species_name:{NAME_W}} {active_str:{ACTV_W}}  {type_str:{TYPE_W}}  hp: {mon['hp']:>6}  status: {mon['status'].lower():7}  {stats_str}"
                print(line1)
                
                # Secondary Row (Minimalist)
                line2 = f"{TAB}item: {mon['item'].lower():17}  ably: {mon['ability'].lower():16}  moves: {mon.get('moves', [])}"
                print(line2)
                
            print("-" * 30)
            for i, mon in enumerate(desc['opp_team']):
                active_str = "[actv]" if mon.get('active') else "      "
                s = mon['stats']
                stats_str = f"{s['hp']}/{s['atk']}/{s['def']}/{s['spa']}/{s['spd']}/{s['spe']}"
                type_str = mon['types'].lower()
                species_name = mon['species'].lower()
                
                line1 = f"[OPP {i}] {species_name:{NAME_W}} {active_str:{ACTV_W}}  {type_str:{TYPE_W}}  hp: {mon['hp']:>6}  status: {mon['status'].lower():7}  {stats_str}"
                print(line1)
                
                line2 = f"{TAB}item: {mon['item'].lower():17}  ably: {mon['ability'].lower():16}  moves: {mon.get('moves', [])}"
                print(line2)
            
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

    def forward_internal(self, obs):
        """Internal forward pass that constructs the combined feature vector without the final projection."""
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
        species_info = pk_layout['species']
        species_idx = species_info['offset'] + species_info['layout']['species_id']['offset']
        species_ids = pokemon_part[:, :, species_idx].long()
        
        # Move IDs & Move Type IDs
        moves_info = pk_layout['moves']
        moves_offset = moves_info['offset']
        moves_layout = moves_info['layout']
        move_id_tensors = []
        move_type_id_tensors = []
        for i in range(4):
            slot_idx = moves_offset + moves_layout['slots'][i]['offset']
            # Move ID at index 0 of slot
            move_id_tensors.append(pokemon_part[:, :, slot_idx].long().unsqueeze(2))
            # Move Type ID at index 4 of slot
            move_type_id_tensors.append(pokemon_part[:, :, slot_idx + 4].long().unsqueeze(2))
            
        all_move_ids = torch.cat(move_id_tensors, dim=2) # [B, 12, 4]
        all_move_type_ids = torch.cat(move_type_id_tensors, dim=2) # [B, 12, 4]
        
        # Item ID
        items_info = pk_layout['items']
        items_layout = items_info['layout']
        item_idx = items_info['offset'] + items_layout['id']['offset']
        item_ids = pokemon_part[:, :, item_idx].long() # [B, 12]
        
        # Ability ID
        abilities_info = pk_layout['abilities']
        abilities_layout = abilities_info['layout']
        ability_idx = abilities_info['offset'] + abilities_layout['id']['offset']
        ability_ids = pokemon_part[:, :, ability_idx].long() # [B, 12]
        
        # Pokémon Type IDs (2 IDs in the 'types' block)
        types_info = pk_layout['types']
        types_layout = types_info['layout']
        type1_ids = pokemon_part[:, :, types_info['offset'] + types_layout['type1']['offset']].long()
        type2_ids = pokemon_part[:, :, types_info['offset'] + types_layout['type2']['offset']].long()
        
        # 3. Diagnostic Trace (Only in real forward)
        # We skip this in dummy pass by checking if it's training or testing
        if self.training or not hasattr(self, 'dummy_pass'):
             pass # Trace logic handled in forward()

        # 4. Embed Everything
        embedded_species = self.species_embedding(species_ids) # [B, 12, 32]
        
        embedded_moves = self.move_embedding(all_move_ids) # [B, 12, 4, 16]
        embedded_move_types = self.type_embedding(all_move_type_ids) # [B, 12, 4, 16]
        embedded_items = self.item_embedding(item_ids) # [B, 12, 16]
        embedded_abilities = self.ability_embedding(ability_ids) # [B, 12, 16]
        
        # Pokémon Types: sum of embeddings (E1 + E2)
        embedded_t1 = self.type_embedding(type1_ids) # [B, 12, 16]
        embedded_t2 = self.type_embedding(type2_ids) # [B, 12, 16]
        embedded_pk_types = embedded_t1 + embedded_t2 # [B, 12, 16]
        
        # 5. Construct the enriched Pokemon vector by stitching
        # Part A: Stats (between Species (0) and Items)
        part_a = pokemon_part[:, :, species_idx + 1 : items_info['offset']] # [B, 12, 36]
        
        # Part B: Item remnants (Known flag)
        item_remnant_idx = items_info['offset'] + items_layout['known']['offset']
        item_remnant = pokemon_part[:, :, item_remnant_idx : item_remnant_idx + 1] # [B, 12, 1]
        
        # Part C: Ability remnants (Known flag)
        ability_remnant_idx = abilities_info['offset'] + abilities_layout['known']['offset']
        ability_remnant = pokemon_part[:, :, ability_remnant_idx : ability_remnant_idx + 1] # [B, 12, 1]
        
        # Part D: Condition (between Abilities and Moves)
        part_d = pokemon_part[:, :, abilities_info['offset'] + 25 : moves_offset] # [B, 12, 8]
        
        # Part E: Move remnants (Power, Secondary, Recoil - everything but ID and Type)
        move_remnants = []
        for i in range(4):
            slot_start = moves_offset + moves_layout['slots'][i]['offset']
            # Indices: 1 (Power), 2 (Secondary), 3 (Recoil)
            move_remnants.append(pokemon_part[:, :, slot_start + 1 : slot_start + 4])
            # Index 5, 6, 7 are extra remnants in the 8-dim slot
            move_remnants.append(pokemon_part[:, :, slot_start + 5 : slot_start + 8])
        all_move_remnants = torch.cat(move_remnants, dim=2) # [B, 12, 24] (4 slots * 6 remnants)
        
        # Part F: Move Known Flags
        known_idx = moves_offset + moves_layout['known']['offset']
        known_dim = moves_layout['known']['dim']
        known_flags = pokemon_part[:, :, known_idx : known_idx + known_dim] # [B, 12, 4]
        
        # Part G: HP and Active Flag
        hp_and_active = pokemon_part[:, :, -2:] # [B, 12, 2]
        
        # Final stitch
        # --- SHARED MOVE PROCESSING (Step 1) ---
        # Reshape move remnants and flags to align with the 4 slots
        move_remnants_reshaped = all_move_remnants.reshape(batch_size, 12, 4, 6)
        known_flags_reshaped = known_flags.reshape(batch_size, 12, 4, 1)
        
        # Combine all move features into [B, 12, 4, 39]
        move_features = torch.cat([
            embedded_moves, 
            embedded_move_types, 
            move_remnants_reshaped, 
            known_flags_reshaped
        ], dim=3)
        
        # Process through shared network
        processed_moves = self.move_network(move_features.reshape(-1, move_features.shape[-1]))
        processed_moves = processed_moves.reshape(batch_size, 12, -1) # [B, 12, 4 * 32 = 128]
        
        pokemon_enriched = torch.cat([
            embedded_species,      # 32
            part_a,                # 36
            embedded_items,        # 16
            item_remnant,          # 1
            embedded_pk_types,     # 16
            embedded_abilities,    # 16
            ability_remnant,       # 1
            part_d,                # 8
            processed_moves,       # 128 (Shared Processor Output)
            hp_and_active          # 2
        ], dim=2) # [B, 12, 256]
        
        # --- POKÉMON ROLE ENCODER (Step 2) ---
        # Reshape to [B*12, 226] for shared processing
        role_tokens = self.role_encoder(pokemon_enriched.reshape(-1, 226))
        role_tokens = role_tokens.reshape(batch_size, 12, 128) # [B, 12, 128]
        
        pokemon_flat = role_tokens.reshape(batch_size, -1) # [B, 1536]
        combined = torch.cat([pokemon_flat, remaining_part], dim=1)
        return combined

    def forward(self, obs):
        combined = self.forward_internal(obs)
        
        # Diagnostic Trace logic
        current_time = time.time()
        if current_time - self.last_trace_time > TRACE_INTERVAL:
            self.last_trace_time = current_time
            # For trace, we need the original parts again (or pass them through)
            # Re-extracting for trace is fine since it's infrequent
            x = obs["observation"]
            parts = self.layout['parts']
            ot = parts['our_team']
            our_team = x[:, ot['start']:ot['end']].reshape(x.shape[0], *ot['reshape'])
            opt = parts['opp_team']
            opp_team = x[:, opt['start']:opt['end']].reshape(x.shape[0], *opt['reshape'])
            pokemon_part = torch.cat([our_team, opp_team], dim=1)
            
            pk_layout = self.layout['pokemon']
            species_info = pk_layout['species']
            species_idx = species_info['offset'] + species_info['layout']['species_id']['offset']
            species_ids = pokemon_part[:, :, species_idx].long()
            
            self._print_deep_trace(x, pokemon_part, species_ids)
        
        return self.activation(self.projection(combined))
