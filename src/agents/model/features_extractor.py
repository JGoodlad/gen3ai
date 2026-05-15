import torch
import numpy as np
import time
from gymnasium import spaces
from typing import Dict, Any
from agents.observation.state_encoder import Gen3ObservationEncoder
from agents.observation.constants import (
    TRACE_INTERVAL, 
    TEAM_SIZE, 
    GLOBAL_ENV_DIM, 
    POKEMON_FULL_DIM
)
from utils.logging.rate_limiter import RateLimitedLogger
from utils.logging.levels import LogLevel

class Gen3FeaturesExtractor(torch.nn.Module):
    """
    Custom feature extractor for Gen 3 Pokémon battles.
    Uses a dynamic layout provided by the Observation Encoder to avoid magic constants.
    Supports shared latent embeddings for Species, Moves, Items, Abilities, and Types.
    """
    def __init__(self, observation_space: spaces.Dict, layout: Dict[str, Any] = None, mappings: Dict[str, Any] = None, log_level: LogLevel = LogLevel.QUIET):
        super().__init__()
        self.layout = layout
        self.mappings = mappings
        self.log_level = log_level
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
        # Input: Move Embedding (16) + Type Embedding (16) + Remnants (4) + Known (1)
        # Context: HP (1) + Turn (1) + Weather (6) + Fainted (2) + Spikes (2) = 12
        # Matchups: Effectiveness against 6 potential targets = 6
        # Remnants: Power, Secondary, Recoil, Category (known extracted separately below)
        # Total: 37 + 12 + 6 = 55
        move_input_dim = layout['move_embedding_dim'] + layout['type_embedding_dim'] + 4 + 1 + 12 + 6
        self.move_network = torch.nn.Sequential(
            torch.nn.Linear(move_input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32)
        )
        
        # 1.6 Pokémon Role Encoder (Step 2)
        # pokemon_enriched dim is move_processor_output (128) + context_broadcasted (11) + others
        self.role_token_size = 128
        role_input_dim = 237 # (32 + 6 + 16 + 1 + 16 + 16 + 1 + 8 + 128 + 2) + 11
        self.role_encoder = torch.nn.Sequential(
            torch.nn.Linear(role_input_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.role_token_size) # Final Role Token Size
        )
        
        # 1.7 Team-Wide Attention (Step 4)
        # 1.7.1 Status Biases (0: Our Active, 1: Our Bench, 2: Their Active, 3: Their Bench)
        self.status_embedding = torch.nn.Embedding(4, self.role_token_size)

        # 1.7.2 Multi-Path Attention Heads
        self.pressure_attn = torch.nn.MultiheadAttention(embed_dim=self.role_token_size, num_heads=4, batch_first=True)
        self.safety_attn = torch.nn.MultiheadAttention(embed_dim=self.role_token_size, num_heads=4, batch_first=True)
        self.synergy_attn = torch.nn.MultiheadAttention(embed_dim=self.role_token_size, num_heads=4, batch_first=True)

        # 1.7.3 LayerNorm for Residual Stability
        self.norm1 = torch.nn.LayerNorm(self.role_token_size)
        self.norm2 = torch.nn.LayerNorm(self.role_token_size)
        self.norm3 = torch.nn.LayerNorm(self.role_token_size)
        
        # 2. Dynamic Input Dimension Discovery (Dummy Forward)
        # We run a single fake observation through the logic to determine the exact projection dimension.
        with torch.no_grad():
            dummy_obs = torch.zeros((1, layout['total_dim']))
            dummy_dict = {"observation": dummy_obs}
            # We call forward_internal which does everything except the projection
            sample_output = self.forward_internal(dummy_dict)
            self.projection_input_dim = sample_output.shape[1]
        
        # 3. Projection layer
        self.projection_dim = 512
        self.projection = torch.nn.Linear(self.projection_input_dim, self.projection_dim)
        self.activation = torch.nn.ReLU()
        self.features_dim = self.projection_dim
        self.trace_logger = RateLimitedLogger(interval_seconds=TRACE_INTERVAL)
        
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
            print(f"Fainted: {momentum.get('fainted_our', 0)} (Us) / {momentum.get('fainted_opp', 0)} (Them)")
            
            # --- MATCHUP MATRIX ---
            our_vs_their = momentum.get('our_vs_their') # [6, 4, 6]
            if our_vs_their is not None:
                print("\n--- MATCHUPS (Our Moves vs Their Team) ---")
                opp_mons = [m['species'][:6] for m in desc['opp_team']]
                if opp_mons:
                    header = "               " + "".join([f"{m:>8}" for m in opp_mons])
                    print(header)
                    for i, mon in enumerate(desc['our_team']):
                        moves = mon.get('moves', [])
                        for m_idx in range(4):
                            if m_idx < len(moves):
                                move_name = moves[m_idx]
                                row_label = f"[{i}] {mon['species'][:3]}:{move_name[:7]}"
                                vals = "".join([f"{our_vs_their[i, m_idx, j]:>8.1f}x" for j in range(len(opp_mons))])
                                print(f"{row_label:15} {vals}")
                else:
                    print("No opponent Pokémon revealed yet.")

            their_vs_our = momentum.get('their_vs_our') # [6, 4, 6]
            if their_vs_our is not None and np.any(their_vs_our):
                print("\n--- MATCHUPS (Their Moves vs Our Team) ---")
                our_mons = [m['species'][:6] for m in desc['our_team']]
                header = "               " + "".join([f"{m:>8}" for m in our_mons])
                print(header)
                for i, mon in enumerate(desc['opp_team']):
                    moves = mon.get('moves', [])
                    for m_idx in range(4):
                        if m_idx < len(moves):
                            move_name = moves[m_idx]
                            row_label = f"[{i}] {mon['species'][:3]}:{move_name[:7]}"
                            vals = "".join([f"{their_vs_our[i, m_idx, j]:>8.1f}x" for j in range(len(our_mons))])
                            print(f"{row_label:15} {vals}")
            
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
        
        # 1.1 Context Features for Move Selection
        # Global: Weather (offset 0), Spikes (6), Turn (8)
        global_start = parts['global']['start'] - ctx['start']
        weather_feature = remaining_part[:, global_start : global_start + 6] # [B, 6]
        spikes_feature = remaining_part[:, global_start + 6 : global_start + 8] # [B, 2]
        turn_feature = remaining_part[:, global_start + 8 : global_start + 9] # [B, 1]
        
        # Fainted counts (Phase) are at Reactive block offset 8
        reactive_start = parts['reactive']['start'] - ctx['start']
        fainted_feature = remaining_part[:, reactive_start + 8 : reactive_start + 10] # [B, 2]
        
        # 1.2 Matchup Matrix Extraction (New)
        # Extract 144 dims for Our Matchups and 144 for Theirs
        our_matchups_flat = remaining_part[:, reactive_start + 16 : reactive_start + 160]
        their_matchups_flat = remaining_part[:, reactive_start + 160 : reactive_start + 304]
        
        # Reshape to [B, 6, 4, 6] to align with Move Processor
        our_matchups = our_matchups_flat.reshape(batch_size, TEAM_SIZE, 4, TEAM_SIZE)
        their_matchups = their_matchups_flat.reshape(batch_size, TEAM_SIZE, 4, TEAM_SIZE)
        matchups_all = torch.cat([our_matchups, their_matchups], dim=1) # [B, 12, 4, 6]
        
        pokemon_part = torch.cat([our_team, opp_team], dim=1) # [B, 2 * TEAM_SIZE, POKEMON_FULL_DIM]
        
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
        embedded_species = self.species_embedding(species_ids) # [B, 2 * TEAM_SIZE, 32]
        
        embedded_moves = self.move_embedding(all_move_ids) # [B, 12, 4, 16]
        embedded_move_types = self.type_embedding(all_move_type_ids) # [B, 12, 4, 16]
        embedded_items = self.item_embedding(item_ids) # [B, 12, 16]
        embedded_abilities = self.ability_embedding(ability_ids) # [B, 12, 16]
        
        # Pokémon Types: sum of embeddings (E1 + E2)
        embedded_t1 = self.type_embedding(type1_ids) # [B, 12, 16]
        embedded_t2 = self.type_embedding(type2_ids) # [B, 12, 16]
        embedded_pk_types = embedded_t1 + embedded_t2 # [B, 12, 16]
        
        # 5. Construct the enriched Pokemon vector by stitching
        # Part A: Stats (between Species ID and Items)
        species_id_layout = species_info['layout']['species_id']
        stats_start = species_idx + species_id_layout['dim']
        part_a = pokemon_part[:, :, stats_start : items_info['offset']] # [B, 12, 6]
        
        # Part B: Item remnants (Known flag)
        item_remnant_idx = items_info['offset'] + items_layout['known']['offset']
        item_remnant = pokemon_part[:, :, item_remnant_idx : item_remnant_idx + items_layout['known']['dim']] # [B, 12, 1]
        
        # Part C: Ability remnants (Known flag)
        ability_remnant_idx = abilities_info['offset'] + abilities_layout['known']['offset']
        ability_remnant = pokemon_part[:, :, ability_remnant_idx : ability_remnant_idx + abilities_layout['known']['dim']] # [B, 12, 1]
        
        # Part D: Condition (between Abilities and Moves)
        condition_start = abilities_info['offset'] + abilities_info['dim']
        part_d = pokemon_part[:, :, condition_start : moves_offset] # [B, 12, 8]
        
        # Part E: Move remnants (Power, Secondary, Recoil - everything but ID and Type)
        move_remnants = []
        # Fallback for models trained before slot_layout was added to the saved layout
        if 'slot_layout' not in moves_layout:
            moves_layout['slot_layout'] = {
                "id": {"offset": 0, "dim": 1},
                "power": {"offset": 1, "dim": 1},
                "secondary": {"offset": 2, "dim": 1},
                "recoil": {"offset": 3, "dim": 1},
                "type": {"offset": 4, "dim": 1},
                "category": {"offset": 5, "dim": 1},
                "known": {"offset": 6, "dim": 1}
            }
        m_slot_layout = moves_layout['slot_layout']
        for i in range(4):
            slot_info = moves_layout['slots'][i]
            slot_start = moves_offset + slot_info['offset']
            
            # Power, Secondary, Recoil  [3 dims]
            rem1_start = slot_start + m_slot_layout['power']['offset']
            rem1_end = slot_start + m_slot_layout['type']['offset']
            move_remnants.append(pokemon_part[:, :, rem1_start : rem1_end])

            # Category  [1 dim] — stop before known flag to avoid duplication
            rem2_start = slot_start + m_slot_layout['type']['offset'] + m_slot_layout['type']['dim']
            rem2_end = slot_start + m_slot_layout['known']['offset']
            move_remnants.append(pokemon_part[:, :, rem2_start : rem2_end])
        all_move_remnants = torch.cat(move_remnants, dim=2) # [B, 12, 16]
        
        # Part F: Move Known Flags (Now interleaved in slots)
        known_flags_tensors = []
        for i in range(4):
            slot_start = moves_offset + moves_layout['slots'][i]['offset']
            # Known flag is at index 6 of the slot
            known_flags_tensors.append(pokemon_part[:, :, slot_start + 6 : slot_start + 7])
        known_flags = torch.cat(known_flags_tensors, dim=2) # [B, 12, 4]
        
        # Part G: HP and Active Flag
        hp_offset = pk_layout['hp']['offset']
        # Extract from HP offset to the end of the Pokémon vector (includes Active Flag)
        hp_and_active = pokemon_part[:, :, hp_offset:] 
        
        # Final stitch
        # --- SHARED MOVE PROCESSING (Step 1) ---
        # Reshape move remnants and flags to align with the 4 slots
        move_remnants_reshaped = all_move_remnants.reshape(batch_size, 12, 4, 4)
        known_flags_reshaped = known_flags.reshape(batch_size, 12, 4, 1)
        
        # Combine all move features into [B, 12, 4, 51]
        # Context: HP (1), Turn (1), Weather (6), Fainted (2), Spikes (2)
        hp_feature = hp_and_active[:, :, 0:1] # [B, 12, 1]
        turn_expanded = turn_feature.unsqueeze(1).expand(-1, 2 * TEAM_SIZE, -1) # [B, 12, 1]
        weather_expanded = weather_feature.unsqueeze(1).expand(-1, 2 * TEAM_SIZE, -1) # [B, 12, 6]
        fainted_expanded = fainted_feature.unsqueeze(1).expand(-1, 2 * TEAM_SIZE, -1) # [B, 12, 2]
        spikes_expanded = spikes_feature.unsqueeze(1).expand(-1, 2 * TEAM_SIZE, -1) # [B, 12, 2]
        
        # Combine into a [B, 12, 12] context vector
        move_context = torch.cat([
            hp_feature, turn_expanded, weather_expanded, fainted_expanded, spikes_expanded
        ], dim=2)
        # Expand to each move slot [B, 12, 4, 12]
        move_context_final = move_context.unsqueeze(2).expand(-1, -1, 4, -1)
        
        move_features = torch.cat([
            embedded_moves, 
            embedded_move_types, 
            move_remnants_reshaped, 
            known_flags_reshaped,
            move_context_final,
            matchups_all
        ], dim=3)
        
        # Process through shared network
        processed_moves = self.move_network(move_features.reshape(-1, move_features.shape[-1]))
        processed_moves = processed_moves.reshape(batch_size, 2 * TEAM_SIZE, -1) # [B, 12, 4 * 32 = 128]
        
        pokemon_enriched = torch.cat([
            embedded_species,      # 32
            part_a,                # 6
            embedded_items,        # 16
            item_remnant,          # 1
            embedded_pk_types,     # 16
            embedded_abilities,    # 16
            ability_remnant,       # 1
            part_d,                # 8
            processed_moves,       # 128 (Shared Processor Output)
            hp_and_active          # 2
        ], dim=2) # [B, 12, 226]
        
        # --- POKÉMON ROLE ENCODER (Step 2) ---
        # Add Global context to the role encoder to make roles phase-aware and environment-aware
        global_context = torch.cat([turn_feature, weather_feature, fainted_feature, spikes_feature], dim=1) # [B, GLOBAL_ENV_DIM]
        context_broadcasted = global_context.unsqueeze(1).expand(-1, 2 * TEAM_SIZE, -1) # [B, 12, 11]
        pokemon_enriched_with_context = torch.cat([pokemon_enriched, context_broadcasted], dim=2)
        
        # Reshape to [B*12, role_input_dim] for shared processing
        role_tokens = self.role_encoder(pokemon_enriched_with_context.reshape(-1, pokemon_enriched_with_context.shape[-1]))
        role_tokens = role_tokens.reshape(batch_size, 2 * TEAM_SIZE, self.role_token_size) # [B, 12, 128]
        
        # --- TEAM-WIDE ATTENTION (Step 4) ---
        # 1. Find who is active on both teams
        active_flags = hp_and_active[:, :, 1] # [B, 12]
        our_active_idx = torch.argmax(active_flags[:, 0:TEAM_SIZE], dim=1) # [B]
        opp_active_idx = torch.argmax(active_flags[:, TEAM_SIZE:2*TEAM_SIZE], dim=1) + TEAM_SIZE # [B]

        # 2. Apply Status Biases (Pre-Attention)
        status_idx = torch.ones((batch_size, 12), device=role_tokens.device).long() # Default to Bench
        status_idx[:, 0:TEAM_SIZE] = 1   # Our Bench
        status_idx[:, TEAM_SIZE:2*TEAM_SIZE] = 3  # Their Bench
        status_idx[torch.arange(batch_size), our_active_idx] = 0 # Our Active
        status_idx[torch.arange(batch_size), opp_active_idx] = 2 # Their Active
        
        role_tokens = role_tokens + self.status_embedding(status_idx)

        # 3. Split Teams for Logic Paths
        our_team = role_tokens[:, 0:TEAM_SIZE, :]
        their_team = role_tokens[:, TEAM_SIZE:2*TEAM_SIZE, :]
        our_active = role_tokens[torch.arange(batch_size), our_active_idx].unsqueeze(1)
        their_active = role_tokens[torch.arange(batch_size), opp_active_idx].unsqueeze(1)

        # --- ATTENTION PATHS WITH RESIDUALS ---

        # Path 1: The Pressure (Update Our Active with Opponent Context)
        pressure_delta, _ = self.pressure_attn(our_active, their_team, their_team)
        our_active = self.norm1(our_active + pressure_delta)
        
        # Path 2: The Safety (Update Our Team with Opponent Active Context)
        safety_delta, _ = self.safety_attn(our_team, their_active, their_active)
        our_team = self.norm2(our_team + safety_delta)
        
        # Path 3: Synergy (Update Our Team with Internal Context)
        synergy_delta, _ = self.synergy_attn(our_team, our_team, our_team)
        our_team = self.norm3(our_team + synergy_delta)

        # --- FINAL AGGREGATION ---
        # Combine our refined team, the opponent's final role tokens, and the raw context
        our_team_flat = our_team.reshape(batch_size, -1)
        their_team_flat = their_team.reshape(batch_size, -1)
        our_active_refined = our_active.squeeze(1)

        combined = torch.cat([our_team_flat, their_team_flat, our_active_refined, remaining_part], dim=1)
        return combined

    def forward(self, obs):
        combined = self.forward_internal(obs)
        
        # Diagnostic Trace logic
        if self.log_level >= LogLevel.PERIODIC and self.trace_logger.should_log():
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
