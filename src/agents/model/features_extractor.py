import torch
import numpy as np
import time
from gymnasium import spaces
from typing import Dict, Any, Optional
from agents.observation.state_encoder import Gen3ObservationEncoder
from agents.observation.constants import (
    TRACE_INTERVAL,
    TEAM_SIZE,
    GLOBAL_ENV_DIM,
    POKEMON_FULL_DIM
)
from agents.observation.turn_delta_encoder import TURN_DELTA_DIM
from utils.logging.rate_limiter import RateLimitedLogger
from utils.logging.levels import LogLevel

class Gen3FeaturesExtractor(torch.nn.Module):
    """
    Custom feature extractor for Gen 3 Pokémon battles.
    Uses a dynamic layout provided by the Observation Encoder to avoid magic constants.
    Supports shared latent embeddings for Species, Moves, Items, Abilities, and Types.
    """
    def __init__(self, observation_space: spaces.Dict, layout: Optional[Dict[str, Any]] = None, mappings: Optional[Dict[str, Any]] = None, log_level: LogLevel = LogLevel.QUIET):
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
        # Input: move_emb + type_emb + remnants + known(1) + context + matchups(TEAM_SIZE) + validity(1)
        # Remnants: power, secondary, recoil, category, current_pp, max_pp (known extracted separately)
        # Context: HP(1) + turn(1) + weather + fainted + spikes
        _msl = layout['pokemon']['moves']['layout']['slot_layout']
        _rem1 = _msl['type']['offset'] - _msl['power']['offset']              # power+secondary+recoil = 3
        _rem2 = _msl['known']['offset'] - (_msl['type']['offset'] + _msl['type']['dim'])  # category = 1
        _rem3 = (_msl['max_pp']['offset'] + _msl['max_pp']['dim']) - _msl['current_pp']['offset']  # pp = 2
        self.move_remnant_dim = _rem1 + _rem2 + _rem3
        _gl = layout.get('global_layout', {})
        _rl = layout.get('reactive_layout', {})
        _move_ctx_dim = (1 + 1                                      # hp + turn
                         + _gl.get('weather', {}).get('dim', 6)     # weather
                         + _rl.get('fainted', {}).get('dim', 2)     # fainted
                         + _gl.get('hazards', {}).get('dim', 2))    # spikes
        move_input_dim = (layout['move_embedding_dim'] + layout['type_embedding_dim']
                          + self.move_remnant_dim + 1               # remnants + known
                          + _move_ctx_dim                           # context
                          + TEAM_SIZE + 1)                          # matchups + validity
        self.move_network = torch.nn.Sequential(
            torch.nn.Linear(move_input_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32)
        )
        
        # 1.6 Pokémon Role Encoder
        # pokemon_enriched (241) + global_context broadcast (16) + switch_valid (1) + struggle_from_prev (1) = 259
        # pokemon_enriched: species(32) + stats(6) + item_emb(16) + item_known(1) +
        #   pk_types(32) + ability_emb(16) + ability_known(1) + condition(7) + moves(128) + hp+active(2)
        # global_context: turn(1) + weather(6) + fainted(2) + spikes(2) + struggle(1) + screens(4) = 16
        # switch_valid: was this slot a valid switch target in the previous turn's mask (1)
        # struggle_from_prev: was struggle the only option last turn (1)
        self.role_token_size = 128
        role_input_dim = 259
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
        # Path 4: Threat — their_team queries our_active (which of their bench threatens us most?)
        self.threat_attn = torch.nn.MultiheadAttention(embed_dim=self.role_token_size, num_heads=4, batch_first=True)

        # 1.7.3 LayerNorm for Residual Stability
        self.norm1 = torch.nn.LayerNorm(self.role_token_size)
        self.norm2 = torch.nn.LayerNorm(self.role_token_size)
        self.norm3 = torch.nn.LayerNorm(self.role_token_size)
        self.norm4 = torch.nn.LayerNorm(self.role_token_size)
        
        # 1.8 Active Context Encoder
        # Encodes boosts, volatiles, and other per-active-mon context (22 dims each side)
        # into a dense 32-dim token. Shared weights between our side and opponent side.
        active_ctx_dim = layout.get('active_context_dim', 22)
        self.active_ctx_encoder = torch.nn.Sequential(
            torch.nn.Linear(active_ctx_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
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
            # Pass the full obs (including prev_mask + TurnDelta tail) so describe_vector
            # can delegate to turn_delta_encoder.describe_vector for the tail block.
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

            # --- TURN DELTA ---
            td = desc.get("turn_delta")
            if td is not None:
                def _action_str(switched, failed, cant, move):
                    if switched: return "SWITCH"
                    if failed: return f"FAIL({cant or '?'})"
                    if move: return f"move pwr={move['power']} sec={int(move['secondary'])} recoil={int(move['recoil'])}"
                    return "none (first turn)"
                print(f"\n--- LAST TURN (TurnDelta) ---")
                print(f"  Us:  {_action_str(td['our_switched'], td['our_failed'], td['our_cant'], td['our_move'])}  hp_delta={td['our_hp_delta']:+.2f}  fainted={int(td['we_fainted'])}")
                print(f"  Opp: {_action_str(td['opp_switched'], td['opp_failed'], td['opp_cant'], td['opp_move'])}  hp_delta={td['opp_hp_delta']:+.2f}  fainted={int(td['opp_fainted'])}  known={int(td['opp_move_known'])}")

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
        base_dim = self.layout['base_dim']

        # Layouts (derived from observation encoder; avoid hardcoded offsets below)
        reactive_layout = self.layout.get('reactive_layout', {})
        global_layout   = self.layout.get('global_layout', {})
        moves_info      = self.layout['pokemon']['moves']
        moves_layout    = moves_info['layout']
        m_slot_layout   = moves_layout['slot_layout']
        num_moves       = len(moves_layout['slots'])

        # Extract prev_mask (11 dims) and TurnDelta block (29 dims) from observation tail
        prev_mask = x[:, base_dim : base_dim + 11]                               # [B, 11]
        turn_delta_feat = x[:, base_dim + 11 : base_dim + 11 + TURN_DELTA_DIM]  # [B, 29]
        switch_mask  = prev_mask[:, 0:TEAM_SIZE]                                 # [B, 6]
        move_mask    = prev_mask[:, TEAM_SIZE:TEAM_SIZE + num_moves]             # [B, 4]
        struggle_mask = prev_mask[:, TEAM_SIZE + num_moves:TEAM_SIZE + num_moves + 1]  # [B, 1]

        # 1. Extract parts using dynamic layout (all offsets are within the base 1021-dim obs)
        parts = self.layout['parts']
        ot = parts['our_team']
        our_team = x[:, ot['start']:ot['end']].reshape(batch_size, *ot['reshape'])
        opt = parts['opp_team']
        opp_team = x[:, opt['start']:opt['end']].reshape(batch_size, *opt['reshape'])
        ctx = parts['context']
        remaining_part = x[:, ctx['start']:base_dim]  # stop before prev_mask tail

        # Reactive layout offsets — extracted here so all downstream code can reference them
        reactive_layout = self.layout.get('reactive_layout', {})
        struggle_offset = reactive_layout.get('forced_struggle', {}).get('offset', 11)
        matchup_offset  = reactive_layout.get('our_matchups', {}).get('offset', 12)

        # 1.1 Context Features for Move Selection
        global_start = parts['global']['start'] - ctx['start']
        _w  = global_layout.get('weather', {});  _w_off, _w_dim  = _w.get('offset', 0), _w.get('dim', 6)
        _hz = global_layout.get('hazards', {});  _hz_off, _hz_dim = _hz.get('offset', 6), _hz.get('dim', 2)
        _ck = global_layout.get('clock',   {});  _ck_off, _ck_dim = _ck.get('offset', 8), _ck.get('dim', 1)
        weather_feature = remaining_part[:, global_start + _w_off  : global_start + _w_off  + _w_dim]   # [B, 6]
        spikes_feature  = remaining_part[:, global_start + _hz_off : global_start + _hz_off + _hz_dim]  # [B, 2]
        turn_feature    = remaining_part[:, global_start + _ck_off : global_start + _ck_off + _ck_dim]  # [B, 1]
        
        reactive_start = parts['reactive']['start'] - ctx['start']
        _f   = reactive_layout.get('fainted', {});      _f_off, _f_dim  = _f.get('offset', 8),   _f.get('dim', 2)
        _om  = reactive_layout.get('our_matchups', {}); _om_off, _om_dim = _om.get('offset', 16), _om.get('dim', 144)
        _tm  = reactive_layout.get('their_matchups', {}); _tm_off, _tm_dim = _tm.get('offset', 160), _tm.get('dim', 144)
        fainted_feature     = remaining_part[:, reactive_start + _f_off  : reactive_start + _f_off  + _f_dim]   # [B, 2]
        our_matchups_flat   = remaining_part[:, reactive_start + _om_off : reactive_start + _om_off + _om_dim]  # [B, 144]
        their_matchups_flat = remaining_part[:, reactive_start + _tm_off : reactive_start + _tm_off + _tm_dim]  # [B, 144]

        our_matchups   = our_matchups_flat.reshape(batch_size, TEAM_SIZE, num_moves, TEAM_SIZE)
        their_matchups = their_matchups_flat.reshape(batch_size, TEAM_SIZE, num_moves, TEAM_SIZE)
        matchups_all = torch.cat([our_matchups, their_matchups], dim=1) # [B, n_poke, num_moves, TEAM_SIZE]
        
        pokemon_part = torch.cat([our_team, opp_team], dim=1) # [B, 2 * TEAM_SIZE, POKEMON_FULL_DIM]
        
        # 2. Extract IDs for embedding
        pk_layout = self.layout['pokemon']
        
        # Species ID
        species_info = pk_layout['species']
        species_idx = species_info['offset'] + species_info['layout']['species_id']['offset']
        species_ids = pokemon_part[:, :, species_idx].long()
        
        # Move IDs & Move Type IDs
        moves_offset = moves_info['offset']
        _type_off = m_slot_layout['type']['offset']
        move_id_tensors = []
        move_type_id_tensors = []
        for i in range(num_moves):
            slot_idx = moves_offset + moves_layout['slots'][i]['offset']
            move_id_tensors.append(pokemon_part[:, :, slot_idx].long().unsqueeze(2))
            move_type_id_tensors.append(pokemon_part[:, :, slot_idx + _type_off].long().unsqueeze(2))
            
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
        
        # 3. Embed Everything
        embedded_species = self.species_embedding(species_ids) # [B, 2 * TEAM_SIZE, 32]
        
        embedded_moves = self.move_embedding(all_move_ids) # [B, 12, 4, 16]
        embedded_move_types = self.type_embedding(all_move_type_ids) # [B, 12, 4, 16]
        embedded_items = self.item_embedding(item_ids) # [B, 12, 16]
        embedded_abilities = self.ability_embedding(ability_ids) # [B, 12, 16]
        
        # Pokémon Types: concatenate E1 and E2 (32 dims) — sum loses type-pair signal
        embedded_t1 = self.type_embedding(type1_ids) # [B, 12, 16]
        embedded_t2 = self.type_embedding(type2_ids) # [B, 12, 16]
        embedded_pk_types = torch.cat([embedded_t1, embedded_t2], dim=-1) # [B, 12, 32]
        
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
        
        # Part E: Move remnants (power, secondary, recoil, category, current_pp, max_pp)
        _known_off = m_slot_layout['known']['offset']
        move_remnants = []
        for i in range(num_moves):
            slot_start = moves_offset + moves_layout['slots'][i]['offset']
            # power, secondary, recoil  [3 dims]
            move_remnants.append(pokemon_part[:, :, slot_start + m_slot_layout['power']['offset'] : slot_start + m_slot_layout['type']['offset']])
            # category  [1 dim]
            move_remnants.append(pokemon_part[:, :, slot_start + m_slot_layout['type']['offset'] + m_slot_layout['type']['dim'] : slot_start + _known_off])
            # current_pp, max_pp  [2 dims]
            move_remnants.append(pokemon_part[:, :, slot_start + m_slot_layout['current_pp']['offset'] : slot_start + m_slot_layout['max_pp']['offset'] + m_slot_layout['max_pp']['dim']])
        all_move_remnants = torch.cat(move_remnants, dim=2) # [B, 2*TEAM_SIZE, num_moves * move_remnant_dim]

        # Part F: Move Known Flags (interleaved in slots)
        known_flags_tensors = []
        for i in range(num_moves):
            slot_start = moves_offset + moves_layout['slots'][i]['offset']
            known_flags_tensors.append(pokemon_part[:, :, slot_start + _known_off : slot_start + _known_off + 1])
        known_flags = torch.cat(known_flags_tensors, dim=2) # [B, 2*TEAM_SIZE, num_moves]
        
        # Part G: HP and Active Flag
        hp_offset = pk_layout['hp']['offset']
        # Extract from HP offset to the end of the Pokémon vector (includes Active Flag)
        hp_and_active = pokemon_part[:, :, hp_offset:] 
        
        # --- SHARED MOVE PROCESSING (Step 1) ---
        # Reshape move remnants and flags to align with per-slot dims
        n_poke = 2 * TEAM_SIZE
        move_remnants_reshaped = all_move_remnants.reshape(batch_size, n_poke, num_moves, self.move_remnant_dim)
        known_flags_reshaped   = known_flags.reshape(batch_size, n_poke, num_moves, 1)
        
        # Combine all move features into [B, 12, 4, 51]
        # Context: HP (1), Turn (1), Weather (6), Fainted (2), Spikes (2)
        hp_feature = hp_and_active[:, :, 0:1]                                             # [B, n_poke, 1]
        turn_expanded    = turn_feature.unsqueeze(1).expand(-1, n_poke, -1)               # [B, n_poke, 1]
        weather_expanded = weather_feature.unsqueeze(1).expand(-1, n_poke, -1)            # [B, n_poke, 6]
        fainted_expanded = fainted_feature.unsqueeze(1).expand(-1, n_poke, -1)            # [B, n_poke, 2]
        spikes_expanded  = spikes_feature.unsqueeze(1).expand(-1, n_poke, -1)             # [B, n_poke, 2]

        move_context = torch.cat([hp_feature, turn_expanded, weather_expanded, fainted_expanded, spikes_expanded], dim=2)
        move_context_final = move_context.unsqueeze(2).expand(-1, -1, num_moves, -1)      # [B, n_poke, num_moves, ctx_dim]

        # Move validity from prev_mask: our moves get their validity bit, opp gets all-ones
        move_validity_ours = move_mask.unsqueeze(1).unsqueeze(3).expand(-1, TEAM_SIZE, -1, -1).float()  # [B, 6, num_moves, 1]
        move_validity_opp  = torch.ones(batch_size, TEAM_SIZE, num_moves, 1, device=x.device)
        move_validity = torch.cat([move_validity_ours, move_validity_opp], dim=1)                       # [B, n_poke, num_moves, 1]

        move_features = torch.cat([
            embedded_moves,
            embedded_move_types,
            move_remnants_reshaped,
            known_flags_reshaped,
            move_context_final,
            matchups_all,
            move_validity,
        ], dim=3)
        
        # Process through shared network
        processed_moves = self.move_network(move_features.reshape(-1, move_features.shape[-1]))
        processed_moves = processed_moves.reshape(batch_size, n_poke, -1)  # [B, n_poke, num_moves * 32]
        
        pokemon_enriched = torch.cat([
            embedded_species,      # 32
            part_a,                # 6
            embedded_items,        # 16
            item_remnant,          # 1
            embedded_pk_types,     # 32 (concatenated type pair, was 16)
            embedded_abilities,    # 16
            ability_remnant,       # 1
            part_d,                # 7 (condition one-hot, unused slot removed)
            processed_moves,       # 128 (Shared Processor Output)
            hp_and_active          # 2
        ], dim=2) # [B, 12, 241]
        
        # --- POKÉMON ROLE ENCODER ---
        _fs = reactive_layout.get('forced_struggle', {}); struggle_offset = _fs.get('offset', 11)
        struggle_feature = remaining_part[:, reactive_start + struggle_offset : reactive_start + struggle_offset + 1] # [B, 1]
        # S3: screens (Reflect/Light Screen both sides) are critical switch-safety context
        _sc = global_layout.get('screens', {}); _sc_off, _sc_dim = _sc.get('offset', 9), _sc.get('dim', 4)
        screen_feature = remaining_part[:, global_start + _sc_off : global_start + _sc_off + _sc_dim] # [B, 4]
        global_context = torch.cat([turn_feature, weather_feature, fainted_feature, spikes_feature, struggle_feature, screen_feature], dim=1) # [B, 16]
        context_broadcasted = global_context.unsqueeze(1).expand(-1, n_poke, -1)

        switch_validity_ours = switch_mask.unsqueeze(2)                                          # [B, 6, 1]
        switch_validity_opp  = torch.ones(batch_size, TEAM_SIZE, 1, device=x.device)
        switch_validity = torch.cat([switch_validity_ours, switch_validity_opp], dim=1)         # [B, n_poke, 1]

        struggle_from_prev = struggle_mask.unsqueeze(1).expand(-1, n_poke, -1)                  # [B, n_poke, 1]

        pokemon_enriched_with_context = torch.cat([
            pokemon_enriched, context_broadcasted, switch_validity, struggle_from_prev
        ], dim=2)  # [B, 12, 259]

        role_tokens = self.role_encoder(pokemon_enriched_with_context.reshape(-1, pokemon_enriched_with_context.shape[-1]))
        role_tokens = role_tokens.reshape(batch_size, n_poke, self.role_token_size)

        # --- TEAM-WIDE ATTENTION ---
        # 1. Find who is active on both teams
        active_flags = hp_and_active[:, :, 1] # [B, 12]
        # QW3: guard argmax — if no active flag is set (opponent not revealed, or dummy forward),
        # argmax silently returns 0; clamp to valid range explicitly.
        our_active_flags = active_flags[:, 0:TEAM_SIZE]
        opp_active_flags = active_flags[:, TEAM_SIZE:2*TEAM_SIZE]
        our_active_idx = torch.where(
            our_active_flags.any(dim=1),
            torch.argmax(our_active_flags, dim=1),
            torch.zeros(batch_size, dtype=torch.long, device=x.device)
        )  # [B] — 0-5
        opp_active_idx = torch.where(
            opp_active_flags.any(dim=1),
            torch.argmax(opp_active_flags, dim=1),
            torch.zeros(batch_size, dtype=torch.long, device=x.device)
        ) + TEAM_SIZE  # [B] — 6-11

        # 2. Apply Status Biases (Pre-Attention)
        status_idx = torch.ones((batch_size, n_poke), device=role_tokens.device).long()
        status_idx[:, 0:TEAM_SIZE] = 1
        status_idx[:, TEAM_SIZE:2*TEAM_SIZE] = 3
        status_idx[torch.arange(batch_size), our_active_idx] = 0
        status_idx[torch.arange(batch_size), opp_active_idx] = 2

        role_tokens = role_tokens + self.status_embedding(status_idx)

        # 3. Split Teams for Attention Paths
        our_team = role_tokens[:, 0:TEAM_SIZE, :]           # [B, 6, 128]
        their_team = role_tokens[:, TEAM_SIZE:2*TEAM_SIZE, :]  # [B, 6, 128]
        our_active_pre = role_tokens[torch.arange(batch_size), our_active_idx].unsqueeze(1)   # [B, 1, 128]
        their_active = role_tokens[torch.arange(batch_size), opp_active_idx].unsqueeze(1)     # [B, 1, 128]

        # --- ATTENTION PATHS WITH RESIDUALS ---

        # Path 1: Pressure — our_active learns what threatens it from their bench
        pressure_delta, _ = self.pressure_attn(our_active_pre, their_team, their_team)
        our_active_post_pressure = self.norm1(our_active_pre + pressure_delta)  # [B, 1, 128]

        # Write Pressure result back into our_team so Safety/Synergy see the updated active slot
        our_team = our_team.clone()
        our_team[torch.arange(batch_size), our_active_idx] = our_active_post_pressure.squeeze(1)

        # Path 2: Safety — our bench learns which opponent active threatens each slot
        safety_delta, _ = self.safety_attn(our_team, their_active, their_active)
        our_team = self.norm2(our_team + safety_delta)

        # Path 3: Synergy — our team learns internal role cohesion
        # QW1: mask fainted slots as keys so dead Pokémon don't pollute the key space.
        # Always unmask the active slot — if all keys were masked softmax produces NaN.
        fainted_mask_ours = (hp_and_active[:, 0:TEAM_SIZE, 0] == 0)  # [B, 6] True = fainted
        fainted_mask_ours[torch.arange(batch_size), our_active_idx] = False
        synergy_delta, _ = self.synergy_attn(our_team, our_team, our_team,
                                              key_padding_mask=fainted_mask_ours)
        our_team = self.norm3(our_team + synergy_delta)

        # Path 4: Threat — their bench learns which of them threatens our active most (P2)
        threat_delta, _ = self.threat_attn(their_team, our_active_post_pressure, our_active_post_pressure)
        their_team = self.norm4(their_team + threat_delta)

        # --- FINAL AGGREGATION ---
        our_team_flat = our_team.reshape(batch_size, -1)       # [B, 768]
        their_team_flat = their_team.reshape(batch_size, -1)   # [B, 768]

        # P4: extract our_active_refined from our_team after ALL paths (not Pressure-only)
        our_active_refined = our_team[torch.arange(batch_size), our_active_idx]  # [B, 128]

        # P3: encode active_context (boosts, volatiles) through a learned MLP
        active_ctx_dim = self.layout.get('active_context_dim', 22)
        our_ctx_raw = remaining_part[:, 0 : active_ctx_dim]           # [B, 22]
        opp_ctx_raw = remaining_part[:, active_ctx_dim : 2 * active_ctx_dim]  # [B, 22]
        our_ctx_enc = self.active_ctx_encoder(our_ctx_raw)             # [B, 32]
        opp_ctx_enc = self.active_ctx_encoder(opp_ctx_raw)             # [B, 32]

        # P1: strip matchup matrix from projection input (already encoded in role tokens)
        # Keep: active_ctx replaced by encoded tokens + global env + reactive scalars (before matchups)
        non_matchup_rest = remaining_part[:, 2 * active_ctx_dim : reactive_start + matchup_offset]  # global + scalars

        combined = torch.cat([
            our_team_flat, their_team_flat, our_active_refined,
            our_ctx_enc, opp_ctx_enc,
            non_matchup_rest,
            turn_delta_feat,
        ], dim=1)
        return combined

    def forward(self, obs):
        combined = self.forward_internal(obs)
        
        # Diagnostic Trace logic
        if self.log_level >= LogLevel.PERIODIC and self.trace_logger.should_log():
            x = obs["observation"]
            x_base = x[:, :self.layout['base_dim']]
            parts = self.layout['parts']
            ot = parts['our_team']
            our_team = x_base[:, ot['start']:ot['end']].reshape(x_base.shape[0], *ot['reshape'])
            opt = parts['opp_team']
            opp_team = x_base[:, opt['start']:opt['end']].reshape(x_base.shape[0], *opt['reshape'])
            pokemon_part = torch.cat([our_team, opp_team], dim=1)

            pk_layout = self.layout['pokemon']
            species_info = pk_layout['species']
            species_idx = species_info['offset'] + species_info['layout']['species_id']['offset']
            species_ids = pokemon_part[:, :, species_idx].long()

            # Pass the full obs so describe_vector can delegate TurnDelta to its encoder
            self._print_deep_trace(x, pokemon_part, species_ids)
        
        return self.activation(self.projection(combined))
