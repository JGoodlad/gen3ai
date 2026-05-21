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
from agents.observation.turn_delta_encoder import TURN_DELTA_DIM, EFF_DIM, ORDER_DIM
from utils.logging.rate_limiter import RateLimitedLogger
from utils.logging.levels import LogLevel

# Strategic TurnDelta slice: always the tail of the TurnDelta block (effectiveness + order).
TD_STRATEGIC_DIM    = EFF_DIM * 2 + ORDER_DIM      # 10: our_eff(4) + opp_eff(4) + order(2)
TD_STRATEGIC_OFFSET = TURN_DELTA_DIM - TD_STRATEGIC_DIM  # 29

# Architecture constants — single source of truth.
# ModelVersion imports these so model_config.json always reflects the live values.
# When you change any of these, also bump MODEL_CONFIG_VERSION in model_version.py.
ROLE_TOKEN_SIZE = 128
PROJECTION_DIM = 512
MOVE_NET_HIDDEN = [64, 32]        # [hidden, output] of shared move processor
ROLE_ENCODER_HIDDEN = [256, 128]  # [hidden, output] of per-Pokémon role encoder
ACTIVE_CTX_HIDDEN = [64, 32]      # [hidden, output] of active context encoder
NET_ARCH = [512, 512]             # MLP policy layers (SB3 policy_kwargs["net_arch"])


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
            torch.nn.Linear(move_input_dim, MOVE_NET_HIDDEN[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(MOVE_NET_HIDDEN[0], MOVE_NET_HIDDEN[1])
        )

        # 1.6 Pokémon Role Encoder
        # pokemon_enriched (244) + global_context broadcast (16) + switch_valid (1) + struggle_from_prev (1) = 262
        # pokemon_enriched: species(32) + stats(6) + item_emb(16) + item_known(1) +
        #   pk_types(32) + ability_emb(16) + ability_known(1) + condition(7) + moves(128) + hp+species_known+active+counters(5)
        # global_context: turn(1) + weather(6) + fainted(2) + spikes(2) + struggle(1) + screens(4) = 16
        # switch_valid: was this slot a valid switch target in the previous turn's mask (1)
        # struggle_from_prev: was struggle the only option last turn (1)
        self.role_token_size = ROLE_TOKEN_SIZE
        role_input_dim = 262
        self.role_encoder = torch.nn.Sequential(
            torch.nn.Linear(role_input_dim, ROLE_ENCODER_HIDDEN[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(ROLE_ENCODER_HIDDEN[0], ROLE_ENCODER_HIDDEN[1])
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

        # N3 — Opponent synergy: their_team attends to itself so the opponent's
        # bench can learn internal role cohesion (mirrors our Path 3).
        self.opp_synergy_attn = torch.nn.MultiheadAttention(embed_dim=self.role_token_size, num_heads=4, batch_first=True)
        self.norm_opp_synergy = torch.nn.LayerNorm(self.role_token_size)

        # N4 — Within-Pokémon move self-attention: before folding the 4 processed
        # move slots into the role encoder, let them attend to each other so the
        # role encoder can see "this mon has two physical attacks + a spread move."
        self.move_self_attn = torch.nn.MultiheadAttention(embed_dim=32, num_heads=2, batch_first=True)
        self.move_self_norm = torch.nn.LayerNorm(32)

        # 1.7.4 Attention pool — replaces the slot-order team flatten with a
        # permutation-equivariant pooled token per team. A single learned query
        # attends over the 6 role tokens (with fainted slots key-masked) and
        # produces one 128-dim pooled vector per side.
        self.our_pool_query = torch.nn.Parameter(torch.randn(1, 1, self.role_token_size) * 0.02)
        self.their_pool_query = torch.nn.Parameter(torch.randn(1, 1, self.role_token_size) * 0.02)
        self.our_pool_attn = torch.nn.MultiheadAttention(embed_dim=self.role_token_size, num_heads=4, batch_first=True)
        self.their_pool_attn = torch.nn.MultiheadAttention(embed_dim=self.role_token_size, num_heads=4, batch_first=True)
        self.norm_pool_our = torch.nn.LayerNorm(self.role_token_size)
        self.norm_pool_their = torch.nn.LayerNorm(self.role_token_size)

        # 1.8 Active Context Encoder
        # Encodes boosts, volatiles, and other per-active-mon context (22 dims each side)
        # into a dense 32-dim token. Shared weights between our side and opponent side.
        active_ctx_dim = layout.get('active_context_dim', 22)
        self.active_ctx_encoder = torch.nn.Sequential(
            torch.nn.Linear(active_ctx_dim, ACTIVE_CTX_HIDDEN[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(ACTIVE_CTX_HIDDEN[0], ACTIVE_CTX_HIDDEN[1]),
        )

        # 1.9 Active Context → Role Token Injection
        # Projects active context (boosts/volatiles) into role token space so the 5
        # attention paths can see boost/volatile state. 2-layer MLP captures non-linear
        # interactions (e.g. "+2 SpA AND +2 Spe" is qualitatively different from either alone).
        # Shared weights between our side and opponent side. Added additively to the active
        # slot only — bench mons have no boosts/volatiles in Gen 3.
        self.active_ctx_to_role = torch.nn.Sequential(
            torch.nn.Linear(active_ctx_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, ROLE_TOKEN_SIZE),
        )

        # 1.10 TurnDelta Conditioner
        # Maps the 10-dim strategic TurnDelta slice (effectiveness + order) into role
        # token space. Applied additively to the active slots before all 5 attention paths,
        # so the attention mechanism can reason about last-turn matchup quality and speed.
        # Shared MLP — our and opp active tokens use perspective-flipped inputs.
        self.td_conditioner = torch.nn.Sequential(
            torch.nn.Linear(TD_STRATEGIC_DIM, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, ROLE_TOKEN_SIZE),
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
        # Pre-projection LayerNorm equalises the very different per-block scales
        # in `combined` (embedding outputs, 0/1 validity bits, HP fractions,
        # ±1 TurnDelta deltas) before they hit a single Linear.
        self.pre_proj_norm = torch.nn.LayerNorm(self.projection_input_dim)
        self.projection_dim = PROJECTION_DIM
        self.projection = torch.nn.Linear(self.projection_input_dim, self.projection_dim)
        self.activation = torch.nn.ReLU()
        self.features_dim = self.projection_dim
        self.trace_logger = RateLimitedLogger(interval_seconds=30)
        self._trace_buffer: list = []
        self._trace_collect_remaining: int = 0
        
    def _print_one_turn(self, obs_np: np.ndarray, label: str):
        """Print one turn's decoded state from a full 1093-dim obs array."""
        desc = self._encoder.describe_vector(obs_np)
        world = desc.get('world', {})
        NAME_W, ACTV_W, TYPE_W = 12, 6, 18
        TAB = "    "

        print(f"\n{'─' * 60}")
        print(f"  {label}   |  Turn: {world.get('turn', '?')} | Weather: {world.get('weather', 'NONE')} | Spikes: {world.get('our_spikes', 0)}/{world.get('opp_spikes', 0)}")
        print(f"{'─' * 60}")

        ctx = desc.get('our_active', {})
        print(f"Active ctx — Boosts: {ctx.get('boosts', {})} | Volatiles: {ctx.get('volatiles', [])}")

        print("\n--- TEAMS ---")
        for i, mon in enumerate(desc['our_team']):
            active_str = "[actv]" if mon.get('active') else "      "
            s = mon['stats']
            print(f"[OUR {i}] {mon['species'].lower():{NAME_W}} {active_str:{ACTV_W}}  {mon['types'].lower():{TYPE_W}}  hp: {mon['hp']:>6}  status: {mon['status'].lower():7}  {s['hp']}/{s['atk']}/{s['def']}/{s['spa']}/{s['spd']}/{s['spe']}")
            print(f"{TAB}item: {mon['item'].lower():17}  ably: {mon['ability'].lower():16}  moves: {mon.get('moves', [])}")
        print("-" * 30)
        for i, mon in enumerate(desc['opp_team']):
            active_str = "[actv]" if mon.get('active') else "      "
            s = mon['stats']
            print(f"[OPP {i}] {mon['species'].lower():{NAME_W}} {active_str:{ACTV_W}}  {mon['types'].lower():{TYPE_W}}  hp: {mon['hp']:>6}  status: {mon['status'].lower():7}  {s['hp']}/{s['atk']}/{s['def']}/{s['spa']}/{s['spd']}/{s['spe']}")
            print(f"{TAB}item: {mon['item'].lower():17}  ably: {mon['ability'].lower():16}  moves: {mon.get('moves', [])}")

        momentum = desc.get('momentum', {})
        print(f"\nFainted: {momentum.get('fainted_our', 0)} (Us) / {momentum.get('fainted_opp', 0)} (Them)")

        td = desc.get("turn_delta")
        if td is not None:
            def _action_str(switched, failed, cant, move):
                move_str = None
                if move and move.get("move_id", 0) > 0:
                    name = move.get("move_name") or f"#{move['move_id']}"
                    type_ = (move.get("move_type") or "").title()
                    pwr = move["power"]
                    meta = []
                    if type_: meta.append(type_)
                    if pwr > 0: meta.append(f"{pwr}bp")
                    if move["secondary"]: meta.append("+eff")
                    if move["recoil"]: meta.append("recoil")
                    suffix = f" [{', '.join(meta)}]" if meta else ""
                    move_str = f"{name}{suffix}"
                if switched:
                    # Phaze: they moved first, then were forced out
                    return f"{move_str} → phazed" if move_str else "switch"
                if failed:
                    return f"✗ {cant or '?'}"
                if move_str:
                    return move_str
                return "(first turn)"

            W = 36
            opp_known = "" if td["opp_move_known"] else "  [unconfirmed]"
            faint_parts = (["us"] if td["we_fainted"] else []) + (["opp"] if td["opp_fainted"] else [])
            faint_str = f"   💀 fainted: {'/'.join(faint_parts)}" if faint_parts else ""
            print(f"--- Last Turn ---")
            print(f"  Us:  {_action_str(td['our_switched'], td['our_failed'], td['our_cant'], td['our_move']):{W}}")
            print(f"  Opp: {_action_str(td['opp_switched'], td['opp_failed'], td['opp_cant'], td['opp_move']) + opp_known:{W}}")
            print(f"  ΔHP  us={td['our_hp_delta']:+.2f}  opp={td['opp_hp_delta']:+.2f}{faint_str}")

        warnings, is_critical = self._encoder.integrity_check(obs_np)
        if warnings:
            print("\n⚠️ [INTEGRITY CHECK WARNINGS]")
            for w in warnings:
                print(f"  - {w}")
        if is_critical:
            raise ValueError(f"CRITICAL INTEGRITY FAILURE: {warnings}")

    def _print_deep_trace(self):
        """Print the last 3 buffered turns as a succession."""
        if self._encoder is None and self.mappings:
            self._encoder = Gen3ObservationEncoder(self.mappings)
        if not self._encoder or not self._trace_buffer:
            return

        n = len(self._trace_buffer)
        print("\n" + "🧬" * 30)
        print(f"🧬 [DEEP TRACE — {n} turns ending at {time.strftime('%H:%M:%S')}]")
        print("=" * 60)
        labels = [f"turn -2 (oldest)", "turn -1", "turn 0 (current)"][-n:]
        for obs_t, label in zip(self._trace_buffer, labels):
            self._print_one_turn(obs_t.cpu().numpy(), label)
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
        turn_delta_feat = x[:, base_dim + 11 : base_dim + 11 + TURN_DELTA_DIM]  # [B, TURN_DELTA_DIM]
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
        processed_moves = processed_moves.reshape(batch_size, n_poke, num_moves, 32)  # [B, n_poke, 4, 32]

        # N4 — Within-Pokémon move self-attention: let the 4 move slots attend to
        # each other so the role encoder sees "2 physical attackers + coverage move."
        mv_in = processed_moves.reshape(batch_size * n_poke, num_moves, 32)
        mv_delta, _ = self.move_self_attn(mv_in, mv_in, mv_in)
        mv_out = self.move_self_norm(mv_in + mv_delta)
        processed_moves = mv_out.reshape(batch_size, n_poke, -1)  # [B, n_poke, num_moves * 32]
        
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
            hp_and_active          # 3 (hp, species_known, active)
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

        # Extract active context raw vectors here so they are available for pre-attention
        # injection below and for the direct encoder call later in the function.
        active_ctx_dim = self.layout.get('active_context_dim', 22)
        our_ctx_raw = remaining_part[:, 0 : active_ctx_dim]                   # [B, 23]
        opp_ctx_raw = remaining_part[:, active_ctx_dim : 2 * active_ctx_dim]  # [B, 23]

        # --- TEAM-WIDE ATTENTION ---
        # 1. Find who is active on both teams
        active_flags = hp_and_active[:, :, 2] # [B, 12] — active flag is now at index 2 (hp, species_known, active)
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

        # 1.5 Inject active context (boosts/volatiles) into each active slot's role token
        # before attention runs. All 5 attention paths can now see boost/volatile state.
        # Bench slots are not injected — they have no boosts/volatiles in Gen 3.
        our_ctx_bias = self.active_ctx_to_role(our_ctx_raw)   # [B, 128]
        opp_ctx_bias = self.active_ctx_to_role(opp_ctx_raw)   # [B, 128]
        batch_idx = torch.arange(batch_size, device=x.device)
        role_tokens[batch_idx, our_active_idx] = role_tokens[batch_idx, our_active_idx] + our_ctx_bias
        role_tokens[batch_idx, opp_active_idx] = role_tokens[batch_idx, opp_active_idx] + opp_ctx_bias

        # 1.10 Inject TurnDelta strategic signals (effectiveness + order) into active role
        # tokens before team attention runs. Perspective is flipped for the opp side so the
        # shared MLP sees consistent [attacker_eff, defender_eff, we_first, opp_first] framing.
        td_strategic = turn_delta_feat[:, TD_STRATEGIC_OFFSET:]  # [B, 10]
        our_td_bias = self.td_conditioner(td_strategic)           # [B, 128]
        opp_td_input = torch.cat([
            td_strategic[:, 4:8],   # opp_eff → "our eff" from opp's view
            td_strategic[:, 0:4],   # our_eff → "opp eff" from opp's view
            td_strategic[:, 9:10],  # opp_first → "we first" from opp's view
            td_strategic[:, 8:9],   # we_first  → "opp first" from opp's view
        ], dim=1)                                                  # [B, 10]
        opp_td_bias = self.td_conditioner(opp_td_input)           # [B, 128]
        role_tokens[batch_idx, our_active_idx] += our_td_bias
        role_tokens[batch_idx, opp_active_idx] += opp_td_bias

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

        # Fainted masks — True = fainted slot. Always unmask the active slot on
        # each side: if every key gets masked, softmax produces NaN; if every
        # query gets masked, the slot's delta multiplier collapses to zero
        # (harmless but redundant).
        fainted_mask_ours = (hp_and_active[:, 0:TEAM_SIZE, 0] == 0)            # [B, 6]
        fainted_mask_ours[torch.arange(batch_size), our_active_idx] = False
        fainted_mask_opp_local = (hp_and_active[:, TEAM_SIZE:2*TEAM_SIZE, 0] == 0)  # [B, 6]
        # opp_active_idx is in [6, 11]; convert to local [0, 5] for the opp-team mask
        opp_active_local = opp_active_idx - TEAM_SIZE
        fainted_mask_opp_local[torch.arange(batch_size), opp_active_local] = False
        live_ours = (~fainted_mask_ours).float().unsqueeze(-1)                 # [B, 6, 1]
        live_opp  = (~fainted_mask_opp_local).float().unsqueeze(-1)            # [B, 6, 1]

        # Path 1: Pressure — our_active learns what threatens it from their bench
        pressure_delta, _ = self.pressure_attn(our_active_pre, their_team, their_team)
        our_active_post_pressure = self.norm1(our_active_pre + pressure_delta)  # [B, 1, 128]

        # Write Pressure result back into our_team so Safety/Synergy see the updated active slot
        our_team = our_team.clone()
        our_team[torch.arange(batch_size), our_active_idx] = our_active_post_pressure.squeeze(1)

        # Path 2: Safety — our bench learns which opponent active threatens each slot.
        # Zero the delta at fainted query rows so dead slots don't accumulate noise.
        safety_delta, _ = self.safety_attn(our_team, their_active, their_active)
        our_team = self.norm2(our_team + safety_delta * live_ours)

        # Path 3: Synergy — our team learns internal role cohesion.
        # Mask fainted slots as keys so dead Pokémon don't pollute the key space.
        synergy_delta, _ = self.synergy_attn(our_team, our_team, our_team,
                                              key_padding_mask=fainted_mask_ours)
        our_team = self.norm3(our_team + synergy_delta * live_ours)

        # Path 4: Threat — their bench learns which of them threatens our active most (P2).
        # Zero the delta at fainted opp-query rows for the same reason as Safety.
        threat_delta, _ = self.threat_attn(their_team, our_active_post_pressure, our_active_post_pressure)
        their_team = self.norm4(their_team + threat_delta * live_opp)

        # Path 5: Opp Synergy — their team attends to itself (mirrors our Path 3).
        # Fainted slots key-masked; output zeroed for fainted queries.
        opp_synergy_delta, _ = self.opp_synergy_attn(their_team, their_team, their_team,
                                                      key_padding_mask=fainted_mask_opp_local)
        their_team = self.norm_opp_synergy(their_team + opp_synergy_delta * live_opp)

        # --- FINAL AGGREGATION ---
        # N1: replace slot-order flatten with attention pooling (permutation-equivariant).
        # One learned query per team attends over the 6 role tokens; fainted slots are
        # key-masked so they cannot dominate the pool.
        our_pool_q   = self.our_pool_query.expand(batch_size, -1, -1)         # [B, 1, 128]
        their_pool_q = self.their_pool_query.expand(batch_size, -1, -1)
        our_pool_out, _   = self.our_pool_attn(our_pool_q, our_team, our_team,
                                                key_padding_mask=fainted_mask_ours)
        their_pool_out, _ = self.their_pool_attn(their_pool_q, their_team, their_team,
                                                  key_padding_mask=fainted_mask_opp_local)
        our_team_pooled   = self.norm_pool_our(our_pool_out).squeeze(1)        # [B, 128]
        their_team_pooled = self.norm_pool_their(their_pool_out).squeeze(1)    # [B, 128]

        # P4: extract our_active_refined from our_team after ALL paths (not Pressure-only)
        our_active_refined = our_team[torch.arange(batch_size), our_active_idx]  # [B, 128]

        # P3: encode active_context (boosts, volatiles) through a learned MLP for the
        # direct projection path. (our_ctx_raw / opp_ctx_raw extracted earlier.)
        our_ctx_enc = self.active_ctx_encoder(our_ctx_raw)             # [B, 32]
        opp_ctx_enc = self.active_ctx_encoder(opp_ctx_raw)             # [B, 32]

        # P1: strip matchup matrix from projection input (already encoded in role tokens)
        # Keep: active_ctx replaced by encoded tokens + global env + reactive scalars (before matchups)
        non_matchup_rest = remaining_part[:, 2 * active_ctx_dim : reactive_start + matchup_offset]  # global + scalars

        # N2 — embed TurnDelta move and type IDs through the shared embedding tables.
        # Positions 0/4 (our move id, our type id) and 5/9 (opp move id, opp type id)
        # carry raw int IDs (stored as float32); the 35 remaining positions are scalars.
        our_move_id   = turn_delta_feat[:, 0].long()
        our_type_id   = turn_delta_feat[:, 4].long()
        opp_move_id   = turn_delta_feat[:, 5].long()
        opp_type_id   = turn_delta_feat[:, 9].long()
        # Clamp to embedding range — at init (zeros obs), ids are 0 which is valid
        our_move_id   = our_move_id.clamp(0, self.move_embedding.num_embeddings - 1)
        our_type_id   = our_type_id.clamp(0, self.type_embedding.num_embeddings - 1)
        opp_move_id   = opp_move_id.clamp(0, self.move_embedding.num_embeddings - 1)
        opp_type_id   = opp_type_id.clamp(0, self.type_embedding.num_embeddings - 1)
        our_move_emb  = self.move_embedding(our_move_id)   # [B, 16]
        our_type_emb  = self.type_embedding(our_type_id)   # [B, 16]
        opp_move_emb  = self.move_embedding(opp_move_id)   # [B, 16]
        opp_type_emb  = self.type_embedding(opp_type_id)   # [B, 16]
        # Scalar remnants: power/sec/recoil for each side + all 19 event scalars
        td_scalars = torch.cat([
            turn_delta_feat[:, 1:4],    # our: power, secondary, recoil
            turn_delta_feat[:, 6:9],    # opp: power, secondary, recoil
            turn_delta_feat[:, 10:],    # event scalars: switched, failed, cant, hp_delta, fainted, move_known, eff, order
        ], dim=1)  # [B, 35]
        turn_delta_embedded = torch.cat([
            our_move_emb, our_type_emb, opp_move_emb, opp_type_emb, td_scalars
        ], dim=1)  # [B, 99]

        combined = torch.cat([
            our_team_pooled, their_team_pooled, our_active_refined,
            our_ctx_enc, opp_ctx_enc,
            non_matchup_rest,
            turn_delta_embedded,
        ], dim=1)
        return combined

    def forward(self, obs):
        combined = self.forward_internal(obs)
        
        # Diagnostic Trace logic — collect only 3 turns when the timer fires, free otherwise
        if self.log_level >= LogLevel.PERIODIC:
            if self.trace_logger.should_log():
                self._trace_buffer = []
                self._trace_collect_remaining = 3
            if self._trace_collect_remaining > 0:
                x = obs["observation"]
                self._trace_buffer.append(x[0].detach().clone())
                self._trace_collect_remaining -= 1
                if self._trace_collect_remaining == 0:
                    self._print_deep_trace()
        
        return self.activation(self.projection(self.pre_proj_norm(combined)))
