import torch
import numpy as np
from gymnasium import spaces
from typing import Dict, Any, Optional
from agents.observation.constants import (
    TRACE_INTERVAL,
    TEAM_SIZE,
    GLOBAL_ENV_DIM,
    POKEMON_FULL_DIM
)
from agents.observation.turn_delta_encoder import (
    TURN_DELTA_DIM,
    EFF_DIM,
    ORDER_DIM,
    OFFSET_OUR_ACTOR_SPECIES,
    OFFSET_OPP_ACTOR_SPECIES,
    OFFSET_OUR_TARGET_SPECIES,
    OFFSET_OPP_TARGET_SPECIES,
    OFFSET_OUR_SWITCH_TO_SPEC,
    OFFSET_OPP_SWITCH_TO_SPEC,
)
from agents.action.constants import ACTION_SPACE_SIZE
from utils.logging.levels import LogLevel

# Strategic TurnDelta slice: always the tail of the TurnDelta block (effectiveness + order).
# Kept exported because external tests reference these constants.
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
N_HISTORY_TURNS = 10              # number of consecutive TurnDeltas in the observation

# Unified transformer hyperparameters. d_model matches ROLE_TOKEN_SIZE so team
# role tokens enter the transformer without a projection step.
D_MODEL = ROLE_TOKEN_SIZE         # 128
TRANSFORMER_N_LAYERS = 2
TRANSFORMER_N_HEADS = 4
TRANSFORMER_FFN_DIM = 256

# Token group ids for the unified transformer's type embedding.
TOKEN_TYPE_OUR_TEAM = 0
TOKEN_TYPE_THEIR_TEAM = 1
TOKEN_TYPE_HISTORY = 2
TOKEN_TYPE_GLOBAL = 3
NUM_TOKEN_TYPES = 4


class Gen3FeaturesExtractor(torch.nn.Module):
    """
    Custom feature extractor for Gen 3 Pokémon battles.

    Architecture (ai_v4 — unified transformer):
      1. Per-Pokémon role tokens via the shared move processor + role encoder (unchanged from v3).
      2. A single L=2 transformer stack attends over 23 tokens:
           - 6 our-team role tokens
           - 6 their-team role tokens
           - N_HISTORY_TURNS history tokens (raw TurnDelta → embedded → projected to d_model)
           - 1 global token (active contexts + global env + reactive scalars → projected)
         Token type and (history-only) positional embeddings precede the transformer.
         Fainted team slots and empty history slots are key-padding-masked.
      3. Two learned CLS queries cross-attend over each side's 6 team output tokens to
         produce permutation-equivariant team pools.
      4. The projection input concatenates: our_pool, their_pool, our_active_out,
         active context encodings, and the non-matchup scalar remainder.
    """
    def __init__(self, observation_space: spaces.Dict, layout: Optional[Dict[str, Any]] = None, mappings: Optional[Dict[str, Any]] = None, log_level: LogLevel = LogLevel.QUIET):
        super().__init__()
        self.layout = layout
        self.mappings = mappings
        self.log_level = log_level

        # ------------------------------------------------------------------
        # 1. Embedding tables
        # ------------------------------------------------------------------
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

        # ------------------------------------------------------------------
        # 2. Shared Move Processor + within-Pokémon move self-attention
        # ------------------------------------------------------------------
        # Input: move_emb + type_emb + remnants + known(1) + context + matchups(TEAM_SIZE) + validity(1)
        # Remnants: power, secondary, recoil, category, current_pp, max_pp (known extracted separately)
        # Context: HP(1) + turn(1) + weather + fainted + spikes
        _msl = layout['pokemon']['moves']['layout']['slot_layout']
        _rem1 = _msl['type']['offset'] - _msl['power']['offset']              # power+secondary+recoil = 3
        _rem2 = _msl['known']['offset'] - (_msl['type']['offset'] + _msl['type']['dim'])  # category = 1
        _rem3 = (_msl['max_pp']['offset'] + _msl['max_pp']['dim']) - _msl['current_pp']['offset']  # pp = 2
        self.move_remnant_dim = _rem1 + _rem2 + _rem3
        _gl = layout['global_layout']
        _rl = layout['reactive_layout']
        _move_ctx_dim = (1 + 1                       # hp + turn
                         + _gl['weather']['dim']     # weather
                         + _rl['fainted']['dim']     # fainted
                         + _gl['hazards']['dim'])    # spikes
        move_input_dim = (layout['move_embedding_dim'] + layout['type_embedding_dim']
                          + self.move_remnant_dim + 1               # remnants + known
                          + _move_ctx_dim                           # context
                          + TEAM_SIZE + 1)                          # matchups + validity
        self.move_network = torch.nn.Sequential(
            torch.nn.Linear(move_input_dim, MOVE_NET_HIDDEN[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(MOVE_NET_HIDDEN[0], MOVE_NET_HIDDEN[1])
        )

        # Within-Pokémon move self-attention: lets the role encoder see
        # "this mon has 2 physical attackers + a coverage move."
        self.move_self_attn = torch.nn.MultiheadAttention(
            embed_dim=MOVE_NET_HIDDEN[1], num_heads=2, batch_first=True
        )
        self.move_self_norm = torch.nn.LayerNorm(MOVE_NET_HIDDEN[1])

        # ------------------------------------------------------------------
        # 3. Pokémon Role Encoder
        # ------------------------------------------------------------------
        # Input = pokemon_enriched + broadcasted global_context + switch_valid + struggle_from_prev.
        # Computed dynamically so a change to any embedding dim or component automatically propagates.
        _pk_layout = layout['pokemon']
        _num_moves = len(_pk_layout['moves']['layout']['slots'])
        _abilities_info = _pk_layout['abilities']
        _condition_dim = _pk_layout['moves']['offset'] - (_abilities_info['offset'] + _abilities_info['dim'])
        _hp_and_active_dim = POKEMON_FULL_DIM - _pk_layout['hp']['offset']
        _global_ctx_dim = (
            1                          # turn
            + _gl['weather']['dim']
            + _rl['fainted']['dim']
            + _gl['hazards']['dim']
            + 1                        # struggle
            + _gl['screens']['dim']
        )
        role_input_dim = (
            layout['species_embedding_dim']             # embedded species
            + 6                                         # base stats
            + layout['item_embedding_dim']              # embedded item
            + 2                                         # item known + consumed
            + 2 * layout['type_embedding_dim']          # embedded type pair
            + layout['ability_embedding_dim']           # embedded ability
            + 1                                         # ability known
            + _condition_dim                            # condition one-hot
            + MOVE_NET_HIDDEN[1] * _num_moves           # processed moves (4×32)
            + _hp_and_active_dim                        # hp + species_known + sleep + toxic + spread + hp_block + active
            + _global_ctx_dim                           # broadcasted global context
            + 1                                         # switch_validity
            + 1                                         # struggle_from_prev
        )
        self.role_token_size = ROLE_TOKEN_SIZE
        self.role_encoder = torch.nn.Sequential(
            torch.nn.Linear(role_input_dim, ROLE_ENCODER_HIDDEN[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(ROLE_ENCODER_HIDDEN[0], ROLE_ENCODER_HIDDEN[1])
        )

        # ------------------------------------------------------------------
        # 4. Active context encoder (kept for projection input only)
        # ------------------------------------------------------------------
        # Active context (boosts, volatiles) for each side enters the transformer
        # through the global token and is also encoded separately into 32-dim
        # tokens that go directly into the projection alongside the team pools.
        active_ctx_dim = layout['active_context_dim']
        self.active_ctx_encoder = torch.nn.Sequential(
            torch.nn.Linear(active_ctx_dim, ACTIVE_CTX_HIDDEN[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(ACTIVE_CTX_HIDDEN[0], ACTIVE_CTX_HIDDEN[1]),
        )

        # ------------------------------------------------------------------
        # 5. Unified transformer
        # ------------------------------------------------------------------
        # Token group identity (4: our_team / their_team / history / global).
        self.token_type_emb = torch.nn.Embedding(NUM_TOKEN_TYPES, D_MODEL)

        # History token: embed each raw TurnDelta identically (reuse move/type/species
        # tables), project to d_model, then add a learned positional encoding so "2 turns
        # ago" is distinguishable from "8 turns ago." Team slots stay permutation-
        # equivariant (no positional encoding); the global token sits alone.
        #
        # The slot's 4 move/type raw ints + 6 species raw ints (actor/target/switch_to
        # for both sides) are replaced by their embedded vectors; the remaining
        # TURN_DELTA_DIM-10 dims pass through as raw scalars.
        self._td_embed_dim = (
            2 * layout['move_embedding_dim']
            + 2 * layout['type_embedding_dim']
            + 6 * layout['species_embedding_dim']
            + TURN_DELTA_DIM - 10
        )
        self.history_proj = torch.nn.Linear(self._td_embed_dim, D_MODEL)
        self.turn_history_pos_emb = torch.nn.Embedding(N_HISTORY_TURNS, D_MODEL)

        # Global token input: active contexts (both sides) + non-matchup scalars
        # (global env + reactive scalars before the matchup matrices).
        reactive_layout = layout['reactive_layout']
        _matchup_offset_in_reactive = reactive_layout['our_matchups']['offset']
        # `non_matchup_rest` in forward is `remaining_part[:, 2*ctx : reactive_start + matchup_offset]`,
        # where remaining_part is sliced starting at OFFSET_CONTEXT. The dim of that
        # span is the global env dim + the pre-matchup portion of the reactive block.
        self._non_matchup_rest_dim = GLOBAL_ENV_DIM + _matchup_offset_in_reactive
        self._global_token_input_dim = 2 * active_ctx_dim + self._non_matchup_rest_dim
        self.global_proj = torch.nn.Linear(self._global_token_input_dim, D_MODEL)

        # The transformer itself. norm_first=False keeps the post-LN formulation
        # used by `nn.TransformerEncoderLayer`'s default; matches the design.
        # dropout=0 because the rest of the network is dropout-free and we want
        # forward_internal to be deterministic outside of train mode (the snapshot
        # round-trip test compares feature outputs across save/load).
        self.transformer_layers = torch.nn.ModuleList([
            torch.nn.TransformerEncoderLayer(
                d_model=D_MODEL,
                nhead=TRANSFORMER_N_HEADS,
                dim_feedforward=TRANSFORMER_FFN_DIM,
                dropout=0.0,
                activation="relu",
                batch_first=True,
                norm_first=False,
            )
            for _ in range(TRANSFORMER_N_LAYERS)
        ])

        # CLS cross-attention pooling: one learned query per team attends over the 6
        # post-transformer team tokens (fainted slots key-masked).
        self.our_cls = torch.nn.Parameter(torch.randn(1, 1, D_MODEL) * 0.02)
        self.their_cls = torch.nn.Parameter(torch.randn(1, 1, D_MODEL) * 0.02)
        self.our_cls_attn = torch.nn.MultiheadAttention(
            embed_dim=D_MODEL, num_heads=TRANSFORMER_N_HEADS, batch_first=True
        )
        self.their_cls_attn = torch.nn.MultiheadAttention(
            embed_dim=D_MODEL, num_heads=TRANSFORMER_N_HEADS, batch_first=True
        )
        self.norm_pool_our = torch.nn.LayerNorm(D_MODEL)
        self.norm_pool_their = torch.nn.LayerNorm(D_MODEL)

        # Token-count constants (handy for slicing transformer output).
        self._our_token_slice = slice(0, TEAM_SIZE)
        self._their_token_slice = slice(TEAM_SIZE, 2 * TEAM_SIZE)
        self._history_token_slice = slice(2 * TEAM_SIZE, 2 * TEAM_SIZE + N_HISTORY_TURNS)
        self._total_tokens = 2 * TEAM_SIZE + N_HISTORY_TURNS + 1

        # ------------------------------------------------------------------
        # 6. Dynamic projection-input dim discovery (dummy forward)
        # ------------------------------------------------------------------
        with torch.no_grad():
            dummy_obs = torch.zeros((1, layout['total_dim']))
            sample_output = self.forward_internal({"observation": dummy_obs})
            self.projection_input_dim = sample_output.shape[1]

        # ------------------------------------------------------------------
        # 7. Projection
        # ------------------------------------------------------------------
        # Pre-projection LayerNorm equalises per-block scales before the final Linear.
        self.pre_proj_norm = torch.nn.LayerNorm(self.projection_input_dim)
        self.projection_dim = PROJECTION_DIM
        self.projection = torch.nn.Linear(self.projection_input_dim, self.projection_dim)
        self.activation = torch.nn.ReLU()
        self.features_dim = self.projection_dim

        if log_level >= LogLevel.PERIODIC and mappings:
            from agents.model.observation_debugger import ObservationDebugger
            self._debugger: Optional[ObservationDebugger] = ObservationDebugger(mappings)
        else:
            self._debugger = None

    @staticmethod
    def _locate_active_slot(active_flags: torch.Tensor) -> torch.Tensor:
        """[B, TEAM_SIZE] active-flag tensor → [B] long index of the live active slot,
        or 0 when no flag is set. Centralises the fallback so a future change to the
        "no active" behaviour is a single-site edit.
        """
        B = active_flags.shape[0]
        return torch.where(
            active_flags.any(dim=1),
            torch.argmax(active_flags, dim=1),
            torch.zeros(B, dtype=torch.long, device=active_flags.device),
        )

    @staticmethod
    def _embed_delta_slot(slot, move_embedding, type_embedding, species_embedding):
        """Embed one [B, TURN_DELTA_DIM] raw TurnDelta vector → [B, _td_embed_dim].

        The slot carries 4 raw move/type IDs (positions 0, 4, 5, 9) and 6 raw species IDs
        (positions OFFSET_OUR_ACTOR_SPECIES..OFFSET_OPP_SWITCH_TO_SPEC, contiguous at the
        tail of the slot). All ten positions get looked up against their embedding tables
        and concatenated alongside the remaining scalar dims.
        """
        m_our = slot[:, 0].long().clamp(0, move_embedding.num_embeddings - 1)
        t_our = slot[:, 4].long().clamp(0, type_embedding.num_embeddings - 1)
        m_opp = slot[:, 5].long().clamp(0, move_embedding.num_embeddings - 1)
        t_opp = slot[:, 9].long().clamp(0, type_embedding.num_embeddings - 1)
        # Species IDs are contiguous at the tail; slice once and clamp.
        species_block = slot[:, OFFSET_OUR_ACTOR_SPECIES:OFFSET_OPP_SWITCH_TO_SPEC + 1].long()
        species_block = species_block.clamp(0, species_embedding.num_embeddings - 1)
        s_our_actor   = species_embedding(species_block[:, 0])
        s_opp_actor   = species_embedding(species_block[:, 1])
        s_our_target  = species_embedding(species_block[:, 2])
        s_opp_target  = species_embedding(species_block[:, 3])
        s_our_switch  = species_embedding(species_block[:, 4])
        s_opp_switch  = species_embedding(species_block[:, 5])
        # Pass-through scalars: everything that isn't a raw embedding ID.
        # Positions 1-3, 6-8, 10..OFFSET_OUR_ACTOR_SPECIES.
        scalars = torch.cat([
            slot[:, 1:4],
            slot[:, 6:9],
            slot[:, 10:OFFSET_OUR_ACTOR_SPECIES],
        ], dim=1)
        return torch.cat([
            move_embedding(m_our), type_embedding(t_our),
            move_embedding(m_opp), type_embedding(t_opp),
            s_our_actor, s_opp_actor, s_our_target, s_opp_target,
            s_our_switch, s_opp_switch,
            scalars,
        ], dim=1)

    def forward_internal(self, obs):
        """Internal forward pass: build the combined feature vector without the final projection."""
        x = obs["observation"]
        batch_size = x.shape[0]
        base_dim = self.layout['base_dim']

        # Layouts (derived from observation encoder; avoid hardcoded offsets below)
        reactive_layout = self.layout['reactive_layout']
        global_layout   = self.layout['global_layout']
        moves_info      = self.layout['pokemon']['moves']
        moves_layout    = moves_info['layout']
        m_slot_layout   = moves_layout['slot_layout']
        num_moves       = len(moves_layout['slots'])

        # Extract prev_mask (ACTION_SPACE_SIZE dims) and turn-history block (N * TURN_DELTA_DIM) from obs tail
        prev_mask = x[:, base_dim : base_dim + ACTION_SPACE_SIZE]                # [B, ACTION_SPACE_SIZE]
        history_dim = N_HISTORY_TURNS * TURN_DELTA_DIM
        turn_history_raw = x[:, base_dim + ACTION_SPACE_SIZE : base_dim + ACTION_SPACE_SIZE + history_dim]  # [B, N*39]
        switch_mask  = prev_mask[:, 0:TEAM_SIZE]                                 # [B, 6]
        move_mask    = prev_mask[:, TEAM_SIZE:TEAM_SIZE + num_moves]             # [B, 4]
        struggle_mask = prev_mask[:, TEAM_SIZE + num_moves:TEAM_SIZE + num_moves + 1]  # [B, 1]

        # ------------------------------------------------------------------
        # 1. Extract parts using dynamic layout (offsets read from layout, not hardcoded)
        # ------------------------------------------------------------------
        parts = self.layout['parts']
        ot = parts['our_team']
        our_team_raw = x[:, ot['start']:ot['end']].reshape(batch_size, *ot['reshape'])
        opt = parts['opp_team']
        opp_team_raw = x[:, opt['start']:opt['end']].reshape(batch_size, *opt['reshape'])
        ctx = parts['context']
        remaining_part = x[:, ctx['start']:base_dim]  # stop before prev_mask tail

        # Reactive layout offsets — `struggle_offset` is read again below where it is used.
        matchup_offset  = reactive_layout['our_matchups']['offset']

        # 1.1 Context features for move selection
        global_start = parts['global']['start'] - ctx['start']
        _w  = global_layout['weather'];  _w_off,  _w_dim  = _w['offset'], _w['dim']
        _hz = global_layout['hazards'];  _hz_off, _hz_dim = _hz['offset'], _hz['dim']
        _ck = global_layout['clock'];    _ck_off, _ck_dim = _ck['offset'], _ck['dim']
        weather_feature = remaining_part[:, global_start + _w_off  : global_start + _w_off  + _w_dim]   # [B, 6]
        spikes_feature  = remaining_part[:, global_start + _hz_off : global_start + _hz_off + _hz_dim]  # [B, 2]
        turn_feature    = remaining_part[:, global_start + _ck_off : global_start + _ck_off + _ck_dim]  # [B, 1]

        reactive_start = parts['reactive']['start'] - ctx['start']
        _f  = reactive_layout['fainted'];        _f_off,  _f_dim  = _f['offset'],  _f['dim']
        _om = reactive_layout['our_matchups'];   _om_off, _om_dim = _om['offset'], _om['dim']
        _tm = reactive_layout['their_matchups']; _tm_off, _tm_dim = _tm['offset'], _tm['dim']
        fainted_feature     = remaining_part[:, reactive_start + _f_off  : reactive_start + _f_off  + _f_dim]   # [B, 2]
        our_matchups_flat   = remaining_part[:, reactive_start + _om_off : reactive_start + _om_off + _om_dim]  # [B, 144]
        their_matchups_flat = remaining_part[:, reactive_start + _tm_off : reactive_start + _tm_off + _tm_dim]  # [B, 144]

        our_matchups   = our_matchups_flat.reshape(batch_size, TEAM_SIZE, num_moves, TEAM_SIZE)
        their_matchups = their_matchups_flat.reshape(batch_size, TEAM_SIZE, num_moves, TEAM_SIZE)
        matchups_all = torch.cat([our_matchups, their_matchups], dim=1)  # [B, 2*TEAM_SIZE, num_moves, TEAM_SIZE]

        pokemon_part = torch.cat([our_team_raw, opp_team_raw], dim=1)    # [B, 2*TEAM_SIZE, POKEMON_FULL_DIM]

        # ------------------------------------------------------------------
        # 2. Extract IDs for embedding
        # ------------------------------------------------------------------
        pk_layout = self.layout['pokemon']

        species_info = pk_layout['species']
        species_idx = species_info['offset'] + species_info['layout']['species_id']['offset']
        species_ids = pokemon_part[:, :, species_idx].long()

        moves_offset = moves_info['offset']
        _type_off = m_slot_layout['type']['offset']
        move_id_tensors = []
        move_type_id_tensors = []
        for i in range(num_moves):
            slot_idx = moves_offset + moves_layout['slots'][i]['offset']
            move_id_tensors.append(pokemon_part[:, :, slot_idx].long().unsqueeze(2))
            move_type_id_tensors.append(pokemon_part[:, :, slot_idx + _type_off].long().unsqueeze(2))
        all_move_ids = torch.cat(move_id_tensors, dim=2)            # [B, 12, 4]
        all_move_type_ids = torch.cat(move_type_id_tensors, dim=2)  # [B, 12, 4]

        items_info = pk_layout['items']
        items_layout = items_info['layout']
        item_idx = items_info['offset'] + items_layout['id']['offset']
        item_ids = pokemon_part[:, :, item_idx].long()

        abilities_info = pk_layout['abilities']
        abilities_layout = abilities_info['layout']
        ability_idx = abilities_info['offset'] + abilities_layout['id']['offset']
        ability_ids = pokemon_part[:, :, ability_idx].long()

        types_info = pk_layout['types']
        types_layout = types_info['layout']
        type1_ids = pokemon_part[:, :, types_info['offset'] + types_layout['type1']['offset']].long()
        type2_ids = pokemon_part[:, :, types_info['offset'] + types_layout['type2']['offset']].long()

        # ------------------------------------------------------------------
        # 3. Embed everything
        # ------------------------------------------------------------------
        embedded_species = self.species_embedding(species_ids)
        embedded_moves = self.move_embedding(all_move_ids)
        embedded_move_types = self.type_embedding(all_move_type_ids)
        embedded_items = self.item_embedding(item_ids)
        embedded_abilities = self.ability_embedding(ability_ids)

        # Pokémon types: concatenate E1 and E2 (32 dims) — sum loses type-pair signal
        embedded_t1 = self.type_embedding(type1_ids)
        embedded_t2 = self.type_embedding(type2_ids)
        embedded_pk_types = torch.cat([embedded_t1, embedded_t2], dim=-1)

        # ------------------------------------------------------------------
        # 4. Stitch the enriched per-Pokémon vector
        # ------------------------------------------------------------------
        species_id_layout = species_info['layout']['species_id']
        stats_start = species_idx + species_id_layout['dim']
        part_a = pokemon_part[:, :, stats_start : items_info['offset']]   # base stats [B, 12, 6]

        item_remnant_idx = items_info['offset'] + items_layout['known']['offset']
        item_remnant = pokemon_part[:, :, item_remnant_idx : item_remnant_idx + items_layout['known']['dim']]
        item_consumed_idx = items_info['offset'] + items_layout['consumed']['offset']
        item_consumed = pokemon_part[:, :, item_consumed_idx : item_consumed_idx + items_layout['consumed']['dim']]

        ability_remnant_idx = abilities_info['offset'] + abilities_layout['known']['offset']
        ability_remnant = pokemon_part[:, :, ability_remnant_idx : ability_remnant_idx + abilities_layout['known']['dim']]

        condition_start = abilities_info['offset'] + abilities_info['dim']
        part_d = pokemon_part[:, :, condition_start : moves_offset]       # condition one-hot

        _known_off = m_slot_layout['known']['offset']
        move_remnants = []
        for i in range(num_moves):
            slot_start = moves_offset + moves_layout['slots'][i]['offset']
            move_remnants.append(pokemon_part[:, :, slot_start + m_slot_layout['power']['offset'] : slot_start + m_slot_layout['type']['offset']])
            move_remnants.append(pokemon_part[:, :, slot_start + m_slot_layout['type']['offset'] + m_slot_layout['type']['dim'] : slot_start + _known_off])
            move_remnants.append(pokemon_part[:, :, slot_start + m_slot_layout['current_pp']['offset'] : slot_start + m_slot_layout['max_pp']['offset'] + m_slot_layout['max_pp']['dim']])
        all_move_remnants = torch.cat(move_remnants, dim=2)

        known_flags_tensors = []
        for i in range(num_moves):
            slot_start = moves_offset + moves_layout['slots'][i]['offset']
            known_flags_tensors.append(pokemon_part[:, :, slot_start + _known_off : slot_start + _known_off + 1])
        known_flags = torch.cat(known_flags_tensors, dim=2)

        hp_offset = pk_layout['hp']['offset']
        hp_and_active = pokemon_part[:, :, hp_offset:]                    # [B, 12, _hp_and_active_dim]

        # ------------------------------------------------------------------
        # 5. Shared move processing
        # ------------------------------------------------------------------
        n_poke = 2 * TEAM_SIZE
        move_remnants_reshaped = all_move_remnants.reshape(batch_size, n_poke, num_moves, self.move_remnant_dim)
        known_flags_reshaped   = known_flags.reshape(batch_size, n_poke, num_moves, 1)

        hp_feature       = hp_and_active[:, :, 0:1]
        turn_expanded    = turn_feature.unsqueeze(1).expand(-1, n_poke, -1)
        weather_expanded = weather_feature.unsqueeze(1).expand(-1, n_poke, -1)
        fainted_expanded = fainted_feature.unsqueeze(1).expand(-1, n_poke, -1)
        spikes_expanded  = spikes_feature.unsqueeze(1).expand(-1, n_poke, -1)

        move_context = torch.cat([hp_feature, turn_expanded, weather_expanded, fainted_expanded, spikes_expanded], dim=2)
        move_context_final = move_context.unsqueeze(2).expand(-1, -1, num_moves, -1)

        # Move validity from prev_mask: only the active slot gets the real move mask;
        # bench slots get all-ones (they have no move-validity context from prev turn).
        # The active flag is always the LAST dim of `hp_and_active` — keep this anchored
        # to `-1` so future per-slot additions (more counters, candidate blocks, etc.)
        # don't silently break it.
        _our_active_idx_early = self._locate_active_slot(hp_and_active[:, 0:TEAM_SIZE, -1])
        move_validity_ours = torch.ones(batch_size, TEAM_SIZE, num_moves, 1, device=x.device)
        move_validity_ours[torch.arange(batch_size, device=x.device), _our_active_idx_early] = \
            move_mask.unsqueeze(-1).float()
        move_validity_opp  = torch.ones(batch_size, TEAM_SIZE, num_moves, 1, device=x.device)
        move_validity = torch.cat([move_validity_ours, move_validity_opp], dim=1)

        move_features = torch.cat([
            embedded_moves,
            embedded_move_types,
            move_remnants_reshaped,
            known_flags_reshaped,
            move_context_final,
            matchups_all,
            move_validity,
        ], dim=3)

        processed_moves = self.move_network(move_features.reshape(-1, move_features.shape[-1]))
        processed_moves = processed_moves.reshape(batch_size, n_poke, num_moves, MOVE_NET_HIDDEN[1])

        # Within-Pokémon move self-attention.
        mv_in = processed_moves.reshape(batch_size * n_poke, num_moves, MOVE_NET_HIDDEN[1])
        mv_delta, _ = self.move_self_attn(mv_in, mv_in, mv_in)
        mv_out = self.move_self_norm(mv_in + mv_delta)
        processed_moves = mv_out.reshape(batch_size, n_poke, -1)

        pokemon_enriched = torch.cat([
            embedded_species,
            part_a,
            embedded_items,
            item_remnant,
            item_consumed,
            embedded_pk_types,
            embedded_abilities,
            ability_remnant,
            part_d,
            processed_moves,
            hp_and_active,
        ], dim=2)

        # ------------------------------------------------------------------
        # 6. Role encoder (per-Pokémon → 128-dim role token)
        # ------------------------------------------------------------------
        _fs = reactive_layout['forced_struggle']; struggle_offset = _fs['offset']
        struggle_feature = remaining_part[:, reactive_start + struggle_offset : reactive_start + struggle_offset + 1]
        _sc = global_layout['screens']; _sc_off, _sc_dim = _sc['offset'], _sc['dim']
        screen_feature = remaining_part[:, global_start + _sc_off : global_start + _sc_off + _sc_dim]
        global_context = torch.cat([turn_feature, weather_feature, fainted_feature, spikes_feature, struggle_feature, screen_feature], dim=1)
        context_broadcasted = global_context.unsqueeze(1).expand(-1, n_poke, -1)

        switch_validity_ours = switch_mask.unsqueeze(2)
        switch_validity_opp  = torch.ones(batch_size, TEAM_SIZE, 1, device=x.device)
        switch_validity = torch.cat([switch_validity_ours, switch_validity_opp], dim=1)

        struggle_from_prev = struggle_mask.unsqueeze(1).expand(-1, n_poke, -1)

        pokemon_enriched_with_context = torch.cat([
            pokemon_enriched, context_broadcasted, switch_validity, struggle_from_prev
        ], dim=2)

        role_tokens = self.role_encoder(
            pokemon_enriched_with_context.reshape(-1, pokemon_enriched_with_context.shape[-1])
        )
        role_tokens = role_tokens.reshape(batch_size, n_poke, self.role_token_size)   # [B, 12, 128]

        # Active context — encoded directly into the projection input alongside the team pools.
        active_ctx_dim = self.layout['active_context_dim']
        our_ctx_raw = remaining_part[:, 0 : active_ctx_dim]
        opp_ctx_raw = remaining_part[:, active_ctx_dim : 2 * active_ctx_dim]

        # Locate active slots for downstream extraction.
        # The active flag is always the LAST dim of `hp_and_active` (see comment above).
        active_flags = hp_and_active[:, :, -1]                  # [B, 12]
        our_active_idx = self._locate_active_slot(active_flags[:, 0:TEAM_SIZE])           # [B] in [0, 5]
        opp_active_local = self._locate_active_slot(active_flags[:, TEAM_SIZE:2 * TEAM_SIZE])  # [B] in [0, 5]
        batch_idx = torch.arange(batch_size, device=x.device)

        # Fainted masks (True = fainted). Always unmask the active slot on each side
        # so attention has at least one live key/query.
        fainted_mask_ours = (hp_and_active[:, 0:TEAM_SIZE, 0] == 0)
        fainted_mask_ours[batch_idx, our_active_idx] = False
        fainted_mask_opp = (hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, 0] == 0)
        fainted_mask_opp[batch_idx, opp_active_local] = False

        # ------------------------------------------------------------------
        # 7. Unified transformer over 23 tokens
        # ------------------------------------------------------------------
        # 7a. History tokens — embed each raw TurnDelta, project to d_model, add positional encoding.
        history_slots = turn_history_raw.view(batch_size, N_HISTORY_TURNS, TURN_DELTA_DIM)  # [B, N, 39]
        # An empty (padding) history slot is all-zero. Detect before projection because
        # projection biases would otherwise produce non-zero outputs from a zero input.
        empty_history = (history_slots.abs().sum(dim=-1) == 0)  # [B, N]
        embedded_history = torch.stack(
            [self._embed_delta_slot(history_slots[:, t, :],
                                    self.move_embedding,
                                    self.type_embedding,
                                    self.species_embedding)
             for t in range(N_HISTORY_TURNS)],
            dim=1,
        )  # [B, N, _td_embed_dim]
        history_tokens = self.history_proj(embedded_history)                       # [B, N, D_MODEL]
        positions = torch.arange(N_HISTORY_TURNS, device=x.device)
        history_tokens = history_tokens + self.turn_history_pos_emb(positions)     # [B, N, D_MODEL]

        # 7b. Global token — active contexts + non-matchup scalars projected into d_model.
        non_matchup_rest = remaining_part[:, 2 * active_ctx_dim : reactive_start + matchup_offset]
        global_token_input = torch.cat([our_ctx_raw, opp_ctx_raw, non_matchup_rest], dim=1)
        global_token = self.global_proj(global_token_input).unsqueeze(1)           # [B, 1, D_MODEL]

        # 7c. Add token-type embeddings to each group.
        our_team_tokens   = role_tokens[:, 0:TEAM_SIZE, :]                          # [B, 6, 128]
        their_team_tokens = role_tokens[:, TEAM_SIZE:2 * TEAM_SIZE, :]              # [B, 6, 128]

        tt = self.token_type_emb
        our_team_tokens   = our_team_tokens   + tt(torch.full((1,), TOKEN_TYPE_OUR_TEAM,   dtype=torch.long, device=x.device))
        their_team_tokens = their_team_tokens + tt(torch.full((1,), TOKEN_TYPE_THEIR_TEAM, dtype=torch.long, device=x.device))
        history_tokens    = history_tokens    + tt(torch.full((1,), TOKEN_TYPE_HISTORY,    dtype=torch.long, device=x.device))
        global_token      = global_token      + tt(torch.full((1,), TOKEN_TYPE_GLOBAL,     dtype=torch.long, device=x.device))

        # 7d. Build the token sequence and the key-padding mask.
        tokens = torch.cat([our_team_tokens, their_team_tokens, history_tokens, global_token], dim=1)
        # key_padding_mask: True at padding positions. Fainted team slots and empty
        # history slots are masked; the global token is always live.
        global_pad = torch.zeros(batch_size, 1, dtype=torch.bool, device=x.device)
        key_padding_mask = torch.cat([
            fainted_mask_ours,            # [B, 6]
            fainted_mask_opp,             # [B, 6]
            empty_history,                # [B, N]
            global_pad,                   # [B, 1]
        ], dim=1)                          # [B, 23]

        # 7e. Run the transformer stack.
        for layer in self.transformer_layers:
            tokens = layer(tokens, src_key_padding_mask=key_padding_mask)

        # 7f. Slice out the team outputs; discard the history and global outputs
        # (their information has already flowed into team tokens via attention).
        our_team_out   = tokens[:, self._our_token_slice, :]    # [B, 6, 128]
        their_team_out = tokens[:, self._their_token_slice, :]  # [B, 6, 128]

        # ------------------------------------------------------------------
        # 8. Team pooling via CLS cross-attention
        # ------------------------------------------------------------------
        our_cls_q   = self.our_cls.expand(batch_size, -1, -1)
        their_cls_q = self.their_cls.expand(batch_size, -1, -1)
        our_pool_out, _   = self.our_cls_attn(our_cls_q,   our_team_out,   our_team_out,
                                              key_padding_mask=fainted_mask_ours)
        their_pool_out, _ = self.their_cls_attn(their_cls_q, their_team_out, their_team_out,
                                                 key_padding_mask=fainted_mask_opp)
        our_team_pooled   = self.norm_pool_our(our_pool_out).squeeze(1)             # [B, 128]
        their_team_pooled = self.norm_pool_their(their_pool_out).squeeze(1)         # [B, 128]

        # Our-active output from the transformer (after all attention layers).
        our_active_refined = our_team_out[batch_idx, our_active_idx]                # [B, 128]

        # ------------------------------------------------------------------
        # 9. Projection input
        # ------------------------------------------------------------------
        our_ctx_enc = self.active_ctx_encoder(our_ctx_raw)                          # [B, 32]
        opp_ctx_enc = self.active_ctx_encoder(opp_ctx_raw)                          # [B, 32]

        combined = torch.cat([
            our_team_pooled,
            their_team_pooled,
            our_active_refined,
            our_ctx_enc,
            opp_ctx_enc,
            non_matchup_rest,
        ], dim=1)
        return combined

    def forward(self, obs):
        combined = self.forward_internal(obs)
        if self._debugger is not None:
            self._debugger.on_forward(obs["observation"])
        return self.activation(self.projection(self.pre_proj_norm(combined)))
