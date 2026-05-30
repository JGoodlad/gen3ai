import torch
import numpy as np
from dataclasses import dataclass
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple
from agents.observation.constants import (
    TRACE_INTERVAL,
    TEAM_SIZE,
    GLOBAL_ENV_DIM,
    POKEMON_FULL_DIM,
    POKEMON_HP_PROBS_OFFSET,
    POKEMON_SPECIES_KNOWN_OFFSET,
)
from agents.observation.moves import HIDDEN_POWER_MOVE_NUM
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
MOVE_NET_HIDDEN = [96, 32]        # [hidden, output] of shared move processor
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


def locate_active_slot(active_flags: torch.Tensor) -> torch.Tensor:
    """[B, TEAM_SIZE] active-flag tensor → [B] long index of the live active slot,
    or 0 when no flag is set. Centralises the fallback so a future change to the
    "no active" behaviour is a single-site edit."""
    B = active_flags.shape[0]
    return torch.where(
        active_flags.any(dim=1),
        torch.argmax(active_flags, dim=1),
        torch.zeros(B, dtype=torch.long, device=active_flags.device),
    )


def turn_delta_embed_dim(layout: Dict[str, Any]) -> int:
    """Width of one embedded TurnDelta slot: 4 move/type IDs + 6 species IDs replaced
    by their embeddings, plus the remaining (TURN_DELTA_DIM - 10) pass-through scalars."""
    return (
        2 * layout['move_embedding_dim']
        + 2 * layout['type_embedding_dim']
        + 6 * layout['species_embedding_dim']
        + TURN_DELTA_DIM - 10
    )


@dataclass(eq=False)
class ExtractorContext:
    """Immutable-by-convention bundle of everything `ObsUnpack` peels out of the flat
    observation vector, consumed by the downstream phase modules. Concentrating the
    obs-unpacking width here keeps every phase's forward signature narrow."""
    batch_size: int
    device: torch.device
    # Raw per-Pokémon block [B, 12, POKEMON_FULL_DIM] and the categorical IDs sliced from it.
    pokemon_part: torch.Tensor
    species_ids: torch.Tensor
    all_move_ids: torch.Tensor
    all_move_type_ids: torch.Tensor
    item_ids: torch.Tensor
    ability1_ids: torch.Tensor
    ability2_ids: torch.Tensor
    type1_ids: torch.Tensor
    type2_ids: torch.Tensor
    hp_probs: torch.Tensor
    hp_and_active: torch.Tensor
    # Reactive / global feature slices.
    matchups_all: torch.Tensor
    move_mask: torch.Tensor
    switch_mask: torch.Tensor
    struggle_mask: torch.Tensor
    turn_feature: torch.Tensor
    weather_feature: torch.Tensor
    fainted_feature: torch.Tensor
    spikes_feature: torch.Tensor
    struggle_feature: torch.Tensor
    screen_feature: torch.Tensor
    # Active-slot indices + fainted masks (hoisted here so transformer/pool read them directly).
    our_active_idx: torch.Tensor
    opp_active_local: torch.Tensor
    fainted_mask_ours: torch.Tensor
    fainted_mask_opp: torch.Tensor
    # Transformer / projection inputs.
    turn_history_raw: torch.Tensor
    our_ctx_raw: torch.Tensor
    opp_ctx_raw: torch.Tensor
    non_matchup_rest: torch.Tensor


class Embeddings(torch.nn.Module):
    """Sole owner of the embedding tables, shared between per-Pokémon encoding and
    turn-history encoding. Passed as a forward-time argument to the phases that need
    it so the tables register exactly once in the state_dict."""

    def __init__(self, layout: Dict[str, Any]):
        super().__init__()
        self.species_embedding = torch.nn.Embedding(layout['max_species'], layout['species_embedding_dim'])
        self.move_embedding = torch.nn.Embedding(layout['max_moves'], layout['move_embedding_dim'])
        self.item_embedding = torch.nn.Embedding(layout['max_items'], layout['item_embedding_dim'])
        self.ability_embedding = torch.nn.Embedding(layout['max_abilities'], layout['ability_embedding_dim'])
        # Shared type embedding for both Pokémon and Moves.
        self.type_embedding = torch.nn.Embedding(layout['max_types'], layout['type_embedding_dim'])

        # Hidden Power: map the 16 candidate-type slots in HIDDEN_POWER_TYPE_ORDER to
        # their rows in `type_embedding`. The probability-weighted soft type embedding for
        # an HP move slot is `hp_probs @ type_embedding[hp_type_idx_map]`, which collapses
        # to a direct lookup for one-hot (own-team) distributions.
        from agents.training.hidden_power_tracker import HIDDEN_POWER_TYPE_ORDER
        from agents.observation.types import TypeEncoder
        _hp_rows = [TypeEncoder.TYPE_TO_IDX[t.name] for t in HIDDEN_POWER_TYPE_ORDER]
        self.register_buffer('hp_type_idx_map', torch.tensor(_hp_rows, dtype=torch.long))

        self._td_embed_dim = turn_delta_embed_dim(layout)

    def hp_soft_type(self, hp_probs: torch.Tensor) -> torch.Tensor:
        """[B, 12, 16] HP candidate-type distribution → [B, 12, type_emb] soft type embedding."""
        hp_type_rows = self.type_embedding(self.hp_type_idx_map)   # [16, type_emb]
        return hp_probs @ hp_type_rows

    def embed_delta_slot(self, slot: torch.Tensor) -> torch.Tensor:
        """Embed one [B, TURN_DELTA_DIM] raw TurnDelta vector → [B, _td_embed_dim].

        The slot carries 4 raw move/type IDs (positions 0, 4, 5, 9) and 6 raw species IDs
        (positions OFFSET_OUR_ACTOR_SPECIES..OFFSET_OPP_SWITCH_TO_SPEC, contiguous at the
        tail of the slot). All ten positions get looked up against their embedding tables
        and concatenated alongside the remaining scalar dims."""
        move_embedding, type_embedding, species_embedding = (
            self.move_embedding, self.type_embedding, self.species_embedding
        )
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


class ObsUnpack(torch.nn.Module):
    """Stateless phase that peels the flat observation vector into named tensors
    (`ExtractorContext`). This is the bulk of the gather/slice plumbing; isolating it
    makes the rest of the pipeline read at the tensor level."""

    def __init__(self, layout: Dict[str, Any]):
        super().__init__()
        self.layout = layout

    def forward(self, obs: Dict[str, torch.Tensor]) -> ExtractorContext:
        layout = self.layout
        x = obs["observation"]
        batch_size = x.shape[0]
        base_dim = layout['base_dim']

        reactive_layout = layout['reactive_layout']
        global_layout   = layout['global_layout']
        moves_info      = layout['pokemon']['moves']
        moves_layout    = moves_info['layout']
        m_slot_layout   = moves_layout['slot_layout']
        num_moves       = len(moves_layout['slots'])

        # prev_mask (ACTION_SPACE_SIZE) + turn-history block (N * TURN_DELTA_DIM) from obs tail.
        prev_mask = x[:, base_dim : base_dim + ACTION_SPACE_SIZE]
        history_dim = N_HISTORY_TURNS * TURN_DELTA_DIM
        turn_history_raw = x[:, base_dim + ACTION_SPACE_SIZE : base_dim + ACTION_SPACE_SIZE + history_dim]
        switch_mask  = prev_mask[:, 0:TEAM_SIZE]
        move_mask    = prev_mask[:, TEAM_SIZE:TEAM_SIZE + num_moves]
        struggle_mask = prev_mask[:, TEAM_SIZE + num_moves:TEAM_SIZE + num_moves + 1]

        # Team blocks + the remaining context/global/reactive span.
        parts = layout['parts']
        ot = parts['our_team']
        our_team_raw = x[:, ot['start']:ot['end']].reshape(batch_size, *ot['reshape'])
        opt = parts['opp_team']
        opp_team_raw = x[:, opt['start']:opt['end']].reshape(batch_size, *opt['reshape'])
        ctx_part = parts['context']
        remaining_part = x[:, ctx_part['start']:base_dim]   # stop before prev_mask tail

        matchup_offset = reactive_layout['our_matchups']['offset']

        # Global env feature slices.
        global_start = parts['global']['start'] - ctx_part['start']
        _w  = global_layout['weather'];  _w_off,  _w_dim  = _w['offset'], _w['dim']
        _hz = global_layout['hazards'];  _hz_off, _hz_dim = _hz['offset'], _hz['dim']
        _ck = global_layout['clock'];    _ck_off, _ck_dim = _ck['offset'], _ck['dim']
        weather_feature = remaining_part[:, global_start + _w_off  : global_start + _w_off  + _w_dim]
        spikes_feature  = remaining_part[:, global_start + _hz_off : global_start + _hz_off + _hz_dim]
        turn_feature    = remaining_part[:, global_start + _ck_off : global_start + _ck_off + _ck_dim]
        _sc = global_layout['screens']; _sc_off, _sc_dim = _sc['offset'], _sc['dim']
        screen_feature = remaining_part[:, global_start + _sc_off : global_start + _sc_off + _sc_dim]

        # Reactive feature slices (fainted, matchups, forced-struggle).
        reactive_start = parts['reactive']['start'] - ctx_part['start']
        _f  = reactive_layout['fainted'];        _f_off,  _f_dim  = _f['offset'],  _f['dim']
        _om = reactive_layout['our_matchups'];   _om_off, _om_dim = _om['offset'], _om['dim']
        _tm = reactive_layout['their_matchups']; _tm_off, _tm_dim = _tm['offset'], _tm['dim']
        fainted_feature     = remaining_part[:, reactive_start + _f_off  : reactive_start + _f_off  + _f_dim]
        our_matchups_flat   = remaining_part[:, reactive_start + _om_off : reactive_start + _om_off + _om_dim]
        their_matchups_flat = remaining_part[:, reactive_start + _tm_off : reactive_start + _tm_off + _tm_dim]
        _fs = reactive_layout['forced_struggle']; struggle_offset = _fs['offset']
        struggle_feature = remaining_part[:, reactive_start + struggle_offset : reactive_start + struggle_offset + 1]

        our_matchups   = our_matchups_flat.reshape(batch_size, TEAM_SIZE, num_moves, TEAM_SIZE)
        their_matchups = their_matchups_flat.reshape(batch_size, TEAM_SIZE, num_moves, TEAM_SIZE)
        matchups_all = torch.cat([our_matchups, their_matchups], dim=1)

        pokemon_part = torch.cat([our_team_raw, opp_team_raw], dim=1)

        # Categorical IDs for embedding.
        pk_layout = layout['pokemon']
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
        all_move_ids = torch.cat(move_id_tensors, dim=2)
        all_move_type_ids = torch.cat(move_type_id_tensors, dim=2)

        items_info = pk_layout['items']
        items_layout = items_info['layout']
        item_idx = items_info['offset'] + items_layout['id']['offset']
        item_ids = pokemon_part[:, :, item_idx].long()

        abilities_info = pk_layout['abilities']
        abilities_layout = abilities_info['layout']
        ability1_idx = abilities_info['offset'] + abilities_layout['id1']['offset']
        ability2_idx = abilities_info['offset'] + abilities_layout['id2']['offset']
        ability1_ids = pokemon_part[:, :, ability1_idx].long()
        ability2_ids = pokemon_part[:, :, ability2_idx].long()

        types_info = pk_layout['types']
        types_layout = types_info['layout']
        type1_ids = pokemon_part[:, :, types_info['offset'] + types_layout['type1']['offset']].long()
        type2_ids = pokemon_part[:, :, types_info['offset'] + types_layout['type2']['offset']].long()

        hp_probs = pokemon_part[:, :, POKEMON_HP_PROBS_OFFSET : POKEMON_HP_PROBS_OFFSET + 16]  # [B, 12, 16]
        hp_offset = pk_layout['hp']['offset']
        hp_and_active = pokemon_part[:, :, hp_offset:]                                          # [B, 12, _]

        # Active-slot indices + fainted masks (used by move-validity, transformer, and pool).
        active_flags = hp_and_active[:, :, -1]
        our_active_idx = locate_active_slot(active_flags[:, 0:TEAM_SIZE])
        opp_active_local = locate_active_slot(active_flags[:, TEAM_SIZE:2 * TEAM_SIZE])
        batch_idx = torch.arange(batch_size, device=x.device)
        fainted_mask_ours = (hp_and_active[:, 0:TEAM_SIZE, 0] == 0)
        fainted_mask_ours[batch_idx, our_active_idx] = False
        fainted_mask_opp = (hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, 0] == 0)
        fainted_mask_opp[batch_idx, opp_active_local] = False

        # Active contexts + non-matchup scalar tail (transformer global token + projection).
        active_ctx_dim = layout['active_context_dim']
        our_ctx_raw = remaining_part[:, 0 : active_ctx_dim]
        opp_ctx_raw = remaining_part[:, active_ctx_dim : 2 * active_ctx_dim]
        non_matchup_rest = remaining_part[:, 2 * active_ctx_dim : reactive_start + matchup_offset]

        return ExtractorContext(
            batch_size=batch_size, device=x.device,
            pokemon_part=pokemon_part,
            species_ids=species_ids, all_move_ids=all_move_ids, all_move_type_ids=all_move_type_ids,
            item_ids=item_ids, ability1_ids=ability1_ids, ability2_ids=ability2_ids,
            type1_ids=type1_ids, type2_ids=type2_ids,
            hp_probs=hp_probs, hp_and_active=hp_and_active,
            matchups_all=matchups_all, move_mask=move_mask, switch_mask=switch_mask, struggle_mask=struggle_mask,
            turn_feature=turn_feature, weather_feature=weather_feature, fainted_feature=fainted_feature,
            spikes_feature=spikes_feature, struggle_feature=struggle_feature, screen_feature=screen_feature,
            our_active_idx=our_active_idx, opp_active_local=opp_active_local,
            fainted_mask_ours=fainted_mask_ours, fainted_mask_opp=fainted_mask_opp,
            turn_history_raw=turn_history_raw,
            our_ctx_raw=our_ctx_raw, opp_ctx_raw=opp_ctx_raw, non_matchup_rest=non_matchup_rest,
        )


class PokemonEncoder(torch.nn.Module):
    """Per-Pokémon encoding: embed + stitch the enriched vector, run the shared move
    processor + within-mon move self-attention, then the role encoder → 12×128 role tokens."""

    def __init__(self, layout: Dict[str, Any]):
        super().__init__()
        self.layout = layout
        _msl = layout['pokemon']['moves']['layout']['slot_layout']
        _rem1 = _msl['type']['offset'] - _msl['power']['offset']                                  # power+secondary+recoil = 3
        _rem2 = _msl['known']['offset'] - (_msl['type']['offset'] + _msl['type']['dim'])           # category = 1
        _rem3 = (_msl['max_pp']['offset'] + _msl['max_pp']['dim']) - _msl['current_pp']['offset']  # pp = 2
        _rem4 = (_msl['never_miss']['offset'] + _msl['never_miss']['dim']) - _msl['accuracy']['offset']  # accuracy+never_miss = 2
        self.move_remnant_dim = _rem1 + _rem2 + _rem3 + _rem4
        _gl = layout['global_layout']
        _rl = layout['reactive_layout']
        _move_ctx_dim = (1 + 1                       # hp + turn
                         + _gl['weather']['dim']     # weather
                         + _rl['fainted']['dim']     # fainted
                         + _gl['hazards']['dim'])    # spikes
        HP_PROBS_DIM = 16
        move_input_dim = (layout['move_embedding_dim'] + layout['type_embedding_dim']
                          + self.move_remnant_dim + 1               # remnants + known
                          + _move_ctx_dim                           # context
                          + TEAM_SIZE                                # matchups
                          + TEAM_SIZE                                # matchup validity (per opponent)
                          + HP_PROBS_DIM                             # hp candidate-type distribution
                          + 1)                                       # move validity
        self.move_network = torch.nn.Sequential(
            torch.nn.Linear(move_input_dim, MOVE_NET_HIDDEN[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(MOVE_NET_HIDDEN[0], MOVE_NET_HIDDEN[1])
        )
        self.move_self_attn = torch.nn.MultiheadAttention(
            embed_dim=MOVE_NET_HIDDEN[1], num_heads=2, batch_first=True
        )
        self.move_self_norm = torch.nn.LayerNorm(MOVE_NET_HIDDEN[1])

        _pk_layout = layout['pokemon']
        self.num_moves = len(_pk_layout['moves']['layout']['slots'])
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
            layout['species_embedding_dim']
            + 6                                         # base stats
            + layout['item_embedding_dim']
            + 2                                         # item known + consumed
            + 2 * layout['type_embedding_dim']          # type pair
            + 2 * layout['ability_embedding_dim']       # ability1 + ability2
            + 1                                         # ability dominance
            + 1                                         # ability known
            + _condition_dim
            + MOVE_NET_HIDDEN[1] * self.num_moves       # processed moves
            + _hp_and_active_dim
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

    def forward(self, ctx: ExtractorContext, embeddings: Embeddings) -> torch.Tensor:
        layout = self.layout
        batch_size = ctx.batch_size
        pokemon_part = ctx.pokemon_part
        num_moves = self.num_moves
        n_poke = 2 * TEAM_SIZE

        # --- Embed (incl. Hidden Power soft-type blend) ---
        embedded_species = embeddings.species_embedding(ctx.species_ids)
        embedded_moves = embeddings.move_embedding(ctx.all_move_ids)
        embedded_move_types = embeddings.type_embedding(ctx.all_move_type_ids)
        embedded_items = embeddings.item_embedding(ctx.item_ids)
        embedded_ability1 = embeddings.ability_embedding(ctx.ability1_ids)
        embedded_ability2 = embeddings.ability_embedding(ctx.ability2_ids)

        soft_type_emb_per_pk = embeddings.hp_soft_type(ctx.hp_probs)                            # [B, 12, type_emb]
        soft_type_emb = soft_type_emb_per_pk.unsqueeze(2).expand(-1, -1, num_moves, -1)         # [B, 12, 4, type_emb]
        is_hp_slot = (ctx.all_move_ids == HIDDEN_POWER_MOVE_NUM)                                # [B, 12, 4] bool
        embedded_move_types = torch.where(is_hp_slot.unsqueeze(-1), soft_type_emb, embedded_move_types)

        embedded_t1 = embeddings.type_embedding(ctx.type1_ids)
        embedded_t2 = embeddings.type_embedding(ctx.type2_ids)
        embedded_pk_types = torch.cat([embedded_t1, embedded_t2], dim=-1)

        # --- Stitch the enriched per-Pokémon vector ---
        pk_layout = layout['pokemon']
        species_info = pk_layout['species']
        species_idx = species_info['offset'] + species_info['layout']['species_id']['offset']
        species_id_layout = species_info['layout']['species_id']
        stats_start = species_idx + species_id_layout['dim']
        items_info = pk_layout['items']
        items_layout = items_info['layout']
        part_a = pokemon_part[:, :, stats_start : items_info['offset']]   # base stats [B, 12, 6]

        item_remnant_idx = items_info['offset'] + items_layout['known']['offset']
        item_remnant = pokemon_part[:, :, item_remnant_idx : item_remnant_idx + items_layout['known']['dim']]
        item_consumed_idx = items_info['offset'] + items_layout['consumed']['offset']
        item_consumed = pokemon_part[:, :, item_consumed_idx : item_consumed_idx + items_layout['consumed']['dim']]

        abilities_info = pk_layout['abilities']
        abilities_layout = abilities_info['layout']
        ability_remnant_idx = abilities_info['offset'] + abilities_layout['known']['offset']
        ability_remnant = pokemon_part[:, :, ability_remnant_idx : ability_remnant_idx + abilities_layout['known']['dim']]
        ability_dominance_idx = abilities_info['offset'] + abilities_layout['dominance']['offset']
        ability_dominance = pokemon_part[:, :, ability_dominance_idx : ability_dominance_idx + abilities_layout['dominance']['dim']]

        moves_info = pk_layout['moves']
        moves_offset = moves_info['offset']
        moves_layout = moves_info['layout']
        m_slot_layout = moves_layout['slot_layout']
        condition_start = abilities_info['offset'] + abilities_info['dim']
        part_d = pokemon_part[:, :, condition_start : moves_offset]       # condition one-hot

        _known_off = m_slot_layout['known']['offset']
        move_remnants = []
        for i in range(num_moves):
            slot_start = moves_offset + moves_layout['slots'][i]['offset']
            move_remnants.append(pokemon_part[:, :, slot_start + m_slot_layout['power']['offset'] : slot_start + m_slot_layout['type']['offset']])
            move_remnants.append(pokemon_part[:, :, slot_start + m_slot_layout['type']['offset'] + m_slot_layout['type']['dim'] : slot_start + _known_off])
            move_remnants.append(pokemon_part[:, :, slot_start + m_slot_layout['current_pp']['offset'] : slot_start + m_slot_layout['max_pp']['offset'] + m_slot_layout['max_pp']['dim']])
            move_remnants.append(pokemon_part[:, :, slot_start + m_slot_layout['accuracy']['offset'] : slot_start + m_slot_layout['never_miss']['offset'] + m_slot_layout['never_miss']['dim']])
        all_move_remnants = torch.cat(move_remnants, dim=2)

        known_flags_tensors = []
        for i in range(num_moves):
            slot_start = moves_offset + moves_layout['slots'][i]['offset']
            known_flags_tensors.append(pokemon_part[:, :, slot_start + _known_off : slot_start + _known_off + 1])
        known_flags = torch.cat(known_flags_tensors, dim=2)

        hp_and_active = ctx.hp_and_active

        # --- Shared move processing ---
        move_remnants_reshaped = all_move_remnants.reshape(batch_size, n_poke, num_moves, self.move_remnant_dim)
        known_flags_reshaped   = known_flags.reshape(batch_size, n_poke, num_moves, 1)

        hp_feature       = hp_and_active[:, :, 0:1]
        turn_expanded    = ctx.turn_feature.unsqueeze(1).expand(-1, n_poke, -1)
        weather_expanded = ctx.weather_feature.unsqueeze(1).expand(-1, n_poke, -1)
        fainted_expanded = ctx.fainted_feature.unsqueeze(1).expand(-1, n_poke, -1)
        spikes_expanded  = ctx.spikes_feature.unsqueeze(1).expand(-1, n_poke, -1)

        move_context = torch.cat([hp_feature, turn_expanded, weather_expanded, fainted_expanded, spikes_expanded], dim=2)
        move_context_final = move_context.unsqueeze(2).expand(-1, -1, num_moves, -1)

        # Move validity from prev_mask: only the active slot gets the real move mask;
        # bench slots get all-ones. The active flag is the LAST dim of `hp_and_active`.
        move_validity_ours = torch.ones(batch_size, TEAM_SIZE, num_moves, 1, device=ctx.device)
        move_validity_ours[torch.arange(batch_size, device=ctx.device), ctx.our_active_idx] = \
            ctx.move_mask.unsqueeze(-1).float()
        move_validity_opp  = torch.ones(batch_size, TEAM_SIZE, num_moves, 1, device=ctx.device)
        move_validity = torch.cat([move_validity_ours, move_validity_opp], dim=1)

        # HP candidate-type distribution per move slot: broadcast to the 4 slots, zeroed for non-HP slots.
        hp_probs_per_slot = ctx.hp_probs.unsqueeze(2).expand(-1, -1, num_moves, -1)  # [B, 12, 4, 16]
        hp_probs_per_slot = torch.where(
            is_hp_slot.unsqueeze(-1),
            hp_probs_per_slot,
            torch.zeros_like(hp_probs_per_slot),
        )

        # Per-cell matchup validity: `move_known(slot) AND species_known(opp)`.
        species_known_all = pokemon_part[:, :, POKEMON_SPECIES_KNOWN_OFFSET]  # [B, 12]
        our_species_known = species_known_all[:, :TEAM_SIZE]                  # [B, 6]
        opp_species_known = species_known_all[:, TEAM_SIZE:]                  # [B, 6]
        move_known_all = known_flags_reshaped.squeeze(-1)                     # [B, 12, 4]
        our_match_validity = (
            move_known_all[:, :TEAM_SIZE].unsqueeze(-1)
            * opp_species_known[:, None, None, :]
        )
        their_match_validity = (
            move_known_all[:, TEAM_SIZE:].unsqueeze(-1)
            * our_species_known[:, None, None, :]
        )
        matchup_validity = torch.cat([our_match_validity, their_match_validity], dim=1)  # [B, 12, 4, 6]

        move_features = torch.cat([
            embedded_moves,
            embedded_move_types,
            move_remnants_reshaped,
            known_flags_reshaped,
            move_context_final,
            ctx.matchups_all,
            matchup_validity,
            hp_probs_per_slot,
            move_validity,
        ], dim=3)

        processed_moves = self.move_network(move_features.reshape(-1, move_features.shape[-1]))
        processed_moves = processed_moves.reshape(batch_size, n_poke, num_moves, MOVE_NET_HIDDEN[1])

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
            embedded_ability1,
            embedded_ability2,
            ability_dominance,
            ability_remnant,
            part_d,
            processed_moves,
            hp_and_active,
        ], dim=2)

        # --- Role encoder ---
        global_context = torch.cat([
            ctx.turn_feature, ctx.weather_feature, ctx.fainted_feature,
            ctx.spikes_feature, ctx.struggle_feature, ctx.screen_feature,
        ], dim=1)
        context_broadcasted = global_context.unsqueeze(1).expand(-1, n_poke, -1)

        switch_validity_ours = ctx.switch_mask.unsqueeze(2)
        switch_validity_opp  = torch.ones(batch_size, TEAM_SIZE, 1, device=ctx.device)
        switch_validity = torch.cat([switch_validity_ours, switch_validity_opp], dim=1)

        struggle_from_prev = ctx.struggle_mask.unsqueeze(1).expand(-1, n_poke, -1)

        pokemon_enriched_with_context = torch.cat([
            pokemon_enriched, context_broadcasted, switch_validity, struggle_from_prev
        ], dim=2)

        role_tokens = self.role_encoder(
            pokemon_enriched_with_context.reshape(-1, pokemon_enriched_with_context.shape[-1])
        )
        return role_tokens.reshape(batch_size, n_poke, self.role_token_size)   # [B, 12, 128]


class TeamTransformer(torch.nn.Module):
    """Unified transformer over 23 tokens (6 our + 6 their team role tokens + 10 history
    + 1 global). Adds token-type and history-positional embeddings, applies the encoder
    stack with a fainted/empty-history key-padding mask, returns the two team token blocks."""

    def __init__(self, layout: Dict[str, Any]):
        super().__init__()
        self.token_type_emb = torch.nn.Embedding(NUM_TOKEN_TYPES, D_MODEL)

        self._td_embed_dim = turn_delta_embed_dim(layout)
        self.history_proj = torch.nn.Linear(self._td_embed_dim, D_MODEL)
        self.turn_history_pos_emb = torch.nn.Embedding(N_HISTORY_TURNS, D_MODEL)

        reactive_layout = layout['reactive_layout']
        _matchup_offset_in_reactive = reactive_layout['our_matchups']['offset']
        active_ctx_dim = layout['active_context_dim']
        self._non_matchup_rest_dim = GLOBAL_ENV_DIM + _matchup_offset_in_reactive
        self._global_token_input_dim = 2 * active_ctx_dim + self._non_matchup_rest_dim
        self.global_proj = torch.nn.Linear(self._global_token_input_dim, D_MODEL)

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

        self._our_token_slice = slice(0, TEAM_SIZE)
        self._their_token_slice = slice(TEAM_SIZE, 2 * TEAM_SIZE)
        self._history_token_slice = slice(2 * TEAM_SIZE, 2 * TEAM_SIZE + N_HISTORY_TURNS)
        self._total_tokens = 2 * TEAM_SIZE + N_HISTORY_TURNS + 1   # team×2 + history + global

    def forward(self, role_tokens: torch.Tensor, ctx: ExtractorContext,
                embeddings: Embeddings) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = ctx.batch_size
        device = ctx.device

        # History tokens — embed each raw TurnDelta, project to d_model, add positional encoding.
        history_slots = ctx.turn_history_raw.view(batch_size, N_HISTORY_TURNS, TURN_DELTA_DIM)
        empty_history = (history_slots.abs().sum(dim=-1) == 0)  # [B, N]
        embedded_history = torch.stack(
            [embeddings.embed_delta_slot(history_slots[:, t, :]) for t in range(N_HISTORY_TURNS)],
            dim=1,
        )
        history_tokens = self.history_proj(embedded_history)
        positions = torch.arange(N_HISTORY_TURNS, device=device)
        history_tokens = history_tokens + self.turn_history_pos_emb(positions)

        # Global token — active contexts + non-matchup scalars projected into d_model.
        global_token_input = torch.cat([ctx.our_ctx_raw, ctx.opp_ctx_raw, ctx.non_matchup_rest], dim=1)
        global_token = self.global_proj(global_token_input).unsqueeze(1)

        # Token-type embeddings per group.
        our_team_tokens   = role_tokens[:, 0:TEAM_SIZE, :]
        their_team_tokens = role_tokens[:, TEAM_SIZE:2 * TEAM_SIZE, :]

        tt = self.token_type_emb
        our_team_tokens   = our_team_tokens   + tt(torch.full((1,), TOKEN_TYPE_OUR_TEAM,   dtype=torch.long, device=device))
        their_team_tokens = their_team_tokens + tt(torch.full((1,), TOKEN_TYPE_THEIR_TEAM, dtype=torch.long, device=device))
        history_tokens    = history_tokens    + tt(torch.full((1,), TOKEN_TYPE_HISTORY,    dtype=torch.long, device=device))
        global_token      = global_token      + tt(torch.full((1,), TOKEN_TYPE_GLOBAL,     dtype=torch.long, device=device))

        tokens = torch.cat([our_team_tokens, their_team_tokens, history_tokens, global_token], dim=1)
        global_pad = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        key_padding_mask = torch.cat([
            ctx.fainted_mask_ours,
            ctx.fainted_mask_opp,
            empty_history,
            global_pad,
        ], dim=1)

        for layer in self.transformer_layers:
            tokens = layer(tokens, src_key_padding_mask=key_padding_mask)

        our_team_out   = tokens[:, self._our_token_slice, :]
        their_team_out = tokens[:, self._their_token_slice, :]
        return our_team_out, their_team_out


class CLSPool(torch.nn.Module):
    """Per-side CLS cross-attention pool: one learned query attends over the 6 post-transformer
    team tokens (fainted slots key-masked). Also extracts our active Pokémon's refined token."""

    def __init__(self, layout: Dict[str, Any]):
        super().__init__()
        self.our_cls = torch.nn.Parameter(torch.randn(1, 1, D_MODEL) * 0.02)
        self.their_cls = torch.nn.Parameter(torch.randn(1, 1, D_MODEL) * 0.02)
        self.our_cls_attn = torch.nn.MultiheadAttention(embed_dim=D_MODEL, num_heads=TRANSFORMER_N_HEADS, batch_first=True)
        self.their_cls_attn = torch.nn.MultiheadAttention(embed_dim=D_MODEL, num_heads=TRANSFORMER_N_HEADS, batch_first=True)
        self.norm_pool_our = torch.nn.LayerNorm(D_MODEL)
        self.norm_pool_their = torch.nn.LayerNorm(D_MODEL)

    def forward(self, our_team_out: torch.Tensor, their_team_out: torch.Tensor,
                ctx: ExtractorContext) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = ctx.batch_size
        our_cls_q   = self.our_cls.expand(batch_size, -1, -1)
        their_cls_q = self.their_cls.expand(batch_size, -1, -1)
        our_pool_out, _   = self.our_cls_attn(our_cls_q,   our_team_out,   our_team_out,
                                              key_padding_mask=ctx.fainted_mask_ours)
        their_pool_out, _ = self.their_cls_attn(their_cls_q, their_team_out, their_team_out,
                                                key_padding_mask=ctx.fainted_mask_opp)
        our_team_pooled   = self.norm_pool_our(our_pool_out).squeeze(1)             # [B, 128]
        their_team_pooled = self.norm_pool_their(their_pool_out).squeeze(1)         # [B, 128]

        batch_idx = torch.arange(batch_size, device=ctx.device)
        our_active_refined = our_team_out[batch_idx, ctx.our_active_idx]            # [B, 128]
        return our_team_pooled, their_team_pooled, our_active_refined


class ProjectionAssembler(torch.nn.Module):
    """Assembles the projection input: team pools + our active token + per-side encoded
    active contexts + the non-matchup scalar tail."""

    def __init__(self, layout: Dict[str, Any]):
        super().__init__()
        active_ctx_dim = layout['active_context_dim']
        self.active_ctx_encoder = torch.nn.Sequential(
            torch.nn.Linear(active_ctx_dim, ACTIVE_CTX_HIDDEN[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(ACTIVE_CTX_HIDDEN[0], ACTIVE_CTX_HIDDEN[1]),
        )

    def forward(self, our_team_pooled: torch.Tensor, their_team_pooled: torch.Tensor,
                our_active_refined: torch.Tensor, ctx: ExtractorContext) -> torch.Tensor:
        our_ctx_enc = self.active_ctx_encoder(ctx.our_ctx_raw)                      # [B, 32]
        opp_ctx_enc = self.active_ctx_encoder(ctx.opp_ctx_raw)                      # [B, 32]
        return torch.cat([
            our_team_pooled,
            their_team_pooled,
            our_active_refined,
            our_ctx_enc,
            opp_ctx_enc,
            ctx.non_matchup_rest,
        ], dim=1)


class Gen3FeaturesExtractor(torch.nn.Module):
    """Orchestrates the phase modules. Data flow:
        ObsUnpack → PokemonEncoder → TeamTransformer → CLSPool → ProjectionAssembler
    then a final pre-projection LayerNorm + Linear + ReLU head. The embedding tables live
    in `self.embeddings` (shared) and are passed into the phases that need them. See
    `src/agents/model/CLAUDE.md` for the phase-module contract."""

    def __init__(self, observation_space: spaces.Dict, layout: Optional[Dict[str, Any]] = None,
                 mappings: Optional[Dict[str, Any]] = None, log_level: LogLevel = LogLevel.QUIET):
        super().__init__()
        self.layout = layout
        self.mappings = mappings
        self.log_level = log_level

        # Phase modules (constructed before the dummy forward below).
        self.embeddings = Embeddings(layout)
        self.unpack = ObsUnpack(layout)
        self.pokemon_encoder = PokemonEncoder(layout)
        self.team_transformer = TeamTransformer(layout)
        self.cls_pool = CLSPool(layout)
        self.assembler = ProjectionAssembler(layout)

        self.role_token_size = ROLE_TOKEN_SIZE

        # Discover the projection-input dim via a dummy forward through the assembled phases.
        with torch.no_grad():
            dummy_obs = torch.zeros((1, layout['total_dim']))
            sample_output = self.forward_internal({"observation": dummy_obs})
            self.projection_input_dim = sample_output.shape[1]

        # Projection head. Pre-projection LayerNorm equalises per-block scales.
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

    # Read-only forwarders for the shared embedding tables — they are a model-level concept
    # and several tests/inspectors reach for them by name. Properties add no state_dict keys.
    @property
    def species_embedding(self): return self.embeddings.species_embedding
    @property
    def move_embedding(self): return self.embeddings.move_embedding
    @property
    def item_embedding(self): return self.embeddings.item_embedding
    @property
    def ability_embedding(self): return self.embeddings.ability_embedding
    @property
    def type_embedding(self): return self.embeddings.type_embedding
    @property
    def hp_type_idx_map(self): return self.embeddings.hp_type_idx_map

    @staticmethod
    def _locate_active_slot(active_flags: torch.Tensor) -> torch.Tensor:
        """Back-compat shim — delegates to the module-level `locate_active_slot`."""
        return locate_active_slot(active_flags)

    def forward_internal(self, obs):
        """Build the combined feature vector (pre-projection) by chaining the phase modules."""
        ctx = self.unpack(obs)
        role_tokens = self.pokemon_encoder(ctx, self.embeddings)
        our_team_out, their_team_out = self.team_transformer(role_tokens, ctx, self.embeddings)
        our_team_pooled, their_team_pooled, our_active_refined = self.cls_pool(
            our_team_out, their_team_out, ctx
        )
        return self.assembler(our_team_pooled, their_team_pooled, our_active_refined, ctx)

    def forward(self, obs):
        combined = self.forward_internal(obs)
        if self._debugger is not None:
            self._debugger.on_forward(obs["observation"])
        return self.activation(self.projection(self.pre_proj_norm(combined)))
