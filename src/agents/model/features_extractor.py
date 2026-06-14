import torch
from torch.utils.checkpoint import checkpoint
import numpy as np
from dataclasses import dataclass, replace
from gymnasium import spaces
from typing import Dict, Any, Optional, Tuple
from agents.observation.constants import (
    TRACE_INTERVAL,
    TEAM_SIZE,
    GLOBAL_ENV_DIM,
    POKEMON_FULL_DIM,
    POKEMON_HP_PROBS_OFFSET,
    POKEMON_SPECIES_KNOWN_OFFSET,
    POKEMON_SPREAD_OFFSET,
    POKEMON_SPREAD_DIM,
)
from agents.observation.moves import HIDDEN_POWER_MOVE_NUM
from agents.observation.turn_delta_encoder import (
    TURN_DELTA_DIM,
    EFF_DIM,
    ORDER_DIM,
    TURN_DELTA_EMBEDDED_IDS,
    TURN_DELTA_SCALAR_OFFSETS,
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
    """Width of one embedded TurnDelta slot, derived from the embedded-ID manifest.

    Each entry in TURN_DELTA_EMBEDDED_IDS contributes its table's embedding dim;
    every non-embedded slot position contributes 1 (pass-through scalar). No
    hand-maintained counts — add an ID to the manifest and this updates itself."""
    dim_by_kind = {
        "move": layout['move_embedding_dim'],
        "type": layout['type_embedding_dim'],
        "species": layout['species_embedding_dim'],
    }
    embedded = sum(dim_by_kind[kind] for _, kind in TURN_DELTA_EMBEDDED_IDS)
    return embedded + len(TURN_DELTA_SCALAR_OFFSETS)


def slice_pokemon_categoricals(pokemon_part: torch.Tensor, layout: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """Slice the per-Pokémon categorical IDs + HP blocks from a [B, N, POKEMON_FULL_DIM] block.

    Pure, layout-driven — the SINGLE source of truth for which slot positions carry which embedding
    ID. Shared by `ObsUnpack` (the live obs block) and the latent-belief privileged-target encode (a
    fresh true-opp-team block), so a fresh block is sliced byte-identically to the live one. Returns
    exactly the `ExtractorContext` fields `PokemonEncoder` consumes from the per-mon block."""
    pk_layout = layout['pokemon']
    moves_info = pk_layout['moves']
    moves_layout = moves_info['layout']
    m_slot_layout = moves_layout['slot_layout']
    num_moves = len(moves_layout['slots'])

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

    hp_probs = pokemon_part[:, :, POKEMON_HP_PROBS_OFFSET : POKEMON_HP_PROBS_OFFSET + 16]  # [B, N, 16]
    hp_offset = pk_layout['hp']['offset']
    hp_and_active = pokemon_part[:, :, hp_offset:]                                          # [B, N, _]

    return {
        "species_ids": species_ids, "all_move_ids": all_move_ids,
        "all_move_type_ids": all_move_type_ids, "item_ids": item_ids,
        "ability1_ids": ability1_ids, "ability2_ids": ability2_ids,
        "type1_ids": type1_ids, "type2_ids": type2_ids,
        "hp_probs": hp_probs, "hp_and_active": hp_and_active,
    }


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
    # Per-opp-slot "still hidden" mask [B, 6]: True where species_known==0 (Gen 3 has no team
    # preview, so these are the opponent's un-revealed party mons). Single-sourced here so the
    # in-place belief-slot injection (BeliefSlots) and any future consumer agree on which slots
    # are believed vs revealed. Always computed (cheap); only consumed when belief is enabled.
    opp_believed_mask: torch.Tensor
    # Combined 12-token key-mask (= cat[ours, opp]), single-sourced here so the value-CLS pool and
    # the hidden-opp belief share ONE mask — they rely on the same "both actives force-unmasked ⇒ no
    # all-True row" NaN-safety invariant, which must not be able to drift between two call sites.
    all_fainted: torch.Tensor
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

        # Embedded-ID manifest (single source of truth, from turn_delta_encoder).
        # Precompute, per embedding table, the LongTensor of slot positions that
        # route to it (in manifest order), and the pass-through scalar positions.
        # Registered as buffers so they move with .to(device) and never desync from
        # the encoder layout. embed_delta_slot is then a pure table-driven gather.
        _kind_to_table = {
            "move": "move_embedding", "type": "type_embedding", "species": "species_embedding",
        }
        # Ordered list of (table_attr, position) following the manifest order, so the
        # concatenation order of embedded outputs matches the manifest exactly.
        self._td_embed_plan = tuple(
            (_kind_to_table[kind], pos) for pos, kind in TURN_DELTA_EMBEDDED_IDS
        )
        self.register_buffer(
            "_td_scalar_idx",
            torch.tensor(TURN_DELTA_SCALAR_OFFSETS, dtype=torch.long),
            persistent=False,  # derived from constants — not a learned/saved tensor
        )

    def hp_soft_type(self, hp_probs: torch.Tensor) -> torch.Tensor:
        """[B, 12, 16] HP candidate-type distribution → [B, 12, type_emb] soft type embedding."""
        hp_type_rows = self.type_embedding(self.hp_type_idx_map)   # [16, type_emb]
        return hp_probs @ hp_type_rows

    def embed_delta_slot(self, slot: torch.Tensor) -> torch.Tensor:
        """Embed one [B, TURN_DELTA_DIM] raw TurnDelta vector → [B, _td_embed_dim].

        Fully manifest-driven: TURN_DELTA_EMBEDDED_IDS (in turn_delta_encoder) declares
        which slot positions carry raw embedding IDs and which table each routes to.
        Every other position is a pass-through scalar. There are NO hardcoded positions
        here — the encoder and extractor read the same manifest, so a raw id can never
        silently leak through as a scalar."""
        parts = []
        for table_attr, pos in self._td_embed_plan:
            table = getattr(self, table_attr)
            idx = slot[:, pos].long().clamp(0, table.num_embeddings - 1)
            parts.append(table(idx))
        # Pass-through scalars, gathered in ascending-position order.
        parts.append(slot.index_select(1, self._td_scalar_idx))
        return torch.cat(parts, dim=1)


class ObsUnpack(torch.nn.Module):
    """Stateless phase that peels the flat observation vector into named tensors
    (`ExtractorContext`). This is the bulk of the gather/slice plumbing; isolating it
    makes the rest of the pipeline read at the tensor level."""

    def __init__(self, layout: Dict[str, Any], attend_unrevealed_opponents: bool = False):
        super().__init__()
        self.layout = layout
        # When True, UNREVEALED opp slots (species_known==0, hp filled as 0 — Gen 3 has no
        # team preview, so unseen party mons arrive here as all-zero placeholders) stay
        # ATTENDABLE in the transformer instead of being key-masked identically to fainted
        # mons. Lets the body reason about the still-hidden enemy team. Off = baseline.
        self.attend_unrevealed_opponents = attend_unrevealed_opponents

    def forward(self, obs: Dict[str, torch.Tensor]) -> ExtractorContext:
        layout = self.layout
        x = obs["observation"]
        batch_size = x.shape[0]
        base_dim = layout['base_dim']

        reactive_layout = layout['reactive_layout']
        global_layout   = layout['global_layout']
        # num_moves drives the prev-mask + matchup slices below; the per-mon move-slot slicing moved to
        # slice_pokemon_categoricals (so moves_info/moves_layout/m_slot_layout are no longer needed here).
        num_moves       = len(layout['pokemon']['moves']['layout']['slots'])

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

        # Categorical IDs + HP blocks for embedding (shared, layout-driven slicer — the same one the
        # latent-belief privileged-target encode uses, so a fresh block slices identically).
        _ids = slice_pokemon_categoricals(pokemon_part, layout)
        species_ids = _ids["species_ids"]
        all_move_ids = _ids["all_move_ids"]
        all_move_type_ids = _ids["all_move_type_ids"]
        item_ids = _ids["item_ids"]
        ability1_ids = _ids["ability1_ids"]
        ability2_ids = _ids["ability2_ids"]
        type1_ids = _ids["type1_ids"]
        type2_ids = _ids["type2_ids"]
        hp_probs = _ids["hp_probs"]
        hp_and_active = _ids["hp_and_active"]

        # Active-slot indices + fainted masks (used by move-validity, transformer, and pool).
        active_flags = hp_and_active[:, :, -1]
        our_active_idx = locate_active_slot(active_flags[:, 0:TEAM_SIZE])
        opp_active_local = locate_active_slot(active_flags[:, TEAM_SIZE:2 * TEAM_SIZE])
        batch_idx = torch.arange(batch_size, device=x.device)
        fainted_mask_ours = (hp_and_active[:, 0:TEAM_SIZE, 0] == 0)
        fainted_mask_ours[batch_idx, our_active_idx] = False
        fainted_mask_opp = (hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, 0] == 0)
        # Per-opp-slot "still hidden" mask, single-sourced for both the attendability mask below
        # and the in-place belief-slot injection. species_known==0 ⇒ the opponent's un-revealed party.
        species_known_opp = pokemon_part[:, TEAM_SIZE:2 * TEAM_SIZE, POKEMON_SPECIES_KNOWN_OFFSET]
        opp_believed_mask = species_known_opp < 0.5                                   # [B, 6] bool
        if self.attend_unrevealed_opponents:
            # Keep UNREVEALED opp slots attendable: only REVEALED-then-fainted mons
            # (species_known==1 AND hp==0) stay masked. species_known==0 slots are the
            # opponent's still-hidden party — let the transformer attend to them.
            fainted_mask_opp = fainted_mask_opp & (species_known_opp > 0.5)
        fainted_mask_opp[batch_idx, opp_active_local] = False
        # Combined 12-token key-mask, single-sourced (both actives are now force-unmasked, so it can
        # never be all-True per row — the NaN-safety invariant the value-CLS pool + belief both rely on).
        all_fainted = torch.cat([fainted_mask_ours, fainted_mask_opp], dim=1)

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
            opp_believed_mask=opp_believed_mask,
            all_fainted=all_fainted,
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
        # Runtime-only memory/compute knob (NOT a weight or arch param — never enters
        # model_config.json / the version check). When True, the encoder layers are run
        # under gradient checkpointing during the backward-needing pass, trading one extra
        # forward (on the otherwise-idle GPU) for ~5GB less activation VRAM. Bit-exact:
        # dropout=0.0 and use_reentrant=False make the recompute identical. Toggled per run
        # from --grad-checkpointing via _apply_grad_checkpointing(); a no-op under inference.
        self.grad_checkpointing = False
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

        # Gradient checkpointing only helps when a graph is being built for backward
        # (the PPO update); under inference's no_grad it would be pure overhead, so gate on
        # torch.is_grad_enabled(). use_reentrant=False is the correct variant here (handles
        # non-grad inputs + autocast/RNG state); with dropout=0 the recompute is bit-exact.
        use_ckpt = self.grad_checkpointing and torch.is_grad_enabled()
        for layer in self.transformer_layers:
            if use_ckpt:
                tokens = checkpoint(
                    lambda t, _layer=layer: _layer(t, src_key_padding_mask=key_padding_mask),
                    tokens,
                    use_reentrant=False,
                )
            else:
                tokens = layer(tokens, src_key_padding_mask=key_padding_mask)

        our_team_out   = tokens[:, self._our_token_slice, :]
        their_team_out = tokens[:, self._their_token_slice, :]
        return our_team_out, their_team_out


class CLSPool(torch.nn.Module):
    """Per-side CLS cross-attention pools plus a value-dedicated pool.

    `our_cls`/`their_cls` each attend over their side's 6 post-transformer team tokens
    (fainted slots key-masked) to produce the policy-facing team summaries, and we also
    extract our active Pokémon's refined token. `value_cls` is a third learned query that
    attends over ALL 12 team tokens (both sides) to produce a global "who's winning"
    summary for the value head — a different aggregation than the policy's our-active-centric
    view. History/global information has already flowed into the team tokens via the unified
    transformer, so pooling over the 12 team tokens gives the value query a whole-board read.
    """

    def __init__(self, layout: Dict[str, Any]):
        super().__init__()
        self.our_cls = torch.nn.Parameter(torch.randn(1, 1, D_MODEL) * 0.02)
        self.their_cls = torch.nn.Parameter(torch.randn(1, 1, D_MODEL) * 0.02)
        self.our_cls_attn = torch.nn.MultiheadAttention(embed_dim=D_MODEL, num_heads=TRANSFORMER_N_HEADS, batch_first=True)
        self.their_cls_attn = torch.nn.MultiheadAttention(embed_dim=D_MODEL, num_heads=TRANSFORMER_N_HEADS, batch_first=True)
        self.norm_pool_our = torch.nn.LayerNorm(D_MODEL)
        self.norm_pool_their = torch.nn.LayerNorm(D_MODEL)

        # Value-dedicated CLS pool (Option C): a separate readout for the critic so the
        # shared body's representation is summarised through a value-specific lens.
        self.value_cls = torch.nn.Parameter(torch.randn(1, 1, D_MODEL) * 0.02)
        self.value_cls_attn = torch.nn.MultiheadAttention(embed_dim=D_MODEL, num_heads=TRANSFORMER_N_HEADS, batch_first=True)
        self.norm_pool_value = torch.nn.LayerNorm(D_MODEL)

    def forward(self, our_team_out: torch.Tensor, their_team_out: torch.Tensor,
                ctx: ExtractorContext) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        # Value pool: one query over both teams' 12 tokens (fainted slots key-masked).
        all_team_out = torch.cat([our_team_out, their_team_out], dim=1)             # [B, 12, 128]
        value_cls_q  = self.value_cls.expand(batch_size, -1, -1)
        value_pool_out, _ = self.value_cls_attn(value_cls_q, all_team_out, all_team_out,
                                                key_padding_mask=ctx.all_fainted)
        value_pooled = self.norm_pool_value(value_pool_out).squeeze(1)              # [B, 128]

        return our_team_pooled, their_team_pooled, our_active_refined, value_pooled


class HiddenOppBeliefPool(torch.nn.Module):
    """Hidden-opponent belief: K distinct learned query tokens (DETR object-query / Slot-Attention
    style) that read the post-transformer team tokens and summarise the belief over the opponent's
    still-hidden party.

    Why this exists: once the unrevealed opp slots are unmasked (`attend_unrevealed_opponents`), the
    N unknown slots are *identical* (zeros + `species_known=0`); a permutation-equivariant transformer
    is forced to map identical inputs to identical outputs, so they collapse to one representation —
    the model can know "there are unknowns" but can't represent "slot A leans physical sweeper, slot B
    leans special wall". K learned queries are non-identical *by construction* (independent init), so
    they break that symmetry and can specialise.

    - **K=1** is a single "hidden-opponent CLS" — a set-summary of the whole unrevealed remainder
      (sibling to `our_cls`/`their_cls`/`value_cls`), holding a multimodal belief implicitly without
      ever materialising a phantom mean-mon.
    - **K>1** gives distinct per-slot queries that **coordinate** (the decoder's self-attention lets
      them attend to each other — "board is rock-leaning → I, the 2nd slot, lean rock too" = overload)
      and **read the board** (cross-attention to the 12 team tokens). For K=1 the self-attention is a
      benign near-identity.

    Output `[B, K*D_MODEL]` is concatenated into BOTH the policy and value projection inputs. Hard
    requires `attend_unrevealed_opponents`: with the hidden slots masked there is nothing for the
    belief to summarise. Untrained-capacity caveat: without a dedicated objective (B3 — species-ID /
    BYOL) the RL gradient only weakly shapes these queries; this module is the *structure* those
    objectives later attach to. See `designs/ai_v5/design_offense_and_opponent_belief.md` §B2."""

    def __init__(self, k: int):
        super().__init__()
        if k < 1:
            raise ValueError(f"HiddenOppBeliefPool needs k >= 1, got {k}")
        self.k = k
        # K distinct learned queries — non-identical by construction (independent init) so they do
        # not collapse the way identical zero-slots would. Same 0.02 init scale as the CLSPool queries.
        self.queries = torch.nn.Parameter(torch.randn(1, k, D_MODEL) * 0.02)
        # DETR-style decoder block: query self-attention (coordinate) → cross-attention to the 12
        # team tokens (read the board) → FFN. dropout=0.0 / post-LN match the encoder stack.
        self.decoder = torch.nn.TransformerDecoderLayer(
            d_model=D_MODEL, nhead=TRANSFORMER_N_HEADS, dim_feedforward=TRANSFORMER_FFN_DIM,
            dropout=0.0, activation="relu", batch_first=True, norm_first=False,
        )
        self.norm = torch.nn.LayerNorm(D_MODEL)

    def forward(self, all_team_out: torch.Tensor, all_fainted: torch.Tensor,
                batch_size: int) -> torch.Tensor:
        """all_team_out [B, 12, D_MODEL], all_fainted [B, 12] bool key-mask → [B, K*D_MODEL].

        `all_fainted` always has >=2 False entries (our + opp active are force-unmasked in
        ObsUnpack), so no memory row is fully masked → no attention NaN."""
        queries = self.queries.expand(batch_size, -1, -1)                         # [B, K, D_MODEL]
        belief = self.decoder(queries, all_team_out, memory_key_padding_mask=all_fainted)
        belief = self.norm(belief)                                                # [B, K, D_MODEL]
        return belief.reshape(batch_size, self.k * D_MODEL)                       # [B, K*D_MODEL]


class BeliefSlots(torch.nn.Module):
    """In-place hidden-opponent belief (the live design — supersedes the side-pool `HiddenOppBeliefPool`).

    Instead of summarising the hidden party into K side query tokens (a readout), this REPLACES the
    opponent's un-revealed team slots — which arrive at the encoder as all-zero placeholders (Gen 3
    has no team preview) — with K=TEAM_SIZE **distinct learned "unknown-mon" embeddings**, one per opp
    slot position. The believed mons then sit *in the lineup* and are refined by the SAME 12-token
    `TeamTransformer` and attended over by every downstream readout (`their_cls`, `value_cls`, the
    policy reasoning) — "the model thinks about the hidden mons in latent space" rather than reading a
    side summary.

    Why distinct per-slot params: a permutation-equivariant transformer maps identical inputs to
    identical outputs, so identical zero-slots collapse to one representation (the model can know
    "there are unknowns" but not "slot A leans physical sweeper, slot B special wall"). Independent
    init breaks that symmetry so the slots can specialise — the same trick `HiddenOppBeliefPool` used,
    done in-place. The refined believed tokens are supervised by `BeliefHead` (species + moves),
    which is what makes a slot actually *mean* a Skarmory-shaped wall instead of a generic blob.

    Requires `attend_unrevealed_opponents` (else the believed slots are key-masked out of the
    transformer and never refined). Off ⇒ this module is not built and the opp slots stay zeros
    (baseline arch, byte-for-byte). See `designs/ai_v5/design_offense_and_opponent_belief.md`."""

    def __init__(self):
        super().__init__()
        # One distinct learned token per opponent team-slot position. Same 0.02 init scale as the
        # CLS / belief queries. Slot position is the canonical order the aux labels are matched in.
        self.unknown_slot_emb = torch.nn.Parameter(torch.randn(TEAM_SIZE, D_MODEL) * 0.02)

    def forward(self, role_tokens: torch.Tensor, opp_believed_mask: torch.Tensor) -> torch.Tensor:
        """role_tokens [B, 12, D], opp_believed_mask [B, 6] bool → role_tokens with believed opp
        slots replaced by their learned unknown-token. Revealed (and revealed-then-fainted) opp slots
        keep their encoded token unchanged."""
        batch_size = role_tokens.shape[0]
        our_tokens = role_tokens[:, :TEAM_SIZE, :]
        opp_tokens = role_tokens[:, TEAM_SIZE:, :]                                    # [B, 6, D]
        unknown = self.unknown_slot_emb.unsqueeze(0).expand(batch_size, -1, -1)       # [B, 6, D]
        opp_tokens = torch.where(opp_believed_mask.unsqueeze(-1), unknown, opp_tokens)
        return torch.cat([our_tokens, opp_tokens], dim=1)


class BeliefHead(torch.nn.Module):
    """Auxiliary supervision for the in-place belief slots (the missing "B3" objective).

    Reads the post-transformer opponent team tokens and predicts, per slot, what the hidden mon IS:
    its **species** (cross-entropy) and its **moves** (multi-label BCE). Labels come free from the
    self-play env (it knows the opponent's full team); the loss (computed in `instrumented_ppo`)
    scores ONLY the believed slots, in `BeliefSlots`' canonical slot order. Role is implicit: a
    predicted species routes through the model's existing species/stat/type embeddings, which already
    encode wall-vs-sweeper — so "think Skarmory" supplies the role.

    Returns a dict of logits so a later BYOL/latent-matching target (regress the real hidden mon's
    encoded token) can be added as another key without touching the call sites — the agreed clean
    escalation path off the species+moves v1.

    **Latent escalation (`latent_dim` set, `--opp-belief-latent-coef>0`).** Adds an asymmetric
    SimSiam-style predictor MLP that maps the refined believed-slot token into the `pokemon_encoder`
    role-token space. A cosine loss (`instrumented_ppo`) regresses it toward the STOP-GRAD encoder
    role-token of the TRUE hidden mon — GRADED identity supervision (a "similar wall" is less wrong)
    the hard species CE can't give, in the role geometry a representation probe found the encoder
    amplifies ~7.5×. The discrete species head stays as the banked fallback; the predictor's asymmetry
    + the target encoder being TASK-ANCHORED (shared `pokemon_encoder`, stop-grad) defuses collapse
    without an EMA (a VICReg variance floor + a `latent_std` monitor are the belt-and-braces)."""

    def __init__(self, n_species: int, n_moves: int, latent_dim: "Optional[int]" = None):
        super().__init__()
        self.norm = torch.nn.LayerNorm(D_MODEL)
        self.species_head = torch.nn.Linear(D_MODEL, n_species)
        self.moves_head = torch.nn.Linear(D_MODEL, n_moves)
        self.latent_head = None
        if latent_dim is not None:
            # Asymmetric predictor (own LayerNorm → bottleneck MLP) onto the role-token space.
            self.latent_head = torch.nn.Sequential(
                torch.nn.LayerNorm(D_MODEL),
                torch.nn.Linear(D_MODEL, D_MODEL),
                torch.nn.ReLU(),
                torch.nn.Linear(D_MODEL, latent_dim),
            )

    def forward(self, their_team_out: torch.Tensor) -> Dict[str, torch.Tensor]:
        """their_team_out [B, 6, D] → {"species": [B,6,n_species], "moves": [B,6,n_moves],
        ["latent": [B,6,latent_dim]]}. The latent key is present only when the predictor is built."""
        h = self.norm(their_team_out)
        out = {"species": self.species_head(h), "moves": self.moves_head(h)}
        if self.latent_head is not None:
            out["latent"] = self.latent_head(their_team_out)
        return out


class MoveBelief(torch.nn.Module):
    """Predicts each opponent slot's full MOVESET and REINJECTS that prediction back into the slot
    token, so the believed moves actually FLOW into the representation the policy/value heads read —
    not a dead-end aux readout. This is the "make it meaningful" mechanism.

    Per opp slot: a multi-label move head predicts the moveset; the predicted distribution is
    soft-embedded (`sigmoid(logits) @ move_embedding` — the expected moveset embedding), projected
    back to token space, and ADDED to the slot token (a residual, gated to the slots the mode selects).
    The enriched tokens then feed the CLS pools, so the policy reasons about the believed moves. The
    reinject projection is small-init so the enrichment starts ≈0 (no harm before the prediction
    sharpens). The move head's logits are stashed for the aux loss, which supervises against the real
    moveset (revealed slots → direct; hidden slots → Hungarian). `mode` selects which slots are
    enriched + scored: 'revealed' (seen species — predict its UNREVEALED moves, the surprise-OHKO
    new-move gap), 'unrevealed' (hidden species), or 'both'."""

    def __init__(self, n_moves: int, move_emb_dim: int):
        super().__init__()
        self.move_head = torch.nn.Linear(D_MODEL, n_moves)
        self.reinject = torch.nn.Linear(move_emb_dim, D_MODEL)
        torch.nn.init.normal_(self.reinject.weight, std=0.02)   # start the enrichment ≈0
        torch.nn.init.zeros_(self.reinject.bias)
        self.norm = torch.nn.LayerNorm(D_MODEL)

    def forward(self, opp_tokens: torch.Tensor, apply_mask: torch.Tensor,
                move_embedding: torch.nn.Embedding) -> Tuple[torch.Tensor, torch.Tensor]:
        """opp_tokens [B,6,D], apply_mask [B,6] bool (which slots get the belief), move_embedding table
        → (enriched_tokens [B,6,D], move_logits [B,6,M]). The enrichment is residual + gated to the
        selected slots, so unselected slots pass through unchanged."""
        move_logits = self.move_head(opp_tokens)                                 # [B, 6, M]
        soft_emb = torch.sigmoid(move_logits) @ move_embedding.weight             # [B, 6, move_emb]
        enriched = opp_tokens + apply_mask.unsqueeze(-1) * self.reinject(soft_emb)
        return self.norm(enriched), move_logits


# Differentiable damage operator (`DamageOperator`) constants.
_DMG_BETA = 8.0            # soft-max-over-candidates temperature (→ hard max as β→∞)
_DMG_FEATS = 4            # per-defender features: [phys_chip, spec_chip, phys_pko, spec_pko]


class DamageOperator(torch.nn.Module):
    """Fixed, differentiable gen3 damage calculator run in the GPU forward pass, fed by the
    move-belief head's PREDICTED moves — the "compute the physics, learn the belief" op
    (`designs/ai_v6/design_differentiable_damage_op.md`).

    For the opponent ACTIVE mon (always a revealed mon under move-belief mode revealed/both), it
    computes the incoming damage its *believed* moveset would deal to each of our 6 mons, then
    aggregates per (defender, gen3-type-channel) into the believed-move threat. The damage is a
    differentiable function of the move-belief weights `w_m = sigmoid(last_move_belief_logits[active])`,
    so the gradient sharpens the belief toward the moves that actually threaten KOs. This replaces
    the FIXED usage-prior the CPU `incoming_damage.py` block must use (the belief doesn't exist at
    obs-build time) with the model's LEARNED belief.

    Per defender d, per channel (physical / special — the gen3 TYPE split), two features:
      chip = soft-max over that channel's believed candidates of `w·dmg_frac` (the damage fraction of
             the most-threatening believed move — mirrors `incoming_damage`'s max-over-candidates, made
             differentiable via a temperature soft-max instead of a hard max);
      pko  = soft-max of `w·P(KO|move)` (P(KO) = a clamped roll-band ramp, the continuous limit of the
             16-roll integration).
    Hidden Power (all 17 variants collide on num=237 → unrepresentable on the num axis) is expanded
    into 16 TYPED candidates (BP 70 = gen3 max), weighted `P(HP present)·P(type)` — presence from the
    move belief (`w[237]`), type from the per-mon `hp_probs` obs block (the HP tracker's narrowed
    distribution) — so HP Grass vs HP Ice get distinct type effectiveness.

    Stats: our defenders use their REAL spread (IVs/EVs/nature reconstructed from the obs spread block —
    they are revealed); the hidden-spread attacker uses a fixed de-timid offensive assumption (252 EV,
    31 IV, +nature ×1.1), mirroring `incoming_damage`'s offensive-stat tail. The smooth (un-floored)
    L100 stat + damage formula keeps everything differentiable (the byte-exact floored kernel is the
    proof's; the forward only needs the gradient).

    Leak-safe: reads only the PREDICTED belief + public obs (our HP/types, the opp active's revealed
    species/types) — never a privileged label. Output `[B, 6*_DMG_FEATS]` is appended to BOTH
    projection heads. Zeroed (incl. gradient) when there is no opponent active and per fainted defender.
    Lookup tables are registered as non-persistent float32 buffers (pure physics, recomputable from
    `data/`)."""

    n_feats = _DMG_FEATS

    def __init__(self, layout: Dict[str, Any]):
        super().__init__()
        from agents.model.damage_tables import (
            build_damage_buffers, HIDDEN_POWER_NUM, HIDDEN_POWER_BP,
        )
        bufs = build_damage_buffers(layout['max_moves'], layout['max_species'])
        for name, tensor in bufs.items():
            # Non-persistent: deterministic physics from data/, not learned weights → keep them out of
            # every checkpoint (and out of the state_dict, so a load never demands them).
            self.register_buffer(name, tensor, persistent=False)
        self.hp_num = HIDDEN_POWER_NUM
        self.hp_bp = float(HIDDEN_POWER_BP)

    def _softmax_max(self, score: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
        """Soft (differentiable) max over a channel's candidates: `Σ_c softmax(β·score)·score` with
        off-channel candidates masked out of BOTH the softmax and the value. `score` [B,6,C],
        `channel_mask` [1,1,C] (1=on-channel). Returns [B,6]. Bounded by max(score); no /0; the
        off-channel −1e9 underflows the softmax to exactly 0 (softmax subtracts its max → stable)."""
        masked = _DMG_BETA * score + (1.0 - channel_mask) * (-1e9)
        weights = torch.softmax(masked, dim=-1)
        return (weights * score * channel_mask).sum(dim=-1)

    def forward(self, ctx: 'ExtractorContext', move_belief_logits: torch.Tensor) -> torch.Tensor:
        B = ctx.batch_size
        device = ctx.device
        eps = 1e-6
        ar = torch.arange(B, device=device)
        opp_act = TEAM_SIZE + ctx.opp_active_local                         # [B] global opp-active slot

        # No-opp-active gate (forced switch / battle start / dummy zero-obs): zero the whole block.
        has_opp = ctx.hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, -1].any(dim=1).float()  # [B]

        # --- Attacker = opp active (revealed species; hidden spread → fixed 252 EV / 31 IV / ×1.1) ---
        a_base = self.BASE_STATS[ctx.species_ids[ar, opp_act]]            # [B,6] [hp,atk,def,spa,spd,spe]
        off_const = 31.0 + 252.0 / 4.0 + 5.0                              # IV + EV/4 + 5
        atk = (2.0 * a_base[:, 1] + off_const) * 1.1                      # [B]
        spa = (2.0 * a_base[:, 3] + off_const) * 1.1                      # [B]
        at1 = ctx.type1_ids[ar, opp_act]                                 # [B] TypeEncoder axis
        at2 = ctx.type2_ids[ar, opp_act]

        # --- Defenders = our 6 mons (revealed → REAL spread reconstructed from the obs) ---
        d_base = self.BASE_STATS[ctx.species_ids[:, :TEAM_SIZE]]          # [B,6,6]
        spread = ctx.pokemon_part[:, :TEAM_SIZE,
                                  POKEMON_SPREAD_OFFSET:POKEMON_SPREAD_OFFSET + POKEMON_SPREAD_DIM]
        iv = spread[..., 0:6] * 31.0                                      # [B,6,6] [hp,atk,def,spa,spd,spe]
        ev = spread[..., 6:12] * 252.0
        nat = spread[..., 13:18]                                          # [B,6,5] [atk,def,spa,spd,spe]
        def_stat = (2.0 * d_base[..., 2] + iv[..., 2] + ev[..., 2] / 4.0 + 5.0) * nat[..., 1]   # [B,6]
        spd_stat = (2.0 * d_base[..., 4] + iv[..., 4] + ev[..., 4] / 4.0 + 5.0) * nat[..., 3]   # [B,6]
        maxhp = 2.0 * d_base[..., 0] + iv[..., 0] + ev[..., 0] / 4.0 + 110.0                    # [B,6]
        hp_frac = ctx.hp_and_active[:, :TEAM_SIZE, 0]                     # [B,6]
        cur_hp = hp_frac * maxhp                                          # [B,6]
        defender_alive = (hp_frac > 0).float()                           # [B,6]
        t1d = ctx.type1_ids[:, :TEAM_SIZE]                               # [B,6]
        t2d = ctx.type2_ids[:, :TEAM_SIZE]

        # --- Belief weights at the opp-active slot + the typed-HP candidate weights ---
        w = torch.sigmoid(move_belief_logits[ar, ctx.opp_active_local])   # [B, n_moves]
        w_hp = torch.sigmoid(move_belief_logits[ar, ctx.opp_active_local, self.hp_num])  # [B] P(HP present)
        hp_type_belief = ctx.hp_probs[ar, opp_act]                       # [B,16] P(HP type | present)

        # --- Candidate set: the num-indexed moves + 16 typed Hidden Powers ---
        n_hp = self.HP_TYPE_IDX.shape[0]
        bp_all = torch.cat([self.MOVE_BP, torch.full((n_hp,), self.hp_bp, device=device)])      # [C]
        mty_all = torch.cat([self.MOVE_TYPE_IDX, self.HP_TYPE_IDX])                             # [C]
        phys_all = torch.cat([self.MOVE_PHYS, self.HP_IS_PHYS])                                 # [C]
        w_all = torch.cat([w, w_hp.unsqueeze(-1) * hp_type_belief], dim=1)                      # [B,C]

        # --- gen3 damage per (defender, candidate), all differentiable in w ---
        eff = self.CHART[t1d][..., mty_all] * self.CHART[t2d][..., mty_all]                     # [B,6,C]
        A = phys_all * atk[:, None] + (1.0 - phys_all) * spa[:, None]                           # [B,C]
        D = phys_all[None, None, :] * def_stat[:, :, None] \
            + (1.0 - phys_all)[None, None, :] * spd_stat[:, :, None]                            # [B,6,C]
        is_stab = ((mty_all[None, :] == at1[:, None]) | (mty_all[None, :] == at2[:, None])).float()  # [B,C]
        stab = 1.0 + 0.5 * is_stab                                                              # [B,C]
        core = 42.0 * bp_all[None, None, :] * A[:, None, :] / (D + eps) / 50.0 + 2.0            # [B,6,C]
        dmg = core * stab[:, None, :] * eff * 0.925                                             # [B,6,C]
        dmg = dmg * (bp_all > 0).float()[None, None, :]                  # kill the +2 floor on BP-0 moves
        frac = dmg / (maxhp[:, :, None] + eps)                                                  # [B,6,C]
        # P(KO|move): clamped roll-band ramp — the continuous limit of the 16-roll integration.
        ko_ramp = torch.clamp((dmg - cur_hp[:, :, None]) / (0.15 * dmg + eps), 0.0, 1.0)        # [B,6,C]

        # --- aggregate per (defender, channel): soft-max of belief-weighted damage / KO ---
        wf = w_all[:, None, :] * frac                                    # [B,6,C] (broadcast w over defenders)
        wk = w_all[:, None, :] * ko_ramp
        phys_mask = phys_all[None, None, :]
        spec_mask = 1.0 - phys_mask
        phys_chip = self._softmax_max(wf, phys_mask)                     # [B,6]
        spec_chip = self._softmax_max(wf, spec_mask)
        phys_pko = self._softmax_max(wk, phys_mask)
        spec_pko = self._softmax_max(wk, spec_mask)

        feats = torch.stack([phys_chip, spec_chip, phys_pko, spec_pko], dim=-1)                 # [B,6,4]
        feats = feats * defender_alive[:, :, None] * has_opp[:, None, None]                     # gates
        return feats.reshape(B, TEAM_SIZE * self.n_feats)               # [B, 6*_DMG_FEATS]


class ProjectionAssembler(torch.nn.Module):
    """Assembles the pre-projection inputs for BOTH heads.

    Policy input: team pools + our active token + per-side encoded active contexts + the
    non-matchup scalar tail (unchanged from the single-head design).
    Value input: the value-dedicated pool + the same per-side encoded active contexts +
    non-matchup scalar tail. The active-context encoder and the raw global scalars are
    shared inputs (not the contested body representation), so reusing them for both heads
    is parameter-efficient; the value head's distinct signal comes from `value_pooled`.
    """

    def __init__(self, layout: Dict[str, Any], value_active_readout: bool = False):
        super().__init__()
        active_ctx_dim = layout['active_context_dim']
        self.active_ctx_encoder = torch.nn.Sequential(
            torch.nn.Linear(active_ctx_dim, ACTIVE_CTX_HIDDEN[0]),
            torch.nn.ReLU(),
            torch.nn.Linear(ACTIVE_CTX_HIDDEN[0], ACTIVE_CTX_HIDDEN[1]),
        )
        # When True, the value head ALSO reads our_active_refined (the active mon's refined token).
        # The dual-head (Option C) value readout pools the whole board (value_pooled) but DROPS the
        # active-mon view the policy head keeps — a probe on a real checkpoint found the value rep
        # predicts an incoming self-KO at AUC 0.79 vs the policy rep's 0.90 (and ≈ the raw-obs-linear
        # 0.77, i.e. the critic isn't using the trunk's nonlinear KO reasoning). A critic blind to
        # incoming KOs over-values pre-KO states → the V-tail crater. Off by default (clean A/B).
        self.value_active_readout = value_active_readout

    def forward(self, our_team_pooled: torch.Tensor, their_team_pooled: torch.Tensor,
                our_active_refined: torch.Tensor, value_pooled: torch.Tensor,
                ctx: ExtractorContext,
                hidden_opp_belief: Optional[torch.Tensor] = None,
                damage_block: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        our_ctx_enc = self.active_ctx_encoder(ctx.our_ctx_raw)                      # [B, 32]
        opp_ctx_enc = self.active_ctx_encoder(ctx.opp_ctx_raw)                      # [B, 32]
        pi_parts = [our_team_pooled, their_team_pooled, our_active_refined,
                    our_ctx_enc, opp_ctx_enc, ctx.non_matchup_rest]
        vf_parts = [value_pooled, our_ctx_enc, opp_ctx_enc, ctx.non_matchup_rest]
        # Give the critic the active-mon readout it structurally lacked (see __init__): routes the
        # trunk's nonlinear incoming-KO/threat reasoning into the value head so it can price the tail.
        if self.value_active_readout:
            vf_parts.append(our_active_refined)
        # Hidden-opponent belief (flag-guarded; None when off) feeds BOTH heads — the policy reads
        # the threat over the hidden team, the value reads its winning-ness. Appended last so the
        # off-by-default block layout is unchanged (the dummy forward auto-sizes the projections).
        if hidden_opp_belief is not None:
            pi_parts.append(hidden_opp_belief)
            vf_parts.append(hidden_opp_belief)
        # Differentiable damage operator (flag-guarded; None when off): the believed-move incoming
        # damage to each of our mons, fed to BOTH heads (policy: which threat to dodge; value: price the
        # KO tail). Appended last, after the belief, so off-by-default block layouts are unchanged.
        if damage_block is not None:
            pi_parts.append(damage_block)
            vf_parts.append(damage_block)
        pi_combined = torch.cat(pi_parts, dim=1)
        vf_combined = torch.cat(vf_parts, dim=1)
        return pi_combined, vf_combined


class Gen3FeaturesExtractor(torch.nn.Module):
    """Orchestrates the phase modules. Data flow:
        ObsUnpack → PokemonEncoder → TeamTransformer → CLSPool → [HiddenOppBeliefPool?] → ProjectionAssembler
    then a final pre-projection LayerNorm + Linear + ReLU head. `HiddenOppBeliefPool` is built only
    when `--opp-belief-cls-k > 0` (else `None`); when present its belief feeds both projection heads. The
    embedding tables live in `self.embeddings` (shared) and are passed into the phases that need them.
    See `src/agents/model/CLAUDE.md` for the phase-module contract."""

    def __init__(self, observation_space: spaces.Dict, layout: Optional[Dict[str, Any]] = None,
                 mappings: Optional[Dict[str, Any]] = None, log_level: LogLevel = LogLevel.QUIET,
                 attend_unrevealed_opponents: bool = False, opp_belief_cls_k: int = 0,
                 value_active_readout: bool = False, opp_belief_slots: bool = False,
                 move_belief_mode: str = "off", opp_belief_latent: bool = False,
                 damage_op: bool = False):
        super().__init__()
        self.layout = layout
        self.mappings = mappings
        self.log_level = log_level
        # Behavioral toggle (no weight-shape change): unmask the opponent's still-hidden
        # party so the transformer attends to it. Version-checked, not in ARCH_SIGNATURE.
        self.attend_unrevealed_opponents = attend_unrevealed_opponents
        # Hidden-opponent belief: opp_belief_cls_k = number of learned belief query tokens.
        # 0 = OFF (no module, baseline arch — reproduces it byte-for-byte, so no ARCH_SIGNATURE bump);
        # k>0 builds HiddenOppBeliefPool(k) and widens both projection inputs by k*D_MODEL (a
        # WEIGHT-SHAPE change, version-checked like use_popart). k>0 hard-requires the unmask flag:
        # with the hidden slots masked the belief queries would read a board with them deleted.
        if opp_belief_cls_k < 0:
            raise ValueError(f"opp_belief_cls_k must be >= 0 (0 = off), got {opp_belief_cls_k}")
        self.opp_belief_cls_k = opp_belief_cls_k
        if opp_belief_cls_k > 0 and not attend_unrevealed_opponents:
            raise ValueError(
                "opp_belief_cls_k > 0 requires attend_unrevealed_opponents=True — the hidden-opponent "
                "belief queries read the unrevealed opp slots, which are key-masked out unless the "
                "unmask flag is on. Enable --attend-unrevealed-opponents, or set --opp-belief-cls-k 0."
            )

        # Phase modules (constructed before the dummy forward below).
        self.embeddings = Embeddings(layout)
        self.unpack = ObsUnpack(layout, attend_unrevealed_opponents=attend_unrevealed_opponents)
        self.pokemon_encoder = PokemonEncoder(layout)
        self.team_transformer = TeamTransformer(layout)
        self.cls_pool = CLSPool(layout)
        self.hidden_opp_belief = HiddenOppBeliefPool(opp_belief_cls_k) if opp_belief_cls_k > 0 else None
        # In-place hidden-opponent belief (the live design): distinct learned unknown-mon tokens fill
        # the un-revealed opp slots + a species/moves aux head supervises them. OFF reproduces the
        # baseline arch byte-for-byte (no module, opp slots stay zeros). k>0 the side-pool and this are
        # independent flags; the in-place path supersedes the pool. Hard-requires the unmask flag:
        # masked believed slots would never be refined by the transformer.
        self.opp_belief_slots = opp_belief_slots
        if opp_belief_slots and not attend_unrevealed_opponents:
            raise ValueError(
                "opp_belief_slots=True requires attend_unrevealed_opponents=True — the in-place "
                "belief tokens fill the un-revealed opp slots, which are key-masked out of the "
                "transformer unless the unmask flag is on. Enable --attend-unrevealed-opponents."
            )
        # Latent-belief escalation (the BYOL/SimSiam target): adds an asymmetric predictor to
        # BeliefHead that regresses each believed slot's refined token toward the STOP-GRAD
        # pokemon_encoder role-token of the TRUE hidden mon (computed in forward_internal from the
        # training-only `belief_target_slots` obs key). A state_dict change (extra predictor params),
        # gated in check_compatible like opp_belief_slots; OFF = byte-for-byte baseline. Requires
        # opp_belief_slots (the believed slots + BeliefHead must exist to attach the predictor).
        self.opp_belief_latent = opp_belief_latent
        if opp_belief_latent and not opp_belief_slots:
            raise ValueError(
                "opp_belief_latent=True requires opp_belief_slots=True — the latent predictor attaches "
                "to the BeliefHead over the in-place believed slots. Enable --opp-belief-aux-coef>0 "
                "(which turns on opp_belief_slots), or set --opp-belief-latent-coef 0."
            )
        self.belief_slots = BeliefSlots() if opp_belief_slots else None
        self.belief_head = (
            BeliefHead(layout['max_species'], layout['max_moves'],
                       latent_dim=(D_MODEL if opp_belief_latent else None)) if opp_belief_slots else None
        )
        # Stashed each forward when belief is on (the species/moves[/latent] logits dict, or None);
        # read by the vendored PPO train loop to add the aux loss. Carries grad — read+used in the same
        # backward graph as the forward that produced it (per minibatch). See instrumented_ppo.
        self.last_belief_logits: Optional[Dict[str, torch.Tensor]] = None
        # Stashed STOP-GRAD latent target [B,6,D] (encoder role-tokens of the true hidden mons) when
        # opp_belief_latent is on AND the privileged `belief_target_slots` key is present (training
        # only). None otherwise. Read ONLY by the latent aux loss — NEVER fed into pi/vf (no leak).
        self.last_belief_target_latent: Optional[torch.Tensor] = None
        # Move belief (flag-guarded): predict + REINJECT the opp moveset into the slot tokens so the
        # believed moves flow into the policy/value readout. mode ∈ {off, revealed, unrevealed, both}
        # selects which opp slots are enriched + scored. OFF reproduces the baseline arch byte-for-byte.
        if move_belief_mode not in ("off", "revealed", "unrevealed", "both"):
            raise ValueError(f"move_belief_mode must be off|revealed|unrevealed|both, got {move_belief_mode!r}")
        self.move_belief_mode = move_belief_mode
        if move_belief_mode != "off" and not attend_unrevealed_opponents:
            raise ValueError(
                "move_belief_mode != off requires attend_unrevealed_opponents=True — the move belief "
                "reads/enriches the opp slots (incl. hidden ones), which are key-masked unless the "
                "unmask flag is on. Enable --attend-unrevealed-opponents."
            )
        self.move_belief = (
            MoveBelief(layout['max_moves'], layout['move_embedding_dim']) if move_belief_mode != "off" else None
        )
        self.last_move_belief_logits: Optional[torch.Tensor] = None
        # Differentiable damage operator (flag-guarded): consumes the move belief's PREDICTED moves for
        # the opp active and computes the believed-move incoming damage to each of our mons, fed to BOTH
        # heads. OFF reproduces the baseline arch byte-for-byte (no module, projection widths unchanged).
        # Requires a move-belief mode that SCORES the opp active (a revealed mon): revealed|both. Under
        # off/unrevealed the active-slot logits are unsupervised, so the belief gradient story breaks.
        self.damage_op_enabled = damage_op
        if damage_op and move_belief_mode not in ("revealed", "both"):
            raise ValueError(
                "damage_op=True requires move_belief_mode in {revealed, both} — the operator reads the "
                "opp ACTIVE slot's predicted move logits, which are only supervised/reinjected for a "
                "revealed mon. Set --move-belief-mode revealed (or both), or disable --damage-op."
            )
        self.damage_op = DamageOperator(layout) if damage_op else None
        # Value-head active readout (weight-shape via flag): adds our_active_refined (D_MODEL) to the
        # value projection. OFF reproduces the baseline value head byte-for-byte (no ARCH_SIGNATURE bump).
        self.value_active_readout = value_active_readout
        self.assembler = ProjectionAssembler(layout, value_active_readout=value_active_readout)

        self.role_token_size = ROLE_TOKEN_SIZE

        # Discover the policy/value projection-input dims via a dummy forward through the
        # assembled phases (the assembler returns a (pi_combined, vf_combined) pair).
        with torch.no_grad():
            dummy_obs = torch.zeros((1, layout['total_dim']))
            pi_sample, vf_sample = self.forward_internal({"observation": dummy_obs})
            self.projection_input_dim = pi_sample.shape[1]
            self.value_projection_input_dim = vf_sample.shape[1]

        # Two projection heads, both → PROJECTION_DIM. Pre-projection LayerNorm equalises
        # per-block scales. The value head reads the value-dedicated CLS pool (Option C):
        # the transformer body is shared, but policy and value are summarised + projected
        # through independent paths so the critic isn't fighting the actor over the readout.
        self.projection_dim = PROJECTION_DIM
        self.pre_proj_norm = torch.nn.LayerNorm(self.projection_input_dim)
        self.projection = torch.nn.Linear(self.projection_input_dim, self.projection_dim)
        self.value_pre_norm = torch.nn.LayerNorm(self.value_projection_input_dim)
        self.value_projection = torch.nn.Linear(self.value_projection_input_dim, self.projection_dim)
        self.activation = torch.nn.ReLU()
        # Both heads emit PROJECTION_DIM; SB3 sizes the shared mlp_extractor from this.
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

    def _belief_target_role_tokens(self, ctx: ExtractorContext, target_slots: torch.Tensor) -> torch.Tensor:
        """The latent head's regression TARGET: run the model's OWN `pokemon_encoder` over a privileged
        12-slot block = [live our-team, true hidden-opp-team] and return the opp-half role tokens
        [B, 6, D], DETACHED.

        SimSiam stop-grad: the target encoder IS the shared, task-anchored `pokemon_encoder` (no EMA,
        no collapse — the main losses keep Aero≠Blissey distinct). `target_slots` [B,6,POKEMON_FULL_DIM]
        are the env's fresh per-mon identity encodes (PAD slots zeros). The live ctx supplies the
        context (global / matchups / masks); a believed opp slot's live matchups are already neutral
        (it is hidden), so its target role-token is a clean identity encode. The result is read ONLY by
        the latent aux loss — it is never concatenated into pi/vf, so the privileged future cannot
        reach the policy/value output (leak-safe)."""
        our_part = ctx.pokemon_part[:, :TEAM_SIZE, :]                      # [B, 6, FULL] live our team
        priv_part = torch.cat([our_part, target_slots.to(our_part.dtype)], dim=1)   # [B, 12, FULL]
        ids = slice_pokemon_categoricals(priv_part, self.layout)
        priv_ctx = replace(ctx, pokemon_part=priv_part, **ids)
        with torch.no_grad():
            priv_role = self.pokemon_encoder(priv_ctx, self.embeddings)   # [B, 12, D]
        return priv_role[:, TEAM_SIZE:, :].detach()                       # [B, 6, D] opp half (targets)

    def forward_internal(self, obs):
        """Build the (pi_combined, vf_combined) pre-projection pair by chaining the phases."""
        ctx = self.unpack(obs)
        role_tokens = self.pokemon_encoder(ctx, self.embeddings)
        # In-place hidden-opponent belief: replace the un-revealed opp slots with distinct learned
        # unknown-mon tokens BEFORE the transformer, so the body refines them and every readout
        # attends over them as party members (flag-guarded; None ⇒ baseline zeros).
        if self.belief_slots is not None:
            role_tokens = self.belief_slots(role_tokens, ctx.opp_believed_mask)
        our_team_out, their_team_out = self.team_transformer(role_tokens, ctx, self.embeddings)
        # Aux belief logits over the refined opp tokens — stashed for the PPO aux loss, NOT fed back
        # into the policy/value path (labels would leak). None when belief is off.
        self.last_belief_logits = (
            self.belief_head(their_team_out) if self.belief_head is not None else None
        )
        # Latent-belief target: the STOP-GRAD pokemon_encoder role-tokens of the TRUE hidden mons,
        # computed only when the latent head is on AND the privileged `belief_target_slots` key is in
        # the obs (training only; absent in the __init__ dummy forward + eval/inference). A side branch
        # — its result is stashed for the loss and NEVER concatenated into pi/vf, so the future cannot
        # reach the policy/value output.
        # Gated on torch.is_grad_enabled() so the second pokemon_encoder pass runs ONLY in the
        # backward-needing path (train()'s evaluate_actions), where the latent loss consumes it — not
        # during no-grad rollout/eval/inference action selection (a free per-step saving; the same
        # is_grad_enabled gate the grad-checkpointing path uses).
        self.last_belief_target_latent = None
        if self.opp_belief_latent and self.belief_head is not None and torch.is_grad_enabled():
            target_slots = obs.get("belief_target_slots")
            if target_slots is not None:
                self.last_belief_target_latent = self._belief_target_role_tokens(ctx, target_slots)
        # Move belief: predict each opp slot's moveset and REINJECT it into the slot token (flow-through)
        # so the believed moves reach the CLS pools → both heads. Mode selects which slots: revealed
        # mons, unrevealed (hidden) mons, or both. The move logits are stashed for the aux loss.
        if self.move_belief is not None:
            if self.move_belief_mode == "revealed":
                mb_mask = ~ctx.opp_believed_mask                 # revealed-species slots
            elif self.move_belief_mode == "unrevealed":
                mb_mask = ctx.opp_believed_mask                  # hidden-species slots
            else:                                                # "both"
                mb_mask = torch.ones_like(ctx.opp_believed_mask)
            their_team_out, self.last_move_belief_logits = self.move_belief(
                their_team_out, mb_mask, self.embeddings.move_embedding)
        else:
            self.last_move_belief_logits = None
        our_team_pooled, their_team_pooled, our_active_refined, value_pooled = self.cls_pool(
            our_team_out, their_team_out, ctx
        )
        belief = None
        if self.hidden_opp_belief is not None:
            # Same 12-token memory + the single-sourced ctx.all_fainted key-mask the value CLS pools
            # over (all_team_out is a forward activation, cheap to recompute; the MASK carries the
            # NaN-safety invariant and is single-sourced on the context).
            all_team_out = torch.cat([our_team_out, their_team_out], dim=1)                 # [B, 12, D]
            belief = self.hidden_opp_belief(all_team_out, ctx.all_fainted, ctx.batch_size)
        # Differentiable damage op (flag-guarded; None when off): fed the move belief's PREDICTED moves
        # for the opp active (set just above). Forward-only — leak-free (reads the prediction + public
        # obs); its gradient flows back into the move-belief head via last_move_belief_logits.
        damage_block = None
        if self.damage_op is not None:
            damage_block = self.damage_op(ctx, self.last_move_belief_logits)
        return self.assembler(our_team_pooled, their_team_pooled, our_active_refined, value_pooled,
                              ctx, belief, damage_block)

    def forward(self, obs):
        """Returns a (pi_features, vf_features) tuple — both [B, PROJECTION_DIM].

        The consuming policy (`Gen3DualHeadMaskablePolicy`) unpacks the tuple and routes
        each half to its own mlp_extractor branch. Standard SB3 policies expect a single
        tensor, so this extractor MUST be paired with that custom policy.
        """
        pi_combined, vf_combined = self.forward_internal(obs)
        if self._debugger is not None:
            self._debugger.on_forward(obs["observation"])
        pi_features = self.activation(self.projection(self.pre_proj_norm(pi_combined)))
        vf_features = self.activation(self.value_projection(self.value_pre_norm(vf_combined)))
        return pi_features, vf_features
