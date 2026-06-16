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
    POKEMON_CONDITION_OFFSET,
    POKEMON_SLEEP_BELIEF_OFFSET,
    INCOMING_DMG_OFFSET,
    INCOMING_DMG_DIM,
)
from agents.observation.moves import HIDDEN_POWER_MOVE_NUM
from agents.model.damage_tables import N_SECONDARY as _N_SECONDARY, SECONDARY_COLS as _SECONDARY_COLS
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
# gen3_unified_move_system_v1: context-free per-move LATENT (MoveLatentEncoder) — a mechanics-grounded
# move identity (move/type embeddings ⊕ MOVE_ATTR), routed into the move network AND used as the
# similarity-grading target so Rock Slide ≈ Hidden Power Rock. Flag-gated (`move_latent`); OFF leaves the
# move network byte-identical.
MOVE_LATENT_HIDDEN = 64           # hidden width of the MoveLatentEncoder MLP
MOVE_LATENT_DIM = 32              # output dim of the per-move latent (grading is cosine in this space)
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

    def __init__(self, layout: Dict[str, Any], attend_unrevealed_opponents: bool = False,
                 mask_incoming_damage_obs: bool = False,
                 mask_active_move_scalars_obs: bool = False, mask_move_effects_obs: bool = False):
        super().__init__()
        self.layout = layout
        # When True, UNREVEALED opp slots (species_known==0, hp filled as 0 — Gen 3 has no
        # team preview, so unseen party mons arrive here as all-zero placeholders) stay
        # ATTENDABLE in the transformer instead of being key-masked identically to fainted
        # mons. Lets the body reason about the still-hidden enemy team. Off = baseline.
        self.attend_unrevealed_opponents = attend_unrevealed_opponents
        # When True, ZERO the 51-dim incoming-damage / OHKO belief block out of the obs the MODEL reads
        # (ablation-by-zeroing — the block STAYS in the obs vector at a fixed dim, and the reward PBRS
        # still reads it from `live_view`). Use with the unified DamageOperator to A/B whether the
        # learned belief→damage op replaces the CPU usage-prior collapse — without deleting any code.
        self.mask_incoming_damage_obs = mask_incoming_damage_obs
        # gen3_unified_spread_belief_v1 (the --unified-obs disable-redundant flag): two more obs regions the
        # unified GPU path now subsumes, zeroed from the MODEL's view (block stays in the obs; reward PBRS
        # untouched). mask_active_move_scalars_obs zeros the active-move power+multiplier scalars (8 dims,
        # subsumed by the op's OUTGOING per-move damage — so it requires damage_outgoing); mask_move_effects
        # zeros the 44-dim move-effect block (subsumed by MOVE_ATTR/the move latent + the op effect axes).
        self.mask_active_move_scalars_obs = mask_active_move_scalars_obs
        self.mask_move_effects_obs = mask_move_effects_obs

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
        # --unified-obs disable-redundant masks: zero each now-GPU-subsumed obs region from the MODEL's view.
        # Clone ONCE if any mask is set (non_matchup_rest is a view of the obs — never mutate the shared
        # input). Offsets are derived from named reactive_layout entries (never hardcoded). The reward PBRS
        # reads these from live_view, so it is unaffected.
        if self.mask_incoming_damage_obs or self.mask_active_move_scalars_obs or self.mask_move_effects_obs:
            non_matchup_rest = non_matchup_rest.clone()

            def _zero_region(off: int, dim: int) -> None:
                s = reactive_start + off - 2 * active_ctx_dim
                non_matchup_rest[:, s:s + dim] = 0.0
            if self.mask_incoming_damage_obs:                  # 51-dim incoming-damage belief → DamageOperator
                _zero_region(INCOMING_DMG_OFFSET, INCOMING_DMG_DIM)
            if self.mask_active_move_scalars_obs:              # active-move power(4)+multiplier(4) → op outgoing
                _mp = reactive_layout['move_power']; _mm = reactive_layout['move_multiplier']
                _zero_region(_mp['offset'], _mp['dim'] + _mm['dim'])
            if self.mask_move_effects_obs:                     # 44-dim move-effect block → MOVE_ATTR/move latent
                _me = reactive_layout['move_effects']
                _zero_region(_me['offset'], _me['dim'])

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


class MoveLatentEncoder(torch.nn.Module):
    """Context-free per-move LATENT (gen3_unified_move_system_v1) — a mechanics-grounded move identity:
    ``MLP(concat(move_embedding(id), type_embedding(type), MOVE_ATTR[id])) → MOVE_LATENT_DIM``. Because the
    structured MOVE_ATTR (BP / category / accuracy / priority / drain / per-status secondary chances /
    utility flags) dominates, mechanically-similar moves land near each other — so Rock Slide ≈ Hidden
    Power Rock. Two uses, ONE MLP:
      - ``forward(...)``: per-move-SLOT latent (resolved type incl. the live HP type), concatenated into
        the move network so policy + value read a richer move representation.
      - ``latent_table(...)``: the ``[n_moves, MOVE_LATENT_DIM]`` table over canonical types — the
        stop-grad similarity-grading TARGET for the move-belief latent aux (Stage 3).
    MOVE_ATTR + canonical MOVE_TYPE_IDX are non-persistent buffers (pure data-derived, recomputable)."""

    def __init__(self, layout: Dict[str, Any]):
        super().__init__()
        from agents.model.damage_tables import build_move_attr, build_move_type_idx
        n_moves = layout['max_moves']
        self.register_buffer("MOVE_ATTR", build_move_attr(n_moves), persistent=False)
        self.register_buffer("MOVE_TYPE_IDX", build_move_type_idx(n_moves), persistent=False)
        in_dim = layout['move_embedding_dim'] + layout['type_embedding_dim'] + self.MOVE_ATTR.shape[1]
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim, MOVE_LATENT_HIDDEN),
            torch.nn.ReLU(),
            torch.nn.Linear(MOVE_LATENT_HIDDEN, MOVE_LATENT_DIM),
        )

    def forward(self, move_emb: torch.Tensor, type_emb: torch.Tensor,
                move_ids: torch.Tensor) -> torch.Tensor:
        """Per-slot latent. ``move_emb``/``type_emb`` are the ALREADY-embedded move + (HP-resolved) type
        ``[..., emb]``; ``move_ids`` ``[...]`` gathers MOVE_ATTR. Returns ``[..., MOVE_LATENT_DIM]``."""
        attr = self.MOVE_ATTR[move_ids]                                    # [..., N_MOVE_ATTR]
        return self.mlp(torch.cat([move_emb, type_emb, attr], dim=-1))

    def latent_table(self, embeddings: 'Embeddings') -> torch.Tensor:
        """``[n_moves, MOVE_LATENT_DIM]`` context-free latent over canonical types — the grading target."""
        ids = torch.arange(self.MOVE_ATTR.shape[0], device=self.MOVE_ATTR.device)
        move_emb = embeddings.move_embedding(ids)                          # [n_moves, move_emb]
        type_emb = embeddings.type_embedding(self.MOVE_TYPE_IDX)           # [n_moves, type_emb]
        return self.mlp(torch.cat([move_emb, type_emb, self.MOVE_ATTR], dim=-1))


class PokemonEncoder(torch.nn.Module):
    """Per-Pokémon encoding: embed + stitch the enriched vector, run the shared move
    processor + within-mon move self-attention, then the role encoder → 12×128 role tokens.

    When ``move_latent`` is on (gen3_unified_move_system_v1), a context-free `MoveLatentEncoder` latent
    is concatenated into the move-network input — a mechanics-grounded move identity (widens
    move_input_dim by MOVE_LATENT_DIM; OFF leaves the move network byte-identical)."""

    def __init__(self, layout: Dict[str, Any], move_latent: bool = False):
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
        # gen3_unified_move_system_v1: the mechanics-grounded move latent, concatenated into the move
        # network when on (widens move_input_dim by MOVE_LATENT_DIM; OFF = byte-identical move network).
        self.move_latent = move_latent
        if move_latent:
            self.move_latent_encoder = MoveLatentEncoder(layout)
            move_input_dim += MOVE_LATENT_DIM
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

        move_feature_blocks = [
            embedded_moves,
            embedded_move_types,
            move_remnants_reshaped,
            known_flags_reshaped,
            move_context_final,
            ctx.matchups_all,
            matchup_validity,
            hp_probs_per_slot,
            move_validity,
        ]
        # gen3_unified_move_system_v1: append the context-free move latent (resolved type incl. live HP
        # type) so the move network reads a mechanics-grounded move identity. The same encoder's
        # latent_table feeds the Stage-3 similarity grading; OFF skips this entirely (byte-identical).
        if self.move_latent:
            move_feature_blocks.append(
                self.move_latent_encoder(embedded_moves, embedded_move_types, ctx.all_move_ids))
        move_features = torch.cat(move_feature_blocks, dim=3)

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


class WinProbHead(torch.nn.Module):
    """Auxiliary WIN-PROBABILITY readout — a calibrated P(win | state) the shaped critic can't give.

    The dual-head value (`value_pooled`) estimates expected *shaped* return (material Φ + PBRS terms +
    terminal, PopArt-normalised) — NOT a probability and not interpretable as win odds. This head reads
    the same whole-board `value_pooled` summary and emits ONE logit; sigmoid(logit) = P(win). It is
    supervised (in `instrumented_ppo`) by the Monte-Carlo episode OUTCOME (win=1 / loss=0) propagated to
    every step of the episode, so it learns the actual probability the current state leads to a win — and
    ΔP(win) across a decision is a directly legible "how much did this move change my win odds".

    SIDE readout, leak-safe: the logit is stashed at `features_extractor.last_win_prob_logits` and read
    ONLY by the aux loss + the offline prober/eval — NEVER concatenated into pi/vf, so the privileged
    future OUTCOME label can never reach the acting path. The tri-state `win_prob_mode` controls the
    GRADIENT at the call site (`read_only` feeds a STOP-GRAD `value_pooled` — the head trains its OWN
    params as a pure, risk-free diagnostic that can't perturb the policy; `shaping` feeds it live so the
    win-prediction objective also shapes the shared trunk). `none` = this module is not built (the chain
    is byte-for-byte the baseline)."""

    def __init__(self):
        super().__init__()
        # Small MLP off the value pool: LayerNorm → Linear → ReLU → Linear(→1). A bottleneck (not a bare
        # linear) so `read_only` reports "decodable by a small head" — fairer to the nonlinear trunk.
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(D_MODEL),
            torch.nn.Linear(D_MODEL, D_MODEL),
            torch.nn.ReLU(),
            torch.nn.Linear(D_MODEL, 1),
        )

    def forward(self, value_pooled: torch.Tensor) -> torch.Tensor:
        """value_pooled [B, D_MODEL] → win-probability logit [B, 1] (sigmoid ⇒ P(win))."""
        return self.net(value_pooled)


# Logit at which a REVEALED (certain) move is pinned under prior fusion: sigmoid(10) ≈ 0.99995 ≈ P 1.
_REVEAL_LOGIT = 10.0


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
    new-move gap), 'unrevealed' (hidden species), or 'both'.

    **Prior fusion (`prior_fusion`, the unified two-part belief).** When on, the head's output is treated
    as a LEARNED LOG-ODDS DELTA fused additively with the Smogon move-frequency prior — so the stashed
    `move_logits` carries a proper POSTERIOR over the opponent's moveset, not a from-scratch prediction:
      - REVEALED moves (seen this battle, opp move-id > 0) → pinned CERTAIN (`_REVEAL_LOGIT` ≈ P 1) — the
        "calculate the known moves" half;
      - UNREVEALED moves → `prior_logit(species) + head_delta` — the "pick the unknown, assign a
        probability" half (cold-start = the prior; the head learns the in-battle correction).
    The damage op + the BCE loss both then read ONE coherent posterior (unifying the priors + the learned
    prediction + the damage calc). The prior buffer is a NON-persistent, data-derived lookup; OFF
    reproduces the from-scratch head byte-for-byte. See `designs/ai_v6/design_differentiable_damage_op.md`."""

    def __init__(self, n_moves: int, move_emb_dim: int,
                 prior_fusion: bool = False, n_species: int = 0, move_candidate_floor: float = 0.0):
        super().__init__()
        self.move_head = torch.nn.Linear(D_MODEL, n_moves)
        self.reinject = torch.nn.Linear(move_emb_dim, D_MODEL)
        torch.nn.init.normal_(self.reinject.weight, std=0.02)   # start the enrichment ≈0
        torch.nn.init.zeros_(self.reinject.bias)
        self.norm = torch.nn.LayerNorm(D_MODEL)
        self.prior_fusion = prior_fusion
        if prior_fusion:
            from agents.model.damage_tables import build_move_prior_logits
            # [n_species, n_moves] log-odds base rate (data-derived physics, recomputable → non-persistent).
            # move_candidate_floor>0 enables the LEGALITY-ONLY gate: a move a species can't learn → ~0
            # (impossible), a legal move keeps its TRUE usage (rare moves stay rare-but-liftable, never
            # pruned), a legal-unobserved move gets the small floor base. 0.0 = the legacy un-gated
            # 0.02-floor prior (byte-identical).
            gate = move_candidate_floor > 0.0
            prior_kwargs = {"learnset_gate": gate}
            if gate:
                prior_kwargs["floor"] = move_candidate_floor   # else the builder's default 0.02 (legacy)
            self.register_buffer(
                "move_prior_logits",
                build_move_prior_logits(n_species, n_moves, **prior_kwargs), persistent=False)
            # Zero-init the head so the cold-start delta is EXACTLY 0 → the fused posterior == the prior at
            # step 0 (the cleanest A/B baseline + matches the docstring claim). Only under fusion; the
            # from-scratch (no-fusion) path keeps the default init unchanged.
            torch.nn.init.zeros_(self.move_head.weight)
            torch.nn.init.zeros_(self.move_head.bias)

    def forward(self, opp_tokens: torch.Tensor, apply_mask: torch.Tensor,
                move_embedding: torch.nn.Embedding,
                opp_species_ids: Optional[torch.Tensor] = None,
                opp_move_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """opp_tokens [B,6,D], apply_mask [B,6] bool (which slots get the belief), move_embedding table
        → (enriched_tokens [B,6,D], move_logits [B,6,M]). The enrichment is residual + gated to the
        selected slots, so unselected slots pass through unchanged. When `prior_fusion`, `opp_species_ids`
        [B,6] (national-dex nums, for the prior lookup) and `opp_move_ids` [B,6,4] (revealed opp moves;
        id>0 ⇒ seen) turn the raw head output into the two-part POSTERIOR (prior⊕delta, revealed pinned)."""
        move_logits = self.move_head(opp_tokens)                                 # [B, 6, M] (learned delta)
        if self.prior_fusion and opp_species_ids is not None:
            move_logits = move_logits + self.move_prior_logits[opp_species_ids]  # posterior = prior ⊕ delta
            if opp_move_ids is not None:
                # REVEALED moves are certain → pin to a high logit (sigmoid ≈ 1). id 0 = unknown sentinel.
                revealed = torch.zeros_like(move_logits, dtype=torch.bool)       # [B, 6, M]
                valid = opp_move_ids > 0                                         # [B, 6, 4]
                if bool(valid.any()):
                    bb, ss, _ = valid.nonzero(as_tuple=True)
                    ids = opp_move_ids[valid].clamp(0, move_logits.shape[-1] - 1)  # defensive index clamp
                    revealed[bb, ss, ids] = True
                    move_logits = torch.where(
                        revealed, move_logits.new_full((), _REVEAL_LOGIT), move_logits)
        soft_emb = torch.sigmoid(move_logits) @ move_embedding.weight             # [B, 6, move_emb]
        enriched = opp_tokens + apply_mask.unsqueeze(-1) * self.reinject(soft_emb)
        return self.norm(enriched), move_logits


class SpreadBelief(torch.nn.Module):
    """Predicts the opponent's hidden SPREAD — the 5 battle-relevant DERIVED stats {atk,def,spa,spd,spe} at
    L100 — per opp slot, and REINJECTS it into the slot token. The THIRD belief leg (moves ✓, species ✓,
    STATS) — `gen3_unified_spread_belief_v1`, mirroring MoveBelief 1:1.

    Per opp slot: a usage-weighted PRIOR (mean, std per stat, from the Smogon spreads) ⊕ a learned head
    DELTA (in std units, so the prediction is `mean + delta·std` — the op consumes the stat VALUE directly,
    keeping the "op does the multiplicative physics, the head reasons additively" contract; predicting EVs
    +nature would force the op to re-run the nonlinear stat formula). The DELTA is reinjected as a small
    residual so the enriched token carries the believed spread to both heads. The `DamageOperator` reads
    the believed stats at the opp ACTIVE slot to REPLACE its hand-coded de-timid (offense) / neutral-0-EV
    (bulk) opp-spread constants — so the op's opponent stats are a learned belief, not a fixed guess. The
    offense/defense stats get gradient from the op's damage rolls; the SPEED component's explicit
    supervision (a masked BCE toward observed MOVE ORDER — a public, leak-safe label) is STAGED, not yet
    wired (`spread_belief_coef` is recorded-only until the loss lands). HP is
    skipped (the op uses the obs HP fraction × a neutral maxhp estimate). Zero-init head → cold-start ==
    the prior; the prior buffer is non-persistent (data-derived) → OFF reproduces nothing (the module isn't
    built when off)."""

    def __init__(self, n_species: int):
        super().__init__()
        from agents.model.damage_tables import build_opp_spread_prior, N_SPREAD_STATS
        self.n_stats = N_SPREAD_STATS
        self.stat_head = torch.nn.Linear(D_MODEL, N_SPREAD_STATS)   # learned delta in std-units
        torch.nn.init.zeros_(self.stat_head.weight)                 # cold-start delta == 0 → believed == prior
        torch.nn.init.zeros_(self.stat_head.bias)
        self.reinject = torch.nn.Linear(N_SPREAD_STATS, D_MODEL)
        torch.nn.init.normal_(self.reinject.weight, std=0.02)       # start the enrichment ≈0
        torch.nn.init.zeros_(self.reinject.bias)
        self.norm = torch.nn.LayerNorm(D_MODEL)
        # [n_species, 5, 2] (mean, std) usage prior — non-persistent (recomputable from data/).
        self.register_buffer("spread_prior", build_opp_spread_prior(n_species), persistent=False)

    def forward(self, opp_tokens: torch.Tensor, apply_mask: torch.Tensor,
                opp_species_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """opp_tokens [B,6,D], apply_mask [B,6] bool (which slots get the belief), opp_species_ids [B,6]
        (national-dex nums) → (enriched_tokens [B,6,D], believed_stats [B,6,5]). believed = prior_mean +
        delta·prior_std (clamped ≥1); the residual carries the learned delta into the token (gated to the
        selected slots; unselected pass through). Cold-start (delta 0) ⇒ believed == the usage prior."""
        prior = self.spread_prior[opp_species_ids]                  # [B,6,5,2]
        mean, std = prior[..., 0], prior[..., 1]                    # [B,6,5]
        delta = self.stat_head(opp_tokens)                          # [B,6,5] (std units; cold == 0)
        believed = (mean + delta * std).clamp(min=1.0)              # [B,6,5] the stat VALUE the op consumes
        enriched = opp_tokens + apply_mask.unsqueeze(-1) * self.reinject(delta)
        return self.norm(enriched), believed


# Differentiable damage operator (`DamageOperator`) constants — the unified 3-roll + P(KO) damage
# representation (owner-chosen: "the raw rolls the model can read, PLUS the P(KO) the policy uses").
# Per defender, two gen3 type CHANNELS (physical / special) each carry [low_roll, high_roll, crit, pko,
# accuracy]: the three rolls as a fraction of the defender's MAX HP (stationary "how big is the hit" —
# low = the 0.85 roll, high = the max roll, crit = the ×2 screen-ignoring crit), pko = the
# accuracy-discounted P(KO this turn) vs CURRENT HP (= acc·P(KO|hit), the EXACT realized KO probability —
# independent events), and accuracy = the dominant threat's base hit probability. {pko, accuracy} together
# parameterize the full miss/survive/KO outcome distribution with every product PRE-COMPUTED (so the ReLU
# head never has to multiply — the operator does the multiplicative physics, the head reasons additively).
# Then p_outspeed + threat_provenance:
#   [phys_low, phys_high, phys_crit, phys_pko, phys_acc, spec_low, spec_high, spec_crit, spec_pko, spec_acc,
#    p_outspeed, prov]
# Driven by the LEARNED belief instead of the usage prior. NOT modifier-for-modifier parity: the op
# applies type/STAB/ability-immunity/screens/crit/accuracy; it does NOT (yet) apply weather, burn, defender
# boost stages, or fixed-damage/OHKO/HP-relative moves, and p_outspeed is a point estimate (no para/boost
# speed distribution). Those are the documented v2 follow-ups (the gradient story holds without them).
_DMG_CHANNEL_FEATS = 5          # [low_roll, high_roll, crit, pko, accuracy] per gen3 type-category channel
_DMG_N_CHANNELS = 2             # physical / special (the gen3 TYPE split)
_DMG_PER_MON = _DMG_N_CHANNELS * _DMG_CHANNEL_FEATS + 2   # + p_outspeed + provenance = 12
# Named per-defender feature offsets (the single source of truth for the op's output slot layout — the
# prober decode + outgoing/safe-switch directions index by these, never a literal).
_DMG_IDX_PHYS_LOW, _DMG_IDX_PHYS_HIGH, _DMG_IDX_PHYS_CRIT, _DMG_IDX_PHYS_PKO, _DMG_IDX_PHYS_ACC = 0, 1, 2, 3, 4
_DMG_IDX_SPEC_LOW, _DMG_IDX_SPEC_HIGH, _DMG_IDX_SPEC_CRIT, _DMG_IDX_SPEC_PKO, _DMG_IDX_SPEC_ACC = 5, 6, 7, 8, 9
_DMG_IDX_OUTSPEED, _DMG_IDX_PROVENANCE = 10, 11
_DMG_ROLL_MIN = 0.85            # lowest of the 16 gen3 damage rolls ((85..100)/100); high roll = 1.0
_DMG_CHIP_CAP = 1.5             # clamp on the roll fractions (a 4× STAB hit otherwise fattens the tail)
_DMG_CRIT_CAP = 3.0            # crit can ×2 a capped high roll → a wider cap
# Opp-active believed-effect threat scalars (belief-weighted MAX over the move belief — see the
# aggregation note in forward; a full-axis noisy-OR over ~400 moves saturated to ~1 from the floor alone),
# order == damage_tables MOVE_EFFECT_COLS: [recovery, status, phaze, boost, hazard, protect]. The
# status/utility axis the damage-only CPU block never had.
_DMG_EFFECT = 6
# gen3_unified_move_system_v1: per-status SECONDARY-effect threat the opp active poses (its damaging
# moves' secondaries — Body Slam para, Rock Slide flinch, Ice Beam freeze), belief-weighted + accuracy-
# folded + ×Serene Grace. 10 scalars, order == damage_tables.SECONDARY_COLS. Appended AFTER the 6 effect
# scalars (the existing per-mon/effect layout is untouched). NO speed coupling — flinch's move-first
# dependence is left to attention (owner decision).
_DMG_INCOMING_SEC = _N_SECONDARY            # 10
# gen3_unified_choice_band_v1: the CB-CONDITIONAL physical tail of the INCOMING threat — per our 6 mons, the
# opp's PHYSICAL [high-roll, P(OHKO)] computed WITH the ×1.5 Choice-Band Atk, then ONE shared `p_cb` scalar
# (P(opp active holds CB)). DECORRELATED from the modal (no-CB) line + p_cb so the head weights them itself
# (OHKO is a nonlinear threshold a mean-field blend would blur — same rationale as the crit-split). The op's
# OUTGOING block separately applies the ×1.5 deterministically (our own item is known). Order:
# [phys_high_cb × 6, phys_pko_cb × 6, p_cb].
_DMG_CB_PER_MON = 2                          # phys_high_cb, phys_pko_cb (physical channel only — CB is phys)
_DMG_CB = _DMG_CB_PER_MON * TEAM_SIZE + 1    # + the shared p_cb scalar = 13
_DMG_CRIT_P = 1.0 / 16.0   # gen3 base crit rate (×2 damage) — the crit roll is exposed as crit_frac
_DMG_SPEED_SCALE = 15.0    # logistic scale for P(outspeed) over the speed-stat difference (~one stage)
# gen3_unified_spread_belief_v1: indices into the SpreadBelief's [atk,def,spa,spd,spe] output (== the
# damage_tables SPREAD_STAT_COLS order). When a spread belief is passed, the op consumes these believed opp
# stat VALUES in place of its hand-coded de-timid/neutral-0-EV constants.
_SB_ATK, _SB_DEF, _SB_SPA, _SB_SPD, _SB_SPE = 0, 1, 2, 3, 4
# gen3_unified_op_physics_v1: the active mons' stat-STAGE boosts (DD/CM/Intimidate…) are the worst
# damage-calc edge case (a +2 sweeper's Atk is doubled). The boost stage is read from the active-context
# block (boosts(14) = 7 stats × 2 [pos/6, neg/6]: atk,def,spa,spd,spe,acc,eva), and the gen3 multiplier
# applied to offense/defense/speed. Mirrors incoming_damage.boost_mult. _PARA_SPEED quarters Speed.
_DMG_PARA_SPEED = 0.25
# Burn (½ physical Atk) + paralysis (×0.25 Speed) read the per-mon condition one-hot (None,BRN,PAR,SLP,…)
# at POKEMON_CONDITION_OFFSET; weather (rain ×1.5 Water/×0.5 Fire; sun the reverse) reads the Water/Fire
# type indices on the TypeEncoder axis (the same axis MOVE_TYPE_IDX rides) + ctx.weather_feature.
_COND_BRN_IDX, _COND_PAR_IDX = 1, 2
from agents.observation.types import TypeEncoder as _TypeEncoder
_WATER_TIDX, _FIRE_TIDX = _TypeEncoder.TYPE_TO_IDX["WATER"], _TypeEncoder.TYPE_TO_IDX["FIRE"]

# OUTGOING direction (our active → opp active): per OUR move, in REQUEST-slot order (== action logits
# 6+k) so the policy head can compare move A vs B directly — the equal-effectiveness tie-break (Earthquake
# vs Meteor Mash into a Rock: same 2× multiplier, different resolved damage). Per move [low, high, crit,
# pko] (accuracy is NOT repeated — our moves' accuracy already rides the action-aligned obs move-block;
# pko still folds it), then one p_outspeed (our active vs opp active), then (gen3_unified_move_system_v1)
# the per-move SECONDARY-effect probabilities — "what status can OUR move cause, with what probability"
# (Thunderbolt 10%/20% para under Serene Grace, ×opp Shield Dust), 4 moves × 10 cols appended LAST.
_DMG_OUT_PER_MOVE = 4
_DMG_OUT_N_MOVES = 4
_DMG_OUT_SEC = _DMG_OUT_N_MOVES * _N_SECONDARY            # 4 moves × 10 secondary probs = 40
_DMG_OUTGOING = _DMG_OUT_N_MOVES * _DMG_OUT_PER_MOVE + 1 + _DMG_OUT_SEC   # 16 + p_outspeed + 40 = 57

# gen3_unified_status_landing_v1: the OUTGOING per-OUR-move "will my STATUS move land vs THIS opponent"
# block — the GPU replacement for the masked move-effect block's `status_will_land`. Per move (request-slot
# order == action 6+k): P(the status applies) + a `known` bit (the value rests on a CERTAIN block — type
# immunity / already-statused / Sleep-Clause / Substitute / a revealed ability — vs a Smogon-prior estimate).
# Folds accuracy × type immunity × ability immunity (revealed-or-prior) × already-statused × Sleep Clause ×
# Substitute; adds Leech Seed (Grass-immune). NOTE the delta vs the masked CPU block it replaces: it now FOLDS
# base accuracy (Toxic 0.85 / WoW 0.75 — the CPU returned 1−P(ability blocks), accuracy-free), so the value is
# more correct but value-meaning-different on the A/B. UNCOVERED residual: Yawn (delayed sleep — no
# status_inflicted) and a Leech-Seed-already-seeded target (no leechseed volatile read). Shield Dust is NOT
# relevant here (it only scales SECONDARY effects, never a primary status move — see the incoming-secondary
# block). Appended to the outgoing direction (gated on `damage_outgoing`).
_DMG_STATUS_N_MOVES = _DMG_OUT_N_MOVES
_DMG_STATUS = 2 * _DMG_STATUS_N_MOVES                     # 4 P(lands) + 4 known = 8
_COND_SLP_IDX = 3                                         # condition one-hot [None,BRN,PAR,SLP,FRZ,PSN,TOX]
# Substitute's index into the active-context block (our_ctx_raw / opp_ctx_raw = boosts ++ volatiles), DERIVED
# from the obs layout (never hardcoded). A Substitute blocks EVERY status move (incl. Leech Seed) in gen3.
from agents.observation.gen3_effects import VOLATILE_SLOTS as _VOLATILE_SLOTS
from agents.observation.constants import BOOSTS_DIM as _BOOSTS_DIM
_SUBSTITUTE_CTX_IDX = _BOOSTS_DIM + list(_VOLATILE_SLOTS).index("substitute")


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

    Per defender d, `_DMG_PER_MON` (12) features: per channel (physical / special — the gen3 TYPE split) the
    3-roll `[low, high, crit]` + accuracy-folded `pko` + `acc` (10 = 5×2), plus P(outspeed) and the threat
    provenance. Aggregation is a HARD max over the
    channel's believed candidates (= `incoming_damage`'s max-over-candidates; differentiable via the
    argmax subgradient — the dominant move's belief weight gets gradient — without the candidate-count
    dilution a low-temperature soft-max would suffer over ~400 moves). Plus `_DMG_EFFECT` opp-active
    believed-EFFECT scalars (belief-weighted MAX of the belief × per-move status/utility flags: recovery,
    status, phaze, boost, hazard, protect) — the status-threat axis the damage-only CPU block never had.
    Plus (gen3_unified_move_system_v1) `_DMG_INCOMING_SEC` per-STATUS secondary scalars — the opp active's
    DAMAGING-move secondaries (Body Slam para, Rock Slide flinch, Ice Beam freeze): realized
    `max_m(w_m·chance·acc) × Serene Grace(opp)`, accuracy folded, NO speed coupling (flinch's move-first
    dependence is left to attention). When `outgoing`, each of OUR 4 moves ALSO carries its secondary
    probabilities (`chance·acc × Serene Grace(us) × Shield Dust(opp)`) — "what status can this move cause,
    with what probability". Order == damage_tables.SECONDARY_COLS.
    Hidden Power (all 17 variants collide on num=237 → unrepresentable on the num axis) is expanded
    into 16 TYPED candidates (BP 70 = gen3 max), weighted `P(HP present)·P(type)` — presence from the
    move belief (`w[237]`), type from the per-mon `hp_probs` obs block (the HP tracker's narrowed
    distribution) — so HP Grass vs HP Ice get distinct type effectiveness.

    Stats: our defenders use their REAL spread (IVs/EVs/nature reconstructed from the obs spread block —
    they are revealed); the hidden-spread attacker uses a fixed de-timid offensive assumption (252 EV,
    31 IV, +nature ×1.1), mirroring `incoming_damage`'s offensive-stat tail. The smooth (un-floored)
    L100 stat + damage formula keeps everything differentiable (the byte-exact floored kernel is the
    proof's; the forward only needs the gradient).

    When `outgoing`, the op ALSO appends the per-OUR-move outgoing damage block AND the
    gen3_unified_status_landing_v1 STATUS-LANDING block (per move: P(a dedicated status move lands vs the opp
    active — type/ability/already-statused/Sleep-Clause/Substitute folded, Leech Seed Grass-immune) + a
    `known` bit) — the GPU home for the masked move-effect `status_will_land`.

    Leak-safe: reads only the PREDICTED belief + public obs (our HP/types, the opp active's revealed
    species/types/condition/sub) — never a privileged label. Output `[B, self.out_dim]` (= incoming +,
    when outgoing, the damage + status-landing blocks) is appended to BOTH projection heads. Zeroed (incl.
    gradient) when there is no opponent active and per fainted defender.
    Lookup tables are registered as non-persistent float32 buffers (pure physics, recomputable from
    `data/`)."""

    per_mon = _DMG_PER_MON
    n_effect = _DMG_EFFECT
    n_incoming_sec = _DMG_INCOMING_SEC
    incoming_dim = TEAM_SIZE * _DMG_PER_MON + _DMG_EFFECT + _DMG_INCOMING_SEC + _DMG_CB

    def __init__(self, layout: Dict[str, Any], outgoing: bool = False):
        super().__init__()
        from agents.model.damage_tables import (
            build_damage_buffers, HIDDEN_POWER_NUM, HIDDEN_POWER_BP,
            CHOICE_BAND_ITEM_NUM, CHOICE_BAND_PHYS_MULT,
        )
        bufs = build_damage_buffers(layout['max_moves'], layout['max_species'], layout['max_abilities'])
        for name, tensor in bufs.items():
            # Non-persistent: deterministic physics from data/, not learned weights → keep them out of
            # every checkpoint (and out of the state_dict, so a load never demands them).
            self.register_buffer(name, tensor, persistent=False)
        self.hp_num = HIDDEN_POWER_NUM
        self.hp_bp = float(HIDDEN_POWER_BP)
        self.cb_item_num = CHOICE_BAND_ITEM_NUM            # gen3_unified_choice_band_v1: Choice Band item num
        self.cb_phys_mult = float(CHOICE_BAND_PHYS_MULT)   # ×1.5 physical Atk
        # OUTGOING direction (our active → opp active, per-move action-aligned): off by default. When on,
        # the op ALSO emits the _DMG_OUTGOING block (widens out_dim → both projections auto-size).
        self.outgoing = outgoing
        # The OUTGOING direction carries the per-move damage block + the gen3_unified_status_landing_v1
        # status-landing block (both action-aligned, our active → opp). Off ⇒ neither → baseline byte-identical.
        self.out_dim = self.incoming_dim + (_DMG_OUTGOING + _DMG_STATUS if outgoing else 0)
        # Runtime grad-checkpointing flag (set per run by --grad-checkpointing via
        # _apply_grad_checkpointing) — recompute the op in backward, trading idle-GPU compute for the
        # ~GBs of [B,6,C]-over-~416-candidate activations this op materialises at batch 16384. No-op
        # under inference (gated on is_grad_enabled). Bit-exact (no dropout/RNG in the op).
        self.grad_checkpointing = False
        # Learnable per-channel adapter (the "structure to learn" — answers the review's M3): a gain on
        # each of the out_dim output channels, INITIALISED to put the heterogeneous physics channels on a
        # comparable scale (chip≤1.5, crit_delta≤1/16, the rest in [0,1]) so the shared pre_proj_norm
        # doesn't bury the small ones, then trained. ×only (no bias) → preserves the no-threat zeros (the
        # has_opp / defender_alive gates stay clean). OFF = no module, so this never touches the baseline.
        gain = torch.ones(self.out_dim)
        # per-mon block, 12 feats: [phys_low, phys_high, phys_crit, phys_pko, phys_acc, spec_low,
        # spec_high, spec_crit, spec_pko, spec_acc, p_outspeed, provenance] → pre-scale the rolls onto
        # ~[0,1]: low/high (cap 1.5) ÷1.5, crit (cap 3.0) ÷3.0; pko/acc/outspeed/provenance already [0,1].
        per_mon_init = torch.tensor([1.0 / 1.5, 1.0 / 1.5, 1.0 / 3.0, 1.0, 1.0,
                                     1.0 / 1.5, 1.0 / 1.5, 1.0 / 3.0, 1.0, 1.0, 1.0, 1.0])
        gain[:TEAM_SIZE * self.per_mon] = per_mon_init.repeat(TEAM_SIZE)
        # gen3_unified_choice_band_v1: the CB block tail [phys_high_cb×6, phys_pko_cb×6, p_cb] — scale the
        # CB high-roll like the other high rolls (cap 1.5 → ÷1.5); pko/p_cb already in [0,1] (stay 1.0).
        _cb0 = TEAM_SIZE * self.per_mon + _DMG_EFFECT + _DMG_INCOMING_SEC
        gain[_cb0:_cb0 + TEAM_SIZE] = 1.0 / 1.5                       # the phys_high_cb sub-block
        if outgoing:
            # outgoing block: per move [low, high, crit, pko] (same roll scaling), then p_outspeed.
            out_move_init = torch.tensor([1.0 / 1.5, 1.0 / 1.5, 1.0 / 3.0, 1.0])
            gain[self.incoming_dim:self.incoming_dim + _DMG_OUT_N_MOVES * _DMG_OUT_PER_MOVE] = \
                out_move_init.repeat(_DMG_OUT_N_MOVES)
            # the trailing p_outspeed, the per-move secondary block, and the gen3_unified_status_landing_v1
            # status block (p_land/known) all stay at gain 1.0 — they are already probabilities in [0,1].
        self.out_gain = torch.nn.Parameter(gain)

    def _chan_max(self, value: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
        """Max over a channel's candidates = the most-threatening believed move (exactly
        `incoming_damage.py`'s max-over-candidates). `value` [B,6,C] (≥0), `channel_mask` [1,1,C]
        (1=on-channel). Off-channel candidates are zeroed; since values are ≥0, `amax` returns the max
        on-channel value (or 0 if the channel has no threat). Differentiable via the argmax subgradient —
        the dominant move's belief weight gets gradient — and crucially NOT diluted by the ~400 zero-score
        candidates the way a low-temperature soft-max would be."""
        return (value * channel_mask).amax(dim=-1)

    @staticmethod
    def _boost_mult(stage: torch.Tensor) -> torch.Tensor:
        """gen3_unified_op_physics_v1: gen3 stat-stage multiplier (atk/def/spa/spd/spe). stage≥0 →
        (2+stage)/2, stage<0 → 2/(2−stage), clamped to [−6,6]. Mirrors incoming_damage.boost_mult."""
        s = stage.clamp(-6.0, 6.0)
        return torch.where(s >= 0, (2.0 + s) / 2.0, 2.0 / (2.0 - s))

    @staticmethod
    def _boost_stages(ctx_raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                       torch.Tensor, torch.Tensor]:
        """Read the active mon's [atk,def,spa,spd,spe] boost STAGES ([B] each) from its active-context
        block (boosts = 7 stats × 2 dims [max(0,stage)/6, max(0,−stage)/6]); stage = (pos − neg)·6."""
        b = ctx_raw
        return ((b[:, 0] - b[:, 1]) * 6.0, (b[:, 2] - b[:, 3]) * 6.0, (b[:, 4] - b[:, 5]) * 6.0,
                (b[:, 6] - b[:, 7]) * 6.0, (b[:, 8] - b[:, 9]) * 6.0)

    @staticmethod
    def _weather_mult(weather_feature: torch.Tensor, is_water: torch.Tensor,
                      is_fire: torch.Tensor) -> torch.Tensor:
        """gen3_unified_op_physics_v1: gen3 weather BP modifier — rain (weather idx 2) ×1.5 Water / ×0.5
        Fire; sun (idx 1) ×1.5 Fire / ×0.5 Water; else 1.0. `is_water`/`is_fire` are broadcast-compatible
        per-candidate type-match flags. Sandstorm/Hail have no BP effect (gen3). Mirrors
        incoming_damage.weather_damage_mult."""
        sun = weather_feature[:, 1:2]                                                # [B,1]
        rain = weather_feature[:, 2:3]                                               # [B,1]
        return 1.0 + rain * (0.5 * is_water - 0.5 * is_fire) + sun * (0.5 * is_fire - 0.5 * is_water)

    def _rolls(self, dmg_ns, screen, maxhp, cur_hp, acc, eps: float = 1e-6):
        """The single source of the 3-roll + accuracy-folded-P(KO) physics — BOTH the incoming kernel
        and the outgoing block call this (the DRY core). From pre-screen max-roll damage ``dmg_ns`` + the
        DEFENDER's ``screen`` multiplier + ``maxhp``/``cur_hp`` + per-candidate ``acc`` (all broadcast-
        compatible) → ``(high_frac, low_frac, crit_frac, ko_ramp)``: the max-roll / 0.85-roll / ×2-crit
        damage as a fraction of MAX HP (gen3 crit ignores screens → ×2 the PRE-screen damage; clamped,
        "damage IF it lands"), and the accuracy-discounted P(KO this turn) vs CURRENT HP (``acc·P(KO|hit)``,
        the exact realized KO probability — accuracy and the roll are independent events)."""
        dmg = dmg_ns * screen                                            # post-screen max-roll
        inv = 1.0 / (maxhp + eps)
        high = (dmg * inv).clamp(max=_DMG_CHIP_CAP)
        low = (_DMG_ROLL_MIN * dmg * inv).clamp(max=_DMG_CHIP_CAP)
        crit = (2.0 * dmg_ns * inv).clamp(max=_DMG_CRIT_CAP)
        ko = acc * torch.clamp((dmg - cur_hp) / (0.15 * dmg + eps), 0.0, 1.0)
        return high, low, crit, ko

    def _damage_rolls(self, atk: torch.Tensor, spa: torch.Tensor, at1: torch.Tensor, at2: torch.Tensor,
                      def_stat: torch.Tensor, spd_stat: torch.Tensor, maxhp: torch.Tensor,
                      cur_hp: torch.Tensor, t1d: torch.Tensor, t2d: torch.Tensor, ability1: torch.Tensor,
                      reflect: torch.Tensor, light_screen: torch.Tensor,
                      bp_all: torch.Tensor, mty_all: torch.Tensor, phys_all: torch.Tensor,
                      acc_all: torch.Tensor, fixed_all: torch.Tensor, weather_mult: torch.Tensor,
                      eps: float = 1e-6):
        """Role-parameterized gen3 single-hit damage per ``(defender, candidate)`` — the shared
        physics kernel every DIRECTION reuses (incoming opp→our-6, outgoing our→opp, safe-switch).
        Roles are passed in rather than hardcoded so the SAME math serves attacker/defender swaps.

        Shapes: ``atk``/``spa``/``at1``/``at2`` are ``[B]`` (one attacker); ``def_stat``/``spd_stat``/
        ``maxhp``/``cur_hp``/``t1d``/``t2d``/``ability1`` are ``[B, n_def]``; ``reflect``/``light_screen``
        are ``[B, 1]`` (the DEFENDER's side screens); ``bp_all``/``mty_all``/``phys_all`` are ``[C]``
        (the candidate move axis incl. the 16 typed Hidden Powers); ``acc_all`` is ``[C]`` (per-candidate
        base hit probability). Returns ``(high_frac, low_frac, crit_frac, ko_ramp)``, each ``[B, n_def,
        C]``: the max-roll / 0.85-roll / ×2-crit damage as a fraction of the defender's MAX HP (clamped —
        damage IF it lands), and the **accuracy-discounted** modal no-crit P(KO) vs CURRENT HP
        (``acc · P(KO|hit)`` — so an inaccurate move reads a lower KO-this-turn risk). Pure /
        differentiable (no learned params) — the shared physics every direction reuses."""
        eff = self.CHART[t1d][..., mty_all] * self.CHART[t2d][..., mty_all]                     # [B,n,C]
        # Defender ABILITY immunity/resist (Levitate 0× Ground, Flash Fire 0× Fire, Thick Fat 0.5×
        # Fire/Ice): gathered by the defender's ability and folded into the effectiveness product.
        amul = self.ABILITY_DAMAGE_MULT[ability1]                                               # [B,n,19]
        eff = eff * amul[..., mty_all]                                                          # [B,n,C]
        A = phys_all * atk[:, None] + (1.0 - phys_all) * spa[:, None]                           # [B,C]
        D = phys_all[None, None, :] * def_stat[:, :, None] \
            + (1.0 - phys_all)[None, None, :] * spd_stat[:, :, None]                            # [B,n,C]
        is_stab = ((mty_all[None, :] == at1[:, None]) | (mty_all[None, :] == at2[:, None])).float()  # [B,C]
        stab = 1.0 + 0.5 * is_stab                                                              # [B,C]
        core = 42.0 * bp_all[None, None, :] * A[:, None, :] / (D + eps) / 50.0 + 2.0            # [B,n,C]
        dmg_ns = core * stab[:, None, :] * eff * 0.925                                          # [B,n,C] pre-screen
        dmg_ns = dmg_ns * (bp_all > 0).float()[None, None, :]            # kill the +2 floor on BP-0 moves
        dmg_ns = dmg_ns * weather_mult[:, None, :]      # gen3_unified_op_physics_v1: rain/sun BP modifier [B,1,C]
        # DEFENDER-side screens: Reflect halves physical incoming, Light Screen halves special.
        # gen3 CRIT IGNORES screens, so the crit roll below uses the pre-screen damage (dmg_ns).
        screen = 1.0 - 0.5 * (reflect * phys_all[None, :] + light_screen * (1.0 - phys_all[None, :]))
        # Final 3 rolls + accuracy-folded P(KO) via the shared formula (DRY — same as the outgoing block).
        high, low, crit, ko = self._rolls(dmg_ns, screen[:, None, :], maxhp[:, :, None], cur_hp[:, :, None],
                                          acc_all[None, None, :], eps)
        # gen3_unified_op_physics_v1: FIXED-damage moves (Seismic Toss / Night Shade = 100, Dragon Rage 40,
        # Sonic Boom 20) ignore Atk/Def/roll/crit but RESPECT type/ability immunity. Override the rolls with
        # the constant fraction (all three rolls equal — no variance), gated to 0 where `eff<=0` (Fighting
        # Seismic Toss → 0 vs Ghost; Ghost Night Shade → 0 vs Normal). Otherwise the BP-0 formula reads ~0.
        # gen3_unified_choice_band_v1: the CB-CONDITIONAL physical rolls — recompute with the physical Atk
        # ×1.5 at the STAT level (A_cb), so `core = k·A+2`'s +2 floor isn't itself ×1.5'd (the exact physics,
        # consistent with the outgoing block which scales our_atk). Special candidates unchanged. Only `high_cb`
        # / `ko_cb` are used (the op aggregates the PHYSICAL channel); the fixed-damage override below is
        # applied to them too (fixed damage is CB-independent → reads identically).
        A_cb = A + 0.5 * phys_all * atk[:, None]                                        # [B,C] physical Atk ×1.5
        # Only `high_cb` + `ko_cb` are aggregated (the special channel is CB-invariant), so compute them
        # INLINE rather than via _rolls — skips the unused low/crit rolls (~2×[B,n,C] of activations the
        # grad-checkpoint backward recompute would otherwise double; matters at batch 16384). `dmg_cb` folds
        # the defender screen in (post-screen), matching _rolls' high/ko exactly.
        dmg_cb = (42.0 * bp_all[None, None, :] * A_cb[:, None, :] / (D + eps) / 50.0 + 2.0) \
            * stab[:, None, :] * eff * 0.925 * (bp_all > 0).float()[None, None, :] \
            * weather_mult[:, None, :] * screen[:, None, :]                             # [B,n,C] post-screen
        inv_cb = 1.0 / (maxhp[:, :, None] + eps)
        high_cb = (dmg_cb * inv_cb).clamp(max=_DMG_CHIP_CAP)
        ko_cb = acc_all[None, None, :] * torch.clamp(
            (dmg_cb - cur_hp[:, :, None]) / (0.15 * dmg_cb + eps), 0.0, 1.0)
        is_fixed = (fixed_all > 0)[None, None, :]                                      # [1,1,C]
        not_immune = (eff > 0).float()                                                # [B,n,C] type+ability gate
        fixed_frac = (fixed_all[None, None, :] / (maxhp[:, :, None] + eps)) * not_immune
        fixed_ko = acc_all[None, None, :] * (fixed_all[None, None, :] >= cur_hp[:, :, None]).float() * not_immune
        high = torch.where(is_fixed, fixed_frac, high)
        low = torch.where(is_fixed, fixed_frac, low)
        crit = torch.where(is_fixed, fixed_frac, crit)
        ko = torch.where(is_fixed, fixed_ko, ko)
        high_cb = torch.where(is_fixed, fixed_frac, high_cb)
        ko_cb = torch.where(is_fixed, fixed_ko, ko_cb)
        return high, low, crit, ko, high_cb, ko_cb

    def _outgoing_block(self, ctx: 'ExtractorContext',
                        spread_belief: Optional[torch.Tensor] = None) -> torch.Tensor:
        """OUR active → opp active, PER MOVE in REQUEST-slot order (== action logits 6+k), so the policy
        head can compare move A vs B directly — the equal-effectiveness tie-break (Earthquake vs Meteor
        Mash into a Rock: same 2× multiplier, different resolved damage). Our moves are KNOWN (no belief —
        a hard one-hot), LEGALITY-MASKED via the action mask (Choice-lock / Disable / Taunt / no-PP); the
        opp DEFENDER's bulk is hidden → a NEUTRAL 0-EV estimate (not max-bulk, which would under-price our
        KOs); opp ability immunity is revealed-or-none; OPP-side screens apply. Output `[B, _DMG_OUTGOING]`:
        per move `[low, high, crit, pko]` + one `p_outspeed`. Reuses the shared `_rolls` formula. Leak-safe
        (public obs only); gated to 0 when there is no opp active OR our active is fainted/absent. Our moves
        are certain → no move-belief gradient (correct: we don't learn our own moves), but differentiable
        in the smooth stat/damage formula."""
        B, device, eps = ctx.batch_size, ctx.device, 1e-6
        ar = torch.arange(B, device=device)
        our_act = ctx.our_active_idx                                 # [B] our active slot (0..5)
        opp_act = TEAM_SIZE + ctx.opp_active_local                   # [B] opp active global slot
        has_opp = ctx.hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, -1].any(dim=1).float()  # [B]
        our_alive = (ctx.hp_and_active[ar, our_act, 0] > 0).float()  # [B] our active must exist + be alive
        gate = (has_opp * our_alive)[:, None]                        # [B,1]

        # --- our 4 moves (request-slot order == action logits 6+k), legality-masked ---
        move_ids = ctx.all_move_ids[ar, our_act, :]                  # [B,4]
        move_ty = ctx.all_move_type_ids[ar, our_act, :]             # [B,4] resolved type (incl our HP type)
        legal = ctx.move_mask.float()                              # [B,4] currently-legal (Choice/Disable/PP)
        is_hp = (move_ids == self.hp_num)
        bp = torch.where(is_hp, torch.full_like(move_ty, self.hp_bp, dtype=torch.float32),
                         self.MOVE_BP[move_ids])                     # [B,4] HP → 70 (else dex BP; status → 0)
        phys = self.TYPE_IS_PHYS[move_ty]                          # [B,4] gen3 category by resolved type
        acc = self.MOVE_ACCURACY[move_ids]                        # [B,4] (HP num → 1.0 default)
        usable = legal * (bp > 0).float()                         # [B,4] gate to legal damaging moves

        # --- our active attacker (real spread) ---
        a_base = self.BASE_STATS[ctx.species_ids[ar, our_act]]     # [B,6] [hp,atk,def,spa,spd,spe]
        spread = ctx.pokemon_part[ar, our_act,
                                  POKEMON_SPREAD_OFFSET:POKEMON_SPREAD_OFFSET + POKEMON_SPREAD_DIM]
        iv = spread[:, 0:6] * 31.0
        ev = spread[:, 6:12] * 252.0
        nat = spread[:, 13:18]                                     # [B,5] [atk,def,spa,spd,spe]
        our_atk = (2.0 * a_base[:, 1] + iv[:, 1] + ev[:, 1] / 4.0 + 5.0) * nat[:, 0]   # [B]
        our_spa = (2.0 * a_base[:, 3] + iv[:, 3] + ev[:, 3] / 4.0 + 5.0) * nat[:, 2]   # [B]
        our_spe = (2.0 * a_base[:, 5] + iv[:, 5] + ev[:, 5] / 4.0 + 5.0) * nat[:, 4]   # [B]
        # gen3_unified_op_physics_v1: OUR active's offensive + speed stat-stage boosts (we attack here) +
        # BURN (½ phys atk) + PARALYSIS (×0.25 speed).
        o_b_atk, o_b_def, o_b_spa, o_b_spd, o_b_spe = self._boost_stages(ctx.our_ctx_raw)
        our_burn = ctx.pokemon_part[ar, our_act, POKEMON_CONDITION_OFFSET + _COND_BRN_IDX]   # [B]
        our_para = ctx.pokemon_part[ar, our_act, POKEMON_CONDITION_OFFSET + _COND_PAR_IDX]   # [B]
        # gen3_unified_choice_band_v1: OUR Choice Band ×1.5 physical Atk (our item is KNOWN → deterministic,
        # not a belief). Composes multiplicatively with boosts/burn below; physical only (CB doesn't touch SpA).
        our_cb = (ctx.item_ids[ar, our_act] == self.cb_item_num).float()                     # [B]
        our_atk = our_atk * torch.where(our_cb > 0.5, our_atk.new_tensor(self.cb_phys_mult),
                                        our_atk.new_tensor(1.0))
        our_atk = our_atk * self._boost_mult(o_b_atk) * torch.where(
            our_burn > 0.5, our_atk.new_tensor(0.5), our_atk.new_tensor(1.0))
        our_spa = our_spa * self._boost_mult(o_b_spa)
        our_spe = our_spe * self._boost_mult(o_b_spe) * torch.where(
            our_para > 0.5, our_spe.new_tensor(_DMG_PARA_SPEED), our_spe.new_tensor(1.0))
        at1 = ctx.type1_ids[ar, our_act]                          # [B] our types (STAB)
        at2 = ctx.type2_ids[ar, our_act]

        # --- opp active defender (revealed species/types; ability revealed-or-none) ---
        # Bulk: the SpreadBelief's learned def/spd if provided (gen3_unified_spread_belief_v1), else the
        # legacy NEUTRAL 0-EV estimate (not max-bulk, which would under-price our KOs). maxhp stays the
        # neutral estimate either way (HP EVs vary little + the obs HP fraction carries relative HP).
        d_base = self.BASE_STATS[ctx.species_ids[ar, opp_act]]     # [B,6]
        bs = spread_belief[ar, ctx.opp_active_local] if spread_belief is not None else None  # [B,5] or None
        opp_def = bs[:, _SB_DEF] if bs is not None else (2.0 * d_base[:, 2] + 31.0 + 5.0)    # [B]
        opp_spd = bs[:, _SB_SPD] if bs is not None else (2.0 * d_base[:, 4] + 31.0 + 5.0)
        opp_maxhp = 2.0 * d_base[:, 0] + 31.0 + 110.0
        opp_spe = bs[:, _SB_SPE] if bs is not None else (2.0 * d_base[:, 5] + 31.0 + 5.0)   # believed / neutral
        # gen3_unified_op_physics_v1: OPP active's DEFENSIVE + speed boosts (it's the defender here) + its
        # paralysis (×0.25 speed, for p_outspeed).
        p_b_atk, p_b_def, p_b_spa, p_b_spd, p_b_spe = self._boost_stages(ctx.opp_ctx_raw)
        opp_para = ctx.pokemon_part[ar, opp_act, POKEMON_CONDITION_OFFSET + _COND_PAR_IDX]   # [B]
        opp_def = opp_def * self._boost_mult(p_b_def)
        opp_spd = opp_spd * self._boost_mult(p_b_spd)
        opp_spe = opp_spe * self._boost_mult(p_b_spe) * torch.where(
            opp_para > 0.5, opp_spe.new_tensor(_DMG_PARA_SPEED), opp_spe.new_tensor(1.0))
        opp_cur_hp = ctx.hp_and_active[ar, opp_act, 0] * opp_maxhp  # [B] obs HP frac × est. max HP
        t1d = ctx.type1_ids[ar, opp_act]                          # [B]
        t2d = ctx.type2_ids[ar, opp_act]
        opp_ability = ctx.ability1_ids[ar, opp_act]              # [B] (0 if unrevealed → no immunity mult)

        # --- gen3 damage per move (defender = opp active, candidates = our 4 moves), via the shared rolls ---
        eff = self.CHART[t1d[:, None], move_ty] * self.CHART[t2d[:, None], move_ty]   # [B,4]
        eff = eff * self.ABILITY_DAMAGE_MULT[opp_ability].gather(1, move_ty)          # [B,4] defender immunity
        A = phys * our_atk[:, None] + (1.0 - phys) * our_spa[:, None]                 # [B,4]
        D = phys * opp_def[:, None] + (1.0 - phys) * opp_spd[:, None]                 # [B,4]
        is_stab = ((move_ty == at1[:, None]) | (move_ty == at2[:, None])).float()
        stab = 1.0 + 0.5 * is_stab
        core = 42.0 * bp * A / (D + eps) / 50.0 + 2.0
        weather_mult = self._weather_mult(ctx.weather_feature, (move_ty == _WATER_TIDX).float(),
                                          (move_ty == _FIRE_TIDX).float())             # [B,4] rain/sun
        dmg_ns = core * stab * eff * 0.925 * usable * weather_mult                    # [B,4] (non-usable → 0)
        opp_reflect = ctx.screen_feature[:, 1:2]                                      # OPP-side screens
        opp_ls = ctx.screen_feature[:, 3:4]
        screen = 1.0 - 0.5 * (opp_reflect * phys + opp_ls * (1.0 - phys))             # [B,4]
        high, low, crit, ko = self._rolls(dmg_ns, screen, opp_maxhp[:, None], opp_cur_hp[:, None], acc, eps)
        # gen3_unified_op_physics_v1: OUR fixed-damage moves (Seismic Toss into the opp), immunity-gated +
        # legality-gated (usable). Mirrors the incoming kernel's override.
        fixed = self.MOVE_FIXED_DAMAGE[move_ids] * usable                            # [B,4] (0 if illegal)
        is_fixed = fixed > 0
        not_immune = (eff > 0).float()                                               # [B,4] type+ability gate
        fixed_frac = (fixed / (opp_maxhp[:, None] + eps)) * not_immune
        fixed_ko = acc * (fixed >= opp_cur_hp[:, None]).float() * not_immune
        high = torch.where(is_fixed, fixed_frac, high)
        low = torch.where(is_fixed, fixed_frac, low)
        crit = torch.where(is_fixed, fixed_frac, crit)
        ko = torch.where(is_fixed, fixed_ko, ko)
        p_outspeed = torch.sigmoid((our_spe - opp_spe) / _DMG_SPEED_SCALE)            # [B]

        # gen3_unified_move_system_v1: per OUR move, "what status can it cause + with what probability".
        # realized P(effect k | move) = chance_mk × acc_m × Serene Grace(our active) × Shield Dust(opp
        # active), gated to legal moves (status moves carry 0 secondary → naturally zeroed). Order ==
        # SECONDARY_COLS. [B,4,10].
        our_serene = self.ABILITY_SECONDARY_MULT[ctx.ability1_ids[ar, our_act]]        # [B] our active
        opp_block = self.ABILITY_SECONDARY_BLOCK[opp_ability]                          # [B] opp Shield Dust
        sec = self.MOVE_SECONDARY[move_ids]                                            # [B,4,10] base chance
        sec = sec * (acc * legal)[:, :, None] * (our_serene * opp_block)[:, None, None]
        sec = sec.clamp(max=1.0)                                                        # [B,4,10]

        per_move = torch.stack([low, high, crit, ko], dim=-1)                          # [B,4,4]
        block = torch.cat([per_move.reshape(B, -1), p_outspeed[:, None], sec.reshape(B, -1)], dim=1)  # [B, _DMG_OUTGOING]
        return block * gate

    def _status_landing(self, ctx: 'ExtractorContext') -> torch.Tensor:
        """gen3_unified_status_landing_v1: per OUR move (REQUEST-slot order == action 6+k), P(a dedicated
        STATUS move applies to the opp active) + a `known` bit — the GPU home for the masked move-effect
        block's `status_will_land`. The status MOVES the outgoing DAMAGE block can't price (BP 0 → usable 0).

        P(lands) = is_status_move · accuracy · (1−type_immune) · (1−ability_block) · (1−already_block)
                   · (1−sleep_clause_block), gated to 0 with no opp active / our active dead. Where:
          • type_immune  — per-MOVE gen3 rule (Thunder Wave→Ground, Toxic/Poison→Steel/Poison, Will-O-Wisp
            →Fire, **Leech Seed→Grass**), max over the opp active's two types.
          • ability_block — REVEALED opp ability → exact `ABILITY_STATUS_BLOCK`; UNREVEALED → the species
            Smogon-prior marginal `SPECIES_STATUS_BLOCK_PRIOR` (Snorlax Toxic ≈0.14 Immunity-dominated).
          • already_block — the opp active already carries a major status (can't double-apply); NOT Leech Seed.
          • sleep_clause_block — a SLEEP move fails if ANY opp mon is already asleep via a NON-Rest source
            (`sleep_is_deterministic==0`). Rest self-sleep does NOT consume our cap (the user's rule). The
            per-mon Rest flag is the existing gen3_sleep_wake_belief_v1 `sleep_is_deterministic` (reused).
          • has_sub — the opp active behind a Substitute blocks EVERY status move (incl. Leech Seed); read
            from the public Substitute volatile in `ctx.opp_ctx_raw` at `_SUBSTITUTE_CTX_IDX`.
        `known` = the value rests on CERTAIN (public) info — a type/already-statused/Sleep-Clause/Substitute
        hard block OR a revealed ability — vs a Smogon-prior estimate. No move-belief gradient (OUR moves are
        certain). UNCOVERED residual: Yawn (delayed sleep, no status_inflicted), Leech-Seed-already-seeded."""
        B, device = ctx.batch_size, ctx.device
        ar = torch.arange(B, device=device)
        our_act = ctx.our_active_idx                                  # [B]
        opp_act = TEAM_SIZE + ctx.opp_active_local                    # [B] opp-active global slot
        has_opp = ctx.hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, -1].any(dim=1).float()  # [B]
        our_alive = (ctx.hp_and_active[ar, our_act, 0] > 0).float()   # [B]
        gate = (has_opp * our_alive)[:, None]                         # [B,1]

        move_ids = ctx.all_move_ids[ar, our_act, :]                   # [B,4] request-slot order
        inflicts = self.MOVE_INFLICTS_STATUS[move_ids]               # [B,4]
        acc = self.MOVE_ACCURACY[move_ids]                          # [B,4] (Toxic .85, WoW .75, T-Wave 1, …)
        sidx = self.MOVE_STATUS_CAT[move_ids]                       # [B,4] long (0 = not a status move)
        is_sleep = self.MOVE_IS_SLEEP[move_ids]                     # [B,4]
        blocked_if_statused = self.MOVE_BLOCKED_IF_STATUSED[move_ids]  # [B,4] (0 for Leech Seed)

        # type immunity (per move) — max over the opp active's two types.
        t1 = ctx.type1_ids[ar, opp_act]                            # [B]
        t2 = ctx.type2_ids[ar, opp_act]
        ti = self.MOVE_STATUS_TYPE_IMMUNE[move_ids]                 # [B,4,N_TYPE_IDX]
        type_immune = torch.maximum(ti.gather(2, t1[:, None, None].expand(B, 4, 1)).squeeze(2),
                                    ti.gather(2, t2[:, None, None].expand(B, 4, 1)).squeeze(2))  # [B,4]

        # ability immunity — revealed → exact; unrevealed (id 0) → the species Smogon-prior marginal.
        opp_ability = ctx.ability1_ids[ar, opp_act]                # [B] (0 if unrevealed)
        opp_species = ctx.species_ids[ar, opp_act]                 # [B]
        abl_rev = self.ABILITY_STATUS_BLOCK[opp_ability].gather(1, sidx)           # [B,4]
        abl_prior = self.SPECIES_STATUS_BLOCK_PRIOR[opp_species].gather(1, sidx)   # [B,4]
        revealed = (opp_ability > 0).float()[:, None]              # [B,1]
        ability_block = revealed * abl_rev + (1.0 - revealed) * abl_prior          # [B,4]

        # already-statused (opp active) — any non-None status bit → blocks a MAJOR status (not Leech Seed).
        opp_cond = ctx.pokemon_part[ar, opp_act,
                                    POKEMON_CONDITION_OFFSET + 1:POKEMON_CONDITION_OFFSET + 7]  # [B,6]
        already_statused = (opp_cond.sum(dim=1) > 0.5).float()[:, None]            # [B,1]
        already_block = already_statused * blocked_if_statused                     # [B,4]

        # Sleep Clause — ANY opp mon asleep via a NON-Rest source consumes our one-sleep cap.
        opp_slp = ctx.pokemon_part[:, TEAM_SIZE:2 * TEAM_SIZE, POKEMON_CONDITION_OFFSET + _COND_SLP_IDX]  # [B,6]
        opp_rest = ctx.pokemon_part[:, TEAM_SIZE:2 * TEAM_SIZE, POKEMON_SLEEP_BELIEF_OFFSET]  # [B,6] is_rest
        nonrest_sleep = opp_slp * (1.0 - opp_rest)                                 # [B,6]
        sleep_clause = (nonrest_sleep.sum(dim=1) > 0.5).float()[:, None]           # [B,1]
        sleep_block = sleep_clause * is_sleep                                      # [B,4]

        # Substitute — the opp active behind a Sub blocks EVERY status move (incl. Leech Seed) in gen3. Read
        # the public Substitute volatile from the opp active context (boosts ++ volatiles). Applies to ALL
        # inflicting moves (not just majors), so it folds in as a flat per-channel factor below.
        has_sub = (ctx.opp_ctx_raw[:, _SUBSTITUTE_CTX_IDX] > 0.5).float()[:, None]  # [B,1]

        p_land = (inflicts * acc * (1.0 - type_immune) * (1.0 - ability_block)
                  * (1.0 - already_block) * (1.0 - sleep_block) * (1.0 - has_sub))  # [B,4]
        # `known` = the value rests on CERTAIN info (a hard block — type/already-statused/Sleep-Clause/
        # Substitute, all PUBLIC — or a revealed ability) vs a Smogon-prior estimate.
        certain = torch.clamp(type_immune + already_block + sleep_block + has_sub + revealed, max=1.0)  # [B,4]
        known = inflicts * certain                                                 # [B,4]
        return torch.cat([p_land, known], dim=1) * gate                            # [B, _DMG_STATUS]

    def forward(self, ctx: 'ExtractorContext', move_belief_logits: torch.Tensor,
                spread_belief: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = ctx.batch_size
        device = ctx.device
        eps = 1e-6
        ar = torch.arange(B, device=device)
        opp_act = TEAM_SIZE + ctx.opp_active_local                         # [B] global opp-active slot
        # gen3_unified_spread_belief_v1: the believed opp-active stats [B,5] (atk,def,spa,spd,spe), or None
        # (→ the legacy hand-coded de-timid offense / neutral bulk constants below).
        sb = spread_belief[ar, ctx.opp_active_local] if spread_belief is not None else None

        # No-opp-active gate (forced switch / battle start / dummy zero-obs): zero the whole block.
        has_opp = ctx.hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, -1].any(dim=1).float()  # [B]

        # --- Attacker = opp active (revealed species; hidden spread → fixed 252 EV / 31 IV / ×1.1) ---
        a_base = self.BASE_STATS[ctx.species_ids[ar, opp_act]]            # [B,6] [hp,atk,def,spa,spd,spe]
        off_const = 31.0 + 252.0 / 4.0 + 5.0                              # IV + EV/4 + 5 (legacy de-timid)
        atk = sb[:, _SB_ATK] if sb is not None else (2.0 * a_base[:, 1] + off_const) * 1.1   # [B] believed/legacy
        spa = sb[:, _SB_SPA] if sb is not None else (2.0 * a_base[:, 3] + off_const) * 1.1   # [B]
        # gen3_unified_op_physics_v1: fold the OPP active's OFFENSIVE stat-stage boosts (Dragon Dance /
        # Calm Mind / Swords Dance) into its offense — a +2 sweeper's Atk is doubled (the worst
        # damage-calc edge case). Stages read from the opp active-context; our-side read below for defence.
        opp_b_atk, opp_b_def, opp_b_spa, opp_b_spd, opp_b_spe = self._boost_stages(ctx.opp_ctx_raw)
        atk = atk * self._boost_mult(opp_b_atk)
        spa = spa * self._boost_mult(opp_b_spa)
        # gen3_unified_op_physics_v1: BURN halves the opp attacker's PHYSICAL attack (atk only; spa unhurt).
        opp_burn = ctx.pokemon_part[ar, opp_act, POKEMON_CONDITION_OFFSET + _COND_BRN_IDX]   # [B]
        atk = atk * torch.where(opp_burn > 0.5, atk.new_tensor(0.5), atk.new_tensor(1.0))
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
        # OUR ACTIVE defender's DEFENSIVE boosts (only the active mon carries boosts in gen3 — bench reset).
        our_b_atk, our_b_def, our_b_spa, our_b_spd, our_b_spe = self._boost_stages(ctx.our_ctx_raw)
        def_boost = torch.ones_like(def_stat); def_boost[ar, ctx.our_active_idx] = self._boost_mult(our_b_def)
        spd_boost = torch.ones_like(spd_stat); spd_boost[ar, ctx.our_active_idx] = self._boost_mult(our_b_spd)
        def_stat = def_stat * def_boost
        spd_stat = spd_stat * spd_boost
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
        acc_all = torch.cat([self.MOVE_ACCURACY, torch.ones(n_hp, device=device)])             # [C] HP is 100%
        fixed_all = torch.cat([self.MOVE_FIXED_DAMAGE, torch.zeros(n_hp, device=device)])       # [C] HP not fixed
        # Fixed-damage moves read BP 0 → derived category STATUS → MOVE_PHYS 0; route them onto their TYPE's
        # channel instead (Seismic Toss=Fighting=phys, Night Shade=Ghost=phys), matching the outgoing block.
        phys_all = torch.where(fixed_all > 0, self.TYPE_IS_PHYS[mty_all], phys_all)             # [C]
        # gen3_unified_op_physics_v1: per-candidate WEATHER BP modifier (rain/sun × Water/Fire), [B,C].
        weather_mult = self._weather_mult(ctx.weather_feature, (mty_all == _WATER_TIDX).float()[None, :],
                                          (mty_all == _FIRE_TIDX).float()[None, :])             # [B,C]
        w_all = torch.cat([w, w_hp.unsqueeze(-1) * hp_type_belief], dim=1)                      # [B,C]

        # --- gen3 damage per (defender, candidate), all differentiable in w (the shared physics
        # kernel — incoming roles: attacker = opp active, defenders = our 6, OUR-side screens) ---
        our_reflect = ctx.screen_feature[:, 0:1]                                                # [B,1]
        our_light_screen = ctx.screen_feature[:, 2:3]                                           # [B,1]
        high_frac, low_frac, crit_frac, ko_ramp, high_cb, ko_cb = self._damage_rolls(
            atk, spa, at1, at2, def_stat, spd_stat, maxhp, cur_hp, t1d, t2d,
            ctx.ability1_ids[:, :TEAM_SIZE], our_reflect, our_light_screen,
            bp_all, mty_all, phys_all, acc_all, fixed_all, weather_mult, eps)

        # --- per (defender, channel): HARD max of the belief-weighted roll/KO over the candidates ---
        # The dominant believed move owns each channel (the candidate-count-robust max, NOT a diluting
        # soft-max over ~400 moves). low/high/crit are monotone in damage → the same dominant move; pko is
        # its KO probability. Each feature is `max_c w_c · value_c` on the channel.
        wb = w_all[:, None, :]                                           # [B,1,C] (belief, broadcast over defenders)
        phys_mask = phys_all[None, None, :]
        spec_mask = 1.0 - phys_mask
        phys_low, spec_low = self._chan_max(wb * low_frac, phys_mask), self._chan_max(wb * low_frac, spec_mask)
        phys_high, spec_high = self._chan_max(wb * high_frac, phys_mask), self._chan_max(wb * high_frac, spec_mask)
        phys_crit, spec_crit = self._chan_max(wb * crit_frac, phys_mask), self._chan_max(wb * crit_frac, spec_mask)
        phys_pko, spec_pko = self._chan_max(wb * ko_ramp, phys_mask), self._chan_max(wb * ko_ramp, spec_mask)
        # gen3_unified_choice_band_v1: the CB-CONDITIONAL physical tail — the PHYSICAL-channel high-roll +
        # P(OHKO) computed with the opp Atk ×1.5. Same hard-max aggregation over the believed candidates;
        # special channel is CB-invariant so only the physical max is exposed (paired with p_cb below).
        phys_high_cb = self._chan_max(wb * high_cb, phys_mask)                                   # [B,6]
        phys_pko_cb = self._chan_max(wb * ko_cb, phys_mask)                                      # [B,6]
        # PER-CHANNEL accuracy + PROVENANCE of the dominant (max belief-weighted high-roll) believed move.
        # accuracy is gathered COHERENTLY at the channel's dominant-damage move (the one the rolls describe),
        # so {pko, accuracy} parameterize that threat's full outcome distribution. provenance is the dominant
        # move's belief weight (1.0 ≈ a REVEALED/pinned move, <1.0 = a usage-prior GUESS). argmax detached;
        # the gathered (acc fixed-buffer / belief weight) values carry the right gradient.
        wf = wb * high_frac                                                                      # [B,6,C]
        acc_exp = acc_all[None, None, :].expand(B, TEAM_SIZE, -1)                                # [B,6,C]

        def _chan_acc(channel_mask):
            wfc = wf * channel_mask                                                              # off-channel→0
            dom = wfc.argmax(dim=-1, keepdim=True)                                               # [B,6,1]
            acc = torch.gather(acc_exp, -1, dom).squeeze(-1)                                     # [B,6]
            return torch.where(wfc.amax(dim=-1) > eps, acc, torch.zeros_like(acc))               # 0 if no threat
        phys_acc = _chan_acc(phys_mask)
        spec_acc = _chan_acc(spec_mask)

        dom_idx = wf.argmax(dim=-1, keepdim=True)                                                # [B,6,1] (overall)
        provenance = torch.gather(w_all[:, None, :].expand(-1, TEAM_SIZE, -1), -1, dom_idx).squeeze(-1)
        provenance = torch.where(wf.amax(dim=-1) > eps, provenance, torch.zeros_like(provenance))
        # P(outspeed): our mon's REAL speed vs the opp active's fast-tail speed (252/+nat) — a per-mon
        # point estimate (paralysis/boosts not modelled in v1). Logistic over the stat difference.
        our_spe = (2.0 * d_base[..., 5] + iv[..., 5] + ev[..., 5] / 4.0 + 5.0) * nat[..., 4]     # [B,6]
        opp_spe = sb[:, _SB_SPE] if sb is not None else (2.0 * a_base[:, 5] + off_const) * 1.1   # [B] believed/legacy
        # gen3_unified_op_physics_v1: SPEED boosts (Agility / DD) + PARALYSIS (×0.25) on the active mons fold
        # into p_outspeed — so "I set up DD → I outspeed" / "I paralyze them → I outspeed" are both priced.
        opp_para = ctx.pokemon_part[ar, opp_act, POKEMON_CONDITION_OFFSET + _COND_PAR_IDX]       # [B]
        opp_spe = opp_spe * self._boost_mult(opp_b_spe) * torch.where(
            opp_para > 0.5, opp_spe.new_tensor(_DMG_PARA_SPEED), opp_spe.new_tensor(1.0))
        our_para = ctx.pokemon_part[ar, ctx.our_active_idx, POKEMON_CONDITION_OFFSET + _COND_PAR_IDX]  # [B]
        our_spe_mult = torch.ones_like(our_spe)                                                  # [B,6]
        our_spe_mult[ar, ctx.our_active_idx] = self._boost_mult(our_b_spe) * torch.where(
            our_para > 0.5, our_para.new_tensor(_DMG_PARA_SPEED), our_para.new_tensor(1.0))
        our_spe = our_spe * our_spe_mult
        p_outspeed = torch.sigmoid((our_spe - opp_spe[:, None]) / _DMG_SPEED_SCALE)              # [B,6]

        # Slot order == the named _DMG_IDX_* offsets: [phys_low, phys_high, phys_crit, phys_pko, phys_acc,
        #               spec_low, spec_high, spec_crit, spec_pko, spec_acc, outspeed, prov]
        feats = torch.stack([phys_low, phys_high, phys_crit, phys_pko, phys_acc,
                             spec_low, spec_high, spec_crit, spec_pko, spec_acc,
                             p_outspeed, provenance], dim=-1)                                     # [B,6,12]
        feats = feats * defender_alive[:, :, None] * has_opp[:, None, None]                       # gates

        # --- opp-active believed-EFFECT threat: belief weight of the most-believed move of each category ---
        # p_k = max_m (w_m·flag_mk) = P(the single most-believed move of category k) — the SAME collapse-free
        # max the chip/pko channels use. (A full-axis noisy-OR over ~400 moves saturated to ~1 from the floor
        # alone — the candidate-count dilution `_chan_max` exists to avoid.) [recovery, status, phaze, boost,
        # hazard, protect]; over the num moves only (HP is damaging → flags 0).
        w_eff = w[:, :, None] * self.MOVE_EFFECT_FLAGS[None, :, :]       # [B, M, K]
        p_effect = w_eff.amax(dim=1) * has_opp[:, None]                 # [B, K], gated

        # gen3_unified_move_system_v1: per-STATUS secondary threat from the opp active's DAMAGING moves
        # (Body Slam para, Rock Slide flinch, Ice Beam freeze — the axis the binary `status` flag missed).
        # realized P(effect k) = max_m (w_m · chance_mk · acc_m) × Serene Grace(opp active). Accuracy is
        # folded (a secondary only fires on a hit — the same physics-in-the-op principle as pko: e.g. Zap
        # Cannon's 100% para × 50% acc → 0.5). NO speed coupling — flinch's move-first dependence is left
        # to attention (owner decision). Order == damage_tables.SECONDARY_COLS. (Defender Shield Dust is a
        # rare v2 follow-up — the effect block is opp-active-level, not per-defender.)
        w_sec = (w * self.MOVE_ACCURACY[None, :])[:, :, None] * self.MOVE_SECONDARY[None, :, :]  # [B,M,10]
        opp_serene = self.ABILITY_SECONDARY_MULT[ctx.ability1_ids[ar, opp_act]]                  # [B]
        p_sec = (w_sec.amax(dim=1) * opp_serene[:, None]).clamp(max=1.0) * has_opp[:, None]      # [B,10]

        # gen3_unified_choice_band_v1: P(opp active holds Choice Band), collapsed to 0/1 once its item is
        # revealed (item_id==CB → 1; any OTHER revealed item → 0; unrevealed id==0 → the species usage prior).
        # The op's outgoing block applies CB ×1.5 deterministically for OUR known item; here it's a belief.
        opp_item = ctx.item_ids[ar, opp_act]                                                     # [B]
        cb_prior = self.SPECIES_CB_PRIOR[ctx.species_ids[ar, opp_act]]                           # [B]
        revealed_cb = (opp_item == self.cb_item_num).float()                                     # [B]
        unrevealed = (opp_item == 0).float()                                                     # [B] all-zero id
        p_cb = (revealed_cb + (1.0 - revealed_cb) * unrevealed * cb_prior) * has_opp             # [B]
        # CB-conditional physical tail, gated like the modal per-mon feats (alive defender + opp present).
        cb_gate = defender_alive * has_opp[:, None]                                              # [B,6]
        cb_block = torch.cat([phys_high_cb * cb_gate, phys_pko_cb * cb_gate, p_cb[:, None]], dim=1)  # [B, _DMG_CB]

        block = torch.cat([feats.reshape(B, TEAM_SIZE * self.per_mon), p_effect, p_sec, cb_block], dim=1)  # [B, incoming_dim]
        # OUTGOING (our active → opp active, per-move action-aligned): appended after the incoming block
        # when enabled (widens out_dim; both projections auto-size). Reuses the shared `_rolls` physics.
        # gen3_unified_status_landing_v1: the per-OUR-move status-landing block rides the SAME outgoing
        # direction (status moves the damage block can't price), so it's appended right after it.
        if self.outgoing:
            block = torch.cat([block, self._outgoing_block(ctx, spread_belief),
                               self._status_landing(ctx)], dim=1)  # [B, out_dim]
        # Read-only stash of the PRE-gain physics (the interpretable damage fractions / P(KO) / accuracy),
        # for the prober/forensic decode — the learned out_gain only rescales for the projection.
        self.last_raw_block = block.detach()
        return block * self.out_gain                                    # learnable per-channel adapter (×only)


# Effect-scalar column order (== damage_tables.MOVE_EFFECT_COLS) for the prober decode.
_DMG_EFFECT_COLS = ("recovery", "status", "phaze", "boost", "hazard", "protect")


def decode_damage_block(row, *, outgoing: bool, team_size: int = TEAM_SIZE):
    """Decode ONE `DamageOperator.last_raw_block[i]` row (the PRE-gain physics) into a human-readable dict
    for the prober / forensic tooling — the single source of truth for the operator's output layout, mirrored
    by the TUI. Uses the named `_DMG_IDX_*` offsets: per our mon the incoming threat
    `[low,high,crit,pko,acc]×{phys,spec} + p_outspeed + provenance` (slot 0 = our active, slots 1..5 = the
    **safe-switch** bench reads), then the 6 opp-active believed-EFFECT scalars, then (if `outgoing`) our 4
    moves' outgoing damage `[low,high,crit,pko]` + p_outspeed + per-move secondary, then the
    gen3_unified_status_landing_v1 `status_landing` block — per move `{p_land, known}` (request-slot order =
    action logits 6+k)."""
    r = [float(x) for x in row]

    def _chan(b):
        return {"low": r[b], "high": r[b + 1], "crit": r[b + 2], "pko": r[b + 3], "acc": r[b + 4]}

    incoming = []
    for i in range(team_size):
        b = i * _DMG_PER_MON
        incoming.append({
            "phys": _chan(b + _DMG_IDX_PHYS_LOW), "spec": _chan(b + _DMG_IDX_SPEC_LOW),
            "p_outspeed": r[b + _DMG_IDX_OUTSPEED], "provenance": r[b + _DMG_IDX_PROVENANCE],
        })
    eb = team_size * _DMG_PER_MON
    sb = eb + _DMG_EFFECT                              # incoming per-status secondary base
    cbb = sb + _DMG_INCOMING_SEC                       # gen3_unified_choice_band_v1: CB block base
    out = {"incoming": incoming,
           "effect": {c: r[eb + j] for j, c in enumerate(_DMG_EFFECT_COLS)},
           "incoming_secondary": {c: r[sb + j] for j, c in enumerate(_SECONDARY_COLS)},
           # CB-conditional physical tail (per our mon) + the shared P(opp holds Choice Band).
           "choice_band": {"phys_high_cb": [r[cbb + i] for i in range(team_size)],
                           "phys_pko_cb": [r[cbb + team_size + i] for i in range(team_size)],
                           "p_cb": r[cbb + 2 * team_size]},
           "outgoing": None}
    if outgoing:
        ob = cbb + _DMG_CB                              # outgoing damage base (after the CB block)
        osb = ob + _DMG_OUT_N_MOVES * _DMG_OUT_PER_MOVE + 1   # outgoing per-move secondary base (after p_outspeed)
        slb = ob + _DMG_OUTGOING                        # status-landing base (after the whole outgoing block)
        out["outgoing"] = {
            "moves": [{"low": r[ob + k * _DMG_OUT_PER_MOVE], "high": r[ob + k * _DMG_OUT_PER_MOVE + 1],
                       "crit": r[ob + k * _DMG_OUT_PER_MOVE + 2], "pko": r[ob + k * _DMG_OUT_PER_MOVE + 3]}
                      for k in range(_DMG_OUT_N_MOVES)],
            "p_outspeed": r[ob + _DMG_OUT_N_MOVES * _DMG_OUT_PER_MOVE],
            "secondary": [{c: r[osb + k * _N_SECONDARY + j] for j, c in enumerate(_SECONDARY_COLS)}
                          for k in range(_DMG_OUT_N_MOVES)],
        }
        # gen3_unified_status_landing_v1: per-OUR-move P(status lands) + a `known` bit (request-slot order).
        out["status_landing"] = [{"p_land": r[slb + k], "known": r[slb + _DMG_STATUS_N_MOVES + k]}
                                 for k in range(_DMG_STATUS_N_MOVES)]
    return out


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
    """Orchestrates the phase modules. Data flow (bracketed phases are flag-gated; all off = the
    baseline ObsUnpack → PokemonEncoder → TeamTransformer → CLSPool → ProjectionAssembler):
        ObsUnpack → PokemonEncoder → [BeliefSlots?] → TeamTransformer → [BeliefHead?] → [MoveBelief?]
          → CLSPool → [HiddenOppBeliefPool?] → [DamageOperator?] → ProjectionAssembler
    then a final pre-projection LayerNorm + Linear + ReLU head per side. `MoveBelief` reinjects the
    believed moveset into the opp tokens BEFORE the pools (so it flows to the heads via cross-attention);
    `DamageOperator` runs AFTER the pools and consumes the move-belief logits, appending its features to
    both projection inputs (it does not enter the token stream). The optional `WinProbHead`
    (`win_prob_mode`) reads `value_pooled` AFTER the pools and stashes a P(win) logit as a SIDE readout
    (never in pi/vf — leak-safe). The embedding tables live in
    `self.embeddings` (shared) and are passed into the phases that need them. See
    `src/agents/model/CLAUDE.md` for the phase-module contract."""

    def __init__(self, observation_space: spaces.Dict, layout: Optional[Dict[str, Any]] = None,
                 mappings: Optional[Dict[str, Any]] = None, log_level: LogLevel = LogLevel.QUIET,
                 attend_unrevealed_opponents: bool = False, opp_belief_cls_k: int = 0,
                 value_active_readout: bool = False, opp_belief_slots: bool = False,
                 move_belief_mode: str = "off", opp_belief_latent: bool = False,
                 damage_op: bool = False, move_prior_fusion: bool = False,
                 mask_incoming_damage_obs: bool = False, win_prob_mode: str = "none",
                 damage_outgoing: bool = False, move_candidate_floor: float = 0.0,
                 move_latent: bool = False, spread_belief: bool = False,
                 mask_active_move_scalars_obs: bool = False, mask_move_effects_obs: bool = False):
        super().__init__()
        self.layout = layout
        self.mappings = mappings
        self.log_level = log_level
        # Behavioral toggle (no weight-shape change): unmask the opponent's still-hidden
        # party so the transformer attends to it. Version-checked, not in ARCH_SIGNATURE.
        self.attend_unrevealed_opponents = attend_unrevealed_opponents
        # Ablation toggle (no weight-shape change): zero the incoming-damage obs block out of the model's
        # view (the unified DamageOperator replaces it; the reward still uses it). Version-checked.
        self.mask_incoming_damage_obs = mask_incoming_damage_obs
        # gen3_unified_spread_belief_v1: the other two --unified-obs disable-redundant masks (stored on the
        # root for arch_toggles_from_model threading; passed to ObsUnpack below).
        self.mask_active_move_scalars_obs = mask_active_move_scalars_obs
        self.mask_move_effects_obs = mask_move_effects_obs
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

        # gen3_unified_move_system_v1: the mechanics-grounded move latent (structural toggle — widens the
        # move-network input → state_dict-changing; OFF byte-identical). Required by the Stage-3 latent
        # grading aux (the loss reads its latent_table).
        self.move_latent = move_latent

        # Phase modules (constructed before the dummy forward below).
        self.embeddings = Embeddings(layout)
        self.unpack = ObsUnpack(layout, attend_unrevealed_opponents=attend_unrevealed_opponents,
                                mask_incoming_damage_obs=mask_incoming_damage_obs,
                                mask_active_move_scalars_obs=mask_active_move_scalars_obs,
                                mask_move_effects_obs=mask_move_effects_obs)
        self.pokemon_encoder = PokemonEncoder(layout, move_latent=move_latent)
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
        # Stashed each forward [B,6] bool: which opponent team slots are un-revealed (believed) — the
        # single-sourced `ctx.opp_believed_mask`. A read-only side stash (does NOT change the forward
        # output, so the off/baseline path stays byte-identical) so eval/forensic tooling can decode
        # `last_belief_logits["species"]` for exactly the hidden slots (see inference/belief_decode.py).
        self.last_opp_believed_mask: Optional[torch.Tensor] = None
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
        # Prior fusion (the unified two-part belief): fold the Smogon move-frequency prior into the
        # move-belief head as a log-odds residual + pin revealed moves certain. Requires the head to exist
        # (move_belief_mode != off). OFF reproduces the from-scratch head byte-for-byte (no buffer, no
        # forward change) — a forward-behavior toggle (no weight-shape change, version-checked).
        self.move_prior_fusion = move_prior_fusion
        if move_prior_fusion and move_belief_mode == "off":
            raise ValueError(
                "move_prior_fusion=True requires move_belief_mode != off — the prior fuses INTO the "
                "move-belief head's logits; with no head there is nothing to fuse. Set --move-belief-mode "
                "revealed (or both/unrevealed), or disable --move-prior-fusion."
            )
        self.move_belief = (
            MoveBelief(layout['max_moves'], layout['move_embedding_dim'],
                       prior_fusion=move_prior_fusion, n_species=layout['max_species'],
                       move_candidate_floor=move_candidate_floor)
            if move_belief_mode != "off" else None
        )
        self.last_move_belief_logits: Optional[torch.Tensor] = None
        # gen3_unified_move_system_v1: the [n_moves, MOVE_LATENT_DIM] context-free move-latent table,
        # stashed each forward (training only) for the Stage-3 latent grading aux — a side stash, never
        # fed forward (the per-slot latent is what flows; the table is the grading TARGET).
        self.last_move_latent_table: Optional[torch.Tensor] = None
        # gen3_unified_spread_belief_v1: the THIRD belief leg — predicts the opp's hidden SPREAD (5 derived
        # stats) per slot, reinjected into the opp token, consumed by the DamageOperator (replacing its
        # hand-coded opp-spread constants). STRUCTURAL toggle (widens nothing in the projection — it enriches
        # the opp token like MoveBelief). Requires move_belief_mode != off only if damage_op is on (the op is
        # the consumer); built whenever the flag is set. Stash for the supervision loss + the op.
        self.spread_belief_enabled = spread_belief
        self.spread_belief = SpreadBelief(layout['max_species']) if spread_belief else None
        self.last_spread_belief: Optional[torch.Tensor] = None
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
        # OUTGOING direction (our active → opp active, per-move action-aligned): requires the op itself.
        self.damage_outgoing = damage_outgoing
        if damage_outgoing and not damage_op:
            raise ValueError(
                "damage_outgoing=True requires damage_op=True — the outgoing per-move block is emitted by "
                "the DamageOperator. Enable --damage-op (--unified-damage both), or drop the outgoing flag."
            )
        self.damage_op = DamageOperator(layout, outgoing=damage_outgoing) if damage_op else None
        # Stored on the root so arch_toggles_from_model can thread it to the eval/self-play workers
        # (the move-prior gate is a version-checked forward-behavior toggle).
        self.move_candidate_floor = move_candidate_floor
        # Value-head active readout (weight-shape via flag): adds our_active_refined (D_MODEL) to the
        # value projection. OFF reproduces the baseline value head byte-for-byte (no ARCH_SIGNATURE bump).
        self.value_active_readout = value_active_readout
        self.assembler = ProjectionAssembler(layout, value_active_readout=value_active_readout)

        # Auxiliary WIN-PROBABILITY head (tri-state `win_prob_mode`): a calibrated P(win|state) readout
        # off `value_pooled`. 'none' = no module (baseline byte-for-byte, NOT in pi/vf so projection dims
        # are unchanged either way). 'read_only' = the head trains its OWN params on a STOP-GRAD
        # value_pooled (a pure, risk-free diagnostic — zero gradient to the trunk). 'shaping' = gradient
        # flows into the shared trunk (the win objective shapes the representation). It is a SIDE readout
        # (stashed for the aux loss + prober, never concatenated into pi/vf — leak-safe). The
        # state_dict-changing toggle is 'none'↔head; the mode itself is resume-immutable (version-checked).
        if win_prob_mode not in ("none", "read_only", "shaping"):
            raise ValueError(f"win_prob_mode must be none|read_only|shaping, got {win_prob_mode!r}")
        self.win_prob_mode = win_prob_mode
        self.win_head = WinProbHead() if win_prob_mode != "none" else None
        # Stashed each forward when the head is on (the [B,1] win logit, or None). Read by the PPO aux
        # loss + the offline prober/eval; NEVER fed into pi/vf (the OUTCOME label can't leak).
        self.last_win_prob_logits: Optional[torch.Tensor] = None

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
        # Expose which opp slots are believed (hidden) so eval/forensic tooling can decode the belief
        # head's per-slot species prediction for exactly those slots. Read-only stash — never read by
        # the forward itself, so the off/baseline output is unchanged.
        self.last_opp_believed_mask = ctx.opp_believed_mask
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
            # Prior-fusion inputs (used only when move_prior_fusion): the opp slots' species nums (for the
            # prior lookup) + their revealed move ids (id>0 ⇒ seen ⇒ pinned certain). Slices of ctx; cheap.
            opp_species_ids = ctx.species_ids[:, TEAM_SIZE:]                  # [B, 6]
            opp_move_ids = ctx.all_move_ids[:, TEAM_SIZE:, :]                 # [B, 6, 4]
            their_team_out, self.last_move_belief_logits = self.move_belief(
                their_team_out, mb_mask, self.embeddings.move_embedding,
                opp_species_ids=opp_species_ids, opp_move_ids=opp_move_ids)
        else:
            self.last_move_belief_logits = None
        # gen3_unified_spread_belief_v1: predict + reinject the opp's hidden SPREAD (revealed slots), and
        # stash the believed stats [B,6,5] for the DamageOperator (consumed at the opp active slot, replacing
        # its hand-coded spread constants) + the speed-supervision loss. Enriches the opp tokens before the
        # CLS pools, like MoveBelief. Hidden slots aren't enriched (their species num 0 → flat prior) and the
        # op only reads the (revealed) active slot.
        if self.spread_belief is not None:
            their_team_out, self.last_spread_belief = self.spread_belief(
                their_team_out, ~ctx.opp_believed_mask, ctx.species_ids[:, TEAM_SIZE:])
        else:
            self.last_spread_belief = None
        # gen3_unified_move_system_v1: stash the context-free move-latent table for the Stage-3 latent
        # grading aux (training only — a side stash, never fed forward). Gated on is_grad_enabled so
        # rollout / eval / inference pay nothing.
        self.last_move_latent_table = None
        if self.move_latent and torch.is_grad_enabled():
            self.last_move_latent_table = self.pokemon_encoder.move_latent_encoder.latent_table(self.embeddings)
        our_team_pooled, their_team_pooled, our_active_refined, value_pooled = self.cls_pool(
            our_team_out, their_team_out, ctx
        )
        # Auxiliary win-probability readout (flag-guarded; None when off). Reads the whole-board
        # value_pooled and stashes a [B,1] logit for the aux loss + the prober/eval. NOT fed into the
        # assembler (a side readout — the future OUTCOME label can't leak into pi/vf). `read_only` feeds
        # a STOP-GRAD value_pooled (head-only training, no trunk gradient); `shaping` feeds it live.
        # Computed on EVERY forward (one small MLP) so eval/inference can read P(win) too — its cost is
        # negligible and it is never gated off, since the prober reads it under no_grad.
        if self.win_head is not None:
            wp_in = value_pooled if self.win_prob_mode == "shaping" else value_pooled.detach()
            self.last_win_prob_logits = self.win_head(wp_in)
        else:
            self.last_win_prob_logits = None
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
            # Optional gradient-checkpointing: the op materialises several [B,6,~416] activations, so at
            # batch 16384 it recomputes in backward (idle-GPU compute for ~GBs of VRAM), same gate as the
            # transformer. Bit-exact (no dropout/RNG); a no-op under inference. ctx is a non-tensor arg
            # (passed through by use_reentrant=False); last_move_belief_logits carries the grad.
            if self.damage_op.grad_checkpointing and torch.is_grad_enabled():
                damage_block = checkpoint(self.damage_op, ctx, self.last_move_belief_logits,
                                          self.last_spread_belief, use_reentrant=False)
            else:
                damage_block = self.damage_op(ctx, self.last_move_belief_logits, self.last_spread_belief)
        # Read-only stash for the prober/forensic decode (the operator's per-mon incoming + outgoing
        # damage features) — never read by the forward itself, so the off/baseline output is unchanged.
        self.last_damage_block = damage_block
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
