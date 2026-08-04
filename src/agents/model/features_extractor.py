import torch
from torch.utils.checkpoint import checkpoint
import numpy as np
from dataclasses import dataclass, replace
from gymnasium import spaces
from typing import Callable, Dict, Any, Optional, Sequence, Tuple
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
)
from agents.observation.moves import HIDDEN_POWER_MOVE_NUM
from agents.model.team_signature import TEAM_SIGNATURE_DIM, TEAM_SIGNATURE_MOVES
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
# gen3_arch_constants_v1: the architecture constants now live in `arch_constants.py` so
# `damage_op.py` can import them without a cycle. Re-exported here UNCHANGED — this module
# remains the documented import surface (`from agents.model.features_extractor import D_MODEL`).
from agents.model.arch_constants import (  # noqa: F401  (re-export)
    ROLE_TOKEN_SIZE,
    PROJECTION_DIM,
    MOVE_NET_HIDDEN,
    MOVE_LATENT_HIDDEN,
    MOVE_LATENT_DIM,
    ROLE_ENCODER_HIDDEN,
    ACTIVE_CTX_HIDDEN,
    NET_ARCH,
    N_HISTORY_TURNS,
    ZARCH_DIM,
    ZARCH_ATOM_HIDDEN,
    POINTER_HIDDEN,
    D_MODEL,
    TRANSFORMER_N_LAYERS,
    TRANSFORMER_N_HEADS,
    TRANSFORMER_FFN_DIM,
)

# Token group ids for the unified transformer's type embedding.
TOKEN_TYPE_OUR_TEAM = 0
TOKEN_TYPE_THEIR_TEAM = 1
TOKEN_TYPE_HISTORY = 2
TOKEN_TYPE_GLOBAL = 3
# gen3_entity_move_seats_v1 (v54, Stage 1 of the entity generation): move ENTITY seats in the trunk.
TOKEN_TYPE_OUR_MOVE = 4        # E3 — our active's 4 request-ordered move tokens
TOKEN_TYPE_THEIR_THREAT = 5    # E4 — the opp active's top-K believed threat-move tokens
NUM_TOKEN_TYPES = 6


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
    # gen3_op_move_align_v1: OUR active mon's 4 moves in REQUEST-slot order (action 6+k) — the
    # DamageOperator's OUTGOING per-move blocks read THESE (not all_move_ids[our_active], which is
    # sorted-by-id) so their per-move output aligns with the action logits. [B,4] each:
    #   our_active_req_move_ids       — dex num (HP → 237 regardless of type)
    #   our_active_req_move_type_ids  — TypeEncoder index (our own Hidden Power is typed)
    #   our_active_req_move_legal     — current-decision choosability (1=choosable now), request order
    # (NB move_mask above is the PREV-turn, sorted-by-id mask used as a per-mon-block feature — a
    # different order + freshness, deliberately NOT what the op outgoing should use.)
    our_active_req_move_ids: torch.Tensor
    our_active_req_move_type_ids: torch.Tensor
    our_active_req_move_legal: torch.Tensor
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
        # gen3_cpu_damage_deleted_v1: the three `--unified-obs` ablation masks are GONE along with
        # the blocks they hid. They existed to A/B whether the GPU DamageOperator subsumed the CPU
        # incoming-damage / move-effect / active-move-scalar regions; that A/B is settled and the
        # producers are deleted, so there is nothing left to mask.

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

        # gen3_op_move_align_v1: OUR active's request-order move ids/types/legality (after the matchups,
        # so non_matchup_rest — which stops at the matchup offset — never sees these embedding IDs).
        # ids/type_ids → long for the op's table lookups; legal stays float (a 0/1 gate).
        _arm = reactive_layout['active_req_moves']
        _arm_base = reactive_start + _arm['offset']; _arm_per = _arm['per']
        our_active_req_move_ids = remaining_part[:, _arm_base : _arm_base + _arm_per].long()
        our_active_req_move_type_ids = remaining_part[:, _arm_base + _arm_per : _arm_base + 2 * _arm_per].long()
        our_active_req_move_legal = remaining_part[:, _arm_base + 2 * _arm_per : _arm_base + 3 * _arm_per]

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
            our_active_req_move_ids=our_active_req_move_ids,
            our_active_req_move_type_ids=our_active_req_move_type_ids,
            our_active_req_move_legal=our_active_req_move_legal,
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

    def hp_latent_block(self, embeddings: 'Embeddings', hp_type_idx: torch.Tensor,
                        hp_num: int) -> torch.Tensor:
        """``[n_hp, MOVE_LATENT_DIM]`` TYPED Hidden-Power latents (gen3_unified_topk_incoming_v1): all 17
        HP variants collide on ``hp_num`` (so the bare ``latent_table`` row is type-collapsed), but the
        DamageOperator's top-K candidate axis carries 16 TYPED HP candidates — each must speak its own
        type. Built the SAME way the move network builds its per-slot HP latent: the shared move
        embedding(hp_num) ⊕ the per-type ``type_embedding(hp_type_idx[j])`` ⊕ MOVE_ATTR[hp_num] (all-zero
        for HP), so HP-Rock ≠ HP-Ice in the latent axis. Aligned with the op's ``HP_TYPE_IDX`` order."""
        n_hp = hp_type_idx.shape[0]
        ids = torch.full((n_hp,), int(hp_num), device=self.MOVE_ATTR.device, dtype=torch.long)
        move_emb = embeddings.move_embedding(ids)                          # [n_hp, move_emb] (shared HP emb)
        type_emb = embeddings.type_embedding(hp_type_idx)                  # [n_hp, type_emb] (per-type)
        return self.forward(move_emb, type_emb, ids)                       # [n_hp, MOVE_LATENT_DIM]


class PokemonEncoder(torch.nn.Module):
    """Per-Pokémon encoding: embed + stitch the enriched vector, run the shared move
    processor + within-mon move self-attention, then the role encoder → 12×128 role tokens.

    When ``move_latent`` is on (gen3_unified_move_system_v1), a context-free `MoveLatentEncoder` latent
    is concatenated into the move-network input — a mechanics-grounded move identity (widens
    move_input_dim by MOVE_LATENT_DIM; OFF leaves the move network byte-identical)."""

    def __init__(self, layout: Dict[str, Any], move_latent: bool = False):
        super().__init__()
        # gen3_pointer_native_v1: `forward` ALWAYS stashes the per-(mon, move-slot) tokens — they are
        # the pointer action head's move-scorer input, and the pointer head is THE action head in this
        # generation (no flat action_net exists to fall back to). Cost: one reshape per forward.
        self.last_move_tokens: Optional[torch.Tensor] = None
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
        # gen3_pointer_native_v1: STASH the per-move tokens BEFORE they are flattened into the mon
        # vector below — one addressable vector per (mon, move slot), already contextualised by the
        # within-mon move self-attention above; the flatten on the next line is the only reason they
        # were never reachable. The pointer action head (the ONLY action head) scores move logit k from
        # these. NOTE the slot axis is SORTED-BY-ID order (MovesEncoder.get_sorted_moves), NOT
        # action/request order — the consumer MUST permute (see `_request_order_move_tokens`), which is
        # the `ordering_integrity.py` bug class the pointer head makes unrepresentable at the logits.
        self.last_move_tokens = mv_out.reshape(batch_size, n_poke, num_moves, MOVE_NET_HIDDEN[1])
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
    """Unified transformer over the entity seats: 6 our + 6 their team role tokens +
    N_HISTORY_TURNS history + 1 global, plus (gen3_entity_move_seats_v1) any `extra` entity
    seats appended after the global token (E3 our-move + E4 threat-move seats today). Adds
    token-type and history-positional embeddings, applies the encoder stack with a
    fainted/empty-history/seat key-padding mask, returns the two team token blocks + the
    refined extra seats."""

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
                embeddings: Embeddings,
                between_layers: "Optional[Callable[[torch.Tensor, int], torch.Tensor]]" = None,
                extra: "Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]" = None,
                ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """`extra` (gen3_entity_move_seats_v1): optional `(tokens [B,n,d_model], types [n] long,
        pad [B,n] bool)` — additional entity seats appended AFTER the global token, so every
        absolute slice (team/history/global, the refine callback's tail cat) is position-stable.
        The third return is the refined extra seats (None when no extra)."""
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
        if extra is not None:
            extra_tokens, extra_types, extra_pad = extra
            tokens = torch.cat([tokens, extra_tokens + self.token_type_emb(extra_types)], dim=1)
            key_padding_mask = torch.cat([key_padding_mask, extra_pad], dim=1)

        # Gradient checkpointing only helps when a graph is being built for backward
        # (the PPO update); under inference's no_grad it would be pure overhead, so gate on
        # torch.is_grad_enabled(). use_reentrant=False is the correct variant here (handles
        # non-grad inputs + autocast/RNG state); with dropout=0 the recompute is bit-exact.
        use_ckpt = self.grad_checkpointing and torch.is_grad_enabled()
        for i, layer in enumerate(self.transformer_layers):
            # gen3_iterative_damage_v1: BEFORE each of the first N layers, recompute the lean discrete
            # incoming damage from the CURRENT (being-enriched) tokens and inject it back onto our-mon
            # tokens (the callback no-ops past round N / when refinement is off). So each layer attends
            # over physics derived from the freshest belief — physics-in-the-loop, not one-shot post-hoc.
            if between_layers is not None:
                tokens = between_layers(tokens, i)
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
        extra_out = tokens[:, self._total_tokens:, :] if extra is not None else None
        return our_team_out, their_team_out, extra_out


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

    def species_logits(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens [B,6,D] → species logits [B,6,n_species]. Factored out of `forward` (mirrors
        `MoveBelief.move_logits`) so the bidirectional-threat refine can read P(species) over the
        CURRENT opp tokens MID-transformer (per round) and the final op can read it post-transformer —
        without also running the moves/latent heads. `forward` is left byte-identical (it computes its
        own `norm` once and reuses it for both heads); this standalone path is only called by the
        v36 expected-latent-defender (gated off by default), so the baseline forward is unchanged."""
        read = tokens.detach() if getattr(self, "detach_read", False) else tokens
        return self.species_head(self.norm(read))

    def species_posterior(self, tokens: torch.Tensor) -> torch.Tensor:
        """tokens [B,6,D] → P(species) [B,6,n_species], the v36 expected-latent defender's input.

        ⚠️ THE SPELLING IS LOAD-BEARING — do not "simplify" this back to `torch.softmax`.

        This one op was the ONLY thing in the whole extractor that Inductor could not codegen, and
        therefore the ONLY reason `--compile-extractor` ever set
        `torch._dynamo.config.suppress_errors`. `torch.softmax` lowers to a `[B,6,n_species]`
        numerator buffer plus a `[B,6,1]` denominator, and the CPU scheduler then trips
        `AssertionError: buf<N>` trying to fuse the division. Reproduced by
        `tmp/inductor_crash_repro.py`; `tmp/softmax_variant_probe.py` shows `.contiguous()`,
        `.clone()`, a 2-D reshape and a hand-rolled `exp / sum` ALL still fail — only the
        `log_softmax().exp()` factoring lowers to a shape Inductor accepts.

        Mathematically identical (`exp(log_softmax(x)) == softmax(x)`, and it goes through the same
        max-subtraction for stability); measured max|Δ| vs eager 5.1e-07, the same order as the
        compile's own float-reassociation noise. Pinned by `inductor_compile_test.py`, which fails
        loudly if a future edit reintroduces the uncompilable form."""
        return torch.log_softmax(self.species_logits(tokens), dim=-1).exp()

    def forward(self, their_team_out: torch.Tensor) -> Dict[str, torch.Tensor]:
        """their_team_out [B, 6, D] → {"species": [B,6,n_species], "moves": [B,6,n_moves],
        ["latent": [B,6,latent_dim]]}. The latent key is present only when the predictor is built."""
        # gen3_belief_grad_mode_v1: BeliefHead is a pure readout (no reinject), so `detached` simply stops the
        # aux-supervision gradient (species/moves/latent) from reaching the trunk — train the head only.
        read = their_team_out.detach() if getattr(self, "detach_read", False) else their_team_out
        h = self.norm(read)
        out = {"species": self.species_head(h), "moves": self.moves_head(h)}
        if self.latent_head is not None:
            out["latent"] = self.latent_head(read)
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


class PubValHead(WinProbHead):
    """PUBLIC-information value readout (`gen3_pubval_aux_v1`) — the WinProbHead architecture with a
    different, EXOGENOUS target: the frozen HUMAN-replay-calibrated public value `V_pub(public board)`
    (`agents.training.pubval`, 164k rated gen3ou games, held-out AUC ≈ 0.74, calibrated). Where the
    win-prob head learns P(win) from SELF-PLAY outcomes (inheriting the bootstrap's blind spots — a
    policy that never plays positionally never generates outcomes that price positional value), this
    head regresses the trunk toward how HUMAN game outcomes price the same public board — hazards,
    status, attrition, tempo — as a dense per-step target (V_pub moves turn by turn, so the trunk sees
    WHEN the game swung, not only how it ended: the credit-assignment lever). Under `shaping` its
    gradient flows into the shared trunk; the target is a pure function of PUBLIC state computed
    env-side (leak-free: the POC's turn-1 AUC ≈ 0.51 guard), NEVER in pi/vf, and NEVER in GAE (it is
    V^human, not V^π). Same module shape as WinProbHead (a named subclass so state_dict keys +
    reprs are self-documenting)."""


class ValueDistHead(torch.nn.Module):
    """Distributional VALUE readout — an INTERPRETABILITY side head over the return distribution.

    The scalar critic emits one number, E[Z] (expected shaped return). This head reads the same
    whole-board `value_pooled` summary and emits `bins` logits over a FIXED atom support
    `linspace(vmin, vmax, bins)`: `softmax(logits)` is the critic's predicted return DISTRIBUTION,
    not just its mean. That distribution is what makes "how is the model predicting" legible — a
    sharp spike = confident, a wide spread = uncertain, a bimodal shape = the critic sees a coinflip
    (e.g. "I win if this move hits, else I lose") — all invisible in the scalar V that collapses
    every shape to one mean. The categorical HL-Gauss parameterization (Phase A side head) is the
    `WinProbHead` pattern applied to the value target. Design:
    `designs/ai_v6/design_distributional_value_critic.md`.

    SIDE readout, leak-safe: the logits are stashed at `features_extractor.last_value_dist_logits`
    and read ONLY by the (future) aux loss + the offline prober/eval — NEVER concatenated into pi/vf,
    so the projection dims are unchanged either way (off byte-for-byte). The tri-state
    `value_dist_mode` controls the GRADIENT at the call site (`read_only` feeds a STOP-GRAD
    `value_pooled` — a pure, risk-free diagnostic that can't perturb the policy; `shaping` feeds it
    live so the distributional objective also shapes the shared trunk). `none` = this module is not
    built (the chain is byte-for-byte the baseline). The `atoms` buffer is non-persistent
    (deterministic from `bins`/`vmin`/`vmax`) so it stays out of the state_dict — only the head's
    params (whose final Linear is `bins`-wide) define the loadable shape."""

    def __init__(self, bins: int, vmin: float, vmax: float):
        super().__init__()
        if bins <= 0:
            raise ValueError(f"ValueDistHead bins must be > 0, got {bins}")
        if not vmax > vmin:
            raise ValueError(f"ValueDistHead requires vmax > vmin, got vmin={vmin}, vmax={vmax}")
        self.bins = bins
        # Small MLP off the value pool: LayerNorm → Linear → ReLU → Linear(→bins) — the WinProbHead
        # bottleneck, widened from 1 logit to `bins` (a categorical head over the return support).
        self.net = torch.nn.Sequential(
            torch.nn.LayerNorm(D_MODEL),
            torch.nn.Linear(D_MODEL, D_MODEL),
            torch.nn.ReLU(),
            torch.nn.Linear(D_MODEL, bins),
        )
        # Fixed atom support, non-persistent (deterministic from bins+range → out of the state_dict,
        # like the damage_tables buffers). Read by the loss (target projection) + the prober (atoms →
        # return units) + `mean()` below; the head's forward only needs the net.
        self.register_buffer("atoms", torch.linspace(vmin, vmax, bins), persistent=False)

    def forward(self, value_pooled: torch.Tensor) -> torch.Tensor:
        """value_pooled [B, D_MODEL] → per-atom logits [B, bins] (softmax ⇒ return distribution)."""
        return self.net(value_pooled)

    def mean(self, logits: torch.Tensor) -> torch.Tensor:
        """E[Z] = Σ atomsᵢ·softmax(logits)ᵢ — the scalar the distribution implies, [B, 1]. (Used by
        the prober / diagnostics; the Phase-A side head does NOT feed this into the scalar critic.)"""
        return (torch.softmax(logits, dim=-1) * self.atoms).sum(-1, keepdim=True)


# Logit at which a REVEALED (certain) move is pinned under prior fusion: sigmoid(10) ≈ 0.99995 ≈ P 1.
_REVEAL_LOGIT = 10.0

# gen3_typed_hp_belief_v1: the logit the bare typeless Hidden Power (num 237) is driven to once the
# typed composition has run — sigmoid(-30) ≈ 9.4e-14, i.e. hard off but FINITE, so the move-belief BCE
# sees a ~0 loss with a ~0 gradient there instead of the NaN a true -inf would produce. 237 survives
# only as the belief's internal PRESENCE channel (read BEFORE this is written); no consumer downstream
# of the composition may treat it as a real move.
_HP_PRESENCE_OFF_LOGIT = -30.0


def mask_typeless_hp(move_logits: torch.Tensor) -> torch.Tensor:
    """Drive the bare typeless Hidden Power channel (num 237) hard-off in a move posterior.

    Shared by BOTH `--hp-belief-mode` arms, because it is not part of what the ablation varies: 237
    carries **BP 0**, so it is not a move at all — leaving it live in the damage candidate set is the
    original "the opponent's Hidden Power reads immune" bug. The ablation varies whether the 16 TYPED
    channels are produced by a presence×type factorisation or predicted independently; both must agree
    that the typeless token is never a candidate."""
    out = move_logits.clone()
    out[..., HIDDEN_POWER_MOVE_NUM] = _HP_PRESENCE_OFF_LOGIT
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

    def move_logits(self, opp_tokens: torch.Tensor,
                    opp_species_ids: Optional[torch.Tensor] = None,
                    opp_move_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """The POSTERIOR move logits [B,6,M] (NO reinjection) — the head delta, optionally fused with the
        prior + revealed-pinned. Factored out of `forward` so the ITERATIVE refinement path
        (gen3_iterative_damage_v1) can recompute the belief from the MID-transformer opp tokens each round
        without re-running the soft-embed reinjection. When `prior_fusion`, `opp_species_ids` [B,6]
        (national-dex nums) and `opp_move_ids` [B,6,4] (revealed; id>0 ⇒ seen) turn the raw head output into
        the two-part POSTERIOR (prior⊕delta, revealed pinned certain)."""
        # gen3_belief_grad_mode_v1: in `detached` mode the head READS a stop-grad trunk (so neither the
        # supervised belief loss nor the op/policy gradient through this read can reshape the trunk); the
        # reinject WRITE below still rides the LIVE tokens. `getattr` default-False ⇒ unset == byte-identical.
        read = opp_tokens.detach() if getattr(self, "detach_read", False) else opp_tokens
        logits = self.move_head(read)                                            # [B, 6, M] (learned delta)
        if self.prior_fusion and opp_species_ids is not None:
            logits = logits + self.move_prior_logits[opp_species_ids]            # posterior = prior ⊕ delta
            if opp_move_ids is not None:
                # REVEALED moves are certain → pin to a high logit (sigmoid ≈ 1). id 0 = unknown sentinel.
                # BRANCHLESS + SYNC-FREE: the old form did `if bool(valid.any())` then `valid.nonzero()`
                # — TWO host syncs per call, and `move_logits` runs once per refine round + once in
                # `forward` (3×/forward in the production config), stalling the pipeline on the critical
                # path. `scatter_` writes the same mask with no host round-trip. Exactly equivalent: a
                # VALID slot scatters True at its id (always ≥1, since `id > 0` is the validity test); an
                # INVALID slot scatters False at the clamped index 0, which no valid slot can occupy — so
                # the two never collide and the resulting mask is identical, including the all-invalid
                # case (an all-False `revealed` makes the `where` a no-op, exactly like the old skip).
                valid = opp_move_ids > 0                                         # [B, 6, 4]
                ids = opp_move_ids.clamp(0, logits.shape[-1] - 1)                # defensive index clamp
                revealed = torch.zeros_like(logits, dtype=torch.bool)            # [B, 6, M]
                revealed.scatter_(-1, ids, valid)
                logits = torch.where(revealed, _REVEAL_LOGIT, logits)
        return logits

    def reinject_moves(self, opp_tokens: torch.Tensor, apply_mask: torch.Tensor,
                       move_embedding: torch.nn.Embedding,
                       move_logits: torch.Tensor) -> torch.Tensor:
        """Soft-embed a move posterior and residually enrich the opp tokens → [B,6,D].

        Split out of `forward` (gen3_typed_hp_belief_v1) so the caller can interpose the typed-HP
        composition between the head read and the reinjection. That ordering matters: the soft-embed is
        `Σ_m P(m)·move_emb[m]`, so feeding it the RAW posterior would inject `P(HP)·move_emb[237]` — the
        typeless HP row, which carries no type and whose `MOVE_ATTR`/latent row is deliberately all-zero.
        Fed the COMPOSED posterior it instead injects `Σ_t P(HP_t)·move_emb[355+t]`, i.e. the believed
        Hidden Power as a mixture over real typed moves. The enrichment is gated to the selected slots,
        so unselected slots pass through unchanged."""
        soft_emb = torch.sigmoid(move_logits) @ move_embedding.weight             # [B, 6, move_emb]
        enriched = opp_tokens + apply_mask.unsqueeze(-1) * self.reinject(soft_emb)
        return self.norm(enriched)

    def forward(self, opp_tokens: torch.Tensor, apply_mask: torch.Tensor,
                move_embedding: torch.nn.Embedding,
                opp_species_ids: Optional[torch.Tensor] = None,
                opp_move_ids: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """opp_tokens [B,6,D], apply_mask [B,6] bool (which slots get the belief), move_embedding table
        → (enriched_tokens [B,6,D], move_logits [B,6,M]). Convenience wrapper over
        `move_logits` + `reinject_moves` with NO typed-HP composition — the extractor drives the two
        steps itself so it can compose in between. Kept for direct/unit use."""
        move_logits = self.move_logits(opp_tokens, opp_species_ids, opp_move_ids)  # [B, 6, M] posterior
        return self.reinject_moves(opp_tokens, apply_mask, move_embedding, move_logits), move_logits


# gen3_nature_ev_belief_v1: the EV-delta head output is scaled by this before adding to the EV prior, so a
# unit logit moves the believed EV by ~_EV_DELTA_SCALE points (the head then learns within the clamped [0,252]).
_EV_DELTA_SCALE = 64.0


class SpreadBelief(torch.nn.Module):
    """Predicts the opponent's hidden SPREAD — the 5 battle-relevant DERIVED stats {atk,def,spa,spd,spe} at
    L100 — per opp slot, and REINJECTS it into the slot token. The THIRD belief leg (moves ✓, species ✓,
    STATS) — `gen3_unified_spread_belief_v1`, mirroring MoveBelief 1:1.

    Two parameterisations (the `nature` flag):
      - ADDITIVE (default): a usage-weighted PRIOR (mean, std per stat, Smogon spreads) ⊕ a learned head DELTA
        in std units → `believed = mean + delta·std`. Simple, but the DERIVED stat is a point estimate that
        sits BETWEEN the nature ×1.1/×0.9 modes → the "over-estimates the largest EV" order-statistic bias.
      - NATURE/EV GENERATIVE (`gen3_nature_ev_belief_v1`, `--spread-belief-nature`): predict a NATURE
        categorical ⊕ its Smogon log-prior + a per-stat EV ⊕ its Smogon prior (the prior-fusion pattern of the
        move/HP-type beliefs), assume IV 31, and COMPUTE `believed = (2·base + 31 + E[EV]/4 + 5)·E[nature_mult]`.
        The nature coupling (exactly one stat ×1.1, one ×0.9 — shared probability mass) + the EV budget are now
        STRUCTURAL, so the head can't inflate every stat → the order-statistic bias is fixed at the source. The
        nature distribution + EV are stashed so the supervised loss (nature CE + EV regression) trains the
        decomposition AND the op (`--spread-belief-nature-marginalize`) can marginalise P(KO) over the natures.

    Either way the output is the same `believed [B,6,5]` DERIVED stat the `DamageOperator` consumes at the opp
    ACTIVE slot (replacing its hand-coded constants), reinjected as a small residual so both heads see it. HP is
    skipped (the op uses the obs HP fraction × a neutral maxhp). Zero-init heads → cold-start == the prior; the
    prior buffers are non-persistent (data-derived) → OFF builds no module (reproduces nothing)."""

    def __init__(self, n_species: int, nature: bool = False):
        super().__init__()
        from agents.model.damage_tables import build_opp_spread_prior, N_SPREAD_STATS
        self.n_stats = N_SPREAD_STATS
        self.nature = nature
        self.reinject = torch.nn.Linear(N_SPREAD_STATS, D_MODEL)
        torch.nn.init.normal_(self.reinject.weight, std=0.02)       # start the enrichment ≈0
        torch.nn.init.zeros_(self.reinject.bias)
        self.norm = torch.nn.LayerNorm(D_MODEL)
        # [n_species, 5, 2] (mean, std) usage prior — non-persistent (recomputable from data/).
        self.register_buffer("spread_prior", build_opp_spread_prior(n_species), persistent=False)
        if not nature:
            self.stat_head = torch.nn.Linear(D_MODEL, N_SPREAD_STATS)   # learned delta in std-units
            torch.nn.init.zeros_(self.stat_head.weight)                 # cold-start delta == 0 → believed == prior
            torch.nn.init.zeros_(self.stat_head.bias)
        else:
            from agents.model.damage_tables import (build_species_nature_prior, build_species_ev_prior,
                                                    build_nature_mult, build_species_base_stats, N_NATURES)
            self.n_natures = N_NATURES
            self.nature_head = torch.nn.Linear(D_MODEL, N_NATURES)      # logit DELTA on the nature log-prior
            torch.nn.init.zeros_(self.nature_head.weight)               # cold-start delta 0 → posterior == prior
            torch.nn.init.zeros_(self.nature_head.bias)
            self.ev_head = torch.nn.Linear(D_MODEL, N_SPREAD_STATS)     # EV DELTA on the EV prior (×_EV_DELTA_SCALE)
            torch.nn.init.zeros_(self.ev_head.weight)
            torch.nn.init.zeros_(self.ev_head.bias)
            self.register_buffer("nature_logprior", build_species_nature_prior(n_species), persistent=False)  # [n,25]
            self.register_buffer("ev_prior", build_species_ev_prior(n_species), persistent=False)             # [n,5]
            self.register_buffer("nature_mult", build_nature_mult(), persistent=False)                        # [25,5]
            self.register_buffer("base_nonhp", build_species_base_stats(n_species), persistent=False)         # [n,5]

    def forward(self, opp_tokens: torch.Tensor, apply_mask: torch.Tensor, opp_species_ids: torch.Tensor):
        """opp_tokens [B,6,D], apply_mask [B,6] bool (which slots get the belief), opp_species_ids [B,6] (nums)
        → (enriched_tokens [B,6,D], believed_stats [B,6,5], nature_logits [B,6,25]|None, ev [B,6,5]|None). The
        residual carries the believed-vs-prior delta into the (selected) token; unselected slots pass through.
        Cold-start (zero deltas) ⇒ believed == the usage prior in BOTH parameterisations."""
        prior = self.spread_prior[opp_species_ids]                  # [B,6,5,2]
        mean, std = prior[..., 0], prior[..., 1]                    # [B,6,5]
        # gen3_belief_grad_mode_v1: `detached` READS a stop-grad trunk for the head(s); reinject below keeps
        # the LIVE `opp_tokens` identity term (so normal policy training still shapes the trunk).
        read = opp_tokens.detach() if getattr(self, "detach_read", False) else opp_tokens
        if not self.nature:
            delta = self.stat_head(read)                            # [B,6,5] (std units; cold == 0)
            believed = (mean + delta * std).clamp(min=1.0)          # [B,6,5] the stat VALUE the op consumes
            enriched = opp_tokens + apply_mask.unsqueeze(-1) * self.reinject(delta)
            return self.norm(enriched), believed, None, None
        # NATURE/EV generative path (gen3_nature_ev_belief_v1): posterior nature ⊕ EV → COMPUTE the derived stat.
        nat_logits = self.nature_logprior[opp_species_ids] + self.nature_head(read)         # [B,6,25] prior⊕delta
        e_mult = torch.softmax(nat_logits, dim=-1) @ self.nature_mult                       # [B,6,5] E[mult]
        ev = (self.ev_prior[opp_species_ids]
              + self.ev_head(read) * _EV_DELTA_SCALE).clamp(0.0, 252.0)                     # [B,6,5]
        base = self.base_nonhp[opp_species_ids]                                             # [B,6,5]
        believed = ((2.0 * base + 31.0 + ev / 4.0 + 5.0) * e_mult).clamp(min=1.0)           # [B,6,5] DERIVED stat
        delta = (believed - mean) / std.clamp(min=1.0)                                      # std-unit residual
        enriched = opp_tokens + apply_mask.unsqueeze(-1) * self.reinject(delta)
        return self.norm(enriched), believed, nat_logits, ev


class HPTypeBelief(torch.nn.Module):
    """Predicts the opponent's hidden HIDDEN-POWER TYPE — per opp slot, a 16-way distribution over
    HIDDEN_POWER_TYPE_ORDER (== the op's HP_TYPE_IDX axis) — FUSED with the Smogon HP-type prior in
    log-odds (mirroring MoveBelief's prior fusion). The `DamageOperator` reads this posterior as its
    typed-HP candidate weights, so an opponent's still-unrevealed Hidden Power is priced as a real ~70-BP
    threat of its most-likely type(s) instead of the all-zero obs `hp_probs` (empty until the opp FIRES HP
    — the "opp HP reads immune" GIGO) or a flat prior. gen3_opp_hp_type_belief_v1, the "force the model to
    guess which Hidden Power it is" head.

    The posterior is stashed at `features_extractor.last_hp_type_logits` and (a) fed to the op as the
    per-slot typed-HP candidate weights — whose damage gradient sharpens this head — and (b) supervised by
    the aux CE loss against the privileged true HP type (from agent2's team). ZERO-INIT so the cold-start
    posterior == the Smogon prior (the clean A/B baseline). Leak-safe: the posterior is a model output; the
    HP-type LABEL rides a training-only obs key read ONLY by the loss, never in pi/vf.

    **gen3_opp_hp_type_belief_v2 (robust):** the belief is ALSO reinjected into the opp token (`reinject`,
    a small-init residual like MoveBelief/SpreadBelief) as the PRESENCE-GATED expected typed-HP embedding —
    so the believed HP type flows into the CLS pools + BOTH heads (attention reasons over "this mon is
    probably an ice/grass HP user"), not only the op's damage block. The presence gate (the move belief's
    P(HP present), ≈1 once `hiddenpower` is revealed — the "presence bit") zeroes the signal when HP is
    unlikely. The posterior stays a full 16-way distribution (it does NOT argmax-collapse), so multiple
    un-ruled-out types remain live candidates the op's top-K simulates distinctly + weights by confidence."""

    def __init__(self, n_species: int, type_emb_dim: int, n_hp: int = 16):
        super().__init__()
        self.norm = torch.nn.LayerNorm(D_MODEL)
        self.type_head = torch.nn.Linear(D_MODEL, n_hp)
        # gen3_typed_hp_belief_v1: the 16 typed-HP dex nums (355-370) in HP_TYPE_ORDER order — the
        # destination of the presence×type composition. Non-persistent (data-derived, recomputable).
        from agents.model.damage_tables import _hp_typed_nums
        self.register_buffer("HP_TYPED_NUMS",
                             torch.tensor(list(_hp_typed_nums()), dtype=torch.long), persistent=False)
        torch.nn.init.zeros_(self.type_head.weight)        # cold-start delta = 0 → posterior == prior
        torch.nn.init.zeros_(self.type_head.bias)
        # gen3_opp_hp_type_belief_v2: the token reinjection of the presence-gated expected typed-HP embedding
        # (the expected type emb = embeddings.hp_soft_type(posterior)). Small-init so the enrichment starts
        # ≈0 (no harm before the belief sharpens), mirroring MoveBelief.reinject / SpreadBelief.reinject.
        self.reinject_proj = torch.nn.Linear(type_emb_dim, D_MODEL)
        torch.nn.init.normal_(self.reinject_proj.weight, std=0.02)
        torch.nn.init.zeros_(self.reinject_proj.bias)
        self.reinject_norm = torch.nn.LayerNorm(D_MODEL)
        from agents.model.damage_tables import build_hp_type_prior
        # [n_species, 16] prob prior (sums to 1) — the log-odds fusion base; non-persistent (data-derived).
        self.register_buffer("hp_prior", build_hp_type_prior(n_species), persistent=False)

    def forward(self, opp_tokens: torch.Tensor,
                opp_species_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """opp_tokens [B,6,D], opp_species_ids [B,6] (national-dex nums) → (logits [B,6,16], posterior
        [B,6,16]). logits = head_delta + log(prior[species]); posterior = softmax. Cold-start (delta 0) ⇒
        posterior == the Smogon prior. The logits feed the CE loss; the posterior feeds the op + reinject."""
        read = opp_tokens.detach() if getattr(self, "detach_read", False) else opp_tokens
        delta = self.type_head(self.norm(read))                             # [B,6,16] (cold == 0)
        prior = self.hp_prior[opp_species_ids].clamp_min(1e-6)              # [B,6,16]
        logits = delta + torch.log(prior)                                  # posterior logit = prior ⊕ delta
        return logits, torch.softmax(logits, dim=-1)

    def compose_typed_hp(self, move_logits: torch.Tensor, posterior: torch.Tensor,
                         obs_hp_probs: torch.Tensor,
                         opp_move_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """**gen3_typed_hp_belief_v1 — the one place a typeless Hidden Power becomes typed.**

        Rewrites the raw move-belief posterior [B,6,M] so that Hidden Power exists ONLY as the 16 real
        typed moves, and returns `(typed_logits, presence)`:

          * ``logits[..., 237]`` → `_HP_PRESENCE_OFF_LOGIT` (sigmoid ≈ 1e-13). The typeless num is a
            belief bookkeeping channel, not a move; nothing downstream — the damage op, the top-K, the
            BCE, the latent grading, the soft-embed reinjection, the prober — may see it as a candidate.
          * ``logits[..., 355+t]`` → ``logit(presence · P(type = t))`` for each of the 16 types.

        **The `Σ_t P(HP_t) = presence` identity is what makes the "a revealed HP must exist somewhere"
        constraint STRUCTURAL rather than a penalty term.** `presence` is read from the raw 237 channel,
        which `MoveBelief.move_logits` pins to `_REVEAL_LOGIT` the moment the opponent is seen using
        `hiddenpower`; so on reveal the 16 typed weights sum to ≈1 and the belief cannot conclude "no
        Hidden Power" no matter what the type head does. It can only be UNSURE ACROSS TYPES — which is
        the honest state, since Gen 3 never discloses the type.

        Two eliminations run first, both of them certain facts rather than learned preferences — the
        "discard the ones that don't make sense" half:

        1. **Rule-out by moveset exhaustion.** If all four of the opponent's moves are revealed and none
           is Hidden Power, it has none: presence → 0. Derived from `opp_move_ids` alone, so it needs no
           extra plumbing and agrees by construction with `HiddenPowerTracker.mark_no_hp`.
        2. **Narrowing by observed effectiveness.** Once the opponent has FIRED Hidden Power, the
           tracker's `obs_hp_probs` has hard-zeroed every type inconsistent with the observed
           effectiveness bucket. Those zeros are CERTAIN physics (not a prior), so the believed type
           distribution is restricted to the survivors and renormalised — the model is forced to spend
           its probability mass only on types that could actually have produced the damage it saw. If
           the belief puts ~no mass on the survivors (an off-meta HP), we fall back to uniform over the
           survivors rather than a ~0 vector, which would silently re-immune the move.

        Shapes: `move_logits` [B,6,M]; `posterior` [B,6,16]; `obs_hp_probs` [B,6,16] (OPP slots);
        `opp_move_ids` [B,6,4]. Differentiable in both `presence` and `posterior`, so the damage
        gradient and the move-belief BCE both sharpen the type head through the typed channels."""
        presence = torch.sigmoid(move_logits[..., HIDDEN_POWER_MOVE_NUM])                   # [B,6]
        # (1) moveset exhaustion — 4 revealed, none of them HP ⇒ certainly no Hidden Power.
        n_revealed = (opp_move_ids > 0).sum(dim=-1)                                     # [B,6]
        hp_seen = (opp_move_ids == HIDDEN_POWER_MOVE_NUM).any(dim=-1)                       # [B,6]
        presence = presence * (~((n_revealed >= opp_move_ids.shape[-1]) & ~hp_seen)).float()
        # (2) effectiveness narrowing — the tracker's zeros are certain, so restrict + renormalise.
        has_obs = obs_hp_probs.sum(-1, keepdim=True) > 0                                # HP has fired
        surv = (obs_hp_probs > 0).float()                                               # certain survivors
        narrowed = posterior * surv
        narrowed = torch.where(narrowed.sum(-1, keepdim=True) > 1e-6, narrowed, surv)   # off-meta fallback
        narrowed = narrowed / narrowed.sum(-1, keepdim=True).clamp_min(1e-6)
        hp_type = torch.where(has_obs, narrowed, posterior)                             # [B,6,16]

        typed_p = (presence.unsqueeze(-1) * hp_type).clamp(1e-9, 1.0 - 1e-6)            # [B,6,16]
        out = move_logits.clone()
        out.index_copy_(-1, self.HP_TYPED_NUMS, torch.logit(typed_p))
        out[..., HIDDEN_POWER_MOVE_NUM] = _HP_PRESENCE_OFF_LOGIT
        return out, presence

    def reinject(self, opp_tokens: torch.Tensor, posterior: torch.Tensor, presence: torch.Tensor,
                 apply_mask: torch.Tensor, embeddings: 'Embeddings') -> torch.Tensor:
        """gen3_opp_hp_type_belief_v2: enrich the opp tokens [B,6,D] with the PRESENCE-GATED expected
        typed-HP embedding, so the believed HP type flows into the CLS pools + both heads (not only the op's
        damage block). `posterior` [B,6,16] (the type belief), `presence` [B,6] (P(HP present) — ≈1 on
        reveal; gates the signal to ≈0 when HP is unlikely), `apply_mask` [B,6] float (which slots to
        enrich — revealed). `embeddings.hp_soft_type(posterior)` = Σ_t P(t)·type_emb[t], the expected type
        embedding. Small-init residual → starts ≈0; LayerNorm'd. Returns the enriched [B,6,D]."""
        soft = embeddings.hp_soft_type(posterior)                          # [B,6,type_emb] expected type emb
        gated = presence.unsqueeze(-1) * soft                              # ≈0 when HP unlikely (no spurious)
        enriched = opp_tokens + apply_mask.unsqueeze(-1) * self.reinject_proj(gated)
        return self.reinject_norm(enriched)


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
# gen3_damage_op_split_v1: the DamageOperator (1,689 lines) + its constants + the decode mirror
# now live in `damage_op.py` — 39% of this file for one concern. Re-exported UNCHANGED so every
# historical import path still resolves (the prober, model_version, snapshot and the tests all do
# `from agents.model.features_extractor import DamageOperator / decode_damage_block / _DMG_*`).
from agents.model.damage_op import (  # noqa: F401,E501  (re-export)
    DamageOperator,
    _COND_BRN_IDX,
    _COND_PAR_IDX,
    _COND_SLP_IDX,
    _DMG_CB,
    _DMG_CB_PER_MON,
    _DMG_CHANNEL_FEATS,
    _DMG_CHIP_CAP,
    _DMG_CRIT_CAP,
    _DMG_CRIT_P,
    _DMG_EFFECT,
    _DMG_EFFECT_COLS,
    _DMG_IDX_OUTSPEED,
    _DMG_IDX_PHYS_ACC,
    _DMG_IDX_PHYS_CRIT,
    _DMG_IDX_PHYS_HIGH,
    _DMG_IDX_PHYS_LOW,
    _DMG_IDX_PHYS_PKO,
    _DMG_IDX_PROVENANCE,
    _DMG_IDX_SPEC_ACC,
    _DMG_IDX_SPEC_CRIT,
    _DMG_IDX_SPEC_HIGH,
    _DMG_IDX_SPEC_LOW,
    _DMG_IDX_SPEC_PKO,
    _DMG_IMX_CELL,
    _DMG_IMX_HDR_ACC,
    _DMG_IMX_HDR_EFFECT,
    _DMG_IMX_HDR_PHYS,
    _DMG_IMX_HDR_SEC,
    _DMG_IMX_HDR_W,
    _DMG_IMX_HEADER,
    _DMG_INCOMING_SEC,
    _DMG_N_CHANNELS,
    _DMG_OAX,
    _DMG_OAX_IDX_CRIT,
    _DMG_OAX_IDX_HIGH,
    _DMG_OAX_IDX_LOW,
    _DMG_OAX_IDX_PKO,
    _DMG_OAX_N_MOVES,
    _DMG_OAX_PER_MON,
    _DMG_OAX_PER_MOVE,
    _DMG_OMX,
    _DMG_OMX_CELL,
    _DMG_OMX_IDX_CRIT,
    _DMG_OMX_IDX_HIGH,
    _DMG_OMX_IDX_LOW,
    _DMG_OMX_IDX_MULT,
    _DMG_OMX_IDX_PKO,
    _DMG_OUTGOING,
    _DMG_OUT_N_MOVES,
    _DMG_OUT_PER_MOVE,
    _DMG_OUT_REFINE,
    _DMG_OUT_SEC,
    _DMG_PARA_SPEED,
    _DMG_PER_MON,
    _DMG_REFINE_FEATS,
    _DMG_REFINE_K,
    _DMG_ROLL_MIN,
    _DMG_SPEED_SCALE,
    _DMG_SPEED_STD_K,
    _DMG_STATUS,
    _DMG_STATUS_N_MOVES,
    _DMG_STATUS_REFINE,
    _DMG_TOPK_DEFAULT_K,
    _DMG_TOPK_DMG_PER,
    _DMG_TOPK_IDX_ACC,
    _DMG_TOPK_IDX_HIGH,
    _DMG_TOPK_IDX_LATENT,
    _DMG_TOPK_IDX_PHYS,
    _DMG_TOPK_IDX_PKO,
    _DMG_TOPK_IDX_STATUS,
    _DMG_TOPK_IDX_W,
    _DMG_TOPK_ID_DIM,
    _DMG_TOPK_META,
    _DMG_TOPK_MOVE,
    _FIRE_TIDX,
    _IMMOBILIZE_STATUS_CATS,
    _SB_ATK,
    _SB_DEF,
    _SB_SPA,
    _SB_SPD,
    _SB_SPE,
    _SECONDARY_MAJOR_N,
    _SECONDARY_TO_STATUS_CAT,
    _SUBSTITUTE_CTX_IDX,
    _WATER_TIDX,
    _dmg_imx_dim,
    _dmg_topk_dim,
    decode_damage_block,
)
def _request_order_move_tokens(move_tokens_all, ctx):
    """gen3_pointer_native_v1: permute OUR ACTIVE mon's per-move tokens from the extractor's
    SORTED-BY-ID slot order into ACTION/REQUEST order, by MOVE-NUM IDENTITY (never by position).

    This is the single place the `ordering_integrity.py` bug class is dissolved: the extractor reads
    moves via `MovesEncoder.get_sorted_moves` (sorted by dex num) while action logit `6+k` refers to
    `legal.move_slots[k]` (request order). Both id sources are dex NUMs — `all_move_ids` indexes
    MOVE_BP, and `our_active_req_move_ids` is written as `float(md.num)` in reactive.py — so they are
    directly comparable.

    Returns `(tokens_req [B,4,D], valid [B,4])`. A request slot that matches no sorted slot (an empty
    slot, forced Struggle, or a mon with <4 moves) gets a ZEROED token and `valid=0`, so the head
    contributes exactly 0 there rather than scoring a garbage vector.
    """
    ar = torch.arange(ctx.batch_size, device=ctx.device)
    sorted_ids = ctx.all_move_ids[ar, ctx.our_active_idx]                 # [B,4] dex nums, SORTED order
    req_ids = ctx.our_active_req_move_ids.long()                          # [B,4] dex nums, REQUEST order
    match = (sorted_ids[:, None, :] == req_ids[:, :, None]) & (req_ids[:, :, None] > 0)  # [B,4req,4sorted]
    valid = match.any(-1)                                                 # [B,4] did this slot resolve?
    perm = match.float().argmax(-1)                                       # [B,4] request slot -> sorted slot
    active_tokens = move_tokens_all[ar, ctx.our_active_idx]               # [B,4,D] sorted order
    tokens_req = active_tokens.gather(1, perm[:, :, None].expand(-1, -1, active_tokens.shape[-1]))
    return tokens_req * valid[:, :, None].float(), valid.float()


class EntityMoveSeats(torch.nn.Module):
    """gen3_entity_move_seats_v1 (v54) — Stage 1 of the entity generation: MOVE tokens become
    first-class attention SEATS in the unified trunk (`design_generation_roadmap.md` §3 Stage 1).

    Until now moves existed only INSIDE `PokemonEncoder` (flattened into the mon vector; v51 rescued
    the active's 4 for the pointer head) — attention could never do "Rock Slide THE TOKEN threatens
    Zapdos THE TOKEN". This module builds the two new seat families, appended AFTER the global token
    so every existing absolute slice (team tokens, history, the refine callback's tail-preserving
    cat) is untouched:

      * **E3 — our active's 4 move seats** (unconditional): the SAME request-ordered, identity-
        permuted tokens the pointer head reads (`_request_order_move_tokens` — one permutation,
        shared), projected 32 → d_model. Seat k == action logit 6+k by construction. An unresolved
        request slot (forced Struggle / <4 moves) is a zero token AND key-masked.
      * **E4 — the opp active's top-K believed threat-move seats** (`topk_seats` > 0): candidates
        from `DamageOperator.refine_candidates(k=K)` — the SAME belief-weighted, typed-HP-scattered,
        learnset-gated candidate definition the refine/top-K kernels use (one source, no drift) —
        each seat = `[move latent(32) ⊕ belief w ⊕ accuracy ⊕ is_phys]` projected to d_model.
        Index selection detached, `w` differentiable → the belief gradient rides the seats. All K
        seats key-masked when there is no opponent active.

    The feasibility spike (`entity_spike_benchmark.py`) priced the seat growth at ~+0.19 ms B=1 for
    the full ~50-seat layout — dispatch-bound, not FLOP-bound. NO edges yet (Stage 2); the seats
    enter attention purely as content. Input projections are ordinary trainable Linears (NOT
    zero-init — new-information inputs, the `history_proj`/`global_proj` convention, not the
    residual-injection one)."""

    def __init__(self, topk_seats: int = 0):
        super().__init__()
        self.topk_seats = int(topk_seats)
        self.move_seat_proj = torch.nn.Linear(MOVE_NET_HIDDEN[1], D_MODEL)
        self.threat_seat_proj = (torch.nn.Linear(MOVE_LATENT_DIM + 3, D_MODEL)
                                 if self.topk_seats > 0 else None)

    @property
    def n_seats(self) -> int:
        return 4 + self.topk_seats

    def seat_types(self, device) -> torch.Tensor:
        """Per-seat token-type ids [n_seats] for the transformer's type embedding."""
        return torch.cat([
            torch.full((4,), TOKEN_TYPE_OUR_MOVE, dtype=torch.long, device=device),
            torch.full((self.topk_seats,), TOKEN_TYPE_THEIR_THREAT, dtype=torch.long, device=device),
        ])

    def forward(self, tok_req: torch.Tensor, move_valid: torch.Tensor, ctx: 'ExtractorContext',
                damage_op, move_belief_logits: Optional[torch.Tensor],
                latent_table: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """→ `(seats [B, 4+K, d_model], pad [B, 4+K] bool)` (pad True = masked, the key-mask sense)."""
        seats = [self.move_seat_proj(tok_req)]                                # [B,4,D] (invalid = zeros)
        pads = [move_valid < 0.5]                                             # [B,4]
        if self.topk_seats > 0:
            assert move_belief_logits is not None and latent_table is not None, (
                "E4 threat seats need the pre-transformer move-belief logits + the move latent table "
                "(guaranteed by the __init__ gate: damage_op_prefuse + move_latent)"
            )
            idx, w = damage_op.refine_candidates(ctx, move_belief_logits, k=self.topk_seats)  # [B,K]
            has_opp = ctx.hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, -1].any(dim=1)            # [B] bool
            hdr = torch.cat([
                latent_table[idx],                                            # [B,K,32] typed-HP-aware identity
                w[:, :, None],                                                # belief weight (differentiable)
                damage_op.MOVE_ACCURACY[idx][:, :, None],
                damage_op.MOVE_PHYS[idx][:, :, None],
            ], dim=2)
            e4 = self.threat_seat_proj(hdr) * has_opp[:, None, None].float()  # zeroed when no opp active
            seats.append(e4)
            pads.append(~has_opp[:, None].expand(-1, self.topk_seats))
        return torch.cat(seats, dim=1), torch.cat(pads, dim=1)


class PointerNativeActionHead(torch.nn.Module):
    """gen3_pointer_native_v1: THE action head — score each action FROM THE TOKEN OF THE ENTITY IT
    SELECTS. There is no flat positional head in this generation; these ARE the logits.

    WHY (the fresh-generation commitment — designs/ai_v9/design_pointer_action_head.md §0). A flat
    `Linear(latent, 11)` is position-SENSITIVE: logit row j must independently rediscover what "team
    slot j" means, and per-move physics reaches logit 6+k only by the projection LEARNING the
    alignment. The pointer head is position-EQUIVARIANT: one shared scoring function per entity token
    (permute the team ⇒ the logits permute with it), which dissolves two defect classes structurally:
      * **F2** — switch logits read from a permutation-INVARIANT CLS pool, so a bench mon's own token
        could never reach its own switch logit.
      * **The ordering bug class** — the extractor reads moves SORTED-BY-ID while the action space
        uses REQUEST order; the permutation now happens once, by move-num identity
        (`_request_order_move_tokens`), so a misaligned logit is unrepresentable.

    INPUTS (the information contract — each closes a measured deficit of the v49 delta form):
      * `ctx_vec` = **latent_pi** (the policy tower's output) — the decision context. This is the
        SAME vector the deleted flat head consumed, so everything it saw (the op block, the beliefs,
        FiLM/z_arch modulation) conditions every pointer score (closes G4/G5).
      * move k: its REQUEST-slot token ⊕ its own op cells `[low,high,crit,pko,p_land,known,sec×10]`
        — the lossless per-action physics route (closes G2).
      * switch j: our-team token j (post-transformer — board-aware) ⊕ its incoming row + CB tail
        ⊕ its OAX attacker row — per-candidate defense AND offense (closes G3).
    Cell widths are fixed by the build-time toggle set (`DamageOperator.pointer_*_cell_dim`; 0 when
    the source block is off — a missing block narrows the Linear, never silently zero-pads).

    OWNERSHIP. The module lives on the POLICY as `pointer_head` (`action_net` is a raising stub; built in
    `Gen3DualHeadMaskablePolicy._build`, AFTER SB3's ortho-init apply — so the zero-init survives
    without the M1 guard). The three scorers are zero-init ⇒ every logit is exactly 0 at step 0 ⇒
    the cold-start policy is uniform-over-legal, the correct fresh-run init.

    Output layout is the action space (`agents/action/constants.py`): `[switch x6, move x4, struggle]`
    = SWITCH_START..SWITCH_END, MOVE_START..MOVE_END, STRUGGLE.
    """

    def __init__(self, move_token_dim: int, d_model: int, ctx_dim: int,
                 move_cell_dim: int = 0, switch_cell_dim: int = 0, hidden: int = POINTER_HIDDEN):
        super().__init__()
        self.move_cell_dim = int(move_cell_dim)
        self.switch_cell_dim = int(switch_cell_dim)
        self.ctx_proj = torch.nn.Linear(ctx_dim, hidden)
        self.move_proj = torch.nn.Linear(move_token_dim + self.move_cell_dim, hidden)
        self.switch_proj = torch.nn.Linear(d_model + self.switch_cell_dim, hidden)
        # The three SCORERS are zero-init (weight AND bias) => all logits exactly 0 at init =>
        # uniform-over-legal cold start.
        self.move_score = torch.nn.Linear(hidden, 1)
        self.switch_score = torch.nn.Linear(hidden, 1)
        self.struggle_score = torch.nn.Linear(hidden, 1)
        for lin in (self.move_score, self.switch_score, self.struggle_score):
            torch.nn.init.zeros_(lin.weight)
            torch.nn.init.zeros_(lin.bias)

    def forward(self, ctx_vec: torch.Tensor, move_tokens_req: torch.Tensor,
                move_valid: torch.Tensor, team_tokens: torch.Tensor,
                move_cells: torch.Tensor, switch_cells: torch.Tensor) -> torch.Tensor:
        """`ctx_vec` [B,ctx_dim] = latent_pi; `move_tokens_req` [B,4,move_token_dim] in REQUEST
        order; `move_valid` [B,4] (1.0 where the request slot resolved to a real move);
        `team_tokens` [B,6,d_model]; `move_cells` [B,4,move_cell_dim] / `switch_cells`
        [B,6,switch_cell_dim] (the op's per-action physics; width-0 when the op is off).
        Returns the [B,11] action logits."""
        c = self.ctx_proj(ctx_vec)                                            # [B,H]
        m_in = torch.cat([move_tokens_req, move_cells], dim=-1)               # [B,4,tok+cell]
        s_in = torch.cat([team_tokens, switch_cells], dim=-1)                 # [B,6,d_model+cell]
        m = torch.tanh(self.move_proj(m_in) + c[:, None, :])                  # [B,4,H]
        s = torch.tanh(self.switch_proj(s_in) + c[:, None, :])                # [B,6,H]
        # An unresolved request slot (forced Struggle / <4 moves) contributes exactly 0, never a
        # score computed from a zero token — the action mask already forbids it, but a nonzero logit
        # there would still perturb the softmax normaliser over the LEGAL actions.
        move_logits = self.move_score(m).squeeze(-1) * move_valid             # [B,4]
        switch_logits = self.switch_score(s).squeeze(-1)                      # [B,6]
        struggle_logit = self.struggle_score(torch.tanh(c))                   # [B,1]
        return torch.cat([switch_logits, move_logits, struggle_logit], dim=-1)  # [B,11]


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


class ZArchEncoder(torch.nn.Module):
    """Team-archetype latent z_arch from the INVARIANT parts of OUR team (gen3_zarch_film_v1, v44).

    The amortization-gap storage fix's conditioning signal (designs/learning/
    amortization_gap_and_conditioning.md): a TEAM-STATIC, permutation-invariant DeepSets code over
    OUR 6 mons' invariant facts — species ⊕ item ⊕ ability ⊕ moves (mean of the 4 move embeddings)
    ⊕ the 18-dim spread block (IVs/EVs/nature — our own side, known + invariant all battle). Each
    mon → a shared atom MLP → MEAN over the 6 → LayerNorm → z [B, dim].

    Design properties (each aimed at a measured failure mode):
      - TEAM-STATIC by construction — every input is invariant within a battle, so z cannot
        per-decision "flip" archetype (the composition_robustness_probe's 0.092 per-decision vs
        0.030 static flip rate). Deterministic (no VIB sampling — v1 is the LUT-first operating
        point; per-forward sampling would also break PPO's ratio recompute + eval determinism).
      - DeepSets mean — order-free (a team is a SET) and a one-mon swap moves 1/6 of the sum
        (a twist, not a flip; no single salient mon can dominate the code).
      - DETACHED embedding reads — the shared species/move/item/ability tables are consumed
        value-only, so the z-aux (recon/VICReg) and FiLM gradients cannot reshape the trunk's
        embeddings (the belief_grad_mode='detached' philosophy: fully decoupled from the trunk,
        zero gradient interference).
      - recon_head — species multi-hot reconstruction logits (the day-0 anti-collapse anchor:
        a constant z cannot reconstruct different teams; Species Clause ⇒ multi-hot is lossless).
        A side readout for the aux loss only — never fed forward.

    Leak-trivial: reads OUR OWN team only (fully public to us)."""

    def __init__(self, layout: Dict[str, Any], dim: int):
        super().__init__()
        atom_in = (layout['species_embedding_dim'] + layout['item_embedding_dim']
                   + layout['ability_embedding_dim'] + layout['move_embedding_dim']
                   + POKEMON_SPREAD_DIM)
        self.atom_mlp = torch.nn.Sequential(
            torch.nn.Linear(atom_in, ZARCH_ATOM_HIDDEN),
            torch.nn.ReLU(),
            torch.nn.Linear(ZARCH_ATOM_HIDDEN, dim),
        )
        self.norm = torch.nn.LayerNorm(dim)
        self.recon_head = torch.nn.Linear(dim, layout['max_species'])

    def forward(self, ctx: ExtractorContext, embeddings: Embeddings) -> torch.Tensor:
        # Invariant inputs only (our side = slots 0..5): no HP / status / boosts / PP.
        # .detach() on every table read — see the class docstring (zero trunk interference).
        sp = embeddings.species_embedding(ctx.species_ids[:, :TEAM_SIZE]).detach()      # [B,6,S]
        it = embeddings.item_embedding(ctx.item_ids[:, :TEAM_SIZE]).detach()            # [B,6,I]
        ab = embeddings.ability_embedding(ctx.ability1_ids[:, :TEAM_SIZE]).detach()     # [B,6,A]
        mv = embeddings.move_embedding(ctx.all_move_ids[:, :TEAM_SIZE, :]).detach().mean(dim=2)  # [B,6,M]
        spread = ctx.pokemon_part[:, :TEAM_SIZE,
                                  POKEMON_SPREAD_OFFSET:POKEMON_SPREAD_OFFSET + POKEMON_SPREAD_DIM]
        atoms = self.atom_mlp(torch.cat([sp, it, ab, mv, spread], dim=-1))              # [B,6,dim]
        return self.norm(atoms.mean(dim=1))                                             # [B,dim]

    def recon_logits(self, z: torch.Tensor) -> torch.Tensor:
        """Species multi-hot reconstruction logits [B, max_species] — aux-loss target only."""
        return self.recon_head(z)


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
                 win_prob_mode: str = "none",
                 damage_outgoing: bool = False, move_candidate_floor: float = 0.0,
                 move_latent: bool = False, spread_belief: bool = False, spread_belief_nature: bool = False,
                 spread_belief_nature_marginalize: bool = False,
                 value_dist_mode: str = "none", value_dist_bins: int = 0,
                 value_dist_vmin: float = 0.0, value_dist_vmax: float = 0.0,
                 damage_topk_k: int = 0, damage_reattend: bool = False,
                 move_belief_prefuse: bool = False, move_belief_single_compute: bool = False,
                 damage_candidate_k: int = 0, damage_refine_rounds: int = 0,
                 damage_op_prefuse: bool = False,
                 entity_topk_seats: int = 0,
                 damage_matrices_outgoing: bool = False, damage_matrices_incoming: bool = False,
                 damage_matrices_outgoing_all: bool = False,
                 threat_refine_outgoing: bool = False, threat_unrevealed_outgoing: bool = False,
                 threat_prob_outspeed: bool = False, threat_status_refine: bool = False,
                 hp_belief_mode: str = "composed", belief_grad_mode: str = "shaping",
                 pubval_mode: str = "none",
                 zarch_film: str = "off", zarch_dim: int = 0,
                 zarch_lut: str = "off", zarch_lut_rosters: Optional[Sequence[Sequence[int]]] = None,
                 zarch_lut_init_std: float = 1.0):
        super().__init__()
        self.layout = layout
        self.mappings = mappings
        self.log_level = log_level
        # Behavioral toggle (no weight-shape change): unmask the opponent's still-hidden
        # party so the transformer attends to it. Version-checked, not in ARCH_SIGNATURE.
        self.attend_unrevealed_opponents = attend_unrevealed_opponents
        # gen3_cpu_damage_deleted_v1: the three --unified-obs ablation masks are gone with their blocks.
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

        # gen3_belief_grad_mode_v1: 'detached' makes the STATE-prediction belief heads (move / spread /
        # hp-type / the species-moves-latent aux) READ a stop-grad trunk — so neither their supervised
        # loss nor the op/policy gradient through them can reshape the shared trunk. The belief stays
        # computed, reinjected into the forward, and consumed by the op (fully "in the system"); only the
        # trunk-shaping gradient is cut. The flag is applied per-head via `detach_read` (set just before
        # the dummy forward, once all heads exist). 'shaping' (default) = current behavior, byte-identical.
        if belief_grad_mode not in ("shaping", "detached"):
            raise ValueError(f"belief_grad_mode must be shaping|detached, got {belief_grad_mode!r}")
        self.belief_grad_mode = belief_grad_mode
        self._belief_detach = (belief_grad_mode == "detached")

        # Phase modules (constructed before the dummy forward below).
        self.embeddings = Embeddings(layout)
        self.unpack = ObsUnpack(layout, attend_unrevealed_opponents=attend_unrevealed_opponents)
        # gen3_pointer_native_v1: the pointer action head is THE action head (no flat action_net in this
        # generation), but the MODULE lives on the POLICY (Gen3DualHeadMaskablePolicy._build — its ctx is
        # latent_pi, which does not exist at extractor time). The extractor's side of the contract is the
        # per-forward stash `last_pointer_inputs` (request-ordered move tokens + valid mask + our team
        # tokens + the op's per-action cells), set unconditionally in forward_internal.
        self.last_pointer_inputs: Optional[Tuple[torch.Tensor, ...]] = None
        self.pokemon_encoder = PokemonEncoder(layout, move_latent=move_latent)
        # gen3_entity_move_seats_v1 (v54, Stage 1): move ENTITY seats in the trunk — E3 (our active's
        # 4 request-ordered move tokens, unconditional) + E4 (the opp active's top-`entity_topk_seats`
        # believed threat moves, opt-in). The pointer head then reads the REFINED E3 seats (post-
        # attention, d_model-wide) instead of the raw 32-dim PokemonEncoder tokens. E4's gates: the
        # candidate weights + latent table must exist PRE-transformer, which is exactly the prefuse
        # stack (validated below after those flags are set).
        self.entity_topk_seats = int(entity_topk_seats)
        self.entity_seats = EntityMoveSeats(self.entity_topk_seats)
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
        # move_belief_prefuse (gen3_move_prefuse_v1): reinject the move belief into the opp role tokens
        # BEFORE the TeamTransformer (vs the default POST-transformer reinject), so the believed moves
        # CO-REFINE with the species/team belief through the 2 attention layers (the move prediction for
        # one mon can inform — and be informed by — the rest of the board). FORWARD-BEHAVIOR only: the SAME
        # MoveBelief module, a different call site → state_dict identical, projection widths unchanged, NO
        # ARCH_SIGNATURE bump; OFF byte-for-byte. Gated in check_compatible (a resume flip feeds a different
        # forward). Requires move_belief_mode != off (the head must exist to reinject).
        self.move_belief_prefuse = move_belief_prefuse
        if move_belief_prefuse and move_belief_mode == "off":
            raise ValueError(
                "move_belief_prefuse=True requires move_belief_mode != off — there is no move-belief head "
                "to reinject before the transformer. Set --move-belief-mode revealed (or both/unrevealed), "
                "or disable --move-belief-prefuse."
            )
        # move_belief_single_compute (gen3_belief_single_compute_v1): compute the move belief EXACTLY
        # ONCE per forward — pre-attention — and freeze it. Under prefuse the belief is already
        # predicted + reinjected before the transformer, but the between-layers refine callback then
        # RE-READ `move_logits` off the (reinjected) opp tokens, so the belief was computed twice and
        # the physics consumed a different posterior than the one attention was given. When on, the
        # refine kernels reuse the stashed pre-transformer logits, so:
        #   belief ONCE (pre-attention) → physics ONCE → N attention layers that CANNOT change it.
        # This is the "frozen belief" arm of the iterative-refinement A/B (next_run_plan item 3: the
        # per-round recompute only pays if the belief sharpens through layers — measured near-zero
        # under detached). It is also strictly cheaper: one fewer move-belief head pass per forward.
        # FORWARD-BEHAVIOR only (no new params, state_dict identical, projection widths unchanged, NO
        # ARCH_SIGNATURE bump; OFF byte-for-byte) → gated in check_compatible like move_belief_prefuse.
        # Requires move_belief_prefuse: without it the only belief is computed POST-transformer, so
        # there is nothing stashed for the refine callback to reuse.
        self.move_belief_single_compute = move_belief_single_compute
        if move_belief_single_compute and not move_belief_prefuse:
            raise ValueError(
                "move_belief_single_compute=True requires move_belief_prefuse=True — without prefuse "
                "the move belief is computed AFTER the transformer, so the between-layers refine "
                "callback has no pre-attention belief to reuse. Set --move-belief-prefuse, or disable "
                "--move-belief-single-compute."
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
        # gen3_nature_ev_belief_v1: --spread-belief-nature swaps the additive point-estimate head for the
        # NATURE/EV generative head (prior-fusion → compute the derived stat) to fix the largest-EV over-estimate.
        # Requires --spread-belief (the head IS the SpreadBelief module). STRUCTURAL (different SpreadBelief params).
        if spread_belief_nature and not spread_belief:
            raise ValueError("spread_belief_nature requires spread_belief=True (it parameterises the "
                             "SpreadBelief head). Enable --spread-belief, or drop --spread-belief-nature.")
        # gen3_nature_ev_belief_v1: the op marginalises P(KO) over the head's nature distribution → requires it.
        if spread_belief_nature_marginalize and not spread_belief_nature:
            raise ValueError("spread_belief_nature_marginalize requires spread_belief_nature=True (the op "
                             "marginalises over the generative head's nature distribution).")
        self.spread_belief_enabled = spread_belief
        self.spread_belief_nature = spread_belief_nature
        self.spread_belief_nature_marginalize = spread_belief_nature_marginalize
        self.spread_belief = SpreadBelief(layout['max_species'], nature=spread_belief_nature) if spread_belief else None
        self.last_spread_belief: Optional[torch.Tensor] = None
        self.last_spread_nature_logits: Optional[torch.Tensor] = None   # [B,6,25] (gen3_nature_ev_belief_v1)
        self.last_spread_ev: Optional[torch.Tensor] = None              # [B,6,5] believed EVs
        # gen3_typed_hp_belief_v1 / gen3_hp_belief_ablation_v1: the opponent's Hidden Power is ALWAYS
        # reasoned about as the 16 DISCRETE TYPED moves — the old typeless-BP-0 candidate is gone in both
        # arms, because it was a correctness bug (a 0-damage "immune" reading of a revealed HP), not an
        # ablation. What `hp_belief_mode` varies is HOW the 16 typed channels are produced:
        #
        #   'composed' (DEFAULT) — build `HPTypeBelief` and factor the belief as
        #                          `P(HP_t) = presence · P(type=t)`. Buys the structural constraint
        #                          (a revealed HP must exist as SOME type), the moveset-exhaustion
        #                          rule-out, and the effectiveness narrowing.
        #   'flat'     (ABLATION) — no head. The multi-label move head predicts the 16 typed channels
        #                          INDEPENDENTLY, off their own real per-typed Smogon usage priors, and
        #                          Hidden Power is treated exactly like any other move. No factorisation,
        #                          no constraint, no narrowing.
        #
        # The head is prior-fused + zero-init, so under 'composed' its cold-start posterior IS the Smogon
        # HP-type prior; `--hp-type-belief-coef` controls only whether the privileged CE supervises it on
        # top of the damage + move-BCE gradients it already gets. Neither arm requires `damage_op`: the
        # composition lives in the BELIEF, so the typed posterior reaches the token reinjection, the BCE,
        # the latent grading and the prober even on a run with no damage operator at all.
        self.hp_belief_mode = str(hp_belief_mode)
        if self.hp_belief_mode not in ("composed", "flat"):
            raise ValueError(
                f"hp_belief_mode must be one of composed|flat, got {hp_belief_mode!r}")
        self.hp_type_belief_head = (
            HPTypeBelief(layout['max_species'], layout['type_embedding_dim'])
            if (self.move_belief is not None and self.hp_belief_mode == "composed") else None)
        self.last_hp_type_logits: Optional[torch.Tensor] = None
        self._last_hp_type_post: Optional[torch.Tensor] = None
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
        # gen3_unified_topk_incoming_v1: the DISCRETE top-K incoming block (K = damage_topk_k; 0 = off).
        # Requires the op (it extends it) AND --move-latent (the op gathers each top-K move's LATENT from
        # the MoveLatentEncoder for identity, and the candidate latent table is built only when move_latent).
        self.damage_topk_k = int(damage_topk_k)
        if self.damage_topk_k > 0 and not damage_op:
            raise ValueError(
                "damage_topk_k>0 requires damage_op=True — the top-K incoming block extends the "
                "DamageOperator. Enable --damage-op (--unified-damage), or drop --damage-topk."
            )
        if self.damage_topk_k > 0 and not move_latent:
            raise ValueError(
                "damage_topk_k>0 requires move_latent=True — the top-K block gathers each move's identity "
                "LATENT from the MoveLatentEncoder. Enable --move-latent (--unified-moves), or drop --damage-topk."
            )
        # gen3_per_move_matrices_v1: the OUTGOING per-move DAMAGE MATRIX (our 4 moves × opp active+revealed
        # bench). Requires damage_op (the op physics). Off byte-identical.
        self.damage_matrices_outgoing = bool(damage_matrices_outgoing)
        if self.damage_matrices_outgoing and not damage_op:
            raise ValueError(
                "damage_matrices_outgoing=True requires damage_op=True — the outgoing per-move damage matrix "
                "is emitted by the DamageOperator. Enable --damage-op (--unified-damage), or drop --damage-matrices."
            )
        # gen3_per_move_matrices_v1: the INCOMING per-move DAMAGE MATRIX (enriched top-K). Requires damage_op +
        # move_latent (the latent gather); REUSES damage_topk_k as its K and replaces the lean top-K block.
        self.damage_matrices_incoming = bool(damage_matrices_incoming)
        if self.damage_matrices_incoming and not damage_op:
            raise ValueError(
                "damage_matrices_incoming=True requires damage_op=True — the incoming per-move damage matrix "
                "is emitted by the DamageOperator. Enable --damage-op (--unified-damage), or drop --damage-matrices."
            )
        if self.damage_matrices_incoming and not move_latent:
            raise ValueError(
                "damage_matrices_incoming=True requires move_latent=True — the matrix header gathers each "
                "move's identity LATENT. Enable --move-latent (--unified-moves), or drop the incoming matrix."
            )
        # gen3_per_move_matrices_v1 (v39): the TRANSPOSED outgoing matrix (our 6 mons' moves → opp active — the
        # switch-in offense read). Requires damage_op (the op physics). Off byte-identical.
        self.damage_matrices_outgoing_all = bool(damage_matrices_outgoing_all)
        if self.damage_matrices_outgoing_all and not damage_op:
            raise ValueError(
                "damage_matrices_outgoing_all=True requires damage_op=True — the transposed outgoing matrix "
                "(our 6 mons' moves → opp active) is emitted by the DamageOperator. Enable --damage-op "
                "(--unified-damage), or drop --damage-matrices-outgoing-all."
            )
        # gen3_topk_candidates_v1: the incoming candidate-sweep cap (0 = full ~400-wide sweep).
        self.damage_candidate_k = int(damage_candidate_k)
        if self.damage_candidate_k and not damage_op:
            raise ValueError("damage_candidate_k>0 requires damage_op=True — it caps the DamageOperator's "
                             "incoming candidate sweep, which only exists when the op is built.")
        # The incoming matrix REUSES damage_topk_k as its K (one knob tunes both lean & rich) and REPLACES the
        # lean top-K block — so they never coexist (the op suppresses the lean block when matrices_incoming).
        self.damage_op = (DamageOperator(layout, outgoing=damage_outgoing, topk_k=self.damage_topk_k,
                                         matrices_outgoing=self.damage_matrices_outgoing,
                                         matrices_incoming=self.damage_matrices_incoming,
                                         matrices_outgoing_all=self.damage_matrices_outgoing_all,
                                         prob_outspeed=threat_prob_outspeed,
                                         candidate_k=self.damage_candidate_k)
                          if damage_op else None)
        self.threat_prob_outspeed = bool(threat_prob_outspeed)
        # damage_reattend (gen3_damage_reattend_v1): let attention reason OVER the computed physics. The op's
        # per-OUR-mon INCOMING damage block (incl. the bench = safe-switch reads) is projected onto the 6
        # our-team tokens as a small-init residual, then ONE more TransformerEncoderLayer lets the
        # damage-aware our tokens attend to the opp tokens (threats) + each other; the CLS pools are then
        # RE-DERIVED from the re-attended tokens (forward_internal), so both heads + the switch logits read
        # damage-contextualised summaries. Re-pooling keeps the SAME pooled shapes ⇒ projection widths are
        # UNCHANGED — a state_dict change ONLY via the 3 modules below, so OFF is byte-for-byte (gated in
        # check_compatible like opp_belief_slots; NO ARCH_SIGNATURE bump). Requires damage_op (the source).
        self.damage_reattend_enabled = damage_reattend
        if damage_reattend and not damage_op:
            raise ValueError(
                "damage_reattend=True requires damage_op=True — the re-attend layer reads the operator's "
                "per-mon incoming damage block. Enable --damage-op (--unified-damage), or drop --damage-reattend."
            )
        if damage_reattend:
            # Small-init residual (mirrors MoveBelief.reinject): the damage enrichment starts ≈0 so the
            # damage signal grows over training rather than shocking the tokens at step 0.
            self.reattend_proj = torch.nn.Linear(_DMG_PER_MON, D_MODEL, bias=False)
            torch.nn.init.normal_(self.reattend_proj.weight, std=0.02)
            self.reattend_norm = torch.nn.LayerNorm(D_MODEL)
            self.reattend_layer = torch.nn.TransformerEncoderLayer(
                d_model=D_MODEL, nhead=TRANSFORMER_N_HEADS, dim_feedforward=TRANSFORMER_FFN_DIM,
                dropout=0.0, activation="relu", batch_first=True, norm_first=False)
            # IDENTITY-AT-INIT: zero the layer's two output paths (attention out-proj + FFN second linear)
            # so its residual contributions are 0 at step 0 → reattend_layer(tok) ≈ tok (the post-transformer
            # tokens are already LayerNorm-standardised, so the post-LN passes through). Combined with the
            # ≈0 damage residual, ON then starts ≈ the damage_op baseline (a clean A/B + no re-pool shock to
            # the CLS pools); the layer + projection learn the enrichment from there.
            torch.nn.init.zeros_(self.reattend_layer.self_attn.out_proj.weight)
            torch.nn.init.zeros_(self.reattend_layer.self_attn.out_proj.bias)
            torch.nn.init.zeros_(self.reattend_layer.linear2.weight)
            torch.nn.init.zeros_(self.reattend_layer.linear2.bias)
        else:
            self.reattend_proj = None
            self.reattend_norm = None
            self.reattend_layer = None
        # gen3_iterative_damage_v1: ITERATIVE damage refinement. N>0 recomputes the op's LEAN discrete
        # incoming threat BETWEEN transformer layers (as the opp token / move belief is enriched by
        # attention) and injects it back onto our-mon tokens via `refine_proj`, so each layer attends over
        # physics derived from the CURRENT belief. STRUCTURAL toggle: 0 = off (no module, the transformer
        # callback is None → byte-identical baseline forward); N>0 builds `refine_proj` (weight-tied across
        # rounds, so its SHAPE is N-independent) and changes the forward (so EVERY distinct N — incl. 0↔N
        # AND N↔M — is version-gated). Requires damage_op (→ the op's physics + a move_belief to re-read).
        self.damage_refine_rounds = int(damage_refine_rounds)
        if self.damage_refine_rounds < 0:
            raise ValueError(f"damage_refine_rounds must be >= 0 (0 = off), got {damage_refine_rounds}")
        if self.damage_refine_rounds > 0 and not damage_op:
            raise ValueError(
                "damage_refine_rounds>0 requires damage_op=True — the iterative refinement recomputes the "
                "DamageOperator's lean incoming threat between transformer layers (and re-reads the move "
                "belief, which damage_op requires). Enable --damage-op (--unified-damage / --unified-moves), "
                "or set --damage-refine-rounds 0."
            )
        self.refine_proj = (torch.nn.Linear(_DMG_REFINE_FEATS, D_MODEL)
                            if self.damage_refine_rounds > 0 else None)
        if self.refine_proj is not None:
            # Zero-init → the injected residual is EXACTLY 0 at init, so ON starts byte-identical to the
            # baseline transformer (identity-at-init). The gradient still flows (∂out/∂W = the damage feats,
            # which are non-zero), so the model learns the enrichment from the first optimizer step. No
            # LayerNorm on the residual branch (a LayerNorm of a ~0 vector is ill-conditioned, and post-LN
            # transformer layers already renormalize downstream).
            torch.nn.init.zeros_(self.refine_proj.weight)
            torch.nn.init.zeros_(self.refine_proj.bias)
        # gen3_bidir_threat_trunk_v1: the OUTGOING half of the in-trunk threat field (#1). It reuses the SAME
        # between-layers refine loop (damage_refine_rounds), injecting a per-OPP-mon outgoing-threat residual
        # onto the OPP token slice via a zero-init `outgoing_proj` (symmetric to refine_proj). #2 (the
        # expected-LATENT defender for unrevealed mons) is folded into the kernel when the belief is on and
        # threat_unrevealed_outgoing is set. STRUCTURAL like refine_proj (adds a Linear; OFF byte-identical).
        self.threat_refine_outgoing = bool(threat_refine_outgoing)
        self.threat_unrevealed_outgoing = bool(threat_unrevealed_outgoing)
        if self.threat_refine_outgoing and self.damage_refine_rounds <= 0:
            raise ValueError(
                "threat_refine_outgoing=True requires damage_refine_rounds>0 — the outgoing residual rides "
                "the SAME between-layers refine loop. Set --damage-refine-rounds N (and --damage-op)."
            )
        if self.threat_refine_outgoing and not damage_op:
            raise ValueError("threat_refine_outgoing=True requires damage_op=True (the outgoing physics).")
        if self.threat_unrevealed_outgoing and not self.threat_refine_outgoing:
            raise ValueError(
                "threat_unrevealed_outgoing=True requires threat_refine_outgoing=True — it only enriches the "
                "outgoing residual's UNREVEALED columns (expected-latent defender)."
            )
        self.outgoing_proj = (torch.nn.Linear(_DMG_OUT_REFINE, D_MODEL)
                              if self.threat_refine_outgoing else None)
        if self.outgoing_proj is not None:
            torch.nn.init.zeros_(self.outgoing_proj.weight)   # identity-at-init, same rationale as refine_proj
            torch.nn.init.zeros_(self.outgoing_proj.bias)
        # gen3_status_trunk_v1 (v37): STATUS-LANDING into the trunk (the last CPU-obs deprecation gap). Two
        # zero-init residuals riding the SAME between-layers refine loop: INCOMING status (opp active's
        # believed status moves → our 6, onto OUR tokens) + OUTGOING status (our active's status moves → opp
        # 6, revealed-gated, onto OPP tokens). Status immunity is a computed MECHANICS fact (not learnable
        # strategy); we hand it over so the model spends capacity on HOW to value it, not on re-deriving the
        # gen3 immunity rules across non-local tokens. STRUCTURAL (adds two Linears; OFF byte-identical).
        self.threat_status_refine = bool(threat_status_refine)
        # Prober-only: when True, the between-layers refine_cb stashes its per-round (move_logits, lean
        # incoming-damage) into `last_refine_rounds` for the observability TUI. Default False → the
        # training/rollout forward never captures (byte-for-byte unchanged); the prober flips it per re-run.
        self.capture_refine_rounds = False
        if self.threat_status_refine and not damage_op:
            raise ValueError("threat_status_refine=True requires damage_op=True (the status physics).")
        if self.threat_status_refine and self.damage_refine_rounds <= 0:
            raise ValueError(
                "threat_status_refine=True requires damage_refine_rounds>0 — the status residuals ride the "
                "SAME between-layers refine loop. Set --damage-refine-rounds N."
            )
        self.status_in_proj = (torch.nn.Linear(_DMG_STATUS_REFINE, D_MODEL)
                               if self.threat_status_refine else None)   # incoming → OUR tokens
        self.status_out_proj = (torch.nn.Linear(_DMG_STATUS_REFINE, D_MODEL)
                                if self.threat_status_refine else None)  # outgoing → OPP tokens
        for _p in (self.status_in_proj, self.status_out_proj):
            if _p is not None:
                torch.nn.init.zeros_(_p.weight)   # identity-at-init (same rationale as refine_proj/outgoing_proj)
                torch.nn.init.zeros_(_p.bias)
        # gen3_damage_op_prefuse_v1 (v50): ONE damage computation per forward, PRE-attention. Today the op
        # runs TWICE — a LEAN `discrete_*` recompute inside the between-layers refine loop (2 rounds in the
        # production config) plus the FULL 835-dim block after the transformer — and at B=1 on CPU (the PFSP
        # frozen-opponent regime, which sits on the rollout critical path) the two together are ~75% of a
        # dispatch-bound 6.45 ms forward, against 0.27 ms for the attention layers themselves. ON: the
        # spread/HP-type beliefs + the FULL op run on the PRE-transformer role tokens, the per-OUR-mon
        # incoming rows are injected onto our tokens via the zero-init `prefuse_proj` (the refine_proj
        # convention — identity-at-init), and the SAME block is concatenated to both heads. The refine loop
        # is then redundant, so the two are mutually exclusive.
        #
        # HONEST POSTURE: the justification is the CPU cost. The architectural story ("attention now reasons
        # over full-fidelity physics") is SECONDARY and, on this codebase's evidence, unlikely to pay —
        # physics-into-the-trunk measured NULL 3-for-3 (ledger K9/K10) and the lean kernel was already a good
        # proxy for the full op (K10a). What must be PRESERVED is the head concat, which ledger P1 measured as
        # the policy's largest single dependency; ON keeps it at full width, only sourced from a pre-attention
        # belief. STRUCTURAL (adds `prefuse_proj`) → gated in check_compatible; OFF byte-for-byte.
        self.damage_op_prefuse = bool(damage_op_prefuse)
        if self.damage_op_prefuse and not damage_op:
            raise ValueError(
                "damage_op_prefuse=True requires damage_op=True — there is no operator to run before the "
                "transformer. Enable --damage-op (--unified-damage), or drop --damage-op-prefuse."
            )
        if self.damage_op_prefuse and not move_belief_prefuse:
            raise ValueError(
                "damage_op_prefuse=True requires move_belief_prefuse=True — the op consumes the move "
                "belief, so the belief must itself be computed BEFORE the transformer. Set "
                "--move-belief-prefuse, or drop --damage-op-prefuse."
            )
        if self.damage_op_prefuse and self.damage_refine_rounds > 0:
            raise ValueError(
                f"damage_op_prefuse=True is mutually exclusive with damage_refine_rounds>0 (got "
                f"{self.damage_refine_rounds}) — the prefuse IS the unified single computation the "
                "between-layers refine loop is being replaced by; running both would restore the double "
                "cost it exists to remove. Set --damage-refine-rounds 0 (which also requires dropping "
                "--threat-refine-outgoing / --threat-status-refine), or drop --damage-op-prefuse."
            )
        # Zero-init → the injected residual is EXACTLY 0 at init, so ON starts from the same trunk the
        # baseline transformer sees (identity-at-init; the gradient still flows because the damage feats are
        # non-zero). Richer than the refine loop's lean 4-feature injection: this carries the FULL per-mon
        # incoming row. NOTE the v36/v37 OUTGOING + STATUS trunk residuals are NOT reproduced here — they
        # rode the refine loop and measured null (K10); re-adding them would need their own projections.
        self.prefuse_proj = (torch.nn.Linear(_DMG_PER_MON, D_MODEL)
                             if self.damage_op_prefuse else None)
        if self.prefuse_proj is not None:
            torch.nn.init.zeros_(self.prefuse_proj.weight)
            torch.nn.init.zeros_(self.prefuse_proj.bias)
        # gen3_entity_move_seats_v1: E4 threat seats need the belief-weighted candidate definition
        # (`DamageOperator.refine_candidates`) + the move latent table, both PRE-transformer — i.e.
        # the prefuse stack. E3 is unconditional and needs none of this.
        if self.entity_topk_seats > 0 and not (self.damage_op_prefuse and move_latent):
            raise ValueError(
                f"entity_topk_seats={self.entity_topk_seats} requires damage_op_prefuse=True AND "
                "move_latent=True — the E4 threat seats gather the op's pre-transformer candidate "
                "weights and the move latent table. Enable --damage-op-prefuse + --move-latent "
                "(--unified-moves), or set --entity-topk-seats 0 (E3-only)."
            )
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
        # Stashed EVERY forward (the [B,128] value-CLS pool). The FitNets distillation HINT layer — read by
        # `instrumented_ppo._value_feat_distill` off both the student and teacher forwards. None until the
        # first forward; never fed into pi/vf.
        self.last_value_pooled: Optional[torch.Tensor] = None

        # Auxiliary PUBLIC-VALUE head (tri-state `pubval_mode`, gen3_pubval_aux_v1): the WinProbHead
        # pattern with the frozen HUMAN-replay-calibrated V_pub as the target (dense per-step, exogenous
        # — see PubValHead's docstring). 'none' = no module (baseline byte-for-byte); 'read_only' =
        # head-only training on a STOP-GRAD value_pooled; 'shaping' = the human positional prior also
        # shapes the shared trunk. SIDE readout (stashed for the aux loss, never in pi/vf, never in GAE).
        if pubval_mode not in ("none", "read_only", "shaping"):
            raise ValueError(f"pubval_mode must be none|read_only|shaping, got {pubval_mode!r}")
        self.pubval_mode = pubval_mode
        self.pubval_head = PubValHead() if pubval_mode != "none" else None
        # Stashed each forward when the head is on (the [B,1] V_pub logit, or None). Read by the PPO
        # aux loss; NEVER fed into pi/vf.
        self.last_pubval_logits: Optional[torch.Tensor] = None

        # Distributional VALUE head (tri-state `value_dist_mode`, v29): an interpretability readout off
        # `value_pooled` emitting `value_dist_bins` logits over the support [vmin, vmax]. 'none' = no
        # module (baseline byte-for-byte, NOT in pi/vf so projection dims are unchanged). 'read_only' =
        # trains its OWN params on a STOP-GRAD value_pooled (a risk-free diagnostic — zero trunk
        # gradient). 'shaping' = its gradient also shapes the shared trunk. SIDE readout (stashed for the
        # aux loss + prober, never in pi/vf — and the value target can't leak). The state_dict-changing
        # toggles are 'none'↔head (the head params) AND the atom count `bins` (the head's output width);
        # both + the mode are resume-immutable (version-checked). See ValueDistHead.
        if value_dist_mode not in ("none", "read_only", "shaping"):
            raise ValueError(f"value_dist_mode must be none|read_only|shaping, got {value_dist_mode!r}")
        if value_dist_mode != "none" and value_dist_bins <= 0:
            raise ValueError(
                f"value_dist_mode={value_dist_mode!r} requires value_dist_bins > 0 (the atom count), "
                f"got {value_dist_bins}"
            )
        if value_dist_mode == "none" and value_dist_bins != 0:
            raise ValueError(
                f"value_dist_bins must be 0 when value_dist_mode == 'none', got {value_dist_bins}"
            )
        self.value_dist_mode = value_dist_mode
        self.value_dist_bins = value_dist_bins
        self.value_dist_vmin = value_dist_vmin
        self.value_dist_vmax = value_dist_vmax
        self.value_dist_head = (
            ValueDistHead(value_dist_bins, value_dist_vmin, value_dist_vmax)
            if value_dist_mode != "none" else None
        )
        # Stashed each forward when on (the [B, bins] per-atom logits, or None). Read by the (future)
        # distributional aux loss + the offline prober/eval; NEVER fed into pi/vf (leak-safe side head).
        self.last_value_dist_logits: Optional[torch.Tensor] = None

        # gen3_zarch_film_v1 (v44): the team-archetype latent + head FiLM — the amortization-gap
        # STORAGE fix. 'off' = no modules (byte-identical baseline). 'heads' = build the ZArchEncoder
        # (a team-static DeepSets code over OUR team's invariant facts) + two zero-init FiLM
        # generators, one per root head; forward() then applies `h·(1+Δγ(z)) + Δβ(z)` to each head's
        # post-projection pre-ReLU features. Zero-init ⇒ Δγ=Δβ=0 at init ⇒ ON starts byte-identical
        # to the baseline forward (identity-at-init, the refine_proj convention); the modulation is
        # rank-zarch_dim by construction (low-rank conditioning). STRUCTURAL toggle gated in
        # check_compatible (string + int, the value_dist_mode/bins pattern); NO ARCH_SIGNATURE bump.
        if zarch_film not in ("off", "heads"):
            raise ValueError(f"zarch_film must be off|heads, got {zarch_film!r}")
        if zarch_film != "off" and zarch_dim <= 0:
            raise ValueError(
                f"zarch_film={zarch_film!r} requires zarch_dim > 0 (the latent width), got {zarch_dim}")
        if zarch_film == "off" and zarch_dim != 0:
            raise ValueError(f"zarch_dim must be 0 when zarch_film == 'off', got {zarch_dim}")
        self.zarch_film = zarch_film
        self.zarch_dim = int(zarch_dim)
        if zarch_film != "off":
            self.zarch_encoder = ZArchEncoder(layout, self.zarch_dim)
            # One FiLM generator per head (separate γ/β maps, shared z): Linear(z) → [Δγ ‖ Δβ],
            # both chunks over PROJECTION_DIM. Zero-init weight AND bias → exact identity at init.
            self.film_pi = torch.nn.Linear(self.zarch_dim, 2 * PROJECTION_DIM)
            self.film_vf = torch.nn.Linear(self.zarch_dim, 2 * PROJECTION_DIM)
            for _g in (self.film_pi, self.film_vf):
                torch.nn.init.zeros_(_g.weight)
                torch.nn.init.zeros_(_g.bias)
        else:
            self.zarch_encoder = None
            self.film_pi = None
            self.film_vf = None

        # gen3_zarch_lut_v1 (v46): the per-team LUT — a FREE, unconstrained code per pinned team,
        # added to (or replacing) the DeepSets z. It exists to test ONE thing: is the measured
        # multi-team exploiter ceiling (N=1/3/10 distil cleanly, N=20 stalls) a conditioning-SIGNAL
        # limit or a capacity limit? The DeepSets z is COMPOSITIONAL, so z-similar teams sit at
        # z̄ + tiny ε_i and the FiLM generator's gradient is proportional to that tiny residual —
        # ill-conditioned. A random-init LUT makes the per-team codes LARGE and ~orthogonal by
        # construction, which is exactly the intervention that ill-conditioning story predicts should
        # help. If N=20 still stalls with a free code, the ceiling is NOT signal.
        #   'off'  = no modules (byte-identical).
        #   'add'  = z = LN(z_deepsets + code) — the practical form: composition generalizes to
        #            UNSEEN teams (code row 0 = zeros ⇒ z is exactly the DeepSets z), the code is a
        #            free per-team correction on top.
        #   'only' = z = LN(code) — the sharpest ablation (identity only, no composition).
        # The team is identified from the OBSERVATION (agents.model.team_signature: sorted
        # species(6) ⊕ moves(24)), so nothing in env / eval / prober / frozen-opponent plumbing
        # changes. The table is a PERSISTENT buffer, so the team↔row mapping travels with the
        # checkpoint. STRUCTURAL toggle gated in check_compatible; NO ARCH_SIGNATURE bump.
        if zarch_lut not in ("off", "add", "only"):
            raise ValueError(f"zarch_lut must be off|add|only, got {zarch_lut!r}")
        if zarch_lut != "off" and zarch_film == "off":
            raise ValueError("zarch_lut requires --zarch-film heads (the LUT conditions z, and z is "
                             "only consumed by the FiLM generators)")
        rosters = [list(r) for r in (zarch_lut_rosters or [])]
        if zarch_lut != "off" and not rosters:
            raise ValueError("zarch_lut requires zarch_lut_rosters (the pinned teams' signatures)")
        if zarch_lut == "off" and rosters:
            raise ValueError("zarch_lut_rosters given but zarch_lut == 'off'")
        self.zarch_lut = zarch_lut
        self.zarch_lut_teams = len(rosters)
        if zarch_lut != "off":
            if any(len(r) != TEAM_SIGNATURE_DIM for r in rosters):
                raise ValueError(
                    f"every zarch_lut roster signature must be {TEAM_SIGNATURE_DIM} ints, got "
                    f"{sorted({len(r) for r in rosters})}")
            # PERSISTENT (the team↔row mapping is learned-state-adjacent: a reload with a different
            # table would re-key every code, so it must ride the checkpoint).
            self.register_buffer("zarch_lut_table", torch.tensor(rosters, dtype=torch.long))
            # Row 0 = UNKNOWN team (zeros ⇒ 'add' degrades to exactly the DeepSets z); rows 1..N are
            # random-init so the per-team codes are distinct and ~orthogonal from step 0 — the whole
            # point (a zero-init LUT would reproduce the ill-conditioned starting geometry).
            self.zarch_lut_emb = torch.nn.Embedding(self.zarch_lut_teams + 1, self.zarch_dim)
            # INIT SCALE is an experiment knob, not an architecture field (shapes are identical, and
            # a resume loads saved weights, so it only ever matters at the initial fork — hence NOT
            # version-gated). std>0 = the codes start large + ~orthogonal, which perturbs an already
            # TRAINED FiLM head (arm 1 paid ~-0.04 at the fork and spent ~6M steps recovering).
            # std=0 = identity-at-init: z is exactly the DeepSets z on step 0 and the codes GROW from
            # zero. That does NOT inherit the ill-conditioning — a code is a FREE per-team parameter,
            # so its gradient is the full dL/dz restricted to that team's samples, not something
            # scaled by a tiny compositional residual.
            self.zarch_lut_init_std = float(zarch_lut_init_std)
            if self.zarch_lut_init_std > 0:
                torch.nn.init.normal_(self.zarch_lut_emb.weight, mean=0.0, std=self.zarch_lut_init_std)
            else:
                torch.nn.init.zeros_(self.zarch_lut_emb.weight)
            with torch.no_grad():
                self.zarch_lut_emb.weight[0].zero_()          # row 0 = unknown team, always zero
            self.zarch_lut_norm = torch.nn.LayerNorm(self.zarch_dim)
        else:
            self.zarch_lut_emb = None
            self.zarch_lut_norm = None
        # Stashed each forward when on (else None): the live z [B, zarch_dim] (read by forward()'s
        # FiLM application + the aux loss + probes), the recon logits [B, max_species] + our-side
        # species ids [B, 6] (grad-gated — training epochs only; read ONLY by the recon BCE, so the
        # public our-team composition never routes anywhere new in the forward), and the resolved
        # LUT row [B] (a side read for metrics/probes — 0 = unmatched team).
        self.last_zarch: Optional[torch.Tensor] = None
        self.last_zarch_recon_logits: Optional[torch.Tensor] = None
        self.last_zarch_species_ids: Optional[torch.Tensor] = None
        self.last_zarch_lut_idx: Optional[torch.Tensor] = None

        self.role_token_size = ROLE_TOKEN_SIZE

        # gen3_belief_grad_mode_v1: stamp the per-head trunk-read detach flag now that every belief head
        # exists (before the dummy forward — shapes are identical in either mode, so auto-sizing is
        # unaffected). 'shaping' ⇒ all False ⇒ byte-identical. BeliefSlots has no predictive read (it only
        # swaps in learned tokens pre-transformer), so it is intentionally NOT in this list.
        for _bh in (self.move_belief, self.spread_belief, self.hp_type_belief_head, self.belief_head):
            if _bh is not None:
                _bh.detach_read = self._belief_detach

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

        # gen3_identity_init_guard_v1 — SNAPSHOT the identity-at-init contract. See
        # `restore_identity_init` for why this exists; it must be the LAST thing __init__ does, so
        # every module is built and every deliberate zero-init has already been applied.
        self._identity_init_zeroed: Tuple[str, ...] = tuple(
            name for name, mod in self.named_modules()
            if isinstance(mod, torch.nn.Linear) and not bool(mod.weight.any())
        )

    def disable_observation_debugger(self) -> bool:
        """Detach the `ObservationDebugger`. Returns True if one was attached.

        The debugger runs NUMPY assertions inside `forward`. That is fine eagerly, but `torch.compile`
        cannot trace it — dynamo dies building a guard over a numpy bool ("TypeError: 'numpy.bool'
        object cannot be interpreted as an integer"). It is a LEARNER-side diagnostic and a frozen
        opponent has no use for it, so the compile path drops it.

        This is a METHOD rather than the caller reaching in and setting `fe._debugger = None`, so the
        ownership stays here: if the debugger ever gains teardown state, this is the one place that
        has to learn about it."""
        had = self._debugger is not None
        self._debugger = None
        return had

    def restore_identity_init(self) -> int:
        """Re-zero every Linear this extractor deliberately zero-initialised. Returns the count.

        WHY THIS EXISTS. SB3's `ActorCriticPolicy._build()` runs
        ``self.features_extractor.apply(partial(self.init_weights, gain=sqrt(2)))``
        (stable_baselines3/common/policies.py:617-631), and `init_weights` ORTHOGONALLY
        re-initialises EVERY `nn.Linear` it finds. `ortho_init` defaults True and nothing here
        overrides it — so every deliberate zero-init inside the extractor was silently destroyed the
        moment the policy was built, in every real training run.

        That silently falsified a documented invariant for six shipped features — `refine_proj`
        (v31/v33), `outgoing_proj` (v36), `status_in_proj`/`status_out_proj` (v37) and `film_pi`/
        `film_vf` (v44) all claim "zero-init ⇒ identity-at-init ⇒ ON starts byte-identical" — and,
        more insidiously, the belief heads' cold-start contract: `MoveBelief.move_head`,
        `SpreadBelief.{stat,nature,ev}_head` and `HPTypeBelief.type_head` are zero-init precisely so
        the cold-start posterior EQUALS the Smogon prior. Clobbered, they start at prior ⊕ noise.

        It went unnoticed because every unit test builds the module or a bare extractor DIRECTLY,
        where the zero-init survives; only SB3-wrapped construction destroys it.

        The set is captured by OBSERVATION at the end of `__init__` (any Linear whose weight is
        all-zero once construction finishes was zero-init'd on purpose) rather than by a hand-kept
        list, so a future zero-init module is protected automatically — the failure mode that
        produced this bug cannot recur by omission. Biases are not tracked: SB3's `init_weights`
        already zeroes every bias.
        """
        by_name = dict(self.named_modules())
        n = 0
        for name in self._identity_init_zeroed:
            mod = by_name.get(name)
            if isinstance(mod, torch.nn.Linear):
                torch.nn.init.zeros_(mod.weight)
                if mod.bias is not None:
                    torch.nn.init.zeros_(mod.bias)
                n += 1
        return n

    def set_belief_grad_mode(self, mode: str) -> None:
        """Apply a belief-grad-mode at RUNTIME (the --allow-belief-grad-mode-change migration path).

        SB3's load reconstructs the extractor from the ZIP's saved policy_kwargs, so a resume that
        passes a different --belief-grad-mode would otherwise be a SILENT NO-OP (the 2026-07-21
        incident: the migration notice printed but the loaded extractor kept 'detached' —
        grad/*_norm_shared stayed exactly 0). The mode lives in THREE places (this attr,
        `_belief_detach`, and the `detach_read` flag stamped on each belief head); this is the ONE
        setter that updates them all — call it post-load on the resume path (a no-op when unchanged)."""
        if mode not in ("shaping", "detached"):
            raise ValueError(f"belief_grad_mode must be shaping|detached, got {mode!r}")
        changed = mode != getattr(self, "belief_grad_mode", None)
        self.belief_grad_mode = mode
        self._belief_detach = (mode == "detached")
        for _bh in (self.move_belief, self.spread_belief, self.hp_type_belief_head, self.belief_head):
            if _bh is not None:
                _bh.detach_read = self._belief_detach
        if changed:
            print(f"[Gen3FeaturesExtractor] belief_grad_mode APPLIED at runtime -> {mode!r} "
                  f"(detach_read={'on' if self._belief_detach else 'off'} across the belief heads)")

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

    # gen3_pointer_native_v1: the pointer head's per-action cell widths — the policy sizes the head's
    # Linears from these at build time (0 when the source op block is off, so a missing block narrows
    # the Linear instead of silently feeding zeros at a learned weight).
    @property
    def pointer_move_token_dim(self) -> int:
        # gen3_entity_move_seats_v1: the head reads the REFINED E3 seats (d_model), no longer the raw
        # 32-dim PokemonEncoder move tokens.
        return D_MODEL

    @property
    def pointer_move_cell_dim(self) -> int:
        return self.damage_op.pointer_move_cell_dim if self.damage_op is not None else 0
    @property
    def pointer_switch_cell_dim(self) -> int:
        return self.damage_op.pointer_switch_cell_dim if self.damage_op is not None else 0

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

    def _typed_hp_posterior(self, opp_tokens, ctx, raw_move_logits):
        """Compose the raw move posterior into the TYPED-Hidden-Power one → `(typed_logits, presence,
        hp_type_posterior)` (gen3_typed_hp_belief_v1).

        The HP-type head reads THE SAME `opp_tokens` the move head just read, at the same point in the
        forward, so the two halves of `P(HP_t) = presence · P(t)` can never be sourced from differently
        refined tokens. (Before this, the type head lived in `_spread_hp_damage` — which under
        `--move-belief-prefuse` alone runs POST-transformer while the move head runs PRE-transformer, so
        the factors came from two different states of the same slot.)

        Under the **`flat` ABLATION** (`--hp-belief-mode flat`) there is no head: Hidden Power is just
        16 more ordinary move channels that the multi-label move head predicts INDEPENDENTLY, off
        their own real per-typed Smogon usage priors, with no factorisation, no reveal constraint and
        no tracker narrowing. All that survives is masking the bare 237 — which is not a moderation of
        the ablation but a necessity: 237 carries BP 0, so leaving it in the damage candidate set is
        the original "opp HP reads immune" bug, not an arm of the experiment. See the class docstring
        of `HPTypeBelief` for what the ablation is actually testing."""
        if self.hp_type_belief_head is None:                  # flat ablation — HP is an ordinary move
            return mask_typeless_hp(raw_move_logits), None, None, None
        hp_logits, hp_post = self.hp_type_belief_head(opp_tokens, ctx.species_ids[:, TEAM_SIZE:])
        typed, presence = self.hp_type_belief_head.compose_typed_hp(
            raw_move_logits, hp_post,
            ctx.hp_probs[:, TEAM_SIZE:],                     # [B,6,16] tracker narrowing (OPP slots)
            ctx.all_move_ids[:, TEAM_SIZE:, :])              # [B,6,4] revealed ids (rule-out)
        return typed, presence, hp_post, hp_logits

    def _apply_move_belief(self, opp_tokens, ctx):
        """Predict + reinject the opp moveset into the given opp tokens [B, 6, D] → (enriched, logits).
        Shared by the POST-transformer (default) and PRE-transformer (move_belief_prefuse) call sites —
        only the input tensor + timing differ. The mask selects the slots per move_belief_mode; the
        species/move ids feed prior-fusion (Smogon prior + pin revealed moves certain).

        gen3_typed_hp_belief_v1: the HP-type head + the typed composition run HERE, between the move
        head's read and the reinjection, so the posterior that leaves this method — and therefore the
        one every consumer reads (`last_move_belief_logits`) — is already typed. The reinjection then
        soft-embeds REAL typed moves rather than the typeless 237 row."""
        if self.move_belief_mode == "revealed":
            mb_mask = ~ctx.opp_believed_mask                 # revealed-species slots
        elif self.move_belief_mode == "unrevealed":
            mb_mask = ctx.opp_believed_mask                  # hidden-species slots
        else:                                                # "both"
            mb_mask = torch.ones_like(ctx.opp_believed_mask)
        raw = self.move_belief.move_logits(
            opp_tokens,
            ctx.species_ids[:, TEAM_SIZE:],                                  # [B, 6]
            ctx.all_move_ids[:, TEAM_SIZE:, :])                              # [B, 6, 4]
        logits, presence, hp_post, self.last_hp_type_logits = self._typed_hp_posterior(
            opp_tokens, ctx, raw)
        self._last_hp_type_post = hp_post                     # for the between-layers refine recompose
        enriched = self.move_belief.reinject_moves(
            opp_tokens, mb_mask, self.embeddings.move_embedding, logits)
        # gen3_opp_hp_type_belief_v2: ALSO reinject the presence-gated expected TYPE embedding. This is
        # deliberately not redundant with the move soft-embed above: that one injects believed move
        # IDENTITY (the 355-370 rows), this one injects the believed TYPE in the shared type-embedding
        # space the mon's own types live in — so "this Zapdos threatens ICE" lands in the same geometry
        # attention already uses for type matchups. Revealed slots only. (No head under `flat` — the
        # typed move rows still ride the soft-embed above, which is the point of that ablation.)
        if self.hp_type_belief_head is not None:
            enriched = self.hp_type_belief_head.reinject(
                enriched, hp_post, presence, (~ctx.opp_believed_mask).float(), self.embeddings)
        return enriched, logits

    def _spread_hp_damage(self, opp_tokens, ctx):
        """The spread + HP-type belief legs and the FULL DamageOperator, in ONE place.

        `opp_tokens` [B, 6, D] → `(enriched_opp_tokens, damage_block | None)`. Called from exactly two
        sites, which differ ONLY in WHEN they run:
          * POST-transformer (default) — the historical site: the beliefs read attention-REFINED opp
            tokens, so the 835-dim head block prices physics from a sharpened belief.
          * PRE-transformer (`damage_op_prefuse`, gen3_damage_op_prefuse_v1) — the beliefs read the raw
            role tokens, the op runs ONCE, and its output both seeds the trunk (see `prefuse_proj`) and
            feeds the heads. One op call per forward instead of two (the between-layers refine loop's
            lean recompute + this full one).
        Every stash (`last_spread_belief`, `last_hp_type_logits`, `last_move_latent_table`,
        `last_damage_block`) is written here, so the aux losses and the prober read the same tensors
        either way.
        """
        # gen3_unified_spread_belief_v1: predict + reinject the opp's hidden SPREAD (revealed slots), and
        # stash the believed stats [B,6,5] for the DamageOperator (consumed at the opp active slot, replacing
        # its hand-coded spread constants) + the speed-supervision loss. Enriches the opp tokens before the
        # CLS pools, like MoveBelief. Hidden slots aren't enriched (their species num 0 → flat prior) and the
        # op only reads the (revealed) active slot.
        if self.spread_belief is not None:
            (opp_tokens, self.last_spread_belief,
             self.last_spread_nature_logits, self.last_spread_ev) = self.spread_belief(
                opp_tokens, ~ctx.opp_believed_mask, ctx.species_ids[:, TEAM_SIZE:])
        else:
            self.last_spread_belief = None
            self.last_spread_nature_logits = None
            self.last_spread_ev = None
        # gen3_typed_hp_belief_v1: the opp-HP-TYPE head + its typed composition + its token reinjection all
        # moved UP into `_apply_move_belief`, where the move head reads the same tokens at the same time —
        # so `last_move_belief_logits` is ALREADY typed by the time it reaches here and the op needs no
        # HP-type argument. `last_hp_type_logits` (the aux-CE + prober stash) is written there too.
        # gen3_unified_move_system_v1: the context-free move-latent table — the Stage-3 latent grading aux
        # TARGET (training only; is_grad_enabled-gated, rollout pays nothing) AND
        # (gen3_unified_topk_incoming_v1) the op's top-K candidate latents. The latter must be present in
        # rollout too (the op output feeds both heads), so when topk is on the table is built EVERY forward.
        # One `latent_table()` call, reused for both.
        self.last_move_latent_table = None
        move_latent_all = None
        # The op's candidate latent table is needed in rollout (not just is_grad_enabled) when EITHER the top-K
        # block OR the incoming per-move matrix is on — both gather the per-move latent into the op output —
        # OR (gen3_entity_move_seats_v1) the E4 threat seats are on: they gather the per-candidate latent as
        # the seat identity (and this method runs PRE-transformer under prefuse, which E4 requires — so the
        # stash below is guaranteed to exist by seat-build time).
        need_topk_latent = self.damage_op is not None and (
            self.damage_op.topk_k > 0 or self.damage_op.matrices_incoming
            or self.entity_topk_seats > 0)
        if self.move_latent and (torch.is_grad_enabled() or need_topk_latent):
            enc = self.pokemon_encoder.move_latent_encoder
            latent_table = enc.latent_table(self.embeddings)                     # [n_moves, MOVE_LATENT_DIM]
            if torch.is_grad_enabled():
                self.last_move_latent_table = latent_table                       # grading aux target
            if need_topk_latent:
                # gen3_opp_hp_typed_candidates_v1: the op's candidate axis is C = n_moves — the typed HPs are
                # the real move-nums 355-370, whose latents already carry their type (move_emb[355-370] ⊕ the
                # type emb ⊕ MOVE_ATTR), so a selected HP-Ice candidate gets the genuine typed-move latent. No
                # synthetic append (the old `hp_latent_block` workaround for the 237 collision is obsolete).
                move_latent_all = latent_table                                   # [n_moves, MOVE_LATENT_DIM]
        # gen3_entity_move_seats_v1: LIVE stash for the E4 seat builder (same forward, read in
        # forward_internal right after this returns; live, not detached — the latent gradient rides).
        self._entity_latent_table = move_latent_all if self.entity_topk_seats > 0 else None
        # Differentiable damage op (flag-guarded; None when off): fed the move belief's PREDICTED moves for
        # the opp active. Forward-only, leak-free; its gradient flows back into the move/spread belief heads
        # via last_move_belief_logits / last_spread_belief.
        damage_block = None
        if self.damage_op is not None:
            # gen3_nature_ev_belief_v1: pass the nature posterior to the op ONLY when marginalization is on (the
            # op then marginalises P(KO) over the nature distribution; None → mean-field, byte-identical).
            spread_nat = self.last_spread_nature_logits if self.spread_belief_nature_marginalize else None
            # Optional gradient-checkpointing (same gate as the transformer): the op materialises several
            # [B,6,~416] activations → recompute in backward for ~GBs of VRAM. Bit-exact (no dropout/RNG);
            # a no-op under inference. ctx is a non-tensor arg (use_reentrant=False); the belief tensors carry
            # the grad. move_latent_all (built above) is the op's top-K identity source (None unless topk on).
            if self.damage_op.grad_checkpointing and torch.is_grad_enabled():
                damage_block = checkpoint(self.damage_op, ctx, self.last_move_belief_logits,
                                          self.last_spread_belief, move_latent_all, spread_nat,
                                          use_reentrant=False)
            else:
                damage_block = self.damage_op(ctx, self.last_move_belief_logits, self.last_spread_belief,
                                              move_latent_all, spread_nat)
        # Read-only stash for the prober/forensic decode — never read by the forward, so off is unchanged.
        self.last_damage_block = damage_block
        return opp_tokens, damage_block

    def attach_zarch_lut(self, mode: str, rosters: Sequence[Sequence[int]], init_std: float = 1.0):
        """ATTACH the per-team LUT to an ALREADY-LOADED LUT-less extractor (the exploiter fork).

        `gen3_zarch_lut_v1`. An exploiter always WARM-FORKS from the generalist (measured: 0.84 @2M
        forked vs ~0.65 @20M from scratch), and the generalist never carries a LUT — so "add the LUT"
        is inherently a fork operation, not a fresh-run one. SB3 rebuilds the extractor from the ZIP's
        saved policy_kwargs, so the loaded module has no LUT; this builds it in place, exactly like
        `set_belief_grad_mode` / `set_value_from_dist` apply their post-load migrations.

        Returns the NEW parameters so the caller can hand them to the optimizer as a fresh param
        GROUP — appending rather than reordering, so the existing params keep their positions (SB3
        restores optimizer state BY POSITION; see the resume-optimizer-realign note in
        `src/agents/model/CLAUDE.md`).
        """
        if mode == "off":
            return []
        if self.zarch_film == "off":
            raise ValueError("attach_zarch_lut requires zarch_film != 'off'")
        rosters = [list(r) for r in rosters]
        if any(len(r) != TEAM_SIGNATURE_DIM for r in rosters):
            raise ValueError(f"every roster signature must be {TEAM_SIGNATURE_DIM} ints")
        device = next(self.parameters()).device
        self.zarch_lut = mode
        self.zarch_lut_teams = len(rosters)
        self.register_buffer("zarch_lut_table", torch.tensor(rosters, dtype=torch.long, device=device))
        self.zarch_lut_emb = torch.nn.Embedding(len(rosters) + 1, self.zarch_dim).to(device)
        self.zarch_lut_init_std = float(init_std)
        if self.zarch_lut_init_std > 0:                       # see the __init__ note on init scale
            torch.nn.init.normal_(self.zarch_lut_emb.weight, mean=0.0, std=self.zarch_lut_init_std)
        else:
            torch.nn.init.zeros_(self.zarch_lut_emb.weight)
        with torch.no_grad():
            self.zarch_lut_emb.weight[0].zero_()
        self.zarch_lut_norm = torch.nn.LayerNorm(self.zarch_dim).to(device)
        return list(self.zarch_lut_emb.parameters()) + list(self.zarch_lut_norm.parameters())

    def _zarch_lut_index(self, ctx: ExtractorContext) -> torch.Tensor:
        """Resolve OUR team to its LUT row [B] (long) from the observation. 0 = no match.

        `gen3_zarch_lut_v1`. The signature is the same sorted species(6) ⊕ moves(24) that
        `agents.model.team_signature` builds offline from each pinned team's Showdown export — so
        the extractor identifies the team with ZERO env/eval plumbing. Both blocks are sorted, so
        the match is invariant to team order and move-slot order; both are within-battle invariant
        (species never changes; our own moveset never changes — move-set mutators are rejected at
        table-build time), so a battle's row can never flip mid-game.

        An UNMATCHED team (the generalist's pool, or a probe on some other team) resolves to row 0,
        whose embedding is zero-init'd — so `add` degrades to exactly the DeepSets z rather than
        conditioning on a wrong team's code. Cost is a [B, n_teams, 30] bool compare: negligible.
        """
        species = ctx.species_ids[:, :TEAM_SIZE].sort(dim=1).values                       # [B, 6]
        moves = ctx.all_move_ids[:, :TEAM_SIZE, :].reshape(species.shape[0], -1)
        moves = moves[:, :TEAM_SIGNATURE_MOVES].sort(dim=1).values                        # [B, 24]
        sig = torch.cat([species, moves], dim=1)                                          # [B, 30]
        match = (sig[:, None, :] == self.zarch_lut_table[None, :, :]).all(dim=-1)          # [B, N]
        # +1 because row 0 is the unknown-team slot; `any()` keeps unmatched teams at 0.
        return torch.where(match.any(dim=1), match.long().argmax(dim=1) + 1,
                           torch.zeros(sig.shape[0], dtype=torch.long, device=sig.device))

    def forward_internal(self, obs):
        """Build the (pi_combined, vf_combined) pre-projection pair by chaining the phases."""
        ctx = self.unpack(obs)
        # Expose which opp slots are believed (hidden) so eval/forensic tooling can decode the belief
        # head's per-slot species prediction for exactly those slots. Read-only stash — never read by
        # the forward itself, so the off/baseline output is unchanged.
        self.last_opp_believed_mask = ctx.opp_believed_mask
        self.last_opp_active_local = ctx.opp_active_local   # for the prober's per-round belief-row decode
        # gen3_status_trunk_v1 / observability: prober-only capture of the WITHIN-FORWARD refine trajectory.
        # `capture_refine_rounds` defaults False (set ONLY by the prober before a re-run forward), so the
        # rollout/training path never touches it → byte-for-byte unchanged. Reset the accumulator each forward.
        self.last_refine_rounds = [] if getattr(self, "capture_refine_rounds", False) else None
        # gen3_zarch_film_v1: the team-archetype latent — computed from ctx's INVARIANT our-side
        # facts only (team-static + deterministic ⇒ constant within a battle by construction).
        # Stashed live for forward()'s FiLM application (same call) + the aux loss (same backward
        # graph, the last_move_belief_logits pattern). The recon logits + species ids are grad-gated
        # (training epochs only — rollout/eval pays only the tiny encoder forward FiLM needs).
        self.last_zarch = None
        self.last_zarch_recon_logits = None
        self.last_zarch_species_ids = None
        self.last_zarch_lut_idx = None
        if self.zarch_encoder is not None:
            self.last_zarch = self.zarch_encoder(ctx, self.embeddings)
            if torch.is_grad_enabled():
                # The recon aux always supervises the COMPOSITIONAL code (pre-LUT): it is the
                # anti-collapse anchor on the DeepSets encoder, and reconstructing a roster from a
                # free per-team code would be trivially satisfiable (zero pressure).
                self.last_zarch_recon_logits = self.zarch_encoder.recon_logits(self.last_zarch)
                self.last_zarch_species_ids = ctx.species_ids[:, :TEAM_SIZE].detach()
            # gen3_zarch_lut_v1: fold in the free per-team code. AFTER the recon read (above) so the
            # recon/VICReg aux keeps grading the compositional encoder, not the LUT.
            if self.zarch_lut != "off":
                idx = self._zarch_lut_index(ctx)
                self.last_zarch_lut_idx = idx
                code = self.zarch_lut_emb(idx)
                base = self.last_zarch if self.zarch_lut == "add" else torch.zeros_like(code)
                self.last_zarch = self.zarch_lut_norm(base + code)
        role_tokens = self.pokemon_encoder(ctx, self.embeddings)
        # In-place hidden-opponent belief: replace the un-revealed opp slots with distinct learned
        # unknown-mon tokens BEFORE the transformer, so the body refines them and every readout
        # attends over them as party members (flag-guarded; None ⇒ baseline zeros).
        if self.belief_slots is not None:
            role_tokens = self.belief_slots(role_tokens, ctx.opp_believed_mask)
        # Move belief (gen3_move_prefuse_v1, PRE-transformer variant): when move_belief_prefuse is on,
        # reinject the predicted opp moveset into the opp ROLE tokens BEFORE the transformer, so the
        # believed moves co-refine with the species/team belief through the 2 attention layers (instead
        # of being grafted on afterwards). The logits are stashed here; downstream consumers (damage op +
        # aux loss) read the same `last_move_belief_logits`. Default (off) leaves it None and runs the
        # POST-transformer reinjection below, byte-for-byte unchanged.
        self.last_move_belief_logits = None
        if self.move_belief is not None and self.move_belief_prefuse:
            opp_role, self.last_move_belief_logits = self._apply_move_belief(
                role_tokens[:, TEAM_SIZE:], ctx)
            role_tokens = torch.cat([role_tokens[:, :TEAM_SIZE], opp_role], dim=1)
        # gen3_damage_op_prefuse_v1: run the WHOLE physics stack ONCE, here, PRE-attention. The spread +
        # HP-type beliefs read the raw opp role tokens (the move belief already did, via prefuse), the FULL
        # DamageOperator runs on that belief, and its per-OUR-mon INCOMING rows are injected onto our role
        # tokens through the zero-init `prefuse_proj` — so attention reasons over the physics, then the SAME
        # block is concatenated to both heads below. This replaces the two-computation shape it supersedes
        # (a LEAN op inside the between-layers refine loop + the FULL op after the transformer); the two are
        # mutually exclusive (enforced in __init__). Off ⇒ prefused_damage stays None and the group runs at
        # its historical POST-transformer site, byte-for-byte.
        prefused_damage = None
        if self.damage_op_prefuse:
            opp_role, prefused_damage = self._spread_hp_damage(role_tokens[:, TEAM_SIZE:], ctx)
            inc = prefused_damage[:, :TEAM_SIZE * _DMG_PER_MON].reshape(
                ctx.batch_size, TEAM_SIZE, _DMG_PER_MON)                      # per-OUR-mon incoming rows
            role_tokens = torch.cat(
                [role_tokens[:, :TEAM_SIZE] + self.prefuse_proj(inc), opp_role], dim=1)  # residual (0 at init)
        # gen3_iterative_damage_v1: when refinement is on, recompute the lean discrete incoming damage from
        # the CURRENT (being-enriched) opp tokens BEFORE each of the first N transformer layers and inject it
        # as a residual onto our-mon token positions. The belief is re-read each round (move_belief.move_logits
        # — the per-round gradient sharpens it), the op computes the lean per-our-mon threat, and refine_proj
        # (zero-init → identity-at-init) carries it onto our tokens so the next layer attends over it. None
        # (off) ⇒ the transformer runs byte-identically to the baseline. Composes with prefuse: refine reads
        # whatever opp token the transformer currently holds (prefuse-enriched or not).
        refine_cb = None
        if (self.damage_refine_rounds > 0 and self.damage_op is not None
                and self.move_belief is not None):
            _opp_species_ids = ctx.species_ids[:, TEAM_SIZE:]                 # [B, 6] (prior-fusion lookup)
            _opp_move_ids = ctx.all_move_ids[:, TEAM_SIZE:, :]               # [B, 6, 4] (revealed → pinned)

            def refine_cb(tokens, layer_idx):
                if layer_idx >= self.damage_refine_rounds:
                    return tokens
                opp_tokens = tokens[:, TEAM_SIZE:2 * TEAM_SIZE, :]            # current (mid-transformer) opp
                # gen3_belief_single_compute_v1: FROZEN belief — reuse the pre-transformer posterior
                # instead of re-reading the head off the (being-enriched) tokens, so the belief is
                # computed once per forward and attention cannot revise it. The stash is live (not
                # detached), so the physics gradient still reaches the SAME belief computation the
                # reinjection used — one posterior, one gradient path. Default: re-read per round
                # (physics-in-the-loop), byte-for-byte unchanged.
                if self.move_belief_single_compute and self.last_move_belief_logits is not None:
                    logits = self.last_move_belief_logits                     # [B,6,M] pre-attention (typed)
                else:
                    logits = self.move_belief.move_logits(opp_tokens, _opp_species_ids, _opp_move_ids)  # [B,6,M]
                    # gen3_typed_hp_belief_v1: the re-read is RAW (237 live), so re-run the typed-HP
                    # step before the physics touches it — otherwise this path would price Hidden Power
                    # off a BP-0 typeless row while the head block priced it off real typed rows. Under
                    # 'composed' the type head is re-read from the SAME mid-transformer tokens (matching
                    # this mode's physics-in-the-loop intent; single-compute instead reuses the frozen
                    # pre-attention posterior); under 'flat' this is just the 237 mask.
                    logits, _, _, _ = self._typed_hp_posterior(opp_tokens, ctx, logits)
                # --- INCOMING residuals onto OUR tokens: damage (always) + status (v37, if on) ---
                # ONE candidate build per round, shared by the damage + status kernels (both select the
                # SAME detached top-K over the SAME `logits`; they used to rebuild it independently).
                cand = self.damage_op.refine_candidates(ctx, logits)
                inc = self.damage_op.discrete_incoming(ctx, logits, cand)    # [B,6,4] lean per-our-mon threat
                if self.last_refine_rounds is not None:                      # prober-only capture (axis A)
                    self.last_refine_rounds.append((layer_idx, logits.detach(), inc.detach()))
                refined_our = tokens[:, 0:TEAM_SIZE, :] + self.refine_proj(inc)   # residual (0 at init)
                if self.status_in_proj is not None:
                    # gen3_status_trunk_v1: "will I get statused" — one belief read feeds damage + status.
                    refined_our = refined_our + self.status_in_proj(
                        self.damage_op.discrete_incoming_status(ctx, logits, cand))
                # --- OUTGOING residuals onto OPP tokens: damage (#1/#2, if on) + status (v37, if on) ---
                refined_opp = opp_tokens
                if self.outgoing_proj is not None:
                    # gen3_bidir_threat_trunk_v1: when threat_unrevealed_outgoing + a belief head are on, read
                    # P(species) over the CURRENT opp tokens (gradient sharpens the species belief) so the
                    # unrevealed columns are priced by the expected-latent defender; else revealed-gated.
                    species_probs = None
                    if self.threat_unrevealed_outgoing and self.belief_head is not None:
                        species_probs = self.belief_head.species_posterior(opp_tokens)  # [B,6,n_species]
                    refined_opp = refined_opp + self.outgoing_proj(
                        self.damage_op.discrete_outgoing(ctx, species_probs))           # residual (0 at init)
                if self.status_out_proj is not None:
                    # gen3_status_trunk_v1: "can I status this opp mon" (revealed-gated).
                    refined_opp = refined_opp + self.status_out_proj(
                        self.damage_op.discrete_outgoing_status(ctx))
                if self.outgoing_proj is not None or self.status_out_proj is not None:
                    return torch.cat([refined_our, refined_opp, tokens[:, 2 * TEAM_SIZE:, :]], dim=1)
                return torch.cat([refined_our, tokens[:, TEAM_SIZE:, :]], dim=1)
        # gen3_entity_move_seats_v1 (v54, Stage 1): build the move ENTITY seats and enter them into
        # the trunk's attention. The E3 permutation (sorted-by-id → request order, by move-num
        # identity) happens HERE, pre-transformer — one permutation, shared by the seats and the
        # pointer head (which now reads the REFINED seats below). E4 gathers the op's pre-transformer
        # candidate weights + latents (`_entity_latent_table`, stashed by `_spread_hp_damage` — the
        # prefuse gate guarantees it ran). Seats append AFTER the global token, so every absolute
        # slice above (team/history/global, the refine callback's tail cat) is position-stable.
        _tok_req_raw, _move_valid = _request_order_move_tokens(
            self.pokemon_encoder.last_move_tokens, ctx)
        _seat_tokens, _seat_pad = self.entity_seats(
            _tok_req_raw, _move_valid, ctx, self.damage_op,
            self.last_move_belief_logits,
            getattr(self, "_entity_latent_table", None))
        our_team_out, their_team_out, _seat_out = self.team_transformer(
            role_tokens, ctx, self.embeddings, between_layers=refine_cb,
            extra=(_seat_tokens, self.entity_seats.seat_types(ctx.device), _seat_pad))
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
        # Move belief (default, POST-transformer): predict each opp slot's moveset and REINJECT it into the
        # slot token (flow-through) so the believed moves reach the CLS pools → both heads. Mode selects
        # which slots (revealed / unrevealed / both); the logits are stashed for the aux loss. Skipped when
        # move_belief_prefuse already did this reinjection BEFORE the transformer (handled above, which also
        # already set last_move_belief_logits — left at None when move_belief is off).
        if self.move_belief is not None and not self.move_belief_prefuse:
            their_team_out, self.last_move_belief_logits = self._apply_move_belief(their_team_out, ctx)
        # The spread + HP-type belief legs and the FULL damage op. By DEFAULT they run HERE, on the
        # attention-refined opp tokens (the op is placed BEFORE the CLS pools so the optional damage
        # re-attend can enrich the team tokens and the pools are derived ONCE, on the final tokens —
        # keeping EVERY downstream consumer on the same state). Under `damage_op_prefuse` the whole
        # group already ran PRE-transformer and `prefused_damage` holds its block.
        if self.damage_op_prefuse:
            damage_block = prefused_damage
        else:
            their_team_out, damage_block = self._spread_hp_damage(their_team_out, ctx)
        # damage_reattend (gen3_damage_reattend_v1): let attention reason OVER the computed physics. The op's
        # per-OUR-mon INCOMING rows are projected (small-init residual) onto the 6 our-team tokens, then ONE
        # more encoder layer re-attends the 12 team tokens (our↔opp) so each of our mons' representation is
        # contextualised by ITS incoming damage + the opp threats — and the pools below become damage-AWARE
        # board summaries instead of damage-blind ones. (BOARD-level enrichment; the per-CANDIDATE switch
        # scoring is now first-class regardless — the pointer head reads the final our_team_out per token,
        # so a re-attended bench token flows straight into its own switch logit. gen3_pointer_native_v1.)
        # Identity-at-init
        # (the layer's output paths are zero-init'd in __init__) ⇒ ON starts ≈ the damage_op baseline; the
        # damage signal grows with training. No-op when off (modules None).
        if self.reattend_layer is not None and damage_block is not None:
            inc = damage_block[:, :TEAM_SIZE * _DMG_PER_MON].reshape(
                ctx.batch_size, TEAM_SIZE, _DMG_PER_MON)                          # per-OUR-mon incoming rows
            our_team_out = self.reattend_norm(our_team_out + self.reattend_proj(inc))
            tok = torch.cat([our_team_out, their_team_out], dim=1)                # [B, 12, D_MODEL]
            tok = self.reattend_layer(tok, src_key_padding_mask=ctx.all_fainted)  # our↔opp re-attention
            our_team_out, their_team_out = tok[:, :TEAM_SIZE], tok[:, TEAM_SIZE:]
        # CLS pools — derived ONCE, on the final (possibly damage-re-attended) team tokens, so the policy
        # pools, the value pool, and the side/aux readouts below ALL reflect the same state.
        our_team_pooled, their_team_pooled, our_active_refined, value_pooled = self.cls_pool(
            our_team_out, their_team_out, ctx
        )
        # gen3_pointer_native_v1 / gen3_entity_move_seats_v1: stash the pointer action head's
        # PER-ENTITY inputs for `Gen3DualHeadMaskablePolicy._get_action_dist_from_latent` — the head
        # itself lives on the policy (its ctx is latent_pi, which doesn't exist here). Move logit k
        # now reads the REFINED E3 seat k (post-attention, d_model-wide — the Stage-1 payoff: the
        # token was refined IN the trunk alongside the board, not just inside PokemonEncoder). The
        # request-order permutation happened ONCE, pre-transformer, at the seat build — order is
        # seat-stable through attention, so seat k is still action logit 6+k; `_move_valid` gates
        # unresolved slots to logit 0 exactly as before (their refined content is attention noise the
        # head never scores). Switch scorer j reads the same (possibly re-attended) board-aware team
        # token every pool reads; the op cells are the same post-gain numbers the projection heads
        # consume (width-0 when the op is off — the head's Linears are built correspondingly
        # narrower, never silently zero-padded).
        _tok_req = _seat_out[:, :4, :]
        if self.damage_op is not None and damage_block is not None:
            _mcells, _scells = self.damage_op.pointer_cells(damage_block)
        else:
            _mcells = _tok_req.new_zeros(ctx.batch_size, _tok_req.shape[1], 0)
            _scells = our_team_out.new_zeros(ctx.batch_size, TEAM_SIZE, 0)
        self.last_pointer_inputs = (_tok_req, _move_valid, our_team_out, _mcells, _scells)
        # Read-only stash of the value-CLS pool (the critic's whole-board "who's winning" summary, the
        # 128-dim FitNets HINT layer). Consumed ONLY by the FitNets value-feature distillation
        # (`instrumented_ppo._value_feat_distill`): both student and teacher forwards leave it here, so the
        # distill loop can regress the student's value_pooled toward each teacher's on the teacher-team
        # states. NOT read by the forward → off-path/eval is byte-identical; carries grad on the student pass
        # (a live activation) so the cosine distill gradient flows into the shared trunk.
        self.last_value_pooled = value_pooled
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
        # Auxiliary PUBLIC-VALUE readout (flag-guarded; None when off). Same value_pooled read as the
        # win head → a [B,1] V_pub logit stashed for the aux loss (regressed toward the frozen
        # human-replay-calibrated public value riding the training-only `pubval_target` obs key). NOT
        # fed into the assembler (a side readout). `read_only` feeds a STOP-GRAD value_pooled;
        # `shaping` lets the human positional prior shape the shared trunk.
        if self.pubval_head is not None:
            pv_in = value_pooled if self.pubval_mode == "shaping" else value_pooled.detach()
            self.last_pubval_logits = self.pubval_head(pv_in)
        else:
            self.last_pubval_logits = None
        # Distributional VALUE readout (flag-guarded; None when off). Same value_pooled the win head
        # reads → per-atom return-distribution logits, stashed for the aux loss + prober/eval. NOT fed
        # into the assembler (a side readout — the value target can't leak into pi/vf). `read_only`
        # feeds a STOP-GRAD value_pooled (head-only training); `shaping` feeds it live. Computed on
        # every forward (one small MLP) so eval/inference can read the distribution too.
        if self.value_dist_head is not None:
            vd_in = value_pooled if self.value_dist_mode == "shaping" else value_pooled.detach()
            self.last_value_dist_logits = self.value_dist_head(vd_in)
        else:
            self.last_value_dist_logits = None
        belief = None
        if self.hidden_opp_belief is not None:
            # Same 12-token memory + the single-sourced ctx.all_fainted key-mask the value CLS pools
            # over (all_team_out is a forward activation, cheap to recompute; the MASK carries the
            # NaN-safety invariant and is single-sourced on the context).
            all_team_out = torch.cat([our_team_out, their_team_out], dim=1)                 # [B, 12, D]
            belief = self.hidden_opp_belief(all_team_out, ctx.all_fainted, ctx.batch_size)
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
        pi_pre = self.projection(self.pre_proj_norm(pi_combined))
        vf_pre = self.value_projection(self.value_pre_norm(vf_combined))
        # gen3_zarch_film_v1: FiLM the head features on the team-archetype latent — POST-projection,
        # PRE-ReLU (after the LayerNorm+Linear so the norm can't wash the per-feature scale out;
        # before the activation, the standard FiLM site). `h·(1+Δγ) + Δβ` with zero-init generators
        # ⇒ exact identity at init; each head has its own generator (value is archetype-conditional
        # in its own way — the same board is "winning" for stall, "losing" for offense).
        if self.zarch_film == "heads":
            dg_pi, db_pi = self.film_pi(self.last_zarch).chunk(2, dim=-1)
            dg_vf, db_vf = self.film_vf(self.last_zarch).chunk(2, dim=-1)
            pi_pre = pi_pre * (1.0 + dg_pi) + db_pi
            vf_pre = vf_pre * (1.0 + dg_vf) + db_vf
        pi_features = self.activation(pi_pre)
        vf_features = self.activation(vf_pre)
        return pi_features, vf_features
