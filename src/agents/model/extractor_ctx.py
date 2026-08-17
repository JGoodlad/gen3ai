"""Phase-0 plumbing: the per-forward ExtractorContext, obs unpacking, token-type ids, and the shared embedding tables.

Split out of `features_extractor.py` 2026-08-16 (one responsibility per file); that module
re-exports every name here, so historical import paths still resolve.
"""
import torch
from dataclasses import dataclass
from typing import Dict, Any, Optional, NamedTuple, TYPE_CHECKING
from agents.observation.constants import (
    POKEMON_LAST_ACTION_OFFSET,
    TEAM_SIZE,
    POKEMON_HP_PROBS_OFFSET,
    POKEMON_SPECIES_KNOWN_OFFSET,
)
from agents.observation.turn_delta_encoder import (
    TURN_DELTA_DIM,
    EFF_DIM,
    ORDER_DIM,
    TURN_DELTA_EMBEDDED_IDS,
    TURN_DELTA_SCALAR_OFFSETS,
)

# Strategic TurnDelta slice: always the tail of the TurnDelta block (effectiveness + order).
# Kept exported because external tests reference these constants.


TD_STRATEGIC_DIM    = EFF_DIM * 2 + ORDER_DIM      # 10: our_eff(4) + opp_eff(4) + order(2)
TD_STRATEGIC_OFFSET = TURN_DELTA_DIM - TD_STRATEGIC_DIM  # 29

# Token group ids for the unified transformer's type embedding.
TOKEN_TYPE_OUR_TEAM = 0
TOKEN_TYPE_THEIR_TEAM = 1
TOKEN_TYPE_HISTORY = 2
TOKEN_TYPE_GLOBAL = 3
# gen3_entity_move_seats_v1 (v54, Stage 1 of the entity generation): move ENTITY seats in the trunk.
TOKEN_TYPE_OUR_MOVE = 4        # E3 — our active's 4 request-ordered move tokens
TOKEN_TYPE_THEIR_THREAT = 5    # E4 — the opp active's top-K believed threat-move tokens
NUM_TOKEN_TYPES = 6


class PointerInputs(NamedTuple):
    """The pointer action head's per-forward inputs (`gen3_op_stashes_v1`, extractor half):
    request-ordered move seat tokens + validity, the refined team tokens, and the two
    per-action cell blocks. A NamedTuple so `Gen3DualHeadMaskablePolicy`'s unpack keeps
    working positionally while every consumer can name what it reads."""
    move_tokens: torch.Tensor        # [B, 4, D_MODEL] refined E3 seats, request order
    move_valid: torch.Tensor         # [B, 4]
    team_tokens: torch.Tensor        # [B, 6, D_MODEL] our refined team tokens
    move_cells: torch.Tensor         # [B, 4, move_cell_dim]
    switch_cells: torch.Tensor       # [B, 6, switch_cell_dim]


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
    embedded: int = sum(dim_by_kind[kind] for _, kind in TURN_DELTA_EMBEDDED_IDS)
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

    # Tier H-A1 (gen3_pair_history_v1): the last-action MOVE ID is an embedding id living inside
    # the raw tail — extract it for the move table and ZERO its column in the raw path (a raw
    # dex num must never reach a Linear; the manifest rule). The clone is [B, N, tail] — cheap —
    # and keeps hp_and_active's load-bearing conventions ([..., 0] = hp, [..., -1] = active).
    last_move_ids = pokemon_part[:, :, POKEMON_LAST_ACTION_OFFSET].long()                   # [B, N]
    hp_and_active = hp_and_active.clone()
    hp_and_active[:, :, POKEMON_LAST_ACTION_OFFSET - hp_offset] = 0.0

    return {
        "species_ids": species_ids, "all_move_ids": all_move_ids,
        "all_move_type_ids": all_move_type_ids, "item_ids": item_ids,
        "ability1_ids": ability1_ids, "ability2_ids": ability2_ids,
        "type1_ids": type1_ids, "type2_ids": type2_ids,
        "hp_probs": hp_probs, "hp_and_active": hp_and_active,
        "last_move_ids": last_move_ids,
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
    # Tier H-A1 (gen3_pair_history_v1): the per-mon LAST-ACTION move id [B, 12] — an EMBEDDING id
    # (nonzero only on active slots; 0 = none/switch). Sliced out of the raw tail and routed
    # through the move table; its raw column inside hp_and_active is ZEROED at the slice so the
    # id can never reach a Linear (the embedded-ID manifest rule).
    last_move_ids: torch.Tensor
    # Reactive / global feature slices. (gen3_entity_rehome_v1: matchups_all and
    # struggle_feature are GONE with their obs blocks — the D/V edges own pair physics.)
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
    # Per-opp-slot ADDRESSABILITY [B, 6] bool — "is this slot a legal object to point at":
    # alive-and-revealed OR still-unrevealed. Single-sourced here because `hp == 0` means
    # UNKNOWN, not dead, on an unrevealed opp slot (measured: 1033/1033 unrevealed slots
    # encode hp exactly 0.0), and deriving aliveness from hp at a consumer silently deletes
    # every hidden mon from its candidate set — the bug beta's mask shipped with. Consumers
    # with the *addressable-candidate* semantic (beta; any future intent/target head) read
    # THIS; the op's `alive_j = hp>0` gates are deliberately revealed-only (unrevealed
    # defenders go through the usage-prior path) and must NOT migrate to it.
    opp_addressable: torch.Tensor
    # Combined 12-token key-mask (= cat[ours, opp]), single-sourced here so the value-CLS pool and
    # the hidden-opp belief share ONE mask — they rely on the same "both actives force-unmasked ⇒ no
    # all-True row" NaN-safety invariant, which must not be able to drift between two call sites.
    all_fainted: torch.Tensor
    # Transformer / projection inputs.
    turn_history_raw: torch.Tensor
    our_ctx_raw: torch.Tensor
    opp_ctx_raw: torch.Tensor
    non_matchup_rest: torch.Tensor
    # Tier H-A2 (gen3_pair_history_v1): the pair-history block [B, 6, 6, 5] in OBS order
    # (opp mon i, our mon j, cell) — the `h` edge family permutes to the (our, opp) block
    # convention at the call site. None on pre-pair layouts (never in production).
    pair_history: "Optional[torch.Tensor]" = None
    # gen3_event_window_v1 (Tier H-B): the [B, EVENT_WINDOW_N, EVENT_TOKEN_DIM] typed
    # event-record block (None on a pre-event layout). Ids inside are embedding ids; ONLY the
    # flag-gated EventSeats consumer reads it — no Linear touches it raw.
    event_window: "Optional[torch.Tensor]" = None


class Embeddings(torch.nn.Module):
    """Sole owner of the embedding tables, shared between per-Pokémon encoding and
    turn-history encoding. Passed as a forward-time argument to the phases that need
    it so the tables register exactly once in the state_dict."""

    if TYPE_CHECKING:
        # Registered buffers exist only dynamically, so `Module.__getattr__` types them `Any`.
        # Declaring them keeps every read a `Tensor`. TYPE_CHECKING-only: no runtime effect.
        hp_type_idx_map: torch.Tensor
        _td_scalar_idx: torch.Tensor

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
        return hp_probs @ hp_type_rows  # type: ignore[no-any-return]

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
        # Stage-3 generator half (gen3ai roadmap §3): every block boundary this phase reads comes
        # from the declarative schema's validated slice map — ONE source, tiling-proven at
        # construction (build_schema().validate() throws on any gap/overlap/total mismatch), so a
        # layout drift crashes here instead of silently mis-slicing a consumer. The layout dict
        # stays for sub-structure the schema deliberately doesn't model (per-mon slot internals via
        # slice_pokemon_categoricals, move-slot counts).
        from agents.observation.schema import build_schema
        self._slices = build_schema(layout).slices()
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
        sl = self._slices
        x = obs["observation"]
        batch_size = x.shape[0]

        # num_moves drives the prev-mask + matchup slices below; the per-mon move-slot slicing moved to
        # slice_pokemon_categoricals (so moves_info/moves_layout/m_slot_layout are no longer needed here).
        num_moves = len(layout['pokemon']['moves']['layout']['slots'])

        # prev_mask (ACTION_SPACE_SIZE) + turn-history block (N * TURN_DELTA_DIM) from obs tail.
        prev_mask = x[:, sl['prev_action_mask']]
        turn_history_raw = x[:, sl['turn_history']]
        switch_mask  = prev_mask[:, 0:TEAM_SIZE]
        move_mask    = prev_mask[:, TEAM_SIZE:TEAM_SIZE + num_moves]
        struggle_mask = prev_mask[:, TEAM_SIZE + num_moves:TEAM_SIZE + num_moves + 1]

        # Team blocks (per-mon slot count/width derive from the schema block itself).
        _team = sl['our_team']
        mon_dim = (_team.stop - _team.start) // TEAM_SIZE
        our_team_raw = x[:, sl['our_team']].reshape(batch_size, TEAM_SIZE, mon_dim)
        opp_team_raw = x[:, sl['opp_team']].reshape(batch_size, TEAM_SIZE, mon_dim)

        # Global env feature slices.
        weather_feature = x[:, sl['global_env.weather']]
        spikes_feature  = x[:, sl['global_env.hazards']]
        turn_feature    = x[:, sl['global_env.clock']]
        screen_feature  = x[:, sl['global_env.screens']]

        # Reactive/board feature slices (gen3_entity_rehome_v1: the matchup matrices and the
        # forced_struggle bit no longer exist — the D/V edges own pair physics, and struggle
        # legality lives in the action mask + the all-zero req-move legal bits).
        fainted_feature = x[:, sl['reactive.fainted']]

        # gen3_op_move_align_v1: OUR active's request-order move ids/types/legality (after the matchups,
        # so non_matchup_rest — which stops at the matchup offset — never sees these embedding IDs).
        # ids/type_ids → long for the op's table lookups; legal stays float (a 0/1 gate).
        _arm = sl['reactive.active_req_moves']
        _arm_per = (_arm.stop - _arm.start) // 3   # [ids ×4, type_ids ×4, legal ×4]
        our_active_req_move_ids = x[:, _arm.start : _arm.start + _arm_per].long()
        our_active_req_move_type_ids = x[:, _arm.start + _arm_per : _arm.start + 2 * _arm_per].long()
        our_active_req_move_legal = x[:, _arm.start + 2 * _arm_per : _arm.stop]

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
        last_move_ids = _ids["last_move_ids"]

        # Tier H-A2 (gen3_pair_history_v1): the pair-history block, reshaped to its layout order
        # (opp mon i, our mon j, cell). The schema names the block, so a layout without it (never
        # in production) simply has no slice and the field stays None.
        pair_history = None
        if 'pair_history' in sl:
            _ph = sl['pair_history']
            pair_history = x[:, _ph].reshape(batch_size, TEAM_SIZE, TEAM_SIZE, -1)
        event_window = None
        if 'event_window' in sl:
            _ew = sl['event_window']
            event_window = x[:, _ew].reshape(
                batch_size, self.layout['event_window_n'], self.layout['event_token_dim'])

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
        # Per-opp-slot addressability (see the ExtractorContext field doc): alive-and-revealed
        # OR unrevealed. "A Pokemon cannot faint without being revealed, so an unrevealed mon
        # is alive" — the exactness argument from the beta-mask fix, now single-sourced.
        _opp_hp = hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, 0]
        opp_addressable = (_opp_hp > 0.0) | opp_believed_mask
        # Combined 12-token key-mask, single-sourced (both actives are now force-unmasked, so it can
        # never be all-True per row — the NaN-safety invariant the value-CLS pool + belief both rely on).
        all_fainted = torch.cat([fainted_mask_ours, fainted_mask_opp], dim=1)

        # Active contexts + non-matchup scalar tail (transformer global token + projection).
        _ctx = sl['active_context']
        active_ctx_dim = (_ctx.stop - _ctx.start) // 2
        our_ctx_raw = x[:, _ctx.start : _ctx.start + active_ctx_dim]
        opp_ctx_raw = x[:, _ctx.start + active_ctx_dim : _ctx.stop]
        # Everything between the active contexts and the embedding-ID active_req_moves tail:
        # the global-env block + the 5 raw board scalars (this raw-scalar span never contains
        # an ID by construction).
        non_matchup_rest = x[:, sl['global_env'].start : sl['reactive.active_req_moves'].start]
        return ExtractorContext(
            batch_size=batch_size, device=x.device,
            pokemon_part=pokemon_part,
            species_ids=species_ids, all_move_ids=all_move_ids, all_move_type_ids=all_move_type_ids,
            item_ids=item_ids, ability1_ids=ability1_ids, ability2_ids=ability2_ids,
            type1_ids=type1_ids, type2_ids=type2_ids,
            hp_probs=hp_probs, hp_and_active=hp_and_active,
            last_move_ids=last_move_ids,
            move_mask=move_mask, switch_mask=switch_mask, struggle_mask=struggle_mask,
            our_active_req_move_ids=our_active_req_move_ids,
            our_active_req_move_type_ids=our_active_req_move_type_ids,
            our_active_req_move_legal=our_active_req_move_legal,
            turn_feature=turn_feature, weather_feature=weather_feature, fainted_feature=fainted_feature,
            spikes_feature=spikes_feature, screen_feature=screen_feature,
            our_active_idx=our_active_idx, opp_active_local=opp_active_local,
            fainted_mask_ours=fainted_mask_ours, fainted_mask_opp=fainted_mask_opp,
            opp_believed_mask=opp_believed_mask,
            opp_addressable=opp_addressable,
            all_fainted=all_fainted,
            turn_history_raw=turn_history_raw,
            our_ctx_raw=our_ctx_raw, opp_ctx_raw=opp_ctx_raw, non_matchup_rest=non_matchup_rest,
            pair_history=pair_history,
            event_window=event_window,
        )
