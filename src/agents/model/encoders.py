"""Per-entity encoders: the move-latent encoder and the per-mon role encoder.

Split out of `features_extractor.py` 2026-08-16 (one responsibility per file); that module
re-exports every name here, so historical import paths still resolve.
"""
from agents.model.extractor_ctx import Embeddings, ExtractorContext
import torch
from typing import Dict, Any, Optional
from agents.observation.constants import (
    TEAM_SIZE,
    POKEMON_FULL_DIM,
)
from agents.observation.moves import HIDDEN_POWER_MOVE_NUM
# The LEGAL-BUT-UNOBSERVED move-prior base (the `--move-candidate-floor` default). Legality itself is
# unconditional; this is only the height of the liftable base a legal-unobserved move starts from.

# Strategic TurnDelta slice: always the tail of the TurnDelta block (effectiveness + order).
# Kept exported because external tests reference these constants.
from agents.model.arch_constants import (ROLE_TOKEN_SIZE,
    MOVE_NET_HIDDEN,
    MOVE_LATENT_HIDDEN,
    MOVE_LATENT_DIM,
    ROLE_ENCODER_HIDDEN,
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
        _move_ctx_dim = (1 + _gl['clock']['dim']     # hp + CLOCK (gen3_deadline_clock_v1: 3, not 1 —
                         #                             read the layout, never hardcode the width)
                         + _gl['weather']['dim']     # weather
                         + _rl['fainted']['dim']     # fainted
                         + _gl['hazards']['dim'])    # spikes
        HP_PROBS_DIM = 16
        move_input_dim = (layout['move_embedding_dim'] + layout['type_embedding_dim']
                          + self.move_remnant_dim + 1               # remnants + known
                          + _move_ctx_dim                           # context
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
            _gl['clock']['dim']        # clock (gen3_deadline_clock_v1: log-elapsed + 2 remaining)
            + _gl['weather']['dim']
            + _rl['fainted']['dim']
            + _gl['hazards']['dim']
            + _gl['screens']['dim']
        )
        # gen3_entity_rehome_v1 (E2 injection): each ACTIVE mon's token carries its own side's
        # boosts+volatiles block — the §6-audited entity home for the active context (bench slots
        # read zeros; the block previously reached the model only through the global token and
        # the two projections, which remain — this is additive delivery, not a re-route).
        self._active_ctx_dim = layout['active_context_dim']
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
            + layout['move_embedding_dim']              # H-A1: embedded LAST-ACTION move (active slots)
            + self._active_ctx_dim                      # E2: own side's active ctx (active slot only)
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

        # gen3_entity_rehome_v1: the 288-dim CPU matchup matrices (and their per-cell validity
        # mask) are DELETED — the D/V edge families deliver a strict superset of this signal as
        # attention biases + op cells from real physics and the learned belief.
        move_feature_blocks = [
            embedded_moves,
            embedded_move_types,
            move_remnants_reshaped,
            known_flags_reshaped,
            move_context_final,
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

        # Tier H-A1 (gen3_pair_history_v1): the side's last-action move, embedded through the
        # SAME move table (id 0 — none/switch/bench — embeds like any padding id; the raw column
        # was zeroed at the slice, so this is the id's ONLY route into the network).
        embedded_last_move = embeddings.move_embedding(ctx.last_move_ids)                    # [B, 12, move_emb]

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
            embedded_last_move,
        ], dim=2)

        # --- Role encoder ---
        global_context = torch.cat([
            ctx.turn_feature, ctx.weather_feature, ctx.fainted_feature,
            ctx.spikes_feature, ctx.screen_feature,
        ], dim=1)
        context_broadcasted = global_context.unsqueeze(1).expand(-1, n_poke, -1)

        switch_validity_ours = ctx.switch_mask.unsqueeze(2)
        switch_validity_opp  = torch.ones(batch_size, TEAM_SIZE, 1, device=ctx.device)
        switch_validity = torch.cat([switch_validity_ours, switch_validity_opp], dim=1)

        struggle_from_prev = ctx.struggle_mask.unsqueeze(1).expand(-1, n_poke, -1)

        # gen3_entity_rehome_v1 (E2 injection): scatter each side's active-context block onto its
        # ACTIVE mon's row — the entity owns its own boosts/volatiles (bench rows stay zero).
        batch_idx = torch.arange(batch_size, device=ctx.device)
        active_ctx_inject = torch.zeros(
            batch_size, n_poke, self._active_ctx_dim,
            device=ctx.device, dtype=pokemon_enriched.dtype)
        active_ctx_inject[batch_idx, ctx.our_active_idx] = ctx.our_ctx_raw
        active_ctx_inject[batch_idx, TEAM_SIZE + ctx.opp_active_local] = ctx.opp_ctx_raw

        pokemon_enriched_with_context = torch.cat([
            pokemon_enriched, active_ctx_inject, context_broadcasted, switch_validity,
            struggle_from_prev
        ], dim=2)

        role_tokens = self.role_encoder(
            pokemon_enriched_with_context.reshape(-1, pokemon_enriched_with_context.shape[-1])
        )
        return role_tokens.reshape(batch_size, n_poke, self.role_token_size)   # [B, 12, 128]
