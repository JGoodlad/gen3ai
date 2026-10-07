"""STATIC Pokémon tokens (`gen3_static_tokens_v1`, `--token-encoding static`;
`designs/endstate/design_static_tokens.md`). `legacy` (the default, production) never builds anything here.

A Pokémon's token is the SUM of two parts, and NO board fact enters either:

* **S, the static identity** — a pure function of the mon's SET fields: species, the six level-100 ACTUAL
  stats (exact from the observed spread for our team; the Smogon usage-weighted mean ± std for an opponent,
  ``STAT_PRIOR`` = `belief_tables.build_static_stat_prior`), types, item (id + known), abilities (ids +
  dominance + known), the four moves through the move network and the within-mon self-attention, SUMMED as
  a set (Deep Sets; audit F14), species_known and the Hidden Power block;
* **D, the dynamic state** — what is happening TO the mon: HP, status, the counters, the sleep-wake belief,
  recency, protect odds, its last action, the trap bits, the active flag, the item-consumed bit, its side's
  boosts + volatiles when it is the active (bench rows zero), and a per-move pool
  ``Σ_k ReLU(W [m_k ; pp_k ; legal_k])`` (``m_k`` the move's STATIC token).

The clock, weather, faint counts, Spikes and screens (legacy's per-mon context, audit F4) are read by
NEITHER: they reach a token through the trunk's attention (the global token) and the damage operator.

Because a hidden opponent slot's X5 hypothesis row is its species' DEX ROW (never the active), its whole
token is a function of the species: `static_hypothesis_tokens` encodes the dex table ONCE per forward and
GATHERS (exact — the same function on the same row; no split as `hypothesis_encode` needs for legacy).

The module keeps `PokemonEncoder`'s attribute contract (``layout``, ``num_moves``, ``move_latent``,
``move_latent_encoder``, ``move_network``, ``move_self_attn``, ``move_self_norm``, ``role_encoder``,
``last_move_tokens``) so it is a drop-in at ``extractor.pokemon_encoder``.
"""
from __future__ import annotations

from typing import Any, Dict, NamedTuple, Optional, Tuple

import torch

from agents.model.arch_constants import (MOVE_LATENT_DIM, MOVE_NET_HIDDEN, ROLE_ENCODER_HIDDEN,
                                         ROLE_TOKEN_SIZE, STATIC_DYN_HIDDEN, STATIC_MOVE_POOL_DIM,
                                         STATIC_STAT_SCALE)
from agents.model.belief_tables import build_static_stat_prior
from agents.model.encoders import MoveLatentEncoder
from agents.model.extractor_ctx import Embeddings, ExtractorContext, slice_pokemon_categoricals
from agents.observation.constants import (CONDITION_DIM, POKEMON_ACTIVE_OFFSET, POKEMON_CONDITION_OFFSET,
                                          POKEMON_COUNTER_DIM, POKEMON_COUNTER_OFFSET, POKEMON_HP_BLOCK_DIM,
                                          POKEMON_HP_BLOCK_OFFSET, POKEMON_HP_OFFSET,
                                          POKEMON_LAST_ACTION_DIM, POKEMON_LAST_ACTION_OFFSET,
                                          POKEMON_MAYBE_TRAPPED_OFFSET, POKEMON_PROTECT_OFFSET,
                                          POKEMON_RECENCY_DIM, POKEMON_RECENCY_OFFSET,
                                          POKEMON_SLEEP_BELIEF_DIM, POKEMON_SLEEP_BELIEF_OFFSET,
                                          POKEMON_SPECIES_KNOWN_OFFSET, POKEMON_SPECIES_OFFSET,
                                          POKEMON_SPREAD_OFFSET, POKEMON_TRAPPED_OFFSET, TEAM_SIZE)
from agents.observation.moves import HIDDEN_POWER_MOVE_NUM

#: The `--token-encoding` modes. ``legacy`` is the OFF state: `PokemonEncoder`, byte-identical.
TOKEN_ENCODING_MODES = ("legacy", "static")

#: The base-stat columns of the per-mon slot are ``base / 255`` (`observation/species.py`).
_BASE_STAT_DENOM = 255.0
#: Level-100 stat offsets: HP = 2B + IV + EV/4 + 110; any other = (2B + IV + EV/4 + 5) × nature.
_HP_OFFSET_L100, _STAT_OFFSET_L100 = 110.0, 5.0
#: The stat input S reads: 6 means + 6 stds + spread_known.
STAT_FEATURE_DIM = 13
#: D's input, by part (the docstring's list): hp 1 · status · counters · sleep belief · recency · protect 1 ·
#: last action (its move embedded, + the 5 non-id columns) · trapped + maybe-trapped 2 · active 1 ·
#: item consumed 1 · the active context · the per-move pool.
_LAST_ACTION_BITS = POKEMON_LAST_ACTION_DIM - 1


class StaticParts(NamedTuple):
    """One encoder pass, by part (the tests read S and D separately)."""
    tokens: torch.Tensor        # [B,N,ROLE_TOKEN_SIZE] = static + dynamic
    static: torch.Tensor        # [B,N,ROLE_TOKEN_SIZE] S
    dynamic: torch.Tensor       # [B,N,ROLE_TOKEN_SIZE] D
    move_tokens: torch.Tensor   # [B,N,4,MOVE_NET_HIDDEN[1]] per-move: static move token + its own state


class StaticTokenEncoder(torch.nn.Module):
    """`--token-encoding static`'s per-mon encoder (module docstring). Drop-in for `PokemonEncoder`."""

    STAT_PRIOR: torch.Tensor

    def __init__(self, layout: Dict[str, Any], move_latent: bool = False):
        super().__init__()
        self.last_move_tokens: Optional[torch.Tensor] = None
        self.layout = layout
        pk = layout['pokemon']
        msl = pk['moves']['layout']['slot_layout']
        self.num_moves = len(pk['moves']['layout']['slots'])
        self._msl = msl
        self._moves_offset = pk['moves']['offset']
        self._slot_offsets = [s['offset'] for s in pk['moves']['layout']['slots']]
        # The STATIC move columns: power + secondary + recoil · category · max PP · accuracy + never-miss.
        self.move_static_remnant_dim = ((msl['type']['offset'] - msl['power']['offset'])
                                        + (msl['known']['offset'] - (msl['type']['offset'] + msl['type']['dim']))
                                        + msl['max_pp']['dim']
                                        + (msl['never_miss']['offset'] + msl['never_miss']['dim']
                                           - msl['accuracy']['offset']))
        hp_probs_dim = POKEMON_HP_BLOCK_DIM - 1
        move_input_dim = (layout['move_embedding_dim'] + layout['type_embedding_dim']
                          + self.move_static_remnant_dim + 1        # + known
                          + hp_probs_dim)                           # the HP candidate types (HP slot only)
        self.move_latent = move_latent
        if move_latent:
            self.move_latent_encoder = MoveLatentEncoder(layout)
            move_input_dim += MOVE_LATENT_DIM
        self.move_network = torch.nn.Sequential(
            torch.nn.Linear(move_input_dim, MOVE_NET_HIDDEN[0]), torch.nn.ReLU(),
            torch.nn.Linear(MOVE_NET_HIDDEN[0], MOVE_NET_HIDDEN[1]))
        self.move_self_attn = torch.nn.MultiheadAttention(embed_dim=MOVE_NET_HIDDEN[1], num_heads=2,
                                                          batch_first=True)
        self.move_self_norm = torch.nn.LayerNorm(MOVE_NET_HIDDEN[1])

        items = pk['items']
        abil = pk['abilities']
        self._item_known = items['offset'] + items['layout']['known']['offset']
        self._item_consumed = items['offset'] + items['layout']['consumed']['offset']
        self._ab_dom = abil['offset'] + abil['layout']['dominance']['offset']
        self._ab_known = abil['offset'] + abil['layout']['known']['offset']
        role_input_dim = (layout['species_embedding_dim'] + STAT_FEATURE_DIM
                          + layout['item_embedding_dim'] + 1                 # item + known
                          + 2 * layout['type_embedding_dim']
                          + 2 * layout['ability_embedding_dim'] + 2          # + dominance + known
                          + MOVE_NET_HIDDEN[1]                               # the SET pool
                          + 1                                                # species_known
                          + POKEMON_HP_BLOCK_DIM)
        self.role_token_size = ROLE_TOKEN_SIZE
        self.role_encoder = torch.nn.Sequential(
            torch.nn.Linear(role_input_dim, ROLE_ENCODER_HIDDEN[0]), torch.nn.ReLU(),
            torch.nn.Linear(ROLE_ENCODER_HIDDEN[0], ROLE_TOKEN_SIZE))

        # D: the per-move state (PP, legality) — onto each move token, and pooled (tied to the move) into D.
        self.move_dyn_proj = torch.nn.Linear(2, MOVE_NET_HIDDEN[1])
        self.move_dyn_pool = torch.nn.Linear(MOVE_NET_HIDDEN[1] + 2, STATIC_MOVE_POOL_DIM)
        self._active_ctx_dim = layout['active_context_dim']
        dyn_input_dim = (1 + CONDITION_DIM + POKEMON_COUNTER_DIM + POKEMON_SLEEP_BELIEF_DIM
                         + POKEMON_RECENCY_DIM + 1
                         + layout['move_embedding_dim'] + _LAST_ACTION_BITS
                         + 2 + 1 + 1                                          # trap bits · active · consumed
                         + self._active_ctx_dim + STATIC_MOVE_POOL_DIM)
        self.dynamic_encoder = torch.nn.Sequential(
            torch.nn.Linear(dyn_input_dim, STATIC_DYN_HIDDEN), torch.nn.ReLU(),
            torch.nn.Linear(STATIC_DYN_HIDDEN, ROLE_TOKEN_SIZE))
        self.register_buffer("STAT_PRIOR", build_static_stat_prior(int(layout['max_species'])), persistent=False)

    # ------------------------------------------------------------------------------------------ parts
    def _actual_stats(self, pp: torch.Tensor, species_ids: torch.Tensor) -> torch.Tensor:
        """[...,13] the six level-100 stats (/ STATIC_STAT_SCALE), their std, spread_known. Known spread
        (our team): the smooth formula on the observed IV / EV / nature (std 0). Unknown (the opponent): the
        Smogon prior row of the species."""
        base = pp[..., POKEMON_SPECIES_OFFSET + 1:POKEMON_SPECIES_OFFSET + 7] * _BASE_STAT_DENOM
        sp = pp[..., POKEMON_SPREAD_OFFSET:POKEMON_SPREAD_OFFSET + 18]
        iv, ev, known, nat = sp[..., 0:6] * 31.0, sp[..., 6:12] * 252.0, sp[..., 12:13], sp[..., 13:18]
        pre = 2.0 * base + iv + ev / 4.0
        hp = pre[..., 0:1] + _HP_OFFSET_L100
        rest = (pre[..., 1:] + _STAT_OFFSET_L100) * nat
        exact = torch.cat([hp, rest], dim=-1)
        prior = self.STAT_PRIOR[species_ids]                                   # [...,6,2]
        is_known = known > 0.5
        mean = torch.where(is_known, exact, prior[..., 0])
        std = torch.where(is_known, torch.zeros((), dtype=mean.dtype, device=mean.device), prior[..., 1])
        return torch.cat([mean / STATIC_STAT_SCALE, std / STATIC_STAT_SCALE, known], dim=-1)

    def _move_columns(self, pp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per move: the static remnants [...,4,R], the known bit [...,4,1], current PP [...,4,1]."""
        msl, mo = self._msl, self._moves_offset
        rem, known, cur = [], [], []
        for off in self._slot_offsets:
            s0 = mo + off
            rem.append(torch.cat([
                pp[..., s0 + msl['power']['offset']:s0 + msl['type']['offset']],
                pp[..., s0 + msl['type']['offset'] + msl['type']['dim']:s0 + msl['known']['offset']],
                pp[..., s0 + msl['max_pp']['offset']:s0 + msl['max_pp']['offset'] + msl['max_pp']['dim']],
                pp[..., s0 + msl['accuracy']['offset']:s0 + msl['never_miss']['offset'] + msl['never_miss']['dim']],
            ], dim=-1))
            known.append(pp[..., s0 + msl['known']['offset']:s0 + msl['known']['offset'] + 1])
            cur.append(pp[..., s0 + msl['current_pp']['offset']:s0 + msl['current_pp']['offset'] + 1])
        return torch.stack(rem, dim=-2), torch.stack(known, dim=-2), torch.stack(cur, dim=-2)

    def encode(self, pp: torch.Tensor, ids: Dict[str, torch.Tensor], embeddings: Embeddings,
               move_legal: torch.Tensor, active_ctx: torch.Tensor) -> StaticParts:
        """Encode ``pp`` [B,N,POKEMON_FULL_DIM] (``ids`` its categoricals, `slice_pokemon_categoricals`'s
        keys); ``move_legal`` [B,N,4] each move slot's legality (1 off our active); ``active_ctx``
        [B,N,active_context_dim] the mon's side's active context when it is the active, else 0. Reads
        NOTHING else — that is the static-mode contract."""
        B, N = pp.shape[0], pp.shape[1]
        nm = self.num_moves
        rem, known, cur_pp = self._move_columns(pp)                            # [B,N,4,R] [B,N,4,1] x2
        # ---- S: the moves, as a SET
        mv_emb = embeddings.move_embedding(ids["all_move_ids"])
        mv_type = embeddings.type_embedding(ids["all_move_type_ids"])
        hp_probs = ids["hp_probs"]
        is_hp = (ids["all_move_ids"] == HIDDEN_POWER_MOVE_NUM).unsqueeze(-1)
        soft = embeddings.hp_soft_type(hp_probs).unsqueeze(2).expand(-1, -1, nm, -1)
        mv_type = torch.where(is_hp, soft, mv_type)
        hp_per_slot = torch.where(is_hp, hp_probs.unsqueeze(2).expand(-1, -1, nm, -1),
                                  torch.zeros((), dtype=hp_probs.dtype, device=hp_probs.device))
        blocks = [mv_emb, mv_type, rem, known, hp_per_slot]
        if self.move_latent:
            blocks.append(self.move_latent_encoder(mv_emb, mv_type, ids["all_move_ids"]))
        mv = self.move_network(torch.cat(blocks, dim=-1))                      # [B,N,4,H1]
        mv_in = mv.reshape(B * N, nm, MOVE_NET_HIDDEN[1])
        mv_delta, _ = self.move_self_attn(mv_in, mv_in, mv_in)
        m_static = self.move_self_norm(mv_in + mv_delta).reshape(B, N, nm, MOVE_NET_HIDDEN[1])
        move_set = m_static.sum(dim=2)                                         # [B,N,H1] permutation-invariant
        # ---- S: the identity
        s_in = torch.cat([
            embeddings.species_embedding(ids["species_ids"]),
            self._actual_stats(pp, ids["species_ids"]),
            embeddings.item_embedding(ids["item_ids"]),
            pp[..., self._item_known:self._item_known + 1],
            embeddings.type_embedding(ids["type1_ids"]), embeddings.type_embedding(ids["type2_ids"]),
            embeddings.ability_embedding(ids["ability1_ids"]), embeddings.ability_embedding(ids["ability2_ids"]),
            pp[..., self._ab_dom:self._ab_dom + 1], pp[..., self._ab_known:self._ab_known + 1],
            move_set,
            pp[..., POKEMON_SPECIES_KNOWN_OFFSET:POKEMON_SPECIES_KNOWN_OFFSET + 1],
            pp[..., POKEMON_HP_BLOCK_OFFSET:POKEMON_HP_BLOCK_OFFSET + POKEMON_HP_BLOCK_DIM],
        ], dim=-1)
        static = self.role_encoder(s_in)
        # ---- D: the per-move state, then the mon's
        mdyn = torch.cat([cur_pp, move_legal.unsqueeze(-1).to(pp.dtype)], dim=-1)     # [B,N,4,2]
        move_tokens = m_static + self.move_dyn_proj(mdyn)
        pool = torch.relu(self.move_dyn_pool(torch.cat([m_static, mdyn], dim=-1))).sum(dim=2)
        la = POKEMON_LAST_ACTION_OFFSET
        d_in = torch.cat([
            pp[..., POKEMON_HP_OFFSET:POKEMON_HP_OFFSET + 1],
            pp[..., POKEMON_CONDITION_OFFSET:POKEMON_CONDITION_OFFSET + CONDITION_DIM],
            pp[..., POKEMON_COUNTER_OFFSET:POKEMON_COUNTER_OFFSET + POKEMON_COUNTER_DIM],
            pp[..., POKEMON_SLEEP_BELIEF_OFFSET:POKEMON_SLEEP_BELIEF_OFFSET + POKEMON_SLEEP_BELIEF_DIM],
            pp[..., POKEMON_RECENCY_OFFSET:POKEMON_RECENCY_OFFSET + POKEMON_RECENCY_DIM],
            pp[..., POKEMON_PROTECT_OFFSET:POKEMON_PROTECT_OFFSET + 1],
            embeddings.move_embedding(ids["last_move_ids"]),
            pp[..., la + 1:la + POKEMON_LAST_ACTION_DIM],                      # the id column is embedded
            pp[..., POKEMON_TRAPPED_OFFSET:POKEMON_MAYBE_TRAPPED_OFFSET + 1],
            pp[..., POKEMON_ACTIVE_OFFSET:POKEMON_ACTIVE_OFFSET + 1],
            pp[..., self._item_consumed:self._item_consumed + 1],
            active_ctx,
            pool,
        ], dim=-1)
        dynamic = self.dynamic_encoder(d_in)
        return StaticParts(tokens=static + dynamic, static=static, dynamic=dynamic, move_tokens=move_tokens)

    # ------------------------------------------------------------------------------------- the forward
    @staticmethod
    def ctx_ids(ctx: ExtractorContext) -> Dict[str, torch.Tensor]:
        return {"species_ids": ctx.species_ids, "all_move_ids": ctx.all_move_ids,
                "all_move_type_ids": ctx.all_move_type_ids, "item_ids": ctx.item_ids,
                "ability1_ids": ctx.ability1_ids, "ability2_ids": ctx.ability2_ids,
                "type1_ids": ctx.type1_ids, "type2_ids": ctx.type2_ids, "hp_probs": ctx.hp_probs,
                "last_move_ids": ctx.last_move_ids}

    def move_legality(self, ctx: ExtractorContext) -> torch.Tensor:
        """[B,12,4] per (mon, SORTED move slot): our active's request-order legality matched to its sorted
        slots by MOVE-NUM IDENTITY (the `_request_order_move_tokens` rule, inverted); 1 everywhere else. A
        sorted slot no request slot names (an empty slot) reads 0 on our active."""
        B = ctx.batch_size
        ar = torch.arange(B, device=ctx.device)
        sorted_ids = ctx.all_move_ids[ar, ctx.our_active_idx]                          # [B,4]
        req_ids = ctx.our_active_req_move_ids.long()                                    # [B,4]
        match = (req_ids[:, None, :] == sorted_ids[:, :, None]) & (sorted_ids[:, :, None] > 0)
        legal = (match & (ctx.our_active_req_move_legal[:, None, :] > 0.5)).any(-1)     # [B,4sorted]
        out = torch.ones(B, 2 * TEAM_SIZE, self.num_moves, dtype=ctx.pokemon_part.dtype, device=ctx.device)
        out[ar, ctx.our_active_idx] = legal.to(out.dtype)
        return out

    def active_context(self, ctx: ExtractorContext) -> torch.Tensor:
        """[B,12,active_context_dim]: each side's active context on its ACTIVE mon's row (legacy's E2 rule)."""
        B = ctx.batch_size
        ar = torch.arange(B, device=ctx.device)
        out = torch.zeros(B, 2 * TEAM_SIZE, self._active_ctx_dim, dtype=ctx.pokemon_part.dtype,
                          device=ctx.device)
        out[ar, ctx.our_active_idx] = ctx.our_ctx_raw
        out[ar, TEAM_SIZE + ctx.opp_active_local] = ctx.opp_ctx_raw
        return out

    def parts(self, ctx: ExtractorContext, embeddings: Embeddings) -> StaticParts:
        return self.encode(ctx.pokemon_part, self.ctx_ids(ctx), embeddings, self.move_legality(ctx),
                           self.active_context(ctx))

    def forward(self, ctx: ExtractorContext, embeddings: Embeddings) -> torch.Tensor:
        """[B,12,ROLE_TOKEN_SIZE]; stashes the per-move tokens (SORTED-BY-ID slot order, as legacy)."""
        p = self.parts(ctx, embeddings)
        self.last_move_tokens = p.move_tokens
        return p.tokens


def encode_dex_rows(pe: StaticTokenEncoder, embeddings: Embeddings, rows: torch.Tensor) -> torch.Tensor:
    """[R,ROLE_TOKEN_SIZE] the static-mode token of each OPPONENT dex row in ``rows`` [R,POKEMON_FULL_DIM]
    (a hypothesis is never the active: active context 0, every move slot legal)."""
    pp = rows.unsqueeze(0)
    ids = slice_pokemon_categoricals(pp, pe.layout)
    legal = pp.new_ones(1, rows.shape[0], pe.num_moves)
    act = pp.new_zeros(1, rows.shape[0], pe._active_ctx_dim)
    return pe.encode(pp, ids, embeddings, legal, act).tokens[0]


def static_hypothesis_tokens(pe: StaticTokenEncoder, embeddings: Embeddings, slot_species: torch.Tensor,
                             dex_rows: torch.Tensor) -> torch.Tensor:
    """[B,6,ROLE_TOKEN_SIZE] the opponent slots' hypothesis tokens under `static`: the dex table encoded
    ONCE and GATHERED by ``slot_species`` [B,6] (0 on a revealed slot — never read). A small batch
    (B·6 < the table, T2's buckets) encodes the slots' own rows instead (a static choice per graph)."""
    B = slot_species.shape[0]
    sp = slot_species.long()
    rows = dex_rows.to(next(pe.parameters()).dtype)
    if B * TEAM_SIZE < rows.shape[0]:
        return encode_dex_rows(pe, embeddings, rows[sp.reshape(-1)]).reshape(B, TEAM_SIZE, -1)
    return encode_dex_rows(pe, embeddings, rows)[sp]
