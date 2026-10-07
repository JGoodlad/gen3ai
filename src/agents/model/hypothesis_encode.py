"""X5's hypothesis TOKENS, encoded with the SPECIES half once per forward and GATHERED
(`gen3_x5_hyp_gather_v1`; `designs/endstate/design_x5_belief_tokens.md` §3.4 "as built").
X5 only — the `blob` arm never calls anything here.

WHAT IT COMPUTES. A hypothesis token is `PokemonEncoder` on the hypothesis context (`hypothesis_ctx`:
a hidden opponent slot's row is its hypothesis's DEX ROW) — the per-row pass this replaces ran the whole
encoder over all 12 mons of every row (B x 12 x 4 move-network rows), to read 6 of them. That encoding is
NOT a function of the species alone: five ROW-LEVEL inputs enter it — the clock, weather, fainted and
hazard features (the move network's context) and the screens (with the first four, the role encoder's
broadcast global context) — plus the opponent's active-context scatter (a hypothesis is never the active,
so it reads zeros; it is kept so the split is exact on every slot regardless). With those held at one row's
values, the per-row pass's tokens are bitwise equal per species on real rows; with real rows they differ
by up to 1.06 (`designs/research_state/measurements/x5_hyp_gather_2026-10-05/`, premise).

THE EXACT SPLIT. Each of those inputs reaches the network ONLY through the FIRST Linear of the move
network and of the role encoder, and a Linear is additive over its input columns:
``W·[x_species; x_row] + b = (W_s·x_species + b) + W_r·x_row``. So:

* **once per forward, over the dex table** (S = 400 rows; `species_table`): every species-only column
  (embeddings, the move latent, remnants, the Hidden Power blend, the hp / active cells of the dex row,
  validity 1) through each first Linear, its row-level columns ZERO — ``move_pre`` [S,4,H0], ``role_pre``
  [S,R0];
* **per row** (B): the row-level columns against their own weight columns (no bias);
* **per (row, opponent slot)** (B x 6, not B x 12): gather the table by the hypothesis species, add, then
  everything AFTER the first Linear exactly as the per-row pass runs it (ReLU, the second Linear, the
  within-mon move self-attention + norm, the processed moves' role columns, ReLU, the second Linear).

Equal to the per-row pass up to fp32 REASSOCIATION of the two first-layer sums (the declared bound and its
measurement: the unit's README; `hypothesis_encode_test`). The non-hypothesis slots' outputs are
discarded by `splice_hypothesis_tokens` (they gather table row 0 — finite, never read).

The per-row pass stays THE definition: `PokemonEncoder.forward` on `hypothesis_ctx` is the reference the
tests compare this against, so a change to the encoder that this module does not mirror FAILS there.
"""
from __future__ import annotations

from typing import Any, NamedTuple

import torch
import torch.nn.functional as F

from agents.model.extractor_ctx import Embeddings, ExtractorContext, slice_pokemon_categoricals
from agents.observation.constants import TEAM_SIZE
from agents.observation.moves import HIDDEN_POWER_MOVE_NUM


class SpeciesTable(NamedTuple):
    """The species half of the two first Linears, over the dex table (row s = dex row s)."""
    move_pre: torch.Tensor   # [S,4,H0]  move_network[0] on the species columns (+ its bias)
    role_pre: torch.Tensor   # [S,R0]    role_encoder[0] on the species columns (+ its bias)


class _Cols(NamedTuple):
    move_ctx: slice   # move_network[0]'s row-level columns: clock, weather, fainted, hazards
    role_pm: slice    # role_encoder[0]'s processed-moves columns
    role_act: slice   # its active-context columns
    role_glob: slice  # its broadcast global-context columns


def _cols(pe: Any) -> _Cols:
    """The column blocks of the two first Linears, from the SAME layout arithmetic `PokemonEncoder`
    uses to size them (the move input: embeds · remnants · known · [hp, clock, weather, fainted,
    hazards] · hp-probs · validity · latent; the role input: … · processed moves · hp_and_active ·
    last move · active ctx · global ctx)."""
    layout = pe.layout
    gl, rl = layout['global_layout'], layout['reactive_layout']
    move_ctx_dim = 1 + gl['clock']['dim'] + gl['weather']['dim'] + rl['fainted']['dim'] + gl['hazards']['dim']
    c0 = layout['move_embedding_dim'] + layout['type_embedding_dim'] + pe.move_remnant_dim + 1 + 1  # after hp
    move_ctx = slice(c0, c0 - 1 + move_ctx_dim)
    glob_dim = (gl['clock']['dim'] + gl['weather']['dim'] + rl['fainted']['dim'] + gl['hazards']['dim']
                + gl['screens']['dim'])
    n_in = pe.role_encoder[0].in_features
    g0 = n_in - glob_dim
    a0 = g0 - pe._active_ctx_dim
    from agents.model.arch_constants import MOVE_NET_HIDDEN
    from agents.observation.constants import POKEMON_FULL_DIM
    hp_and_active_dim = POKEMON_FULL_DIM - layout['pokemon']['hp']['offset']
    p1 = a0 - layout['move_embedding_dim'] - hp_and_active_dim
    p0 = p1 - MOVE_NET_HIDDEN[1] * pe.num_moves
    return _Cols(move_ctx=move_ctx, role_pm=slice(p0, p1), role_act=slice(a0, g0), role_glob=slice(g0, n_in))


def species_table(pe: Any, embeddings: Embeddings, dex_rows: torch.Tensor) -> SpeciesTable:
    """``dex_rows`` [S, POKEMON_FULL_DIM] → the species half of both first Linears. Mirrors
    `PokemonEncoder.forward`'s stitch for an OPPONENT mon (move validity 1), with every row-level column
    zero."""
    layout = pe.layout
    pp = dex_rows.unsqueeze(0)                                                         # [1,S,D]
    ids = slice_pokemon_categoricals(pp, layout)
    S = pp.shape[1]
    nm = pe.num_moves
    cols = _cols(pe)

    embedded_species = embeddings.species_embedding(ids["species_ids"])
    embedded_moves = embeddings.move_embedding(ids["all_move_ids"])
    embedded_move_types = embeddings.type_embedding(ids["all_move_type_ids"])
    embedded_items = embeddings.item_embedding(ids["item_ids"])
    embedded_ability1 = embeddings.ability_embedding(ids["ability1_ids"])
    embedded_ability2 = embeddings.ability_embedding(ids["ability2_ids"])
    hp_probs = ids["hp_probs"]
    soft = embeddings.hp_soft_type(hp_probs).unsqueeze(2).expand(-1, -1, nm, -1)
    is_hp_slot = ids["all_move_ids"] == HIDDEN_POWER_MOVE_NUM
    embedded_move_types = torch.where(is_hp_slot.unsqueeze(-1), soft, embedded_move_types)
    embedded_pk_types = torch.cat([embeddings.type_embedding(ids["type1_ids"]),
                                   embeddings.type_embedding(ids["type2_ids"])], dim=-1)

    pk = layout['pokemon']
    species_info = pk['species']
    species_idx = species_info['offset'] + species_info['layout']['species_id']['offset']
    stats_start = species_idx + species_info['layout']['species_id']['dim']
    items_info, items_layout = pk['items'], pk['items']['layout']
    part_a = pp[:, :, stats_start:items_info['offset']]
    i_known = items_info['offset'] + items_layout['known']['offset']
    item_remnant = pp[:, :, i_known:i_known + items_layout['known']['dim']]
    i_cons = items_info['offset'] + items_layout['consumed']['offset']
    item_consumed = pp[:, :, i_cons:i_cons + items_layout['consumed']['dim']]
    ab_info, ab_layout = pk['abilities'], pk['abilities']['layout']
    a_rem = ab_info['offset'] + ab_layout['known']['offset']
    ability_remnant = pp[:, :, a_rem:a_rem + ab_layout['known']['dim']]
    a_dom = ab_info['offset'] + ab_layout['dominance']['offset']
    ability_dominance = pp[:, :, a_dom:a_dom + ab_layout['dominance']['dim']]
    moves_info = pk['moves']
    moves_offset, moves_layout = moves_info['offset'], moves_info['layout']
    msl = moves_layout['slot_layout']
    part_d = pp[:, :, ab_info['offset'] + ab_info['dim']:moves_offset]
    known_off = msl['known']['offset']
    rem, known = [], []
    for i in range(nm):
        s0 = moves_offset + moves_layout['slots'][i]['offset']
        rem.append(pp[:, :, s0 + msl['power']['offset']:s0 + msl['type']['offset']])
        rem.append(pp[:, :, s0 + msl['type']['offset'] + msl['type']['dim']:s0 + known_off])
        rem.append(pp[:, :, s0 + msl['current_pp']['offset']:s0 + msl['max_pp']['offset'] + msl['max_pp']['dim']])
        rem.append(pp[:, :, s0 + msl['accuracy']['offset']:s0 + msl['never_miss']['offset'] + msl['never_miss']['dim']])
        known.append(pp[:, :, s0 + known_off:s0 + known_off + 1])
    move_remnants = torch.cat(rem, dim=2).reshape(1, S, nm, pe.move_remnant_dim)
    known_flags = torch.cat(known, dim=2).reshape(1, S, nm, 1)
    hp_and_active = ids["hp_and_active"]
    n_row_ctx = cols.move_ctx.stop - cols.move_ctx.start
    move_context = torch.cat([hp_and_active[:, :, 0:1], pp.new_zeros(1, S, n_row_ctx)], dim=2)
    hp_probs_per_slot = torch.where(is_hp_slot.unsqueeze(-1), hp_probs.unsqueeze(2).expand(-1, -1, nm, -1),
                                    torch.zeros((), dtype=hp_probs.dtype, device=hp_probs.device))
    blocks = [embedded_moves, embedded_move_types, move_remnants, known_flags,
              move_context.unsqueeze(2).expand(-1, -1, nm, -1), hp_probs_per_slot,
              pp.new_ones(1, S, nm, 1)]
    if pe.move_latent:
        blocks.append(pe.move_latent_encoder(embedded_moves, embedded_move_types, ids["all_move_ids"]))
    move_pre = pe.move_network[0](torch.cat(blocks, dim=3))[0]                         # [S,4,H0]

    embedded_last_move = embeddings.move_embedding(ids["last_move_ids"])
    n_pm = cols.role_pm.stop - cols.role_pm.start
    n_tail = cols.role_glob.stop - cols.role_act.start                                 # active ctx + global ctx
    role_in = torch.cat([embedded_species, part_a, embedded_items, item_remnant, item_consumed,
                         embedded_pk_types, embedded_ability1, embedded_ability2, ability_dominance,
                         ability_remnant, part_d, pp.new_zeros(1, S, n_pm), hp_and_active,
                         embedded_last_move, pp.new_zeros(1, S, n_tail)], dim=2)
    role_pre = pe.role_encoder[0](role_in)[0]                                          # [S,R0]
    return SpeciesTable(move_pre=move_pre, role_pre=role_pre)


def gathered_hypothesis_tokens(pe: Any, embeddings: Embeddings, ctx: ExtractorContext,
                               slot_species: torch.Tensor, dex_rows: torch.Tensor) -> torch.Tensor:
    """[B,6,ROLE_TOKEN_SIZE] the opponent slots' hypothesis-pass tokens (the per-row pass's
    ``pokemon_encoder(hypothesis_ctx(...))[:, 6:12]`` on every hypothesis slot, up to fp32
    reassociation). ``slot_species`` [B,6] is `HypothesisSet.slot_species` (0 on a revealed slot — its
    output is never read). ``ctx`` is the REAL context: only its row-level fields are read."""
    from agents.model.arch_constants import MOVE_NET_HIDDEN
    cols = _cols(pe)
    B = slot_species.shape[0]
    nm = pe.num_moves
    sp = slot_species.long()
    rows = dex_rows.to(ctx.pokemon_part.dtype)
    if B * TEAM_SIZE < rows.shape[0]:
        # SMALL batch (T2's buckets 8 / 64): the slots' own dex rows are fewer than the table's — encode
        # those (the same species half, per slot; a static choice: B is fixed per compiled graph).
        tab = species_table(pe, embeddings, rows[sp.reshape(-1)])
        move_pre = tab.move_pre.reshape(B, TEAM_SIZE, nm, -1)                            # [B,6,4,H0]
        role_pre = tab.role_pre.reshape(B, TEAM_SIZE, -1)                                # [B,6,R0]
    else:
        tab = species_table(pe, embeddings, rows)                                        # once, gathered
        move_pre, role_pre = tab.move_pre[sp], tab.role_pre[sp]
    w_mv = pe.move_network[0].weight
    w_ro = pe.role_encoder[0].weight
    # the move network: species half + the row's context columns, then the rest as the pass runs it
    move_row = F.linear(torch.cat([ctx.turn_feature, ctx.weather_feature, ctx.fainted_feature,
                                   ctx.spikes_feature], dim=1), w_mv[:, cols.move_ctx])   # [B,H0]
    h = move_pre + move_row[:, None, None, :]                                            # [B,6,4,H0]
    mv = pe.move_network[2](pe.move_network[1](h))                                       # [B,6,4,H1]
    mv_in = mv.reshape(B * TEAM_SIZE, nm, MOVE_NET_HIDDEN[1])
    mv_delta, _ = pe.move_self_attn(mv_in, mv_in, mv_in)
    processed = pe.move_self_norm(mv_in + mv_delta).reshape(B, TEAM_SIZE, nm * MOVE_NET_HIDDEN[1])
    # the role encoder: species half gathered + processed moves + the active-context scatter + global ctx
    glob = torch.cat([ctx.turn_feature, ctx.weather_feature, ctx.fainted_feature, ctx.spikes_feature,
                      ctx.screen_feature], dim=1)
    act = F.linear(ctx.opp_ctx_raw, w_ro[:, cols.role_act])                              # [B,R0]
    is_act = F.one_hot(ctx.opp_active_local, TEAM_SIZE).bool().unsqueeze(-1)             # [B,6,1]
    r = (role_pre + F.linear(processed, w_ro[:, cols.role_pm])
         + torch.where(is_act, act.unsqueeze(1), torch.zeros((), dtype=act.dtype, device=act.device))
         + F.linear(glob, w_ro[:, cols.role_glob]).unsqueeze(1))                        # [B,6,R0]
    return pe.role_encoder[2](pe.role_encoder[1](r))  # type: ignore[no-any-return]
