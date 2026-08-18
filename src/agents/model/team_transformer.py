"""The trunk: edge-biased attention (EdgeBias, BiasedEncoderLayer), TeamTransformer, and the event-seat consumer.

Split out of `features_extractor.py` 2026-08-16 (one responsibility per file); that module
re-exports every name here, so historical import paths still resolve.
"""
from agents.model.extractor_ctx import Embeddings, ExtractorContext, NUM_TOKEN_TYPES, TOKEN_TYPE_GLOBAL, TOKEN_TYPE_OUR_TEAM, TOKEN_TYPE_THEIR_TEAM
import torch
from torch.utils.checkpoint import checkpoint
from typing import Callable, Dict, Any, Optional, Tuple
from agents.observation.constants import (
    TEAM_SIZE,
    GLOBAL_ENV_DIM,
)

from agents.model.arch_constants import (
    D_MODEL,
    TRANSFORMER_N_LAYERS,
    TRANSFORMER_N_HEADS,
    TRANSFORMER_FFN_DIM,
)




class BiasedEncoderLayer(torch.nn.Module):
    """gen3_edge_bias_trunk_v1 (v56, Stage 2 of the entity generation): a TransformerEncoderLayer
    clone (post-LN, ReLU, dropout 0 — the literal production kwargs) whose self-attention takes an
    ADDITIVE per-pair per-head float bias `[B, H, n, n]` via `F.scaled_dot_product_attention`'s
    additive-mask path — exactly "logits += bias" pre-softmax. This is the delivery mechanism for
    computed physics as attention EDGES (the closed Stage-2 spike — `entity_spike_benchmark.py`,
    now in git history only — proved the kernel:
    matches a float64 softmax(logits+bias) reference at 1.2e-7 and compiles fullgraph with zero
    graph breaks). The KEY-PADDING mask rides the same tensor as a large negative addend, so
    bias=mask-only reproduces the stock masked layer's math (pinned by
    `edge_bias_test.py::test_layer_matches_stock_transformer_layer`). State_dict keys differ from
    `nn.TransformerEncoderLayer` (`in_proj.*` vs `self_attn.in_proj_*`) — part of the v55
    unconditional break."""

    def __init__(self, d_model: int = D_MODEL, n_heads: int = TRANSFORMER_N_HEADS,
                 ffn_dim: int = TRANSFORMER_FFN_DIM):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.in_proj = torch.nn.Linear(d_model, 3 * d_model)
        self.out_proj = torch.nn.Linear(d_model, d_model)
        self.linear1 = torch.nn.Linear(d_model, ffn_dim)
        self.linear2 = torch.nn.Linear(ffn_dim, d_model)
        self.norm1 = torch.nn.LayerNorm(d_model)
        self.norm2 = torch.nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x [B, n, d_model]; `bias` [B, H, n, n] additive per-pair per-head attention-logit bias
        (already carrying the key-padding addend), or None. Returns the refined [B, n, d_model]."""
        B, n, d = x.shape
        qkv = self.in_proj(x).reshape(B, n, 3, self.n_heads, self.head_dim)
        q, k, v = (qkv[:, :, i].transpose(1, 2) for i in range(3))            # each [B,H,n,hd]
        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        x = self.norm1(x + self.out_proj(attn.transpose(1, 2).reshape(B, n, d)))
        return self.norm2(x + self.linear2(torch.nn.functional.relu(self.linear1(x))))  # type: ignore[no-any-return]


# The key-padding addend on the attention logits. Large-negative-finite rather than -inf: a -inf
# row×col combination under compile/half precision breeds NaN; -1e9 underflows the softmax to
# exactly 0 attention weight at these magnitudes, which is all masking needs.
_KEY_PAD_NEG = -1e9

# Edge-cell widths (the op owns the cell CONTENT — see DamageOperator.pairwise_{outgoing,incoming,
# bench_outgoing} + the status kernels' per_pair branches; these mirror those methods' last dims and
# are pinned against them in edge_bias_test.py).
_EDGE_D1_CELL = 6   # [low, high, crit, pko, type_mult, revealed] per (our move k, opp mon d)
_EDGE_D2_CELL = 4   # [best_high, best_pko, p_outspeed, alive] per (our mon i, opp ACTIVE)
_EDGE_D3_CELL = 5   # [high, pko, eff, is_phys, w] per (their believed move c, our mon i)
_EDGE_S1_CELL = 2   # [land, land·immob] per (our status move k, opp mon d)
_EDGE_S3_CELL = 3   # [land, land·immob, w] per (their believed status move c, our mon i)
_EDGE_V_CELL = 3    # [p_outspeed, both_alive, revealed_j] per (our mon i, opp mon j)
_EDGE_D4_CELL = 4   # [phys_high, spec_high, phys_pko, spec_pko] per (our mon i, opp BENCH mon j)
_EDGE_T_CELL = 2    # [P(i traps j), P(j traps i)] per (our mon i, opp mon j)
_EDGE_X_CELL = 4    # [entry_chip, pursuit_p, pursuit_eff, grounded] per (mon, GLOBAL seat)
_EDGE_G_CELL = 4    # [leftovers, weather_chip, status_tick, leech] per (mon, GLOBAL seat) — signed
_EDGE_C4_CELL = 4   # [is_protect, p_success, net_ours, net_theirs] per (E3 seat, GLOBAL seat)
_EDGE_C1_CELL = 7   # [is_boost, d_best_high, d_best_pko, d_outspeed, hp_cost, d_in_high, d_in_pko]
                    # per (E3 setup-move seat, opp mon) — outgoing (C1, incl. Belly Drum's
                    # half-max-HP price) ⊕ incoming (C1b) consequence deltas
_EDGE_C3_CELL = 3   # [is_recovery, d_in_pko, rest_sleep_turns] per (E3 recovery seat, opp mon) —
                    # the heal-vs-KO flip + Rest's DETERMINISTIC self-sleep cost (2 turns; 1 EB)
_EDGE_C2_CELL = 7   # [is_status, land, d_their_outspeed, d_in_phys_high, d_sched, d_in_all_slp,
                    # e_slp_free_turns] per (E3 status seat, opp mon) — the post-landing
                    # consequence world behind S1 (incl. the sleep-tempo + true-toxic-tick facts)
_EDGE_C5_CELL = 4   # [is_bp, d_best_high, d_best_pko, d_outspeed] per (E3 Baton-Pass seat, OUR
                    # mon) — the receiver's offense inheriting the active's stages (the first
                    # family on the (E3, our-mon) route)
_EDGE_R_CELL = 2    # [is_actor, is_target] per (event seat e, mon token m) — Tier H-C ENTITY
                    # REFERENCE edges (gen3_event_ref_edges_v1, design_history_entity.md §3 H-C):
                    # a STRUCTURAL identity, not a computed quantity — event e's recorded
                    # actor/target IS mon m (species-num equality, SIDE-GATED so a mirror match
                    # across teams cannot false-link; species↔slot is battle-stable). This is
                    # what turns the two critical queries into single attention hops: mon j
                    # attends over the event seats whose target-edge points at j ("what did they
                    # click into this mon"), and a switch-in event's actor-edge names the
                    # arriving mon ("whom did they switch into"). Requires history_events (the
                    # seats are the rows). Events referencing FAINTED mons: the mon KEY is
                    # masked in attention, so the mon→event direction survives while the
                    # event→fainted-mon hop is inert — the token's species content covers it
                    # (the design's accepted v1 nuance).
_EDGE_H_CELL = 5    # [switch_ins, attacks, status_clicks, shared_field_turns, pairing_recency]
                    # per (our mon j, opp mon i) — Tier H-A2 pair-history TENDENCY counts
                    # (gen3_pair_history_v1): folded CPU-side from PUBLIC events by the
                    # EpisodeTracker, log-saturated, delivered from the obs block (an obs-fed
                    # family — the one family whose cell the GPU cannot recompute, since it IS
                    # compiled battle history). A ratio delivery is CORRECT here: tendencies are
                    # relative ("Blissey more than Skarmory"), unlike damage magnitudes.

_EDGE_FAMILIES = {"d1": _EDGE_D1_CELL, "d2": _EDGE_D2_CELL, "d3": _EDGE_D3_CELL,
                  "d4": _EDGE_D4_CELL, "s1": _EDGE_S1_CELL, "s3": _EDGE_S3_CELL,
                  "v": _EDGE_V_CELL, "t": _EDGE_T_CELL, "x": _EDGE_X_CELL,
                  "g": _EDGE_G_CELL, "c4": _EDGE_C4_CELL, "c1": _EDGE_C1_CELL,
                  "c3": _EDGE_C3_CELL, "c2": _EDGE_C2_CELL, "c5": _EDGE_C5_CELL,
                  "h": _EDGE_H_CELL, "r": _EDGE_R_CELL}


class EdgeBias(torch.nn.Module):
    """gen3_edge_bias_trunk_v1 (v55): computed physics as per-pair per-head attention-logit BIASES.

    Stage 2's first slice — the D (damage) family, both quadrants that already have validated
    kernels AND both endpoints seated in the trunk (the v54 move seats made this possible):
      * **D1 outgoing** — our active's move k → their mon d: `DamageOperator.pairwise_outgoing`
        (the v34 `_outgoing_matrix` physics, request-ordered == E3 seat order), written at the
        (E3 seat k, their-mon seat d) pair and its transpose.
      * **D3 incoming** — their believed move c → our mon i: `DamageOperator.pairwise_incoming`
        (the pre-collapse `_incoming_rolls` physics, the SAME detached candidate selection the E4
        seats used — seat c and bias row c always name the same move), written at the
        (E4 seat c, our-mon seat i) pair and its transpose.

    Each family maps its cell through a ZERO-INIT `Linear(cell → 2·n_heads)` (one head-set per
    direction, since "how much the move seat attends to the defender" and the reverse are
    different questions) — so at init every bias is exactly 0 and the trunk is byte-identical to
    the family-off forward (the `prefuse_proj` identity-at-init convention; auto-protected by
    `restore_identity_init`'s observation capture). All seat blocks are CONTIGUOUS index ranges
    (mons [0:6]/[6:12], E3 [20:24], E4 [24:24+K]), so delivery is plain slice assignment — no
    scatter, compile-friendly.

    The op head-concat is NOT deleted here: per the deprecation playbook (and the K9/K10 trunk-null
    history) the edge home is built first; deletion waits on the per-family bias-ablation audit."""

    def __init__(self, families: str):
        super().__init__()
        fams = set() if families in ("", "off") else set(families.split(","))
        if families == "d":
            fams = {"d1", "d3"}   # FROZEN alias (the first-slice pair) — new families are explicit-only,
                                  # so a saved "d" config never silently grows maps under newer code.
        unknown = fams - set(_EDGE_FAMILIES)
        if unknown:
            raise ValueError(f"edge_bias_families: unknown families {sorted(unknown)} "
                             f"(valid: 'off', 'd' [= d1,d3], or a comma list of {sorted(_EDGE_FAMILIES)})")
        self.families = fams
        # One zero-init Linear(cell -> 2·n_heads) per enabled family (a head-set per direction).
        for fam, cell in _EDGE_FAMILIES.items():
            lin = None
            if fam in fams:
                lin = torch.nn.Linear(cell, 2 * TRANSFORMER_N_HEADS)
                torch.nn.init.zeros_(lin.weight)
                torch.nn.init.zeros_(lin.bias)
            setattr(self, f"{fam}_map", lin)

    def _write_block(self, bias: torch.Tensor, m: torch.Tensor,
                     rows: slice, cols: slice) -> None:
        """Add a family's mapped cells at (rows, cols) + the transpose block. `m` [B, R, C, 2H] —
        the first head-set biases row→col attention, the second the reverse direction."""
        H = TRANSFORMER_N_HEADS
        m = m.permute(0, 3, 1, 2)                                              # [B,2H,R,C]
        bias[:, :, rows, cols] = bias[:, :, rows, cols] + m[:, :H]
        bias[:, :, cols, rows] = bias[:, :, cols, rows] + m[:, H:].transpose(-1, -2)

    def forward(self, bias: torch.Tensor, base_seats: int,
                cells: "Dict[str, torch.Tensor]",
                opp_active_onehot: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Write the enabled families into `bias` [B, H, n, n] (already carrying the key-pad addend).
        `base_seats` = the seat count BEFORE the extra block (E3 starts there); `cells` maps family →
        its per-pair cell tensor (see _EDGE_FAMILIES); `opp_active_onehot` [B,6] locates the opp
        active column for the mon↔mon D2 family. Returns `bias`."""
        H = TRANSFORMER_N_HEADS
        e3, e4 = base_seats, base_seats + 4
        our = slice(0, TEAM_SIZE)
        opp = slice(TEAM_SIZE, 2 * TEAM_SIZE)
        if self.d1_map is not None and cells.get("d1") is not None:
            self._write_block(bias, self.d1_map(cells["d1"]), slice(e3, e3 + 4), opp)
        if self.s1_map is not None and cells.get("s1") is not None:
            self._write_block(bias, self.s1_map(cells["s1"]), slice(e3, e3 + 4), opp)
        if self.c1_map is not None and cells.get("c1") is not None:
            # C1 rides the same (E3 seat, opp-mon) route as D1/S1 — a setup-move seat's edge to
            # mon j carries the post-boost CONSEQUENCE deltas instead of this-turn damage.
            self._write_block(bias, self.c1_map(cells["c1"]), slice(e3, e3 + 4), opp)
        if self.c3_map is not None and cells.get("c3") is not None:
            # C3: a recovery seat's edge to mon j carries the heal-vs-their-KO flip.
            self._write_block(bias, self.c3_map(cells["c3"]), slice(e3, e3 + 4), opp)
        if self.c2_map is not None and cells.get("c2") is not None:
            # C2: a status seat's edge to mon j carries what LANDING would do (behind S1's land).
            self._write_block(bias, self.c2_map(cells["c2"]), slice(e3, e3 + 4), opp)
        if self.c5_map is not None and cells.get("c5") is not None:
            # C5: the Baton-Pass seat's edge to OUR mon j — the receiver axis (the first family
            # on the (E3, our-mon) route; the transpose head-set answers "who wants the pass").
            self._write_block(bias, self.c5_map(cells["c5"]), slice(e3, e3 + 4), our)
        if self.d3_map is not None and cells.get("d3") is not None:
            K = cells["d3"].shape[1]
            self._write_block(bias, self.d3_map(cells["d3"]), slice(e4, e4 + K), our)
        if self.s3_map is not None and cells.get("s3") is not None:
            K = cells["s3"].shape[1]
            self._write_block(bias, self.s3_map(cells["s3"]), slice(e4, e4 + K), our)
        if self.d4_map is not None and cells.get("d4") is not None:
            # D4 is the full mon↔mon block too (the active column arrives pre-zeroed by the kernel).
            self._write_block(bias, self.d4_map(cells["d4"]), our, opp)
        if self.c4_map is not None and cells.get("c4") is not None:
            # C4 connects the Protect-family E3 seats to the GLOBAL seat.
            g = 2 * TEAM_SIZE
            self._write_block(bias, self.c4_map(cells["c4"][:, :, None, :]),
                              slice(base_seats, base_seats + 4), slice(g, g + 1))
        if self.r_map is not None and cells.get("r") is not None:
            # R (Tier H-C): event-seat reference edges to the 12 live mon tokens. CONTRACT: the
            # event seats are the LAST N tokens (EventSeats joins the extra seam last — the
            # position-stable append; a family between them and the end would break this slice).
            n_tok = bias.shape[-1]
            N = cells["r"].shape[1]
            self._write_block(bias, self.r_map(cells["r"]),
                              slice(n_tok - N, n_tok), slice(0, 2 * TEAM_SIZE))
        if self.g_map is not None and cells.get("g") is not None:
            # G rides the same (mon, GLOBAL seat) route as X — schedule facts are board-level.
            g = 2 * TEAM_SIZE
            g_our, g_opp = cells["g"]
            self._write_block(bias, self.g_map(g_our[:, :, None, :]), our, slice(g, g + 1))
            self._write_block(bias, self.g_map(g_opp[:, :, None, :]), opp, slice(g, g + 1))
        if self.x_map is not None and cells.get("x") is not None:
            # X connects each mon to the GLOBAL seat (index 2·TEAM_SIZE): entry/
            # exit costs are board-level facts, composable with every mon token through it.
            g = 2 * TEAM_SIZE
            x_our, x_opp = cells["x"]
            self._write_block(bias, self.x_map(x_our[:, :, None, :]), our, slice(g, g + 1))
            self._write_block(bias, self.x_map(x_opp[:, :, None, :]), opp, slice(g, g + 1))
        if self.t_map is not None and cells.get("t") is not None:
            # T is mon↔mon like V (both directions ride the cell's two channels + the two head-sets).
            self._write_block(bias, self.t_map(cells["t"]), our, opp)
        if self.v_map is not None and cells.get("v") is not None:
            # V is the full mon↔mon block — both endpoint sets are static contiguous slices.
            self._write_block(bias, self.v_map(cells["v"]), our, opp)
        if self.h_map is not None and cells.get("h") is not None:
            # H (Tier H-A2) is mon↔mon like V/T — obs-fed pair-history tendencies; the two
            # head-sets let "j reads i's habits" and "i pressures j" bias independently.
            self._write_block(bias, self.h_map(cells["h"]), our, opp)
        if self.d2_map is not None and cells.get("d2") is not None:
            # D2 is mon↔mon with a BATCH-VARYING column (the opp ACTIVE slot) — deliver via the
            # one-hot outer product instead of a static slice.
            assert opp_active_onehot is not None
            m = self.d2_map(cells["d2"])                                       # [B,6,2H]
            m = m.permute(0, 2, 1)                                             # [B,2H,6]
            oh = opp_active_onehot[:, None, None, :]                           # [B,1,1,6] broadcast
            bias[:, :, our, opp] = bias[:, :, our, opp] + m[:, :H, :, None] * oh
            bias[:, :, opp, our] = bias[:, :, opp, our] + (m[:, H:, :, None] * oh).transpose(-1, -2)
        return bias


class TeamTransformer(torch.nn.Module):
    """Unified transformer over the entity seats: 6 our + 6 their team role tokens +
    1 global, plus (gen3_entity_move_seats_v1) any `extra` entity seats appended after the
    global token (E3 our-move + E4 threat-move seats today). Adds
    token-type embeddings, applies the encoder stack with a
    fainted/seat key-padding mask, returns the two team token blocks + the
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

        reactive_layout = layout['reactive_layout']
        _board_scalar_dim = reactive_layout['active_req_moves']['offset']
        active_ctx_dim = layout['active_context_dim']
        self._non_matchup_rest_dim = GLOBAL_ENV_DIM + _board_scalar_dim
        self._global_token_input_dim = 2 * active_ctx_dim + self._non_matchup_rest_dim
        self.global_proj = torch.nn.Linear(self._global_token_input_dim, D_MODEL)

        # gen3_edge_bias_trunk_v1 (v55): the encoder stack is the BIASED clone — same math, same
        # shapes, but attention takes an additive per-pair per-head float bias (the edge-delivery
        # mechanism). The key-padding mask rides the bias tensor as a -1e9 addend, so a mask-only
        # bias reproduces the stock layer's masked attention exactly (test-pinned). Unconditional
        # swap — state_dict keys change → the v55 ARCH_SIGNATURE bump carries it.
        self.transformer_layers = torch.nn.ModuleList(
            BiasedEncoderLayer() for _ in range(TRANSFORMER_N_LAYERS)
        )

        self._our_token_slice = slice(0, TEAM_SIZE)
        self._their_token_slice = slice(TEAM_SIZE, 2 * TEAM_SIZE)
        self._total_tokens = 2 * TEAM_SIZE + 1   # team×2 + global

    def forward(self, role_tokens: torch.Tensor, ctx: ExtractorContext,
                embeddings: Embeddings,
                extra: "Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]" = None,
                edge_bias_fn: "Optional[Callable[[torch.Tensor], torch.Tensor]]" = None,
                ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """`extra` (gen3_entity_move_seats_v1): optional `(tokens [B,n,d_model], types [n] long,
        pad [B,n] bool)` — additional entity seats appended AFTER the global token, so every
        absolute slice (team/history/global) is position-stable.
        The third return is the refined extra seats (None when no extra).

        `edge_bias_fn` (gen3_edge_bias_trunk_v1): optional callable receiving the float attention
        bias [B, H, n, n] ALREADY carrying the key-padding addend, returning it with the edge
        families written in (see `EdgeBias`). The bias is built ONCE and shared by every layer."""
        batch_size = ctx.batch_size
        device = ctx.device

        # Global token — active contexts + non-matchup scalars projected into d_model.
        global_token_input = torch.cat([ctx.our_ctx_raw, ctx.opp_ctx_raw, ctx.non_matchup_rest], dim=1)
        global_token = self.global_proj(global_token_input).unsqueeze(1)

        # Token-type embeddings per group.
        our_team_tokens   = role_tokens[:, 0:TEAM_SIZE, :]
        their_team_tokens = role_tokens[:, TEAM_SIZE:2 * TEAM_SIZE, :]

        tt = self.token_type_emb
        our_team_tokens   = our_team_tokens   + tt(torch.full((1,), TOKEN_TYPE_OUR_TEAM,   dtype=torch.long, device=device))
        their_team_tokens = their_team_tokens + tt(torch.full((1,), TOKEN_TYPE_THEIR_TEAM, dtype=torch.long, device=device))
        global_token      = global_token      + tt(torch.full((1,), TOKEN_TYPE_GLOBAL,     dtype=torch.long, device=device))

        tokens = torch.cat([our_team_tokens, their_team_tokens, global_token], dim=1)
        global_pad = torch.zeros(batch_size, 1, dtype=torch.bool, device=device)
        key_padding_mask = torch.cat([
            ctx.fainted_mask_ours,
            ctx.fainted_mask_opp,
            global_pad,
        ], dim=1)
        if extra is not None:
            extra_tokens, extra_types, extra_pad = extra
            tokens = torch.cat([tokens, extra_tokens + self.token_type_emb(extra_types)], dim=1)
            key_padding_mask = torch.cat([key_padding_mask, extra_pad], dim=1)

        # gen3_edge_bias_trunk_v1: ONE float attention bias [B,H,n,n] carries BOTH the key-padding
        # mask (a -1e9 addend on masked KEYS — softmax weight underflows to exactly 0, matching the
        # stock masked layer's math) AND, when `edge_bias_fn` is given, the computed edge families.
        # Built once, shared by every layer (the edges are pre-attention facts, constant per forward).
        n_tok = tokens.shape[1]
        attn_bias = (key_padding_mask[:, None, None, :].float() * _KEY_PAD_NEG).expand(
            batch_size, TRANSFORMER_N_HEADS, n_tok, n_tok).contiguous()
        if edge_bias_fn is not None:
            attn_bias = edge_bias_fn(attn_bias)

        # Gradient checkpointing only helps when a graph is being built for backward
        # (the PPO update); under inference's no_grad it would be pure overhead, so gate on
        # torch.is_grad_enabled(). use_reentrant=False is the correct variant here (handles
        # non-grad inputs + autocast/RNG state); with dropout=0 the recompute is bit-exact.
        use_ckpt = self.grad_checkpointing and torch.is_grad_enabled()
        for layer in self.transformer_layers:
            if use_ckpt:
                tokens = checkpoint(
                    lambda t, _layer=layer: _layer(t, bias=attn_bias),
                    tokens,
                    use_reentrant=False,
                )
            else:
                tokens = layer(tokens, bias=attn_bias)

        our_team_out   = tokens[:, self._our_token_slice, :]
        their_team_out = tokens[:, self._their_token_slice, :]
        # gen3_unified_value_readout_v2: the REFINED global token, stashed as a side output
        # (the full entity pool reads it as a row; nothing else consumes the stash, so plain
        # attribute assignment keeps the return contract untouched).
        self.last_global_out = tokens[:, self._total_tokens - 1, :]
        extra_out = tokens[:, self._total_tokens:, :] if extra is not None else None
        return our_team_out, their_team_out, extra_out


def _event_reference_cells(event_window: torch.Tensor,
                           species_ids: torch.Tensor) -> torch.Tensor:
    """Tier H-C (`_EDGE_R_CELL`): the [B, N, 12, 2] `[is_actor, is_target]` reference cells.

    Species-num equality between an event row's actor/target columns and the 12 mon slots,
    SIDE-GATED — the actor lives on the event's own side, the target on the opposite side, so
    a mirror species on the other team can never false-link. PAD rows (valid=0) contribute
    nothing. Pure (no parameters) so the identity is testable without a forward."""
    ev = event_window
    sm = species_ids.float()[:, None, :]                                       # [B,1,12]
    ss = torch.cat([torch.ones(TEAM_SIZE), -torch.ones(TEAM_SIZE)]) \
        .to(ev.device)[None, None, :]                                          # [1,1,12]
    valid = (ev[:, :, 18] > 0.5)[:, :, None]
    actor, tgt, eside = ev[:, :, 1:2], ev[:, :, 3:4], ev[:, :, 2:3]
    is_actor = (actor == sm) & (actor > 0) & (eside == ss) & valid             # [B,N,12]
    is_target = (tgt == sm) & (tgt > 0) & (-eside == ss) & valid
    return torch.stack([is_actor.float(), is_target.float()], dim=-1)


class EventSeats(torch.nn.Module):
    """gen3_event_window_v1 (Tier H-B, design_history_entity.md §3 H-B) — the CONSUMER of the
    obs event-window block: the last-N event records become N extra TRUNK SEATS, appended
    through the `extra` seam AFTER the E3/E4/E5 seats (position-stable — every front-indexed
    seat slice keeps its meaning). "The sequential residue becomes queryable": a mon token can
    attend over what happened, in order, with time as CONTENT (the log-saturated recency
    scalar), never as a lag-indexed weight.

    Per record: the type id and status id go through OWN small embeddings; actor/target
    species and the move id go through the SHARED tables (one representation everywhere —
    the design's move-latent rule, satisfied with the embedding the E3 seats also start
    from); the 12 outcome/time scalars ride raw. One Linear projects the concat to d_model
    (the per-type-projection budget deferred until the usage audit says the types need to
    diverge). Seats take TOKEN_TYPE_HISTORY (the E5 precedent: no token-type table growth,
    so the flag stays state_dict-minimal) plus a learned event_marker distinguishing them
    from the TurnDelta frames they will eventually replace. PAD rows (valid=0) are
    key-masked. OFF builds nothing — byte-identical baseline."""

    _KIND_EMB = 16
    _STATUS_EMB = 8
    _CANT_EMB = 6
    _FAINT_EMB = 5
    _ITEMTR_EMB = 4
    _N_SCALARS = 13          # side + [mag, hit, miss, fail, crit, eff×4, we_first] + ago + forced

    def __init__(self, layout: Dict[str, Any]):
        super().__init__()
        from agents.observation.constants import N_EVENT_TYPES
        from agents.observation.gen3_effects import CANT_DIM_LIVE
        from agents.observation.constants import N_ITEM_TRANSITIONS
        from agents.battle.turn_view import FAINT_CAUSE_DIM
        self.n = layout['event_window_n']
        self.kind_emb = torch.nn.Embedding(N_EVENT_TYPES, self._KIND_EMB)
        self.status_emb = torch.nn.Embedding(8, self._STATUS_EMB)
        # gen3_frame_deletion_v1: the cant reason (col 19), 0 = not a CANT row. Sized from
        # CANT_DIM_**LIVE**, not CANT_DIM — the archive vocabulary is frozen at 12 to keep
        # TURN_DELTA_DIM at 159 for the prober, while the live one grew to 13 with `damp`
        # (gen3_damp_cant_v1). Sizing from the frozen number would have clamped damp's id of 13
        # onto 12 = `truant`: a SILENT collision, one blocked Explosion read as loafing.
        self.cant_emb = torch.nn.Embedding(CANT_DIM_LIVE + 1, self._CANT_EMB)
        # gen3_event_semantics_v1: cols 20/21. Both sized +1 from the SAME vocabularies the
        # encoder writes ids from (FAINT_CAUSE_VOCAB / ITEM_TR_*), so extending either widens
        # both sides at once rather than silently clamping a new id onto an existing row.
        self.faint_emb = torch.nn.Embedding(FAINT_CAUSE_DIM + 1, self._FAINT_EMB)
        self.itemtr_emb = torch.nn.Embedding(N_ITEM_TRANSITIONS, self._ITEMTR_EMB)
        in_dim = (self._KIND_EMB + 2 * layout['species_embedding_dim'] +
                  layout['move_embedding_dim'] + self._STATUS_EMB + self._CANT_EMB +
                  self._FAINT_EMB + self._ITEMTR_EMB + self._N_SCALARS)
        self.proj = torch.nn.Linear(in_dim, D_MODEL)
        self.norm = torch.nn.LayerNorm(D_MODEL)
        self.event_marker = torch.nn.Parameter(torch.randn(1, 1, D_MODEL) * 0.02)

    def forward(self, ev: torch.Tensor, embeddings: 'Embeddings'
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ev [B, N, EVENT_TOKEN_DIM] → (seat tokens [B, N, D_MODEL], pad [B, N] bool)."""
        kind = ev[:, :, 0].long().clamp(min=0)
        actor = ev[:, :, 1].long().clamp(min=0)
        target = ev[:, :, 3].long().clamp(min=0)
        move = ev[:, :, 4].long().clamp(min=0)
        status = ev[:, :, 15].long().clamp(min=0, max=7)
        cant = ev[:, :, 19].long().clamp(min=0, max=self.cant_emb.num_embeddings - 1)
        faint = ev[:, :, 20].long().clamp(min=0, max=self.faint_emb.num_embeddings - 1)
        itemtr = ev[:, :, 21].long().clamp(min=0, max=self.itemtr_emb.num_embeddings - 1)
        # the 12 raw scalars: side(2) mag(5) outcome(6:9) crit(9) eff(10:14) we_first(14)
        # ago(16) forced(17) — columns 2,5..14,16,17
        scalars = torch.cat([ev[:, :, 2:3], ev[:, :, 5:15], ev[:, :, 16:18]], dim=-1)
        row = torch.cat([
            self.kind_emb(kind),
            embeddings.species_embedding(actor),
            embeddings.species_embedding(target),
            embeddings.move_embedding(move),
            self.status_emb(status),
            self.cant_emb(cant),
            self.faint_emb(faint),
            self.itemtr_emb(itemtr),
            scalars,
        ], dim=-1)
        tokens = self.norm(self.proj(row)) + self.event_marker
        pad = ev[:, :, 18] < 0.5                                     # valid=0 ⇒ masked
        return tokens, pad
