"""The pointer-native action head: request-slot alignment, EntityMoveSeats, and per-action scoring from entity tokens.

Split out of `features_extractor.py` 2026-08-16 (one responsibility per file); that module
re-exports every name here, so historical import paths still resolve.
"""
from agents.model.extractor_ctx import ExtractorContext, TOKEN_TYPE_OUR_MOVE, TOKEN_TYPE_THEIR_THREAT
import torch
from typing import Any, Optional, Tuple
from agents.observation.constants import (
    TEAM_SIZE,
)
from agents.model.arch_constants import (MOVE_NET_HIDDEN,
    MOVE_LATENT_DIM,
    POINTER_HIDDEN,
    D_MODEL,
)


def _request_order_move_tokens(move_tokens_all: torch.Tensor,
                               ctx: 'ExtractorContext') -> Tuple[torch.Tensor, torch.Tensor]:
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
    so every existing absolute slice (team tokens, history, global) is untouched:

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

    The closed feasibility spike (`entity_spike_benchmark.py`, git history) priced the seat growth
    at ~+0.19 ms B=1 for
    the full ~50-seat layout — dispatch-bound, not FLOP-bound. NO edges yet (Stage 2); the seats
    enter attention purely as content. Input projections are ordinary trainable Linears (NOT
    zero-init — new-information inputs, the `history_proj`/`global_proj` convention, not the
    residual-injection one)."""

    def __init__(self, topk_seats: int = 0, tail_seats: bool = False):
        super().__init__()
        self.topk_seats = int(topk_seats)
        # gen3_entity_tail_seats_v1 (E5): 6 per-opp-mon TAIL-THREAT seats — the truncation
        # insurance. Every candidate consumer top-Ks (E4 seats, D3/D4 edges) and DROPS the tail;
        # these seats carry [p_tail, worst_phys, worst_spec, revealed] of the beyond-rank-K belief
        # mass per opp slot, so a rare-but-lethal candidate below rank K is at least SUMMARIZED
        # (the bimodal-miss finding: truncation loses candidates entirely). Deliberately NO new
        # token-type row (the table growing 6 → 7 would change EVERY model's state_dict and break
        # loading in-generation checkpoints into newer code): tail seats reuse
        # TOKEN_TYPE_THEIR_THREAT + a dedicated learned `tail_marker` added here.
        self.tail_seats = bool(tail_seats)
        self.tail_proj = torch.nn.Linear(4, D_MODEL) if self.tail_seats else None
        self.tail_marker = (torch.nn.Parameter(torch.randn(1, 1, D_MODEL) * 0.02)
                            if self.tail_seats else None)
        # gen3_edge_bias_trunk_v1: the per-forward candidate selection (idx, w), stashed so the D3
        # edge bias prices the SAME K moves the seats represent. None until the first E4 forward.
        self.last_cand: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
        self.move_seat_proj = torch.nn.Linear(MOVE_NET_HIDDEN[1], D_MODEL)
        self.threat_seat_proj = (torch.nn.Linear(MOVE_LATENT_DIM + 3, D_MODEL)
                                 if self.topk_seats > 0 else None)

    @property
    def n_seats(self) -> int:
        """Total seat count: 4 E3 move seats + K E4 threat seats + (E5 tail seats when on)."""
        return 4 + self.topk_seats + (TEAM_SIZE if self.tail_seats else 0)

    def seat_types(self, device: torch.device) -> torch.Tensor:
        """Per-seat token-type ids [n_seats] for the transformer's type embedding. Tail seats reuse
        THEIR_THREAT (distinctness comes from `tail_marker` — see __init__)."""
        n_threat = self.topk_seats + (TEAM_SIZE if self.tail_seats else 0)
        return torch.cat([
            torch.full((4,), TOKEN_TYPE_OUR_MOVE, dtype=torch.long, device=device),
            torch.full((n_threat,), TOKEN_TYPE_THEIR_THREAT, dtype=torch.long, device=device),
        ])

    def forward(self, tok_req: torch.Tensor, move_valid: torch.Tensor, ctx: 'ExtractorContext',
                damage_op: Any, move_belief_logits: Optional[torch.Tensor],
                latent_table: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """→ `(seats [B, 4+K, d_model], pad [B, 4+K] bool)` (pad True = masked, the key-mask sense)."""
        seats = [self.move_seat_proj(tok_req)]                                # [B,4,D] (invalid = zeros)
        pads = [move_valid < 0.5]                                             # [B,4]
        if self.topk_seats > 0:
            assert move_belief_logits is not None and latent_table is not None, (
                "E4 threat seats need the pre-transformer move-belief logits + the move latent table "
                "(guaranteed by the __init__ gate: damage_op + move_latent, and by the tiered order)"
            )
            # gen3_op_candidate_dedup_v1: reuse the op forward's own candidate-weight build —
            # the identical computation on the identical inputs, cleared at op-forward entry so
            # a stale batch is unrepresentable (None ⇒ refine computes standalone).
            idx, w = damage_op.refine_candidates(ctx, move_belief_logits, k=self.topk_seats,
                                                 w_all=damage_op.last_w_all)              # [B,K]
            # gen3_edge_bias_trunk_v1: stash the candidate selection so the D3 edge bias prices the
            # SAME K moves the seats represent (seat c and bias row c must name the same move).
            self.last_cand = (idx, w)
            has_opp = ctx.hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, -1].any(dim=1)            # [B] bool
            hdr = torch.cat([
                latent_table[idx],                                            # [B,K,32] typed-HP-aware identity
                w[:, :, None],                                                # belief weight (differentiable)
                damage_op.MOVE_ACCURACY[idx][:, :, None],
                damage_op.MOVE_PHYS[idx][:, :, None],
            ], dim=2)
            # `threat_seat_proj` / `tail_proj` are built only when their seat count is > 0, which
            # is exactly the branch guard here — an invariant `__init__` owns, not the type.
            e4 = self.threat_seat_proj(hdr) * has_opp[:, None, None].float()  # type: ignore[misc]  # zeroed when no opp active
            seats.append(e4)
            pads.append(~has_opp[:, None].expand(-1, self.topk_seats))
        if self.tail_seats:
            # E5: per opp mon j, the beyond-top-K tail of ITS OWN composed posterior. K = the same
            # entity_topk_seats the E4 seats use (one truncation definition). worst_* are BOUND-ish
            # scores (w · BP/150 · acc, split by category) — defender-independent by design (a
            # token, not an edge); attention composes them with the mon tokens.
            assert move_belief_logits is not None
            w_all = torch.sigmoid(move_belief_logits) * damage_op.HP_CAND_MASK[None, None, :]  # [B,6,M]
            K = max(self.topk_seats, 1)
            topv = w_all.topk(K, dim=-1).values                                   # [B,6,K]
            in_top = w_all >= topv[..., -1:].clamp(min=1e-9)                      # [B,6,M] (ties incl.)
            tail_w = w_all * (~in_top).float()                                    # beyond-rank-K mass
            p_tail = tail_w.sum(-1).clamp(max=1.0)                                # [B,6]
            score = tail_w * (damage_op.MOVE_BP[None, None, :] / 150.0)                     * damage_op.MOVE_ACCURACY[None, None, :]
            phys = damage_op.MOVE_PHYS[None, None, :]
            worst_phys = (score * phys).amax(-1)                                  # [B,6]
            worst_spec = (score * (1.0 - phys)).amax(-1)
            revealed = 1.0 - ctx.opp_believed_mask.float()                        # [B,6]
            has_opp_t = ctx.hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, -1].any(dim=1)
            cells = torch.stack([p_tail, worst_phys, worst_spec, revealed], dim=-1)  # [B,6,4]
            e5 = (self.tail_proj(cells) + self.tail_marker) * has_opp_t[:, None, None].float()  # type: ignore[misc]
            seats.append(e5)
            pads.append(~has_opp_t[:, None].expand(-1, TEAM_SIZE))
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
