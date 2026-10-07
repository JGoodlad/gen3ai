"""The DamageOperator's SPEED PHYSICS — P(we act first at equal priority) under `--speed-physics on`
(gen3_speed_physics_v1, config v143; architecture audit F7b).

A MIXIN, not a module: `DamageOperator` inherits this class (no parameters live here). Every op site that prices
"who moves first" — the incoming per-mon outspeed, the outgoing `p_outspeed`, the outgoing matrix, the pair
outcome's paralysis severity, C1's boost Δ-outspeed, the status consequence's paralysis Δ, the V edge — keeps
its `off` code byte-for-byte and, under `on`, builds its inputs here and calls ONE rule
(`move_order.p_first_same_priority`), so no site can fork the physics:

* OUR six speeds are EXACT (`move_order.gen3_speed_stat` → stage → paralysis, Showdown's integer arithmetic) —
  our spread, item and condition are known;
* THEIR six speeds are DISCRETE: the species' Smogon spreads mixture over the Speed STAT (`SPEED_MIX`,
  `belief_tables.build_species_speed_mix`, gen3_speed_mixture_v1 — real Speed investment is lumpy, max or none,
  which the Gaussian it replaced could not hold), each support point through the SAME exact stage (active row) and
  paralysis arithmetic as ours (`gen3_final_speed`); under X5 fixed_mass's OTHER pass the slot reads the TAIL's
  mixture (``P_tail @ SPEED_MIX``). The learned spread belief is NOT read here (the ``spread_belief`` arguments
  are accepted for the sites' signatures and ignored);
* Quick Claw — FORMAT-GATED (`move_order.quick_claw_live`): BANNED in gen3ou, the format the model plays, so
  `quick_claw_live` is False and both holders read 0 (the term vanishes; nothing is computed from the item
  belief or the prior). In a format that allows it: ours is our item; theirs is revealed exactly, else the ITEM
  BELIEF's P(Quick Claw) for the slot (`op.stash.item_qc_prob`) or, without one, the Smogon species prior
  (`SPECIES_QC_PRIOR`).

Stages live on the ACTIVE rows only (a switch resets them in gen 3); paralysis on every row (it persists on the
bench). A site's hypothetical (C1's post-setup stage, C5's inherited stages, a paralysis it is pricing) is an
explicit override argument, never a second formula.
"""
from typing import Any, Callable, Optional, Tuple, TYPE_CHECKING

import torch

from agents.model.damage_op_layout import (_BS_SPE, _COND_PAR_IDX, _NAT_SPE)
from agents.model.move_order import gen3_final_speed, gen3_speed_stat, p_first_same_priority
from agents.observation.constants import (POKEMON_CONDITION_OFFSET, POKEMON_SPREAD_DIM,
                                          POKEMON_SPREAD_OFFSET, TEAM_SIZE)

if TYPE_CHECKING:  # no runtime import — `ctx` is only ever passed in, never constructed here
    from agents.model.extractor_ctx import ExtractorContext


class DamageOperatorSpeed:

    if TYPE_CHECKING:
        # MIXIN (see `damage_op_pairwise.DamageOperatorPairwise`): every `self.*` is owned by `DamageOperator`.
        def __getattr__(self, name: str) -> Any: ...
        speed_physics: bool
        quick_claw_live: bool
        qc_item_num: int
        _boost_stages: Callable[..., Tuple[torch.Tensor, ...]]
        _x5_avg: Callable[..., torch.Tensor]
        BASE_STATS: torch.Tensor
        SPECIES_SPREAD_PRIOR: torch.Tensor
        SPECIES_QC_PRIOR: torch.Tensor
        SPEED_MIX: torch.Tensor

    def _our_speeds_exact(self, ctx: 'ExtractorContext', stage_rows: Optional[torch.Tensor] = None,
                          para: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Our six mons' EXACT final speeds ``[B,6]``. ``stage_rows`` ``[B,6]`` (None: our active's live speed
        stage on its row, 0 on the bench); ``para`` ``[B,6]`` 0/1 (None: each mon's own paralysis)."""
        B = ctx.batch_size
        ar = torch.arange(B, device=ctx.device)
        base = self.BASE_STATS[ctx.species_ids[:, :TEAM_SIZE]][..., _BS_SPE]                 # [B,6]
        spread = ctx.pokemon_part[:, :TEAM_SIZE,
                                  POKEMON_SPREAD_OFFSET:POKEMON_SPREAD_OFFSET + POKEMON_SPREAD_DIM]
        stat = gen3_speed_stat(base, spread[..., _BS_SPE] * 31.0, spread[..., 6 + _BS_SPE] * 252.0,
                               spread[..., 13 + _NAT_SPE])                                    # [B,6]
        if stage_rows is None:
            stage_rows = torch.zeros_like(stat)
            stage_rows[ar, ctx.our_active_idx] = self._boost_stages(ctx.our_ctx_raw)[4]
        if para is None:
            para = ctx.pokemon_part[:, :TEAM_SIZE, POKEMON_CONDITION_OFFSET + _COND_PAR_IDX]
        out: torch.Tensor = gen3_final_speed(stat, stage_rows, para)
        return out

    def _opp_speed_mix(self, ctx: 'ExtractorContext',
                       para: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Their six mons' DISCRETE final-speed distributions ``(final [B,6,V], cum [B,6,V+1])``: every lattice
        speed v through their stage (active row) and paralysis (`gen3_final_speed`, non-decreasing in v), and the
        cumulative mixture weights (float64, leading 0). ``para`` ``[B,6]`` overrides the observed paralysis (a
        hypothetical). Under X5 fixed_mass's OTHER pass an OTHER slot reads the tail's mixture."""
        B = ctx.batch_size
        ar = torch.arange(B, device=ctx.device)
        w = self.SPEED_MIX[ctx.species_ids[:, TEAM_SIZE:2 * TEAM_SIZE]]                       # [B,6,V]
        x5 = self.stash.x5
        if x5 is not None and x5.override is not None:
            tail = x5.species_probs.to(w.dtype) @ self.SPEED_MIX                               # [B,6,V]
            w = torch.where(x5.override.unsqueeze(-1), tail, w)
        stage = torch.zeros(B, TEAM_SIZE, device=ctx.device)
        stage[ar, ctx.opp_active_local] = self._boost_stages(ctx.opp_ctx_raw)[4]
        if para is None:
            para = ctx.pokemon_part[:, TEAM_SIZE:2 * TEAM_SIZE, POKEMON_CONDITION_OFFSET + _COND_PAR_IDX]
        lattice = torch.arange(w.shape[-1], device=ctx.device, dtype=stage.dtype)
        final: torch.Tensor = gen3_final_speed(lattice.view(1, 1, -1), stage.unsqueeze(-1), para.unsqueeze(-1))
        cum = torch.cat([torch.zeros_like(w[..., :1], dtype=torch.float64),
                         w.to(torch.float64).cumsum(-1)], dim=-1)                              # [B,6,V+1]
        return final, cum

    def _our_quick_claw(self, ctx: 'ExtractorContext') -> torch.Tensor:
        """``[B,6]`` 1 where our mon holds Quick Claw (our items are known); 0 where the format bans it."""
        if not self.quick_claw_live:
            return torch.zeros(ctx.item_ids.shape[0], TEAM_SIZE, device=ctx.device)
        return (ctx.item_ids[:, :TEAM_SIZE] == self.qc_item_num).float()

    def _opp_quick_claw(self, ctx: 'ExtractorContext') -> torch.Tensor:
        """``[B,6]`` P(their mon holds Quick Claw): 1 / 0 once the item is revealed, else the item belief's
        P(Quick Claw) (``op.stash.item_qc_prob``) or the Smogon species prior — the CB rule's exactness gate. 0
        where the format bans it (gen3ou): a banned item has no holder, whatever the stale prior says."""
        if not self.quick_claw_live:
            return torch.zeros(ctx.item_ids.shape[0], TEAM_SIZE, device=ctx.device)
        item = ctx.item_ids[:, TEAM_SIZE:2 * TEAM_SIZE]                                        # [B,6]
        belief = self.stash.item_qc_prob
        prior = belief if belief is not None else self.SPECIES_QC_PRIOR[ctx.species_ids[:, TEAM_SIZE:2 * TEAM_SIZE]]
        revealed = (item == self.qc_item_num).float()
        unrevealed = (item == 0).float()
        p_qc: torch.Tensor = revealed + (1.0 - revealed) * unrevealed * prior
        return p_qc

    def _p_first(self, ours: torch.Tensor, final: torch.Tensor, cum: torch.Tensor,
                 our_qc: torch.Tensor, opp_qc: torch.Tensor) -> torch.Tensor:
        """THE op's P(we act first at equal priority) under `on` — `move_order.p_first_same_priority`. ``ours``
        ``[B,n,Q]`` against ``final`` / ``cum`` ``[B,n,·]`` (n of their mons); the Quick Claw terms broadcast
        against the ``[B,n,Q]`` result."""
        p: torch.Tensor = p_first_same_priority(ours, final, cum, our_qc, opp_qc)
        return p

    def _p_first_vs_six(self, ctx: 'ExtractorContext', ours: torch.Tensor, our_qc: torch.Tensor,
                        para: Optional[torch.Tensor] = None) -> torch.Tensor:
        """``[B,6]`` P(one mon of ours — speed ``ours`` ``[B]``, Quick Claw ``our_qc`` ``[B]`` — acts before each of
        their six); ``para`` overrides their paralysis."""
        final, cum = self._opp_speed_mix(ctx, para=para)
        return self._p_first(ours[:, None, None].expand(-1, TEAM_SIZE, 1), final, cum,
                             our_qc[:, None, None], self._opp_quick_claw(ctx).unsqueeze(-1)).squeeze(-1)

    def _p_first_pairs(self, ctx: 'ExtractorContext') -> torch.Tensor:
        """``[B,6,6]`` P(our mon i acts before their mon j) for every pair — our six EXACT speeds (paralysis on
        every row, the stage on the active's) against each of their mixtures."""
        final, cum = self._opp_speed_mix(ctx)
        ours = self._our_speeds_exact(ctx)                                                      # [B,6]
        p = self._p_first(ours[:, None, :].expand(-1, TEAM_SIZE, -1), final, cum,
                          self._our_quick_claw(ctx)[:, None, :], self._opp_quick_claw(ctx).unsqueeze(-1))
        return p.transpose(1, 2)                                                                # [B,i,j]

    def _p_first_active_at_stage(self, ctx: 'ExtractorContext', spread_belief: Optional[torch.Tensor],
                                 stage: torch.Tensor) -> torch.Tensor:
        """``[B,6]`` P(our ACTIVE, at speed stage ``stage`` ``[B]``, acts before each of their six) — C1's current
        and post-setup worlds."""
        B = ctx.batch_size
        ar = torch.arange(B, device=ctx.device)
        rows = torch.zeros(B, TEAM_SIZE, device=ctx.device)
        rows[ar, ctx.our_active_idx] = stage
        ours = self._our_speeds_exact(ctx, stage_rows=rows)[ar, ctx.our_active_idx]          # [B]
        return self._p_first_vs_six(ctx, ours, self._our_quick_claw(ctx)[ar, ctx.our_active_idx])

    def _p_first_vs_opp_active(self, ctx: 'ExtractorContext', spread_belief: Optional[torch.Tensor],
                               ours: torch.Tensor, our_qc: torch.Tensor) -> torch.Tensor:
        """P(each of ``ours`` (``[B]`` or ``[B,6]``) acts before their ACTIVE at equal priority."""
        B = ctx.batch_size
        ar = torch.arange(B, device=ctx.device)
        final, cum = self._opp_speed_mix(ctx)
        loc = ctx.opp_active_local
        f_a, c_a = final[ar, loc].unsqueeze(1), cum[ar, loc].unsqueeze(1)                    # [B,1,·]
        qc_a = self._opp_quick_claw(ctx)[ar, loc][:, None, None]                              # [B,1,1]
        q = ours.reshape(B, 1, -1)
        p = self._p_first(q, f_a, c_a, our_qc.reshape(B, 1, -1), qc_a)                        # [B,1,Q]
        return p.reshape(ours.shape)
