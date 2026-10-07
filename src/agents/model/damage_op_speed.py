"""The DamageOperator's SPEED PHYSICS — P(we act first at equal priority) under `--speed-physics on`
(gen3_speed_physics_v1, config v143; architecture audit F7b).

A MIXIN, not a module: `DamageOperator` inherits this class (no parameters live here). Every op site that prices
"who moves first" — the incoming per-mon outspeed, the outgoing `p_outspeed`, the outgoing matrix, the pair
outcome's paralysis severity, C1's boost Δ-outspeed, the status consequence's paralysis Δ, the V edge — keeps
its `off` code byte-for-byte and, under `on`, builds its inputs here and calls ONE rule
(`move_order.p_first_same_priority`), so no site can fork the physics:

* OUR six speeds are EXACT (`move_order.gen3_speed_stat` → stage → paralysis, Showdown's integer arithmetic) —
  our spread, item and condition are known;
* THEIR six speeds are the spread BELIEF (the believed speed — the prior mean when the spread belief is off —
  and the Smogon prior's per-species spread, `SPECIES_SPREAD_PRIOR[..., spe, 1]`), scaled by their stage and
  paralysis (`belief_speed_scale`); under X5 fixed_mass's OTHER pass both read the tail averages (`_x5_avg`);
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

from agents.model.damage_op_layout import (_BS_SPE, _COND_PAR_IDX, _NAT_SPE, _SB_SPE)
from agents.model.move_order import (belief_speed_scale, gen3_final_speed, gen3_speed_stat,
                                     p_first_same_priority)
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

    def _opp_speeds_belief(self, ctx: 'ExtractorContext', spread_belief: Optional[torch.Tensor],
                           para: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Their six mons' believed FINAL speed ``(mean [B,6], spread [B,6])``: the spread belief's speed (the
        Smogon prior mean without one) and the prior's per-species spread, both × their stage (active row) and
        paralysis factor. ``para`` ``[B,6]`` overrides the observed paralysis (a hypothetical)."""
        B = ctx.batch_size
        ar = torch.arange(B, device=ctx.device)
        species = ctx.species_ids[:, TEAM_SIZE:2 * TEAM_SIZE]                                  # [B,6]
        prior = self.SPECIES_SPREAD_PRIOR[species, _SB_SPE]                                    # [B,6,2]
        mu = spread_belief[..., _SB_SPE] if spread_belief is not None else prior[..., 0]
        mu = self._x5_avg(mu, "spe")                                                           # X5: OTHER's average
        sigma = self._x5_avg(prior[..., 1], "spe_std")
        stage = torch.zeros_like(mu)
        stage[ar, ctx.opp_active_local] = self._boost_stages(ctx.opp_ctx_raw)[4]
        if para is None:
            para = ctx.pokemon_part[:, TEAM_SIZE:2 * TEAM_SIZE, POKEMON_CONDITION_OFFSET + _COND_PAR_IDX]
        scale = belief_speed_scale(stage, para)
        return mu * scale, sigma * scale

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

    def _p_first(self, ours: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor,
                 our_qc: torch.Tensor, opp_qc: torch.Tensor) -> torch.Tensor:
        """THE op's P(we act first at equal priority) under `on` — `move_order.p_first_same_priority`. All
        five broadcast (the site picks the rows: our active vs their active, our six vs their active, …)."""
        p: torch.Tensor = p_first_same_priority(ours, mu, sigma, our_qc, opp_qc)
        return p

    def _p_first_active_at_stage(self, ctx: 'ExtractorContext', spread_belief: Optional[torch.Tensor],
                                 stage: torch.Tensor) -> torch.Tensor:
        """``[B,6]`` P(our ACTIVE, at speed stage ``stage`` ``[B]``, acts before each of their six) — C1's current
        and post-setup worlds."""
        B = ctx.batch_size
        ar = torch.arange(B, device=ctx.device)
        rows = torch.zeros(B, TEAM_SIZE, device=ctx.device)
        rows[ar, ctx.our_active_idx] = stage
        ours = self._our_speeds_exact(ctx, stage_rows=rows)[ar, ctx.our_active_idx]          # [B]
        mu, sigma = self._opp_speeds_belief(ctx, spread_belief)                                 # [B,6]
        return self._p_first(ours[:, None], mu, sigma, self._our_quick_claw(ctx)[ar, ctx.our_active_idx][:, None],
                             self._opp_quick_claw(ctx))

    def _p_first_vs_opp_active(self, ctx: 'ExtractorContext', spread_belief: Optional[torch.Tensor],
                               ours: torch.Tensor, our_qc: torch.Tensor) -> torch.Tensor:
        """P(each of ``ours`` (``[B]`` or ``[B,6]``) acts before their ACTIVE at equal priority."""
        B = ctx.batch_size
        ar = torch.arange(B, device=ctx.device)
        mu, sigma = self._opp_speeds_belief(ctx, spread_belief)
        loc = ctx.opp_active_local
        mu_a, sd_a, qc_a = mu[ar, loc], sigma[ar, loc], self._opp_quick_claw(ctx)[ar, loc]   # [B]
        if ours.dim() == 2:
            mu_a, sd_a, qc_a = mu_a[:, None], sd_a[:, None], qc_a[:, None]
        return self._p_first(ours, mu_a, sd_a, our_qc, qc_a)
