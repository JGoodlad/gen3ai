"""The DamageOperator's PAIRWISE edge-family physics — the per-(mon, mon) cell producers
behind the EdgeBias grid (d1/d2/d3, status, boost, recovery, speed, trap, entry, baton,
schedule, protect, and the believed-attacker helpers they share).

A MIXIN, not a module: `DamageOperator` inherits this class, so every method still runs as
`self.pairwise_*` against the op's own buffers and stashes — no parameters live here, the
state_dict is byte-identical, and the file split changes nothing the heads see (the production
sha probe pins it). Split out of `damage_op.py` 2026-08-17 (one responsibility per file).
"""
import torch
from typing import Any, Callable, Optional, Tuple, TYPE_CHECKING
from agents.observation.constants import (
    TEAM_SIZE,
    TOXIC_STAGE_MAX,
    POKEMON_SPREAD_OFFSET,
    POKEMON_SPREAD_DIM,
    POKEMON_CONDITION_OFFSET,
    POKEMON_COUNTER_OFFSET,
)
from agents.model.damage_tables import (LEECH_SEED_CAT,
                                        _SLP_CAT as _SLP_STATUS_CAT)
from agents.model.arch_constants import (  # noqa: F401  (re-export)
    ROLE_TOKEN_SIZE,
    PROJECTION_DIM,
    MOVE_NET_HIDDEN,
    MOVE_LATENT_HIDDEN,
    MOVE_LATENT_DIM,
    ROLE_ENCODER_HIDDEN,
    NET_ARCH,
    D_MODEL,
    TRANSFORMER_N_LAYERS,
    TRANSFORMER_N_HEADS,
    TRANSFORMER_FFN_DIM,
)

# The flat-block layout (every _DMG_* constant, OpTensors, decode_damage_block) lives in
# `damage_op_layout.py` (split 2026-08-16); re-imported EXPLICITLY because this module is
# the historical import surface and the operator uses nearly all of it.
from agents.model.damage_op_layout import (  # noqa: F401
    OpTensors, _BOOSTS_DIM, _BS_ATK, _BS_DEF, _BS_HP, _BS_SPA, _BS_SPD, _BS_SPE, _COND_BRN_IDX,
    _COND_PAR_IDX, _COND_SLP_IDX, _DMG_CB, _DMG_CB_PER_MON, _DMG_CHANNEL_FEATS, _DMG_CHIP_CAP,
    _DMG_CRIT_CAP, _DMG_CRIT_P, _DMG_EFFECT, _DMG_IDX_OUTSPEED, _DMG_IDX_PHYS_ACC,
    _DMG_IDX_PHYS_CRIT, _DMG_IDX_PHYS_HIGH, _DMG_IDX_PHYS_LOW, _DMG_IDX_PHYS_PKO,
    _DMG_IDX_PROVENANCE, _DMG_IDX_SPEC_ACC, _DMG_IDX_SPEC_CRIT, _DMG_IDX_SPEC_HIGH,
    _DMG_IDX_SPEC_LOW, _DMG_IDX_SPEC_PKO, _DMG_IMX_CELL, _DMG_IMX_HDR_ACC, _DMG_IMX_HDR_EFFECT,
    _DMG_IMX_HDR_PHYS, _DMG_IMX_HDR_SEC, _DMG_IMX_HDR_W, _DMG_IMX_HEADER, _DMG_IMX_IDX_CRIT,
    _DMG_IMX_IDX_HIGH, _DMG_IMX_IDX_LOW, _DMG_IMX_IDX_MULT, _DMG_IMX_IDX_PKO, _DMG_IMX_IDX_STATUS,
    _DMG_N_CHANNELS, _DMG_OAX, _DMG_OAX_IDX_CRIT, _DMG_OAX_IDX_HIGH, _DMG_OAX_IDX_LOW,
    _DMG_OAX_IDX_PKO, _DMG_OAX_N_MOVES, _DMG_OAX_PER_MON, _DMG_OAX_PER_MOVE, _DMG_OMX,
    _DMG_OMX_CELL, _DMG_OMX_IDX_CRIT, _DMG_OMX_IDX_HIGH, _DMG_OMX_IDX_LOW, _DMG_OMX_IDX_MULT,
    _DMG_OMX_IDX_PKO, _DMG_OUTGOING, _DMG_OUT_N_MOVES, _DMG_OUT_PER_MOVE, _DMG_OUT_SEC,
    _DMG_PARA_SPEED, _DMG_PER_MON, _DMG_REFINE_K, _DMG_ROLL_MIN, _DMG_SPEED_SCALE,
    _DMG_SPEED_STD_K, _DMG_STATUS, _DMG_STATUS_N_MOVES, _DMG_STATUS_REFINE, _DMG_TOPK_DEFAULT_K,
    _ELECTRIC_TIDX, _FIRE_TIDX, _IMMOBILIZE_STATUS_CATS, _LEECH_SEED_CTX_SLOT, _NAT_ATK, _NAT_DEF, _NAT_SPA,
    _NAT_SPD, _NAT_SPE, _N_OUT_SECONDARY, _OUT_SEC_COLS, _OUT_SEC_DROP, _OUT_SEC_KEEP,
    _PAIR_REDUCE_N_CHANNELS, _PTR_MOVE_CELL, _PTR_SWITCH_CELL_IN, _SB_ATK, _SB_DEF, _SB_SPA,
    _SB_SPD, _SB_SPE, _SECONDARY_MAJOR_N, _SECONDARY_TO_STATUS_CAT, _SUBSTITUTE_CTX_IDX,
    _TypeEncoder, _VOLATILE_SLOTS, _WATER_TIDX, _dmg_imx_dim, decode_damage_block,
)

from agents.model.index_max import max_by_index
from agents.model.op_reduction import believed_reduce, noisy_or
from agents.model.damage_kinds import (beatup_base_def, beatup_party_opp, beatup_swap, gather_beatup,
                                       gather_bp, gather_nonformula, nonformula_rolls, override_rolls)

if TYPE_CHECKING:  # no runtime import — `ctx` is only ever passed in, never constructed here
    from agents.model.extractor_ctx import ExtractorContext



def toxic_next_tick(counter_cell: torch.Tensor) -> torch.Tensor:
    """The fraction of MAX HP a badly-poisoned mon's NEXT residual tick costs, from the observation's toxic cell
    (``POKEMON_COUNTER_OFFSET + 1`` = ``min(stage, TOXIC_STAGE_MAX) / TOXIC_STAGE_MAX``, ``gen3_toxic_stage_scale_v1``).

    Gen 3 `tox.onResidual` (deps/pokemon-showdown data/conditions.ts): ``if (stage < 15) stage++`` BEFORE the damage
    ``floor(maxhp/16) · stage`` — so with ``n`` ticks taken the next tick is ``min(n + 1, 15) / 16``: a fresh Toxic
    reads 1/16, the 9th tick 9/16, and the 15th-and-later 15/16 (NOT 16/16). The cell is the engine's stage over its
    true cap, so ``n`` is read exactly (it saturated at 8 before: every tick from the 9th on read as the 8th). The `g`
    ledger (`pairwise_schedule`) and `--eot-residual` (`eot_residual.py`) both decode the cell with this one function.
    (It lives here, not in `damage_op_layout`: the forward's torch ops may only run from `selection_sites.FORWARD_MODULES`.)"""
    cap = float(TOXIC_STAGE_MAX)
    return torch.clamp(counter_cell * cap + 1.0, max=cap) / 16.0


class DamageOperatorPairwise:

    if TYPE_CHECKING:
        # MIXIN, not a module (see the module docstring): every `self.*` below is owned by
        # `DamageOperator`, which composes this class WITH `torch.nn.Module` — so at RUNTIME
        # those names resolve either through the composed class's `__init__` or through
        # `Module.__getattr__` (the damage tables are registered in a loop, so they exist
        # only dynamically). Mirroring `Module.__getattr__`'s signature gives the checker the
        # same view the interpreter has. TYPE_CHECKING-only: no runtime effect.
        def __getattr__(self, name: str) -> Any: ...
        # Scalars + the stash container the composed class sets in `__init__` (see
        # `damage_op.DamageOperator`). Declared so arithmetic against them stays typed.
        baton_num: int
        curse_num: int
        rest_num: int
        toxic_num: int
        rest_sleep_eb: float
        rest_sleep_noeb: float
        sleep_free_eb: float
        sleep_free_noeb: float
        @property
        def last_topk_idx(self) -> Optional[torch.Tensor]: ...
        # Methods the COMPOSED class (`damage_op.DamageOperator`) owns and both mixins call.
        # Declared as callables rather than re-stated signatures: the definition is one file away
        # and mypy checks it there; this only stops the reads decaying to `Any`.
        _rolls: Callable[..., Tuple[torch.Tensor, ...]]
        _damage_rolls: Callable[..., Tuple[torch.Tensor, ...]]
        _crit_p: Callable[..., Optional[torch.Tensor]]
        _crit_p_at: Callable[..., Optional[torch.Tensor]]
        ko_exact: bool
        eot_rule: Any
        _boost_mult: Callable[..., torch.Tensor]
        _boost_stages: Callable[..., Tuple[torch.Tensor, ...]]
        _weather_mult: Callable[..., torch.Tensor]
        _sport_mult: Callable[..., torch.Tensor]
        _field_bp_mult: Callable[..., torch.Tensor]
        _chan_max: Callable[..., torch.Tensor]
        _p_outspeed: Callable[..., torch.Tensor]
        _opp_candidate_weights: Callable[..., torch.Tensor]
        # Methods the SIBLING mixin (`damage_op_blocks.DamageOperatorBlocks`) owns — the pairwise
        # cells are built on the block kernels, so the call goes sideways across the composition.
        discrete_outgoing_status: Callable[..., torch.Tensor]
        opp_ability_view: Callable[..., Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
        _known_or_prior: Callable[..., torch.Tensor]
        unrevealed_species_probs: Callable[..., torch.Tensor]
        _outgoing_matrix: Callable[..., torch.Tensor]
        _outgoing_attacker_matrix: Callable[..., torch.Tensor]
        _incoming_rolls: Callable[..., Tuple[torch.Tensor, ...]]
        # The damage TABLES (`damage_tables.build_damage_buffers`) are registered in a LOOP, so
        # they exist only dynamically; declaring them keeps every read a `Tensor` instead of the
        # `Any` the `__getattr__` above would hand back.
        ABILITY_DAMAGE_MULT: torch.Tensor
        ABILITY_IS_EARLYBIRD: torch.Tensor
        ABILITY_IS_LEVITATE: torch.Tensor
        ABILITY_TRAP: torch.Tensor
        BASE_STATS: torch.Tensor
        CHART: torch.Tensor
        CURSE_BOOSTS: torch.Tensor
        HP_CAND_MASK: torch.Tensor
        MOVE_ACCURACY: torch.Tensor
        MOVE_BOOST_HP_COST: torch.Tensor
        MOVE_BEATUP: torch.Tensor
        MOVE_BP: torch.Tensor
        MOVE_FIXED_DAMAGE: torch.Tensor
        MOVE_TARGET_HP_FRAC: torch.Tensor
        MOVE_ENDEAVOR: torch.Tensor
        MOVE_BP_FLAIL: torch.Tensor
        MOVE_BP_HP_SCALED: torch.Tensor
        MOVE_HEAL_FRACTION: torch.Tensor
        MOVE_INFLICTS_STATUS: torch.Tensor
        MOVE_PHYS: torch.Tensor
        MOVE_SELF_BOOSTS: torch.Tensor
        MOVE_STATUS_CAT: torch.Tensor
        MOVE_TYPE_IDX: torch.Tensor
        MOVE_WEATHER_HEAL: torch.Tensor
        SPECIES_EARLYBIRD_PRIOR: torch.Tensor
        SPECIES_SPREAD_PRIOR: torch.Tensor
        SPECIES_TRAP_PRIOR: torch.Tensor
        SPECIES_TYPE: torch.Tensor
        TYPE_IS_FLYING: torch.Tensor
        TYPE_IS_GHOST: torch.Tensor
        TYPE_IS_PHYS: torch.Tensor
        TYPE_IS_STEEL: torch.Tensor

    def _x5_avg(self, gathered: torch.Tensor, field: str) -> torch.Tensor:
        """X5 fixed_mass, an OTHER-MODE pass only (`hypothesis_tokens.other_roster`): the slots holding
        OTHER read the tail-AVERAGED table ``field`` (``att_base`` [B,6,6], ``spe`` [B,6])
        in place of the per-slot ``gathered`` value. Any other pass (and the belief family off): ``gathered`` itself."""
        x5 = self.stash.x5
        if x5 is None or x5.override is None:
            return gathered
        ov = x5.override.view(*x5.override.shape, *([1] * (gathered.dim() - 2)))
        return torch.where(ov, getattr(x5, field).to(gathered.dtype), gathered)

    def _x5_stab(self, is_stab: torch.Tensor, mty_k: torch.Tensor) -> torch.Tensor:
        """``is_stab`` [B,6,K] with the OTHER slots' STAB the tail's EXPECTED indicator E[1(type(m) ∈
        types(s))] (OTHER-mode pass only; otherwise unchanged)."""
        x5 = self.stash.x5
        if x5 is None or x5.override is None:
            return is_stab
        e = torch.gather(x5.has_type.to(is_stab.dtype), 2, mty_k.long())             # [B,6,K]
        return torch.where(x5.override.unsqueeze(-1), e, is_stab)

    def pairwise_outgoing(self, ctx: 'ExtractorContext',
                          spread_belief: Optional[torch.Tensor] = None,
                          boost_delta: Optional[torch.Tensor] = None,
                          species_probs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """gen3_edge_bias_trunk_v1 (D1): the per-(our move k, opp mon d) outgoing cells for the edge-bias
        delivery — `[B, _DMG_OUT_N_MOVES, TEAM_SIZE, 6]` = `[low, high, crit, pko, type_mult, revealed]`.
        A pure RESHAPE of `_outgoing_matrix`'s flat block (the validated v34 physics: request-ordered
        moves == E3 seat order == action 6+k, CB/boost/burn attacker; unrevealed defenders priced by the
        expected-latent read — gen3_unrevealed_outgoing_prior_v1, `species_probs` overrides the usage
        prior), so the bias and the head concat can never disagree on a value. `boost_delta` [B,5] prices
        the C1 hypothetical post-setup world (None = the current world, byte-identical)."""
        flat = self._outgoing_matrix(ctx, spread_belief, boost_delta, species_probs)      # [B, _DMG_OMX]
        n_cells = _DMG_OUT_N_MOVES * TEAM_SIZE * _DMG_OMX_CELL
        cells = flat[:, :n_cells].reshape(-1, _DMG_OUT_N_MOVES, TEAM_SIZE, _DMG_OMX_CELL)
        revealed = flat[:, n_cells:n_cells + TEAM_SIZE]                                   # [B,6]
        rev = revealed[:, None, :, None].expand(-1, _DMG_OUT_N_MOVES, -1, 1)
        return torch.cat([cells, rev], dim=-1)                                            # [B,4,6,6]

    def _setup_deltas(self, ctx: 'ExtractorContext') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Shared C1/C1b source of truth: per-request-slot setup stage deltas `[B,4,5]`
        (atk,def,spa,spd,spe) + the `is_boost` mask `[B,4]`. Table rows (`MOVE_SELF_BOOSTS`,
        the ~17 pure setup moves) PLUS the runtime NON-GHOST Curse branch (+1 atk/+1 def/−1 spe
        — CurseLax/Curse-Registeel, owner-prioritized): type-CONDITIONAL, so it can't live in
        the type-blind per-move table; resolved from the USER'S live types (the obs encoder's
        `is_boost` convention). A Ghost user's Curse (50% max HP for a target curse) stays an
        unpriced zero row. ONE helper so the outgoing and incoming kernels can never disagree
        on what a setup slot does."""
        ar = torch.arange(ctx.batch_size, device=ctx.device)
        deltas = self.MOVE_SELF_BOOSTS[ctx.our_active_req_move_ids]                  # [B,4,5]
        _at1 = ctx.type1_ids[ar, ctx.our_active_idx]
        _at2 = ctx.type2_ids[ar, ctx.our_active_idx]
        _user_ghost = (self.TYPE_IS_GHOST[_at1] + self.TYPE_IS_GHOST[_at2]) > 0.0    # [B]
        _curse_gate = ((ctx.our_active_req_move_ids == self.curse_num)
                       & ~_user_ghost[:, None]).float()                              # [B,4]
        deltas = deltas + _curse_gate[:, :, None] * self.CURSE_BOOSTS[None, None, :]
        # Belly Drum's fail gate (the +12-clamps-to-max row): below half HP the move FAILS —
        # the whole consequence row zeroes rather than showing a free +6.
        hp_cost = self.MOVE_BOOST_HP_COST[ctx.our_active_req_move_ids]               # [B,4]
        hp_frac = ctx.hp_and_active[ar, ctx.our_active_idx, 0]
        usable = ((hp_cost <= 0) | (hp_frac[:, None] > hp_cost)).float()             # [B,4]
        deltas = deltas * usable[:, :, None]
        return deltas, (deltas.abs().sum(-1) > 0).float(), hp_cost * usable

    def pairwise_boost(self, ctx: 'ExtractorContext',
                       spread_belief: Optional[torch.Tensor] = None,
                       base: Optional[torch.Tensor] = None,
                       species_probs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """gen3_edge_bias_trunk_v1 (C1): the first DAMAGE-consequence edge — "what does clicking the
        setup move at request slot k DO for me". Re-runs the validated `_outgoing_matrix` kernel
        under the HYPOTHETICAL post-boost stat stages (the C4-over-G composition pattern scaled to
        the damage kernel: one input changes, everything else — gates, CB, burn, screens, spread
        belief — rides along, so the delta can never disagree with D1's pricing of either world)
        and emits per-(E3 setup-move seat k, opp mon j) DELTA cells
        `[B, _DMG_OUT_N_MOVES, TEAM_SIZE, 4]` (`_EDGE_C1_CELL`) = `[is_boost, d_best_high,
        d_best_pko, d_outspeed]`:

          * `is_boost` — slot k is a priced setup move: the ~17 declarative `selfBoosts` moves
            (`MOVE_SELF_BOOSTS`, gen3_setup_moves_v1 — Swords Dance/Dragon Dance/Calm Mind/
            Agility/…) PLUS the runtime NON-GHOST Curse branch (`CURSE_BOOSTS` — type-
            conditional, resolved from the user's live types; a Ghost user's Curse is a
            different move and stays a zero row). Belly Drum (HP-cost callback: needs an
            hp_cost cell channel + a fails-below-half gate + the C1b incoming-at-halved-HP
            re-run — recorded TODO, niche) and Defense Curl / the evasion moves stay all-zero
            rows, so their consequence is simply unpriced, never wrong.
          * `d_best_high` / `d_best_pko` — the change in our active's BEST (move-collapsed)
            mean-roll damage / P(KO) vs opp mon j after slot k's stage deltas.
          * `d_outspeed` — the change in P(our active outspeeds opp mon j) after the spe delta
            (the `pairwise_speed` recipe at the active row, WITH stage folding both worlds —
            here the stage IS the signal, unlike V's no-boost coarse convention).

        A pure-DEFENSIVE setup move (Iron Defense, Amnesia, the def/spd halves of Bulk Up /
        Calm Mind) fires `is_boost` with ~0 deltas — the INCOMING-direction delta is the
        declared C1b follow-up. `base` lets the caller hand over an already-computed
        `pairwise_outgoing(ctx, spread_belief)` (the D1 cells) so the current world isn't
        recomputed; the 4 hypothetical worlds are 4 kernel re-runs (correct-by-construction
        beats reconstructing the affine/threshold physics from the base cells)."""
        B, device = ctx.batch_size, ctx.device
        ar = torch.arange(B, device=device)
        opp = slice(TEAM_SIZE, 2 * TEAM_SIZE)
        deltas, is_boost, hp_cost = self._setup_deltas(ctx)                          # [B,4,5], [B,4], [B,4]
        # gen3_unrevealed_outgoing_prior_v1: compute the Species-Clause marginal ONCE and hand it to
        # every kernel re-run below (pure function of ctx — byte-identical, just not recomputed 5×).
        sp_probs = self.unrevealed_species_probs(ctx, species_probs)
        if base is None:
            base = self.pairwise_outgoing(ctx, spread_belief, species_probs=sp_probs)   # [B,4,6,6]
        base_high = max_by_index(base[..., 1], dim=1)                                         # [B,6]
        base_pko = max_by_index(base[..., 3], dim=1)                                          # [B,6]
        # --- speed: our ACTIVE's real-spread spe (the pairwise_speed recipe) × its live stage ---
        d_base = self.BASE_STATS[ctx.species_ids[:, :TEAM_SIZE]]                     # [B,6,6]
        spread = ctx.pokemon_part[:, :TEAM_SIZE,
                                  POKEMON_SPREAD_OFFSET:POKEMON_SPREAD_OFFSET + POKEMON_SPREAD_DIM]
        iv = spread[..., 0:6] * 31.0
        ev = spread[..., 6:12] * 252.0
        nat = spread[..., 13:18]
        our_spe_all = (2.0 * d_base[..., _BS_SPE] + iv[..., _BS_SPE] + ev[..., _BS_SPE] / 4.0
                       + 5.0) * nat[..., _NAT_SPE]     # FIXED 2026-08-06: was SpD (the V GIGO)
        our_para = ctx.pokemon_part[:, :TEAM_SIZE, POKEMON_CONDITION_OFFSET + _COND_PAR_IDX]
        act_spe = (our_spe_all * (1.0 - 0.75 * our_para))[ar, ctx.our_active_idx]    # [B]
        cur_spe_stage = self._boost_stages(ctx.our_ctx_raw)[4]                       # [B]
        opp_species = ctx.species_ids[:, opp]
        if spread_belief is not None:
            opp_spe = spread_belief[..., _SB_SPE]
        else:
            opp_spe = 2.0 * self.BASE_STATS[opp_species][..., _BS_SPE] + 31.0 + 5.0  # neutral 0-EV
        opp_spe = self._x5_avg(opp_spe, "spe")                                      # X5: OTHER's averaged speed
        opp_para = ctx.pokemon_part[:, opp, POKEMON_CONDITION_OFFSET + _COND_PAR_IDX]
        opp_spe = opp_spe * (1.0 - 0.75 * opp_para)                                  # [B,6]
        if self.speed_physics:
            # gen3_speed_physics_v1: our active's EXACT speed at its live stage (+ slot k's speed delta below) vs
            # their six speed mixtures — the one rule (`damage_op_speed`).
            p_base = self._p_first_active_at_stage(ctx, spread_belief, cur_spe_stage)  # [B,6]
        else:
            p_base = self._p_outspeed((act_spe * self._boost_mult(cur_spe_stage))[:, None],
                                      opp_spe)                                       # [B,6]
        revealed = (~ctx.opp_believed_mask).float()                                  # [B,6]
        alive_j = (ctx.hp_and_active[:, opp, 0] > 0).float()
        _x5 = self.stash.x5
        if _x5 is not None:     # X5 (U3 part 3): every live mon's speed is priced (a hypothesis's species',
            revealed, alive_j = torch.ones_like(_x5.alive), _x5.alive   # OTHER's tail average); alive by addressability
        our_alive = (ctx.hp_and_active[ar, ctx.our_active_idx, 0] > 0).float()
        spd_gate = revealed * alive_j * our_alive[:, None]                           # [B,6]
        # --- the 4 hypothetical worlds: slot k's deltas, everything else identical ---
        rows = []
        for k in range(_DMG_OUT_N_MOVES):
            dk = deltas[:, k]                                                        # [B,5]
            boosted = self.pairwise_outgoing(ctx, spread_belief, boost_delta=dk,
                                             species_probs=sp_probs)                # [B,4,6,6]
            d_high = max_by_index(boosted[..., 1], dim=1) - base_high                         # [B,6]
            d_pko = max_by_index(boosted[..., 3], dim=1) - base_pko
            if self.speed_physics:
                p_k = self._p_first_active_at_stage(ctx, spread_belief, cur_spe_stage + dk[:, 4])
            else:
                p_k = self._p_outspeed(
                    (act_spe * self._boost_mult(cur_spe_stage + dk[:, 4]))[:, None], opp_spe)
            d_spd = (p_k - p_base) * spd_gate
            ib = is_boost[:, k:k + 1].expand_as(d_high)
            hc = hp_cost[:, k:k + 1].expand_as(d_high)         # Belly Drum's half-max-HP price
            rows.append(torch.stack([ib, d_high, d_pko, d_spd, hc], dim=-1))         # [B,6,5]
        cells = torch.stack(rows, dim=1)                                             # [B,4,6,5]
        return cells * is_boost[:, :, None, None]

    def _believed_attackers(self, ctx: 'ExtractorContext', move_belief_logits: torch.Tensor,
                            k_cand: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                  torch.Tensor, torch.Tensor, torch.Tensor,
                                                  torch.Tensor, torch.Tensor,
                                                  Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
                                                  torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Shared C1b/C3 attacker block (the D4 recipe with the ACTIVE column KEPT): per opp mon
        j its top-`k_cand` most-believed candidates from ITS OWN slot of the composed posterior
        (selection detached, weights differentiable), de-timid offense, revealed+alive gate.
        → (w_k, bp_k, mty_k, phys_k, acc_k [B,6,K]; atk_j, spa_j, att_gate [B,6]; nf_k = 3×[B,6,K]
        (fixed, target_frac, endeavor); atk_cur_j [B,6]; bu_k [B,6,K] the 0/1 Beat Up flag; crit_k [B,6,K] the
        candidates' crit chance under `--ko-ramp exact`, else None). `bp_k` is the
        EFFECTIVE BP (each mon's HP resolved) and `nf_k` / `atk_cur_j` feed `damage_kinds.nonformula_rolls` —
        gen3_nonformula_damage_v1; `bu_k` feeds `damage_kinds.beatup_swap` — gen3_beatup_exact_v1.

        X5 fixed_mass (U3 part 3, `self.stash.x5`): EVERY live opponent mon attacks — a hypothesis as
        its species (`ctx` is the hypothesis context), "alive" from `opp_addressable`; each mon's
        candidates are the first K of ITS one order over its fixed-mass move presence (no topk), the
        weights that presence (DETACHED, M10), so every max over them is §9 M2 = C's presence-scaled
        max."""
        opp = slice(TEAM_SIZE, 2 * TEAM_SIZE)
        _x5 = self.stash.x5
        if _x5 is None:
            w_all = torch.sigmoid(move_belief_logits) * self.HP_CAND_MASK[None, None, :]  # [B,6,M]
            K = min(int(k_cand), w_all.shape[-1])
            topk_idx = w_all.detach().topk(K, dim=-1).indices                            # [B,6,K]
            w_k = w_all.gather(-1, topk_idx)                                             # diff'able
        else:
            K = min(int(k_cand), _x5.move_w.shape[-1])
            topk_idx = _x5.move_order[..., :K]                                           # [B,6,K] one order
            w_k = _x5.move_w.gather(-1, topk_idx)                                        # presence (detached)
        hp_j = ctx.hp_and_active[:, opp, 0]                                          # [B,6]
        bp_k = gather_bp(self, topk_idx, hp_j[:, :, None])                           # [B,6,K] effective
        mty_k = self.MOVE_TYPE_IDX[topk_idx]
        phys_k = self.MOVE_PHYS[topk_idx]
        acc_k = self.MOVE_ACCURACY[topk_idx]
        nf_k = gather_nonformula(self, topk_idx)                                     # 3×[B,6,K]
        a_base = self._x5_avg(self.BASE_STATS[ctx.species_ids[:, opp]], "att_base")  # [B,6,6] (X5: OTHER = E_tail)
        off_const = 31.0 + 252.0 / 4.0 + 5.0
        atk_j = (2.0 * a_base[..., _BS_ATK] + off_const) * 1.1                       # de-timid
        spa_j = (2.0 * a_base[..., _BS_SPA] + off_const) * 1.1
        atk_cur_j = hp_j * (2.0 * a_base[..., _BS_HP] + 31.0 + 110.0)                # neutral max HP
        att_gate: torch.Tensor = ((1.0 - ctx.opp_believed_mask.float())
                                  * (hp_j > 0).float())                              # [B,6]
        if _x5 is not None:
            att_gate = _x5.alive
        bu_k = gather_beatup(self, topk_idx)                                         # [B,6,K]
        # gen3_endstate_facts_v1 (`--ko-ramp exact`): each candidate's crit chance (None under `ramp`).
        crit_k = self._crit_p(topk_idx)                                              # [B,6,K] | None
        return w_k, bp_k, mty_k, phys_k, acc_k, atk_j, spa_j, att_gate, nf_k, atk_cur_j, bu_k, crit_k

    def _attacker_total(self, ctx: 'ExtractorContext',
                        move_belief_logits: torch.Tensor) -> Optional[torch.Tensor]:
        """gen3_op_reduction_principled_v1: each opp mon's TOTAL move presence over the whole move axis `[B,6]`
        — the denominator of its α (`op_reduction.believed_reduce`), so a candidate cut off the kernels' top-K is
        unpriced mass, never renormalised away. None under `--op-reduction max` (the legacy hard max needs none).
        X5: the roster's fixed-mass presence (Σ = 4 on a full set); else the sigmoid posterior the kernels gather."""
        if not self.op_principled:
            return None
        _x5 = self.stash.x5
        if _x5 is not None:
            tot: torch.Tensor = _x5.move_w.sum(dim=-1)                                   # [B,6]
            return tot
        return (torch.sigmoid(move_belief_logits) * self.HP_CAND_MASK[None, None, :]).sum(dim=-1)

    def _active_defender(self, ctx: 'ExtractorContext') -> Tuple[torch.Tensor, ...]:
        """Consequence-kernel defender block (C2; C1b/C3 keep inline variants — C1b needs the
        UNFOLDED stats to fold per-world stages itself): OUR ACTIVE's real-spread stats with
        its CURRENT def/spd stages folded (named indices — the v58 rule). → (def_c, spd_c,
        maxhp, cur_hp [B]; at1, at2 [B] long; amul [B,T])."""
        ar = torch.arange(ctx.batch_size, device=ctx.device)
        d_base = self.BASE_STATS[ctx.species_ids[ar, ctx.our_active_idx]]            # [B,6]
        spr = ctx.pokemon_part[ar, ctx.our_active_idx,
                               POKEMON_SPREAD_OFFSET:POKEMON_SPREAD_OFFSET + POKEMON_SPREAD_DIM]
        iv = spr[:, 0:6] * 31.0; ev = spr[:, 6:12] * 252.0; nat = spr[:, 13:18]
        def_stat = (2.0 * d_base[:, _BS_DEF] + iv[:, _BS_DEF] + ev[:, _BS_DEF] / 4.0
                    + 5.0) * nat[:, _NAT_DEF]
        spd_stat = (2.0 * d_base[:, _BS_SPD] + iv[:, _BS_SPD] + ev[:, _BS_SPD] / 4.0
                    + 5.0) * nat[:, _NAT_SPD]
        _ba, b_def, _bs2, b_spd, _be = self._boost_stages(ctx.our_ctx_raw)
        def_c = def_stat * self._boost_mult(b_def)
        spd_c = spd_stat * self._boost_mult(b_spd)
        maxhp = 2.0 * d_base[:, _BS_HP] + iv[:, _BS_HP] + ev[:, _BS_HP] / 4.0 + 110.0
        cur_hp = ctx.hp_and_active[ar, ctx.our_active_idx, 0] * maxhp
        at1 = ctx.type1_ids[ar, ctx.our_active_idx]
        at2 = ctx.type2_ids[ar, ctx.our_active_idx]
        amul = self.ABILITY_DAMAGE_MULT[ctx.ability1_ids[ar, ctx.our_active_idx]]
        return def_c, spd_c, maxhp, cur_hp, at1, at2, amul

    def pairwise_status_consequence(self, ctx: 'ExtractorContext',
                                    move_belief_logits: torch.Tensor,
                                    spread_belief: Optional[torch.Tensor] = None,
                                    k_cand: int = 6,
                                    species_probs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """gen3_edge_bias_trunk_v1 (C2): what LANDING our status move DOES — the consequence
        world behind S1's "will it land". Per (E3 status-move seat k, opp mon j) the cells
        `[is_status, land, d_their_outspeed, d_in_phys_high, d_sched, d_in_all_slp,
        e_slp_free_turns]` `[B, 4, TEAM_SIZE, 7]`:

          * `is_status` — slot k inflicts a MAJOR status (cats par/brn/frz/slp/psn — Leech Seed
            is deliberately NOT here: its tick is the G ledger's fact, its landing S1's).
          * `land` — the validated v27/v37 per-pair landing probability (the same
            `discrete_outgoing_status(per_pair=True)` physics S1 delivers; duplicated into the
            cell so C2 is self-contained when S1 is off).
          * `d_their_outspeed` — PARALYSIS: Δ P(our active outspeeds mon j) with j's speed
            ×0.25 (≥ 0 — T-Wave makes the matchup faster for us; the true gen3ou reason to
            click it). Believed/neutral opp speed, named indices, current our-stage folded.
          * `d_in_phys_high` — BURN: Δ of mon j's worst believed PHYSICAL hit on our active
            with its Atk halved (≤ 0 — WoW as damage control; special candidates untouched).
          * `d_sched` — the residual tick the landing adds: brn/psn −1/8 flat, TOXIC its TRUE
            first tick −1/16 (the ramp thereafter is the G ledger's live fact via the public
            toxic counter — owner-prioritized 2026-08-06; par/slp read 0).
          * `d_in_all_slp` / `e_slp_free_turns` — SLEEP's consequence: their whole believed
            threat suspended (−worst hit, any category) for E[free turns] from the VERIFIED
            sleep hazard tables (`sleep_belief.expected_free_turns` — 2.5 no-EB / 1.0 Early
            Bird, marginalised per mon over the revealed-exact/Smogon-prior P(EB); /4-normed).

        Deltas are RAW (not multiplied by `land`) — the decorrelated provide-the-facts form:
        the head composes consequence × probability itself, exactly like pko vs accuracy."""
        B, device, eps = ctx.batch_size, ctx.device, 1e-6
        ar = torch.arange(B, device=device)
        opp = slice(TEAM_SIZE, 2 * TEAM_SIZE)
        ids = ctx.our_active_req_move_ids
        sidx = self.MOVE_STATUS_CAT[ids]                                             # [B,4]
        inflicts = self.MOVE_INFLICTS_STATUS[ids]
        is_status = inflicts * ((sidx > 0) & (sidx != LEECH_SEED_CAT)).float()       # [B,4]
        is_par = (sidx == 1).float()
        is_brn = (sidx == 2).float()
        is_slp = (sidx == _SLP_STATUS_CAT).float()
        is_tox = (ids == self.toxic_num).float()                                     # cat 5 splits here
        is_psn = ((sidx == 5).float() - is_tox).clamp(min=0.0)                       # plain poison
        land = self.discrete_outgoing_status(ctx, per_pair=True)[..., 0]             # [B,4,6]
        # --- paralysis: their speed ×0.25 → Δ P(we outspeed) (the pairwise_speed recipe) ---
        d_base = self.BASE_STATS[ctx.species_ids[:, :TEAM_SIZE]]
        spread = ctx.pokemon_part[:, :TEAM_SIZE,
                                  POKEMON_SPREAD_OFFSET:POKEMON_SPREAD_OFFSET + POKEMON_SPREAD_DIM]
        iv = spread[..., 0:6] * 31.0; ev = spread[..., 6:12] * 252.0; nat = spread[..., 13:18]
        our_spe_all = (2.0 * d_base[..., _BS_SPE] + iv[..., _BS_SPE] + ev[..., _BS_SPE] / 4.0
                       + 5.0) * nat[..., _NAT_SPE]
        our_para = ctx.pokemon_part[:, :TEAM_SIZE, POKEMON_CONDITION_OFFSET + _COND_PAR_IDX]
        act_spe = (our_spe_all * (1.0 - 0.75 * our_para))[ar, ctx.our_active_idx]    # [B]
        act_spe = act_spe * self._boost_mult(self._boost_stages(ctx.our_ctx_raw)[4])
        opp_species = ctx.species_ids[:, opp]
        if spread_belief is not None:
            opp_spe = spread_belief[..., _SB_SPE]
        else:
            opp_spe = 2.0 * self.BASE_STATS[opp_species][..., _BS_SPE] + 31.0 + 5.0
        opp_para = ctx.pokemon_part[:, opp, POKEMON_CONDITION_OFFSET + _COND_PAR_IDX]
        if self.speed_physics:
            # gen3_speed_physics_v1: their speed mixture now vs with them paralysed (the stage stays on their
            # active), our active's EXACT speed — the one rule (`damage_op_speed`).
            _ours = self._our_speeds_exact(ctx)[ar, ctx.our_active_idx]                # [B]
            _qc_u = self._our_quick_claw(ctx)[ar, ctx.our_active_idx]
            p_now = self._p_first_vs_six(ctx, _ours, _qc_u)                              # [B,6]
            p_par = self._p_first_vs_six(ctx, _ours, _qc_u, para=torch.ones_like(opp_para))   # [B,6]
        else:
            p_now = self._p_outspeed(act_spe[:, None], opp_spe * (1.0 - 0.75 * opp_para))
            p_par = self._p_outspeed(act_spe[:, None], opp_spe * 0.25)               # [B,6]
        d_outspeed = (p_par - p_now)[:, None, :] * is_par[:, :, None]                # [B,4,6]
        # --- burn: mon j's worst believed PHYSICAL hit on our active, Atk halved ---
        (w_k, bp_k, mty_k, phys_k, acc_k, atk_j, spa_j,
         att_gate, nf_k, atk_cur_j, bu_k, crit_k) = self._believed_attackers(ctx, move_belief_logits, k_cand)
        w_tot = self._attacker_total(ctx, move_belief_logits)                        # [B,6] | None (max)
        def_c, spd_c, maxhp, cur_hp, at1, at2, amul = self._active_defender(ctx)
        # gen3_beatup_exact_v1: the opp party's Σ base Atk / hit count vs OUR active's BASE Def.
        bu_S, bu_N = beatup_party_opp(self, ctx, species_probs)                      # [B], [B]
        bu_def = beatup_base_def(self, ctx.species_ids[ar, ctx.our_active_idx])      # [B]
        eff = (torch.gather(self.CHART[at1][:, None, :].expand(B, TEAM_SIZE, -1), 2, mty_k)
               * torch.gather(self.CHART[at2][:, None, :].expand(B, TEAM_SIZE, -1), 2, mty_k)
               * torch.gather(amul[:, None, :].expand(B, TEAM_SIZE, -1), 2, mty_k))  # [B,6,K]
        is_stab = ((mty_k == ctx.type1_ids[:, opp][:, :, None])
                   | (mty_k == ctx.type2_ids[:, opp][:, :, None])).float()
        reflect, ls = ctx.screen_feature[:, 0:1], ctx.screen_feature[:, 2:3]
        screen = (1.0 - 0.5 * (reflect[:, :, None] * phys_k
                               + ls[:, :, None] * (1.0 - phys_k)))                   # [B,6,K]
        # gen3_nonformula_damage_v1: a non-formula move counts as damaging (Seismic Toss is a hit) and its
        # rolls are the declared damage — Atk-independent, so burn moves it by exactly 0.
        nf_any = (nf_k[0] + nf_k[1] + nf_k[2]) > 0                                   # [B,6,K]
        phys_mask = phys_k * ((bp_k > 0) | nf_any).float()
        dmg_mask = ((bp_k > 0) | nf_any).float()
        nf = nonformula_rolls(nf_k, cur_hp[:, None, None], maxhp[:, None, None], atk_cur_j[:, :, None],
                              eff, acc_k, eps)

        def _worst(atk_mult: float, mask: torch.Tensor) -> torch.Tensor:
            A = phys_k * atk_j[:, :, None] * atk_mult + (1.0 - phys_k) * spa_j[:, :, None]
            D = phys_k * def_c[:, None, None] + (1.0 - phys_k) * spd_c[:, None, None]
            A, D, plus2 = beatup_swap(bu_k, A, D, bu_S[:, None, None], bu_def[:, None, None],
                                      bu_N[:, None, None])
            core = 42.0 * bp_k * A / (D + eps) / 50.0 + plus2
            dmg_ns = core * (1.0 + 0.5 * is_stab) * eff * 0.925 * (bp_k > 0).float()
            high, _l, _c, _k = override_rolls(self._rolls(dmg_ns, screen, maxhp[:, None, None],
                                                          cur_hp[:, None, None], acc_k, eps,
                                                          crit_p=crit_k), nf)  # [B,6,K]
            return believed_reduce(w_k * high * mask, w_tot)                        # [B,6]

        d_in_phys = ((_worst(0.5, phys_mask) - _worst(1.0, phys_mask))[:, None, :]
                     * is_brn[:, :, None])                                           # [B,4,6] ≤0
        # --- the residual-tick consequence: brn/psn flat −1/8; TOXIC lands at its TRUE first
        # tick −1/16 (the ramp thereafter is the G ledger's live fact once the counter exists —
        # owner-prioritized 2026-08-06, was a flat −1/8 that over-priced turn 1 and hid the ramp) ---
        d_sched = (-(1.0 / 8.0) * (is_brn + is_psn) - (1.0 / 16.0) * is_tox)
        d_sched = d_sched[:, :, None].expand(-1, -1, TEAM_SIZE)                      # [B,4,6]
        # --- SLEEP (owner-prioritized): the consequence of landing it — their whole believed
        # threat SUSPENDED (d_in_all = −worst hit of ANY category) for an EXPECTED number of
        # free turns from the VERIFIED hazard tables, Early-Bird-marginalised per mon (revealed
        # ability → exact; else the Smogon prior). Both RAW, decorrelated from `land`. ---
        # gen3_op_ability_known_v1: revealed = the `known` flag (an unrevealed id1 is the top-1 PRIOR).
        opp_ability, abl_known, opp_sp = self.opp_ability_view(ctx)                  # [B,6] ×3
        p_eb = self._known_or_prior(self.ABILITY_IS_EARLYBIRD[opp_ability],
                                    self.SPECIES_EARLYBIRD_PRIOR[opp_sp], abl_known)  # [B,6]
        e_free = (self.sleep_free_noeb
                  + (self.sleep_free_eb - self.sleep_free_noeb) * p_eb) / 4.0        # [B,6] (/max 4)
        e_slp_free = e_free[:, None, :] * is_slp[:, :, None]                         # [B,4,6]
        d_in_all = (-_worst(1.0, dmg_mask))[:, None, :] * is_slp[:, :, None]         # [B,4,6] ≤0
        our_alive = (ctx.hp_and_active[ar, ctx.our_active_idx, 0] > 0).float()
        row_gate = (is_status[:, :, None] * att_gate[:, None, :]
                    * our_alive[:, None, None])                                      # [B,4,6]
        ib = is_status[:, :, None].expand(-1, -1, TEAM_SIZE)
        cells = torch.stack([ib, land, d_outspeed, d_in_phys, d_sched,
                             d_in_all, e_slp_free], dim=-1)
        return cells * row_gate[..., None]                                           # [B,4,6,7]

    def pointer_intent_status_operands(self, ctx: 'ExtractorContext',
                                       move_belief_logits: torch.Tensor,
                                       spread_belief: Optional[torch.Tensor] = None,
                                       k_cand: int = 6,
                                       c2_cells: Optional[torch.Tensor] = None,
                                       species_probs: Optional[torch.Tensor] = None,
                                       other_u: Optional[torch.Tensor] = None,
                                       ) -> Tuple[torch.Tensor, ...]:
        """gen3_intent_move_cell_v1 (G3): the RAW operands for the alpha-conditioned c2
        re-delivery through the pointer MOVE cell (`agents.model.intent_move_cell` weights them
        by the published alpha at T2 — alpha does not exist when this runs, the same
        T2-object/T1-producer split as `last_pair_cells`).

        Returns ``(base [B,4,4], d_burn_k [B,K], d_slp_k [B,K], is_brn [B,4], is_slp [B,4])``:

          * ``base`` — the k-INDEPENDENT c2 consequence columns, gathered at the opp ACTIVE (the
            mon our status move hits when they stay): `[is_status, d_their_outspeed, d_sched,
            e_slp_free_turns]`. `land` is deliberately absent — the move cell already carries the
            richer unified `_status_landing` p_land/known vs the same recipient. Reuses the c2
            edge grid when the caller already built it (`c2_cells`), else computes it fresh —
            the identical function either way.
          * ``d_burn_k`` — per seat-candidate k, the burn damage-control delta on OUR active:
            `high_k(atk × 0.5) − high_k(atk)` (≤ 0; exactly 0 for special/fixed candidates since
            only the physical attack stat moves). Computed by the SAME `_damage_rolls` kernel and
            the SAME attacker pricing as the op's incoming block (believed spread when on, else
            de-timid; offensive boosts; an existing burn), so the delta is consistent with the
            `high` the model already reads.
          * ``d_slp_k`` — per candidate k, sleep's suspended threat: `−high_k` (any category).
          * ``is_brn`` / ``is_slp`` — our request slots' status-category masks (c2's definitions).

        The candidate axis is the op's OWN top-K (`last_topk_idx`, real move nums) — the axis
        `intent_axis_alignment_test` pins element-wise to alpha's seats — so alignment is by
        construction rather than by convention. Fails loud when the top-K stash is missing (no
        `damage_matrices_incoming`/`damage_topk_k`): silently substituting a different candidate
        selection would mis-weight every term (the named `op move-order` bug class).

        ``other_u`` [B,M] (X5 U4, fixed_mass only — `FixedMassMoves.other_u`): append OTHER_move's
        column to ``d_burn_k`` / ``d_slp_k`` (→ [B,K+1]): the same kernel on the FULL candidate axis,
        contracted with the renormalised tail (E_tail, never a zero column)."""
        B, device, eps = ctx.batch_size, ctx.device, 1e-6
        ar = torch.arange(B, device=device)
        nums = self.last_topk_idx                                                    # [B,K] move NUMS
        _mix = self.stash.seat_mix         # X5 fixed_mass: price the EXTENDED seat axis, contract at the end
        if _mix is not None:
            nums = self.stash.seat_ext_idx
        if nums is None:
            raise RuntimeError(
                "pointer_intent_status_operands needs the op's top-K candidate stash "
                "(last_topk_idx) to align with alpha's seats, and none was recorded. "
                "intent_move_cell requires damage_topk_k>0 (and the incoming matrix that "
                "computes it).")
        has_opp = ctx.hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, -1].any(dim=1).float()  # [B]
        our_alive = (ctx.hp_and_active[ar, ctx.our_active_idx, 0] > 0).float()          # [B]
        # --- the k-independent c2 columns vs the opp ACTIVE (c2's own row_gate already folds
        # our_alive + the active's revealed/alive gate; has_opp guards the no-active gather) ---
        c2 = c2_cells if c2_cells is not None else self.pairwise_status_consequence(
            ctx, move_belief_logits, spread_belief, k_cand=k_cand, species_probs=species_probs)
        row = c2[ar, :, ctx.opp_active_local, :]                                     # [B,4,7]
        base = torch.stack([row[..., 0], row[..., 2], row[..., 4], row[..., 6]],
                           dim=-1) * has_opp[:, None, None]                          # [B,4,4]
        # --- our request slots' status categories (c2's definitions) ---
        ids = ctx.our_active_req_move_ids                                            # [B,4]
        sidx = self.MOVE_STATUS_CAT[ids]
        is_brn = (sidx == 2).float()
        is_slp = (sidx == _SLP_STATUS_CAT).float()
        # --- attacker = opp active, priced EXACTLY as the incoming block prices it (believed
        # spread when on, else de-timid ×1.1; offensive stages; an existing burn) ---
        opp_act = TEAM_SIZE + ctx.opp_active_local                                   # [B]
        sb = spread_belief[ar, ctx.opp_active_local] if spread_belief is not None else None
        a_base = self.BASE_STATS[ctx.species_ids[ar, opp_act]]                       # [B,6]
        off_const = 31.0 + 252.0 / 4.0 + 5.0
        atk = sb[:, _SB_ATK] if sb is not None else (2.0 * a_base[:, 1] + off_const) * 1.1
        spa = sb[:, _SB_SPA] if sb is not None else (2.0 * a_base[:, 3] + off_const) * 1.1
        opp_b_atk, _obd, opp_b_spa, _obs2, _obe = self._boost_stages(ctx.opp_ctx_raw)
        atk = atk * self._boost_mult(opp_b_atk)
        spa = spa * self._boost_mult(opp_b_spa)
        opp_burn = ctx.pokemon_part[ar, opp_act, POKEMON_CONDITION_OFFSET + _COND_BRN_IDX]
        atk = atk * torch.where(opp_burn > 0.5, atk.new_tensor(0.5), atk.new_tensor(1.0))
        at1 = ctx.type1_ids[ar, opp_act]
        at2 = ctx.type2_ids[ar, opp_act]
        # --- defender = OUR ACTIVE (real spread, current def/spd stages folded) ---
        def_c, spd_c, maxhp, cur_hp, d_t1, d_t2, _amul = self._active_defender(ctx)
        our_abl = ctx.ability1_ids[ar, ctx.our_active_idx]                           # [B]
        # --- the seat candidates' move data (effective BP + the non-formula kinds, like the forward) ---
        opp_hp_frac = ctx.hp_and_active[ar, opp_act, 0]                              # [B]
        bp_k = gather_bp(self, nums, opp_hp_frac[:, None])                           # [B,K]
        mty_k = self.MOVE_TYPE_IDX[nums]
        phys_k = self.MOVE_PHYS[nums]
        acc_k = self.MOVE_ACCURACY[nums]
        nf_k = gather_nonformula(self, nums)                                         # 3×[B,K]
        atk_cur = opp_hp_frac * (2.0 * a_base[:, 0] + 31.0 + 110.0)                  # [B] neutral max HP
        weather_k = self._field_bp_mult(ctx, mty_k)                                  # [B,K] × sports
        reflect = ctx.screen_feature[:, 0:1]                                         # [B,1] OUR side
        light_screen = ctx.screen_feature[:, 2:3]
        # gen3_beatup_exact_v1: the opp party's Σ base Atk / hit count vs OUR active's BASE Def.
        bu_k = gather_beatup(self, nums)                                             # [B,K]
        bu_S, bu_N = beatup_party_opp(self, ctx, species_probs)                      # [B], [B]
        bu_party = (bu_S, bu_N, beatup_base_def(self, ctx.species_ids[ar, ctx.our_active_idx])[:, None])
        def _high(atk_x: torch.Tensor) -> torch.Tensor:
            return self._damage_rolls(
                atk_x, spa, at1, at2, def_c[:, None], spd_c[:, None], maxhp[:, None],
                cur_hp[:, None], d_t1[:, None], d_t2[:, None], our_abl[:, None],
                reflect, light_screen, bp_k, mty_k, phys_k, acc_k, nf_k, atk_cur,
                bu_k, bu_party, weather_k, eps)[0][:, 0, :]                          # [B,K]
        high_full = _high(atk)
        high_half = _high(atk * 0.5)
        gate = (has_opp * our_alive)[:, None]                                        # [B,1]
        d_burn_k = (high_half - high_full) * gate                                    # [B,K] ≤ 0
        d_slp_k = -high_full * gate                                                  # [B,K] ≤ 0
        if _mix is not None:                                                         # [B,K+16] → [B,K]
            d_burn_k = torch.einsum("be,bke->bk", d_burn_k, _mix.to(d_burn_k.dtype))
            d_slp_k = torch.einsum("be,bke->bk", d_slp_k, _mix.to(d_slp_k.dtype))
        if other_u is not None and _mix is not None:
            # X5 U4: OTHER_move's column — the SAME kernel over every move num, the tail contraction.
            nums_f = torch.arange(other_u.shape[-1], device=device).expand(B, -1)   # [B,M]
            bp_k = gather_bp(self, nums_f, opp_hp_frac[:, None])
            mty_k = self.MOVE_TYPE_IDX[nums_f]
            phys_k = self.MOVE_PHYS[nums_f]
            acc_k = self.MOVE_ACCURACY[nums_f]
            nf_k = gather_nonformula(self, nums_f)
            weather_k = self._field_bp_mult(ctx, mty_k)
            bu_k = gather_beatup(self, nums_f)
            hf_f = _high(atk)
            hh_f = _high(atk * 0.5)
            u = other_u.to(hf_f.dtype)
            d_burn_o = ((hh_f - hf_f) * gate * u).sum(-1, keepdim=True)                # [B,1]
            d_slp_o = (-hf_f * gate * u).sum(-1, keepdim=True)
            d_burn_k = torch.cat([d_burn_k, d_burn_o.to(d_burn_k.dtype)], dim=-1)      # [B,K+1]
            d_slp_k = torch.cat([d_slp_k, d_slp_o.to(d_slp_k.dtype)], dim=-1)
        return base, d_burn_k, d_slp_k, is_brn, is_slp

    def pairwise_boost_incoming(self, ctx: 'ExtractorContext',
                                move_belief_logits: torch.Tensor,
                                k_cand: int = 6,
                                species_probs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """gen3_edge_bias_trunk_v1 (C1b): the INCOMING half of the setup consequence — "after
        clicking slot k's boost, how much LESS does each of their mons hurt me". Per (E3 setup
        seat k, opp mon j) the DELTA cells `[d_in_high, d_in_pko]` `[B, 4, TEAM_SIZE, 2]`
        (post-boost − current, ≤ 0: a defensive boost SHRINKS the worst believed hit; an
        offense-only setup move reads ~0 BY PHYSICS, not by gate). Completes Curse's +1 Def /
        Bulk Up / Calm Mind / Amnesia / Iron Defense — the half the outgoing kernel can't see.

        Attackers: the D4 recipe — per opp mon j its top-`k_cand` most-believed candidates from
        ITS OWN slot of the composed posterior (selection detached, weights differentiable),
        de-timid offense, revealed+alive-gated — but with the ACTIVE column KEPT (unlike D4):
        the opp active is exactly who you boost in front of. Defender: OUR ACTIVE only — real
        spread + CURRENT def/spd stages (the pairwise_boost speed convention: here the stage IS
        the signal) + our screens/ability/types. The 5 worlds (current + 4 slots) ride a WORLD
        axis so the attacker side is computed once and only the defensive divisor varies."""
        B, device, eps = ctx.batch_size, ctx.device, 1e-6
        ar = torch.arange(B, device=device)
        opp = slice(TEAM_SIZE, 2 * TEAM_SIZE)
        deltas, is_boost, hp_cost = self._setup_deltas(ctx)                          # [B,4,5], [B,4], [B,4]
        (w_k, bp_k, mty_k, phys_k, acc_k, atk_j, spa_j,
         att_gate, nf_k, atk_cur_j, bu_k, crit_k) = self._believed_attackers(ctx, move_belief_logits, k_cand)
        # --- defender: OUR ACTIVE (real spread; CURRENT def/spd stages) ---
        d_base = self.BASE_STATS[ctx.species_ids[ar, ctx.our_active_idx]]            # [B,6]
        spr = ctx.pokemon_part[ar, ctx.our_active_idx,
                               POKEMON_SPREAD_OFFSET:POKEMON_SPREAD_OFFSET + POKEMON_SPREAD_DIM]
        iv = spr[:, 0:6] * 31.0; ev = spr[:, 6:12] * 252.0; nat = spr[:, 13:18]
        def_stat = (2.0 * d_base[:, _BS_DEF] + iv[:, _BS_DEF] + ev[:, _BS_DEF] / 4.0
                    + 5.0) * nat[:, _NAT_DEF]                                        # [B]
        spd_stat = (2.0 * d_base[:, _BS_SPD] + iv[:, _BS_SPD] + ev[:, _BS_SPD] / 4.0
                    + 5.0) * nat[:, _NAT_SPD]
        maxhp = 2.0 * d_base[:, _BS_HP] + iv[:, _BS_HP] + ev[:, _BS_HP] / 4.0 + 110.0
        cur_hp = ctx.hp_and_active[ar, ctx.our_active_idx, 0] * maxhp                # [B]
        _ba, b_def, _bs, b_spd, _be = self._boost_stages(ctx.our_ctx_raw)            # current [B]
        # --- 5 worlds: current + the 4 slots' def/spd hypotheticals → divisors [B,W] ---
        zero = torch.zeros(B, 1, device=device)
        def_w = def_stat[:, None] * self._boost_mult(b_def[:, None]
                                                     + torch.cat([zero, deltas[..., 1]], dim=1))
        spd_w = spd_stat[:, None] * self._boost_mult(b_spd[:, None]
                                                     + torch.cat([zero, deltas[..., 3]], dim=1))
        # --- damage per (world w, attacker j, candidate c) vs OUR ACTIVE → [B,W,6,K] ---
        at1 = ctx.type1_ids[ar, ctx.our_active_idx]
        at2 = ctx.type2_ids[ar, ctx.our_active_idx]
        amul = self.ABILITY_DAMAGE_MULT[ctx.ability1_ids[ar, ctx.our_active_idx]]    # [B,T]
        eff = (torch.gather(self.CHART[at1][:, None, :].expand(B, TEAM_SIZE, -1), 2, mty_k)
               * torch.gather(self.CHART[at2][:, None, :].expand(B, TEAM_SIZE, -1), 2, mty_k)
               * torch.gather(amul[:, None, :].expand(B, TEAM_SIZE, -1), 2, mty_k))  # [B,6,K]
        A = phys_k * atk_j[:, :, None] + (1.0 - phys_k) * spa_j[:, :, None]          # [B,6,K]
        D = (phys_k[:, None] * def_w[:, :, None, None]
             + (1.0 - phys_k)[:, None] * spd_w[:, :, None, None])                    # [B,W,6,K]
        is_stab = ((mty_k == ctx.type1_ids[:, opp][:, :, None])
                   | (mty_k == ctx.type2_ids[:, opp][:, :, None])).float()           # [B,6,K]
        is_stab = self._x5_stab(is_stab, mty_k)                                      # X5: OTHER's E[STAB]
        # gen3_beatup_exact_v1: Beat Up ignores the defender's stat stages (its Def is the species BASE Def,
        # `event.modifier = 1`), so a defensive setup move moves it by exactly 0 across the worlds.
        bu_S, bu_N = beatup_party_opp(self, ctx, species_probs)                      # [B], [B]
        A4, D, plus2 = beatup_swap(bu_k[:, None], A[:, None], D, bu_S[:, None, None, None],
                                   beatup_base_def(self, ctx.species_ids[ar, ctx.our_active_idx]
                                                   )[:, None, None, None], bu_N[:, None, None, None])
        core = 42.0 * bp_k[:, None] * A4 / (D + eps) / 50.0 + plus2                  # [B,W,6,K]
        dmg_ns = (core * ((1.0 + 0.5 * is_stab) * eff * 0.925)[:, None]
                  * (bp_k > 0).float()[:, None])
        reflect, ls = ctx.screen_feature[:, 0:1], ctx.screen_feature[:, 2:3]         # our side
        screen = (1.0 - 0.5 * (reflect[:, :, None] * phys_k
                               + ls[:, :, None] * (1.0 - phys_k)))                   # [B,6,K]
        high, _low, _crit, ko = self._rolls(dmg_ns, screen[:, None],
                                            maxhp[:, None, None, None],
                                            cur_hp[:, None, None, None],
                                            acc_k[:, None], eps,
                                            crit_p=None if crit_k is None else crit_k[:, None])   # [B,W,6,K]
        # gen3_nonformula_damage_v1: the declared non-formula damage (Def-independent → identical in
        # every world, so a defensive boost moves Seismic Toss by exactly 0 — the true physics).
        nf = nonformula_rolls(tuple(t[:, None] for t in nf_k), cur_hp[:, None, None, None],
                              maxhp[:, None, None, None], atk_cur_j[:, None, :, None],
                              eff[:, None], acc_k[:, None], eps)
        high, _low, _crit, ko = override_rolls((high, _low, _crit, ko), nf)
        w_tot = self._attacker_total(ctx, move_belief_logits)                        # [B,6] | None (max)
        w_tot = None if w_tot is None else w_tot[:, None]                            # [B,1,6]
        worst_high = believed_reduce(w_k[:, None] * high, w_tot)                    # [B,W,6]
        worst_pko = believed_reduce(w_k[:, None] * ko, w_tot)
        d_high = worst_high[:, 1:] - worst_high[:, 0:1]                              # [B,4,6]
        d_pko = worst_pko[:, 1:] - worst_pko[:, 0:1]
        our_alive = (ctx.hp_and_active[ar, ctx.our_active_idx, 0] > 0).float()       # [B]
        gate = is_boost[:, :, None] * att_gate[:, None, :] * our_alive[:, None, None]
        return torch.stack([d_high, d_pko], dim=-1) * gate[..., None]                # [B,4,6,2]

    def pairwise_recovery(self, ctx: 'ExtractorContext',
                          move_belief_logits: torch.Tensor,
                          k_cand: int = 6,
                          species_probs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """gen3_edge_bias_trunk_v1 (C3): the RECOVERY-FLIP consequence — "after clicking slot
        k's heal, does their believed hit still KO me". Per (E3 recovery seat k, opp mon j) the
        cells `[is_recovery, d_in_pko, rest_sleep_turns]` `[B, 4, TEAM_SIZE, 3]` (the delta ≤ 0,
        and it hits −w exactly when healing crosses the threshold from "they KO me" to "they
        don't" — the heal-vs-attack decision fact, the C1b machinery with the HYPOTHETICAL
        input moved from the def/spd stages to the HP total). The shared believed-attacker
        block vs OUR ACTIVE; the damage itself is computed ONCE (a heal changes no divisor) and
        only the validated `_rolls` KO ramp is re-evaluated at the 5 worlds' post-heal HP
        (`min(maxhp, cur + MOVE_HEAL_FRACTION·maxhp)`). **Rest's self-sleep COST is priced
        (owner-prioritized 2026-08-06)**: `rest_sleep_turns` = the DETERMINISTIC lost turns —
        EXACTLY 2 (1 with Early Bird; our own ability is KNOWN, so this is exact, never a
        prior), /4-normed, from the same verified `sleep_belief.expected_free_turns` tables as
        C2's opp-sleep channel; zero on every non-Rest slot. The weather heals ride a flat 0.5
        approximation; Wish is excluded (delayed — the wish obs scalars own it). See
        `build_recovery_tables`."""
        B, device, eps = ctx.batch_size, ctx.device, 1e-6
        ar = torch.arange(B, device=device)
        opp = slice(TEAM_SIZE, 2 * TEAM_SIZE)
        frac = self.MOVE_HEAL_FRACTION[ctx.our_active_req_move_ids]                  # [B,4]
        is_rec = (frac > 0).float()
        # Weather heals fold LIVE weather (was a flat 0.5): gen3 Moonlight/Morning Sun/Synthesis
        # heal 2/3 under sun, 1/4 under ANY other weather, 1/2 clear (weather one-hot
        # [NONE, SUN, RAIN, SAND, HAIL]) — sun-stall Synthesis is a real heal, sand-era one isn't.
        wh = self.MOVE_WEATHER_HEAL[ctx.our_active_req_move_ids]                     # [B,4]
        sun = ctx.weather_feature[:, 1:2]
        other_w = ctx.weather_feature[:, 2:5].sum(dim=-1, keepdim=True)
        w_frac = (2.0 / 3.0) * sun + 0.25 * other_w + 0.5 * (1.0 - sun - other_w)    # [B,1]
        frac = torch.where(wh > 0, w_frac.expand_as(frac), frac)
        (w_k, bp_k, mty_k, phys_k, acc_k, atk_j, spa_j,
         att_gate, nf_k, atk_cur_j, bu_k, crit_k) = self._believed_attackers(ctx, move_belief_logits, k_cand)
        # --- defender: OUR ACTIVE (real spread; CURRENT stages — a heal changes no stage) ---
        d_base = self.BASE_STATS[ctx.species_ids[ar, ctx.our_active_idx]]            # [B,6]
        spr = ctx.pokemon_part[ar, ctx.our_active_idx,
                               POKEMON_SPREAD_OFFSET:POKEMON_SPREAD_OFFSET + POKEMON_SPREAD_DIM]
        iv = spr[:, 0:6] * 31.0; ev = spr[:, 6:12] * 252.0; nat = spr[:, 13:18]
        def_stat = (2.0 * d_base[:, _BS_DEF] + iv[:, _BS_DEF] + ev[:, _BS_DEF] / 4.0
                    + 5.0) * nat[:, _NAT_DEF]
        spd_stat = (2.0 * d_base[:, _BS_SPD] + iv[:, _BS_SPD] + ev[:, _BS_SPD] / 4.0
                    + 5.0) * nat[:, _NAT_SPD]
        _ba, b_def, _bs2, b_spd, _be = self._boost_stages(ctx.our_ctx_raw)
        def_c = def_stat * self._boost_mult(b_def)                                   # [B]
        spd_c = spd_stat * self._boost_mult(b_spd)
        maxhp = 2.0 * d_base[:, _BS_HP] + iv[:, _BS_HP] + ev[:, _BS_HP] / 4.0 + 110.0
        hp_frac = ctx.hp_and_active[ar, ctx.our_active_idx, 0]
        cur_hp = hp_frac * maxhp                                                     # [B]
        zero = torch.zeros(B, 1, device=device)
        hp_w = torch.minimum(maxhp[:, None],
                             cur_hp[:, None] + torch.cat([zero, frac], dim=1) * maxhp[:, None])
        # --- damage ONCE (world-independent), the KO ramp per world ---
        at1 = ctx.type1_ids[ar, ctx.our_active_idx]
        at2 = ctx.type2_ids[ar, ctx.our_active_idx]
        amul = self.ABILITY_DAMAGE_MULT[ctx.ability1_ids[ar, ctx.our_active_idx]]
        eff = (torch.gather(self.CHART[at1][:, None, :].expand(B, TEAM_SIZE, -1), 2, mty_k)
               * torch.gather(self.CHART[at2][:, None, :].expand(B, TEAM_SIZE, -1), 2, mty_k)
               * torch.gather(amul[:, None, :].expand(B, TEAM_SIZE, -1), 2, mty_k))  # [B,6,K]
        A = phys_k * atk_j[:, :, None] + (1.0 - phys_k) * spa_j[:, :, None]
        D = phys_k * def_c[:, None, None] + (1.0 - phys_k) * spd_c[:, None, None]    # [B,6,K]
        is_stab = ((mty_k == ctx.type1_ids[:, opp][:, :, None])
                   | (mty_k == ctx.type2_ids[:, opp][:, :, None])).float()
        is_stab = self._x5_stab(is_stab, mty_k)                                      # X5: OTHER's E[STAB]
        # gen3_beatup_exact_v1: the opp party's Σ base Atk / hit count vs OUR active's BASE Def.
        bu_S, bu_N = beatup_party_opp(self, ctx, species_probs)                      # [B], [B]
        A, D, plus2 = beatup_swap(bu_k, A, D, bu_S[:, None, None],
                                  beatup_base_def(self, ctx.species_ids[ar, ctx.our_active_idx])[:, None, None],
                                  bu_N[:, None, None])
        core = 42.0 * bp_k * A / (D + eps) / 50.0 + plus2
        dmg_ns = core * (1.0 + 0.5 * is_stab) * eff * 0.925 * (bp_k > 0).float()     # [B,6,K]
        reflect, ls = ctx.screen_feature[:, 0:1], ctx.screen_feature[:, 2:3]
        screen = (1.0 - 0.5 * (reflect[:, :, None] * phys_k
                               + ls[:, :, None] * (1.0 - phys_k)))                   # [B,6,K]
        _h, _l, _c, ko_w = self._rolls(dmg_ns[:, None], screen[:, None],
                                       maxhp[:, None, None, None],
                                       hp_w[:, :, None, None], acc_k[:, None], eps,
                                       crit_p=None if crit_k is None else crit_k[:, None])  # [B,W,6,K]
        # gen3_nonformula_damage_v1: the declared non-formula damage vs each world's post-heal HP
        # (Super Fang / OHKO / Endeavor read the healed HP; Seismic Toss its fixed 100).
        nf = nonformula_rolls(tuple(t[:, None] for t in nf_k), hp_w[:, :, None, None],
                              maxhp[:, None, None, None], atk_cur_j[:, None, :, None],
                              eff[:, None], acc_k[:, None], eps)
        _h, _l, _c, ko_w = override_rolls((_h, _l, _c, ko_w), nf)
        w_tot = self._attacker_total(ctx, move_belief_logits)                        # [B,6] | None (max)
        worst_pko = believed_reduce(w_k[:, None] * ko_w,
                                    None if w_tot is None else w_tot[:, None])      # [B,W,6]
        d_pko = worst_pko[:, 1:] - worst_pko[:, 0:1]                                 # [B,4,6]
        # --- Rest's deterministic self-sleep cost (our OWN ability → exact, never a prior) ---
        own_eb = self.ABILITY_IS_EARLYBIRD[ctx.ability1_ids[ar, ctx.our_active_idx]]  # [B]
        rest_turns = (self.rest_sleep_noeb
                      + (self.rest_sleep_eb - self.rest_sleep_noeb) * own_eb) / 4.0   # [B]
        is_rest_slot = (ctx.our_active_req_move_ids == self.rest_num).float()         # [B,4]
        rest_ch = (rest_turns[:, None] * is_rest_slot)[:, :, None].expand(-1, -1, TEAM_SIZE)
        gate = (is_rec[:, :, None] * att_gate[:, None, :]
                * (hp_frac > 0).float()[:, None, None])
        ib = is_rec[:, :, None].expand(-1, -1, TEAM_SIZE)
        return torch.stack([ib, d_pko, rest_ch], dim=-1) * gate[..., None]           # [B,4,6,3]

    def pairwise_bench_outgoing(self, ctx: 'ExtractorContext',
                                spread_belief: Optional[torch.Tensor] = None,
                                inherit_stages: bool = False) -> torch.Tensor:
        """gen3_edge_bias_trunk_v1 (D2): per OUR mon, its best offense vs the OPP ACTIVE — the mon↔mon
        edge cell `[best_high, best_pko, p_outspeed, alive]` `[B, TEAM_SIZE, 4]`, a move-collapsed
        reshape of the validated v39 `_outgoing_attacker_matrix` (the switch-in-offense kernel: the
        active row reproduces `_outgoing_block`; bench rows neutral-boost). Written at the
        (our-mon seat i, opp-ACTIVE seat) pair — the "what would this switch-in DO" edge."""
        flat = self._outgoing_attacker_matrix(ctx, spread_belief, inherit_stages)         # [B, _DMG_OAX]
        n_cells = TEAM_SIZE * _DMG_OAX_PER_MON
        cells = flat[:, :n_cells].reshape(-1, TEAM_SIZE, _DMG_OAX_N_MOVES, _DMG_OAX_PER_MOVE)
        p_outspeed = flat[:, n_cells:n_cells + TEAM_SIZE]                                  # [B,6]
        alive = flat[:, n_cells + TEAM_SIZE:n_cells + 2 * TEAM_SIZE]                       # [B,6]
        best_high = max_by_index(cells[..., _DMG_OAX_IDX_HIGH])                             # [B,6]
        best_pko = max_by_index(cells[..., _DMG_OAX_IDX_PKO])                               # [B,6]
        return torch.stack([best_high, best_pko, p_outspeed, alive], dim=-1)               # [B,6,4]

    def pairwise_baton(self, ctx: 'ExtractorContext',
                       spread_belief: Optional[torch.Tensor] = None) -> torch.Tensor:
        """gen3_edge_bias_trunk_v1 (C5): the BATON PASS receiver edge — "who inherits my boosts
        best". The FIRST family on the (E3 seat, OUR-mon) route: per (E3 Baton-Pass seat k, OUR
        mon j) the DELTA cells `[is_bp, d_best_high, d_best_pko, d_outspeed]`
        `[B, 4, TEAM_SIZE, 4]` — receiver j's best offense / P(KO) / P(outspeed) vs the opp
        ACTIVE with the active's CURRENT stages INHERITED (the validated v39 switch-in kernel
        re-run under `inherit_stages=True` — the hypothetical world is one flag away from the
        world D2 already prices) minus its neutral-switch-in baseline. With no stages up, both
        worlds coincide and every delta reads 0 (honest: an unboosted BP is just a slow pivot).
        The ACTIVE column is ZEROED (you pass to the bench); receivers alive-gated. v1
        residuals (documented): negative stages pass too (priced — the delta is signed);
        Substitute/volatile passing is unpriced; the receiver's INCOMING world is C1b-shaped
        follow-up territory."""
        B, device = ctx.batch_size, ctx.device
        ar = torch.arange(B, device=device)
        is_bp = (ctx.our_active_req_move_ids == self.baton_num).float()              # [B,4]
        base = self.pairwise_bench_outgoing(ctx, spread_belief)                      # [B,6,4]
        passed = self.pairwise_bench_outgoing(ctx, spread_belief, inherit_stages=True)
        d_high = passed[..., 0] - base[..., 0]                                       # [B,6]
        d_pko = passed[..., 1] - base[..., 1]
        d_spd = passed[..., 2] - base[..., 2]
        alive = base[..., 3]
        not_active = torch.ones(B, TEAM_SIZE, device=device)
        not_active[ar, ctx.our_active_idx] = 0.0
        our_alive = (ctx.hp_and_active[ar, ctx.our_active_idx, 0] > 0).float()
        gate = (is_bp[:, :, None] * (alive * not_active)[:, None, :]
                * our_alive[:, None, None])                                          # [B,4,6]
        ib = is_bp[:, :, None].expand(-1, -1, TEAM_SIZE)
        cells = torch.stack([ib, d_high[:, None, :].expand(-1, 4, -1),
                             d_pko[:, None, :].expand(-1, 4, -1),
                             d_spd[:, None, :].expand(-1, 4, -1)], dim=-1)
        return cells * gate[..., None]                                               # [B,4,6,4]

    def pairwise_speed(self, ctx: 'ExtractorContext',
                       spread_belief: Optional[torch.Tensor] = None) -> torch.Tensor:
        """gen3_edge_bias_trunk_v1 (V): the full mon↔mon speed edge — P(our mon i outspeeds opp mon j)
        for every (i, j) pair, `[B, TEAM_SIZE, TEAM_SIZE, 3]` = `[p_outspeed, both_alive, revealed_j]`.
        v1 conventions (lean, mirroring `discrete_incoming`'s coarse-signal contract): NO stage boosts
        either side (gen3 resets on switch — the active's live boost is the authoritative incoming
        block's job), PARA folded per-mon ×0.25 from the PUBLIC condition one-hot BOTH sides. Our
        speeds from the REAL spread reconstruction; opp speeds from the believed spread when given
        (the prefuse SpreadBelief) else the neutral sentinel-free estimate; the uncertainty-aware
        sigmoid (`prob_outspeed`) divides by the believed per-species speed STD. Unrevealed opp slots
        carry the `revealed_j` channel so the head can discount the guess."""
        # --- our 6: real spread reconstruction (the _incoming_rolls defender recipe) ---
        d_base = self.BASE_STATS[ctx.species_ids[:, :TEAM_SIZE]]                          # [B,6,6]
        spread = ctx.pokemon_part[:, :TEAM_SIZE,
                                  POKEMON_SPREAD_OFFSET:POKEMON_SPREAD_OFFSET + POKEMON_SPREAD_DIM]
        iv = spread[..., 0:6] * 31.0
        ev = spread[..., 6:12] * 252.0
        nat = spread[..., 13:18]
        # FIXED 2026-08-06 (the SpD-as-speed GIGO): these reads shipped as index 4 / nat[...,3]
        # — Special Defense — so the V edge priced "outspeed" off bulk for both trained
        # generations. Named indices now; guarded by the Aerodactyl-vs-Snorlax regression test.
        our_spe = (2.0 * d_base[..., _BS_SPE] + iv[..., _BS_SPE] + ev[..., _BS_SPE] / 4.0
                   + 5.0) * nat[..., _NAT_SPE]                                             # [B,6]
        our_para = ctx.pokemon_part[:, :TEAM_SIZE, POKEMON_CONDITION_OFFSET + _COND_PAR_IDX]
        our_spe = our_spe * (1.0 - 0.75 * our_para)                                       # gen3 para ×0.25
        # --- opp 6: believed spread else the neutral sentinel-free estimate ---
        opp_species = ctx.species_ids[:, TEAM_SIZE:2 * TEAM_SIZE]                          # [B,6]
        if spread_belief is not None:
            opp_spe = spread_belief[..., _SB_SPE]                                          # [B,6]
        else:
            opp_base = self.BASE_STATS[opp_species]                                        # [B,6,6]
            opp_spe = 2.0 * opp_base[..., _BS_SPE] + 31.0 + 5.0                            # neutral 0-EV
        opp_spe = self._x5_avg(opp_spe, "spe")                                             # X5: OTHER's averaged speed
        opp_para = ctx.pokemon_part[:, TEAM_SIZE:2 * TEAM_SIZE,
                                    POKEMON_CONDITION_OFFSET + _COND_PAR_IDX]
        opp_spe = opp_spe * (1.0 - 0.75 * opp_para)                                        # [B,6]
        if self.speed_physics:
            # gen3_speed_physics_v1: every (our i, their j) pair by the one rule — paralysis on every row, the
            # stage on each side's ACTIVE row only (a bench mon's stages are reset), Quick Claw folded.
            p = self._p_first_pairs(ctx)                                                     # [B,6,6]
        else:
            p = self._p_outspeed(our_spe[:, :, None], opp_spe[:, None, :])                   # [B,6,6]
        alive_i = (ctx.hp_and_active[:, :TEAM_SIZE, 0] > 0).float()                        # [B,6]
        alive_j = (ctx.hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, 0] > 0).float()           # [B,6]
        _x5 = self.stash.x5
        if _x5 is not None:      # X5 (U3 part 3): a hypothesis is alive by addressability (its speed: its species')
            alive_j = _x5.alive
        both = alive_i[:, :, None] * alive_j[:, None, :]                                   # [B,6,6]
        revealed_j = (1.0 - ctx.opp_believed_mask.float())[:, None, :].expand_as(both)     # [B,6,6]
        return torch.stack([p * both, both, revealed_j * both], dim=-1)                    # [B,6,6,3]

    def pairwise_bench_incoming(self, ctx: 'ExtractorContext',
                                move_belief_logits: torch.Tensor,
                                k_bench: int = 6,
                                species_probs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """gen3_edge_bias_trunk_v1 (D4): the MISSING quadrant — every OPP mon's believed threat to every
        OUR mon ("after I KO, what comes in and what does it threaten"). `[B, TEAM_SIZE(our i),
        TEAM_SIZE(opp j), 4]` = `[phys_high, spec_high, phys_pko, spec_pko]` per (defender i,
        attacker j): per opp mon j the top-`k_bench` most-believed candidates from ITS OWN slot of the
        composed posterior (selection detached, weights differentiable — the belief gradient now reaches
        the BENCH slots' move heads for the first time), the de-timid attacker recipe (the
        `discrete_incoming` coarse-signal convention: no boosts/burn/weather — correct for a bench mon,
        which switches in at neutral stages), our REAL-spread defenders + our-side screens. REVEALED- and
        alive-gated per attacker; the opp ACTIVE's column is ZEROED (that quadrant is D3's job at full
        candidate width — decorrelation, not duplication). Delivered at the (our-mon, opp-mon) block."""
        B, device, eps = ctx.batch_size, ctx.device, 1e-6
        ar = torch.arange(B, device=device)
        # --- per-slot candidate selection (the _opp_candidate_weights math, per opp mon j) ---
        _x5 = self.stash.x5
        if _x5 is None:
            w_all = torch.sigmoid(move_belief_logits) * self.HP_CAND_MASK[None, None, :]   # [B,6,M]
            K = min(int(k_bench), w_all.shape[-1])
            topk_idx = w_all.detach().topk(K, dim=-1).indices                              # [B,6,K] DETACHED
            w_k = w_all.gather(-1, topk_idx)                                               # [B,6,K] diff'able
        else:   # X5 (U3 part 3): the per-mon one order over the fixed-mass presence (`_believed_attackers`)
            K = min(int(k_bench), _x5.move_w.shape[-1])
            topk_idx = _x5.move_order[..., :K]
            w_k = _x5.move_w.gather(-1, topk_idx)
        hp_j = ctx.hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, 0]                        # [B,6] attacker HP
        bp_k = gather_bp(self, topk_idx, hp_j[:, :, None])                             # [B,6,K] effective
        mty_k = self.MOVE_TYPE_IDX[topk_idx]
        phys_k = self.MOVE_PHYS[topk_idx]
        acc_k = self.MOVE_ACCURACY[topk_idx]
        nf_k = gather_nonformula(self, topk_idx)                                       # 3×[B,6,K]
        bu_k = gather_beatup(self, topk_idx)                                           # [B,6,K] gen3_beatup_exact_v1
        # --- attackers = the opp 6 (de-timid; revealed+alive-gated; active column zeroed) ---
        opp = slice(TEAM_SIZE, 2 * TEAM_SIZE)
        a_base = self._x5_avg(self.BASE_STATS[ctx.species_ids[:, opp]], "att_base")    # [B,6,6] (X5: OTHER = E_tail)
        off_const = 31.0 + 252.0 / 4.0 + 5.0
        atk_j = (2.0 * a_base[..., 1] + off_const) * 1.1                               # [B,6]
        spa_j = (2.0 * a_base[..., 3] + off_const) * 1.1
        at1 = ctx.type1_ids[:, opp]
        at2 = ctx.type2_ids[:, opp]
        revealed_j: torch.Tensor = (1.0 - ctx.opp_believed_mask.float())               # [B,6]
        alive_j = (ctx.hp_and_active[:, opp, 0] > 0).float()
        not_active_j = torch.ones(B, TEAM_SIZE, device=device)
        not_active_j[ar, ctx.opp_active_local] = 0.0
        att_gate = revealed_j * alive_j * not_active_j                                 # [B,6]
        if _x5 is not None:      # every live bench mon attacks; a hypothesis as its species
            att_gate = _x5.alive * not_active_j
        # --- defenders = our 6 (the _incoming_rolls real-spread recipe) ---
        d_base = self.BASE_STATS[ctx.species_ids[:, :TEAM_SIZE]]
        spread = ctx.pokemon_part[:, :TEAM_SIZE,
                                  POKEMON_SPREAD_OFFSET:POKEMON_SPREAD_OFFSET + POKEMON_SPREAD_DIM]
        iv = spread[..., 0:6] * 31.0
        ev = spread[..., 6:12] * 252.0
        nat = spread[..., 13:18]
        def_stat = (2.0 * d_base[..., 2] + iv[..., 2] + ev[..., 2] / 4.0 + 5.0) * nat[..., 1]  # [B,6]
        spd_stat = (2.0 * d_base[..., 4] + iv[..., 4] + ev[..., 4] / 4.0 + 5.0) * nat[..., 3]
        maxhp = 2.0 * d_base[..., 0] + iv[..., 0] + ev[..., 0] / 4.0 + 110.0
        hp_frac = ctx.hp_and_active[:, :TEAM_SIZE, 0]
        cur_hp = hp_frac * maxhp
        def_alive = (hp_frac > 0).float()                                              # [B,6]
        t1d = ctx.type1_ids[:, :TEAM_SIZE]
        t2d = ctx.type2_ids[:, :TEAM_SIZE]
        amul = self.ABILITY_DAMAGE_MULT[ctx.ability1_ids[:, :TEAM_SIZE]]               # [B,6,T]
        # --- damage per (defender i, attacker j, candidate c) → [B,6,6,K] ---
        idx = mty_k[:, None, :, :].expand(B, TEAM_SIZE, TEAM_SIZE, K)                  # [B,6i,6j,K]
        eff = (torch.gather(self.CHART[t1d][:, :, None, :].expand(B, TEAM_SIZE, TEAM_SIZE, -1), 3, idx)
               * torch.gather(self.CHART[t2d][:, :, None, :].expand(B, TEAM_SIZE, TEAM_SIZE, -1), 3, idx)
               * torch.gather(amul[:, :, None, :].expand(B, TEAM_SIZE, TEAM_SIZE, -1), 3, idx))
        A = phys_k * atk_j[:, :, None] + (1.0 - phys_k) * spa_j[:, :, None]            # [B,6j,K]
        D = (phys_k[:, None, :, :] * def_stat[:, :, None, None]
             + (1.0 - phys_k)[:, None, :, :] * spd_stat[:, :, None, None])             # [B,6i,6j,K]
        is_stab = ((mty_k == at1[:, :, None]) | (mty_k == at2[:, :, None])).float()    # [B,6j,K]
        is_stab = self._x5_stab(is_stab, mty_k)                                        # X5: OTHER's E[STAB]
        stab = (1.0 + 0.5 * is_stab)[:, None, :, :]
        # gen3_beatup_exact_v1: the opp party's Σ base Atk / hit count (the SIDE, whichever mon j uses it) vs
        # each of OUR six defenders' BASE Def.
        bu_S, bu_N = beatup_party_opp(self, ctx, species_probs)                        # [B], [B]
        A4, D, plus2 = beatup_swap(bu_k[:, None], A[:, None, :, :], D, bu_S[:, None, None, None],
                                   beatup_base_def(self, ctx.species_ids[:, :TEAM_SIZE])[:, :, None, None],
                                   bu_N[:, None, None, None])
        core = 42.0 * bp_k[:, None, :, :] * A4 / (D + eps) / 50.0 + plus2
        dmg_ns = core * stab * eff * 0.925
        dmg_ns = dmg_ns * (bp_k > 0).float()[:, None, :, :]                            # kill the +2 floor
        reflect, light_screen = ctx.screen_feature[:, 0:1], ctx.screen_feature[:, 2:3]  # [B,1] our side
        screen = (1.0 - 0.5 * (reflect[:, :, None] * phys_k
                               + light_screen[:, :, None] * (1.0 - phys_k)))           # [B,6j,K]
        high, _low, _crit, ko = self._rolls(dmg_ns, screen[:, None, :, :],
                                            maxhp[:, :, None, None], cur_hp[:, :, None, None],
                                            acc_k[:, None, :, :], eps,
                                            crit_p=self._crit_p_at(topk_idx, 1))   # each [B,6i,6j,K]
        # gen3_nonformula_damage_v1: the declared non-formula damage per (defender i, attacker j, cand c).
        atk_cur_j = hp_j * (2.0 * a_base[..., 0] + 31.0 + 110.0)                       # [B,6j] neutral max HP
        nf = nonformula_rolls(tuple(t[:, None] for t in nf_k), cur_hp[:, :, None, None],
                              maxhp[:, :, None, None], atk_cur_j[:, None, :, None],
                              eff, acc_k[:, None, :, :], eps)
        high, _low, _crit, ko = override_rolls((high, _low, _crit, ko), nf)
        wb = w_k[:, None, :, :]
        pm = phys_k[:, None, :, :]
        w_tot = self._attacker_total(ctx, move_belief_logits)                          # [B,6j] | None (max)
        w_tot = None if w_tot is None else w_tot[:, None, :]                           # [B,1,6j]
        cells = torch.stack([
            believed_reduce(wb * high * pm, w_tot), believed_reduce(wb * high * (1.0 - pm), w_tot),
            believed_reduce(wb * ko * pm, w_tot), believed_reduce(wb * ko * (1.0 - pm), w_tot),
        ], dim=-1)                                                                     # [B,6i,6j,4]
        return cells * def_alive[:, :, None, None] * att_gate[:, None, :, None]

    def pairwise_schedule(self, ctx: 'ExtractorContext') -> Tuple[torch.Tensor, torch.Tensor]:
        """gen3_edge_bias_trunk_v1 (G): the per-mon END-OF-TURN HP LEDGER — what a mon bleeds or
        heals per turn while active, delivered at the (mon, GLOBAL seat) pair like X. → `(our_cells
        [B,6,4], opp_cells [B,6,4])`, cell (signed maxhp fractions; heal +, chip −):

          * leftovers  — +1/16 (our item exact; opp REVEALED item exact, unrevealed 0 — the
            leftovers usage prior is a documented follow-up, the CB p_cb pattern)
          * weather    — −1/16 sand (Rock/Ground/Steel immune) / hail (Ice immune), from the live
            weather one-hot [NONE, SUN, RAIN, SAND, HAIL]
          * status     — −1/8 burn/poison; Toxic charged FLAT −1/8 in v1 (the ramp needs the toxic
            STAGE, an E2 follow-up — documented approximation)
          * leech      — −1/8 for the mon CURRENTLY seeded (ACTIVES only, correctly: Leech Seed is
            an active volatile and clears on switch; the drained credit side is deliberately NOT
            cross-charged — cross-mon maxhp scaling is a GIGO trap, the head composes it)
        Alive-gated; opp weather/status legs revealed-gated (unknown types/condition provenance)."""
        if self.eot_rule is not None:
            # gen3_endstate_facts_v1 (`--g-ledger eot`): THE end-of-turn rule (`eot_residual.EotResidualRule`, the one
            # `--eot-residual` reads) grouped into the ledger's four columns (`eot_residual.g_cells`). Their side is
            # gated as the rule gates it: alive × concrete (X5: the roster's), and the roster's alive under X5.
            from agents.model.eot_residual import g_cells
            _x5g = self.stash.x5
            our_g, opp_g = g_cells(self.eot_rule(ctx, self, None if _x5g is None else _x5g.concrete))
            if _x5g is not None:
                opp_g = opp_g * _x5g.alive[:, :, None]
            return our_g, opp_g
        B, device = ctx.batch_size, ctx.device
        ar = torch.arange(B, device=device)
        from agents.model.damage_tables import _T2I
        LEFTOVERS_NUM = 234
        sand = ctx.weather_feature[:, 3:4]                                        # [B,1]
        hail = ctx.weather_feature[:, 4:5]

        def _side(sl: slice, ctx_raw: torch.Tensor, active_local: torch.Tensor,
                  exact_side: bool) -> torch.Tensor:
            types1 = ctx.type1_ids[:, sl]
            types2 = ctx.type2_ids[:, sl]
            def _has(tname: str) -> torch.Tensor:
                ti = _T2I[tname]
                return ((types1 == ti) | (types2 == ti)).float()                  # [B,6]
            sand_immune = (_has("ROCK") + _has("GROUND") + _has("STEEL")).clamp(max=1.0)
            hail_immune = _has("ICE")
            weather = -(1.0 / 16.0) * (sand * (1.0 - sand_immune) + hail * (1.0 - hail_immune))
            items = ctx.item_ids[:, sl]
            lefties = (items == LEFTOVERS_NUM).float() * (1.0 / 16.0)             # revealed-exact
            cond = ctx.pokemon_part[:, sl,
                                    POKEMON_CONDITION_OFFSET:POKEMON_CONDITION_OFFSET + 7]
            # Burn/poison flat −1/8; TOXIC now carries its RAMP (owner-prioritized 2026-08-06):
            # the obs toxic counter (min(stage,15)/15, PUBLIC both sides; it saturated at 8 until config v151) →
            # the NEXT tick costs min(stage+1, 15)/16 — a fresh Toxic reads −1/16, a 5-turn one −6/16, a 12-turn
            # one −13/16. The old flat −1/8 under-priced late-stall Toxic 3-4× (the CurseLax/stall war fact).
            tox_next = toxic_next_tick(ctx.pokemon_part[:, sl, POKEMON_COUNTER_OFFSET + 1])    # [B,6]
            status = (-(1.0 / 8.0) * (cond[..., _COND_BRN_IDX] + cond[..., 5])
                      - cond[..., 6] * tox_next)
            leech_active = ctx_raw[:, _BOOSTS_DIM + _LEECH_SEED_CTX_SLOT]         # [B] active volatile
            leech = torch.zeros(B, TEAM_SIZE, device=device)
            leech[ar, active_local] = -(1.0 / 8.0) * (leech_active > 0.5).float()
            alive = (ctx.hp_and_active[:, sl, 0] > 0).float()
            cells = torch.stack([lefties, weather.expand(-1, TEAM_SIZE) if weather.shape[1] == 1
                                 else weather, status, leech], dim=-1)            # [B,6,4]
            if not exact_side:
                revealed = (1.0 - ctx.opp_believed_mask.float())[:, :, None]
                _x5 = self.stash.x5
                if _x5 is not None:      # X5 (U3 part 3): a hypothesis's types are its species'
                    revealed, alive = _x5.concrete[:, :, None], _x5.alive
                cells = cells * revealed
            return cells * alive[:, :, None]

        our_cells = _side(slice(0, TEAM_SIZE), ctx.our_ctx_raw, ctx.our_active_idx, True)
        opp_cells = _side(slice(TEAM_SIZE, 2 * TEAM_SIZE), ctx.opp_ctx_raw,
                          ctx.opp_active_local, False)
        return our_cells, opp_cells

    _PROTECT_NUMS = (182, 197, 203)   # protect / detect / endure (pinned by trap_edges_test)

    def pairwise_protect(self, ctx: 'ExtractorContext',
                         protect_odds_our: torch.Tensor) -> torch.Tensor:
        """gen3_edge_bias_trunk_v1 (C4): the PROTECT-CONSEQUENCE edge — "if I click Protect this
        turn, what happens for free" — at the (E3 move seat k, GLOBAL seat) pair, gated to the
        request slots that ARE Protect/Detect/Endure. `[B, 4, 4]` per request slot k:
        `[is_protect_k, p_success, net_ours, net_theirs]` where p_success is the obs
        floored-doubling protect odds (`gen3_protect_odds_v1`, the counter-derived scalar) and
        net_* are the two ACTIVES' end-of-turn ledgers (the G-family `pairwise_schedule` sums —
        Leftovers/weather/status/Leech) — the turn a successful Protect banks: their Toxic ramps,
        our Leftovers ticks, and nothing else happens. The head composes p·net; channels stay
        decorrelated (the provide-facts convention)."""
        B, device = ctx.batch_size, ctx.device
        ar = torch.arange(B, device=device)
        move_ids = ctx.our_active_req_move_ids.long()                             # [B,4]
        is_prot = torch.zeros_like(move_ids, dtype=torch.float32)
        for n in self._PROTECT_NUMS:
            is_prot = is_prot + (move_ids == n).float()
        is_prot = is_prot.clamp(max=1.0)
        our_g, opp_g = self.pairwise_schedule(ctx)                                # [B,6,4] each
        net_ours = our_g[ar, ctx.our_active_idx].sum(-1, keepdim=True)            # [B,1]
        net_theirs = opp_g[ar, ctx.opp_active_local].sum(-1, keepdim=True)
        cells = torch.cat([
            is_prot[:, :, None],
            protect_odds_our[:, None, None].expand(B, 4, 1),
            net_ours[:, None, :].expand(B, 4, 1),
            net_theirs[:, None, :].expand(B, 4, 1),
        ], dim=-1)                                                                # [B,4,4]
        return cells * is_prot[:, :, None]                                        # non-Protect slots 0

    def spikes_entry(self, ctx: 'ExtractorContext') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """THE gen-3 Spikes ENTRY rule, per mon — `(chip_our, chip_opp, grounded_our, grounded_opp)`, each [B,6],
        UN-gated (no alive / revealed gate: each caller applies its own). ONE definition, read by the `x` edge
        cell (`pairwise_entry`) and by `--mon-hazard-cost`'s per-mon fact (`gen3_static_port_v1`), so the two
        can never drift.

        Verified in `deps/pokemon-showdown` (gen 3 inherits gen 4's scripts): `spikes`' `onEntryHazard` (the
        gen-4 mod; gen 4's `runSwitch` fires `runEvent('EntryHazard')` on every switch-in) damages
        `[0, 3, 4, 6][layers] * maxhp / 24` = 1/8, 1/6, 1/4 of max HP iff `pokemon.isGrounded()`; in gen 3
        `isGrounded` is false only for a FLYING type or LEVITATE (Gravity, Ingrain-grounding, Iron Ball, Magnet
        Rise, Air Balloon, Roost, Smack Down are gen 4+; Magic Guard, num 98, is a gen-4 ability; Heavy-Duty
        Boots is gen 8). Levitate is read from the SPECIES (`SPECIES_TRAP_PRIOR[:, 3]`, exactly 0 or 1 in gen 3), never
        the current-ability column. The fraction is the nominal one (Showdown floors the HP lost).

        The Flying check reads the mon's BASE (species) types, never the obs type columns: a switch-IN happens
        after the mon left the field and `clearVolatile` reverted its Color Change / Transform / Conversion /
        Forecast types AND its Trace / Role Play / Skill Swap / Transform ability, so an active Kecleon that turned
        Flying, or a Gardevoir that Traced Levitate, still pays on its next entry (GIGO fix 2026-10-09).

        Under X5 the caller passes the HYPOTHESIS context: a hidden slot's species is its hypothesis'."""
        chip_table = torch.tensor([0.0, 1.0 / 8, 1.0 / 6, 1.0 / 4], device=ctx.device)
        # --- grounded: BOTH the Flying and the Levitate checks read the SPECIES, never a current column ---
        # BASE types, from the species (`SPECIES_TYPE`), NOT the obs type columns: those hold a mon's CURRENT types
        # (Color Change, Transform, Conversion/Conversion 2, Castform's Forecast forme change), and a switch-IN runs
        # on the mon after `clearVolatile` -> `setSpecies(baseSpecies)` has reverted all of them. A benched mon's
        # columns already read base; the ACTIVE mon's do not. Under X5 `ctx` is the hypothesis context, whose
        # `species_ids` are the hypothesis' species.
        sp_i = ctx.species_ids[:, :TEAM_SIZE]
        sp_j = ctx.species_ids[:, TEAM_SIZE:2 * TEAM_SIZE]
        fly_i = (self.TYPE_IS_FLYING[self.SPECIES_TYPE[sp_i]].sum(-1)).clamp(max=1.0)   # [B,6]
        fly_j = (self.TYPE_IS_FLYING[self.SPECIES_TYPE[sp_j]].sum(-1)).clamp(max=1.0)
        # Levitate: a switch-in resets the ability to the BASE one (`clearVolatile`: `this.ability = baseAbility`), and
        # Trace / Role Play / Skill Swap / Transform change only the CURRENT one, so the current-ability column
        # (`ability1_ids`) is the wrong read. In gen 3 Levitate is the SOLE ability of its 17 species (Gastly, Haunter,
        # Gengar, Koffing, Weezing, Misdreavus, Unown, Duskull, Lunatone, Solrock, Baltoy, Claydol, Chimecho, Vibrava,
        # Flygon, Latias, Latios; verified against Showdown's gen-3 pokedex and `data/`), so the species' Smogon
        # P(Levitate) (`SPECIES_TRAP_PRIOR[:, 3]`) is exactly 0 or 1 -- EXACT for every mon on both sides, revealed
        # or not. (If a species ever held Levitate beside another ability this would have to fall back to the known
        # ability for a revealed mon: `test_every_gen3_species_levitate_prior_is_zero_or_one` pins the premise.)
        gr_i = (1.0 - fly_i) * (1.0 - self.SPECIES_TRAP_PRIOR[sp_i, 3])
        gr_j = (1.0 - fly_j) * (1.0 - self.SPECIES_TRAP_PRIOR[sp_j, 3])
        # --- entry chip from each side's OWN hazards (spikes_feature = [our_side, opp_side] / 3) ---
        layers = (ctx.spikes_feature * 3.0).round().long().clamp(0, 3)           # [B,2]
        chip_our = chip_table[layers[:, 0]][:, None] * gr_i                       # [B,6]
        chip_opp = chip_table[layers[:, 1]][:, None] * gr_j
        return chip_our, chip_opp, gr_i, gr_j

    def pairwise_entry(self, ctx: 'ExtractorContext',
                       move_belief_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """gen3_edge_bias_trunk_v1 (X): the ENTRY/EXIT edge — what switching a mon IN or OUT costs,
        per mon, delivered at the (mon seat, GLOBAL seat) pair. → `(our_cells [B,6,4],
        opp_cells [B,6,4])`, cell `[entry_chip, pursuit_p, pursuit_eff, grounded]`:

          * entry_chip — gen3 Spikes on the mon's OWN entry side (1/2/3 layers → 1/8, 1/6, 1/4 of
            maxhp), grounded-gated (Flying / Levitate immune; opp Levitate revealed-exact else the
            SPECIES_TRAP_PRIOR levitate column — the T-family fold, reused).
          * pursuit_p — P(the OTHER side carries Pursuit): vs OUR mons the belief-composed max over
            alive opp slots (revealed rides pinned ≈1); vs OPP mons exact (our movesets are known).
          * pursuit_eff — Dark effectiveness at the victim's types (decorrelated from p).
        Victim-alive-gated; opp cells revealed-gated (unknown types). The "switching is not free"
        facts, attention-composable with every mon token via the global seat."""
        from agents.model.damage_tables import _pursuit_num, _T2I
        pur = _pursuit_num()
        dark = _T2I["DARK"]
        # --- the Spikes entry rule: ONE function (`spikes_entry`), shared with `--mon-hazard-cost`'s per-mon fact ---
        chip_our, chip_opp, gr_i, gr_j = self.spikes_entry(ctx)                   # [B,6] ×4
        # --- Pursuit exposure ---
        alive_i = (ctx.hp_and_active[:, :TEAM_SIZE, 0] > 0).float()
        alive_j = (ctx.hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, 0] > 0).float()
        _x5 = self.stash.x5
        if _x5 is None:
            w_all = torch.sigmoid(move_belief_logits) * self.HP_CAND_MASK[None, None, :]  # [B,6,M]
            p_pur_vs_us = (noisy_or(w_all[:, :, pur] * alive_j, keepdim=True) if self.op_principled
                           else max_by_index(w_all[:, :, pur] * alive_j, keepdim=True))     # [B,1]
        else:
            # X5 (U3 part 3): a class-M max over the opponent MONS (§9 M2 = C) — each live mon's Pursuit
            # presence (its fixed-mass move presence; 1 revealed) scaled by the MON's presence (1 revealed,
            # π a hypothesis).
            alive_j = _x5.alive
            if self.op_principled:
                # gen3_op_reduction_principled_v1: P(SOME mon of theirs holds Pursuit) — the noisy-OR over the
                # mons (and OTHER) of the same presence products the max read; max kept the likeliest holder's.
                p_pur_vs_us = noisy_or(_x5.slot_pi * _x5.move_w[:, :, pur] * alive_j, keepdim=True)
                if _x5.other_any is not None:
                    p_pur_vs_us = 1.0 - (1.0 - p_pur_vs_us) * (1.0 - (_x5.other_any * _x5.other_pursuit)[:, None])
            else:
                p_pur_vs_us = max_by_index(_x5.slot_pi * _x5.move_w[:, :, pur] * alive_j, keepdim=True)
                if _x5.other_any is not None:   # OTHER enters the max with presence 1 − Π(1 − π) (F4 (b))
                    p_pur_vs_us = torch.maximum(p_pur_vs_us, (_x5.other_any * _x5.other_pursuit)[:, None])
        we_have_pur = ((ctx.all_move_ids[:, :TEAM_SIZE] == pur).any(-1).float()
                       * alive_i).amax(dim=-1, keepdim=True)                      # [B,1]
        eff_i = self.CHART[ctx.type1_ids[:, :TEAM_SIZE]][..., dark]                 * self.CHART[ctx.type2_ids[:, :TEAM_SIZE]][..., dark]             # [B,6]
        eff_j = self.CHART[ctx.type1_ids[:, TEAM_SIZE:2 * TEAM_SIZE]][..., dark]                 * self.CHART[ctx.type2_ids[:, TEAM_SIZE:2 * TEAM_SIZE]][..., dark]
        revealed_j = (1.0 - ctx.opp_believed_mask.float())
        if _x5 is not None:      # a hypothesis's types (Dark eff, grounded) are its species'
            revealed_j = _x5.concrete
        our_cells = torch.stack([
            chip_our, p_pur_vs_us.expand(-1, TEAM_SIZE), eff_i.clamp(max=4.0) / 4.0, gr_i,
        ], dim=-1) * alive_i[:, :, None]                                          # [B,6,4]
        opp_cells = torch.stack([
            chip_opp, we_have_pur.expand(-1, TEAM_SIZE), eff_j.clamp(max=4.0) / 4.0, gr_j,
        ], dim=-1) * (alive_j * revealed_j)[:, :, None]                           # [B,6,4]
        return our_cells, opp_cells

    def pairwise_trap(self, ctx: 'ExtractorContext') -> torch.Tensor:
        """gen3_edge_bias_trunk_v1 (T): the mon↔mon TRAPPING edge — gen3-critical (Shadow Tag /
        Arena Trap / Magnet Pull; the `trap_core` archetype exists for this). `[B, TEAM_SIZE(our i),
        TEAM_SIZE(opp j), 2]` = `[P(our i traps opp j), P(opp j traps our i)]` per pair:
        `p = p_shadowtag + p_arenatrap·grounded(victim) + p_magnetpull·steel(victim)` (a mon has ONE
        ability → the events are disjoint; clamped ≤1). OUR trapper/victim abilities are KNOWN
        (exact one-hots); the OPP side uses revealed-ability-exact else the Smogon species prior
        (`SPECIES_TRAP_PRIOR`), and grounded folds the Levitate prior the same way. An UNREVEALED
        opp VICTIM is gated to 0 in direction A (its types are unknown — the D4 convention);
        both-alive-gated. "My Dugtrio traps their weakened Blissey" is a plan-defining edge."""
        our_ab = ctx.ability1_ids[:, :TEAM_SIZE]                                   # [B,6]
        # gen3_op_ability_known_v1: ability known = the `known` flag (an unrevealed id1 is the top-1 PRIOR).
        opp_ab, ab_rev_j, opp_sp = self.opp_ability_view(ctx)                      # [B,6] ×3
        revealed_j = (1.0 - ctx.opp_believed_mask.float())                         # [B,6] species known
        alive_i = (ctx.hp_and_active[:, :TEAM_SIZE, 0] > 0).float()
        alive_j = (ctx.hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, 0] > 0).float()
        _x5 = self.stash.x5
        if _x5 is not None:      # X5 (U3 part 3): a hypothesis's species is known under it; alive by addressability
            revealed_j, alive_j = _x5.concrete, _x5.alive
        both = alive_i[:, :, None] * alive_j[:, None, :]                           # [B,6i,6j]

        def _victim(steel_t1: torch.Tensor, steel_t2: torch.Tensor, fly_t1: torch.Tensor,
                    fly_t2: torch.Tensor,
                    p_lev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            steel = (steel_t1 + steel_t2).clamp(max=1.0)
            flying = (fly_t1 + fly_t2).clamp(max=1.0)
            grounded = (1.0 - flying) * (1.0 - p_lev)
            return steel, grounded

        # victim = opp j (direction A): types from revealed species; levitate revealed-exact else prior
        st_j, gr_j = _victim(self.TYPE_IS_STEEL[ctx.type1_ids[:, TEAM_SIZE:2 * TEAM_SIZE]],
                             self.TYPE_IS_STEEL[ctx.type2_ids[:, TEAM_SIZE:2 * TEAM_SIZE]],
                             self.TYPE_IS_FLYING[ctx.type1_ids[:, TEAM_SIZE:2 * TEAM_SIZE]],
                             self.TYPE_IS_FLYING[ctx.type2_ids[:, TEAM_SIZE:2 * TEAM_SIZE]],
                             ab_rev_j * self.ABILITY_IS_LEVITATE[opp_ab]
                             + (1.0 - ab_rev_j) * self.SPECIES_TRAP_PRIOR[opp_sp, 3])
        # victim = our i (direction B): everything exact
        st_i, gr_i = _victim(self.TYPE_IS_STEEL[ctx.type1_ids[:, :TEAM_SIZE]],
                             self.TYPE_IS_STEEL[ctx.type2_ids[:, :TEAM_SIZE]],
                             self.TYPE_IS_FLYING[ctx.type1_ids[:, :TEAM_SIZE]],
                             self.TYPE_IS_FLYING[ctx.type2_ids[:, :TEAM_SIZE]],
                             self.ABILITY_IS_LEVITATE[our_ab])
        # trapper probs [.., 3] = [shadowtag, arenatrap, magnetpull]
        tr_i = self.ABILITY_TRAP[our_ab]                                           # [B,6,3] exact
        tr_j = (ab_rev_j[:, :, None] * self.ABILITY_TRAP[opp_ab]
                + (1.0 - ab_rev_j)[:, :, None] * revealed_j[:, :, None]
                * self.SPECIES_TRAP_PRIOR[opp_sp, :3])                             # [B,6,3]
        p_a = (tr_i[:, :, None, 0] + tr_i[:, :, None, 1] * gr_j[:, None, :]
               + tr_i[:, :, None, 2] * st_j[:, None, :]).clamp(max=1.0)            # [B,6i,6j]
        p_a = p_a * revealed_j[:, None, :]                                         # unknown victim → 0
        p_b = (tr_j[:, None, :, 0] + tr_j[:, None, :, 1] * gr_i[:, :, None]
               + tr_j[:, None, :, 2] * st_i[:, :, None]).clamp(max=1.0)            # [B,6i,6j]
        return torch.stack([p_a * both, p_b * both], dim=-1)                       # [B,6,6,2]

    def pairwise_incoming(self, ctx: 'ExtractorContext',
                          move_belief_logits: torch.Tensor,
                          cand: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                          spread_belief: Optional[torch.Tensor] = None,
                          species_probs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """gen3_edge_bias_trunk_v1 (D3): the UN-collapsed per-(candidate, defender) incoming cells for the
        edge-bias delivery — `[B, K, TEAM_SIZE, 5]` = `[high, pko, eff, is_phys, w]` per (their believed
        move c, our mon i). The same `_incoming_rolls` physics and the same detached candidate selection as
        `discrete_incoming` (pass the SAME `cand` the E4 seats used so seat c and its bias row agree on
        which move c IS); the collapse is simply not taken, so attention sees WHICH candidate threatens
        WHICH defender instead of the worst-case max. Decorrelated: physics is w-independent, `w` rides as
        its own channel (the belief gradient path). Gated to 0 with no opp active / per fainted defender."""
        high, ko, eff, phys_k, w_topk, defender_alive, has_opp = self._incoming_rolls(
            ctx, move_belief_logits, cand, spread_belief=spread_belief, species_probs=species_probs)
        cells = torch.stack([
            high, ko, eff.clamp(max=4.0) / 4.0,
            phys_k[:, None, :].expand_as(high),
            w_topk[:, None, :].expand_as(high),
        ], dim=-1)                                                                        # [B,6,K,5]
        cells = cells * defender_alive[:, :, None, None] * has_opp[:, None, None, None]
        return cells.permute(0, 2, 1, 3).contiguous()                                     # [B,K,6,5]
