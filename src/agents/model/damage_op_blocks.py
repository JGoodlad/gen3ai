"""The DamageOperator's flat-BLOCK builders — the outgoing/incoming/status region producers
that fill the layout `damage_op_layout.py` declares (the outgoing block + matrix, the OAX
kernel that survives as d2's engine, status-landing, the incoming matrix + rolls, and the
discrete status probes).

A MIXIN, not a module: `DamageOperator` inherits this class — no parameters live here, the
state_dict is byte-identical, and the split changes nothing the heads see (the production sha
probe pins it). Split out of `damage_op.py` 2026-08-17 (one responsibility per file).
"""
import torch
from typing import Any, Callable, Optional, Tuple, TYPE_CHECKING
from agents.observation.constants import (
    TEAM_SIZE,
    POKEMON_SPREAD_OFFSET,
    POKEMON_SPREAD_DIM,
    POKEMON_CONDITION_OFFSET,
    POKEMON_SLEEP_BELIEF_OFFSET,
)
from agents.model.arch_constants import (  # noqa: F401  (re-export)
    ROLE_TOKEN_SIZE,
    PROJECTION_DIM,
    MOVE_NET_HIDDEN,
    MOVE_LATENT_HIDDEN,
    MOVE_LATENT_DIM,
    ROLE_ENCODER_HIDDEN,
    NET_ARCH,
    N_HISTORY_TURNS,
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
    _FIRE_TIDX, _IMMOBILIZE_STATUS_CATS, _LEECH_SEED_CTX_SLOT, _NAT_ATK, _NAT_DEF, _NAT_SPA,
    _NAT_SPD, _NAT_SPE, _N_OUT_SECONDARY, _OUT_SEC_COLS, _OUT_SEC_DROP, _OUT_SEC_KEEP,
    _PAIR_REDUCE_N_CHANNELS, _PTR_MOVE_CELL, _PTR_SWITCH_CELL_IN, _SB_ATK, _SB_DEF, _SB_SPA,
    _SB_SPD, _SB_SPE, _SECONDARY_MAJOR_N, _SECONDARY_TO_STATUS_CAT, _SUBSTITUTE_CTX_IDX,
    _TypeEncoder, _VOLATILE_SLOTS, _WATER_TIDX, _dmg_imx_dim, decode_damage_block,
)

if TYPE_CHECKING:  # no runtime import — `ctx` is only ever passed in, never constructed here
    from agents.model.extractor_ctx import ExtractorContext
    from agents.model.damage_op import OpStashes


class DamageOperatorBlocks:

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
        cb_item_num: int
        cb_phys_mult: float
        hp_bp: float
        hp_num: int
        stash: 'OpStashes'
        # Methods the COMPOSED class (`damage_op.DamageOperator`) owns and both mixins call.
        # Declared as callables rather than re-stated signatures: the definition is one file away
        # and mypy checks it there; this only stops the reads decaying to `Any`.
        _rolls: Callable[..., Tuple[torch.Tensor, ...]]
        _damage_rolls: Callable[..., Tuple[torch.Tensor, ...]]
        _boost_mult: Callable[..., torch.Tensor]
        _boost_stages: Callable[..., Tuple[torch.Tensor, ...]]
        _weather_mult: Callable[..., torch.Tensor]
        _chan_max: Callable[..., torch.Tensor]
        _p_outspeed: Callable[..., torch.Tensor]
        _opp_candidate_weights: Callable[..., torch.Tensor]
        # The damage TABLES (`damage_tables.build_damage_buffers`) are registered in a LOOP, so
        # they exist only dynamically; declaring them keeps every read a `Tensor` instead of the
        # `Any` the `__getattr__` above would hand back.
        ABILITY_DAMAGE_MULT: torch.Tensor
        ABILITY_SECONDARY_BLOCK: torch.Tensor
        ABILITY_SECONDARY_MULT: torch.Tensor
        ABILITY_STATUS_BLOCK: torch.Tensor
        BASE_STATS: torch.Tensor
        CHART: torch.Tensor
        MOVE_ACCURACY: torch.Tensor
        MOVE_BLOCKED_IF_STATUSED: torch.Tensor
        MOVE_BP: torch.Tensor
        MOVE_EFFECT_FLAGS: torch.Tensor
        MOVE_FIXED_DAMAGE: torch.Tensor
        MOVE_INFLICTS_STATUS: torch.Tensor
        MOVE_IS_SLEEP: torch.Tensor
        MOVE_PHYS: torch.Tensor
        MOVE_SECONDARY: torch.Tensor
        MOVE_STATUS_CAT: torch.Tensor
        MOVE_STATUS_TYPE_IMMUNE: torch.Tensor
        MOVE_TYPE_IDX: torch.Tensor
        SPECIES_EXP_MULT: torch.Tensor
        SPECIES_SPREAD_PRIOR: torch.Tensor
        SPECIES_STATUS_BLOCK_PRIOR: torch.Tensor
        SPECIES_USAGE_PRIOR: torch.Tensor
        TYPE_IS_PHYS: torch.Tensor
        _OUT_SEC_KEEP_IDX: torch.Tensor
        _SEC_CAT_IDX: torch.Tensor

    def _outgoing_block(self, ctx: 'ExtractorContext',
                        spread_belief: Optional[torch.Tensor] = None) -> torch.Tensor:
        """OUR active → opp active, PER MOVE in REQUEST-slot order (== action logits 6+k), so the policy
        head can compare move A vs B directly — the equal-effectiveness tie-break (Earthquake vs Meteor
        Mash into a Rock: same 2× multiplier, different resolved damage). Our moves are KNOWN (no belief —
        a hard one-hot), LEGALITY-MASKED via the action mask (Choice-lock / Disable / Taunt / no-PP); the
        opp DEFENDER's bulk is hidden → a NEUTRAL 0-EV estimate (not max-bulk, which would under-price our
        KOs); opp ability immunity is revealed-or-none; OPP-side screens apply. Output `[B, _DMG_OUTGOING]`:
        per move `[low, high, crit, pko]` + one `p_outspeed`. Reuses the shared `_rolls` formula. Leak-safe
        (public obs only); gated to 0 when there is no opp active OR our active is fainted/absent. Our moves
        are certain → no move-belief gradient (correct: we don't learn our own moves), but differentiable
        in the smooth stat/damage formula."""
        B, device, eps = ctx.batch_size, ctx.device, 1e-6
        ar = torch.arange(B, device=device)
        our_act = ctx.our_active_idx                                 # [B] our active slot (0..5)
        opp_act = TEAM_SIZE + ctx.opp_active_local                   # [B] opp active global slot
        has_opp = ctx.hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, -1].any(dim=1).float()  # [B]
        our_alive = (ctx.hp_and_active[ar, our_act, 0] > 0).float()  # [B] our active must exist + be alive
        gate = (has_opp * our_alive)[:, None]                        # [B,1]

        # --- our 4 moves in REQUEST-slot order (action logits 6+k), legality-masked ---
        # gen3_op_move_align_v1: read the request-ordered obs slice (NOT all_move_ids[our_act], which is
        # sorted-by-id), so slot k's output ↔ action 6+k. `legal` is the CURRENT-decision choosability in
        # request order (was ctx.move_mask = prev-turn, sorted-by-id — a stale + misordered gate).
        move_ids = ctx.our_active_req_move_ids                       # [B,4] request order
        move_ty = ctx.our_active_req_move_type_ids                   # [B,4] resolved type (incl our HP type)
        legal = ctx.our_active_req_move_legal                        # [B,4] currently-legal (Choice/Disable/PP)
        is_hp = (move_ids == self.hp_num)
        bp = torch.where(is_hp, torch.full_like(move_ty, self.hp_bp, dtype=torch.float32),
                         self.MOVE_BP[move_ids])                     # [B,4] HP → 70 (else dex BP; status → 0)
        phys = self.TYPE_IS_PHYS[move_ty]                          # [B,4] gen3 category by resolved type
        acc = self.MOVE_ACCURACY[move_ids]                        # [B,4] (HP num → 1.0 default)
        usable = legal * (bp > 0).float()                         # [B,4] gate to legal damaging moves

        # --- our active attacker (real spread) ---
        a_base = self.BASE_STATS[ctx.species_ids[ar, our_act]]     # [B,6] [hp,atk,def,spa,spd,spe]
        spread = ctx.pokemon_part[ar, our_act,
                                  POKEMON_SPREAD_OFFSET:POKEMON_SPREAD_OFFSET + POKEMON_SPREAD_DIM]
        iv = spread[:, 0:6] * 31.0
        ev = spread[:, 6:12] * 252.0
        nat = spread[:, 13:18]                                     # [B,5] [atk,def,spa,spd,spe]
        our_atk = (2.0 * a_base[:, 1] + iv[:, 1] + ev[:, 1] / 4.0 + 5.0) * nat[:, 0]   # [B]
        our_spa = (2.0 * a_base[:, 3] + iv[:, 3] + ev[:, 3] / 4.0 + 5.0) * nat[:, 2]   # [B]
        our_spe = (2.0 * a_base[:, 5] + iv[:, 5] + ev[:, 5] / 4.0 + 5.0) * nat[:, 4]   # [B]
        # gen3_unified_op_physics_v1: OUR active's offensive + speed stat-stage boosts (we attack here) +
        # BURN (½ phys atk) + PARALYSIS (×0.25 speed).
        o_b_atk, o_b_def, o_b_spa, o_b_spd, o_b_spe = self._boost_stages(ctx.our_ctx_raw)
        our_burn = ctx.pokemon_part[ar, our_act, POKEMON_CONDITION_OFFSET + _COND_BRN_IDX]   # [B]
        our_para = ctx.pokemon_part[ar, our_act, POKEMON_CONDITION_OFFSET + _COND_PAR_IDX]   # [B]
        # gen3_unified_choice_band_v1: OUR Choice Band ×1.5 physical Atk (our item is KNOWN → deterministic,
        # not a belief). Composes multiplicatively with boosts/burn below; physical only (CB doesn't touch SpA).
        our_cb = (ctx.item_ids[ar, our_act] == self.cb_item_num).float()                     # [B]
        our_atk = our_atk * torch.where(our_cb > 0.5, our_atk.new_tensor(self.cb_phys_mult),
                                        our_atk.new_tensor(1.0))
        our_atk = our_atk * self._boost_mult(o_b_atk) * torch.where(
            our_burn > 0.5, our_atk.new_tensor(0.5), our_atk.new_tensor(1.0))
        our_spa = our_spa * self._boost_mult(o_b_spa)
        our_spe = our_spe * self._boost_mult(o_b_spe) * torch.where(
            our_para > 0.5, our_spe.new_tensor(_DMG_PARA_SPEED), our_spe.new_tensor(1.0))
        at1 = ctx.type1_ids[ar, our_act]                          # [B] our types (STAB)
        at2 = ctx.type2_ids[ar, our_act]

        # --- opp active defender (revealed species/types; ability revealed-or-none) ---
        # Bulk: the SpreadBelief's learned def/spd if provided (gen3_unified_spread_belief_v1), else the
        # legacy NEUTRAL 0-EV estimate (not max-bulk, which would under-price our KOs). maxhp stays the
        # neutral estimate either way (HP EVs vary little + the obs HP fraction carries relative HP).
        d_base = self.BASE_STATS[ctx.species_ids[ar, opp_act]]     # [B,6]
        bs = spread_belief[ar, ctx.opp_active_local] if spread_belief is not None else None  # [B,5] or None
        opp_def = bs[:, _SB_DEF] if bs is not None else (2.0 * d_base[:, 2] + 31.0 + 5.0)    # [B]
        opp_spd = bs[:, _SB_SPD] if bs is not None else (2.0 * d_base[:, 4] + 31.0 + 5.0)
        opp_maxhp = 2.0 * d_base[:, 0] + 31.0 + 110.0
        opp_spe = bs[:, _SB_SPE] if bs is not None else (2.0 * d_base[:, 5] + 31.0 + 5.0)   # believed / neutral
        # gen3_unified_op_physics_v1: OPP active's DEFENSIVE + speed boosts (it's the defender here) + its
        # paralysis (×0.25 speed, for p_outspeed).
        p_b_atk, p_b_def, p_b_spa, p_b_spd, p_b_spe = self._boost_stages(ctx.opp_ctx_raw)
        opp_para = ctx.pokemon_part[ar, opp_act, POKEMON_CONDITION_OFFSET + _COND_PAR_IDX]   # [B]
        opp_def = opp_def * self._boost_mult(p_b_def)
        opp_spd = opp_spd * self._boost_mult(p_b_spd)
        opp_spe = opp_spe * self._boost_mult(p_b_spe) * torch.where(
            opp_para > 0.5, opp_spe.new_tensor(_DMG_PARA_SPEED), opp_spe.new_tensor(1.0))
        opp_cur_hp = ctx.hp_and_active[ar, opp_act, 0] * opp_maxhp  # [B] obs HP frac × est. max HP
        t1d = ctx.type1_ids[ar, opp_act]                          # [B]
        t2d = ctx.type2_ids[ar, opp_act]
        opp_ability = ctx.ability1_ids[ar, opp_act]              # [B] (0 if unrevealed → no immunity mult)

        # --- gen3 damage per move (defender = opp active, candidates = our 4 moves), via the shared rolls ---
        eff = self.CHART[t1d[:, None], move_ty] * self.CHART[t2d[:, None], move_ty]   # [B,4]
        eff = eff * self.ABILITY_DAMAGE_MULT[opp_ability].gather(1, move_ty)          # [B,4] defender immunity
        A = phys * our_atk[:, None] + (1.0 - phys) * our_spa[:, None]                 # [B,4]
        D = phys * opp_def[:, None] + (1.0 - phys) * opp_spd[:, None]                 # [B,4]
        is_stab = ((move_ty == at1[:, None]) | (move_ty == at2[:, None])).float()
        stab = 1.0 + 0.5 * is_stab
        core = 42.0 * bp * A / (D + eps) / 50.0 + 2.0
        weather_mult = self._weather_mult(ctx.weather_feature, (move_ty == _WATER_TIDX).float(),
                                          (move_ty == _FIRE_TIDX).float())             # [B,4] rain/sun
        dmg_ns = core * stab * eff * 0.925 * usable * weather_mult                    # [B,4] (non-usable → 0)
        opp_reflect = ctx.screen_feature[:, 1:2]                                      # OPP-side screens
        opp_ls = ctx.screen_feature[:, 3:4]
        screen = 1.0 - 0.5 * (opp_reflect * phys + opp_ls * (1.0 - phys))             # [B,4]
        high, low, crit, ko = self._rolls(dmg_ns, screen, opp_maxhp[:, None], opp_cur_hp[:, None], acc, eps)
        # gen3_unified_op_physics_v1: OUR fixed-damage moves (Seismic Toss into the opp), immunity-gated +
        # legality-gated (usable). Mirrors the incoming kernel's override.
        fixed = self.MOVE_FIXED_DAMAGE[move_ids] * usable                            # [B,4] (0 if illegal)
        is_fixed = fixed > 0
        not_immune = (eff > 0).float()                                               # [B,4] type+ability gate
        fixed_frac = (fixed / (opp_maxhp[:, None] + eps)) * not_immune
        fixed_ko = acc * (fixed >= opp_cur_hp[:, None]).float() * not_immune
        high = torch.where(is_fixed, fixed_frac, high)
        low = torch.where(is_fixed, fixed_frac, low)
        crit = torch.where(is_fixed, fixed_frac, crit)
        ko = torch.where(is_fixed, fixed_ko, ko)
        opp_spe_std = self.SPECIES_SPREAD_PRIOR[ctx.species_ids[ar, opp_act], _SB_SPE, 1]   # [B] (#3)
        p_outspeed = self._p_outspeed(our_spe, opp_spe, opp_spe_std)                  # [B]

        # gen3_unified_move_system_v1: per OUR move, "what status can it cause + with what probability".
        # realized P(effect k | move) = chance_mk × acc_m × Serene Grace(our active) × Shield Dust(opp
        # active), gated to legal moves (status moves carry 0 secondary → naturally zeroed). Order ==
        # `_OUT_SEC_COLS` = SECONDARY_COLS minus slp/psn/tox (gen3_op_block_trim_v1 — those three carry no
        # move any pool team runs, so they were structural zeros). [B,4,7].
        our_serene = self.ABILITY_SECONDARY_MULT[ctx.ability1_ids[ar, our_act]]        # [B] our active
        opp_block = self.ABILITY_SECONDARY_BLOCK[opp_ability]                          # [B] opp Shield Dust
        sec = self.MOVE_SECONDARY[move_ids][..., self._OUT_SEC_KEEP_IDX]                # [B,4,7] base chance
        sec = sec * (acc * legal)[:, :, None] * (our_serene * opp_block)[:, None, None]
        sec = sec.clamp(max=1.0)                                                        # [B,4,7]

        per_move = torch.stack([low, high, crit, ko], dim=-1)                          # [B,4,4]
        block = torch.cat([per_move.reshape(B, -1), p_outspeed[:, None], sec.reshape(B, -1)], dim=1)  # [B, _DMG_OUTGOING]
        return block * gate

    def unrevealed_species_probs(self, ctx: 'ExtractorContext',
                                 species_probs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """gen3_unrevealed_outgoing_prior_v1: ``[B, TEAM_SIZE, n_species]`` P(species) per OPP slot —
        the gen3ou usage prior (`SPECIES_USAGE_PRIOR`) under SPECIES CLAUSE: every species already
        REVEALED on the opponent's side this battle (`ctx.species_ids[opp]` where `~opp_believed_mask`)
        is zeroed out, then the distribution renormalizes. Rows are meaningful only at UNREVEALED
        slots (`ctx.opp_believed_mask`) — the expected-latent defender read; revealed slots' rows are
        never consumed. `species_probs` (e.g. a learned belief posterior ``[B,6,n_species]``) overrides
        the prior entirely (returned as-is) — the future-learned-belief seam."""
        if species_probs is not None:
            return species_probs
        B, device = ctx.batch_size, ctx.device
        n_species = self.SPECIES_USAGE_PRIOR.shape[0]
        # Species Clause: scatter zeros at every revealed opp species num. Unrevealed slots contribute
        # the sentinel column 0, whose prior mass is exactly 0 by construction — a harmless no-op.
        revealed_num = ctx.species_ids[:, TEAM_SIZE:2 * TEAM_SIZE] * (~ctx.opp_believed_mask).long()
        keep = torch.ones(B, n_species, device=device).scatter(1, revealed_num, 0.0)
        p = self.SPECIES_USAGE_PRIOR[None, :] * keep                                  # [B, n_species]
        # ⚠️ SHAPE is load-bearing for torch.compile (the gen3_species_posterior_spelling_v1
        # precedent, this module's second instance): returning the marginal EXPANDED to
        # [B, TEAM_SIZE, n_species] fused the normalize + expand into an Inductor CPU kernel
        # that .store()d a SCALAR lane as if vectorized (`request for member 'store' in
        # 'tmp6', which is of non-class type 'float'` — the gen-4 launch CompilePrewarm
        # failure; respelling the div did NOT dodge it). The prior marginal is IDENTICAL for
        # all six slots anyway, so it stays [B, n_species] and consumers broadcast the
        # RESULTS (also 6× less matmul). A learned per-slot override keeps full rank.
        return p / p.sum(dim=1, keepdim=True).clamp_min(1e-12)

    def _outgoing_matrix(self, ctx: 'ExtractorContext',
                         spread_belief: Optional[torch.Tensor] = None,
                         boost_delta: Optional[torch.Tensor] = None,
                         species_probs: Optional[torch.Tensor] = None) -> torch.Tensor:
        """gen3_per_move_matrices_v1: OUR active's 4 moves → the opp's 6 mons (active + bench). The
        bench extension of `_outgoing_block` (which prices only the opp active): per (our move k, opp mon d)
        a `[low, high, crit, pko, type_mult]` cell so the policy prices a KO on a SWITCH-IN, plus a per-opp-mon
        `revealed` bit. An UNREVEALED opp slot (Gen3 no team preview) is priced against the EXPECTED-LATENT
        defender (gen3_unrevealed_outgoing_prior_v1 — the Species-Clause-filtered usage prior, or the
        `species_probs` override): E[mult]/E[def/spd]/E[maxhp] marginals, forced-alive, P(KO) NULLED, the
        `revealed` bit still 0. Reuses the `_outgoing_block` physics
        (attacker = our active with CB/boost/burn; OPP-side screens; per-defender bulk = SpreadBelief or the
        neutral 0-EV estimate; only the opp ACTIVE carries boosts — bench is reset), broadcast over the 6
        defenders. Output `[B, _DMG_OMX]` (grouped by move, action-aligned). Gated to 0 with no opp / fainted
        or absent our active. Leak-safe (revealed species + the fixed usage prior + our known moves only)."""
        B, device, eps = ctx.batch_size, ctx.device, 1e-6
        ar = torch.arange(B, device=device)
        our_act = ctx.our_active_idx
        opp = slice(TEAM_SIZE, 2 * TEAM_SIZE)
        has_opp = ctx.hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, -1].any(dim=1).float()  # [B]
        our_alive = (ctx.hp_and_active[ar, our_act, 0] > 0).float()                     # [B]
        gate = (has_opp * our_alive)[:, None]                                           # [B,1]

        # --- our 4 moves in REQUEST-slot order (action 6+k), legality-masked (== _outgoing_block) ---
        # gen3_op_move_align_v1: request-ordered obs slice + current-decision legality (see _outgoing_block).
        move_ids = ctx.our_active_req_move_ids                                          # [B,4] request order
        move_ty = ctx.our_active_req_move_type_ids                                      # [B,4]
        legal = ctx.our_active_req_move_legal                                           # [B,4]
        is_hp = (move_ids == self.hp_num)
        bp = torch.where(is_hp, torch.full_like(move_ty, self.hp_bp, dtype=torch.float32),
                         self.MOVE_BP[move_ids])                                        # [B,4]
        phys = self.TYPE_IS_PHYS[move_ty]                                               # [B,4]
        acc = self.MOVE_ACCURACY[move_ids]                                              # [B,4]
        usable = legal * (bp > 0).float()                                               # [B,4]

        # --- our active attacker (real spread; CB ×1.5 phys, offensive boosts, burn) — same as _outgoing_block ---
        a_base = self.BASE_STATS[ctx.species_ids[ar, our_act]]
        spr = ctx.pokemon_part[ar, our_act, POKEMON_SPREAD_OFFSET:POKEMON_SPREAD_OFFSET + POKEMON_SPREAD_DIM]
        iv = spr[:, 0:6] * 31.0; ev = spr[:, 6:12] * 252.0; nat = spr[:, 13:18]
        our_atk = (2.0 * a_base[:, 1] + iv[:, 1] + ev[:, 1] / 4.0 + 5.0) * nat[:, 0]
        our_spa = (2.0 * a_base[:, 3] + iv[:, 3] + ev[:, 3] / 4.0 + 5.0) * nat[:, 2]
        o_b_atk, _odf, o_b_spa, _osd, _ose = self._boost_stages(ctx.our_ctx_raw)
        if boost_delta is not None:
            # gen3_edge_bias_trunk_v1 (C1): price the HYPOTHETICAL post-setup world — advance the
            # offensive stages by the setup move's deltas (the _boost_mult clamp bounds ±6). None
            # (every existing caller) is byte-identical to the pre-C1 kernel.
            o_b_atk = o_b_atk + boost_delta[:, 0]
            o_b_spa = o_b_spa + boost_delta[:, 2]
        our_burn = ctx.pokemon_part[ar, our_act, POKEMON_CONDITION_OFFSET + _COND_BRN_IDX]
        our_cb = (ctx.item_ids[ar, our_act] == self.cb_item_num).float()
        our_atk = our_atk * torch.where(our_cb > 0.5, our_atk.new_tensor(self.cb_phys_mult), our_atk.new_tensor(1.0))
        our_atk = our_atk * self._boost_mult(o_b_atk) * torch.where(
            our_burn > 0.5, our_atk.new_tensor(0.5), our_atk.new_tensor(1.0))
        our_spa = our_spa * self._boost_mult(o_b_spa)
        at1 = ctx.type1_ids[ar, our_act]; at2 = ctx.type2_ids[ar, our_act]              # [B] (STAB)
        A = phys * our_atk[:, None] + (1.0 - phys) * our_spa[:, None]                   # [B,4]

        # --- opp 6 defenders (bulk = SpreadBelief or neutral 0-EV; boosts only on the active slot) ---
        d_base = self.BASE_STATS[ctx.species_ids[:, opp]]                               # [B,6,6]
        if spread_belief is not None:
            opp_def = spread_belief[:, :, _SB_DEF]; opp_spd = spread_belief[:, :, _SB_SPD]   # [B,6]
        else:
            opp_def = 2.0 * d_base[..., 2] + 31.0 + 5.0; opp_spd = 2.0 * d_base[..., 4] + 31.0 + 5.0
        opp_maxhp = 2.0 * d_base[..., 0] + 31.0 + 110.0                                 # [B,6]
        # gen3_unrevealed_outgoing_prior_v1: UNREVEALED defender slots are priced against the
        # EXPECTED-LATENT defender instead of zeroed (design_conditional_opponent_cells §4.1 — a
        # revealed-gated read is misleading exactly when switching matters most, the typeless-HP-
        # immune bug class). The v36 `discrete_outgoing` marginals, at the Species-Clause-filtered
        # usage prior (or the `species_probs` override): E[mult] via SPECIES_EXP_MULT, E[def/spd]
        # via the spread-prior means, E[maxhp] via E[base HP], forced-alive full-HP switch-in.
        # P(KO) stays NULLED (owner rule: a full-HP switch-in is ~never OHKO'd) and the trailing
        # `revealed` channel stays 0 — magnitudes change, epistemics don't. REVEALED slots pass
        # through every `torch.where` untouched — byte-identical to the pre-fix kernel.
        believed = ctx.opp_believed_mask                                                # [B,6] bool
        sp_probs = self.unrevealed_species_probs(ctx, species_probs)                    # [B,S] | [B,6,S]
        e_bulk = sp_probs @ self.SPECIES_SPREAD_PRIOR[..., 0]                           # [B,5] | [B,6,5]
        e_maxhp = 2.0 * (sp_probs @ self.BASE_STATS[:, 0]) + 31.0 + 110.0               # [B] | [B,6]
        if sp_probs.dim() == 2:
            # Prior path: one marginal per battle — broadcast the RESULTS over the 6 slots
            # (see unrevealed_species_probs: the [B,6,S] expand mis-vectorizes under Inductor).
            e_def_u, e_spd_u, e_maxhp_u = (e_bulk[:, _SB_DEF, None], e_bulk[:, _SB_SPD, None],
                                           e_maxhp[:, None])
        else:
            e_def_u, e_spd_u, e_maxhp_u = e_bulk[..., _SB_DEF], e_bulk[..., _SB_SPD], e_maxhp
        opp_def = torch.where(believed, e_def_u, opp_def)
        opp_spd = torch.where(believed, e_spd_u, opp_spd)
        opp_maxhp = torch.where(believed, e_maxhp_u, opp_maxhp)
        _pa, p_b_def, _ps, p_b_spd, _pe = self._boost_stages(ctx.opp_ctx_raw)
        def_boost = torch.ones_like(opp_def); def_boost[ar, ctx.opp_active_local] = self._boost_mult(p_b_def)
        spd_boost = torch.ones_like(opp_spd); spd_boost[ar, ctx.opp_active_local] = self._boost_mult(p_b_spd)
        opp_def = opp_def * def_boost; opp_spd = opp_spd * spd_boost
        opp_hp_frac = ctx.hp_and_active[:, opp, 0]                                      # [B,6]
        opp_hp_frac = torch.where(believed, torch.ones_like(opp_hp_frac), opp_hp_frac)  # hidden = full-HP switch-in
        opp_cur_hp = opp_hp_frac * opp_maxhp                                            # [B,6]
        t1d = ctx.type1_ids[:, opp]; t2d = ctx.type2_ids[:, opp]                        # [B,6]
        opp_ability = ctx.ability1_ids[:, opp]                                          # [B,6]
        revealed = (~ctx.opp_believed_mask).float()                                     # [B,6] species known
        def_gate = revealed * (opp_hp_frac > 0).float()                                 # [B,6] revealed live target
        target_gate = def_gate + believed.float()                                       # [B,6] + hidden forced-alive

        # --- type effectiveness eff[B,4,6] = CHART[t1d]·CHART[t2d]·ability_mult, gathered by move type ---
        T = self.CHART.shape[-1]
        mty_e = move_ty[:, :, None, None].expand(B, _DMG_OUT_N_MOVES, TEAM_SIZE, 1)      # [B,4,6,1]
        def _gather_type(table_per_def: torch.Tensor) -> torch.Tensor:                                                 # table [B,6,T] → [B,4,6]
            t = table_per_def[:, None].expand(B, _DMG_OUT_N_MOVES, TEAM_SIZE, T)
            return torch.gather(t, 3, mty_e).squeeze(-1)
        eff = (_gather_type(self.CHART[t1d]) * _gather_type(self.CHART[t2d])
               * _gather_type(self.ABILITY_DAMAGE_MULT[opp_ability]))                    # [B,4,6]
        # gen3_unrevealed_outgoing_prior_v1: E[mult] (type chart × expected ability immunity, one
        # matmul with P(species)) replaces the sentinel-neutral chart read at unrevealed slots.
        e_mult = sp_probs @ self.SPECIES_EXP_MULT                                        # [B,T] | [B,6,T]
        if e_mult.dim() == 2:
            eff_unrev = e_mult.gather(1, move_ty.long())[:, :, None]                     # [B,4,1] → bcast
        else:
            eff_unrev = _gather_type(e_mult)                                             # [B,4,6]
        eff = torch.where(believed[:, None, :], eff_unrev, eff)
        # --- gen3 damage per (move, defender) → [B,4,6] (the _outgoing_block physics, broadcast over 6) ---
        D = phys[:, :, None] * opp_def[:, None, :] + (1.0 - phys)[:, :, None] * opp_spd[:, None, :]   # [B,4,6]
        is_stab = ((move_ty == at1[:, None]) | (move_ty == at2[:, None])).float()        # [B,4]
        stab = (1.0 + 0.5 * is_stab)[:, :, None]                                          # [B,4,1]
        core = 42.0 * bp[:, :, None] * A[:, :, None] / (D + eps) / 50.0 + 2.0             # [B,4,6]
        weather = self._weather_mult(ctx.weather_feature, (move_ty == _WATER_TIDX).float(),
                                     (move_ty == _FIRE_TIDX).float())[:, :, None]         # [B,4,1]
        dmg_ns = core * stab * eff * 0.925 * usable[:, :, None] * weather                 # [B,4,6]
        opp_reflect = ctx.screen_feature[:, 1:2]; opp_ls = ctx.screen_feature[:, 3:4]     # OPP-side screens
        screen = (1.0 - 0.5 * (opp_reflect * phys + opp_ls * (1.0 - phys)))[:, :, None]   # [B,4,1]
        high, low, crit, ko = self._rolls(dmg_ns, screen, opp_maxhp[:, None, :],
                                          opp_cur_hp[:, None, :], acc[:, :, None], eps)    # each [B,4,6]
        # fixed-damage moves (Seismic Toss into a defender): immunity + legality gated, CB-invariant.
        fixed = (self.MOVE_FIXED_DAMAGE[move_ids] * usable)[:, :, None]                   # [B,4,1]
        is_fixed = fixed > 0
        not_immune = (eff > 0).float()                                                   # [B,4,6]
        fixed_frac = (fixed / (opp_maxhp[:, None, :] + eps)) * not_immune
        fixed_ko = acc[:, :, None] * (fixed >= opp_cur_hp[:, None, :]).float() * not_immune
        high = torch.where(is_fixed, fixed_frac, high); low = torch.where(is_fixed, fixed_frac, low)
        crit = torch.where(is_fixed, fixed_frac, crit); ko = torch.where(is_fixed, fixed_ko, ko)
        ko = ko * revealed[:, None, :]        # gen3_unrevealed_outgoing_prior_v1: P(KO) NULLED at hidden slots

        cell = torch.stack([low, high, crit, ko, eff], dim=-1)                            # [B,4,6,_DMG_OMX_CELL]
        cell = cell * (usable[:, :, None, None] * target_gate[:, None, :, None])          # gate move-legal × target
        out = torch.cat([cell.reshape(B, _DMG_OUT_N_MOVES * TEAM_SIZE * _DMG_OMX_CELL), def_gate], dim=1)  # [B,_DMG_OMX]
        return out * gate

    def _outgoing_attacker_matrix(self, ctx: 'ExtractorContext',
                                  spread_belief: Optional[torch.Tensor] = None,
                                  inherit_stages: bool = False) -> torch.Tensor:
        """gen3_per_move_matrices_v1 (v39): the TRANSPOSE of `_outgoing_matrix` — our 6 MONS' 4 moves → the opp
        ACTIVE. On a FORCED SWITCH our active is fainted, so `_outgoing_block`/`_outgoing_matrix` (which only
        price the current active attacker) ZERO and the policy picks switch-ins BLIND to offense; this prices
        every candidate switch-in's offense vs the opp active. The ACTIVE row reproduces `_outgoing_block`
        byte-for-byte (its boosts/CB/burn + request-ordered moves + the same opp-active defender + the same
        `_rolls` kernel); bench rows reuse the SAME `_rolls` physics with NEUTRAL boosts (gen3 resets boosts on
        switch) + the per-mon sorted-by-id moves (bench mons have no current-decision request order). Output
        `[B, _DMG_OAX]` = all (attacker, move) cells `[low,high,crit,pko]` ++ a per-attacker `p_outspeed[6]` ++
        an `alive[6]` gate. Gated to 0 with no opp active; each attacker gated by its alive bit. Leak-safe
        (public obs + the believed opp spread only — same inputs as `_outgoing_block`)."""
        B, device, eps = ctx.batch_size, ctx.device, 1e-6
        ar = torch.arange(B, device=device)
        our = slice(0, TEAM_SIZE)
        opp_act = TEAM_SIZE + ctx.opp_active_local                     # [B] opp active global slot
        has_opp = ctx.hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, -1].any(dim=1).float()  # [B] (== _outgoing_block)

        # --- our 6 attackers' moves: per-mon sorted-by-id block, ACTIVE row OVERWRITTEN with the request slice ---
        # The active uses the SAME request-ordered slice _outgoing_block reads (==action 6+k) → byte-for-byte
        # parity on the active row; bench mons (no request order) use all_move_ids[:, :TEAM_SIZE] (sorted-by-id).
        move_ids = ctx.all_move_ids[:, our].clone()                    # [B,6,4] sorted-by-id
        move_ty = ctx.all_move_type_ids[:, our].clone()                # [B,6,4]
        legal = torch.ones(B, TEAM_SIZE, _DMG_OAX_N_MOVES, device=device)   # bench: all moves available
        move_ids[ar, ctx.our_active_idx] = ctx.our_active_req_move_ids        # active → request order (parity)
        move_ty[ar, ctx.our_active_idx] = ctx.our_active_req_move_type_ids
        legal[ar, ctx.our_active_idx] = ctx.our_active_req_move_legal         # active → current-decision legality
        is_hp = (move_ids == self.hp_num)
        bp = torch.where(is_hp, torch.full_like(move_ty, self.hp_bp, dtype=torch.float32),
                         self.MOVE_BP[move_ids])                       # [B,6,4]
        phys = self.TYPE_IS_PHYS[move_ty]                              # [B,6,4]
        acc = self.MOVE_ACCURACY[move_ids]                            # [B,6,4]
        usable = legal * (bp > 0).float()                            # [B,6,4] legal damaging moves

        # --- our 6 attackers (real spread; CB ×1.5 phys, burn; boosts only on the active slot, bench reset) ---
        a_base = self.BASE_STATS[ctx.species_ids[:, our]]            # [B,6,6] [hp,atk,def,spa,spd,spe]
        spread = ctx.pokemon_part[:, our,
                                  POKEMON_SPREAD_OFFSET:POKEMON_SPREAD_OFFSET + POKEMON_SPREAD_DIM]   # [B,6,SPREAD]
        iv = spread[..., 0:6] * 31.0
        ev = spread[..., 6:12] * 252.0
        nat = spread[..., 13:18]                                     # [B,6,5] [atk,def,spa,spd,spe]
        our_atk = (2.0 * a_base[..., 1] + iv[..., 1] + ev[..., 1] / 4.0 + 5.0) * nat[..., 0]   # [B,6]
        our_spa = (2.0 * a_base[..., 3] + iv[..., 3] + ev[..., 3] / 4.0 + 5.0) * nat[..., 2]   # [B,6]
        our_spe = (2.0 * a_base[..., 5] + iv[..., 5] + ev[..., 5] / 4.0 + 5.0) * nat[..., 4]   # [B,6]
        # Boosts: the ACTIVE row carries our_ctx_raw's stages; bench rows neutral (mult 1.0) — gen3 resets on
        # switch (mirrors _outgoing_matrix's defender-boost handling exactly). `inherit_stages`
        # (C5 Baton Pass) is the HYPOTHETICAL post-pass world: EVERY row gets the active's stages
        # (the receiver inherits them) — False is byte-identical to the pre-C5 kernel.
        o_b_atk, _odf, o_b_spa, _osd, o_b_spe = self._boost_stages(ctx.our_ctx_raw)   # active only, [B]
        if inherit_stages:
            atk_boost = self._boost_mult(o_b_atk)[:, None].expand(B, TEAM_SIZE)
            spa_boost = self._boost_mult(o_b_spa)[:, None].expand(B, TEAM_SIZE)
            spe_boost = self._boost_mult(o_b_spe)[:, None].expand(B, TEAM_SIZE)
        else:
            atk_boost = torch.ones(B, TEAM_SIZE, device=device); atk_boost[ar, ctx.our_active_idx] = self._boost_mult(o_b_atk)
            spa_boost = torch.ones(B, TEAM_SIZE, device=device); spa_boost[ar, ctx.our_active_idx] = self._boost_mult(o_b_spa)
            spe_boost = torch.ones(B, TEAM_SIZE, device=device); spe_boost[ar, ctx.our_active_idx] = self._boost_mult(o_b_spe)
        # Burn / Choice Band compose PER MON (each mon's own KNOWN condition/item).
        our_burn = ctx.pokemon_part[:, our, POKEMON_CONDITION_OFFSET + _COND_BRN_IDX]   # [B,6]
        our_para = ctx.pokemon_part[:, our, POKEMON_CONDITION_OFFSET + _COND_PAR_IDX]   # [B,6]
        our_cb = (ctx.item_ids[:, our] == self.cb_item_num).float()                     # [B,6]
        our_atk = our_atk * torch.where(our_cb > 0.5, our_atk.new_tensor(self.cb_phys_mult), our_atk.new_tensor(1.0))
        our_atk = our_atk * atk_boost * torch.where(our_burn > 0.5, our_atk.new_tensor(0.5), our_atk.new_tensor(1.0))
        our_spa = our_spa * spa_boost
        our_spe = our_spe * spe_boost * torch.where(our_para > 0.5, our_spe.new_tensor(_DMG_PARA_SPEED),
                                                    our_spe.new_tensor(1.0))
        at1 = ctx.type1_ids[:, our]; at2 = ctx.type2_ids[:, our]                        # [B,6] (STAB)
        A = phys * our_atk[:, :, None] + (1.0 - phys) * our_spa[:, :, None]             # [B,6,4]

        # --- opp ACTIVE defender (identical to _outgoing_block: believed/neutral bulk, defensive boosts) ---
        d_base = self.BASE_STATS[ctx.species_ids[ar, opp_act]]        # [B,6]
        bs = spread_belief[ar, ctx.opp_active_local] if spread_belief is not None else None   # [B,5] or None
        opp_def = bs[:, _SB_DEF] if bs is not None else (2.0 * d_base[:, 2] + 31.0 + 5.0)     # [B]
        opp_spd = bs[:, _SB_SPD] if bs is not None else (2.0 * d_base[:, 4] + 31.0 + 5.0)
        opp_maxhp = 2.0 * d_base[:, 0] + 31.0 + 110.0
        opp_spe = bs[:, _SB_SPE] if bs is not None else (2.0 * d_base[:, 5] + 31.0 + 5.0)
        _pa, p_b_def, _ps, p_b_spd, p_b_spe = self._boost_stages(ctx.opp_ctx_raw)
        opp_para = ctx.pokemon_part[ar, opp_act, POKEMON_CONDITION_OFFSET + _COND_PAR_IDX]   # [B]
        opp_def = opp_def * self._boost_mult(p_b_def)
        opp_spd = opp_spd * self._boost_mult(p_b_spd)
        opp_spe = opp_spe * self._boost_mult(p_b_spe) * torch.where(
            opp_para > 0.5, opp_spe.new_tensor(_DMG_PARA_SPEED), opp_spe.new_tensor(1.0))
        opp_cur_hp = ctx.hp_and_active[ar, opp_act, 0] * opp_maxhp    # [B]
        t1d = ctx.type1_ids[ar, opp_act]; t2d = ctx.type2_ids[ar, opp_act]   # [B]
        opp_ability = ctx.ability1_ids[ar, opp_act]                  # [B] (0 if unrevealed)

        # --- type effectiveness eff[B,6,4] = CHART[t1d]·CHART[t2d]·ability_mult (single defender, gathered) ---
        eff = (self.CHART[t1d][:, None, None, :].expand(B, TEAM_SIZE, _DMG_OAX_N_MOVES, self.CHART.shape[-1])
               .gather(3, move_ty[..., None]).squeeze(-1))            # [B,6,4]
        eff = eff * (self.CHART[t2d][:, None, None, :].expand(B, TEAM_SIZE, _DMG_OAX_N_MOVES, self.CHART.shape[-1])
                     .gather(3, move_ty[..., None]).squeeze(-1))
        eff = eff * (self.ABILITY_DAMAGE_MULT[opp_ability][:, None, None, :]
                     .expand(B, TEAM_SIZE, _DMG_OAX_N_MOVES, self.ABILITY_DAMAGE_MULT.shape[-1])
                     .gather(3, move_ty[..., None]).squeeze(-1))      # [B,6,4] defender immunity

        # --- gen3 damage per (attacker, move) → [B,6,4] (the _outgoing_block physics, single opp defender) ---
        D = phys * opp_def[:, None, None] + (1.0 - phys) * opp_spd[:, None, None]       # [B,6,4]
        is_stab = ((move_ty == at1[:, :, None]) | (move_ty == at2[:, :, None])).float() # [B,6,4]
        stab = 1.0 + 0.5 * is_stab
        core = 42.0 * bp * A / (D + eps) / 50.0 + 2.0                                   # [B,6,4]
        # gen3 weather BP modifier (== _weather_mult): sun/rain are [B,1] → broadcast as [B,1,1] over [B,6,4].
        is_water = (move_ty == _WATER_TIDX).float(); is_fire = (move_ty == _FIRE_TIDX).float()   # [B,6,4]
        sun = ctx.weather_feature[:, 1:2, None]; rain = ctx.weather_feature[:, 2:3, None]        # [B,1,1]
        weather = 1.0 + rain * (0.5 * is_water - 0.5 * is_fire) + sun * (0.5 * is_fire - 0.5 * is_water)  # [B,6,4]
        dmg_ns = core * stab * eff * 0.925 * usable * weather                           # [B,6,4]
        opp_reflect = ctx.screen_feature[:, 1:2]; opp_ls = ctx.screen_feature[:, 3:4]   # OPP-side screens [B,1]
        screen = 1.0 - 0.5 * (opp_reflect[:, :, None] * phys + opp_ls[:, :, None] * (1.0 - phys))  # [B,6,4]
        high, low, crit, ko = self._rolls(dmg_ns, screen, opp_maxhp[:, None, None],
                                          opp_cur_hp[:, None, None], acc, eps)           # each [B,6,4]
        # fixed-damage moves (Seismic Toss into the opp active): immunity + legality gated, CB-invariant.
        fixed = self.MOVE_FIXED_DAMAGE[move_ids] * usable                              # [B,6,4]
        is_fixed = fixed > 0
        not_immune = (eff > 0).float()                                                 # [B,6,4]
        fixed_frac = (fixed / (opp_maxhp[:, None, None] + eps)) * not_immune
        fixed_ko = acc * (fixed >= opp_cur_hp[:, None, None]).float() * not_immune
        high = torch.where(is_fixed, fixed_frac, high); low = torch.where(is_fixed, fixed_frac, low)
        crit = torch.where(is_fixed, fixed_frac, crit); ko = torch.where(is_fixed, fixed_ko, ko)

        # --- p_outspeed per attacker (our_spe [B,6] vs the shared believed opp speed) ---
        opp_spe_std = self.SPECIES_SPREAD_PRIOR[ctx.species_ids[ar, opp_act], _SB_SPE, 1]   # [B]
        p_outspeed = self._p_outspeed(our_spe, opp_spe[:, None].expand(B, TEAM_SIZE),
                                      opp_spe_std[:, None].expand(B, TEAM_SIZE))             # [B,6]

        # --- assemble + gate ---
        per_move = torch.stack([low, high, crit, ko], dim=-1)                           # [B,6,4,4]
        alive = (ctx.hp_and_active[:, our, 0] > 0).float()                              # [B,6] attacker exists+alive
        per_move = per_move * (usable[:, :, :, None] * alive[:, :, None, None])         # gate legal × alive attacker
        p_outspeed = p_outspeed * alive
        out = torch.cat([per_move.reshape(B, TEAM_SIZE * _DMG_OAX_N_MOVES * _DMG_OAX_PER_MOVE),
                         p_outspeed, alive], dim=1)                                     # [B, _DMG_OAX]
        return out * has_opp[:, None]                                                   # zeroed when no opp active


    def _status_landing(self, ctx: 'ExtractorContext') -> torch.Tensor:
        """gen3_unified_status_landing_v1: per OUR move (REQUEST-slot order == action 6+k), P(a dedicated
        STATUS move applies to the opp active) + a `known` bit — the GPU home for the masked move-effect
        block's `status_will_land`. The status MOVES the outgoing DAMAGE block can't price (BP 0 → usable 0).

        P(lands) = is_status_move · accuracy · (1−type_immune) · (1−ability_block) · (1−already_block)
                   · (1−sleep_clause_block), gated to 0 with no opp active / our active dead. Where:
          • type_immune  — per-MOVE gen3 rule (Thunder Wave→Ground, Toxic/Poison→Steel/Poison, Will-O-Wisp
            →Fire, **Leech Seed→Grass**), max over the opp active's two types.
          • ability_block — REVEALED opp ability → exact `ABILITY_STATUS_BLOCK`; UNREVEALED → the species
            Smogon-prior marginal `SPECIES_STATUS_BLOCK_PRIOR` (Snorlax Toxic ≈0.14 Immunity-dominated).
          • already_block — the opp active already carries a major status (can't double-apply); NOT Leech Seed.
          • sleep_clause_block — a SLEEP move fails if ANY opp mon is already asleep via a NON-Rest source
            (`sleep_is_deterministic==0`). Rest self-sleep does NOT consume our cap (the user's rule). The
            per-mon Rest flag is the existing gen3_sleep_wake_belief_v1 `sleep_is_deterministic` (reused).
          • has_sub — the opp active behind a Substitute blocks EVERY status move (incl. Leech Seed); read
            from the public Substitute volatile in `ctx.opp_ctx_raw` at `_SUBSTITUTE_CTX_IDX`.
        `known` = the value rests on CERTAIN (public) info — a type/already-statused/Sleep-Clause/Substitute
        hard block OR a revealed ability — vs a Smogon-prior estimate. No move-belief gradient (OUR moves are
        certain). UNCOVERED residual: Yawn (delayed sleep, no status_inflicted), Leech-Seed-already-seeded."""
        B, device = ctx.batch_size, ctx.device
        ar = torch.arange(B, device=device)
        our_act = ctx.our_active_idx                                  # [B]
        opp_act = TEAM_SIZE + ctx.opp_active_local                    # [B] opp-active global slot
        has_opp = ctx.hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, -1].any(dim=1).float()  # [B]
        our_alive = (ctx.hp_and_active[ar, our_act, 0] > 0).float()   # [B]
        gate = (has_opp * our_alive)[:, None]                         # [B,1]

        # gen3_op_move_align_v1: request-ordered obs slice so p_land[k] ↔ action 6+k (was
        # all_move_ids[our_act], sorted-by-id → the output was positionally misaligned with the actions).
        move_ids = ctx.our_active_req_move_ids                        # [B,4] request order
        inflicts = self.MOVE_INFLICTS_STATUS[move_ids]               # [B,4]
        acc = self.MOVE_ACCURACY[move_ids]                          # [B,4] (Toxic .85, WoW .75, T-Wave 1, …)
        sidx = self.MOVE_STATUS_CAT[move_ids]                       # [B,4] long (0 = not a status move)
        is_sleep = self.MOVE_IS_SLEEP[move_ids]                     # [B,4]
        blocked_if_statused = self.MOVE_BLOCKED_IF_STATUSED[move_ids]  # [B,4] (0 for Leech Seed)

        # type immunity (per move) — max over the opp active's two types.
        t1 = ctx.type1_ids[ar, opp_act]                            # [B]
        t2 = ctx.type2_ids[ar, opp_act]
        ti = self.MOVE_STATUS_TYPE_IMMUNE[move_ids]                 # [B,4,N_TYPE_IDX]
        type_immune = torch.maximum(ti.gather(2, t1[:, None, None].expand(B, 4, 1)).squeeze(2),
                                    ti.gather(2, t2[:, None, None].expand(B, 4, 1)).squeeze(2))  # [B,4]

        # ability immunity — revealed → exact; unrevealed (id 0) → the species Smogon-prior marginal.
        opp_ability = ctx.ability1_ids[ar, opp_act]                # [B] (0 if unrevealed)
        opp_species = ctx.species_ids[ar, opp_act]                 # [B]
        abl_rev = self.ABILITY_STATUS_BLOCK[opp_ability].gather(1, sidx)           # [B,4]
        abl_prior = self.SPECIES_STATUS_BLOCK_PRIOR[opp_species].gather(1, sidx)   # [B,4]
        revealed = (opp_ability > 0).float()[:, None]              # [B,1]
        ability_block = revealed * abl_rev + (1.0 - revealed) * abl_prior          # [B,4]

        # already-statused (opp active) — any non-None status bit → blocks a MAJOR status (not Leech Seed).
        opp_cond = ctx.pokemon_part[ar, opp_act,
                                    POKEMON_CONDITION_OFFSET + 1:POKEMON_CONDITION_OFFSET + 7]  # [B,6]
        already_statused = (opp_cond.sum(dim=1) > 0.5).float()[:, None]            # [B,1]
        already_block = already_statused * blocked_if_statused                     # [B,4]

        # Sleep Clause — ANY opp mon asleep via a NON-Rest source consumes our one-sleep cap.
        opp_slp = ctx.pokemon_part[:, TEAM_SIZE:2 * TEAM_SIZE, POKEMON_CONDITION_OFFSET + _COND_SLP_IDX]  # [B,6]
        opp_rest = ctx.pokemon_part[:, TEAM_SIZE:2 * TEAM_SIZE, POKEMON_SLEEP_BELIEF_OFFSET]  # [B,6] is_rest
        nonrest_sleep = opp_slp * (1.0 - opp_rest)                                 # [B,6]
        sleep_clause = (nonrest_sleep.sum(dim=1) > 0.5).float()[:, None]           # [B,1]
        sleep_block = sleep_clause * is_sleep                                      # [B,4]

        # Substitute — the opp active behind a Sub blocks EVERY status move (incl. Leech Seed) in gen3. Read
        # the public Substitute volatile from the opp active context (boosts ++ volatiles). Applies to ALL
        # inflicting moves (not just majors), so it folds in as a flat per-channel factor below.
        has_sub = (ctx.opp_ctx_raw[:, _SUBSTITUTE_CTX_IDX] > 0.5).float()[:, None]  # [B,1]

        p_land = (inflicts * acc * (1.0 - type_immune) * (1.0 - ability_block)
                  * (1.0 - already_block) * (1.0 - sleep_block) * (1.0 - has_sub))  # [B,4]
        # `known` = the value rests on CERTAIN info (a hard block — type/already-statused/Sleep-Clause/
        # Substitute, all PUBLIC — or a revealed ability) vs a Smogon-prior estimate.
        certain = torch.clamp(type_immune + already_block + sleep_block + has_sub + revealed, max=1.0)  # [B,4]
        known = inflicts * certain                                                 # [B,4]
        return torch.cat([p_land, known], dim=1) * gate                            # [B, _DMG_STATUS]

    def _incoming_status_lands(self, ctx: 'ExtractorContext', topk_idx: torch.Tensor,
                               high_topk: torch.Tensor) -> torch.Tensor:
        """gen3_unified_topk_incoming_v1: per (OUR defender d, top-K move k), P(move k applies a status to
        defender d) — the immunity-folded per-pivot safe-switch read (Thunder Wave → a Ground pivot = 0).
        Combines two mutually-exclusive paths, taking the max:
          • DEDICATED status move (Thunder Wave/Toxic/Will-O-Wisp/Spore/Leech Seed, BP 0): `inflicts · acc ·
            (1−type_immune@our_def_types) · (1−ability_block@our_def_ability) · (1−already)` — the
            per-MOVE type immunity (Thunder Wave→Ground, Toxic→Steel/Poison, WoW→Fire, Leech Seed→Grass)
            evaluated at OUR DEFENDER's types (the incoming mirror of `_status_landing`'s opp lookup).
          • DAMAGING-move MAJOR-status SECONDARY (Body Slam para, Ice Beam frz): `max_col(chance_col ·
            (1−ability_block[cat(col)])) · acc · Serene-Grace(opp) · 1[damage lands on this pivot] ·
            (1−already)` — gated by `high_topk>0`, so a pivot immune to the DAMAGE (Ghost vs Body Slam)
            shows 0 status risk too. gen3 has no type-based para/freeze immunity beyond that gate.
        All inputs are buffers + OUR-side public obs (types/ability/condition known) + the opp's revealed
        Serene Grace → w-INDEPENDENT (the belief gradient rides `w_topk`, not this). HP candidates carry no
        status (extended with zeros). v2 residual: incoming Sleep-Clause / our-Substitute (the owner's named
        case is type immunity)."""
        B, device, eps = ctx.batch_size, ctx.device, 1e-6
        K = topk_idx.shape[1]
        ar = torch.arange(B, device=device)
        opp_act = TEAM_SIZE + ctx.opp_active_local
        n_type = self.MOVE_STATUS_TYPE_IMMUNE.shape[1]
        # --- candidate-axis (C = n_moves; the typed HP nums 355-370 carry no status/secondary — all-zero
        # in these buffers, verified) move-status attributes → gather top-K (gen3_opp_hp_typed_candidates_v1) ---
        inflicts = self.MOVE_INFLICTS_STATUS[topk_idx]                                        # [B,K]
        acc = self.MOVE_ACCURACY[topk_idx]                                                    # [B,K]
        sidx = self.MOVE_STATUS_CAT[topk_idx]                                                 # [B,K]
        blocked = self.MOVE_BLOCKED_IF_STATUSED[topk_idx]                                     # [B,K]
        ti = self.MOVE_STATUS_TYPE_IMMUNE[topk_idx]                                           # [B,K,n_type]
        sec = self.MOVE_SECONDARY[topk_idx]                                                   # [B,K,10]

        # --- our 6 defenders' (known) types / ability / already-statused ---
        t1d = ctx.type1_ids[:, :TEAM_SIZE]                                                    # [B,6]
        t2d = ctx.type2_ids[:, :TEAM_SIZE]
        abl = self.ABILITY_STATUS_BLOCK[ctx.ability1_ids[:, :TEAM_SIZE]]                       # [B,6,7]
        our_cond = ctx.pokemon_part[:, :TEAM_SIZE,
                                    POKEMON_CONDITION_OFFSET + 1:POKEMON_CONDITION_OFFSET + 7]  # [B,6,6]
        already = (our_cond.sum(-1) > 0.5).float()                                            # [B,6]

        # --- DEDICATED status move landing: type immunity @ our defender types (max over the 2 types) ---
        ti_dk = ti[:, None, :, :].expand(B, TEAM_SIZE, K, n_type)                              # [B,6,K,n_type]
        ti1 = torch.gather(ti_dk, 3, t1d[:, :, None, None].expand(B, TEAM_SIZE, K, 1)).squeeze(-1)
        ti2 = torch.gather(ti_dk, 3, t2d[:, :, None, None].expand(B, TEAM_SIZE, K, 1)).squeeze(-1)
        t_imm = torch.maximum(ti1, ti2)                                                       # [B,6,K]
        abl_block = torch.gather(abl, 2, sidx[:, None, :].expand(B, TEAM_SIZE, K))             # [B,6,K]
        already_block = already[:, :, None] * blocked[:, None, :]                             # [B,6,K]
        dedicated = (inflicts[:, None, :] * acc[:, None, :] * (1.0 - t_imm)
                     * (1.0 - abl_block) * (1.0 - already_block))                             # [B,6,K]

        # --- DAMAGING-move MAJOR-status SECONDARY (gated by the damage actually landing on this pivot) ---
        opp_serene = self.ABILITY_SECONDARY_MULT[ctx.ability1_ids[ar, opp_act]]               # [B]
        sec_major = sec[..., :_SECONDARY_MAJOR_N]                                             # [B,K,6]
        abl_per_col = abl[..., self._SEC_CAT_IDX]                                             # [B,6,6] per status cat
        sec_land = (sec_major[:, None, :, :] * (1.0 - abl_per_col)[:, :, None, :]).amax(dim=-1)  # [B,6,K]
        damage_gate = (high_topk > eps).float()                                               # [B,6,K]
        secondary = (sec_land * acc[:, None, :] * opp_serene[:, None, None]
                     * damage_gate * (1.0 - already[:, :, None])).clamp(max=1.0)              # [B,6,K]
        return torch.maximum(dedicated, secondary)                                            # [B,6,K]

    def _incoming_matrix(self, ctx: 'ExtractorContext', w_all: torch.Tensor, low_frac: torch.Tensor,
                         high_frac: torch.Tensor, crit_frac: torch.Tensor, ko_ramp: torch.Tensor,
                         acc_all: torch.Tensor, phys_all: torch.Tensor, move_latent_all: torch.Tensor,
                         has_opp: torch.Tensor, defender_alive: torch.Tensor,
                         matrix_k: int, cand_nums: Optional[torch.Tensor] = None) -> torch.Tensor:
        """gen3_per_move_matrices_v1: the INCOMING per-move DAMAGE MATRIX — the ENRICHED top-K block (replaces
        it). For the opp active's top-`matrix_k` most-believed candidates (selection DETACHED): a per-move
        HEADER [latent(32), belief w (→ sharpens the belief), accuracy, is_phys, EXPLICIT effect bits (6,
        from MOVE_EFFECT_FLAGS — recovery/status/phaze/boost/hazard/protect, per move, un-collapsed), EXPLICIT
        secondary chances (10, from MOVE_SECONDARY)] and a per-(OUR mon, move) CELL [low, high, crit, pko,
        type_mult, status_lands]. Damage rolls GATHER from the SAME validated `_damage_rolls` tensors as the
        worst-case block (so an immune pivot reads 0); type_mult is the effectiveness at OUR defender's types;
        status_lands reuses `_incoming_status_lands`. Decorrelated (belief gradient rides `w`, latent rides the
        gather). Meaningful-K gate (zero the 5th+ slot once all 4 opp moves are revealed). HP candidates carry
        zero effect/secondary (extended with zeros). Output `[B, _dmg_imx_dim(matrix_k)]`."""
        B, device, eps = ctx.batch_size, ctx.device, 1e-6
        K = matrix_k
        ar = torch.arange(B, device=device)
        topk_idx = w_all.detach().topk(K, dim=-1).indices                          # [B,K] (detached selection)
        w_topk = w_all.gather(-1, topk_idx)                                        # [B,K] → belief gradient
        # gen3_topk_candidates_v1: `topk_idx` indexes the (possibly TRUNCATED) candidate axis, so any
        # gather into the FULL move space — the latent table, the effect/secondary buffers, the type
        # ids, the status-landing physics and the prober's stash — must go through `cand_nums` to get
        # the REAL move-num. None ⇒ no truncation ⇒ the reduced index IS the move-num.
        real_idx = topk_idx if cand_nums is None else cand_nums.gather(-1, topk_idx)   # [B,K] move-nums
        self.stash.topk_idx = real_idx.detach()                                     # prober: exact move names
        # The CANDIDATE-AXIS index (not the move num) — what a consumer must gather with to line a
        # [B,J,C,F] cells tensor up with alpha's K seats. `last_topk_idx` holds move NUMS, which
        # name the seats but cannot index the candidate axis; using it as an index would silently
        # read the wrong columns.
        self.stash.topk_cand_idx = topk_idx.detach()
        self.stash.topk_w = w_topk.detach()
        # --- per-move header: latent (→ MoveLatentEncoder grad) + belief + accuracy + is_phys + effect + secondary ---
        latent_topk = move_latent_all[real_idx]                                    # [B,K,32] differentiable
        acc_topk = acc_all.gather(-1, topk_idx)                                    # [B,K]
        phys_topk = phys_all.gather(-1, topk_idx)                                  # [B,K]
        # HP at the typed nums 355-370 carries no effect/secondary (all-zero in these buffers, verified);
        # C = n_moves (gen3_opp_hp_typed_candidates_v1 — the typed HP are ordinary move-num candidates).
        eff_flags = self.MOVE_EFFECT_FLAGS[real_idx]                               # [B,K,6]
        sec = self.MOVE_SECONDARY[real_idx]                                        # [B,K,10]
        # --- per-(defender, move) cell: gather the RAW physics rolls (w-INDEPENDENT) + type_mult + status ---
        idxd = topk_idx[:, None, :].expand(B, TEAM_SIZE, K)                        # [B,6,K]
        low_topk = low_frac.gather(-1, idxd)                                       # [B,6,K]
        high_topk = high_frac.gather(-1, idxd)
        crit_topk = crit_frac.gather(-1, idxd)
        pko_topk = ko_ramp.gather(-1, idxd)
        # type_mult @ OUR defenders' types/ability for the top-K move types (the immune/resist pivot read)
        mty_topk = self.MOVE_TYPE_IDX[real_idx]                                    # [B,K]
        idx2 = mty_topk[:, None, :].expand(B, TEAM_SIZE, K)                         # [B,6,K]
        t1d = ctx.type1_ids[:, :TEAM_SIZE]; t2d = ctx.type2_ids[:, :TEAM_SIZE]
        amul = self.ABILITY_DAMAGE_MULT[ctx.ability1_ids[:, :TEAM_SIZE]]            # [B,6,T]
        type_mult = (torch.gather(self.CHART[t1d], 2, idx2) * torch.gather(self.CHART[t2d], 2, idx2)
                     * torch.gather(amul, 2, idx2))                                 # [B,6,K]
        status_topk = self._incoming_status_lands(ctx, real_idx, high_topk)        # [B,6,K]
        # --- meaningful-K gate (== _topk_block): once all 4 opp moves revealed, the 5th+ slot is closed ---
        opp_act = TEAM_SIZE + ctx.opp_active_local
        n_revealed = (ctx.all_move_ids[ar, opp_act] > 0).sum(-1)                   # [B]
        slot_live = ((torch.arange(K, device=device)[None, :] < 4)
                     | (n_revealed[:, None] < 4)).float()                          # [B,K]
        header = torch.cat([latent_topk, w_topk[..., None], acc_topk[..., None],
                            phys_topk[..., None], eff_flags, sec], dim=-1)          # [B,K,_DMG_IMX_HEADER]
        header = header * (has_opp[:, None, None] * slot_live[:, :, None])
        cell = torch.stack([low_topk, high_topk, crit_topk, pko_topk, type_mult, status_topk], dim=-1)  # [B,6,K,6]
        cell = cell * (has_opp[:, None, None, None] * defender_alive[:, :, None, None]
                       * slot_live[:, None, :, None])
        return torch.cat([header.reshape(B, K * _DMG_IMX_HEADER),
                          cell.reshape(B, TEAM_SIZE * K * _DMG_IMX_CELL)], dim=1)

    def refine_candidates(self, ctx: 'ExtractorContext',
                          move_belief_logits: torch.Tensor,
                          k: Optional[int] = None,
                          w_all: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """The SHARED candidate selection over the move belief → `(topk_idx, w_topk)`.

        `discrete_incoming` and `discrete_incoming_status` are called with the
        SAME `move_belief_logits` object, so each was independently rebuilding an identical `[B, n_moves]`
        sigmoid + typed-HP scatter and an identical top-K — 4 redundant candidate builds and 2 redundant
        top-Ks per forward in the production config. Hoisting it here lets the caller compute once and
        pass the result to both (they still fall back to computing it when called standalone).

        `k` overrides the default `_DMG_REFINE_K` — the E4 entity-seat builder
        (`gen3_entity_move_seats_v1`) reuses this selection at its own `entity_topk_seats` K, so the
        seats and the refine kernels share ONE candidate definition (the index selection stays
        DETACHED; the gathered weights stay differentiable so the belief gradient rides them).

        `w_all` (gen3_op_candidate_dedup_v1): the caller may pass the op forward's own
        `last_w_all` — the IDENTICAL computation on the identical inputs — so the [B, n_moves]
        build runs once per forward instead of twice. The top-K itself stays here (it is cheap,
        and `k` may differ from the matrix's K). None ⇒ compute standalone, byte-identical to
        the pre-dedup path."""
        if w_all is None:
            w_all = self._opp_candidate_weights(ctx, move_belief_logits)                 # [B, n_moves]
        K = min(_DMG_REFINE_K if k is None else int(k), w_all.shape[1])
        topk_idx = w_all.detach().topk(K, dim=-1).indices                                # [B,K] (DETACHED)
        return topk_idx, w_all.gather(-1, topk_idx)                                      # → belief gradient

    def _incoming_rolls(self, ctx: 'ExtractorContext',
                        move_belief_logits: torch.Tensor,
                        cand: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                        spread_belief: Optional[torch.Tensor] = None
                        ) -> Tuple[torch.Tensor, ...]:
        """The SHARED lean incoming physics (gen3_iterative_damage_v1 / gen3_edge_bias_trunk_v1): the opp
        active's top-K believed candidate moves vs our 6 defenders, PRE-collapse. Factored out of
        `discrete_incoming` verbatim so its refine consumer and the D3 edge-bias consumer
        (`pairwise_incoming`) price the SAME physics from the SAME candidate selection — one body, no
        drift. v1 semantics unchanged: LEGACY de-timid attacker offense (no spread belief / boost / burn /
        weather / fixed-damage — the coarse signal; the full post-transformer op is authoritative).

        → `(high [B,6,K], ko [B,6,K], eff [B,6,K], phys_k [B,K], w_topk [B,K],
            defender_alive [B,6], has_opp [B])`."""
        B, device, eps = ctx.batch_size, ctx.device, 1e-6
        ar = torch.arange(B, device=device)
        opp_act = TEAM_SIZE + ctx.opp_active_local                                        # [B] global opp-active
        has_opp = ctx.hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, -1].any(dim=1).float()    # [B]
        # --- Attacker = opp active (legacy de-timid offense; the coarse refinement signal) ---
        a_base = self.BASE_STATS[ctx.species_ids[ar, opp_act]]                            # [B,6]
        off_const = 31.0 + 252.0 / 4.0 + 5.0
        if spread_belief is not None:
            # gen3_op_lean_forward_v1 (`believed_lean`): the BELIEVED attacker offense — the same
            # SpreadBelief read the full op prices with, replacing the legacy de-timid fiction at
            # the last site the edges consume (the B-spread correctness fix, applied to d3).
            sb = spread_belief[ar, ctx.opp_active_local]                                  # [B,5]
            atk = sb[:, _SB_ATK]                                                          # [B]
            spa = sb[:, _SB_SPA]                                                          # [B]
        else:
            atk = (2.0 * a_base[:, 1] + off_const) * 1.1                                  # [B]
            spa = (2.0 * a_base[:, 3] + off_const) * 1.1                                  # [B]
        at1 = ctx.type1_ids[ar, opp_act]                                                  # [B]
        at2 = ctx.type2_ids[ar, opp_act]
        # --- Defenders = our 6 (REAL spread reconstructed from the obs, like forward) ---
        d_base = self.BASE_STATS[ctx.species_ids[:, :TEAM_SIZE]]                          # [B,6,6]
        spread = ctx.pokemon_part[:, :TEAM_SIZE,
                                  POKEMON_SPREAD_OFFSET:POKEMON_SPREAD_OFFSET + POKEMON_SPREAD_DIM]
        iv = spread[..., 0:6] * 31.0
        ev = spread[..., 6:12] * 252.0
        nat = spread[..., 13:18]                                                          # [B,6,5]
        def_stat = (2.0 * d_base[..., 2] + iv[..., 2] + ev[..., 2] / 4.0 + 5.0) * nat[..., 1]   # [B,6]
        spd_stat = (2.0 * d_base[..., 4] + iv[..., 4] + ev[..., 4] / 4.0 + 5.0) * nat[..., 3]   # [B,6]
        maxhp = 2.0 * d_base[..., 0] + iv[..., 0] + ev[..., 0] / 4.0 + 110.0              # [B,6]
        hp_frac = ctx.hp_and_active[:, :TEAM_SIZE, 0]                                     # [B,6]
        cur_hp = hp_frac * maxhp
        defender_alive = (hp_frac > 0).float()                                           # [B,6]
        t1d = ctx.type1_ids[:, :TEAM_SIZE]                                               # [B,6]
        t2d = ctx.type2_ids[:, :TEAM_SIZE]
        ability = ctx.ability1_ids[:, :TEAM_SIZE]                                        # [B,6]
        # --- Belief at the opp active → w [B, n_moves] (same source as forward; the lean refine passes no
        # learned posterior → the prior FLOOR resolves the typed-HP belief, scattered onto 355-370; the bare
        # 237 is masked — gen3_opp_hp_typed_candidates_v1) ---
        # --- Candidate axis attributes: C = n_moves (the typed HP 355-370 carry real BP/type; no append) ---
        bp_all = self.MOVE_BP                                                            # [n_moves]
        mty_all = self.MOVE_TYPE_IDX                                                     # [n_moves]
        phys_all = self.MOVE_PHYS                                                        # [n_moves]
        acc_all = self.MOVE_ACCURACY                                                     # [n_moves]
        # --- SELECT the top-K most-believed candidates (selection DETACHED; gathered values differentiable).
        # Reused from the caller when the sibling status kernel already built it (`refine_candidates`). ---
        topk_idx, w_topk = cand if cand is not None else self.refine_candidates(ctx, move_belief_logits)
        K = topk_idx.shape[1]
        bp_k = bp_all[topk_idx]                                                          # [B,K]
        mty_k = mty_all[topk_idx]                                                        # [B,K] (long, TypeEncoder)
        phys_k = phys_all[topk_idx]                                                      # [B,K]
        acc_k = acc_all[topk_idx]                                                        # [B,K]
        # --- gen3 damage for the K candidates × 6 defenders → [B,6,K] (the lean per-K mirror of _damage_rolls) ---
        idxd = mty_k[:, None, :].expand(B, TEAM_SIZE, K)                                  # [B,6,K] type indices
        eff = torch.gather(self.CHART[t1d], 2, idxd) * torch.gather(self.CHART[t2d], 2, idxd)  # [B,6,K]
        amul = self.ABILITY_DAMAGE_MULT[ability]                                          # [B,6,T] defender ability
        eff = eff * torch.gather(amul, 2, idxd)                                           # [B,6,K] fold immunity
        A = phys_k * atk[:, None] + (1.0 - phys_k) * spa[:, None]                         # [B,K]
        D = (phys_k[:, None, :] * def_stat[:, :, None]
             + (1.0 - phys_k)[:, None, :] * spd_stat[:, :, None])                         # [B,6,K]
        is_stab = ((mty_k == at1[:, None]) | (mty_k == at2[:, None])).float()             # [B,K]
        stab = 1.0 + 0.5 * is_stab                                                        # [B,K]
        core = 42.0 * bp_k[:, None, :] * A[:, None, :] / (D + eps) / 50.0 + 2.0           # [B,6,K]
        dmg_ns = core * stab[:, None, :] * eff * 0.925                                    # [B,6,K] pre-screen
        dmg_ns = dmg_ns * (bp_k > 0).float()[:, None, :]                                  # kill the +2 floor on BP-0
        reflect, light_screen = ctx.screen_feature[:, 0:1], ctx.screen_feature[:, 2:3]    # [B,1] OUR-side screens
        screen = 1.0 - 0.5 * (reflect * phys_k + light_screen * (1.0 - phys_k))           # [B,K]
        high, _low, _crit, ko = self._rolls(dmg_ns, screen[:, None, :], maxhp[:, :, None],
                                            cur_hp[:, :, None], acc_k[:, None, :], eps)     # each [B,6,K]
        return high, ko, eff, phys_k, w_topk, defender_alive, has_opp


    def discrete_incoming_status(self, ctx: 'ExtractorContext',
                                 move_belief_logits: torch.Tensor,
                                 cand: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
                                 per_pair: bool = False) -> torch.Tensor:
        """gen3_status_trunk_v1 (INCOMING): per OUR mon, the belief-weighted `[P(major status lands),
        P(immobilizing status lands = para/frz/slp)]` from the opp active's top-`_DMG_REFINE_K` believed
        DEDICATED status moves (Thunder Wave / Toxic / Will-O-Wisp / Spore / Leech Seed). The "will I get
        statused" anticipation signal — injected onto OUR-mon tokens (the incoming mirror of the damage
        refine). Reuses the `_incoming_status_lands` DEDICATED-move immunity physics (type @ OUR def types,
        ability block, already-statused); the damaging-move secondary-para path stays at the heads. The
        major-vs-immobilize split is the decorrelation that matters for a SWITCH (a Ground pivot reads 0
        T-Wave immobilize even if it eats Toxic). Belief-weighted hard-max over K → the per-round gradient
        rides `w_topk` and sharpens the move belief toward status threats. `[B, TEAM_SIZE, _DMG_STATUS_REFINE]`."""
        B, device = ctx.batch_size, ctx.device
        ar = torch.arange(B, device=device)
        has_opp = ctx.hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, -1].any(dim=1).float()   # [B]
        n_type = self.MOVE_STATUS_TYPE_IMMUNE.shape[1]
        # candidate selection — the SAME detached top-K over the move belief as discrete_incoming. C =
        # n_moves: the typed HP nums 355-370 are ordinary candidates (the belief scattered onto them, bare
        # 237 masked) and carry NO status (all-zero in these buffers, verified) — gen3_opp_hp_typed_candidates_v1.
        # Shared with `discrete_incoming` for the same round when the caller passes it (`refine_candidates`).
        topk_idx, w_topk = cand if cand is not None else self.refine_candidates(ctx, move_belief_logits)
        K = topk_idx.shape[1]
        inflicts = self.MOVE_INFLICTS_STATUS[topk_idx]                                    # [B,K]
        acc = self.MOVE_ACCURACY[topk_idx]                                                # [B,K]
        sidx = self.MOVE_STATUS_CAT[topk_idx]                                             # [B,K]
        blocked = self.MOVE_BLOCKED_IF_STATUSED[topk_idx]                                 # [B,K]
        ti = self.MOVE_STATUS_TYPE_IMMUNE[topk_idx]                                       # [B,K,n_type]
        # our 6 defenders' KNOWN types / ability-block / already-statused / alive
        t1d = ctx.type1_ids[:, :TEAM_SIZE]
        t2d = ctx.type2_ids[:, :TEAM_SIZE]
        abl = self.ABILITY_STATUS_BLOCK[ctx.ability1_ids[:, :TEAM_SIZE]]                   # [B,6,N_STATUS_CAT]
        our_cond = ctx.pokemon_part[:, :TEAM_SIZE,
                                    POKEMON_CONDITION_OFFSET + 1:POKEMON_CONDITION_OFFSET + 7]
        already = (our_cond.sum(-1) > 0.5).float()                                        # [B,6]
        defender_alive = (ctx.hp_and_active[:, :TEAM_SIZE, 0] > 0).float()                # [B,6]
        ti_dk = ti[:, None, :, :].expand(B, TEAM_SIZE, K, n_type)
        ti1 = torch.gather(ti_dk, 3, t1d[:, :, None, None].expand(B, TEAM_SIZE, K, 1)).squeeze(-1)
        ti2 = torch.gather(ti_dk, 3, t2d[:, :, None, None].expand(B, TEAM_SIZE, K, 1)).squeeze(-1)
        t_imm = torch.maximum(ti1, ti2)                                                   # [B,6,K]
        abl_block = torch.gather(abl, 2, sidx[:, None, :].expand(B, TEAM_SIZE, K))         # [B,6,K]
        already_block = already[:, :, None] * blocked[:, None, :]                         # [B,6,K]
        land = (inflicts[:, None, :] * acc[:, None, :] * (1.0 - t_imm)
                * (1.0 - abl_block) * (1.0 - already_block))                              # [B,6,K]
        is_immob: torch.Tensor = sum(  # type: ignore[union-attr]
            (sidx == c) for c in _IMMOBILIZE_STATUS_CATS).float().clamp(max=1.0)              # [B,K]
        if per_pair:
            # gen3_edge_bias_trunk_v1 (S3): the UN-collapsed per-(candidate, defender) status cells for
            # the edge bias — [B, K, 6, 3] = [land, land·is_immob, w] per (their believed status move c,
            # our mon i). Same physics, same candidate selection; the collapse is simply not taken.
            # Decorrelated: land is w-independent, w rides as its own channel (the belief gradient path).
            cells = torch.stack([
                land, land * is_immob[:, None, :],
                w_topk[:, None, :].expand_as(land),
            ], dim=-1)                                                                    # [B,6,K,3]
            cells = cells * defender_alive[:, :, None, None] * has_opp[:, None, None, None]
            return cells.permute(0, 2, 1, 3).contiguous()                                 # [B,K,6,3]
        w_b = w_topk[:, None, :]
        p_major = (w_b * land).amax(dim=-1)                                               # [B,6]
        p_immob = (w_b * land * is_immob[:, None, :]).amax(dim=-1)                         # [B,6]
        feats = torch.stack([p_major, p_immob], dim=-1)                                   # [B,6,_DMG_STATUS_REFINE]
        return feats * defender_alive[:, :, None] * has_opp[:, None, None]

    def discrete_outgoing_status(self, ctx: 'ExtractorContext', per_pair: bool = False) -> torch.Tensor:
        """gen3_status_trunk_v1 (OUTGOING): per OPP mon (REVEALED-gated), the `[P(major status from OUR
        active's status moves lands), P(immobilizing status lands)]` — the in-trunk home for the masked
        move-effect block's `status_will_land`, extended over the opp's 6 mons (the active is ALWAYS
        revealed = the deprecation requirement; revealed bench = bonus; unrevealed zeroed in v1). Reuses the
        `_status_landing` immunity physics (type @ opp types, ability revealed-exact else species prior,
        already-statused, Sleep-Clause, Substitute @ the active slot) per OUR move, reduced by category over
        our 4 moves. OUR moves are KNOWN → no belief gradient. `[B, TEAM_SIZE, _DMG_STATUS_REFINE]`."""
        B, device = ctx.batch_size, ctx.device
        ar = torch.arange(B, device=device)
        our_act = ctx.our_active_idx
        has_opp = ctx.hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, -1].any(dim=1).float()
        our_alive = (ctx.hp_and_active[ar, our_act, 0] > 0).float()
        gate = has_opp * our_alive                                                        # [B]
        n_type = self.MOVE_STATUS_TYPE_IMMUNE.shape[1]
        # our 4 status moves (gen3_op_move_align_v1: the request-ordered obs slice, NOT the sorted-by-id
        # all_move_ids[our_act] — consistent with every other our-move op read). Output max-pools over the
        # 4 moves so the ORDER is invariant here; no legality gate (parity with _status_landing + the CPU
        # move-effect block, which both KEEP disabled moves — legality is the action mask's job).
        move_ids = ctx.our_active_req_move_ids                                             # [B,4] request order
        inflicts = self.MOVE_INFLICTS_STATUS[move_ids]                                     # [B,4]
        acc = self.MOVE_ACCURACY[move_ids]                                                # [B,4]
        sidx = self.MOVE_STATUS_CAT[move_ids]                                             # [B,4]
        is_sleep = self.MOVE_IS_SLEEP[move_ids]                                           # [B,4]
        blocked = self.MOVE_BLOCKED_IF_STATUSED[move_ids]                                 # [B,4]
        ti = self.MOVE_STATUS_TYPE_IMMUNE[move_ids]                                       # [B,4,n_type]
        is_immob: torch.Tensor = sum(  # type: ignore[union-attr]
            (sidx == c) for c in _IMMOBILIZE_STATUS_CATS).float().clamp(max=1.0)              # [B,4]
        # opp 6 defenders
        opp_t1 = ctx.type1_ids[:, TEAM_SIZE:2 * TEAM_SIZE]
        opp_t2 = ctx.type2_ids[:, TEAM_SIZE:2 * TEAM_SIZE]
        opp_ability = ctx.ability1_ids[:, TEAM_SIZE:2 * TEAM_SIZE]                         # [B,6]
        opp_species = ctx.species_ids[:, TEAM_SIZE:2 * TEAM_SIZE]                          # [B,6]
        revealed_slot: torch.Tensor = (1.0 - ctx.opp_believed_mask.float())               # [B,6] 1 = revealed
        defender_alive = (ctx.hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, 0] > 0).float()    # [B,6]
        ti_dm = ti[:, None, :, :].expand(B, TEAM_SIZE, 4, n_type)                          # [B,6,4,n_type]
        timm1 = torch.gather(ti_dm, 3, opp_t1[:, :, None, None].expand(B, TEAM_SIZE, 4, 1)).squeeze(-1)
        timm2 = torch.gather(ti_dm, 3, opp_t2[:, :, None, None].expand(B, TEAM_SIZE, 4, 1)).squeeze(-1)
        t_imm = torch.maximum(timm1, timm2)                                               # [B,6,4]
        ab_rev = torch.gather(self.ABILITY_STATUS_BLOCK[opp_ability], 2,
                              sidx[:, None, :].expand(B, TEAM_SIZE, 4))                    # [B,6,4]
        ab_pri = torch.gather(self.SPECIES_STATUS_BLOCK_PRIOR[opp_species], 2,
                              sidx[:, None, :].expand(B, TEAM_SIZE, 4))                    # [B,6,4]
        is_rev = (opp_ability > 0).float()[:, :, None]                                     # [B,6,1]
        ability_block = is_rev * ab_rev + (1.0 - is_rev) * ab_pri                          # [B,6,4]
        opp_cond = ctx.pokemon_part[:, TEAM_SIZE:2 * TEAM_SIZE,
                                    POKEMON_CONDITION_OFFSET + 1:POKEMON_CONDITION_OFFSET + 7]
        already = (opp_cond.sum(-1) > 0.5).float()                                        # [B,6]
        already_block = already[:, :, None] * blocked[:, None, :]                         # [B,6,4]
        # Sleep-Clause (global): any opp asleep via a non-Rest source → our sleep moves fail
        opp_slp = ctx.pokemon_part[:, TEAM_SIZE:2 * TEAM_SIZE, POKEMON_CONDITION_OFFSET + _COND_SLP_IDX]
        opp_rest = ctx.pokemon_part[:, TEAM_SIZE:2 * TEAM_SIZE, POKEMON_SLEEP_BELIEF_OFFSET]
        sleep_clause = ((opp_slp * (1.0 - opp_rest)).sum(-1) > 0.5).float()[:, None, None]  # [B,1,1]
        sleep_block = sleep_clause * is_sleep[:, None, :]                                  # [B,6,4]
        # Substitute — only the opp ACTIVE slot can hold a Sub (blocks every status move)
        has_sub = (ctx.opp_ctx_raw[:, _SUBSTITUTE_CTX_IDX] > 0.5).float()                  # [B]
        is_active = torch.zeros(B, TEAM_SIZE, device=device)
        is_active[ar, ctx.opp_active_local] = 1.0
        sub_block = (has_sub[:, None] * is_active)[:, :, None]                             # [B,6,1]
        land = (inflicts[:, None, :] * acc[:, None, :] * (1.0 - t_imm) * (1.0 - ability_block)
                * (1.0 - already_block) * (1.0 - sleep_block) * (1.0 - sub_block))         # [B,6,4]
        if per_pair:
            # gen3_edge_bias_trunk_v1 (S1): the UN-collapsed per-(our move, opp mon) status cells for
            # the edge bias — [B, 4, 6, 2] = [land, land·is_immob] per (our status move k in REQUEST
            # order == E3 seat k, opp mon d). Same physics + gates; the max over moves is not taken.
            cells = torch.stack([land, land * is_immob[:, None, :]], dim=-1)               # [B,6,4,2]
            cells = (cells * revealed_slot[:, :, None, None] * defender_alive[:, :, None, None]
                     * gate[:, None, None, None])
            return cells.permute(0, 2, 1, 3).contiguous()                                  # [B,4,6,2]
        p_major = land.amax(dim=-1)                                                        # [B,6]
        p_immob = (land * is_immob[:, None, :]).amax(dim=-1)                               # [B,6]
        feats = torch.stack([p_major, p_immob], dim=-1)                                    # [B,6,_DMG_STATUS_REFINE]
        return feats * revealed_slot[:, :, None] * defender_alive[:, :, None] * gate[:, None, None]
