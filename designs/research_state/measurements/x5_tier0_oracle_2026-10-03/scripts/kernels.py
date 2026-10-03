"""The two damage-op kernels this measurement needs, VECTORISED over an entity axis (every species at once, or one
averaged entity), copied from the production op and checked against it by `parity.py`.

- `outgoing(...)`: OUR active's 4 request-order moves -> an UNREVEALED opponent defender, i.e. the hidden-defender
  columns of `DamageOperatorBlocks._outgoing_matrix` (damage_op_blocks.py:287-427): per-defender bulk from the
  Species-Clause/belief marginals (E[def], E[spd] via SPECIES_SPREAD_PRIOR, E[maxhp] via base HP, E[mult] via
  SPECIES_EXP_MULT), forced full HP, OPP-side screens, weather/sport BP, CB/boost/burn on our attacker, fixed-damage
  moves. A ONE-HOT distribution gives exactly what the op computes for that species on its hidden path; any other
  distribution gives the blob's averaged defender. P(KO) is returned UN-nulled; the caller nulls it where the op does.
- `incoming(...)`: an opponent BENCH attacker -> OUR active, i.e. the (i = our active, j) cell of
  `DamageOperatorPairwise.pairwise_bench_incoming` (D4, damage_op_pairwise.py:771-849): top-K candidates of the
  attacker's own composed move posterior, de-timid offense, our real-spread defender, OUR-side screens, the
  presence-scaled max over candidates (M2 = C on the move axis, exactly today's formula). The production D4 gates
  every UNREVEALED attacker off; this kernel prices one as X5 would (a hypothesis = a concrete revealed-path mon).

The only re-implementation choice that has no production counterpart is the AVERAGED ATTACKER that M3 (c) gives
OTHER (`averaged_attacker`); see its docstring (FINDING: the design note does not specify it).
"""
from __future__ import annotations

from typing import Dict

import torch

from agents.model.damage_op_layout import _COND_BRN_IDX, _SB_DEF, _SB_SPD
from agents.observation.constants import (POKEMON_CONDITION_OFFSET, POKEMON_SPREAD_DIM, POKEMON_SPREAD_OFFSET,
                                          TEAM_SIZE)

EPS = 1e-6


# ------------------------------------------------------------------------------------------------ our side
def our_attacker(op, ctx) -> Dict[str, torch.Tensor]:
    """The per-move attacker block of `_outgoing_matrix` (no boost_delta), [B,4] each."""
    B = ctx.batch_size
    ar = torch.arange(B)
    our_act = ctx.our_active_idx
    has_opp = ctx.hp_and_active[:, TEAM_SIZE:2 * TEAM_SIZE, -1].any(dim=1).float()
    our_alive = (ctx.hp_and_active[ar, our_act, 0] > 0).float()
    move_ids = ctx.our_active_req_move_ids
    move_ty = ctx.our_active_req_move_type_ids
    legal = ctx.our_active_req_move_legal
    is_hp = (move_ids == op.hp_num)
    bp = torch.where(is_hp, torch.full_like(move_ty, op.hp_bp, dtype=torch.float32), op.MOVE_BP[move_ids])
    phys = op.TYPE_IS_PHYS[move_ty]
    acc = op.MOVE_ACCURACY[move_ids]
    usable = legal * (bp > 0).float()
    a_base = op.BASE_STATS[ctx.species_ids[ar, our_act]]
    spr = ctx.pokemon_part[ar, our_act, POKEMON_SPREAD_OFFSET:POKEMON_SPREAD_OFFSET + POKEMON_SPREAD_DIM]
    iv = spr[:, 0:6] * 31.0; ev = spr[:, 6:12] * 252.0; nat = spr[:, 13:18]
    our_atk = (2.0 * a_base[:, 1] + iv[:, 1] + ev[:, 1] / 4.0 + 5.0) * nat[:, 0]
    our_spa = (2.0 * a_base[:, 3] + iv[:, 3] + ev[:, 3] / 4.0 + 5.0) * nat[:, 2]
    o_b_atk, _odf, o_b_spa, _osd, _ose = op._boost_stages(ctx.our_ctx_raw)
    our_burn = ctx.pokemon_part[ar, our_act, POKEMON_CONDITION_OFFSET + _COND_BRN_IDX]
    our_cb = (ctx.item_ids[ar, our_act] == op.cb_item_num).float()
    our_atk = our_atk * torch.where(our_cb > 0.5, our_atk.new_tensor(op.cb_phys_mult), our_atk.new_tensor(1.0))
    our_atk = our_atk * op._boost_mult(o_b_atk) * torch.where(our_burn > 0.5, our_atk.new_tensor(0.5),
                                                               our_atk.new_tensor(1.0))
    our_spa = our_spa * op._boost_mult(o_b_spa)
    at1 = ctx.type1_ids[ar, our_act]; at2 = ctx.type2_ids[ar, our_act]
    A = phys * our_atk[:, None] + (1.0 - phys) * our_spa[:, None]
    is_stab = ((move_ty == at1[:, None]) | (move_ty == at2[:, None])).float()
    weather = op._field_bp_mult(ctx, move_ty)
    opp_reflect = ctx.screen_feature[:, 1:2]; opp_ls = ctx.screen_feature[:, 3:4]
    screen = 1.0 - 0.5 * (opp_reflect * phys + opp_ls * (1.0 - phys))
    fixed = op.MOVE_FIXED_DAMAGE[move_ids] * usable
    return dict(move_ty=move_ty, bp=bp, phys=phys, acc=acc, usable=usable, A=A, stab=1.0 + 0.5 * is_stab,
                weather=weather, screen=screen, fixed=fixed, gate=has_opp * our_alive)


def our_defender(op, ctx) -> Dict[str, torch.Tensor]:
    """OUR active as D4's defender (real spread, NO stages: D4's bench recipe), [B] / [B,T]."""
    B = ctx.batch_size
    ar = torch.arange(B)
    i = ctx.our_active_idx
    d_base = op.BASE_STATS[ctx.species_ids[ar, i]]
    spr = ctx.pokemon_part[ar, i, POKEMON_SPREAD_OFFSET:POKEMON_SPREAD_OFFSET + POKEMON_SPREAD_DIM]
    iv = spr[:, 0:6] * 31.0; ev = spr[:, 6:12] * 252.0; nat = spr[:, 13:18]
    def_stat = (2.0 * d_base[:, 2] + iv[:, 2] + ev[:, 2] / 4.0 + 5.0) * nat[:, 1]
    spd_stat = (2.0 * d_base[:, 4] + iv[:, 4] + ev[:, 4] / 4.0 + 5.0) * nat[:, 3]
    maxhp = 2.0 * d_base[:, 0] + iv[:, 0] + ev[:, 0] / 4.0 + 110.0
    hp_frac = ctx.hp_and_active[ar, i, 0]
    return dict(def_stat=def_stat, spd_stat=spd_stat, maxhp=maxhp, cur_hp=hp_frac * maxhp, hp_frac=hp_frac,
                t1=ctx.type1_ids[ar, i], t2=ctx.type2_ids[ar, i],
                amul=op.ABILITY_DAMAGE_MULT[ctx.ability1_ids[ar, i]],
                reflect=ctx.screen_feature[:, 0], ls=ctx.screen_feature[:, 2], alive=(hp_frac > 0).float())


# ------------------------------------------------------------------------------------------------ defenders
def defender_tables(op) -> Dict[str, torch.Tensor]:
    """Per-species hidden-path defender rows: what a one-hot `species_probs` row makes the op compute."""
    return dict(def_=op.SPECIES_SPREAD_PRIOR[:, _SB_DEF, 0], spd=op.SPECIES_SPREAD_PRIOR[:, _SB_SPD, 0],
                maxhp=2.0 * op.BASE_STATS[:, 0] + 31.0 + 110.0, mult=op.SPECIES_EXP_MULT)


def averaged_defender(op, p: torch.Tensor) -> Dict[str, torch.Tensor]:
    """p [B,S] -> the blob's expected-latent defender, spelled as `_outgoing_matrix` spells it: [B,1] / [B,1,T]."""
    e_bulk = p @ op.SPECIES_SPREAD_PRIOR[..., 0]
    e_maxhp = 2.0 * (p @ op.BASE_STATS[:, 0]) + 31.0 + 110.0
    return dict(def_=e_bulk[:, _SB_DEF, None], spd=e_bulk[:, _SB_SPD, None], maxhp=e_maxhp[:, None],
                mult=(p @ op.SPECIES_EXP_MULT)[:, None, :])


def outgoing(op, att: Dict[str, torch.Tensor], dfn: Dict[str, torch.Tensor]):
    """-> (high, ko) [B,4,E]: max-roll damage fraction of the defender's max HP (capped), and acc x P(OHKO) on a
    full-HP defender (UN-nulled). `dfn` rows are [E] (shared) or [B,E] (per decision)."""
    def bE(x):
        return x[None, :] if x.dim() == 1 else x
    d_def, d_spd, d_maxhp = bE(dfn["def_"]), bE(dfn["spd"]), bE(dfn["maxhp"])
    mult = dfn["mult"]
    if mult.dim() == 2:
        mult = mult[None]                                                     # [1,E,T]
    B = att["A"].shape[0]
    E = d_def.shape[1]
    mty = att["move_ty"].long()
    eff = torch.gather(mult.expand(B, E, mult.shape[-1]), 2,
                       mty[:, None, :].expand(B, E, 4)).transpose(1, 2)       # [B,4,E]
    phys = att["phys"][:, :, None]
    D = phys * d_def[:, None, :] + (1.0 - phys) * d_spd[:, None, :]
    core = 42.0 * att["bp"][:, :, None] * att["A"][:, :, None] / (D + EPS) / 50.0 + 2.0
    dmg_ns = core * att["stab"][:, :, None] * eff * 0.925 * att["usable"][:, :, None] * att["weather"][:, :, None]
    maxhp = d_maxhp[:, None, :]
    high, _low, _crit, ko = op._rolls(dmg_ns, att["screen"][:, :, None], maxhp, maxhp, att["acc"][:, :, None], EPS)
    fixed = att["fixed"][:, :, None]
    is_fixed = fixed > 0
    not_immune = (eff > 0).float()
    high = torch.where(is_fixed, (fixed / (maxhp + EPS)) * not_immune, high)
    ko = torch.where(is_fixed, att["acc"][:, :, None] * (fixed >= maxhp).float() * not_immune, ko)
    g = (att["usable"] * att["gate"][:, None])[:, :, None]
    return high * g, ko * g


# ------------------------------------------------------------------------------------------------ attackers
def species_move_logits(fx) -> torch.Tensor:
    """[S,M] the cold-start COMPOSED move posterior of a species-s slot with no revealed move: the MoveBelief prior
    row (zero head delta) through `HPTypeBelief.compose_typed_hp` with the species' own HP-type prior."""
    mb, hpb = fx.move_belief, fx.hp_type_belief_head
    S, M = mb.move_prior_logits.shape
    raw = mb.move_prior_logits[None]                                          # [1,S,M]
    post = torch.softmax(torch.log(hpb.hp_prior.clamp_min(1e-6)), dim=-1)[None]   # == forward() at delta 0
    typed, _ = hpb.compose_typed_hp(raw, post, torch.zeros(1, S, 16), torch.zeros(1, S, 4, dtype=torch.long))
    return typed[0]


def hidden_slot_move_logits(fx, p: torch.Tensor) -> torch.Tensor:
    """[B,M] today's HIDDEN-slot composed posterior at cold start for the species marginal p: the E10 mixture
    (`hidden_slot_prior_logits`) composed with the species-0 HP-type prior row (what a hidden slot's id 0 reads)."""
    mb, hpb = fx.move_belief, fx.hp_type_belief_head
    B = p.shape[0]
    raw = mb.hidden_slot_prior_logits(p.float())[:, None, :]                  # [B,1,M]
    post = torch.softmax(torch.log(hpb.hp_prior[0].clamp_min(1e-6)), dim=-1)[None, None].expand(B, 1, 16)
    typed, _ = hpb.compose_typed_hp(raw, post, torch.zeros(B, 1, 16), torch.zeros(B, 1, 4, dtype=torch.long))
    return typed[:, 0]


def attacker_tables(op, fx, K: int) -> Dict[str, torch.Tensor]:
    """Per-species D4 attacker rows: top-K candidates of its own composed posterior, de-timid offense, STAB."""
    logits = species_move_logits(fx)
    w_all = torch.sigmoid(logits) * op.HP_CAND_MASK[None, :]                  # [S,M]
    idx = w_all.topk(K, dim=-1).indices
    off_const = 31.0 + 252.0 / 4.0 + 5.0
    st = op.SPECIES_TYPE
    mty = op.MOVE_TYPE_IDX[idx]
    stab = 1.0 + 0.5 * ((mty == st[:, :1]) | (mty == st[:, 1:2])).float()
    return dict(w=w_all.gather(-1, idx), bp=op.MOVE_BP[idx], mty=mty, phys=op.MOVE_PHYS[idx],
                acc=op.MOVE_ACCURACY[idx], stab=stab,
                atk=(2.0 * op.BASE_STATS[:, 1] + off_const) * 1.1, spa=(2.0 * op.BASE_STATS[:, 3] + off_const) * 1.1,
                w_all=w_all)


def averaged_attacker(op, fx, tabs, p: torch.Tensor, K: int) -> Dict[str, torch.Tensor]:
    """The AVERAGED attacker M3 (c) gives OTHER (no production counterpart: today every hidden attacker is gated
    off). Declared here, as the f(E[x]) analogue of the averaged defender:
      - offense: E[atk], E[spa] under p (linear in base stats, so the stat of the mean base stat);
      - moves: today's HIDDEN-slot composed move posterior for p (the E10 mixture), top-K;
      - STAB of candidate m: the presence-weighted expected STAB, sum_s p_s w_s(m) stab(s,m) / sum_s p_s w_s(m)
        over the per-species composed posteriors (1.0 where the denominator is 0)."""
    w_all = torch.sigmoid(hidden_slot_move_logits(fx, p)) * op.HP_CAND_MASK[None, :]   # [B,M]
    idx = w_all.topk(K, dim=-1).indices                                                # [B,K]
    st = op.SPECIES_TYPE
    mty_all = op.MOVE_TYPE_IDX                                                         # [M]
    stab_sm = 1.0 + 0.5 * ((mty_all[None, :] == st[:, :1]) | (mty_all[None, :] == st[:, 1:2])).float()  # [S,M]
    num = p @ (tabs["w_all"] * stab_sm)
    den = p @ tabs["w_all"]
    stab_all = torch.where(den > 0, num / den.clamp_min(1e-30), torch.ones_like(den))
    return dict(w=w_all.gather(-1, idx), bp=op.MOVE_BP[idx], mty=op.MOVE_TYPE_IDX[idx], phys=op.MOVE_PHYS[idx],
                acc=op.MOVE_ACCURACY[idx], stab=stab_all.gather(-1, idx),
                atk=p @ tabs["atk"], spa=p @ tabs["spa"], _per_decision=True)


def incoming(op, dfn: Dict[str, torch.Tensor], att: Dict[str, torch.Tensor]):
    """-> (worst, pko) [B,E]: D4's (our active, attacker) cell reduced to max(phys_high, spec_high) and
    max(phys_pko, spec_pko). `att` rows are [E,K] (per species, shared) or [B,K] (one averaged entity, E = 1)."""
    if att.get("_per_decision", False):
        w, bp, mty, phys, acc, stab = (att[k][:, None, :] for k in ("w", "bp", "mty", "phys", "acc", "stab"))  # [B,1,K]
        atk, spa = att["atk"][:, None], att["spa"][:, None]                                                    # [B,1]
    else:
        w, bp, mty, phys, acc, stab = (att[k][None] for k in ("w", "bp", "mty", "phys", "acc", "stab"))       # [1,E,K]
        atk, spa = att["atk"][None], att["spa"][None]                                                          # [1,E]
    B = dfn["maxhp"].shape[0]
    E, K = w.shape[1], w.shape[2]
    T = op.CHART.shape[-1]
    idx = mty.expand(B, E, K).long()
    def g(tab):                                                                      # tab [B,T] -> [B,E,K]
        return torch.gather(tab[:, None, :].expand(B, E, T), 2, idx)
    eff = g(op.CHART[dfn["t1"]]) * g(op.CHART[dfn["t2"]]) * g(dfn["amul"])
    A = phys * atk[..., None] + (1.0 - phys) * spa[..., None]
    D = phys * dfn["def_stat"][:, None, None] + (1.0 - phys) * dfn["spd_stat"][:, None, None]
    core = 42.0 * bp * A / (D + EPS) / 50.0 + 2.0
    dmg_ns = core * stab * eff * 0.925
    dmg_ns = dmg_ns * (bp > 0).float()
    screen = 1.0 - 0.5 * (dfn["reflect"][:, None, None] * phys + dfn["ls"][:, None, None] * (1.0 - phys))
    high, _l, _c, ko = op._rolls(dmg_ns, screen, dfn["maxhp"][:, None, None], dfn["cur_hp"][:, None, None], acc, EPS)
    ph = (w * high * phys).amax(-1); sh = (w * high * (1.0 - phys)).amax(-1)
    pk = (w * ko * phys).amax(-1); sk = (w * ko * (1.0 - phys)).amax(-1)
    al = dfn["alive"][:, None]
    return torch.maximum(ph, sh) * al, torch.maximum(pk, sk) * al, torch.stack([ph, sh, pk, sk], -1) * al[..., None]
