"""gen3_move_resolution_v1 — the MOVE-RESOLUTION family (`--move-resolution on`; architecture audit F11, §9).

**The owner's ruling (2026-10-06).** FACTS — the hard-to-compute mechanics of what will actually happen if I
press this — are kept and consolidated into ONE family; JUDGMENTS — opinions of the right play — are dropped. The
seven hand-designed per-action blocks (`intent_move_cell`, `intent_threshold`, `intent_conditional`,
`pair_outcome_cell`, `switch_branch_cell` on the move cell; `pair_outcome_switch`, `conditional_threat_cell` on
the switch cell) are REPLACED by this one family when the flag is on: their 55 fact coordinates are kept (six of
them rebuilt where the shipped spelling disagreed with the simulator, one with its production bug fixed) and their
10 judgments (`tempo_cost`, `wasted_ko`, `neutralization`, the Focus Punch / Substitute / Endure / Endeavor hand
thresholds, the two hazard-stake products) are gone. The inventory is `design_arch_audit.md` §9.

## The new fact: P(the action resolves as stated)

For every legal MOVE action m, the family computes by the exact gen-3 rules (`move_resolution_rules`, each rule
verified at the Showdown source) the probability that the move does what it says — it LANDS, is not BLOCKED, the
target is not IMMUNE, and it is not a NO-OP — INTENT-WEIGHTED wherever the opponent's choice matters:

    p_resolve(m) = Σ_k α_k · A_k(m) · L(m | k)   +   a_un · A_un(m) · L_un(m)   +   α_SWITCH · p_act(m) · L_sw(m)

* `α_k` — the published opponent-intent α over their believed move seats (stop-grad, the meaningful-K mask applied,
  never renormalised); `α_SWITCH` its switch mass; `a_un = 1 − Σα − α_SWITCH` the mass of a seat the model does not
  name (the target stays, nothing about its move is claimed).
* `A_k(m)` — P(we get to execute m given they click seat k): our own state (`p_act`: asleep with the observed wake
  odds, frozen with its 1/5 thaw, paralysis 1/4, confusion 1/2, infatuation 1/2) × not being KO'd, put to sleep /
  frozen / paralysed or flinched FIRST by seat k. "First" is the exact ORDER: a higher-priority seat always moves
  first, an equal one with P(they outspeed) = 1 − the op's `p_outspeed`. **Destiny Bond** is the one exception: its
  bond is up if WE move first, or — the persisted volatile — if we bonded last turn.
* `L(m | k)` — P(the stated effect applies given seat k): the target's rules (type immunity incl. an UNREVEALED
  immunity ability on their active through its Smogon prior; an existing status / volatile; Sleep, Freeze Clause;
  Safeguard; a Substitute and the `bypasssub` set; Soundproof), our own rules (Substitute at ≤ 1/4 HP, a heal at full
  HP, a boost at +6, spikes at 3 layers, a screen or weather already up, …), and seat k's interference in move
  order: their faster Protect / Detect (the `protect` flag, their stall odds), Magic Coat (the `reflectable` flag),
  Substitute or Taunt. Counter / Mirror Coat need seat k to HIT us (not our Substitute) with the right category
  first (Hidden Power: always Counter, never Mirror Coat); Focus Punch fails exactly when it does.
* `L_sw(m)` — the same rules on the β-weighted ARRIVAL (a switching opponent attacks nothing; Encore / Disable find
  no last move; a Ghost arrival blocks Rapid Spin; Counter returns nothing).

Its decorrelated factors ride beside it (`p_lands_stay`, `p_lands_switch`, `p_ko_first`, `p_act`): a thin tanh
scorer does not multiply its own inputs, so the product is formed here AND the factors are shipped.

**Destiny Bond (owner, explicit): NO threshold.** `dbond_p_ko` = P(the opponent KOs us this turn) =
`Σ_k α_k · ko_k`, the intent × the operator's accuracy-folded KO estimates; `p_resolve` adds the exact trigger rule.

## The INCOMING status coordinates, corrected (the entity-coverage audit's rank-2 gaps)

The op's `_incoming_status_lands` ignores five gen-3 rules (verified: Safeguard `data/moves.ts:15587-15615`; Sleep
Clause `data/rulesets.ts:1378-1402`; Freeze Clause `data/rulesets.ts:1451-1471`, in gen3ou's rule set; Substitute
`gen4/moves.ts:1283-1320`). The family corrects the six incoming status coordinates it delivers: OUR side's
Safeguard zeroes all six, OUR live non-Rest sleeper zeroes `p_slp` (incoming Sleep Clause), ANY frozen mon of ours
zeroes `p_frz` (Freeze Clause), OUR active's Substitute zeroes its own row (a bench candidate switching in has
none). The op itself is untouched, so `--move-resolution off` stays byte-identical.

**One more op defect, corrected the same way.** The op reads an opponent's ability as REVEALED when its id is
non-zero (`_status_landing`'s `revealed = opp_ability > 0`, and `ABILITY_DAMAGE_MULT` / `ABILITY_STATUS_BLOCK`
indexed by the raw id elsewhere), but the observation writes an UNREVEALED opponent's most likely ability into
that slot with `known = 0`. The family reads the `known` flag (and re-runs `_status_landing` with the unrevealed
ids zeroed, so its Smogon-prior branch is the one taken); the production op still carries the defect.

## Contract

Zero-init projections built as `IsolatedLinear`s (SB3's orthogonal re-init skips them and they draw nothing from
the global RNG, so building the family moves no other parameter's initial bytes); ON-at-init contributes exactly 0.
α / β are the stop-grad publications. The family is BLOB-only (`--belief-tokens fixed_mass` is refused at build).
"""
from __future__ import annotations

import dataclasses
from typing import Any, Dict, NamedTuple, Optional, Tuple

import torch

from agents.model.arch_constants import (MOVE_RESOLUTION_MOVE_DIM, MOVE_RESOLUTION_SWITCH_DIM,
                                         _MOVE_RESOLUTION_MOVE_RAW, _MOVE_RESOLUTION_SWITCH_RAW)
from agents.model.move_resolution_rules import (
    ABILITY_INNER_FOCUS, ABILITY_OWN_TEMPO, ABILITY_SLEEP_BLOCK, ABILITY_SOUNDPROOF, BELLY_DRUM_HP_FRACTION,
    MOVE_RESOLUTION_MOVE_COORDS, MOVE_RESOLUTION_SWITCH_COORDS, P_CONFUSION_SELF_HIT, P_FULL_PARA,
    P_INFATUATION, P_THAW, PAIR_FACT_COORDS, PURSUIT_SWITCH_MULT, ROLL_WINDOW, SUB_HP_FRACTION)
from agents.model.move_resolution_tables import (C_BRN, C_FRZ, C_PAR, C_PSN, C_SLP, C_TOX, FLAG_IDX, KIND_IDX,
                                                 NAMED_ABILITIES, S_LS_OURS, S_MIST_OURS, S_REFLECT_OURS, S_SG_OPP,
                                                 S_SG_OURS, SEAT_KIND_IDX, WEATHER_COL,
                                                 build_move_tables, build_species_tables)

assert len(MOVE_RESOLUTION_MOVE_COORDS) == _MOVE_RESOLUTION_MOVE_RAW, \
    "MOVE_RESOLUTION_MOVE_COORDS and _MOVE_RESOLUTION_MOVE_RAW disagree — one was edited alone."
assert len(MOVE_RESOLUTION_SWITCH_COORDS) == _MOVE_RESOLUTION_SWITCH_RAW, \
    "MOVE_RESOLUTION_SWITCH_COORDS and _MOVE_RESOLUTION_SWITCH_RAW disagree — one was edited alone."

_EPS = 1e-6


def _volatile_index() -> Dict[str, int]:
    from agents.observation.gen3_effects import VOLATILE_SLOTS
    return {n: i for i, n in enumerate(VOLATILE_SLOTS)}


VOL = _volatile_index()


def _type_idx(name: str) -> int:
    from agents.observation.types import TypeEncoder
    return int(TypeEncoder.TYPE_TO_IDX[name])


T_GHOST = _type_idx("GHOST")
_PAIR = {n: i for i, n in enumerate(PAIR_FACT_COORDS)}


# =========================================================================================== the OPERANDS
class MoveResolutionOps(NamedTuple):
    """Everything the family reads, as tensors (built from a forward by `gather_ops`, or by hand in a test).
    ``B`` batch · ``K`` their seats · request slots 4 · team slots 6 · ``T`` = 19 attacking types."""
    gate: torch.Tensor             # [B,1]  our active alive × an opponent active exists
    # --- our action
    req_ids: torch.Tensor          # [B,4] long  our request-slot move nums
    req_type: torch.Tensor         # [B,4] long  their resolved TypeEncoder types (Beat Up typeless)
    is_dmg: torch.Tensor           # [B,4]  the op prices damage (formula BP > 0 or a non-formula kind)
    acc: torch.Tensor              # [B,4]  base accuracy
    prio: torch.Tensor             # [B,4]  priority
    inflicts: torch.Tensor         # [B,4]  a major status / Leech Seed inflictor
    p_land: torch.Tensor           # [B,4]  the op's `_status_landing` P(applies to their active)
    st_type_imm: torch.Tensor      # [B,4,T] the per-move status TYPE immunity row
    st_cat: torch.Tensor           # [B,4] long  status category (0 none · 1-5 majors · 6 Leech Seed)
    st_blocked: torch.Tensor       # [B,4]  blocked when the target already has a major status
    self_boost: torch.Tensor       # [B,4,5] the self-boost stages (atk def spa spd spe), Curse resolved by type
    sec_flinch: torch.Tensor       # [B,4]  P(our move flinches) — chance × accuracy × Serene Grace × Shield Dust
    # --- our side
    our_stage: torch.Tensor        # [B,5]  our active's stages
    our_hp: torch.Tensor           # [B]    our active's HP fraction
    our_cond: torch.Tensor         # [B,6,7] our six condition one-hots
    our_alive: torch.Tensor        # [B,6]
    our_rest: torch.Tensor         # [B,6]  1 = a Rest sleep
    our_active: torch.Tensor       # [B] long
    our_p_wake: torch.Tensor       # [B]    P(our active wakes at its next move) — the obs sleep belief
    our_vol: torch.Tensor          # [B,V]  our active's volatiles
    our_protect_odds: torch.Tensor  # [B]
    our_named_abl: torch.Tensor    # [B,n]  our active's named abilities (revealed: exact)
    our_is_ghost: torch.Tensor     # [B,6]
    our_hp_all: torch.Tensor       # [B,6]
    beatup_n: torch.Tensor         # [B]    our party's alive, unstatused members (the user included)
    our_wish: torch.Tensor         # [B]    our side's pending-Wish observation (> 0 = a Wish is pending)
    # --- their side
    opp_active: torch.Tensor       # [B] long
    opp_vol: torch.Tensor          # [B,V]  their active's volatiles
    opp_hp: torch.Tensor           # [B]    their active's HP fraction
    opp_protect_odds: torch.Tensor  # [B]
    opp_last_move: torch.Tensor    # [B] long  their active's last move num (0 = none / a switch)
    opp_p_wake: torch.Tensor       # [B]    P(their active wakes at its next move) — the obs sleep belief
    opp_alive_total: torch.Tensor  # [B]    their mons not known fainted
    opp_cond: torch.Tensor         # [B,6,7]
    opp_rest: torch.Tensor         # [B,6]
    opp_alive: torch.Tensor        # [B,6]  revealed-and-alive OR unrevealed
    imm_dmg: torch.Tensor          # [B,6,T] P(their slot is immune to an attacking type) — types + ability
    chart0: torch.Tensor           # [B,6,T] P(their slot's TYPES make it immune)
    p_type: torch.Tensor           # [B,6,T] P(their slot has type t)
    abl_block: torch.Tensor        # [B,6,C] P(their slot's ability blocks status category c)
    opp_named_abl: torch.Tensor    # [B,6,n] P(their slot has a named ability)
    screens: torch.Tensor          # [B,8]
    weather: torch.Tensor          # [B,W]
    spikes: torch.Tensor           # [B,2]  [ours, theirs] / 3
    # --- the opponent model
    alpha: torch.Tensor            # [B,K]  α move slice (stop-grad, meaningful-K masked)
    a_switch: torch.Tensor         # [B,1]
    beta: torch.Tensor             # [B,6]  β (zero when there is no legal switch-in)
    seat_nums: torch.Tensor        # [B,K] long
    seat_prio: torch.Tensor        # [B,K]
    seat_kind: torch.Tensor        # [B,K,S]
    seat_flag: torch.Tensor        # [B,K,F]
    seat_phys: torch.Tensor        # [B,K]  gen-3 type-based category
    seat_flinch: torch.Tensor      # [B,K]  their seat's flinch chance × their Serene Grace
    pair_in: torch.Tensor          # [B,6,K,14] the op's unified outcome grid (the 12 facts read)
    pair_gate: torch.Tensor        # [B,6,1]
    pair_type_mult: torch.Tensor   # [B,6,K]
    p_out: torch.Tensor            # [B,1]  P(our active outspeeds theirs)
    out_cells: torch.Tensor        # [B,4,6,5] our move × their mon [low, high, crit, pko, type_mult]
    c2_base: torch.Tensor          # [B,4,4] [is_status, d_their_outspeed, d_sched, e_slp_free]
    d_burn_k: torch.Tensor         # [B,K]
    d_slp_k: torch.Tensor          # [B,K]
    is_brn: torch.Tensor           # [B,4]
    is_slp: torch.Tensor           # [B,4]


def _g(t: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """``t[idx]`` with the index clamped into the table (a padded / out-of-vocabulary num reads row 0's zeros)."""
    return t[idx.clamp(min=0, max=t.shape[0] - 1)]


# ===================================================================================== the FACTS (pure)
def incoming_status_correction(o: MoveResolutionOps) -> torch.Tensor:
    """``[B,6,6]`` the multiplier on the six incoming status coordinates (par brn frz slp psn tox) per OUR
    defender: OUR Safeguard (all six), incoming Sleep Clause (a live non-Rest sleeper of ours ⇒ no sleep), Freeze
    Clause (any frozen mon of ours ⇒ no freeze), and OUR active's Substitute (its own row only)."""
    sg = o.screens[:, S_SG_OURS]                                                      # [B]
    slp_live = o.our_cond[..., C_SLP] * o.our_alive * (1.0 - o.our_rest)
    clause_slp = (slp_live.sum(-1) > 0.5).to(sg.dtype)                                # [B]
    clause_frz = (o.our_cond[..., C_FRZ].sum(-1) > 0.5).to(sg.dtype)                  # [B]
    one = torch.ones_like(sg)
    col = torch.stack([one, one, 1.0 - clause_frz, 1.0 - clause_slp, one, one], dim=-1)   # [B,6] par brn frz slp psn tox
    sub = o.our_vol[:, VOL["substitute"]].to(sg.dtype)                                # [B]
    act = torch.nn.functional.one_hot(o.our_active, 6).to(sg.dtype)                   # [B,6]
    mult: torch.Tensor = ((1.0 - sg)[:, None, None] * col[:, None, :]
                          * (1.0 - act * sub[:, None])[:, :, None])                   # [B,6,6]
    return mult


def move_facts(o: MoveResolutionOps, kind_t: torch.Tensor, flag_t: torch.Tensor) -> torch.Tensor:
    """``[B,4,len(MOVE_RESOLUTION_MOVE_COORDS)]`` — the move-cell facts. ``kind_t`` / ``flag_t`` are the
    per-move-num `KIND` / `FLAG` tables."""
    dt = o.alpha.dtype
    B, K = o.alpha.shape
    ar = torch.arange(B, device=o.alpha.device)
    kind = _g(kind_t, o.req_ids).to(dt)                                               # [B,4,NK]
    flag = _g(flag_t, o.req_ids).to(dt)                                               # [B,4,NF]

    def kd(n: str) -> torch.Tensor:
        return kind[..., KIND_IDX[n]]

    def fl(n: str) -> torch.Tensor:
        return flag[..., FLAG_IDX[n]]

    def sk(n: str) -> torch.Tensor:
        return o.seat_kind[..., SEAT_KIND_IDX[n]].to(dt)

    alpha = o.alpha.to(dt)
    a_sw = o.a_switch.to(dt)                                                          # [B,1]
    a_un = (1.0 - alpha.sum(-1, keepdim=True) - a_sw).clamp(min=0.0)                  # [B,1]
    p_out = o.p_out.to(dt)
    # ---------------------------------------------------------------- ORDER: does seat k act before move m?
    pm = o.prio[:, :, None]
    pk = o.seat_prio[:, None, :]
    pre = (pk > pm).to(dt) + (pk == pm).to(dt) * (1.0 - p_out)[:, :, None]            # [B,4,K]
    pre_un = (o.prio < 0).to(dt) + (o.prio == 0).to(dt) * (1.0 - p_out)               # [B,4]
    # ---------------------------------------------------------------- their seats vs OUR ACTIVE
    corr = incoming_status_correction(o)                                              # [B,6,6]
    pin_a = o.pair_in[ar, o.our_active]                                               # [B,K,14]
    ko_k = pin_a[..., _PAIR["ko_ramp"]]                                               # acc · P(KO | hit)
    acc_k = pin_a[..., _PAIR["acc"]]
    high_k = pin_a[..., _PAIR["high"]]
    tm_a = o.pair_type_mult[ar, o.our_active]                                         # [B,K]
    seat_dmg = o.seat_flag[..., FLAG_IDX["damaging"]].to(dt)
    seat_hp = o.seat_flag[..., FLAG_IDX["is_hp"]].to(dt)
    hits_k = seat_dmg * acc_k * (tm_a > 0).to(dt)                                     # P(seat k's hit lands on us)
    corr_a = corr[ar, o.our_active]                                                   # [B,6]
    st_a = pin_a[..., 6:12] * corr_a[:, None, :]                                      # [B,K,6] par brn frz slp psn tox
    immob_k = (st_a[..., 3] + (1.0 - P_THAW) * st_a[..., 2] + P_FULL_PARA * st_a[..., 0]).clamp(0.0, 1.0)
    our_inner = o.our_named_abl[:, NAMED_ABILITIES.index(ABILITY_INNER_FOCUS)]       # [B]
    fl_k = hits_k * o.seat_flinch.to(dt) * (1.0 - our_inner)[:, None]
    blocked_k = 1.0 - (1.0 - ko_k) * (1.0 - immob_k) * (1.0 - fl_k)                  # [B,K]
    act_k = 1.0 - pre * blocked_k[:, None, :]                                         # [B,4,K]
    # ---------------------------------------------------------------- p_act: our own state
    cond_a = o.our_cond[ar, o.our_active]                                             # [B,7]
    asleep, frozen, par = cond_a[:, C_SLP], cond_a[:, C_FRZ], cond_a[:, C_PAR]
    st_talk = kd("sleeptalk")
    ok_slp = (asleep[:, None] * (st_talk * (1.0 - o.our_p_wake[:, None]) + (1.0 - st_talk) * o.our_p_wake[:, None])
              + (1.0 - asleep)[:, None] * (1.0 - st_talk))
    ok_frz = frozen[:, None] * (fl("defrost") + (1.0 - fl("defrost")) * P_THAW) + (1.0 - frozen)[:, None]
    conf = o.our_vol[:, VOL["confusion"]]
    attr = o.our_vol[:, VOL["attract"]]
    p_act = (ok_slp * ok_frz * (1.0 - P_FULL_PARA * par)[:, None] * (1.0 - P_CONFUSION_SELF_HIT * conf)[:, None]
             * (1.0 - P_INFATUATION * attr)[:, None])                                # [B,4]
    # ---------------------------------------------------------------- the TARGET (their active)
    oa = o.opp_active
    imm_a = o.imm_dmg[ar, oa].gather(1, o.req_type)                                   # [B,4]
    chart0_a = o.chart0[ar, oa].gather(1, o.req_type)
    ov = o.opp_vol

    def ovl(n: str) -> torch.Tensor:
        return ov[:, VOL[n]][:, None]

    opp_cond_a = o.opp_cond[ar, oa]                                                   # [B,7]
    opp_statused = (opp_cond_a[:, 1:].sum(-1) > 0.5).to(dt)[:, None]
    opp_asleep = opp_cond_a[:, C_SLP][:, None]
    named_a = o.opp_named_abl[ar, oa]                                                 # [B,n]
    sp_sound = named_a[:, NAMED_ABILITIES.index(ABILITY_SOUNDPROOF)][:, None]
    sp_tempo = named_a[:, NAMED_ABILITIES.index(ABILITY_OWN_TEMPO)][:, None]
    abl_blk_a = o.abl_block[ar, oa]                                                   # [B,C]
    slp_cat = 4
    abl_slp_a = abl_blk_a[:, slp_cat][:, None]
    sg_opp = o.screens[:, S_SG_OPP][:, None]
    opp_slp_live = o.opp_cond[..., C_SLP] * o.opp_alive * (1.0 - o.opp_rest)
    clause = (opp_slp_live.sum(-1) > 0.5).to(dt)[:, None]                             # our one sleep is used
    sub_up = ovl("substitute")
    our_ghost_a = o.our_is_ghost[ar, o.our_active].to(dt)[:, None]                    # [B,1]
    # a non-Ghost user's Curse aims at itself (`onModifyMove`: nonGhostTarget)
    foe = fl("foe") * (1.0 - kd("curse") * (1.0 - our_ghost_a))
    status_cat = fl("status_cat")
    subblock = foe * (1.0 - fl("bypasssub"))
    has_last = (o.opp_last_move > 0).to(dt)[:, None]
    last_fe = _g(flag_t, o.opp_last_move)[:, FLAG_IDX["failencore"]].to(dt)[:, None]
    major = ((o.st_cat > 0) & (o.st_cat < 6)).to(dt)
    is_counter, is_mc, is_fp, is_db = kd("counter"), kd("mirrorcoat"), kd("focuspunch"), kd("destinybond")
    special_dmg = is_counter + is_mc
    is_dmg = o.is_dmg.to(dt) * (1.0 - special_dmg)
    # damaging moves (Counter / Mirror Coat apart): accuracy × not immune × the move's own target condition
    l_dmg = o.acc * (1.0 - imm_a)
    l_dmg = l_dmg * (1.0 - kd("dreameater") * (1.0 - opp_asleep * (1.0 - sub_up)))
    l_dmg = l_dmg * (1.0 - kd("beatup") * (o.beatup_n <= 0.5).to(dt)[:, None])
    # the six-status / Leech Seed inflictors: the op's p_land, + Glare's Ghost immunity, Safeguard, Soundproof,
    # an already-seeded target
    l_inf = (o.p_land * (1.0 - fl("status_type_imm") * chart0_a) * (1.0 - major * sg_opp)
             * (1.0 - fl("sound") * sp_sound) * (1.0 - kd("leechseed") * ovl("leechseed")))
    # every other status move, by kind
    our_stage = o.our_stage
    can_raise = ((o.self_boost > 0).to(dt) * (our_stage[:, None, :] < 5.5).to(dt)
                 + (o.self_boost < 0).to(dt) * (our_stage[:, None, :] > -5.5).to(dt)).sum(-1).clamp(max=1.0)
    has_boost = ((o.self_boost != 0).to(dt).sum(-1) > 0.5).to(dt)
    our_hp = o.our_hp[:, None]
    our_cond_any = (o.our_cond[..., 1:].sum(-1) > 0.5).to(dt)                          # [B,6]
    our_vol = o.our_vol
    alive_others = (o.our_alive.sum(-1) - o.our_alive[ar, o.our_active] > 0.5).to(dt)[:, None]
    my_slp_block = (o.our_named_abl[:, NAMED_ABILITIES.index(ABILITY_SLEEP_BLOCK[0])]
                    + o.our_named_abl[:, NAMED_ABILITIES.index(ABILITY_SLEEP_BLOCK[1])]).clamp(max=1.0)[:, None]
    weather_up = torch.zeros_like(o.acc)
    for w, col in WEATHER_COL.items():
        weather_up = weather_up + kd(w) * o.weather[:, col][:, None]
    sc = o.screens
    cond_status = (
        kd("taunt") * (1.0 - ovl("taunt"))
        + kd("encore") * (1.0 - ovl("encore")) * (1.0 - last_fe)          # × has_last per seat (below)
        + kd("disable") * (1.0 - ovl("disable"))                           # × has_last per seat (below)
        + kd("torment") * (1.0 - ovl("torment"))
        + kd("yawn") * (1.0 - opp_statused) * (1.0 - ovl("yawn")) * (1.0 - abl_slp_a) * (1.0 - clause) * (1.0 - sg_opp)
        + kd("confuse") * (1.0 - ovl("confusion")) * (1.0 - sp_tempo) * (1.0 - sg_opp)
        + kd("trap") * (1.0 - ovl("trapped"))
        + kd("attract") * (1.0 - ovl("attract"))
        + kd("nightmare") * opp_asleep * (1.0 - ovl("nightmare"))
        + kd("phaze") * (o.opp_alive_total > 1.5).to(dt)[:, None] * (1.0 - ovl("ingrain"))
        + kd("substitute") * (our_hp > SUB_HP_FRACTION).to(dt) * (1.0 - our_vol[:, VOL["substitute"]][:, None])
        + kd("bellydrum") * (our_hp > BELLY_DRUM_HP_FRACTION).to(dt) * (our_stage[:, 0:1] < 5.5).to(dt)
        + kd("rest") * (1.0 - my_slp_block)            # asleep: p_act (it must wake to move); full HP: per seat
        + kd("heal")                                     # full HP: per seat (a faster hit makes room to heal)
        + kd("refresh") * (cond_a[:, C_PSN] + cond_a[:, C_TOX] + cond_a[:, C_PAR] + cond_a[:, C_BRN])[:, None]
        + kd("cleric") * (our_cond_any * o.our_alive).sum(-1).clamp(max=1.0)[:, None]
        + kd("sleeptalk")                                                   # p_act carries "asleep"
        + kd("batonpass") * alive_others
        + kd("wish") * (o.our_wish <= 0.0).to(dt)[:, None]
        + kd("focusenergy") * (1.0 - our_vol[:, VOL["focusenergy"]][:, None])
        + kd("ingrain") * (1.0 - our_vol[:, VOL["ingrain"]][:, None])
        + kd("reflect") * (1.0 - sc[:, S_REFLECT_OURS][:, None]) + kd("lightscreen") * (1.0 - sc[:, S_LS_OURS][:, None])
        + kd("safeguard") * (1.0 - sc[:, S_SG_OURS][:, None]) + kd("mist") * (1.0 - sc[:, S_MIST_OURS][:, None])
        + kd("spikes") * (o.spikes[:, 1:2] < 0.99).to(dt)
        + (kd("sunnyday") + kd("raindance") + kd("sandstorm") + kd("hail")) * (1.0 - weather_up)
        + (kd("protect") + kd("endure")) * o.our_protect_odds[:, None]
        + kd("magiccoat") + kd("destinybond")                               # per seat (below)
        + kd("curse") * (our_ghost_a * (1.0 - ovl("curse")) + (1.0 - our_ghost_a) * can_raise)
    )
    named_status = kind[..., [KIND_IDX[n] for n in (
        "taunt", "encore", "disable", "torment", "yawn", "confuse", "trap", "attract", "nightmare", "phaze",
        "substitute", "bellydrum", "rest", "heal", "refresh", "cleric", "sleeptalk", "batonpass", "wish",
        "focusenergy",
        "ingrain", "reflect", "lightscreen", "safeguard", "mist", "spikes", "sunnyday", "raindance", "sandstorm",
        "hail", "protect", "endure", "magiccoat", "destinybond", "curse")]].sum(-1).clamp(max=1.0)
    generic = (1.0 - named_status) * (has_boost * can_raise + (1.0 - has_boost))      # a boost move at +6 is a no-op
    # ...unless a faster seat lowers / resets our stats first (per seat), or an Intimidate arrival drops our Atk
    cap_block = (1.0 - named_status) * has_boost * (1.0 - can_raise) * (1.0 - fl("foe"))
    full = (our_hp >= 1.0).to(dt)                                                     # a heal / Rest at full HP fails
    hp_gate = (kd("heal") + kd("rest")) * full                                        # ...unless a faster hit lands
    is_pf = kd("protect") + kd("endure")                                              # fails if NO action follows it
    acc_foe = foe * o.acc + (1.0 - foe)                                               # accuracy only aims at a foe
    l_status = (acc_foe * (1.0 - status_cat * subblock * sub_up) * (1.0 - fl("sound") * sp_sound)
                * (cond_status + generic))
    inf = o.inflicts.to(dt)
    l_base = (is_dmg * l_dmg + (1.0 - is_dmg - special_dmg) * (inf * l_inf + (1.0 - inf) * l_status)
              + special_dmg * (1.0 - imm_a))                                          # [B,4]  Counter / MC: ×cond_k
    # ---------------------------------------------------------------- seat k's interference (move ORDER)
    prot_odds = o.opp_protect_odds.to(dt)[:, None, None]
    blk = pre * sk("protect")[:, None, :] * prot_odds * fl("protectable")[:, :, None]
    bounce = pre * sk("magiccoat")[:, None, :] * fl("reflectable")[:, :, None]
    sub_first = (pre * sk("substitute")[:, None, :] * (o.opp_hp > SUB_HP_FRACTION).to(dt)[:, None, None]
                 * (1.0 - sub_up)[:, :, None] * (status_cat * subblock)[:, :, None])
    taunted_first = pre * sk("taunt")[:, None, :] * status_cat[:, :, None] * (1.0 - our_vol[:, VOL["taunt"]])[:, None, None]
    seat_mult = (1.0 - blk) * (1.0 - bounce) * (1.0 - sub_first) * (1.0 - taunted_first)   # [B,4,K]
    cat_ok_c = (o.seat_phys.to(dt) + seat_hp).clamp(max=1.0)
    cat_ok_m = (1.0 - o.seat_phys.to(dt)) * (1.0 - seat_hp)
    our_sub = our_vol[:, VOL["substitute"]][:, None, None]
    hit_first = pre * hits_k[:, None, :] * (1.0 - our_sub)                            # a hit on US, before m
    lowers_k = o.seat_flag[..., FLAG_IDX["lowers_foe"]].to(dt)                        # [B,K]
    l_k = (l_base[:, :, None] + cap_block[:, :, None] * pre * lowers_k[:, None, :]) * seat_mult
    l_k = l_k * (1.0 - hp_gate[:, :, None] * (1.0 - hit_first))
    # three states a FASTER seat can change before our move, so a zero there is not certain:
    #  * their self-cure (Refresh / Rest / a cleric) lifts "already statused" on our status move (acc remains);
    cures_k = o.seat_flag[..., FLAG_IDX["cures_self"]].to(dt)                          # [B,K]
    st_blk = (inf * major * o.st_blocked.to(dt) * opp_statused)[:, :, None]             # [B,4,1]
    #    so does their thaw (1/5, or a defrost move) or their waking at a move that comes BEFORE ours;
    thaw_k = P_THAW + (1.0 - P_THAW) * o.seat_flag[..., FLAG_IDX["defrost"]].to(dt)    # [B,K]
    ends_k = (cures_k + (1.0 - cures_k) * (opp_cond_a[:, C_FRZ][:, None] * thaw_k
                                           + opp_asleep * o.opp_p_wake.to(dt)[:, None])).clamp(max=1.0)
    l_k = l_k + st_blk * pre * ends_k[:, None, :] * o.acc[:, :, None] * (1.0 - sub_up)[:, :, None]
    #  * a hit on OUR Substitute may break it (its HP is not observed: counted as breaking) — our Sub then works;
    sub_blk = (kd("substitute") * our_vol[:, VOL["substitute"]][:, None]
               * (our_hp > SUB_HP_FRACTION).to(dt))[:, :, None]
    l_k = l_k + sub_blk * pre * hits_k[:, None, :]
    #  * their status landing on US first gives Refresh something to cure.
    p_cureable_k = (st_a[..., 0] + st_a[..., 1] + st_a[..., 4] + st_a[..., 5]).clamp(max=1.0)   # par brn psn tox
    l_k = l_k + (kd("refresh") * (1.0 - l_base))[:, :, None] * pre * p_cureable_k[:, None, :]
    l_k = l_k.clamp(max=1.0)
    l_k = l_k * (1.0 - is_pf[:, :, None] * pre)                                       # their move must come AFTER
    l_k = l_k * (1.0 - is_counter[:, :, None] * (1.0 - hit_first * cat_ok_c[:, None, :]))
    l_k = l_k * (1.0 - is_mc[:, :, None] * (1.0 - hit_first * cat_ok_m[:, None, :]))
    l_k = l_k * (1.0 - is_fp[:, :, None] * hit_first)
    last_eff = torch.maximum(has_last[:, :, None].expand_as(pre), pre)                  # they move first ⇒ a last move
    enc_dis = (kd("encore") + kd("disable"))[:, :, None]
    l_k = l_k * (1.0 - enc_dis * (1.0 - last_eff))
    refl_k = o.seat_flag[..., FLAG_IDX["reflectable"]].to(dt)                          # [B,K]
    l_k = l_k * (1.0 - kd("magiccoat")[:, :, None] * (1.0 - refl_k[:, None, :] * (1.0 - pre)))
    l_k = l_k * (1.0 - is_db[:, :, None] * (1.0 - ko_k[:, None, :]))                   # the bond triggers on their KO
    # the unmodeled seat: they stay and click something we do not name — no hit, no block is claimed
    l_un = l_base * (1.0 - is_counter - is_mc - kd("magiccoat") - is_db).clamp(min=0.0)
    l_un = l_un * (1.0 - kd("encore") - kd("disable")) + l_base * (kd("encore") + kd("disable")) * torch.maximum(
        has_last, pre_un)
    l_un = l_un * (1.0 - hp_gate) * (1.0 - is_pf * pre_un)
    # ---------------------------------------------------------------- the SWITCH branch (β-weighted arrival)
    beta = o.beta.to(dt)                                                              # [B,6]
    imm_j = o.imm_dmg.gather(2, o.req_type[:, None, :].expand(-1, 6, -1)).transpose(1, 2)   # [B,4,6]
    chart0_j = o.chart0.gather(2, o.req_type[:, None, :].expand(-1, 6, -1)).transpose(1, 2)
    asleep_j = o.opp_cond[..., C_SLP][:, None, :]                                     # [B,1,6]
    statused_j = (o.opp_cond[..., 1:].sum(-1) > 0.5).to(dt)[:, None, :]
    ti_j = torch.einsum("bmt,bjt->bmj", o.st_type_imm.to(dt), o.p_type.to(dt)).clamp(max=1.0)
    ti_j = torch.maximum(ti_j, fl("status_type_imm")[:, :, None] * chart0_j)
    abl_j = o.abl_block.to(dt).gather(2, o.st_cat[:, None, :].expand(-1, 6, -1)).transpose(1, 2)  # [B,4,6]
    named_j = o.opp_named_abl.to(dt)                                                  # [B,6,n]
    sound_j = named_j[..., NAMED_ABILITIES.index(ABILITY_SOUNDPROOF)][:, None, :]
    tempo_j = named_j[..., NAMED_ABILITIES.index(ABILITY_OWN_TEMPO)][:, None, :]
    slp_blk_j = o.abl_block.to(dt)[..., slp_cat][:, None, :]
    is_sleep = (o.st_cat == slp_cat).to(dt)[:, :, None]
    la_dmg = o.acc[:, :, None] * (1.0 - imm_j) * (1.0 - kd("dreameater")[:, :, None] * (1.0 - asleep_j))
    la_dmg = la_dmg * (1.0 - kd("beatup") * (o.beatup_n <= 0.5).to(dt)[:, None])[:, :, None]
    # the switch branch's Sleep Clause: a departing Natural Cure sleeper is cured on the way out
    nc_a = named_a[:, NAMED_ABILITIES.index("naturalcure")][:, None]
    act_slp = (opp_cond_a[:, C_SLP] * (1.0 - o.opp_rest[ar, oa]))[:, None]
    bench_slp = (opp_slp_live.sum(-1, keepdim=True) - act_slp).clamp(min=0.0)
    clause_sw = 1.0 - (1.0 - (bench_slp > 0.5).to(dt)) * (1.0 - act_slp * (1.0 - nc_a))  # [B,1]
    la_inf = (o.acc[:, :, None] * (1.0 - ti_j) * (1.0 - abl_j) * (1.0 - statused_j * o.st_blocked[:, :, None])
              * (1.0 - is_sleep * clause_sw[:, :, None]) * (1.0 - (major * sg_opp)[:, :, None])
              * (1.0 - fl("sound")[:, :, None] * sound_j))
    la_cond = (kd("taunt") + kd("torment") + kd("trap") + kd("attract") + kd("phaze") * (o.opp_alive_total > 1.5).to(dt)[:, None]
               )[:, :, None] + (kd("yawn")[:, :, None] * (1.0 - statused_j) * (1.0 - slp_blk_j) * (1.0 - clause_sw)[:, :, None]
                                * (1.0 - sg_opp)[:, :, None]) + kd("confuse")[:, :, None] * (1.0 - tempo_j) * (1.0 - sg_opp)[:, :, None] \
        + kd("nightmare")[:, :, None] * asleep_j + kd("curse")[:, :, None] * our_ghost_a[:, :, None]
    foe_named = kind[..., [KIND_IDX[n] for n in ("taunt", "encore", "disable", "torment", "yawn", "confuse", "trap",
                                                    "attract", "nightmare", "phaze", "magiccoat")]].sum(-1).clamp(max=1.0)
    foe_generic = foe * status_cat * (1.0 - foe_named) * (1.0 - kd("curse"))
    la_status = (o.acc[:, :, None] * (1.0 - fl("sound")[:, :, None] * sound_j)
                 * (la_cond + foe_generic[:, :, None]))
    la = is_dmg[:, :, None] * la_dmg + ((1.0 - is_dmg) * (1.0 - special_dmg))[:, :, None] * (
        inf[:, :, None] * la_inf + (1.0 - inf)[:, :, None] * la_status)
    l_sw_foe = (la * beta[:, None, :]).sum(-1)                                        # [B,4]
    pursuit = kd("pursuit")
    intim = (named_j[..., NAMED_ABILITIES.index("intimidate")] * beta).sum(-1, keepdim=True)   # [B,1]
    raises_atk = (o.self_boost[..., 0] > 0).to(dt)
    l_self_sw = (l_base + cap_block * raises_atk * intim) * (1.0 - hp_gate) * (1.0 - is_pf)
    l_sw = foe * (1.0 - pursuit) * l_sw_foe + pursuit * (1.0 - imm_a) + (1.0 - foe) * l_self_sw * (
        1.0 - kd("magiccoat") - is_db).clamp(min=0.0)
    # ---------------------------------------------------------------- p_resolve and its factors
    a_k = p_act[:, :, None] * act_k                                                   # [B,4,K]
    db_up = our_vol[:, VOL["destinybond"]][:, None, None]
    a_k_db = (1.0 - pre) * p_act[:, :, None] + pre * db_up
    a_k = a_k * (1.0 - is_db[:, :, None]) + a_k_db * is_db[:, :, None]
    # Sleep Talk while awake: a faster sleep move from seat k puts us to sleep first, and then it works
    a_k = a_k + (st_talk * (1.0 - asleep[:, None]))[:, :, None] * pre * (st_a[..., 3] * (1.0 - ko_k))[:, None, :]
    stay = (alpha[:, None, :] * a_k * l_k).sum(-1) + a_un * p_act * l_un
    p_resolve = stay + a_sw * p_act * l_sw
    lands_stay = ((alpha[:, None, :] * l_k).sum(-1) + a_un * l_un) / (
        alpha.sum(-1, keepdim=True) + a_un).clamp(min=_EPS)
    p_ko_first = (alpha[:, None, :] * pre * ko_k[:, None, :]).sum(-1)
    p_ko_us = (alpha * ko_k).sum(-1, keepdim=True).expand(-1, 4)
    # ---------------------------------------------------------------- the consolidated facts
    is_dmg_seat = seat_dmg
    e_c = (alpha * cat_ok_c * is_dmg_seat * high_k).sum(-1, keepdim=True)
    e_m = (alpha * cat_ok_m * is_dmg_seat * high_k).sum(-1, keepdim=True)
    first_us = (alpha[:, None, :] * (1.0 - pre)).sum(-1) + a_un * (1.0 - pre_un)
    p_flinch = o.sec_flinch * (1.0 - imm_a) * p_act * first_us
    oc = o.out_cells
    out_high_a = oc[ar, :, oa, 1]                                                     # [B,4]
    pko_a = oc[ar, :, oa, 3]
    is_pursuit = pursuit
    is_prot = kd("protect")
    is_status_seat = o.seat_flag[..., FLAG_IDX["status_cat"]].to(dt)
    e_dmg_avoided = (alpha * high_k).sum(-1, keepdim=True)
    e_status_avoided = (alpha * is_status_seat).sum(-1, keepdim=True)
    attack_mass = (alpha * is_dmg_seat).sum(-1, keepdim=True)
    is_boom = kd("boom")
    pko_arrival = (oc[..., 3] * beta[:, None, :]).sum(-1)
    e_high_sw = (oc[..., 1] * beta[:, None, :]).sum(-1)
    e_mult_sw = (oc[..., 4] * beta[:, None, :]).sum(-1)
    e_burn = (alpha * o.d_burn_k.to(dt)).sum(-1, keepdim=True) * o.is_brn
    e_slp = (alpha * o.d_slp_k.to(dt)).sum(-1, keepdim=True) * o.is_slp
    row_a = torch.einsum("bk,bkf->bf", alpha, pin_a[..., :12]) * o.pair_gate[ar, o.our_active]   # [B,12]
    row_a = torch.cat([row_a[:, :6], row_a[:, 6:12] * corr_a], dim=-1)
    cols = [
        p_resolve, lands_stay, l_sw, p_ko_first, p_act,
        p_ko_us, is_db * p_ko_us,
        is_counter * e_c, is_mc * e_m, p_flinch, is_pursuit * a_sw, is_pursuit * a_sw * out_high_a,
        is_prot * e_dmg_avoided, is_prot * e_status_avoided, is_prot * attack_mass,
        is_prot * attack_mass * o.our_protect_odds[:, None],
        is_boom * ((1.0 - a_sw) * pko_a + a_sw * pko_arrival),
        e_high_sw, (oc[..., 3] * beta[:, None, :]).sum(-1), e_mult_sw, a_sw.expand(-1, 4),
        o.c2_base[..., 1], e_burn, o.c2_base[..., 2], e_slp, o.c2_base[..., 3],
    ] + [row_a[:, i:i + 1].expand(-1, 4) for i in range(12)]
    valid = (o.req_ids > 0).to(dt)                                                    # an empty request slot reads 0
    return torch.stack(cols, dim=-1) * (o.gate.to(dt) * valid)[:, :, None]            # [B,4,RAW]


def switch_facts(o: MoveResolutionOps, seat_spin: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``[B,6,len(MOVE_RESOLUTION_SWITCH_COORDS)]`` — the switch-cell facts, per OUR mon j."""
    dt = o.alpha.dtype
    B, K = o.alpha.shape
    ar = torch.arange(B, device=o.alpha.device)
    alpha = o.alpha.to(dt)
    gate = o.pair_gate.to(dt)                                                         # [B,6,1]
    corr = incoming_status_correction(o)                                              # [B,6,6]
    rows = torch.einsum("bk,bjkf->bjf", alpha, o.pair_in[..., :12].to(dt)) * gate    # [B,6,12]
    rows = torch.cat([rows[..., :6], rows[..., 6:] * corr], dim=-1)
    spin = o.seat_kind[..., SEAT_KIND_IDX["rapidspin"]].to(dt) if seat_spin is None else seat_spin
    a_spin = (alpha * spin).sum(-1, keepdim=True)                                     # [B,1]
    p_spin_denied = o.our_is_ghost.to(dt) * a_spin                                    # [B,6]
    e_pko = torch.einsum("bk,bjk->bj", alpha, o.pair_in[..., 3].to(dt))              # acc counted ONCE
    e_type = torch.einsum("bk,bjk->bj", alpha, o.pair_type_mult.to(dt))
    hp = o.our_hp_all.to(dt)
    m_high = torch.einsum("bk,bjk->bj", alpha, o.pair_in[..., 1].to(dt)) - hp
    m_crit = torch.einsum("bk,bjk->bj", alpha, o.pair_in[..., 2].to(dt)) - hp
    pin_a = o.pair_in[ar, o.our_active]                                               # [B,K,14]
    h2 = PURSUIT_SWITCH_MULT * pin_a[..., 1]
    ko2 = torch.clamp((h2 - o.our_hp[:, None]) / (ROLL_WINDOW * h2 + _EPS), 0.0, 1.0)
    pur = o.seat_kind[..., SEAT_KIND_IDX["pursuit"]].to(dt)
    p_sw = 1.0 - (alpha * pur * ko2).sum(-1, keepdim=True)                            # [B,1]
    per = torch.stack([p_spin_denied, e_pko, e_type, m_high, m_crit], dim=-1) * gate  # [B,6,5]
    return torch.cat([rows, per, (p_sw * gate[..., 0])[..., None]], dim=-1)          # [B,6,RAW]


# =========================================================================================== the MODULE
class MoveResolutionCell(torch.nn.Module):
    """gen3_move_resolution_v1 — `MoveResolutionOps` → the extra pointer MOVE-cell block ``[B,4,out_m]`` and the
    extra SWITCH-cell block ``[B,6,out_s]``. Both projections are zero-init `IsolatedLinear`s (no global RNG draw,
    skipped by SB3's orthogonal re-init), so ON-at-init contributes exactly 0 to every logit."""

    KIND: torch.Tensor
    FLAG: torch.Tensor
    SEAT_KIND: torch.Tensor
    SPECIES_CHART0: torch.Tensor
    SPECIES_ABL_IMM: torch.Tensor
    SPECIES_P_IMM: torch.Tensor
    SPECIES_HAS_TYPE: torch.Tensor
    ABILITY_IMM: torch.Tensor
    ABILITY_NAMED: torch.Tensor
    SPECIES_NAMED_PRIOR: torch.Tensor

    def __init__(self, damage_op: Any, out_move: int = MOVE_RESOLUTION_MOVE_DIM,
                 out_switch: int = MOVE_RESOLUTION_SWITCH_DIM) -> None:
        super().__init__()
        from agents.model.hypothesis_set import IsolatedLinear
        n_moves = int(damage_op.MOVE_BP.shape[0])
        for name, t in build_move_tables(n_moves).items():
            self.register_buffer(name, t, persistent=False)
        for name, t in build_species_tables(damage_op.CHART.detach().cpu(), damage_op.ABILITY_DAMAGE_MULT.detach().cpu(),
                                            damage_op.SPECIES_TYPE.detach().cpu()).items():
            self.register_buffer(name, t, persistent=False)
        self.move_proj = IsolatedLinear(_MOVE_RESOLUTION_MOVE_RAW, int(out_move), zero=True)
        self.switch_proj = IsolatedLinear(_MOVE_RESOLUTION_SWITCH_RAW, int(out_switch), zero=True)

    def raw(self, ops: MoveResolutionOps) -> Tuple[torch.Tensor, torch.Tensor]:
        """The un-projected facts ``([B,4,RAW_M], [B,6,RAW_S])`` (what the tests and the fuzz read)."""
        return move_facts(ops, self.KIND, self.FLAG), switch_facts(ops)

    def forward(self, ops: MoveResolutionOps) -> Tuple[torch.Tensor, torch.Tensor]:
        m, s = self.raw(ops)
        return self.move_proj(m), self.switch_proj(s)


# =========================================================================================== the GATHER
def gather_ops(fe: Any, ctx: Any, alpha_logits: Optional[torch.Tensor], beta_logits: Optional[torch.Tensor],
               imc_ops: Optional[Tuple[torch.Tensor, ...]]) -> MoveResolutionOps:
    """Build `MoveResolutionOps` from one extractor forward, at the pointer stash (T2: α / β exist; every op stash
    was written at T1). Fails loud when a stash is missing — a silent zero is indistinguishable from a null result."""
    from agents.model.damage_kinds import gather_bp, is_priced, typeless_move_type
    from agents.model.damage_op_layout import _BOOSTS_DIM
    from agents.model.pair_outcome import pair_alpha_full
    from agents.observation.constants import (GLOBAL_ENV_DIM, POKEMON_CONDITION_OFFSET, POKEMON_PROTECT_OFFSET,
                                              POKEMON_SLEEP_BELIEF_OFFSET, POKEMON_ABILITIES_OFFSET, TEAM_SIZE)
    cell: MoveResolutionCell = fe.move_resolution_cell
    op = fe.damage_op
    stash = op.stash
    if (alpha_logits is None or beta_logits is None or imc_ops is None or op.last_pair_in is None
            or op.last_pair_seat_live is None or op.last_pair_type_mult is None or op.last_out_cells is None
            or op.last_topk_idx is None or op.last_pair_gate is None or stash.opp_species_post is None
            or op.last_raw_block is None):
        raise RuntimeError(
            "move_resolution is on but α / β or an op stash is missing — the family would silently contribute "
            "nothing, which is indistinguishable from a null RESULT. Requires opp_intent + damage_op + "
            "damage_outgoing + damage_matrices_incoming / outgoing + damage_topk_k > 0.")
    B = ctx.batch_size
    dev = ctx.device
    ar = torch.arange(B, device=dev)
    our_act = ctx.our_active_idx
    opp_act = ctx.opp_active_local
    opp_g = TEAM_SIZE + opp_act
    pp = ctx.pokemon_part
    ours, opp = slice(0, TEAM_SIZE), slice(TEAM_SIZE, 2 * TEAM_SIZE)
    has_opp = ctx.hp_and_active[:, opp, -1].any(dim=1).float()
    our_alive_a = (ctx.hp_and_active[ar, our_act, 0] > 0).float()
    gate = (has_opp * our_alive_a)[:, None]
    # --- our request moves
    ids = ctx.our_active_req_move_ids
    req_type = typeless_move_type(op, ids, ctx.our_active_req_move_type_ids)
    our_hp = ctx.hp_and_active[ar, our_act, 0]
    bp = torch.where(ids == op.hp_num, torch.full_like(ids, 1, dtype=torch.float32) * op.hp_bp, op.MOVE_BP[ids])
    bp = gather_bp(op, ids, our_hp[:, None], bp=bp)
    is_dmg = is_priced(op, ids, bp)
    acc = op.MOVE_ACCURACY[ids]
    our_types = torch.stack([ctx.type1_ids[:, ours], ctx.type2_ids[:, ours]], dim=-1)    # [B,6,2]
    our_is_ghost = ((our_types[..., 0] == T_GHOST) | (our_types[..., 1] == T_GHOST)).float()
    sb = op.MOVE_SELF_BOOSTS[ids].clone()                                                 # [B,4,5]
    is_curse = cell.KIND[ids][..., KIND_IDX["curse"]]
    nonghost = (1.0 - our_is_ghost[ar, our_act])[:, None, None]
    sb = sb + is_curse[..., None] * nonghost * op.CURSE_BOOSTS[None, None, :]
    our_abl = ctx.ability1_ids[ar, our_act]
    opp_abl = ctx.ability1_ids[ar, opp_g]
    from agents.model.damage_tables import SECONDARY_FLINCH_IDX
    sec_flinch = (op.MOVE_SECONDARY[ids][..., SECONDARY_FLINCH_IDX] * acc
                  * op.ABILITY_SECONDARY_MULT[our_abl][:, None] * op.ABILITY_SECONDARY_BLOCK[opp_abl][:, None]
                  ).clamp(max=1.0)
    # --- both sides' per-mon state
    cond = pp[..., POKEMON_CONDITION_OFFSET:POKEMON_CONDITION_OFFSET + 7]                  # [B,12,7]
    rest = pp[..., POKEMON_SLEEP_BELIEF_OFFSET]
    alive_all = (ctx.hp_and_active[..., 0] > 0).float()
    opp_alive = alive_all[:, opp] * (~ctx.opp_believed_mask).float() + ctx.opp_believed_mask.float()
    vol_our = ctx.our_ctx_raw[:, _BOOSTS_DIM:]
    vol_opp = ctx.opp_ctx_raw[:, _BOOSTS_DIM:]
    stages = torch.stack(op._boost_stages(ctx.our_ctx_raw), dim=-1)                      # [B,5]
    named_our = cell.ABILITY_NAMED[our_abl]                                               # [B,n] (ours: revealed)
    # --- their per-slot immunity / type / ability marginals (revealed exact; the species posterior where not)
    post = stash.opp_species_post
    post = post[:, None, :] if post.dim() == 2 else post                                  # [B,1|6,S]
    believed = ctx.opp_believed_mask[:, :, None]                                          # [B,6,1]
    sp = ctx.species_ids[:, opp]                                                          # [B,6]
    t1, t2 = ctx.type1_ids[:, opp], ctx.type2_ids[:, opp]
    chart0_rev = ((op.CHART[t1] * op.CHART[t2]) == 0).float()                             # [B,6,19]
    a_rev = ctx.ability1_ids[:, opp]
    # REVEALED is the ability block's `known` flag — NOT `id > 0`: an unrevealed opponent's slot carries its
    # species' most likely ability in id1 (`[top1, top2, dominance, known=0]`, `observation/abilities.py`), so
    # reading `id > 0` as "revealed" would assert Snorlax's Immunity on a Thick Fat Snorlax (a fuzz-found GIGO
    # the production op still carries — see the module docstring).
    known_all = pp[..., POKEMON_ABILITIES_OFFSET + 3]                                    # [B,12]
    abl_known = known_all[:, opp][:, :, None]
    abl_imm_rev = abl_known * cell.ABILITY_IMM[a_rev] + (1.0 - abl_known) * cell.SPECIES_ABL_IMM[sp]
    imm_rev = 1.0 - (1.0 - chart0_rev) * (1.0 - abl_imm_rev)
    imm_post = torch.matmul(post, cell.SPECIES_P_IMM).expand(-1, TEAM_SIZE, -1)
    chart0_post = torch.matmul(post, cell.SPECIES_CHART0).expand(-1, TEAM_SIZE, -1)
    type_rev = cell.SPECIES_HAS_TYPE.new_zeros(B, TEAM_SIZE, 19)
    type_rev = type_rev.scatter(2, t1[..., None], 1.0).scatter(2, t2[..., None], 1.0)
    type_rev[..., 0] = 0.0
    type_post = torch.matmul(post, cell.SPECIES_HAS_TYPE).expand(-1, TEAM_SIZE, -1)
    blk_rev = abl_known * op.ABILITY_STATUS_BLOCK[a_rev] + (1.0 - abl_known) * op.SPECIES_STATUS_BLOCK_PRIOR[sp]
    blk_post = torch.matmul(post, op.SPECIES_STATUS_BLOCK_PRIOR).expand(-1, TEAM_SIZE, -1)
    named_rev = abl_known * cell.ABILITY_NAMED[a_rev] + (1.0 - abl_known) * cell.SPECIES_NAMED_PRIOR[sp]
    named_post = torch.matmul(post, cell.SPECIES_NAMED_PRIOR).expand(-1, TEAM_SIZE, -1)
    imm_dmg = torch.where(believed, imm_post, imm_rev)
    chart0 = torch.where(believed, chart0_post, chart0_rev)
    p_type = torch.where(believed, type_post, type_rev)
    abl_block = torch.where(believed, blk_post, blk_rev)
    opp_named = torch.where(believed, named_post, named_rev)
    # --- the opponent model (stop-grad publications)
    seat_live = op.last_pair_seat_live
    alpha, a_sw, _a_stay = pair_alpha_full(alpha_logits, seat_live)
    has_cand = torch.isfinite(beta_logits).any(-1, keepdim=True).to(alpha.dtype)
    beta = torch.softmax(beta_logits.detach().float().clamp(min=-1e9), dim=-1).to(alpha.dtype) * has_cand
    nums = op.last_topk_idx
    seat_phys = op.MOVE_PHYS[nums]
    seat_flinch = (op.MOVE_SECONDARY[nums][..., SECONDARY_FLINCH_IDX]
                   * op.ABILITY_SECONDARY_MULT[opp_abl][:, None])
    base, d_burn_k, d_slp_k, is_brn, is_slp = imc_ops
    # Every op value the family reads is PRE-gain: the op's learned `out_gain` is one scalar per block channel —
    # per REQUEST SLOT for the per-move outgoing channels, so a post-gain read would scale the same move by where
    # it is listed (trained X5 arms learned 1.365 vs 1.146 on the KO channel of slots 0 / 3). `out_cells`,
    # `pair_cells` / `pair_in` are pre-gain stashes; P(outspeed) is read from the pre-gain block here.
    ot = op.tensors_from_block(op.last_raw_block)
    p_out = ot.out_p_outspeed
    p_out = p_out if p_out.dim() == 2 else p_out[:, None]
    opp_fainted_rev = ((1.0 - alive_all[:, opp]) * (~ctx.opp_believed_mask).float()).sum(-1)
    return MoveResolutionOps(
        gate=gate, req_ids=ids, req_type=req_type, is_dmg=is_dmg, acc=acc, prio=op.MOVE_PRIORITY[ids],
        inflicts=op.MOVE_INFLICTS_STATUS[ids],
        p_land=op._status_landing(dataclasses.replace(
            ctx, ability1_ids=ctx.ability1_ids * (known_all > 0.5).long()))[:, :4],
        st_type_imm=op.MOVE_STATUS_TYPE_IMMUNE[ids], st_cat=op.MOVE_STATUS_CAT[ids].long(),
        st_blocked=op.MOVE_BLOCKED_IF_STATUSED[ids], self_boost=sb, sec_flinch=sec_flinch,
        our_stage=stages, our_hp=our_hp, our_cond=cond[:, ours], our_alive=alive_all[:, ours],
        our_rest=rest[:, ours], our_active=our_act, our_p_wake=pp[ar, our_act, POKEMON_SLEEP_BELIEF_OFFSET + 1],
        our_vol=vol_our, our_protect_odds=pp[ar, our_act, POKEMON_PROTECT_OFFSET], our_named_abl=named_our,
        our_is_ghost=our_is_ghost, our_hp_all=ctx.hp_and_active[:, ours, 0],
        beatup_n=_beatup_count(op, ctx), our_wish=ctx.non_matchup_rest[:, GLOBAL_ENV_DIM + 3],
        opp_active=opp_act, opp_vol=vol_opp, opp_hp=ctx.hp_and_active[ar, opp_g, 0],
        opp_protect_odds=pp[ar, opp_g, POKEMON_PROTECT_OFFSET], opp_last_move=ctx.last_move_ids[ar, opp_g],
        opp_p_wake=pp[ar, opp_g, POKEMON_SLEEP_BELIEF_OFFSET + 1],
        opp_alive_total=TEAM_SIZE - opp_fainted_rev, opp_cond=cond[:, opp], opp_rest=rest[:, opp],
        opp_alive=opp_alive, imm_dmg=imm_dmg, chart0=chart0, p_type=p_type, abl_block=abl_block,
        opp_named_abl=opp_named, screens=ctx.screen_feature, weather=ctx.weather_feature,
        spikes=ctx.spikes_feature,
        alpha=alpha, a_switch=a_sw, beta=beta, seat_nums=nums, seat_prio=op.MOVE_PRIORITY[nums],
        seat_kind=cell.SEAT_KIND[nums], seat_flag=cell.FLAG[nums], seat_phys=seat_phys, seat_flinch=seat_flinch,
        pair_in=op.last_pair_in, pair_gate=op.last_pair_gate, pair_type_mult=op.last_pair_type_mult,
        p_out=p_out, out_cells=op.last_out_cells, c2_base=base, d_burn_k=d_burn_k, d_slp_k=d_slp_k,
        is_brn=is_brn, is_slp=is_slp)


def _beatup_count(op: Any, ctx: Any) -> torch.Tensor:
    from agents.model.damage_kinds import beatup_party_ours
    return beatup_party_ours(op, ctx)[1]
