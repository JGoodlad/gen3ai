"""gen3_nonformula_damage_v1 — the DamageOperator's damage model for every move whose damage is
NOT the plain gen-3 base-power formula, applied identically by every kernel (outgoing AND incoming).

The tables are built and DECLARED in `damage_tables.DAMAGE_MODELS` (one row per such move, with its
Showdown citation); this module is the one place the kernels turn those tables into damage. Two
pure functions, no parameters, no state:

``effective_bp``
    `MOVE_BP` is the move's base power AT THE ATTACKER'S FULL HP. Two kinds depend on the
    attacker's current HP fraction ``f``:

    * ``bp_hp_scaled`` (Eruption / Water Spout): ``BP = 150 · f`` (min 1) —
      `data/moves.ts` eruption ``basePowerCallback``: ``move.basePower * pokemon.hp / pokemon.maxhp``.
    * ``bp_flail`` (Flail / Reversal): the gen-3 step table on
      ``ratio = max(floor(48 · f), 1)`` — `data/mods/gen3/moves.ts` flail / reversal
      ``basePowerCallback`` (200 / 150 / 100 / 80 / 40 / 20 at ratio < 2 / 5 / 10 / 17 / 33 / else).
      A ``1e-4`` nudge inside the floor makes an exact quotient land on its integer: with maxhp ≤ 714
      a non-integer ``48·hp/maxhp`` sits ≥ 1/714 from the next integer, so the nudge never moves a
      genuine value across a step (standing rule 8 — no decision is made within a rounding error).

``nonformula_rolls``
    The damage IN HP of the kinds the BP formula cannot express, each exact given the HP the op
    already carries (Showdown `sim/battle-actions.ts` ``getDamage``: immunity first, then ``ohko`` →
    ``target.hp`` in gen 3, then ``damageCallback``, then ``damage: 'level' | N``):

    * ``fixed`` — a constant: Seismic Toss / Night Shade = the user's level (100), Dragon Rage 40,
      Sonic Boom 20, Psywave its EXPECTATION 100 at level 100 (declared approximation).
    * ``target_hp_frac`` — a fraction of the TARGET's CURRENT HP: Super Fang ½; the OHKO moves 1
      (gen 3 OHKO deals ``target.hp``).
    * ``endeavor`` — ``max(0, target_hp − attacker_hp)`` (the gen-4 mod's ``onTry`` fails it when
      the user is not lower).

    They ignore Atk/Def, the roll, crits, screens and weather, and RESPECT type / ability immunity
    (``eff > 0``) — so all three rolls are one value and P(KO) is ``acc · [the hit KOs]``.

``beatup_*`` (gen3_beatup_exact_v1) — gen-3 Beat Up, the ONE kind that is still the BP formula, with the
    formula's inputs swapped (so it rides the kernels' own roll / screen / crit / P(KO) arithmetic, not
    `nonformula_rolls`). `data/mods/gen3/moves.ts` beatup: the move is TYPELESS (``'???'`` — no STAB, no
    effectiveness, no ability read) and SPECIAL (Light Screen halves it; Reflect and burn do not); it hits
    once per ally in ``side.pokemon.filter(a => !a.fainted && !a.status)`` (the USER counts iff healthy);
    hit *i* is the ordinary formula at BP 10 with ``A`` = ally *i*'s species BASE Atk and ``D`` = the
    TARGET's species BASE Def (``event.modifier = 1``: no boost, item or ability reaches either stat). Every
    hit's pre-roll damage is ``(42/50)·10·A_i/D + 2``, so the sum over the party is

        core = (42/50) · 10 · S / D_base + 2 · N       with  S = Σ ally base Atk,  N = ally count

    — `S` and `N` are all a kernel needs of the attacking side, and `D_base` all it needs of the target.
    The kernels then multiply by the same ``0.925`` mean roll, apply the same screen, crit ×2 and KO ramp
    as for any other move (a per-hit roll is independent in the sim; the op's one shared roll over the
    summed damage over-states the spread of the KO ramp by ≈ √N — the P(KO) edges are exact, the middle is
    smoother than the sim's).

    *Our side* is fully known (our party's HP + status are in the obs). *The opponent's side*: a revealed
    mon is eligible iff alive with no status; **an unrevealed slot is eligible with certainty** (a mon that
    never entered the battle cannot have fainted or been statused) and contributes its EXPECTED base Atk
    under the Species-Clause-filtered usage prior (`DamageOperator.unrevealed_species_probs` — the op's one
    convention for a hidden mon). The expectation is exact for the sum (linearity), so E[damage] is exact
    given those marginals; X5's U3 revisits hidden-mon handling. An unrevealed DEFENDER uses
    ``D_eff = 1 / E[1 / D_base]`` (the harmonic mean), which makes the same formula return E[damage]
    exactly. Declared limits: a forme reads its base species' stats (the op's num-keyed convention); Beat
    Up into a Substitute and the per-hit crit roll are not modelled.
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from agents.model.damage_op_layout import _DMG_CHIP_CAP, _DMG_CRIT_CAP
from agents.observation.constants import CONDITION_DIM, POKEMON_CONDITION_OFFSET, TEAM_SIZE
from agents.observation.types import TypeEncoder

#: The TypeEncoder index of '???' — the type Beat Up has in battle (neutral row/column of every table).
TYPELESS_TYPE_IDX = TypeEncoder.TYPE_TO_IDX["???"]
# Condition one-hot [None, BRN, PAR, SLP, FRZ, PSN, TOX] — columns 1.. are the six MAJOR statuses
# (`ally.status` in Showdown); the all-zero placeholder an unrevealed slot carries reads "no status".
_STATUS_COLS = slice(POKEMON_CONDITION_OFFSET + 1, POKEMON_CONDITION_OFFSET + CONDITION_DIM)
_BS_ATK_COL, _BS_DEF_COL = 1, 2             # BASE_STATS columns [hp, atk, def, spa, spd, spe]

# Flail / Reversal (gen 3): ratio thresholds and the BP each band takes (`data/mods/gen3/moves.ts`).
_FLAIL_RATIO_STEPS = (2.0, 5.0, 10.0, 17.0, 33.0)
_FLAIL_BP = (200.0, 150.0, 100.0, 80.0, 40.0, 20.0)
_FLAIL_FLOOR_NUDGE = 1e-4


def flail_bp(hp_frac: torch.Tensor) -> torch.Tensor:
    """Gen-3 Flail / Reversal base power from the attacker's current HP fraction."""
    ratio = torch.floor(hp_frac * 48.0 + _FLAIL_FLOOR_NUDGE).clamp(min=1.0)
    bp = torch.full_like(hp_frac, _FLAIL_BP[-1])
    # Walk the bands from the widest down so the narrowest (lowest HP) band wins.
    for thr, val in reversed(list(zip(_FLAIL_RATIO_STEPS, _FLAIL_BP[:-1]))):
        bp = torch.where(ratio < thr, torch.full_like(bp, val), bp)
    return bp


def effective_bp(bp: torch.Tensor, hp_scaled: torch.Tensor, flail: torch.Tensor,
                 attacker_hp_frac: torch.Tensor) -> torch.Tensor:
    """The BP the formula should use: `bp` (= MOVE_BP, the BP at full HP) with the two
    attacker-HP-dependent kinds resolved. `hp_scaled` / `flail` are the gathered 0/1 tables;
    all four BROADCAST (the caller passes the attacker axis with a 1-wide move axis)."""
    bp, hp_scaled, flail, f = torch.broadcast_tensors(bp, hp_scaled, flail, attacker_hp_frac)
    out = torch.where(hp_scaled > 0, (bp * f).clamp(min=1.0), bp)
    return torch.where(flail > 0, flail_bp(f), out)


def nonformula_rolls(nf_tabs: Tuple[torch.Tensor, ...],
                     tgt_cur_hp: torch.Tensor, tgt_maxhp: torch.Tensor, atk_cur_hp: torch.Tensor,
                     eff: torch.Tensor, acc: torch.Tensor,
                     eps: float = 1e-6) -> Tuple[torch.Tensor, ...]:
    """→ ``(is_nf, high, crit, ko)`` broadcast to a common shape. ``is_nf`` marks the cells whose
    rolls must be REPLACED; ``high`` (== low: no roll) and ``crit`` (no crit) are the damage as a
    fraction of the target's max HP, clamped like the formula rolls; ``ko = acc · [the hit KOs a live
    target]`` (an OHKO always; a fixed amount iff ≥ the remaining HP; Endeavor / Super Fang never). Every
    output is 0 where the target is immune (``eff <= 0``). ``nf_tabs`` = ``(fixed, target_frac,
    endeavor)`` as `gather_nonformula` returns them (each broadcastable to the output)."""
    fixed, target_frac, endeavor = nf_tabs
    is_nf = (fixed + target_frac + endeavor) > 0
    dmg = fixed + target_frac * tgt_cur_hp + endeavor * (tgt_cur_hp - atk_cur_hp).clamp(min=0.0)
    not_immune = (eff > 0).float()
    frac = dmg / (tgt_maxhp + eps)
    high = frac.clamp(max=_DMG_CHIP_CAP) * not_immune
    crit = frac.clamp(max=_DMG_CRIT_CAP) * not_immune
    # KO: a full-current-HP kind (OHKO, target_frac 1) KOs any live target BY CONSTRUCTION — tested on
    # the table, never as `dmg >= hp` (an exact equality is not a near-tie to be margined); a fixed kind
    # KOs iff it covers the remaining HP (the one genuine threshold — a declared MARGIN site); Endeavor
    # leaves the target at the attacker's HP and Super Fang at half, so neither KOs. APPROXIMATION:
    # Super Fang's 1-HP floor (clampIntRange(hp/2, 1) KOs a 1-HP target) is not modelled.
    alive = (tgt_cur_hp > 0).float()
    full_hp_kind = target_frac >= 1.0                       # (one comparison per line: the
    covers = fixed >= tgt_cur_hp                            #  selection-site recorder reads lines)
    kills = full_hp_kind | covers
    ko = acc * kills.float() * alive * not_immune
    return is_nf, high, crit, ko


def gather_bp(op: Any, idx: torch.Tensor, attacker_hp_frac: torch.Tensor,
              bp: Optional[torch.Tensor] = None) -> torch.Tensor:
    """`effective_bp` over the op's tables at move nums `idx`. `bp` overrides the gathered MOVE_BP
    (the outgoing kernels' Hidden Power → 70 substitution); `attacker_hp_frac` broadcasts against
    `idx` (same rank, a 1-wide move axis)."""
    base: torch.Tensor = op.MOVE_BP[idx] if bp is None else bp
    return effective_bp(base, op.MOVE_BP_HP_SCALED[idx], op.MOVE_BP_FLAIL[idx], attacker_hp_frac)


def gather_nonformula(op: Any, idx: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(fixed, target_frac, endeavor)`` at move nums `idx` — the `nonformula_rolls` inputs."""
    return op.MOVE_FIXED_DAMAGE[idx], op.MOVE_TARGET_HP_FRAC[idx], op.MOVE_ENDEAVOR[idx]


def is_priced(op: Any, idx: torch.Tensor, bp: torch.Tensor) -> torch.Tensor:
    """0/1: the op has a damage model for the move — a positive formula BP OR a non-formula kind.
    The `unmodelled` rows of `DAMAGE_MODELS` (and every Status move) read 0, by declaration."""
    nonformula: torch.Tensor = op.MOVE_NONFORMULA[idx]
    return ((bp > 0) | (nonformula > 0)).float()


def override_rolls(rolls: Tuple[torch.Tensor, ...], nf: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, ...]:
    """Replace the formula ``(high, low, crit, ko)`` with `nonformula_rolls`'s ``(is_nf, high, crit, ko)``
    at the non-formula cells (low == high: no roll)."""
    is_nf, nf_high, nf_crit, nf_ko = nf
    high, low, crit, ko = rolls
    return (torch.where(is_nf, nf_high, high), torch.where(is_nf, nf_high, low),
            torch.where(is_nf, nf_crit, crit), torch.where(is_nf, nf_ko, ko))


# --------------------------------------------------------------------------- gen3_beatup_exact_v1
def gather_beatup(op: Any, idx: torch.Tensor) -> torch.Tensor:
    """0/1 ``MOVE_BEATUP`` at move nums `idx` — the cells whose formula inputs `beatup_swap` replaces."""
    flag: torch.Tensor = op.MOVE_BEATUP[idx]
    return flag


def typeless_move_type(op: Any, move_ids: torch.Tensor, move_ty: torch.Tensor) -> torch.Tensor:
    """OUR move's TypeEncoder index as the battle resolves it: '???' for Beat Up, else `move_ty` (the obs'
    resolved type — which is the DEX type, Dark, for Beat Up). The incoming kernels get the same fact from
    the `MOVE_TYPE_IDX` row `build_damage_buffers` writes; the outgoing kernels read the obs type, so they
    route it through here. → STAB, the chart, the ability multipliers and the weather / sport modifiers all
    read neutral."""
    return torch.where(op.MOVE_BEATUP[move_ids] > 0, torch.full_like(move_ty, TYPELESS_TYPE_IDX), move_ty)


def beatup_base_def(op: Any, species_ids: torch.Tensor) -> torch.Tensor:
    """The TARGET's species BASE Def (what Beat Up's ``onFoeModifySpD`` returns). ``clamp(min=1)`` keeps the
    sentinel species 0 (an empty / unrevealed slot — always gated downstream) off a 1/0; every real species
    has base Def ≥ 5, so the clamp never moves a real value."""
    d: torch.Tensor = op.BASE_STATS[species_ids][..., _BS_DEF_COL]
    return d.clamp(min=1.0)


def beatup_opp_target_def(op: Any, ctx: Any, sp_probs: torch.Tensor) -> torch.Tensor:
    """``[B,6]`` the base Def Beat Up meets at each of the six OPP slots. A revealed slot: that species'
    base Def. An UNREVEALED slot: ``1 / E[1 / D_base]`` under `sp_probs` (``unrevealed_species_probs`` —
    ``[B,S]`` prior or ``[B,6,S]`` override), the harmonic mean, so the kernel's ``… / D`` returns
    E[damage] exactly. The expectation runs over the SAME marginal the other expected-latent reads use."""
    revealed = beatup_base_def(op, ctx.species_ids[:, TEAM_SIZE:2 * TEAM_SIZE])              # [B,6]
    inv_all = 1.0 / op.BASE_STATS[:, _BS_DEF_COL].clamp(min=1.0)                             # [S]
    harmonic = 1.0 / (sp_probs @ inv_all).clamp(min=1e-12)                                   # [B] | [B,6]
    harmonic = harmonic[:, None] if harmonic.dim() == 1 else harmonic                        # [B,1] | [B,6]
    return torch.where(ctx.opp_believed_mask, harmonic, revealed)


def _healthy(ctx: Any, sl: slice) -> torch.Tensor:
    """``[B,6]`` 0/1: alive (HP fraction > 0) AND no major status — Beat Up's ally filter
    ``!ally.fainted && !ally.status``. (An unrevealed opp slot encodes HP 0: the caller handles it.)"""
    alive = (ctx.hp_and_active[:, sl, 0] > 0).float()
    status = ctx.pokemon_part[:, sl, _STATUS_COLS].sum(dim=-1)
    ok: torch.Tensor = alive * (status < 0.5).float()
    return ok


def beatup_party_ours(op: Any, ctx: Any) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(S, N)`` ``[B]`` each for OUR party as Beat Up's attacker: ``S`` = Σ base Atk over our healthy
    mons, ``N`` = how many (the user included iff healthy — it is one of the six). Fully known."""
    ours = slice(0, TEAM_SIZE)
    ok = _healthy(ctx, ours)                                                      # [B,6]
    atk = op.BASE_STATS[ctx.species_ids[:, ours]][..., _BS_ATK_COL]               # [B,6]
    return (ok * atk).sum(dim=-1), ok.sum(dim=-1)


def beatup_party_opp(op: Any, ctx: Any, species_probs: Optional[torch.Tensor] = None
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(S, N)`` ``[B]`` each for the OPPONENT's party as Beat Up's attacker. A REVEALED mon counts iff alive
    with no status; every UNREVEALED slot (`ctx.opp_believed_mask`) counts with certainty — a mon that never
    entered the battle cannot have fainted or been statused — at its EXPECTED base Atk under the op's one
    hidden-mon belief: `unrevealed_species_probs(ctx, species_probs)` — the Species-Clause usage prior, or the
    T0 species belief the extractor hands every pricing site (``[B,S]`` team-level, or a per-slot
    ``[B,6,S]``). The sum's expectation is exact (linear in the per-slot marginals)."""
    opp = slice(TEAM_SIZE, 2 * TEAM_SIZE)
    hidden = ctx.opp_believed_mask.float()                                        # [B,6]
    ok = _healthy(ctx, opp) * (1.0 - hidden)                                      # [B,6] revealed + eligible
    atk = op.BASE_STATS[ctx.species_ids[:, opp]][..., _BS_ATK_COL]                # [B,6]
    e_atk = op.unrevealed_species_probs(ctx, species_probs) @ op.BASE_STATS[:, _BS_ATK_COL]   # [B] | [B,6]
    hidden_atk = hidden.sum(dim=-1) * e_atk if e_atk.dim() == 1 else (hidden * e_atk).sum(dim=-1)   # [B]
    return (ok * atk).sum(dim=-1) + hidden_atk, ok.sum(dim=-1) + hidden.sum(dim=-1)


def beatup_swap(is_bu: torch.Tensor, A: torch.Tensor, D: torch.Tensor, atk_sum: torch.Tensor,
                defence: torch.Tensor, hits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The kernels' ``core = 42·bp·A/(D+eps)/50 + plus2`` inputs, with Beat Up's cells swapped: ``A`` → the
    party's Σ base Atk, ``D`` → the target's base Def, ``plus2`` → ``2·hits`` (the formula's ``+2`` is per
    hit). Every other cell keeps ``(A, D, 2.0)`` EXACTLY (a `where`, no arithmetic), so a table with no Beat
    Up in it is bit-identical to the kernel before. All args broadcast to `A` / `D`'s shape."""
    bu = is_bu > 0
    return (torch.where(bu, atk_sum, A), torch.where(bu, defence, D), torch.where(bu, 2.0 * hits, 2.0))
