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
"""
from __future__ import annotations

from typing import Any, Optional, Tuple

import torch

from agents.model.damage_op_layout import _DMG_CHIP_CAP, _DMG_CRIT_CAP

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
