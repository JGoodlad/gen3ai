"""THE EXACT P(KO) — `--ko-ramp exact` (`gen3_endstate_facts_v1`, config v152; `design_hand_computed_features.md`
D9 / §4 rank 3, an APPROXIMATE FACT made exact), computed in CLOSED FORM (`gen3_ko_exact_closed_v1`, 2026-10-10).

**What it replaces.** Every P(KO) the damage operator hands the model was the continuous ramp
``acc · clamp((dmg − hp) / (0.15 · dmg), 0, 1)``, with ``dmg`` the op's MEAN-roll damage (its kernels multiply the
gen-3 formula by ``0.925``, the mean of the 16 rolls). That is two approximations of one fact: it smooths the 16
discrete rolls into a line, and it ANCHORS the line at the mean roll (so a hit whose mean roll exactly equals the
target's HP reads P(KO) = 0, where the exact answer is 8 of 16 rolls) — and it omits the critical hit entirely.

**The rule (gen 3, VERIFIED in `deps/pokemon-showdown` at the pinned commit; `[Gen 3] OU` = the gen3 mod over gen4
over the base):**

* the damage ROLL: ``sim/battle.ts`` ``randomizer``: ``floor(floor(base · (100 − random(16))) / 100)`` — r uniform on
  the 16 integers 85 … 100, applied LAST (`data/mods/gen3/scripts.ts` ``modifyDamage``, after STAB and the type
  chart). So the 100-roll damage is the op's mean-roll damage ÷ 0.925, and roll r deals ``top · r / 100``;
* the CRITICAL HIT: ``sim/battle-actions.ts`` ``getDamage``: gen ≤ 5 ``critMult = [0, 16, 8, 4, 3, 2]`` indexed by
  ``move.critRatio`` (default 1, ``sim/dex-moves.ts``) → 1/16 for an ordinary move, 1/8 for a ``critRatio: 2`` move
  (Slash, Razor Leaf, Crabhammer, Karate Chop, Cross Chop, Aeroblast, Air Cutter, Blaze Kick, Leaf Blade, Poison Tail,
  Razor Wind, Sky Attack — `data/pokemon/gen3_moves.json` ``critRatio``). A crit DOUBLES the damage after the ``+2``
  (`modifyDamage`: ``baseDamage = modify(baseDamage, 2)``) and IGNORES Reflect / Light Screen (their
  ``ModifyDamagePhase1`` handlers skip a crit), so the crit's 100-roll damage is twice the PRE-screen top — the op's
  ``crit`` column (``2 · dmg_ns``) is exactly that, at the mean roll;
* the KO: damage ``≥`` the target's current HP faints it. Damage and HP are integers, the op's damage is a real
  number (no floors), and the TARGET's HP is known only to the observation's precision: **ours exactly** (an integer
  HP of an exact max HP — resolved over ±½ HP, below the op's own precision), **theirs as a percentage**
  (`data/rulesets.ts` 'Standard AG' carries 'HP Percentage Mod'; `sim/pokemon.ts` ``getHealth`` reports
  ``ceil(100 · hp / maxhp)`` (99 while not full), so a reported p < 100 % means the true fraction lies in
  (p − 1 %, p]; a reported 100 % is exactly full). Each roll's KO indicator is therefore the probability over
  that HP interval — a ramp across the interval, which is EXACT given what is observed.

    P(KO | hit) = (1 − c) · (1/16) Σ_r ramp(top · r/100)  +  c · (1/16) Σ_r ramp(top_crit · r/100)
    ramp(d)     = clamp((d − hp_lo) / (hp_hi − hp_lo), 0, 1)        (HP uniform on [hp_lo, hp_hi])

and every caller folds accuracy in as before (``acc · P(KO | hit)``; the two are independent events).

**Why the resolved interval, and not a staircase count of rolls.** A count of KOing rolls is a step function of the
damage estimate, and the damage estimate moves with the weights (the spread / species beliefs reach it): a STEP there is
a discrete threshold on a score, which K9(b)'s behaviour check (`selection_sites`, `tie_margins`) must exclude every row
of that sits within a rounding error of a step — with ~2,300 (defender, candidate) cells per row, most rows. Resolving
each step over the observed HP interval makes the function CONTINUOUS (piecewise linear) and DIFFERENTIABLE (its true
gradient, where gradients flow: the spread belief reaches the damage), and it is the exact probability given the
observation rather than a smoothing.

**THE CLOSED FORM (owner 2026-10-10: "optimise the 16 rolls, since it's a known formula with a crit chance"; then
"remove the old one").** Because the op's damage is continuous, the 16 terms ``(top · r/100 − hp_lo) / w`` are an
EXACT arithmetic sequence in r, and their clamped sum has an exact closed form (`roll_ko_prob`): the terms reading 0
and the terms reading 1 are COUNTED, the middle run is its count times its mean — O(1) per cell where the first build
summed 16 passes. It is the SAME real function as that sum, for our exact HP and their percentage bin alike; there is
no flooring error to quantify because neither spelling floors a roll (verified 2026-10-10 at the pinned submodule:
`sim/battle.ts` ``randomizer`` is ``tr(tr(base · (100 − random(16))) / 100)`` and `data/mods/gen3/scripts.ts`
``modifyDamage`` applies it after the type chart and the ``ModifyDamage`` event, then floors once with a minimum of 1 —
the per-roll floor and that minimum are named residuals below). The 16-roll sum survives as the REFERENCE ORACLE of
`ko_exact_test.py`, which derives and pins the rounding bound between the two (``8 eps (1 + (|top| + |hp_lo|) / w)``,
float64 and float32) and the gradients. The counts are a floor / ceil of a score, but the SUM is continuous across
each count's step (the term that changes class sits on the clamp's edge), so they are tie-safe — declared
``COUNT_CONTINUOUS`` in `selection_sites`, never a K9(b) margin. (A brief `--ko-ramp exact_closed` value carried this
form beside the sum on main at `6e0a1a7a`; it is RETIRED — `model_version/retired_levers.RETIRED_VALUES`.)

**Residuals (named, not modelled):** the attacker's Focus Energy (+2 crit stages), Scope Lens (+1), Lucky Punch /
Stick (+2) and the defender's Battle Armor / Shell Armor (no crit) — every site reads the move's own ratio only; a
crit's disregard of the attacker's negative / the defender's positive stat stages (the op keeps the stages for the
crit column, as before); the per-hit floors of the formula and its 1-HP minimum (the op's damage is continuous).

Torch-only leaf module (no extractor import): every op kernel, `intent_threshold` and `move_resolution` call it.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import torch

#: `--ko-ramp`'s legal values. ``ramp`` = the legacy mean-anchored 15 %-window ramp, no crit (production;
#: byte-identical — every site keeps its own inline expression); ``exact`` = this module (the closed form).
KO_RAMP_MODES: Tuple[str, ...] = ("ramp", "exact")

#: The 16 gen-3 damage rolls (percent of the 100-roll damage), `sim/battle.ts` ``randomizer``.
ROLLS: Tuple[int, ...] = tuple(range(85, 101))
N_ROLLS = len(ROLLS)
#: The op's damage numbers are MEAN-roll damage: every kernel multiplies the formula by this (the mean of 85 … 100
#: is exactly 92.5). The 100-roll ("top") damage is the op's number divided by it.
MEAN_ROLL = 0.925
#: Gen ≤ 5 crit chance by ``critRatio`` (`sim/battle-actions.ts` ``critMult = [0, 16, 8, 4, 3, 2]``).
CRIT_P_BY_RATIO: Dict[int, float] = {1: 1.0 / 16.0, 2: 1.0 / 8.0, 3: 1.0 / 4.0, 4: 1.0 / 3.0, 5: 1.0 / 2.0}
CRIT_P_BASE = CRIT_P_BY_RATIO[1]
#: The opponent's HP precision under HP Percentage Mod: a reported p < 100 % is a fraction in (p − 1 %, p].
OPP_HP_PERCENT = 0.01

_TINY = 1e-6
#: `roll_ko_prob`'s floor on the per-roll step in the COUNT division only (the value uses the true step): a
#: step below it (|top| < 1e-28 · width) can mis-count only terms of magnitude <= 16 · 1e-30, so the bound is far
#: below fp32's resolution of anything the sum reads. Normal in fp32 (smallest normal ~1.2e-38).
_STEP_FLOOR = 1e-30


def roll_ko_prob(mean_dmg: torch.Tensor, hp_lo: torch.Tensor, hp_hi: torch.Tensor) -> torch.Tensor:
    """P(the hit KOs | no crit) over the 16 gen-3 rolls, the target's HP uniform on ``[hp_lo, hp_hi]`` (the
    observation's precision), IN CLOSED FORM. ``mean_dmg`` is the op's MEAN-roll damage in the SAME units as the HP
    bounds (HP points, or fractions of max HP); every argument broadcasts. Continuous, piecewise linear in the damage.

    The 16 terms ``x_r = (top · r/100 − lo) / w`` are an EXACT arithmetic sequence in r (the op's damage is
    continuous — no per-roll floor), so with the sequence ascending from ``x_start`` (the r = 85 term for top >= 0,
    the r = 100 term for top < 0) by ``step = |top| / (100 w)``:

        Σ_r clamp(x_r, 0, 1) = (16 − n1) + (n1 − n0) · (x_start + step · (n0 + n1 − 1) / 2)
        n0 = #{j : x_j < 0} = clamp(ceil(−x_start / step), 0, 16)        (the terms that read 0)
        n1 = #{j : x_j <= 1} = clamp(floor((1 − x_start) / step) + 1, 0, 16)   (16 − n1 terms read 1)

    the middle run's n1 − n0 terms summed as their count times their mean. The two counts are a floor / ceil of a
    SCORE, but the sum is CONTINUOUS across each count's step: the term that changes class there is 0 (at n0) or 1
    (at n1) to the quotient's rounding, so a count off by one moves the value by that rounding, never a jump
    (`selection_sites` REASONS ``COUNT_CONTINUOUS``). The counts are read on DETACHED operands (piecewise constant:
    their gradient is 0, and a zero step must not put 0/0 into the backward); the value reads the true
    ``x_start`` / ``step``, so its gradient is the 16-roll sum's term by term — Σ over the middle run of
    ∂x_r — including at top = 0, where ``torch.minimum``'s tie splits the two end terms' slopes (0.85 and 1.0 per
    100 w) evenly into the mean roll's 0.925 and ``abs``'s derivative is 0. Both ends are inclusive (a term exactly at
    0 or 1 is in the middle run), as `torch.clamp`'s subgradient is."""
    top = mean_dmg / MEAN_ROLL
    width = (hp_hi - hp_lo).clamp_min(_TINY)
    x_lo_roll = (top * (ROLLS[0] / 100.0) - hp_lo) / width            # the r = 85 term
    x_hi_roll = (top * (ROLLS[-1] / 100.0) - hp_lo) / width           # the r = 100 term
    x_start = torch.minimum(x_lo_roll, x_hi_roll)
    step = top.abs() * 0.01 / width
    n0, n1 = run_counts(x_start, step)
    return closed_sum(x_start, step, n0, n1) / float(N_ROLLS)


def run_counts(x_start: torch.Tensor, step: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """`roll_ko_prob`'s two run counts (FLOAT tensors, on DETACHED operands): ``n0`` = the terms below 0,
    ``n1`` = the terms at most 1, of the ascending sequence ``x_start + step · j`` (j = 0 … 15)."""
    xs, sd = x_start.detach(), step.detach().clamp_min(_STEP_FLOOR)
    n0 = (-xs / sd).ceil().clamp(0.0, float(N_ROLLS))
    n1 = ((1.0 - xs) / sd).floor().add(1.0).clamp(0.0, float(N_ROLLS))
    return n0, n1


def closed_sum(x_start: torch.Tensor, step: torch.Tensor, n0: torch.Tensor, n1: torch.Tensor) -> torch.Tensor:
    """Σ_j clamp(x_start + step · j, 0, 1) given the run counts: the ``16 − n1`` terms that read 1, plus the middle
    run's ``n1 − n0`` terms as their count times their mean. Any counts may be passed (the continuity pin reads it at
    a neighbouring count)."""
    mid = x_start + step * (0.5 * (n0 + n1 - 1.0))
    total: torch.Tensor = (float(N_ROLLS) - n1) + (n1 - n0) * mid
    return total


def ko_given_hit(mean_dmg: torch.Tensor, crit_mean_dmg: torch.Tensor, hp_lo: torch.Tensor, hp_hi: torch.Tensor,
                 crit_p: Optional[torch.Tensor] = None) -> torch.Tensor:
    """P(KO | the move hits) = the no-crit roll probability and the crit roll probability, mixed by the move's crit
    chance ``crit_p`` (None → the base 1/16). ``crit_mean_dmg`` is the op's crit column (2 × the PRE-screen
    mean-roll damage: a crit ignores the screens)."""
    c = CRIT_P_BASE if crit_p is None else crit_p
    return (1.0 - c) * roll_ko_prob(mean_dmg, hp_lo, hp_hi) + c * roll_ko_prob(crit_mean_dmg, hp_lo, hp_hi)


def ours_hp_bounds(cur_hp: torch.Tensor, one_hp: Union[torch.Tensor, float]) -> Tuple[torch.Tensor, torch.Tensor]:
    """OUR mon's HP is exact: the step is resolved over ±½ HP (``one_hp`` = 1 HP in the caller's units: 1 in HP
    points, ``1 / maxhp`` in fractions)."""
    return cur_hp - 0.5 * one_hp, cur_hp + 0.5 * one_hp


def opp_hp_bounds(cur_hp: torch.Tensor, hp_frac: torch.Tensor, maxhp: Union[torch.Tensor, float],
                  one_hp: Union[torch.Tensor, float]) -> Tuple[torch.Tensor, torch.Tensor]:
    """THEIR mon's HP as HP Percentage Mod reports it: a reported fraction p < 1 means (p − 1 %, p]; a full one is
    exact (resolved over ±½ HP, as ours). ``cur_hp`` = p in the caller's units (``p · maxhp`` in HP points, ``p`` in
    fractions); ``maxhp`` = the max HP in those units (1.0 for fractions); ``one_hp`` = 1 HP in them."""
    full = (hp_frac >= 1.0).to(cur_hp.dtype)
    lo = cur_hp - (1.0 - full) * (OPP_HP_PERCENT * maxhp) - full * (0.5 * one_hp)
    return lo, cur_hp + full * (0.5 * one_hp)


def crit_p_table(n_moves: int) -> torch.Tensor:
    """``[n_moves]`` each move num's crit chance from its gen-3 ``critRatio`` (`data/pokemon/gen3_moves.json`, read
    through the data facade's raw records; absent = 1). A ratio outside `CRIT_P_BY_RATIO` RAISES."""
    from agents import gen3_data
    out = torch.full((n_moves,), CRIT_P_BASE, dtype=torch.float32)
    for mid, rec in gen3_data.moves.raw().items():
        md = gen3_data.moves.get(mid)
        if md is None or not (0 <= int(md.num) < n_moves):
            continue
        ratio = int(rec.get("critRatio") or 1)
        if ratio not in CRIT_P_BY_RATIO:
            raise ValueError(f"ko_exact: move {mid!r} has critRatio {ratio}, outside the gen-3 table")
        out[int(md.num)] = CRIT_P_BY_RATIO[ratio]
    return out


__all__ = ["KO_RAMP_MODES", "ROLLS", "N_ROLLS", "MEAN_ROLL", "CRIT_P_BY_RATIO", "CRIT_P_BASE",
           "OPP_HP_PERCENT", "roll_ko_prob", "run_counts", "closed_sum", "ko_given_hit",
           "ours_hp_bounds", "opp_hp_bounds", "crit_p_table"]
