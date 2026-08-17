"""Incoming-damage / OHKO belief — the pure math core (ai_v5 incoming-damage feature).

We do **not** compute "the damage"; we compute a calibrated **belief about being KO'd** under hidden
opponent info (move / spread / item unknown). This module is the perf-light, deterministic core:
the Gen-3 damage formula (incl. the gen≤4 Explosion/Self-Destruct Def-halve), the roll→P(KO)
closed form blended with a gen3 critical-hit tail term (``_CRIT_P``: a crit does ×2 and ignores
screens, so a move that only KOs on a crit reads a calibrated ~6% instead of P(KO)~0), and
P(outspeed) over a Speed distribution. Battle-state integration — reading our team
+ the opp active + field and building the ``Candidate`` / ``Defender`` / ``AttackerThreat`` beliefs —
lives in ``incoming_damage_encoder.py`` and calls these.

Design: `designs/ai_v5/design_incoming_damage_obs.md` §6–§7. All functions are pure (no poke-env,
no torch) so they unit-test without a battle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import numpy as np

from agents.enums import PokemonType, Status
from agents.gen3_mechanics import effective_multiplier_by_types

# Gen-3 categories are TYPE-based (pre-gen4 physical/special split). Physical types use Atk vs the
# defender's Def; special types use SpA vs SpD. This is also why Hidden Power Bug is physical.
_PHYSICAL_TYPES = frozenset({
    PokemonType.NORMAL, PokemonType.FIGHTING, PokemonType.POISON, PokemonType.GROUND,
    PokemonType.FLYING, PokemonType.BUG, PokemonType.ROCK, PokemonType.GHOST, PokemonType.STEEL,
})

# Per-mon output width (gen3_incoming_crit_split):
#   [phys_expdmg, spec_expdmg, phys_pko_nocrit, spec_pko_nocrit, phys_crit_delta, spec_crit_delta,
#    p_outspeed, threat_revealed]
# P(KO) is the modal no-crit line; the crit risk is exposed as the DELTA (pko_crit − pko_nocrit ∈
# [0, _CRIT_P]) rather than the near-redundant absolute crit-inclusive line — the crit "tax" is its
# own decorrelated feature a small net can read (crit-inclusive = nocrit + tiny tail, ≤6% apart, gets
# buried after standardization). The model prices the modal outcome and treats crit as a priced tail.
# ``threat_revealed`` is the provenance of the dominant KO threat (1.0 = a revealed move, <1.0 = a
# usage-prior guess; 0.0 when no candidate can KO — read it jointly with the pko channels).
PER_MON = 8
# Named field offsets WITHIN a per-mon slot — the single source of truth for the block layout.
# ANY code that reads one field of the block (reward shaping, prober, fuzz tests) MUST index via
# these names, never a hardcoded literal: a stale magic offset is exactly how the reward PBRS came
# to read `phys_crit_delta` as `p_outspeed` when the crit-split inserted the two delta fields and
# pushed outspeed from 4 → 6. The producer below assembles the slot FROM these names, and
# `_PER_MON_FIELDS == PER_MON` is asserted at import, so inserting a field forces every index to
# move in lockstep (or fail loudly) instead of silently desyncing a consumer.
IDX_PHYS_EXP = 0          # physical expected-damage fraction
IDX_SPEC_EXP = 1          # special  expected-damage fraction
IDX_PHYS_PKO = 2          # physical modal (no-crit) P(KO)
IDX_SPEC_PKO = 3          # special  modal (no-crit) P(KO)
IDX_PHYS_CRIT_DELTA = 4   # physical crit-risk delta (crit-inclusive − no-crit ∈ [0, _CRIT_P])
IDX_SPEC_CRIT_DELTA = 5   # special  crit-risk delta
IDX_OUTSPEED = 6          # P(our mon outspeeds the opp active)
IDX_THREAT_REVEALED = 7   # provenance of the dominant KO threat (1.0 revealed … 0.0 none)
_PER_MON_FIELDS = 8
assert _PER_MON_FIELDS == PER_MON, "named per-mon field count must equal PER_MON"

# Trailing opp-active scalars: [recovery_rate, cures_status(P rest), recovery_known]
RECOVERY = 3
IDX_RECOVERY_RATE = 0
IDX_RECOVERY_CURES = 1
IDX_RECOVERY_KNOWN = 2
_MEAN_ROLL = 0.925   # mean of the 16 damage rolls (85..100)/100

# Gen-3 single-hit damage at level 100, max roll (R=100), before the random roll:
#   core = floor( floor( floor(2*L/5 + 2) * power * A / D ) / 50 ) + 2
# 2*100/5 + 2 = 42.
_L_TERM = 42

# Fixed / level / fraction damage moves — basePower=0 in the dex, so the multiplicative formula
# reads 0 and gen3_data buckets them STATUS. They are exactly the unpriced KOs on stall mons, so
# the encoder tags a Candidate with ``fixed_dmg`` (the constant damage at level 100) and
# ``_channel_threat`` resolves them ignoring Atk/Def/roll but respecting type immunity.
# HP-relative (Super Fang/Endeavor), reflective (Counter/Mirror Coat) and the unreliable OHKO
# moves (Sheer Cold/Fissure/…) are NOT priced in v1 — they need the live HP / our-damage context
# the obs hot path doesn't carry; deferred to a v2 context-aware pass.
FIXED_DAMAGE: dict[str, int] = {
    "seismictoss": 100, "nightshade": 100,   # level-100 fixed
    "dragonrage": 40, "sonicboom": 20,        # constant
}

# Gen-3 base critical-hit rate (most moves; high-crit-ratio moves are higher, not modelled). A crit
# does ×2 damage and IGNORES screens (Reflect/Light Screen). Folded into P(KO) as a small tail term
# so a hit that only KOs on a crit surfaces a calibrated ~6% KO risk instead of reading P(KO)~0.
# Set to 0.0 to disable the crit term (recovers the pre-crit belief — used by the A/B calibration probe).
_CRIT_P = 1.0 / 16.0

_PARA_SPEED = 0.25      # gen3 paralysis quarters Speed
_BOOST_STAGES = {  # standard gen3 stat-stage multipliers (atk/spa/spe share this table)
    -6: 2 / 8, -5: 2 / 7, -4: 2 / 6, -3: 2 / 5, -2: 2 / 4, -1: 2 / 3,
    0: 1.0, 1: 3 / 2, 2: 4 / 2, 3: 5 / 2, 4: 6 / 2, 5: 7 / 2, 6: 8 / 2,
}
# NOTE: gen3 Sandstorm gives Rock-types NO SpD boost — the ×1.5 is a gen4+ mechanic. The Showdown
# gen3 mod explicitly nulls it (`data/mods/gen3/conditions.ts`: `onModifySpD: undefined`), so we do
# NOT apply it here. Sandstorm has no BP effect either; its only combat effect is residual chip,
# which is folded into HP elsewhere, not the per-hit damage belief.


def type_is_physical(ptype: PokemonType) -> bool:
    return ptype in _PHYSICAL_TYPES


def boost_mult(stage: int) -> float:
    """Gen-3 stat-stage multiplier for atk/spa/spe (clamped to [-6, 6])."""
    return _BOOST_STAGES[max(-6, min(6, int(stage)))]


def gen3_damage_max(power: int, atk: int, defense: int, *, stab: bool, type_eff: float,
                    screen: bool = False, weather: float = 1.0, burned: bool = False) -> int:
    """Gen-3 max-roll (R=100) damage of one hit. Modifiers applied multiplicatively with a final
    floor (a belief, not a frame-exact calc): STAB 1.5, type effectiveness, screen ×0.5,
    weather (×1.5/×0.5 for boosted/weakened), burn ×0.5 (physical). Returns 0 on immunity."""
    if power <= 0 or type_eff <= 0.0 or defense <= 0:
        return 0
    core = ((_L_TERM * power * atk) // defense) // 50 + 2
    mod = type_eff * weather
    if stab:
        mod *= 1.5
    if screen:
        mod *= 0.5
    if burned:
        mod *= 0.5
    return int(core * mod)


def p_ko(dmg_max: int, remaining_hp: int) -> float:
    """P(KO this hit) integrating the Gen-3 damage roll (R ∈ 85..100, 16 equal values).

    damage(R) = floor(dmg_max * R / 100); KO ⟺ damage(R) ≥ remaining_hp. Closed form: count the
    R in [85,100] that reach the threshold. dmg_max is the R=100 damage."""
    if remaining_hp <= 0:
        return 1.0
    if dmg_max <= 0:
        return 0.0
    if (dmg_max * 85) // 100 >= remaining_hp:   # even the low roll KOs
        return 1.0
    if dmg_max < remaining_hp:                   # even the high roll can't
        return 0.0
    hits = sum(1 for r in range(85, 101) if (dmg_max * r) // 100 >= remaining_hp)
    return hits / 16.0


def p_outspeed(our_spe: int, opp_spe_dist: Sequence[Tuple[int, float]], *,
               our_boost: int = 0, opp_boost: int = 0,
               our_para: bool = False, opp_para: bool = False) -> float:
    """P(we move first) over the opponent's Speed **distribution** (the hidden nature/EV).

    ``opp_spe_dist`` = ``[(spe_stat, weight)]`` (e.g. ``priors.stat_distribution(species,'spe')``).
    Our Speed is exact; observed boosts + paralysis fold in on both sides. Speed ties → ½ (gen3
    coin flip). Returns 0.5 when the opponent's Speed is unknown (no distribution)."""
    if not opp_spe_dist:
        return 0.5
    ours = our_spe * boost_mult(our_boost) * (_PARA_SPEED if our_para else 1.0)
    faster = ties = total = 0.0
    for spe, w in opp_spe_dist:
        theirs = spe * boost_mult(opp_boost) * (_PARA_SPEED if opp_para else 1.0)
        if ours > theirs:
            faster += w
        elif ours == theirs:
            ties += w
        total += w
    if total <= 0:
        return 0.5
    return (faster + 0.5 * ties) / total


def percentile(dist: Sequence[Tuple[int, float]], q: float) -> Optional[int]:
    """Weighted q-quantile (0..1) of a ``[(value, weight)]`` distribution (e.g. the offensive-tail
    Atk/SpA stat for the worst-case magnitude, or the mean via q=0.5-ish). None if empty."""
    if not dist:
        return None
    items = sorted(dist)
    total = sum(w for _, w in items)
    if total <= 0:
        return items[-1][0]
    acc = 0.0
    for val, w in items:
        acc += w
        if acc >= q * total:
            return val
    return items[-1][0]


def weighted_mean(dist: Sequence[Tuple[int, float]]) -> Optional[float]:
    """Usage-weighted mean of a ``[(value, weight)]`` distribution (the 'expected' magnitude)."""
    if not dist:
        return None
    total = sum(w for _, w in dist)
    return sum(v * w for v, w in dist) / total if total > 0 else None


# ---------------------------------------------------------------------------
# Battle-integration: the per-defender incoming-KO belief block
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Candidate:
    """One plausible opponent attacking move vs our team. ``p_in_set`` = revealed(1.0) or the
    usage prior over remaining slots. ``fixed_dmg`` is set for fixed-damage moves (Seismic Toss
    etc.) — they ignore Atk/Def/roll but still respect type immunity. ``halves_defense`` flags
    Explosion / Self-Destruct, which halve the target's Def in the Gen-3 calc."""
    move_type: PokemonType
    power: int
    p_in_set: float
    fixed_dmg: Optional[int] = None
    halves_defense: bool = False


@dataclass(frozen=True)
class Defender:
    """One of our mons (exact, known). Boosts are stat stages; ``has_sub`` only for the active.
    ``status`` is the raw poke-env :class:`~agents.enums.Status` enum (or None) — fed straight to
    the effectiveness primitive (which keys Flash Fire off FRZ) and folded into P(outspeed) for
    paralysis."""
    def_stat: int
    spd_stat: int
    hp_remaining: int
    hp_max: int
    spe: int
    type1: PokemonType
    type2: Optional[PokemonType]
    ability: Optional[str]
    status: Optional[Status]
    boost_def: int = 0
    boost_spd: int = 0
    boost_spe: int = 0
    has_sub: bool = False


@dataclass(frozen=True)
class AttackerThreat:
    """The opponent active as a *belief*: types known; offensive stats as usage percentiles
    (boost already folded in by the caller); candidate moves split by gen3 type-category."""
    types: Tuple[PokemonType, ...]
    atk_tail: float
    atk_mean: float
    spa_tail: float
    spa_mean: float
    spe_dist: Sequence[Tuple[int, float]]
    boost_spe: int = 0
    para: bool = False
    burn: bool = False
    phys: Sequence[Candidate] = field(default_factory=tuple)
    spec: Sequence[Candidate] = field(default_factory=tuple)
    our_reflect: bool = False
    our_light_screen: bool = False
    weather: Optional[str] = None
    recovery_rate: float = 0.0
    cures_status: float = 0.0
    recovery_known: float = 0.0


def weather_damage_mult(move_type: PokemonType, weather: Optional[str]) -> float:
    """Gen-3 weather BP modifier: rain ×1.5 Water/×0.5 Fire; sun ×1.5 Fire/×0.5 Water.
    (Sandstorm has no BP effect, and gen3 gives Rock-types no SpD boost — that's a gen4+ mechanic.)"""
    if not weather:
        return 1.0
    w = weather.lower()
    if "rain" in w:
        return 1.5 if move_type == PokemonType.WATER else 0.5 if move_type == PokemonType.FIRE else 1.0
    if "sun" in w:
        return 1.5 if move_type == PokemonType.FIRE else 0.5 if move_type == PokemonType.WATER else 1.0
    return 1.0


def _channel_threat(cands: Sequence[Candidate], d: Defender, atk_tail: float, atk_mean: float, *,
                    a: AttackerThreat, screen: bool, is_phys: bool) -> Tuple[float, float, float, float]:
    """(pko_nocrit, pko_crit, expdmg_frac, revealed) = max over a channel's candidates of
    p_in_set·(KO prob / dmg fraction), plus the provenance of the dominant threat.

    The KO prob is SPLIT into two channels the policy/critic see separately (gen3_incoming_crit_split):
    ``pko_nocrit`` is the modal-line roll integration (what happens if you DON'T get crit — the
    outcome you plan around); ``pko_crit`` additionally blends the ``_CRIT_P`` crit branch (×2,
    screen-ignoring) — the rare tail you hope to dodge. Their gap is the crit risk, made explicit so
    the model can price the modal line without over-weighting the coinflip. Expected-damage stays the
    mean-stat chip (no crit term). ``revealed`` is the ``p_in_set`` of the dominant (crit-weighted)
    threat: 1.0 = a REVEALED move (we KNOW), <1.0 = a usage-prior GUESS (the provenance / how-much-
    are-we-guessing signal). The KO magnitude rides ``atk_tail``; expected-damage rides ``atk_mean``."""
    if not cands or d.hp_max <= 0:
        return 0.0, 0.0, 0.0, 0.0
    defense = d.def_stat if is_phys else d.spd_stat
    defense = max(1, int(defense * boost_mult(d.boost_def if is_phys else d.boost_spd)))
    best_nc = best_cr = best_exp = best_revealed = 0.0
    for c in cands:
        eff = effective_multiplier_by_types(c.move_type, d.type1, d.type2, d.ability, d.status)
        if eff <= 0.0:                      # immune (incl. Seismic Toss vs Ghost)
            continue
        stab = c.move_type in a.types
        if c.fixed_dmg is not None:         # fixed damage: ignores Atk/Def/roll, respects immunity
            pko_nc = pko_cr = 1.0 if c.fixed_dmg >= d.hp_remaining else 0.0
            exp = c.fixed_dmg / d.hp_max
        else:
            # Explosion / Self-Destruct halve the target's Def in the Gen-3 calc.
            cdef = max(1, defense // 2) if c.halves_defense else defense
            w = weather_damage_mult(c.move_type, a.weather)
            burned = a.burn and is_phys
            dmax = gen3_damage_max(c.power, int(atk_tail), cdef, stab=stab, type_eff=eff,
                                   screen=screen, weather=w, burned=burned)
            pko_nc = p_ko(dmax, d.hp_remaining)
            pko_cr = pko_nc
            if _CRIT_P > 0.0:
                # gen3 crit: ×2 damage and IGNORES screens (so compute it screen-free, then double).
                # Burn still applies; we keep the (boosted) Def — a defender at +Def being OHKO'd is
                # rare, so this stays a calibrated tail term, not a worst-case calc. Blending the
                # crit branch lifts a move that only KOs on a crit out of the P(KO)~0 floor.
                dmax_crit = 2 * gen3_damage_max(c.power, int(atk_tail), cdef, stab=stab, type_eff=eff,
                                                screen=False, weather=w, burned=burned)
                pko_cr = (1.0 - _CRIT_P) * pko_nc + _CRIT_P * p_ko(dmax_crit, d.hp_remaining)
            # expected dmg ≈ the max-roll damage scaled by the mean/tail stat ratio × mean roll
            # (damage is ~linear in Atk; avoids a second full calc on the obs hot path).
            ratio = (atk_mean / atk_tail) if atk_tail > 0 else 1.0
            exp = (dmax * ratio * _MEAN_ROLL) / d.hp_max
        w_cr = c.p_in_set * pko_cr
        if w_cr > best_cr:                  # the dominant threat owns the provenance scalar
            best_cr, best_revealed = w_cr, c.p_in_set
        best_nc = max(best_nc, c.p_in_set * pko_nc)
        best_exp = max(best_exp, c.p_in_set * min(1.5, exp))
    return best_nc, best_cr, best_exp, best_revealed


def compute_team_block(defenders: Sequence[Optional[Defender]],
                       attacker: Optional[AttackerThreat],
                       n_slots: int) -> np.ndarray:
    """The incoming-KO belief block: per our mon (slot-aligned to the team), the phys/spec
    expected-damage-fraction + the modal no-crit P(KO) + the crit-risk DELTA per channel + P(outspeed)
    + the dominant threat's revealed-provenance; then the 3 recovery scalars.

    Width = ``n_slots * PER_MON + RECOVERY``. All zeros when there is no opponent active (forced
    switch / battle start). A defender behind a Substitute can't be KO'd this turn → its KO/dmg
    rows are zeroed (the hit eats the Sub).

    ``defenders`` slots may be ``None`` (empty / fainted / stats not yet known) — the loop skips
    them, leaving that mon's row zeroed. That has always been the behaviour; the parameter type
    used to say ``List[Defender]`` and simply did not describe it."""
    out = np.zeros(n_slots * PER_MON + RECOVERY, dtype=np.float32)
    if attacker is None:
        return out
    for i, d in enumerate(defenders[:n_slots]):
        if d is None:
            continue
        if d.has_sub:
            phys_nc = phys_cr = spec_nc = spec_cr = phys_exp = spec_exp = revealed = 0.0
        else:
            phys_nc, phys_cr, phys_exp, phys_rev = _channel_threat(
                attacker.phys, d, attacker.atk_tail, attacker.atk_mean, a=attacker,
                screen=attacker.our_reflect, is_phys=True)
            spec_nc, spec_cr, spec_exp, spec_rev = _channel_threat(
                attacker.spec, d, attacker.spa_tail, attacker.spa_mean, a=attacker,
                screen=attacker.our_light_screen, is_phys=False)
            revealed = phys_rev if phys_cr >= spec_cr else spec_rev   # provenance of the dominant channel
        outspeed = p_outspeed(d.spe, attacker.spe_dist, our_boost=d.boost_spe,
                              opp_boost=attacker.boost_spe,
                              our_para=(d.status == Status.PAR), opp_para=attacker.para)
        # Expose the crit RISK as the delta (crit-inclusive − no-crit ∈ [0, _CRIT_P]) — a decorrelated
        # "crit tax" feature, not the near-redundant absolute crit-inclusive line.
        phys_crit_delta = max(0.0, phys_cr - phys_nc)
        spec_crit_delta = max(0.0, spec_cr - spec_nc)
        # Write each field by NAME (not a positional tuple) so the producer and every consumer
        # share one layout contract — inserting a field is a one-line add here, and a stale
        # consumer offset can't silently read the wrong channel.
        base = i * PER_MON
        out[base + IDX_PHYS_EXP] = phys_exp
        out[base + IDX_SPEC_EXP] = spec_exp
        out[base + IDX_PHYS_PKO] = phys_nc
        out[base + IDX_SPEC_PKO] = spec_nc
        out[base + IDX_PHYS_CRIT_DELTA] = phys_crit_delta
        out[base + IDX_SPEC_CRIT_DELTA] = spec_crit_delta
        out[base + IDX_OUTSPEED] = outspeed
        out[base + IDX_THREAT_REVEALED] = revealed
    rec = n_slots * PER_MON
    out[rec + IDX_RECOVERY_RATE] = attacker.recovery_rate
    out[rec + IDX_RECOVERY_CURES] = attacker.cures_status
    out[rec + IDX_RECOVERY_KNOWN] = attacker.recovery_known
    return out
