"""Incoming-damage / OHKO belief — the pure math core (ai_v5 incoming-damage feature).

We do **not** compute "the damage"; we compute a calibrated **belief about being KO'd** under hidden
opponent info (move / spread / item unknown). This module is the perf-light, deterministic core:
the Gen-3 damage formula (incl. the Explosion Def-halve + Sandstorm SpD mechanics), the roll→P(KO)
closed form, and P(outspeed) over a Speed distribution. Battle-state integration — reading our team
+ the opp active + field and building the ``Candidate`` / ``Defender`` / ``AttackerThreat`` beliefs —
lives in ``incoming_damage_encoder.py`` and calls these.

Design: `designs/ai_v5/design_incoming_damage_obs.md` §6–§7. All functions are pure (no poke-env,
no torch) so they unit-test without a battle.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from agents.enums import PokemonType, Status
from agents.gen3_mechanics import effective_multiplier_by_types

# Gen-3 categories are TYPE-based (pre-gen4 physical/special split). Physical types use Atk vs the
# defender's Def; special types use SpA vs SpD. This is also why Hidden Power Bug is physical.
_PHYSICAL_TYPES = frozenset({
    PokemonType.NORMAL, PokemonType.FIGHTING, PokemonType.POISON, PokemonType.GROUND,
    PokemonType.FLYING, PokemonType.BUG, PokemonType.ROCK, PokemonType.GHOST, PokemonType.STEEL,
})

# Per-mon output width: [phys_expdmg, spec_expdmg, phys_pko, spec_pko, p_outspeed]
PER_MON = 5
# Trailing opp-active scalars: [recovery_rate, cures_status(P rest), recovery_known]
RECOVERY = 3
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

_PARA_SPEED = 0.25      # gen3 paralysis quarters Speed
_BOOST_STAGES = {  # standard gen3 stat-stage multipliers (atk/spa/spe share this table)
    -6: 2 / 8, -5: 2 / 7, -4: 2 / 6, -3: 2 / 5, -2: 2 / 4, -1: 2 / 3,
    0: 1.0, 1: 3 / 2, 2: 4 / 2, 3: 5 / 2, 4: 6 / 2, 5: 7 / 2, 6: 8 / 2,
}
_SANDSTORM_SPD_MULT = 1.5   # gen3 Sandstorm gives Rock-types ×1.5 SpD


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
    (Sandstorm has no BP effect — its Rock-type SpD boost is applied in ``_channel_threat``.)"""
    if not weather:
        return 1.0
    w = weather.lower()
    if "rain" in w:
        return 1.5 if move_type == PokemonType.WATER else 0.5 if move_type == PokemonType.FIRE else 1.0
    if "sun" in w:
        return 1.5 if move_type == PokemonType.FIRE else 0.5 if move_type == PokemonType.WATER else 1.0
    return 1.0


def _is_sandstorm(weather: Optional[str]) -> bool:
    return weather is not None and "sand" in weather.lower()


def _channel_threat(cands, d: Defender, atk_tail: float, atk_mean: float, *,
                    a: AttackerThreat, screen: bool, is_phys: bool) -> Tuple[float, float]:
    """(pko, expdmg_frac) = max over a channel's candidates of p_in_set·(KO prob / dmg fraction)."""
    if not cands or d.hp_max <= 0:
        return 0.0, 0.0
    defense = d.def_stat if is_phys else d.spd_stat
    defense = max(1, int(defense * boost_mult(d.boost_def if is_phys else d.boost_spd)))
    # Gen-3 Sandstorm gives Rock-types ×1.5 SpD (special channel only — no effect on Def).
    if not is_phys and _is_sandstorm(a.weather) and PokemonType.ROCK in (d.type1, d.type2):
        defense = max(1, int(defense * _SANDSTORM_SPD_MULT))
    best_pko = best_exp = 0.0
    for c in cands:
        eff = effective_multiplier_by_types(c.move_type, d.type1, d.type2, d.ability, d.status)
        if eff <= 0.0:                      # immune (incl. Seismic Toss vs Ghost)
            continue
        stab = c.move_type in a.types
        if c.fixed_dmg is not None:         # fixed damage: ignores Atk/Def/roll, respects immunity
            pko = 1.0 if c.fixed_dmg >= d.hp_remaining else 0.0
            exp = c.fixed_dmg / d.hp_max
        else:
            # Explosion / Self-Destruct halve the target's Def in the Gen-3 calc.
            cdef = max(1, defense // 2) if c.halves_defense else defense
            w = weather_damage_mult(c.move_type, a.weather)
            burned = a.burn and is_phys
            dmax = gen3_damage_max(c.power, int(atk_tail), cdef, stab=stab, type_eff=eff,
                                   screen=screen, weather=w, burned=burned)
            pko = p_ko(dmax, d.hp_remaining)
            # expected dmg ≈ the max-roll damage scaled by the mean/tail stat ratio × mean roll
            # (damage is ~linear in Atk; avoids a second full calc on the obs hot path).
            ratio = (atk_mean / atk_tail) if atk_tail > 0 else 1.0
            exp = (dmax * ratio * _MEAN_ROLL) / d.hp_max
        best_pko = max(best_pko, c.p_in_set * pko)
        best_exp = max(best_exp, c.p_in_set * min(1.5, exp))
    return best_pko, best_exp


def compute_team_block(defenders: List[Defender], attacker: Optional[AttackerThreat],
                       n_slots: int) -> np.ndarray:
    """The incoming-KO belief block: per our mon (slot-aligned to the team), the phys/spec
    expected-damage-fraction + mode-max P(KO) + P(outspeed); then the 3 recovery scalars.

    Width = ``n_slots * PER_MON + RECOVERY``. All zeros when there is no opponent active (forced
    switch / battle start). A defender behind a Substitute can't be KO'd this turn → its KO/dmg
    rows are zeroed (the hit eats the Sub)."""
    out = np.zeros(n_slots * PER_MON + RECOVERY, dtype=np.float32)
    if attacker is None:
        return out
    for i, d in enumerate(defenders[:n_slots]):
        if d is None:
            continue
        if d.has_sub:
            phys_pko = spec_pko = phys_exp = spec_exp = 0.0
        else:
            phys_pko, phys_exp = _channel_threat(attacker.phys, d, attacker.atk_tail,
                                                  attacker.atk_mean, a=attacker,
                                                  screen=attacker.our_reflect, is_phys=True)
            spec_pko, spec_exp = _channel_threat(attacker.spec, d, attacker.spa_tail,
                                                 attacker.spa_mean, a=attacker,
                                                 screen=attacker.our_light_screen, is_phys=False)
        outspeed = p_outspeed(d.spe, attacker.spe_dist, our_boost=d.boost_spe,
                              opp_boost=attacker.boost_spe,
                              our_para=(d.status == Status.PAR), opp_para=attacker.para)
        base = i * PER_MON
        out[base:base + PER_MON] = (phys_exp, spec_exp, phys_pko, spec_pko, outspeed)
    out[n_slots * PER_MON:] = (attacker.recovery_rate, attacker.cures_status, attacker.recovery_known)
    return out
