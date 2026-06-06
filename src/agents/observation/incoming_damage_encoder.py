"""Battle → incoming-damage belief block (the glue over the pure ``incoming_damage`` math core).

Single public entry point: :func:`encode_block`. It reads the live battle (our team, the opponent
active, the field via the :class:`~agents.battle.live_view.LiveView` read-model), turns the hidden
opponent set into a ``Candidate`` / ``AttackerThreat`` belief (revealed moves ∪ Smogon-usage priors;
offensive stats as a usage distribution), and folds the per-defender KO/chip/outspeed belief through
``incoming_damage.compute_team_block``. Everything else here is private.

The math is deliberately kept poke-env-free in ``incoming_damage.py``; this module owns the *only*
poke-env / data-facade reads. Per-species static reads (the usage stat distribution and the prior
candidate moves) are ``lru_cache``d — they can't change mid-battle — which is what keeps the block
inside the obs-build benchmark budget.

Design: `designs/ai_v5/design_incoming_damage_obs.md`.
"""
from __future__ import annotations

import functools
from typing import List, Optional, Tuple

import numpy as np

from agents import gen3_data
from agents.enums import Status
from agents.gen3_mechanics import RECOVERY_MOVES
from agents.observation import incoming_damage as inc
from agents.observation.constants import TEAM_SIZE

# Smogon-usage prior knobs for the unrevealed-slot candidate moves.
_PRIOR_MOVE_MIN_P = 0.12          # drop prior moves below this P(in set) — usage noise, not a threat
_MAX_CANDIDATES_PER_CHANNEL = 4   # cap per phys/spec channel (a set runs ≤4 attacking moves)
_OFFENSIVE_TAIL_Q = 0.85          # worst-case offensive-stat percentile (the 252-EV+ tail)
_FALLBACK_BASE = 80               # neutral base stat when a species has no dex entry
_FALLBACK_EV = 252                # the no-prior fallback assumes a max-invested neutral spread

# Rest is the recovery discriminator (cures the Toxic clock AND fully heals → an unbreakable wall
# for a chip-based team), so it gets its own ``cures_status`` scalar.
_REST = "rest"

# Explosion / Self-Destruct halve the target's Def in Gen 3 — the math core prices this off a
# per-Candidate flag, so the move-id → flag classification lives here (the core stays id-agnostic).
_HALVE_DEF_MOVES = frozenset({"explosion", "selfdestruct"})


@functools.lru_cache(maxsize=None)
def _offensive_stat(species: str, stat: str) -> Tuple[float, float, tuple]:
    """(tail, weighted_mean, dist) for an opponent's offensive stat from the spread prior; falls
    back to a max-EV / neutral-nature stat off the base when the species has no prior. Static per
    (species, stat) → cached (priors don't change mid-battle)."""
    dist = tuple(gen3_data.priors.stat_distribution(species, stat))
    tail = inc.percentile(dist, _OFFENSIVE_TAIL_Q)
    mean = inc.weighted_mean(dist)
    if tail is None or mean is None:
        sp = gen3_data.species.get(species)
        base = sp.base_stats.get(stat, _FALLBACK_BASE) if sp is not None else _FALLBACK_BASE
        fb = float(gen3_data.priors.gen3_stat(base, _FALLBACK_EV, 1.0))
        return fb, fb, ((fb, 1.0),)
    return float(tail), float(mean), dist


def _is_damaging(mid: str, md) -> bool:
    """Does move ``mid`` deal damage we should price? A positive-BP move or a fixed-damage move
    (Seismic Toss/… read 0 BP in the dex but hit for a constant)."""
    return md is not None and ((md.base_power or 0) > 0 or mid in inc.FIXED_DAMAGE)


def _make_candidate(mid: str, md, p_in_set: float) -> inc.Candidate:
    """Build a :class:`incoming_damage.Candidate` for move ``mid`` at probability ``p_in_set``."""
    return inc.Candidate(
        md.type, int(md.base_power or 0), float(p_in_set),
        fixed_dmg=inc.FIXED_DAMAGE.get(mid),
        halves_defense=(mid in _HALVE_DEF_MOVES),
    )


@functools.lru_cache(maxsize=None)
def _prior_candidates(species: str) -> Tuple[tuple, tuple]:
    """Cached per-species prior (unrevealed-slot) candidate moves, split by gen3 type-category.
    Static per species → computed once. Revealed moves are merged in per decision by
    :func:`_candidates`; duplicates are harmless because the belief takes the max over candidates
    (the revealed P=1 copy wins)."""
    phys: list = []
    spec: list = []
    pc = sc = 0
    for p, mid in sorted(((p, mid) for mid, p in gen3_data.priors.moves(species).items()
                          if p >= _PRIOR_MOVE_MIN_P), reverse=True):
        md = gen3_data.moves.get(mid)
        if not _is_damaging(mid, md):
            continue
        cand = _make_candidate(mid, md, p)
        if inc.type_is_physical(md.type):
            if pc < _MAX_CANDIDATES_PER_CHANNEL:
                pc += 1
                phys.append(cand)
        elif sc < _MAX_CANDIDATES_PER_CHANNEL:
            sc += 1
            spec.append(cand)
    return tuple(phys), tuple(spec)


def _candidates(opp_active, species: str) -> Tuple[list, list]:
    """Physical / special candidate moves vs us = revealed (P=1, per decision) ∪ cached prior
    moves. Fixed-damage moves (Seismic Toss/…) carry constant damage despite the dex STATUS tag."""
    phys: list = []
    spec: list = []
    for mid in (getattr(opp_active, "moves", {}) or {}):
        md = gen3_data.moves.get(mid)
        if _is_damaging(mid, md):
            cand = _make_candidate(mid, md, 1.0)
            (phys if inc.type_is_physical(md.type) else spec).append(cand)
    pp, sp = _prior_candidates(species)
    return phys + list(pp), spec + list(sp)


def _attacker_threat(opp_active, live) -> Optional[inc.AttackerThreat]:
    """The opponent active as a belief (None if there is no opp active / no species yet): types
    known, offensive stats as usage tail+mean (boost folded in), candidate moves split by channel,
    plus our screens / the weather (via the LiveView read-model) and the recovery scalars."""
    if opp_active is None:
        return None
    species = getattr(opp_active, "species", None)
    if not species:
        return None
    boosts = getattr(opp_active, "boosts", {}) or {}
    atk_b, spa_b = inc.boost_mult(boosts.get("atk", 0)), inc.boost_mult(boosts.get("spa", 0))
    atk_tail, atk_mean, _ = _offensive_stat(species, "atk")
    spa_tail, spa_mean, _ = _offensive_stat(species, "spa")
    _, _, spe_dist = _offensive_stat(species, "spe")
    phys, spec = _candidates(opp_active, species)

    # Recovery scalars (Suicune-Rest discriminator): revealed move → certain; else its usage prior.
    prior_mv = gen3_data.priors.moves(species)
    revealed = set(getattr(opp_active, "moves", {}) or {})
    rate = min(1.0, sum(1.0 if rm in revealed else prior_mv.get(rm, 0.0) for rm in RECOVERY_MOVES))
    cures = 1.0 if _REST in revealed else prior_mv.get(_REST, 0.0)
    known = 1.0 if revealed & RECOVERY_MOVES else 0.0

    # Field via the read-model (LiveView) — our screens + weather (strict-API boundary; global_env
    # reads screens/weather the same way). None on the plain-battle / unit-test path → no screens /
    # no weather (a small, safe over-pessimism in that rare path).
    reflect = lscreen = False
    weather = None
    if live is not None:
        oursc = getattr(live.ours, "side_conditions", {}) or {}
        reflect = "reflect" in oursc
        lscreen = "light_screen" in oursc
        lw = getattr(live, "weather", None)
        weather = getattr(lw, "weather", None) if lw is not None else None

    opp_types = tuple(t for t in (opp_active.type_1, opp_active.type_2) if t is not None)
    status = getattr(opp_active, "status", None)
    return inc.AttackerThreat(
        types=opp_types,
        atk_tail=atk_tail * atk_b, atk_mean=atk_mean * atk_b,
        spa_tail=spa_tail * spa_b, spa_mean=spa_mean * spa_b,
        spe_dist=spe_dist, boost_spe=int(boosts.get("spe", 0)),
        para=(status == Status.PAR), burn=(status == Status.BRN),
        phys=phys, spec=spec, our_reflect=reflect, our_light_screen=lscreen, weather=weather,
        recovery_rate=rate, cures_status=float(cures), recovery_known=known,
    )


def _defender(mon, active_mon) -> Optional[inc.Defender]:
    """One of our mons as an exact ``Defender`` (None if absent / fainted / stats not yet known).
    ``status`` stays the raw :class:`~agents.enums.Status` enum; ``has_sub`` is only set for the
    active mon (a Substitute eats the incoming hit, so a benched mon can't be 'behind' one)."""
    if mon is None or getattr(mon, "fainted", False):
        return None
    stats = getattr(mon, "stats", None) or {}
    d, sd, spe = stats.get("def"), stats.get("spd"), stats.get("spe")
    hp, hpx = getattr(mon, "current_hp", None), getattr(mon, "max_hp", None)
    if not (d and sd and spe and hp and hpx):
        return None
    boosts = getattr(mon, "boosts", {}) or {}
    has_sub = mon is active_mon and any(
        getattr(e, "name", "") == "SUBSTITUTE" for e in (getattr(mon, "effects", {}) or {}))
    return inc.Defender(
        def_stat=int(d), spd_stat=int(sd), hp_remaining=int(hp), hp_max=int(hpx), spe=int(spe),
        type1=mon.type_1, type2=mon.type_2, ability=getattr(mon, "ability", None),
        status=getattr(mon, "status", None), boost_def=int(boosts.get("def", 0)),
        boost_spd=int(boosts.get("spd", 0)), boost_spe=int(boosts.get("spe", 0)), has_sub=has_sub,
    )


def encode_block(battle, our_team: List, live=None) -> np.ndarray:
    """The incoming-damage / OHKO belief block (incoming_damage_v1) for one decision.

    Per our mon (slot-aligned to the team): the opponent active's phys/spec expected-chip + P(KO) +
    P(outspeed) under the hidden-set belief, then 3 opp-active recovery scalars. Defensive: any
    missing field (no opp active, opp not in priors, our mon without stats) degrades to zeros for
    that piece. ``live`` (LiveView) supplies our screens + the weather via the strict-API boundary.
    """
    threat = _attacker_threat(getattr(battle, "opponent_active_pokemon", None), live)
    active = getattr(battle, "active_pokemon", None)
    defenders = [_defender(our_team[i] if i < len(our_team) else None, active)
                 for i in range(TEAM_SIZE)]
    return inc.compute_team_block(defenders, threat, TEAM_SIZE)
