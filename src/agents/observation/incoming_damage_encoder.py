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

# Smogon-usage prior knobs for the unrevealed-slot candidate moves. The floor is deliberately LOW
# and the cap generous so a low-usage but super-effective coverage move (e.g. a 4× Hidden Power)
# survives into the pool — the real "is this a threat vs THIS defender" gate is the per-defender max
# over p_in_set·P(KO|move) in incoming_damage._channel_threat (P(KO) embeds type effectiveness), so
# extra low-usage candidates only ever surface a genuine SE threat; they can't inflate a neutral one.
_PRIOR_MOVE_MIN_P = 0.05          # drop prior moves below this P(in set) — usage noise, not a threat
_MAX_CANDIDATES_PER_CHANNEL = 6   # cap per phys/spec channel (covers STAB + the popular coverage)
_OFFENSIVE_TAIL_Q = 0.95          # offensive-stat tail percentile for the KO magnitude (the max-EV+
                                  # tail; expected-damage re-normalises to the mean, so raising this
                                  # lifts P(KO) on near-OHKOs WITHOUT inflating the chip belief)
_FALLBACK_BASE = 80               # neutral base stat when a species has no dex entry
_FALLBACK_EV = 252                # the no-prior fallback assumes a max-invested neutral spread

# Rest is the recovery discriminator (cures the Toxic clock AND fully heals → an unbreakable wall
# for a chip-based team), so it gets its own ``cures_status`` scalar.
_REST = "rest"

# Explosion / Self-Destruct halve the target's Def in Gen 3 — the math core prices this off a
# per-Candidate flag, so the move-id → flag classification lives here (the core stays id-agnostic).
_HALVE_DEF_MOVES = frozenset({"explosion", "selfdestruct"})

# Variable-power moves that read base_power 0 in the dex (power is computed live from happiness):
# Return/Frustration cap at 102 BP, and a maxed-happiness Return is the STAB workhorse of many
# physical sets — without a power they were silently dropped (a "no threat" that can OHKO). Priced
# at the competitive max; the full damage formula (Atk/Def/type/STAB/roll) still applies.
_VARIABLE_POWER: dict[str, int] = {"return": 102, "frustration": 102}

# The bare (un-typed) Hidden Power id. poke-env reveals HP as this generic id (the type is inferred
# from observed effectiveness), and it reads base_power 0 → was dropped. We expand it into per-type
# candidates (priced from the typed dex variants, ~70 BP) so HP Ice/Grass/etc. coverage is visible.
_HIDDEN_POWER = "hiddenpower"
_HIDDEN_POWER_FALLBACK_BP = 60    # if a typed dex variant is somehow missing, price HP at ~60 BP
_HP_TYPE_MIN_P = 0.02             # drop negligible HP-type mass (a usage prior has a long ~0 tail);
                                  # keeps the expansion to the few real coverage types (perf)


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
    """Does move ``mid`` deal damage we should price? A positive-BP move, a fixed-damage move
    (Seismic Toss/… read 0 BP in the dex but hit for a constant), or a variable-power move
    (Return/Frustration, also 0 BP in the dex)."""
    return md is not None and ((md.base_power or 0) > 0 or mid in inc.FIXED_DAMAGE
                               or mid in _VARIABLE_POWER)


def _make_candidate(mid: str, md, p_in_set: float) -> inc.Candidate:
    """Build a :class:`incoming_damage.Candidate` for move ``mid`` at probability ``p_in_set``.
    Variable-power moves (Return/Frustration) substitute their competitive-max BP for the dex 0."""
    power = _VARIABLE_POWER.get(mid, int(md.base_power or 0))
    return inc.Candidate(
        md.type, power, float(p_in_set),
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


def _hp_type_dist(species: str, hp_tracker) -> dict:
    """``{type_name: P(HP type)}`` for a revealed Hidden Power on ``species`` — the tracker's
    observation-narrowed distribution if it has one (the per-episode signal: e.g. a 2× HP Ice on a
    Dragon already rules out the non-Ice types), else the species' Smogon HP-type prior. Normalised
    to sum 1 (HP IS in the set when revealed; only the type is uncertain)."""
    dist: dict = {}
    if hp_tracker is not None and species:
        probs = hp_tracker.get_probs(species)
        if probs is not None and getattr(probs, "sum", lambda: 0.0)() > 0:
            from agents.training.hidden_power_tracker import HIDDEN_POWER_TYPE_ORDER
            dist = {HIDDEN_POWER_TYPE_ORDER[i].name.lower(): float(probs[i])
                    for i in range(len(HIDDEN_POWER_TYPE_ORDER)) if probs[i] > 0}
    if not dist:
        dist = {k: float(v) for k, v in gen3_data.priors.hidden_power(species).items()}
    total = sum(dist.values())
    return {k: v / total for k, v in dist.items()} if total > 0 else {}


def _hidden_power_candidates(species: str, hp_tracker, p_total: float) -> list:
    """Expand a Hidden Power of total set-probability ``p_total`` into per-type ``Candidate``s.
    Each is priced from its typed dex variant (``hiddenpower<type>``, ~70 BP, correct gen3 category)
    at ``p_total · P(type)``, so a revealed bare ``hiddenpower`` (dex BP 0 → previously dropped)
    makes its Ice/Grass/Fighting coverage visible. The per-defender max in ``_channel_threat`` then
    surfaces whichever HP type is super-effective vs each of our mons."""
    out: list = []
    for type_name, prob in _hp_type_dist(species, hp_tracker).items():
        if prob < _HP_TYPE_MIN_P:
            continue
        mid = _HIDDEN_POWER + type_name
        md = gen3_data.moves.get(mid)
        if md is None:
            continue
        power = int(md.base_power or 0) or _HIDDEN_POWER_FALLBACK_BP
        out.append(inc.Candidate(md.type, power, float(p_total * prob)))
    return out


def _candidates(opp_active, species: str, hp_tracker=None) -> Tuple[list, list]:
    """Physical / special candidate moves vs us = revealed (P=1, per decision) ∪ cached prior
    moves. Fixed-damage moves (Seismic Toss/…) carry constant damage despite the dex STATUS tag; a
    revealed bare ``hiddenpower`` is expanded into per-type candidates (its type is hidden, so it
    reads 0 BP in the dex) via the HP tracker / prior."""
    phys: list = []
    spec: list = []
    for mid in (getattr(opp_active, "moves", {}) or {}):
        if mid == _HIDDEN_POWER:   # revealed but un-typed → expand into per-type candidates
            for cand in _hidden_power_candidates(species, hp_tracker, 1.0):
                (phys if inc.type_is_physical(cand.move_type) else spec).append(cand)
            continue
        md = gen3_data.moves.get(mid)
        if _is_damaging(mid, md):
            cand = _make_candidate(mid, md, 1.0)
            (phys if inc.type_is_physical(md.type) else spec).append(cand)
    pp, sp = _prior_candidates(species)
    return phys + list(pp), spec + list(sp)


def _attacker_threat(opp_active, live, hp_tracker=None) -> Optional[inc.AttackerThreat]:
    """The opponent active as a belief (None if there is no opp active / no species yet): types
    known, offensive stats as usage tail+mean (boost folded in), candidate moves split by channel,
    plus our screens / the weather (via the LiveView read-model) and the recovery scalars.
    ``hp_tracker`` (optional) types a revealed Hidden Power from its narrowed distribution."""
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
    phys, spec = _candidates(opp_active, species, hp_tracker)

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


def encode_block(battle, our_team: List, live=None, hp_tracker=None) -> np.ndarray:
    """The incoming-damage / OHKO belief block (incoming_damage_v1) for one decision.

    Per our mon (slot-aligned to the team): the opponent active's phys/spec expected-chip + P(KO) +
    P(outspeed) under the hidden-set belief, then 3 opp-active recovery scalars. Defensive: any
    missing field (no opp active, opp not in priors, our mon without stats) degrades to zeros for
    that piece. ``live`` (LiveView) supplies our screens + the weather via the strict-API boundary;
    ``hp_tracker`` (optional) types a revealed Hidden Power from its observation-narrowed prior.
    """
    threat = _attacker_threat(getattr(battle, "opponent_active_pokemon", None), live, hp_tracker)
    active = getattr(battle, "active_pokemon", None)
    defenders = [_defender(our_team[i] if i < len(our_team) else None, active)
                 for i in range(TEAM_SIZE)]
    return inc.compute_team_block(defenders, threat, TEAM_SIZE)
