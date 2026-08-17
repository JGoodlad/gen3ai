"""LiveView → incoming-damage belief block (the glue over the pure ``incoming_damage`` math core).

Single public entry point: :func:`encode_block`. It reads the current board **only** through the
vetted :class:`~agents.battle.live_view.LiveView` read-model (no raw poke-env battle), turns the
hidden opponent set into a ``Candidate`` / ``AttackerThreat`` belief (revealed moves ∪ Smogon-usage
priors; offensive stats as a usage distribution), and folds the per-defender KO/chip/outspeed belief
through ``incoming_damage.compute_team_block``. Everything else here is private.

The math is deliberately kept poke-env-free in ``incoming_damage.py``; this module owns the *only*
data-facade reads (the usage stat distribution + prior candidate moves), all of them per-species and
``lru_cache``d — which is what keeps the block inside the obs-build benchmark budget. Sourcing the
board from ``LiveView`` (primitives) instead of the raw battle keeps the whole obs+reward belief on
the strict-API read-model: the SAME ``LiveView`` already built per decision feeds both the obs path
(``reactive.py``) and the reward-shaping path (``reward_manager.py`` PBRS), with no duplicate raw
reads. ``hp_tracker`` (optional) types a revealed Hidden Power from its observation-narrowed prior;
the obs path threads it, the reward path passes None (falls back to the Smogon HP-type prior).

Design: `designs/ai_v5/design_incoming_damage_obs.md`.
"""
from __future__ import annotations

import functools
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import numpy as np

from agents import gen3_data
from agents.gen3_data.moves import MoveData
from agents.enums import PokemonType, Status
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


def _ptype(name: Optional[str]) -> Optional[PokemonType]:
    """LiveView type-id string (e.g. ``'fire'``) → ``PokemonType`` enum (None passes through).
    Inverse of LiveView's ``_enum_name`` (``PokemonType.FIRE`` ↔ ``'fire'``), so the belief sees the
    exact same enum the raw-``mon.type_1`` path used — byte-identical."""
    return PokemonType[name.upper()] if name else None


def _pstatus(name: Optional[str]) -> Optional[Status]:
    """LiveView status-id string (e.g. ``'par'``) → ``Status`` enum (None passes through). Only FRZ
    changes an effectiveness result (Flash Fire) and PAR folds into P(outspeed); the rest are inert
    to the belief but converted for consistency."""
    return Status[name.upper()] if name else None


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


def _is_damaging(mid: str, md: Optional[MoveData]) -> bool:
    """Does move ``mid`` deal damage we should price? A positive-BP move, a fixed-damage move
    (Seismic Toss/… read 0 BP in the dex but hit for a constant), or a variable-power move
    (Return/Frustration, also 0 BP in the dex)."""
    return md is not None and ((md.base_power or 0) > 0 or mid in inc.FIXED_DAMAGE
                               or mid in _VARIABLE_POWER)


def _make_candidate(mid: str, md: MoveData, p_in_set: float) -> inc.Candidate:
    """Build a :class:`incoming_damage.Candidate` for move ``mid`` at probability ``p_in_set``.
    Variable-power moves (Return/Frustration) substitute their competitive-max BP for the dex 0."""
    power = _VARIABLE_POWER.get(mid, int(md.base_power or 0))
    return inc.Candidate(
        md.type, power, float(p_in_set),
        fixed_dmg=inc.FIXED_DAMAGE.get(mid),
        halves_defense=(mid in _HALVE_DEF_MOVES),
    )


@functools.lru_cache(maxsize=None)
def _prior_candidates(species: str) -> Tuple[Tuple[inc.Candidate, ...], Tuple[inc.Candidate, ...]]:
    """Cached per-species prior (unrevealed-slot) candidate moves, split by gen3 type-category.
    Static per species → computed once. Revealed moves are merged in per decision by
    :func:`_candidates`; duplicates are harmless because the belief takes the max over candidates
    (the revealed P=1 copy wins)."""
    phys: List[inc.Candidate] = []
    spec: List[inc.Candidate] = []
    pc = sc = 0
    for p, mid in sorted(((p, mid) for mid, p in gen3_data.priors.moves(species).items()
                          if p >= _PRIOR_MOVE_MIN_P), reverse=True):
        md = gen3_data.moves.get(mid)
        if not _is_damaging(mid, md):
            continue
        # `_is_damaging` is True only when `md is not None`, but that guard lives INSIDE the
        # helper, so no narrowing reaches here. `cast` is the erasure-free narrowing tool — it
        # returns its argument unchanged, so the emitted candidate stays byte-identical.
        cand = _make_candidate(mid, cast(MoveData, md), p)
        if inc.type_is_physical(cast(MoveData, md).type):
            if pc < _MAX_CANDIDATES_PER_CHANNEL:
                pc += 1
                phys.append(cand)
        elif sc < _MAX_CANDIDATES_PER_CHANNEL:
            sc += 1
            spec.append(cand)
    return tuple(phys), tuple(spec)


def _hp_type_dist(species: str, hp_tracker: Any) -> Dict[str, float]:
    """``{type_name: P(HP type)}`` for a revealed Hidden Power on ``species`` — the tracker's
    observation-narrowed distribution if it has one (the per-episode signal: e.g. a 2× HP Ice on a
    Dragon already rules out the non-Ice types), else the species' Smogon HP-type prior. Normalised
    to sum 1 (HP IS in the set when revealed; only the type is uncertain)."""
    dist: Dict[str, float] = {}
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


def _hidden_power_candidates(species: str, hp_tracker: Any, p_total: float) -> List[inc.Candidate]:
    """Expand a Hidden Power of total set-probability ``p_total`` into per-type ``Candidate``s.
    Each is priced from its typed dex variant (``hiddenpower<type>``, ~70 BP, correct gen3 category)
    at ``p_total · P(type)``, so a revealed bare ``hiddenpower`` (dex BP 0 → previously dropped)
    makes its Ice/Grass/Fighting coverage visible. The per-defender max in ``_channel_threat`` then
    surfaces whichever HP type is super-effective vs each of our mons."""
    out: List[inc.Candidate] = []
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


def _candidates(revealed_move_ids: Sequence[str], species: str,
                hp_tracker: Any = None) -> Tuple[List[inc.Candidate], List[inc.Candidate]]:
    """Physical / special candidate moves vs us = revealed (P=1, per decision) ∪ cached prior
    moves. Fixed-damage moves (Seismic Toss/…) carry constant damage despite the dex STATUS tag; a
    revealed bare ``hiddenpower`` is expanded into per-type candidates (its type is hidden, so it
    reads 0 BP in the dex) via the HP tracker / prior. The belief takes the max over candidates, so
    the revealed/prior order does not affect the output."""
    phys: List[inc.Candidate] = []
    spec: List[inc.Candidate] = []
    for mid in revealed_move_ids:
        if mid == _HIDDEN_POWER:   # revealed but un-typed → expand into per-type candidates
            for cand in _hidden_power_candidates(species, hp_tracker, 1.0):
                (phys if inc.type_is_physical(cand.move_type) else spec).append(cand)
            continue
        md = gen3_data.moves.get(mid)
        if _is_damaging(mid, md):
            # See `_prior_candidates`: `_is_damaging` owns the not-None guard, so narrow with
            # a no-op `cast` rather than re-testing (which would change the code, not its types).
            cand = _make_candidate(mid, cast(MoveData, md), 1.0)
            (phys if inc.type_is_physical(cast(MoveData, md).type) else spec).append(cand)
    pp, sp = _prior_candidates(species)
    return phys + list(pp), spec + list(sp)


def _attacker_threat(live: Any, hp_tracker: Any = None) -> Optional[inc.AttackerThreat]:
    """The opponent active as a belief (None if there is no opp active / no species yet): types
    known, offensive stats as usage tail+mean (boost folded in), candidate moves split by channel,
    plus our screens / the weather, all from the LiveView read-model, and the recovery scalars.
    ``hp_tracker`` (optional) types a revealed Hidden Power from its narrowed distribution."""
    opp = live.opp.active
    if opp is None:
        return None
    species = opp.species
    if not species:
        return None
    boosts = opp.boosts or {}
    atk_b, spa_b = inc.boost_mult(boosts.get("atk", 0)), inc.boost_mult(boosts.get("spa", 0))
    atk_tail, atk_mean, _ = _offensive_stat(species, "atk")
    spa_tail, spa_mean, _ = _offensive_stat(species, "spa")
    _, _, spe_dist = _offensive_stat(species, "spe")
    revealed_ids = opp.move_ids
    phys, spec = _candidates(revealed_ids, species, hp_tracker)

    # Recovery scalars (Suicune-Rest discriminator): revealed move → certain; else its usage prior.
    prior_mv = gen3_data.priors.moves(species)
    revealed = set(revealed_ids)
    rate = min(1.0, sum(1.0 if rm in revealed else prior_mv.get(rm, 0.0) for rm in RECOVERY_MOVES))
    cures = 1.0 if _REST in revealed else prior_mv.get(_REST, 0.0)
    known = 1.0 if revealed & RECOVERY_MOVES else 0.0

    # Field via the read-model — our screens + the weather. None on the no-LiveView path.
    oursc = live.ours.side_conditions or {}
    reflect = "reflect" in oursc
    lscreen = "light_screen" in oursc
    lw = getattr(live, "weather", None)
    weather = getattr(lw, "weather", None) if lw is not None else None

    opp_types = tuple(t for t in (_ptype(opp.types[0]) if opp.types else None,
                                  _ptype(opp.types[1]) if len(opp.types) > 1 else None)
                      if t is not None)
    status = opp.status
    return inc.AttackerThreat(
        types=opp_types,
        atk_tail=atk_tail * atk_b, atk_mean=atk_mean * atk_b,
        spa_tail=spa_tail * spa_b, spa_mean=spa_mean * spa_b,
        spe_dist=spe_dist, boost_spe=int(boosts.get("spe", 0)),
        para=(status == "par"), burn=(status == "brn"),
        phys=phys, spec=spec, our_reflect=reflect, our_light_screen=lscreen, weather=weather,
        recovery_rate=rate, cures_status=float(cures), recovery_known=known,
    )


def _defender(lm: Any) -> Optional[inc.Defender]:
    """One of our mons as an exact ``Defender`` (None if absent / fainted / stats not yet known),
    built from its :class:`~agents.battle.live_view.LivePokemon`. ``status`` is converted to the raw
    :class:`~agents.enums.Status` enum; ``has_sub`` is only set for the active mon (a Substitute eats
    the incoming hit, so a benched mon can't be 'behind' one)."""
    if lm is None or lm.fainted:
        return None
    stats = lm.stats or {}
    d, sd, spe = stats.get("def"), stats.get("spd"), stats.get("spe")
    hp, hpx = lm.current_hp, lm.max_hp
    if not (d and sd and spe and hp and hpx):
        return None
    boosts = lm.boosts or {}
    has_sub = lm.active and lm.has_volatile("substitute")
    return inc.Defender(
        def_stat=int(d), spd_stat=int(sd), hp_remaining=int(hp), hp_max=int(hpx), spe=int(spe),
        # ⚠️ `Defender.type1` is declared NON-optional and `effective_multiplier_by_types`
        # requires it, but this expression yields None for a typeless `lm` — a LATENT
        # mismatch, not a behaviour change made here, so it is DECLARED rather than
        # silently repaired. Believed unreachable (a LivePokemon always carries ≥1 type);
        # if it ever fires, the fix belongs in `_defender`, not in this annotation.
        type1=_ptype(lm.types[0]) if lm.types else None,  # type: ignore[arg-type]
        type2=_ptype(lm.types[1]) if len(lm.types) > 1 else None,
        ability=lm.ability,
        status=_pstatus(lm.status), boost_def=int(boosts.get("def", 0)),
        boost_spd=int(boosts.get("spd", 0)), boost_spe=int(boosts.get("spe", 0)), has_sub=has_sub,
    )


def encode_block(live: Any, hp_tracker: Any = None) -> np.ndarray:
    """The incoming-damage / OHKO belief block (incoming_damage_v2) for one decision, from the
    :class:`~agents.battle.live_view.LiveView` read-model.

    Per our mon (slot-aligned to ``live.ours.mons``, which mirrors ``get_team_list`` order): the
    opponent active's phys/spec expected-chip + P(KO) + P(outspeed) under the hidden-set belief, then
    3 opp-active recovery scalars. Defensive: ``live`` is None / no opp active / our mon without
    stats degrades to zeros for that piece. ``hp_tracker`` (optional) types a revealed Hidden Power
    from its observation-narrowed prior (obs path threads it; the reward PBRS passes None → the
    Smogon HP-type prior)."""
    if live is None:
        return np.zeros(TEAM_SIZE * inc.PER_MON + inc.RECOVERY, dtype=np.float32)
    threat = _attacker_threat(live, hp_tracker)
    mons = live.ours.mons
    defenders = [_defender(mons[i]) if i < len(mons) else None for i in range(TEAM_SIZE)]
    return inc.compute_team_block(defenders, threat, TEAM_SIZE)
