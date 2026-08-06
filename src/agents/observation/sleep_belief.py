"""gen3_sleep_wake_belief_v1 — the per-mon sleep WAKE belief.

poke-env exposes only ``Status.SLP`` + a noisy turn counter (``status_counter``); it does NOT
expose the rolled sleep duration, the remaining time, or the source move (Rest vs Spore). So a
policy reading the raw counter has to *learn* the gen3 sleep RNG — exactly the kind of "annoying
edge case" we'd rather COMPUTE and hand it. This module turns the observable counter + the
event-log sleep source + the (own-known / opp-prior) Early Bird ability into a calibrated
**P(wake on the next move attempt)**, plus two interpretable bits.

Verified gen3 mechanics (``deps/pokemon-showdown/data/mods/gen3/conditions.ts`` slp +
``data/moves.ts`` rest; cross-checked against Bulbapedia gen3 sleep + an adversarial
turn-by-turn re-simulation):

* Move/opponent sleep sets ``time = random(2,6)`` ∈ {2,3,4,5} uniform (the gen3 mod OVERRIDES the
  modern base ``random(2,5)``) → 1–4 forced-asleep turns.
* **Rest** sets ``time = 3`` fixed (deterministic) → 2 forced-asleep turns.
* ``onBeforeMove`` decrements ``time`` (TWICE with Early Bird → halved, rounded down) then wakes &
  acts when ``time <= 0`` — emitting NO ``|cant|`` on the wake turn, so poke-env's counter never
  reaches the wake value. Counter ``K`` = number of cant-turns already observed (still asleep), and
  the belief is P(wake on the next, i.e. (K+1)-th, ``onBeforeMove``) = P(T = K+1) / P(T ≥ K+1).

The four resulting tables (counter K → P(wake), unreachable rows held at 1.0 as a sentinel that the
reliability bit guards):

================  K=0    K=1    K=2   K=3  K=4  K≥5
opp,   no EB     0      1/4    1/3   1/2  1    1
opp,   Early Bird 1/4   2/3    1     1    1    1
Rest,  no EB     0      0      1     1    1    1
Rest,  Early Bird 0     1      1     1    1    1

**Sleep Talk / Snore caveat (empirically verified):** a sleep-usable-move turn ticks poke-env's
counter by +3 (one ``|cant|`` + two ``|move|`` lines), so K is corrupted for those episodes. We do
NOT reconstruct Showdown's ``skippedTime`` switch refund (rare; the opponent's true ``time`` is
hidden anyway). Instead a ``sleep_counter_reliable`` bit goes to 0 the moment a sleep-usable move is
seen this episode, so the model can discount the (now-unreliable) P(wake) scalar.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from agents.battle.battle_event import EventKind, OURS, OPP
from agents.gen3_data import priors as _priors

# --- Verified P(wake | observed counter K) tables (index by K, clamped to _MAX_K) -------------
# Unreachable rows are 1.0 sentinels; the reliability bit / clamp keep them from lying.
_OPP_NOEB: Tuple[float, ...] = (0.0, 0.25, 1.0 / 3.0, 0.5, 1.0, 1.0)
_OPP_EB: Tuple[float, ...] = (0.25, 2.0 / 3.0, 1.0, 1.0, 1.0, 1.0)
_REST_NOEB: Tuple[float, ...] = (0.0, 0.0, 1.0, 1.0, 1.0, 1.0)
_REST_EB: Tuple[float, ...] = (0.0, 1.0, 1.0, 1.0, 1.0, 1.0)
_MAX_K = 5

_SLEEP_USABLE_MOVES = frozenset({"sleeptalk", "snore"})
_EARLY_BIRD = "earlybird"
_UNKNOWN_ABILITY = "unknownability"


def expected_free_turns(is_rest: bool, p_earlybird: float) -> float:
    """E[number of FULL turns a freshly-slept mon cannot act], derived FROM the verified hazard
    tables above (never hand-asserted — the C2 sleep-consequence kernel's one source):
    E = Σ_k P(free ≥ k) over the survival curve of the wake hazards at counters 0,1,2,….
    Opp sleep no-EB = 2.5, with Early Bird = 1.0, Rest no-EB = 2.0 exactly; marginalised
    linearly over ``p_earlybird`` (expectation of a mixture)."""
    def _e(table: Tuple[float, ...]) -> float:
        surv, e = 1.0, 0.0
        for h in table:
            surv *= (1.0 - h)
            e += surv
            if surv <= 0.0:
                break
        return e

    if is_rest:
        return (1.0 - p_earlybird) * _e(_REST_NOEB) + p_earlybird * _e(_REST_EB)
    return (1.0 - p_earlybird) * _e(_OPP_NOEB) + p_earlybird * _e(_OPP_EB)


def sleep_wake_probability(counter: int, is_rest: bool, p_earlybird: float) -> float:
    """P(the mon wakes & can act on its NEXT move attempt) given the observed poke-env sleep
    ``counter`` (cant-turns already seen, still asleep), whether the sleep is Rest-deterministic,
    and P(Early Bird). Pure: no poke-env / battle objects, so it unit-tests against the 4 tables.

    For the OPPONENT, Early Bird is only a Smogon prior, so we marginalise:
    ``p = P(EB)·table_EB[K] + (1-P(EB))·table_noEB[K]``. For our own mon (and a revealed opp) the
    prior collapses to 0/1, picking one table exactly.
    """
    k = 0 if counter < 0 else (counter if counter <= _MAX_K else _MAX_K)
    eb, noeb = (_REST_EB, _REST_NOEB) if is_rest else (_OPP_EB, _OPP_NOEB)
    p = float(p_earlybird)
    if p <= 0.0:
        return noeb[k]
    if p >= 1.0:
        return eb[k]
    return p * eb[k] + (1.0 - p) * noeb[k]


def early_bird_probability(mon: Any) -> float:
    """P(``mon`` has Early Bird). Own team / revealed opponent → exact 1.0/0.0 from ``mon.ability``;
    an unrevealed opponent → the species' Smogon ``earlybird`` usage prior (≈0 for almost every
    gen3ou mon, so the no-EB table dominates in practice)."""
    ability = getattr(mon, "ability", None)
    if ability and str(ability) != _UNKNOWN_ABILITY:
        return 1.0 if str(ability) == _EARLY_BIRD else 0.0
    species = getattr(mon, "species", None)
    if not species:
        return 0.0
    return float(_priors.ability(species).get(_EARLY_BIRD, 0.0))


def _reason_is_rest(reason: Optional[str]) -> bool:
    """True iff a slp STATUS event's ``[from]`` reason names Rest (``move: rest`` or the bundled
    bare ``rest``). Rest is the only gen3 self-cure sleep with a FIXED duration → deterministic."""
    if not reason:
        return False
    r = str(reason).lower()
    if ":" in r:
        r = r.split(":", 1)[1]
    return r.strip().replace(" ", "") == "rest"


def build_sleep_sources(battle: Any) -> Dict[Tuple[str, str], Tuple[bool, bool]]:
    """Fold the whole-battle event log into ``{(side, species): (is_rest_sleep,
    sleep_usable_move_seen)}`` for each mon's CURRENT sleep episode. One backward-agnostic linear
    pass over ``battle.events`` (gen3 STATUS events carry the ``[from]`` source poke-env discards).
    Callers gate this on "is any mon asleep?" so it costs nothing on the common no-sleep decision.

    ``is_rest_sleep`` selects the deterministic Rest table; ``sleep_usable_move_seen`` drives the
    reliability bit (the +3 counter-noise guard). Keyed by the most-recent slp application per mon,
    so a re-sleep after waking is tracked correctly.
    """
    events = getattr(battle, "events", None)
    if not events:
        return {}
    slp_seq: Dict[Tuple[str, str], int] = {}
    is_rest: Dict[Tuple[str, str], bool] = {}
    for e in events:
        if e.kind is EventKind.STATUS and e.status == "slp" and e.side and e.actor_species:
            key = (e.side, e.actor_species)
            slp_seq[key] = e.seq                       # most-recent slp application wins
            is_rest[key] = _reason_is_rest(e.reason)
    if not slp_seq:
        return {}
    usable: Dict[Tuple[str, str], bool] = {k: False for k in slp_seq}
    for e in events:
        if e.kind is EventKind.MOVE and e.side and e.actor_species and e.move_id in _SLEEP_USABLE_MOVES:
            key = (e.side, e.actor_species)
            seq0 = slp_seq.get(key)
            if seq0 is not None and e.seq > seq0:      # a sleep-usable move SINCE this sleep applied
                usable[key] = True
    return {k: (is_rest[k], usable[k]) for k in slp_seq}


def sleep_belief_features(
    counter: int, mon: Any, is_own: bool, sleep_sources: Optional[Dict[Tuple[str, str], Tuple[bool, bool]]]
) -> Tuple[float, float, float]:
    """The 3 per-mon obs features for a CURRENTLY-ASLEEP mon, in obs order:
    ``(sleep_is_deterministic, p_wake, sleep_counter_reliable)``.

    - ``sleep_is_deterministic`` = 1.0 when the sleep is Rest (a known fixed schedule), else 0.0.
    - ``p_wake`` = :func:`sleep_wake_probability` over the observed counter, source and Early Bird.
    - ``sleep_counter_reliable`` = 0.0 once a Sleep Talk / Snore turn has corrupted the counter.

    ``sleep_sources`` is the map from :func:`build_sleep_sources` (None on the mock / non-Gen3Battle
    path → treated as a non-Rest, reliable, prior-only sleep, the common opponent case)."""
    side = OURS if is_own else OPP
    species = getattr(mon, "species", None)
    is_rest, usable_seen = (False, False)
    if sleep_sources and species is not None:
        is_rest, usable_seen = sleep_sources.get((side, species), (False, False))
    p_eb = early_bird_probability(mon)
    p_wake = sleep_wake_probability(counter, is_rest, p_eb)
    return (1.0 if is_rest else 0.0, p_wake, 0.0 if usable_seen else 1.0)
