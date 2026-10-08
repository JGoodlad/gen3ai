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

**What lives here now.** The per-mon belief block is written by the Rust encoder (its wake tables are
`layout.rs`'s ``SLEEP_*``); the Python writer (``sleep_belief_features``, the event-log source fold
``build_sleep_sources``, ``early_bird_probability``) is DELETED with the Python encoder's encode path (T27 P6
slice 6d-2). Kept: the verified tables, ``expected_free_turns`` (the model's C2 sleep-consequence kernel,
``agents.model.damage_op``) and the pure ``sleep_wake_probability`` the tables are tested through.
"""
from __future__ import annotations

from typing import Tuple


# --- Verified P(wake | observed counter K) tables (index by K, clamped to _MAX_K) -------------
# Unreachable rows are 1.0 sentinels; the reliability bit / clamp keep them from lying.
_OPP_NOEB: Tuple[float, ...] = (0.0, 0.25, 1.0 / 3.0, 0.5, 1.0, 1.0)
_OPP_EB: Tuple[float, ...] = (0.25, 2.0 / 3.0, 1.0, 1.0, 1.0, 1.0)
_REST_NOEB: Tuple[float, ...] = (0.0, 0.0, 1.0, 1.0, 1.0, 1.0)
_REST_EB: Tuple[float, ...] = (0.0, 1.0, 1.0, 1.0, 1.0, 1.0)
_MAX_K = 5


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
