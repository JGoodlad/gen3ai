"""Loss attribution — categorize a loss's DECISIVE turning point into a fixed taxonomy.

The taxonomy is the single place to extend: one `_Cat` entry (a name, the LEVER it implicates, a
blurb, and a predicate over the feature dict). `attribute_turning_point` assigns the FIRST match,
so the buckets are non-overlapping by construction.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Loss attribution — categorize the DECISIVE turning point of a loss into a
# fixed taxonomy, so a whole run's losses can be RANKED by which lever (obs /
# reward / self-play / critic / upstream) would recover the most rating.
# ---------------------------------------------------------------------------
# The taxonomy is the single place to extend: add one `_Cat` entry (a name, the
# LEVER it implicates, a one-line blurb, and a predicate over the feature dict).
# `attribute_turning_point` assigns the FIRST matching category (ordered
# most-diagnostic-first), so the buckets are non-overlapping by construction.
#
# A feature dict (built model-free by ProbeSession._turning_point_features from
# the saved summary + npz at the worst-ΔV decision) carries:
#   turns:int|None  is_switch:bool  is_setup:bool  our_hp:float|None
#   our_hp_delta:float|None  faint:bool  active_pko:float|None
#   active_outspeed:float|None  max_pko:float|None  n_healthy_bench:int
#   min_other_pko:float|None  delta_v:float|None  td:float|None  v_at:float|None
#   wp_at:float|None        recorded P(win) at the cliff (win-prob head); the CALIBRATED winning-vs-
#                           losing signal that re-centers the grind/throw split (see `_was_winning`)
#   wp_even / v_even        the winning thresholds triage stamps on (defaults WP_EVEN_DEFAULT / 0.0)
# Every field is optional-tolerant: a predicate must treat None as "unknown"
# (never assume), so a trace missing the belief block still categorizes (it
# just falls through to a coarser bucket) rather than crashing.

STALL_NEAR_CAP = 220          # a loss at/near the 250-turn forfeit cap = a no-progress timeout, not combat
BELIEF_FIRED_PKO = 0.6        # active_pko at/above this = the OHKO belief WARNED of the threat
BELIEF_UNDERREAD_PKO = 0.3    # active_pko below this on a healthy-mon death = the belief MISSED it
HEALTHY_HP = 0.6              # our active counts as "healthy" (a real mon thrown away) above this HP
FAINTED_HP = 0.02            # our active's PRE-decision HP at/below this = already fainted (forced replacement)
CRITIC_CONFIDENT_V = 0.0     # FALLBACK winning threshold on V when no win-prob recorded. NB: V's zero is NOT
                             # "even" — V is a shaped/discounted RETURN with a structural NEGATIVE offset (a
                             # measured self-mirror 50/50 reads V≈−6.5; PopArt μ≈−3.6), so V>0 OVER-counts grinds.
                             # Prefer the calibrated P(win) split (`_was_winning`); re-center this only via v_even.
WP_EVEN_DEFAULT = 0.5        # PRIMARY winning threshold: P(win) at the cliff ≥ this = the model rated the position
                             # WINNING. Calibrated win-odds (the win-prob head), so it is correctly centered at 0.5
                             # — unlike V's sign. (The head can carry a small absolute optimism bias; pass wp_even to
                             # de-bias by a per-checkpoint self-mirror offset if you have one.)

# Common gen3 boosting/setup moves — used at the decisive turn = "set up into a threat" (greedy).
SETUP_MOVES = frozenset({
    "dragondance", "swordsdance", "calmmind", "nastyplot", "bulkup", "curse", "agility",
    "irondefense", "amnesia", "growth", "meditate", "sharpen", "acidarmor", "barrier",
    "cosmicpower", "bellydrum", "tailglow", "rockpolish", "shellsmash", "workup",
})


@dataclass(frozen=True)
class _Cat:
    name: str
    lever: str          # which system axis a fix would touch (the prioritization output)
    blurb: str
    test: "callable"    # (feat) -> bool


def _f(feat, k):
    """feat[k] or None — tolerant of missing keys."""
    return feat.get(k)


def _was_winning(f) -> bool:
    """Did the critic rate the position as WINNING right before the value cratered? This is the
    grind-vs-throw boundary, and it must NOT be the sign of V: V is a shaped/discounted return with a
    structural negative offset (a self-mirror 50/50 reads V≈−6.5), so V>0 systematically UNDER-counts
    "was winning" and over-attributes losses to `positional_grind`. So PREFER the calibrated win-prob
    head — P(win) ≥ wp_even (default 0.5) — and fall back to V > v_even (default 0, re-centerable via the
    structural even-point) only when no win-prob was recorded. Returns False on unknown (no signal)."""
    wp = _f(f, "wp_at")
    if wp is not None:
        return wp >= f.get("wp_even", WP_EVEN_DEFAULT)
    v = _f(f, "v_at")
    return v is not None and v > f.get("v_even", CRITIC_CONFIDENT_V)


# Ordered most-specific / most-diagnostic first. First match wins. The death buckets
# (surprise / ignored / doomed / attrition) are split to separate the LEVERS: a death the
# belief UNDER-read is an OBS gap; a death the belief FIRED on (we had a pivot, didn't take it)
# is a POLICY/REWARD gap; a death with no pivot left is UPSTREAM; the rest is attrition.
LOSS_TAXONOMY = (
    _Cat("stall_timeout", "self-play / stall reward (Φ-price heal moves in the mirror)",
         "lost at/near the 250-turn cap — a no-progress timeout, not a combat loss",
         lambda f: (_f(f, "turns") or 0) >= STALL_NEAR_CAP),

    _Cat("post_faint_replacement",
         "MEASUREMENT/UPSTREAM (worst-ΔV is a forced post-faint pick — the causal turn is earlier; re-scan turn N-1)",
         "our active had ALREADY fainted — this is a forced replacement, not the decision that lost the mon",
         lambda f: _f(f, "our_hp") is not None and f["our_hp"] <= FAINTED_HP),

    _Cat("surprise_ohko", "obs (surprise-OHKO coverage — price unrevealed/just-switched threats)",
         "a HEALTHY mon DIED but the incoming belief UNDER-READ it (unseen / just-switched attacker)",
         lambda f: _f(f, "faint")
                   and (_f(f, "our_hp") is None or f["our_hp"] >= HEALTHY_HP)
                   and _f(f, "active_pko") is not None and f["active_pko"] < BELIEF_UNDERREAD_PKO),

    _Cat("ignored_threat_death",
         "reward/policy (belief FIRED but the policy didn't switch out — the under-switch / doomed_stay target)",
         "the incoming belief FIRED (high P(KO)) and we had a healthy pivot, yet our mon DIED (stayed or pivoted into it)",
         lambda f: _f(f, "faint")
                   and _f(f, "active_pko") is not None and f["active_pko"] >= BELIEF_FIRED_PKO
                   and (_f(f, "n_healthy_bench") or 0) >= 1),

    _Cat("doomed_already", "UPSTREAM (the loss was decided earlier — sequencing/material, look back)",
         "the belief fired high but NO healthy mon left to switch to — the position was already lost",
         lambda f: _f(f, "faint")
                   and _f(f, "active_pko") is not None and f["active_pko"] >= BELIEF_FIRED_PKO
                   and (_f(f, "n_healthy_bench") or 0) == 0),

    _Cat("greedy_setup", "reward/critic (anti-greedy: setup-into-threat + critic tail-blindness)",
         "the decisive move was a SETUP/boost move that got punished",
         lambda f: bool(_f(f, "is_setup"))),

    _Cat("attrition_death", "obs/critic (chip / partial-belief death — a worn-down mon died, belief only partly fired)",
         "our mon DIED with the belief only PARTLY fired (mid P(KO)) or already chipped below healthy — attrition, not a clean surprise",
         lambda f: bool(_f(f, "faint"))),

    _Cat("critic_blindspot", "critic capacity / obs (the critic rated the position WINNING then it craters — confident-wrong: more value capacity / a missing obs feature)",
         "no death this turn, but right before the cliff the model rated the position WINNING — P(win)≥0.5 "
         "(or, no win-prob head, V above its even-point) — a confident-wrong critic miss (a THROW, coachable)",
         lambda f: not _f(f, "faint") and _was_winning(f)),

    _Cat("positional_grind", "UPSTREAM / material (the model already knew it was behind — a slow positional / material loss, not a critic miss)",
         "no death this turn and the model ALREADY rated itself behind (P(win)<0.5, or V below its even-point) "
         "right before the cliff — a gradual positional / material grind (was never ahead to throw)",
         lambda f: not _f(f, "faint")),

    _Cat("other", "unattributed (drill in with analyze)", "did not match a known failure pattern",
         lambda f: True),
)


def attribute_turning_point(feat: dict) -> dict:
    """Assign one loss's decisive turning point to the FIRST matching taxonomy bucket.

    Returns ``{category, lever, blurb}``. Pure + total (the final ``other`` rule matches
    anything), so it never raises on a partial feature dict."""
    for cat in LOSS_TAXONOMY:
        try:
            if cat.test(feat):
                return {"category": cat.name, "lever": cat.lever, "blurb": cat.blurb}
        except Exception:  # noqa: BLE001 — a predicate must never crash the scan; treat as no-match
            continue
    return {"category": "other", "lever": LOSS_TAXONOMY[-1].lever, "blurb": LOSS_TAXONOMY[-1].blurb}
