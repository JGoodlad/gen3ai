"""The team-strength read's decision classes (boundaries, rule 8) and the matched-strength piloting gap: a gap that is
all team STRENGTH reads ~0 once matched, and a real familiarity gap survives the matching."""
from __future__ import annotations

import math

import numpy as np

from main.team_strength import fit as F
from main.team_strength import report as RP


def test_competitive_class_boundaries():
    assert RP.competitive_class([-80.0, -40.0], 35.0)["class"] == "MATERIALLY WEAKER"
    c = RP.competitive_class([-30.0, -5.0], 35.0)
    assert c["class"] == "COMPETITIVE" and c["detected_weaker"]
    assert not RP.competitive_class([-30.0, 10.0], 35.0)["detected_weaker"]
    assert RP.competitive_class([-60.0, 10.0], 35.0)["class"] == "UNDECIDED"
    assert RP.competitive_class([-35.0 + 1e-12, 5.0], 35.0)["class"] == "UNDECIDED"     # within 1e-9: rule 8


def test_agreement_and_gap_classes():
    assert RP.agreement_class([0.1, 0.6]) == "AGREE"
    assert RP.agreement_class([-0.1, 0.6]) == "NOT SHOWN"
    assert RP.agreement_class([1e-12, 0.6]) == "UNDECIDED"
    assert RP.gap_class([0.01, 0.2]) == "DETECTED"
    assert RP.gap_class([-0.05, 0.02]) == "GONE"
    assert RP.gap_class([-0.05, 0.10]) == "UNDECIDED"


def _rating(strength: dict, B: int = 200, seed: int = 0) -> F.Rating:
    """A Rating carrying ``strength`` exactly, with small replicate noise."""
    ids = sorted(strength)
    r = np.array([strength[t] for t in ids])
    rng = np.random.default_rng(seed)
    reps = r[None, :] + rng.normal(0, 0.05, (B, len(ids)))
    return F.Rating(ids=ids, r=r, reps=reps, games={t: 20 for t in ids}, score={t: 0.5 for t in ids}, seed=seed,
                    prior_sd=2.0, anchor=ids[:1])


def _games(strength: dict, unfamiliar: set, gap_logit: float, per_team: int, seed: int) -> list:
    rng = np.random.default_rng(seed)
    rows = []
    for t in sorted(strength):
        u = t in unfamiliar
        p = 1.0 / (1.0 + math.exp(-(strength[t] - (gap_logit if u else 0.0))))
        for _ in range(per_team):
            rows.append({"team": t, "unfamiliar": u, "win": bool(rng.random() < p)})
    return rows


def test_a_gap_that_is_all_strength_is_explained_and_a_real_one_survives():
    # unfamiliar teams are WEAKER (strength -0.6 .. 0.0) than familiar ones (0.0 .. +0.6), overlapping at 0
    fam = {f"f{i}": 0.6 * i / 19 for i in range(20)}
    unf = {f"u{i}": -0.6 + 0.6 * i / 19 for i in range(20)}
    strength = {**fam, **unf}
    rt = _rating(strength)
    none = RP.matched_gap(_games(strength, set(unf), 0.0, 200, 1), rt, B=200, seed=2)
    assert none["raw"] > 0.10                                    # the raw drop is large ...
    assert none["matched_ci"][0] < 0.0 < none["matched_ci"][1]   # ... and all of it is strength
    assert none["class"] in ("GONE", "UNDECIDED")
    real = RP.matched_gap(_games(strength, set(unf), 0.8, 200, 3), rt, B=200, seed=4)
    assert real["class"] == "DETECTED" and real["matched"] > 0.12
