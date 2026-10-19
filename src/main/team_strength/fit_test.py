"""The team-strength fit: Bradley-Terry on team identity recovers planted strengths, is anchored on the reference
panel, stays finite on a perfect record, and its bootstrap / set means / agreement are deterministic and honest."""
from __future__ import annotations

import math

import numpy as np
import pytest

from main.team_strength import fit as F

REFS = [f"r{i}" for i in range(8)]


def simulate(truth: dict, pairs: int, seed: int) -> list:
    """Each non-reference team plays ``pairs`` mirrored pairs against cycling references (two games per pair)."""
    rng = np.random.default_rng(seed)
    games = []
    for t in sorted(truth):
        if t in REFS:
            continue
        for j in range(pairs):
            r = REFS[j % len(REFS)]
            p = 1.0 / (1.0 + math.exp(-(truth[t] - truth[r])))
            for g in range(2):
                games.append(F.Game(a=t, b=r, s=float(rng.random() < p), owner=t, cluster=f"{t}:{j}"))
    return games


@pytest.fixture(scope="module")
def planted():
    truth = {r: 0.0 for r in REFS}
    truth.update({"strong": 1.0, "weak": -1.0, "mid": 0.0})
    return truth, simulate(truth, pairs=600, seed=1)


def test_the_fit_recovers_planted_strengths_and_orders_them(planted):
    truth, games = planted
    gs = F.Games.of(games, REFS)
    r = F.fit(gs, prior_sd=10.0, anchor=REFS)
    est = {t: float(r[gs.ids.index(t)]) for t in ("strong", "mid", "weak")}
    # 1,200 games per team: the MLE is within ~0.15 logits of the truth; a sign or a pairing error is off by ~2
    assert abs(est["strong"] - 1.0) < 0.25 and abs(est["weak"] + 1.0) < 0.25 and abs(est["mid"]) < 0.25
    assert est["strong"] > est["mid"] > est["weak"]


def test_the_reference_panel_mean_is_zero(planted):
    _truth, games = planted
    gs = F.Games.of(games, REFS)
    r = F.fit(gs, prior_sd=2.0, anchor=REFS)
    assert abs(float(np.mean([r[gs.ids.index(x)] for x in REFS]))) < 1e-12


def test_a_perfect_record_stays_finite_under_the_prior():
    games = [F.Game(a="t", b=REFS[j % 8], s=1.0, owner="t", cluster=f"t:{j}") for j in range(20)]
    games += [F.Game(a="u", b=REFS[j % 8], s=float(j % 2), owner="u", cluster=f"u:{j}") for j in range(20)]
    gs = F.Games.of(games, REFS)
    r = F.fit(gs, prior_sd=2.0, anchor=REFS)
    assert np.all(np.isfinite(r))
    assert r[gs.ids.index("t")] > r[gs.ids.index("u")] + 1.0


def test_a_self_game_and_a_bad_score_are_refused():
    with pytest.raises(F.FitError):
        F.Games.of([F.Game(a="t", b="t", s=1.0, owner="t", cluster="c")])
    with pytest.raises(F.FitError):
        F.Games.of([F.Game(a="t", b="u", s=0.7, owner="t", cluster="c")])


def test_the_bootstrap_is_deterministic_and_set_means_carry_team_sampling():
    truth = {r: 0.0 for r in REFS}
    spread = {f"s{i}": (i - 5) * 0.3 for i in range(11)}
    truth.update(spread)
    games = simulate(truth, pairs=40, seed=3)
    a = F.rate(games, prior_sd=2.0, anchor=REFS, B=200, seed=7)
    b = F.rate(games, prior_sd=2.0, anchor=REFS, B=200, seed=7)
    assert np.array_equal(a.reps, b.reps)
    m = F.set_means(a, {"S": sorted(spread)}, seed=11)
    lo, hi = m["S"]["ci"]
    glo, ghi = m["S"]["ci_games_only"]
    # teams differ by up to 3 logits: resampling the members must widen the interval well past game noise alone
    assert (hi - lo) > 1.5 * (ghi - glo)
    assert lo < m["S"]["mean_elo"] < hi


def test_agreement_of_two_pilots_who_see_the_same_strengths_is_high_and_reliability_is_sane():
    truth = {r: 0.0 for r in REFS}
    spread = {f"s{i}": (i - 10) * 0.15 for i in range(21)}
    truth.update(spread)
    ra = F.rate(simulate(truth, pairs=60, seed=4), prior_sd=2.0, anchor=REFS, B=150, seed=1)
    rb = F.rate(simulate(truth, pairs=60, seed=5), prior_sd=2.0, anchor=REFS, B=150, seed=2)
    ag = F.agreement(ra, rb, sorted(spread), seed=3)
    assert ag["rho"] > 0.8 and ag["ci"][0] > 0.5
    assert 0.5 < ag["reliability_a"] <= 1.0 and 0.5 < ag["reliability_b"] <= 1.0
    # an INVERTED second pilot reads negative: the statistic is not sign-blind
    inv = {**truth, **{k: -v for k, v in spread.items()}}
    rc = F.rate(simulate(inv, pairs=60, seed=6), prior_sd=2.0, anchor=REFS, B=150, seed=3)
    assert F.agreement(ra, rc, sorted(spread), seed=3)["rho"] < -0.8


def test_elo_scale():
    assert abs(F.elo_to_p(F.ELO * 1.0) - 1.0 / (1.0 + math.exp(-1.0))) < 1e-12
    assert abs(F.elo_to_p(0.0) - 0.5) < 1e-15
