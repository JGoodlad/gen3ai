"""THE RATING of the team-strength meter (``gen3_tsr_fit_v1``): a Bradley-Terry fit on TEAM IDENTITY, its bootstrap,
per-set means, and the agreement of two pilots' ratings.

THE MODEL. A game between teams ``a`` and ``b`` (both piloted by the SAME agent) scores ``s`` for ``a`` (win 1, draw
1/2, loss 0); ``P(a beats b) = sigmoid(r_a - r_b)``. The ratings are the MAP estimate under independent
``N(0, prior_sd^2)`` priors (``prior_sd`` in logits; it keeps a team that won every game finite and shrinks a thinly
measured team toward the panel), found by Newton's method to a step below ``1e-10``, then SHIFTED so the reference
panel's mean is 0. A rating is reported in Elo points: ``ELO * r`` (``ELO = 400 / ln 10``), so +100 Elo against the
panel mean is a 64 % expected score.

THE BOOTSTRAP (deterministic: ``numpy.random.default_rng(seed)``, ``B`` replicates). Each game belongs to an OWNER
(the rated team whose schedule produced it) and a CLUSTER (a mirrored pair: its two games share a battle seed and are
not independent; an unmirrored game is its own cluster). A replicate resamples, within each owner, that owner's
clusters with replacement, and REFITS (warm-started). It gives each team's game-noise interval. A SET MEAN's
replicate also resamples the set's members with replacement (the set as a sample of its population of teams) and
averages their refitted ratings: a two-stage interval, teams then games. A difference of two set means uses the same
replicate for both (the shared refit) with independent member draws. Intervals are percentile 95 %.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SCHEMA = "gen3_tsr_fit_v1"
ELO = 400.0 / math.log(10.0)
STEP_TOL = 1e-10
MAX_ITERS = 200


class FitError(RuntimeError):
    pass


@dataclass(frozen=True)
class Game:
    a: str
    b: str
    s: float            # a's score: 1 win, 0.5 draw, 0 loss
    owner: str
    cluster: str


@dataclass
class Games:
    """Games over an id table, as arrays."""

    ids: List[str]
    a: np.ndarray
    b: np.ndarray
    s: np.ndarray
    owner_clusters: List[np.ndarray]        # per owner: the cluster indices it owns
    cluster_games: List[np.ndarray]         # per cluster: its game indices

    @classmethod
    def of(cls, games: Sequence[Game], ids: Optional[Sequence[str]] = None) -> "Games":
        if not games:
            raise FitError("no games to fit")
        all_ids = sorted({g.a for g in games} | {g.b for g in games} | set(ids or ()))
        ix = {t: i for i, t in enumerate(all_ids)}
        for g in games:
            if g.a == g.b:
                raise FitError(f"a game of team {g.a} against itself")
            if g.s not in (0.0, 0.5, 1.0):
                raise FitError(f"score {g.s!r} is not 0 / 0.5 / 1")
        clusters: Dict[str, List[int]] = {}
        owner_of: Dict[str, str] = {}
        for i, g in enumerate(games):
            clusters.setdefault(g.cluster, []).append(i)
            if owner_of.setdefault(g.cluster, g.owner) != g.owner:
                raise FitError(f"cluster {g.cluster} has two owners")
        cnames = sorted(clusters)
        cix = {c: i for i, c in enumerate(cnames)}
        owners: Dict[str, List[int]] = {}
        for c in cnames:
            owners.setdefault(owner_of[c], []).append(cix[c])
        return cls(ids=all_ids, a=np.array([ix[g.a] for g in games]), b=np.array([ix[g.b] for g in games]),
                   s=np.array([g.s for g in games], dtype=float),
                   owner_clusters=[np.array(owners[o]) for o in sorted(owners)],
                   cluster_games=[np.array(clusters[c]) for c in cnames])


def log_posterior(gs: Games, r: np.ndarray, w: np.ndarray, lam: float) -> float:
    d = r[gs.a] - r[gs.b]
    # s log sigmoid(d) + (1 - s) log sigmoid(-d), stably
    ll = gs.s * -np.logaddexp(0.0, -d) + (1.0 - gs.s) * -np.logaddexp(0.0, d)
    return float(np.sum(w * ll) - 0.5 * lam * np.dot(r, r))


def fit(gs: Games, *, prior_sd: float, anchor: Sequence[str], w: Optional[np.ndarray] = None,
        r0: Optional[np.ndarray] = None) -> np.ndarray:
    """The MAP ratings (logits), shifted so the mean over ``anchor`` is 0 (module docs). DAMPED Newton: a step is
    halved until the log posterior does not fall (a full Newton step from a warm start can overshoot and cycle — a
    bootstrap replicate did, 2026-10-19); converged when the step or the gradient is below :data:`STEP_TOL`."""
    n = len(gs.ids)
    lam = 1.0 / float(prior_sd) ** 2
    w = np.ones(len(gs.s)) if w is None else np.asarray(w, dtype=float)
    r = np.zeros(n) if r0 is None else np.array(r0, dtype=float)
    a, b, s = gs.a, gs.b, gs.s
    cur = log_posterior(gs, r, w, lam)
    for _ in range(MAX_ITERS):
        p = 1.0 / (1.0 + np.exp(-(r[a] - r[b])))
        resid = w * (s - p)
        grad = np.bincount(a, resid, n) - np.bincount(b, resid, n) - lam * r
        if float(np.max(np.abs(grad))) < STEP_TOL:
            break
        h = w * p * (1.0 - p)
        H = np.zeros((n, n))
        np.add.at(H, (a, a), h)
        np.add.at(H, (b, b), h)
        np.add.at(H, (a, b), -h)
        np.add.at(H, (b, a), -h)
        H[np.diag_indices(n)] += lam
        step = np.linalg.solve(H, grad)
        t = 1.0
        while True:
            nxt = log_posterior(gs, r + t * step, w, lam)
            if nxt >= cur - 1e-12 * max(1.0, abs(cur)) or t < 1e-6:
                break
            t *= 0.5
        r, cur = r + t * step, nxt
        if float(np.max(np.abs(t * step))) < STEP_TOL:
            break
    else:
        raise FitError(f"Newton did not converge in {MAX_ITERS} steps")
    ax = [gs.ids.index(t) for t in anchor]
    if not ax:
        raise FitError("an empty anchor")
    return r - float(np.mean(r[ax]))


def replicate_weights(gs: Games, rng: np.random.Generator) -> np.ndarray:
    """One replicate's per-game multiplicities: each owner's clusters resampled with replacement."""
    w = np.zeros(len(gs.s))
    for oc in gs.owner_clusters:
        pick = oc[rng.integers(0, len(oc), len(oc))]
        for c in pick:
            w[gs.cluster_games[int(c)]] += 1.0
    return w


@dataclass
class Rating:
    ids: List[str]
    r: np.ndarray                 # point ratings, logits
    reps: np.ndarray              # B x n replicate ratings
    games: Dict[str, int]
    score: Dict[str, float]       # each team's mean score over its games
    seed: int
    prior_sd: float
    anchor: List[str]

    def elo(self, t: str) -> float:
        return ELO * float(self.r[self.ids.index(t)])

    def ci(self, t: str) -> Tuple[float, float]:
        col = self.reps[:, self.ids.index(t)] * ELO
        lo, hi = np.percentile(col, [2.5, 97.5])
        return float(lo), float(hi)


def rate(games: Sequence[Game], *, prior_sd: float, anchor: Sequence[str], B: int, seed: int) -> Rating:
    gs = Games.of(games, anchor)
    r = fit(gs, prior_sd=prior_sd, anchor=anchor)
    rng = np.random.default_rng(int(seed))
    reps = np.empty((int(B), len(gs.ids)))
    for i in range(int(B)):
        reps[i] = fit(gs, prior_sd=prior_sd, anchor=anchor, w=replicate_weights(gs, rng), r0=r)
    n = np.bincount(gs.a, minlength=len(gs.ids)) + np.bincount(gs.b, minlength=len(gs.ids))
    tot = np.bincount(gs.a, gs.s, len(gs.ids)) + np.bincount(gs.b, 1.0 - gs.s, len(gs.ids))
    return Rating(ids=gs.ids, r=r, reps=reps, games={t: int(n[i]) for i, t in enumerate(gs.ids)},
                  score={t: (float(tot[i] / n[i]) if n[i] else float("nan")) for i, t in enumerate(gs.ids)},
                  seed=int(seed), prior_sd=float(prior_sd), anchor=list(anchor))


def set_means(rt: Rating, sets: Mapping[str, Sequence[str]], seed: int) -> Dict[str, Any]:
    """Per set: the mean Elo of its members, and the two-stage replicate means (module docs). ``draws`` holds the
    B replicate means per set (for differences)."""
    rng = np.random.default_rng(int(seed))
    out: Dict[str, Any] = {"draws": {}}
    ix = {t: i for i, t in enumerate(rt.ids)}
    for name, members in sets.items():
        cols = np.array([ix[t] for t in members])
        if not len(cols):
            raise FitError(f"set {name} has no rated member")
        pick = cols[rng.integers(0, len(cols), (rt.reps.shape[0], len(cols)))]
        draws = np.take_along_axis(rt.reps, pick, axis=1).mean(axis=1) * ELO
        game_only = rt.reps[:, cols].mean(axis=1) * ELO
        lo, hi = np.percentile(draws, [2.5, 97.5])
        glo, ghi = np.percentile(game_only, [2.5, 97.5])
        out[name] = {"n_teams": int(len(cols)), "mean_elo": float(rt.r[cols].mean() * ELO),
                     "ci": [float(lo), float(hi)], "ci_games_only": [float(glo), float(ghi)],
                     "sd_elo": float(rt.r[cols].std(ddof=1) * ELO) if len(cols) > 1 else 0.0}
        out["draws"][name] = draws
    return out


def diff(means: Mapping[str, Any], x: str, y: str) -> Dict[str, Any]:
    d = np.asarray(means["draws"][x]) - np.asarray(means["draws"][y])
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"est": float(means[x]["mean_elo"] - means[y]["mean_elo"]), "ci": [float(lo), float(hi)]}


def elo_to_p(e: float) -> float:
    return 1.0 / (1.0 + 10.0 ** (-float(e) / 400.0))


def _ranks(x: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(x)


def spearman(x: Sequence[float], y: Sequence[float]) -> float:
    rx, ry = _ranks(np.asarray(x, float)), _ranks(np.asarray(y, float))
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def reliability(rt: Rating, members: Sequence[str]) -> float:
    """1 - (mean bootstrap variance of a member's rating) / (variance of the members' point ratings): the share of
    the observed spread that is not game noise."""
    cols = [rt.ids.index(t) for t in members]
    obs = float(np.var(rt.r[cols], ddof=1))
    noise = float(np.mean(np.var(rt.reps[:, cols], axis=0, ddof=1)))
    return 1.0 - noise / obs if obs > 0 else float("nan")


def agreement(ra: Rating, rb: Rating, common: Sequence[str], seed: int) -> Dict[str, Any]:
    """Spearman's rho of two pilots' ratings over ``common`` teams, its bootstrap (teams resampled, each pilot's
    replicate ratings), both reliabilities and the attenuation-corrected rho."""
    common = sorted(common)
    if len(common) < 4:
        raise FitError(f"only {len(common)} teams rated by both pilots")
    ia, ib = [ra.ids.index(t) for t in common], [rb.ids.index(t) for t in common]
    rho = spearman(ra.r[ia], rb.r[ib])
    B = min(ra.reps.shape[0], rb.reps.shape[0])
    rng = np.random.default_rng(int(seed))
    draws = []
    for k in range(B):
        pick = rng.integers(0, len(common), len(common))
        draws.append(spearman(ra.reps[k, ia][pick], rb.reps[k, ib][pick]))
    d = np.asarray([x for x in draws if not math.isnan(x)])
    lo, hi = np.percentile(d, [2.5, 97.5])
    rel_a, rel_b = reliability(ra, common), reliability(rb, common)
    corr = rho / math.sqrt(rel_a * rel_b) if rel_a > 0 and rel_b > 0 else float("nan")
    return {"n_teams": len(common), "rho": rho, "ci": [float(lo), float(hi)], "reliability_a": rel_a,
            "reliability_b": rel_b, "rho_disattenuated": corr,
            "pearson_elo": float(np.corrcoef(ra.r[ia], rb.r[ib])[0, 1])}
