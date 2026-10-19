"""THE READ of a team-strength registration: both pilots' ratings, per-set strength, the decision classes, the
cross-pilot agreement, and the MATCHED-STRENGTH piloting gap (``gen3_tsr_report_v1``).

The decision rules are the registration's (``reg["decisions"]`` names them; the code is here so a test pins them):

* :func:`competitive_class` — a set's mean minus a comparator set's mean, in Elo, with its two-stage 95 % interval:
  **MATERIALLY WEAKER** iff the upper bound < ``-margin``; **COMPETITIVE** iff the lower bound > ``-margin``;
  otherwise **UNDECIDED**, and UNDECIDED whenever a bound lies within 1e-9 of ``-margin`` (standing rule 8). The
  flag ``detected_weaker`` (upper bound < 0) is reported beside it.
* :func:`agreement_class` — **AGREE** iff Spearman's rho's 95 % lower bound > 0, **NOT SHOWN** otherwise (UNDECIDED
  within 1e-9 of 0).
* :func:`matched_gap` — the familiar-minus-unfamiliar PILOTING gap at matched team strength (below), **DETECTED** iff
  its lower bound > 0, **GONE** iff its upper bound < +0.030, otherwise **UNDECIDED** (and UNDECIDED within 1e-9).

THE MATCHED-STRENGTH GAP. Games where the agent PILOTS a team ``t`` against a fixed opponent distribution, split into
FAMILIAR (``U = 0``) and UNFAMILIAR (``U = 1``) piloted teams, each with the strength rating ``z_t`` (logits) of its
team. A logistic fit ``P(win) = sigmoid(a + g U + b z)`` (ties are not wins), and the gap is the average over the
pooled games of ``sigmoid(a + b z_i) - sigmoid(a + g + b z_i)``: what being unfamiliar costs at each game's strength.
Its bootstrap resamples the piloted teams within each group, then each drawn team's games, and draws ``z`` from the
rating's own replicate of the same index (so rating noise is carried). The RAW gap (familiar minus unfamiliar win rate
on the same games) is reported beside it, under the same resampling.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from main.team_strength import fit as F

SCHEMA = "gen3_tsr_report_v1"
NEAR = 1e-9


def _near(x: float, b: float) -> bool:
    return abs(float(x) - float(b)) <= NEAR


def competitive_class(ci: Sequence[float], margin: float) -> Dict[str, Any]:
    lo, hi = float(ci[0]), float(ci[1])
    if _near(lo, -margin) or _near(hi, -margin) or _near(hi, 0.0):
        cls = "UNDECIDED"
    elif hi < -margin:
        cls = "MATERIALLY WEAKER"
    elif lo > -margin:
        cls = "COMPETITIVE"
    else:
        cls = "UNDECIDED"
    return {"class": cls, "margin_elo": float(margin), "detected_weaker": hi < 0.0}


def agreement_class(ci: Sequence[float]) -> str:
    if _near(ci[0], 0.0):
        return "UNDECIDED"
    return "AGREE" if ci[0] > 0.0 else "NOT SHOWN"


def gap_class(ci: Sequence[float], gone_below: float = 0.030) -> str:
    lo, hi = float(ci[0]), float(ci[1])
    if _near(lo, 0.0) or _near(hi, gone_below):
        return "UNDECIDED"
    if lo > 0.0:
        return "DETECTED"
    if hi < gone_below:
        return "GONE"
    return "UNDECIDED"


def _sig(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logistic(X: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
    """Weighted logistic regression by Newton's method (a tiny ridge, 1e-6, keeps a separable resample finite)."""
    w = np.ones(len(y)) if w is None else w
    beta = np.zeros(X.shape[1])
    for _ in range(100):
        p = _sig(X @ beta)
        g = X.T @ (w * (y - p)) - 1e-6 * beta
        H = (X * (w * p * (1 - p))[:, None]).T @ X + 1e-6 * np.eye(X.shape[1])
        step = np.linalg.solve(H, g)
        beta = beta + step
        if float(np.max(np.abs(step))) < 1e-10:
            return beta
    raise F.FitError("the logistic fit did not converge")


def _gap_stats(team: np.ndarray, U: np.ndarray, y: np.ndarray, z_of_team: np.ndarray, w: np.ndarray
               ) -> Tuple[float, float]:
    z = z_of_team[team]
    X = np.column_stack([np.ones(len(y)), U, z])
    a, g, b = logistic(X, y, w)
    m = float(np.sum(w * (_sig(a + b * z) - _sig(a + g + b * z))) / np.sum(w))
    raw = float(np.sum(w * y * (U == 0)) / np.sum(w * (U == 0)) - np.sum(w * y * (U == 1)) / np.sum(w * (U == 1)))
    return m, raw


def matched_gap(rows: Sequence[Mapping[str, Any]], rating: F.Rating, *, B: int, seed: int) -> Dict[str, Any]:
    """``rows``: ``{"team": id, "unfamiliar": bool, "win": bool}`` per game (module docs). Every row's team must be
    rated."""
    teams = sorted({r["team"] for r in rows})
    missing = [t for t in teams if t not in rating.ids]
    if missing:
        raise F.FitError(f"{len(missing)} piloted teams have no rating (first: {missing[:3]})")
    tix = {t: i for i, t in enumerate(teams)}
    col = np.array([rating.ids.index(t) for t in teams])
    team = np.array([tix[r["team"]] for r in rows])
    U = np.array([1.0 if r["unfamiliar"] else 0.0 for r in rows])
    y = np.array([1.0 if r["win"] else 0.0 for r in rows])
    if len({bool(r["unfamiliar"]) for r in rows}) != 2:
        raise F.FitError("both groups (familiar and unfamiliar) need games")
    est, raw = _gap_stats(team, U, y, rating.r[col], np.ones(len(y)))
    groups: Dict[Tuple[int, bool], List[int]] = {}
    for i, r in enumerate(rows):
        groups.setdefault((tix[r["team"]], bool(r["unfamiliar"])), []).append(i)
    by_u: Dict[bool, List[List[int]]] = {False: [], True: []}
    for (t, u), idx in sorted(groups.items()):
        by_u[u].append(idx)
    rng = np.random.default_rng(int(seed))
    nb = min(int(B), rating.reps.shape[0])
    ms, raws = [], []
    for k in range(nb):
        w = np.zeros(len(y))
        for u in (False, True):
            tl = by_u[u]
            for ti in rng.integers(0, len(tl), len(tl)):
                idx = tl[int(ti)]
                for gi in rng.integers(0, len(idx), len(idx)):
                    w[idx[int(gi)]] += 1.0
        m, rw = _gap_stats(team, U, y, rating.reps[k, col], w)
        ms.append(m)
        raws.append(rw)
    ci = [float(x) for x in np.percentile(ms, [2.5, 97.5])]
    rci = [float(x) for x in np.percentile(raws, [2.5, 97.5])]
    return {"n_games": int(len(y)), "n_teams": {"familiar": len(by_u[False]), "unfamiliar": len(by_u[True])},
            "matched": est, "matched_ci": ci, "class": gap_class(ci), "raw": raw, "raw_ci": rci,
            "explained_by_strength": raw - est, "B": nb, "seed": int(seed)}


def rating_table(rt: F.Rating, membership: Mapping[str, Sequence[str]]) -> List[Dict[str, Any]]:
    rows = []
    for t in rt.ids:
        lo, hi = rt.ci(t)
        rows.append({"team_id": t, "sets": list(membership.get(t, [])), "elo": rt.elo(t), "ci": [lo, hi],
                     "games": rt.games[t], "score": rt.score[t]})
    rows.sort(key=lambda r: -r["elo"])
    return rows


def finite(x: float) -> Optional[float]:
    return None if (x is None or (isinstance(x, float) and math.isnan(x))) else float(x)
