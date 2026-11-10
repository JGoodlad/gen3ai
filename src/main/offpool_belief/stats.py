"""Team-clustered statistics: a panel's number is the mean over TEAMS of each team's mean, and every interval is a
seeded bootstrap over teams (rows of one team are one sample, never many).

Deterministic by construction: the resampling indices are a pure function of ``(seed, n_teams, draws)``, one index
matrix is shared by every column of a comparison (so a difference is paired), and a team with no unit for a metric
is ABSENT from that metric (never a zero).
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

DEFAULT_DRAWS = 10_000
DEFAULT_SEED = 20261109


def team_means(sums: Sequence[float], n: Sequence[float]) -> np.ndarray:
    """Per-team mean ``sums / n``; NaN where the team has no unit."""
    s, c = np.asarray(sums, dtype=np.float64), np.asarray(n, dtype=np.float64)
    if s.shape != c.shape:
        raise ValueError(f"sums {s.shape} and counts {c.shape} differ")
    out = np.full(s.shape, np.nan)
    np.divide(s, c, out=out, where=c > 0)
    return out


def boot_index(n_teams: int, draws: int = DEFAULT_DRAWS, seed: int = DEFAULT_SEED) -> np.ndarray:
    """``[draws, n_teams]`` team indices, resampled with replacement."""
    return np.random.default_rng([seed, n_teams]).integers(0, n_teams, size=(draws, n_teams))


def mean_ci(values: Sequence[float], index: Optional[np.ndarray] = None) -> Dict[str, Optional[float]]:
    """The mean over the teams that have a value, with the team-bootstrap 95 % percentile interval.

    ``values`` is one number per team (NaN = the team has no unit). A resample that draws no team with a value is
    dropped (and counted in ``empty_draws``); with fewer than two valued teams there is no interval."""
    v = np.asarray(values, dtype=np.float64)
    ok = ~np.isnan(v)
    n = int(ok.sum())
    if n == 0:
        return {"mean": None, "lo": None, "hi": None, "n_teams": 0, "empty_draws": 0}
    mean = float(v[ok].mean())
    if n < 2:
        return {"mean": mean, "lo": None, "hi": None, "n_teams": n, "empty_draws": 0}
    idx = boot_index(len(v)) if index is None else index
    if idx.shape[1] != len(v):
        raise ValueError(f"bootstrap index is for {idx.shape[1]} teams, the column has {len(v)}")
    drawn, got = v[idx], ok[idx]
    cnt = got.sum(axis=1)
    tot = np.where(got, drawn, 0.0).sum(axis=1)
    keep = cnt > 0
    boot = tot[keep] / cnt[keep]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"mean": mean, "lo": float(lo), "hi": float(hi), "n_teams": n, "empty_draws": int((~keep).sum())}


def paired_diff(a: Sequence[float], b: Sequence[float], index: Optional[np.ndarray] = None
                ) -> Dict[str, Optional[float]]:
    """``a − b`` per team (a team missing either side is absent), its mean and team-bootstrap interval."""
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"paired columns differ in shape: {x.shape} vs {y.shape}")
    return mean_ci(x - y, index)
