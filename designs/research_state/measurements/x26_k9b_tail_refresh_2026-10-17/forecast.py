"""Refresh of the 177f61c0 K9(b) tail forecast for rb_x26_s2601 against Amendment 3's ceilings.

    python3 forecast.py --series tb_series.json --violations behaviour_violations.jsonl --out forecast.json

Inputs: `tb_extract.py`'s full-precision TensorBoard series (the probe runs every 10th update, ~98.3k steps)
and the run's violation records. The u13900 probe raised before TensorBoard was written (177f61c0 F6), so it
is carried as a sensitivity (the run resumed from checkpoint_136000153, so u13900 is on an abandoned branch).

Two independent views, each with at least two model forms:

A. EVENT RATE (the 1e-4 bar). A binomial GLM per probe on the indicator judged-max >= 1e-4 with
   logit p = a (constant), a + b*s (log-linear in steps) or a + b*log s (power in steps); MLE, the
   likelihood-ratio test of the trend, and a profile 90 % interval on b.

B. TRAJECTORIES (the ceilings). For x in {judged max, p99}: log x_i = T(s_i) + r_i with three trend forms
   fitted by least squares on s >= --fit-from:
     linear       T = log(a + b s)          (177f61c0's form)
     exponential  T = a + b s
     power        T = a + b log s
   The residual r is taken as stationary: its body is the empirical distribution, its upper tail beyond the
   95th percentile a generalised Pareto (GPD) fit by MLE, so a ceiling far above the data is an
   extrapolation of a FITTED tail, labelled as such. Per future probe j (every 98.3k steps to the 500M cap)
   p_j = P(r >= log c - T(s_j)); P(cross before cap) = 1 - prod(1 - p_j); the first-crossing distribution is
   F(t) = 1 - prod_{s_j <= t}(1 - p_j). Uncertainty: a moving-block bootstrap (blocks of 50 probes, ~5M
   steps, which keeps the residuals' autocorrelation) refits the trend AND the tail; the reported interval
   is the 5 / 50 / 95 % points of the MIXTURE of the replicates' first-crossing distributions, and
   P(cross) is the mixture's mass below the cap. A hold-out check fits each form on s <= 137M (the data
   177f61c0 had) and scores its prediction of the 137M..now medians.
"""
from __future__ import annotations

import argparse
import json
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from scipy import optimize, stats

CAP = 500_000_000
CEIL = {"max": 1e-3, "p99": 1e-4}
BAR = 1e-4


# ----------------------------------------------------------------------------------------------- data
def load(series: str, violations: str) -> Dict[str, np.ndarray]:
    s = json.load(open(series))["series"]
    m = dict((int(k), v) for k, v in s["behaviour/max_abs_dlogp_judged"])
    q = dict((int(k), v) for k, v in s["behaviour/p99_abs_dlogp_current"])
    recs = [json.loads(line) for line in open(violations)]
    # The run resumed after the u13900 FATAL from checkpoint_136000153 (a startup-gate refusal of the exception
    # save; design_x26_readiness.md §4.2), so 136.2M..137.0M is logged twice and the later file (read last by
    # tb_extract.py) is the LIVE lineage. The u13900 probe belongs to the abandoned branch and never reached
    # TensorBoard (177f61c0 F6): the series is the live lineage only, and the u13900 event is carried as a
    # SENSITIVITY (`abandoned`), never silently merged.
    st = np.array(sorted(set(m) & set(q)), dtype=np.float64)
    live, abandoned = [], []
    for r in recs:
        mv = next(c["value"] for c in r["conditions"] if c["statistic"] == "max")
        hit = [k for k in st if 0 < k - r["num_timesteps"] <= 1.5e5 and abs(m[int(k)] - mv) < 1e-12]
        (live if hit else abandoned).append(int(r["n_updates"]))
    return {"s": st, "max": np.array([m[int(k)] for k in st]), "p99": np.array([q[int(k)] for k in st]),
            "live_records": np.array(live), "abandoned_records": np.array(abandoned), "event_steps": np.array([r["num_timesteps"] for r in recs])}


# ------------------------------------------------------------------------------------- A. event rate
def glm(s: np.ndarray, y: np.ndarray, form: str) -> Dict[str, Any]:
    x = {"const": np.zeros_like(s), "loglinear": s / 1e8, "power": np.log(s / 1e8)}[form]

    def nll(th: np.ndarray) -> float:
        z = th[0] + (th[1] * x if form != "const" else 0.0)
        return float(np.sum(np.logaddexp(0, z) - y * z))
    th0 = np.array([np.log(max(y.mean(), 1e-6) / (1 - y.mean())), 0.0])
    res = optimize.minimize(nll, th0, method="Nelder-Mead", options={"xatol": 1e-8, "fatol": 1e-10, "maxiter": 20000})
    out: Dict[str, Any] = {"form": form, "a": float(res.x[0]), "b": float(res.x[1]) if form != "const" else 0.0,
                           "nll": float(res.fun)}
    if form != "const":                  # profile likelihood 90 % interval on b (chi2_1 0.90 = 2.706)
        def prof(b: float) -> float:
            r = optimize.minimize_scalar(lambda a: nll(np.array([a, b])), bounds=(-40, 20), method="bounded")
            return float(r.fun)
        target = res.fun + stats.chi2.ppf(0.90, 1) / 2
        grid = np.linspace(res.x[1] - 15, res.x[1] + 15, 1201)
        pv = np.array([prof(b) for b in grid])
        inside = grid[pv <= target]
        out["b_ci90"] = [float(inside.min()), float(inside.max())]
    return out


def glm_p(fit: Dict[str, Any], s: np.ndarray, b: float = None) -> np.ndarray:  # type: ignore[assignment]
    b = fit["b"] if b is None else b
    x = {"const": np.zeros_like(s), "loglinear": s / 1e8, "power": np.log(s / 1e8)}[fit["form"]]
    return 1 / (1 + np.exp(-(fit["a"] + b * x)))


# ------------------------------------------------------------------------------------ B. trajectories
FORMS: Dict[str, Tuple[Callable[..., np.ndarray], List[float]]] = {
    "linear": (lambda s, a, b: np.log(np.maximum(a + b * s / 1e8, 1e-30)), [1e-5, 1e-5]),
    "exponential": (lambda s, a, b: a + b * s / 1e8, [-11.0, 0.5]),
    "power": (lambda s, a, b: a + b * np.log(s / 1e8), [-11.0, 0.5]),
}


def fit_trend(s: np.ndarray, lx: np.ndarray, form: str) -> np.ndarray:
    f, p0 = FORMS[form]
    if form == "linear":     # scale the start to the data
        p0 = [float(np.exp(np.median(lx))) * 0.5, float(np.exp(np.median(lx))) * 0.3]
    p, _ = optimize.curve_fit(f, s, lx, p0=p0, maxfev=20000)
    return p


class Tail:
    """The residual's survival function: empirical below u (its 95th percentile), GPD above."""

    def __init__(self, r: np.ndarray) -> None:
        self.r = np.sort(r)
        self.u = float(np.quantile(r, 0.95))
        exc = r[r > self.u] - self.u
        self.pu = exc.size / r.size
        self.xi, _, self.sigma = stats.genpareto.fit(exc, floc=0)

    def sf(self, z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=np.float64)
        emp = 1 - np.searchsorted(self.r, z, side="left") / self.r.size
        gp = self.pu * stats.genpareto.sf(z - self.u, self.xi, loc=0, scale=self.sigma)
        return np.where(z <= self.u, emp, gp)


def crossing(s_fut: np.ndarray, T: np.ndarray, tail: Tail, c: float) -> np.ndarray:
    """Per future probe, P(x >= c)."""
    return np.clip(tail.sf(np.log(c) - T), 0, 1)


def first_cross_cdf(p: np.ndarray) -> np.ndarray:
    return 1 - np.exp(np.cumsum(np.log1p(-np.minimum(p, 1 - 1e-15))))


def block_idx(n: int, L: int, rng: np.random.Generator) -> np.ndarray:
    starts = rng.integers(0, n - L + 1, size=int(np.ceil(n / L)))
    return np.concatenate([np.arange(a, a + L) for a in starts])[:n]


def trajectory(d: Dict[str, np.ndarray], key: str, fit_from: float, B: int, seed: int) -> Dict[str, Any]:
    s_all, x_all = d["s"], d[key]
    k = s_all >= fit_from
    s, lx = s_all[k], np.log(x_all[k])
    step = float(np.median(np.diff(s_all)))
    s_fut = np.arange(s_all[-1] + step, CAP + 1, step)
    c = CEIL[key]
    out: Dict[str, Any] = {"ceiling": c, "fit_from": fit_from, "n_fit": int(k.sum()), "forms": {}}
    rng = np.random.default_rng(seed)
    idxs = [block_idx(s.size, 50, rng) for _ in range(B)]
    for form in FORMS:
        f = FORMS[form][0]
        p = fit_trend(s, lx, form)
        r = lx - f(s, *p)
        tail = Tail(r)
        T = f(s_fut, *p)
        pj = crossing(s_fut, T, tail, c)
        F = first_cross_cdf(pj)
        # hold-out: fit on s <= 137M, score the 137M..now median of log x in 10M windows
        h = s <= 137e6
        ph = fit_trend(s[h], lx[h], form)
        edges = np.arange(140e6, s[-1] + 1, 10e6)
        ho = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            w = (s >= lo) & (s < hi)
            if w.sum() >= 20:
                ho.append(float(np.median(lx[w]) - np.median(f(s[w], *ph) + np.median(lx[h] - f(s[h], *ph)))))
        # bootstrap mixture
        Fb = np.zeros((B, s_fut.size))
        pcap = np.zeros(B)
        for bi, ix in enumerate(idxs):
            try:
                pb = fit_trend(s[ix], lx[ix], form)
                rb = lx[ix] - f(s[ix], *pb)
                tb = Tail(rb)
                Fb[bi] = first_cross_cdf(crossing(s_fut, f(s_fut, *pb), tb, c))
            except (RuntimeError, ValueError):
                Fb[bi] = np.nan
            pcap[bi] = Fb[bi, -1]
        ok = np.isfinite(Fb[:, -1])
        Fm = np.nanmean(Fb[ok], 0)

        def q_of(F_: np.ndarray, qq: float) -> Any:
            j = np.searchsorted(F_, qq)
            return None if j >= F_.size else float(s_fut[j])
        med_now = float(np.exp(f(s_all[-1], *p) + np.median(r)))
        med_cap = float(np.exp(f(np.array([CAP]), *p)[0] + np.median(r)))
        out["forms"][form] = {
            "params": [float(v) for v in p], "rss": float(np.sum(r ** 2)),
            "aic": float(s.size * np.log(np.sum(r ** 2) / s.size) + 4),
            "median_now": med_now, "median_at_cap": med_cap,
            "tail": {"u": tail.u, "xi": float(tail.xi), "sigma": float(tail.sigma), "r_max": float(r.max()),
                     "ceiling_excess_over_trend_at_cap_logunits": float(np.log(c) - f(np.array([CAP]), *p)[0])},
            "point": {"P_cross_before_cap": float(F[-1]), "first_cross_q05": q_of(F, 0.05),
                      "first_cross_q50": q_of(F, 0.5), "first_cross_q95": q_of(F, 0.95)},
            "bootstrap": {"B_ok": int(ok.sum()), "P_cross_before_cap_mixture": float(Fm[-1]),
                          "P_cross_before_cap_replicate_q05_q50_q95": [float(np.nanquantile(pcap[ok], qq)) for qq in (0.05, 0.5, 0.95)],
                          "first_cross_mixture_q05": q_of(Fm, 0.05), "first_cross_mixture_q50": q_of(Fm, 0.5),
                          "first_cross_mixture_q95": q_of(Fm, 0.95)},
            "holdout_137M_log_median_error_per_10M_window": ho,
            # is the residual's spread stationary? its 95th / 99th percentile before vs after 137M
            "residual_q95_q99_before_after_137M": [[float(np.quantile(r[h], qq)) for qq in (0.95, 0.99)],
                                                   [float(np.quantile(r[~h], qq)) for qq in (0.95, 0.99)]],
        }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", required=True)
    ap.add_argument("--violations", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=400)
    a = ap.parse_args()
    d = load(a.series, a.violations)
    s = d["s"]
    out: Dict[str, Any] = {"n_probes": int(s.size), "first_step": float(s[0]), "last_step": float(s[-1]),
                           "probe_step": float(np.median(np.diff(s))),
                           "records_on_live_lineage": d["live_records"].tolist(),
                           "records_on_abandoned_branch": d["abandoned_records"].tolist()}
    # ---------------- A. event rate on the 1e-4 bar
    y = (d["max"] >= BAR).astype(float)
    out["events"] = {"n": int(y.sum()), "steps": s[y > 0].tolist()}
    fits = {f: glm(s, y, f) for f in ("const", "loglinear", "power")}
    y5 = y.copy()                                 # sensitivity: the abandoned u13900 event counted too
    for st_ in [r for r in d["event_steps"] if not any(0 < k - r <= 1.5e5 and y[i] for i, k in enumerate(s))]:
        y5[int(np.argmin(np.abs(s - st_ - 9.8e4)))] = 1.0
    fits5 = {f: glm(s, y5, f) for f in ("const", "loglinear", "power")}
    out["event_rate_with_abandoned_u13900"] = {f: {"b": v["b"], "b_ci90": v.get("b_ci90"), "nll": v["nll"],
                                                   "lr_vs_const": float(2 * (fits5["const"]["nll"] - v["nll"]))}
                                               for f, v in fits5.items()}
    out["events"]["n_with_abandoned"] = int(y5.sum())
    s_fut = np.arange(s[-1] + out["probe_step"], CAP + 1, out["probe_step"])
    ev: Dict[str, Any] = {}
    for f, fit in fits.items():
        e: Dict[str, Any] = {**fit, "lr_vs_const": float(2 * (fits["const"]["nll"] - fit["nll"])),
                             "lr_p": float(stats.chi2.sf(2 * (fits["const"]["nll"] - fit["nll"]), 1)) if f != "const" else None,
                             "p_per_probe_now": float(glm_p(fit, s[-1:])[0]),
                             "p_per_probe_cap": float(glm_p(fit, np.array([float(CAP)]))[0]),
                             "expected_events_to_cap": float(glm_p(fit, s_fut).sum())}
        if f != "const":
            lo, hi = fit["b_ci90"]
            e["expected_events_to_cap_at_b_ci90"] = [float(glm_p(fit, s_fut, b).sum()) for b in (lo, hi)]
        ev[f] = e
    # what 177f61c0's table predicted for 137M..now: per-probe p rises log-linearly in the table's points
    tab_s = np.array([137e6, 200e6, 250e6, 370e6, 500e6])
    tab_p = np.array([0.0037, 0.010, 0.017, 0.039, 0.064])
    w = s > 137.0e6
    pred = np.exp(np.interp(s[w], tab_s, np.log(tab_p))).sum()
    ev["prior_forecast_check"] = {"expected_events_after_137M_to_now": float(pred),
                                  "observed": int(y[w].sum()),
                                  "poisson_p_ge_observed": float(stats.poisson.sf(y[w].sum() - 1, pred)),
                                  "last20M_expected": float(np.exp(np.interp(s[s > s[-1] - 2e7], tab_s, np.log(tab_p))).sum()),
                                  "last20M_observed": int(y[s > s[-1] - 2e7].sum())}
    ev["prior_forecast_check"]["last20M_poisson_p_ge_observed"] = float(
        stats.poisson.sf(ev["prior_forecast_check"]["last20M_observed"] - 1, ev["prior_forecast_check"]["last20M_expected"]))
    out["event_rate"] = ev
    # ---------------- the records themselves: does each violator's whole legal distribution move by a
    # rounding-sized amount (a numerical gap) rather than by a wrong-row / wrong-action amount (>= 1e-2)?
    recs = [json.loads(line) for line in open(a.violations)]
    rr = []
    for r in recs:
        v = r["rows"][0]
        ds = [x - y for x, y in zip(v["dist_stored"], v["dist_recomputed"]) if x is not None and y is not None]
        rr.append({"n_updates": r["n_updates"], "num_timesteps": r["num_timesteps"], "abs_dlogp": v["abs_dlogp"],
                   "n_legal": len(ds), "max_abs_dlogp_all_legal": float(np.max(np.abs(ds))),
                   "legal_sets_equal": [x is None for x in v["dist_stored"]] == [y is None for y in v["dist_recomputed"]],
                   "argmax_equal": v["argmax_stored"] == v["argmax_recomputed"],
                   "tie_margin": v.get("tie_margin"), "tie_site": v.get("tie_site"),
                   "judged_by_flip": bool(v.get("judged_by_flip", False)),
                   "probe_p99": r["p99"], "weights_dumped": "policy_pt" in r.get("artifacts", {})})
    out["records"] = rr
    # ---------------- B. trajectories vs the ceilings
    out["trajectories"] = {}
    for key in ("p99", "max"):
        for ff in (20e6, 100e6):
            out["trajectories"][f"{key}_from{int(ff / 1e6)}M"] = trajectory(d, key, ff, a.B, 20261017)
    # windowed summary (10M)
    win = []
    for lo in np.arange(0, s[-1], 1e7):
        k = (s >= lo) & (s < lo + 1e7)
        if k.sum():
            win.append({"lo_M": float(lo / 1e6), "n": int(k.sum()), "max_median": float(np.median(d["max"][k])),
                        "max_max": float(d["max"][k].max()), "p99_median": float(np.median(d["p99"][k])),
                        "p99_max": float(d["p99"][k].max()), "events": int((d["max"][k] >= BAR).sum())})
    out["windows_10M"] = win
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "windows_10M"}, indent=1))


if __name__ == "__main__":
    main()
