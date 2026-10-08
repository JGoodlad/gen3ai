"""``main.ops.critic_readouts`` — the CRITIC LADDER's statistics, in one place.

Promoted 2026-09-08 out of the 75M read's measurement directory, where the same arithmetic
lived as three loose session scripts
(``designs/research_state/measurements/winprob_critic_75M_read_2026-09-08/identity_readout.py``,
``ipw_pass.py``, ``bootstrap_consistency.py``). Those copies STAY where they are — a committed
measurement must remain reproducible from the artifacts beside it — and this module is the
version the ladder's read imports, so every arm is read by identical code.

**What is left (P6 slice 6d-1, 2026-10-08).** The IDENTITY statistics — the join of ``cf_audit``'s
Monte-Carlo labels to the eval traces and everything computed on that sample (the sampler / capture-rate
reweightings of RULE OF EVIDENCE 17, the bias ``V - p_hat``, Murphy at the rollout level, the clock-tracking
contrast, the reliability cells) — were deleted with ``cf_audit`` (owner decision 2026-10-08), which already
refused every Rust-core trace (F-LH-4: no win-prob head recorded). What the ladder's read still imports
from here is the machinery both remaining halves share: the battle-clustered bootstrap over GROUPS, the
DELTA of two independent bootstraps and its three-way label, :func:`murphy_arrays` (the reference the
conditioning meters are held to), and the scaffolding gauge's arrays and metrics (the capture-rate weights
the gauge applies itself).

**Every interval resamples BATTLES**, never states: outcome labels are per-battle and broadcast,
so an i.i.d. bootstrap over decisions reports an interval roughly ``sqrt(states-per-battle)``
times too tight. A DELTA between two runs resamples the two INDEPENDENTLY and differences the
draws — the difference of independent bootstraps, which is the only interval a
"better/equivalent" sentence may be written from.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

N_BOOT = 4000
BOOT_SEED = 20260908
GATE_METRICS = ("resolution", "reliability", "ece", "skill")
#: strata the gate gates on, in the order the report prints them (`all` is context, never gated)
GATE_STRATA = ("all", "bot", "pool")


def independent_delta(a_point: float, a_draws: np.ndarray, b_point: float, b_draws: np.ndarray,
                      *, seed: int = BOOT_SEED) -> Dict[str, Any]:
    """ARM - CONTROL with the DIFFERENCE OF INDEPENDENT BOOTSTRAPS.

    The two samples are different runs' battles: nothing pairs them, so the delta's sampling
    distribution is the difference of the two marginal ones. Draws are paired POSITIONALLY after
    an independent shuffle of each — pairing the b-th draw of each arm without shuffling would
    import the ordering of two unrelated RNG streams as if it were structure.
    """
    delta = float(a_point) - float(b_point)
    if a_draws.size < 20 or b_draws.size < 20:
        return {"delta": delta, "ci": [float("nan"), float("nan")],
                "n_draws": int(min(a_draws.size, b_draws.size))}
    rng = np.random.default_rng(seed)
    n = int(min(a_draws.size, b_draws.size))
    d = rng.permutation(a_draws)[:n] - rng.permutation(b_draws)[:n]
    lo, hi = np.percentile(d, [2.5, 97.5])
    return {"delta": delta, "ci": [float(lo), float(hi)], "n_draws": n}


# --------------------------------------------------------------------------- the label

NO_FLOOR_NOTE = ("NO REPLICATE FLOOR — deltas are NOT DETECTED unless their CI clears zero, and "
                 "never DETECTED against a floor")


def label_delta(delta: float, ci: Sequence[float], floor: Optional[float]) -> Dict[str, Any]:
    """The registered three-way label (ledger 2026-09-08 REGISTRATION · THE CRITIC LADDER).

    * **DETECTED** — the delta's CI clears zero AND, when a floor exists, clears the floor.
    * **WITHIN FLOOR** — a floor exists and the point estimate sits inside it.
    * **NOT DETECTED** — everything else, including every delta whose CI is undefined.

    With no floor the DETECTED label carries an explicit qualifier: it is a detection against
    ZERO, which is a weaker claim than the registration's, and it must never be quoted as if a
    replicate floor had been cleared.
    """
    lo, hi = (float(ci[0]), float(ci[1])) if ci is not None and len(ci) == 2 else (
        float("nan"), float("nan"))
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return {"label": "NOT DETECTED", "qualifier": "no interval (too few battles)",
                "floor": floor, "clears_zero": False, "clears_floor": False}
    clears_zero = bool(lo > 0.0 or hi < 0.0)
    if floor is None:
        return {"label": "DETECTED" if clears_zero else "NOT DETECTED",
                "qualifier": "vs ZERO — NO FLOOR" if clears_zero else "CI covers zero",
                "floor": None, "clears_zero": clears_zero, "clears_floor": False}
    f = abs(float(floor))
    clears_floor = bool(lo > f or hi < -f)
    if clears_zero and clears_floor:
        return {"label": "DETECTED", "qualifier": f"CI clears the floor {f:+.4f}",
                "floor": f, "clears_zero": True, "clears_floor": True}
    if abs(float(delta)) <= f:
        return {"label": "WITHIN FLOOR", "qualifier": f"|delta| <= floor {f:.4f}",
                "floor": f, "clears_zero": clears_zero, "clears_floor": False}
    return {"label": "NOT DETECTED",
            "qualifier": ("CI covers zero" if not clears_zero else
                          f"CI does not clear the floor {f:.4f}"),
            "floor": f, "clears_zero": clears_zero, "clears_floor": False}


# --------------------------------------------------------------------------- Murphy's decomposition

def murphy_arrays(v: np.ndarray, k: np.ndarray, n: np.ndarray, w: np.ndarray,
                  nbins: int = 10) -> Dict[str, float]:
    """Murphy's decomposition at the ROLLOUT level, plus the BASE-RATE CAP.

    Every MC rollout is one Bernoulli outcome and the forecast is that state's V, so
    ``BS = REL - RES + UNC + (within-bin forecast variance)``; the last term is reported
    explicitly rather than hidden, because the forecast is not constant inside a bin.
    ``resolution_cap_share`` is RES / UNC — resolution as a fraction of the most any forecaster
    could resolve on this slice, which is the "27% of the base-rate cap" the 75M read quoted.

    Array-based on purpose: the cluster bootstrap re-runs this thousands of times, and rebuilding
    a list of row dicts per draw is the difference between seconds and minutes.
    """
    nan = float("nan")
    keys = ("brier", "reliability", "resolution", "uncertainty", "base_rate",
            "within_bin_forecast_var", "residual", "skill_score", "resolution_cap_share")
    N = float((w * n).sum())
    if not np.isfinite(N) or N <= 0:
        return {key: nan for key in keys}
    obar = float((w * k).sum() / N)
    bs = float((w * (n * v * v - 2 * v * k + k)).sum() / N)
    b = np.minimum(nbins - 1, (v * nbins).astype(int))
    rel = res = wbv = 0.0
    for j in range(nbins):
        m = b == j
        if not m.any():
            continue
        wn = float((w[m] * n[m]).sum())
        if wn <= 0:
            continue
        vbar = float((w[m] * n[m] * v[m]).sum() / wn)
        ok = float((w[m] * k[m]).sum() / wn)
        rel += wn / N * (vbar - ok) ** 2
        res += wn / N * (ok - obar) ** 2
        wbv += float((w[m] * n[m] * (v[m] - vbar) ** 2).sum() / N)
    unc = obar * (1 - obar)
    return {"brier": bs, "reliability": rel, "resolution": res, "uncertainty": unc,
            "base_rate": obar, "within_bin_forecast_var": wbv,
            "residual": bs - (rel - res + unc + wbv),
            "skill_score": 1 - bs / unc if unc else nan,
            "resolution_cap_share": res / unc if unc else nan}


# --------------------------------------------------------------------------- gate statistics

def group_boot_draws(groups: Sequence, stat_fn: Callable[[np.ndarray], float], *,
                     draws: int, seed: int) -> Tuple[float, np.ndarray]:
    """``(point, the raw draws)`` for ``stat_fn(row_indices)``, resampling GROUPS (battles)."""
    g = np.asarray(groups).ravel()
    uniq, inverse = np.unique(g, return_inverse=True)
    point = float(stat_fn(np.arange(g.size)))
    if uniq.size < 2:
        return point, np.empty(0)
    by_group = [np.nonzero(inverse == k)[0] for k in range(uniq.size)]
    rng = np.random.default_rng(seed)
    out: List[float] = []
    for _ in range(int(draws)):
        idx = np.concatenate([by_group[k] for k in rng.integers(0, uniq.size, uniq.size)])
        try:
            val = float(stat_fn(idx))
        except (ValueError, ZeroDivisionError, np.linalg.LinAlgError):
            continue
        if np.isfinite(val):
            out.append(val)
    return point, np.asarray(out, dtype=float)


def ci_of(point: float, draws: np.ndarray) -> List[float]:
    if draws.size < 20:
        return [float("nan"), float("nan")]
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return [float(lo), float(hi)]


def gauge_arrays(run_dir: str, step: int, *, seed: int = 0,
                 say=lambda _m: None) -> Tuple[Dict[str, dict], dict]:
    """``{stratum: {p, y, battles, w}}`` for ONE eval cycle, plus a coverage record.

    Every array is the scaffolding gauge's own — ``collect_slices`` for the rows,
    ``true_win_rates`` + ``selection_weights`` for the CAPTURE-RATE weights (rule 17), and
    ``main.scaffolding_gauge.opponent_class`` for the bot/pool split. Nothing is re-derived: a
    second estimator of a published statistic is a second answer to the same question.
    """
    from main.scaffolding_gauge import (SelectionWeightError, collect_slices, opponent_class,
                                        selection_weights, true_win_rates)

    slices, coverage = collect_slices(run_dir, seed=seed, say=say)
    if int(step) not in slices:
        raise KeyError(f"no usable rows at step {step} in {run_dir}")
    sl = slices[int(step)]
    try:
        rates = true_win_rates(run_dir).get(int(step), {})
    except SelectionWeightError as exc:
        raise KeyError(str(exc)) from exc
    w, wnote = selection_weights(sl["outcomes"], sl["battles"], sl["opponents"], rates)
    p = np.asarray(sl["win_probs"], dtype=np.float64)
    y = np.asarray(sl["outcomes"], dtype=np.float64)
    b = np.asarray(sl["battles"])
    cls = np.array([opponent_class(o) for o in np.asarray(sl["opponents"]).tolist()])
    out: Dict[str, dict] = {"all": {"p": p, "y": y, "battles": b, "w": w}}
    for c in ("bot", "pool"):
        m = cls == c
        if m.any():
            out[c] = {"p": p[m], "y": y[m], "battles": b[m], "w": w[m]}
    cov = {"per_step": coverage.get("per_step", {}).get(str(int(step)))
           or coverage.get("per_step", {}).get(int(step)),
           "n_traces_seen": coverage.get("n_traces_seen"),
           "n_traces_read": coverage.get("n_traces_read"),
           "n_traces_no_npz": coverage.get("n_traces_no_npz"),
           "n_traces_no_winprob": coverage.get("n_traces_no_winprob"),
           "selection_note": wnote}
    return out, cov


def gate_metric_draws(arr: dict, metric: str, *, bins: int = 10, draws: int,
                      seed: int) -> Tuple[float, np.ndarray]:
    """``(point, bootstrap draws)`` for one gauge metric on one stratum's arrays."""
    from agents.training.scaffolding import reliability_table

    p, y, w = arr["p"], arr["y"], arr["w"]

    def stat(idx):
        return float(reliability_table(p[idx], y[idx], bins=bins, weights=w[idx])[metric])
    return group_boot_draws(arr["battles"], stat, draws=draws, seed=seed)
