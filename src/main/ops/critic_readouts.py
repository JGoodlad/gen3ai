"""``main.ops.critic_readouts`` — the CRITIC LADDER's statistics, in one place.

Promoted 2026-09-08 out of the 75M read's measurement directory, where the same arithmetic
lived as three loose session scripts
(``designs/research_state/measurements/winprob_critic_75M_read_2026-09-08/identity_readout.py``,
``ipw_pass.py``, ``bootstrap_consistency.py``). Those copies STAY where they are — a committed
measurement must remain reproducible from the artifacts beside it — and this module is the
version the ladder's read imports, so every arm is read by identical code.

**Three selections stack on the identity sample, and they are different corrections.**

1. ``cf_audit`` draws a STRATIFIED sample (confidence decile x outcome x turn tercile, with a 4x
   boost on high-confidence-from-lost-battles). :func:`pop_weights` recombines it at the frame's
   own (decile, outcome) mass. This corrects the SAMPLER against the trace tree.
2. The trace tree is itself LOSS-ENRICHED by the eval quota (first 10 losses / 5 draws / 5 wins
   per opponent per cycle). :func:`capture_weights` reads each cycle's ``eval_manifest.json`` and
   returns ``1 / capture_rate(opponent, outcome)``. This corrects the TREE against the eval
   population, and it is RULE OF EVIDENCE 17 (ledger 2026-09-08): *any statistic on the
   eval-trace tree is reweighted by each cycle's recorded capture rate, or it is not a
   measurement.*
3. The ANCHOR arm is neither. Its estimand — "does the offline driver reproduce the recorded
   outcome under recorded dice?" — is a property of the REPLAY DRIVER, not of the eval
   population, and its frame is a census of the bot battles it draws from. Reweighting it would
   answer a question nobody asked. It is reported raw, and :func:`anchor_note` says so in print.

**Every interval resamples BATTLES**, never states: outcome labels are per-battle and broadcast,
so an i.i.d. bootstrap over decisions reports an interval roughly ``sqrt(states-per-battle)``
times too tight. A DELTA between two runs resamples the two INDEPENDENTLY and differences the
draws — the difference of independent bootstraps, which is the only interval a
"better/equivalent" sentence may be written from.
"""
from __future__ import annotations

import json
import math
import os
from collections import Counter, defaultdict
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

N_BOOT = 4000
BOOT_SEED = 20260908
TURN_BUCKETS = ("early (turn<=10)", "mid (11-24)", "late (turn>=25)")
GATE_METRICS = ("resolution", "reliability", "ece", "skill")
#: strata the gate gates on, in the order the report prints them (`all` is context, never gated)
GATE_STRATA = ("all", "bot", "pool")


def anchor_note() -> str:
    return ("the anchor (label-trust) arm is a CENSUS of the bot battles it draws from and its "
            "estimand is the REPLAY DRIVER's fidelity, not a property of the eval population — "
            "capture-rate reweighting does NOT apply to it and none is applied.")


def turn_bucket(turn: int) -> str:
    return TURN_BUCKETS[0] if turn <= 10 else (TURN_BUCKETS[1] if turn <= 24 else TURN_BUCKETS[2])


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


# --------------------------------------------------------------------------- selection weights

def capture_weights(run_dir: str, step: int) -> Optional[Dict[Tuple[str, int], float]]:
    """``{(opponent, outcome): 1 / capture_rate}`` from the cycle's ``eval_manifest.json``.

    ``outcome`` is 1 for a win and 0 for a loss, matching the frame's own coding. Returns
    ``None`` when the manifest records no selection block — a tree without one is SELECTION
    UNKNOWN and the caller must SAY so rather than silently read raw (rule 17).

    An opponent whose quota captured only one outcome class carries no usable rate on the empty
    side; that pair is omitted, and :func:`apply_capture` reports the rows it could not weight
    instead of leaving them at 1.0 beside corrected ones.
    """
    path = os.path.join(run_dir, "eval_traces", f"step_{int(step)}", "eval_manifest.json")
    if not os.path.exists(path):
        return None
    with open(path) as fh:
        man = json.load(fh)
    sel = man.get("selection")
    if not isinstance(sel, dict):
        return None
    per = sel.get("opponents") or sel.get("per_opponent") or {}
    out: Dict[Tuple[str, int], float] = {}
    for name, rec in per.items():
        for key, y in (("capture_rate_win", 1), ("capture_rate_loss", 0)):
            rate = rec.get(key)
            if isinstance(rate, (int, float)) and rate > 0:
                out[(str(name), y)] = 1.0 / float(rate)
    return out or None


def apply_capture(rows: Sequence[dict], w: np.ndarray,
                  cap: Optional[Dict[Tuple[str, int], float]]) -> Tuple[np.ndarray, float]:
    """Multiply per-row weights by the capture-rate IPW factor. Returns ``(w, covered)``.

    ``covered`` is the share of rows for which a capture rate existed. Rows with no rate get
    weight 0 — never 1 — because mixing corrected rows with uncorrected ones and calling the
    result corrected is the exact failure rule 17 was written against.
    """
    if cap is None:
        return np.asarray(w, dtype=float), 0.0
    f = np.array([cap.get((str(r["opponent"]), 1 if str(r["outcome"]) == "win" else 0), 0.0)
                  for r in rows], dtype=float)
    covered = float((f > 0).mean()) if len(rows) else 0.0
    out = np.asarray(w, dtype=float) * f
    m = out.mean()
    return (out / m if m else out), covered


def cell_of(r: dict) -> tuple:
    return (min(9, int(r["v"] * 10)), r["outcome"], r["opp_class"], turn_bucket(r["turn"]))


def pop_weights(rows: Sequence[dict], frame_cells: Dict[str, int]) -> Tuple[np.ndarray, float]:
    """``cf_audit``'s own recombination, restricted to the stratum being reported.

    ``w_i`` = (the stratum's frame mass in the row's (decile, outcome) cell) / (how many of that
    cell the sampler drew INSIDE the stratum). Returns the weights (mean 1) and the share of the
    stratum's frame mass the draw actually covers — a cell with zero draws is mass this readout
    cannot speak for, and it is reported rather than dropped in silence.
    """
    mass: Counter = Counter()
    keep = {(c[1], c[2], c[3]) for c in (cell_of(r) for r in rows)}
    for key, m in frame_cells.items():
        dec, oc, cls, tb = key.split("|")
        if (oc, cls, tb) in keep:
            mass[(int(dec), oc)] += m
    drawn: Counter = Counter((c[0], c[1]) for c in (cell_of(r) for r in rows))
    w = np.array([mass.get((c[0], c[1]), 0) / drawn[(c[0], c[1])]
                  for c in (cell_of(r) for r in rows)], dtype=float)
    covered = sum(m for k, m in mass.items() if drawn.get(k))
    total = sum(mass.values()) or 1
    return (w / w.mean() if w.mean() else w), covered / total


# --------------------------------------------------------------------------- bootstraps

def _by_battle(rows: Sequence[dict]) -> Tuple[List[str], Dict[str, List[int]]]:
    idx: Dict[str, List[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        idx[r["battle_key"]].append(i)
    return sorted(idx), idx


def cluster_boot(rows: Sequence[dict], stat: Callable[[np.ndarray, np.ndarray], float], *,
                 w: np.ndarray, draws: int = N_BOOT,
                 seed: int = BOOT_SEED) -> Tuple[float, float, float]:
    """``(point, lo, hi)`` — resampling BATTLES with replacement, weights carried per row."""
    keys, idx_by_battle = _by_battle(rows)
    point = stat(np.arange(len(rows)), w)
    if len(keys) < 2:
        return float(point), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    out: List[float] = []
    for _ in range(draws):
        pick = rng.integers(0, len(keys), len(keys))
        val = stat(np.concatenate([idx_by_battle[keys[j]] for j in pick]), w)
        if val is not None and np.isfinite(val):
            out.append(float(val))
    if len(out) < 20:
        return float(point), float("nan"), float("nan")
    lo, hi = np.percentile(np.asarray(out), [2.5, 97.5])
    return float(point), float(lo), float(hi)


def boot_draws(rows: Sequence[dict], stat: Callable[[np.ndarray, np.ndarray], float], *,
               w: np.ndarray, draws: int, seed: int) -> Tuple[float, np.ndarray]:
    """``(point, the bootstrap draws)`` — the raw draws, so a DELTA can difference two of them."""
    keys, idx_by_battle = _by_battle(rows)
    point = float(stat(np.arange(len(rows)), w))
    if len(keys) < 2:
        return point, np.empty(0)
    rng = np.random.default_rng(seed)
    out: List[float] = []
    for _ in range(draws):
        val = stat(np.concatenate([idx_by_battle[keys[j]] for j in rng.integers(
            0, len(keys), len(keys))]), w)
        if val is not None and np.isfinite(val):
            out.append(float(val))
    return point, np.asarray(out, dtype=float)


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


# --------------------------------------------------------------------------- identity statistics

def build_identity_payload(labels_path: str, traces: str) -> dict:
    """Join a ``cf_audit`` label file to the eval traces it came from.

    Same join, field for field, as the 75M read's ``identity_readout.py --build`` — it returns
    the payload instead of writing it, so the caller owns the path.
    """
    from agents.training.cf_audit import build_frame

    frame, skipped = build_frame(traces)
    by_key = {(d.battle, d.inv): d for d in frame}
    frame_mass: Counter = Counter()
    frame_cells: Counter = Counter()
    for d in frame:
        frame_mass[(min(9, int(d.win_prob * 10)), d.outcome)] += 1
        frame_cells[(min(9, int(d.win_prob * 10)), d.outcome, d.opp_class,
                     turn_bucket(d.turn))] += 1
    rows: List[dict] = []
    with open(labels_path) as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            base = r["battle"][: -len("_reconstruction.json")]
            d = by_key[(base, r["decision_idx"])]
            rows.append({
                "battle": d.short, "battle_key": d.short, "inv": d.inv, "turn": d.turn,
                "opponent": d.opponent, "opp_class": d.opp_class, "outcome": d.outcome,
                "v": d.win_prob, "value": d.value,
                "wins": int(round(r["label"] * r["n_rollouts"])),
                "n": int(r["n_rollouts"]), "mc": r["label"],
                "wilson_lo": r["wilson_lo"], "wilson_hi": r["wilson_hi"],
            })
    return {
        "schema": 1, "source_labels": os.path.abspath(labels_path),
        "traces": os.path.abspath(traces),
        "frame_decisions": len(frame), "frame_battles": len({d.battle for d in frame}),
        "frame_skipped": dict(skipped),
        "frame_mass": {f"{k[0]}/{k[1]}": v for k, v in sorted(frame_mass.items())},
        "frame_cells": {"|".join(map(str, k)): v for k, v in sorted(frame_cells.items())},
        "rows": rows,
    }


def bias_stat(rows: Sequence[dict]) -> Callable[[np.ndarray, np.ndarray], float]:
    v = np.array([r["v"] for r in rows], dtype=float)
    mc = np.array([r["mc"] for r in rows], dtype=float)

    def f(idx, w):
        ww = w[idx]
        return float(np.average((v - mc)[idx], weights=ww)) if ww.sum() > 0 else float("nan")
    return f


def _wcorr(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> float:
    if w.sum() <= 0:
        return float("nan")
    ma, mb = np.average(a, weights=w), np.average(b, weights=w)
    va = np.average((a - ma) ** 2, weights=w)
    vb = np.average((b - mb) ** 2, weights=w)
    if va <= 0 or vb <= 0:
        return float("nan")
    return float(np.average((a - ma) * (b - mb), weights=w) / math.sqrt(va * vb))


def turn_contrast_stat(rows: Sequence[dict]) -> Callable[[np.ndarray, np.ndarray], float]:
    """``corr(turn, V) - corr(turn, MC)`` — the registered clock-tracking contrast.

    Arm A's committed value (+0.3089 [+0.0828, +0.5101], design note
    ``winprob_critic_ladder_2026-09-08.md`` §L2) is the POOLED, UNWEIGHTED Pearson pair; that is
    the estimand every ladder arm is compared against, so the registered call passes ``w = 1``.
    The population-weighted variant is computed beside it and reported as a SEPARATE row, never
    substituted for the registered one.
    """
    t = np.array([r["turn"] for r in rows], dtype=float)
    v = np.array([r["v"] for r in rows], dtype=float)
    mc = np.array([r["mc"] for r in rows], dtype=float)

    def f(idx, w):
        ww = w[idx]
        return _wcorr(t[idx], v[idx], ww) - _wcorr(t[idx], mc[idx], ww)
    return f


def turn_corr_parts(rows: Sequence[dict], w: np.ndarray) -> Dict[str, float]:
    t = np.array([r["turn"] for r in rows], dtype=float)
    v = np.array([r["v"] for r in rows], dtype=float)
    mc = np.array([r["mc"] for r in rows], dtype=float)
    return {"corr_turn_v": _wcorr(t, v, w), "corr_turn_mc": _wcorr(t, mc, w)}


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


def _mc_cols(rows: Sequence[dict]) -> "Tuple[np.ndarray, np.ndarray, np.ndarray]":
    return (np.array([r["v"] for r in rows], dtype=float),
            np.array([r["wins"] for r in rows], dtype=float),
            np.array([r["n"] for r in rows], dtype=float))


def murphy(rows: Sequence[dict], w: np.ndarray, nbins: int = 10) -> Dict[str, float]:
    v, k, n = _mc_cols(rows)
    return murphy_arrays(v, k, n, np.asarray(w, dtype=float), nbins)


def murphy_stat(rows: Sequence[dict], term: str,
                nbins: int = 10) -> Callable[[np.ndarray, np.ndarray], float]:
    v, k, n = _mc_cols(rows)

    def f(idx, w):
        return murphy_arrays(v[idx], k[idx], n[idx], w[idx], nbins)[term]
    return f


def reliability_cells(rows: Sequence[dict], w: np.ndarray, nbins: int = 10) -> List[dict]:
    """Per-bin reliability rows. Wilson is on the UNWEIGHTED rollout counts of the cell — the
    weights change the estimand, not the number of coin flips behind it."""
    v = np.array([r["v"] for r in rows], dtype=float)
    k = np.array([r["wins"] for r in rows], dtype=float)
    n = np.array([r["n"] for r in rows], dtype=float)
    b = np.minimum(nbins - 1, (v * nbins).astype(int))
    cells: List[dict] = []
    for j in range(nbins):
        m = b == j
        if not m.any():
            continue
        wn = float((w[m] * n[m]).sum())
        if wn <= 0:
            continue
        vbar = float((w[m] * n[m] * v[m]).sum() / wn)
        obar = float((w[m] * k[m]).sum() / wn)
        lo, hi = wilson(int(k[m].sum()), int(n[m].sum()))
        cells.append({"bin": f"[{j / nbins:.1f},{(j + 1) / nbins:.1f})",
                      "n_states": int(m.sum()), "n_rollouts": int(n[m].sum()),
                      "mean_v_pop": vbar, "mc_pop": obar,
                      "wilson_lo": lo, "wilson_hi": hi, "gap_pop": vbar - obar})
    return cells


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
