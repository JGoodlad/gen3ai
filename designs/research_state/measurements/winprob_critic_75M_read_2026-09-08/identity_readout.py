#!/usr/bin/env python3
"""THE IDENTITY-TEST READOUT — V(s) against the Monte-Carlo win fraction, 75M read.

Two modes, and the split is the point: `--build` joins a `cf_audit` label file to the eval
traces it came from (that needs `models/`, which lives only in the MAIN checkout) and writes
`identity_labels.json` BESIDE this script; every other invocation reads that committed file and
recomputes every number in `identity_test.md` from it. So the report is reproducible from the
tree alone, which is the property the measurement-readout gate exists to protect.

    python identity_readout.py --build --labels <cf_labels/*.jsonl> --traces <eval_traces/step_N>
    python identity_readout.py                 # recompute the tables from identity_labels.json
    python identity_readout.py --check         # inputs resolve? (the gate's contract)

WHY THE WEIGHTS. `cf_audit` draws a STRATIFIED sample (confidence decile x outcome x turn
tercile) with a 4x boost on the high-confidence-from-lost-battles cell, on top of eval traces
that are already loss-enriched by an explicit quota. A pooled mean over that sample measures the
sampler. Every aggregate here is therefore reported twice: SAMPLE (what was drawn) and
POPULATION (recombined at the frame's own (decile, outcome) mass, the same recombination
`cf_audit.resolution_cells` uses).

WHY THE BOOTSTRAP IS OVER BATTLES. Decisions inside one battle share a board, a matchup and a
dice stream. Pooling them as independent is the Simpson trap this tree has already been bitten
by; every interval below resamples BATTLES with replacement.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from typing import Dict, List, Sequence

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "identity_labels.json")

N_BOOT = 10000
BOOT_SEED = 20260908
TURN_BUCKETS = ("early (turn<=10)", "mid (11-24)", "late (turn>=25)")


def turn_bucket(turn: int) -> str:
    return TURN_BUCKETS[0] if turn <= 10 else (TURN_BUCKETS[1] if turn <= 24 else TURN_BUCKETS[2])


# ---------------------------------------------------------------------------
# build: join labels -> traces
# ---------------------------------------------------------------------------

def build(labels_path: str, traces: str, out: str) -> None:
    sys.path.insert(0, os.path.join(os.environ.get("GEN3AI_SRC", ""), "")) if os.environ.get(
        "GEN3AI_SRC") else None
    from agents.training.cf_audit import build_frame  # needs PYTHONPATH=src

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
            r = json.loads(line)
            base = r["battle"][: -len("_reconstruction.json")]
            d = by_key[(base, r["decision_idx"])]
            rows.append({
                "battle": d.short,               # "<opponent>/<name>" — UNIQUE (a bare basename repeats across opponent dirs)
                "battle_key": d.short,
                "inv": d.inv,
                "turn": d.turn,
                "opponent": d.opponent,
                "opp_class": d.opp_class,
                "outcome": d.outcome,
                "v": d.win_prob,               # the head's recorded P(win) at this state
                "value": d.value,
                "wins": int(round(r["label"] * r["n_rollouts"])),
                "n": int(r["n_rollouts"]),
                "mc": r["label"],
                "wilson_lo": r["wilson_lo"], "wilson_hi": r["wilson_hi"],
            })
    payload = {
        "schema": 1,
        "source_labels": os.path.abspath(labels_path),
        "traces": os.path.abspath(traces),
        "frame_decisions": len(frame),
        "frame_battles": len({d.battle for d in frame}),
        "frame_skipped": dict(skipped),
        "frame_mass": {f"{k[0]}/{k[1]}": v for k, v in sorted(frame_mass.items())},
        # decile | outcome | opp_class | turn bucket — the FINEST frame mass, so any stratum
        # below is recombined against ITS OWN population rather than the whole frame's mix.
        "frame_cells": {"|".join(map(str, k)): v for k, v in sorted(frame_cells.items())},
        "rows": rows,
    }
    with open(out, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"wrote {out}: {len(rows)} labels / {len({r['battle_key'] for r in rows})} battles")


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def wilson(k: int, n: int, z: float = 1.96) -> "tuple[float, float]":
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def cell_of(r: dict) -> tuple:
    return (min(9, int(r["v"] * 10)), r["outcome"], r["opp_class"], turn_bucket(r["turn"]))


def pop_weights(rows: Sequence[dict], frame_cells: Dict[str, int]
                ) -> "tuple[np.ndarray, float]":
    """w_i = (this stratum's frame mass in the row's (decile, outcome) cell) / (how many of that
    cell the sampler drew INSIDE this stratum) — cf_audit's recombination, restricted to the
    stratum being reported so a bot/sentinel or turn-bucket row is not reweighted to the whole
    frame's decile mix. Returns the weights (mean 1) and the share of the stratum's frame mass
    that the draw actually covers — a cell with zero draws is mass this readout cannot speak for,
    and it is reported rather than dropped in silence."""
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


def _bias(v: np.ndarray, mc: np.ndarray, w: np.ndarray) -> float:
    return float(np.average(v - mc, weights=w))


def cluster_boot(rows: Sequence[dict], stat, *, w: np.ndarray, draws: int = N_BOOT,
                 seed: int = BOOT_SEED) -> "tuple[float, float, float]":
    """(point, lo, hi) — resampling BATTLES with replacement, weights carried per row."""
    idx_by_battle: "Dict[str, List[int]]" = defaultdict(list)
    for i, r in enumerate(rows):
        idx_by_battle[r["battle_key"]].append(i)
    keys = sorted(idx_by_battle)
    all_idx = np.arange(len(rows))
    point = stat(all_idx, w)
    rng = np.random.default_rng(seed)
    out = np.empty(draws, dtype=float)
    n_ok = 0
    for b in range(draws):
        pick = rng.integers(0, len(keys), len(keys))
        idx = np.concatenate([idx_by_battle[keys[j]] for j in pick])
        val = stat(idx, w)
        if val is not None and np.isfinite(val):
            out[n_ok] = val
            n_ok += 1
    lo, hi = np.percentile(out[:n_ok], [2.5, 97.5])
    return float(point), float(lo), float(hi)


def bias_stat(rows: Sequence[dict]):
    v = np.array([r["v"] for r in rows])
    mc = np.array([r["mc"] for r in rows])

    def f(idx, w):
        return float(np.average((v - mc)[idx], weights=w[idx]))
    return f


def reliability(rows: Sequence[dict], w: np.ndarray, nbins: int = 10,
                boot_draws: int = 4000) -> List[dict]:
    v = np.array([r["v"] for r in rows])
    k = np.array([r["wins"] for r in rows], dtype=float)
    n = np.array([r["n"] for r in rows], dtype=float)
    b = np.minimum(nbins - 1, (v * nbins).astype(int))
    cells = []
    for j in range(nbins):
        m = b == j
        if not m.any():
            continue
        wn = (w[m] * n[m]).sum()
        vbar = float((w[m] * n[m] * v[m]).sum() / wn)
        obar = float((w[m] * k[m]).sum() / wn)
        # Wilson on the UNWEIGHTED rollout counts of the cell (the honest sampling interval;
        # the weights change the estimand, not the number of coin flips behind it).
        lo, hi = wilson(int(k[m].sum()), int(n[m].sum()))
        cells.append({
            "bin": f"[{j / nbins:.1f},{(j + 1) / nbins:.1f})",
            "n_states": int(m.sum()), "n_battles": len({rows[i]["battle_key"]
                                                        for i in np.flatnonzero(m)}),
            "n_rollouts": int(n[m].sum()),
            "mean_v_sample": float(v[m].mean()), "mean_v_pop": vbar,
            "mc_sample": float(k[m].sum() / n[m].sum()), "mc_pop": obar,
            "wilson_lo": lo, "wilson_hi": hi,
            "gap_pop": vbar - obar,
        })
        sub = [rows[i] for i in np.flatnonzero(m)]
        wsub = w[m]

        def _phat(idx, ww, _k=k[m], _n=n[m]):
            return float((ww[idx] * _k[idx]).sum() / (ww[idx] * _n[idx]).sum())

        def _gap(idx, ww, _v=v[m], _k=k[m], _n=n[m]):
            wn = (ww[idx] * _n[idx]).sum()
            return float((ww[idx] * _n[idx] * _v[idx]).sum() / wn
                         - (ww[idx] * _k[idx]).sum() / wn)
        _, plo, phi = cluster_boot(sub, _phat, w=wsub, draws=boot_draws)
        _, glo, ghi = cluster_boot(sub, _gap, w=wsub, draws=boot_draws)
        cells[-1]["mc_pop_ci"] = [plo, phi]
        cells[-1]["gap_pop_ci"] = [glo, ghi]
    return cells


def brier(rows: Sequence[dict], w: np.ndarray, nbins: int = 10) -> dict:
    """Murphy's decomposition at the ROLLOUT level: every MC rollout is one Bernoulli outcome
    y and the forecast is that state's V. BS = REL - RES + UNC + (within-bin term), the last
    reported explicitly rather than hidden, because the forecast is not constant inside a bin."""
    v = np.array([r["v"] for r in rows])
    k = np.array([r["wins"] for r in rows], dtype=float)
    n = np.array([r["n"] for r in rows], dtype=float)
    W = w * 1.0
    N = float((W * n).sum())
    obar = float((W * k).sum() / N)
    bs = float((W * (n * v * v - 2 * v * k + k)).sum() / N)
    b = np.minimum(nbins - 1, (v * nbins).astype(int))
    rel = res = wbv = 0.0
    for j in range(nbins):
        m = b == j
        if not m.any():
            continue
        wn = float((W[m] * n[m]).sum())
        vbar = float((W[m] * n[m] * v[m]).sum() / wn)
        ok = float((W[m] * k[m]).sum() / wn)
        rel += wn / N * (vbar - ok) ** 2
        res += wn / N * (ok - obar) ** 2
        wbv += float((W[m] * n[m] * (v[m] - vbar) ** 2).sum() / N)
    unc = obar * (1 - obar)
    return {"brier": bs, "reliability": rel, "resolution": res, "uncertainty": unc,
            "base_rate": obar, "within_bin_forecast_var": wbv,
            "residual": bs - (rel - res + unc + wbv),
            "skill_score": 1 - bs / unc if unc else float("nan")}


def brier_ci(rows: Sequence[dict], w: np.ndarray, *, draws: int = 4000) -> dict:
    """Cluster-bootstrap (over BATTLES) intervals for every Murphy term, each draw recomputing
    the decomposition from scratch on the resampled battles — a CI whose resamples ran different
    arithmetic from the point estimate is a CI of nothing."""
    out = {}
    for term in ("brier", "reliability", "resolution", "uncertainty", "skill_score"):
        def f(idx, ww, _t=term, _rows=rows):
            return brier([_rows[i] for i in idx], ww[idx])[_t]
        p, lo, hi = cluster_boot(rows, f, w=w, draws=draws)
        out[term] = {"point": p, "ci": [lo, hi]}
    return out


def sd_true_excess(rows: Sequence[dict], w: np.ndarray) -> float:
    """The registered meter's estimator, over the whole labelled set: the within-set spread of
    the true p, with the R-rollout binomial floor subtracted."""
    p = np.array([r["mc"] for r in rows])
    n = np.array([r["n"] for r in rows], dtype=float)
    mu = float(np.average(p, weights=w))
    var = float(np.average((p - mu) ** 2, weights=w))
    floor = float(np.average(p * (1 - p) / (n - 1), weights=w))
    return math.sqrt(max(0.0, var - floor))


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def analyse(payload: dict) -> dict:
    rows = payload["rows"]
    fc = payload["frame_cells"]
    w_pop, cover = pop_weights(rows, fc)
    w_one = np.ones(len(rows))

    def block(sub: Sequence[dict], name: str) -> dict:
        if not sub:
            return {"stratum": name, "n": 0}
        wp, cov = pop_weights(sub, fc)
        wo = np.ones(len(sub))
        f = bias_stat(sub)
        p_s, lo_s, hi_s = cluster_boot(sub, f, w=wo)
        p_p, lo_p, hi_p = cluster_boot(sub, f, w=wp)
        return {
            "stratum": name, "n": len(sub),
            "n_battles": len({r["battle_key"] for r in sub}),
            "frame_mass_covered": cov,
            "mean_v": float(np.mean([r["v"] for r in sub])),
            "mean_mc": float(np.mean([r["mc"] for r in sub])),
            "bias_sample": p_s, "ci_sample": [lo_s, hi_s],
            "bias_pop": p_p, "ci_pop": [lo_p, hi_p],
            "brier_pop": brier(sub, wp),
            "sd_true_excess_pop": sd_true_excess(sub, wp),
        }

    out = {
        "n_labels": len(rows),
        "n_battles": len({r["battle_key"] for r in rows}),
        "n_rollouts": int(sum(r["n"] for r in rows)),
        "overall": block(rows, "ALL"),
        "reliability_sample_and_pop": reliability(rows, w_pop),
        "reliability_unweighted": reliability(rows, w_one),
        "brier_sample": brier(rows, w_one),
        "brier_pop": brier(rows, w_pop),
        "frame_mass_covered": cover,
        "brier_pop_ci": brier_ci(rows, w_pop),
        "brier_sample_ci": brier_ci(rows, w_one),
        "by_turn": [block([r for r in rows if turn_bucket(r["turn"]) == nm], nm)
                    for nm in TURN_BUCKETS],
        "by_stratum": [block([r for r in rows if r["opp_class"] == c], c)
                       for c in ("bot", "sentinel")],
        "by_outcome": [block([r for r in rows if r["outcome"] == o], o) for o in ("win", "loss")],
        "by_opponent": [block([r for r in rows if r["opponent"] == o], o)
                        for o in sorted({r["opponent"] for r in rows})],
        "boot": {"draws": N_BOOT, "seed": BOOT_SEED, "unit": "battle"},
    }
    return out


def main(argv: "Sequence[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="inputs resolve? (readout gate)")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--traces", default=None)
    ap.add_argument("--out", default=RAW)
    ap.add_argument("--json-out", default=os.path.join(HERE, "identity_stats.json"))
    args = ap.parse_args(argv)

    if args.check:
        missing = [p for p in (RAW,) if not os.path.exists(p)]
        for p in missing:
            print(f"MISSING: {p}", file=sys.stderr)
        return 1 if missing else 0
    if args.build:
        if not (args.labels and args.traces):
            print("--build needs --labels and --traces", file=sys.stderr)
            return 2
        build(args.labels, args.traces, args.out)
        return 0

    with open(RAW) as fh:
        payload = json.load(fh)
    stats = analyse(payload)
    with open(args.json_out, "w") as fh:
        json.dump(stats, fh, indent=1)
    o = stats["overall"]
    print(f"labels {stats['n_labels']} / battles {stats['n_battles']} / "
          f"rollouts {stats['n_rollouts']}")
    print(f"bias V-p_hat  SAMPLE {o['bias_sample']:+.4f} "
          f"[{o['ci_sample'][0]:+.4f}, {o['ci_sample'][1]:+.4f}]   "
          f"POP {o['bias_pop']:+.4f} [{o['ci_pop'][0]:+.4f}, {o['ci_pop'][1]:+.4f}]")
    b = stats["brier_pop"]
    print(f"brier(pop) {b['brier']:.4f} = REL {b['reliability']:.4f} - RES {b['resolution']:.4f} "
          f"+ UNC {b['uncertainty']:.4f} + WBV {b['within_bin_forecast_var']:.4f} "
          f"(resid {b['residual']:+.2e}); base rate {b['base_rate']:.4f}")
    for c in stats["reliability_sample_and_pop"]:
        print(f"  {c['bin']}  n={c['n_states']:4d}  V={c['mean_v_pop']:.3f}  "
              f"mc={c['mc_pop']:.3f}  [{c['wilson_lo']:.3f},{c['wilson_hi']:.3f}]")
    for row in stats["by_turn"] + stats["by_stratum"]:
        if row["n"]:
            print(f"  {row['stratum']:<18} n={row['n']:4d}  bias_pop {row['bias_pop']:+.4f} "
                  f"[{row['ci_pop'][0]:+.4f}, {row['ci_pop'][1]:+.4f}]")
    print(f"wrote {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# ---------------------------------------------------------------------------
# CONTRASTS — a stratum-vs-stratum sentence needs the DIFFERENCE's interval, never two
# overlapping bars (rule of evidence: a bar vs a point estimate is vacuous).
# ---------------------------------------------------------------------------

def contrast(rows_a: Sequence[dict], rows_b: Sequence[dict], fc: Dict[str, int], *,
             draws: int = 4000, seed: int = BOOT_SEED) -> dict:
    """POP-weighted bias(A) - bias(B), resampling battles INDEPENDENTLY inside each arm."""
    wa, _ = pop_weights(rows_a, fc)
    wb, _ = pop_weights(rows_b, fc)
    fa, fb = bias_stat(rows_a), bias_stat(rows_b)

    def arm(rows, w, f):
        idx_by_battle: "Dict[str, List[int]]" = defaultdict(list)
        for i, r in enumerate(rows):
            idx_by_battle[r["battle_key"]].append(i)
        keys = sorted(idx_by_battle)
        return keys, idx_by_battle, w, f

    A, B = arm(rows_a, wa, fa), arm(rows_b, wb, fb)
    point = A[3](np.arange(len(rows_a)), wa) - B[3](np.arange(len(rows_b)), wb)
    rng = np.random.default_rng(seed)
    out = np.empty(draws)
    for d in range(draws):
        vals = []
        for keys, ibb, w, f in (A, B):
            pick = rng.integers(0, len(keys), len(keys))
            vals.append(f(np.concatenate([ibb[keys[j]] for j in pick]), w))
        out[d] = vals[0] - vals[1]
    lo, hi = np.percentile(out, [2.5, 97.5])
    return {"delta": float(point), "ci": [float(lo), float(hi)], "draws": draws}
