"""Monte-Carlo OPERATING CHARACTERISTICS of the SPRT promotion rule (T6, ``gen3_sprt_promotion_v1``).

CPU only, no battles. For each TRUE per-game score of a candidate vs the pool (0.45 / 0.50 / 0.525 / 0.55 /
0.60) it simulates many SPRTs EXACTLY as ``agents.training.sprt.SprtState`` runs them — pentanomial GSPRT,
H0 0.50 vs H1 0.55, alpha = beta = 0.05, decisions only at BATCH boundaries — and reports the promotion
(accept) rate, the rejection rate, and the expected / median / p95 pairs to decide.

THE GAME MODEL (stated, because the operating characteristics depend on it):

* a pair is played against one of K = 5 sentinels, round-robin (the eval's pool); sentinel k carries a
  logit offset o_k (``--heterogeneity``: o = h * (-1, -0.5, 0, 0.5, 1)) — the pool is not one opponent;
* the pair's TEAM IMBALANCE d ~ Normal(0, tau^2) in logits: game 1 (candidate on team A) is won with
  probability sigmoid(m + o_k + d), game 2 (candidate on team B, same seed) with sigmoid(m + o_k - d) —
  the mirror. tau = 0 is "no team effect" (the two games independent given the matchup); tau = 1 and 2 are
  moderate and strong team effects (the within-pair NEGATIVE correlation mirroring exploits);
* no draws (0 drawn in 145k archived eval traces);
* m is solved so the EXPECTED per-game score over (k, d) equals the stated true rate exactly.

PHASE 1 runs UNTRUNCATED Wald tests (a large safety cap) at every tau to measure the pairs-to-decide
distribution; the cap is the p95 of the WORST cell (the indifference point, 0.525, at the weakest team
effect), rounded UP to a whole batch. PHASE 2 re-runs every cell WITH that cap (reaching it = REJECT) for
each candidate rule — Wald or Siegmund-overshoot-corrected bounds x a minimum pair count — and applies the
PRE-DECLARED choice (see ``choice_rule`` in the output). PHASE 3 checks the chosen rule on a homogeneous pool.

Usage (writes results.json + results.md beside this script):
    python designs/research_state/measurements/sprt_promotion/simulate.py [--runs 20000] [--seed 0]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RATES = (0.45, 0.50, 0.525, 0.55, 0.60)
TAUS = (0.0, 1.0, 2.0)
BASE_TAU, BASE_H = 1.0, 0.4
K = 5
SCORES = np.arange(5) / 4.0


def _sprt_mod():
    sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "src"))
    from agents.training import sprt
    return sprt


def _lam_many(p: np.ndarray, mu: float) -> np.ndarray:
    lo = np.full(len(p), -1.0 / (1.0 - mu) + 1e-12)
    hi = np.full(len(p), 1.0 / mu - 1e-12)
    d = SCORES - mu
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f = (p * d / (1.0 + mid[:, None] * d)).sum(axis=1)
        pos = f > 0
        lo = np.where(pos, mid, lo)
        hi = np.where(pos, hi, mid)
    return 0.5 * (lo + hi)


def llr_many(counts: np.ndarray, p0: float, p1: float, reg: float, with_var: bool = False):
    """Vectorized ``sprt.llr`` (and ``sprt.llr_increment_var``) over rows of pentanomial counts — the same
    regularization, the same 200-step bisection."""
    n = counts.sum(axis=1).astype(float)
    c = np.where(counts > 0, counts.astype(float), reg)
    p = c / c.sum(axis=1, keepdims=True)
    l0, l1 = _lam_many(p, p0), _lam_many(p, p1)
    t = np.log1p(l0[:, None] * (SCORES - p0)) - np.log1p(l1[:, None] * (SCORES - p1))
    m = (p * t).sum(axis=1)
    out = np.where(n > 0, n * m, 0.0)
    if not with_var:
        return out
    var = np.maximum(0.0, (p * (t - m[:, None]) ** 2).sum(axis=1))
    return out, np.where(n > 0, var, 0.0)


def offsets(h: float) -> np.ndarray:
    return h * np.array([-1.0, -0.5, 0.0, 0.5, 1.0])


def expected_score(m: float, tau: float, h: float) -> float:
    """E[per-game score] over sentinels and team imbalance — Gauss-Hermite quadrature over d."""
    x, w = np.polynomial.hermite_e.hermegauss(80)
    w = w / w.sum()
    o = offsets(h)[:, None]
    d = tau * x[None, :]
    s = 0.5 * (1 / (1 + np.exp(-(m + o + d))) + 1 / (1 + np.exp(-(m + o - d))))
    return float((s * w[None, :]).sum(axis=1).mean())


def solve_m(rate: float, tau: float, h: float) -> float:
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if expected_score(mid, tau, h) < rate:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def pair_probs(m: float, tau: float, h: float) -> np.ndarray:
    """Per-sentinel pair-category probabilities [K, 5] (no draws ⇒ only categories 0, 2, 4)."""
    x, w = np.polynomial.hermite_e.hermegauss(80)
    w = w / w.sum()
    out = np.zeros((K, 5))
    for k, o in enumerate(offsets(h)):
        a = 1 / (1 + np.exp(-(m + o + tau * x)))
        b = 1 / (1 + np.exp(-(m + o - tau * x)))
        out[k, 4] = (w * a * b).sum()
        out[k, 0] = (w * (1 - a) * (1 - b)).sum()
        out[k, 2] = 1.0 - out[k, 4] - out[k, 0]
    return out


def simulate(rate: float, tau: float, h: float, *, runs: int, batch: int, cap: int, p0: float, p1: float,
             alpha: float, beta: float, reg: float, rng: np.random.Generator, min_pairs: int = 0,
             mode: str = "wald", rho: float = 0.5826) -> dict:
    """``runs`` SPRTs exactly as ``SprtState`` runs them: checks only at batch boundaries, no bound decision
    before ``min_pairs``, ``mode`` "wald" or "siegmund" (both bounds moved inward by rho * sigma_check, the
    per-batch LLR standard deviation estimated from the observed pentanomial), the cap = REJECT."""
    m = solve_m(rate, tau, h)
    probs = pair_probs(m, tau, h)
    lo_b, hi_b = math.log(beta / (1 - alpha)), math.log((1 - beta) / alpha)
    counts = np.zeros((runs, 5), dtype=np.int64)
    decided = np.zeros(runs, dtype=bool)
    accept = np.zeros(runs, dtype=bool)
    by_cap = np.zeros(runs, dtype=bool)
    n_at = np.zeros(runs, dtype=np.int64)
    played = 0
    while played < cap and not decided.all():
        b = min(batch, cap - played)
        live = ~decided
        nl = int(live.sum())
        # round-robin over the K sentinels: pair i of the test is against sentinel i mod K
        add = np.zeros((nl, 5), dtype=np.int64)
        for j in range(b):
            k = (played + j) % K
            cat = rng.choice(5, size=nl, p=probs[k])
            np.add.at(add, (np.arange(nl), cat), 1)
        counts[live] += add
        played += b
        if played < min_pairs:
            continue
        v, var = llr_many(counts[live], p0, p1, reg, with_var=True)
        lo, hi = np.full(nl, lo_b), np.full(nl, hi_b)
        if mode == "siegmund":
            shift = rho * np.sqrt(batch * var)
            lo, hi = np.minimum(lo_b + shift, 0.0), np.maximum(hi_b - shift, 0.0)
        idx = np.flatnonzero(live)
        acc = v >= hi
        rej = (v <= lo) & ~acc
        hit = acc | rej
        decided[idx[hit]] = True
        accept[idx[acc]] = True
        n_at[idx[hit]] = played
    # cap reached without a decision ⇒ REJECT
    rest = ~decided
    by_cap[rest] = True
    n_at[rest] = played
    return {"rate": rate, "tau": tau, "heterogeneity": h, "mode": mode, "min_pairs": min_pairs, "cap": cap,
            "m_logit": round(m, 6), "runs": runs,
            "accept_rate": float(accept.mean()), "reject_rate": float(1 - accept.mean()),
            "reject_by_cap_rate": float(by_cap.mean()),
            "mean_pairs": float(n_at.mean()), "median_pairs": float(np.median(n_at)),
            "p95_pairs": float(np.percentile(n_at, 95)), "max_pairs": int(n_at.max())}


def _errors_ok(rows, alpha, beta) -> bool:
    """Both error rates at or under nominal BY A MONTE-CARLO MARGIN — the rate's upper 95% bound
    (rate + 1.96 * its binomial standard error over the runs) must clear the nominal value, so a cell that
    sits within simulation noise of the bar does not pass by luck (checks pass or fail deterministically)."""
    def up(x, n):
        return x + 1.96 * math.sqrt(max(x * (1 - x), 0.0) / n)
    return all((up(r["accept_rate"], r["runs"]) <= alpha if r["rate"] <= 0.50 else True)
               and (up(r["reject_rate"], r["runs"]) <= beta if r["rate"] >= 0.55 else True) for r in rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=40, help="pairs per SPRT check (the shipped batch)")
    ap.add_argument("--safety-cap", type=int, default=20000, help="phase-1 cap (pairs) — effectively untruncated")
    ap.add_argument("--min-grid", default="40,80,160", help="minimum-pair counts compared")
    ap.add_argument("--out", default=HERE)
    args = ap.parse_args(argv)

    sprt = _sprt_mod()
    p0, p1, alpha, beta, reg = sprt.P0, sprt.P1, sprt.ALPHA, sprt.BETA, sprt.REGULARIZE
    mins = [int(x) for x in args.min_grid.split(",")]

    # self-check: the vectorized LLR and increment variance equal the shipped scalar ones
    chk = np.random.default_rng(123).integers(0, 40, size=(50, 5))
    chk[::7, 1:4] = 0
    want = np.array([sprt.llr(list(map(int, r)), p0, p1) for r in chk])
    wvar = np.array([sprt.llr_increment_var(list(map(int, r)), p0, p1) for r in chk])
    got, gvar = llr_many(chk, p0, p1, reg, with_var=True)
    assert np.allclose(want, got, atol=1e-9) and np.allclose(wvar, gvar, atol=1e-9), \
        "the vectorized LLR disagrees with agents.training.sprt"

    t0 = time.time()
    rng = np.random.default_rng(args.seed)
    kw = dict(runs=args.runs, batch=args.batch, p0=p0, p1=p1, alpha=alpha, beta=beta, reg=reg, rng=rng)

    # PHASE 1 — Wald, first check after one batch, UNTRUNCATED, every team-effect strength: the cap.
    phase1 = [simulate(r, t, BASE_H, cap=args.safety_cap, min_pairs=args.batch, **kw) for t in TAUS for r in RATES]
    worst = max(phase1, key=lambda row: row["p95_pairs"])
    cap = int(math.ceil(worst["p95_pairs"] / args.batch) * args.batch)

    # PHASE 2 — the candidate rules WITH the cap, at every team-effect strength.
    cands = [(mode, mn) for mode in ("wald", "siegmund") for mn in mins]
    phase2 = {f"{mode}/min{mn}": [simulate(r, t, BASE_H, cap=cap, min_pairs=mn, mode=mode, **kw)
                                   for t in TAUS for r in RATES] for mode, mn in cands}

    def mean_pairs(rows):
        return float(np.mean([r["mean_pairs"] for r in rows]))

    # THE PRE-DECLARED CHOICE. min_pairs: the SMALLEST of the grid whose Wald rule holds both error rates
    # at or under nominal (by the Monte-Carlo margin) in every cell. Bounds: Wald, unless a Siegmund rule
    # (any minimum) also holds them by the margin AND spends <= 85% of the chosen Wald rule's mean pairs;
    # among several such, the cheapest.
    ok_mins = [mn for mn in mins if _errors_ok(phase2[f"wald/min{mn}"], alpha, beta)]
    chosen_min = ok_mins[0] if ok_mins else max(mins)
    wald_mean = mean_pairs(phase2[f"wald/min{chosen_min}"])
    sieg = {mn: {"errors_ok": _errors_ok(phase2[f"siegmund/min{mn}"], alpha, beta),
                 "saving_vs_chosen_wald": round(1.0 - mean_pairs(phase2[f"siegmund/min{mn}"]) / wald_mean, 4)}
            for mn in mins}
    qualifying = [mn for mn in mins if sieg[mn]["errors_ok"] and sieg[mn]["saving_vs_chosen_wald"] >= 0.15]
    chosen_mode = "wald"
    if qualifying:
        chosen_min = max(qualifying, key=lambda mn: sieg[mn]["saving_vs_chosen_wald"])
        chosen_mode = "siegmund"
    saving = sieg[chosen_min]["saving_vs_chosen_wald"]
    siegmund_ok = sieg[chosen_min]["errors_ok"]
    hetero = [simulate(r, BASE_TAU, 0.0, cap=cap, min_pairs=chosen_min, mode=chosen_mode, **kw) for r in RATES]

    out = {"schema": sprt.SCHEMA, "seed": args.seed, "runs_per_cell": args.runs, "batch_pairs": args.batch,
           "p0": p0, "p1": p1, "alpha": alpha, "beta": beta, "regularize": reg,
           "llr_bounds": list(sprt.bounds(alpha, beta)), "siegmund_rho": sprt.SIEGMUND_RHO, "sentinels": K,
           "heterogeneity": BASE_H, "taus": list(TAUS),
           "cap_rule": "p95 pairs-to-decide at the worst (true rate, tau) cell of the untruncated Wald test, "
                       "rounded up to a whole batch",
           "cap_pairs": cap, "cap_from": {"rate": worst["rate"], "tau": worst["tau"], "p95_pairs": worst["p95_pairs"]},
           "choice_rule": "min_pairs = smallest grid value whose Wald rule holds both error rates <= nominal "
                          "everywhere BY THE MONTE-CARLO MARGIN (upper 95% bound); bounds = Wald unless a "
                          "Siegmund rule holds them by that margin too AND spends <= 85% of the Wald rule's pairs",
           "chosen": {"min_pairs": chosen_min, "bounds_mode": chosen_mode,
                      "siegmund_saving_vs_wald": round(saving, 4), "siegmund_errors_ok": siegmund_ok,
                      "siegmund_by_min": {str(k): v for k, v in sieg.items()},
                      "wald_errors_ok_by_min": {str(mn): _errors_ok(phase2[f"wald/min{mn}"], alpha, beta)
                                                for mn in mins}},
           "phase1_untruncated_wald": phase1, "phase2_truncated": phase2,
           "phase3_chosen_homogeneous_pool": hetero, "seconds": round(time.time() - t0, 1)}
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(args.out, "results.md"), "w") as f:
        f.write(render(out))
    print(render(out))
    return 0


def _table(rows, title):
    s = [f"### {title}", "",
         "| tau | true rate | P(promote) | P(reject) | of which by cap | mean pairs | median | p95 |",
         "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        s.append(f"| {r['tau']:g} | {r['rate']:.3f} | {r['accept_rate']:.4f} | {r['reject_rate']:.4f} | "
                 f"{r['reject_by_cap_rate']:.4f} | {r['mean_pairs']:.0f} | {r['median_pairs']:.0f} | "
                 f"{r['p95_pairs']:.0f} |")
    return "\n".join(s) + "\n"


def _compare(out) -> str:
    s = ["### Wald vs Siegmund-corrected bounds, by minimum pair count (with the cap; every tau pooled)", "",
         "| rule | max P(promote) at rate <= 0.50 | max P(reject) at rate >= 0.55 | mean pairs (all cells) | "
         "mean pairs at 0.525 | worst p95 |", "|---|---|---|---|---|---|"]
    for key, rows in out["phase2_truncated"].items():
        fp = max(r["accept_rate"] for r in rows if r["rate"] <= 0.50)
        fr = max(r["reject_rate"] for r in rows if r["rate"] >= 0.55)
        mid = [r["mean_pairs"] for r in rows if r["rate"] == 0.525]
        s.append(f"| {key} | {fp:.4f} | {fr:.4f} | {np.mean([r['mean_pairs'] for r in rows]):.0f} | "
                 f"{np.mean(mid):.0f} | {max(r['p95_pairs'] for r in rows):.0f} |")
    return "\n".join(s) + "\n"


def render(out: dict) -> str:
    ch = out["chosen"]
    key = f"{ch['bounds_mode']}/min{ch['min_pairs']}"
    head = (f"# SPRT promotion — operating characteristics\n\n"
            f"`{out['schema']}` · H0 {out['p0']} vs H1 {out['p1']} (mean per-game SCORE, a draw = 1/2) · "
            f"alpha = beta = {out['alpha']} · Wald LLR bounds [{out['llr_bounds'][0]:.3f}, {out['llr_bounds'][1]:.3f}] · "
            f"pentanomial GSPRT over mirrored pairs (Van den Bergh constrained MLE, lambda by bisection) · "
            f"checked every {out['batch_pairs']} pairs · {out['runs_per_cell']:,} runs per cell · seed {out['seed']} · "
            f"{out['sentinels']} sentinels round-robin (logit offsets {out['heterogeneity']} x (-1,-0.5,0,0.5,1)) · "
            f"team-imbalance tau in {out['taus']} logits\n\n"
            f"**Cap: {out['cap_pairs']} pairs** ({2 * out['cap_pairs']} games) — {out['cap_rule']} "
            f"(p95 = {out['cap_from']['p95_pairs']:.0f} at rate {out['cap_from']['rate']}, tau {out['cap_from']['tau']:g}). "
            f"Reaching the cap = REJECT.\n\n"
            f"**Chosen: {key}** — {out['choice_rule']}. Siegmund vs the chosen Wald rule, by minimum: "
            + "; ".join(f"min{k}: errors held by the margin {v['errors_ok']}, pairs saved "
                        f"{100 * v['saving_vs_chosen_wald']:.1f}%" for k, v in ch["siegmund_by_min"].items())
            + ".\n\n"
            f"P(promote) at a true rate <= 0.50 is the FALSE-PROMOTION rate (target <= alpha); P(reject) at a true "
            f"rate >= 0.55 is the FALSE-REJECTION rate (target <= beta). Between them is the indifference zone.\n\n")
    body = head + _compare(out) + "\n" + _table(out["phase2_truncated"][key], f"The shipped rule ({key}, with the cap)")
    body += "\n" + _table(out["phase3_chosen_homogeneous_pool"], "The shipped rule on a HOMOGENEOUS pool (heterogeneity 0)")
    for k, rows in out["phase2_truncated"].items():
        if k != key:
            body += "\n" + _table(rows, f"{k} (with the cap)")
    body += "\n" + _table(out["phase1_untruncated_wald"], "Untruncated Wald, first check after one batch (how the cap was chosen)")
    return body


if __name__ == "__main__":
    raise SystemExit(main())
