"""X55 / LG-0, part 1: the league HEADROOM from EXISTING games only (no game is played here).

    PYTHONPATH=<checkout>/src python headroom_existing.py [--run-dir models/rb_x26_s2601] [--draws 500] > headroom_existing.json

Reads (read-only) X26's in-run snapshot ladder (`snapshot_ladder/games.jsonl`) and its in-run eval rows
(`eval_results.jsonl`, the trainee vs 5 pool sentinels per 2M cycle, 100 games each, greedy, symmetric teams), and
X35's frozen reference population file. Prints one JSON document:

  (a) THE NASH over X26's archive as of P* (the ladder's nodes <= 270M): the point LP Nash the cycle monitor uses
      (`agents.training.cycle_monitor.nash_lp`, unplayed edges filled by the Bradley-Terry spine), the MAX-ENTROPY Nash
      of the point matrix (SLSQP over the Nash polytope, started at the LP point), and the POSTERIOR (Dirichlet(c + 1/2)
      draws per edge, an LP Nash per draw): per node P(in support) and mean weight. Also for the last-80M window.
  (b) OVERLAP: the Nash mass held by C0's seeded pool (the 14 snapshots 164-268M), by ladder nodes within +-5M of
      A1's 8 archive nodes, and by ladder nodes within +-5M of R's A-block.
  (c) THE P*-NEIGHBOURHOOD ROW from the in-run eval: the trainee at 270-290M (P* +- 10M) against each pool sentinel,
      wins pooled per sentinel step (a proxy for P*'s row; the registered row is part 2's `play_pstar_row.sh`).
  (d) WHAT PFSP COULD REWEIGHT: A2's factor recency x (1 + 2.5 (1 - p)) over C0's pool, and stable PFSP's
      max(0.05, 1 - p) over A1's archive, from (c)'s rates; each node's share relative to uniform.
  (e) THE SPINE'S extrapolated score of the 268M node against each ladder node (to be checked against part 2's
      measured row: a steep spine means the ladder Nash cannot see old nodes).

Deterministic: every random draw is seeded from a fixed string.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

from agents.training import cycle_monitor as CM
from main.cycle_monitor import ladder_edges

PSTAR = 280_000_170
C0_POOL = [164000130, 166000198, 176000224, 194000195, 200000120, 220000027, 224000053, 228000030, 236000155,
           244000047, 246000132, 252000090, 254000018, 268000171]          # x35_pool: 14 snapshots 164-268M
A1 = [30000045, 60000135, 100000037, 130000065, 160000085, 190000228, 230000074, 260000235]
A_BLOCK = [280000170, 270000128, 80000111, 140000062, 210000149]
NEAR = 5_000_000


def seed(tag: str) -> int:
    return int.from_bytes(hashlib.sha256(tag.encode()).digest()[:8], "big")


def maxent_nash(a: np.ndarray, x0: np.ndarray, tol: float = 1e-7):
    """The max-entropy point of the Nash polytope of the symmetric zero-sum game ``a`` (value 0):
    max H(x) s.t. a^T x >= -tol, sum x = 1, x >= 0."""
    k = a.shape[0]
    eps = 1e-12

    def f(x):
        x = np.clip(x, eps, None)
        return float((x * np.log(x)).sum())

    def g(x):
        return np.log(np.clip(x, eps, None)) + 1.0

    cons = [{"type": "eq", "fun": lambda x: x.sum() - 1.0, "jac": lambda x: np.ones(k)},
            {"type": "ineq", "fun": lambda x: a.T @ x + tol, "jac": lambda x: a.T}]
    start = 0.999 * x0 + 0.001 / k
    res = minimize(f, start, jac=g, bounds=[(0.0, 1.0)] * k, constraints=cons, method="SLSQP",
                   options={"maxiter": 500, "ftol": 1e-12})
    x = np.clip(res.x, 0.0, None)
    x = x / x.sum()
    if not res.success or (a.T @ x).min() < -1e-4:      # failed / infeasible drift: fall back to the LP point
        return x0, False
    return x, True


def nash_block(mx: CM.Matrix, draws: int, tag: str) -> dict:
    k = mx.n
    n, m, var_m, y, w = CM.edge_stats(mx.counts)
    r = CM.wls_fit(k, mx.ei, mx.ej, y, w)
    a = CM.payoff(k, mx.ei, mx.ej, m, r)
    x_lp = CM.nash_lp(a)
    x_me, me_ok = maxent_nash(a, x_lp)
    rng = np.random.Generator(np.random.PCG64(seed("x55:posterior:" + tag)))
    post = np.stack([rng.dirichlet(mx.counts[e] + 0.5, size=draws) for e in range(len(n))], axis=1)
    pm = (post * CM.SCORES).sum(-1)
    xs = np.stack([CM.nash_lp(CM.payoff(k, mx.ei, mx.ej, pm[i], r)) for i in range(draws)])
    p_in = (xs > 0.01).mean(0)
    w_mean = xs.mean(0)
    supp = (xs > 0.01).sum(1)

    def ent(x):
        x = x[x > 0]
        return float(-(x * np.log(x)).sum())

    nodes = [{"step": mx.steps[i], "lp_w": round(float(x_lp[i]), 4), "maxent_w": round(float(x_me[i]), 4),
              "post_p_in_support": round(float(p_in[i]), 3), "post_mean_w": round(float(w_mean[i]), 4),
              "spine_elo_vs_newest": round(float(r[i] - r[-1]) * CM.ELO_PER_LOGIT, 1)}
             for i in range(k)]
    return {"n_nodes": k, "n_edges": int(len(n)), "pairs": int(n.sum()),
            "lp_support": int((x_lp > 0.01).sum()), "maxent_support": int((x_me > 0.01).sum()),
            "lp_entropy": round(ent(x_lp), 4), "maxent_entropy": round(ent(x_me), 4),
            "maxent_minus_lp_l1": round(float(np.abs(x_me - x_lp).sum()), 4), "maxent_solver_ok": me_ok,
            "post_support_mean": round(float(supp.mean()), 2),
            "post_support_q05_q95": [int(np.quantile(supp, 0.05)), int(np.quantile(supp, 0.95))],
            "post_weight_effective_n": round(float(1.0 / (w_mean ** 2).sum()), 2),
            "nodes": nodes, "_x": w_mean, "_steps": list(mx.steps), "_spine": r}


def mass_near(steps, wts, targets, near=NEAR):
    hit = set()
    for t in targets:
        for i, s in enumerate(steps):
            if abs(s - t) <= near:
                hit.add(i)
    return round(float(sum(wts[i] for i in hit)), 4), sorted(steps[i] for i in hit)


def sentinel_row(run_dir: Path, lo: int, hi: int) -> dict:
    agg: dict = {}
    trainees = []
    for line in (run_dir / "eval_results.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if not (lo <= int(r["step"]) <= hi):
            continue
        trainees.append(int(r["step"]))
        for s in r.get("sentinels", []):
            w, g = s["counts"]
            a = agg.setdefault(int(s["step"]), [0, 0, 0])
            a[0] += int(w)
            a[1] += int(g)
            a[2] += 1
    out = []
    for st, (w, g, c) in sorted(agg.items()):
        p = w / g
        out.append({"sentinel_step": st, "wins": w, "games": g, "cycles": c, "p": round(p, 4),
                    "se_binom": round(math.sqrt(p * (1 - p) / g), 4)})
    return {"trainee_steps": [min(trainees), max(trainees)] if trainees else None, "n_cycles": len(trainees),
            "rows": out}


def nearest_rate(row, step):
    best = min(row, key=lambda r: abs(r["sentinel_step"] - step))
    return best["p"], best["sentinel_step"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="/home/goodlad/dev/gen3ai/models/rb_x26_s2601")
    ap.add_argument("--draws", type=int, default=500)
    a = ap.parse_args()
    run = Path(a.run_dir)
    edges, meta = ladder_edges(run)
    pre = [e for e in edges if e[1] <= 270_000_000 and e[3] <= 270_000_000]
    mx = CM.build_matrix(pre)
    out = {"source": str(run), "ladder_meta": meta, "edges_pre_pstar": len(pre)}

    full = nash_block(mx, a.draws, "full")
    win = CM.window(mx, 188_000_000)
    late = nash_block(win, a.draws, "late80")
    for name, blk in (("archive_to_268M", full), ("window_188_268M", late)):
        st, wm = blk["_steps"], blk["_x"]
        blk["mass_c0_pool"] = round(float(sum(wm[i] for i, s in enumerate(st) if s in C0_POOL)), 4)
        blk["mass_near_a1"], blk["near_a1_nodes"] = mass_near(st, wm, A1)
        blk["mass_near_a1_early4"], _ = mass_near(st, wm, A1[:4])
        blk["mass_near_a1_late4"], _ = mass_near(st, wm, A1[4:])
        blk["mass_near_a_block"], blk["near_a_block_nodes"] = mass_near(st, wm, A_BLOCK)
        blk["mass_ge_130M"] = round(float(sum(wm[i] for i, s in enumerate(st) if s >= 130_000_000)), 4)
        blk["mass_lt_130M"] = round(float(sum(wm[i] for i, s in enumerate(st) if s < 130_000_000)), 4)
        # (e) the spine's extrapolated score of the newest node (268M) against each node
        r = blk["_spine"]
        blk["spine_score_newest_vs"] = {str(s): round(float(1 / (1 + math.exp(-(r[-1] - r[i])))), 3)
                                       for i, s in enumerate(st)}
        for k in ("_x", "_steps", "_spine"):
            blk.pop(k)
    out["nash"] = {"archive_to_268M": full, "window_188_268M": late}

    near = sentinel_row(run, 270_000_000, 290_000_216)
    wide = sentinel_row(run, 240_000_000, 300_000_189)
    out["pstar_neighbourhood_row"] = {"p_star_pm_10M": near, "trainee_240_300M": wide}

    # (d) PFSP reweighting, from the 270-290M row (nearest sentinel step)
    rows = near["rows"]
    ps = {s: nearest_rate(rows, s) for s in C0_POOL}
    rec = {s: 1.0 + 0.3 * (s - C0_POOL[0]) / (C0_POOL[-1] - C0_POOL[0]) for s in C0_POOL}
    wa2 = {s: rec[s] * (1.0 + 2.5 * (1.0 - ps[s][0])) for s in C0_POOL}
    tot_c0, tot_a2 = sum(rec.values()), sum(wa2.values())
    out["a2_pfsp_over_c0_pool"] = {
        "formula": "recency (1 + 0.3 (step - oldest)/span) x (1 + 2.5 (1 - p)); p = nearest sentinel's pooled rate",
        "nodes": [{"step": s, "p_used": ps[s][0], "p_from_sentinel": ps[s][1],
                   "share_c0": round(rec[s] / tot_c0, 4), "share_a2": round(wa2[s] / tot_a2, 4),
                   "ratio": round((wa2[s] / tot_a2) / (rec[s] / tot_c0), 3)} for s in C0_POOL]}
    rr = [n["ratio"] for n in out["a2_pfsp_over_c0_pool"]["nodes"]]
    out["a2_pfsp_over_c0_pool"]["ratio_range"] = [min(rr), max(rr)]
    out["a2_pfsp_over_c0_pool"]["tv_distance"] = round(0.5 * sum(
        abs(n["share_a2"] - n["share_c0"]) for n in out["a2_pfsp_over_c0_pool"]["nodes"]), 4)

    pa = {s: nearest_rate(wide["rows"], s) for s in A1}
    wst = {s: max(0.05, 1.0 - pa[s][0]) for s in A1}
    tw = sum(wst.values())
    out["stable_pfsp_over_a1_archive_PROVISIONAL"] = {
        "formula": "max(0.05, 1 - p) over the un-mastered (p < 0.80) nodes; p = nearest sentinel's rate, trainee "
                   "240-300M (PROVISIONAL: part 2's measured P* row replaces it)",
        "nodes": [{"step": s, "p_used": pa[s][0], "p_from_sentinel": pa[s][1], "mastered": pa[s][0] >= 0.8,
                   "share": round(wst[s] / tw, 4), "ratio_vs_uniform": round(8 * wst[s] / tw, 3)} for s in A1]}
    json.dump(out, sys.stdout, indent=1, default=float)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
