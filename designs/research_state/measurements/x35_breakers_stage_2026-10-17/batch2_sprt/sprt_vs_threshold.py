"""The SPRT arm's MECHANISM prediction (REPLAN_2026-10-20 §2): per-candidate promotion probability under T6's SPRT rule
vs today's threshold rule, at a true per-game edge over the pool of 0.51-0.54. CPU only, no battles, < 1 min.

* SPRT: ``sprt_promotion/simulate.py``'s ``simulate()`` with the SHIPPED rule (``results.json``'s chosen rule: Wald
  bounds, min 40 pairs, batch 40, cap 1,680 pairs = REJECT), its base game model (tau 1, sentinel heterogeneity 0.4).
* Today's rule: promote iff the eval cycle's win rate over ~500 games exceeds 0.55 -- BINOMIAL (X26's measured
  promotions exceed it, REPLAN F-RP-4), so the SPRT / threshold ratio is an upper bound.

    python designs/research_state/measurements/x35_breakers_stage_2026-10-17/batch2_sprt/sprt_vs_threshold.py
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "..", "..", "sprt_promotion")
sys.path.insert(0, SIM)
import simulate as S  # noqa: E402

EDGES = (0.51, 0.52, 0.53, 0.54)
RUNS, SEED, TAU = 5000, 20261020, 1.0
GAMES, BAR = 500, 0.55


def threshold_rate(p: float, n: int = GAMES, bar: float = BAR, strict: bool = True) -> float:
    """P(wins / n > bar) (``strict``, the code's ``win_rate_vs_pool > promote_threshold``) or P(wins / n >= bar),
    Binomial(n, p): the exact tail."""
    k0 = math.floor(bar * n) + 1 if strict else math.ceil(bar * n)
    return math.fsum(math.comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(k0, n + 1))


def main() -> int:
    res = json.load(open(os.path.join(SIM, "results.json")))
    ch = res["chosen"]
    sprt = S._sprt_mod()
    rng = np.random.default_rng(SEED)
    rows = []
    for e in EDGES:
        r = S.simulate(e, TAU, res["heterogeneity"], runs=RUNS, batch=res["batch_pairs"], cap=res["cap_pairs"],
                       p0=sprt.P0, p1=sprt.P1, alpha=sprt.ALPHA, beta=sprt.BETA, reg=sprt.REGULARIZE, rng=rng,
                       min_pairs=ch["min_pairs"], mode=ch["bounds_mode"])
        rows.append({"edge": e, "sprt_accept": round(r["accept_rate"], 4), "sprt_mean_pairs": round(r["mean_pairs"], 1),
                     "sprt_by_cap": round(r["reject_by_cap_rate"], 4),
                     "threshold_promote_binomial": round(threshold_rate(e), 4),
                     "threshold_promote_binomial_ge": round(threshold_rate(e, strict=False), 4)})
    out = {"runs": RUNS, "seed": SEED, "tau": TAU, "heterogeneity": res["heterogeneity"], "cap_pairs": res["cap_pairs"],
           "min_pairs": ch["min_pairs"], "bounds": ch["bounds_mode"], "threshold_rule": f"Binomial({GAMES}) > {BAR} (the code's strict >; the _ge column is >= {BAR})",
           "rows": rows}
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
