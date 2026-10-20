"""Operating characteristics of the windowed cyclic-size estimate (design_evaluation.md §8.2 "The size bar"):
synthetic 14-check climb (Elo 300 ln(1+k)), the registered hybrid plan at 500 mirrored pairs per edge, an independent
cyclic logit perturbation of SD tau pp per edge; the size on the last 8 checks' nodes; 2,000 bootstrap replicates.
Usage: PYTHONPATH=src python size_bar_oc.py <runs per cell>"""
import math, sys
import numpy as np
from agents.training import cycle_monitor as CM
D = 10_000_000
def pent(p, n, rng):
    """n mirrored pairs of two independent games at win probability p (pentanomial from the newer side)."""
    wins = (rng.random((n, 2)) < p).sum(1)
    return [float((wins == 0).sum()), 0.0, float((wins == 1).sum()), 0.0, float((wins == 2).sum())]
def run(tau, n_ck, pairs, seed):
    rng = np.random.default_rng(seed)
    elo = [300 * math.log(1 + k) for k in range(n_ck)]
    steps = [(k + 1) * D for k in range(n_ck)]
    edges = []
    for rp in CM.plan_rows(steps):
        for _l, s in rp.edges:
            i, j = steps.index(rp.newest), steps.index(s)
            x = (elo[i] - elo[j]) * math.log(10) / 400 + rng.normal(0, tau / 100 * 4)
            edges.append((f"n{i}", rp.newest, f"n{j}", s, pent(1 / (1 + math.exp(-x)), pairs, rng)))
    mx = CM.build_matrix(edges)
    g = steps[-1]
    return CM.cyclic_size(mx, g - 8 * D - 500_000 + 1, seed=seed, n_boot=2000)
bar = 1.7
for tau in (0.0, 1.0, 1.7, 2.5, 4.0):
    st = {"MAT": 0, "NEG": 0, "INC": 0}; sz = []
    for s in range(int(sys.argv[1])):
        r = run(tau, 14, 500, 1000 + s)
        sz.append(r["size_pp"])
        st["MAT" if r["lower_pp"] > bar else "NEG" if r["upper_pp"] < bar else "INC"] += 1
    print(f"tau {tau}: mean size {np.mean(sz):.2f} sd {np.std(sz):.2f} nodes {r['nodes']} edges {r['edges']} df {r['df']} phi {r['phi']:.3f} {st}", flush=True)
