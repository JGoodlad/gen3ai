"""Seed-to-seed spread of FRESH runs, from the banked head-to-head crosses (boundary1_scratch DRAFT_DESIGN §3).

For every banked cross matrix H (rows = arm A's seeds, columns = arm B's seeds, cell = A's score in pp over that
cell's mirrored pairs) it prints the SD of the row means and of the column means, and the same with the cell noise
removed (two-way residual mean square / the number of cells in the mean). A row mean's SD is the seed-to-seed SD
of "this seed's score against the other arm's seeds", the unit the cross statistic's variance is built from
(`main.h2h.cross.cross_stat`: V = s2_rows / n + s2_cols / m).

CPU only, seconds. From the repo root:
    python designs/research_state/measurements/boundary1_scratch_2026-11-09/seed_spread.py > .../seed_spread.log
"""
import json
import math
from pathlib import Path

import numpy as np

M = Path(__file__).resolve().parent.parent
FILES = [
    "x5ab_look3_2026-10-07/result.json",
    "static_screen_look3_2026-10-09/result.json",
    "endstate_closing_test_2026-10-09/cross/verdict.json",
    "bf16_strength_ab_2026-10-12/stage1_cross/result/verdict.json",
    "bf16_strength_ab_2026-10-12/stage2_cross/result/verdict.json",
]


def matrices(node, out):
    if isinstance(node, dict):
        h = node.get("h_pp")
        if h is not None and not any(np.array_equal(np.array(h, float), m) for m in out):
            out.append(np.array(h, float))
        for v in node.values():
            matrices(v, out)
    elif isinstance(node, list):
        for v in node:
            matrices(v, out)


def main():
    var_true = []
    for f in FILES:
        found = []
        matrices(json.loads((M / f).read_text()), found)
        for h in found:
            n, m = h.shape
            r, c = h.mean(1), h.mean(0)
            res = h - r[:, None] - c[None, :] + h.mean()
            ms = (res ** 2).sum() / ((n - 1) * (m - 1))
            vr, vc = max(r.var(ddof=1) - ms / m, 0.0), max(c.var(ddof=1) - ms / n, 0.0)
            var_true += [(vr, n - 1), (vc, m - 1)]
            print(f"{f}: {n}x{m} mean-50 {h.mean() - 50:+.2f} pp | SD of row means {r.std(ddof=1):.2f}, of column "
                  f"means {c.std(ddof=1):.2f} | cell-noise SD {math.sqrt(ms):.2f} | noise-removed seed SD rows "
                  f"{math.sqrt(vr):.2f}, columns {math.sqrt(vc):.2f}")
    v = np.array([x for x, _ in var_true])
    w = np.array([d for _, d in var_true], float)
    print(f"arm groups {len(v)}: noise-removed seed SD min {math.sqrt(v.min()):.2f}, max {math.sqrt(v.max()):.2f}, "
          f"pooled (df-weighted) {math.sqrt((v * w).sum() / w.sum()):.2f} pp on {int(w.sum())} df")
    print("toy_crosses_2026-10-16/RESULTS.md (3 T-ENT seeds x 8 production finals): per-seed means "
          "R-T5 (46-48M) +5.90/+6.34/+7.21 -> SD "
          f"{np.std([5.90, 6.34, 7.21], ddof=1):.2f} pp; R-T6 (100M cap) +8.10/+11.24/+11.27 -> SD "
          f"{np.std([8.10, 11.24, 11.27], ddof=1):.2f} pp (s4001b carries the canary hazard, RESULTS finding 2)")


if __name__ == "__main__":
    main()
