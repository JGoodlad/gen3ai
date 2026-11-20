"""The fresh-cell frame's sigma amendment and power record, from BOUNDARY-1's banked 50M look (Q6-N and Q5-N PREREG §5).

FRAME.md §4 left the seed SD (sigma) UNVERIFIED and asked the first card registered after B1's 50M look to record the
banked cell's spread at 50M. This script does that for cell C and recomputes the frame's two tables with it. It adds
NO statistic: the rule is ``main.b1.cross.simple_effect`` + ``classify`` (the ``b1_fresh_cell`` rule), and the
simulation is B1's own ``power_matrix.py`` (``simulate`` takes one seed SD a run, as its part (9) uses it;
``simple_np`` / ``classes_100`` are its numpy copy of the rule; ``check_simple`` runs the first 1,000 draws of every
row through ``main.b1.cross`` itself, and a mismatch exits 1).

(A) C's own spread at 50M on view N: in each of B1's three simple effects against C (W, Br, WB), the variance of C's
    three run means (``s2_raw``; ``record/look_50M/verdict.json``, commit eff47b0df). Each run mean averages 3 cells, so
    it carries cell noise 0.68**2 / 3; sigma_C**2 = the mean s2_raw minus that. Three runs are 2 df: the chi-square
    bounds are printed with it. The three s2_raw share the same three C runs, so they are ONE 2-df estimate, not three.
(B) the 50M rule (CONTINUE iff the estimate > +0.005 pp), closed form, one seed SD a cell.
(C) the 100M classes, simulated, one seed SD a cell; the banked cell is drawn like the fresh one (the power is over
    both cells' seeds: the three banked runs are in fact already trained, and their own luck is unknown).

Deterministic (numpy default_rng(20261120)); CPU only, under a minute:
    PYTHONPATH=<checkout>/src CUDA_VISIBLE_DEVICES="" python power_c50m.py > power_c50m.log
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np
from scipy import stats

from utils.paths import repo_path

B1 = repo_path("designs", "research_state", "measurements", "boundary1_scratch_2026-11-09")
sys.path.insert(0, str(B1))

import power_matrix as P  # noqa: E402  (B1's committed helpers; the path is B1's folder)
from main.b1 import cross as X  # noqa: E402

SEED = 20261120
N = 3
DELTAS_50 = (-4.0, -2.0, 0.0, 2.0, 4.0, 6.0)
DELTAS_100 = (-4.0, 0.0, 2.0, 4.0, 6.0, 8.0)


def c_spread() -> tuple:
    """(sigma_C point, its 95 % upper bound, the mean raw variance) on view N at 50M, printed as it is read."""
    doc = json.loads((B1 / "record" / "look_50M" / "verdict.json").read_text())
    cell = P.cell_sd("N")
    noise = cell ** 2 / N
    print(f"(A) cell C at 50M, view N: B1's look of {doc['look']} at pin {doc['pin'][:9]}; cell noise SD {cell:.2f} pp, "
          f"so a run mean over {N} cells carries variance {noise:.3f}; floor {X.FLOOR_PP} pp (variance "
          f"{X.FLOOR_PP ** 2:.4f})")
    raws = []
    for other in ("W", "Br", "WB"):
        for view in ("N", "forum72"):
            st = doc["simple_effects"][f"{other}_vs_C"][view]
            g = {x["arm"]: x for x in st["groups"]}
            lo, hi = st["interval_pp"]
            print(f"  {other} vs C on {view}: {st['estimate_pp']:+.2f} [{lo:+.2f}, {hi:+.2f}] | C run means "
                  + " / ".join(f"{m:.2f}" for m in g["C"]["means_pp"]) + f", s2_raw {g['C']['s2_raw']:.4f} "
                  f"(floored: {g['C']['floored']}) | {other} s2_raw {g[other]['s2_raw']:.4f} (floored: {g[other]['floored']})")
            if view == "N":
                raws.append(g["C"]["s2_raw"])
    raw = sum(raws) / len(raws)
    df = N - 1
    up = raw * df / stats.chi2.ppf(0.05, df)
    down = raw * df / stats.chi2.ppf(0.95, df)
    point = math.sqrt(max(raw - noise, 0.0))
    upper = math.sqrt(max(up - noise, 0.0))
    print(f"  C on N: mean s2_raw {raw:.4f} (SD {math.sqrt(raw):.2f} pp) on {df} df; minus the cell noise: sigma_C = "
          f"{point:.2f} pp; chi-square 5 % / 95 % bounds on the raw variance {down:.3f} / {up:.3f}, so sigma_C lies "
          f"in [{math.sqrt(max(down - noise, 0.0)):.2f}, {upper:.2f}] pp (the lower bound is under the cell noise: 0)")
    print(f"  every C variance above is under the floor: the BANKED side of a fresh cell's interval is the floor, "
          f"{X.FLOOR_PP ** 2:.4f} / {N}. The half-width is then at least "
          f"{stats.t.ppf(0.5 + X.LEVEL / 2.0, 2 * (N - 1)) * math.sqrt(2 * X.FLOOR_PP ** 2 / N):.2f} pp whatever the "
          f"fresh cell's spread")
    return point, upper


def sd_of_estimate(s_fresh: float, s_banked: float) -> float:
    return math.sqrt(s_fresh ** 2 / N + s_banked ** 2 / N + P.cell_sd("N") ** 2 / N ** 2)


def main() -> int:
    rng = np.random.default_rng(SEED)
    point, upper = c_spread()
    # (seed SD of the fresh cell, of the banked cell): C's measured point on the banked side against four fresh
    # spreads; C's 95 % upper bound; then the frame's three columns (one SD both sides), kept as registered.
    rows = [(point, point), (1.76, point), (2.5, point), (3.1, point), (1.76, upper), (1.76, 1.76), (2.5, 2.5), (3.1, 3.1)]
    print("\n(B) THE 50M RULE: P(CONTINUE) = P(estimate > +0.005 pp), closed form, at a true 50M effect (pp) of "
          + ", ".join(f"{d:+.0f}" for d in DELTAS_50))
    for sf, sb in rows:
        sd = sd_of_estimate(sf, sb)
        print(f"  seed SD fresh {sf:.2f} / banked {sb:.2f} | SD of the estimate {sd:.2f} | "
              + " | ".join(f"{stats.norm.cdf((d - X.EPS_PP) / sd):.3f}" for d in DELTAS_50))
    print(f"\n(C) THE 100M CLASSES on N, {N} runs against {N}: P(WIN) / P(LOSS) / P(NO-USEFUL-GAIN) / P(UNDECIDED) at a "
          f"true 100M effect (pp); {P.REPS} draws a row, the first {P.CHECK} through main.b1.cross")
    for sf, sb in rows:
        sig = np.repeat(np.array([sb, sf, 0.0, 0.0]), N)   # arm order C, W, Br, WB: C is the banked cell, W the fresh
        out = []
        halves = []
        for d in DELTAS_100:
            f = P.simulate(rng, N, sig, P.cell_sd("N"), {"C": 0.0, "W": d, "Br": 0.0, "WB": 0.0})
            dd, half = P.simple_np(f, N, "W", "C")
            c = P.classes_100(dd, half)
            P.check_simple(f, N, "W", "C", c)
            halves.append(float(np.median(half)))
            out.append(f"{d:+.0f}: " + " / ".join(f"{P.frac(c, k):.3f}" for k in (X.WIN, X.LOSS, X.NO_USEFUL_GAIN, X.UNDECIDED)))
        print(f"  seed SD fresh {sf:.2f} / banked {sb:.2f} | median half-width {halves[2]:.2f} | " + " | ".join(out))
    print(f"\nRULE CODE vs B1'S NUMPY FORMULA: {P.MISMATCH[0]} mismatches in {P.MISMATCH[1]} classes")
    return 1 if P.MISMATCH[0] else 0


if __name__ == "__main__":
    sys.exit(main())
