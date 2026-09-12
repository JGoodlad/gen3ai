"""L1 (separation-of-raced) at MATCHED SEARCH WIDTH.

Why this file exists. L1 — the mechanism row of the leaf-quality meter registered on 2026-09-11 —
is measured under a per-decision WALL-CLOCK budget, so what it reports depends on how much width
that clock bought. The contemporaneous `ctrl10M` anchors run for this battery make the dependence
impossible to miss: the SAME head reads L1 0.058 / 0.128 / 0.331 at realized K = 3.6 / 5.1 / 7.1
worlds. A bar set on quiet-box cells cannot be applied to a loaded-box cell, and a head cannot be
compared to another head at a different K.

The fix is to compare heads WITHIN bands of realized width. Every row already carries its own
per-battle `realized_mean.k_worlds` and its own `n_defensive_raced` / `n_defensive_separated`, so
L1 can be recomputed over the battles whose width falls in a common band, with no re-running.

    python3 l1_width_matched.py <data_dir> [<data_dir> ...]
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
import sys
from collections import defaultdict

FNAME = re.compile(r"(?P<cell>[a-zA-Z0-9]+)__(?P<head>[a-zA-Z0-9_]+)__s(?P<lo>\d+)\.jsonl$")
BANDS = ((0.0, 3.0), (3.0, 4.5), (4.5, 6.0), (6.0, 8.0), (8.0, 99.0))


def wilson(k: int, n: int, z: float = 1.96):
    if n <= 0:
        return (None, None, None)
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, c - h), min(1.0, c + h))


def main(dirs):
    # (cell, head) -> band -> [raced, separated, battles]
    agg = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    for d in dirs:
        for path in sorted(glob.glob(os.path.join(d, "*.jsonl"))):
            m = FNAME.search(os.path.basename(path))
            if not m or m["cell"] == "grid":
                continue
            key = (m["cell"], m["head"])
            for line in open(path):
                r = json.loads(line)
                k = (r.get("realized_mean") or {}).get("k_worlds")
                raced = int(r.get("n_defensive_raced", 0) or 0)
                if not k or not raced:
                    continue
                for lo, hi in BANDS:
                    if lo <= k < hi:
                        a = agg[key][(lo, hi)]
                        a[0] += raced
                        a[1] += int(r.get("n_defensive_separated", 0) or 0)
                        a[2] += 1
                        break

    print("L1 = separated / raced, WITHIN bands of realized width K (worlds per contested "
          "decision).\nA head is comparable to another head only inside a band.\n")
    head = f"{'cell/head':26}" + "".join(f"{f'K {lo:g}-{hi:g}':>22}" for lo, hi in BANDS)
    print(head)
    for key in sorted(agg):
        row = f"{key[0] + '/' + key[1]:26}"
        for band in BANDS:
            raced, sep, n = agg[key].get(band, [0, 0, 0])
            if raced < 200:
                row += f"{'-':>22}"
            else:
                p, lo_, hi_ = wilson(sep, raced)
                row += f"{f'{p:.3f} [{lo_:.3f},{hi_:.3f}]':>22}"
        print(row)
    print("\n(a band is printed only where the cell raced >= 200 candidate decisions in it; "
          "the Wilson interval is over RACED decisions, which is the denominator L1 is defined on)")


if __name__ == "__main__":
    main(sys.argv[1:] or ["."])
