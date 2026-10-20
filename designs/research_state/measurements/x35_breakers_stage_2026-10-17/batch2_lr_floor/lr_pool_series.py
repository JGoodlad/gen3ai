"""The learning rate in force and the self-play pool's staleness, from one run's TensorBoard scalars (read-only).

    python lr_pool_series.py <tb_scalars.json.gz> <reads in M, comma-separated> <label> > out.md

The input is the archive ``tb_extract.py`` writes (``../../x26_collapse_battery_2026-10-20/tb_extract.py``: every scalar
tag, a restart's duplicate step resolved to its last write). Printed, as markdown:

1. every change of ``train/learning_rate`` (the adaptive-KL controller's rungs, 3e-4 / 1.2^k);
2. per read r: the trailing-10M mean of the learning rate, approx KL, gradient norm, clip fraction and
   ``eval/win_rate_vs_pool``; the promotions in the last 50M; and the pool's median member age and oldest member.
   The pool at step t is the 20 latest promotions <= t (FIFO, ``--pfsp-scale 0``), the battery's B-pool rule.

A read inside a (r - 10M, r + 0.5M] window with no row prints NA, never 0.
"""
from __future__ import annotations

import gzip
import json
import statistics
import sys

POOL_SIZE = 20
TAGS = ("train/learning_rate", "train/approx_kl", "train/grad_norm", "train/clip_fraction", "eval/win_rate_vs_pool")


def trailing_mean(rows: list[list[float]], r_m: int, width_m: int = 10) -> float | None:
    vals = [v for s, v, _w in rows if (r_m - width_m) * 1e6 < s <= r_m * 1e6 + 0.5e6]
    return statistics.mean(vals) if vals else None


def lr_changes(rows: list[list[float]]) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for s, v, _w in rows:
        if not out or abs(v - out[-1][1]) / out[-1][1] > 1e-3:
            out.append((s, v))
    return out


def main(src: str, reads_arg: str, label: str) -> int:
    with gzip.open(src, "rt") as fh:
        tags = json.load(fh)["tags"]
    reads = [int(x) for x in reads_arg.split(",")]
    print(f"## {label}\n\n**Learning-rate changes** (step: rate): " + "; ".join(
        f"{s / 1e6:.2f}M: {v:.4e}" for s, v in lr_changes(tags["train/learning_rate"])) + "\n")
    prom = sorted(v for _s, v, _w in tags["train/selfplay_promoted_steps"])
    print("| read | learning rate | approx KL | grad norm | clip fraction | eval win rate vs pool "
          "| promotions in the last 50M | pool median age (M) | oldest member (M) |")
    print("|---|---|---|---|---|---|---|---|---|")
    for r in reads:
        cells = [f"{r}M"]
        for tag in TAGS:
            m = trailing_mean(tags[tag], r)
            cells.append("NA" if m is None else f"{m:.4g}")
        t = r * 1e6 + 0.5e6
        upto = [p for p in prom if p <= t]
        pool = upto[-POOL_SIZE:]
        cells.append(str(sum(1 for p in upto if p > t - 50e6)))
        cells.append(f"{statistics.median((t - p) / 1e6 for p in pool):.0f}" if pool else "NA")
        cells.append(f"{min(pool) / 1e6:.0f}" if pool else "NA")
        print("| " + " | ".join(cells) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
