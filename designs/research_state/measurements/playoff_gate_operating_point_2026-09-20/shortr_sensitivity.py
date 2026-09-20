"""THE SHORT-R SENSITIVITY — does dropping the guard-refused shards move the cell?

    python3 shortr_sensitivity.py <battery dir> --out shortr.json

WHY. `playoff.short_r_refusal` is a ROW-level check: it raises on the FIRST game whose realized
mean R falls below 0.9 x the requested R, which kills that shard for good. Two of this battery's
eight playoff shards were refused that way even though their POOLED realized R was 3.96 and 3.98 —
a single slow game is enough. The rows those shards had already written are real rows at a realized
R of ~4, but they come from cells the production guard declared invalid, so the cell is read BOTH
ways and both numbers are published:

  * ALL   — every pfk05 row, which is what `report_battery.py` reads;
  * CLEAN — only the shards that never tripped the guard.

If the two disagree materially the refusal is load-bearing and the CLEAN number is the one to
quote. If they agree, the refusal cost pairs and bought nothing, which is a finding about the
guard's granularity rather than about the arm.

Also reported: the per-ROW realized-R distribution, which is what the row-level rule actually sees.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict


def rows_by_shard(d):
    out = {}
    for p in sorted(glob.glob(os.path.join(d, "pfk05__*.jsonl"))):
        out[os.path.basename(p)] = [json.loads(x) for x in open(p) if x.strip()]
    return out


def refused(log_dir, name):
    log = os.path.join(log_dir, name.replace(".jsonl", ".log"))
    if not os.path.exists(log):
        return False
    return "REFUSED: --playoff-rollouts" in open(log, errors="replace").read()


def l2(rows):
    by = defaultdict(dict)
    for r in rows:
        if not int(r.get("finished", 0)):
            continue
        by[int(r["game"])][int(r.get("orientation", 0) or 0)] = (
            0.5 if int(r.get("tied", 0) or 0) else float(int(r.get("won", 0))))
    xs = [sum(o.values()) / 2.0 for o in by.values() if len(o) == 2]
    n = len(xs)
    if n < 2:
        return {"n": n, "mean": (sum(xs) / n if n else None), "ci": None, "sd": None}
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    h = 1.96 * sd / math.sqrt(n)
    return {"n": n, "mean": m, "ci": [m - h, m + h], "sd": sd}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    shards = rows_by_shard(args.data_dir)
    log_dir = os.path.join(args.data_dir, "logs")
    flag = {k: refused(log_dir, k) for k in shards}

    allrows = [r for rs in shards.values() for r in rs]
    clean = [r for k, rs in shards.items() if not flag[k] for r in rs]
    per_row_R = sorted(float(r.get("playoff_r_total", 0) or 0)
                       / max(1, int(r.get("n_playoff_ran", 0) or 0))
                       for r in allrows if int(r.get("n_playoff_ran", 0) or 0))
    out = {
        "shards": {k: {"rows": len(v), "guard_refused": flag[k],
                       "pooled_realized_R": (sum(r.get("playoff_r_total", 0) for r in v)
                                             / max(1, sum(r.get("n_playoff_ran", 0) for r in v)))}
                   for k, v in sorted(shards.items())},
        "n_shards_refused": sum(flag.values()),
        "L2_all": l2(allrows), "L2_clean": l2(clean),
        "per_row_realized_R": {
            "n": len(per_row_R),
            "min": (per_row_R[0] if per_row_R else None),
            "median": (per_row_R[len(per_row_R) // 2] if per_row_R else None),
            "frac_below_3_6": (sum(1 for x in per_row_R if x < 3.6) / len(per_row_R)
                               if per_row_R else None),
        },
    }
    print(json.dumps(out, indent=1))
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
