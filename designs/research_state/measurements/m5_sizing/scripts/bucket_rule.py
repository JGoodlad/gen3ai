#!/usr/bin/env python3
"""REGISTRATION §3: T2 buckets at N from the measured rows-per-opponent-slot histogram.

    buckets = (8, b_opp, b_tr):  b_tr = min(N, 512);  b_opp = the p95 of rows per opponent slot per
    flush rounded up to a multiple of 16 (dropped when <= 8 or >= b_tr);  bound: the TIME padding costs over the
    opponent chunks <= 15 % under T2's replay cost model (a fixed overhead + a per-row slope, fitted to T2's
    per-bucket replay times) and T2's packing (largest bucket first, then the smallest holding the remainder) — a violation adds ONE bucket (the waste-minimising multiple of 16), at most 4 buckets.

    bucket_rule.py provisional N            # the (T-b) first-pass set
    bucket_rule.py rule <split_n<N>.json>   # the rule's set from the histogram + its padding waste
"""
from __future__ import annotations

import json
import math
import sys
from typing import Dict, List, Sequence, Tuple

B_CAP = 512
MEAN_ROWS_PER_SLOT_PER_ENV = 0.95 * 0.88 / 20     # Lane G N = 48: ~40 opponent rows over ~45.6 policy envs


def pow2_ceil(x: float) -> int:
    return 1 << max(0, math.ceil(math.log2(max(1.0, x))))


QUANTUM = 16


def round_up(x: float) -> int:
    """b_opp: the p95 rounded up to a multiple of QUANTUM (T2 takes any bucket size; 48 is today's)."""
    return max(QUANTUM, QUANTUM * math.ceil(float(x) / QUANTUM))


def finish(n: int, b_opp: int) -> List[int]:
    b_tr = min(n, B_CAP)
    out = {8, b_tr}
    if 8 < b_opp < b_tr:
        out.add(b_opp)
    return sorted(out)


def provisional(n: int) -> List[int]:
    q = 1.5 * MEAN_ROWS_PER_SLOT_PER_ENV * n
    return finish(n, round_up(q) if q > 8 else 8)


def pack(rows: int, buckets: Sequence[int]) -> int:
    """Padded rows T2 serves for ``rows`` on one slot: the largest bucket while it fits, then the
    smallest bucket that holds the remainder (T2 DESIGN: packing)."""
    bs = sorted(buckets)
    served, left = 0, int(rows)
    while left > bs[-1]:
        served += bs[-1]
        left -= bs[-1]
    if left > 0:
        served += next(b for b in bs if b >= left)
    return served


def chunks(rows: int, buckets: Sequence[int]) -> List[Tuple[int, int]]:
    """(bucket, real rows) per chunk T2 replays for ``rows`` on one slot (the packing above)."""
    bs = sorted(buckets)
    out, left = [], int(rows)
    while left > bs[-1]:
        out.append((bs[-1], bs[-1]))
        left -= bs[-1]
    if left > 0:
        out.append((next(b for b in bs if b >= left), left))
    return out


#: T2's per-replay cost model (ms) = REPLAY_MS0 + REPLAY_MS_PER_ROW * rows: the least-squares line through
#: T2's per-bucket replay times on 2.5.1 (`static_buckets_graph.jsonl`: B = 2 / 8 / 48 / 128 -> 1.06 / 1.23 /
#: 1.97 / 2.94 ms). Its job is the SHAPE (a fixed overhead plus a per-row slope), not absolute speed.
REPLAY_MS0, REPLAY_MS_PER_ROW = 1.119, 0.01465
WASTE_BOUND = 0.15


def cost(rows: float) -> float:
    return REPLAY_MS0 + REPLAY_MS_PER_ROW * rows


def waste(hist: Dict[int, int], buckets: Sequence[int]) -> float:
    """The TIME share padding costs over the opponent chunks: sum(cost(bucket) - cost(real rows)) /
    sum(cost(bucket)), under the replay cost model (padding a small bucket is nearly free; a large one
    is not)."""
    spent = pad = 0.0
    for k, v in hist.items():
        for b, r in chunks(k, buckets):
            spent += cost(b) * v
            pad += (cost(b) - cost(r)) * v
    return pad / spent if spent else 0.0


def p95(hist: Dict[int, int]) -> int:
    tot = sum(hist.values())
    acc = 0
    for k in sorted(hist):
        acc += hist[k]
        if acc >= 0.95 * tot:
            return k
    return max(hist)


def rule(n: int, hist: Dict[int, int]) -> Tuple[List[int], float, List[str]]:
    notes: List[str] = []
    q = p95(hist)
    bs = finish(n, round_up(q) if q > 8 else 8)        # a p95 the 8 bucket holds needs no b_opp
    w = waste(hist, bs)
    while w > WASTE_BOUND and len(bs) < 4:
        cands = [b for b in range(QUANTUM, bs[-1], QUANTUM) if b not in bs and b > 8]
        if not cands:
            break
        mid = min(cands, key=lambda b: (waste(hist, sorted(set(bs) | {b})), b))
        bs = sorted(set(bs) | {mid})
        notes.append(f"waste {w:.3f} > {WASTE_BOUND}: added {mid} (the waste-minimising multiple of {QUANTUM})")
        w = waste(hist, bs)
    if w > WASTE_BOUND:
        notes.append(f"BOUND VIOLATED at 4 buckets: waste {w:.3f}")
    return bs, w, notes


def main(argv: Sequence[str]) -> int:
    if argv[:1] == ["provisional"]:
        print(",".join(str(b) for b in provisional(int(argv[1]))))
        return 0
    if argv[:1] == ["rule"]:
        d = json.load(open(argv[1]))
        n = int(d["regime"]["n_envs"])
        res = next(iter(d["per_lanes"].values()))
        hist = {int(k): int(v) for k, v in res["rows_per_slot_hist"].items()}
        bs, w, notes = rule(n, hist)
        print(json.dumps({"n": n, "p95_rows_per_slot": p95(hist), "buckets": bs, "opponent_padding_waste": w,
                          "measured_with": res.get("buckets"), "notes": notes}))
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
