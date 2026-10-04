"""The COMPILED UPDATE-TIME A/B across commits from the profile runs' own phase markers (2026-10-03).

    python update_ab.py --arm NAME PHASES.jsonl CPU.jsonl [CAPTURE.log] ... --out update_ab.json

Per arm: every `update` phase (one PPO update, 98,304 rows x 10 epochs) EXCEPT the startup dry update and
the first real one (warm-up), the update that runs the compile canary (the 11th `train` call: learn
update 10), any update overlapping an nsys capture window, a torch-profiled update (> 100 s), and any
update whose window read CONTENDED (the windowed cpu_meter factor >= 1.05 on any sampler row inside it —
standing rule 8: a contaminated window is excluded, not tolerated). The update is GPU-bound (K8: six CPU
bystanders moved it +0.04 %), so the exclusion is belt and braces. Reports n, median, the bootstrap 95 %
CI of the median, and each arm's median minus the first arm's with the bootstrap CI of that difference.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics as st


def updates(phases, cpu, cap):
    rows = [json.loads(line) for line in open(phases)]
    u = [r for r in rows if r.get("phase") == "update"]
    caps = []
    if cap:
        t = [float(x) for x in re.findall(r"(?:start|stop) at ([0-9.]+)", open(cap).read())]
        caps = [(t[i], t[i + 1]) for i in range(0, len(t) - 1, 2)]
    samp = [json.loads(line) for line in open(cpu)]
    samp = [s for s in samp if "t" in s]
    keep, why = [], []
    for i, r in enumerate(u):
        d = r["t1"] - r["t0"]
        if i < 2:
            why.append((i, round(d, 2), "warm-up"))
            continue
        if i == 10:
            why.append((i, round(d, 2), "canary update"))
            continue
        if d > 100:
            why.append((i, round(d, 2), "torch-profiled"))
            continue
        if any(not (r["t1"] < a or r["t0"] > b) for a, b in caps):
            why.append((i, round(d, 2), "nsys capture"))
            continue
        f = [s["factor"] for s in samp if r["t0"] <= s["t"] - s["dt"] and s["t"] <= r["t1"]]
        if not f or max(f) >= 1.05:
            why.append((i, round(d, 2), f"contended (max factor {max(f) if f else 'n/a'})"))
            continue
        keep.append(d)
    return keep, why


def boot_median(x, n=10000, seed=0):
    rng = random.Random(seed)
    m = sorted(st.median(rng.choices(x, k=len(x))) for _ in range(n))
    return m[int(0.025 * n)], m[int(0.975 * n)]


def boot_diff(a, b, n=10000, seed=1):
    rng = random.Random(seed)
    d = sorted(st.median(rng.choices(b, k=len(b))) - st.median(rng.choices(a, k=len(a))) for _ in range(n))
    return d[int(0.025 * n)], d[int(0.975 * n)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", nargs="+", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = {}
    base = None
    for arm in a.arm:
        name, ph, cpu = arm[:3]
        cap = arm[3] if len(arm) > 3 else None
        k, why = updates(ph, cpu, cap)
        lo, hi = boot_median(k)
        row = {"n": len(k), "median_s": st.median(k), "ci95_median": [lo, hi], "kept_s": [round(x, 3) for x in k],
               "excluded": why}
        if base is None:
            base = k
        else:
            dl, dh = boot_diff(base, k)
            row["delta_vs_first_s"] = st.median(k) - st.median(base)
            row["delta_ci95"] = [dl, dh]
            row["delta_pct"] = 100 * (st.median(k) - st.median(base)) / st.median(base)
        res[name] = row
        print(name, {kk: v for kk, v in row.items() if kk not in ("kept_s", "excluded")})
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
