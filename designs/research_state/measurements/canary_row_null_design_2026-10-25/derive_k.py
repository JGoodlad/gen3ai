"""The per-row dither-null margin's CALIBRATION TABLE (the design's k, K and floor; README here), from a directory of
`margin_read.py` reads (`read_<label>.json`: CPU from the x26c margin read, or CUDA from gpu_census.sh).

Per checkpoint and fixture row: the arm's |d log pi| from the fp32 reference (Ebf at B = 64; Cbf and Ebf at the canary's
B = 2048 when the read ran `--compiled`) over the row's fp32 dither null (`dither.E32~_per_row_max`, the max over the
read's seeds). Prints, per arm and absolute floor, the healthy ratio's quantiles and maximum with its (checkpoint, row),
the rule's k = R1_BAR_K (4, every R1 bar's safety factor) x that maximum, and how often the old global margin and the
proposed per-row bar would each have refused the healthy read. With per-seed rows (`--rows` covering every row, as
gpu_census.sh asks), also the null's K-sensitivity: the max over the first K seeds against all of them.

    python derive_k.py <dir> [<dir> …] [--floors 0.01,0.02,0.05,0.1] [--json out.json]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np

R1_BAR_K = 4.0
M = 0.72


def rows_of(d: Dict[str, Any]) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """{arm: (shift per row, null per row)} for every arm the read carries."""
    null = np.asarray(d["dither"]["E32~_per_row_max"], dtype=float)
    out = {"Ebf_B64": (np.asarray(d["per_row"]["abs_dlogp_Ebf"], dtype=float), null)}
    c = d.get("compiled")
    if c:
        for arm in ("Cbf_vs_E32", "Ebf_vs_E32"):
            out[f"{arm.split('_')[0]}_B{c['batch']}_{c['device']}"] = (np.asarray(c["per_row"][arm], dtype=float), null)
    return out


def k_sensitivity(d: Dict[str, Any]) -> Dict[str, float]:
    rows = d["dither"].get("rows", {})
    if len(rows) < 8:
        return {}
    per = np.array([rows[str(r)]["E32~"] for r in sorted(map(int, rows))])          # [rows, seeds]
    n = per.shape[1]
    full = per.max(1)
    out = {}
    for k in (2, 4, 8, 16, 32):
        if k < n:
            sub = per[:, :k].max(1)
            ok = full > 0.01
            out[f"K{k}_of_{n}_median_share"] = float(np.median(sub[ok] / full[ok])) if ok.any() else float("nan")
            out[f"K{k}_of_{n}_min_share"] = float((sub[ok] / full[ok]).min()) if ok.any() else float("nan")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--floors", default="0.01,0.02,0.05,0.1")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    reads = []
    for dd in a.dirs:
        for f in sorted(glob.glob(os.path.join(dd, "read_*.json"))):
            d = json.load(open(f))
            if "dither" in d:
                reads.append(d)
        for f in sorted(glob.glob(os.path.join(dd, "compiled_*.json"))):   # a --compiled read made without --dither
            c = json.load(open(f))
            for d in reads:
                if d["label"] == c["label"] and "compiled" not in d and "compiled" in c:
                    d["compiled"] = c["compiled"]
    table: Dict[str, Any] = {"reads": [r["label"] for r in reads], "arms": {}, "k_sensitivity": {}}
    arms: Dict[str, List[Tuple[str, int, float, float]]] = {}
    for d in reads:
        for arm, (s, n) in rows_of(d).items():
            arms.setdefault(arm, []).extend((d["label"], i, float(s[i]), float(n[i])) for i in range(len(s)))
        ks = k_sensitivity(d)
        if ks:
            table["k_sensitivity"][d["label"]] = ks
    floors = [float(x) for x in a.floors.split(",")]
    for arm, rs in arms.items():
        s = np.array([r[2] for r in rs])
        n = np.array([r[3] for r in rs])
        print(f"\n== {arm}: {len(rs)} row-reads over {len({r[0] for r in rs})} checkpoints; old global rule (M {M}) "
              f"refuses {int((s > M).sum())}; null < 0.01 on {int((n < 0.01).sum())} rows "
              f"(their max shift {s[n < 0.01].max() if (n < 0.01).any() else float('nan'):.3g})")
        per_floor = {}
        for fl in floors:
            r = s / np.maximum(n, fl)
            j = int(r.argmax())
            k = R1_BAR_K * float(r.max())
            bar = np.maximum(k * n, fl * k)
            per_floor[str(fl)] = {"p50": float(np.median(r)), "p90": float(np.quantile(r, 0.9)),
                                  "p99": float(np.quantile(r, 0.99)), "max": float(r.max()),
                                  "argmax": [rs[j][0], rs[j][1], rs[j][2], rs[j][3]], "k_4x": k,
                                  "per_row_bar_median": float(np.median(bar)), "per_row_bar_max": float(bar.max()),
                                  "rows_with_bar_below_M": float((bar < M).mean())}
            print(f"  floor {fl:<5g} ratio p50 {np.median(r):.3f} p90 {np.quantile(r, .9):.3f} p99 "
                  f"{np.quantile(r, .99):.3f} max {r.max():.3f} at {rs[j][0]} row {rs[j][1]} "
                  f"(shift {rs[j][2]:.3f}, null {rs[j][3]:.3f}) -> k = 4 x max = {k:.2f}; per-row bar median "
                  f"{np.median(bar):.3f}, max {bar.max():.3f}, tighter than M on {100 * (bar < M).mean():.0f} % of rows")
        table["arms"][arm] = per_floor
    if table["k_sensitivity"]:
        print("\n== the null's K-sensitivity (max over the first K seeds / over all, rows with null > 0.01):")
        for lab, ks in table["k_sensitivity"].items():
            print(f"  {lab}: " + ", ".join(f"{k} {v:.3f}" for k, v in ks.items()))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(table, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
