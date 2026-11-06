"""The census REPLICATE (2026-11-05; README here, "The census and its replicate"): the second CUDA census of the 20
X26-lineage checkpoints against W0's (the basis of `canary_row_bar.ROW_BAR_K`), and the driver question.

The replicate (`gpu_census.sh` into ~/gen3ai_archive/canary_census_2026-10-25) read 14 checkpoints on 10-25 under NVIDIA
driver 595.91.07 and the last 6 on 11-05 under 595.99.02 (`linux-modules-nvidia-595-open`, kernel 7.0.0-42). W0
(~/.claude/jobs/x35/w0_2026-10-28/census) read all 20 on 10-28 under 595.91.07. Same script, same env (torch
2.8.0+cu126, triton 3.4.0). Prints, CPU only, seconds:

  1. per checkpoint, replicate vs W0: the max |difference| of every per-row quantity (fp32 null, eager bf16 at B = 64
     and 2,048, compiled bf16 at 2,048), batch by batch -- the driver test on the SAME weights;
  2. the compiled arm's draw-to-draw spread, same-driver pairs against cross-driver pairs;
  3. the RULED k (`canary_recalibration_2026-10-28.md` §1, `canary_row_bar_test.test_k_RE_DERIVES_from_the_W0_census`):
     round_up_2sig(4 x max shift / max(16-seed null, 0.02)) over the compiled AND eager arms at B = 2,048, on W0, on the
     replicate, per driver batch, and with both compile draws pooled; plus stage 1's backtest on the replicate.

    python census_replicate.py <replicate dir> <W0 census dir>
"""
from __future__ import annotations

import glob
import json
import math
import os
import sys
from typing import Dict, Optional, Set, Tuple

import numpy as np

NEW_DRIVER = ("rb_x26c_s2601_470M", "rb_x26c_s2601_480M", "rb_x26c_s2601_481M", "rb_x26c_s2601_482M",
              "x26c_u1000_run_478M", "x26c_u1000_run_479M")
R1_BAR_K, FLOOR, SEEDS, K_RULED, CAP = 4.0, 0.02, 16, 6.0, 0.72
Read = Dict[str, np.ndarray]


def round_up_2sig(x: float) -> float:
    e = math.floor(math.log10(x)) - 1
    return float(f"{math.ceil(x / 10 ** e - 1e-9) * 10 ** e:.10g}")


def load(d: str) -> Dict[str, Read]:
    out: Dict[str, Read] = {}
    for f in sorted(glob.glob(os.path.join(d, "read_*.json"))):
        with open(f) as fh:
            x = json.load(fh)
        c, dd = x["compiled"]["per_row"], x["dither"]
        per = np.array([dd["rows"][str(i)]["E32~"] for i in range(64)])
        out[x["label"]] = {"null32": np.asarray(dd["E32~_per_row_max"], float), "null_seeds": per,
                           "null16": per[:, :SEEDS].max(1),
                           "Ebf~_seeds": np.array([dd["rows"][str(i)]["Ebf~"] for i in range(64)]),
                           "Ebf64": np.asarray(x["per_row"]["abs_dlogp_Ebf"], float),
                           "Ebf": np.asarray(c["Ebf_vs_E32"], float), "Cbf": np.asarray(c["Cbf_vs_E32"], float),
                           "zip": np.array(os.path.basename(x["zip"]))}
    return out


def worst(src: Dict[str, Read], arms: Tuple[str, ...], labels: Optional[Set[str]] = None
          ) -> Tuple[float, str, str, int, float, float]:
    best: Tuple[float, str, str, int, float, float] = (-1.0, "", "", -1, 0.0, 0.0)
    for lab, e in src.items():
        if labels is not None and lab not in labels:
            continue
        nn = np.maximum(e["null16"], FLOOR)
        for arm in arms:
            r = e[arm] / nn
            j = int(r.argmax())
            if r[j] > best[0]:
                best = (float(r[j]), lab, arm, j, float(e[arm][j]), float(e["null16"][j]))
    return best


def main() -> int:
    rep, w0 = load(sys.argv[1]), load(sys.argv[2])
    assert set(rep) == set(w0) and len(rep) == 20 and set(NEW_DRIVER) <= set(rep)
    new = set(NEW_DRIVER)
    old = set(rep) - new
    print("== 1. replicate vs W0 on the SAME checkpoint: max |difference| per quantity")
    for lab in sorted(rep, key=lambda s: (s in new, s)):
        a, w = rep[lab], w0[lab]
        assert str(a["zip"]) == str(w["zip"])
        cells = " ".join(f"{k} {np.abs(a[k] - w[k]).max():.1e}"
                         for k in ("null32", "null_seeds", "Ebf~_seeds", "Ebf64", "Ebf", "Cbf"))
        print(f"  {'NEW 11-05' if lab in new else 'old 10-25'} vs W0 10-28  {lab:22s} {cells}")
    print("\n== 2. the compiled arm's draw-to-draw |difference| per row (replicate vs W0)")
    for name, labs in (("same driver  (10-25 vs 10-28)", old), ("cross driver (11-05 vs 10-28)", new)):
        d = np.concatenate([np.abs(rep[lab]["Cbf"] - w0[lab]["Cbf"]) for lab in sorted(labs)])
        print(f"  {name}: n {d.size}, median {np.median(d):.4f}, p90 {np.quantile(d, .9):.4f}, p99 "
              f"{np.quantile(d, .99):.4f}, max {d.max():.4f}, bit-equal rows {100 * np.mean(d == 0):.0f} %")
    print(f"\n== 3. the RULED k: round_up_2sig({R1_BAR_K:g} x max shift / max({SEEDS}-seed null, {FLOOR}))")
    pooled = {lab: {**w0[lab], "Cbf": np.maximum(w0[lab]["Cbf"], rep[lab]["Cbf"])} for lab in w0}
    rows = [("W0 (the committed basis)", w0, None), ("replicate, all 20", rep, None),
            ("replicate, old driver 14", rep, old), ("  W0 on those 14", w0, old),
            ("replicate, NEW driver 6", rep, new), ("  W0 on those 6", w0, new),
            ("both compile draws pooled", pooled, None)]
    for name, src, labs in rows:
        for arms in (("Cbf",), ("Ebf",), ("Cbf", "Ebf")):
            m, lab, arm, j, s, n = worst(src, arms, labs)
            print(f"  {name:27s} {'+'.join(arms):8s} max {m:.3f} at {lab} {arm} row {j} (shift {s:.3f}, null16 "
                  f"{n:.3f}) -> k {round_up_2sig(R1_BAR_K * m)}")
    refusing, top = 0, (0.0, "")
    for lab, e in rep.items():
        bars = np.minimum(K_RULED * np.maximum(e["null16"], FLOOR), CAP)
        refusing += int((e["Cbf"] > bars).any())
        top = max(top, (float((e["Cbf"] / bars).max()), lab))
    print(f"  stage 1 (k {K_RULED}, K {SEEDS}, floor {FLOOR}, cap {CAP}) on the replicate's compiled arm: "
          f"{refusing} of 20 checkpoints refuse; the nearest reads {top[0]:.3f} of its bar ({top[1]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
