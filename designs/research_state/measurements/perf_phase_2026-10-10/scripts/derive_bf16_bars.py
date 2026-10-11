"""DERIVE the bf16 PARITY RULE's bars (`compile_regions.BF16_DRIFT_BARS` / `BF16_ENVELOPE_K`) from GPU envelope reads
(`bf16_envelope.py`, one JSON per read). PLAN.md §2a states the rule; this script is its arithmetic, and
`compile_regions_bf16_test` re-runs it on the committed inputs and fails when a constant differs.

    python derive_bf16_bars.py ../results/window_D/E_bf16_envelope.json [more.json ...] --out ../results/bf16_parity_bars.json

THE ARITHMETIC (every number from a healthy read: the compiled graph is the shipped one, no fault planted):

* For every read (a weight state x a fixture slice), d(X, ref) for X in {eager bf16 (E), compiled bf16 (C)} against
  the fp32 eager reference: loss rel |l_X - l_ref| / max(1, |l_ref|), the global gradient's relative L2, and every
  parameter's relative error over the parameters whose reference gradient norm is above `_PARAM_GRAD_FLOOR` (1e-3)
  x the largest — the judged set, exactly as the rule judges.
* DRIFT bars = R1_BAR_K (4) x the healthy MAXIMUM of d(E, ref) over every read and both eager repeats (Ebf, Ebf2),
  per quantity, rounded UP to 2 significant digits.
* ENVELOPE K = R1_BAR_K x the healthy MAXIMUM of d(C, ref) / max(d(E, ref), floor) over every read and both
  compiled repeats (Cbf, Cbf2) against both eager repeats, per quantity, the floor being the fp32 bar the rule uses
  (loss `BF16_LOSS_FLOOR` = max(1e-4, bf16's unit roundoff 2^-9); global `BF16_GRAD_FLOOR`; per parameter `R1_PARAM_BAR[regime]`, the regime being
  `fresh` for the `fresh` state and `trained` for every other), rounded UP to 2 significant digits, and never
  below 1.0 (a compiled graph is never held tighter than eager bf16's own error).
* RULE 8 (deterministic): every healthy read then sits at most 1 / R1_BAR_K = 25 % of the way to its bar, so no
  healthy input is within a rounding error of a boundary.

It also reports the TEETH: for each read, the per-parameter error a backward miscompile must exceed to be refused,
K x max(d(E, ref), floor), as a distribution over the judged parameters (median, p90, max).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.environ.get("DIAG_SRC") or os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "..", "src"))
sys.path.insert(0, SRC)

R1_BAR_K = 4.0


def up2(x: float) -> float:
    """Round UP to 2 significant digits."""
    if x <= 0 or not math.isfinite(x):
        return x
    e = math.floor(math.log10(x)) - 1
    return math.ceil(x / 10 ** e - 1e-9) * 10 ** e


def derive(reads: list, *, param_floor_frac: float, loss_floor: float, grad_floor: float,
           param_bar: dict) -> dict:
    drift = {"loss": 0.0, "grad_rel_l2": 0.0}
    env = {"loss": 0.0, "grad_rel_l2": 0.0, "param": 0.0}
    where = {}
    teeth = []
    n_reads = 0
    for src, state, slice_name, r in reads:
        pairs = r["pairs"]
        ratio_ref = r["grad_norm_ratio_E32"]
        judged = [i for i, x in enumerate(ratio_ref) if x > param_floor_frac]
        floor_p = param_bar["fresh" if state == "fresh" else "trained"]
        n_reads += 1
        for e in ("Ebf", "Ebf2"):
            pe = pairs[f"{e}_vs_E32"]
            for q, v in (("loss", pe["loss_rel"]), ("grad_rel_l2", pe["rel_l2"])):
                if v > drift[q]:
                    drift[q] = v
                    where[f"drift_{q}"] = f"{src}:{state}:{slice_name}:{e}"
            for c in ("Cbf", "Cbf2"):
                pc = pairs[f"{c}_vs_E32"]
                for q, dc, de, fl in (("loss", pc["loss_rel"], pe["loss_rel"], loss_floor),
                                      ("grad_rel_l2", pc["rel_l2"], pe["rel_l2"], grad_floor)):
                    rt = dc / max(de, fl)
                    if rt > env[q]:
                        env[q] = rt
                        where[f"env_{q}"] = f"{src}:{state}:{slice_name}:{c}/{e}"
                for i in judged:
                    rt = pc["per_param_rel"][i] / max(pe["per_param_rel"][i], floor_p)
                    if rt > env["param"]:
                        env["param"] = rt
                        where["env_param"] = f"{src}:{state}:{slice_name}:{c}/{e}:#{i}"
        pe = pairs["Ebf_vs_E32"]
        teeth.append((state, slice_name, sorted(pe["per_param_rel"][i] for i in judged), floor_p))
    drift_bars = {q: up2(R1_BAR_K * v) for q, v in drift.items()}
    env_k = {q: max(1.0, up2(R1_BAR_K * v)) for q, v in env.items()}
    teeth_out = []
    for state, sl, errs, floor_p in teeth:
        thr = sorted(env_k["param"] * max(x, floor_p) for x in errs)
        if thr:
            teeth_out.append({"state": state, "slice": sl, "n_params": len(thr),
                              "refuse_above_median": thr[len(thr) // 2],
                              "refuse_above_p90": thr[int(0.9 * (len(thr) - 1))], "refuse_above_max": thr[-1]})
    return {"n_reads": n_reads, "healthy_max": {"drift": drift, "envelope_ratio": env}, "argmax": where,
            "BF16_DRIFT_BARS": drift_bars, "BF16_ENVELOPE_K": env_k, "R1_BAR_K": R1_BAR_K, "teeth": teeth_out,
            "floors": {"loss": loss_floor, "grad_rel_l2": grad_floor, "param": param_bar,
                       "param_floor_frac": param_floor_frac}}


def load_reads(paths: list) -> list:
    reads = []
    for p in paths:
        d = json.load(open(p))
        for state, s in d["states"].items():
            for sl in ("slice0", "slice1"):
                if sl in s:
                    reads.append((os.path.basename(p), state, sl, s[sl]))
    return reads


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out")
    a = ap.parse_args()
    from agents.model import compile_regions as cr
    from agents.model import compile_trainer as ct
    out = derive(load_reads(a.inputs), param_floor_frac=ct._PARAM_GRAD_FLOOR, loss_floor=cr.BF16_LOSS_FLOOR,
                 grad_floor=cr.BF16_GRAD_FLOOR, param_bar=dict(cr.R1_PARAM_BAR))
    out["inputs"] = [os.path.relpath(p, os.path.join(HERE, "..")) for p in a.inputs]
    txt = json.dumps(out, indent=1)
    if a.out:
        with open(a.out, "w") as f:
            f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main()
