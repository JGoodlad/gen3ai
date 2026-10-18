"""PREREGISTRATION §4, part 2: THE READ. Committed before any anchor's output on a bank state exists; every rule is
code (``src/main/hidden_info_study/maturity_rules.py``, loaded by path) and this file only assembles its inputs.

Per anchor (on the COMMON states: truth ok, F ok and no root tie at EVERY anchor read, so each across-anchor contrast is
paired state for state; ONE battle-clustered bootstrap over their battles):
  * IMP of O, F, FN at LEAF-K8 / LEAF-K4 / TRUNC-4-K8 / CRITIC (descriptive);
  * G_F = IMP_O − IMP_F at LEAF-K8, all-state and per stage bucket (k, turn, material), and per stratum (descriptive);
  * TODAY = FN's p-equivalent on F's M8 LEAF-K8 curve, Tier 1's 0.05 grid, clamped to [0, 1] per draw (on the common
    states whose FN row is ok at every anchor read);
  * the per-bucket GATE (R-STAGE-SEARCH on one network). In LEAN every bucket holds < 150 states, so no bucket can be
    GATE-OPEN by the n rule: the gate statistics are printed and FRONTIER is NOT READ unless the first and the last
    anchor are both FULL.
Across anchors: R-MATURITY (``maturity_rules.r_maturity``): GAP-SHRINKS / GROWS per cell, BELIEF-RISES, FRONTIER,
CONVERGED (the last two anchors).

    python -I read_mat.py [--partial] [--cp x26_025M,...]   -> result_mat.json (this folder)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mat_common as E  # noqa: E402

R = E.src_module("maturity_rules")
LEAF = "LEAF-K8"
SCORERS = ("LEAF-K8", "LEAF-K4", "TRUNC-4-K8", "CRITIC")
PF = np.round(np.linspace(0.0, 1.0, 21), 2)
SOURCES = ("O", "F", "FN", "F+M", "F+U")


def load_table() -> dict:
    p = E.ARCH / "score" / "table.pkl"
    b = p.read_bytes()
    want = (p.parent / "table.sha256").read_text().split()[0]
    if hashlib.sha256(b).hexdigest() != want:
        raise SystemExit("table.pkl does not match table.sha256")
    return pickle.loads(b)


def _ok(rec: dict, src: str) -> bool:
    return bool(rec["src"].get(src, {}).get("ok"))


def common_ids(tab: dict, labels: List[str]):
    """(sorted common ids, drops {reason: n}) — states valid at EVERY anchor read."""
    by = {lab: {r["id"]: r for r in tab[lab]["recs"]} for lab in labels}
    every = set.intersection(*(set(by[lab]) for lab in labels))
    drops = {"tie": 0, "truth": 0, "F": 0}
    keep = []
    for did in sorted(every):
        rs = [by[lab][did] for lab in labels]
        if any(r["tie"] for r in rs):
            drops["tie"] += 1
        elif not all(_ok(r, "O") for r in rs):
            drops["truth"] += 1
        elif not all(_ok(r, "F") for r in rs):
            drops["F"] += 1
        else:
            keep.append(did)
    return keep, drops, by


def vec(recs: List[dict], src: str, z: str) -> np.ndarray:
    return np.array([r["src"][src]["imp"][z] if _ok(r, src) else np.nan for r in recs])


def anchor_read(t: dict, recs: List[dict], bt, fn_all: np.ndarray, gates: bool):
    """(CheckpointRead, descriptive dict) for one anchor on the common states (``recs`` in common-id order)."""
    g = vec(recs, "O", LEAF) - vec(recs, "F", LEAF)
    cells = {"all": np.ones(len(recs), bool)}
    for ax, (fld, buckets) in R.AXES.items():
        for b in buckets:
            cells[f"{ax}:{b}"] = np.array([r.get(fld) == b for r in recs])
    G = {c: bt.mean(g, m) for c, m in cells.items()}
    desc: dict = {"label": t["label"], "step": t["step"], "mode": t["mode"], "n_common": len(recs),
                  "G_F": {c: R.summary(*G[c], n=int(m.sum())) for c, m in cells.items()},
                  "imp": {s: {z: R.summary(*bt.mean(vec(recs, s, z))) for z in SCORERS}
                          for s in SOURCES if any(_ok(r, s) for r in recs)}}
    for s in sorted({r.get("stratum") for r in recs} - {None}):
        m = np.array([r.get("stratum") == s for r in recs])
        desc["G_F"][f"stratum:{s}"] = R.summary(*bt.mean(g, m), n=int(m.sum()))
    today = None
    if fn_all.any():
        curve = np.stack([r["dk"]["F"]["M8|LEAF-K8"][::5] for r in recs])          # [n, 21] on the 0.05 grid
        cd = [bt.mean(curve[:, i], fn_all) for i in range(len(PF))]
        ce = np.array([c[0] for c in cd])
        cv = np.stack([c[1] for c in cd], 1)
        fe, fv = bt.mean(vec(recs, "FN", LEAF), fn_all)
        today = R.today_clamped(ce, fe, cv, fv, PF)
        desc["TODAY"] = {"n": int(fn_all.sum()), "p_eq": today[0], "ci": R.ci(today[1]),
                         "IMP_FN": R.summary(fe, fv)}
    gate_tab: Dict[str, Dict[str, str]] = {}
    desc["gates"] = {}
    t4c = vec(recs, "F", "TRUNC-4-K8") - vec(recs, "F", "CRITIC")
    t4 = vec(recs, "F", "TRUNC-4-K8")
    for ax, (fld, buckets) in R.AXES.items():
        gate_tab[ax] = {}
        for b in buckets:
            m = cells[f"{ax}:{b}"]
            c1, c2 = bt.mean(t4c, m), bt.mean(t4, m)
            st = R.gate_status(int(m.sum()), c1[1], c2[1])
            gate_tab[ax][b] = st
            desc["gates"][f"{ax}:{b}"] = {"n": int(m.sum()), "status": st, "TRUNC4_minus_CRITIC": R.summary(*c1),
                                          "TRUNC4_minus_CRITIC_996": R.summary(*c1, level=R.BONF_LEVEL),
                                          "TRUNC4": R.summary(*c2)}
    cr = R.CheckpointRead(label=t["label"], step=t["step"], g=G, today=today, gates=gate_tab if gates else {})
    return cr, desc


def read(tab: dict, labels: Optional[List[str]] = None) -> dict:
    labels = sorted(labels or tab, key=lambda x: tab[x]["step"])
    ids, drops, by = common_ids(tab, labels)
    if not ids:
        raise SystemExit("no common states")
    recs0 = [by[labels[0]][x] for x in ids]
    bt = R.Boot([r["battle"] for r in recs0])
    fn_all = np.array([all(_ok(by[lab][x], "FN") for lab in labels) for x in ids])
    full = [tab[lab]["mode"] == "full" for lab in labels]
    reads, desc = [], {}
    for lab in labels:
        cr, d = anchor_read(tab[lab], [by[lab][x] for x in ids], bt, fn_all, gates=full[0] and full[-1])
        reads.append(cr)
        desc[lab] = d
    out = {"anchors": labels, "n_common": len(ids), "drops": drops, "battles": bt.nb,
           "common_ids_sha256": hashlib.sha256("".join(ids).encode()).hexdigest(), "per_anchor": desc}
    if len(labels) >= 2:
        rm = R.r_maturity(reads)
        if not (full[0] and full[-1]):
            rm["FRONTIER"] = {"verdict": "NOT READ (LEAN: every bucket holds < 150 states; read between FULL anchors)"}
        out["R-MATURITY"] = rm
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--cp", default=None)
    a = ap.parse_args()
    if a.partial:
        import subprocess

        subprocess.run([sys.executable, "-I", str(E.HERE / "score_mat.py"), "--partial"], check=True)
        return
    tab = load_table()
    res = read(tab, a.cp.split(",") if a.cp else None)
    res["table_sha256"] = (E.ARCH / "score" / "table.sha256").read_text().split()[0]
    (E.HERE / "result_mat.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=float) + "\n")
    print(json.dumps(res.get("R-MATURITY", {}), indent=1, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
