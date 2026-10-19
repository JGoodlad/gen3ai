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
    python -I read_mat.py plateau                           -> result_mat_plateau.json (A1: the plateau step's read)

AMENDMENT A1 (2026-10-19; PREREGISTRATION §9, committed before any plateau-step output): ``plateau`` assembles the
registered reads of the plateau step from the two tables ``score_mat.py --bank <b> --tag plateau_step`` writes:
  1. THE RECORD: the plateau bank, R-MATURITY over [25M, P*] (both FULL, 800 states: GAP, BELIEF, FRONTIER, the gates;
     CONVERGED NOT READ on one bank, STILL-MOVING);
  2. ITS REPLICATION: the interim bank over the same FULL pair [25M, P*];
  3. the interim LEAN trajectory [25M, 100M, 175M, 225M, P*] on the LEAN states (descriptive trajectory; its verdicts are
     quoted beside the record, never instead of it);
  4. R3 at P* on each bank (E2's rule, ``maturity_rules.r3_read``; all states PRIMARY, each stratum secondary);
  5. BANK-DEPENDENT (§2.5) for every verdict of 1 against its twin in 2 (and R3's plateau against interim);
  6. CONVERGED pooled over both banks (:func:`converged_pooled`): read only when BOTH banks hold P* and the interim
     anchor before it; else NOT READ, with the reason.
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
SOURCES = ("O", "F", "FN", "F+M", "F+U", "O-M", "O-U")
R3_SOURCES = ("O", "F", "F+M", "F+U", "O-M", "O-U")
PLATEAU_TAG = "plateau_step"
FULL_MODES = ("full", "full_hide")


def table_sha(bank: str = "interim", tag: str = "table") -> str:
    with E.using_bank(bank):
        return (E.d("score") / f"{tag}.sha256").read_text().split()[0]


def load_table(bank: str = "interim", tag: str = "table") -> dict:
    with E.using_bank(bank):
        p = E.d("score") / f"{tag}.pkl"
    b = p.read_bytes()
    if hashlib.sha256(b).hexdigest() != table_sha(bank, tag):
        raise SystemExit(f"{p} does not match its sha256 file")
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


def read(tab: dict, labels: Optional[List[str]] = None, converged_on_this_bank: bool = True) -> dict:
    labels = sorted(labels or tab, key=lambda x: tab[x]["step"])
    ids, drops, by = common_ids(tab, labels)
    if not ids:
        raise SystemExit("no common states")
    recs0 = [by[labels[0]][x] for x in ids]
    bt = R.Boot([r["battle"] for r in recs0])
    fn_all = np.array([all(_ok(by[lab][x], "FN") for lab in labels) for x in ids])
    # A1 (README FINDING 4): a read that names a LEAN anchor runs on the LEAN states, so the gates and FRONTIER are read
    # only when EVERY anchor named is FULL (the original LEAN read is unchanged: no anchor of it was FULL)
    full_read = all(tab[lab]["mode"] in FULL_MODES for lab in labels)
    reads, desc = [], {}
    for lab in labels:
        cr, d = anchor_read(tab[lab], [by[lab][x] for x in ids], bt, fn_all, gates=full_read)
        reads.append(cr)
        desc[lab] = d
    out = {"anchors": labels, "n_common": len(ids), "drops": drops, "battles": bt.nb,
           "common_ids_sha256": hashlib.sha256("".join(ids).encode()).hexdigest(), "per_anchor": desc}
    if len(labels) >= 2:
        rm = R.r_maturity(reads, converged_on_this_bank=converged_on_this_bank)
        if not full_read:
            rm["FRONTIER"] = {"verdict": "NOT READ (LEAN: every bucket holds < 150 states; read between FULL anchors)"}
        out["R-MATURITY"] = rm
    return out


def valid_at(tab: dict, label: str) -> List[dict]:
    """One anchor's own valid states (truth ok, no root tie), in id order: R3's states (a one-anchor read)."""
    return sorted((r for r in tab[label]["recs"] if not r["tie"] and _ok(r, "O")), key=lambda r: r["id"])


def r3_at(tab: dict, label: str) -> dict:
    """A1: R3 (E2's rule) at one anchor, LEAF-K8; all states PRIMARY, each stratum SECONDARY (same draws)."""
    recs = valid_at(tab, label)
    if not recs:
        return {"verdict": "NOT READ (no valid state)"}
    if not any(_ok(r, "O-M") for r in recs):
        return {"verdict": f"NOT READ ({label} has no hide-only rows on this bank)"}
    bt = R.Boot([r["battle"] for r in recs])
    imp = {s: vec(recs, s, LEAF) for s in R3_SOURCES}
    out = {"label": label, "n_valid": len(recs), "battles": bt.nb, "all": R.r3_read(bt, imp)}
    for st in sorted({r.get("stratum") for r in recs} - {None}):
        out[f"stratum:{st}"] = R.r3_read(bt, imp, np.array([r.get("stratum") == st for r in recs]))
    out["verdict"] = out["all"]["verdict"]
    return out


def converged_pooled(tabs: Dict[str, dict], last: str, prev: str) -> dict:
    """A1: CONVERGED pooled over the banks: per bank its common states for [prev, last], concatenated, ONE bootstrap
    over every battle of both banks (a battle belongs to one bank). NOT READ unless every bank holds both anchors."""
    missing = {b: [x for x in (prev, last) if x not in t] for b, t in tabs.items()}
    missing = {b: v for b, v in missing.items() if v}
    if missing:
        return {"verdict": f"NOT READ (amendment A1: pooled CONVERGED needs every bank at {prev} and {last}; "
                           f"missing {missing})", "between": [prev, last]}
    recs = {prev: [], last: []}
    n_by = {}
    for b, t in tabs.items():
        ids, _drops, by = common_ids(t, [prev, last])
        n_by[b] = len(ids)
        for lab in (prev, last):
            recs[lab] += [by[lab][x] for x in ids]
    bt = R.Boot([r["battle"] for r in recs[prev]])
    g = {lab: bt.mean(vec(recs[lab], "O", LEAF) - vec(recs[lab], "F", LEAF)) for lab in (prev, last)}
    out = R.converged(g[prev], g[last])
    out.update({"between": [prev, last], "n_common_per_bank": n_by, "battles": bt.nb})
    return out


def _bank_dep(record: dict, repl: dict) -> dict:
    """§2.5 BANK-DEPENDENT for every verdict the record and its replication both hold."""
    rm, rr = record.get("R-MATURITY", {}), repl.get("R-MATURITY", {})
    out = {}
    for c in sorted(set(rm.get("GAP", {})) & set(rr.get("GAP", {}))):
        out[f"GAP:{c}"] = R.bank_dependent("GAP", rm["GAP"][c]["verdict"], rr["GAP"][c]["verdict"])
    for k in ("BELIEF", "FRONTIER"):
        if k in rm and k in rr:
            out[k] = R.bank_dependent(k, rm[k]["verdict"], rr[k]["verdict"])
    return out


def plateau_read() -> dict:
    """A1's registered read of the plateau step (the module docstring's items 1-6)."""
    MS = E.src_module("maturity_series")
    pstar = MS.plateau_label(E.series())
    first = "x26_025M"
    ti, tp = load_table("interim", PLATEAU_TAG), load_table("plateau", PLATEAU_TAG)
    for b, t in (("interim", ti), ("plateau", tp)):
        for lab in (first, pstar):
            if lab not in t or t[lab]["mode"] not in ("full", "full_hide"):
                raise SystemExit(f"{b}: {lab} is not a FULL anchor of the table")
    record = read(tp, [first, pstar], converged_on_this_bank=False)
    repl = read(ti, [first, pstar], converged_on_this_bank=False)
    lean = [x for x in ti if x != pstar] + [pstar]
    traj = read(ti, lean, converged_on_this_bank=False)
    r3 = {"plateau (record)": r3_at(tp, pstar), "interim (replication)": r3_at(ti, pstar)}
    bdep = _bank_dep(record, repl)
    bdep["R3"] = R.bank_dependent("R3", r3["plateau (record)"]["verdict"], r3["interim (replication)"]["verdict"])
    for st in ("snapshot", "bot"):
        k = f"stratum:{st}"
        if k in r3["plateau (record)"] and k in r3["interim (replication)"]:
            bdep[f"R3:{k}"] = R.bank_dependent("R3", r3["plateau (record)"][k]["verdict"],
                                               r3["interim (replication)"][k]["verdict"])
    prev = sorted((x for x in ti if x != pstar), key=lambda x: ti[x]["step"])[-1]
    return {"amendment": "A1 (PREREGISTRATION §9, 2026-10-19)", "plateau_anchor": pstar, "first_anchor": first,
            "1_record_plateau_bank": record, "2_replication_interim_full_pair": repl,
            "3_interim_lean_trajectory": traj, "4_R3_at_plateau": r3, "5_BANK-DEPENDENT": bdep,
            "6_CONVERGED_pooled": converged_pooled({"interim": ti, "plateau": tp}, pstar, prev),
            "tables_sha256": {"interim": table_sha("interim", PLATEAU_TAG),
                              "plateau": table_sha("plateau", PLATEAU_TAG)}}


def main():
    if sys.argv[1:2] == ["plateau"]:
        res = plateau_read()
        (E.HERE / "result_mat_plateau.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=float)
                                                        + "\n")
        print(json.dumps({k: res[k] for k in ("5_BANK-DEPENDENT", "6_CONVERGED_pooled")}, indent=1, default=float))
        return
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
    res["table_sha256"] = table_sha()
    (E.HERE / "result_mat.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=float) + "\n")
    print(json.dumps(res.get("R-MATURITY", {}), indent=1, sort_keys=True, default=float))


if __name__ == "__main__":
    main()
