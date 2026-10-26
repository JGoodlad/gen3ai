"""L1's REGISTERED REPLICATION on the maturity battery's PLATEAU bank (`PREREGISTRATION.md`; scoring only, zero playouts).

Loads the plateau bank's banked rows (truth O + honest arm F at the two FULL anchors 25M and P*), applies L1's rules
at the values FITTED ON TIER 1 (`../search_l1_l5_2026-10-20/fit_l1.json`, never refitted), and reads the registered
cell. Pure numpy over the archives; READ-ONLY on every input; the output is `result_l1_plateau.json` beside this file.

    python -I l1_plateau.py read
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MEAS = HERE.parent
L1DIR = MEAS / "search_l1_l5_2026-10-20"
MATDIR = MEAS / "hidden_info_maturity_x26_2026-10-18"
sys.path.insert(0, str(L1DIR / "scripts"))
import l1l5_data as L  # noqa: E402  (the registered loader's row readers and per-state arrays)
import l1_read as R  # noqa: E402  (the registered rules, frame, fit-free summaries)

ARCH = L.MAT_ARCH / "plateau"
LABELS = ("x26_025M", "x26_P280M")
OUT = HERE / "result_l1_plateau.json"
PRIMARY = R.PRIMARY
CELL = ("PIKL", "TRUNC-4-K8")          # the registered cell (L1's one PAYS-SCOPED cell)


def load(log: dict):
    tp_ = ARCH / "score" / "plateau_step.pkl"
    want = (ARCH / "score" / "plateau_step.sha256").read_text().split()[0]
    got = L.sha_file(tp_)
    reg = json.loads((MATDIR / "result_mat_plateau.json").read_text())
    rec1 = reg["1_record_plateau_bank"]
    if got != want or got != reg["tables_sha256"]["plateau"]:
        raise SystemExit("plateau table sha256 differs from its file / the registered plateau read")
    tab = pickle.loads(tp_.read_bytes())
    st = json.loads((MATDIR / "states_mat_x26_plateau.json").read_text())["states"]
    by = {lab: {r["id"]: r for r in tab[lab]["recs"]} for lab in LABELS}
    common = []
    for did in sorted(set.intersection(*(set(by[lab]) for lab in LABELS))):
        rs = [by[lab][did] for lab in LABELS]
        if any(r["tie"] for r in rs):
            continue
        if all(r["src"].get("O", {}).get("ok") and r["src"].get("F", {}).get("ok") for r in rs):
            common.append(did)
    csha = hashlib.sha256("".join(common).encode()).hexdigest()
    if csha != rec1["common_ids_sha256"] or len(common) != rec1["n_common"]:
        raise SystemExit("the common states differ from the registered plateau read's")
    log.update({"table_sha256": got, "n_common": len(common), "common_ids_sha256": csha, "per_anchor": {}})
    out = []
    for lab in LABELS:
        roots = L.read_dir(ARCH / "roots", f"{lab}.jsonl")
        truth = L.read_dir(ARCH / "rows" / lab / "O", "shard_*.jsonl")
        rowsF = L.read_dir(ARCH / "rows" / lab / "F", "rows_*.jsonl")
        n = {"F_ok": 0, "F_argmax_mismatch": 0, "O_argmax_mismatch": 0}
        for did in common:
            rt = roots[did]
            tp = L.truth_part(truth[did], rt["logits"], rt["top"])
            m = st[did]
            rec = {"bank": "plateau", "cp": lab, "id": did, "battle": m["battle"], **tp["rec"], "k": int(m["k"]),
                   "turn": int(m["turn"]), "material": int(m["material"]), "opp": m["stratum"]}
            rec["F"] = L.world_part(rowsF[did], tp)
            for src, key in (("F", "F"), ("O", "O")):
                mine = L.argmax_imps(rec[src], rec["Tp"])
                reg_imp = by[lab][did]["src"][key]["imp"]
                zs = L.SCORERS if src == "F" else ("LEAF-K8",)       # the O rows' registered K8 arrays: LEAF-K8 only here
                if any(abs(mine[z] - reg_imp[z]) > 1e-9 for z in zs):
                    n[f"{src}_argmax_mismatch"] += 1
            n["F_ok"] += 1
            out.append(rec)
        log["per_anchor"][lab] = n
    return out, reg


def gates(log: dict) -> dict:
    bad = []
    if log["n_common"] != 719:
        bad.append(f"common {log['n_common']} != 719")
    for lab, n in log["per_anchor"].items():
        if n["F_ok"] != 719:
            bad.append(f"{lab}: {n['F_ok']} F rows != 719")
        if n["F_argmax_mismatch"] or n["O_argmax_mismatch"]:
            bad.append(f"{lab}: ARGMAX mismatches {n}")
    if bad:
        raise SystemExit("GIGO gate FAILED: " + "; ".join(bad))
    return {"G1_G3": "PASS", "counts": {lab: n["F_ok"] for lab, n in log["per_anchor"].items()}}


def read() -> dict:
    log: dict = {}
    recs, reg = load(log)
    gate = gates(log)
    fit = json.loads((L1DIR / "fit_l1.json").read_text())
    fitted = {Rl: {z: (float("inf") if fit[Rl][z]["fitted"] == "inf" else float(fit[Rl][z]["fitted"]))
                   for z in R.SCORERS} for Rl in R.RULES}
    for r in recs:
        r["stratum_bt"] = r["cp"]
    F = R.Frame(recs, False)
    out: dict = {"status": "L1 REPLICATION on the plateau bank (PREREGISTRATION.md)", "referee": "SELF-REFEREE (every cell)",
                 "gigo": {**gate, **{k: log[k] for k in ("table_sha256", "common_ids_sha256", "n_common", "per_anchor")}},
                 "fit_l1_sha256": L.sha_file(L1DIR / "fit_l1.json"),
                 "fitted_on_tier1": {Rl: {z: R.gx(x) for z, x in v.items()} for Rl, v in fitted.items()},
                 "n_rows": F.n, "n_battles": int(F.bt.nb), "level_verdict": R.LEVEL_VERDICT}
    primary = {Rl: {"plateau": {z: R.read_rule(F, "F", z, Rl, fitted[Rl][z]) for z in R.SCORERS}} for Rl in R.RULES}
    out["primary"] = primary
    out["ARGMAX"] = {z: R.summ(F.bt, F.imp("F", z, "ARGMAX", 0.0)) for z in R.SCORERS}
    # THE REGISTERED CELL
    c = primary[CELL[0]]["plateau"][CELL[1]]
    lo = c["IMP"]["ci_verdict"][0]
    out["registered_cell"] = {"cell": "|".join(CELL) + "|plateau", "IMP": c["IMP"], "minus_ARGMAX": c["minus_ARGMAX"],
                              "lower_98.33": lo, "clear": bool(R.ST.gt(lo, R.EDGE))}
    out["verdict_replication"] = "REPLICATES" if out["registered_cell"]["clear"] else "NOT REPLICATED"
    # the single-bank form of PREREG 4.3 (descriptive label)
    cells_clear = [f"{Rl}|{z}" for Rl in R.RULES for z in PRIMARY
                   if R.ST.gt(primary[Rl]["plateau"][z]["IMP"]["ci_verdict"][0], R.EDGE)]
    pays = [Rl for Rl in R.RULES if all(R.ST.gt(primary[Rl]["plateau"][z]["IMP"]["ci_verdict"][0], R.EDGE) for z in PRIMARY)]
    bounded = all(R.ST.lt(primary[Rl]["plateau"][z]["IMP"]["ci_verdict"][1], R.MATERIAL) for Rl in R.RULES for z in PRIMARY)
    out["single_bank_4_3"] = {"label": "PAYS" if pays else "PAYS-SCOPED" if cells_clear else "NULL" if bounded else "NOT DETECTED",
                              "rules_paying_both_primaries": pays, "clear_cells": cells_clear, "every_upper_below_material": bounded}
    # SECONDARIES (descriptive)
    sec: dict = {}
    sub = {}
    for lab in LABELS:
        sub[f"anchor {lab}"] = np.array([r["cp"] == lab for r in recs])
    for o in sorted({r["opp"] for r in recs}):
        sub[f"stratum {o} (SELF-REFEREE)"] = np.array([r["opp"] == o for r in recs])
    for k in (1, 2, 3, 4, 5, 6):
        sub[f"k {k}"] = np.array([(r["k"] if r["k"] <= 5 else 6) == k for r in recs])
    sec["by_cell"] = {nm: {Rl: {z: R.summ(F.bt, F.imp("F", z, Rl, fitted[Rl][z]), m) for z in PRIMARY} for Rl in R.RULES}
                      for nm, m in sub.items() if m.sum() >= 30}
    sec["ARGMAX_by_cell"] = {nm: {z: R.summ(F.bt, F.imp("F", z, "ARGMAX", 0.0), m) for z in PRIMARY}
                             for nm, m in sub.items() if m.sum() >= 30}
    sec["heldout_curves (an upper envelope, NOT a verdict)"] = {
        Rl: {z: [{"x": R.gx(x), **R.summ(F.bt, F.imp("F", z, Rl, x))} for x in R.GRIDS[Rl]] for z in PRIMARY} for Rl in R.RULES}
    sec["anchored_rollout_minus_anchored_critic"] = {
        Rl: {z: R.summ(F.bt, F.imp("F", z, Rl, fitted[Rl][z]) - F.imp("F", "CRITIC", Rl, fitted[Rl]["CRITIC"]))
             for z in PRIMARY} for Rl in R.RULES}
    sec["oracle_world_ARGMAX (O, LEAF-K8; context)"] = R.summ(F.bt, np.array([_o_imp(r) for r in recs]))
    out["secondary (descriptive)"] = sec
    OUT.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    return out


def _o_imp(r: dict) -> float:
    q = np.asarray(r["O"]["W"], dtype=np.float64).mean(1)
    return float(r["Tp"][R.first_max(q)] - r["Tp"][0])


if __name__ == "__main__":
    if sys.argv[1:] != ["read"]:
        raise SystemExit("usage: l1_plateau.py read")
    res = read()
    rc = res["registered_cell"]
    print(json.dumps({"fitted": res["fitted_on_tier1"], "gigo": res["gigo"]["G1_G3"], "registered_cell": rc,
                      "replication": res["verdict_replication"], "single_bank": res["single_bank_4_3"]}, indent=1, default=str))
