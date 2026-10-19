"""PREREGISTRATION §4, part 1: THE TABLE. Per anchor and bank state: the stage facts, the anchor's own root (top-1, tie),
the referee T, and for every source (the truth O; the arms F, FN, and FULL's F+M / F+U) whether it has a valid row and
each scorer's IMP; for F, the M8 belief-dose curves (§6.9) that TODAY reads. Pure numpy over the archive; Tier 1's
`t1_data` (`truth_arrays`, `world_arrays`, `scorer_values`, `pick`) and `score_blind.dk_family` are IMPORTED, never
copied, so every number is computed exactly as Tier 1 computed it.

    python -I score_mat.py [--partial]      -> <archive>/score/table.pkl (+ its sha256 in table.sha256)
    python -I score_mat.py --bank plateau --tag plateau_step   -> <archive>/plateau/score/plateau_step.pkl (A1)

A1 (2026-10-19): ``--bank`` scores one bank (its namespace; each anchor at ITS mode on that bank, every anchor that
holds the bank by default), the hide-only arms O−M / O−U are scored like every arm, and ``--tag`` names the table so
the LEAN read's frozen ``table.pkl`` is never overwritten.

``--partial`` prints counts only (a progress read; writes nothing). No outcome is printed in either mode.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mat_common as E  # noqa: E402

E.add_paths()
import score_blind as SB  # noqa: E402  (Tier 1's dose family)
import t1_data as D  # noqa: E402  (Tier 1's per-state arrays)

M8_KEYS = ("M8|LEAF-K8", "M8|TRUNC-4-K8", "M8|CRITIC")
KEEP_SCORERS = ("LEAF-K8", "LEAF-K4", "TRUNC-4-K8", "CRITIC")


def score_anchor(label: str, partial: bool) -> dict:
    meta = E.bank_states()
    reg = E.registered_ckpts()[label]
    mode = E.anchor_mode(label)
    if mode is None:
        raise SystemExit(f"{label} is not scored on the {E.bank().name} bank")
    roots = E.read_dir(E.d("roots"), f"{label}.jsonl")
    ids = E.mode_ids(mode)
    truth = E.read_dir(E.rows_dir(label, "O"), "shard_*.jsonl")
    arms = E.MODE_ARMS[mode]
    rows = {arm: E.read_dir(E.rows_dir(label, arm), "rows_*.jsonl") for arm in arms}
    recs = []
    cnt = {"states": len(ids), "no_root": 0, "tie": 0, "no_truth": 0, "truth_refused": 0,
           "arms": {arm: {"ok": 0, "missing": 0, "excluded": 0, "defect": 0, "refused": 0} for arm in arms}}
    for did in ids:
        m = meta[did]
        rt = roots.get(did)
        if rt is None:
            cnt["no_root"] += 1
            continue
        r1 = truth.get(did)
        if r1 is None:
            cnt["no_truth"] += 1
            continue
        rec = {"id": did, "cp": label, "battle": m.get("battle", did.rsplit("#", 2)[0]), "tie": bool(rt["tie"]),
               "src": {}, "dk": {}}
        rec.update({k: m.get(k) for k in ("k", "k_bin", "turn", "turn_bin", "material", "mat_bin", "stratum",
                                         "outcome", "lean")})
        cnt["tie"] += int(rt["tie"])
        if not r1.get("ok"):
            cnt["truth_refused"] += 1
            rec["src"]["O"] = {"ok": False, "error": str(r1.get("error"))[:200]}
            recs.append(rec)
            continue
        S1 = D.truth_arrays(r1, {"logits": rt["logits"], "top": rt["top"]})
        Tv, t1 = S1["T"], S1["top1"]
        rec["n_legal"] = len(S1["acts"])

        def put(name, vals):
            rec["src"][name] = {"ok": True, "imp": {z: float(Tv[D.pick(v, S1["order"])] - Tv[t1])
                                                    for z, v in vals.items() if z in KEEP_SCORERS}}

        put("O", D.scorer_values(S1))
        for arm in arms:
            rw = rows[arm].get(did)
            c = cnt["arms"][arm]
            if rw is None:
                c["missing"] += 1
                continue
            if not rw.get("ok"):
                c["defect" if rw.get("defect") else "excluded" if rw.get("excluded") else "refused"] += 1
                rec["src"][arm] = {"ok": False, "excluded": bool(rw.get("excluded")), "defect": bool(rw.get("defect")),
                                   "error": str(rw.get("error"))[:200]}
                continue
            Sw = D.world_arrays(rw, S1["acts"], S1["turn"])
            put(arm, D.scorer_values(Sw))
            c["ok"] += 1
            if arm == "F":
                fam = SB.dk_family(S1, Sw, None)
                rec["dk"]["F"] = {k: np.asarray(fam[k], dtype=np.float64) for k in M8_KEYS}
        recs.append(rec)
    if not partial and (cnt["no_root"] or cnt["no_truth"]):
        raise SystemExit(f"{label}: {cnt['no_root']} states without a root, {cnt['no_truth']} without a truth row "
                         "(use --partial)")
    return {"label": label, "step": reg["step"], "mode": mode, "bank": E.bank().name, "recs": recs, "counts": cnt}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--cp", default=None, help="comma list of anchors (default: every anchor that holds the bank)")
    ap.add_argument("--bank", default="interim", choices=sorted(E.BANKS))
    ap.add_argument("--tag", default="table", help="the table's name under <bank>/score/ (default: the LEAN read's)")
    a = ap.parse_args()
    E.set_bank(a.bank)
    labels = a.cp.split(",") if a.cp else sorted(x for x in E.registered_ckpts() if E.anchor_mode(x) is not None)
    tab = {lab: score_anchor(lab, a.partial) for lab in labels}
    if a.partial:
        print(json.dumps({lab: {"scored": len(t["recs"]), **t["counts"]} for lab, t in tab.items()}, indent=1,
                         sort_keys=True))
        return
    b = pickle.dumps(tab, protocol=4)
    out = E.d("score") / f"{a.tag}.pkl"
    out.write_bytes(b)
    sha = hashlib.sha256(b).hexdigest()
    (E.d("score") / f"{a.tag}.sha256").write_text(f"{sha}  {a.tag}.pkl\n")
    print("table:", out, sha)


if __name__ == "__main__":
    main()
