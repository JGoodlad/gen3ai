"""A DRESS REHEARSAL of the read (no Tier-1 outcome exists or is read): the real table of the TRUTH (Phase 1) and the
pilot's ALREADY-READ v2 base P (`det2`), with every Tier-1 arm code and every hand-eval code FABRICATED as a copy of P
(the hand-eval scorers mapped from P's critic scorers), pushed through every function of `unblind_t1.py`. It catches a
crash or a shape error before the scripts are frozen; its numbers mean nothing (every arm equals P) and are not
written anywhere.

    python -I rehearsal.py
"""
import copy
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_blind as SB  # noqa: E402
import t1_common as T  # noqa: E402
import unblind_t1 as U  # noqa: E402

HE_MAP = {"HE-TRUNC-1-K4": "TRUNC-1-K4", "HE-TRUNC-1-K8": "TRUNC-1-K8", "HE-TRUNC-2-K4": "TRUNC-2-K4",
          "HE-TRUNC-2-K8": "TRUNC-2-K8", "HE-TRUNC-4-K4": "TRUNC-4-K4", "HE-TRUNC-4-K8": "LEAF-K8",
          "HE-1PLY": "LEAF-K4", "VT-TRUNC-4-K8": "TRUNC-4-K8", "VT-CRITIC": "CRITIC"}


def main():
    import t1_data as D

    D.coded_dirs = lambda cp: []          # never load a real Tier-1 arm output (they may exist by now)
    D.he_rows = lambda cp, code: {}
    tab = {cp: SB.score_cp(cp, partial=True) for cp in T.CPS}
    name = {"O": "O", "P": "P", "HE@O": "H8e895870", "HE@P": "H8503aeeb", "HE@F": "H528ede2b"}
    for i, lab in enumerate(T.ARMS):
        name[lab] = f"R{i:02d}"
    for cp in T.CPS:
        tab[cp]["recs"] = [r for r in tab[cp]["recs"] if "P" in r["src"]]
        rng = np.random.default_rng(0)
        for r in tab[cp]["recs"]:
            r["set"] = "registered" if rng.random() < 0.8 else "supp"      # exercise both state sets
            for lab in T.ARMS:
                r["src"][name[lab]] = copy.deepcopy(r["src"]["P"])
                d = dict(r["dk"]["P"])
                d.update({f"M5X|{z}": d[f"M5W|{z}"] for z in SB.DK_SCORERS})
                r["dk"][name[lab]] = d
                r["w"][name[lab]] = r["w"]["P"]
                r["hits"][name[lab]] = [int(x) for x in rng.integers(1, 9, size=4)]
            for h in ("HE@O", "HE@P", "HE@F"):
                base = r["src"]["O" if h == "HE@O" else "P"]
                r["src"][name[h]] = {"ok": True, "imp": {k: base["imp"][v] for k, v in HE_MAP.items()},
                                     "pick": {k: base["pick"][v] for k, v in HE_MAP.items()}}
    fr_reg, fr_all = U.frames(tab, "registered"), U.frames(tab, "all")
    out = {"size": U.size_rules(fr_reg, fr_all, name), "validity": U.validity(fr_all, name, tab),
           "moves": U.r_moves(fr_reg, name, True)["verdict"], "U": U.r_rival_u(fr_reg, name)["verdict"],
           "cond": U.condition_robust(fr_reg["pooled"], name)["verdict"],
           "recover": U.r_recover(fr_reg["pooled"], name)["verdict"], "eval": U.r_eval(fr_reg, name)["verdict_bound"],
           "voi": {k: v["verdict"] for k, v in U.stage_voi(fr_all, name).items()},
           "axis": {k: v["verdict"] for k, v in U.stage_axis(fr_all, name).items()},
           "mech": {k: v["verdict"] for k, v in U.stage_mech(fr_all, name).items()},
           "class_k": {b: v["verdict"] for b, v in U.stage_class(fr_all, name)["k"].items()},
           "frontier": U.stage_search(fr_all, name)["FRONTIER"]}
    dz = U.dose(fr_all, name)
    out["dose_all"] = {k: dz["all"][k] for k in ("P-STAR", "TODAY", "REACH", "SHAPE", "p_half")}
    out["determinization_first"] = {ax: v["DETERMINIZATION-FIRST"] for ax, v in dz["axes"].items()}
    out["natural_mix"] = U.natural_mix(fr_all, name)
    out["root_choice_dropped"] = U.root_choice_sensitivity(fr_reg, name)["states_dropped"]
    cd = U.class_dose(fr_all, name)
    out["class_dose_k"] = {b: v.get("CLASS-FIRST") for b, v in cd["axes"]["k"].items()}
    out["class_dose_all_M"] = {k: cd["all"]["M"][k] for k in ("P-STAR", "SHAPE", "p_half", "TODAY_proxy")}
    U.describe(fr_reg, name)
    res = U.assemble(tab, name, "rehearsal", {"verdict": "PASS"}, fr_reg, fr_all, out["size"])
    scratch = T.ARCH / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    (scratch / "result_rehearsal.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n")
    import render_t1

    sys.argv = ["render_t1.py", str(scratch / "result_rehearsal.json"), str(scratch / "result_rehearsal.md")]
    render_t1.main()
    print(json.dumps(out, indent=1, sort_keys=True, default=str)[:3000])
    print("REHEARSAL OK (every arm is a copy of P: the numbers mean nothing)")


if __name__ == "__main__":
    main()
