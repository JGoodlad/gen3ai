"""STAGE 1's REGISTERED READ (``PREREGISTRATION.md`` §6, part 2). Committed before any arm's output on a registered state
exists; every rule is ``rules_s1`` (code, tested by ``read_s1_test.py`` on planted effects); this file assembles inputs.

THE STATES: the table's COMPLETE states (every arm holds a play row), minus a root tie or a refused truth (none
expected). ONE battle-clustered bootstrap over their battles (2,000 draws, seed 20261019), every quantity on the same
draws; each contrast is read on the states where BOTH its sources are valid (an exclusion is counted, never imputed).

PRIMARY (per X in B, C_ladder, C_forum_strict, C_pool, NS; design §6.2):
  * Δ_X = IMP_X − IMP_F_legal at LEAF-K8 and TRUNC-4-K8 -> PAYS / HURTS / NULL / NOT DETECTED (``delta_verdict``);
  * MATERIAL: PAYS and r_X = Δ_X / G_F_legal (LEAF-K8, a ratio of means on the same draws) lower bound >= 0.25;
  * SEARCH-ENABLED_X: lower95[IMP_X] > +0.0005 at LEAF-K8 AND lower95[IMP_X − IMP_CRITIC,X] > +0.0005 at TRUNC-4-K8;
  * TODAY_X on F_legal's M8 LEAF-K8 dose curve (0.05 grid, clamped), beside q_SM,X, the exact-world mass of X's own
    worlds; CALIBRATED iff every X's TODAY interval contains its q_SM (``contains``);
SECONDARY (same verdict rule, labelled): F_legal − F (banked), F_off − F, F_legal − F_off (the legality rule alone),
F_rolls − F_legal, F_rolls_w − F_legal (likelihood-weighted), F_rolls_w − F_rolls (the weighting alone).
DESCRIPTIVE: every source's IMP at every scorer, G_X, C_pool − C_forum_strict, Δ per opponent stratum, exclusions, the
world meters, the corpus backoff's λ, F_rolls' weight ESS, the cost; the 1b TRIGGER; the design's meaning key.

    python read_s1.py            -> result_s1.json (this folder)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rules_s1 as R  # noqa: E402
import s1_common as E  # noqa: E402

LEAF, T4, CRIT = "LEAF-K8", "TRUNC-4-K8", "CRITIC"
SCORERS = ("LEAF-K8", "LEAF-K4", "TRUNC-4-K8", "CRITIC")
SECONDARY = (("F_legal", "F"), ("F_off", "F"), ("F_legal", "F_off"), ("F_rolls", "F_legal"),
             ("F_rolls_w", "F_legal"), ("F_rolls_w", "F_rolls"))


def _ok(r: dict, s: str) -> bool:
    return bool(r["src"].get(s, {}).get("ok"))


def vec(recs: List[dict], s: str, z: str) -> np.ndarray:
    return np.array([r["src"][s]["imp"][z] if _ok(r, s) and z in r["src"][s]["imp"] else np.nan for r in recs])


def okv(recs: List[dict], *srcs: str) -> np.ndarray:
    return np.array([all(_ok(r, s) for s in srcs) for r in recs])


def contrast(bt: "R.Boot", recs: List[dict], x: str, base: str) -> dict:
    sel = okv(recs, x, base)
    d = {z: bt.mean(vec(recs, x, z) - vec(recs, base, z), sel) for z in (LEAF, T4)}
    cis = {z: R.ci(d[z][1]) for z in d}
    out = {"n": int(sel.sum()), "delta": {z: R.summary(*d[z], n=int(sel.sum())) for z in d},
           "verdict": R.delta_verdict(cis[LEAF], cis[T4]), "inside_pm_0.01": R.inside(cis[LEAF], R.NULL_MARGIN)}
    return out


def world_mean(bt: "R.Boot", recs: List[dict], arm: str, key: str):
    v = np.array([r["worlds"].get(arm, {}).get(key, np.nan) if _ok(r, arm) else np.nan for r in recs], dtype=float)
    return bt.mean(v)


def read(tab: dict) -> dict:
    recs = [r for r in tab["recs"] if not r["tie"] and _ok(r, "O")]
    if not recs:
        raise SystemExit("no valid state")
    bt = R.Boot([r["battle"] for r in recs])
    out: dict = {"n_complete": tab["counts"]["complete"], "n_valid": len(recs), "battles": bt.nb,
                 "complete_is_registered_prefix": tab["counts"]["is_prefix"], "ids_sha256": tab["ids_sha256"],
                 "dropped_tie_or_truth": len(tab["recs"]) - len(recs), "exclusions": tab["counts"]["arms"]}
    srcs = ["O", "F", "FN", *E.ARMS, "F_rolls_w"]
    out["imp"] = {s: {z: R.summary(*bt.mean(vec(recs, s, z)), n=int(okv(recs, s).sum())) for z in SCORERS
                      if not np.all(np.isnan(vec(recs, s, z)))} for s in srcs}
    out["G"] = {s: R.summary(*bt.mean(vec(recs, "O", LEAF) - vec(recs, s, LEAF), okv(recs, "O", s)))
                for s in srcs if s != "O"}
    # ---- PRIMARY
    prim: Dict[str, dict] = {}
    curve_all = np.stack([r["dk"]["F_legal"]["M8|LEAF-K8"][::5] if "F_legal" in r["dk"] else np.full(21, np.nan)
                          for r in recs])
    curve_t4 = np.stack([r["dk"]["F_legal"]["M8|TRUNC-4-K8"] if "F_legal" in r["dk"] else np.full(21, np.nan)
                         for r in recs])
    g = vec(recs, "O", LEAF) - vec(recs, "F_legal", LEAF)
    for x in E.PRIMARY:
        c = contrast(bt, recs, x, "F_legal")
        sel = okv(recs, x, "F_legal")
        dx = vec(recs, x, LEAF) - vec(recs, "F_legal", LEAF)
        r_est, r_vec = bt.ratio(dx, g, sel)
        c["r"] = R.summary(r_est, r_vec)
        c["MATERIAL"] = R.material(c["verdict"], R.ci(r_vec))
        selx = okv(recs, x)
        imp_leaf = bt.mean(vec(recs, x, LEAF), selx)
        t4c = bt.mean(vec(recs, x, T4) - vec(recs, x, CRIT), selx)
        c["IMP_X_leaf"] = R.summary(*imp_leaf, n=int(selx.sum()))
        c["IMP_X_T4_minus_CRITIC"] = R.summary(*t4c)
        c["SEARCH-ENABLED"] = R.search_enabled(R.ci(imp_leaf[1]), R.ci(t4c[1]))
        for name, cv in (("TODAY", curve_all), ("TODAY_T4 (descriptive)", curve_t4)):
            z = LEAF if name == "TODAY" else T4
            cd = [bt.mean(cv[:, i], sel) for i in range(cv.shape[1])]
            te, tv = bt.mean(vec(recs, x, z), sel)
            p, pv = R.today_clamped(np.array([q[0] for q in cd]), te, np.stack([q[1] for q in cd], 1), tv)
            c[name] = {"p_eq": p, "ci": R.ci(pv)}
        q = world_mean(bt, recs, x, "qSM")
        c["qSM"] = R.summary(*q)
        c["today_contains_qSM"] = R.contains(c["TODAY"]["ci"], q[0])
        c["per_stratum"] = {}
        for st in sorted({r.get("stratum") for r in recs} - {None}):
            m = sel & np.array([r.get("stratum") == st for r in recs])
            c["per_stratum"][st] = {z: R.summary(*bt.mean(vec(recs, x, z) - vec(recs, "F_legal", z), m), n=int(m.sum()))
                                    for z in (LEAF, T4)}
        prim[x] = c
    out["PRIMARY"] = prim
    out["CALIBRATED"] = R.calibrated({x: prim[x]["today_contains_qSM"] for x in E.PRIMARY})
    out["SECONDARY"] = {f"{a} − {b}": contrast(bt, recs, a, b) for a, b in SECONDARY}
    out["C_pool − C_forum_strict"] = contrast(bt, recs, "C_pool", "C_forum_strict")
    out["TRIGGER_1b"] = R.trigger_1b({x: {z: prim[x]["delta"][z]["mean"] for z in (LEAF, T4)} for x in prim},
                                     {x: prim[x]["SEARCH-ENABLED"] for x in prim})
    out["MEANING"] = R.meaning({x: prim[x]["verdict"] for x in prim}, {x: prim[x]["MATERIAL"] for x in prim},
                               {x: prim[x]["qSM"]["mean"] if prim[x]["qSM"]["mean"] is not None else 1.0 for x in prim})
    # ---- DESCRIPTIVE world meters, the backoff, the weights, the cost
    wm: Dict[str, dict] = {}
    for a in E.ARMS:
        blk = {k: R.summary(*world_mean(bt, recs, a, k)) for k in ("qSM", "U_exact", "M_exact", "hit", "near4",
                                                                      "cost_build_cpu_s", "cost_play_cpu_s")}
        if a == "F_rolls":
            blk["ess"] = R.summary(*world_mean(bt, recs, a, "ess"))
        props = [r["worlds"].get(a, {}).get("prop") for r in recs if _ok(r, a)]
        props = [p for p in props if p]
        if props:
            keys = sorted({k for p in props for k in p})
            blk["prop_mean"] = {k: round(float(np.mean([p.get(k, 0) for p in props])), 4) for k in keys}
            if "lambda" in keys:
                blk["share_states_n_gt_0"] = round(float(np.mean([p.get("n", 0) > 0 for p in props])), 4)
        wm[a] = blk
    out["worlds"] = wm
    return out


def main():
    import score_s1 as SC

    tab = SC.load_table()
    res = read(tab)
    res["table_sha256"] = (E.d("score") / "table.sha256").read_text().split()[0]
    (E.HERE / "result_s1.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=float) + "\n")
    print(json.dumps({x: {"verdict": res["PRIMARY"][x]["verdict"], "MATERIAL": res["PRIMARY"][x]["MATERIAL"],
                          "SEARCH-ENABLED": res["PRIMARY"][x]["SEARCH-ENABLED"]} for x in res["PRIMARY"]},
                     indent=1), res["CALIBRATED"], res["TRIGGER_1b"])


if __name__ == "__main__":
    main()
