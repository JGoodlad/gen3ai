"""L3's FAMILIARITY COUNTER-TEST, THE REGISTERED READ (``PREREGISTRATION.md`` §4–§6). Pure numpy over the driver's
rows (``decoy_run.py``) and L3's archived rows (gate G5); no model, no core. Reuses L3's registered read
(``../l3_read.py``: the battle-clustered bootstrap, the tie band, the per-state AUC) unchanged.

    python -I decoy_read.py --rows <archive>/rows.jsonl --recheck <archive>/recheck.jsonl \\
        --l3-rows ~/gen3ai_archive/search_l3/rows_0_of_1.jsonl --out result_decoy.json
    python -I decoy_read.py --selfcheck
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import l3_read as R  # noqa: E402  (L3's registered read: Boot, ci, summ, auc_state, TIE, J)

IDS_SHA256 = "cf8d8c6fc09c9a85ce976a85a6195eb52977900659ac9be218b47562d9a09dfb"
#: G5: the truth's and the honest worlds' log L and conf must reproduce L3's archived values within this many nats
REPRO_NATS = 1e-4
#: G5 / G2: more than this many failing states STOPS the read
MAX_BAD = 2
#: the read claims a verdict only with at least this many states holding a decoy
MIN_STATES = 60
CHANCE, MATERIAL, BAND = 0.50, 0.55, 0.005


def above(x: Optional[float], c: float) -> bool:
    return x is not None and x > c + BAND


def below(x: Optional[float], c: float) -> bool:
    return x is not None and x < c - BAND


def pair_auc(a: float, bs) -> float:
    """The share of the values ``bs`` that ``a`` beats (ties within ``R.TIE`` count ½): ``R.auc_state``."""
    return R.auc_state(a, list(bs))


def td_verdict(lohi) -> str:
    """PREREGISTRATION §5 (a): the truth vs its decoys."""
    lo, hi = lohi
    if lo is None or hi is None:
        return "NO-DATA"
    if above(lo, CHANCE):
        return "TRUTH-ABOVE"
    if below(hi, CHANCE):
        return "DECOY-ABOVE"
    if below(hi, MATERIAL):
        return "TIED"
    return "INCONCLUSIVE"


def dh_verdict(lohi) -> str:
    """PREREGISTRATION §5 (b): the decoys vs the honest worlds."""
    lo, hi = lohi
    if lo is None or hi is None:
        return "NO-DATA"
    if above(lo, CHANCE):
        return "POOL-LIFT"
    if below(hi, MATERIAL):
        return "NO-POOL-LIFT"
    return "INCONCLUSIVE"


def combined(td: str, dh: str, th_lohi, n: int) -> str:
    """PREREGISTRATION §5 (c), in this order."""
    if n < MIN_STATES:
        return "UNDERPOWERED"
    if td == "TRUTH-ABOVE":
        return "TRUTH-PLUS-POOL" if dh == "POOL-LIFT" else ("TRUTH-SPECIFIC" if dh == "NO-POOL-LIFT"
                                                             else "TRUTH-ABOVE-POOL-UNCLEAR")
    if td in ("TIED", "DECOY-ABOVE"):
        if dh == "POOL-LIFT":
            return "FAMILIARITY"
        if not above(th_lohi[0], CHANCE):
            return "NO-SEPARATION-ON-SUBSET"
        return "INCONSISTENT"
    return "NOT-DETECTED"


def read(rows: List[dict], recheck: Optional[List[dict]], l3_rows: Optional[List[dict]],
         ids_sha256: Optional[str] = IDS_SHA256) -> dict:
    ids = sorted(r["id"] for r in rows)
    g: Dict[str, object] = {"n_rows": len(rows)}
    g["ids_sha256"] = hashlib.sha256("".join(ids).encode()).hexdigest()
    g["G1_ids"] = ids_sha256 is None or g["ids_sha256"] == ids_sha256
    defects = [r for r in rows if r.get("defect")]
    excl = [r for r in rows if not r.get("ok") and not r.get("defect")]
    ok = [r for r in rows if r.get("ok")]
    g["defects"] = [{"id": r["id"], "error": str(r.get("error"))[:300]} for r in defects]
    g["G2_defects"] = len(defects) <= MAX_BAD
    g["excluded"] = len(excl)
    if recheck is not None:
        by = {r["id"]: r for r in ok}
        same = [r["id"] for r in recheck if r.get("ok") and r["id"] in by and r["ll"] == by[r["id"]]["ll"]
                and r["conf"] == by[r["id"]]["conf"]]
        g["G3_determinism"] = {"rechecked": len(recheck), "identical": len(same)}
        g["G3"] = len(recheck) > 0 and len(same) == len(recheck)
    else:
        g["G3"] = None
    if l3_rows is not None:
        l3 = {r["id"]: r for r in l3_rows if r.get("ok")}
        bad, diffs, missing = [], [], 0
        for r in ok:
            o = l3.get(r["id"])
            if o is None:
                missing += 1
                continue
            d = max(float(np.max(np.abs(np.asarray(r["ll"][:R.J + 1]) - np.asarray(o["ll"])))),
                    float(np.max(np.abs(np.asarray(r["conf"][:R.J + 1]) - np.asarray(o["conf"])))))
            diffs.append(d)
            if d > REPRO_NATS:
                bad.append(r["id"])
        g["G5_reproduction"] = {"compared": len(diffs), "not_in_l3_ok": missing, "failing": bad[:20],
                                "n_failing": len(bad), "max_abs_nats": max(diffs) if diffs else None,
                                "l3_ok_not_ok_here": len(set(l3) - {r["id"] for r in ok})}
        g["G5"] = len(bad) + missing <= MAX_BAD
    else:
        g["G5"] = None
    stop = not (g["G1_ids"] and g["G2_defects"] and g["G3"] is not False and g["G5"] is not False)
    out: Dict[str, object] = {"gates": g, "stopped": stop}
    if stop or not ok:
        return out

    dec = [r for r in ok if r.get("n_decoys", 0) >= 1]
    for r in dec:
        if len(r["ll"]) != R.J + 1 + r["n_decoys"]:
            raise SystemExit(f"{r['id']}: {len(r['ll'])} likelihoods for the truth + {R.J} + {r['n_decoys']}")
    n_dec_states = len(dec)
    out["counts"] = {
        "ok": len(ok), "excluded": len(excl), "defects": len(defects), "with_decoy": n_dec_states,
        "decoys_total": int(sum(r["n_decoys"] for r in dec)),
        "with_decoy_by_stratum": {s: sum(1 for r in dec if r["stratum"] == s) for s in sorted({r["stratum"] for r in ok})},
        "ok_by_stratum": {s: sum(1 for r in ok if r["stratum"] == s) for s in sorted({r["stratum"] for r in ok})},
        "no_decoy_reasons": {
            "no_eligible_candidate": sum(1 for r in ok if r.get("n_decoys", 0) == 0 and r["decoy"]["eligible"] == 0),
            "none_passed": sum(1 for r in ok if r.get("n_decoys", 0) == 0 and r["decoy"]["eligible"] > 0)},
    }
    if not dec:
        out["verdict"] = "UNDERPOWERED"
        return out

    def per_state(r, key="ll"):
        v = r[key]
        t, h, d = v[0], v[1:R.J + 1], v[R.J + 1:]
        return (float(np.mean([pair_auc(t, [x]) for x in d])),                       # truth vs decoys
                float(np.mean([pair_auc(x, h) for x in d])),                         # decoys vs honest
                pair_auc(t, h))                                                      # truth vs honest (L3's A_s)

    A = np.array([per_state(r) for r in dec])
    C = np.array([per_state(r, "conf") for r in dec])
    A_all = np.array([pair_auc(r["ll"][0], r["ll"][1:R.J + 1]) for r in ok])
    shared = np.array([float(np.mean(r["decoy"]["shared_sets"])) for r in dec])
    strata = np.array([r["stratum"] for r in dec])
    bt = R.Boot([r["battle"] for r in dec])

    def share_ci(sel=None):
        """(AUC_dh − ½) / (AUC_th − ½), per bootstrap draw (descriptive)."""
        _, vd, _ = bt.mean(A[:, 1], sel)
        e_t, vt, _ = bt.mean(A[:, 2], sel)
        e_d, _, _ = bt.mean(A[:, 1], sel)
        with np.errstate(invalid="ignore", divide="ignore"):
            vec = (vd - 0.5) / (vt - 0.5)
            est = (e_d - 0.5) / (e_t - 0.5) if e_t != 0.5 else float("nan")
        return R.summ(est, vec, int(np.sum(sel)) if sel is not None else len(dec))

    def block(sel=None) -> dict:
        res: Dict[str, object] = {}
        for i, name in enumerate(("AUC_truth_vs_decoy", "AUC_decoy_vs_honest", "AUC_truth_vs_honest")):
            res[name] = R.summ(*bt.mean(A[:, i], sel))
        for i, name in enumerate(("AUC_conf_truth_vs_decoy", "AUC_conf_decoy_vs_honest", "AUC_conf_truth_vs_honest")):
            res[name] = R.summ(*bt.mean(C[:, i], sel))
        td = td_verdict(res["AUC_truth_vs_decoy"]["ci"])
        dh = dh_verdict(res["AUC_decoy_vs_honest"]["ci"])
        n = res["AUC_truth_vs_decoy"]["n"]
        res["verdict_truth_vs_decoy"] = td + ("-MATERIAL" if td == "TRUTH-ABOVE"
                                              and above(res["AUC_truth_vs_decoy"]["ci"][0], MATERIAL) else "")
        res["verdict_decoy_vs_honest"] = dh
        res["VERDICT"] = combined(td, dh, res["AUC_truth_vs_honest"]["ci"], n)
        res["pool_share_of_L3"] = share_ci(sel)
        return res

    out["pooled"] = block()
    out["by_stratum"] = {s: block(strata == s) for s in sorted(set(strata))}
    out["by_shared_sets"] = {"mean<2": R.summ(*bt.mean(A[:, 0], shared < 2)),
                             "mean>=2": R.summ(*bt.mean(A[:, 0], shared >= 2))}
    bt_all = R.Boot([r["battle"] for r in ok])
    out["AUC_truth_vs_honest_all_ok"] = R.summ(*bt_all.mean(A_all))
    out["verdict"] = out["pooled"]["VERDICT"]               # type: ignore[index]
    return out


def selfcheck() -> None:
    assert td_verdict([0.51, 0.7]) == "TRUTH-ABOVE" and td_verdict([0.504, 0.6]) == "INCONCLUSIVE"
    assert td_verdict([0.45, 0.544]) == "TIED" and td_verdict([0.45, 0.546]) == "INCONCLUSIVE"
    assert td_verdict([0.3, 0.494]) == "DECOY-ABOVE"
    assert dh_verdict([0.506, 0.6]) == "POOL-LIFT" and dh_verdict([0.45, 0.54]) == "NO-POOL-LIFT"
    assert combined("TIED", "POOL-LIFT", [0.6, 0.7], 100) == "FAMILIARITY"
    assert combined("TRUTH-ABOVE", "NO-POOL-LIFT", [0.6, 0.7], 100) == "TRUTH-SPECIFIC"
    assert combined("TRUTH-ABOVE", "POOL-LIFT", [0.6, 0.7], 100) == "TRUTH-PLUS-POOL"
    assert combined("TRUTH-ABOVE", "POOL-LIFT", [0.6, 0.7], 59) == "UNDERPOWERED"
    rng = np.random.default_rng(3)

    def fake(t_shift: float, d_shift: float, n: int = 287) -> List[dict]:
        rows = []
        for i in range(n):
            h = rng.normal(0.0, 1.0, R.J).tolist()
            nd = int(rng.integers(0, 5))
            d = rng.normal(d_shift, 1.0, nd).tolist()
            t = float(rng.normal(t_shift, 1.0))
            rows.append({"id": f"s{i}", "battle": f"b{i // 3}", "stratum": "bot" if i % 2 else "snapshot", "ok": True,
                         "ll": [t] + h + d, "conf": [0.0] * (R.J + 1 + nd), "n_decoys": nd,
                         "decoy": {"eligible": nd, "shared_sets": [1] * nd or [0]}})
        return rows

    fam = read(fake(1.0, 1.0), None, None, None)
    assert fam["verdict"] == "FAMILIARITY", fam["pooled"]
    spec = read(fake(1.0, 0.0), None, None, None)
    assert spec["verdict"] == "TRUTH-SPECIFIC", spec["pooled"]
    mix = read(fake(1.5, 0.7), None, None, None)
    assert mix["verdict"] == "TRUTH-PLUS-POOL", mix["pooled"]
    print("[l3 decoy read] selfcheck PASS")


def load(path: str) -> List[dict]:
    return R.load(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows")
    ap.add_argument("--recheck")
    ap.add_argument("--l3-rows")
    ap.add_argument("--out")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        selfcheck()
        return
    res = read(load(a.rows), load(a.recheck) if a.recheck else None, load(a.l3_rows) if a.l3_rows else None)
    text = json.dumps(res, indent=1, sort_keys=True)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
    print(text)
    if res.get("stopped"):
        sys.exit(3)


if __name__ == "__main__":
    main()
