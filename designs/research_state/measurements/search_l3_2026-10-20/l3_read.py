"""L3, THE REGISTERED READ (``PREREGISTRATION.md`` §4–§6): does the TRUE world out-rank the honest worlds under the
likelihood of the opponent's own observed choices? Pure numpy over the driver's rows (``l3_run.py``); no model, no core.

    python l3_read.py --rows <archive>/rows.jsonl [--recheck <archive>/recheck.jsonl] --out result_l3.json
    python l3_read.py --selfcheck          # the rules on synthetic states (no data)

Every number is computed on the SAME battle-clustered bootstrap draws (2,000, ``default_rng(20261020)``, percentile 95 %
intervals, states weighted equally). RULE 8: a bound within ±0.005 of an AUC boundary (0.50, 0.55) is not clear.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from typing import Dict, List, Optional, Sequence

import numpy as np

B = 2000
SEED = 20261020
#: two log-likelihoods within this many nats are a TIE (rule 8: exclude the rounding band, never break it by chance)
TIE = 1e-6
#: the AUC boundaries and the rule-8 band around each
CHANCE, MATERIAL, BAND = 0.50, 0.55, 0.005
J = 8
N_STATES = 287
#: the defect budget: more than this many defect rows (a guard raised) STOPS the read
MAX_DEFECTS = 2
#: the informative-decision bins of S5 (the truth's scored steps with ≥ 2 legal actions)
BINS = ((0, 0), (1, 4), (5, 9), (10, 10 ** 6))


# -------------------------------------------------------------------------------------------- statistics

class Boot:
    """Battles resampled with replacement; a mean over a selection is a ratio of per-draw sums (states equal)."""

    def __init__(self, battles: Sequence[str]):
        self.ub = sorted(set(battles))
        ix = {b: i for i, b in enumerate(self.ub)}
        self.bi = np.array([ix[b] for b in battles], dtype=np.int64)
        nb = len(self.ub)
        rng = np.random.default_rng(SEED)
        self.C = np.zeros((B, nb))
        draws = rng.integers(0, nb, size=(B, nb))
        for r in range(B):
            np.add.at(self.C[r], draws[r], 1.0)
        self.nb = nb

    def mean(self, x, sel=None):
        x = np.asarray(x, dtype=np.float64)
        m = np.ones(len(x), bool) if sel is None else np.asarray(sel, bool).copy()
        m &= np.isfinite(x)
        S = np.bincount(self.bi[m], weights=x[m], minlength=self.nb)
        N = np.bincount(self.bi[m], minlength=self.nb).astype(np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            vec = (self.C @ S) / (self.C @ N)
        est = float(S.sum() / N.sum()) if N.sum() > 0 else float("nan")
        return est, vec, int(m.sum())


def ci(vec) -> List[Optional[float]]:
    v = np.asarray(vec, dtype=np.float64)
    v = v[np.isfinite(v)]
    if len(v) < 20:
        return [None, None]
    lo, hi = np.percentile(v, [2.5, 97.5])
    return [float(lo), float(hi)]


def summ(est, vec, n) -> dict:
    lo, hi = ci(vec)
    r = (lambda x: None if x is None or not np.isfinite(x) else round(float(x), 4))
    return {"mean": r(est), "ci": [r(lo), r(hi)], "n": int(n)}


def above(x: Optional[float], c: float) -> bool:
    return x is not None and x > c + BAND


def below(x: Optional[float], c: float) -> bool:
    return x is not None and x < c - BAND


def verdict(lohi: Sequence[Optional[float]]) -> str:
    """PREREGISTRATION §5, on the AUC's 95 % interval [lo, hi]."""
    lo, hi = lohi
    if lo is None or hi is None:
        return "NO-DATA"
    if above(lo, MATERIAL):
        return "WORKS"
    if above(lo, CHANCE):
        return "WEAK"
    if below(hi, CHANCE):
        return "REVERSED"
    if below(hi, MATERIAL):
        return "NO-EVIDENCE"
    return "NOT-DETECTED"


# -------------------------------------------------------------------------------------------- per-state quantities

def auc_state(t: float, ws: Sequence[float]) -> float:
    """The share of the honest worlds the truth beats (a tie within ``TIE`` counts ½)."""
    d = t - np.asarray(ws, dtype=np.float64)
    return float(np.mean(np.where(d > TIE, 1.0, np.where(d < -TIE, 0.0, 0.5))))


def top1_state(t: float, ws: Sequence[float]) -> float:
    """1 when the truth beats EVERY honest world clearly, else 0."""
    return float(np.all(t - np.asarray(ws, dtype=np.float64) > TIE))


def weight_of_truth(t: float, ws: Sequence[float]) -> float:
    """The truth's self-normalised likelihood weight among {truth} ∪ the honest worlds (uniform: 1 / (J + 1))."""
    v = np.concatenate([[t], np.asarray(ws, dtype=np.float64)])
    e = np.exp(v - v.max())
    return float(e[0] / e.sum())


def lift(ll: Sequence[float], x: Sequence[Optional[float]]) -> float:
    """Σ_j w_j x_j − mean_j x_j over the honest worlds whose x is defined, w = softmax(ll) over those worlds."""
    ok = [j for j, v in enumerate(x) if v is not None]
    if not ok:
        return float("nan")
    v = np.asarray([ll[j] for j in ok], dtype=np.float64)
    xv = np.asarray([x[j] for j in ok], dtype=np.float64)
    w = np.exp(v - v.max())
    w /= w.sum()
    return float(w @ xv - xv.mean())


def informative(t: float, ws: Sequence[float]) -> bool:
    return bool(np.any(np.abs(t - np.asarray(ws, dtype=np.float64)) > TIE))


def spearman(x, y) -> Optional[float]:
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    if len(x) < 3:
        return None

    def rank(a):
        o = np.argsort(a, kind="mergesort")
        r = np.empty(len(a))
        r[o] = np.arange(len(a))
        for v in np.unique(a):                        # ties: the average rank
            m = a == v
            r[m] = r[m].mean()
        return r

    rx, ry = rank(x), rank(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return round(float(np.corrcoef(rx, ry)[0, 1]), 4)


# -------------------------------------------------------------------------------------------- the read

def load(path: str) -> List[dict]:
    out: Dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r["id"]] = r                      # the LAST row per id (a resumed run rewrites none)
    return list(out.values())


def read(rows: List[dict], recheck: Optional[List[dict]] = None, ids_sha256: Optional[str] = None) -> dict:
    ids = sorted(r["id"] for r in rows)
    gates: Dict[str, object] = {"n_rows": len(rows)}
    sha = hashlib.sha256("".join(ids).encode()).hexdigest()
    gates["ids_sha256"] = sha
    gates["G1_ids"] = ids_sha256 is None or sha == ids_sha256
    defects = [r for r in rows if r.get("defect")]
    excl = [r for r in rows if not r.get("ok") and not r.get("defect")]
    ok = [r for r in rows if r.get("ok")]
    gates["defects"] = [{"id": r["id"], "error": str(r.get("error"))[:300]} for r in defects]
    gates["G2_defects"] = len(defects) <= MAX_DEFECTS
    gates["excluded"] = len(excl)
    gates["exclusion_reasons"] = sorted({str(r.get("error"))[:80] for r in excl})[:20]
    if recheck is not None:
        by = {r["id"]: r for r in ok}
        same = [r["id"] for r in recheck if r.get("ok") and r["id"] in by and r["ll"] == by[r["id"]]["ll"]
                and r["conf"] == by[r["id"]]["conf"]]
        gates["G3_determinism"] = {"rechecked": len(recheck), "identical": len(same)}
        gates["G3"] = len(recheck) > 0 and len(same) == len(recheck)
    else:
        gates["G3"] = None
    for r in ok:
        if len(r["ll"]) != J + 1 or len(r["conf"]) != J + 1:
            raise SystemExit(f"{r['id']}: {len(r['ll'])} likelihoods, expected the truth + {J} worlds")
    enc = [r for r in ok if r.get("enc_pin_equal") is not None]
    gates["G4_encoder_identity"] = {"checked": len(enc), "equal": sum(1 for r in enc if r["enc_pin_equal"])}
    stop = not (gates["G1_ids"] and gates["G2_defects"] and gates["G3"] is not False)
    out: Dict[str, object] = {"gates": gates, "stopped": stop}
    if stop or not ok:
        return out

    bt = Boot([r["battle"] for r in ok])
    strata = np.array([r["stratum"] for r in ok])
    T = np.array([r["ll"][0] for r in ok])
    Wd = [r["ll"][1:] for r in ok]
    Ct = np.array([r["conf"][0] for r in ok])
    Cw = [r["conf"][1:] for r in ok]
    auc = np.array([auc_state(t, w) for t, w in zip(T, Wd)])
    auc_c = np.array([auc_state(t, w) for t, w in zip(Ct, Cw)])
    auc_r = np.array([auc_state(t - c, np.asarray(w) - np.asarray(cw)) for t, w, c, cw in zip(T, Wd, Ct, Cw)])
    top1 = np.array([top1_state(t, w) for t, w in zip(T, Wd)])
    pt = np.array([weight_of_truth(t, w) for t, w in zip(T, Wd)])
    inf = np.array([informative(t, w) for t, w in zip(T, Wd)], dtype=np.float64)
    lift_m = np.array([lift(r["ll"][1:], r["match_m"]) for r in ok])
    lift_u = np.array([lift(r["ll"][1:], r["hit_u"]) for r in ok])
    lift_m_exact = np.array([lift(r["ll"][1:], r["exact_m"]) for r in ok])
    n_inf = np.array([r["n_informative"][0] for r in ok])

    def block(sel=None) -> dict:
        e, v, n = bt.mean(auc, sel)
        res = {"AUC": dict(summ(e, v, n), verdict=verdict(ci(v)))}
        for name, x in (("AUC_conf", auc_c), ("AUC_regret", auc_r), ("top1", top1), ("p_truth", pt),
                        ("informative_share", inf), ("lift_moves", lift_m), ("lift_moves_exact", lift_m_exact),
                        ("lift_species", lift_u)):
            res[name] = summ(*bt.mean(x, sel))
        res["p_truth"]["uniform"] = round(1.0 / (J + 1), 4)
        res["top1"]["chance_if_exchangeable"] = round(1.0 / (J + 1), 4)
        return res

    out["pooled"] = block()
    out["by_stratum"] = {s: block(strata == s) for s in sorted(set(strata))}
    out["by_informative_decisions"] = {f"{lo}-{hi if hi < 10 ** 6 else 'inf'}": summ(*bt.mean(auc, (n_inf >= lo) & (n_inf <= hi)))
                                       for lo, hi in BINS}
    out["spearman_auc_vs_n_informative"] = spearman(n_inf, auc)
    out["counts"] = {
        "ok": len(ok), "excluded": len(excl), "defects": len(defects),
        "steps_scored_mean": round(float(np.mean([r["n_scored"] for r in ok])), 2),
        "steps_informative_mean": round(float(np.mean(n_inf)), 2),
        "steps_unobserved_total": int(sum(r["n_unobserved"] for r in ok)),
        "steps_pending_total": int(sum(r["n_pending"] for r in ok)),
        "states_with_pending": int(sum(1 for r in ok if r["n_pending"])),
        "by_stratum": {s: int((strata == s).sum()) for s in sorted(set(strata))},
    }
    eq = np.array([bool(r.get("enc_pin_equal")) for r in ok])
    if eq.any() and not eq.all():
        e, v, n = bt.mean(auc, eq)
        out["robustness_encoder_identical_states"] = dict(summ(e, v, n), verdict=verdict(ci(v)))
    out["verdict"] = out["pooled"]["AUC"]["verdict"]          # type: ignore[index]
    return out


# -------------------------------------------------------------------------------------------- the selfcheck

def selfcheck() -> None:
    """The rules on synthetic states: chance worlds read near 0.5; a truth always ahead reads WORKS; ties count ½."""
    assert auc_state(0.0, [0.0] * J) == 0.5
    assert auc_state(1.0, [0.0] * 4 + [2.0] * 4) == 0.5
    assert auc_state(0.0, [1e-7] * J) == 0.5                 # inside the tie band
    assert top1_state(1.0, [0.0] * J) == 1.0 and top1_state(1.0, [1.0] + [0.0] * 7) == 0.0
    assert abs(weight_of_truth(0.0, [0.0] * J) - 1 / 9) < 1e-12
    assert abs(lift([0.0] * J, [1.0, 0.0] * 4)) < 1e-12 and lift([5.0] + [0.0] * 7, [1.0] + [0.0] * 7) > 0.8
    assert verdict([0.56, 0.7]) == "WORKS" and verdict([0.553, 0.7]) == "WEAK" and verdict([0.506, 0.6]) == "WEAK"
    assert verdict([0.40, 0.494]) == "REVERSED" and verdict([0.47, 0.54]) == "NO-EVIDENCE"
    assert verdict([0.48, 0.546]) == "NOT-DETECTED" and verdict([0.504, 0.53]) == "NO-EVIDENCE"
    rng = np.random.default_rng(1)

    def fake(shift: float, n: int = 300) -> List[dict]:
        rows = []
        for i in range(n):
            ws = rng.normal(0.0, 1.0, J).tolist()
            rows.append({"id": f"s{i}", "battle": f"b{i // 3}", "stratum": "bot" if i % 2 else "snapshot", "ok": True,
                         "ll": [float(rng.normal(shift, 1.0))] + ws, "conf": [0.0] * (J + 1),
                         "match_m": [0.5] * J, "exact_m": [0.0] * J, "hit_u": [None] * J,
                         "n_informative": [5] * (J + 1), "n_scored": 5, "n_unobserved": 0, "n_pending": 0})
        return rows

    null = read(fake(0.0))
    assert null["verdict"] in ("NO-EVIDENCE", "NOT-DETECTED"), null["pooled"]
    strong = read(fake(1.5))
    assert strong["verdict"] == "WORKS", strong["pooled"]
    rev = read(fake(-1.5))
    assert rev["verdict"] == "REVERSED", rev["pooled"]
    bad = fake(0.0, 30)
    bad[0]["defect"] = True
    bad[1]["defect"] = True
    bad[2]["defect"] = True
    assert read(bad)["stopped"]
    print("[l3 read] selfcheck PASS")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows")
    ap.add_argument("--recheck")
    ap.add_argument("--ids-sha256")
    ap.add_argument("--out")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        selfcheck()
        return
    rows = load(a.rows)
    rec = load(a.recheck) if a.recheck else None
    res = read(rows, rec, a.ids_sha256)
    text = json.dumps(res, indent=1, sort_keys=True)
    if a.out:
        with open(a.out, "w") as f:
            f.write(text + "\n")
    print(text)
    if res.get("stopped"):
        sys.exit(3)


if __name__ == "__main__":
    main()
