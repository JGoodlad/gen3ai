"""STAGE 1c, THE REGISTERED READ (``PREREGISTRATION.md`` §4–§6, §8). Two parts, each read ONCE:

    python read_1c.py a [--interim] --out result_1c_a.json   # PHASE A: the worlds' nearness to the truth, the bar
    python read_1c.py b --out result_1c_b.json               # PHASE B: the playouts (only when the bar was met)
    python read_1c.py selfcheck                              # the rules on synthetic states (no data)

PHASE A is pure numpy over the driver's rows (and the module's own pure resample functions, for gate G5). PHASE B reads
the pilot's playout rows with Tier 1's array helpers and L1's acting rules (imported, never copied). Every interval is a
battle-clustered percentile bootstrap (2,000 draws, ``default_rng(20261020)``, states weighted equally), every contrast
PAIRED on the same draws. RULE 8: a bound within ±0.0005 of its boundary is not clear; a zero edge is written
"> +0.0005"; "inside ±m" means lower > −m + 0.0005 and upper < m − 0.0005.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c1c_common as K  # noqa: E402

B = 2000
SEED = 20261020
EDGE = 0.0005
NULL_BAND = 0.01
#: MATERIAL: the share of the binding hidden facts held rises by at least this much (10x L3's reweighting lift)
MATERIAL_H = 0.05
#: DOSE: the posterior mass on worlds exact on both binding classes reaches half the dose curve's p* (0.2)
DOSE_Q = 0.10
MAX_DEFECTS = 2
#: the meters (PREREGISTRATION §4): name -> (numerator, denominator) of the per-world facts, or a 0/1 fact
METERS = ("H", "M_hit", "M_exact", "U_hit", "U_exact", "qSM", "item", "spread", "U_moves", "near4")
#: L1's fitted anchors (``../search_l1_l5_2026-10-20/fit_l1.json``), re-checked at the read
L1_FIT = {"PIKL": {"LEAF-K8": 0.1, "TRUNC-4-K8": 0.015, "CRITIC": 0.015},
          "GUMBEL": {"LEAF-K8": 0.1, "TRUNC-4-K8": 0.2, "CRITIC": 0.05},
          "THRESH": {"LEAF-K8": 5 / 32, "TRUNC-4-K8": 0.02, "CRITIC": 0.02}}
PRIMARY_RULE = "PIKL"
SCORERS = ("LEAF-K8", "TRUNC-4-K8", "CRITIC")
PRIMARY_SCORERS = ("LEAF-K8", "TRUNC-4-K8")


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
    r = (lambda x: None if x is None or not np.isfinite(x) else round(float(x), 5))
    return {"mean": r(est), "ci": [r(lo), r(hi)], "n": int(n)}


def clear_above(lo: Optional[float], c: float = 0.0) -> bool:
    """lower bound clear above c (a zero edge carries the margin: > +0.0005; any other c: > c + 0.0005)."""
    return lo is not None and lo > c + EDGE


def clear_below(hi: Optional[float], c: float = 0.0) -> bool:
    return hi is not None and hi < c - EDGE


def inside(lo: Optional[float], hi: Optional[float], m: float) -> bool:
    return lo is not None and hi is not None and lo > -m + EDGE and hi < m - EDGE


def verdict_a(s: dict) -> dict:
    """PREREGISTRATION §5 on one primary arm's ΔH interval (+ its q_SM level's interval)."""
    lo, hi = s["dH"]["ci"]
    if clear_above(lo):
        v = "NEARER"
    elif clear_below(hi):
        v = "FARTHER"
    elif inside(lo, hi, NULL_BAND):
        v = "NULL"
    else:
        v = "NOT-DETECTED"
    return {"verdict": v, "material": bool(v == "NEARER" and clear_above(lo, MATERIAL_H)),
            "dose": bool(clear_above(s["qSM"]["ci"][0], DOSE_Q))}


def verdict_b(imp: Dict[str, dict], dif: Dict[str, dict]) -> str:
    """PREREGISTRATION §6 on one played primary arm (the anchored rule's IMP, and its difference from F8)."""
    los = [imp[z]["ci"][0] for z in PRIMARY_SCORERS]
    his = [imp[z]["ci"][1] for z in PRIMARY_SCORERS]
    if all(clear_above(x) for x in los):
        return "SEARCH-PAYS"
    if all(clear_below(x) for x in his):
        return "HURTS"
    lo, hi = imp["LEAF-K8"]["ci"]
    if inside(lo, hi, NULL_BAND):
        return "NULL"
    return "NOT-DETECTED"


# -------------------------------------------------------------------------------------------- per-state meters

def facts_values(f: dict) -> Dict[str, float]:
    """One world's meter values from its fact counts (``c1c_common.world_facts``); NaN where the state has none."""
    nan = float("nan")
    mt, ut = f["mt"], f["ut"]
    return {"H": (f["mh"] + f["uh"]) / (mt + ut) if mt + ut else nan,
            "M_hit": f["mh"] / mt if mt else nan, "M_exact": float(f["me"]) if mt else nan,
            "U_hit": f["uh"] / ut if ut else nan, "U_exact": float(f["ue"]) if ut else nan,
            "qSM": float(f["ue"] * f["me"]),
            "item": f["ih"] / f["it"] if f["it"] else nan, "spread": f["sh"] / f["st"] if f["st"] else nan,
            "U_moves": f["umh"] / ut if ut else nan}


def arm_parts(row: dict, arm: str, temper: float = 1.0) -> Optional[Tuple[np.ndarray, List[dict], List[int]]]:
    """(the arm's weights over its proposals, their facts, their near-copy flags), or None if the arm is not valid.
    ``temper`` scales the opponent-choice log-likelihood (1: the registered weight; ``K.TEMPER``: the descriptive)."""
    from main.value_search.opp_likelihood import normalise, posterior_log_weights

    parts, wt = K.ARMS[arm]
    sets = [s for s, _ in parts]
    if not row.get("ok") or row.get("defect") or any(not (row.get(s) or {}).get("ok") for s in sets):
        return None
    if any("facts" not in row[s] for s in sets):
        return None
    facts = K.arm_slice({s: row[s]["facts"] for s in sets}, arm)
    near = K.arm_slice({s: row[s]["near4"] for s in sets}, arm)
    n = len(facts)
    if wt == "uniform":
        return np.full(n, 1.0 / n), facts, near
    dice = K.arm_slice({s: row[s]["log_w"] for s in sets}, arm)
    ll = [temper * x for x in K.arm_slice({s: row[s]["ll"] for s in sets}, arm)] if wt == "dice*opp" else None
    return normalise(posterior_log_weights(n, dice, ll)), facts, near


def arm_meters(row: dict, arm: str, temper: float = 1.0) -> Optional[Dict[str, float]]:
    p = arm_parts(row, arm, temper)
    if p is None:
        return None
    w, facts, near = p
    vals = [facts_values(f) for f in facts]
    out = {}
    for m in METERS:
        x = np.array([float(v) for v in near], dtype=np.float64) if m == "near4" else \
            np.array([v[m] for v in vals], dtype=np.float64)
        out[m] = float(w @ x) if np.all(np.isfinite(x)) else float("nan")
    # the posterior mass on the NEAREST proposals (max H in the set) minus their uniform share
    h = np.array([v["H"] for v in vals], dtype=np.float64)
    if np.all(np.isfinite(h)):
        top = h >= h.max() - 1e-12
        out["NW_lift"] = float(w[top].sum() - top.mean())
    else:
        out["NW_lift"] = float("nan")
    out["ESS"] = float(w.sum() ** 2 / np.sum(w ** 2))
    out["ESS_share"] = out["ESS"] / len(w)
    return out


# -------------------------------------------------------------------------------------------- phase A

def gates_a(rows: List[dict], recheck: Optional[List[dict]], l3: Optional[Dict[str, dict]],
            order: Optional[List[str]]) -> dict:
    from main.value_search.opp_likelihood import normalise, posterior_log_weights, resample_uniform, systematic

    g: Dict[str, object] = {"n_rows": len(rows)}
    ids = [r["id"] for r in rows]
    if order is not None:
        g["G1_prefix"] = ids == order[:len(ids)] and len(set(ids)) == len(ids)
    else:
        g["G1_prefix"] = len(set(ids)) == len(ids)
    g["ids_sha256"] = K.ids_sha(sorted(ids))
    bad = [r for r in rows if r.get("defect") or r.get("parked")]
    g["defects"] = [{"id": r["id"], "parked": bool(r.get("parked")), "error": str(r.get("error") or
                     (r.get("F") or {}).get("error") or (r.get("C") or {}).get("error"))[:300]} for r in bad]
    g["G2_defects"] = len(bad) <= MAX_DEFECTS
    g["excluded"] = {s: sum(1 for r in rows if not r.get("defect") and not r.get("parked")
                            and not (r.get(s) or {}).get("ok")) for s in ("F", "C")}
    g["exclusion_reasons"] = sorted({str((r.get(s) or {}).get("error"))[:90] for r in rows for s in ("F", "C")
                                     if (r.get(s) or {}).get("error") and not r.get("defect")})[:20]
    if recheck is not None:
        by = {r["id"]: r for r in rows}
        keys = ("ll_truth", "resample")
        same = 0
        for r in recheck:
            m = by.get(r["id"])
            if m is None:
                continue
            ok = all(json.dumps(r.get(k), sort_keys=True) == json.dumps(m.get(k), sort_keys=True) for k in keys)
            for s in ("F", "C"):
                for k in ("ok", "log_w", "ll", "facts", "near4"):
                    ok = ok and json.dumps((r.get(s) or {}).get(k), sort_keys=True) == \
                        json.dumps((m.get(s) or {}).get(k), sort_keys=True)
            same += int(ok)
        g["G3_determinism"] = {"rechecked": len(recheck), "identical": same}
        g["G3"] = len(recheck) > 0 and same == len(recheck)
    else:
        g["G3"] = None
    # G4 (descriptive): the truth's likelihood equals L3's on the common states
    if l3 is not None:
        both = [r for r in rows if r.get("ok") and "ll_truth" in r and (l3.get(r["id"]) or {}).get("ok")]
        g["G4_l3_truth_ll"] = {"compared": len(both),
                               "equal": sum(1 for r in both if abs(r["ll_truth"] - l3[r["id"]]["ll"][0]) <= 1e-9)}
    # G5: every resample index recomputed from the stored weights by the module's own functions
    n5 = bad5 = 0
    for r in rows:
        for arm, rec in (r.get("resample") or {}).items():
            parts, wt = K.ARMS[arm]
            sets = [s for s, _ in parts]
            dice = K.arm_slice({s: r[s]["log_w"] for s in sets}, arm)
            ll = K.arm_slice({s: r[s]["ll"] for s in sets}, arm) if wt == "dice*opp" else None
            idx = systematic(normalise(posterior_log_weights(len(dice), dice, ll)), K.J,
                             resample_uniform(K.RESAMPLE_KEY.format(arm=arm, id=r["id"])))
            n5 += 1
            bad5 += int(idx != rec["idx"])
        for s, nn in (("F", K.N_F), ("C", K.N_C)):
            x = r.get(s) or {}
            if x.get("ok") and (len(x.get("log_w", [])) != nn or not all(math.isfinite(v) for v in x["log_w"])):
                bad5 += 1
    g["G5_resample"] = {"checked": n5, "mismatch": bad5}
    g["G5"] = bad5 == 0
    return g


def read_a(rows: List[dict], recheck: Optional[List[dict]] = None, l3: Optional[Dict[str, dict]] = None,
           order: Optional[List[str]] = None, interim: bool = False) -> dict:
    g = gates_a(rows, recheck, l3, order)
    stop = not (g["G1_prefix"] and g["G2_defects"] and g["G3"] is not False and g["G5"])
    out: Dict[str, object] = {"gates": g, "stopped": stop, "interim": interim}
    if stop:
        return out
    ok = [r for r in rows if r.get("ok") and not r.get("defect") and not r.get("parked")]
    if not ok:
        return out
    bt = Boot([r["battle"] for r in ok])
    strata = np.array([str(r.get("stratum")) for r in ok])
    met = {arm: [arm_meters(r, arm) for r in ok] for arm in K.ARMS}
    met_t = {arm: [arm_meters(r, arm, K.TEMPER) for r in ok] for arm in K.PRIMARY}

    def col(arm: str, m: str) -> np.ndarray:
        return np.array([x[m] if x is not None else np.nan for x in met[arm]], dtype=np.float64)

    def colt(arm: str, m: str) -> np.ndarray:
        return np.array([x[m] if x is not None else np.nan for x in met_t[arm]], dtype=np.float64)

    def contrast(a: str, b: str, m: str, sel=None) -> dict:
        return summ(*bt.mean(col(a, m) - col(b, m), sel))

    def level(a: str, m: str, sel=None) -> dict:
        return summ(*bt.mean(col(a, m), sel))

    def block(sel=None) -> dict:
        res: Dict[str, dict] = {"levels": {a: {m: level(a, m, sel) for m in METERS} for a in K.ARMS}}
        res["vs_F8"] = {a: {m: contrast(a, K.BASELINE, m, sel) for m in METERS} for a in K.ARMS if a != K.BASELINE}
        # the likelihood's own selection (same proposals: dice*opp vs dice only) and the proposal effects
        res["likelihood_selection"] = {"F128L-F128d": {m: contrast("F128L", "F128d", m, sel) for m in METERS},
                                       "C128L-C128d": {m: contrast("C128L", "C128d", m, sel) for m in METERS}}
        res["proposals"] = {"F128d-F8": {m: contrast("F128d", "F8", m, sel) for m in METERS},
                            "C8-F8": {m: contrast("C8", "F8", m, sel) for m in METERS},
                            "C128d-C8": {m: contrast("C128d", "C8", m, sel) for m in METERS}}
        res["n_scaling"] = {f"{a}-{b}": {m: contrast(a, b, m, sel) for m in ("H", "M_hit", "U_hit", "qSM")}
                            for a, b in (("F128L", "F64L"), ("C128L", "C64L"), ("M256L", "M128L"))}
        res["tempered_0.5"] = {a: {m: summ(*bt.mean(colt(a, m) - col(K.BASELINE, m), sel)) for m in ("H", "qSM")}
                               for a in K.PRIMARY}
        wa = ("F128L", "C128L", "M256L", "F64L", "C64L", "M128L", "F128d", "C128d")
        res["NW_lift"] = {a: level(a, "NW_lift", sel) for a in wa}
        res["ESS"] = {a: {"ESS": level(a, "ESS", sel), "ESS_share": level(a, "ESS_share", sel),
                          "share_ESS_below_2": summ(*bt.mean((col(a, "ESS") < 2).astype(float)
                                                             * np.where(np.isfinite(col(a, "ESS")), 1, np.nan), sel))}
                      for a in wa}
        return res

    pooled = block()
    out["pooled"] = pooled
    out["by_stratum"] = {s: block(strata == s) for s in sorted(set(strata))}
    def head(a: str) -> dict:
        x = {"dH": pooled["vs_F8"][a]["H"], "qSM": pooled["levels"][a]["qSM"]}
        return x if interim else {**x, **verdict_a(x)}     # an INTERIM read carries no verdict (§7)

    prim = {a: head(a) for a in K.PRIMARY}
    out["primary"] = prim
    out["secondary_64"] = {a: head(a) for a in K.SECONDARY_64}
    if not interim:
        nearer = [a for a in K.PRIMARY if prim[a]["verdict"] == "NEARER"]
        play = []
        if nearer:
            play = ["F8"] + nearer + (["C8"] if any(a in nearer for a in ("C128L", "M256L")) else [])
        out["bar"] = {"met": bool(nearer), "nearer": nearer, "play": play}
    out["counts"] = {
        "ok": len(ok), "by_stratum": {s: int((strata == s).sum()) for s in sorted(set(strata))},
        "valid": {a: int(sum(1 for x in met[a] if x is not None)) for a in K.ARMS},
        "steps_scored_mean": round(float(np.mean([r.get("n_scored", 0) for r in ok])), 2),
        "corpus_lambda_mean": round(float(np.mean([((r.get("C") or {}).get("prop") or {}).get("lambda", 0.0)
                                                   for r in ok if (r.get("C") or {}).get("ok")] or [0.0])), 4),
        "cost_cpu_s_mean": {s: round(float(np.mean([r[s]["cost"]["cpu_s"] for r in ok if (r.get(s) or {}).get("cost")]
                                                   or [0.0])), 1) for s in ("F", "C")},
        "cost_ll_cpu_s_mean": round(float(np.mean([r.get("cost_ll_cpu_s", 0.0) for r in ok])), 1),
    }
    return out


# -------------------------------------------------------------------------------------------- phase B

def read_b(play: Dict[str, Dict[str, dict]], truth: Dict[str, dict], roots: Dict[str, dict], states: Dict[str, dict],
           ids: List[str], fit: Optional[dict] = None, identity: Optional[dict] = None) -> dict:
    """``play``: {arm: {id: the pilot's playout row}}; ``truth`` / ``roots``: the banked 225M O rows and roots."""
    sys.path.insert(0, str(K.L1DIR / "scripts"))
    import l1_read as L1                                # L1's rules (pick), imported unchanged
    import l1l5_data as LD                              # the per-state arrays in the POLICY's order

    if fit is not None:
        for rule, zs in L1_FIT.items():
            for z, x in zs.items():
                if abs(float(fit[rule][z]["fitted"]) - x) > 1e-12:
                    raise SystemExit(f"fit_l1.json's {rule} {z} = {fit[rule][z]['fitted']}, registered {x}")
    arms = [a for a in play]
    if "F8" not in arms:
        raise SystemExit("phase B needs the F8 baseline")
    recs, refused, excluded = [], {a: 0 for a in arms}, {a: 0 for a in arms}
    for did in ids:
        r1, rt = truth.get(did), roots.get(did)
        if r1 is None or not r1.get("ok") or rt is None:
            continue
        tp = LD.truth_part(r1, rt["logits"], rt["top"])
        rec = {"id": did, "battle": states[did]["battle"], "stratum": states[did].get("stratum"), **tp["rec"]}
        for a in arms:
            rw = play[a].get(did)
            if rw is None or not rw.get("ok"):
                if rw is not None:
                    (excluded if rw.get("excluded") and not rw.get("defect") else refused)[a] += 1
                continue
            rec[a] = LD.world_part(rw, tp)
        recs.append(rec)
    stopped = [a for a in arms if refused[a] > MAX_DEFECTS]
    if identity is not None and not (identity.get("identical") == identity.get("states") == K.N_IDENTITY):
        return {"stopped": True, "reason": "the identity tooth did not pass", "identity": identity}
    if "F8" in stopped:
        return {"stopped": True, "reason": "the F8 baseline has more than 2 refusals", "refused": refused}
    bt = Boot([r["battle"] for r in recs])
    strata = np.array([str(r["stratum"]) for r in recs])
    key = {"LEAF-K8": "W", "TRUNC-4-K8": "X4", "CRITIC": "crit"}

    def imp(a: str, z: str, rule: str) -> np.ndarray:
        out = np.full(len(recs), np.nan)
        x = L1_FIT[rule][z] if rule != "ARGMAX" else 0.0
        for i, r in enumerate(recs):
            s = r.get(a)
            if s is None:
                continue
            q = np.asarray(s[key[z]], dtype=np.float64).mean(1)
            out[i] = r["Tp"][L1.pick(rule, x, q, r["logit"])] - r["Tp"][0]
        return out

    res: Dict[str, object] = {"n_states": len(recs), "refused": refused, "excluded": excluded,
                              "arms_stopped": stopped, "identity": identity, "stopped": False}
    tab: Dict[str, dict] = {}
    for rule in ("PIKL", "ARGMAX", "GUMBEL", "THRESH"):
        tab[rule] = {}
        for a in arms:
            tab[rule][a] = {}
            for z in SCORERS:
                v = imp(a, z, rule)
                f8 = imp("F8", z, rule)
                tab[rule][a][z] = {"imp": summ(*bt.mean(v)), "minus_F8": summ(*bt.mean(v - f8)),
                                   "by_stratum": {s: summ(*bt.mean(v, strata == s)) for s in sorted(set(strata))}}
                if a in ("C128L", "M256L") and "C8" in arms:
                    tab[rule][a][z]["minus_C8"] = summ(*bt.mean(v - imp("C8", z, rule)))
    res["table"] = tab
    verd = {}
    for a in arms:
        if a not in K.PRIMARY:
            continue
        if a in stopped:
            verd[a] = {"verdict": "STOPPED (more than 2 refused worlds)", "beats_F8": False}
            continue
        t = tab[PRIMARY_RULE][a]
        v = verdict_b({z: t[z]["imp"] for z in PRIMARY_SCORERS}, {z: t[z]["minus_F8"] for z in PRIMARY_SCORERS})
        beats = all(clear_above(t[z]["minus_F8"]["ci"][0]) for z in PRIMARY_SCORERS)
        verd[a] = {"verdict": v, "beats_F8": beats}
    res["primary"] = verd
    return res


def bcheck(E, order: List[str]) -> dict:
    """PHASE B's PLUMBING CHECK (before any playout of this unit): the BANKED 225M F rows (the maturity battery's,
    already read and published) through ``read_b``'s own arrays and rules must reproduce the registered LEAN table's
    per-state ARGMAX IMP of F exactly, at every scorer. Prints the mismatch count only."""
    import hashlib
    import pickle

    sys.path.insert(0, str(K.L1DIR / "scripts"))
    import l1_read as L1
    import l1l5_data as LD

    tp_ = E.MAT_ARCH / "score" / "table.pkl"
    want = (E.MAT_ARCH / "score" / "table.sha256").read_text().split()[0]
    blob = tp_.read_bytes()
    if hashlib.sha256(blob).hexdigest() != want:
        raise SystemExit("the LEAN table does not match its sha256")
    tab = {r["id"]: r for r in pickle.loads(blob)[E.ANCHOR]["recs"]}
    truth: Dict[str, dict] = {}
    for p in sorted((E.MAT_ARCH / "rows" / E.ANCHOR / "O").glob("shard_*.jsonl")):
        truth.update({r["id"]: r for r in K.read_rows(p)})
    roots = {r["id"]: r for r in K.read_rows(E.MAT_ARCH / "roots" / f"{E.ANCHOR}.jsonl")}
    banked = K.read_by_id(E.MAT_ARCH / "rows" / E.ANCHOR / "F" / "rows_0_of_1.jsonl")
    key = {"LEAF-K8": "W", "TRUNC-4-K8": "X4", "CRITIC": "crit"}
    n = bad = 0
    for did in order:
        r1, rt, rw = truth.get(did), roots.get(did), banked.get(did)
        reg = (tab.get(did) or {}).get("src", {}).get("F", {})
        if r1 is None or rt is None or rw is None or not rw.get("ok") or not reg.get("ok"):
            continue
        tp = LD.truth_part(r1, rt["logits"], rt["top"])
        wp = LD.world_part(rw, tp)
        for z in SCORERS:
            q = np.asarray(wp[key[z]], dtype=np.float64).mean(1)
            v = tp["rec"]["Tp"][L1.pick("ARGMAX", 0.0, q, tp["rec"]["logit"])] - tp["rec"]["Tp"][0]
            n += 1
            bad += int(abs(v - reg["imp"][z]) > 1e-9)
    return {"compared": n, "mismatch": bad, "pass": n > 0 and bad == 0}


def unit_verdict(a: dict, b: Optional[dict]) -> str:
    """PREREGISTRATION §5–§6: the unit's one line."""
    if a.get("stopped"):
        return "STOPPED (a GIGO gate)"
    if not a["bar"]["met"]:
        return "NOT-NEARER: no proposal set, weighted by the opponent's choices, holds worlds clearly nearer the truth"
    if b is None:
        return "NEARER: phase B pending"
    pays = [x for x, v in b["primary"].items() if v["verdict"] == "SEARCH-PAYS"]
    if pays:
        return "SEARCH-PAYS: " + ", ".join(pays)
    return "NEARER-BUT-NOT-PAYING: " + ", ".join(f"{x} {v['verdict']}" for x, v in b["primary"].items())


# -------------------------------------------------------------------------------------------- the selfcheck

def _facts(mh, mt, me, uh, ut, ue) -> dict:
    return dict(mh=mh, mt=mt, me=me, uh=uh, ut=ut, ue=ue, ih=0, it=1, sh=0, st=1, umh=0)


def selfcheck() -> None:
    """The meters and rules on synthetic states: a set whose likelihood picks its true-ish worlds reads NEARER; a
    likelihood that ignores them reads NULL; the resample gate catches a wrong index."""
    from main.value_search.opp_likelihood import normalise, posterior_log_weights, resample_uniform, systematic

    v = facts_values(_facts(2, 4, 0, 1, 2, 0))
    assert v["H"] == 0.5 and v["M_hit"] == 0.5 and v["U_hit"] == 0.5 and v["qSM"] == 0.0
    assert math.isnan(facts_values(_facts(0, 0, 1, 0, 0, 1))["H"]) and facts_values(_facts(0, 0, 1, 0, 0, 1))["qSM"] == 1
    assert verdict_a({"dH": {"ci": [0.0006, 0.02]}, "qSM": {"ci": [0.0, 0.01]}})["verdict"] == "NEARER"
    assert verdict_a({"dH": {"ci": [0.0005, 0.02]}, "qSM": {"ci": [0.0, 0.01]}})["verdict"] == "NOT-DETECTED"
    assert verdict_a({"dH": {"ci": [-0.009, 0.009]}, "qSM": {"ci": [0.0, 0.01]}})["verdict"] == "NULL"
    assert verdict_a({"dH": {"ci": [-0.0095, 0.009]}, "qSM": {"ci": [0.0, 0.01]}})["verdict"] == "NOT-DETECTED"
    assert verdict_a({"dH": {"ci": [-0.03, -0.001]}, "qSM": {"ci": [0.0, 0.01]}})["verdict"] == "FARTHER"
    m = verdict_a({"dH": {"ci": [0.051, 0.08]}, "qSM": {"ci": [0.1006, 0.2]}})
    assert m["material"] and m["dose"]
    m = verdict_a({"dH": {"ci": [0.0505, 0.08]}, "qSM": {"ci": [0.1005, 0.2]}})
    assert not m["material"] and not m["dose"]
    imp = {z: {"ci": [0.001, 0.02]} for z in PRIMARY_SCORERS}
    assert verdict_b(imp, imp) == "SEARCH-PAYS"
    assert verdict_b({"LEAF-K8": {"ci": [0.001, 0.02]}, "TRUNC-4-K8": {"ci": [-0.001, 0.02]}}, imp) == "NOT-DETECTED"
    assert verdict_b({z: {"ci": [-0.009, 0.009]} for z in PRIMARY_SCORERS}, imp) == "NULL"
    assert verdict_b({z: {"ci": [-0.03, -0.001]} for z in PRIMARY_SCORERS}, imp) == "HURTS"
    rng = np.random.default_rng(5)

    def fake(strength: float, n: int = 240) -> List[dict]:
        """Each proposal holds 0..4 of 4 hidden facts; its opponent likelihood rises with them by ``strength``."""
        rows = []
        for i in range(n):
            r = {"id": f"s{i:03d}", "battle": f"b{i // 2}", "stratum": "bot" if i % 2 else "snapshot", "ok": True,
                 "n_scored": 5, "resample": {}}
            for s, nn in (("F", K.N_F), ("C", K.N_C)):
                hits = rng.integers(0, 5, size=nn)
                facts = [_facts(int(h), 4, int(h == 4), 0, 0, 1) for h in hits]
                r[s] = {"ok": True, "log_w": [0.0] * nn, "ll": [float(strength * h + rng.normal(0, 0.5)) for h in hits],
                        "facts": facts, "near4": [0] * nn, "cost": {"cpu_s": 1.0}, "prop": {"lambda": 0.5}}
            for arm in K.RESAMPLED:
                parts, wt = K.ARMS[arm]
                sets = [s for s, _ in parts]
                dice = K.arm_slice({s: r[s]["log_w"] for s in sets}, arm)
                ll = K.arm_slice({s: r[s]["ll"] for s in sets}, arm) if wt == "dice*opp" else None
                lw = posterior_log_weights(len(dice), dice, ll)
                r["resample"][arm] = {"idx": systematic(normalise(lw), K.J,
                                                        resample_uniform(K.RESAMPLE_KEY.format(arm=arm, id=r["id"]))),
                                      "ess": 0.0}
            rows.append(r)
        return rows

    assert "verdict" not in read_a(fake(1.5, 40), interim=True)["primary"]["F128L"]
    strong = read_a(fake(1.5))
    assert not strong["stopped"], strong["gates"]
    assert all(strong["primary"][a]["verdict"] == "NEARER" for a in K.PRIMARY), strong["primary"]
    assert strong["primary"]["F128L"]["material"], strong["primary"]
    assert strong["bar"]["play"] == ["F8", "F128L", "C128L", "M256L", "C8"]
    assert all(strong["secondary_64"][a]["verdict"] == "NEARER" for a in K.SECONDARY_64)
    null = read_a(fake(0.0))
    assert all(null["primary"][a]["verdict"] in ("NULL", "NOT-DETECTED") for a in K.PRIMARY), null["primary"]
    assert not null["bar"]["met"] and null["bar"]["play"] == []
    assert unit_verdict(null, None).startswith("NOT-NEARER")
    bad = fake(1.0, 30)
    bad[3]["resample"]["F128L"]["idx"] = [0] * K.J
    assert read_a(bad)["stopped"]                      # G5: the stored resample is not the weights' resample
    bad = fake(1.0, 30)
    for i in range(3):
        bad[i]["defect"] = True
    assert read_a(bad)["stopped"]                      # G2
    rows = fake(1.0, 30)
    assert read_a(rows, order=[r["id"] for r in rows][1:])["stopped"]      # G1: not the registered prefix
    print("[1c read] selfcheck PASS")


# -------------------------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("part", choices=("a", "b", "bcheck", "selfcheck"))
    ap.add_argument("--interim", action="store_true")
    ap.add_argument("--out")
    a = ap.parse_args()
    if str(K.REPO / "src") not in sys.path:
        sys.path.insert(1, str(K.REPO / "src"))
    if a.part == "selfcheck":
        selfcheck()
        return
    sys.path.insert(1, str(K.STAGE1))
    import s1_common as E                              # the registered order (read-only)

    order = E.ordered_ids()
    rows_by = K.read_by_id(K.ARCH / "rows" / "rows.jsonl")
    rows = [rows_by[i] for i in order if i in rows_by]
    if a.part == "bcheck":
        print(json.dumps(bcheck(E, order), indent=1, sort_keys=True))
        return
    if a.part == "a":
        rec = K.read_rows(K.ARCH / "recheck" / "rows.jsonl") or None
        l3 = K.read_by_id(K.L3_ARCH / "rows_0_of_1.jsonl") or None
        res = read_a(rows, rec, l3, order, interim=a.interim)
        if not res.get("stopped") and not a.interim:
            res["unit"] = unit_verdict(res, None)
    else:
        ra = json.loads((HERE / "result_1c_a.json").read_text())
        if not ra.get("bar", {}).get("met"):
            raise SystemExit("phase A's bar was not met: phase B is not read")
        play = {arm: K.read_by_id(K.ARCH / "play" / arm / "rows.jsonl") for arm in ra["bar"]["play"]}
        truth: Dict[str, dict] = {}
        for p in sorted((E.MAT_ARCH / "rows" / E.ANCHOR / "O").glob("shard_*.jsonl")):
            truth.update({r["id"]: r for r in K.read_rows(p)})
        roots = {r["id"]: r for r in K.read_rows(E.MAT_ARCH / "roots" / f"{E.ANCHOR}.jsonl")}
        fit = json.loads((K.L1DIR / "fit_l1.json").read_text())
        ids = [r["id"] for r in rows if all(r["id"] in play[x] for x in play)]
        idp = K.ARCH / "play" / "identity" / "identity.json"
        identity = json.loads(idp.read_text()) if idp.exists() else {"states": 0, "identical": 0}
        res = read_b(play, truth, roots, E.mat_states(), ids, fit, identity)
        if not res.get("stopped"):
            res["unit"] = unit_verdict(ra, res)
    text = json.dumps(res, indent=1, sort_keys=True, default=float)
    if a.out:
        Path(a.out).write_text(text + "\n")
    print(text[:4000])
    if res.get("stopped"):
        sys.exit(3)


if __name__ == "__main__":
    main()
