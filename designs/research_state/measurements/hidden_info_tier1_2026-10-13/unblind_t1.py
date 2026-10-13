"""TIER 1's READ, part 3: UNBLIND the frozen table and apply the REGISTERED RULES as code (registration §12.3–§12.7,
§12.10, §12.13.4; this unit's README §6 states every reading of an ambiguous rule, fixed BEFORE any arm output existed).

    python -I unblind_t1.py --stage size     # the size rules ONLY (§12.10, §12.13.2): read before any verdict
    python -I unblind_t1.py                  # every rule → result_t1.json, result_t1.md (this folder)

It refuses a table whose sha256 differs from the committed ``blinded_table.sha256`` and a key whose sha256 differs from
the committed ``arm_key.sha256``. No stratum, exclusion or threshold here may change after the key is opened; a change is
a disclosed amendment with both reads reported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_common as T  # noqa: E402
import t1_stats as ST  # noqa: E402
from score_blind import P7, P101, PF, TABLE  # noqa: E402

LEAF = "LEAF-K8"
BOTS = ("aggressive", "heuristic", "random", "setup_sweep", "staller")
P7_IN_P101 = [int(round(p * 100)) for p in P7]
P7_IN_PF = [int(round(p * 20)) for p in P7]
STAGE_AXES = {"k": ("k_bin", T.K_BINS), "turn": ("turn_bin", T.TURN_BINS), "material": ("mat_bin", T.MAT_BINS)}
EARLY_LATE = {"k": (lambda r: r["k"] <= 3, lambda r: r["k"] >= 5),
              "material": (lambda r: r["material"] >= 10, lambda r: r["material"] <= 7),
              "turn": (lambda r: r["turn"] <= 7, lambda r: r["turn"] >= 22)}
DOSE_P = {"F": 0.0, "D50": 0.5, "D90": 0.9, "F+M": 1.0}
BONF_LEVEL = 99.6                     # §12.13.4 R-STAGE-SEARCH: pooled at the 99.6 % level (13 primary buckets)


# ------------------------------------------------------------------------------------------------
# loading
# ------------------------------------------------------------------------------------------------

def load(check: bool = True):
    b = TABLE.read_bytes()
    sha = hashlib.sha256(b).hexdigest()
    want = (T.HERE / "blinded_table.sha256").read_text().split()[0]
    if check and sha != want:
        raise SystemExit(f"the blinded table's sha256 {sha} != the committed {want}")
    kb = T.key_path().read_bytes()
    ksha = hashlib.sha256(kb).hexdigest()
    kwant = (T.HERE / "arm_key.sha256").read_text().split()[0]
    if ksha != kwant:
        raise SystemExit(f"the arm key's sha256 {ksha} != the committed {kwant}")
    key = T.load_key()
    tab = pickle.loads(b)
    name = {"O": "O", "P": "P", "HE@O": "H8e895870", "HE@P": "H8503aeeb", "HE@F": "H528ede2b", **key}
    return tab, name, sha


class Frame:
    """A set of states (one checkpoint or both) with its bootstrap."""

    def __init__(self, recs: List[dict], pooled: bool):
        self.recs = recs
        self.bt = ST.Boot([r["battle"] for r in recs], [r["cp"] for r in recs] if pooled else None)
        self.n = len(recs)

    def imp(self, src: str, scorer: str = LEAF) -> np.ndarray:
        return np.array([r["src"][src]["imp"][scorer] if r["src"].get(src, {}).get("ok") else np.nan
                         for r in self.recs])

    def pick(self, src: str, scorer: str = LEAF) -> np.ndarray:
        return np.array([r["src"][src]["pick"][scorer] if r["src"].get(src, {}).get("ok") else -1 for r in self.recs])

    def ok(self, *srcs: str) -> np.ndarray:
        return np.array([all(r["src"].get(s, {}).get("ok") for s in srcs) for r in self.recs], bool)

    def sel(self, pred) -> np.ndarray:
        return np.array([bool(pred(r)) for r in self.recs], bool)

    def dk(self, base: str, key: str) -> np.ndarray:
        """[n, grid] (NaN rows where the base has no dose family)."""
        rows = [r["dk"].get(base, {}).get(key) for r in self.recs]
        g = next((len(x) for x in rows if x is not None), 0)
        return np.array([x if x is not None else np.full(g, np.nan) for x in rows], dtype=np.float64)


def frames(tab: dict, which: str) -> Dict[str, Frame]:
    out = {}
    for cp in T.CPS:
        out[cp] = Frame([r for r in tab[cp]["recs"] if which == "all" or r["set"] == "registered"], False)
    out["pooled"] = Frame([r for cp in T.CPS for r in tab[cp]["recs"] if which == "all" or r["set"] == "registered"],
                          True)
    return out


def S(est, vec, n=None, level=95.0):
    return ST.summary(est, vec, n, level)


def lo(vec, level=95.0):
    return ST.ci(vec, level)[0]


def hi(vec, level=95.0):
    return ST.ci(vec, level)[1]


# ------------------------------------------------------------------------------------------------
# the attribution arithmetic (§6.3)
# ------------------------------------------------------------------------------------------------

def gap_and_shares(F: Frame, name: Dict[str, str], classes=("M", "U")) -> dict:
    o, f = F.imp("O"), F.imp(name["F"])
    base = F.ok("O", name["F"])
    g_est, g_vec = F.bt.mean(o - f, base)
    out = {"G": S(g_est, g_vec, int(base.sum())), "_G": (g_est, g_vec), "shares": {}}
    for X in classes:
        fx, ox = F.imp(name[f"F+{X}"]), F.imp(name[f"O-{X}"])
        ms = base & F.ok(name[f"F+{X}"])
        mn = base & F.ok(name[f"O-{X}"])
        s_est, s_vec = F.bt.ratio(fx - f, o - f, ms)
        n_est, n_vec = F.bt.ratio(o - ox, o - f, mn)
        ds_est, ds_vec = F.bt.mean(fx - f, ms)
        dn_est, dn_vec = F.bt.mean(o - ox, mn)
        out["shares"][X] = {"s": S(s_est, s_vec, int(ms.sum())), "n": S(n_est, n_vec, int(mn.sum())),
                            "sigma": S(ds_est, ds_vec), "nu": S(dn_est, dn_vec),
                            "_s": (s_est, s_vec), "_n": (n_est, n_vec), "_sigma": (ds_est, ds_vec),
                            "_nu": (dn_est, dn_vec)}
    return out


def contrast(F: Frame, a: str, b: str, scorer=LEAF, sel=None, level=95.0):
    """IMP_a − IMP_b on the states both hold (paired)."""
    m = F.ok(a, b) if sel is None else (F.ok(a, b) & sel)
    est, vec = F.bt.mean(F.imp(a, scorer) - F.imp(b, scorer), m)
    return S(est, vec, int(m.sum()), level), (est, vec)


def slope_of_dose(F: Frame, name: Dict[str, str]) -> dict:
    """B6's paired-bootstrap OLS slope of IMP on p over the registered dose points PRESENT in Tier 1 (README §6:
    p ∈ {0 (F), 0.5 (D50), 0.9 (D90), 1.0 (F+M)}; D25 / D75 are Tier 2)."""
    labs = list(DOSE_P)
    m = F.ok(*[name[x] for x in labs])
    ps = np.array([DOSE_P[x] for x in labs])
    means, vecs = [], []
    for x in labs:
        e, v = F.bt.mean(F.imp(name[x]), m)
        means.append(e)
        vecs.append(v)
    X = np.column_stack([np.ones(len(ps)), ps])
    proj = np.linalg.pinv(X)[1]
    est = float(proj @ np.array(means))
    vec = np.stack(vecs, 1) @ proj
    return {"grid": ps.tolist(), "slope": S(est, vec, int(m.sum())), "_v": (est, vec),
            "means": [round(x, 5) for x in means]}


# ------------------------------------------------------------------------------------------------
# §12.3 R-MOVES, §12.4 R-RIVALS
# ------------------------------------------------------------------------------------------------

def r_moves(fr: Dict[str, Frame], name: Dict[str, str], gigo_pass: bool) -> dict:
    P = fr["pooled"]
    gs = gap_and_shares(P, name)
    g_est, g_vec = gs["_G"]
    res = {"G": gs["G"], "B": {}, "R": {}, "shares": {X: {k: v for k, v in d.items() if not k.startswith("_")}
                                                       for X, d in gs["shares"].items()}}
    B = res["B"]
    B["B1"] = {"holds": False, "R-GIGO": "PASS" if gigo_pass else "FAIL",
               "R-REPLICATE": "NOT AVAILABLE (the independent implementation is compared by a third party, §13.1)"}
    B["B2"] = {"holds": ST.gt(lo(g_vec), 0.02), "G_lower": lo(g_vec)}
    if not B["B2"]["holds"]:
        res["verdict"] = "NO GAP TO APPORTION (G lower <= 0.02)"
        return res
    sM, nM = gs["shares"]["M"]["_s"], gs["shares"]["M"]["_n"]
    sU, nU = gs["shares"]["U"]["_s"], gs["shares"]["U"]["_n"]
    B["B3"] = {"holds": ST.ge(sM[0], 0.5) and ST.ge(lo(sM[1]), 0.25), "s_M": S(*sM)}
    B["B4"] = {"holds": ST.ge(nM[0], 0.5) and ST.ge(lo(nM[1]), 0.25), "n_M": S(*nM)}
    B["B5"] = {"holds": bool(sM[0] >= sU[0] and nM[0] >= nU[0]), "classes_compared": ["U"],
               "not_in_tier1": ["I", "A", "S"], "s_U": S(*sU), "n_U": S(*nU)}
    d90 = contrast(P, name["D90"], name["F"])
    sl = slope_of_dose(P, name)
    B["B6"] = {"holds": ST.gt(d90[0]["ci"][0], 0.0005) and ST.gt(sl["slope"]["ci"][0], 0.0),
               "D90_minus_F": d90[0], "slope": sl["slope"], "slope_grid": sl["grid"], "dose_means": sl["means"],
               "note": "the slope grid holds the Tier-1 doses only (D25 / D75 are Tier 2)"}
    fm9 = contrast(P, name["D90"], name["W90"])
    w90 = contrast(P, name["W90"], name["F"])
    B["B7"] = {"holds": ST.gt(fm9[0]["ci"][0], 0.0005) and ST.lt(w90[0]["ci"][1], 0.01),
               "D90_minus_W90": fm9[0], "W90_minus_F": w90[0]}
    b8 = {}
    for cp in T.CPS:
        c1 = contrast(fr[cp], name["F+M"], name["F"])[0]
        c2 = contrast(fr[cp], "O", name["O-M"])[0]
        b8[cp] = {"F+M_minus_F": c1, "O_minus_O-M": c2,
                  "holds": ST.gt(c1["ci"][0], 0.0005) and ST.gt(c2["ci"][0], 0.0005)}
    B["B8"] = {"holds": all(v["holds"] for v in b8.values()), **b8}
    # the refutation rules
    R = res["R"]
    R["R1"] = ST.lt(hi(sM[1]), 0.25) and ST.lt(hi(nM[1]), 0.25)
    R["R2"] = ST.inside(fm9[0]["ci"], 0.01)
    m_sU = P.ok("O", name["F"], name["F+M"], name["F+U"])
    m_nU = P.ok("O", name["F"], name["O-M"], name["O-U"])
    o, f = P.imp("O"), P.imp(name["F"])
    dsx = P.bt.ratio(P.imp(name["F+U"]) - P.imp(name["F+M"]), o - f, m_sU)
    dnx = P.bt.ratio(P.imp(name["O-M"]) - P.imp(name["O-U"]), o - f, m_nU)
    R["R3"] = ST.gt(lo(dsx[1]), 0.0) and ST.gt(lo(dnx[1]), 0.0)
    R["R3_detail"] = {"s_U_minus_s_M": S(*dsx), "n_U_minus_n_M": S(*dnx)}
    sum_s = P.bt.ratio(P.imp(name["F+M"]) + P.imp(name["F+U"]) - 2 * f, o - f, m_sU)
    res["NON-ADDITIVE"] = {"flag": ST.lt(hi(sum_s[1]), 0.5), "sum_s_M_U": S(*sum_s),
                           "note": "classes M and U only (I, A, S are Tier 2)"}
    b2_8 = all(B[k]["holds"] for k in ("B2", "B3", "B4", "B5", "B6", "B7", "B8"))
    if b2_8 and B["B1"]["holds"]:
        v = "MOVES-BINDING"
    elif R["R1"] or R["R2"] or R["R3"]:
        v = "MOVES-REFUTED (" + ", ".join(k for k in ("R1", "R2", "R3") if R[k]) + ")"
    elif b2_8:
        v = "B2-B8 HOLD; MOVES-BINDING PENDING B1 (R-REPLICATE) and B5 for I/A/S (Tier 2)"
    elif ST.ge(lo(nM[1]), 0.25) and ST.lt(hi(sM[1]), 0.25):
        v = "MOVES-NECESSARY-NOT-SUFFICIENT"
    elif ST.gt(lo(sM[1]), 0.0) or ST.gt(lo(nM[1]), 0.0):
        v = "MOVES-SHARED"
    else:
        v = "MOVES-NOT-DETECTED"
    res["verdict"] = v
    res["modifiers"] = ["PROVISIONAL 'at 15M, current architecture' (§10.5)",
                        "V8 GUARD UNRESOLVED: M-STABLE and G(15M) - G(5M) need the trajectory arm (Tier 3, not run)"]
    return res


def r_rival_u(fr: Dict[str, Frame], name: Dict[str, str]) -> dict:
    P = fr["pooled"]
    gs = gap_and_shares(P, name)
    g_vec = gs["_G"][1]
    sU, nU = gs["shares"]["U"]["_s"], gs["shares"]["U"]["_n"]
    sM, nM = gs["shares"]["M"]["_s"], gs["shares"]["M"]["_n"]
    b8 = {}
    for cp in T.CPS:
        c1 = contrast(fr[cp], name["F+U"], name["F"])[0]
        c2 = contrast(fr[cp], "O", name["O-U"])[0]
        b8[cp] = {"F+U_minus_F": c1, "O_minus_O-U": c2,
                  "holds": ST.gt(c1["ci"][0], 0.0005) and ST.gt(c2["ci"][0], 0.0005)}
    conds = {"B2": ST.gt(lo(g_vec), 0.02),
             "B3": ST.ge(sU[0], 0.5) and ST.ge(lo(sU[1]), 0.25),
             "B4": ST.ge(nU[0], 0.5) and ST.ge(lo(nU[1]), 0.25),
             "B5 (vs M only)": bool(sU[0] >= sM[0] and nU[0] >= nM[0]),
             "B8": all(v["holds"] for v in b8.values())}
    if all(conds.values()):
        v = "U-BINDING (B5 vs M only; I/A/S are Tier 2)"
    elif ST.lt(hi(sU[1]), 0.25) and ST.lt(hi(nU[1]), 0.25):
        v = "U-REFUTED"
    elif ST.gt(lo(sU[1]), 0.0) or ST.gt(lo(nU[1]), 0.0):
        v = "U-SHARED"
    else:
        v = "U-NOT-DETECTED"
    return {"verdict": v, "conditions": conds, "s_U": S(*sU), "n_U": S(*nU), "B8": b8}


def condition_robust(P: Frame, name: Dict[str, str]) -> dict:
    """§10.4 (the conditions available in Tier 1: the opponent class and the game phase): s_M point >= 0.5 in every
    condition with >= 150 states (pooled)."""
    conds = {"opp:bot": lambda r: r["opp_class"].split("_")[0] in BOTS or r["opp_class"].rstrip("2") in BOTS,
             "opp:sentinel": lambda r: r["opp_class"].startswith("sentinel"),
             "phase:2-10": lambda r: 2 <= r["turn"] <= 10, "phase:11-25": lambda r: 11 <= r["turn"] <= 25,
             "phase:26-40": lambda r: 26 <= r["turn"] <= 40}
    o, f, fm = P.imp("O"), P.imp(name["F"]), P.imp(name["F+M"])
    out, fails = {}, []
    for c, pred in conds.items():
        m = P.sel(pred) & P.ok("O", name["F"], name["F+M"])
        e, v = P.bt.ratio(fm - f, o - f, m)
        ge, gv = P.bt.mean(o - f, m)
        out[c] = {"n": int(m.sum()), "s_M": S(e, v), "G": S(ge, gv)}
        if m.sum() >= 150 and not ST.ge(e, 0.5):
            fails.append(c)
    return {"conditions": out, "verdict": "CONDITION-ROBUST" if not fails else "CONDITION-DEPENDENT",
            "failing": fails, "note": "the away bank and the long-term-planning tags are Tier 3 (not run)"}


# ------------------------------------------------------------------------------------------------
# §12.5 R-RECOVER, §12.6 R-EVAL
# ------------------------------------------------------------------------------------------------

def r_recover(P: Frame, name: Dict[str, str]) -> dict:
    o, f = P.imp("O"), P.imp(name["F"])
    out = {}
    for X in ("POOL", "POOLSETS", "MAP100", "D50", "D90", "F+M"):
        m = P.ok("O", name["F"], name[X])
        e, v = P.bt.ratio(P.imp(name[X]) - f, o - f, m)
        out[X] = {"r": S(e, v, int(m.sum())), "_r": (e, v)}
    rp = out["POOL"]["_r"]
    if ST.ge(lo(rp[1]), 0.5):
        v1 = "BELIEF-RECOVERABLE (on this distribution)"
    elif ST.lt(hi(rp[1]), 0.25):
        v1 = "LARGELY-IRREDUCIBLE"
    else:
        v1 = "PARTIAL"
    pm = ST.gt(lo(out["POOLSETS"]["_r"][1]), 0.25)
    sharp = contrast(P, name["MAP100"], name["F"])[0]
    p_half = next((p for X, p in (("D50", 0.5), ("D90", 0.9), ("F+M", 1.0)) if ST.ge(out[X]["_r"][0], 0.5)), None)
    return {"verdict": v1, "PRIOR-MISMATCH": pm, "HONEST-SHARPENING-HELPS": ST.gt(sharp["ci"][0], 0.0005),
            "MAP100_minus_F": sharp, "p_half": p_half,
            "r": {X: d["r"] for X, d in out.items()}}


def r_eval(fr: Dict[str, Frame], name: Dict[str, str]) -> dict:
    P = fr["pooled"]
    out = {"worlds": {}}
    for W in ("F", "P"):
        h = name[f"HE@{W}"]
        c1 = contrast_scorers(P, h, "HE-TRUNC-4-K8", h, "VT-TRUNC-4-K8")
        c2 = contrast_scorers(P, h, "HE-1PLY", h, "VT-CRITIC")
        eb = ST.gt(c1["ci"][0], 0.0005) or ST.gt(c2["ci"][0], 0.0005)
        nb = ST.lt(c1["ci"][1], 0.005) and ST.lt(c2["ci"][1], 0.005)
        out["worlds"][W] = {"HE-TRUNC-4-K8_minus_TRUNC-4-K8": c1, "HE-1PLY_minus_CRITIC": c2,
                            "EVALUATOR-BOUND": eb, "EVALUATOR-NOT-BINDING": nb}
    eF, eP = out["worlds"]["F"]["EVALUATOR-BOUND"], out["worlds"]["P"]["EVALUATOR-BOUND"]
    out["verdict_bound"] = ("EVALUATOR-BOUND (F and P)" if eF and eP else "EVALUATOR-BOUND in F only" if eF
                            else "EVALUATOR-BOUND in P only" if eP else "not EVALUATOR-BOUND")
    nF, nP = out["worlds"]["F"]["EVALUATOR-NOT-BINDING"], out["worlds"]["P"]["EVALUATOR-NOT-BINDING"]
    out["verdict_not_binding"] = ("EVALUATOR-NOT-BINDING (F and P)" if nF and nP else "EVALUATOR-NOT-BINDING in F only"
                                  if nF else "EVALUATOR-NOT-BINDING in P only" if nP else "neither")
    hO, hF = name["HE@O"], name["HE@F"]
    m = P.ok(hO, hF)
    gHE = P.bt.mean(P.imp(hO, "HE-TRUNC-4-K8") - P.imp(hF, "HE-TRUNC-4-K8"), m)
    gCR = P.bt.mean(P.imp(hO, "VT-TRUNC-4-K8") - P.imp(hF, "VT-TRUNC-4-K8"), m)
    ratio = P.bt.ratio(P.imp(hO, "HE-TRUNC-4-K8") - P.imp(hF, "HE-TRUNC-4-K8"),
                       P.imp(hO, "VT-TRUNC-4-K8") - P.imp(hF, "VT-TRUNC-4-K8"), m)
    out["INFORMATION-BOUND"] = {"holds": ST.gt(lo(gHE[1]), 0.0005) and ST.ge(ratio[0], 0.5),
                                "g_HE": S(*gHE, int(m.sum())), "g_CRIT": S(*gCR), "ratio": S(*ratio)}
    # descriptive: every HE scorer per world
    out["imp"] = {W: {z: S(*P.bt.mean(P.imp(name[f"HE@{W}"], z))) for z in
                      ("HE-TRUNC-1-K8", "HE-TRUNC-2-K8", "HE-TRUNC-4-K4", "HE-TRUNC-4-K8", "HE-1PLY",
                       "VT-TRUNC-4-K8", "VT-CRITIC")} for W in ("O", "P", "F")}
    out["per_cp"] = {cp: {W: {"HE-TRUNC-4-K8_minus_TRUNC-4-K8":
                              contrast_scorers(fr[cp], name[f"HE@{W}"], "HE-TRUNC-4-K8", name[f"HE@{W}"],
                                               "VT-TRUNC-4-K8"),
                              "HE-1PLY_minus_CRITIC": contrast_scorers(fr[cp], name[f"HE@{W}"], "HE-1PLY",
                                                                       name[f"HE@{W}"], "VT-CRITIC")}
                          for W in ("F", "P")} for cp in T.CPS}
    return out


def contrast_scorers(F: Frame, a: str, za: str, b: str, zb: str, sel=None) -> dict:
    m = F.ok(a, b) if sel is None else F.ok(a, b) & sel
    e, v = F.bt.mean(F.imp(a, za) - F.imp(b, zb), m)
    return S(e, v, int(m.sum()))


# ------------------------------------------------------------------------------------------------
# §12.13.4 the STAGE rules
# ------------------------------------------------------------------------------------------------

def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    return (x - x.mean()) / x.std()


def stage_voi(fr: Dict[str, Frame], name: Dict[str, str]) -> dict:
    out = {}
    for ax, (fld, bins) in STAGE_AXES.items():
        res = {"G": {}, "per_cp": {}}
        vals = {}
        for cpn in list(T.CPS) + ["pooled"]:
            F = fr[cpn]
            g = F.imp("O") - F.imp(name["F"])
            base = F.ok("O", name["F"])
            vals[cpn] = {b: F.bt.mean(g, base & F.sel(lambda r, b=b: r[fld] == b)) for b in bins}
            raw = np.array([r[ax] if ax != "material" else r["material"] for r in F.recs], dtype=np.float64)
            beta, bs = F.bt.ols(g, zscore(raw)[:, None], base)
            vals[cpn]["_slope"] = (float(beta[1]), bs[:, 1])
        P = vals["pooled"]
        res["G"] = {b: S(*P[b]) for b in bins}
        res["per_cp"] = {cp: {b: S(*vals[cp][b]) for b in bins} for cp in T.CPS}
        res["slope_per_sd"] = S(*P["_slope"])
        res["slope_per_cp"] = {cp: S(*vals[cp]["_slope"]) for cp in T.CPS}
        peaks = []
        for b in bins:
            if all(ST.gt(lo(P[b][1] - P[c][1]), 0.0005) for c in bins if c != b):
                rep = all(all(vals[cp][b][0] > vals[cp][c][0] for c in bins if c != b) for cp in T.CPS)
                peaks.append({"bucket": b, "replicated": rep})
        sl_lo, sl_hi = ST.ci(P["_slope"][1])
        verdicts = [f"PEAK({p['bucket']})" + ("" if p["replicated"] else " NOT REPLICATED") for p in peaks]
        if ST.lt(sl_hi, -0.0005):
            rep = all(vals[cp]["_slope"][0] < 0 for cp in T.CPS)
            verdicts.append("TREND-DOWN" + ("" if rep else " NOT REPLICATED"))
        elif ST.gt(sl_lo, 0.0005):
            rep = all(vals[cp]["_slope"][0] > 0 for cp in T.CPS)
            verdicts.append("TREND-UP" + ("" if rep else " NOT REPLICATED"))
        if not verdicts:
            bmax = max(bins, key=lambda b: P[b][0])
            bmin = min(bins, key=lambda b: P[b][0])
            if ST.inside(ST.ci(P[bmax][1] - P[bmin][1]), 0.04):
                verdicts.append("FLAT")
            else:
                verdicts.append("NOT DETECTED")
        res["verdict"] = verdicts
        out[ax] = res
    return out


def stage_axis(fr: Dict[str, Frame], name: Dict[str, str]) -> dict:
    out = {}
    coefs = {}
    for cpn in list(T.CPS) + ["pooled"]:
        F = fr[cpn]
        g = F.imp("O") - F.imp(name["F"])
        base = F.ok("O", name["F"])
        Z = np.column_stack([zscore([r[c] for r in F.recs]) for c in ("k", "turn", "material")])
        beta, bs = F.bt.ols(g, Z, base)
        coefs[cpn] = (beta, bs)
    beta, bs = coefs["pooled"]
    for i, ax in enumerate(("k", "turn", "material"), start=1):
        l, h = ST.ci(bs[:, i])
        carries = ST.gt(l, 0.0005) or ST.lt(h, -0.0005)
        rep = all(np.sign(coefs[cp][0][i]) == np.sign(beta[i]) for cp in T.CPS)
        out[ax] = {"partial_slope": S(beta[i], bs[:, i]),
                   "per_cp": {cp: round(float(coefs[cp][0][i]), 5) for cp in T.CPS},
                   "verdict": ("CARRIES" + ("" if rep else " NOT REPLICATED")) if carries else "does not carry"}
    return out


def stage_mech(fr: Dict[str, Frame], name: Dict[str, str]) -> dict:
    out = {}
    for ax, (fe, fl) in EARLY_LATE.items():
        res = {}
        vals = {}
        for cpn in list(T.CPS) + ["pooled"]:
            F = fr[cpn]
            base = F.ok("O", name["F"])
            d = (F.pick("O") != F.pick(name["F"])).astype(np.float64)
            gain = F.imp("O") - F.imp(name["F"])
            v = {}
            for side, pred in (("early", fe), ("late", fl)):
                m = base & F.sel(pred)
                v[side] = {"c": F.bt.mean(d, m), "s": F.bt.ratio(gain, d, m), "G": F.bt.mean(gain, m), "n": int(m.sum())}
            vals[cpn] = v
        P = vals["pooled"]
        dc = P["early"]["c"][1] - P["late"]["c"][1]
        ds = P["early"]["s"][1] - P["late"]["s"][1]
        dc_e = P["early"]["c"][0] - P["late"]["c"][0]
        ds_e = P["early"]["s"][0] - P["late"]["s"][0]
        cbar = (P["early"]["c"][0] + P["late"]["c"][0]) / 2
        sbar = (P["early"]["s"][0] + P["late"]["s"][0]) / 2
        fq = ST.gt(lo(dc), 0.0005) and ST.inside(ST.ci(ds), 0.04)
        st = ST.gt(lo(ds), 0.0005) and ST.inside(ST.ci(dc), 0.05)
        both = ST.gt(lo(dc), 0.0005) and ST.gt(lo(ds), 0.0005)
        v = "BOTH" if both else "FREQUENCY-LED" if fq else "STAKES-LED" if st else "NOT DETECTED"
        rep = all(np.sign(vals[cp]["early"]["c"][0] - vals[cp]["late"]["c"][0]) == np.sign(dc_e) for cp in T.CPS) \
            if v in ("FREQUENCY-LED", "BOTH") else True
        if v in ("STAKES-LED", "BOTH"):
            rep = rep and all(np.sign(vals[cp]["early"]["s"][0] - vals[cp]["late"]["s"][0]) == np.sign(ds_e)
                              for cp in T.CPS)
        res = {"early": {k: (S(*P["early"][k]) if k != "n" else P["early"][k]) for k in ("c", "s", "G", "n")},
               "late": {k: (S(*P["late"][k]) if k != "n" else P["late"][k]) for k in ("c", "s", "G", "n")},
               "delta_c": S(dc_e, dc), "delta_s": S(ds_e, ds),
               "split": {"delta_c_times_sbar": round(dc_e * sbar, 5), "cbar_times_delta_s": round(cbar * ds_e, 5)},
               "verdict": v + ("" if rep else " NOT REPLICATED")}
        out[ax] = res
    return out


def stage_class(fr: Dict[str, Frame], name: Dict[str, str]) -> dict:
    P = fr["pooled"]
    o, f = P.imp("O"), P.imp(name["F"])
    fm, fu, om, ou = (P.imp(name[x]) for x in ("F+M", "F+U", "O-M", "O-U"))
    allok = P.ok("O", name["F"], name["F+M"], name["F+U"], name["O-M"], name["O-U"])
    out = {}
    for ax, (fld, bins) in STAGE_AXES.items():
        out[ax] = {}
        for b in bins:
            if ax == "k" and b == "6":
                out[ax][b] = {"verdict": "UNDEFINED (no never-seen mon at k = 6)"}
                continue
            m = allok & P.sel(lambda r, b=b: r[fld] == b)
            sM, sU = P.bt.mean(fm - f, m), P.bt.mean(fu - f, m)
            nM, nU = P.bt.mean(o - om, m), P.bt.mean(o - ou, m)
            dS, dN = P.bt.mean(fm - fu, m), P.bt.mean(ou - om, m)          # σ_M − σ_U ; ν_M − ν_U
            dirS = "M" if ST.gt(lo(dS[1]), 0.0005) else "U" if ST.lt(hi(dS[1]), -0.0005) else None
            dirN = "M" if ST.gt(lo(dN[1]), 0.0005) else "U" if ST.lt(hi(dN[1]), -0.0005) else None
            if dirS is not None and dirS == dirN:
                v = f"{dirS}-LED"
            elif (dirS is None) != (dirN is None) or (dirS is not None and dirN is not None):
                v = "SIDES-DISAGREE"
            elif ST.gt(lo(sM[1]), 0.0005) and ST.gt(lo(sU[1]), 0.0005):
                v = "MIXED"
            elif ST.lt(hi(sM[1]), 0.01) and ST.lt(hi(sU[1]), 0.01):
                v = "NEITHER"
            else:
                v = "NOT DETECTED"
            g = P.bt.mean(o - f, m)
            out[ax][b] = {"n": int(m.sum()), "G": S(*g), "sigma_M": S(*sM), "sigma_U": S(*sU), "nu_M": S(*nM),
                          "nu_U": S(*nU), "sigma_M_minus_U": S(*dS), "nu_M_minus_U": S(*dN),
                          "phi_M": round(0.5 * (sM[0] + nM[0]), 5), "phi_U": round(0.5 * (sU[0] + nU[0]), 5),
                          "interaction_M": round(abs(sM[0] - nM[0]), 5), "interaction_U": round(abs(sU[0] - nU[0]), 5),
                          "verdict": v}
    return out


def stage_search(fr: Dict[str, Frame], name: Dict[str, str]) -> dict:
    out, frontier = {}, []
    fname = name["F"]
    for ax, (fld, bins) in STAGE_AXES.items():
        out[ax] = {}
        for b in bins:
            res = {"per_cp": {}}
            per_ok = True
            for cp in T.CPS:
                F = fr[cp]
                m = F.ok(fname) & F.sel(lambda r, b=b: r[fld] == b)
                c1 = F.bt.mean(F.imp(fname, "TRUNC-4-K8") - F.imp(fname, "CRITIC"), m)
                c2 = F.bt.mean(F.imp(fname, "TRUNC-4-K8"), m)
                c3 = F.bt.mean(F.imp(fname, LEAF) - F.imp(fname, "CRITIC"), m)
                ok = int(m.sum()) >= 150 and ST.gt(lo(c1[1]), 0.0005) and ST.gt(lo(c2[1]), 0.0005)
                per_ok &= ok
                res["per_cp"][cp] = {"n": int(m.sum()), "TRUNC4_minus_CRITIC": S(*c1), "TRUNC4": S(*c2),
                                     "LEAF8_minus_CRITIC": S(*c3), "holds": ok}
            P = fr["pooled"]
            m = P.ok(fname) & P.sel(lambda r, b=b: r[fld] == b)
            c1 = P.bt.mean(P.imp(fname, "TRUNC-4-K8") - P.imp(fname, "CRITIC"), m)
            c2 = P.bt.mean(P.imp(fname, "TRUNC-4-K8"), m)
            c3 = P.bt.mean(P.imp(fname, LEAF) - P.imp(fname, "CRITIC"), m)
            bonf = ST.gt(lo(c1[1], BONF_LEVEL), 0.0005) and ST.gt(lo(c2[1], BONF_LEVEL), 0.0005)
            p95 = ST.gt(lo(c1[1]), 0.0005) and ST.gt(lo(c2[1]), 0.0005)
            v = "GATE-OPEN" if (per_ok and bonf) else "GATE-WEAK" if p95 else "GATE-CLOSED"
            res.update({"pooled": {"n": int(m.sum()), "TRUNC4_minus_CRITIC": S(*c1),
                                   "TRUNC4_minus_CRITIC_996": S(*c1, level=BONF_LEVEL), "TRUNC4": S(*c2),
                                   "TRUNC4_996": S(*c2, level=BONF_LEVEL), "LEAF8_minus_CRITIC": S(*c3)},
                        "verdict": v})
            if v == "GATE-OPEN":
                frontier.append(f"{ax}:{b}")
            out[ax][b] = res
    return {"buckets": out, "FRONTIER": frontier}


# ------------------------------------------------------------------------------------------------
# §6.9 / §12.13.4 R-DOSE
# ------------------------------------------------------------------------------------------------

def persistent_first(mask):
    if not len(mask) or not mask[-1]:
        return None
    i = len(mask) - 1
    while i > 0 and mask[i - 1]:
        i -= 1
    return i


def eq_p(curve, target, grid):
    """The p at which the curve first reaches the target (linear interpolation); 0 at or below the p = 0 value;
    None above the p = 1 value."""
    if not np.isfinite(target) or not np.all(np.isfinite(curve)):
        return None
    if target <= curve[0]:
        return 0.0
    for i in range(1, len(curve)):
        if curve[i] >= target:
            a, b = curve[i - 1], curve[i]
            return float(grid[i - 1] + (grid[i] - grid[i - 1]) * (target - a) / (b - a)) if b > a else float(grid[i])
    return None


def dose_cell(P: Frame, name: Dict[str, str], sel: np.ndarray) -> dict:
    fname, fn = name["F"], name["FN"]
    m = sel & P.ok("O", fname)
    leaf = P.dk(fname, f"M8|{LEAF}")                       # [n, 101]
    crit = P.dk(fname, "M8|CRITIC")                        # [n, 21]
    res = {"n": int(m.sum())}
    if m.sum() == 0:
        return res
    # the curves on the owner's grid
    L7 = leaf[:, P7_IN_P101]
    C7 = crit[:, P7_IN_PF]
    cur = {}
    for gi, p in enumerate(P7):
        cur[p] = {"leaf": P.bt.mean(L7[:, gi], m), "crit": P.bt.mean(C7[:, gi], m),
                  "leaf_minus_crit": P.bt.mean(L7[:, gi] - C7[:, gi], m),
                  "leaf_minus_crit0": P.bt.mean(L7[:, gi] - C7[:, 0], m)}
    res["curve"] = {f"{p:g}": {k: S(*v) for k, v in d.items()} for p, d in cur.items()}

    def pstar(key):
        mask = np.array([ST.gt(lo(cur[p][key][1]), 0.0005) for p in P7])
        i = persistent_first(mask)
        return None if i is None else float(P7[i])

    res["P-STAR"] = {"vs_policy": pstar("leaf"), "vs_critic_same_p": pstar("leaf_minus_crit"),
                     "vs_critic_today": pstar("leaf_minus_crit0")}
    # TODAY: the p-equivalent of FN on the 0.05 grid
    mf = m & P.ok(fn)
    LF = leaf[:, ::5]                                     # PF
    lc_e, lc_v = [], []
    for gi in range(len(PF)):
        e, v = P.bt.mean(LF[:, gi], mf)
        lc_e.append(e)
        lc_v.append(v)
    lc_e, lc_v = np.array(lc_e), np.stack(lc_v, 1)
    fe, fv = P.bt.mean(P.imp(fn), mf)
    t_pt = eq_p(lc_e, fe, PF)
    draws = np.array([9.0 if (x := eq_p(c, t, PF)) is None else x for c, t in zip(lc_v, fv)])
    t_up = float(np.percentile(draws, 97.5)) if len(draws) else None
    res["TODAY"] = {"n": int(mf.sum()), "p_eq": t_pt, "upper95": None if t_up is None or t_up > 1.5 else round(t_up, 4),
                    "upper95_above_curve": bool(t_up is not None and t_up > 1.5), "IMP_FN": S(fe, fv),
                    "interval": [None if x > 1.5 else round(float(x), 4) for x in np.percentile(draws, [2.5, 97.5])]}
    ps = res["P-STAR"]["vs_critic_same_p"]
    tu = 1.0 if res["TODAY"]["upper95_above_curve"] else res["TODAY"]["upper95"]
    if ps is not None and tu is not None and not ST.gt(ps, tu + 0.10):
        reach = "WITHIN-REACH"
    elif ps is None or (tu is not None and ST.gt(ps, tu + 0.25)):
        reach = "FAR"
    else:
        reach = "INTERMEDIATE"
    res["REACH"] = reach
    # SHAPE
    d2 = P.bt.mean(leaf[:, 50] - 0.5 * (leaf[:, 0] + leaf[:, 100]), m)
    res["SHAPE"] = {"delta2": S(*d2), "verdict": "CONCAVE" if ST.gt(lo(d2[1]), 0.0005) else
                    "CONVEX" if ST.lt(hi(d2[1]), -0.0005) else "LINEAR"}
    full_e = np.array([P.bt.mean(leaf[:, i], m)[0] for i in range(101)])
    half = full_e[0] + 0.5 * (full_e[-1] - full_e[0])
    res["p_half"] = next((float(P101[i]) for i in range(101) if full_e[i] >= half), None)
    # CONTENT, OVER-CONFIDENCE, CRITIC-INFO-LIMITED
    i35 = list(P7).index(0.35)
    i75 = list(P7).index(0.75)
    m5 = P.dk(fname, f"M5|{LEAF}")
    m5w = P.dk(fname, f"M5W|{LEAF}")
    m5x = P.dk(fname, f"M5X|{LEAF}")
    cm = P.bt.mean(m5[:, i35] - m5w[:, i35], m)
    oc = P.bt.mean(m5x[:, i75] - m5x[:, 0], m)
    ci_ = P.bt.mean(crit[:, -1] - crit[:, 0], m)
    res["CONTENT-MATTERS"] = {"holds": ST.gt(lo(cm[1]), 0.0005), "M5_minus_M5W_at_0.35": S(*cm)}
    res["OVERCONFIDENCE-COSTS"] = {"holds": ST.lt(hi(oc[1]), -0.0005), "M5X_0.75_minus_0": S(*oc)}
    res["CRITIC-INFO-LIMITED"] = {"holds": ST.gt(lo(ci_[1]), 0.0005), "CRITIC_1_minus_0": S(*ci_)}
    res["M5"] = {f"{p:g}": S(*P.bt.mean(m5[:, i], m)) for i, p in enumerate(P7)}
    res["M5W"] = {f"{p:g}": S(*P.bt.mean(m5w[:, i], m)) for i, p in enumerate(P7)}
    res["M5X"] = {f"{p:g}": S(*P.bt.mean(m5x[:, i], m)) for i, p in enumerate(P7)}
    return res


def dose(fr: Dict[str, Frame], name: Dict[str, str]) -> dict:
    P = fr["pooled"]
    out = {"all": dose_cell(P, name, np.ones(P.n, bool)), "per_cp": {}, "axes": {}}
    for cp in T.CPS:
        out["per_cp"][cp] = dose_cell(fr[cp], name, np.ones(fr[cp].n, bool))
    fname = name["F"]
    leaf = P.dk(fname, f"M8|{LEAF}")
    for ax, (fld, bins) in STAGE_AXES.items():
        cells = {b: dose_cell(P, name, P.sel(lambda r, b=b: r[fld] == b)) for b in bins}
        # DETERMINIZATION-FIRST: the payoff of the next 0.10 of belief from TODAY(b), per draw
        pi = {}
        for b in bins:
            t = cells[b].get("TODAY", {}).get("p_eq")
            if t is None:
                continue
            i0 = int(round(min(t, 0.9) * 100))
            m = P.ok("O", fname) & P.sel(lambda r, b=b: r[fld] == b)
            pi[b] = P.bt.mean(leaf[:, i0 + 10] - leaf[:, i0], m)
            cells[b]["PI"] = S(*pi[b])
        first = [b for b in pi if all(ST.gt(lo(pi[b][1] - pi[c][1]), 0.0005) for c in pi if c != b)]
        out["axes"][ax] = {"cells": cells, "DETERMINIZATION-FIRST": first}
    return out


# ------------------------------------------------------------------------------------------------
# validity, size rules, descriptive tables
# ------------------------------------------------------------------------------------------------

def size_rules(fr_reg: Dict[str, Frame], fr_all: Dict[str, Frame], name: Dict[str, str]) -> dict:
    gs = gap_and_shares(fr_reg["pooled"], name, classes=("M",))
    sM, nM = gs["shares"]["M"]["_s"], gs["shares"]["M"]["_n"]
    hw = lambda v: None if None in ST.ci(v) else round((ST.ci(v)[1] - ST.ci(v)[0]) / 2, 4)   # noqa: E731
    out = {"s_M_halfwidth": hw(sM[1]), "n_M_halfwidth": hw(nM[1])}
    out["rule_12_10_triggered"] = bool((out["s_M_halfwidth"] or 0) > 0.20 or (out["n_M_halfwidth"] or 0) > 0.20)
    P = fr_all["pooled"]
    g = P.imp("O") - P.imp(name["F"])
    base = P.ok("O", name["F"])
    kb = {}
    for b in T.K_BINS:
        e, v = P.bt.mean(g, base & P.sel(lambda r, b=b: r["k_bin"] == b))
        kb[b] = hw(v)
    out["G_F_halfwidth_by_k"] = kb
    out["rule_12_13_2_bins_triggered"] = [b for b, h in kb.items() if h is not None and h > 0.025]
    return out


def validity(fr_all: Dict[str, Frame], name: Dict[str, str], tab: dict) -> dict:
    P = fr_all["pooled"]
    out = {}
    k6 = P.sel(lambda r: r["k"] == 6)
    for a, b, what in (("F+U", "F", "F+U = F at k = 6"), ("O-U", "O", "O-U = O (seeds 0-7) at k = 6")):
        na, nb_ = name[a], name[b]
        m = k6 & P.ok(na, nb_)
        diffs = {}
        for z in ("LEAF-K8", "TRUNC-4-K8", "CRITIC"):
            if b == "O" and z == "LEAF-K8":
                pass
            x = P.imp(na, z) - P.imp(nb_, z)
            diffs[z] = int(np.sum(np.abs(x[m]) > 1e-9))
        out[what] = {"states": int(m.sum()), "states_differing": diffs}
    for W in ("F", "O"):
        h = name[f"HE@{W}"]
        src = name["F"] if W == "F" else "O"
        m = P.ok(h, src)
        d1 = np.abs(P.imp(h, "VT-TRUNC-4-K8") - P.imp(src, "TRUNC-4-K8"))[m]
        d2 = np.abs(P.imp(h, "VT-CRITIC") - P.imp(src, "CRITIC"))[m]
        out[f"HE@{W} re-play = the arm's own critic scorers"] = {
            "states": int(m.sum()), "TRUNC-4-K8_differing": int(np.sum(d1 > 1e-9)), "CRITIC_differing": int(np.sum(d2 > 1e-9))}
    excl = {}
    for lab in list(T.ARMS):
        c = name[lab]
        reg = [r for r in P.recs if r["set"] == "registered" or T.ARMS[lab]["states"] == "all"]
        n_ok = sum(1 for r in reg if r["src"].get(c, {}).get("ok"))
        n_ex = sum(1 for r in reg if r["src"].get(c, {}).get("ok") is False)
        excl[lab] = {"states": len(reg), "ok": n_ok, "excluded_or_refused": n_ex, "missing": len(reg) - n_ok - n_ex,
                     "share_excluded": round((len(reg) - n_ok) / max(1, len(reg)), 4)}
    out["exclusions"] = excl
    out["F_excluded_over_10pct"] = excl["F"]["share_excluded"] > 0.10
    return out


def describe(fr: Dict[str, Frame], name: Dict[str, str]) -> dict:
    """Every arm's IMP for every scorer (registered 600: per checkpoint and pooled)."""
    out = {}
    labels = ["O", "P"] + list(T.ARMS)
    for cpn, F in fr.items():
        out[cpn] = {}
        for lab in labels:
            src = name[lab]
            out[cpn][lab] = {z: S(*F.bt.mean(F.imp(src, z)), int(np.isfinite(F.imp(src, z)).sum()))
                             for z in ("LEAF-K1", "LEAF-K4", "LEAF-K8", "TRUNC-1-K8", "TRUNC-2-K8", "TRUNC-4-K4",
                                       "TRUNC-4-K8", "CRITIC")}
    return out


def natural_mix(fr_all: Dict[str, Frame], name: Dict[str, str]) -> dict:
    """G on all states re-weighted to the pool's k mix: weight = N_pool(bin) / N_sample(bin), per checkpoint."""
    out = {}
    P = fr_all["pooled"]
    wts = []
    for r in P.recs:
        cen = json.loads((T.HERE / f"census_{r['cp']}.json").read_text())
        npool = cen["eligible_pool"]["k_bin"] if isinstance(cen["eligible_pool"], dict) else None
        if npool is None:
            raise SystemExit("census eligible_pool block missing")
        nsamp = sum(1 for x in P.recs if x["cp"] == r["cp"] and x["k_bin"] == r["k_bin"])
        wts.append(npool[r["k_bin"]] / nsamp)
    g = P.imp("O") - P.imp(name["F"])
    base = P.ok("O", name["F"])
    out["G_all_k_weighted"] = S(*P.bt.mean(g, base, np.array(wts)))
    out["G_all_unweighted"] = S(*P.bt.mean(g, base))
    return out


def root_choice_sensitivity(fr_reg: Dict[str, Frame], name: Dict[str, str]) -> dict:
    """README §6 item 9: G and the M / U shares with the states whose opponent root-turn choice is RECORDED before the
    searched side's (Phase 1 ``other_open`` false) dropped."""
    P = fr_reg["pooled"]
    keep = P.sel(lambda r: r.get("other_open", True))
    sub = Frame([r for r, k in zip(P.recs, keep) if k], True)
    gs = gap_and_shares(sub, name)
    return {"states_dropped": int((~keep).sum()), "G": gs["G"],
            "shares": {X: {k: d[k] for k in ("s", "n")} for X, d in gs["shares"].items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all", choices=("size", "all"))
    ap.add_argument("--gigo", default=None, help="the Tier-1 R-GIGO read (world_audit.json); default this folder's")
    a = ap.parse_args()
    tab, name, sha = load()
    fr_reg = frames(tab, "registered")
    fr_all = frames(tab, "all")
    size = size_rules(fr_reg, fr_all, name)
    if a.stage == "size":
        print(json.dumps(size, indent=1, sort_keys=True))
        (T.HERE / "result_size_rules.json").write_text(json.dumps(size, indent=1, sort_keys=True) + "\n")
        return
    gp = Path(a.gigo) if a.gigo else T.HERE / "world_audit.json"
    gigo = json.loads(gp.read_text()) if gp.exists() else {"verdict": "NOT READ"}
    res = {"table_sha256": sha, "size_rules": size, "validity": validity(fr_all, name, tab),
           "R-GIGO (Tier 1 worlds)": gigo.get("verdict"),
           "R-MOVES": r_moves(fr_reg, name, gigo.get("verdict") == "PASS"),
           "R-RIVALS": {"U": r_rival_u(fr_reg, name)},
           "CONDITION": condition_robust(fr_reg["pooled"], name),
           "R-RECOVER": r_recover(fr_reg["pooled"], name),
           "R-EVAL": r_eval(fr_reg, name),
           "R-STAGE-VOI": stage_voi(fr_all, name), "R-STAGE-AXIS": stage_axis(fr_all, name),
           "R-STAGE-MECH": stage_mech(fr_all, name), "R-STAGE-CLASS": stage_class(fr_all, name),
           "R-STAGE-SEARCH": stage_search(fr_all, name), "R-DOSE": dose(fr_all, name),
           "per_cp_shares": {cp: {X: {k: v for k, v in d.items() if not k.startswith("_")}
                                  for X, d in gap_and_shares(fr_reg[cp], name)["shares"].items()}
                             for cp in T.CPS},
           "per_cp_G": {cp: gap_and_shares(fr_reg[cp], name)["G"] for cp in T.CPS},
           "natural_mix": natural_mix(fr_all, name),
           "root_choice_sensitivity": root_choice_sensitivity(fr_reg, name),
           "describe_registered": describe(fr_reg, name)}
    (T.HERE / "result_t1.json").write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({"R-MOVES": res["R-MOVES"]["verdict"], "U": res["R-RIVALS"]["U"]["verdict"],
                      "R-RECOVER": res["R-RECOVER"]["verdict"], "R-EVAL": res["R-EVAL"]["verdict_bound"],
                      "FRONTIER": res["R-STAGE-SEARCH"]["FRONTIER"]}, indent=1))


if __name__ == "__main__":
    main()
