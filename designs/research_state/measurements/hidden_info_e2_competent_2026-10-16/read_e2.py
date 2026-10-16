"""E2's PRE-REGISTERED READ (`PREREGISTRATION.md` §3-§5), as code, committed BEFORE any E2 arm output exists.
PURE numpy over the archive (no model, no engine). Read-only on every input.

    python -I read_e2.py [--partial] [--no-tier1]      # -> result_e2.json (+ the verdict lines on stdout)

Per state (`states_e2_<cp>.json`): the truth (Phase 1, rows/O) and the five arms (rows/<arm>), each turned into the
per-action arrays of Tier 1's `t1_data` (`truth_arrays`, `world_arrays`; imported) in POLICY order, and the IMP under
every scorer of the red team's section C (`red_team.scorer_imps`, imported): IMP = T(the scorer's pick) − T(the policy's
top1), T = the mean outcome over the TRUE world's referee seeds 16..39.

PRIMARY (§3.4): pooled end2 + prod, LEAF-K8 (ties to the policy's first): Δs = IMP(F+U) − IMP(F+M), Δn = IMP(O−M) −
IMP(O−U) → U-FIRST / M-FIRST / SHARED / NOT DETECTED with the qualifiers. Rule 8 at ±0.0005 (`t1_stats`: a boundary c
is clear only beyond c ± 0.0005, so the MATERIAL flag at 0.005 reads lower > 0.0055). Bootstrap: `t1_stats.Boot` (2,000
draws, seed 20261011, battle-clustered, pooled within checkpoint).
SECONDARY (§4): per checkpoint; LEAF-K8|RAND (SCORER-DEPENDENT when its verdict differs) and every other scorer of the
section C list (no verdict); the shares + B3 / B4; Tier 1's bot / snapshot populations (descriptive); by k bin and traced
outcome. VALIDITY (§5): the exclusion / defect / missing counts per arm, the k = 6 identities, the 1 % defect flag.

``--partial`` reads whatever exists (for a progress check of COUNTS only; it refuses to print any verdict or number of
an incomplete read unless ``--allow-incomplete-numbers`` is also given — never during the crawl).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2_common as E  # noqa: E402

E.add_paths()
import t1_data as D  # noqa: E402
import t1_stats as ST  # noqa: E402

OUT = E.HERE / "result_e2.json"
SOURCES = ("O",) + tuple(E.ARMS)
PRIMARY = "LEAF-K8"
TIE_FREE = "LEAF-K8|RAND"
EDGE = ST.EDGE
MATERIAL = 0.005
CLOSE = 0.01


def scorers():
    import red_team as RT

    return RT.SCORERS_C


# ------------------------------------------------------------------------------------------------
# loading (I/O)
# ------------------------------------------------------------------------------------------------

def records(cp: str) -> List[dict]:
    """One record per E2 state: its meta, and per source {ok, imp{scorer: value}} or a status."""
    import red_team as RT
    from score_blind import fact_hits

    st = E.e2_states(cp)
    truth = E.read_dir(E.rows_dir(cp, "O"), "shard_*.jsonl")
    rows = {a: E.read_dir(E.cp_dir(cp) / "rows" / E.arm_dir(a), "rows_*.jsonl") for a in E.ARMS}
    worlds = {a: E.read_dir(E.cp_dir(cp) / "worlds" / E.arm_dir(a), "worlds_*.jsonl") for a in E.ARMS}
    out = []
    for did in sorted(st):
        e = st[did]
        rec = {"id": did, "cp": cp, "battle": f"{cp}:{e['battle']}", "k": int(e["k"]), "outcome": e["outcome"],
               "k_bin": "1-2" if int(e["k"]) <= 2 else str(int(e["k"])), "src": {}}
        ou_w = worlds["O-U"].get(did)
        rec["move_slots"] = (fact_hits(ou_w)[1] / 8.0) if ou_w and ou_w.get("ok") else float("nan")
        r1 = truth.get(did)
        if r1 is None:
            rec["src"]["O"] = {"status": "missing"}
            out.append(rec)
            continue
        if not r1.get("ok"):
            rec["src"]["O"] = {"status": "refused", "error": str(r1.get("error"))[:200]}
            rec["src"].update({a: {"status": "no truth"} for a in E.ARMS})     # §5: no referee, no contrast
            out.append(rec)
            continue
        A = D.truth_arrays(r1, {"logits": e["logits"], "top": e["top"]})
        o = np.array(A["order"])
        Tp = A["T"][o]
        rec["src"]["O"] = {"status": "ok", "imp": RT.scorer_imps(
            {"W": A["W"][o][:, :E.J], "X4": A["Xd"][4][o][:, :E.J], "crit": A["crit"][o]}, Tp)}
        for a in E.ARMS:
            w = worlds[a].get(did)
            rd = rows[a].get(did)
            if w is not None and not w.get("ok"):
                rec["src"][a] = {"status": "defect" if w.get("crash") else "excluded", "error": str(w.get("error"))[:200]}
                continue
            if rd is None:
                rec["src"][a] = {"status": "missing"}
                continue
            if not rd.get("ok"):
                rec["src"][a] = {"status": "refused", "error": str(rd.get("error"))[:200]}
                continue
            Aw = D.world_arrays(rd, A["acts"], A["turn"])
            rec["src"][a] = {"status": "ok", "imp": RT.scorer_imps(
                {"W": Aw["W"][o], "X4": Aw["Xd"][4][o], "crit": Aw["crit"][o]}, Tp),
                "out": rd["out"]}
        rec["truth_out8"] = {a: s[:E.J] for a, s in r1["out"].items()}
        out.append(rec)
    return out


# ------------------------------------------------------------------------------------------------
# the read (pure)
# ------------------------------------------------------------------------------------------------

def imp(recs: List[dict], src: str, z: str) -> np.ndarray:
    return np.array([r["src"][src]["imp"][z] if r["src"].get(src, {}).get("status") == "ok" else np.nan for r in recs])


def ok_of(*xs) -> np.ndarray:
    m = np.ones(len(xs[0]), bool)
    for x in xs:
        m &= np.isfinite(x)
    return m


def S(est, vec, n=None) -> dict:
    return ST.summary(est, vec, n)


def lo(vec):
    return ST.ci(vec)[0]


def hi(vec):
    return ST.ci(vec)[1]


def verdict(ds_vec, dn_vec, det: Dict[str, bool]) -> dict:
    """§3.4: the class verdict from the two numerators' draw vectors and the per-class detection."""
    ls, hs, ln, hn = lo(ds_vec), hi(ds_vec), lo(dn_vec), hi(dn_vec)

    def direction(lw, hg):
        return "U" if ST.gt(lw, 0.0) else "M" if ST.lt(hg, 0.0) else "none"

    d_s, d_n = direction(ls, hs), direction(ln, hn)
    if d_s == "U" and d_n == "U":
        v = "U-FIRST"
    elif d_s == "M" and d_n == "M":
        v = "M-FIRST"
    elif det["M"] and det["U"]:
        v = "SHARED"
    else:
        v = "NOT DETECTED"
    q = {"direction_reveal_only": d_s, "direction_hide_only": d_n,
         "sides_disagree": {d_s, d_n} == {"U", "M"},
         "material_reveal_only": bool(ST.gt(ls, MATERIAL) or ST.lt(hs, -MATERIAL)),
         "material_hide_only": bool(ST.gt(ln, MATERIAL) or ST.lt(hn, -MATERIAL)),
         "close": bool(ST.inside([ls, hs], CLOSE) and ST.inside([ln, hn], CLOSE)),
         "classes_detected": [x for x in ("M", "U") if det[x]]}
    label = v + (" (SIDES-DISAGREE)" if v == "SHARED" and q["sides_disagree"] else "")
    return {"verdict": v, "label": label, "qualifiers": q}


def read_frame(recs: List[dict], pooled: bool, z: str) -> dict:
    """Every §3-§4 quantity of one frame (a checkpoint, or both pooled) under scorer z."""
    bt = ST.Boot([r["battle"] for r in recs], [r["cp"] for r in recs] if pooled else None)
    o, f = imp(recs, "O", z), imp(recs, "F", z)
    fm, fu, om, ou = (imp(recs, a, z) for a in ("F+M", "F+U", "O-M", "O-U"))
    ms, mn = ok_of(fu, fm), ok_of(om, ou)
    ds, dn = bt.mean(fu - fm, ms), bt.mean(om - ou, mn)
    res = {"n_states": len(recs), "reveal_only Δs = IMP(F+U) − IMP(F+M)": S(*ds, int(ms.sum())),
           "hide_only Δn = IMP(O−M) − IMP(O−U)": S(*dn, int(mn.sum()))}
    det, cls = {}, {}
    for X, fx, ox in (("M", fm, om), ("U", fu, ou)):
        sig = bt.mean(fx - f, ok_of(fx, f))
        nu = bt.mean(o - ox, ok_of(o, ox))
        cls[f"σ_{X} = IMP(F+{X}) − IMP(F)"] = S(*sig, int(ok_of(fx, f).sum()))
        cls[f"ν_{X} = IMP(O) − IMP(O−{X})"] = S(*nu, int(ok_of(o, ox).sum()))
        det[X] = bool(ST.gt(lo(sig[1]), 0.0) or ST.gt(lo(nu[1]), 0.0))
    res["class_contrasts"] = cls
    res.update(verdict(ds[1], dn[1], det))
    base = ok_of(o, f)
    g = bt.mean(o - f, base)
    res["G = IMP(O) − IMP(F)"] = S(*g, int(base.sum()))
    res["GAP"] = bool(ST.gt(lo(g[1]), 0.0))
    res["IMP per source"] = {s: S(*bt.mean(x, ok_of(x)), int(ok_of(x).sum()))
                             for s, x in (("O", o), ("F", f), ("F+M", fm), ("F+U", fu), ("O-M", om), ("O-U", ou))}
    sh = {}
    for X, fx, ox in (("M", fm, om), ("U", fu, ou)):
        sh[f"s_{X}"] = S(*bt.ratio(fx - f, o - f, base & ok_of(fx)))
        sh[f"n_{X}"] = S(*bt.ratio(o - ox, o - f, base & ok_of(ox)))
    sh["s_U − s_M"] = S(*bt.ratio(fu - fm, o - f, base & ms))
    sh["n_U − n_M"] = S(*bt.ratio(om - ou, o - f, base & mn))
    s_m, n_m = bt.ratio(fm - f, o - f, base & ok_of(fm)), bt.ratio(o - om, o - f, base & ok_of(om))
    sh["B3 (s_M ≥ 0.50, lower ≥ 0.25)"] = bool(ST.ge(s_m[0], 0.5) and ST.ge(lo(s_m[1]), 0.25))
    sh["B4 (n_M ≥ 0.50, lower ≥ 0.25)"] = bool(ST.ge(n_m[0], 0.5) and ST.ge(lo(n_m[1]), 0.25))
    sh["shares_uninformative (no GAP)"] = not res["GAP"]
    res["shares (descriptive)"] = sh
    res["_draws"] = {"ds": ds[1], "dn": dn[1]}
    return res


def by_groups(recs: List[dict], z: str) -> dict:
    """§4.5 (descriptive): Δs, Δn by k bin and by traced outcome, pooled."""
    bt = ST.Boot([r["battle"] for r in recs], [r["cp"] for r in recs])
    fm, fu, om, ou = (imp(recs, a, z) for a in ("F+M", "F+U", "O-M", "O-U"))
    ms, mn = ok_of(fu, fm), ok_of(om, ou)
    out = {}
    groups = {f"k {b}": (lambda r, b=b: r["k_bin"] == b) for b in ("1-2", "3", "4", "5", "6")}
    groups.update({"traced win": lambda r: str(r["outcome"]).startswith("win"),
                   "traced loss/draw": lambda r: not str(r["outcome"]).startswith("win")})
    for g, pred in groups.items():
        sel = np.array([pred(r) for r in recs])
        out[g] = {"n": int(sel.sum()), "Δs": S(*bt.mean(fu - fm, sel & ms)), "Δn": S(*bt.mean(om - ou, sel & mn))}
    k = np.array([r["k"] for r in recs], dtype=np.float64)
    slots = np.array([r["move_slots"] for r in recs], dtype=np.float64)
    Z = np.column_stack([6 - k, slots])
    reg = {}
    for lab, y, m in (("Δs", fu - fm, ms), ("Δn", om - ou, mn)):
        mm = m & np.isfinite(slots)
        if mm.sum() > 10:
            beta, bs = bt.ols(y, Z, mm)
            # the k-adjusted mean: the fitted value at the MEAN never-seen count and move slots of the frame
            zbar = np.array([1.0, float((6 - k)[mm].mean()), float(slots[mm].mean())])
            reg[lab] = {"coef": {nm: round(float(beta[j]), 5) for j, nm in
                                 enumerate(("intercept", "never_seen_mons", "move_slots"))},
                        "adjusted_at_frame_mean": S(float(zbar @ beta), bs @ zbar)}
    out["volume_regression"] = reg
    return out


def validity(recs: List[dict]) -> dict:
    """§5: per-arm statuses, the 1 % defect flag, the k = 6 identities (outcome for outcome)."""
    st = {}
    for s in SOURCES:
        c = {}
        for r in recs:
            x = r["src"].get(s, {"status": "no truth"} if s != "O" else {"status": "missing"})
            c[x["status"]] = c.get(x["status"], 0) + 1
        st[s] = c
    defects = {a: [r["id"] for r in recs if r["src"].get(a, {}).get("status") == "defect"] for a in E.ARMS}
    n = max(1, len(recs))
    ident = {}
    for a, b in (("F+U", "F"), ("O-U", "O")):
        k6_all = [r for r in recs if r["k"] == 6 and "O" in r["src"] and r["src"]["O"].get("status") == "ok"]
        # a status difference (one side built, the other not) is a difference; both not ok is consistent, uncompared
        diff = [r["id"] for r in k6_all if (r["src"].get(a, {}).get("status") == "ok")
                != (r["src"].get(b, {}).get("status") == "ok")]
        k6 = [r for r in k6_all if r["src"].get(a, {}).get("status") == "ok"
              and r["src"].get(b, {}).get("status") == "ok"]
        for r in k6:
            xa = r["src"][a]["out"]
            xb = r["truth_out8"] if b == "O" else r["src"][b]["out"]
            if xa != xb:
                diff.append(r["id"])
        ident[f"{a} ≡ {b} at k = 6"] = {"states": len(k6), "differing": len(diff), "examples": diff[:5],
                                        "holds": not diff}
    return {"status_counts": st, "defects": defects,
            "defect_share_over_1pct": {a: len(v) / n > 0.01 for a, v in defects.items()},
            "F_excluded_over_10pct": (st["F"].get("excluded", 0) + st["F"].get("defect", 0)) / n > 0.10,
            "k6_identities": ident, "valid": all(v["holds"] for v in ident.values())}


def tier1_populations(e2_draws: Dict[str, np.ndarray], z: str) -> dict:
    """§4.4 (descriptive): E2's numerators minus Tier 1's on its bot-game and snapshot-game states (registered 600 +
    supplement, pooled), independent samples on disjoint battles: a difference of draw vectors, Tier 1's draws
    permuted once (rng 20261016) so the two equal-seed bootstraps share no draw index."""
    import red_team as RT
    import unblind_t1 as U

    tab, name, _ = U.load()
    P = U.frames(tab, "all")["pooled"]
    I = RT.Imps(P, name, RT.truth_arrays_all(), ["F+M", "F+U", "O-M", "O-U"])
    fm, fu, om, ou = (I(x, z) for x in ("F+M", "F+U", "O-M", "O-U"))
    ms, mn = ok_of(fu, fm), ok_of(om, ou)
    sent = P.sel(lambda r: RT.family(r) == "sentinel")
    perm = np.random.default_rng(20261016).permutation(ST.B)
    out = {}
    for g, sel in (("Tier-1 bot-game states", ~sent), ("Tier-1 snapshot-game states", sent)):
        es, vs = P.bt.mean(fu - fm, sel & ms)
        en, vn = P.bt.mean(om - ou, sel & mn)
        out[g] = {"n": int(sel.sum()), "Δs": S(es, vs), "Δn": S(en, vn),
                  "E2 − this: Δs": S(e2_draws["ds_est"] - es, e2_draws["ds"] - vs[perm]),
                  "E2 − this: Δn": S(e2_draws["dn_est"] - en, e2_draws["dn"] - vn[perm])}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true")
    ap.add_argument("--allow-incomplete-numbers", action="store_true")
    ap.add_argument("--no-tier1", action="store_true", help="skip §4.4 (needs Tier 1's frozen table)")
    a = ap.parse_args()
    recs = {cp: records(cp) for cp in E.ORDER}
    allr = [r for cp in E.ORDER for r in recs[cp]]
    val = validity(allr)
    complete = all(r["src"].get(s, {}).get("status") not in (None, "missing") for r in allr for s in SOURCES) and \
        all(len(recs[cp]) == 5 * E.PER_QUINTILE for cp in E.ORDER)
    if not complete and not (a.partial and a.allow_incomplete_numbers):
        print(json.dumps({"complete": False, "status_counts": val["status_counts"]}, indent=1))
        if not a.partial:
            raise SystemExit("the read is incomplete (pass --partial for counts only)")
        return
    res = {"complete": complete, "preregistration": "PREREGISTRATION.md (5881ee83)", "validity": val, "scorers": {}}
    for z in scorers():
        fz = {"pooled": read_frame(allr, True, z), **{cp: read_frame(recs[cp], False, z) for cp in E.ORDER}}
        res["scorers"][z] = fz
    prim = res["scorers"][PRIMARY]["pooled"]
    tf = res["scorers"][TIE_FREE]["pooled"]
    res["PRIMARY"] = {"verdict": prim["label"], "scorer_dependent": tf["verdict"] != prim["verdict"],
                      "tie_free_verdict": tf["label"], "per_checkpoint": {cp: res["scorers"][PRIMARY][cp]["label"]
                                                                         for cp in E.ORDER},
                      "valid": val["valid"],
                      "quote_as": ("computed" if val["valid"] and not any(val["defect_share_over_1pct"].values())
                                   else "computed, guard defects open" if val["valid"] else "INVALID (k = 6 identity)")}
    res["by_groups (descriptive)"] = {z: by_groups(allr, z) for z in (PRIMARY, TIE_FREE)}
    if not a.no_tier1:
        res["tier1_populations (descriptive)"] = {}
        for z in (PRIMARY, TIE_FREE):
            p = res["scorers"][z]["pooled"]
            draws = {"ds": p["_draws"]["ds"], "dn": p["_draws"]["dn"],
                     "ds_est": p["reveal_only Δs = IMP(F+U) − IMP(F+M)"]["mean"],
                     "dn_est": p["hide_only Δn = IMP(O−M) − IMP(O−U)"]["mean"]}
            res["tier1_populations (descriptive)"][z] = tier1_populations(draws, z)
    for z in res["scorers"]:
        for fr in res["scorers"][z].values():
            fr.pop("_draws", None)
    OUT.write_text(json.dumps(res, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps(res["PRIMARY"], indent=1))


if __name__ == "__main__":
    main()
