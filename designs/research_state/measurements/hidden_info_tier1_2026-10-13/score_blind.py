"""TIER 1's READ, part 2: the BLINDED TABLE (registration §12.10). It reads every arm by its opaque CODE only and never
opens the arm key. Per checkpoint and state: the stage facts, the referee T, and for every SOURCE (the truth O, the
pilot's v2 base P, each coded arm Tnn, each hand-eval code H…) whether it has a valid row and each scorer's IMP and pick
(`t1_data`); for every world-based source (each code and P) the BELIEF-QUALITY DOSE family with that source as the
honest base (§6.9: M8 on a 0.01 grid for LEAF-K8 and the 0.05 grid for TRUNC-4-K8 / CRITIC; the literal K = 5 design M5,
its placebos M5W and M5X, on the owner's 7-point grid). A dose family is computed for EVERY world source alike, so the
table does not reveal which code is the honest base F.

    python -I score_blind.py [--partial]      → <archive>/blinded_table.pkl, result_blinded.json (this folder)

``blinded_table.sha256`` (this folder) freezes the table before `unblind_t1.py` maps codes to labels. ``--partial``
reads whatever exists (a progress read: prints counts only, writes nothing).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_common as T  # noqa: E402
import t1_data as D  # noqa: E402
import t1_stats as ST  # noqa: E402

P7 = np.array([0.0, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0])           # the owner's grid (§6.9)
PF = np.round(np.linspace(0.0, 1.0, 21), 2)                      # 0.05 (the registered interpolation grid)
P101 = np.round(np.linspace(0.0, 1.0, 101), 2)                   # 0.01 (LEAF-K8 only: TODAY + δ, README §6)
SUB4 = np.array(list(combinations(range(8), 4)))                 # 70 honest 4-subsets
OTHER4 = {w: np.array(list(combinations([j for j in range(8) if j != w], 4))) for w in range(8)}   # 35 each
DK_SCORERS = ("LEAF-K8", "TRUNC-4-K8", "CRITIC")
TABLE = T.ARCH / "blinded_table.pkl"


def imp_of(S, Tp):
    """S [..., A] in POLICY order (index 0 = top1): pick = the first action within TIE of the max."""
    m = S.max(-1, keepdims=True)
    idx = (S >= m - D.TIE).argmax(-1)
    return Tp[idx] - Tp[0]


def dk_family(S1: dict, Sb: dict, worst: int | None) -> dict:
    """The dose family with honest base ``Sb`` (world arrays) and the truth ``S1`` (truth arrays), in policy order."""
    ordr = np.array(S1["order"])
    Tp = S1["T"][ordr]
    comp = {
        "LEAF-K8": (S1["W"][ordr][:, :8].mean(1), S1["W"][ordr].T, Sb["W"][ordr].T),
        "TRUNC-4-K8": (S1["Xd"][4][ordr][:, :8].mean(1), S1["Xd"][4][ordr].T, Sb["Xd"][4][ordr].T),
        "CRITIC": (S1["crit"][ordr].mean(1), S1["crit"][ordr].T, Sb["crit"][ordr].T),
    }
    out = {}
    for z, (Tm, Ts, H) in comp.items():
        Hm = H.mean(0)
        grid = P101 if z == "LEAF-K8" else PF
        out[f"M8|{z}"] = imp_of(grid[:, None] * Tm[None, :] + (1 - grid)[:, None] * Hm[None, :], Tp)
        Hsub = H[SUB4].mean(1)                                                       # [70, A]
        p = P7[:, None, None, None]
        out[f"M5|{z}"] = imp_of(p * Ts[None, :, None, :] + (1 - p) * Hsub[None, None, :, :], Tp).mean((1, 2))
        heavy = np.concatenate([np.repeat(H[w_][None, :], len(J), axis=0) for w_, J in OTHER4.items()])   # [280, A]
        light = np.concatenate([H[J].mean(1) for _w, J in OTHER4.items()])
        out[f"M5W|{z}"] = imp_of(P7[:, None, None] * heavy[None] + (1 - P7)[:, None, None] * light[None], Tp).mean(1)
        if worst is not None:
            J = OTHER4[worst]
            hv = np.repeat(H[worst][None, :], len(J), axis=0)
            lt_ = H[J].mean(1)
            out[f"M5X|{z}"] = imp_of(P7[:, None, None] * hv[None] + (1 - P7)[:, None, None] * lt_[None], Tp).mean(1)
    return out


def fact_hits(wr: dict) -> list:
    """Amendment 3's descriptive belief proxy, from the builder's provenance (no outcome), summed over the worlds:
    [seen mons' unrevealed TRUE moves held, their slots, never-seen slots holding a TRUE never-seen species, never-seen
    slots]."""
    mh = ms = sh = sn = 0
    for fl in wr.get("facts") or []:
        for f in fl:
            if f.get("seen"):
                mv = f.get("moves", {})
                if "true_hits" in mv:
                    mh += int(mv["true_hits"])
                    ms += int(mv.get("slots", 0))
            else:
                sn += 1
                sh += int(bool(f.get("species_fact", {}).get("match")))
    return [mh, ms, sh, sn]


def score_cp(cp: str, partial: bool) -> dict:
    meta = D.stage_meta(cp)
    r1 = {**D.phase1_rows(cp, "registered"), **D.phase1_rows(cp, "supp")}
    P = D.p_rows(cp)
    codes = D.coded_dirs(cp)
    rows = {c: D.code_rows(cp, c) for c in codes}
    worlds = {c: D.code_worlds(cp, c) for c in codes}
    he = {h: D.he_rows(cp, h) for h in D.HE_CODES}
    recs, counts, tally = [], {"states": len(meta), "no_truth": [], "truth_refused": []}, {}
    cnt = {s: {"ok": 0, "missing": 0, "excluded": 0, "refused": 0} for s in ["P"] + codes + list(D.HE_CODES)}
    cost = {c: {"play_cpu_s": 0.0, "crit_cpu_s": 0.0, "states": 0} for c in codes}
    wcost = {c: {"cpu_s": 0.0, "states": 0} for c in codes}
    for did in sorted(meta):
        m = meta[did]
        r = r1.get(did)
        if r is None:
            counts["no_truth"].append(did)
            continue
        if not r.get("ok"):
            counts["truth_refused"].append(did)
            continue
        S1 = D.truth_arrays(r, m)
        Tv, t1 = S1["T"], S1["top1"]
        rec = {k: m[k] for k in ("set", "battle", "k", "turn", "material", "k_bin", "turn_bin", "mat_bin", "outcome",
                                 "opp_class", "cycle_opp")}
        rec.update({"id": did, "cp": cp, "n_legal": len(S1["acts"]), "upper": float(Tv.max() - Tv[t1]), "src": {},
                    "dk": {}, "w": {}, "hits": {}, "other_open": bool(r.get("other_open"))})

        def put(name, vals):
            rec["src"][name] = {"ok": True,
                                "pick": {z: int(D.pick(v, S1["order"])) for z, v in vals.items()},
                                "imp": {z: float(Tv[D.pick(v, S1["order"])] - Tv[t1]) for z, v in vals.items()}}

        put("O", D.scorer_values(S1))
        world_sources = []
        if m["set"] == "registered":
            rp = P.get(did)
            if rp is None:
                cnt["P"]["missing"] += 1
            elif not rp.get("ok"):
                cnt["P"]["refused"] += 1
            else:
                Sp = D.world_arrays(rp, S1["acts"], S1["turn"])
                put("P", D.scorer_values(Sp))
                world_sources.append(("P", Sp, None))
                cnt["P"]["ok"] += 1
        for c in codes:
            rc = rows[c].get(did)
            if rc is None:
                cnt[c]["missing"] += 1
                continue
            if not rc.get("ok"):
                cnt[c]["excluded" if rc.get("excluded") else "refused"] += 1
                rec["src"][c] = {"ok": False, "excluded": bool(rc.get("excluded")), "error": str(rc.get("error"))[:200]}
                continue
            Sc = D.world_arrays(rc, S1["acts"], S1["turn"])
            put(c, D.scorer_values(Sc))
            cnt[c]["ok"] += 1
            cost[c]["play_cpu_s"] += rc["cost"]["play"]["cpu_s"]
            cost[c]["crit_cpu_s"] += rc["cost"]["crit"]["cpu_s"]
            cost[c]["states"] += 1
            wr = worlds[c].get(did, {})
            if wr.get("cost"):
                wcost[c]["cpu_s"] += wr["cost"]["cpu_s"]
                wcost[c]["states"] += 1
            corr = wr.get("correct")
            worst = int(np.argmin(corr)) if corr else None              # ties: the lowest world index
            world_sources.append((c, Sc, worst))
            rec["hits"][c] = fact_hits(wr)
        ordr = np.array(S1["order"])
        rec["Tp"] = S1["T"][ordr]
        for name, Sb, worst in world_sources:
            rec["dk"][name] = {k: np.asarray(v, dtype=np.float64) for k, v in dk_family(S1, Sb, worst).items()}
            # Amendment 3 (§12.14): every world source's per-world arrays in POLICY order, so the per-class dose
            # curves (two arms' worlds re-weighted) are computed after unblinding from this frozen table
            rec["w"][name] = {"W": Sb["W"][ordr].astype(np.float32), "X4": Sb["Xd"][4][ordr].astype(np.float32),
                              "crit": Sb["crit"][ordr].astype(np.float32)}
        for h in D.HE_CODES:
            rh = he[h].get(did)
            if rh is None:
                if m["set"] == "registered":
                    cnt[h]["missing"] += 1
                continue
            if not rh.get("ok"):
                cnt[h]["refused"] += 1
                rec["src"][h] = {"ok": False, "error": str(rh.get("error"))[:200]}
                continue
            put(h, D.he_arrays(rh, S1["acts"], S1["turn"], tally))
            cnt[h]["ok"] += 1
        recs.append(rec)
    counts.update({"sources": cnt, "cost_play": cost, "cost_worlds": wcost, "he_tally": tally,
                   "no_truth_n": len(counts["no_truth"]), "truth_refused_n": len(counts["truth_refused"])})
    if not partial and counts["no_truth"]:
        raise SystemExit(f"{cp}: {len(counts['no_truth'])} states have no truth row (use --partial)")
    return {"recs": recs, "counts": counts, "codes": codes}


def blinded_summary(tab: dict) -> dict:
    """Per source × scorer: the mean IMP per checkpoint and pooled (registered 600 and all states), with intervals."""
    out = {}
    for which in ("registered", "all"):
        res = {}
        for cp in list(T.CPS) + ["pooled"]:
            recs = [r for c in (T.CPS if cp == "pooled" else [cp]) for r in tab[c]["recs"]
                    if which == "all" or r["set"] == "registered"]
            if not recs:
                continue
            bt = ST.Boot([r["battle"] for r in recs], [r["cp"] for r in recs] if cp == "pooled" else None)
            srcs = sorted({s for r in recs for s in r["src"]})
            res[cp] = {}
            for s in srcs:
                names = sorted({z for r in recs if r["src"].get(s, {}).get("ok") for z in r["src"][s]["imp"]})
                res[cp][s] = {}
                for z in names:
                    x = np.array([r["src"][s]["imp"][z] if r["src"].get(s, {}).get("ok") else np.nan for r in recs])
                    est, vec = bt.mean(x)
                    res[cp][s][z] = ST.summary(est, vec, int(np.isfinite(x).sum()))
        out[which] = res
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true")
    a = ap.parse_args()
    tab = {cp: score_cp(cp, a.partial) for cp in T.CPS}
    if a.partial:
        print(json.dumps({cp: {"states_scored": len(tab[cp]["recs"]), "no_truth": tab[cp]["counts"]["no_truth_n"],
                               "sources": tab[cp]["counts"]["sources"]} for cp in tab}, indent=1, sort_keys=True))
        return
    b = pickle.dumps(tab, protocol=4)
    TABLE.write_bytes(b)
    sha = hashlib.sha256(b).hexdigest()
    (T.HERE / "blinded_table.sha256").write_text(sha + "  blinded_table.pkl (archive)\n")
    doc = {"table_sha256": sha, "counts": {cp: tab[cp]["counts"] for cp in tab},
           "codes": {cp: tab[cp]["codes"] for cp in tab}, "imp": blinded_summary(tab)}
    for cp in doc["counts"]:
        doc["counts"][cp] = {k: v for k, v in doc["counts"][cp].items() if k not in ("no_truth", "truth_refused")}
    (T.HERE / "result_blinded.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print("blinded table:", sha)


if __name__ == "__main__":
    main()
