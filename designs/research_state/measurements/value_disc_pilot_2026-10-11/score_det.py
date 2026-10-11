"""PHASE 2's READ (README §8): the DETERMINIZED (honest-information) scorers against Phase 1's TRUE-world
truth (seeds 16–39), paired with Phase 1's ORACLE scorers on the same states.

    python -I score_det.py <cp> [<cp> ...] [--out-json result_det.json] [--out-md result_det.md] [--partial]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score as S1  # noqa: E402

HERE = Path(__file__).resolve().parent
ARCH = S1.ARCH
DEC = S1.DEC


def det_rows(cp):
    rows = {}
    for p in sorted((ARCH / cp / "det").glob("rows_*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rows[r["id"]] = r
    return rows


def det_arrays(r1, rd, e):
    """Phase 1's state arrays with the leaf / trace / critic replaced by the determinized ones."""
    S = S1.state_arrays(r1, e)              # truth, logits, order, top1 from Phase 1 (true world)
    acts = S["acts"]
    if sorted(rd["out"], key=int) != acts:
        raise RuntimeError(f"{r1['id']}: Phase 2 actions differ from Phase 1's")
    O = np.array([[DEC[ch] for ch in rd["out"][a]] for a in acts])           # [A, 16]
    T0 = int(rd["turn"])
    nw = O.shape[1]
    Xd = {d: np.empty((len(acts), nw)) for d in S1.TRUNC_D}
    X1p = np.empty((len(acts), nw))
    rows_b = np.empty((len(acts), nw))
    frac_d = {d: np.empty((len(acts), nw)) for d in S1.TRUNC_D}
    for i, a in enumerate(acts):
        for j in range(nw):
            t = rd["trace"][a][j]
            tt, vv = t["t"], t["v"]
            term = S1.w(O[i, j])
            X1p[i, j] = vv[0] if vv else term
            rows_b[i, j] = max(1, len(vv))
            for d in S1.TRUNC_D:
                k = next((q for q, x in enumerate(tt) if x >= T0 + d), None)
                Xd[d][i, j] = vv[k] if k is not None else term
                frac_d[d][i, j] = ((k + 1) if k is not None else len(vv)) / max(1, len(vv))
    cost = rd["cost"]["play"]
    D = dict(S)
    D.update({"O": O, "W": S1.w(O), "Xd": Xd, "X1p": X1p,
              "crit": np.array([S1.w(np.mean(rd["crit"][a]["q"])) for a in acts]),
              "cpu_row": cost["cpu_s"] / max(1, cost["rows"]), "rows": rows_b / rows_b.sum() * cost["rows"],
              "frac_d": frac_d, "crit_cpu": rd["cost"]["crit"]["cpu_s"],
              "cpu_playout": cost["cpu_s"] / max(1, cost["playouts"])})
    return S, D


def read_cp(cp, partial):
    st, r1, missing1, _ = S1.load(cp, True)
    rd = det_rows(cp)
    worlds = {}
    for p in sorted((ARCH / cp / "det").glob("worlds_*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                x = json.loads(line)
                worlds[x["id"]] = x
    E = st["states"]
    ids = sorted(i for i in E if r1.get(i, {}).get("ok") and rd.get(i, {}).get("ok"))
    excluded = sorted(i for i in E if r1.get(i, {}).get("ok") and not rd.get(i, {}).get("ok"))
    if (missing1 or len(ids) + len(excluded) < len(E)) and not partial:
        raise SystemExit(f"{cp}: incomplete rows (use --partial)")
    X, Xo, costs = [], [], {}
    for did in ids:
        So, Sd = det_arrays(r1[did], rd[did], E[did])
        do, _ = S1.decisions(So)
        n16 = S1.N_LEAF
        S1.N_LEAF = Sd["W"].shape[1]                 # sequential halving's per-action cap = the worlds played
        try:
            dd, spd = S1.decisions(Sd)
        finally:
            S1.N_LEAF = n16
        T, t1 = So["truth"], So["top1"]
        X.append({k: T[v] - T[t1] for k, v in dd.items()})
        Xo.append({k: T[v] - T[t1] for k, v in do.items()})
        for k in dd:
            costs.setdefault(k, []).append(S1.cost_of(Sd, k, spd))
    e = [E[i] for i in ids]
    bt = S1.Boot([x["battle"] for x in e], S1.reweights(st, e))
    nw = len(rd[ids[0]]["out"][next(iter(rd[ids[0]]["out"]))])

    def fits(k):
        """A key whose per-action sample count exceeds the worlds played is not computable."""
        if k.startswith("SH-"):
            return int(k.split("n*")[1]) <= nw
        name = k.split("|")[1]
        return name == "CRITIC" or int(name.rsplit("K", 1)[1]) <= nw
    keys = sorted(k for k in X[0] if fits(k))
    A = {k: np.array([p[k] for p in X]) for k in keys}
    O = {k: np.array([p[k] for p in Xo]) for k in keys}
    res = {"cp": cp, "n_states": len(ids), "excluded_no_world_set": len(excluded),
           "n_battles": len(bt.ub), "imp_det": {}, "imp_oracle_same_states": {}, "cost_det": {}, "contrasts": {},
           "strata": {}}
    for k in keys:
        res["imp_det"][k] = bt.stat(A[k])
        res["imp_oracle_same_states"][k] = bt.stat(O[k])
        res["cost_det"][k] = round(float(np.mean(costs[k])), 3)
    C = {"R-HONEST: det LEAF-K8 - det CRITIC": A["ARGMAX|LEAF-K8"] - A["ARGMAX|CRITIC"],
         "R-INFO: oracle LEAF-K8 - det LEAF-K8": O["ARGMAX|LEAF-K8"] - A["ARGMAX|LEAF-K8"],
         "det LEAF-K8 - oracle CRITIC": A["ARGMAX|LEAF-K8"] - O["ARGMAX|CRITIC"],
         "det CRITIC - oracle CRITIC": A["ARGMAX|CRITIC"] - O["ARGMAX|CRITIC"],
         "det LEAF-K8 - det LEAF-K4": A["ARGMAX|LEAF-K8"] - A["ARGMAX|LEAF-K4"],
         "det GUMBEL - ARGMAX at LEAF-K4": A["GUMBEL|LEAF-K4"] - A["ARGMAX|LEAF-K4"],
         "det SH-4 (B=n*2) - ARGMAX LEAF-K2": A["SH-4|B=n*2"] - A["ARGMAX|LEAF-K2"]}
    for K in (1, 4):
        C[f"R-INFO at K{K}: oracle - det LEAF-K{K}"] = O[f"ARGMAX|LEAF-K{K}"] - A[f"ARGMAX|LEAF-K{K}"]
    for d in S1.TRUNC_D:
        C[f"det TRUNC-{d}-K8 - det CRITIC"] = A[f"ARGMAX|TRUNC-{d}-K8"] - A["ARGMAX|CRITIC"]
        C[f"det TRUNC-{d}-K8 - det LEAF-K8"] = A[f"ARGMAX|TRUNC-{d}-K8"] - A["ARGMAX|LEAF-K8"]
    C["det CV-K8 - det LEAF-K8"] = A["ARGMAX|CV-K8"] - A["ARGMAX|LEAF-K8"]
    for k, v in C.items():
        res["contrasts"][k] = bt.stat(v)
    # the share of the oracle leaf's improvement the honest leaf keeps (ratio of means, same draws)
    for K in (1, 4, 8):
        a, o = A[f"ARGMAX|LEAF-K{K}"], O[f"ARGMAX|LEAF-K{K}"]
        nb = len(bt.ub)
        Sa = np.bincount(bt.bi, weights=a, minlength=nb)
        So_ = np.bincount(bt.bi, weights=o, minlength=nb)
        ratio = (bt.C @ Sa) / (bt.C @ So_)
        res[f"retained_share_K{K}"] = {"mean": round(float(Sa.sum() / So_.sum()), 4),
                                       "ci": np.percentile(ratio, [2.5, 97.5]).round(4).tolist()}
    for lab, lo, hi in (("unrev_0-1", 0, 1), ("unrev_2-3", 2, 3), ("unrev_4-5", 4, 5)):
        m = np.array([lo <= x["unrevealed"] <= hi for x in e])
        if m.sum():
            res["strata"][lab] = {"det LEAF-K8": bt.stat(A["ARGMAX|LEAF-K8"], sel=m),
                                  "oracle LEAF-K8": bt.stat(O["ARGMAX|LEAF-K8"], sel=m),
                                  "det CRITIC": bt.stat(A["ARGMAX|CRITIC"], sel=m),
                                  "oracle - det LEAF-K8": bt.stat(O["ARGMAX|LEAF-K8"] - A["ARGMAX|LEAF-K8"], sel=m)}
    v = {}
    lo, hi = res["contrasts"]["R-HONEST: det LEAF-K8 - det CRITIC"]["ci"]
    v["R-HONEST"] = "LEAF-AHEAD" if lo > S1.EDGE else ("CRITIC-EQUIVALENT" if lo > -0.01 and hi < 0.01 else "NOT DETECTED")
    lo, hi = res["contrasts"]["R-INFO: oracle LEAF-K8 - det LEAF-K8"]["ci"]
    v["R-INFO"] = "INFO-COSTS" if lo > S1.EDGE else ("INFO-FREE" if lo > -0.01 and hi < 0.01 else "NOT DETECTED")
    res["verdicts"] = v
    wl = [worlds[i] for i in ids if i in worlds]
    res["worlds"] = {"tries_mean": round(float(np.mean([t for x in wl for t in x["tries"]])), 3),
                     "rewritten_moves": int(sum(x["leak"]["rewritten_moves"] for x in wl)),
                     "rewritten_switches": int(sum(x["leak"]["rewritten_switches"] for x in wl)),
                     "ambiguous_moves_kept": int(sum(len(x["leak"]["moves"]) for x in wl)),
                     "root_choice_kept": int(sum(x["leak"]["root_choice"] is not None for x in wl)),
                     "new_slots_mean": round(float(np.mean([s["new_slots"] for x in wl for s in x["stats"]])), 3),
                     "redrawn_move_slots_mean": round(float(np.mean([s["redrawn_move_slots"] for x in wl for s in x["stats"]])), 3)}
    return res


def md(results):
    L = ["# Value-discrimination pilot, PHASE 2 (determinized worlds): result tables", "",
         "IMP against the TRUE world's 24 truth seeds; det = honest-information worlds (Smogon-prior samples of the",
         "opponent's hidden team, rejected unless the searched side's observations replay byte-equal).", ""]
    for r in results:
        L += [f"## {r['cp']} ({r['n_states']} states, {r['n_battles']} battles; excluded {r['excluded_no_world_set']})",
              "", "Verdicts: " + ", ".join(f"{k} **{x}**" for k, x in r["verdicts"].items()), "",
              "| scorer (ARGMAX) | det IMP [95 %] | oracle IMP, same states | det CPU-s / decision |", "|---|---|---|---|"]
        for k, x in r["imp_det"].items():
            if not k.startswith("ARGMAX|"):
                continue
            o = r["imp_oracle_same_states"][k]
            L.append(f"| {k} | {x['mean']:+.4f} [{x['ci'][0]:+.4f}, {x['ci'][1]:+.4f}] | {o['mean']:+.4f} | "
                     f"{r['cost_det'][k]:.2f} |")
        L += ["", "| rule (det) | IMP [95 %] |", "|---|---|"]
        for k, x in r["imp_det"].items():
            if k.startswith(("GUMBEL|LEAF", "SH-", "PRIOR2|LEAF-K8", "GATE")):
                L.append(f"| {k} | {x['mean']:+.4f} [{x['ci'][0]:+.4f}, {x['ci'][1]:+.4f}] |")
        L += ["", "| contrast | Δ [95 %] |", "|---|---|"]
        for k, x in r["contrasts"].items():
            L.append(f"| {k} | {x['mean']:+.4f} [{x['ci'][0]:+.4f}, {x['ci'][1]:+.4f}] |")
        for K in (1, 4, 8):
            x = r[f"retained_share_K{K}"]
            L.append(f"| retained share, LEAF-K{K} (det / oracle) | {x['mean']:.3f} {x['ci']} |")
        L += ["", "| unrevealed opponent slots | n | det LEAF-K8 | oracle LEAF-K8 | det CRITIC | oracle − det |",
              "|---|---|---|---|---|---|"]
        for s, d in r["strata"].items():
            L.append(f"| {s} | {d['det LEAF-K8']['n']} | {d['det LEAF-K8']['mean']:+.4f} | {d['oracle LEAF-K8']['mean']:+.4f} | "
                     f"{d['det CRITIC']['mean']:+.4f} | {d['oracle - det LEAF-K8']['mean']:+.4f} {d['oracle - det LEAF-K8']['ci']} |")
        L += ["", f"Worlds: {r['worlds']}", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cps", nargs="+")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--partial", action="store_true")
    a = ap.parse_args()
    res = [read_cp(c, a.partial) for c in a.cps]
    if a.out_json:
        Path(a.out_json).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    if a.out_md:
        Path(a.out_md).write_text(md(res) + "\n")


if __name__ == "__main__":
    main()
