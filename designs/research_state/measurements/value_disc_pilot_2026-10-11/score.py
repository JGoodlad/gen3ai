"""THE PILOT'S READ (README §3–§5; B3): every scorer × acting rule's DECISION IMPROVEMENT over the policy's
own pick against the 24 held-out truth seeds, by stratum, with battle-clustered bootstrap CIs (paired);
the cost per decision; the pairing diagnostics; the registered verdicts. Pure numpy, no model.

    python -I score.py <cp> [<cp> ...] [--out-json result.json] [--out-md result.md] [--partial]

Reads the archive's ``<cp>/rows/shard_*.jsonl`` and this folder's ``states_<cp>.json``. ``--partial`` reads
whatever rows exist (a progress read, never a verdict); otherwise every sampled state must have a row.
"""
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARCH = Path.home() / "gen3ai_archive" / "value_disc_pilot"
N_LEAF, N_TRUTH = 16, 24
KS = (1, 2, 4, 8, 16)
TRUNC_D = (1, 2, 4)
TRUNC_K = (1, 4, 8, 16)
MIX_K = (1, 4, 8)
SH_K = (1, 2, 4, 8)
GATE_K = (4, 8, 16)
C_VISIT, C_SCALE, C_PUCT = 50.0, 0.1, 1.25
TIE = 1e-6
B = 2000
SEED = 20261011
EDGE = 0.0005


def w(x):
    return (np.asarray(x, dtype=np.float64) + 1.0) / 2.0


DEC = {"+": 1.0, "-": -1.0, "0": 0.0}


def load(cp, partial):
    st = json.loads((HERE / f"states_{cp}.json").read_text())
    rows = {}
    for p in sorted((ARCH / cp / "rows").glob("shard_*.jsonl")):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows[r["id"]] = r
    missing = sorted(set(st["states"]) - set(rows))
    refused = sorted(i for i, r in rows.items() if not r.get("ok"))
    if missing and not partial:
        raise SystemExit(f"{cp}: {len(missing)} sampled states have no row (use --partial for a progress read)")
    return st, rows, missing, refused


# ------------------------------------------------------------------------------------------------
# per-state arrays
# ------------------------------------------------------------------------------------------------

def state_arrays(r, e):
    acts = sorted(r["out"], key=int)
    O = np.array([[DEC[ch] for ch in r["out"][a]] for a in acts])           # [A, 40] ±1/0
    T0 = int(r["turn"])
    n_play = O.shape[1]
    Xd = {d: np.empty((len(acts), n_play)) for d in TRUNC_D}
    X1p = np.empty((len(acts), n_play))
    rows_b = np.empty((len(acts), n_play))
    frac_d = {d: np.empty((len(acts), n_play)) for d in TRUNC_D}
    for i, a in enumerate(acts):
        for s in range(n_play):
            t = r["trace"][a][s]
            tt, vv = t["t"], t["v"]
            term = w(O[i, s])
            X1p[i, s] = vv[0] if vv else term
            rows_b[i, s] = max(1, len(vv))
            for d in TRUNC_D:
                k = next((j for j, x in enumerate(tt) if x >= T0 + d), None)
                Xd[d][i, s] = vv[k] if k is not None else term
                frac_d[d][i, s] = ((k + 1) if k is not None else len(vv)) / max(1, len(vv))
    crit = np.array([w(np.mean(r["crit"][a]["q"])) for a in acts])
    lg = np.array([e["logits"][int(a)] for a in acts], dtype=np.float64)
    order = sorted(range(len(acts)), key=lambda i: (-lg[i], int(acts[i])))
    top1 = order[0]
    if int(acts[top1]) != int(e["top"][0]):
        raise RuntimeError(f"{r['id']}: top1 {acts[top1]} != registered {e['top'][0]}")
    cost = r["cost"]["play"]
    cpu_row = cost["cpu_s"] / max(1, cost["rows"])          # all-in CPU per forward row (sim + python)
    share = rows_b / rows_b.sum() * cost["rows"]             # rows per branch, both sides
    return {"acts": acts, "O": O, "W": w(O), "Xd": Xd, "X1p": X1p, "crit": crit, "lg": lg, "order": order,
            "top1": top1, "truth": w(O[:, N_LEAF:N_LEAF + N_TRUTH]).mean(1), "cpu_row": cpu_row,
            "rows": share, "frac_d": frac_d, "crit_cpu": r["cost"]["crit"]["cpu_s"],
            "cpu_playout": cost["cpu_s"] / max(1, cost["playouts"])}


def scorers(S):
    W, out = S["W"], {"CRITIC": S["crit"]}
    for K in KS:
        out[f"LEAF-K{K}"] = W[:, :K].mean(1)
        out[f"CV-K{K}"] = W[:, :K].mean(1) - (S["X1p"][:, :K].mean(1) - S["crit"])
    for d in TRUNC_D:
        for K in TRUNC_K:
            out[f"TRUNC-{d}-K{K}"] = S["Xd"][d][:, :K].mean(1)
    for K in MIX_K:
        out[f"MIX-K{K}"] = 0.5 * S["crit"] + 0.5 * W[:, :K].mean(1)
    return out


def n_of(name):
    if name == "CRITIC":
        return 8
    return int(name.rsplit("K", 1)[1])


def pick(score, order, allowed=None):
    cand = [i for i in order if allowed is None or i in allowed]
    m = max(score[i] for i in cand)
    return next(i for i in cand if score[i] >= m - TIE)


def gumbel_score(score, lg, idx, nmax):
    v = np.array([score[i] for i in idx])
    lo, hi = v.min(), v.max()
    q = (v - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(v)
    out = np.full(len(score), -np.inf)
    out[idx] = lg[idx] + (C_VISIT + nmax) * C_SCALE * q
    return out


def softmax(lg):
    z = np.exp(lg - lg.max())
    return z / z.sum()


def seq_halving(S, K, m):
    """Gumbel AlphaZero's allocation without noise (README §4). Returns (pick, spent, capped)."""
    W, lg, order = S["W"], S["lg"], S["order"]
    n = len(order)
    budget = n * K
    surv = list(order[: (n if m == "all" else min(m, n))])
    R = max(1, math.ceil(math.log2(len(surv)))) if len(surv) > 1 else 1
    cnt = np.zeros(n, dtype=int)
    spent, capped = 0, False
    while True:
        per = max(1, budget // (R * len(surv)))
        for i in surv:
            add = min(per, N_LEAF - cnt[i])
            capped |= add < per
            cnt[i] += add
            spent += add
        if len(surv) == 1:
            break
        mean = np.array([W[i, :cnt[i]].mean() if cnt[i] else 0.0 for i in range(n)])
        g = gumbel_score(mean, lg, surv, cnt.max())
        keep = math.ceil(len(surv) / 2)
        ranked = sorted(surv, key=lambda i: (-g[i], order.index(i)))
        surv = ranked[:keep]
        if keep == 1:
            break
    return surv[0], spent, capped


def decisions(S):
    """{rule∘scorer: picked action index} plus spends."""
    sc = scorers(S)
    lg, order = S["lg"], S["order"]
    n = len(order)
    allidx = list(range(n))
    pi = softmax(lg)
    out, spend = {}, {}
    for name, v in sc.items():
        out[f"ARGMAX|{name}"] = pick(v, order)
        g = gumbel_score(v, lg, allidx, n_of(name))
        out[f"GUMBEL|{name}"] = pick(g, order)
        K = n_of(name)
        out[f"PUCT|{name}"] = pick(v + C_PUCT * pi * math.sqrt(n * K) / (1 + K), order)
        for k in (2, 3):
            out[f"PRIOR{k}|{name}"] = pick(v, order, set(order[:k]))
    for K in SH_K:
        for m in (4, "all"):
            p, sp, cap = seq_halving(S, K, m)
            out[f"SH-{m}|B=n*{K}"] = p
            spend[f"SH-{m}|B=n*{K}"] = (sp, cap)
    W, t1 = S["W"], S["top1"]
    for K in GATE_K:
        a2 = pick(W[:, :K].mean(1), order)
        dd = W[a2, :K] - W[t1, :K]
        se = max(dd.std(ddof=1) / math.sqrt(K) if K > 1 else 0.0, 0.5 / K)
        out[f"GATE2SE|LEAF-K{K}"] = a2 if (a2 != t1 and dd.mean() >= 2 * se) else t1
    return out, spend


def cost_of(S, key, spend):
    """CPU-s per decision for a rule∘scorer (README §5)."""
    rule, name = key.split("|")
    rows = S["rows"]
    if rule.startswith("SH-"):
        return spend[key][0] * S["cpu_playout"]
    if name == "CRITIC":
        return S["crit_cpu"]
    nA = rows.shape[0]
    allowed = range(nA)
    if rule.startswith("PRIOR"):
        allowed = S["order"][: int(rule[5:])]
    allowed = list(allowed)
    K = n_of(name)
    if name.startswith("TRUNC"):
        d = int(name.split("-")[1])
        r = (rows[allowed, :K] * S["frac_d"][d][allowed, :K]).sum()
    else:
        r = rows[allowed, :K].sum()
    c = r * S["cpu_row"]
    if name.startswith("CV") or name.startswith("MIX"):
        c += S["crit_cpu"]
    return c


# ------------------------------------------------------------------------------------------------
# bootstrap
# ------------------------------------------------------------------------------------------------

class Boot:
    def __init__(self, battles, weights):
        self.ub = sorted(set(battles))
        ix = {b: i for i, b in enumerate(self.ub)}
        self.bi = np.array([ix[b] for b in battles])
        rng = np.random.default_rng(SEED)
        nb = len(self.ub)
        draws = rng.integers(0, nb, size=(B, nb))
        self.C = np.zeros((B, nb))
        for r in range(B):
            np.add.at(self.C[r], draws[r], 1.0)
        self.wt = np.asarray(weights, dtype=np.float64)

    def stat(self, x, sel=None, weighted=False):
        x = np.asarray(x, dtype=np.float64)
        wt = self.wt if weighted else np.ones(len(x))
        m = np.ones(len(x), bool) if sel is None else np.asarray(sel, bool)
        nb = len(self.ub)
        S = np.bincount(self.bi[m], weights=(x * wt)[m], minlength=nb)
        N = np.bincount(self.bi[m], weights=wt[m], minlength=nb)
        est = S.sum() / N.sum()
        num, den = self.C @ S, self.C @ N
        ok = den > 0
        bs = num[ok] / den[ok]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return {"mean": round(float(est), 5), "ci": [round(float(lo), 5), round(float(hi), 5)],
                "n": int(m.sum())}


def verdict3(ci, pos="HELPS", neg="HURTS"):
    lo, hi = ci
    if lo > EDGE and abs(lo - EDGE) > 1e-12:
        return pos
    if hi < -EDGE:
        return neg
    return "NOT DETECTED"


# ------------------------------------------------------------------------------------------------
# the read
# ------------------------------------------------------------------------------------------------

def read_cp(cp, partial):
    st, rows, missing, refused = load(cp, partial)
    E = st["states"]
    ids = sorted(i for i in E if i in rows and rows[i].get("ok"))
    picks, spends, per, costs = [], [], [], defaultdict(list)
    pairs, cvbias = [], []
    for did in ids:
        S = state_arrays(rows[did], E[did])
        d, sp = decisions(S)
        T = S["truth"]
        imp = {k: T[v] - T[S["top1"]] for k, v in d.items()}
        imp["ORACLE_UPPER"] = T.max() - T[S["top1"]]
        per.append(imp)
        spends.append(sp)
        for k in d:
            costs[k].append(cost_of(S, k, sp))
        pairs.append(pair_stats(S))
        picks.append(d)
        cvbias.append(float((S["X1p"][:, :N_LEAF].mean(1) - S["crit"]).mean()))
    keys = sorted(per[0])
    e = [E[i] for i in ids]
    battles = [x["battle"] for x in e]
    wts = reweights(st, e)
    bt = Boot(battles, wts)
    strata = {"all": np.ones(len(ids), bool)}
    for q in range(1, 6):
        strata[f"gap_q{q}"] = np.array([x["quintile"] == q for x in e])
    strata["torn_q1-2"] = np.array([x["quintile"] <= 2 for x in e])
    for lab, lo, hi in (("unrev_0-1", 0, 1), ("unrev_2-3", 2, 3), ("unrev_4-5", 4, 5)):
        strata[lab] = np.array([lo <= x["unrevealed"] <= hi for x in e])
    for lab, lo, hi in (("turn_2-10", 2, 10), ("turn_11-25", 11, 25), ("turn_26-40", 26, 40)):
        strata[lab] = np.array([lo <= x["turn"] <= hi for x in e])
    for o in ("win", "loss", "draw"):
        strata[f"outcome_{o}"] = np.array([x["outcome"] == o for x in e])
    X = {k: np.array([p[k] for p in per]) for k in keys}
    res = {"cp": cp, "n_states": len(ids), "missing": len(missing), "refused": refused,
           "n_battles": len(set(battles)), "imp": {}, "imp_reweighted": {}, "strata": {}, "cost": {},
           "contrasts": {}}
    for k in keys:
        res["imp"][k] = bt.stat(X[k])
        res["imp_reweighted"][k] = bt.stat(X[k], weighted=True)
        if k in costs:
            res["cost"][k] = round(float(np.mean(costs[k])), 3)
    focus = ["ARGMAX|CRITIC", "ARGMAX|LEAF-K1", "ARGMAX|LEAF-K4", "ARGMAX|LEAF-K8", "ARGMAX|LEAF-K16",
             "ARGMAX|TRUNC-1-K8", "ARGMAX|TRUNC-2-K8", "ARGMAX|TRUNC-4-K8", "ARGMAX|CV-K8", "ARGMAX|CV-K1",
             "GUMBEL|LEAF-K4", "SH-4|B=n*2", "ORACLE_UPPER"]
    for sname, m in strata.items():
        if sname == "all":
            continue
        if m.sum() > 0:
            res["strata"][sname] = {k: bt.stat(X[k], sel=m) for k in focus}
    C = {
        "R-LEAF: LEAF-K8 - CRITIC": X["ARGMAX|LEAF-K8"] - X["ARGMAX|CRITIC"],
        "R-K: LEAF-K8 - LEAF-K4": X["ARGMAX|LEAF-K8"] - X["ARGMAX|LEAF-K4"],
        "LEAF-K16 - LEAF-K8": X["ARGMAX|LEAF-K16"] - X["ARGMAX|LEAF-K8"],
        "LEAF-K8 - LEAF-K1": X["ARGMAX|LEAF-K8"] - X["ARGMAX|LEAF-K1"],
        "R-PRIOR: GUMBEL - ARGMAX at LEAF-K4": X["GUMBEL|LEAF-K4"] - X["ARGMAX|LEAF-K4"],
        "R-SH: SH-4 (B=n*2) - ARGMAX LEAF-K2": X["SH-4|B=n*2"] - X["ARGMAX|LEAF-K2"],
    }
    for d in TRUNC_D:
        C[f"TRUNC-{d}-K8 - LEAF-K8"] = X[f"ARGMAX|TRUNC-{d}-K8"] - X["ARGMAX|LEAF-K8"]
        C[f"TRUNC-{d}-K8 - CRITIC"] = X[f"ARGMAX|TRUNC-{d}-K8"] - X["ARGMAX|CRITIC"]
    for K in (1, 4, 8):
        C[f"CV-K{K} - LEAF-K{K}"] = X[f"ARGMAX|CV-K{K}"] - X[f"ARGMAX|LEAF-K{K}"]
        C[f"MIX-K{K} - LEAF-K{K}"] = X[f"ARGMAX|MIX-K{K}"] - X[f"ARGMAX|LEAF-K{K}"]
    for K in (1, 2, 4, 8):
        C[f"SH-4 (B=n*{K}) - LEAF-K{K}"] = X[f"SH-4|B=n*{K}"] - X[f"ARGMAX|LEAF-K{K}"]
        C[f"SH-all (B=n*{K}) - LEAF-K{K}"] = X[f"SH-all|B=n*{K}"] - X[f"ARGMAX|LEAF-K{K}"]
    for K in (1, 4, 8):
        C[f"GUMBEL - ARGMAX at LEAF-K{K}"] = X[f"GUMBEL|LEAF-K{K}"] - X[f"ARGMAX|LEAF-K{K}"]
    C["GUMBEL - ARGMAX at CRITIC"] = X["GUMBEL|CRITIC"] - X["ARGMAX|CRITIC"]
    for k, v in C.items():
        res["contrasts"][k] = bt.stat(v)
    res["sh_spend"] = {k: {"mean_spent": round(float(np.mean([s[k][0] for s in spends])), 2),
                           "capped_share": round(float(np.mean([s[k][1] for s in spends])), 3)}
                       for k in spends[0]}
    res["agree_top1"] = {k: round(float(np.mean([abs(p[k]) < 1e-12 for p in per])), 3) for k in keys}
    res["pairing"] = pairing_summary(pairs, bt)
    # CV validity (descriptive): the leaf seeds' first-decision V and the critic's one-ply estimate
    # measure the same quantity on different seeds, so their mean difference must straddle 0
    res["cv_validity_mean_X1p_minus_critic"] = bt.stat(np.array(cvbias))
    res["verdicts"] = verdicts(res)
    res["mean_legal"] = round(float(np.mean([x["n_legal"] for x in e])), 3)
    return res


def reweights(st, e):
    """weight = battles of that outcome played / traced, per (cycle, opponent) (README §5)."""
    out = []
    cap = st["capture"]
    for x in e:
        c = cap[x["cycle_opp"]]
        rate = {"win": c.get("capture_rate_win"), "loss": c.get("capture_rate_loss"),
                "draw": c.get("capture_rate_draw")}[x["outcome"]]
        out.append(1.0 / rate if rate else 1.0)
    return out


def pair_stats(S):
    """Per state: for every action pair, per depth (1, 2, 3, 5, 8 turns, terminal) the paired and
    unpaired sample variances over all 40 playout seeds, and the derangement (terminal)."""
    W = S["W"]
    A = W.shape[0]
    out = {}
    depth = {"terminal": W}
    for d in TRUNC_D:
        depth[f"d{d}"] = S["Xd"][d]
    acc = {}
    for name, M in depth.items():
        sp = su = cov = sq = 0.0
        for i in range(A):
            for j in range(i + 1, A):
                a, b = M[i], M[j]
                va, vb = a.var(ddof=1), b.var(ddof=1)
                c = np.cov(a, b, ddof=1)[0, 1]
                sp += (a - b).var(ddof=1)
                su += va + vb
                cov += c
                sq += math.sqrt(va * vb)
        acc[name] = (sp, su, cov, sq)
    sp = su = 0.0
    for i in range(A):
        for j in range(i + 1, A):
            a, b = W[i], np.roll(W[j], -1)
            sp += (a - b).var(ddof=1)
            su += a.var(ddof=1) + b.var(ddof=1)
    acc["derangement"] = (sp, su, 0.0, 0.0)
    out["acc"] = acc
    return out


def pairing_summary(pairs, bt):
    res = {}
    for name in pairs[0]["acc"]:
        sp = np.array([p["acc"][name][0] for p in pairs])
        su = np.array([p["acc"][name][1] for p in pairs])
        cv = np.array([p["acc"][name][2] for p in pairs])
        sq = np.array([p["acc"][name][3] for p in pairs])
        nb = len(bt.ub)
        f = lambda x: np.bincount(bt.bi, weights=x, minlength=nb)  # noqa: E731
        Sp, Su, Cv, Sq = f(sp), f(su), f(cv), f(sq)
        vrf = Su.sum() / Sp.sum()
        bv = (bt.C @ Su) / (bt.C @ Sp)
        r = {"VRF": round(float(vrf), 4), "VRF_ci": np.percentile(bv, [2.5, 97.5]).round(4).tolist()}
        if name != "derangement" and Sq.sum() > 0:
            br = (bt.C @ Cv) / (bt.C @ Sq)
            r["rho"] = round(float(Cv.sum() / Sq.sum()), 4)
            r["rho_ci"] = np.percentile(br, [2.5, 97.5]).round(4).tolist()
        res[name] = r
    return res


def verdicts(res):
    c = res["contrasts"]
    v = {}
    lo, hi = c["R-LEAF: LEAF-K8 - CRITIC"]["ci"]
    if lo > EDGE:
        v["R-LEAF"] = "LEAF-AHEAD"
    elif lo > -0.01 and hi < 0.01:
        v["R-LEAF"] = "CRITIC-EQUIVALENT"
    else:
        v["R-LEAF"] = "NOT DETECTED"
    v["R-TRUNC"] = "NONE"
    for d in TRUNC_D:
        if c[f"TRUNC-{d}-K8 - LEAF-K8"]["ci"][0] > -0.01 and c[f"TRUNC-{d}-K8 - CRITIC"]["ci"][0] > EDGE:
            v["R-TRUNC"] = f"d = {d}"
            break
    lo, hi = c["R-K: LEAF-K8 - LEAF-K4"]["ci"]
    v["R-K"] = "SATURATING" if hi < 0.01 else ("STILL CLIMBING" if lo > EDGE else "UNRESOLVED")
    if hi < 0.01 and lo > EDGE:
        v["R-K"] = "STILL CLIMBING (and below +0.01)"
    v["R-PRIOR"] = verdict3(c["R-PRIOR: GUMBEL - ARGMAX at LEAF-K4"]["ci"], "PRIOR HELPS", "PRIOR HURTS")
    v["R-SH"] = verdict3(c["R-SH: SH-4 (B=n*2) - ARGMAX LEAF-K2"]["ci"], "SH HELPS", "SH HURTS")
    p = res["pairing"]
    sig = p["d1"]["rho_ci"][0] >= 0.30 and p["terminal"]["rho_ci"][1] < 0.20
    v["R-PAIR"] = "ALIGNMENT-LOSS SIGNATURE" if sig else "NO SIGNATURE"
    dl = p["derangement"]["VRF_ci"]
    v["M1 validity (derangement CI contains 1)"] = bool(dl[0] <= 1.0 <= dl[1])
    return v


def md(results):
    L = ["# Value-discrimination pilot: result tables", "",
         "Generated by `score.py`. IMP = decision improvement over the policy's own top-1, win probability per",
         "decision, against the 24 held-out truth seeds; [95 % battle-clustered bootstrap]. ORACLE world.", ""]
    for res in results:
        L += [f"## {res['cp']} ({res['n_states']} states, {res['n_battles']} battles; missing {res['missing']}, "
              f"refused {len(res['refused'])})", "", "### Verdicts", ""]
        for k, x in res["verdicts"].items():
            L.append(f"- {k}: **{x}**")
        L += ["", "### IMP by scorer (ARGMAX over all legal actions) and cost", "",
              "| scorer | IMP [95 %] | re-weighted | agrees with top1 | CPU-s / decision |", "|---|---|---|---|---|"]
        for k, x in res["imp"].items():
            if not k.startswith("ARGMAX|") and k != "ORACLE_UPPER":
                continue
            rw = res["imp_reweighted"][k]
            L.append(f"| {k} | {x['mean']:+.4f} [{x['ci'][0]:+.4f}, {x['ci'][1]:+.4f}] | {rw['mean']:+.4f} | "
                     f"{res['agree_top1'][k]:.2f} | {res['cost'].get(k, float('nan')):.2f} |")
        L += ["", "### Acting rules", "", "| rule∘scorer | IMP [95 %] | CPU-s / decision |", "|---|---|---|"]
        for k, x in res["imp"].items():
            if k.startswith("ARGMAX|") or k == "ORACLE_UPPER":
                continue
            if not any(s in k for s in ("CRITIC", "LEAF-K1", "LEAF-K2", "LEAF-K4", "LEAF-K8", "LEAF-K16",
                                        "SH-", "GATE", "TRUNC-4-K8", "CV-K8")):
                continue
            L.append(f"| {k} | {x['mean']:+.4f} [{x['ci'][0]:+.4f}, {x['ci'][1]:+.4f}] | "
                     f"{res['cost'].get(k, float('nan')):.2f} |")
        L += ["", "### Paired contrasts", "", "| contrast | Δ [95 %] |", "|---|---|"]
        for k, x in res["contrasts"].items():
            L.append(f"| {k} | {x['mean']:+.4f} [{x['ci'][0]:+.4f}, {x['ci'][1]:+.4f}] |")
        L += ["", "### Strata (ARGMAX unless named)", "", "| stratum | n | " + " | ".join(
            k for k in next(iter(res["strata"].values()))) + " |",
              "|---|---|" + "---|" * len(next(iter(res["strata"].values())))]
        for s, d in res["strata"].items():
            n = next(iter(d.values()))["n"]
            L.append(f"| {s} | {n} | " + " | ".join(f"{x['mean']:+.3f}" for x in d.values()) + " |")
        L += ["", "### Pairing (all 40 playout seeds, all action pairs)", "", "| depth | VRF [95 %] | ρ [95 %] |",
              "|---|---|---|"]
        for k, x in res["pairing"].items():
            L.append(f"| {k} | {x['VRF']} {x['VRF_ci']} | {x.get('rho', '')} {x.get('rho_ci', '')} |")
        L += ["", "### Sequential halving spend", "", "| rule | mean spent | capped share |", "|---|---|---|"]
        for k, x in res["sh_spend"].items():
            L.append(f"| {k} | {x['mean_spent']} | {x['capped_share']} |")
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cps", nargs="+")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--partial", action="store_true")
    a = ap.parse_args()
    results = [read_cp(c, a.partial) for c in a.cps]
    if a.out_json:
        Path(a.out_json).write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
    if a.out_md:
        Path(a.out_md).write_text(md(results) + "\n")
    if not a.out_json and not a.out_md:
        for r in results:
            print(json.dumps({"cp": r["cp"], "n": r["n_states"], "verdicts": r["verdicts"]}))


if __name__ == "__main__":
    main()
