"""Architecture diagnostic, the value reads (2026-11-09). numpy only; no model, no games.

    value_read.py <out_dir>     (reads <out_dir>/acts.npz from model_read.py, writes <out_dir>/value_read.json)

(A) P* against Kakuna and against PastaMittens: H-MID's exact reconstruction (1,600 games a read, V at every one of
    P*'s decisions, `~/gen3ai_archive/h_mid_value_test_2026-10-26/`). Per panel, over ALL decisions: reliability, ECE,
    bias, AUC, and the logistic recalibration y ~ sigmoid(a + b * logit V). A UNIFORM level error moves `a` and leaves
    `b` and the AUC alone; a RANKING failure lowers `b` and the AUC. Intervals: 95 % percentile bootstrap over the
    reads' 20-game units, resampled within each panel (H-MID's own unit), B = 1,000.
(B) P* greedy self-play rows (48 games a panel, descriptive): the same per set, by game.
(C) Is the head's hidden layer used? The trained logit against its best LINEAR fit on LayerNorm(value_pooled)."""
from __future__ import annotations

import gzip
import json
import os
import sys

import numpy as np

HMID = "/home/goodlad/gen3ai_archive/h_mid_value_test_2026-10-26"
CELLS = {"home72": "home72", "hpfix": "H", "forum72": "F", "p72": "P72"}
B = 1000
SEED = 20261109
EPS = 1e-6
EDGES = np.linspace(0, 1, 11)


def logit(p):
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


def auc(v, y):
    order = np.argsort(v, kind="mergesort")
    v, y = v[order], y[order]
    r = np.arange(1, len(v) + 1, dtype=np.float64)
    # average ranks over ties
    _uniq, start, cnt = np.unique(v, return_index=True, return_counts=True)
    csum = np.concatenate([[0.0], np.cumsum(r)])
    avg = (csum[start + cnt] - csum[start]) / cnt
    ranks = np.repeat(avg, cnt)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ece(v, y, w=None):
    w = np.ones_like(v) if w is None else w
    b = np.clip(np.digitize(v, EDGES[1:-1]), 0, 9)
    tot = w.sum(); e = 0.0
    for k in range(10):
        m = b == k
        if m.any():
            e += w[m].sum() / tot * abs(np.average(v[m], weights=w[m]) - np.average(y[m], weights=w[m]))
    return float(e)


def recal(v, y, slope=True, iters=30):
    """Newton fit of y ~ sigmoid(a + b * x), x = logit V; slope=False pins b = 1."""
    x = logit(v)
    a, b = 0.0, 1.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(a + b * x)))
        g_a = (p - y).sum(); w = p * (1 - p)
        if slope:
            g_b = ((p - y) * x).sum()
            h = np.array([[w.sum(), (w * x).sum()], [(w * x).sum(), (w * x * x).sum()]]) + 1e-9 * np.eye(2)
            da, db = np.linalg.solve(h, [g_a, g_b])
            a, b = a - da, b - db
        else:
            a -= g_a / max(w.sum(), 1e-9)
    return float(a), float(b)


def stats(v, y):
    a1, _ = recal(v, y, slope=False)
    a2, b2 = recal(v, y, slope=True)
    v_shift = 1 / (1 + np.exp(-(logit(v) + a1)))
    v_full = 1 / (1 + np.exp(-(a2 + b2 * logit(v))))
    return {"n_rows": len(v), "win_rate": float(y.mean()), "mean_V": float(v.mean()), "bias_V_minus_win": float(v.mean() - y.mean()),
            "ece": ece(v, y), "ece_after_uniform_logit_shift": ece(v_shift, y), "ece_after_shift_and_slope": ece(v_full, y),
            "brier": float(((v - y) ** 2).mean()), "auc": auc(v, y), "recal_intercept_at_slope1": a1,
            "recal_intercept": a2, "recal_slope": b2}


KEYS = ("win_rate", "mean_V", "bias_V_minus_win", "ece", "ece_after_uniform_logit_shift", "ece_after_shift_and_slope",
        "brier", "auc", "recal_intercept_at_slope1", "recal_intercept", "recal_slope")


def reliability(v, y):
    b = np.clip(np.digitize(v, EDGES[1:-1]), 0, 9)
    return [{"bin": f"{EDGES[k]:.1f}-{EDGES[k+1]:.1f}", "share": float((b == k).mean()),
             "mean_V": float(v[b == k].mean()) if (b == k).any() else None,
             "win_rate": float(y[b == k].mean()) if (b == k).any() else None} for k in range(10)]


def boot(units_by_set: dict, fn, rng):
    """units_by_set: {set: [ (v, y) per unit ]}; fn(dict set -> (v, y)) -> dict of floats. Units resampled within set."""
    draws = []
    for _ in range(B):
        samp = {}
        for s, units in units_by_set.items():
            idx = rng.integers(0, len(units), len(units))
            samp[s] = (np.concatenate([units[i][0] for i in idx]), np.concatenate([units[i][1] for i in idx]))
        draws.append(fn(samp))
    out = {}
    for k in draws[0]:
        a = np.array([d[k] for d in draws], dtype=np.float64)
        out[k] = [float(np.nanquantile(a, 0.025)), float(np.nanquantile(a, 0.975))]
    return out


def hmid_read(name, rng):
    with gzip.open(os.path.join(HMID, f"recon_{name}.jsonl.gz"), "rt") as fh:
        rows = [json.loads(x) for x in fh]
    assert len(rows) == 1600 and all(r["ok"] for r in rows)
    out = {"games": len(rows), "ties_excluded": 0}

    def build(select):
        by = {p: {} for p in CELLS.values()}
        for r in rows:
            if r["result"] == 0.5:
                continue
            panel = CELLS[r["unit"].rsplit("_", 1)[0].split("_", 1)[1]]
            dec = select(r["decisions"])
            if not dec:
                continue
            u = by[panel].setdefault(r["unit"], ([], []))
            u[0].extend(d[2] for d in dec); u[1].extend([r["result"]] * len(dec))
        return {p: [(np.array(v), np.array(y)) for v, y in us.values()] for p, us in by.items()}

    out["ties_excluded"] = sum(r["result"] == 0.5 for r in rows)

    def first_at(t):
        def f(decs):
            for d in decs:
                if d[1] >= t:
                    return [d]
            return []
        return f

    views = {"all decisions": lambda d: d, "turn 1 (one row a game)": first_at(1),
             "turn 10 (one row a game, games that reach it)": first_at(10),
             "turn 20 (one row a game, games that reach it)": first_at(20)}
    for vname, sel in views.items():
        by = build(sel)
        by_u = {"home72": by["home72"], "U": by["H"] + by["F"] + by["P72"]}

        def point(samp):
            d = {}
            for s, (v, y) in samp.items():
                st = stats(v, y)
                for k in KEYS:
                    d[f"{s}:{k}"] = st[k]
            if "home72" in samp and "U" in samp:
                for k in KEYS:
                    d[f"U-home72:{k}"] = d[f"U:{k}"] - d[f"home72:{k}"]
            return d

        full = {s: (np.concatenate([u[0] for u in us]), np.concatenate([u[1] for u in us])) for s, us in by_u.items()}
        per_panel = {s: (np.concatenate([u[0] for u in us]), np.concatenate([u[1] for u in us])) for s, us in by.items()}
        res = {"point": point(full), "ci95": boot(by_u, point, rng),
               "n_rows": {s: len(v) for s, (v, y) in full.items()},
               "per_panel": {s: stats(v, y) for s, (v, y) in per_panel.items()}}
        if vname == "all decisions":
            res["reliability"] = {s: reliability(v, y) for s, (v, y) in full.items()}
        out[vname] = res
        print(name, vname, "done", flush=True)
    return out


def selfplay_read(acts, rng):
    names = sorted({k.split(":", 1)[0] for k in acts.files}, key=int)
    out = {}
    sets = {}
    for i in names:
        nm = str(acts[f"{i}:name"])
        v = 1 / (1 + np.exp(-acts[f"{i}:win_logit"].astype(np.float64)))
        y = acts[f"{i}:y"]; game = acts[f"{i}:game"]
        sets[nm] = [(v[game == g], y[game == g]) for g in np.unique(game)]
    grp = {"home72 (both sides)": [k for k in sets if "home72" in k], "pilots U": [k for k in sets if "pilots U" in k],
           "home team facing U": [k for k in sets if "faces U" in k]}
    by = {g: [u for k in ks for u in sets[k]] for g, ks in grp.items()}

    def point(samp):
        d = {}
        for s, (v, y) in samp.items():
            st = stats(v, y)
            for k in KEYS:
                d[f"{s}:{k}"] = st[k]
        return d
    full = {s: (np.concatenate([u[0] for u in us]), np.concatenate([u[1] for u in us])) for s, us in by.items()}
    out = {"point": point(full), "ci95": boot(by, point, rng), "games": {s: len(us) for s, us in by.items()},
           "n_rows": {s: len(v) for s, (v, y) in full.items()}}
    return out


def linearity_read(acts):
    """The trained win logit against its best linear fit on LayerNorm(value_pooled) (no affine: standardised rows)."""
    names = sorted({k.split(":", 1)[0] for k in acts.files}, key=int)
    x = np.concatenate([acts[f"{i}:value_pooled"] for i in names]).astype(np.float64)
    z = np.concatenate([acts[f"{i}:win_logit"] for i in names]).astype(np.float64)
    game = np.concatenate([acts[f"{i}:game"] + 1000 * int(i) for i in names])
    xn = (x - x.mean(1, keepdims=True)) / np.sqrt(x.var(1, keepdims=True) + 1e-5)
    ug = np.unique(game)
    rng = np.random.default_rng(SEED)
    fold = dict(zip(ug.tolist(), rng.integers(0, 2, len(ug)).tolist()))
    tr = np.array([fold[g] == 0 for g in game.tolist()])
    out = {}
    for lab, feats in (("LayerNorm(value_pooled)", xn), ("raw value_pooled", x)):
        a = np.c_[feats, np.ones(len(feats))]
        w, *_ = np.linalg.lstsq(a[tr], z[tr], rcond=None)
        pred = a @ w
        def ss(m, pred=pred):
            return 1 - ((z[m] - pred[m]) ** 2).sum() / ((z[m] - z[m].mean()) ** 2).sum()
        vt, vl = 1 / (1 + np.exp(-z[~tr])), 1 / (1 + np.exp(-pred[~tr]))
        out[lab] = {"r2_train": float(ss(tr)), "r2_heldout_games": float(ss(~tr)),
                    "heldout_mean_abs_dV": float(np.abs(vt - vl).mean()), "heldout_p95_abs_dV": float(np.quantile(np.abs(vt - vl), 0.95)),
                    "heldout_logit_sd": float(z[~tr].std()), "heldout_resid_sd": float((z[~tr] - pred[~tr]).std())}
    out["n_rows"] = len(z); out["n_train"] = int(tr.sum())
    return out


def main(out_dir):
    rng = np.random.default_rng(SEED)
    res = {"B": B, "seed": SEED}
    for name in ("kakuna", "pm"):
        res[f"hmid:{name}"] = hmid_read(name, rng)
    acts = np.load(os.path.join(out_dir, "acts.npz"))
    res["selfplay"] = selfplay_read(acts, rng)
    res["head_linearity"] = linearity_read(acts)
    with open(os.path.join(out_dir, "value_read.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("done")


if __name__ == "__main__":
    main(sys.argv[1])
