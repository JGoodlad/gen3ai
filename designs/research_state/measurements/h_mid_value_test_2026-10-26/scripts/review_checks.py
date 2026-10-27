"""Adversarial review checks on H-MID and the piloting-gap forensics (REVIEW.md, 2026-10-27).

Usage (numpy only, no model):
    python review_checks.py --games GAMES.jsonl.gz --recon kakuna=K.jsonl.gz --recon pm=P.jsonl.gz --out review.json

Reuses ``analyze.py``'s loader and state definitions unchanged. Every check is descriptive and post hoc (nothing here
was registered); intervals are 95 % percentile bootstraps over 20-game units resampled within each panel cell.
  R1  per-panel decomposition at level-10 states (Simpson: is U's Delta-bias carried by one panel?)
  R2  sensitivity of Delta-bias and the matched-V gap to the level band, the turn and the V binning
  R3  the turn-1 prior: is the turn-10 V gap the critic's opening view of the team, carried forward?
  R4  where E[V] separates the sets over the game (V as the state proxy the forensics lacked)
  R6  per-mon KOs: the within-team seen-vs-unseen DiD on U, and the seen-set home72 - U drop within won / lost games
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze as A  # the registered loader and definitions

B = 1000
RNG_SEED = 20261027
PANELS = ["home72", "H", "F", "P72"]


def units_by_panel(games):
    cells = defaultdict(lambda: defaultdict(list))
    for i, g in enumerate(games):
        cells[g["panel"]][g["unit"]].append(i)
    return {p: [np.array(ix) for ix in u.values()] for p, u in cells.items()}


def boot(games, stat, rng, b=B):
    """stat(index array) -> dict of floats; returns point and 95 % intervals per key."""
    cells = units_by_panel(games)
    point = stat(np.arange(len(games)))
    reps = defaultdict(list)
    for _ in range(b):
        idx = []
        for units in cells.values():
            pick = rng.integers(0, len(units), len(units))
            idx.extend(units[j] for j in pick)
        r = stat(np.concatenate(idx))
        for k, v in r.items():
            reps[k].append(v)
    out = {}
    for k, v in point.items():
        arr = np.array(reps[k], float)
        arr = arr[np.isfinite(arr)]
        out[k] = [round(float(v), 4), round(float(np.percentile(arr, 2.5)), 4),
                  round(float(np.percentile(arr, 97.5)), 4)] if len(arr) else [v, None, None]
    return out


def arrays(games, t, band, v_turn=None):
    """Per game: panel, y, V at t, V at v_turn (or None), level-at-t flag for |D(t)| < band (band None = all)."""
    pan, y, v, v0, keep = [], [], [], [], []
    for g in games:
        vt = A.v_at(g, t)
        ok = vt is not None and (band is None or abs(A.state_at(g, t)) < band)
        pan.append(g["panel"])
        y.append(g["y"])
        v.append(vt if vt is not None else np.nan)
        v0.append(A.v_at(g, v_turn) if v_turn else np.nan)
        keep.append(ok)
    return np.array(pan), np.array(y, float), np.array(v, float), np.array(v0, float), np.array(keep)


def wmean(x, mask, pan, panels):
    """Mean over the panels named, each panel equal weight (X51's U), games equal within a panel."""
    ms = []
    for p in panels:
        m = mask & (pan == p)
        if m.sum() == 0:
            return np.nan
        ms.append(x[m].mean())
    return float(np.mean(ms))


def matched_gap(y, v, pan, mask_h, mask_u_panels, edges):
    """Sum_b w_h(b) (W_h(b) - W_U(b)); U's bins equal-weight its panels; empty bins drop, weights renormalise."""
    bins = np.digitize(v, edges)
    num = den = 0.0
    nb = len(edges) + 1
    for b in range(nb):
        mh = mask_h & (bins == b)
        if mh.sum() == 0:
            continue
        wu = []
        for p in mask_u_panels:
            m = (pan == p) & (bins == b) & mask_u_panels[p]
            if m.sum():
                wu.append(y[m].mean())
        if not wu:
            continue
        w = mh.sum()
        num += w * (y[mh].mean() - np.mean(wu))
        den += w
    return num / den if den else np.nan


def contrast_stat(pan, y, v, keep, upanels, edges=A.V_EDGES):
    def stat(idx):
        P, Y, V, K = pan[idx], y[idx], v[idx], keep[idx]
        mh = K & (P == "home72")
        Wh, Vh = Y[mh].mean() if mh.any() else np.nan, V[mh].mean() if mh.any() else np.nan
        Wu, Vu = wmean(Y, K, P, upanels), wmean(V, K, P, upanels)
        mu = {p: K for p in upanels}
        return {"G_y": Wh - Wu, "G_V": Vh - Vu, "dbias": (Wh - Wu) - (Vh - Vu),
                "matched_V_gap": matched_gap(Y, V, P, mh, mu, edges)}
    return stat


def r1_per_panel(games, rng):
    pan, y, v, _, keep = arrays(games, 10, 1.0)
    out = {}
    for p in PANELS:
        m = keep & (pan == p)
        out[p] = {"n": int(m.sum()), "W": round(float(y[m].mean()), 4), "Vbar": round(float(v[m].mean()), 4),
                  "bias": round(float(v[m].mean() - y[m].mean()), 4)}
    for p in ["H", "F", "P72"]:
        out[f"home72_minus_{p}"] = boot(games, contrast_stat(pan, y, v, keep, [p]), rng)
    # all games, turn 1 (no level filter): the turn-1 profile per panel
    pan1, y1, v1, _, k1 = arrays(games, 1, None)
    out["turn1_all"] = {p: {"n": int((k1 & (pan1 == p)).sum()), "W": round(float(y1[k1 & (pan1 == p)].mean()), 4),
                            "V1": round(float(v1[k1 & (pan1 == p)].mean()), 4)} for p in PANELS}
    return out


def r2_sensitivity(games, rng):
    out = {}
    for t in [5, 10, 15, 20]:
        for band in [0.4, 1.0, 2.0, None]:
            pan, y, v, _, keep = arrays(games, t, band)
            out[f"t{t}_band{band}"] = boot(games, contrast_stat(pan, y, v, keep, ["H", "F", "P72"]), rng, b=500)
            out[f"t{t}_band{band}"]["n_h_u"] = [int((keep & (pan == "home72")).sum()),
                                               int((keep & (pan != "home72")).sum())]
    pan, y, v, _, keep = arrays(games, 10, 1.0)
    for name, edges in {"deciles": list(np.linspace(0.1, 0.9, 9)),
                        "quintile_of_pooled": list(np.quantile(v[keep], [0.2, 0.4, 0.6, 0.8])),
                        "three_bins": [0.45, 0.6]}.items():
        out[f"t10_level_bins_{name}"] = boot(games, contrast_stat(pan, y, v, keep, ["H", "F", "P72"], edges),
                                             rng, b=500)
    # linear-probability regression y ~ V + U (weights: U panels equal), the gap at equal V without bins
    def lp(idx):
        P, Y, V, K = pan[idx], y[idx], v[idx], keep[idx]
        m = K
        w = np.zeros(len(P))
        for p in PANELS:
            mp = m & (P == p)
            if mp.sum():
                w[mp] = (1.0 if p == "home72" else 1.0 / 3.0) / mp.sum()
        X = np.column_stack([np.ones(m.sum()), V[m], (P[m] != "home72").astype(float)])
        W = w[m]
        beta = np.linalg.lstsq(X * np.sqrt(W)[:, None], Y[m] * np.sqrt(W), rcond=None)[0]
        return {"slope_V": beta[1], "U_coef": beta[2]}
    out["t10_level_linear_y_on_V_plus_U"] = boot(games, lp, rng, b=500)
    return out


def r3_turn1_prior(games, rng):
    """V(10) - V(1): the critic's in-game update; its set gap vs the outcome gap."""
    pan, y, v, v1, keep = arrays(games, 10, 1.0, v_turn=1)
    keep = keep & np.isfinite(v1)
    up = ["H", "F", "P72"]

    def stat(idx):
        P, Y, V, V1, K = pan[idx], y[idx], v[idx], v1[idx], keep[idx]
        mh = K & (P == "home72")
        d = V - V1
        return {"G_V1": V1[mh].mean() - wmean(V1, K, P, up), "G_V10": V[mh].mean() - wmean(V, K, P, up),
                "G_update": d[mh].mean() - wmean(d, K, P, up), "G_y": Y[mh].mean() - wmean(Y, K, P, up),
                "matched_V1_gap": matched_gap(Y, V1, P, mh, {p: K for p in up}, A.V_EDGES)}
    return boot(games, stat, rng)


def r4_v_profile(games, rng):
    """E[V(t)] by set over ALL games, with V(t) = y once the game is over: where the critic's gap opens."""
    ts = [1, 5, 10, 15, 20, 25, 30, 40, 50, 60]
    pan = np.array([g["panel"] for g in games])
    y = np.array([g["y"] for g in games], float)
    mat = np.full((len(games), len(ts)), np.nan)
    for i, g in enumerate(games):
        for j, t in enumerate(ts):
            vt = A.v_at(g, t)
            mat[i, j] = g["y"] if vt is None else vt
    up = ["H", "F", "P72"]
    keep = np.ones(len(games), bool)

    def stat(idx):
        P, M, Y = pan[idx], mat[idx], y[idx]
        out = {f"gap_EV_t{t}": M[P == "home72", j].mean() - wmean(M[:, j], keep[idx], P, up) for j, t in enumerate(ts)}
        out["gap_y"] = Y[P == "home72"].mean() - wmean(Y, keep[idx], P, up)
        return out
    return boot(games, stat, rng, b=500)


def r6_within_team(fz_path, rng):
    """Per-mon KOs per entry (P* minus Kakuna), the forensics' rel measure, read three ways (X51 cells only):
    (a) within U games: P*'s seen-set minus unseen-set mons, minus Kakuna's (same games: outcome held fixed);
    (b) home72 - U on seen sets, within won and within lost games (is the +0.093 the win gap restated?);
    (c) the same per panel. Imports the forensics' own features() and Familiarity (HEAD's pool)."""
    import gzip
    fdir = Path(__file__).resolve().parents[2] / "piloting_gap_forensics_2026-10-26" / "scripts"
    sys.path.insert(0, str(fdir))
    import importlib.util
    spec = importlib.util.spec_from_file_location("forensics_analyze", fdir / "analyze.py")
    FA = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(FA)
    from teams import Familiarity
    fam = Familiarity()
    cellp = {"pstar_home72": "home72", "pstar_hpfix": "H", "pstar_forum72": "F", "pstar_p72": "P72"}
    rows = []
    with gzip.open(fz_path, "rt") as fh:
        for line in fh:
            g = json.loads(line)
            if g["read"] != "x51":
                continue
            f = FA.features(g, fam)
            r = np.zeros(8)  # f seen k,n ; f unseen k,n ; o seen k,n ; o unseen k,n
            for m in f["mons"]:
                if not m["entered"]:
                    continue
                j = (0 if m["side"] == "f" else 4) + (0 if m["seen"] else 2)
                r[j] += m["kills"]
                r[j + 1] += 1
            rows.append({"panel": cellp[g["cell"]], "unit": g["unit"], "y": float(g["result"]), "r": r})
    pan = np.array([x["panel"] for x in rows])
    y = np.array([x["y"] for x in rows])
    R = np.array([x["r"] for x in rows])

    games = [{"panel": x["panel"], "unit": x["unit"]} for x in rows]

    def stat(idx):
        P = pan[idx]
        Rr, Y = R[idx], y[idx]

        def rr(m, k, n):
            a = Rr[m]
            return a[:, k].sum() / a[:, n].sum() if a[:, n].sum() else np.nan

        def rel_seen(m):   # P* seen minus Kakuna seen
            return rr(m, 0, 1) - rr(m, 4, 5)

        def rel_all(m):
            a = Rr[m]
            return (a[:, 0].sum() + a[:, 2].sum()) / (a[:, 1].sum() + a[:, 3].sum()) - \
                   (a[:, 4].sum() + a[:, 6].sum()) / (a[:, 5].sum() + a[:, 7].sum())

        def did_within(m):  # (P* seen - P* unseen) - (Kakuna seen - Kakuna unseen)
            return (rr(m, 0, 1) - rr(m, 2, 3)) - (rr(m, 4, 5) - rr(m, 6, 7))
        hm = P == "home72"
        res = {"U_within_team_DiD": float(np.mean([did_within(P == p) for p in ("H", "F", "P72")])),
               "seen_rel_home72": rel_seen(hm),
               "seen_rel_U": float(np.mean([rel_seen(P == p) for p in ("H", "F", "P72")])),
               "all_rel_home72": rel_all(hm),
               "all_rel_U": float(np.mean([rel_all(P == p) for p in ("H", "F", "P72")]))}
        for lab, ym in (("won", Y == 1), ("lost", Y == 0)):
            res[f"seen_rel_home72_minus_U_{lab}"] = rel_seen(hm & ym) - float(
                np.mean([rel_seen((P == p) & ym) for p in ("H", "F", "P72")]))
        for p in ("H", "F", "P72"):
            res[f"seen_rel_home72_minus_{p}"] = rel_seen(hm) - rel_seen(P == p)
            res[f"U_within_team_DiD_{p}"] = did_within(P == p)
        res["seen_rel_home72_minus_U"] = res["seen_rel_home72"] - res["seen_rel_U"]
        return res
    return boot(games, stat, rng, b=500)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", required=True)
    ap.add_argument("--recon", action="append", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    recon = dict(x.split("=", 1) for x in a.recon)
    reads = A.load(a.games, recon)
    rng = np.random.default_rng(RNG_SEED)
    res = {}
    for name, rd in reads.items():
        g = rd["games"]
        res[name] = {"R1_per_panel": r1_per_panel(g, rng), "R3_turn1_prior": r3_turn1_prior(g, rng),
                     "R4_V_profile": r4_v_profile(g, rng)}
        if name == "kakuna":
            res[name]["R2_sensitivity"] = r2_sensitivity(g, rng)
    res["R6_within_team_kakuna"] = r6_within_team(a.games, rng)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
