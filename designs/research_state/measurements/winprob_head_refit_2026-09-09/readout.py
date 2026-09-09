"""READ each refit condition on the SAME meters the two prior measurements used, so the rows are
comparable to the numbers already on the record rather than to a new scale invented here.

Four meters, all imported rather than re-derived:

  * THE MIXTURE IDENTITY (`winprob_mixture_diagnostic_2026-09-09/analyze.py`) — the between-opponent
    spread of the prediction against the between-opponent spread of the OUTCOME, each corrected for
    its own sampling noise, cells weighted equally. For any calibrated critic the two are EQUAL, so
    the ratio is 1.0 by construction for a head that conditions and < 1 for one that emits the
    mixture's marginal. The online head reads 0.334 at turns 1-3.
  * THE BIAS SLOPE (same module) — OLS of (prediction − true win rate) on opponent Elo/100 with
    cycle fixed effects. The online head reads +0.0171 per 100 Elo.
  * THE PROBE DECODES (`winprob_probe_read_2026-09-09/decode.py`) — the grouped-CV weighted-ridge
    decode of the opponent's CLASS and the own team's leave-one-out win rate FROM THE PREDICTION,
    with `value_pooled` re-run as the reference row. The online head reads AUC 0.532 / R² 0.010 at
    turn 1 against pooled's 0.846 / 0.671.
  * BRIER, decomposed (reliability − resolution + uncertainty) and as a skill score against the
    base rate, on HELD-OUT battles — because a refit that separates opponents but forecasts worse
    has not recovered anything.

RULES OF EVIDENCE, enforced here rather than described:
  * every interval is a BATTLE-CLUSTERED bootstrap, resampling battles WITHIN each (cycle,
    opponent) cell, with the outcome side's own binomial noise redrawn (100 games per cell);
  * every condition is evaluated on the SAME draw, so every delta is PAIRED and gets its own CI —
    a delta whose CI straddles zero is NOT DETECTED, whatever the two point estimates look like;
  * every statistic is HT-weighted by the manifest capture rates unless the run says otherwise.

🚨 The prediction columns are the OUT-OF-FOLD ones. `refit.py` also stores the in-fold columns and
`--train` reads those instead: the difference between the two runs IS the overfitting
counter-hypothesis, and at 951 battles it is not small enough to leave unmeasured.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MEAS = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(MEAS, "winprob_mixture_diagnostic_2026-09-09"))
sys.path.insert(0, os.path.join(MEAS, "winprob_probe_read_2026-09-09"))

import analyze as MIX      # noqa: E402  the mixture identity + the bias slope
import decode as DEC       # noqa: E402  the grouped-CV ridge probe

#: `online_rec` is NOT a fit and NOT the same object as `online`. It is the RECORDED win
#: probability — each cycle's OWN model on its own states — where `online` is the ONE frozen
#: snapshot re-forwarded over every cycle, which is what the refits' features come from. The two
#: differ (probe-read hazard 5), and they differ in the direction that flatters a refit: on arm A
#: the frozen forward's turn-1-3 spread ratio is 0.149 against the recorded 0.334. `online` is the
#: INTERNALLY CONSISTENT comparator (same weights, same features, same states as every refit);
#: `online_rec` is the CONSERVATIVE one and reproduces the mixture diagnostic's published numbers
#: exactly. Every headline delta is reported against BOTH.
FITTED = ("online", "lin_term", "mlp_term", "mlp_term_ft",
          "lin_cond", "mlp_cond", "mlp_cond_ft", "cond_oracle")
CONDITIONS = FITTED + ("online_rec",)
BUCKETS = ("t1", "t1_3", "all")
N_GAMES = 100.0
#: The comparisons the reading needs, each judged on the DELTA'S OWN battle-clustered CI and never
#: on two overlapping bars. `x-online` is the reference row of every table; `mlp_term-mlp_term_ft`
#: is the ORDER hazard (scratch vs a fine-tune of the online weights — if scratch separates and the
#: fine-tune does not, the online head sits in a basin); `mlp_cond-mlp_term` isolates the TARGET at
#: fixed architecture and optimiser; `mlp_*-lin_*` isolates CAPACITY at fixed target.
PAIRS = [("lin_term", "online"), ("mlp_term", "online"), ("mlp_term_ft", "online"),
         ("lin_cond", "online"), ("mlp_cond", "online"), ("mlp_cond_ft", "online"),
         ("cond_oracle", "online"), ("online", "online_rec"),
         ("mlp_term", "online_rec"), ("mlp_term_ft", "online_rec"),
         ("mlp_cond", "online_rec"), ("mlp_cond_ft", "online_rec"),
         ("cond_oracle", "online_rec"),
         ("mlp_term", "mlp_term_ft"), ("mlp_cond", "mlp_cond_ft"),
         ("mlp_cond", "mlp_term"), ("mlp_term", "lin_term"), ("mlp_cond", "lin_cond"),
         ("cond_oracle", "mlp_cond")]


def as_states(meta, V):
    """The structured array the mixture's `rollup` consumes, with `V` swapped for a prediction."""
    dt = np.dtype([("cycle", "i8"), ("opponent", "U24"), ("opp_class", "U10"),
                   ("battle", "U80"), ("y", "f8"), ("turn", "i8"), ("w", "f8"),
                   ("team", "U12"), ("strength", "f8"), ("true_wr", "f8"), ("V", "f8")])
    out = np.empty(len(meta), dt)
    for f in dt.names:
        out[f] = V if f == "V" else meta[f]
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Brier, decomposed. Murphy's identity: Brier = reliability − resolution + uncertainty.
# `uncertainty` is the base-rate forecaster's Brier, so `skill` is the fraction of it removed.
# ──────────────────────────────────────────────────────────────────────────────
def brier(p, y, w, nbins=10):
    W = w.sum()
    b = float((w * (p - y) ** 2).sum() / W)
    obar = float((w * y).sum() / W)
    unc = obar * (1 - obar)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, nbins + 1)))
    if len(edges) < 3:
        return {"brier": b, "reliability": np.nan, "resolution": np.nan,
                "uncertainty": unc, "skill": (unc - b) / unc if unc > 0 else np.nan}
    k = np.clip(np.searchsorted(edges, p, "right") - 1, 0, len(edges) - 2)
    Wk = np.bincount(k, weights=w, minlength=len(edges) - 1)
    Pk = np.bincount(k, weights=w * p, minlength=len(edges) - 1)
    Ok = np.bincount(k, weights=w * y, minlength=len(edges) - 1)
    m = Wk > 0
    pk, ok_ = Pk[m] / Wk[m], Ok[m] / Wk[m]
    rel = float((Wk[m] * (pk - ok_) ** 2).sum() / W)
    res = float((Wk[m] * (ok_ - obar) ** 2).sum() / W)
    return {"brier": b, "reliability": rel, "resolution": res, "uncertainty": unc,
            "skill": (unc - b) / unc if unc > 0 else np.nan}


# ──────────────────────────────────────────────────────────────────────────────
def decodes(meta, preds, pooled, seed, n_perm, n_boot, buckets=("t1", "t1_3")):
    """The probe read's decode, run on each condition's PREDICTION as a one-column feature set,
    with `value_pooled` as the reference row. Same grouped-by-battle nested CV, same HT weights,
    same permutation null through the identical pipeline, same battle-clustered delta CIs."""
    tgts, _b, _bi, _bt = DEC.build_targets(meta)
    out = {}
    for bname in buckets:
        lo, hi = DEC.BUCKETS[bname]
        for tname in ("opp_class", "own_team_wr"):
            tgt = tgts[tname]
            idx = DEC.cap_per_battle(meta, (meta["turn"] >= lo) & (meta["turn"] <= hi)
                                     & tgt["mask"], 2, seed)
            y = tgt["y"][idx].astype(float)
            w = meta["w"][idx].astype(float)
            g = meta["battle"][idx]
            feats = {c: preds[c][idx][:, None] for c in CONDITIONS}
            feats["pooled"] = pooled[idx].astype(np.float64)
            cv = {k: DEC.GroupedRidgeCV(np.asarray(v, np.float64), w, g, seed=seed)
                  for k, v in feats.items()}
            oof = {k: cv[k].oof(y, w, tgt["task"])[0] for k in feats}
            pt = {k: DEC._score(y, oof[k], w, tgt["task"]) for k in feats}
            pairs = PAIRS + [(c, "pooled") for c in CONDITIONS]
            ci, dci = DEC.boot_ci(y, oof, w, g, tgt["task"], n_boot, seed + 5, extra=pairs)
            # permutation null, identical pipeline (penalty selection included)
            _, binv = np.unique(g, return_inverse=True)
            nb = int(binv.max()) + 1
            fo = DEC.first_of_group(binv, nb)
            b_team, b_y = meta["team"][idx][fo], meta["y"][idx][fo]
            b_w = meta["w"][idx][fo].astype(float)
            rngp = np.random.default_rng(seed + 77)
            nulls = {k: [] for k in feats}
            for _ in range(n_perm):
                if tgt["null_kind"] == "team_assign":
                    lab_b = DEC._loo_team_wr(DEC.shuffle_team_assignment(b_team, rngp), b_y, b_w,
                                             DEC.MIN_TEAM_BATTLES_WR)
                    yp = np.nan_to_num(lab_b, nan=float(np.nanmean(lab_b)))[binv]
                else:
                    yp = DEC._perm_labels(y, binv, "battle", None, rngp)
                for k in feats:
                    nulls[k].append(DEC._score(yp, cv[k].oof(yp, w, tgt["task"])[0], w,
                                               tgt["task"]))
            out[f"{bname}|{tname}"] = {
                "task": tgt["task"], "n_states": int(len(idx)), "n_battles": nb,
                "score": {k: round(float(pt[k]), 4) for k in feats},
                "ci": ci, "delta_ci": dci,
                "null_p95": {k: round(float(np.nanpercentile(v, 95)), 4) for k, v in nulls.items()},
                "null_mean": {k: round(float(np.nanmean(v)), 4) for k, v in nulls.items()}}
            print(f"  decode {bname}/{tname}: "
                  + " ".join(f"{k}={pt[k]:.3f}" for k in ("pooled",) + CONDITIONS), flush=True)
    return out


# ──────────────────────────────────────────────────────────────────────────────
def spread_and_slope(meta, preds, n_boot, seed):
    """The mixture identity and the bias slope, for every condition, on ONE shared bootstrap so
    every delta is paired. Battles are resampled WITHIN their (cycle, opponent) cell and the
    outcome side's 100-game binomial noise is redrawn on the same draw."""
    base = MIX.rollup(as_states(meta, preds["online"]))
    cyc, opps, oidx, cid = MIX.cell_index(base)
    n_cells = len(cyc) * len(opps)
    cyc_of_cell = np.repeat(cyc, len(opps))
    opp_of_cell = np.tile(np.array(opps), len(cyc))
    is_bot = np.isin(opp_of_cell, MIX.BOTS)
    strength = np.full(n_cells, np.nan)
    true_wr = np.full(n_cells, np.nan)
    for c, o, s, t in zip(base["cycle"], base["opponent"], base["strength"], base["true_wr"]):
        strength[np.searchsorted(cyc, c) * len(opps) + oidx[o]] = s
        true_wr[np.searchsorted(cyc, c) * len(opps) + oidx[o]] = t
    rolls = {c: MIX.rollup(as_states(meta, preds[c])) for c in CONDITIONS}

    # battle slots per cell, for the within-cell cluster resample (the mixture's own machinery:
    # a draw has exactly as many entries as the original, so the CELL CODE must travel with the
    # SLOT — pairing a resampled battle with the original cell table was a real defect there).
    order = np.argsort(cid, kind="stable")
    cid_s = cid[order]
    starts = np.searchsorted(cid_s, np.arange(n_cells), "left")
    sizes = np.searchsorted(cid_s, np.arange(n_cells), "right") - starts
    slots = [order[starts[k]:starts[k] + sizes[k]] for k in range(n_cells)]
    all_slots = np.concatenate([s for s in slots if s.size])
    off = np.concatenate([np.full(s.size, i) for i, s in enumerate(slots) if s.size])
    size_per = sizes[off].astype(float)
    start_per, p = np.empty(len(off), int), 0
    for k, s in enumerate(slots):
        if s.size:
            start_per[np.where(off == k)[0]] = p
            p += s.size

    def stats(sel, cid_sel, wr):
        o = {}
        for c in CONDITIONS:
            for bk in BUCKETS:
                st = MIX.cell_stats(rolls[c], sel, cid_sel, n_cells, bk, wr)
                keep = st["n_battles"] > 0
                sc = MIX.spread_corrected(st, keep, wr)
                o[f"{c}|{bk}|ratio"] = sc["ratio"]
                # the UNCLAMPED companion: `ratio` subtracts each side's sampling noise and
                # clamps a negative variance to 0, which is a biased, non-monotone operator —
                # it is why a point estimate can sit below its own bootstrap interval (the
                # mixture diagnostic reports the same shape). `ratio_raw` does not correct and
                # does not clamp, so it always contains its point; it is inflated by cell size
                # on BOTH sides equally, so it ranks conditions even where the corrected one
                # bottoms out at zero.
                o[f"{c}|{bk}|ratio_raw"] = (sc["sd_V_raw"] / sc["sd_y_raw"]
                                            if sc["sd_y_raw"] else np.nan)
                o[f"{c}|{bk}|sd_V"] = sc["sd_V"]
                o[f"{c}|{bk}|sd_y"] = sc["sd_y"]
                o[f"{c}|{bk}|delta"] = sc["delta"]
                if bk == "all":
                    o[f"{c}|slope"] = MIX.ols_slope(st["bias"], strength, cyc_of_cell, keep)
                    o[f"{c}|slope_bot"] = MIX.ols_slope(st["bias"], strength, cyc_of_cell,
                                                        keep & is_bot)
        return o

    sel0 = np.arange(base["y"].size)
    pt = stats(sel0, cid, true_wr)
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in pt}
    for _ in range(n_boot):
        j = start_per + (rng.random(all_slots.size) * size_per).astype(int)
        wr = rng.binomial(N_GAMES, np.nan_to_num(true_wr, nan=0.5)) / N_GAMES
        wr = np.where(np.isfinite(true_wr), wr, np.nan)
        d = stats(all_slots[j], off, wr)
        for k in pt:
            draws[k].append(d[k])

    def q(v):
        v = np.asarray(v, float)
        v = v[np.isfinite(v)]
        return [None, None] if v.size < 20 else [round(float(np.percentile(v, 2.5)), 4),
                                                 round(float(np.percentile(v, 97.5)), 4)]
    res = {"point": {k: (None if not np.isfinite(v) else round(float(v), 4))
                     for k, v in pt.items()},
           "ci": {k: q(v) for k, v in draws.items()},
           "delta": {}, "cells": {"n": n_cells, "cycles": cyc.tolist(),
                                  "opponents": list(opps)}}
    for ca, cb in PAIRS:
        for suf in ([f"{bk}|ratio" for bk in BUCKETS]
                    + [f"{bk}|ratio_raw" for bk in BUCKETS] + ["slope", "slope_bot"]):
            a_, b_ = f"{ca}|{suf}", f"{cb}|{suf}"
            dr = np.asarray(draws[a_], float) - np.asarray(draws[b_], float)
            d = pt[a_] - pt[b_]
            res["delta"][f"{ca}-{cb}|{suf}"] = {
                "point": round(float(d), 4) if np.isfinite(d) else None, "ci": q(dr)}
    return res


# ──────────────────────────────────────────────────────────────────────────────
def brier_block(meta, preds, ptr, w, n_boot, seed):
    """Brier + its decomposition on HELD-OUT battles, and the in-fold value beside it."""
    y = meta["y"].astype(float)
    battles, binv = np.unique(meta["battle"], return_inverse=True)
    members = [np.where(binv == i)[0] for i in range(len(battles))]
    masks = {"all": np.ones(len(y), bool), "t1_3": meta["turn"] <= 3}
    out = {}
    rng = np.random.default_rng(seed + 31)
    picks = [rng.integers(0, len(battles), len(battles)) for _ in range(n_boot)]
    for mname, m in masks.items():
        idx = np.where(m)[0]
        held = {c: brier(preds[c][idx], y[idx], w[idx]) for c in CONDITIONS}
        tr = {}
        for c in CONDITIONS:
            ok = idx[np.isfinite(ptr[c][idx])]
            tr[c] = brier(ptr[c][ok], y[ok], w[ok]) if len(ok) > 20 else None
        sub = [np.intersect1d(mem, idx, assume_unique=False) for mem in members]
        draws = {c: [] for c in CONDITIONS}
        dd = {f"{a_}-{b_}": [] for a_, b_ in PAIRS}
        for pk in picks:
            sel = np.concatenate([sub[i] for i in pk if sub[i].size])
            if sel.size < 50:
                continue
            cur = {}
            for c in CONDITIONS:
                bb = float((w[sel] * (preds[c][sel] - y[sel]) ** 2).sum() / w[sel].sum())
                draws[c].append(bb)
                cur[c] = bb
            for a_, b_ in PAIRS:
                dd[f"{a_}-{b_}"].append(cur[a_] - cur[b_])

        def q(v):
            v = np.asarray(v, float)
            v = v[np.isfinite(v)]
            return [None, None] if v.size < 20 else [round(float(np.percentile(v, 2.5)), 5),
                                                     round(float(np.percentile(v, 97.5)), 5)]
        out[mname] = {"held_out": {c: {k: (None if not np.isfinite(v) else round(float(v), 5))
                                       for k, v in held[c].items()} for c in CONDITIONS},
                      "in_fold": {c: (None if tr[c] is None else
                                      {k: (None if not np.isfinite(v) else round(float(v), 5))
                                       for k, v in tr[c].items()}) for c in CONDITIONS},
                      "brier_ci": {c: q(v) for c, v in draws.items()},
                      "brier_delta_ci": {c: q(v) for c, v in dd.items()}}
    return out


def main(a):
    meta = np.load(f"{a.dir}/meta.npy")
    pooled = np.load(f"{a.dir}/pooled.npy")
    z = np.load(a.preds)
    key = "ptr_" if a.train else "p_"
    preds = {c: z[f"{key}{c}"] for c in FITTED}
    ptr = {c: z[f"ptr_{c}"] for c in FITTED}
    preds["online_rec"] = meta["V_rec"].astype(float)   # the RECORDED per-cycle win prob
    ptr["online_rec"] = meta["V_rec"].astype(float)
    if a.train:                    # in-fold columns are NaN on the held-out rows
        keep = np.isfinite(np.stack([preds[c] for c in CONDITIONS])).all(0)
        meta, pooled = meta[keep], pooled[keep]
        preds = {c: v[keep] for c, v in preds.items()}
        ptr = {c: v[keep] for c, v in ptr.items()}

    out = {"dir": a.dir, "preds": a.preds, "train_columns": bool(a.train),
           "n_states": int(len(meta)), "n_battles": int(len(set(meta["battle"].tolist()))),
           "conditional_target": json.load(open(a.preds.replace(".npz", "_fit.json")))
           ["conditional_target"]}
    out["conditional_target"].pop("grid", None)
    print("spread + slope ...", flush=True)
    out["spread"] = spread_and_slope(meta, preds, a.boot_spread, a.seed)
    print("brier ...", flush=True)
    out["brier"] = brier_block(meta, preds, ptr, meta["w"].astype(float), a.boot_brier, a.seed)
    if not a.no_decode:
        print("decodes ...", flush=True)
        out["decode"] = decodes(meta, preds, pooled, a.seed, a.perm, a.boot_decode)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", action="store_true", help="read the IN-FOLD columns instead")
    ap.add_argument("--no-decode", action="store_true")
    ap.add_argument("--boot-spread", type=int, default=1500)
    ap.add_argument("--boot-brier", type=int, default=2000)
    ap.add_argument("--boot-decode", type=int, default=1500)
    ap.add_argument("--perm", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260909)
    main(ap.parse_args())
