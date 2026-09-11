"""READ every (N, condition) fit on the SAME meters the head refit used, on ONE fixed held-out
battle set, so the only thing that moves along a row is the amount of stationary training data.

Five meters, all imported from the measurements that defined them rather than re-derived:

  * THE MIXTURE IDENTITY (`winprob_mixture_diagnostic_2026-09-09/analyze.py`) — the between-cell
    spread of the prediction against the between-cell spread of the OUTCOME, each corrected for
    its own sampling noise. Equal for any calibrated critic; the online head reads 0.066 at
    turns 1-3 on this substrate.
  * THE TURN-1 DECODES (`winprob_probe_read_2026-09-09/decode.py`) — the grouped-CV weighted-ridge
    decode of the opponent's CLASS and of the own team's leave-one-battle-out win rate FROM THE
    PREDICTION, with `value_pooled` re-run as the reference row.
  * BRIER, decomposed (`winprob_head_refit_2026-09-09/readout.py::brier`) — reliability,
    resolution and the skill score against the base rate.
  * THE CALIBRATION SLOPE — weighted logistic regression of the outcome on `logit(p)`; 1.0 is
    perfectly dispersed, below 1.0 over-dispersed, above 1.0 under-dispersed (the head hedges).

🚨 TWO THINGS ARE DELIBERATELY DIFFERENT FROM THE HEAD REFIT AND BOTH CHANGE HOW A ROW READS.

  1. THE OUTCOME SIDE OF THE IDENTITY IS THE MANIFEST'S FULL-SAMPLE WIN RATE, at that cell's own
     `n_games` (400 or 800), not the held-out subsample's. The outcome's true between-cell spread
     is a property of the policy and the opponent set, so estimating it from 400-800 games rather
     than from the ~67 held-out battles in the cell is strictly better AND is the same quantity at
     every N — which is what an N-curve needs. The prediction side is the held-out battles only.
  2. THE CELLS ARE (cycle, opponent) WHERE `cycle` IS THE EVAL SEED, NOT A TRAINING STEP. The three
     trees are three seeds of one frozen checkpoint, so the three cells of one opponent are
     REPLICATES. That is a free noise check the head refit could not run: on a stationary dataset
     the within-opponent, between-seed spread of the outcome is pure sampling noise, and the
     `opponent` keying (12 cells) is reported beside the `cell` keying (36) for exactly that
     reason. The head refit's hazard 2 — a conditional target with no cycle term is unreadable on
     a rapidly improving run — CANNOT bite here, and `extract_meta.json` records the per-tree
     overall win rate so a reader can check that rather than take it on trust.

Every interval is a BATTLE-CLUSTERED bootstrap on the held-out set; every condition is evaluated
on the SAME draw, so every delta is paired and gets its own CI. A delta whose CI straddles zero is
NOT DETECTED whatever the two point estimates look like.
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
sys.path.insert(0, os.path.join(MEAS, "winprob_head_refit_2026-09-09"))

import analyze as MIX          # noqa: E402
import decode as DEC           # noqa: E402
import readout as HR           # noqa: E402  (only `brier`, the Murphy decomposition)

BUCKETS = ("t1", "t1_3", "all")
FITS = ("lin_term", "mlp_term", "lin_cond", "mlp_cond", "cond_oracle")


def as_states(meta, V):
    dt = np.dtype([("cycle", "i8"), ("opponent", "U24"), ("opp_class", "U10"),
                   ("battle", "U90"), ("y", "f8"), ("turn", "i8"), ("w", "f8"),
                   ("team", "U12"), ("strength", "f8"), ("true_wr", "f8"), ("V", "f8")])
    out = np.empty(len(meta), dt)
    for f in dt.names:
        out[f] = V if f == "V" else meta[f]
    return out


def spread_corrected_vec(st, keep, true_wr, n_games):
    """`MIX.spread_corrected` with a PER-CELL `n_games` instead of a scalar 100.

    Verbatim arithmetic — the only change is that the outcome side's binomial noise is
    `p(1-p)/n_c` with the cell's own game count, because this frame's cells were played 400 or
    800 times rather than the 100 the prior measurements had.
    """
    m = keep & np.isfinite(st["meanV_b"]) & np.isfinite(true_wr) & np.isfinite(st["seV2_b"])
    if m.sum() < 3:
        return {k: np.nan for k in ("sd_V", "sd_y", "ratio", "delta",
                                    "sd_V_raw", "sd_y_raw", "noise_V", "noise_y")}
    V, Y, NG = st["meanV_b"][m], true_wr[m], np.asarray(n_games)[m]
    nV = float(np.mean(st["seV2_b"][m]))
    nY = float(np.mean(Y * (1 - Y) / NG))
    vV, vY = float(np.var(V, ddof=1)), float(np.var(Y, ddof=1))
    sV, sY = np.sqrt(max(vV - nV, 0.0)), np.sqrt(max(vY - nY, 0.0))
    return {"sd_V": sV, "sd_y": sY, "ratio": (sV / sY) if sY > 0 else np.nan,
            "delta": sV - sY, "sd_V_raw": np.sqrt(vV), "sd_y_raw": np.sqrt(vY),
            "noise_V": np.sqrt(nV), "noise_y": np.sqrt(nY)}


def calib_slope(p, y, w, iters=50):
    """Weighted logistic regression y ~ a + b*logit(p). `b` is the calibration slope: 1.0 is
    perfectly dispersed, < 1 over-confident, > 1 UNDER-dispersed (the head hedges toward the
    marginal) — which is the direction a mixture-emitting critic fails in."""
    z = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    if np.std(z) < 1e-9:
        return np.nan
    X = np.stack([np.ones_like(z), z], 1)
    beta = np.array([0.0, 1.0])
    for _ in range(iters):
        eta = X @ beta
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))
        W = w * mu * (1 - mu) + 1e-12
        g = X.T @ (w * (y - mu))
        H = X.T @ (X * W[:, None])
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            return np.nan
        beta = beta + step
        if np.max(np.abs(step)) < 1e-9:
            break
    return float(beta[1])


# ──────────────────────────────────────────────────────────────────────────────
def spread_block(meta, preds, key, n_boot, seed):
    """The identity for every condition, on ONE shared bootstrap so every delta is paired.

    `key="cell"`   -> 36 (eval-seed, opponent) cells, the head refit's keying.
    `key="opponent"` -> 12 opponent cells, the three eval seeds POOLED. On a stationary frame the
    three seeds of one opponent are replicates, so this is the same question with 3x the games per
    cell on the outcome side; the head refit could not run it (its cycles were training steps).
    """
    conds = list(preds)
    proto = MIX.rollup(as_states(meta, preds[conds[0]]))
    # the OUTCOME side, straight from the manifest: won / played at the cell's OWN game count.
    # manifest rows are constant within a (cycle, opponent) pair
    seen, won, played = {}, {}, {}
    for c, o, twr, ng in zip(proto["cycle"], proto["opponent"], proto["true_wr"],
                             proto["n_games"]):
        k = (int(c), o)
        if k in seen:
            continue
        seen[k] = True
        ck = f"{0 if key == 'opponent' else int(c)}|{o}"
        won[ck] = won.get(ck, 0.0) + twr * ng
        played[ck] = played.get(ck, 0.0) + ng

    rolls = {}
    for c in conds:
        r = MIX.rollup(as_states(meta, preds[c]))
        if key == "opponent":
            r["cycle"] = np.zeros_like(r["cycle"])
        rolls[c] = r
    cyc, opps, oidx, cid = MIX.cell_index(rolls[conds[0]])
    n_cells = len(cyc) * len(opps)
    true_wr = np.full(n_cells, np.nan)
    n_games = np.full(n_cells, np.nan)
    for ck, pl in played.items():
        cstr, o = ck.split("|", 1)
        idx = np.searchsorted(cyc, int(cstr)) * len(opps) + oidx[o]
        true_wr[idx] = won[ck] / pl
        n_games[idx] = pl

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
        for c in conds:
            for bk in BUCKETS:
                st = MIX.cell_stats(rolls[c], sel, cid_sel, n_cells, bk, wr)
                keep = st["n_battles"] > 0
                sc = spread_corrected_vec(st, keep, wr, n_games)
                o[f"{c}|{bk}|ratio"] = sc["ratio"]
                o[f"{c}|{bk}|ratio_raw"] = (sc["sd_V_raw"] / sc["sd_y_raw"]
                                            if sc["sd_y_raw"] else np.nan)
                o[f"{c}|{bk}|sd_V"] = sc["sd_V"]
                o[f"{c}|{bk}|sd_y"] = sc["sd_y"]
        return o

    pt = stats(np.arange(rolls[conds[0]]["y"].size), cid, true_wr)
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in pt}
    ng_safe = np.where(np.isfinite(n_games), n_games, 1.0).astype(int)
    for _ in range(n_boot):
        j = start_per + (rng.random(all_slots.size) * size_per).astype(int)
        wr = rng.binomial(ng_safe, np.nan_to_num(true_wr, nan=0.5)) / ng_safe
        wr = np.where(np.isfinite(true_wr), wr, np.nan)
        d = stats(all_slots[j], off, wr)
        for k in pt:
            draws[k].append(d[k])
    return pt, draws, {"n_cells": int(n_cells), "key": key,
                       "cells_with_data": int(np.isfinite(true_wr).sum()),
                       "games_per_cell": sorted(set(int(x) for x in ng_safe
                                                    if np.isfinite(x) and x > 1))}


def q(v, nd=4):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    return [None, None] if v.size < 20 else [round(float(np.percentile(v, 2.5)), nd),
                                             round(float(np.percentile(v, 97.5)), nd)]


# ──────────────────────────────────────────────────────────────────────────────
def scalar_block(meta, preds, n_boot, seed):
    """Brier (decomposed) and the calibration slope, on the held-out battles, one shared
    battle-clustered bootstrap so every delta is paired."""
    conds = list(preds)
    y = meta["y"].astype(float)
    w = meta["w"].astype(float)
    battles, binv = np.unique(meta["battle"], return_inverse=True)
    order = np.argsort(binv, kind="stable")
    binv_s = binv[order]
    starts = np.searchsorted(binv_s, np.arange(len(battles)), "left")
    ends = np.searchsorted(binv_s, np.arange(len(battles)), "right")
    members = [order[starts[i]:ends[i]] for i in range(len(battles))]
    masks = {"t1": meta["turn"] == 1, "t1_3": meta["turn"] <= 3,
             "all": np.ones(len(y), bool)}
    rng = np.random.default_rng(seed + 31)
    picks = [rng.integers(0, len(battles), len(battles)) for _ in range(n_boot)]
    out = {}
    for mname, m in masks.items():
        idx = np.where(m)[0]
        held = {c: HR.brier(preds[c][idx], y[idx], w[idx]) for c in conds}
        for c in conds:
            held[c]["calib_slope"] = calib_slope(preds[c][idx], y[idx], w[idx])
        inm = np.zeros(len(y), bool)
        inm[idx] = True
        sub = [mem[inm[mem]] for mem in members]
        draws = {f"{c}|{k}": [] for c in conds for k in ("brier", "calib_slope", "resolution")}
        for pk in picks:
            sel = np.concatenate([sub[i] for i in pk if sub[i].size])
            if sel.size < 50:
                continue
            for c in conds:
                b = HR.brier(preds[c][sel], y[sel], w[sel])
                draws[f"{c}|brier"].append(b["brier"])
                draws[f"{c}|resolution"].append(b["resolution"])
                draws[f"{c}|calib_slope"].append(calib_slope(preds[c][sel], y[sel], w[sel]))
        out[mname] = {"point": {c: {k: (None if not np.isfinite(v) else round(float(v), 5))
                                    for k, v in held[c].items()} for c in conds},
                      "draws": draws}
    return out


# ──────────────────────────────────────────────────────────────────────────────
def decode_block(meta_full, held_states, preds, pooled_full, seed, n_perm, n_boot, pairs):
    """The probe read's decode, run on each condition's PREDICTION as a one-column feature set.

    🚨 The own-team win-rate LABEL is built on the FULL pooled dataset (leave-one-battle-out), not
    on the held-out subsample: at 2,400 held-out battles over hundreds of teams a within-holdout
    LOO win rate is mostly noise and the meter would read the label's noise rather than the head's
    conditioning. LOO still removes the scored battle's own outcome, so nothing the decoder sees
    contains the label of the state it is scoring."""
    tgts, _b, _bi, _bt = DEC.build_targets(meta_full)
    conds = list(preds)
    out = {}
    for tname in ("opp_class", "own_team_wr"):
        tgt = tgts[tname]
        sub = np.zeros(len(meta_full), bool)
        sub[held_states] = True
        mask = sub & (meta_full["turn"] == 1) & tgt["mask"]
        idx = DEC.cap_per_battle(meta_full, mask, 2, seed)
        pos = np.searchsorted(held_states, idx)       # held_states is sorted
        y = tgt["y"][idx].astype(float)
        w = meta_full["w"][idx].astype(float)
        g = meta_full["battle"][idx]
        feats = {c: preds[c][pos][:, None] for c in conds}
        feats["pooled"] = pooled_full[idx].astype(np.float64)
        cv = {k: DEC.GroupedRidgeCV(np.asarray(v, np.float64), w, g, seed=seed)
              for k, v in feats.items()}
        oof = {k: cv[k].oof(y, w, tgt["task"])[0] for k in feats}
        pt = {k: DEC._score(y, oof[k], w, tgt["task"]) for k in feats}
        pp = pairs + [(c, "pooled") for c in conds]
        ci, dci = DEC.boot_ci(y, oof, w, g, tgt["task"], n_boot, seed + 5, extra=pp)
        _, binv = np.unique(g, return_inverse=True)
        nb = int(binv.max()) + 1
        fo = DEC.first_of_group(binv, nb)
        b_team, b_y = meta_full["team"][idx][fo], meta_full["y"][idx][fo]
        b_w = meta_full["w"][idx][fo].astype(float)
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
                nulls[k].append(DEC._score(yp, cv[k].oof(yp, w, tgt["task"])[0], w, tgt["task"]))
        out[tname] = {"task": tgt["task"], "n_states": int(len(idx)), "n_battles": nb,
                      "score": {k: round(float(pt[k]), 4) for k in feats},
                      "ci": ci, "delta_ci": dci,
                      "null_p95": {k: round(float(np.nanpercentile(v, 95)), 4)
                                   for k, v in nulls.items()}}
        print(f"  decode {tname}: pooled={pt['pooled']:.3f} online={pt['online']:.3f}",
              flush=True)
    return out


# ──────────────────────────────────────────────────────────────────────────────
def main(a):
    meta = np.load(f"{a.dir}/meta.npy")
    pooled = np.load(f"{a.dir}/pooled.npy")
    z = np.load(a.preds)
    fitinfo = json.load(open(a.preds.replace(".npz", "_fit.json")))
    held = z["held_states"]
    hm = meta[held]
    cols = ["online", "online_rec"] + [k for k in z.files
                                       if "|" in k]
    preds = {c: z[c].astype(float) for c in cols}
    ns = fitinfo["ns"]

    # the comparisons the reading needs, each on the DELTA'S OWN CI
    pairs = []
    for N in ns:
        for c in FITS:
            k = f"{N}|{c}"
            if k in preds:
                pairs.append((k, "online"))
        if f"{N}|mlp_cond" in preds:
            pairs.append((f"{N}|mlp_cond", f"{N}|mlp_term"))
            pairs.append((f"{N}|mlp_term", f"{N}|lin_term"))
            pairs.append((f"{N}|cond_oracle", f"{N}|mlp_cond"))
        if N != ns[0] and f"{N}|mlp_term" in preds:
            pairs.append((f"{N}|mlp_term", f"{ns[0]}|mlp_term"))   # the N-TREND row
            pairs.append((f"{N}|mlp_cond", f"{ns[0]}|mlp_cond"))
    for c in ("mlp_term", "mlp_cond"):
        k = f"{ns[-1]}|{c}_long"
        if k in preds:
            pairs.append((k, f"{ns[-1]}|{c}"))
            pairs.append((k, "online"))

    out = {"dir": a.dir, "preds": a.preds, "ns": ns,
           "n_holdout_states": int(len(held)),
           "n_holdout_battles": int(len(z["held_b"])),
           "conditions": cols, "pairs": [list(p) for p in pairs],
           "conditional_target": fitinfo["conditional"], "fit_log": fitinfo["fits"],
           "fit_frame": {k: fitinfo[k] for k in ("holdout_battles", "pool_battles", "n_states",
                                                 "n_battles", "n_teams", "batch", "lr",
                                                 "max_steps", "eval_every", "patience",
                                                 "long_multiplier")}}

    print("spread (cell key) ...", flush=True)
    for key in ("cell", "opponent"):
        pt, draws, cinfo = spread_block(hm, preds, key, a.boot_spread, a.seed)
        blk = {"info": cinfo,
               "point": {k: (None if not np.isfinite(v) else round(float(v), 4))
                         for k, v in pt.items()},
               "ci": {k: q(v) for k, v in draws.items()}, "delta": {}}
        for ca, cb in pairs:
            for suf in [f"{bk}|ratio" for bk in BUCKETS] + [f"{bk}|ratio_raw" for bk in BUCKETS]:
                ka, kb = f"{ca}|{suf}", f"{cb}|{suf}"
                if ka not in draws or kb not in draws:
                    continue
                dr = np.asarray(draws[ka], float) - np.asarray(draws[kb], float)
                d = pt[ka] - pt[kb]
                blk["delta"][f"{ca}-{cb}|{suf}"] = {
                    "point": round(float(d), 4) if np.isfinite(d) else None, "ci": q(dr)}
        out[f"spread_{key}"] = blk
        print(f"  spread[{key}] done", flush=True)

    print("brier + calibration ...", flush=True)
    sb = scalar_block(hm, preds, a.boot_brier, a.seed)
    out["scalar"] = {}
    for bk, blk in sb.items():
        d = blk["draws"]
        entry = {"point": blk["point"], "ci": {k: q(v, 5) for k, v in d.items()}, "delta": {}}
        for ca, cb in pairs:
            for met in ("brier", "resolution", "calib_slope"):
                ka, kb = f"{ca}|{met}", f"{cb}|{met}"
                if ka not in d or kb not in d:
                    continue
                dr = np.asarray(d[ka], float) - np.asarray(d[kb], float)
                entry["delta"][f"{ca}-{cb}|{met}"] = {
                    "point": round(float(blk["point"][ca][met] - blk["point"][cb][met]), 5)
                    if blk["point"][ca][met] is not None and blk["point"][cb][met] is not None
                    else None, "ci": q(dr, 5)}
        out["scalar"][bk] = entry

    if not a.no_decode:
        print("decodes ...", flush=True)
        out["decode"] = decode_block(meta, held, preds, pooled, a.seed, a.perm,
                                     a.boot_decode, pairs)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot-spread", type=int, default=1500)
    ap.add_argument("--boot-brier", type=int, default=800)
    ap.add_argument("--boot-decode", type=int, default=800)
    ap.add_argument("--perm", type=int, default=120)
    ap.add_argument("--no-decode", action="store_true")
    ap.add_argument("--seed", type=int, default=20260910)
    main(ap.parse_args())
