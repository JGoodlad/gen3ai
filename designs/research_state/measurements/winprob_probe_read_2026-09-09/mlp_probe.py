"""THE NON-LINEARITY CHECK — a small MLP probe on the headline cells.

A linear probe can miss a non-linearly encoded quantity, and "the value path does not carry it"
is exactly the claim that failure mode would fake. So the headline cells are re-run with a
1-hidden-layer MLP (64 tanh units, weight decay, early-stopped on an inner grouped split) under
the SAME grouped-by-battle folds, the SAME IPW weights, the SAME battle-clustered bootstrap and
the SAME permutation null as `decode.py`.

Read it as a CEILING on the linear read, never as a replacement: if the MLP recovers what the
ridge does not, the signal is present but non-linearly coded, and the reading changes.

CPU only, tiny (a few thousand parameters), `torch` with 2 threads.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from decode import (BUCKETS, FSETS, _fold_map, _score, build_targets, cap_per_battle,
                    first_of_group)

HIDDEN = 64
EPOCHS = 300


def _fit_mlp(Xtr, ytr, wtr, Xte, task, seed, wd=1e-3):
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    Ztr = torch.as_tensor(((Xtr - mu) / sd).astype(np.float32))
    Zte = torch.as_tensor(((Xte - mu) / sd).astype(np.float32))
    yt = torch.as_tensor(ytr.astype(np.float32))
    wt = torch.as_tensor((wtr / wtr.mean()).astype(np.float32))
    net = torch.nn.Sequential(torch.nn.Linear(Ztr.shape[1], HIDDEN), torch.nn.Tanh(),
                              torch.nn.Linear(HIDDEN, 1))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=wd)
    ym, ys = float(yt.mean()), float(yt.std()) or 1.0
    tgt = (yt - ym) / ys
    for _ in range(EPOCHS):
        opt.zero_grad()
        p = net(Ztr).reshape(-1)
        loss = (wt * (p - tgt) ** 2).mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return (net(Zte).reshape(-1).numpy() * ys + ym)


def oof_mlp(X, y, w, groups, task, seed, k=5):
    folds = _fold_map(groups, k, seed)
    pred = np.full(len(y), np.nan)
    for f in range(k):
        te, tr = np.where(folds == f)[0], np.where(folds != f)[0]
        if len(te) == 0 or len(tr) < 20:
            continue
        pred[te] = _fit_mlp(X[tr], y[tr], w[tr], X[te], task, seed + f)
    return pred


def main(a):
    feats = {k: np.load(f"{a.dir}/{k}.npy") for k in ("raw", "pi", "vf", "pooled")}
    meta = np.load(f"{a.dir}/meta.npy")
    feats["V"] = meta["V_fwd"].astype(np.float32)[:, None]
    tgts, _b, _bi, _bt = build_targets(meta)
    lo, hi = BUCKETS[a.bucket]
    out = {"dir": a.dir, "bucket": a.bucket, "hidden": HIDDEN, "epochs": EPOCHS, "cells": []}
    for tname in a.targets.split(","):
        tgt = tgts[tname]
        if "ovr" in tgt:
            print(f"skip {tname}: one-vs-rest macro target is linear-probe only")
            continue
        idx = cap_per_battle(meta, (meta["turn"] >= lo) & (meta["turn"] <= hi) & tgt["mask"],
                             a.cap, a.seed)
        y = tgt["y"][idx].astype(float)
        w = meta["w"][idx].astype(float)
        g = meta["battle"][idx]
        sets = [s for s in FSETS if s != "V"]
        preds = {k: oof_mlp(np.asarray(feats[k])[idx].astype(np.float64), y, w, g,
                            tgt["task"], a.seed) for k in sets}
        pt = {k: _score(y, preds[k], w, tgt["task"]) for k in sets}
        # battle-clustered CI + a permutation null through the identical pipeline
        rng = np.random.default_rng(a.seed + 3)
        uq, inv = np.unique(g, return_inverse=True)
        members = [np.where(inv == i)[0] for i in range(len(uq))]
        draws = {k: [] for k in sets}
        for _ in range(a.boot):
            pick = rng.integers(0, len(uq), len(uq))
            sel = np.concatenate([members[i] for i in pick])
            for k in sets:
                draws[k].append(_score(y[sel], preds[k][sel], w[sel], tgt["task"]))
        from decode import _perm_labels
        b_team = meta["team"][idx][first_of_group(inv, len(uq))]
        rngp = np.random.default_rng(a.seed + 77)
        nulls = {k: [] for k in sets}
        for _ in range(a.perm):
            yp = _perm_labels(y, inv, b_team, tgt["perm"], rngp)
            for k in sets:
                nulls[k].append(_score(yp, oof_mlp(np.asarray(feats[k])[idx].astype(np.float64),
                                                   yp, w, g, tgt["task"], a.seed), w, tgt["task"]))

        def q(v):
            v = np.asarray(v, float)
            v = v[~np.isnan(v)]
            return [None, None] if len(v) < 10 else [round(float(np.percentile(v, 2.5)), 4),
                                                     round(float(np.percentile(v, 97.5)), 4)]
        cell = {"target": tname, "task": tgt["task"], "n_states": int(len(idx)),
                "n_battles": int(len(uq)),
                "score": {k: round(float(pt[k]), 4) for k in sets},
                "ci": {k: q(v) for k, v in draws.items()},
                "null": {k: {"mean": round(float(np.nanmean(v)), 4),
                             "p95": round(float(np.nanpercentile(v, 95)), 4), "n": len(v)}
                         for k, v in nulls.items()}}
        out["cells"].append(cell)
        print(tname, cell["score"], flush=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bucket", default="t1_3")
    ap.add_argument("--targets", default="opp_elo,opp_class,own_team_wr")
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--cap", type=int, default=2)
    ap.add_argument("--perm", type=int, default=8)
    ap.add_argument("--boot", type=int, default=1000)
    main(ap.parse_args())
