"""THE N-CURVE: refit the win head on a STATIONARY dataset at 1k -> ~21k battles and watch the
conditioning meters as the amount of frozen-policy data grows.

The head refit (`winprob_head_refit_2026-09-09`) showed that the TERMINAL 0/1 label reproduces the
online conditioning failure offline, from a FROZEN policy, at ~951 battles — and argued the result
is scale-free because the between-cell share of the objective is a property of the loss and not of
n. That is an ARGUMENT. This measures it: the same refit, the same architecture, the same target,
on nested subsets of ~24,000 battles drawn from ONE checkpoint against ONE opponent set.

THE OWNER'S QUESTION (2026-09-10): would the win head condition if the policy moved SLOWER — if
non-stationarity were removed so the head could accumulate data on a fixed target? A stationary
dataset 23x the refit's frame is the direct test, and the three readings are registered in §4 of
the README before any number is produced:

  (i)   the terminal-label curves RISE with N toward the conditional target's level
        => non-stationarity / discard is the binding constraint; a slower policy or a value replay
           on a stationary window helps, and the curve says by how much per doubling.
  (ii)  they stay FLAT from 1k to 21k while the conditional target conditions at EVERY N
        => the target's NOISE SHARE is the constraint; more stationary data does not help, the
           online arms' null is explained, and the owner's hypothesis is refuted FOR THIS REGIME.
  (iii) they rise only past ~10k battles
        => quantify the data an online head would need per stationary window against what a
           rollout supplies (48 envs x 2048 steps ~ 3,300 episodes).

DESIGN DIFFERENCES FROM THE HEAD REFIT, and why each one:
  * ONE FIXED HELD-OUT BATTLE SET, drawn FIRST and excluded from every training subset, instead of
    5-fold CV. An N-curve compares fits at different N, so they must be scored on the SAME states;
    with k-fold, the held-out population would itself change with N and the curve would confound
    "more training data" with "a different test set". It also buys 6x the compute back.
  * THE OPTIMISATION BUDGET IS IN GRADIENT STEPS, NOT EPOCHS. An epoch at 21k battles is 21x an
    epoch at 1k, so "the same number of epochs" hands the large-N fits 21x the optimisation and
    makes a rise in the curve uninterpretable. Every fit here gets the same step budget, the same
    validation cadence and the same patience, and `--long` re-runs the largest N with 5x the steps
    and NO early stopping as the separate "not enough passes" control.
  * THE CONDITIONAL TARGET IS BUILT FROM THE TRAINING SUBSET ONLY, and applied to the held-out
    battles WITHOUT leave-one-out (they contributed nothing to it). So the ceiling row is honestly
    out of sample at every N, and the conditional label's OWN improvement with N is part of the
    curve rather than a constant handed to every point.

HT weights are 1.0 everywhere — `extract.py` asserts full capture — so the IPW machinery is kept
but inert, and every "weighted" mean here is an unweighted one. Said once, here, rather than
implied by silence.

CPU only, nothing written under models/.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

SHRINK_GRID = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 1e9)
CONDITIONS = ("lin_term", "mlp_term", "lin_cond", "mlp_cond")


# ──────────────────────────────────────────────────────────────────────────────
def battle_level(meta):
    battles, first, binv = np.unique(meta["battle"], return_index=True, return_inverse=True)
    return {"battle": battles, "y": meta["y"][first].astype(float),
            "w": meta["w"][first].astype(float), "team": meta["team"][first],
            "opponent": meta["opponent"][first], "cycle": meta["cycle"][first],
            "true_wr": meta["true_wr"][first], "n_games": meta["n_games"][first],
            "binv": binv, "n": len(battles)}


def _group_sums(codes, y, w, n_groups):
    return (np.bincount(codes, weights=w, minlength=n_groups),
            np.bincount(codes, weights=w * y, minlength=n_groups),
            np.bincount(codes, minlength=n_groups).astype(float))


def conditional_factors(bl, idx, key):
    """Fit the additive conditional target's factors on the battles `idx` ONLY.

    `p = clip(base + s_o*(opp_wr - base) + s_t*(team_wr - base))`, additive in (opponent, own team)
    on the win-rate scale, each factor shrunk by n/(n+k). The two k are chosen by minimising the
    weighted battle-level log loss of the LEAVE-ONE-BATTLE-OUT prediction inside `idx` — honest,
    because the label of battle i is built without battle i.

    Returns (loo_label_on_idx, apply_fn, info) where `apply_fn(sel)` scores ANY battle set with the
    full (non-LOO) factors; for held-out battles that is exactly right, since they contributed
    nothing to the factors in the first place.
    """
    y, w = bl["y"][idx], bl["w"][idx]
    if key == "opponent":
        okey = bl["opponent"][idx].astype("U40")
    else:
        okey = np.array([f"{c}|{o}" for c, o in zip(bl["cycle"][idx], bl["opponent"][idx])])
    ouniq, ocode = np.unique(okey, return_inverse=True)
    tuniq, tcode = np.unique(bl["team"][idx], return_inverse=True)
    sw, swy = w.sum(), (w * y).sum()
    base_loo = (swy - w * y) / (sw - w)
    base_full = swy / sw
    o_sw, o_swy, o_cnt = _group_sums(ocode, y, w, len(ouniq))
    t_sw, t_swy, t_cnt = _group_sums(tcode, y, w, len(tuniq))

    def _loo(swg, swyg, code):
        den = swg[code] - w
        ok = den > 1e-12
        return np.where(ok, (swyg[code] - w * y) / np.where(ok, den, 1.0), np.nan)

    _o, _t = _loo(o_sw, o_swy, ocode), _loo(t_sw, t_swy, tcode)
    o_loo = np.where(np.isfinite(_o), _o, base_loo)
    t_loo = np.where(np.isfinite(_t), _t, base_loo)
    o_n, t_n = o_cnt[ocode] - 1.0, t_cnt[tcode] - 1.0

    def build(ko, kt, base, owr, twr, on, tn):
        # a group with NO other battles (LOO count 0) must shrink to the base rate, not to 0/0
        so = (np.where(on + ko > 0, on / np.where(on + ko > 0, on + ko, 1.0), 0.0)
              if ko < 1e8 else np.zeros_like(on))
        st = (np.where(tn + kt > 0, tn / np.where(tn + kt > 0, tn + kt, 1.0), 0.0)
              if kt < 1e8 else np.zeros_like(tn))
        return np.clip(base + so * (owr - base) + st * (twr - base), 0.02, 0.98)

    best, best_k, grid = np.inf, (0.0, 0.0), []
    for ko in SHRINK_GRID:
        for kt in SHRINK_GRID:
            p = build(ko, kt, base_loo, o_loo, t_loo, o_n, t_n)
            ll = float(-(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum() / w.sum())
            grid.append({"k_opp": ko, "k_team": kt, "logloss": round(ll, 6)})
            if ll < best:
                best, best_k = ll, (ko, kt)
    p_loo = build(*best_k, base_loo, o_loo, t_loo, o_n, t_n)

    o_wr_full = np.where(o_sw > 0, o_swy / np.where(o_sw > 0, o_sw, 1.0), base_full)
    t_wr_full = np.where(t_sw > 0, t_swy / np.where(t_sw > 0, t_sw, 1.0), base_full)
    omap = {k: i for i, k in enumerate(ouniq)}
    tmap = {k: i for i, k in enumerate(tuniq)}

    def apply_fn(sel):
        if key == "opponent":
            k2 = bl["opponent"][sel].astype("U40")
        else:
            k2 = np.array([f"{c}|{o}" for c, o in zip(bl["cycle"][sel], bl["opponent"][sel])])
        oi = np.array([omap.get(k, -1) for k in k2])
        ti = np.array([tmap.get(k, -1) for k in bl["team"][sel]])
        owr = np.where(oi >= 0, o_wr_full[np.clip(oi, 0, None)], base_full)
        twr = np.where(ti >= 0, t_wr_full[np.clip(ti, 0, None)], base_full)
        on = np.where(oi >= 0, o_cnt[np.clip(oi, 0, None)], 0.0)
        tn = np.where(ti >= 0, t_cnt[np.clip(ti, 0, None)], 0.0)
        return build(*best_k, base_full, owr, twr, on, tn)

    info = {"k_opp": best_k[0], "k_team": best_k[1], "logloss_loo": round(best, 6),
            "logloss_marginal": float(-(w * (y * np.log(base_loo)
                                             + (1 - y) * np.log(1 - base_loo))).sum() / w.sum()),
            "opponent_key": key, "n_opponent_cells": int(len(ouniq)), "n_teams": int(len(tuniq)),
            "base_rate": float(base_full),
            "corr_loo_with_own_outcome": float(np.corrcoef(p_loo, y)[0, 1]),
            "sd": float(np.std(p_loo)), "grid": grid}
    return p_loo, apply_fn, info


# ──────────────────────────────────────────────────────────────────────────────
def _win_prob_head(torch, d=128):
    """`agents.model.aux_value_heads.WinProbHead` verbatim: LayerNorm -> Linear -> ReLU -> Linear."""
    return torch.nn.Sequential(torch.nn.LayerNorm(d), torch.nn.Linear(d, d),
                               torch.nn.ReLU(), torch.nn.Linear(d, 1))


def _linear_head(torch, d=128):
    """The CAPACITY FLOOR: the same LayerNorm, no hidden layer, so the contrast is the hidden
    layer and not feature scaling."""
    return torch.nn.Sequential(torch.nn.LayerNorm(d), torch.nn.Linear(d, 1))


def fit_head(Xt, yt, wt, tr, va, arch, seed, lr, batch, max_steps, eval_every, patience,
             early_stop=True):
    """Train ONE head with a STEP budget (not an epoch budget) so every N gets the same
    optimisation. Validation every `eval_every` steps on a grouped-by-battle split of the
    training battles; stop after `patience` checks without improvement."""
    import torch
    torch.manual_seed(seed)
    net = _win_prob_head(torch) if arch == "mlp" else _linear_head(torch)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    Xv, yv, wv = Xt[va], yt[va], wt[va]
    wv = wv / wv.mean()
    rng = np.random.default_rng(seed)
    ntr = len(tr)
    best, best_state, best_step, bad, step = np.inf, None, 0, 0, 0
    perm, pos = rng.permutation(ntr), 0
    while step < max_steps:
        if pos + batch > ntr:
            perm, pos = rng.permutation(ntr), 0
        j = tr[perm[pos:pos + batch]]
        pos += batch
        wb = wt[j] / wt[j].mean()
        opt.zero_grad()
        ((bce(net(Xt[j]).reshape(-1), yt[j], reduction="none") * wb).mean()).backward()
        opt.step()
        step += 1
        if step % eval_every == 0:
            with torch.no_grad():
                vl = float((bce(net(Xv).reshape(-1), yv, reduction="none") * wv).mean())
            if vl < best - 1e-6:
                best, bad, best_step = vl, 0, step
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if early_stop and bad >= patience:
                    break
    if early_stop and best_state is not None:
        net.load_state_dict(best_state)
    return net, {"steps_run": step, "best_step": best_step, "best_val_bce": round(best, 6),
                 "early_stop": bool(early_stop), "stopped_early": bool(step < max_steps)}


def predict(net, X, idx, batch=16384):
    import torch
    out = np.empty(len(idx), np.float64)
    with torch.no_grad():
        for i in range(0, len(idx), batch):
            j = idx[i:i + batch]
            out[i:i + batch] = torch.sigmoid(net(X[j]).reshape(-1)).numpy()
    return out


# ──────────────────────────────────────────────────────────────────────────────
def main(a):
    import torch
    torch.set_num_threads(a.threads)
    t0 = time.time()
    meta = np.load(f"{a.dir}/meta.npy")
    X = np.load(f"{a.dir}/pooled.npy")
    assert len(X) == len(meta)
    bl = battle_level(meta)
    binv = bl["binv"]
    rng = np.random.default_rng(a.seed)

    # ── the HELD-OUT battle set, drawn FIRST, stratified by OPPONENT so every cell keeps the
    # same number of scored battles at every N (the identity is a spread ACROSS opponents; an
    # unstratified draw would give it a different cell-size profile per substrate).
    opps = np.unique(bl["opponent"])
    per = a.holdout // len(opps)
    held_b = []
    for o in opps:
        w = np.where(bl["opponent"] == o)[0]
        held_b.append(rng.permutation(w)[:per])
    held_b = np.sort(np.concatenate(held_b))
    pool_b = np.setdiff1d(np.arange(bl["n"]), held_b)
    pool_b = rng.permutation(pool_b)                    # ONE order; subsets are nested prefixes

    held_mask = np.zeros(bl["n"], bool)
    held_mask[held_b] = True
    held_states = np.where(held_mask[binv])[0]

    ns = [n for n in a.ns if 0 < n <= len(pool_b)]
    if a.ns and a.ns[-1] == 0:
        ns.append(len(pool_b))
    ns = sorted(set(ns))

    Xt = torch.as_tensor(X)
    y_term_all = meta["y"].astype(float)
    w_all = meta["w"].astype(float)
    yt_term = torch.as_tensor(y_term_all.astype(np.float32))
    wt = torch.as_tensor(w_all.astype(np.float32))

    out = {"n": {}, "conditional": {}, "fits": {}}
    preds = {}
    info = {"dir": a.dir, "seed": a.seed, "holdout_battles": int(len(held_b)),
            "holdout_states": int(len(held_states)), "pool_battles": int(len(pool_b)),
            "n_states": int(len(meta)), "n_battles": int(bl["n"]),
            "n_teams": int(len(np.unique(bl["team"]))), "ns": ns,
            "opponent_key": a.cond_key, "batch": a.batch, "lr": a.lr,
            "max_steps": a.max_steps, "eval_every": a.eval_every, "patience": a.patience,
            "long_multiplier": a.long_mult, "long_cap": a.long_cap}

    for N in ns:
        sub_b = np.sort(pool_b[:N])
        # grouped early-stopping split of the TRAINING battles
        r2 = np.random.default_rng(a.seed + 991)
        pick = r2.permutation(len(sub_b))
        n_va = max(1, int(round(a.val_frac * len(sub_b))))
        va_mask = np.zeros(bl["n"], bool)
        va_mask[sub_b[pick[:n_va]]] = True
        sub_mask = np.zeros(bl["n"], bool)
        sub_mask[sub_b] = True
        in_sub = sub_mask[binv]
        is_va = va_mask[binv]
        tr = np.where(in_sub & ~is_va)[0]
        va = np.where(in_sub & is_va)[0]

        p_loo_b, apply_fn, cinfo = conditional_factors(bl, sub_b, a.cond_key)
        y_cond_all = np.zeros(len(meta))
        y_cond_all[in_sub] = p_loo_b[np.searchsorted(sub_b, binv[in_sub])]
        yt_cond = torch.as_tensor(y_cond_all.astype(np.float32))
        # the CEILING row on the held-out battles: the target itself, factors from the subset,
        # no LOO needed (a held-out battle contributed nothing to them).
        cond_held_b = apply_fn(held_b)
        preds[f"{N}|cond_oracle"] = cond_held_b[np.searchsorted(held_b, binv[held_states])]

        spec = {"lin_term": ("lin", yt_term), "mlp_term": ("mlp", yt_term),
                "lin_cond": ("lin", yt_cond), "mlp_cond": ("mlp", yt_cond)}
        out["conditional"][str(N)] = {k: v for k, v in cinfo.items() if k != "grid"}
        out["fits"][str(N)] = {}
        for cname, (arch, yv) in spec.items():
            net, log = fit_head(Xt, yv, wt, tr, va, arch, a.seed + 7, a.lr, a.batch,
                                a.max_steps, a.eval_every, a.patience)
            preds[f"{N}|{cname}"] = predict(net, Xt, held_states)
            out["fits"][str(N)][cname] = log
            print(f"  N={N} {cname}: {log['steps_run']} steps (best {log['best_step']}) "
                  f"val {log['best_val_bce']:.5f}  [{time.time()-t0:.0f}s]", flush=True)
        out["n"][str(N)] = {"battles": int(N), "states_train": int(len(tr)),
                            "states_val": int(len(va))}

        # ── the MORE-OPTIMISATION control, at the largest N only: no early stopping, 5x steps.
        if a.long_mult and N == ns[-1]:
            for cname, arch in (("mlp_term", "mlp"), ("mlp_cond", "mlp")):
                used = out["fits"][str(N)][cname]["steps_run"]
                ms = min(a.long_cap, int(a.long_mult * used))
                net, log = fit_head(Xt, spec[cname][1], wt, tr, va, arch, a.seed + 7, a.lr,
                                    a.batch, ms, a.eval_every, a.patience, early_stop=False)
                preds[f"{N}|{cname}_long"] = predict(net, Xt, held_states)
                out["fits"][str(N)][f"{cname}_long"] = log
                print(f"  N={N} {cname}_long: {log['steps_run']} steps (no early stop) "
                      f"[{time.time()-t0:.0f}s]", flush=True)

    np.savez(a.out, held_states=held_states, held_b=held_b,
             y=y_term_all[held_states], w=w_all[held_states],
             online=meta["V_fwd"][held_states].astype(float),
             online_rec=meta["V_rec"][held_states].astype(float),
             **preds)
    info.update(out)
    info["seconds"] = round(time.time() - t0, 1)
    with open(a.out.replace(".npz", "_fit.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps({k: v for k, v in info.items() if k not in ("fits", "conditional", "n")},
                     indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ns", default="1000,2000,4000,8000,16000,0",
                    help="battle counts; a trailing 0 means 'every remaining battle'")
    ap.add_argument("--holdout", type=int, default=2400)
    ap.add_argument("--cond-key", default="cell", choices=("cell", "opponent"))
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--max-steps", type=int, default=60000)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--patience", type=int, default=400)
    ap.add_argument("--long-mult", type=float, default=5.0)
    ap.add_argument("--long-cap", type=int, default=250000)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260910)
    a = ap.parse_args()
    a.ns = [int(x) for x in a.ns.split(",")]
    main(a)
