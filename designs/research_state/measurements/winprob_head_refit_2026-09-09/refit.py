"""REFIT ONLY THE WIN HEAD on a frozen `value_pooled`, and see which condition recovers the
separation the online head does not have.

The probe read (`winprob_probe_read_2026-09-09`) established that `value_pooled` — the tensor the
win head literally reads — decodes the opponent's class at AUC 0.846 and the trainee's own-team
win rate at R² 0.671 on turn-1 states, while `V = sigmoid(win_head(value_pooled))` reads 0.532 and
0.010. One MLP sits between those numbers. Three things could put it there, and they have three
different treatments:

  (1) the ONLINE OPTIMISATION  — the target is learnable from these features at this data scale and
      the online run did not learn it (non-stationary target under a moving policy, the head's
      effective lr / epochs / gradient share, buffer reuse).
      => how the head is TRAINED: a value replay, a periodic head refit, a head-specific lr.
  (2) the LABEL VARIANCE       — one 0/1 draw per episode is too noisy for the head to see the
      conditional through, at its effective sample size.
      => target-variance levers: MC / counterfactual labels, search leaves, outcome balancing.
  (3) the MAPPING              — features -> outcome is not expressible by this head.
      => head CAPACITY, a separate value trunk.

The discriminating experiment is to hold the features frozen and vary only the TARGET and the
OPTIMISER, which is exactly what this script does. Seven conditions, one feature tensor:

  online          the checkpoint's OWN head applied to the same features (= `V_fwd`; no fit).
  lin_term        LOGISTIC head, terminal 0/1 target.          the capacity FLOOR for (a).
  mlp_term        WinProbHead architecture, from SCRATCH,      the (a) row.
                  terminal 0/1 target.
  mlp_term_ft     the same, initialised FROM THE ONLINE HEAD.  the ORDER hazard: if scratch
                                                               separates and the fine-tune does
                                                               not, the online head is in a basin.
  lin_cond        LOGISTIC head, CONDITIONAL target.           the capacity floor for (b).
  mlp_cond        WinProbHead from scratch, CONDITIONAL.       the (b) row.
  mlp_cond_ft     WinProbHead from the online head, CONDITIONAL.
  cond_oracle     the CONDITIONAL TARGET ITSELF, emitted as if it were a prediction. Not a fit —
                  the CEILING row. It says what the meters would read for a head that fitted the
                  conditional target EXACTLY, which is the only way to tell "the target does not
                  carry the separation" (a ceiling near the online head's) apart from "the fit did
                  not reach it" (a ceiling near 1.0 with the fits far below). Without it a failed
                  (b) is uninterpretable.

🚨 EVERY prediction this script emits is OUT OF FOLD, `online` included (it is a fixed function, so
its "OOF" prediction is itself). Comparing an in-sample refit against an out-of-sample online head
would hand the refit the whole overfitting budget as a free win, and at 951 battles that budget is
large — §"train vs held-out" in the README is the report of exactly how large.

THE FOLDS ARE GROUPED BY BATTLE, at both levels: the outer 5-fold that produces the held-out
predictions, and the inner split that early-stops the fit. States inside a battle share the team,
the opponent AND the outcome, so a battle split across folds would let the head read its own label.

EVERY fit and every score is Horvitz-Thompson weighted by the manifest's capture rates: the trace
quota prefers losses (arm A's traced sample wins 0.42 of its battles against a true 0.83), so an
unweighted refit would be a refit for a population that does not exist. `--no-ipw` runs the whole
thing unweighted as the counter-hypothesis.

THE CONDITIONAL TARGET, and why it is not a per-(opponent, team) cell mean. 951 battles over 14
opponents and 216 teams leaves ~0.3 battles in the average joint cell, so the literal cell mean is
undefined almost everywhere. The target is therefore ADDITIVE in the two factors, on the win-rate
scale, with each factor's leave-one-battle-out IPW mean SHRUNK toward the leave-one-out base rate
by n/(n+k), and the two k's chosen by minimising the HT-weighted battle-level log loss of the LOO
prediction itself (an honest out-of-sample criterion, since the label for battle i never contains
battle i). `--cond raw_cell` swaps in the literal per-(opponent, team) LOO cell mean wherever the
cell has >= `--cell-min` battles, backing off to the additive elsewhere, as the sensitivity.
LOO is what stops the label carrying the outcome of the battle it labels — the leakage
counter-hypothesis — and it is checked by `y_cond`'s own correlation with `y` inside a battle.

CPU only, torch with 2 threads, nothing written under models/.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

CONDITIONS = ("online", "lin_term", "mlp_term", "mlp_term_ft",
              "lin_cond", "mlp_cond", "mlp_cond_ft", "cond_oracle")
SHRINK_GRID = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 1e9)
EPS = 1e-6


# ──────────────────────────────────────────────────────────────────────────────
# Folds — grouped by BATTLE, a seeded permutation (never a hash: a hash of a battle id
# correlates with nothing but is not reproducible across a rename of the id scheme).
# ──────────────────────────────────────────────────────────────────────────────
def fold_map(groups, k, seed):
    uniq = np.unique(groups)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(uniq))
    gf = {g: int(perm[i] % k) for i, g in enumerate(uniq)}
    return np.array([gf[g] for g in groups])


def battle_level(meta):
    """One row per battle: outcome, IPW weight, team, opponent, cycle."""
    battles, first, binv = np.unique(meta["battle"], return_index=True, return_inverse=True)
    return {"battle": battles, "y": meta["y"][first].astype(float),
            "w": meta["w"][first].astype(float), "team": meta["team"][first],
            "opponent": meta["opponent"][first], "cycle": meta["cycle"][first],
            "binv": binv, "n": len(battles)}


def _loo_group_wr(codes, y, w, n_groups):
    """Leave-one-battle-out IPW win rate of each battle's GROUP, plus the group's battle count.

    Closed form, so it is exact LOO rather than an approximation of it: the group's weighted sum
    minus this battle's own contribution, over the group's weight minus this battle's weight."""
    sw = np.bincount(codes, weights=w, minlength=n_groups)
    swy = np.bincount(codes, weights=w * y, minlength=n_groups)
    cnt = np.bincount(codes, minlength=n_groups).astype(float)
    den = sw[codes] - w
    ok = den > 1e-12
    wr = np.where(ok, (swy[codes] - w * y) / np.where(ok, den, 1.0), np.nan)
    return wr, cnt[codes]


def conditional_target(bl, kind, cell_min, seed):
    """Per-battle CONDITIONAL win probability — the terminal label with its per-episode variance
    removed, and nothing else changed. Additive in (opponent, own team) on the win-rate scale;
    every ingredient leave-one-battle-out and IPW-weighted."""
    y, w = bl["y"], bl["w"]
    # 🚨 THE OPPONENT FACTOR IS KEYED BY (CYCLE, OPPONENT), NOT BY OPPONENT. The identity the
    # readout scores is computed over (cycle, opponent) CELLS — that is the unit the manifest
    # reports a 100-game win rate for — and a trainee's own strength moves between cycles. On the
    # ladder CONTROL it moves enormously (its per-cycle mean true win rate runs 0.547 at 2M to
    # 0.896 at 8M), so a target with no cycle term cannot reproduce the cell means however well it
    # is fitted, and its CEILING reads ~0.42 for a reason that has nothing to do with the head.
    # Keying the factor by the cell removes that confound and leaves (a) untouched — the terminal
    # target has no factors at all. `--cond additive_pooled` restores the opponent-only key as the
    # sensitivity, and the two differ by almost nothing on arm A, whose four cycles are flat.
    if kind == "additive_pooled":
        okey = bl["opponent"].astype("U40")
    else:
        okey = np.array([f"{c}|{o}" for c, o in zip(bl["cycle"], bl["opponent"])])
    ocode = np.unique(okey, return_inverse=True)[1]
    tcode = np.unique(bl["team"], return_inverse=True)[1]
    # LOO base rate: the overall IPW win rate with this battle removed.
    sw, swy = w.sum(), (w * y).sum()
    base = (swy - w * y) / (sw - w)
    o_wr, o_n = _loo_group_wr(ocode, y, w, ocode.max() + 1)
    t_wr, t_n = _loo_group_wr(tcode, y, w, tcode.max() + 1)
    o_wr = np.where(np.isfinite(o_wr), o_wr, base)
    t_wr = np.where(np.isfinite(t_wr), t_wr, base)

    def build(ko, kt):
        so = o_n / (o_n + ko) if ko < 1e8 else np.zeros_like(o_n)
        st = t_n / (t_n + kt) if kt < 1e8 else np.zeros_like(t_n)
        return np.clip(base + so * (o_wr - base) + st * (t_wr - base), 0.02, 0.98)

    # k chosen on the LOO prediction's OWN weighted log loss — an honest out-of-sample criterion,
    # because the label of battle i is built without battle i.
    best, best_k = np.inf, (0.0, 0.0)
    grid = []
    for ko in SHRINK_GRID:
        for kt in SHRINK_GRID:
            p = build(ko, kt)
            ll = float(-(w * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum() / w.sum())
            grid.append({"k_opp": ko, "k_team": kt, "logloss": round(ll, 6)})
            if ll < best:
                best, best_k = ll, (ko, kt)
    p = build(*best_k)
    info = {"k_opp": best_k[0], "k_team": best_k[1], "logloss": round(best, 6),
            "opponent_key": "opponent" if kind == "additive_pooled" else "(cycle, opponent)",
            "n_opponent_cells": int(ocode.max() + 1),
            "logloss_marginal": float(-(w * (y * np.log(base) + (1 - y) * np.log(1 - base))).sum()
                                      / w.sum()),
            "grid": grid, "kind": "additive_loo"}

    if kind == "raw_cell":                 # per-(cycle, opponent, team) once okey carries the cycle
        # SENSITIVITY: the literal per-(opponent, team) LOO cell mean where the cell is big enough.
        ccode = np.unique(np.stack([ocode, tcode], 1), axis=0, return_inverse=True)[1]
        c_wr, c_n = _loo_group_wr(ccode, y, w, ccode.max() + 1)
        use = np.isfinite(c_wr) & (c_n >= cell_min)
        info["kind"] = "raw_cell_backoff"
        info["cell_min"] = cell_min
        info["cell_coverage"] = float(use.mean())
        p = np.where(use, np.clip(c_wr, 0.02, 0.98), p)
    info["mean"] = float(np.average(p, weights=w))
    info["sd"] = float(np.sqrt(np.average((p - info["mean"]) ** 2, weights=w)))
    # LEAKAGE CHECK: a LOO label must not track its own battle's outcome beyond what the
    # group means explain. Reported, not asserted — the number is the evidence.
    info["corr_with_own_outcome"] = float(np.corrcoef(p, y)[0, 1])
    return p, info


# ──────────────────────────────────────────────────────────────────────────────
# The heads
# ──────────────────────────────────────────────────────────────────────────────
def _win_prob_head(torch, d=128):
    """The ONLINE architecture, verbatim from `agents.model.aux_value_heads.WinProbHead`:
    LayerNorm(D_MODEL) -> Linear(D,D) -> ReLU -> Linear(D,1). Built here rather than imported so
    the refit does not drag the whole extractor in, and asserted against the checkpoint's own
    parameter shapes by `dump_head.py`."""
    return torch.nn.Sequential(torch.nn.LayerNorm(d), torch.nn.Linear(d, d),
                               torch.nn.ReLU(), torch.nn.Linear(d, 1))


def _linear_head(torch, d=128):
    """The CAPACITY FLOOR: a bare logistic map. It keeps the LayerNorm so the two heads see the
    SAME input distribution and the only difference between them is the hidden layer — otherwise
    'the MLP beat the linear head' could be a statement about feature scaling."""
    return torch.nn.Sequential(torch.nn.LayerNorm(d), torch.nn.Linear(d, 1))


def _load_init(torch, net, head_npz):
    z = np.load(head_npz)
    sd = net.state_dict()
    pairs = [("0.weight", "net.0.weight"), ("0.bias", "net.0.bias"),
             ("1.weight", "net.1.weight"), ("1.bias", "net.1.bias"),
             ("3.weight", "net.3.weight"), ("3.bias", "net.3.bias")]
    for dst, src in pairs:
        if dst not in sd:
            raise SystemExit(f"REFUSED: fine-tune init has no slot {dst} — the refit head is not "
                             f"the online architecture and the basin question is not being asked.")
        sd[dst] = torch.as_tensor(np.asarray(z[src], np.float32))
    net.load_state_dict(sd)
    return net


def fit_head(X, y, w, tr, va, arch, init_npz, seed, lr, max_epochs, patience, batch, wd):
    """Train ONE head on `tr`, early-stop on `va` (a grouped-by-battle split of the training
    battles), return the fitted net. The stopping criterion is the VALIDATION weighted BCE against
    the SAME target the arm is trained on — a scratch head stopped on the terminal label and a
    conditional head stopped on the conditional label, so neither arm is handed the other's
    objective as a tie-break."""
    import torch
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    net = _win_prob_head(torch) if arch == "mlp" else _linear_head(torch)
    if init_npz is not None:
        if arch != "mlp":
            raise SystemExit("REFUSED: only the MLP arm can be initialised from the online head.")
        _load_init(torch, net, init_npz)
    Xt = torch.as_tensor(X.astype(np.float32))
    yt = torch.as_tensor(y.astype(np.float32))
    wt = torch.as_tensor(w.astype(np.float32))
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    Xv, yv, wv = Xt[va], yt[va], wt[va]
    wv = wv / wv.mean()
    best, best_state, bad, ep = np.inf, None, 0, 0
    rng = np.random.default_rng(seed)
    ntr = len(tr)
    for ep in range(1, max_epochs + 1):
        perm = rng.permutation(ntr)
        for i in range(0, ntr, batch):
            j = tr[perm[i:i + batch]]
            wb = wt[j] / wt[j].mean()
            opt.zero_grad()
            loss = (bce(net(Xt[j]).reshape(-1), yt[j], reduction="none") * wb).mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            vl = float((bce(net(Xv).reshape(-1), yv, reduction="none") * wv).mean())
        if vl < best - 1e-6:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return net, {"epochs_run": ep, "best_val_bce": round(best, 6)}


def predict(net, X, batch=8192):
    import torch
    out = np.empty(len(X), np.float64)
    with torch.no_grad():
        for i in range(0, len(X), batch):
            out[i:i + batch] = torch.sigmoid(
                net(torch.as_tensor(X[i:i + batch].astype(np.float32))).reshape(-1)).numpy()
    return out


# ──────────────────────────────────────────────────────────────────────────────
def main(a):
    t0 = time.time()
    meta = np.load(f"{a.dir}/meta.npy")
    X = np.load(f"{a.dir}/pooled.npy")
    n = len(meta)
    assert len(X) == n, (len(X), n)
    w = meta["w"].astype(float) if not a.no_ipw else np.ones(n)
    bl = battle_level(meta)
    binv = bl["binv"]

    y_term = meta["y"].astype(float)
    p_cond_b, cond_info = conditional_target(bl, a.cond, a.cell_min, a.seed)
    y_cond = p_cond_b[binv]

    folds_b = fold_map(bl["battle"], a.k, a.seed)          # per BATTLE
    folds = folds_b[binv]                                  # broadcast to states
    preds = {c: np.full(n, np.nan) for c in CONDITIONS}
    preds_train = {c: np.full(n, np.nan) for c in CONDITIONS}
    preds["online"] = meta["V_fwd"].astype(float)
    preds_train["online"] = meta["V_fwd"].astype(float)
    preds["cond_oracle"] = y_cond.copy()          # the ceiling row: the target, not a fit
    preds_train["cond_oracle"] = y_cond.copy()

    spec = {"lin_term": ("lin", y_term, None), "mlp_term": ("mlp", y_term, None),
            "mlp_term_ft": ("mlp", y_term, a.head), "lin_cond": ("lin", y_cond, None),
            "mlp_cond": ("mlp", y_cond, None), "mlp_cond_ft": ("mlp", y_cond, a.head)}
    fitlog = {c: [] for c in spec}

    for f in range(a.k):
        te = np.where(folds == f)[0]
        tr_all_b = np.where(folds_b != f)[0]
        # inner grouped split of the TRAINING battles for early stopping
        rng = np.random.default_rng(a.seed + 991 + f)
        pick = rng.permutation(len(tr_all_b))
        n_va = max(1, int(round(a.val_frac * len(tr_all_b))))
        va_b = set(bl["battle"][tr_all_b[pick[:n_va]]].tolist())
        in_tr = (folds != f)
        is_va = np.array([b in va_b for b in meta["battle"]])
        tr = np.where(in_tr & ~is_va)[0]
        va = np.where(in_tr & is_va)[0]
        for cname, (arch, yv, init) in spec.items():
            net, log = fit_head(X, yv, w, tr, va, arch, init, a.seed + 7 * f, a.lr,
                                a.max_epochs, a.patience, a.batch, a.wd)
            preds[cname][te] = predict(net, X[te])
            preds_train[cname][tr] = predict(net, X[tr])
            log.update(fold=f, n_tr=len(tr), n_va=len(va), n_te=len(te))
            fitlog[cname].append(log)
        print(f"  fold {f}: {time.time()-t0:.0f}s", flush=True)

    for c in CONDITIONS:
        if np.isnan(preds[c]).any():
            raise SystemExit(f"REFUSED: condition {c} left {int(np.isnan(preds[c]).sum())} states "
                             f"without a held-out prediction.")

    np.savez(a.out, y_term=y_term, y_cond=y_cond, folds=folds, w=w,
             **{f"p_{c}": preds[c] for c in CONDITIONS},
             **{f"ptr_{c}": preds_train[c] for c in CONDITIONS})
    info = {"dir": a.dir, "n_states": n, "n_battles": bl["n"], "k": a.k, "seed": a.seed,
            "ipw": not a.no_ipw, "lr": a.lr, "max_epochs": a.max_epochs, "patience": a.patience,
            "batch": a.batch, "weight_decay": a.wd, "val_frac": a.val_frac,
            "conditional_target": cond_info, "fits": fitlog,
            "seconds": round(time.time() - t0, 1)}
    with open(a.out.replace(".npz", "_fit.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps({k: v for k, v in info.items() if k not in ("fits", "conditional_target")},
                     indent=1))
    print("cond target:", {k: v for k, v in cond_info.items() if k != "grid"})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="the probe read's extract dir")
    ap.add_argument("--head", required=True, help="head.npz from dump_head.py (fine-tune init)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cond", default="additive",
                    choices=("additive", "additive_pooled", "raw_cell"))
    ap.add_argument("--cell-min", type=int, default=3)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-epochs", type=int, default=1500)
    ap.add_argument("--patience", type=int, default=60)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--no-ipw", action="store_true")
    ap.add_argument("--seed", type=int, default=20260909)
    main(ap.parse_args())
