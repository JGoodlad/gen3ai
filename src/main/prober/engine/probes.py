"""Representation probing — fit a small LINEAR probe on the model's internal activations."""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Representation probing — fit a small LINEAR probe on the model's INTERNAL
# activations to predict a derived game quantity (is-faster, damage-taken,
# faint-soon). The decisive test of "is X already in the representation": if a
# linear probe recovers X from the trunk embedding, the model HAS computed it
# (so handing X over as a feature is redundant); if a linear probe CAN'T, X is
# an EXTRACTION gap — a real obs lever (per the provide-vs-learn rule, "let it
# learn" has hit this small net's capacity wall for X, so provide it).
#
# Pure numpy (no sklearn). Standardized ridge (regression) / logistic
# (classification) with k-fold OUT-OF-FOLD predictions, scored overall AND per
# group — so we see whether the representation knows X on the HARD/contested
# cases, not just on average (a model can encode speed on an obvious matchup yet
# fail the Leftovers/Sandstorm-timing inference exactly where it's decision-relevant).
# ---------------------------------------------------------------------------

def _kfold_indices(n: int, k: int, seed: int) -> "list[np.ndarray]":
    """k interleaved test folds over a seeded permutation (deterministic)."""
    perm = np.random.default_rng(seed).permutation(n)
    return [perm[i::k] for i in range(min(k, n))]


def _standardize(train: np.ndarray, test: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    mu = train.mean(0)
    sd = train.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return (train - mu) / sd, (test - mu) / sd


def _ridge_fit(X: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + l2 * np.eye(d), X.T @ y)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


def _logistic_fit(X: np.ndarray, y: np.ndarray, l2: float,
                  iters: int = 400, lr: float = 0.5) -> "tuple[np.ndarray, float]":
    """L2-regularized logistic regression by full-batch gradient descent on
    standardized inputs (robust + dependency-free; converges in a few hundred steps)."""
    n, d = X.shape
    w = np.zeros(d)
    b = 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        for _ in range(iters):
            p = _sigmoid(X @ w + b)
            g = p - y
            w_new = w - lr * (X.T @ g / n + l2 * w / n)
            b -= lr * g.mean()
            if not np.isfinite(w_new).all():    # a weak-l2 grid point diverged — keep the last finite w
                break
            w = w_new
    return w, b


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based ROC AUC (Mann–Whitney); nan if a class is absent."""
    pos, neg = p[y == 1], p[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allp = np.concatenate([pos, neg])
    ranks = allp.argsort().argsort().astype(float) + 1.0  # average-ish ranks (ties rare on floats)
    r_pos = ranks[:len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


_L2_GRID = (0.1, 1.0, 10.0, 100.0, 1000.0)


def _oof_predict(X, y, task, l2, folds, seed) -> np.ndarray:
    """Out-of-fold predictions for one l2 (every row scored by a model that didn't see it)."""
    n = len(y)
    oof = np.full(n, np.nan)
    for te in _kfold_indices(n, folds, seed):
        tr = np.setdiff1d(np.arange(n), te)
        if len(tr) < 2:
            continue
        Xtr, Xte = _standardize(X[tr], X[te])
        if task == "classification":
            w, b = _logistic_fit(Xtr, y[tr], l2)
            oof[te] = _sigmoid(Xte @ w + b)
        else:
            yc = y[tr].mean()
            oof[te] = Xte @ _ridge_fit(Xtr, y[tr] - yc, l2) + yc
    return oof


def _selection_score(y, oof, task) -> float:
    """The scalar used to pick l2: AUC (classification) / R² (regression), over the usable rows."""
    ok = ~np.isnan(oof)
    yy, pp = y[ok], oof[ok]
    if len(yy) < 5:
        return -np.inf
    if task == "classification":
        a = _auc(yy, pp)
        return a if not np.isnan(a) else -np.inf
    ss_tot = float(((yy - yy.mean()) ** 2).sum())
    return 1.0 - float(((yy - pp) ** 2).sum()) / ss_tot if ss_tot > 0 else -np.inf


def fit_probe(X, y, task: str, groups=None, seed: int = 0, folds: int = 5, l2=None) -> dict:
    """Fit a cross-validated linear probe and score its OUT-OF-FOLD predictions.

    ``task='classification'`` → logistic, reports accuracy / AUC / base_rate / lift (accuracy −
    majority-class). ``task='regression'`` → ridge, reports r2 / rmse. ``groups`` (per-sample
    labels) adds a per-group breakdown — the easy-vs-contested contrast is the real signal.

    ``l2=None`` (default) **auto-tunes** the ridge/logistic penalty over a grid by the OOF
    selection score — essential because the activation probe is high-dim (≈512) and a fixed weak
    penalty overfits when d≈n (negative OOF R²). Both the representation probe and the 1-D provided
    baseline get the SAME grid, so the comparison stays fair. Pure; no torch, no sklearn. ``overall``
    is None when n is too small."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(y)
    grid = (float(l2),) if l2 is not None else _L2_GRID
    chosen_l2, oof = grid[0], np.full(n, np.nan)
    if n >= folds:
        best = -np.inf
        for cand in grid:
            cand_oof = _oof_predict(X, y, task, cand, folds, seed)
            score = _selection_score(y, cand_oof, task)
            if score > best:
                best, chosen_l2, oof = score, cand, cand_oof

    def _metrics(mask) -> "dict | None":
        yy, pp = y[mask], oof[mask]
        ok = ~np.isnan(pp)
        yy, pp = yy[ok], pp[ok]
        if len(yy) < 5:
            return None
        if task == "classification":
            base = float(max(yy.mean(), 1.0 - yy.mean()))
            acc = float(((pp >= 0.5).astype(float) == yy).mean())
            return {"n": int(len(yy)), "accuracy": round(acc, 4), "auc": round(_auc(yy, pp), 4),
                    "base_rate": round(base, 4), "lift": round(acc - base, 4),
                    "pos_rate": round(float(yy.mean()), 4)}
        ss_res = float(((yy - pp) ** 2).sum())
        ss_tot = float(((yy - yy.mean()) ** 2).sum())
        return {"n": int(len(yy)), "r2": round(1.0 - ss_res / ss_tot, 4) if ss_tot > 0 else None,
                "rmse": round(float(np.sqrt(ss_res / len(yy))), 4),
                "target_std": round(float(yy.std()), 4)}

    out = {"task": task, "n": n, "l2": chosen_l2, "overall": _metrics(np.ones(n, dtype=bool))}
    if groups is not None:
        g = np.asarray(groups)
        out["by_group"] = {str(k): _metrics(g == k) for k in sorted({str(v) for v in g.tolist()})}
    return out
