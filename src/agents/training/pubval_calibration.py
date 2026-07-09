"""One-time V_pub calibration: fit the public-value logistic on the human replay corpus.

    python -m agents.training.pubval_calibration [--games N] [--min-rating R] \\
        [--corpus replays/showdown/gen3ou] [--out data/gen3_pubval.json]

Parses the rated gen3ou ladder replay logs into per-turn PUBLIC snapshots (both perspectives,
labeled by the terminal outcome), fits the 17-feature logistic (`agents.training.pubval` is the
single feature definition — the SAME code the live env evaluates), reports the held-out-by-GAME
AUC + the turn-bucket leakage guard (turn-1 must be ~0.5) + calibration, and writes the frozen
artifact `data/gen3_pubval.json` with provenance. CPU-only, minutes on the full corpus; mirrors
the `bot_elo_calibration` artifact pattern (a derived, committed calibration input in `data/`).

The POC (designs/ai_v8/design_public_info_value.md) validated this exact recipe at 0.734 test AUC,
calibrated, leakage-clean; richer identity features overfit — do not add them here.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import random
from datetime import datetime, timezone

import numpy as np

from agents.training.pubval import (
    DEFAULT_PUBVAL_PATH, PUBVAL_FEATURE_NAMES, PubValModel, features, parse_replay_log,
)

DEFAULT_CORPUS = os.path.join("replays", "showdown", "gen3ou")


def build_dataset(files, min_rating: int = 0, progress_every: int = 10000):
    """Parse logs → (X [n,17], y [n], game_id [n], turn [n], games_kept). Both perspectives per
    position (mirrored); rated games with a resolvable winner only; split-by-game downstream."""
    X, y, gid, turn = [], [], [], []
    kept = 0
    for i, path in enumerate(files):
        if progress_every and i and i % progress_every == 0:
            print(f"  ... {i}/{len(files)} files, {kept} games kept, {len(y)} positions", flush=True)
        try:
            with open(path, encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except OSError:
            continue
        positions, winner, (r1, r2), is_rated = parse_replay_log(text)
        if winner is None or not is_rated or not positions:
            continue
        if min_rating and (r1 < min_rating or r2 < min_rating):
            continue
        for (t, s1, s2, weather) in positions:
            X.append(features(s1, s2, t, weather)); y.append(1 if winner == 0 else 0)
            X.append(features(s2, s1, t, weather)); y.append(1 if winner == 1 else 0)
            gid.extend((kept, kept)); turn.extend((t, t))
        kept += 1
    return (np.asarray(X, np.float32), np.asarray(y, np.int8),
            np.asarray(gid, np.int64), np.asarray(turn, np.int64), kept)


def auc(scores, labels) -> float:
    """Rank AUC (Mann-Whitney)."""
    s = np.asarray(scores, np.float64)
    yy = np.asarray(labels).astype(bool)
    pos, neg = int(yy.sum()), int((~yy).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[yy].sum() - pos * (pos + 1) / 2) / (pos * neg))


def train_logistic(X, yv, iters: int = 500, lr: float = 0.5, l2: float = 1e-4):
    """Full-batch gradient-descent logistic on standardized features → (mu, sd, w, b)."""
    mu, sd = X.mean(0).astype(np.float64), (X.std(0) + 1e-6).astype(np.float64)
    Xn = ((X - mu) / sd).astype(np.float64)
    w = np.zeros(Xn.shape[1], np.float64)
    b = 0.0
    yv = yv.astype(np.float64)
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-(Xn @ w + b)))
        grad = p - yv
        w -= lr * (Xn.T @ grad / len(yv) + l2 * w)
        b -= lr * float(grad.mean())
    return mu, sd, w, b


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=0, help="max games to parse (0 = the full corpus)")
    ap.add_argument("--min-rating", type=int, default=0, help="drop games where either player is below this")
    ap.add_argument("--corpus", default=DEFAULT_CORPUS, help=f"replay corpus root (default {DEFAULT_CORPUS})")
    ap.add_argument("--out", default=DEFAULT_PUBVAL_PATH, help=f"artifact path (default {DEFAULT_PUBVAL_PATH})")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.corpus, "*", "*.log")))
    if not files:
        print(f"no replay logs under {args.corpus!r} (expected <corpus>/<date>/*.log)")
        return 2
    rng = random.Random(args.seed)
    rng.shuffle(files)
    if args.games:
        files = files[: args.games]
    print(f"parsing {len(files)} replay files (min_rating={args.min_rating}) ...", flush=True)
    X, y, gid, turn, kept = build_dataset(files, args.min_rating)
    print(f"kept {kept} rated games -> {len(y)} positions; base win-rate {y.mean():.3f}")

    # Held-out split BY GAME (positions within a game are correlated — the POC's key control).
    ug = np.unique(gid)
    rng.shuffle(ug)
    cut = int(0.8 * len(ug))
    tr = np.isin(gid, ug[:cut])
    te = ~tr
    mu, sd, w, b = train_logistic(X[tr], y[tr])
    model = PubValModel(mu=mu, sd=sd, w=w, b=b, feature_names=PUBVAL_FEATURE_NAMES)
    z = ((X[te].astype(np.float64) - mu) / sd) @ w + b
    p_te = 1.0 / (1.0 + np.exp(-z))
    auc_te = auc(p_te, y[te])
    brier = float(np.mean((p_te - y[te]) ** 2))
    print(f"\nheld-out (by game): AUC={auc_te:.4f}  Brier={brier:.4f}   (POC reference: 0.734)")

    print("AUC by turn bucket (leakage guard — turn 1 must be ~0.5):")
    tt = turn[te]
    bucket_aucs = {}
    for lo, hi in [(1, 1), (2, 5), (6, 10), (11, 20), (21, 50), (51, 999)]:
        m = (tt >= lo) & (tt <= hi)
        if m.sum() > 50:
            a = auc(p_te[m], y[te][m])
            bucket_aucs[f"{lo}-{hi}"] = round(a, 3)
            print(f"  turn {lo:>2}-{hi:<3} n={int(m.sum()):7}  AUC={a:.3f}")
    print("calibration (predicted bucket -> realized win-rate):")
    for lo in (0.0, 0.2, 0.4, 0.6, 0.8):
        m = (p_te >= lo) & (p_te < lo + 0.2)
        if m.sum() > 50:
            print(f"  pred [{lo:.1f},{lo + 0.2:.1f})  n={int(m.sum()):7}  realized={y[te][m].mean():.3f}")

    t1 = bucket_aucs.get("1-1")
    if t1 is not None and abs(t1 - 0.5) > 0.05:
        print(f"\nWARNING: turn-1 AUC {t1} is far from 0.5 — possible outcome leakage; NOT writing.")
        return 1

    try:
        from utils.git import get_git_hash
        git_hash = get_git_hash()
    except Exception:
        git_hash = "unknown"
    model.meta = {
        "n_files": len(files), "n_games": int(kept), "n_positions": int(len(y)),
        "min_rating": args.min_rating, "auc_test": round(auc_te, 4), "brier_test": round(brier, 4),
        "auc_by_turn": bucket_aucs, "corpus": args.corpus, "seed": args.seed,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "git_hash": git_hash,
        "parser_version": "gen3_pubval_aux_v1",
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(model.to_json(), f, indent=1)
    print(f"\nwrote {args.out}  (games={kept}, AUC={auc_te:.4f})")
    # Round-trip sanity: the artifact must load through the exact runtime path.
    PubValModel.load(args.out)
    print("artifact reload OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
