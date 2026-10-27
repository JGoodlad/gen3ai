"""S-U adversarial review (REVIEW.md): two zero-playout checks over banked rows, READ-ONLY on every input.

1. `noise`: is S's deviation from pi signal or noise? On the maturity battery's PLATEAU bank (L1's registered loader,
   `../../search_l1_plateau_2026-10-25/l1_plateau.py`; 719 common states per anchor, pool teams, self-play), the
   registered acting rule (piKL lambda 0.015 at TRUNC-4) is applied to the 8-world mean and to each 4-world HALF
   (worlds 0-3 vs 4-7). Reported: the deviation rate, the halves' pick agreement, the within-state Spearman of the
   halves' Q (reliability) and of the 8-world Q against the truth referee T (validity, SELF-REFEREE), and the sign of
   T(pick) - T(pi top-1) given a deviation. Descriptive; battle-clustered bootstrap (B 1,000, seed 20261027).
2. `cost`: the per-searched-game CPU of S-U's preset over X51's P*-vs-Kakuna game lengths (the forensics' rows,
   `~/gen3ai_archive/piloting_gap_forensics_2026-10-26/games.jsonl.gz`), from stage 0's ONE searched game's cost curve
   (AMENDMENT_1_DRAFT.md section 1) on the REFERENCE builder (the pin's) and on the LEAN screen (2eb65b3d9: 18.0
   search CPU-min for the same game, a decision's median ~5x from turns 10-19 to 50-59), two extrapolations past its
   66 turns, and the share of games whose searched twin would pass the 3-h unit bound (`main.x35.kakuna.UNIT_SEARCH_S`).
   An ESTIMATE from one game.

    python -I review_checks.py   (CPU, < 1 min, < 1 GB; writes review_results.json beside this folder)
"""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MEAS = HERE.parent.parent
sys.path.insert(0, str(MEAS / "search_l1_plateau_2026-10-25"))
import l1_plateau as P  # L1's registered plateau loader: sha-checked table, common ids

LAM = 0.015                     # r0s4's piKL lambda (= L1's fitted value)
TIE = 1e-9
FORENSICS = Path.home() / "gen3ai_archive" / "piloting_gap_forensics_2026-10-26" / "games.jsonl.gz"


def log_softmax(x: np.ndarray) -> np.ndarray:
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())


def pick(q: np.ndarray, logit: np.ndarray) -> int:
    s = q + LAM * log_softmax(logit)
    return int(np.argmax(s >= s.max() - TIE))     # policy order: ties to the earlier action, L1's rule


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def boot(vals: dict, B: int = 1000, seed: int = 20261027):
    """Battle-clustered percentile bootstrap of the pooled mean of per-state values (NaN dropped)."""
    keys = sorted(vals)
    arrs = [np.asarray([v for v in vals[k] if not np.isnan(v)], dtype=float) for k in keys]
    s = np.array([a.sum() for a in arrs])
    n = np.array([len(a) for a in arrs])
    rng = np.random.default_rng(seed)
    est = s.sum() / max(n.sum(), 1)
    bs = []
    for _ in range(B):
        i = rng.integers(0, len(keys), len(keys))
        bs.append(s[i].sum() / max(n[i].sum(), 1))
    return [round(float(est), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4),
            int(n.sum())]


def noise() -> dict:
    log: dict = {}
    recs, _ = P.load(log)
    out = {"loader": {k: log[k] for k in ("table_sha256", "n_common", "common_ids_sha256")}}
    for lab in P.LABELS:
        rs = [r for r in recs if r["cp"] == lab]
        acc = {k: {} for k in ("dev8", "devA", "agree_given_both_dev", "same_dev", "rel", "val", "sign", "imp")}
        n_legal = []
        for r in rs:
            X = np.asarray(r["F"]["X4"], dtype=float)             # [A, 8] in policy order
            lg = np.asarray(r["logit"], dtype=float)
            T = np.asarray(r["Tp"], dtype=float)
            b = r["battle"]
            n_legal.append(len(lg))
            p8, pa, pb = pick(X.mean(1), lg), pick(X[:, :4].mean(1), lg), pick(X[:, 4:].mean(1), lg)
            for k, v in (("dev8", float(p8 != 0)), ("devA", float(pa != 0)),
                         ("same_dev", float(pa != 0 and pa == pb)),
                         ("agree_given_both_dev", float(pa == pb) if (pa != 0 and pb != 0) else float("nan")),
                         ("rel", spearman(X[:, :4].mean(1), X[:, 4:].mean(1)) if len(lg) > 2 else float("nan")),
                         ("val", spearman(X.mean(1), T) if len(lg) > 2 else float("nan")),
                         ("sign", float(T[p8] > T[0]) if p8 != 0 and T[p8] != T[0] else float("nan")),
                         ("imp", float(T[p8] - T[0]))):
                acc[k].setdefault(b, []).append(v)
        res = {k: boot(v) for k, v in acc.items()}
        # chance agreement of two halves that both deviate, if each picked uniformly among the non-top actions
        res["chance_agree_given_both_dev"] = round(float(np.mean([1.0 / (k - 1) for k in n_legal if k > 1])), 4)
        res["n_states"] = len(rs)
        out[lab] = res
    return out


def builder(a: float, x: float, dpt: float = 1.12):
    """Search CPU-min of one searched game of T turns: c(t) = a * t**x CPU-s a decision, `dpt` decisions a turn from
    turn 10; `growth` keeps growing past 66, `flat` freezes c at its turn-66 value (optimistic), `window` stops at 30."""
    e = x + 1.0

    def cum(t):
        return a * dpt * (t ** e - 10 ** e) / e if t > 10 else 0.0

    return {"growth": lambda t: cum(t) / 60,
            "flat": lambda t: (cum(min(t, 66)) + max(t - 66, 0) * dpt * a * 66 ** x) / 60,
            "window": lambda t: cum(min(t, 30)) / 60}


#: fitted to stage 0's game: the REFERENCE builder 77.8 search CPU-min at 66 turns, a decision ~6 s at turns 10-19 and
#: ~130 at 50-59 (exponent 2.4); the LEAN screen 18.0 at 66 turns, ~5.8 s and ~27 s (exponent 1.24)
BUILDERS = {"reference": (0.00922, 2.4), "lean": (0.184, 1.24)}


def cost() -> dict:

    with gzip.open(FORENSICS, "rt") as fh:
        rows = [json.loads(x) for x in fh]
    rows = [r for r in rows if r["opp"] == "Kakuna34" and r["model"] == "P*"]
    bound = 180.0                                  # ~3 h of wall at ~1x CPU (stage 0: 77.8 CPU-min, 82.7 min wall)
    out = {"model": "c(t)=a*t^x CPU-s/decision, 1.12 decisions/turn from turn 10; fit: stage 0 game 1 (ONE game)",
           "builders": {k: {"a": a, "x": x} for k, (a, x) in BUILDERS.items()}, "n_games": len(rows)}
    for name, (a, x) in BUILDERS.items():
        f = builder(a, x)
        res: dict = {}
        for p in ("home72", "H", "F", "P72", "ALL"):
            T = [r["turns"] for r in rows if p == "ALL" or r["our_panel"] == p]
            res[p] = {"n": len(T), "median_turns": float(np.median(T)),
                      "mean_cpu_min_growth": round(float(np.mean([f["growth"](t) for t in T])), 1),
                      "mean_cpu_min_flat": round(float(np.mean([f["flat"](t) for t in T])), 1),
                      "median_cpu_min": round(float(np.median([f["flat"](t) for t in T])), 1),
                      "mean_cpu_min_window_10_30": round(float(np.mean([f["window"](t) for t in T])), 2),
                      "share_over_3h_flat": round(float(np.mean([f["flat"](t) > bound for t in T])), 3),
                      "share_over_3h_growth": round(float(np.mean([f["growth"](t) > bound for t in T])), 3),
                      "share_turns_gt_100": round(float(np.mean([t > 100 for t in T])), 3)}
        m = res["ALL"]
        res["full_read_cpu_h"] = {"flat": round(m["mean_cpu_min_flat"] * 2400 / 60),
                                  "growth": round(m["mean_cpu_min_growth"] * 2400 / 60),
                                  "window_10_30": round(m["mean_cpu_min_window_10_30"] * 2400 / 60)}
        out[name] = res
    return out


if __name__ == "__main__":
    res = {"noise": noise(), "cost": cost()}
    (HERE.parent / "review_results.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
