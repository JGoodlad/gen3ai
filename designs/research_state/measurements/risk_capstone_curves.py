#!/usr/bin/env python3
"""Risk-capstone trace-only instruments — accuracy-tradeoff + explosion-timing curves.

The two trace-only instruments of designs/ai_v12/probe_risk_modulation_capstone.md §1,
runnable on ANY set of run dirs carrying eval_traces (gen-15 baseline now, the ai_v12
arms later — pass the run dirs, nothing else changes):

  1. ACCURACY-TRADEOFF (registered primary): decisions where a same-type power/accuracy
     pair is simultaneously legal (pairs resolved from data/pokemon/gen3_moves.json by
     criteria, never hardcoded). Endpoint = dP(chose inaccurate)/d(win-prob).
     Because the team pool almost never carries both members of a pair on one moveset
     (measured: 9 of 4,566 pool mons), the script ALSO reports a defined companion:
  1b. ACCURACY-CLASS companion (explicitly confounded — drops the same-type control):
     decisions where the active mon has BOTH a legal 100-acc damaging move and a legal
     sub-100-acc damaging move (confound-filtered), conditioned on choosing a damaging
     move. Moveset fixed-effects slope reported beside pooled (the roster confound).
  2. EXPLOSION TIMING: P(boom chosen | Explosion/Selfdestruct legal) vs recorded
     win-prob, pooled + moveset-FE, indicator (greedy argmax) and probability-mass forms.

All slopes are OLS in probability-per-unit-win-prob, cluster-bootstrapped over BATTLES
(one trace file = one battle = one cluster; the Simpson-trap rule).

Run (CPU, trace-only, no model loads, no battles):
  python designs/research_state/measurements/risk_capstone_curves.py \
      --runs 'models/ai_v9_7*' --out designs/research_state/measurements/out.json
(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

REPO_ROOT = Path(__file__).resolve().parents[3]
MOVES_JSON = REPO_ROOT / "data" / "pokemon" / "gen3_moves.json"
BOOM_MOVES = ("explosion", "selfdestruct")  # named by the capstone registration itself
WP_BINS = np.linspace(0.0, 1.0, 11)


# ---------------------------------------------------------------- pair resolution
def load_moves() -> dict:
    with open(MOVES_JSON) as f:
        return json.load(f)


def eligible_damaging(d: dict) -> bool:
    """A move usable as a member of an accuracy-tradeoff comparison.

    Filters remove drawbacks OTHER than accuracy that would confound the tradeoff:
    charge turns (Solar Beam), recoil (Double-Edge), self stat drops (Overheat),
    never-miss mechanics, sub-50 BP filler.
    """
    return (
        d.get("basePower", 0) >= 50
        and isinstance(d.get("accuracy"), (int, float))
        and not d.get("never_miss")
        and not d.get("isCharge")
        and not d.get("hasRecoil")
        and float(d.get("recoilFraction", 0)) == 0.0
        and not d.get("selfDrops")
    )


def resolve_pairs(moves: dict) -> list[tuple[str, str]]:
    """All (accurate, inaccurate) same-type pairs: strictly higher power AND strictly
    lower accuracy on the inaccurate member, both confound-eligible."""
    el = {m: d for m, d in moves.items() if eligible_damaging(d)}
    pairs = []
    for a, da in el.items():
        for i, di in el.items():
            if (
                a != i
                and da["type"] == di["type"]
                and di["basePower"] > da["basePower"]
                and di["accuracy"] < da["accuracy"]
            ):
                pairs.append((a, i))
    return sorted(pairs)


# ---------------------------------------------------------------- trace scanning
def iter_decisions(summary_path: str):
    """Yield (k, inv, names, wp, probs, chosen_idx) per move_selection decision."""
    npz_path = summary_path.replace("_summary.json", "_states.npz")
    if not os.path.exists(npz_path):
        return
    try:
        with open(summary_path) as f:
            d = json.load(f)
        z = np.load(npz_path)
    except Exception as e:  # noqa: BLE001 - a corrupt trace is skipped, counted by caller
        print(f"  [skip] {summary_path}: {e}", file=sys.stderr)
        return
    inv = d.get("invocations", [])
    if "win_probs" not in z or len(inv) != z["win_probs"].shape[0]:
        return
    wps = z["win_probs"]
    logits = z["logits"]
    masks = z["action_mask"]
    acts = z["actions"]
    has = z["has_state"] if "has_state" in z else np.ones(len(inv), dtype=np.int8)
    for k, rec in enumerate(inv):
        if rec.get("phase") != "move_selection" or not has[k]:
            continue
        wp = float(wps[k])
        if not np.isfinite(wp):
            continue
        names = list(rec["actions"].keys())
        if len(names) != masks.shape[1]:
            continue
        lg = np.where(masks[k], logits[k], -np.inf)
        p = np.exp(lg - lg.max())
        p /= p.sum()
        yield k, rec, names, wp, p, int(acts[k])


class Rows:
    """Per-instrument event rows: (battle, group, wp, indicator, mass)."""

    def __init__(self):
        self.battle: list[int] = []
        self.group: list = []
        self.wp: list[float] = []
        self.ind: list[float] = []
        self.mass: list[float] = []

    def add(self, battle, group, wp, ind, mass):
        self.battle.append(battle)
        self.group.append(group)
        self.wp.append(wp)
        self.ind.append(ind)
        self.mass.append(mass)

    def arrays(self):
        return (
            np.asarray(self.battle),
            np.asarray(self.wp),
            np.asarray(self.ind, dtype=float),
            np.asarray(self.mass, dtype=float),
        )


# ---------------------------------------------------------------- statistics
def _cluster_slope_ci(bat, x, y, n_boot, rng):
    """OLS slope of y on x + percentile 95% CI, cluster bootstrap over battles."""
    if len(x) < 3 or np.ptp(x) == 0:
        return None
    ub, inv = np.unique(bat, return_inverse=True)
    nb = len(ub)
    # per-cluster sufficient statistics
    S = np.zeros((nb, 5))
    np.add.at(S, inv, np.column_stack([np.ones_like(x), x, y, x * y, x * x]))

    def slope_from(sums):
        n, sx, sy, sxy, sxx = sums
        den = n * sxx - sx * sx
        if den == 0:
            return np.nan
        return (n * sxy - sx * sy) / den

    point = slope_from(S.sum(axis=0))
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, nb, nb)
        draws[b] = slope_from(S[idx].sum(axis=0))
    draws = draws[np.isfinite(draws)]
    lo, hi = (np.percentile(draws, [2.5, 97.5]) if len(draws) else (np.nan, np.nan))
    return {
        "slope": float(point),
        "ci95": [float(lo), float(hi)],
        "n_rows": int(len(x)),
        "n_battles": int(nb),
        "n_boot_finite": int(len(draws)),
    }


def _fe_demean(groups, x, y):
    """Two-step fixed effects: demean x and y within GROUP (moveset) using the full
    sample, then the through-origin cluster-bootstrapped slope on the residuals."""
    idx_of = {}
    gid = np.empty(len(x), dtype=int)
    for i, g in enumerate(groups):
        gid[i] = idx_of.setdefault(g, len(idx_of))
    ng = len(idx_of)
    sums_x = np.zeros(ng)
    sums_y = np.zeros(ng)
    cnt = np.zeros(ng)
    np.add.at(sums_x, gid, x)
    np.add.at(sums_y, gid, y)
    np.add.at(cnt, gid, 1)
    return x - sums_x[gid] / cnt[gid], y - sums_y[gid] / cnt[gid]


def _cluster_slope_fe(bat, groups, x, y, n_boot, rng):
    xd, yd = _fe_demean(groups, x, y)
    if len(xd) < 3 or np.ptp(xd) == 0:
        return None
    ub, inv = np.unique(bat, return_inverse=True)
    nb = len(ub)
    S = np.zeros((nb, 2))
    np.add.at(S, inv, np.column_stack([xd * yd, xd * xd]))
    tot = S.sum(axis=0)
    point = tot[0] / tot[1] if tot[1] else np.nan
    draws = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, nb, nb)
        sxy, sxx = S[idx].sum(axis=0)
        draws[b] = sxy / sxx if sxx else np.nan
    draws = draws[np.isfinite(draws)]
    lo, hi = (np.percentile(draws, [2.5, 97.5]) if len(draws) else (np.nan, np.nan))
    return {
        "slope": float(point),
        "ci95": [float(lo), float(hi)],
        "n_rows": int(len(x)),
        "n_battles": int(nb),
        "n_groups": int(len(set(groups))),
    }


def binned_curve(wp, y):
    out = []
    which = np.digitize(wp, WP_BINS[1:-1])
    for b in range(len(WP_BINS) - 1):
        sel = which == b
        n = int(sel.sum())
        out.append(
            {
                "bin": [round(float(WP_BINS[b]), 2), round(float(WP_BINS[b + 1]), 2)],
                "n": n,
                "mean": (float(y[sel].mean()) if n else None),
            }
        )
    return out


def analyze(rows: Rows, n_boot, rng, fe=True):
    bat, wp, ind, mass = rows.arrays()
    if len(wp) == 0:
        return {"n_rows": 0}
    out = {
        "n_rows": int(len(wp)),
        "n_battles": int(len(set(rows.battle))),
        "rate_overall": float(ind.mean()),
        "mass_overall": float(mass.mean()),
        "slope_indicator": _cluster_slope_ci(bat, wp, ind, n_boot, rng),
        "slope_mass": _cluster_slope_ci(bat, wp, mass, n_boot, rng),
        "curve_indicator": binned_curve(wp, ind),
        "curve_mass": binned_curve(wp, mass),
    }
    if fe:
        out["slope_indicator_fe"] = _cluster_slope_fe(bat, rows.group, wp, ind, n_boot, rng)
        out["slope_mass_fe"] = _cluster_slope_fe(bat, rows.group, wp, mass, n_boot, rng)
    mid = (wp >= 0.45) & (wp <= 0.55)
    out["rate_at_equality_0.45_0.55"] = (
        {"n": int(mid.sum()), "rate": float(ind[mid].mean()), "mass": float(mass[mid].mean())}
        if mid.any()
        else {"n": 0}
    )
    return out


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="run-dir globs, each containing eval_traces/ (e.g. 'models/ai_v9_7*')",
    )
    ap.add_argument("--out", required=True, help="output JSON path")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plots", action="store_true", help="write PNG curves beside --out")
    args = ap.parse_args()

    run_dirs = sorted({d for pat in args.runs for d in glob.glob(pat) if os.path.isdir(d)})
    files = sorted(
        f
        for rd in run_dirs
        for f in glob.glob(os.path.join(rd, "eval_traces", "*", "*", "*_summary.json"))
    )
    print(f"runs: {len(run_dirs)}  trace files: {len(files)}")
    if not files:
        sys.exit("no trace files found under --runs")

    moves = load_moves()
    pairs = resolve_pairs(moves)
    pair_set = {p: i for i, p in enumerate(pairs)}
    print(f"resolved {len(pairs)} same-type power/accuracy pairs from {MOVES_JSON.name}")

    rng = np.random.default_rng(args.seed)
    pair_rows: dict[tuple[str, str], Rows] = defaultdict(Rows)
    pair_pooled = Rows()
    acc_rows = Rows()
    boom_rows = Rows()
    per_move_boom: dict[str, Rows] = defaultdict(Rows)
    n_dec = 0
    moveset_census = Counter()

    for bi, f in enumerate(files):
        for _k, rec, names, wp, p, a_idx in iter_decisions(f):
            n_dec += 1
            legal = {
                n
                for j, n in enumerate(names)
                if rec["actions"][n]["valid"] and not n.startswith("switch:") and n != "struggle"
            }
            moveset = tuple(sorted(n for n in names if not n.startswith("switch:") and n != "struggle"))
            moveset_census[moveset] += 1
            chosen = names[a_idx]
            pidx = {n: j for j, n in enumerate(names)}

            # instrument 1 — registered same-type pairs
            for A, I in pair_set:
                if A in legal and I in legal:
                    pA, pI = p[pidx[A]], p[pidx[I]]
                    share = pI / (pA + pI) if (pA + pI) > 0 else np.nan
                    if chosen in (A, I):
                        ind = 1.0 if chosen == I else 0.0
                        pair_rows[(A, I)].add(bi, moveset, wp, ind, share)
                        pair_pooled.add(bi, moveset, wp, ind, share)

            # instrument 1b — accuracy-class companion (same-type control dropped)
            dmg = [
                n
                for n in legal
                if n in moves and eligible_damaging(moves[n])
            ]
            acc100 = [n for n in dmg if moves[n]["accuracy"] >= 100]
            sub100 = [n for n in dmg if moves[n]["accuracy"] < 100]
            if acc100 and sub100 and chosen in dmg:
                ind = 1.0 if chosen in sub100 else 0.0
                m_sub = sum(p[pidx[n]] for n in sub100)
                m_acc = sum(p[pidx[n]] for n in acc100)
                share = m_sub / (m_sub + m_acc) if (m_sub + m_acc) > 0 else np.nan
                acc_rows.add(bi, moveset, wp, ind, share)

            # instrument 2 — explosion timing
            booms = [n for n in BOOM_MOVES if n in legal]
            if booms:
                ind = 1.0 if chosen in booms else 0.0
                m_boom = sum(p[pidx[n]] for n in booms)
                boom_rows.add(bi, moveset, wp, ind, m_boom)
                for n in booms:
                    per_move_boom[n].add(bi, moveset, wp, 1.0 if chosen == n else 0.0, p[pidx[n]])

    print(f"move-selection decisions scanned: {n_dec}")
    nb = args.bootstrap

    result = {
        "meta": {
            "script": "risk_capstone_curves.py",
            "runs": run_dirs,
            "n_trace_files": len(files),
            "n_move_decisions": n_dec,
            "n_distinct_movesets": len(moveset_census),
            "bootstrap": nb,
            "seed": args.seed,
            "pair_criteria": "same type; inaccurate member strictly higher basePower AND strictly "
            "lower accuracy; both >=50 BP, numeric accuracy, no never_miss/isCharge/recoil/selfDrops",
            "n_pairs_resolved": len(pairs),
        },
        "accuracy_tradeoff_pairs": {
            "support": {
                f"{A}|{I}": analyze(pair_rows[(A, I)], nb, rng, fe=False)
                for (A, I) in pairs
                if (A, I) in pair_rows
            },
            "pairs_with_zero_support": len(pairs) - len(pair_rows),
            "pooled": analyze(pair_pooled, nb, rng, fe=False),
        },
        "accuracy_class_companion": analyze(acc_rows, nb, rng),
        "explosion_timing": {
            "pooled": analyze(boom_rows, nb, rng),
            "per_move": {n: analyze(r, nb, rng) for n, r in per_move_boom.items()},
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(result, f, indent=1)
    print(f"wrote {out}")

    if args.plots:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        for key, rows_obj, title in [
            ("explosion", boom_rows, "P(boom | boom legal) vs recorded win-prob"),
            ("accclass", acc_rows, "P(chose sub-100-acc damaging | both classes legal) vs win-prob"),
        ]:
            bat, wp, ind, mass = rows_obj.arrays()
            if len(wp) == 0:
                continue
            fig, ax = plt.subplots(figsize=(7, 4.5))
            centers, means, ns = [], [], []
            for b in binned_curve(wp, ind):
                if b["n"]:
                    centers.append(sum(b["bin"]) / 2)
                    means.append(b["mean"])
                    ns.append(b["n"])
            ax.plot(centers, means, "o-", label="indicator (greedy choice)")
            centers2, means2 = [], []
            for b in binned_curve(wp, mass):
                if b["n"]:
                    centers2.append(sum(b["bin"]) / 2)
                    means2.append(b["mean"])
            ax.plot(centers2, means2, "s--", label="policy probability mass")
            for cx, m, n in zip(centers, means, ns):
                ax.annotate(str(n), (cx, m), textcoords="offset points", xytext=(0, 6), fontsize=7)
            ax.set_xlabel("recorded win-prob at decision")
            ax.set_ylabel("P(risky option)")
            ax.set_title(title, fontsize=10)
            ax.legend()
            ax.grid(alpha=0.3)
            png = out.with_name(out.stem + f"_{key}.png")
            fig.tight_layout()
            fig.savefig(png, dpi=120)
            print(f"wrote {png}")

    # console digest
    def fmt(s):
        if not s:
            return "n/a"
        return f"{s['slope']:+.4f} CI[{s['ci95'][0]:+.4f},{s['ci95'][1]:+.4f}] (n={s['n_rows']})"

    print("\n=== DIGEST ===")
    ap_ = result["accuracy_tradeoff_pairs"]
    print(f"pairs resolved {len(pairs)}, with any support: {len(ap_['support'])}")
    for k, v in ap_["support"].items():
        print(f"  {k}: n={v['n_rows']}")
    print("pooled pair slope:", fmt(ap_["pooled"].get("slope_indicator")))
    ac = result["accuracy_class_companion"]
    print("acc-class pooled:", fmt(ac.get("slope_indicator")), " FE:", fmt(ac.get("slope_indicator_fe")))
    ex = result["explosion_timing"]["pooled"]
    print("explosion pooled:", fmt(ex.get("slope_indicator")), " FE:", fmt(ex.get("slope_indicator_fe")))
    print("explosion mass  :", fmt(ex.get("slope_mass")), " FE:", fmt(ex.get("slope_mass_fe")))


if __name__ == "__main__":
    main()
