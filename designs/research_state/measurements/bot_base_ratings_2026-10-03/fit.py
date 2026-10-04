"""Rustboro-era BOT BASE RATINGS — the fit half: Bradley-Terry (``elo.fit_pairwise``) over the §0b rows of
``bot_rr.py``, anchored as the ladder's anchor is (``random`` pinned at ``base`` = 1000, every other bot free),
with a PAIR-bootstrap CI and a HodgeRank read.

    export PYTHONPATH=<checkout>/src
    python fit.py <dir>/ledger [--out <dir>/result.json] [--anchor-out <dir>/gen3_bot_elo_anchors.rustboro.json]
                  [--bootstrap 1000] [--size-target 10]

* POINT ESTIMATE — a draw is HALF a win: each edge enters ``fit_pairwise`` as ``(A, B, 2W + D, 2G)`` (half-point
  units; ``_aggregate`` takes integers). The MLE of a half-win BT fit is exactly this fit's.
* CI — the half-point units double the binomial Fisher information, so the Hessian SE is reported x sqrt(2)
  (``se_hessian``: the independent-games SE). The CI quoted is the PAIR BOOTSTRAP: each edge's pentanomial
  pair counts are resampled (multinomial, the edge's own n_pairs) and the whole fit re-run; ``se_boot`` is the
  SD over replicates and ``ci95`` the percentile interval. Pairs are the independent unit of a mirrored design.
* HODGE — ``hodge.hodge_decompose`` over ``(A, B, round(W + D/2), G)`` (whole games; the rounding is <= half a
  game on thousands), triangle scope, parametric-bootstrap null.
* SIZING — ``--size-target T`` prints the pairs/edge at which every free bot's 1.96 x se_boot would reach T Elo,
  scaling the current SE by sqrt(n_now / n).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import random
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone


HERE = os.path.dirname(os.path.abspath(__file__))
N_EDGES = 36            # the nine-bot roster's round robin


def check(ledger_dir: str) -> int:
    """``--check`` (``src/measurements_readout_gate_test.py``): stdlib only — the shards resolve, every row
    parses, all 36 edges are present, and the banked ``result.json`` sits beside the script."""
    import gzip

    shards = sorted(glob.glob(os.path.join(ledger_dir, "ledger.*.jsonl"))
                    + glob.glob(os.path.join(ledger_dir, "ledger.*.jsonl.gz")))
    problems = [] if shards else [f"no ledger shards under {ledger_dir}"]
    edges = set()
    for sh in shards:
        with (gzip.open(sh, "rt") if sh.endswith(".gz") else open(sh)) as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    edges.add((r["player"]["id"], r["opponent"]["id"]))
    if len(edges) != N_EDGES:
        problems.append(f"{len(edges)} edges in the rows, expected {N_EDGES}")
    if not os.path.isfile(os.path.join(HERE, "result.json")):
        problems.append("result.json is missing")
    for p in problems:
        print(f"[check] MISSING: {p}")
    print(f"[check] {len(shards)} shards, {len(edges)} edges" + ("" if problems else " — OK"))
    return 1 if problems else 0


def load_edges(ledger_dir: str):
    from agents.training import eval_ledger as L

    rows = L.read_rows(ledger_dir)
    regimes = {r["regime"]["regime_id"] for r in rows}
    if len(regimes) != 1:
        raise SystemExit(f"refusing to pool {len(regimes)} regimes: {sorted(regimes)}")
    e = defaultdict(lambda: {"w": 0, "l": 0, "d": 0, "pc": [0] * 5, "rows": 0, "timeouts": 0, "a_p1": 0})
    commits = set()
    for r in rows:
        if r["purpose"] != "anchor":
            continue
        a, b = r["player"]["id"].split(":", 1)[1], r["opponent"]["id"].split(":", 1)[1]
        x = e[(a, b)]
        for k in "wld":
            x[k] += r["counts"][k]
        x["pc"] = [p + q for p, q in zip(x["pc"], r["pairs"]["counts"])]
        x["rows"] += 1
        x["timeouts"] += int(r["compute"].get("timeouts", 0))
        x["a_p1"] += int(r["compute"].get("player_p1_games", 0))
        commits.add(r["commit"])
    return dict(e), rows[0]["regime"], sorted(commits), len(rows)


def bt(edges, base, pc_override=None):
    from agents.training import elo as elo_mod

    res = []
    for (a, b), x in edges.items():
        pc = pc_override[(a, b)] if pc_override else x["pc"]
        pts = sum(i * c for i, c in enumerate(pc))          # half-points to A (0..4 per pair)
        res.append((a, b, pts, 4 * sum(pc)))                 # 4 half-points per pair = 2 games
    return elo_mod.fit_pairwise(res, pinned={"random": base}, base=base)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ledger", nargs="?", default=os.path.join(HERE, "ledger"))
    ap.add_argument("--check", action="store_true", help="only resolve the inputs (the readout gate)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--anchor-out", default=None)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--hodge-bootstrap", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--size-target", type=float, default=10.0)
    ap.add_argument("--old-anchor", default="data/gen3_bot_elo_anchors.json")
    args = ap.parse_args()
    if args.check:
        return check(args.ledger)

    from agents.training import elo as elo_mod
    from agents.training import hodge as H
    from agents.training.eval_callback import eval_opponent_names

    names = list(eval_opponent_names())
    edges, regime, commits, n_rows = load_edges(args.ledger)
    missing = [(a, b) for i, a in enumerate(names) for b in names[i + 1:] if (a, b) not in edges]
    if missing:
        raise SystemExit(f"incomplete round robin: missing {missing}")
    n_pairs = {k: sum(x["pc"]) for k, x in edges.items()}
    base = 1000.0
    ratings, se_h, conv = bt(edges, base)
    se_hessian = {n: se_h.get(n, 0.0) * math.sqrt(2) for n in names}

    rng = random.Random(args.seed)
    reps = defaultdict(list)
    for _ in range(args.bootstrap):
        over = {}
        for k, x in edges.items():
            n = n_pairs[k]
            cats = rng.choices(range(5), weights=x["pc"], k=n)
            over[k] = [cats.count(i) for i in range(5)]
        r, _s, _c = bt(edges, base, over)
        for nm in names:
            reps[nm].append(r[nm])
    se_boot, ci = {}, {}
    for nm in names:
        v = sorted(reps[nm])
        se_boot[nm] = statistics.pstdev(v)
        ci[nm] = [v[int(0.025 * len(v))], v[min(len(v) - 1, int(0.975 * len(v)))]]

    # pair-level win matrix and fit quality (predicted vs observed score)
    score, games = {}, {}
    errs = []
    for (a, b), x in edges.items():
        g = x["w"] + x["l"] + x["d"]
        s = (x["w"] + 0.5 * x["d"]) / g
        score[(a, b)], score[(b, a)] = s, 1 - s
        games[f"{a} vs {b}"] = g
        errs.append(abs(elo_mod.win_prob(ratings[a], ratings[b]) - s))
    fit_quality = {"mean_abs_err": round(sum(errs) / len(errs), 4), "max_abs_err": round(max(errs), 4),
                   "n_pairs": len(errs)}

    hres = [(a, b, int(round(x["w"] + 0.5 * x["d"])), x["w"] + x["l"] + x["d"]) for (a, b), x in edges.items()]
    hf = H.hodge_decompose(hres, bootstrap=args.hodge_bootstrap, seed=args.seed)
    hodge = None
    if hf is not None:
        hodge = {"spine_spread_elo": round(hf.spine_spread * H.ELO_PER_LOGIT, 1),
                 "width_rms_raw_elo": round(hf.width_rms_raw * H.ELO_PER_LOGIT, 2),
                 "width_rms_null_elo": round(hf.width_rms_null * H.ELO_PER_LOGIT, 2),
                 "width_rms_excess_elo": round(hf.width_rms_excess * H.ELO_PER_LOGIT, 2),
                 "p_value": hf.p_value, "cyclic_energy_fraction": round(hf.cyclic_energy_fraction, 4),
                 "cyclic_energy_fraction_excess": round(hf.cyclic_energy_fraction_excess, 4),
                 "n_triangles": hf.n_triangles, "n_width_edges": hf.n_width_edges, "n_bootstrap": hf.n_bootstrap,
                 "cycles": [c.__dict__ if hasattr(c, "__dict__") else str(c) for c in hf.cycles],
                 "caveats": hf.caveats}

    old = None
    try:
        with open(args.old_anchor) as f:
            old = json.load(f)
    except OSError:
        pass

    free = [n for n in names if n != "random"]
    n_now = min(n_pairs.values())
    worst = max(1.96 * se_boot[n] for n in free)
    need = math.ceil(n_now * (worst / args.size_target) ** 2)

    table = []
    for nm in sorted(names, key=lambda n: -ratings[n]):
        row = {"bot": nm, "rating": round(ratings[nm], 1), "se_boot": round(se_boot[nm], 1),
               "ci95": [round(ci[nm][0], 1), round(ci[nm][1], 1)], "se_hessian": round(se_hessian[nm], 1)}
        if old:
            row["old_rating"] = old["ratings"].get(nm)
            row["delta"] = round(ratings[nm] - old["ratings"].get(nm, float("nan")), 1)
        table.append(row)

    out = {
        "schema": "gen3_bot_base_ratings_v1", "era": "Rustboro", "computed_at": datetime.now(timezone.utc).isoformat(),
        "commits": commits, "regime": regime, "n_rows": n_rows,
        "anchor": {"pinned": {"random": base}, "rule": "the ladder's anchor convention: random pinned at base 1000 "
                   "(bot_elo_calibration._fit_and_write); the ladder then pins all nine bots to these ratings"},
        "draws": "half a win (fit over half-points)", "bootstrap": args.bootstrap, "seed": args.seed,
        "converged": conv, "pairs_per_edge": n_pairs and {f"{a}:{b}": n for (a, b), n in n_pairs.items()},
        "games_total": sum(x["w"] + x["l"] + x["d"] for x in edges.values()),
        "timeouts_total": sum(x["timeouts"] for x in edges.values()),
        "draws_total": sum(x["d"] for x in edges.values()),
        "table": table, "fit_quality": fit_quality, "hodge": hodge,
        "sizing": {"target_ci95_halfwidth": args.size_target, "worst_free_ci95_halfwidth_now": round(worst, 2),
                   "pairs_per_edge_now": n_now, "pairs_per_edge_needed": need},
        "edges": {f"{a}:{b}": {"w": x["w"], "l": x["l"], "d": x["d"], "pentanomial": x["pc"],
                               "player_p1_games": x["a_p1"], "timeouts": x["timeouts"]} for (a, b), x in edges.items()},
        "old_anchor": ({"git_hash": old.get("git_hash"), "computed_at": old.get("computed_at")} if old else None),
    }
    print(f"[fit] {out['games_total']} games, {len(edges)} edges, min {n_now} pairs/edge; regime {regime['regime_id']}; "
          f"commits {commits}")
    print(f"{'bot':>16} {'rating':>8} {'95% CI':>17} {'se_boot':>7} {'se_hess':>7} {'old':>8} {'delta':>7}")
    for r in table:
        print(f"{r['bot']:>16} {r['rating']:>8.1f} [{r['ci95'][0]:>7.1f},{r['ci95'][1]:>7.1f}] {r['se_boot']:>7.1f} "
              f"{r['se_hessian']:>7.1f} {r.get('old_rating', ''):>8} {r.get('delta', ''):>7}")
    print(f"[fit] fit quality {fit_quality}; hodge {json.dumps({k: v for k, v in (hodge or {}).items() if k != 'cycles'})}")
    print(f"[fit] cycles: {hodge['cycles'] if hodge else None}")
    print(f"[fit] sizing: worst free-bot CI half-width {worst:.2f} at {n_now} pairs/edge -> {need} pairs/edge "
          f"for +-{args.size_target}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=1, default=str)
            f.write("\n")
    if args.anchor_out:
        anchor = {"version": 1, "era": "Rustboro", "target_games": 2 * n_now, "min_pair_games": 2 * n_now,
                  "complete": True, "computed_at": out["computed_at"], "git_hash": commits[0] if len(commits) == 1 else commits,
                  "elo_per_decade": elo_mod.ELO_PER_DECADE, "base": base, "converged": conv,
                  "ratings": {n: round(ratings[n], 1) for n in names},
                  "se": {n: round(se_boot[n], 1) for n in names},
                  "win_matrix": {a: {b: round(score[(a, b)], 4) for b in names if b != a} for a in names},
                  "pair_games": games, "fit_quality": fit_quality,
                  "provenance": "designs/research_state/measurements/bot_base_ratings_2026-10-03/ (bot_rr.py + fit.py); "
                                "draws = half a win; se = pair-bootstrap SD"}
        with open(args.anchor_out, "w") as f:
            json.dump(anchor, f, indent=1)
            f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
