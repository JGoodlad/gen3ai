"""Score DEFENSIVE PAIRED SEARCH iteration 2 — "spend the bank" — against its registered bars.

    python designs/research_state/measurements/defensive_search_iter2_report.py \
        "tmp/sd2/shardA.jsonl,tmp/sd2/shardB.jsonl,tmp/sd2/shardC.jsonl" \
        designs/research_state/measurements/defensive_search_iter2_2026-08-29.json

(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)

The ONE change under test is ``--defensive-contested-deadline-s 3.0``: the gate already forces
~74% of decisions and spends nothing on them, so a contested decision may now draw on that bank
instead of being held to the uniform ``--budget``. Everything else is iteration 1's cell verbatim
— gate threshold 0.15, ``seq`` rule, floor 5, win-prob leaf, depth 1, confirm off, checkpoint,
``--games-seed 7``, side-swap on.

**Four comparisons, in increasing strength**, because these arms did not merely play a similar
experiment — on the overlapping indices they played the SAME battles:

* **vs the null.** A mirror's no-effect point is 0.50 by CONSTRUCTION (the same network with
  search structurally off on one side), so "does the interval reach it / clear it" is the whole
  primary and stretch test.
* **vs ``honest_1s``, unpaired** (Newcombe) and **paired on the game index** — the historical bar.
* **vs ITERATION 1, paired on the game index.** Both cells ran at ``--games-seed 7`` off the same
  checkpoint, so game *g* is the same pinned dice and the same team draw in both, and the pair
  difference is a statement about the DEADLINE rather than about which side drew the better six.
  This is the row the registered "no regression" bar is read off.

⚠️ The iteration-2 arm is played across three SHARDS over disjoint game-index windows. That is a
scheduling split and not three experiments: :func:`~main.search_dividend.battery.game_seed` and
``team_pair`` are functions of the index alone, so the rows concatenate into exactly the file one
process would have written. The script asserts the windows really were disjoint rather than
trusting the launch script.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

REPO = "/home/goodlad/dev/gen3ai"

#: ITERATION 1's rows, read from the COMMITTED artifact beside this script rather than from a
#: scratch directory — the record is the thing a later reader has, and a comparison that depends
#: on someone's `tmp/` is a comparison that stops reproducing.
ITER1 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "defensive_search_first_cell_2026-08-29_rows.jsonl.gz")

#: The battery's own prior mirror arms — QUOTED from the files they were played into, never re-run.
HISTORICAL = [
    ("mirror_base", "tmp/search_dividend/mirror_base.jsonl",
     "the policy alone — the control; 0.500 by construction"),
    ("mirror_honest_1s", "tmp/search_dividend/mirror_honest_1s.jsonl",
     "THE HISTORICAL BAR: plain depth-1 grid search, belief-determinized, 1 s/decision"),
    ("mirror_honest_3s", "tmp/search_dividend/mirror_honest_3s.jsonl",
     "the same at a UNIFORM 3 s — the arm that says the deadline alone is not the mechanism"),
    ("mirror_oracle_1s", "tmp/search_dividend/mirror_oracle_1s.jsonl",
     "grid search on the TRUE hidden team, 1 s"),
    ("mirror_oracle_3s", "tmp/search_dividend/mirror_oracle_3s.jsonl", "...at 3 s"),
    ("playoff_10s", "tmp/search_dividend/playoff_10s.jsonl",
     "the screen + paired terminal rollouts at 20 s — the best prior arm"),
]


def rows(path: str) -> List[dict]:
    out: List[dict] = []
    if not os.path.exists(path):
        return out
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as fh:                                # type: ignore[operator]
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    break                                     # a torn final line stops the read
    return out


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n <= 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def newcombe(k1: int, n1: int, k2: int, n2: int, z: float = 1.96) -> tuple:
    """The 95% interval on ``p1 - p2`` for two INDEPENDENT proportions (Newcombe's method 10)."""
    p1, l1, u1 = wilson(k1, n1, z)
    p2, l2, u2 = wilson(k2, n2, z)
    lo = (p1 - p2) - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = (p1 - p2) + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (p1 - p2, max(-1.0, lo), min(1.0, hi))


def pair_scores(rs: Sequence[dict]) -> Dict[int, float]:
    """``{game_index: paired score}`` over games with BOTH orientations finished.

    A pair scores 1 / 0.5 / 0 for won-both / split / lost-both, and a TIE counts 0.5 — which is
    exactly the null's own prediction, so it adds no information rather than being convenient.
    """
    by: Dict[int, Dict[int, float]] = defaultdict(dict)
    for r in rs:
        if r.get("opponent") != "self" or not int(r.get("finished", 0)):
            continue
        s = 0.5 if int(r.get("tied", 0) or 0) else float(int(r.get("won", 0)))
        by[int(r["game"])][int(r.get("orientation", 0) or 0)] = s
    return {g: sum(o.values()) / 2.0 for g, o in by.items() if len(o) == 2}


def mean_ci(xs: Sequence[float], z: float = 1.96) -> tuple:
    n = len(xs)
    if n == 0:
        return (None, None, None, 0)
    m = sum(xs) / n
    if n < 2:
        return (m, None, None, n)
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    half = z * math.sqrt(var / n)
    return (m, m - half, m + half, n)


def unpaired(rs: Sequence[dict]) -> dict:
    fin = [r for r in rs if int(r.get("finished", 0)) and r.get("opponent") == "self"]
    ties = sum(int(r.get("tied", 0) or 0) for r in fin)
    won = sum(int(r.get("won", 0)) for r in fin)
    decisive = len(fin) - ties
    p, lo, hi = wilson(won, decisive)
    return {"orientation_games": len(rs), "finished": len(fin), "decisive": decisive,
            "won": won, "ties": ties,
            "errors": sum(1 for r in rs if r.get("error")),
            "unfinished": sum(1 for r in rs if not int(r.get("finished", 0))),
            "win_rate": round(p, 4), "ci95": [round(lo, 4), round(hi, 4)]}


def defensive_fold(rs: Sequence[dict]) -> dict:
    keys = sorted({k for r in rs for k in r if k.startswith("n_defensive")})
    out = {k: sum(int(r.get(k, 0) or 0) for r in rs) for k in keys}
    out["defensive_banked_s"] = round(sum(float(r.get("defensive_banked_s", 0.0) or 0.0)
                                          for r in rs), 2)
    return out


def arm_block(cell: Sequence[dict], label: str, contested_s: Optional[float]) -> dict:
    """Everything one arm reports about itself. Shared by iteration 1 and 2 so the two rows are
    computed by ONE function rather than by two that could drift."""
    u = unpaired(cell)
    ps = pair_scores(cell)
    m, lo, hi, npairs = mean_ci(list(ps.values()))
    dec = sum(int(r.get("n_decisions", 0) or 0) for r in cell)
    srch = sum(int(r.get("n_searched", 0) or 0) for r in cell)
    chg = sum(int(r.get("n_changed", 0) or 0) for r in cell)
    fold = defensive_fold(cell)
    nd = max(1, fold.get("n_defensive", 0))
    ndr = max(1, fold.get("n_defensive_raced", 0))
    nfut = fold.get("n_defensive_futility", 0)
    n_race = max(1, sum(int(r.get("n_racing", 0) or 0) for r in cell))
    rounds = sum(int(r.get("racing_rounds_total", 0) or 0) for r in cell)
    wall = sum(float(r.get("wall_s", 0.0)) for r in cell)
    search_s = sum(float((r.get("realized_mean") or {}).get("elapsed", 0.0) or 0.0)
                   * int(r.get("n_searched", 0) or 0) for r in cell)
    budget = float(cell[0].get("budget") or 1.0)
    fallbacks: Dict[str, int] = defaultdict(int)
    for r in cell:
        for k, v in (r.get("fallbacks") or {}).items():
            fallbacks[k] += int(v)
    # `n_defensive_futility_deadline` does not exist on iteration-1 rows (the counter is what
    # iteration 2 added). `None` says "not measured"; a 0 would say "measured and none", which is
    # the opposite claim.
    fut_dl = (fold["n_defensive_futility_deadline"]
              if "n_defensive_futility_deadline" in fold else None)
    return {
        "label": label, "arm": cell[0].get("arm"), "budget_s": budget,
        "contested_deadline_s": contested_s,
        "root_strategy": cell[0].get("root_strategy"), "score_mode": cell[0].get("score_mode"),
        **u,
        "paired_win_rate": round(m, 4) if m is not None else None,
        "paired_ci95": [round(lo, 4), round(hi, 4)] if lo is not None else None,
        "paired_ci_halfwidth": round((hi - lo) / 2, 4) if lo is not None else None,
        "n_pairs": npairs, "decisions": dec, "searched": srch, "changed": chg,
        "searched_frac": round(srch / dec, 4) if dec else None,
        "change_rate": round(chg / srch, 4) if srch else None,
        "fallbacks": dict(fallbacks),
        "rates": {
            "forced": round(fold.get("n_defensive_forced", 0) / nd, 4),
            "forced_wp": round(fold.get("n_defensive_forced_wp", 0) / nd, 4),
            "forced_n_legal": round(fold.get("n_defensive_forced_n_legal", 0) / nd, 4),
            "raced": round(fold.get("n_defensive_raced", 0) / nd, 4),
            "separated_of_raced": round(fold.get("n_defensive_separated", 0) / ndr, 4),
            "futility_of_raced": round(nfut / ndr, 4),
            "kept_of_raced": round(fold.get("n_defensive_kept", 0) / ndr, 4),
            "overrule_of_all": round(fold.get("n_defensive_overruled", 0) / nd, 4),
            "overrule_of_raced": round(fold.get("n_defensive_overruled", 0) / ndr, 4),
            # THE SPLIT iteration 2 added. A futility stop the CLOCK ended is a budget finding; a
            # race that ran its supply out and still could not separate is probe I's U-shape. The
            # first cell could not tell them apart — all 3,301 of its stops were both.
            "futility_deadline_frac": (round(fut_dl / nfut, 4)
                                       if (fut_dl is not None and nfut) else None),
        },
        "counts": fold,
        "futility_split": {
            "total": nfut,
            "deadline_truncated": fut_dl,
            "genuine_non_separation": (nfut - fut_dl) if fut_dl is not None else None,
            "note": ("iteration-1 rows predate the counter, so `deadline_truncated` is None there "
                     "rather than 0 — the record states it was 100% by the exact identity "
                     "futility == deadline_truncated over its 3,301 stops"),
        },
        "race": {
            "raced": n_race, "rounds_total": rounds,
            "rounds_per_race": round(rounds / n_race, 3),
            "seq_floor": 5,
            "max_rounds_supply": 64,
            "eliminated_per_race": round(sum(int(r.get("racing_eliminated_total", 0) or 0)
                                             for r in cell) / n_race, 3),
            "arms_saved_total": sum(int(r.get("racing_arms_saved_total", 0) or 0) for r in cell),
            "rounds_incomplete": sum(int(r.get("racing_rounds_incomplete_total", 0) or 0)
                                     for r in cell),
            "deadline_truncated_decisions": sum(int(r.get("deadline_truncated", 0) or 0)
                                                for r in cell),
            "mean_search_s_per_raced_decision": round(search_s / max(1, srch), 4),
        },
        # THE REALIZED ENVELOPE — measured, never asserted from the plan. `search_s_per_game` is
        # what the strategy actually spent thinking; `wall_s_per_game` is the whole battle
        # including the two live policy forwards and the sim, so the two are not interchangeable.
        "envelope": {
            "mean_wall_s_per_game": round(wall / max(1, len(cell)), 2),
            "total_wall_h": round(wall / 3600.0, 2),
            "search_s_per_game": round(search_s / max(1, len(cell)), 2),
            "contested_decisions_per_game": round(ndr / max(1, len(cell)), 2),
            "uniform_notional_s_per_game": round(budget * nd / max(1, len(cell)), 2),
            "inside_uniform_notional": bool(search_s <= budget * nd),
            # DERIVED from measured spend, and this is the bank figure to quote across the two
            # iterations. The raw `defensive_banked_s` counter is per-branch — a forced decision
            # banks `budget_s`, a raced one banks its own deadline's residual — so once the two
            # deadlines differ the counter mixes scales and its ratio against a 1 s notional can
            # exceed 1. Spend is one scale, always.
            "banked_vs_uniform_s_per_game": round((budget * nd - search_s) / max(1, len(cell)), 2),
            "banked_frac_of_uniform": round(1.0 - search_s / max(1e-9, budget * nd), 4),
        },
        "banked": {
            "total_s": fold["defensive_banked_s"],
            "per_decision_s": round(fold["defensive_banked_s"] / nd, 4),
            "per_game_s": round(fold["defensive_banked_s"] / max(1, len(cell)), 3),
            "of_notional_budget": round(fold["defensive_banked_s"] / max(1e-9, nd * budget), 4),
            "caveat": ("the RAW counter, mixing scales once the contested deadline differs from "
                       "--budget: a forced decision banks --budget while a raced one banks the "
                       "residual of its own (larger) deadline. Quote envelope.banked_frac_of_"
                       "uniform instead, which is derived from measured SPEND on one scale."),
        },
        "_pairs": ps,
    }


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    paths = (argv[0] if argv else "tmp/sd2/shardA.jsonl").split(",")
    out_path = argv[1] if len(argv) > 1 else None

    shards = {}
    cell: List[dict] = []
    for p in paths:
        rs = rows(p.strip())
        idx = sorted({int(r["game"]) for r in rs})
        shards[os.path.basename(p.strip())] = {
            "orientation_games": len(rs),
            "game_index_range": [idx[0], idx[-1]] if idx else None,
        }
        cell += rs
    if not cell:
        print(f"no rows in {paths}", file=sys.stderr)
        return 1
    # DISJOINT windows, asserted rather than trusted: an overlap would double-count a game index
    # and hand `pair_scores` two different battles under one key.
    seen: Dict[int, str] = {}
    for name, p in zip(shards, paths):
        for r in rows(p.strip()):
            g = int(r["game"])
            if seen.setdefault(g, name) != name:
                raise SystemExit(f"shards {seen[g]} and {name} both played game index {g} — the "
                                 "windows overlap and the paired read would be wrong")
    # ...and ONE cell, not three: every row must carry the same arm/budget/opponent/strategy.
    sig = {(r.get("arm"), r.get("budget"), r.get("opponent"), r.get("root_strategy"),
            r.get("score_mode")) for r in cell}
    if len(sig) != 1:
        raise SystemExit(f"the shards did not play one cell: {sorted(sig)}")

    it2 = arm_block(cell, "defensive_1s_contested3s", 3.0)
    it1_rows = rows(ITER1)
    it1 = arm_block(it1_rows, "defensive_1s (iteration 1)", None) if it1_rows else None

    hist = {}
    for label, rel, what in HISTORICAL:
        hr = rows(os.path.join(REPO, rel))
        if not hr:
            continue
        u = unpaired(hr)
        ps = pair_scores(hr)
        m, lo, hi, n = mean_ci(list(ps.values()))
        dec = sum(int(r.get("n_decisions", 0) or 0) for r in hr)
        srch = sum(int(r.get("n_searched", 0) or 0) for r in hr)
        chg = sum(int(r.get("n_changed", 0) or 0) for r in hr)
        hist[label] = {
            "what": what, "path": rel, "budget_s": hr[0].get("budget"), "arm": hr[0].get("arm"),
            **u,
            "paired_win_rate": round(m, 4) if m is not None else None,
            "paired_ci95": [round(lo, 4), round(hi, 4)] if lo is not None else None,
            "n_pairs": n, "decisions": dec, "searched": srch, "changed": chg,
            "searched_frac": round(srch / dec, 4) if dec else None,
            "change_rate": round(chg / srch, 4) if srch else None,
            "mean_wall_s": round(sum(float(r.get("wall_s", 0.0)) for r in hr) / len(hr), 2),
            "_pairs": ps,
        }

    def paired_vs(other: dict, note: str) -> dict:
        shared = sorted(set(it2["_pairs"]) & set(other["_pairs"]))
        diffs = [it2["_pairs"][g] - other["_pairs"][g] for g in shared]
        pm, plo, phi, pn = mean_ci(diffs)
        return {"n_shared_games": pn,
                "delta": round(pm, 4) if pm is not None else None,
                "ci95": [round(plo, 4), round(phi, 4)] if plo is not None else None,
                "excludes_zero": bool(plo is not None and (plo > 0 or phi < 0)),
                "note": note}

    comparisons = {}
    bar = hist.get("mirror_honest_1s")
    if bar:
        d, dlo, dhi = newcombe(it2["won"], it2["decisive"], bar["won"], bar["decisive"])
        comparisons["vs_honest_1s_unpaired"] = {
            "delta": round(d, 4), "ci95": [round(dlo, 4), round(dhi, 4)],
            "excludes_zero": bool(dlo > 0 or dhi < 0)}
        comparisons["vs_honest_1s_paired_on_game"] = paired_vs(
            bar, "both cells ran at --games-seed 7 off the same checkpoint, so game g is the same "
                 "pinned dice and the same team draw in both arms")
    if it1:
        d, dlo, dhi = newcombe(it2["won"], it2["decisive"], it1["won"], it1["decisive"])
        comparisons["vs_iteration1_unpaired"] = {
            "delta": round(d, 4), "ci95": [round(dlo, 4), round(dhi, 4)],
            "excludes_zero": bool(dlo > 0 or dhi < 0)}
        comparisons["vs_iteration1_paired_on_game"] = paired_vs(
            it1, "THE REGISTERED NO-REGRESSION ROW: the two cells differ in exactly one flag "
                 "(--defensive-contested-deadline-s), and on the shared indices they are the same "
                 "pinned dice and the same team draw")

    r2, r1 = it2["rates"], (it1["rates"] if it1 else {})
    bars = {
        "primary_no_regression": {
            "registered": "win rate >= iteration 1 (NO REGRESSION is the primary bar)",
            "iteration1_paired": it1["paired_win_rate"] if it1 else None,
            "iteration2_paired": it2["paired_win_rate"],
            "paired_delta_on_shared_games": comparisons.get(
                "vs_iteration1_paired_on_game", {}).get("delta"),
            "held": bool(it1 and it2["paired_win_rate"] >= it1["paired_win_rate"]),
            "held_within_ci": bool(
                it1 and (comparisons.get("vs_iteration1_paired_on_game", {}).get("ci95") or
                         [None])[0] is not None
                and comparisons["vs_iteration1_paired_on_game"]["ci95"][1] > 0),
        },
        "stretch_ci_above_null": {
            "registered": ("paired CI above 0.50 — resolves only if the true rate is >= ~0.525; a "
                           "0.51-ish point estimate is 'real but unresolved', PRE-STATED"),
            "paired_win_rate": it2["paired_win_rate"],
            "paired_ci95": it2["paired_ci95"],
            "paired_ci_above_null": bool(it2["paired_ci95"] and it2["paired_ci95"][0] > 0.5),
            "unpaired_ci_above_null": bool(it2["ci95"][0] > 0.5),
            "resolvable_at_this_n": ("a true rate >= "
                                     f"{round(0.5 + (it2['paired_ci_halfwidth'] or 0), 4)} would "
                                     "have cleared 0.50 at this n"),
        },
        "prediction_separated_of_raced": {
            "registered": "rises from 0.157 toward probe I's ~0.48 ceiling",
            "iteration1": r1.get("separated_of_raced"),
            "iteration2": r2["separated_of_raced"],
            "probe_i_ceiling": 0.478,
            "held": bool(it1 and r2["separated_of_raced"] > r1.get("separated_of_raced", 0)),
        },
        "prediction_overrule_rate": {
            "registered": [0.06, 0.12],
            "iteration1": r1.get("overrule_of_all"),
            "iteration2": r2["overrule_of_all"],
            "in_range": bool(0.06 <= r2["overrule_of_all"] <= 0.12),
        },
        "envelope_claim": {
            "registered": ("~11 contested/game x 3 s ~ 33 s/game stays inside the same total-time "
                           "envelope the uniform-1s notional implies (~42 s) — reported so the "
                           "claim is checked, not assumed"),
            "contested_decisions_per_game": it2["envelope"]["contested_decisions_per_game"],
            "search_s_per_game": it2["envelope"]["search_s_per_game"],
            "uniform_notional_s_per_game": it2["envelope"]["uniform_notional_s_per_game"],
            "inside_uniform_notional": it2["envelope"]["inside_uniform_notional"],
            "still_banked_frac": it2["envelope"]["banked_frac_of_uniform"],
            "iteration1_banked_frac": (it1["envelope"]["banked_frac_of_uniform"]
                                       if it1 else None),
        },
    }

    out = {
        "date": "2026-08-29",
        "iteration": 2,
        "one_change": ("--defensive-contested-deadline-s 3.0 — a decision that PASSES the triage "
                       "gate is granted 3 s instead of the uniform --budget of 1 s. Gate "
                       "threshold 0.15, seq rule, elimination floor 5, win-prob leaf, depth 1, "
                       "confirm 0, max_rounds supply 64: ALL UNCHANGED."),
        "checkpoint": ("models/ai_v9_29_rev1_0823/checkpoints/checkpoint_9995088_steps.zip "
                       "(the SAME checkpoint iteration 1 and the historical mirror arms played)"),
        "invocation": ("python -m main.search_dividend <ckpt> --arm honest --budget 1 "
                       "--root-strategy defensive --defensive-leaf winprob "
                       "--defensive-wp-margin 0.15 --defensive-confirm 0 "
                       "--defensive-contested-deadline-s 3.0 --games-start <lo> --games <n> "
                       "--games-seed 7 --opponents self --battle-timeout-s 1800 "
                       "--battle-idle-s 120"),
        "shards": shards,
        "null": 0.5,
        "arm": {k: v for k, v in it2.items() if k != "_pairs"},
        "iteration1": {k: v for k, v in it1.items() if k != "_pairs"} if it1 else None,
        "historical": {k: {kk: vv for kk, vv in v.items() if kk != "_pairs"}
                       for k, v in hist.items()},
        "comparisons": comparisons,
        "bars": bars,
    }
    if out_path:
        with open(out_path, "w") as fh:
            json.dump(out, fh, indent=1, sort_keys=False)
            fh.write("\n")
    print(json.dumps({k: v for k, v in out.items() if k != "historical"}, indent=1))
    print("\nHISTORICAL (quoted, not re-run):")
    for k, v in out["historical"].items():
        print(f"  {k:20} n={v['decisive']:<4} wr={v['win_rate']:.4f} {v['ci95']} "
              f"paired={v['paired_win_rate']} searched_frac={v['searched_frac']} "
              f"change_rate={v['change_rate']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
