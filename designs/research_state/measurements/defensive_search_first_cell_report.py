"""Score the DEFENSIVE PAIRED SEARCH first mirror cell against its registered bars.

    export PYTHONPATH=$PYTHONPATH:src
    python designs/research_state/measurements/defensive_search_first_cell_report.py \
        tmp/defensive/cell.jsonl designs/research_state/measurements/defensive_search_first_cell_2026-08-29.json

(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)

Reads the cell's append-only rows plus the battery's HISTORICAL mirror arms (quoted, never
re-run), and emits the JSON the record is rendered from. Three comparisons, in increasing
strength, because the arms did not merely play a similar experiment — they played the SAME
battles:

* **vs the null.** A mirror's no-effect point is 0.50 by CONSTRUCTION (same network, search off on
  one side), so "does the interval reach it" is the whole primary test.
* **vs honest_1s, UNPAIRED.** Newcombe's hybrid-score interval on the difference of two
  proportions — it inherits Wilson's behaviour near the boundary, which a normal-approximation
  difference does not.
* **vs honest_1s, PAIRED on the game index.** Both cells were played at ``--games-seed 7`` off the
  same checkpoint, so game ``g`` is the same pinned dice and the same team draw in both. The team
  draw is most of the variance in a mirror at these n (the exploiter work measured an "edge" that
  was entirely team draw), and pairing on ``g`` differences it out. This is the reportable one on
  the overlap.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

REPO = "/home/goodlad/dev/gen3ai"

#: The battery's own prior mirror arms — QUOTED from the files they were played into, never
#: re-run. Each entry is (label, path, what it is), and the paths are in the MAIN checkout because
#: that is where the battery ran.
HISTORICAL = [
    ("mirror_base", "tmp/search_dividend/mirror_base.jsonl",
     "the policy alone — the control; 0.500 by construction"),
    ("mirror_honest_1s", "tmp/search_dividend/mirror_honest_1s.jsonl",
     "THE PRIMARY BAR: plain depth-1 grid search, belief-determinized, 1 s/decision"),
    ("mirror_honest_3s", "tmp/search_dividend/mirror_honest_3s.jsonl", "the same at 3 s"),
    ("mirror_oracle_1s", "tmp/search_dividend/mirror_oracle_1s.jsonl",
     "grid search on the TRUE hidden team, 1 s"),
    ("mirror_oracle_3s", "tmp/search_dividend/mirror_oracle_3s.jsonl", "...at 3 s"),
    ("playoff_10s", "tmp/search_dividend/playoff_10s.jsonl",
     "the screen + paired terminal rollouts at 20 s — the only historical arm that did not lose"),
]


def rows(path: str) -> List[dict]:
    out: List[dict] = []
    if not os.path.exists(path):
        return out
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    break
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
    """The 95% interval on ``p1 - p2`` for two INDEPENDENT proportions (Newcombe's method 10).

    Built from the two Wilson intervals rather than from a pooled normal SE, because at these n
    and away from 0.5 the normal difference interval leaves [-1, 1] and reports impossible bounds
    — the same reason the battery uses Wilson for a single rate.
    """
    p1, l1, u1 = wilson(k1, n1, z)
    p2, l2, u2 = wilson(k2, n2, z)
    lo = (p1 - p2) - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = (p1 - p2) + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return (p1 - p2, max(-1.0, lo), min(1.0, hi))


def pair_scores(rs: Sequence[dict]) -> Dict[int, float]:
    """``{game_index: paired score}`` over the games with BOTH orientations finished.

    A pair scores 1 / 0.5 / 0 for won-both / split / lost-both, and a TIE counts 0.5 — which is
    exactly the null's own prediction, so it is the value that adds no information rather than a
    convenient one.
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
    keys = [k for k in rs[0] if k.startswith("n_defensive")] if rs else []
    out = {k: sum(int(r.get(k, 0) or 0) for r in rs) for k in keys}
    out["defensive_banked_s"] = round(sum(float(r.get("defensive_banked_s", 0.0) or 0.0)
                                          for r in rs), 2)
    return out


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    cell_path = argv[0] if argv else "tmp/defensive/cell.jsonl"
    out_path = argv[1] if len(argv) > 1 else None
    cell = rows(cell_path)
    if not cell:
        print(f"no rows in {cell_path}", file=sys.stderr)
        return 1

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
            "what": what, "path": rel, "budget_s": hr[0].get("budget"),
            "arm": hr[0].get("arm"), **u,
            "paired_win_rate": round(m, 4) if m is not None else None,
            "paired_ci95": [round(lo, 4), round(hi, 4)] if lo is not None else None,
            "n_pairs": n, "decisions": dec, "searched": srch, "changed": chg,
            "searched_frac": round(srch / dec, 4) if dec else None,
            "change_rate": round(chg / srch, 4) if srch else None,
            "mean_wall_s": round(sum(float(r.get("wall_s", 0.0)) for r in hr) / len(hr), 2),
            "_pairs": ps,
        }

    u = unpaired(cell)
    ps = pair_scores(cell)
    m, lo, hi, npairs = mean_ci(list(ps.values()))
    dec = sum(int(r.get("n_decisions", 0) or 0) for r in cell)
    srch = sum(int(r.get("n_searched", 0) or 0) for r in cell)
    chg = sum(int(r.get("n_changed", 0) or 0) for r in cell)
    fold = defensive_fold(cell)
    nd = max(1, fold.get("n_defensive", 0))
    ndr = max(1, fold.get("n_defensive_raced", 0))
    fallbacks: Dict[str, int] = defaultdict(int)
    for r in cell:
        for k, v in (r.get("fallbacks") or {}).items():
            fallbacks[k] += int(v)

    arm = {
        "label": "defensive_1s", "arm": cell[0].get("arm"), "budget_s": cell[0].get("budget"),
        "root_strategy": cell[0].get("root_strategy"), "score_mode": cell[0].get("score_mode"),
        **u,
        "paired_win_rate": round(m, 4) if m is not None else None,
        "paired_ci95": [round(lo, 4), round(hi, 4)] if lo is not None else None,
        "n_pairs": npairs, "decisions": dec, "searched": srch, "changed": chg,
        "searched_frac": round(srch / dec, 4) if dec else None,
        "change_rate": round(chg / srch, 4) if srch else None,
        "mean_wall_s": round(sum(float(r.get("wall_s", 0.0)) for r in cell) / len(cell), 2),
        "total_wall_h": round(sum(float(r.get("wall_s", 0.0)) for r in cell) / 3600.0, 2),
        "fallbacks": dict(fallbacks),
        "rates": {
            "forced": round(fold.get("n_defensive_forced", 0) / nd, 4),
            "forced_wp": round(fold.get("n_defensive_forced_wp", 0) / nd, 4),
            "forced_n_legal": round(fold.get("n_defensive_forced_n_legal", 0) / nd, 4),
            "raced": round(fold.get("n_defensive_raced", 0) / nd, 4),
            "separated_of_raced": round(fold.get("n_defensive_separated", 0) / ndr, 4),
            "futility_of_raced": round(fold.get("n_defensive_futility", 0) / ndr, 4),
            "kept_of_raced": round(fold.get("n_defensive_kept", 0) / ndr, 4),
            "overrule_of_all": round(fold.get("n_defensive_overruled", 0) / nd, 4),
            "overrule_of_raced": round(fold.get("n_defensive_overruled", 0) / ndr, 4),
        },
        "counts": fold,
        # What the RACE itself did, which is what decides whether the futility mass is the game's
        # U-shape (probe I) or merely this budget's round supply. `rounds_per_race` against the
        # `seq` rule's floor of 5 is the number that tells them apart: below the floor NO
        # elimination is legal at all, so a race that never got there did not fail to separate —
        # it was never allowed to try.
        "race": {
            "raced": sum(int(r.get("n_racing", 0) or 0) for r in cell),
            "rounds_total": sum(int(r.get("racing_rounds_total", 0) or 0) for r in cell),
            "rounds_per_race": round(
                sum(int(r.get("racing_rounds_total", 0) or 0) for r in cell)
                / max(1, sum(int(r.get("n_racing", 0) or 0) for r in cell)), 3),
            "seq_floor": 5,
            "eliminated_per_race": round(
                sum(int(r.get("racing_eliminated_total", 0) or 0) for r in cell)
                / max(1, sum(int(r.get("n_racing", 0) or 0) for r in cell)), 3),
            "arms_saved_total": sum(int(r.get("racing_arms_saved_total", 0) or 0) for r in cell),
            "rounds_incomplete": sum(int(r.get("racing_rounds_incomplete_total", 0) or 0)
                                     for r in cell),
            "deadline_truncated_decisions": sum(int(r.get("deadline_truncated", 0) or 0)
                                                for r in cell),
            "mean_search_s_per_raced_decision": round(
                sum(float((r.get("realized_mean") or {}).get("elapsed", 0.0) or 0.0)
                    * int(r.get("n_searched", 0) or 0) for r in cell)
                / max(1, srch), 4),
        },
        "banked": {
            "total_s": fold["defensive_banked_s"],
            "per_decision_s": round(fold["defensive_banked_s"] / nd, 4),
            "per_game_s": round(fold["defensive_banked_s"] / max(1, len(cell)), 3),
            # What the SAME decisions would have cost a search that never refused: one full budget
            # per decision the gate declined, plus the residual a futility stop handed back.
            "of_notional_budget": round(
                fold["defensive_banked_s"] / max(1e-9, nd * float(cell[0].get("budget") or 1)), 4),
        },
    }

    bar = hist.get("mirror_honest_1s")
    comparisons = {}
    if bar:
        d, dlo, dhi = newcombe(u["won"], u["decisive"], bar["won"], bar["decisive"])
        comparisons["vs_honest_1s_unpaired"] = {
            "delta": round(d, 4), "ci95": [round(dlo, 4), round(dhi, 4)],
            "excludes_zero": bool(dlo > 0 or dhi < 0),
        }
        shared = sorted(set(ps) & set(bar["_pairs"]))
        diffs = [ps[g] - bar["_pairs"][g] for g in shared]
        pm, plo, phi, pn = mean_ci(diffs)
        comparisons["vs_honest_1s_paired_on_game"] = {
            "n_shared_games": pn,
            "delta": round(pm, 4) if pm is not None else None,
            "ci95": [round(plo, 4), round(phi, 4)] if plo is not None else None,
            "excludes_zero": bool(plo is not None and (plo > 0 or phi < 0)),
            "note": ("both cells ran at --games-seed 7 off the same checkpoint, so game g is the "
                     "same pinned dice and the same team draw in both arms"),
        }

    bars = {
        "primary": {
            "registered": ("DEFENSIVE decisively above the historical honest_1s mirror arm "
                           "(0.292) with its CI reaching 0.50 — 'search stops losing'"),
            "above_honest_1s": bool(bar and u["ci95"][0] > bar["win_rate"]),
            "ci_reaches_null": bool(u["ci95"][1] >= 0.5),
            "paired_ci_reaches_null": bool(hi is not None and hi >= 0.5),
        },
        "stretch": {
            "registered": "CI above 0.50 — 'search finally pays'",
            "unpaired_ci_above_null": bool(u["ci95"][0] > 0.5),
            "paired_ci_above_null": bool(lo is not None and lo > 0.5),
        },
        "prediction_overrule_rate": {
            "registered": [0.08, 0.17],
            "measured_of_all_decisions": arm["rates"]["overrule_of_all"],
            "measured_of_raced": arm["rates"]["overrule_of_raced"],
            "in_range": bool(0.08 <= arm["rates"]["overrule_of_all"] <= 0.17),
        },
    }

    out = {
        "date": "2026-08-29",
        "checkpoint": ("models/ai_v9_29_rev1_0823/checkpoints/checkpoint_9995088_steps.zip "
                       "(the SAME checkpoint the historical mirror arms played)"),
        "invocation": ("python -m main.search_dividend <ckpt> --arm honest --budget 1 "
                       "--root-strategy defensive --defensive-leaf winprob "
                       "--defensive-wp-margin 0.15 --defensive-confirm 0 --games 200 "
                       "--games-seed 7 --opponents self --battle-timeout-s 1800 "
                       "--battle-idle-s 120"),
        "null": 0.5,
        "arm": arm,
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
