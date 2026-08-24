"""Reading the battery: per-cell win rates with Wilson intervals, and anchored-ELO deltas.

Two rules the report obeys, both inherited from things this project has already got wrong once:

* **A win rate is quoted with an interval, never alone.** At the smoke's N=10 the Wilson interval
  on 0.6 runs [0.31, 0.83]; a bare "0.60 vs 0.50" reads like a result and is not one.
* **An ELO here is a DELTA between arms fitted from the same games, and it is not a ladder ELO.**
  The anchored Bradley-Terry fit re-solves every node on every add and systematically inflates the
  newest one, so a mid-run absolute number is not reportable. What IS reportable is the difference
  between two arms measured against the SAME pinned bots on MATCHED games — the anchors hold the
  gauge fixed, so the arms are on one scale by construction.
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Optional, Sequence


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    """The Wilson score interval for ``k`` successes in ``n`` trials.

    Wilson rather than normal-approximation because the battery's cells are small and often near
    0 or 1, exactly where the normal interval leaves the unit interval and reports impossible
    bounds."""
    if n <= 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (p, max(0.0, centre - half), min(1.0, centre + half))


def per_cell(rows: Sequence[dict]) -> List[dict]:
    """One summary per ``(arm, budget, opponent)`` cell, with a decision-level fold beside it.

    Unfinished games (a crash, a transport error) are counted SEPARATELY and excluded from the win
    rate — a timeout / error is never a semantic outcome, and folding one in as a loss is how a
    starved box gets reported as a weaker policy."""
    agg: Dict[tuple, dict] = defaultdict(lambda: {
        "games": 0, "finished": 0, "won": 0, "errors": 0, "wall_s": 0.0,
        "n_decisions": 0, "n_searched": 0, "n_changed": 0, "deadline_truncated": 0,
        "worlds_gate_failed": 0, "fallbacks": defaultdict(int),
        "_m": [], "_k": [], "_r": [], "_arms": [], "_elapsed": [],
    })
    for r in rows:
        a = agg[(r["arm"], float(r["budget"]), r["opponent"])]
        a["games"] += 1
        a["finished"] += int(r.get("finished", 0))
        a["won"] += int(r.get("won", 0))
        a["errors"] += 1 if r.get("error") else 0
        a["wall_s"] += float(r.get("wall_s", 0.0))
        for key in ("n_decisions", "n_searched", "n_changed", "deadline_truncated",
                    "worlds_gate_failed"):
            a[key] += int(r.get(key, 0) or 0)
        for reason, n in (r.get("fallbacks") or {}).items():
            a["fallbacks"][reason] += int(n)
        rm = r.get("realized_mean") or {}
        for src, dst in (("m_opp", "_m"), ("k_worlds", "_k"), ("r_dice", "_r"),
                         ("arms", "_arms"), ("elapsed", "_elapsed")):
            if rm.get(src):
                a[dst].append(float(rm[src]))

    out: List[dict] = []
    for (arm, budget, opp), a in sorted(agg.items()):
        p, lo, hi = wilson(a["won"], a["finished"])
        out.append({
            "arm": arm, "budget": budget, "opponent": opp,
            "games": a["games"], "finished": a["finished"], "won": a["won"],
            "errors": a["errors"],
            "win_rate": round(p, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "mean_wall_s": round(a["wall_s"] / max(1, a["games"]), 2),
            "decisions": a["n_decisions"], "searched": a["n_searched"],
            "changed": a["n_changed"],
            "change_rate": round(a["n_changed"] / a["n_searched"], 4) if a["n_searched"] else None,
            "deadline_truncated": a["deadline_truncated"],
            "worlds_gate_failed": a["worlds_gate_failed"],
            "fallbacks": dict(a["fallbacks"]),
            "realized_mean": {
                "opp_candidates": _mean(a["_m"]), "worlds": _mean(a["_k"]),
                "dice": _mean(a["_r"]), "arms_scored": _mean(a["_arms"]),
                "search_s": _mean(a["_elapsed"]),
            },
        })
    return out


def _mean(v: Sequence[float]) -> Optional[float]:
    return round(sum(v) / len(v), 3) if v else None


def elo_by_arm(rows: Sequence[dict], anchors_path: Optional[str] = None) -> dict:
    """Anchored Bradley-Terry ELO per ``(arm, budget)``, fitted from the battery's own win records.

    Each arm/budget cell is entered as one ``EvalRow`` whose ``step`` encodes the cell (the fitter
    keys snapshots by step), so all arms are solved against the SAME pinned bot anchors in one
    system — which is what makes the DELTA between arms meaningful even though no single absolute
    number here should be quoted as a ladder ELO.
    """
    from agents.training.elo import BOT_ANCHORS_PATH, EvalRow, fit_elo, load_bot_anchors

    per = per_cell(rows)
    cells = sorted({(c["arm"], c["budget"]) for c in per})
    if not cells:
        return {"cells": [], "note": "no rows"}
    index = {c: i for i, c in enumerate(cells)}
    eval_rows: List[EvalRow] = []
    games_by_cell: Dict[tuple, int] = {}
    for (arm, budget) in cells:
        bots = {c["opponent"]: c["win_rate"] for c in per
                if (c["arm"], c["budget"]) == (arm, budget) and c["finished"] > 0}
        n = max((c["finished"] for c in per if (c["arm"], c["budget"]) == (arm, budget)),
                default=0)
        games_by_cell[(arm, budget)] = n
        if bots:
            eval_rows.append(EvalRow(step=index[(arm, budget)], n_games=n, bots=bots))
    if not eval_rows:
        return {"cells": [], "note": "no finished games"}
    anchors = load_bot_anchors(anchors_path or BOT_ANCHORS_PATH)
    pin = anchors["ratings"] if anchors else None
    base = float(anchors["base"]) if anchors and "base" in anchors else 1000.0
    fit = fit_elo(eval_rows, pin_ratings=pin, base=base)
    curve = {step: (elo, se) for (step, elo, se) in fit.snapshot_curve()}
    out = []
    for cell, i in index.items():
        elo, se = curve.get(i, (None, None))
        out.append({"arm": cell[0], "budget": cell[1], "games": games_by_cell[cell],
                    "elo": round(elo, 1) if elo is not None else None,
                    "se": round(se, 1) if se is not None else None})
    base_elo = next((c["elo"] for c in out if c["arm"] == "base"), None)
    for c in out:
        c["delta_vs_base"] = (round(c["elo"] - base_elo, 1)
                              if (c["elo"] is not None and base_elo is not None) else None)
    return {
        "cells": out, "anchored": bool(anchors), "base": base,
        # Named, not implied: an ELO fitted from bot win rates alone is only as separable as those
        # win rates. If every arm saturates a bot, the fit has nothing to separate them with.
        "caveats": [
            "Deltas are between arms fitted from the SAME matched games against the SAME pinned "
            "bot anchors; no single number here is a ladder ELO.",
            "A saturated bot roster compresses every arm onto the anchors and the deltas go to "
            "zero for want of signal, not for want of a dividend.",
        ],
    }


def format_report(rows: Sequence[dict], anchors_path: Optional[str] = None) -> str:
    """The human-readable summary the CLI prints."""
    per = per_cell(rows)
    lines = ["arm      budget  opponent          n   wr     ci95            "
             "chg    m/k/r/arms          s/dec  fallbacks"]
    for c in per:
        rm = c["realized_mean"]
        widths = f"{rm['opp_candidates']}/{rm['worlds']}/{rm['dice']}/{rm['arms_scored']}"
        fb = ",".join(f"{k}:{v}" for k, v in sorted(c["fallbacks"].items())) or "-"
        lines.append(
            f"{c['arm']:<8} {c['budget']:<6g}  {c['opponent']:<16} {c['finished']:<3} "
            f"{c['win_rate']:<6} [{c['ci95'][0]:.2f},{c['ci95'][1]:.2f}]   "
            f"{(c['change_rate'] if c['change_rate'] is not None else 0):<6} "
            f"{widths:<19} {rm['search_s'] or 0:<6} {fb}")
    elo = elo_by_arm(rows, anchors_path)
    lines.append("")
    lines.append("anchored ELO by arm (delta vs base is the reportable quantity):")
    for c in elo.get("cells", []):
        lines.append(f"  {c['arm']:<8} @{c['budget']:<5g} elo={c['elo']} "
                     f"se={c['se']} delta_vs_base={c['delta_vs_base']} (n={c['games']})")
    for note in elo.get("caveats", []):
        lines.append(f"  ! {note}")
    return "\n".join(lines)
