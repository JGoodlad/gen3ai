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
        "games": 0, "finished": 0, "won": 0, "tied": 0, "errors": 0, "wall_s": 0.0,
        "n_decisions": 0, "n_searched": 0, "n_changed": 0, "n_deepened": 0,
        "deadline_truncated": 0,
        "worlds_gate_failed": 0, "fallbacks": defaultdict(int),
        "_m": [], "_k": [], "_r": [], "_arms": [], "_elapsed": [], "_depth": [], "_beam": [],
    })
    for r in rows:
        a = agg[(r["arm"], float(r["budget"]), r["opponent"])]
        a["games"] += 1
        a["finished"] += int(r.get("finished", 0))
        a["won"] += int(r.get("won", 0))
        a["tied"] += int(r.get("tied", 0) or 0)
        a["errors"] += 1 if r.get("error") else 0
        a["wall_s"] += float(r.get("wall_s", 0.0))
        for key in ("n_decisions", "n_searched", "n_changed", "n_deepened",
                    "deadline_truncated", "worlds_gate_failed"):
            a[key] += int(r.get(key, 0) or 0)
        for reason, n in (r.get("fallbacks") or {}).items():
            a["fallbacks"][reason] += int(n)
        rm = r.get("realized_mean") or {}
        for src, dst in (("m_opp", "_m"), ("k_worlds", "_k"), ("r_dice", "_r"),
                         ("arms", "_arms"), ("elapsed", "_elapsed"), ("depth", "_depth"),
                         ("beam", "_beam")):
            if rm.get(src):
                a[dst].append(float(rm[src]))

    out: List[dict] = []
    for (arm, budget, opp), a in sorted(agg.items()):
        # TIES ARE EXCLUDED FROM THE DENOMINATOR, and reported beside it. A gen3 tie is a real
        # outcome (poke-env: finished, `won is None`) and it used to be recorded as a LOSS,
        # because the battery inferred the result from a win COUNTER that cannot see a draw. The
        # convention here is "win rate among DECISIVE games"; the alternative — scoring a tie 0.5
        # — needs a different interval than Wilson's, and a half-success is not a Bernoulli trial.
        decisive = a["finished"] - a["tied"]
        p, lo, hi = wilson(a["won"], decisive)
        out.append({
            "arm": arm, "budget": budget, "opponent": opp,
            "games": a["games"], "finished": a["finished"], "decisive": decisive,
            "won": a["won"], "tied": a["tied"], "errors": a["errors"],
            "win_rate": round(p, 4), "ci95": [round(lo, 4), round(hi, 4)],
            "mean_wall_s": round(a["wall_s"] / max(1, a["games"]), 2),
            "decisions": a["n_decisions"], "searched": a["n_searched"],
            "changed": a["n_changed"], "deepened": a["n_deepened"],
            "change_rate": round(a["n_changed"] / a["n_searched"], 4) if a["n_searched"] else None,
            "deepen_rate": (round(a["n_deepened"] / a["n_searched"], 4)
                            if a["n_searched"] else None),
            "deadline_truncated": a["deadline_truncated"],
            "worlds_gate_failed": a["worlds_gate_failed"],
            "fallbacks": dict(a["fallbacks"]),
            "realized_mean": {
                "opp_candidates": _mean(a["_m"]), "worlds": _mean(a["_k"]),
                "dice": _mean(a["_r"]), "arms_scored": _mean(a["_arms"]),
                "search_s": _mean(a["_elapsed"]), "depth": _mean(a["_depth"]),
                "beam": _mean(a["_beam"]),
            },
        })
    return out


def _mean(v: Sequence[float]) -> Optional[float]:
    return round(sum(v) / len(v), 3) if v else None


MIRROR = "self"


def mirror_report(rows: Sequence[dict]) -> dict:
    """The MIRROR cells read against their own null, with the team draw differenced out.

    A mirror cell is the searched side against the SAME network unsearched, so the no-effect point
    is **exactly 0.50** — not an estimate, a construction. That is what makes it the sensitive
    contrast: the scripted roster saturates near 90% and would hide a real dividend inside its
    ceiling, whereas here every point above 50 is the search.

    Two readings, and the second is the one to quote at small n:

    * **unpaired** — wins over DECISIVE games with a Wilson interval, plus whether that interval
      excludes 0.50. Ties are counted and excluded from the denominator (see :func:`per_cell`).
    * **paired** — the side-swap read. Each game index is played in both orientations off ONE
      pinned seed, and the pair scores 1 / 0.5 / 0 for won-both / split / lost-both. The team draw
      is common to the pair and cancels in the difference, which matters because it is most of the
      variance at these n: a mirror's two sides pilot two different teams, and the exploiter work
      already measured an "edge" that was entirely team draw and vanished under an equal-pilot
      control. A split pair scores 0.5 because it is exactly the null's prediction.

    ``paired_ci`` is a normal interval on the mean of a bounded score, so it is honest only once
    there are enough pairs for the mean to be near-normal; ``n_pairs`` is published beside it so a
    reader can refuse it rather than be misled by it.
    """
    per = {(c["arm"], c["budget"]): c for c in per_cell(rows) if c["opponent"] == MIRROR}
    if not per:
        return {"cells": [], "note": "no mirror (--opponents self) rows"}

    scores: Dict[tuple, Dict[int, Dict[int, float]]] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r.get("opponent") != MIRROR or not int(r.get("finished", 0)):
            continue
        s = 0.5 if int(r.get("tied", 0) or 0) else float(int(r.get("won", 0)))
        scores[(r["arm"], float(r["budget"]))][int(r["game"])][int(r.get("orientation", 0) or 0)] = s

    cells: List[dict] = []
    for key, cell in sorted(per.items()):
        pairs = [sum(o.values()) / 2.0 for o in scores[key].values() if len(o) == 2]
        n = len(pairs)
        mean = sum(pairs) / n if n else None
        if n > 1:
            var = sum((p - mean) ** 2 for p in pairs) / (n - 1)
            half = 1.96 * math.sqrt(var / n)
            ci = [round(max(0.0, mean - half), 4), round(min(1.0, mean + half), 4)]
        else:
            ci = None
        rm = cell["realized_mean"]
        cells.append({
            "arm": cell["arm"], "budget": cell["budget"],
            "games": cell["games"], "finished": cell["finished"],
            "decisive": cell["decisive"], "won": cell["won"], "ties": cell["tied"],
            "errors": cell["errors"],
            "win_rate": cell["win_rate"], "ci95": cell["ci95"],
            # The null is 0.50 by construction, so "does the interval exclude it" is the whole
            # test — and it is stated rather than left for the reader to eyeball off two numbers.
            "beats_null": bool(cell["decisive"] and cell["ci95"][0] > 0.5),
            "worse_than_null": bool(cell["decisive"] and cell["ci95"][1] < 0.5),
            "n_pairs": n,
            "paired_win_rate": round(mean, 4) if mean is not None else None,
            "paired_ci95": ci,
            "unpaired_games": cell["finished"] - 2 * n,
            "change_rate": cell["change_rate"], "deepen_rate": cell["deepen_rate"],
            "searched": cell["searched"], "decisions": cell["decisions"],
            "fallbacks": cell["fallbacks"],
            "realized_mean": rm,
        })
    return {
        "cells": cells, "null": 0.5,
        "notes": [
            "A mirror cell's no-effect point is 0.50 BY CONSTRUCTION (same network, search off on "
            "one side) — every point above it is the search, not a stronger opponent model.",
            "Ties are excluded from the win-rate denominator and reported as `ties`; a paired "
            "score counts a tie as 0.5, which is exactly the null's prediction.",
            "The PAIRED rate is the one to quote at small n: it differences out the team draw, "
            "which is most of the variance in a mirror. `unpaired_games` counts orientation-games "
            "with no partner yet — they contribute to `win_rate` and not to `paired_win_rate`.",
        ],
    }


def elo_by_arm(rows: Sequence[dict], anchors_path: Optional[str] = None) -> dict:
    """Anchored Bradley-Terry ELO per ``(arm, budget)``, fitted from the battery's own win records.

    Each arm/budget cell is entered as one ``EvalRow`` whose ``step`` encodes the cell (the fitter
    keys snapshots by step), so all arms are solved against the SAME pinned bot anchors in one
    system — which is what makes the DELTA between arms meaningful even though no single absolute
    number here should be quoted as a ladder ELO.
    """
    from agents.training.elo import BOT_ANCHORS_PATH, EvalRow, fit_elo, load_bot_anchors

    # MIRROR cells are excluded, and this is not a technicality. The fit is ANCHORED — every
    # rating is pinned by the fixed bots' known ratings, which is precisely what puts all the arms
    # on one gauge. `self` has no anchor: its rating is whatever the trainee's own rating is, so a
    # mirror cell would enter the system as a free parameter matched against another free
    # parameter and drag the arms it shares a node with. The mirror's reading is its own — a win
    # rate against a null of exactly 0.50 (:func:`mirror_report`) — and it needs no ELO to be one.
    per = [c for c in per_cell(rows) if c["opponent"] != MIRROR]
    cells = sorted({(c["arm"], c["budget"]) for c in per})
    if not cells:
        return {"cells": [], "note": "no anchored (non-mirror) rows — see the mirror report"}
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
    lines = ["arm      budget  opponent          n   wr     ci95           tie "
             "chg    dep    m/k/r/arms/depth        s/dec  fallbacks"]
    for c in per:
        rm = c["realized_mean"]
        widths = (f"{rm['opp_candidates']}/{rm['worlds']}/{rm['dice']}/{rm['arms_scored']}"
                  f"/{rm['depth']}")
        fb = ",".join(f"{k}:{v}" for k, v in sorted(c["fallbacks"].items())) or "-"
        lines.append(
            f"{c['arm']:<8} {c['budget']:<6g}  {c['opponent']:<16} {c['decisive']:<3} "
            f"{c['win_rate']:<6} [{c['ci95'][0]:.2f},{c['ci95'][1]:.2f}]  {c['tied']:<3} "
            f"{(c['change_rate'] if c['change_rate'] is not None else 0):<6} "
            f"{(c['deepen_rate'] if c['deepen_rate'] is not None else 0):<6} "
            f"{widths:<23} {rm['search_s'] or 0:<6} {fb}")

    mirror = mirror_report(rows)
    if mirror.get("cells"):
        lines.append("")
        lines.append("MIRROR cells — the searched side vs the SAME network unsearched "
                     "(null = 0.50 by construction):")
        for c in mirror["cells"]:
            verdict = ("ABOVE null" if c["beats_null"] else
                       ("BELOW null" if c["worse_than_null"] else "null not excluded"))
            paired = ("-" if c["paired_win_rate"] is None else
                      f"{c['paired_win_rate']:.4f}"
                      + (f" [{c['paired_ci95'][0]:.2f},{c['paired_ci95'][1]:.2f}]"
                         if c["paired_ci95"] else ""))
            lines.append(
                f"  {c['arm']:<8} @{c['budget']:<5g} wr={c['win_rate']:.4f} "
                f"[{c['ci95'][0]:.2f},{c['ci95'][1]:.2f}] n={c['decisive']} ties={c['ties']} "
                f"— {verdict}")
            lines.append(
                f"           paired={paired} over {c['n_pairs']} swap-pairs "
                f"({c['unpaired_games']} unpaired) | changed={c['change_rate']} "
                f"deepened={c['deepen_rate']} width={c['realized_mean']['opp_candidates']}"
                f"/{c['realized_mean']['worlds']}/{c['realized_mean']['dice']} "
                f"depth={c['realized_mean']['depth']} "
                # 0, not None: a beam of zero is the honest statement "no ply was ever afforded",
                # whereas a bare None reads like a missing measurement.
                f"beam={c['realized_mean']['beam'] or 0}")
        for note in mirror.get("notes", []):
            lines.append(f"  ! {note}")

    elo = elo_by_arm(rows, anchors_path)
    lines.append("")
    lines.append("anchored ELO by arm (delta vs base is the reportable quantity):")
    if not elo.get("cells"):
        lines.append(f"  (skipped: {elo.get('note')})")
    for c in elo.get("cells", []):
        lines.append(f"  {c['arm']:<8} @{c['budget']:<5g} elo={c['elo']} "
                     f"se={c['se']} delta_vs_base={c['delta_vs_base']} (n={c['games']})")
    for note in elo.get("caveats", []):
        lines.append(f"  ! {note}")
    if mirror.get("cells"):
        lines.append("  ! MIRROR cells are excluded from this fit — `self` carries no anchor, so "
                     "it would enter as a free parameter. Read them in the MIRROR block above.")
    return "\n".join(lines)
