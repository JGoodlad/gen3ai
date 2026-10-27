"""SEARCH COST: S-U's stage-0 report over the searched games a read (or a plain anchor run) played
(``designs/endstate/design_search_u_headroom.md`` §4.4, S-1 / S-6).

    python -m main.reads search-cost --state DIR              # every searched unit of a read's state, + G-PREFIX
    python -m main.reads search-cost --cells DIR [DIR ...]    # main.anchors --out dirs (a smoke)

It reads each searched game's per-decision log (``search_decisions.jsonl``, :mod:`main.anchors.search_side`) and its
``summary.json``, and reports, per game and pooled:

* the per-decision cost of a SEARCHED decision: CPU-s (the worker thread + the replay children) and wall-s, as median,
  mean and the 10th / 90th percentiles;
* the share of our decisions from the search's first turn that have >= 2 legal actions (searched or excluded: the
  decisions the search is asked to make; S-6);
* per game: the searched decisions, the exclusions (no consistent world), the deviation rate (the pick is not π's
  greedy action), the search CPU-min (the decisions' own), the WHOLE unit's CPU-min (``kakuna``'s ``units.jsonl``,
  the conservative figure: model load, peer, every thread and child) and the game's wall-min;
* the registered re-size input: the MEDIAN searched game's search CPU-min against the design's 2 × central (50 CPU-min;
  §4.4: above it, phase 2 is dropped). It is REPORTED here, decided by the read's rule.

Cost is never an outcome: nothing here reads a result.
"""
from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

REPORT_SCHEMA = "gen3_search_cost_v1"
#: §4.4's re-size threshold: a median searched game above 2 × the central 25 CPU-min.
RESIZE_CPU_MIN = 50.0
LOG = "search_decisions.jsonl"


def _q(xs: List[float], p: float) -> Optional[float]:
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return round(s[lo] + (s[hi] - s[lo]) * (k - lo), 3)


def _dist(xs: List[float]) -> Dict[str, Any]:
    return {"n": len(xs), "median": _q(xs, 0.5), "mean": round(statistics.fmean(xs), 3) if xs else None,
            "p10": _q(xs, 0.1), "p90": _q(xs, 0.9), "max": round(max(xs), 3) if xs else None}


def unit_cpu(kakuna_state: Path, unit: str) -> Optional[float]:
    """The unit's whole-process CPU-s as its last ``end`` event in ``units.jsonl`` recorded it (``kakuna._wait_cpu``:
    the model load, the peer, the search and its children), or None."""
    try:
        lines = (kakuna_state / "units.jsonl").read_text().splitlines()
    except OSError:
        return None
    got = None
    for ln in lines:
        try:
            ev = json.loads(ln)
        except ValueError:
            continue
        if ev.get("event") == "end" and ev.get("unit") == unit and ev.get("cpu_s") is not None:
            got = float(ev["cpu_s"])
    return got


def game_report(cell: Path) -> Optional[Dict[str, Any]]:
    """One searched game's row, or None when the cell holds no decision log (a raw unit)."""
    p = cell / LOG
    if not p.exists():
        return None
    recs = [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]
    try:
        summary = json.loads((cell / "summary.json").read_text())
    except (OSError, ValueError):
        summary = {}
    search = {r.get("search") for r in recs}
    from_turn = None
    if recs:
        from main.anchors.search_presets import parse

        from_turn = parse(sorted(x for x in search if x)[0])[1] if any(search) else None
    late = [r for r in recs if from_turn is not None and int(r["turn"]) >= from_turn]
    searched = [r for r in recs if r.get("searched")]
    excluded = [r for r in recs if r.get("why") == "excluded"]
    asked = [r for r in late if r.get("n_legal", 0) >= 2]
    cpu = sum(float(r.get("cpu_s") or 0.0) for r in searched + excluded)
    peer = ((summary.get("provenance") or {}).get("peer_report") or {})
    whole_cpu_s = unit_cpu(cell.parent.parent, cell.name)
    return {"cell": str(cell), "search": sorted(x for x in search if x), "decisions": len(recs),
            "from_turn": from_turn, "from_turn_decisions": len(late), "asked": len(asked),
            "ge2_legal_share": round(len(asked) / len(late), 4) if late else None,
            "searched": len(searched), "excluded": len(excluded),
            "deviations": sum(1 for r in searched if r.get("deviation")),
            "deviation_rate": round(sum(1 for r in searched if r.get("deviation")) / len(searched), 4) if searched
            else None,
            "search_cpu_min": round(cpu / 60.0, 3), "game_wall_min": (round(float(peer["wall_s"]) / 60.0, 3)
                                                                      if peer.get("wall_s") is not None else None),
            "unit_cpu_min": None if whole_cpu_s is None else round(whole_cpu_s / 60.0, 3),
            "turns": [g.get("turns") for g in _games(cell)], "status": summary.get("status"),
            "decision_cpu_s": [float(r["cpu_s"]) for r in searched if r.get("cpu_s") is not None],
            "decision_wall_s": [float(r["wall_s"]) for r in searched if r.get("wall_s") is not None]}


def _games(cell: Path) -> List[Dict[str, Any]]:
    try:
        return [json.loads(ln) for ln in (cell / "games.jsonl").read_text().splitlines() if ln.strip()]
    except (OSError, ValueError):
        return []


def report(cells: Iterable[Path], *, prefix: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The pooled report over ``cells`` (module docstring); ``prefix``: G-PREFIX's per-group result, carried."""
    games = [g for g in (game_report(Path(c)) for c in cells) if g is not None]
    cpu = [x for g in games for x in g["decision_cpu_s"]]
    wall = [x for g in games for x in g["decision_wall_s"]]
    late = sum(g["from_turn_decisions"] for g in games)
    asked = sum(g["asked"] for g in games)
    searched = sum(g["searched"] for g in games)
    game_cpu = [g["search_cpu_min"] for g in games if g["status"] == "OK"]
    med = _q(game_cpu, 0.5)
    rows = [{k: v for k, v in g.items() if k not in ("decision_cpu_s", "decision_wall_s")} for g in games]
    return {"schema": REPORT_SCHEMA, "games": len(games), "per_game": rows,
            "searched_decision_cpu_s": _dist(cpu), "searched_decision_wall_s": _dist(wall),
            "ge2_legal_share": round(asked / late, 4) if late else None, "from_turn_decisions": late,
            "searched": searched, "excluded": sum(g["excluded"] for g in games),
            "deviation_rate": (round(sum(g["deviations"] for g in games) / searched, 4) if searched else None),
            "game_search_cpu_min": _dist(game_cpu),
            "game_wall_min": _dist([g["game_wall_min"] for g in games if g["game_wall_min"] is not None]),
            # the WHOLE unit's CPU (kakuna's units.jsonl): the per-decision cpu_s leaves out the validator's persistent
            # Node child and every other thread, so this is the conservative figure for a searched game
            "unit_cpu_min": _dist([g["unit_cpu_min"] for g in games if g["unit_cpu_min"] is not None]),
            "resize": {"median_game_cpu_min": med, "threshold_cpu_min": RESIZE_CPU_MIN,
                       "above": (None if med is None else med > RESIZE_CPU_MIN),
                       "note": "REPORTED: design §4.4 drops phase 2 when the median searched game costs more"},
            **({"g_prefix": prefix} if prefix is not None else {})}


def state_report(state: Path) -> Dict[str, Any]:
    """Every searched unit of a read's state, with G-PREFIX per searched group."""
    from main.reads import layout as LY
    from main.reads import prefix as PX
    from main.reads.runner import load_state

    spec = load_state(state)
    cells: List[Path] = []
    pre: Dict[str, Any] = {}
    for g, a in spec.anchor.items():
        searched = [lab for lab in a.models if "search" in spec.models[lab].load]
        for lab in searched:
            d = LY.anchor_state(state, g, lab) / "cells"
            if d.is_dir():
                cells += sorted(p for p in d.iterdir() if p.is_dir() and ".partial." not in p.name)
        if searched and a.per_game:
            pre[g] = PX.check_group(spec, state, g)
    return report(cells, prefix=pre)
