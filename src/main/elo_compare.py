"""``python -m main.elo compare RUN RUN [RUN ...]`` — the cross-run ELO read at matched snapshot COUNT.

The runbook's "Reporting an ELO has FOUR rules" (``designs/ops/training_runbook.md``) and "Reading an ELO"
rules 2-5 (``designs/training/eval_and_rating.md``) say how two runs' ratings may be set side by side. This is
the small CLI that applies them, so no one hand-reads ``ladder.json``'s n-th node (wrong on a groomed or forked
run). It PLAYS NOTHING and WRITES NOTHING (safe on a live run and on ``models/``).

    python -m main.elo compare RUN_A RUN_B [RUN_C ...] [--at-snapshots N] [--allow-transport-mix] [--json]

Each ``RUN`` is a run dir, ``models/<run>``, or a NAME from ``designs/baselines.json``. The FIRST run is the
reference: every other run's delta is quoted against it.

What it does, and what it REFUSES (exit 2, nothing concluded):

* the COUNT is over every OWN snapshot EVER rated: ``snapshot_ladder.own_rated_steps_ever`` (the append-only
  ``games.jsonl`` endpoints plus the committed nodes, above the lineage ``fork_step``), never the committed
  ``ladder.json``'s survivors of a groomed pool. ``--at-snapshots N`` defaults to the largest count every side
  has; a larger N is refused.
* a side with NO ``games.jsonl`` cannot be refit. It is read from its committed file only when that file's
  recipe stamp is current, it rates exactly N own snapshots, and its eval record names no snapshot the file
  has dropped (a groomed pool); else REFUSED.
* every other side is REFIT on its first N own-rated snapshots (``fit_ladder(first_n=N, steps=...)``, strict
  prefix) even when its node count already matches: a committed file was written by the code its run is pinned
  to. A prefix that rates fewer than N nodes on its own, or does not converge, is REFUSED (an UNMATCHED fit size
  is not a read).
* the sides must share one transport (``check_same_transport``); ``--allow-transport-mix`` reads them with the
  caveat printed.
* a cross-era comparison and an unfinished run are LABELLED, never silent (``cross_era_warning``).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

from agents.training import baselines
from agents.training import elo as elo_mod
from agents.training import snapshot_ladder as sl
from agents.training.lineage import fork_step_of
from utils.era import cross_era_warning, era_label, read_run_era

Node = Tuple[int, float, float]      # (step, elo, se)


class CompareRefusal(Exception):
    """A matched-count read that cannot honestly be made. ``str(e)`` names the side and the fix."""


def _label(run_dir: str) -> str:
    return os.path.basename(os.path.normpath(run_dir))


def _resolve(ref: str) -> str:
    """A baseline NAME, ``models/<run>`` typed in a worktree, or a path, to the run dir."""
    from utils.paths import resolve_archive_ref
    if baselines.is_name(ref):
        resolved = baselines.run_dir(ref)
        if resolved is None:
            raise CompareRefusal(f"baseline {ref!r} names a run, but there is no models/ archive in this checkout")
        ref = resolved
    path = resolve_archive_ref(ref)
    if not os.path.isdir(path):
        raise CompareRefusal(f"{ref!r} is not a run directory")
    return path


def _load_committed(run_dir: str) -> dict:
    path = sl.ladder_json_path(run_dir)
    if not os.path.exists(path):
        raise CompareRefusal(
            f"{_label(run_dir)}: no {path}. The endpoint is the DENSE ladder (+-10), not `eval/elo` (+-29); "
            f"there is no substitute. Build it with: python -m agents.training.snapshot_ladder {run_dir} --backfill")
    try:
        with open(path) as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as e:
        raise CompareRefusal(f"{_label(run_dir)}: {path} is unreadable as JSON: {e}") from e
    if not isinstance(doc, dict) or not isinstance(doc.get("ratings"), dict):
        raise CompareRefusal(f"{_label(run_dir)}: {path} has no `ratings` map; written by a different schema")
    return doc


def _nodes(doc: dict) -> List[Node]:
    se = doc.get("se") or {}
    return sorted((int(k), float(v), float(se.get(k, float("nan")))) for k, v in doc["ratings"].items())


def _sentinel_steps_ever(run_dir: str) -> List[int]:
    """Every pool snapshot an eval cycle ever drew as a sentinel: the pool's history when ``games.jsonl`` is gone."""
    path = os.path.join(run_dir, "eval_results.jsonl")
    out: set = set()
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                for s in (row.get("sentinels") or []) if isinstance(row, dict) else []:
                    if isinstance(s, dict) and isinstance(s.get("step"), (int, float)):
                        out.add(int(s["step"]))
    return sorted(out)


def _own_steps(run_dir: str, committed: List[Node]) -> List[int]:
    """The side's OWN snapshot steps EVER rated; refuses a log-less side whose pool was groomed."""
    steps = sl.own_rated_steps_ever(run_dir, [s for s, _, _ in committed])
    if os.path.exists(sl.games_log_path(run_dir)):
        return steps
    floor = fork_step_of(run_dir)
    floor = -1 if floor is None else floor
    lost = [s for s in _sentinel_steps_ever(run_dir) if s > floor and s not in set(steps)]
    if lost:
        raise CompareRefusal(
            f"{_label(run_dir)}: no snapshot_ladder/games.jsonl and its committed ladder.json no longer rates "
            f"{len(lost)} snapshot(s) its own eval cycles drew from the pool (first: {lost[:3]}). The pool was "
            "GROOMED and the committed file rates the SURVIVORS, so its 'first n' are not the run's first n "
            "(ledger 6c4ae35d). A matched-count read needs the append-only games.jsonl; nothing is concluded.")
    return steps


def _finished(run_dir: str) -> bool:
    return any(os.path.exists(os.path.join(run_dir, n)) for n in ("final_model.zip", "final_model_interrupted.zip"))


def _side_transport(run_dir: str, committed: dict) -> str:
    if os.path.exists(sl.games_log_path(run_dir)):
        try:
            return sl.fit_transport(run_dir, allow_mixed=True)
        except sl.LadderTransportError as e:       # allow_mixed cannot raise; belt and braces
            raise CompareRefusal(f"{_label(run_dir)}: {e}") from e
    return sl.ladder_transport(committed)


def _refit_first_n(run_dir: str, n: int, steps: Sequence[int], allow_mix: bool) -> List[Node]:
    games_path = sl.games_log_path(run_dir)
    try:
        doc = sl.fit_ladder(run_dir, first_n=n, write=False, steps=list(steps), allow_mixed_transport=allow_mix)
    except sl.LadderTransportError as e:
        raise CompareRefusal(f"{_label(run_dir)}: {e}") from e
    nodes = _nodes({**doc})
    if len(nodes) != n:
        raise CompareRefusal(
            f"{_label(run_dir)}: the first-{n} prefix of its ladder rates {len(nodes)} node(s) on its own, not {n}: "
            "the prefix has no measured edge inside itself (its early nodes are rated only through edges to LATER "
            "snapshots), so a matched-size fit does not exist for it. An UNMATCHED fit size is not a read; "
            f"nothing is concluded ({games_path}).")
    if not doc.get("converged"):
        raise CompareRefusal(f"{_label(run_dir)}: the first-{n} BT fit did not converge; nothing is concluded.")
    if not doc.get("anchored_to_bots"):
        raise CompareRefusal(f"{_label(run_dir)}: the first-{n} fit is not anchored to the bots "
                             "(no bot anchor file); a cross-run absolute comparison needs the anchor.")
    return nodes


def _committed_side(run_dir: str, committed: dict, nodes: List[Node], own: List[int], n: int) -> List[Node]:
    """The log-less fallback: the committed file's OWN nodes, only when they are exactly the first n and current."""
    status, detail = sl.recipe_status(committed)
    if status != "current":
        raise CompareRefusal(
            sl.recipe_refusal(sl.ladder_json_path(run_dir), status, detail, run_dir, what=_label(run_dir))
            + "\n(This side has no games.jsonl, so it cannot be refit here either.)")
    if len(own) != n:
        raise CompareRefusal(
            f"{_label(run_dir)}: no games.jsonl to refit from, and it rates {len(own)} own snapshots, not {n}; "
            f"its first-{n} fit does not exist. Rebuild the ladder with: python -m agents.training.snapshot_ladder "
            f"{run_dir} --backfill")
    if not committed.get("converged"):
        raise CompareRefusal(f"{_label(run_dir)}: the committed ladder did not converge; nothing is concluded.")
    if not committed.get("anchored_to_bots"):
        raise CompareRefusal(f"{_label(run_dir)}: the committed ladder is not anchored to the bots.")
    return [t for t in nodes if t[0] in set(own)]


def compare(run_refs: Sequence[str], at_snapshots: Optional[int] = None, *,
            allow_transport_mix: bool = False) -> Dict[str, Any]:
    """The matched-count read over ``run_refs`` (first = reference). Raises :class:`CompareRefusal`."""
    if len(run_refs) < 2:
        raise CompareRefusal("compare needs at least two runs")
    dirs = [_resolve(r) for r in run_refs]
    if len({os.path.realpath(d) for d in dirs}) != len(dirs):
        raise CompareRefusal("the same run was named twice; a run compared with itself is not a comparison")
    docs = {d: _load_committed(d) for d in dirs}
    transports = {_label(d): _side_transport(d, docs[d]) for d in dirs}
    try:
        notes = sl.check_same_transport({k: {"recipe": {"transport": t}} for k, t in transports.items()},
                                        allow=allow_transport_mix)
    except sl.LadderTransportError as e:
        raise CompareRefusal(f"ladder transport: {e}\n  (pass --allow-transport-mix to read it with the caveat)") from e
    own = {d: _own_steps(d, _nodes(docs[d])) for d in dirs}
    avail = min(len(v) for v in own.values())
    n = avail if at_snapshots is None else at_snapshots
    if n < 1 or n > avail:
        raise CompareRefusal(
            f"--at-snapshots {n}: " + ", ".join(f"{_label(d)} has rated {len(own[d])}" for d in dirs)
            + f" own snapshot(s) (every step ever rated, above any fork step), so the matched count is at most "
              f"{avail}. Compare at MATCHED COUNT, never at matched step.")
    sides = []
    for d in dirs:
        has_log = os.path.exists(sl.games_log_path(d))
        nodes = (_refit_first_n(d, n, own[d][:n], allow_transport_mix) if has_log
                 else _committed_side(d, docs[d], _nodes(docs[d]), own[d], n))
        committed_steps = {s for s, _, _ in _nodes(docs[d])}
        node = nodes[n - 1]
        sides.append({
            "run": _label(d), "run_dir": d, "transport": transports[_label(d)],
            "source": "refit from games.jsonl" if has_log else "committed ladder.json (no games.jsonl)",
            "era": era_label(read_run_era(d)), "finished": _finished(d), "fork_step": fork_step_of(d),
            "n_own_rated_ever": len(own[d]), "n_groomed_out": sum(1 for s in own[d] if s not in committed_steps),
            "nodes": [{"i": i + 1, "step": s, "elo": e, "se": v, "ci95": elo_mod.ci95(v)}
                      for i, (s, e, v) in enumerate(nodes)],
            "at_count": {"step": node[0], "elo": node[1], "se": node[2], "ci95": elo_mod.ci95(node[2])},
        })
    ref = sides[0]["at_count"]
    for s in sides[1:]:
        a = s["at_count"]
        se = math.sqrt(sum(x ** 2 for x in (a["se"], ref["se"]) if math.isfinite(x)))
        s["delta_vs_first"] = {"elo": a["elo"] - ref["elo"], "se": se, "ci95": elo_mod.ci95(se)}
    eras = [w for w in (cross_era_warning(dirs[0], d) for d in dirs[1:]) if w]
    return {"at_snapshots": n, "available_matched": avail, "reference": sides[0]["run"], "sides": sides,
            "transport_warning": notes, "era_warning": eras,
            "unfinished": [s["run"] for s in sides if not s["finished"]]}


def render(rep: Dict[str, Any]) -> str:
    n = rep["at_snapshots"]
    out = [f"MATCHED-COUNT ladder read at n={n} own-rated snapshots (max matched {rep['available_matched']}); "
           f"reference = {rep['reference']}"]
    for s in rep["sides"]:
        a = s["at_count"]
        out.append(f"\n{s['run']}   [{s['source']}]   [transport: {s['transport']}]   [era: {s['era']}]"
                   f"   [{'finished' if s['finished'] else 'UNFINISHED'}]")
        out.append(f"  rated ever (own): {s['n_own_rated_ever']}   groomed out of the committed file: "
                   f"{s['n_groomed_out']}   fork_step: {s['fork_step']}")
        for nd in s["nodes"]:
            out.append(f"    n={nd['i']:<3} {nd['step'] / 1e6:8.2f}M   ELO {nd['elo']:7.1f} +- {nd['ci95']:.1f}")
        out.append(f"  AT n={n}: {a['step'] / 1e6:.2f}M -> ELO {a['elo']:.1f} +- {a['ci95']:.1f}")
        d = s.get("delta_vs_first")
        if d:
            out.append(f"  DELTA vs {rep['reference']}: {d['elo']:+.1f} Elo (95% CI {d['elo'] - d['ci95']:+.1f} to "
                       f"{d['elo'] + d['ci95']:+.1f}, ci95 = {d['ci95']:.1f})")
    for w in rep["transport_warning"] + rep["era_warning"]:
        out.append(f"\n{w}")
    if rep["unfinished"]:
        out.append(f"\nUNFINISHED: {', '.join(rep['unfinished'])}. Both sides are refit on the same first n, so the "
                   "newest-node inflation cancels; quote this as PROVISIONAL and re-read at run end (rule 1).")
    return "\n".join(out)


def compare_main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m main.elo compare",
        description="Cross-run dense-ladder ELO at matched snapshot COUNT (reads only; plays and writes nothing).")
    ap.add_argument("runs", nargs="+", help="two or more run dirs / models/<run> / baseline names; the first is "
                                            "the reference")
    ap.add_argument("--at-snapshots", type=int, default=None, metavar="N",
                    help="the matched count (default: the largest count every run has)")
    ap.add_argument("--allow-transport-mix", action="store_true",
                    help="read ladders played on different transports, with the caveat printed")
    ap.add_argument("--json", action="store_true", help="print the report as JSON instead of the table")
    a = ap.parse_args(argv)
    try:
        rep = compare(a.runs, a.at_snapshots, allow_transport_mix=a.allow_transport_mix)
    except CompareRefusal as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2
    print(json.dumps(rep, indent=2) if a.json else render(rep))
    return 0
