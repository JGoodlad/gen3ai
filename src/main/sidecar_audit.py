"""SIDECAR AUDIT — does every checkpoint agree with its run about which commit ran?

    python -m main.sidecar_audit models/
    python -m main.sidecar_audit models/run_20260901_120000 models/ai_v9_171
    python -m main.sidecar_audit models/ --json
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)

WHY IT EXISTS. Until 2026-09-05 ``agents.model.snapshot.record_checkpoint`` resolved a
checkpoint's ``git_hash`` as ``git rev-parse HEAD`` **in the process CWD**, and the launcher
spawns its training child with no ``cwd=`` — so a run pinned to one commit stamped every
sidecar with whatever HEAD the un-pinned main checkout happened to be at. That is now fixed at
the root (``utils.git.get_git_hash`` is anchored at the IMPORTED checkout, and one resolver
serves the run metadata and every sidecar), but the historical sidecars on disk still carry
whatever they carried. This tool sizes that: how many, in which runs, by how much.

It reads **JSON only** — no torch, no checkpoint loads — so it works on every archived run
regardless of architecture drift, and it never opens a ``.zip``.

WHAT A "MISMATCH" MEANS, and why it is not always a defect. The run-level scalar ``git_hash``
is REWRITTEN on every save, so on a run that restarts every 3 h it names the last code to
touch the run, not the code that ran most of it. A sidecar written under an earlier commit is
therefore *expected* to differ — which is why ``metadata.json`` now also carries an
append-only ``pin_history``. A mismatch whose hash appears in that history is EXPLAINED; one
that appears nowhere is unexplained and is the shape the 2026-09-05 defect leaves behind.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------------------

def _load_json(path: str) -> Optional[dict]:
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def is_run_dir(path: str) -> bool:
    """A run dir is one that has a run-level ``metadata.json``."""
    return os.path.isfile(os.path.join(path, "metadata.json"))


def discover_runs(paths: List[str]) -> List[str]:
    """Expand each argument into run dirs: a run dir is itself, anything else is scanned.

    Order is preserved and duplicates dropped, so ``models/ models/run_x`` lists run_x once.
    """
    out: List[str] = []
    seen = set()

    def _add(p: str) -> None:
        rp = os.path.realpath(p)
        if rp not in seen:
            seen.add(rp)
            out.append(p)

    for p in paths:
        if not os.path.isdir(p):
            print(f"⚠️  not a directory, skipped: {p}", file=sys.stderr)
            continue
        if is_run_dir(p):
            _add(p)
            continue
        for name in sorted(os.listdir(p)):
            child = os.path.join(p, name)
            if os.path.isdir(child) and is_run_dir(child):
                _add(child)
    return out


def _sidecar_paths(run_dir: str) -> List[str]:
    """Every per-checkpoint sidecar in a run: ``<x>.json`` sitting beside an ``<x>.zip``.

    Keyed off the ZIP, deliberately — that is what makes a file a *checkpoint sidecar* rather
    than one of the run-level JSONs (``metadata.json`` / ``model_config.json`` / manifests),
    and it needs no filename allow-list that a new checkpoint prefix could fall out of.
    Covers the current ``checkpoints/`` layout, the legacy run-root one, ``best_model/`` and
    ``snapshots/``.
    """
    found: List[str] = []
    for root, dirs, files in os.walk(run_dir):
        dirs[:] = [d for d in dirs if d not in ("tb", "eval_traces", "crashes")]
        for fn in files:
            if not fn.endswith(".zip"):
                continue
            side = os.path.join(root, fn[:-4] + ".json")
            if os.path.isfile(side):
                found.append(side)
    return sorted(found)


# --------------------------------------------------------------------------------------
# Auditing
# --------------------------------------------------------------------------------------

def audit_run(run_dir: str) -> Dict[str, Any]:
    """The per-run record: the run-level pin, its history, and every sidecar's hash."""
    meta = _load_json(os.path.join(run_dir, "metadata.json")) or {}
    run_hash = meta.get("git_hash")
    pin_history = meta.get("pin_history")
    if not isinstance(pin_history, list):
        pin_history = []
    known = {str(e.get("git_hash")) for e in pin_history if isinstance(e, dict)}
    if run_hash:
        known.add(str(run_hash))

    sidecars: List[Dict[str, Any]] = []
    for path in _sidecar_paths(run_dir):
        entry = _load_json(path) or {}
        h = entry.get("git_hash")
        matches = (h == run_hash)
        sidecars.append({
            "path": os.path.relpath(path, run_dir),
            "git_hash": h,
            "matches_run": matches,
            # A hash the run's own history knows about is a RESTART span, not a misattribution.
            "explained_by_history": bool(h) and str(h) in known,
        })

    mismatched = [s for s in sidecars if not s["matches_run"]]
    return {
        "run_dir": run_dir,
        "run_name": os.path.basename(os.path.normpath(run_dir)),
        "git_hash": run_hash,
        "pin_source": meta.get("pin_source"),
        "pin_history": pin_history,
        "pin_split": len(pin_history) > 1,
        "sidecars": sidecars,
        "n_sidecars": len(sidecars),
        "n_mismatched": len(mismatched),
        "n_unexplained": sum(1 for s in mismatched if not s["explained_by_history"]),
    }


def _short(h: Optional[str]) -> str:
    return (str(h)[:8] if h else "—")


def print_run(rec: Dict[str, Any], *, verbose: bool) -> None:
    print(f"\n== {rec['run_name']} ==")
    src = f"  (pin_source={rec['pin_source']})" if rec.get("pin_source") else ""
    print(f"   run-level git_hash : {_short(rec['git_hash'])}{src}")

    hist = rec["pin_history"]
    if not hist:
        print("   pin_history        : (absent — pre-2026-09-05 run; the scalar above is all "
              "this run records, and it is the LAST commit to touch it, not the only one)")
    else:
        tag = "  ⚠️  PIN-SPLIT" if rec["pin_split"] else ""
        print(f"   pin_history        : {len(hist)} span(s){tag}")
        for i, e in enumerate(hist, 1):
            if not isinstance(e, dict):
                continue
            derived = "  (derived)" if e.get("derived") else ""
            psrc = f" pin_source={e.get('pin_source')}" if e.get("pin_source") else ""
            print(f"       [{i}] {_short(e.get('git_hash'))}{psrc}  "
                  f"steps {e.get('first_step')} → {e.get('last_step')}{derived}")

    n, m, u = rec["n_sidecars"], rec["n_mismatched"], rec["n_unexplained"]
    flag = "" if m == 0 else f", {m} differ from the run-level hash ({u} UNEXPLAINED)"
    print(f"   sidecars           : {n}{flag}")
    for s in rec["sidecars"]:
        if s["matches_run"] and not verbose:
            continue
        mark = "  " if s["matches_run"] else ("~ " if s["explained_by_history"] else "⚠️ ")
        note = ""
        if not s["matches_run"]:
            note = ("  (a span in pin_history)" if s["explained_by_history"]
                    else "  (in NO recorded span — misattributed)")
        print(f"       {mark}{s['path']}: {_short(s['git_hash'])}{note}")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m main.sidecar_audit",
        description="Compare every checkpoint sidecar's git_hash against its run's recorded "
                    "pin and pin_history. JSON only — no torch, no checkpoint loads.",
    )
    ap.add_argument("paths", nargs="+",
                    help="A run dir, or a models/ dir whose children are run dirs.")
    ap.add_argument("--json", action="store_true", dest="as_json",
                    help="Emit the full record as JSON instead of the human report.")
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="List every sidecar, not only the differing ones.")
    ap.add_argument("--strict", action="store_true",
                    help="Exit 1 when any sidecar's hash is in NO recorded span.")
    args = ap.parse_args(argv)

    runs = discover_runs(args.paths)
    if not runs:
        print("no run dirs found (a run dir is one containing metadata.json)", file=sys.stderr)
        return 2

    records = [audit_run(r) for r in runs]

    if args.as_json:
        json.dump({"runs": records}, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        for rec in records:
            print_run(rec, verbose=args.verbose)
        n_side = sum(r["n_sidecars"] for r in records)
        n_mis = sum(r["n_mismatched"] for r in records)
        n_unexp = sum(r["n_unexplained"] for r in records)
        runs_mis = sum(1 for r in records if r["n_mismatched"])
        runs_split = sum(1 for r in records if r["pin_split"])
        runs_nohist = sum(1 for r in records if not r["pin_history"])
        print(f"\nSUMMARY: {len(records)} run(s) · {n_side} sidecar(s) · "
              f"{n_mis} differing from their run hash in {runs_mis} run(s) "
              f"({n_unexp} UNEXPLAINED by any recorded span) · "
              f"{runs_split} run(s) PIN-SPLIT · {runs_nohist} run(s) with no pin_history")

    if args.strict and any(r["n_unexplained"] for r in records):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
