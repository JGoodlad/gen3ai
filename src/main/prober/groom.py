"""Groom a run's eval data so it doesn't grow without bound.

Training writes a forensic trace pair (``*_summary.json`` + ``*_states.npz``) per
sampled eval battle plus, with ``--keep-eval-snapshots``, a ~27MB weight snapshot
per cycle — under ``<run_dir>/eval_traces/step_<N>/``. Over a long run this piles
up. This groomer enforces retention, **scoped strictly to ``eval_traces/``**:

- keep full traces for the **K most-recent eval steps**; delete older step dirs;
- among the steps kept, keep ``snapshot.zip`` only for the **N most-recent**
  (drop the older snapshots but keep their traces).

**Dry-run by default** — it reports what it *would* delete and the bytes
reclaimed. Pass ``--apply`` to actually delete. Designed to be run out-of-band
(manually, on a ``/loop``/`/schedule`, or from cron):

    python -m main.prober.groom <run_dir> [--keep-trace-steps 10] [--keep-snapshots 5] [--apply]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil

_SNAPSHOT_NAME = "snapshot.zip"
_STEP_RE = re.compile(r"^step_(\d+)$")


def _dir_size(path: str) -> int:
    total = 0
    for root, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return total


def _eval_root(path: str) -> str:
    if os.path.basename(os.path.normpath(path)) == "eval_traces":
        return path
    return os.path.join(path, "eval_traces")


def _step_dirs(eval_root: str) -> "list[tuple[int, str]]":
    out = []
    if not os.path.isdir(eval_root):
        return out
    for name in os.listdir(eval_root):
        m = _STEP_RE.match(name)
        full = os.path.join(eval_root, name)
        if m and os.path.isdir(full):
            out.append((int(m.group(1)), full))
    return sorted(out, key=lambda sp: sp[0], reverse=True)   # most-recent first


def groom_run(run_dir: str, keep_trace_steps: int = 10, keep_snapshots: int = 5,
              apply: bool = False) -> dict:
    """Plan (and optionally apply) retention on a run's eval_traces. Returns a report."""
    eval_root = _eval_root(run_dir)
    steps = _step_dirs(eval_root)

    plan = []
    reclaim = 0
    for i, (step, path) in enumerate(steps):
        if i >= keep_trace_steps:
            size = _dir_size(path)
            reclaim += size
            plan.append({"step": step, "action": "remove_step", "bytes": size})
        else:
            snap = os.path.join(path, _SNAPSHOT_NAME)
            if i >= keep_snapshots and os.path.exists(snap):
                size = os.path.getsize(snap)
                reclaim += size
                plan.append({"step": step, "action": "drop_snapshot", "bytes": size})
            else:
                plan.append({"step": step, "action": "keep", "bytes": 0})

    if apply:
        for entry, (_step, path) in zip(plan, steps):
            if entry["action"] == "remove_step":
                shutil.rmtree(path, ignore_errors=True)
            elif entry["action"] == "drop_snapshot":
                try:
                    os.remove(os.path.join(path, _SNAPSHOT_NAME))
                except OSError:
                    pass

    return {
        "run_dir": os.path.abspath(run_dir),
        "eval_root": eval_root,
        "applied": apply,
        "keep_trace_steps": keep_trace_steps,
        "keep_snapshots": keep_snapshots,
        "n_steps": len(steps),
        "removed_steps": [e["step"] for e in plan if e["action"] == "remove_step"],
        "dropped_snapshots": [e["step"] for e in plan if e["action"] == "drop_snapshot"],
        "bytes_reclaimed": reclaim,
        "mb_reclaimed": round(reclaim / 1e6, 1),
        "plan": plan,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        prog="python -m main.prober.groom",
        description="Prune a run's eval_traces to a retention window (dry-run by default).",
    )
    p.add_argument("run_dir", help="a run dir (or an eval_traces dir)")
    p.add_argument("--keep-trace-steps", type=int, default=10,
                   help="keep full traces for the N most-recent eval steps (default 10)")
    p.add_argument("--keep-snapshots", type=int, default=10,
                   help="keep snapshot.zip for the N most-recent steps (default 10, matching training)")
    p.add_argument("--apply", action="store_true",
                   help="actually delete (default: dry-run — only report)")
    args = p.parse_args()
    print(json.dumps(groom_run(args.run_dir, args.keep_trace_steps,
                               args.keep_snapshots, args.apply), indent=2))


if __name__ == "__main__":
    main()
