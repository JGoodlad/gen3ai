"""Bound a run's append-only debug-artifact dirs (``stalls/`` + ``crashes/``).

Two debug dirs grow without bound under a run dir over a multi-day session:

- ``<run_dir>/stalls/stall_*.html`` — a full Showdown replay saved the first time a
  battle hits the 250-turn stall cap (``StallLogger``). ~80 KB each, and a self-play
  run generates thousands (3000+ over 50M steps) — the dominant offender.
- ``<run_dir>/crashes/restart_err_*.txt`` — a per-crash diagnostic the launcher writes
  before auto-restarting (``launcher/run.py``). Tiny, but unbounded across a run's
  restarts.

Neither is worth keeping beyond the most-recent handful for debugging, so this keeps
the **N most-recent of each** (by mtime — which matches the timestamp embedded in every
filename) and deletes the rest. **Scoped strictly to those two dirs and their own
filename prefixes** — checkpoints, ``eval_traces/``, ``best_model``, ``metadata.json``,
``snapshots/`` are never matched, never touched.

This is the same *producer-grooms-its-own-data* contract as the eval-trace retention
(``eval_callback.prune_eval_traces``): the trainer calls :func:`prune_run_artifacts`
every eval cycle (grouped next to ``prune_eval_traces`` in both eval callbacks), so a
live run self-bounds with no external task. The CLI is the manual fallback for a
finished run, a different retention, or a one-off sweep of every run under ``models/``
(**dry-run by default** — mirrors ``python -m main.prober.groom``):

    python -m agents.training.artifact_retention <run_dir | models_dir> \
        [--keep-stalls 50] [--keep-crashes 10] [--apply]
"""

from __future__ import annotations

import argparse
import json
import os

# Defaults — the retention the trainer applies every cycle. 0 = keep all (the
# `prune_eval_traces` convention), so a run can disable either prune independently.
KEEP_STALLS_DEFAULT = 50
KEEP_CRASHES_DEFAULT = 10

# (subdir, filename-prefix, filename-suffix) for each artifact kind. The prefix+suffix
# guard means a stray file in the dir is left alone — we only ever delete what the
# producer wrote (StallLogger -> stall_*.html, launcher -> restart_err_*.txt).
_STALLS = ("stalls", "stall_", ".html")
_CRASHES = ("crashes", "restart_err_", ".txt")


def _victims(directory: str, prefix: str, suffix: str, keep_n: int) -> "tuple[list[str], int]":
    """All matching files in ``directory`` except the ``keep_n`` most-recent (by mtime).

    Returns ``(paths_to_delete, bytes)``. ``keep_n <= 0`` keeps everything (no victims).
    A missing/unreadable dir yields no victims — never raises.
    """
    if keep_n <= 0:
        return [], 0
    try:
        names = os.listdir(directory)
    except OSError:
        return [], 0
    files = []
    for name in names:
        if not (name.startswith(prefix) and name.endswith(suffix)):
            continue
        full = os.path.join(directory, name)
        try:
            st = os.stat(full)
        except OSError:
            continue
        if os.path.isfile(full):
            files.append((full, st.st_mtime, st.st_size))
    files.sort(key=lambda t: t[1], reverse=True)            # most-recent first
    victims = files[keep_n:]
    return [p for p, _m, _s in victims], sum(s for _p, _m, s in victims)


def prune_run_artifacts(model_dir: "str | None",
                        keep_stalls: int = KEEP_STALLS_DEFAULT,
                        keep_crashes: int = KEEP_CRASHES_DEFAULT,
                        *, apply: bool = True) -> dict:
    """Keep the N most-recent stalls/ + crashes/ files in one run dir; delete older.

    The single unit of work — called in-process every eval cycle by both eval callbacks,
    and per run dir by the CLI. Silent and never raises (mirrors ``prune_eval_traces``);
    deletes only when ``apply`` (``apply=False`` is the CLI dry-run). Returns a report.
    """
    report = {"run_dir": os.path.abspath(model_dir) if model_dir else None,
              "applied": apply, "bytes_reclaimed": 0}
    if not model_dir:
        return report
    total = 0
    for key, (subdir, prefix, suffix), keep in (
        ("stalls", _STALLS, keep_stalls),
        ("crashes", _CRASHES, keep_crashes),
    ):
        directory = os.path.join(model_dir, subdir)
        victims, nbytes = _victims(directory, prefix, suffix, keep)
        if apply:
            for path in victims:
                try:
                    os.remove(path)
                except OSError:
                    pass
        report[key] = {"dir": directory, "keep": keep,
                       "removed": len(victims), "bytes": nbytes}
        total += nbytes
    report["bytes_reclaimed"] = total
    report["mb_reclaimed"] = round(total / 1e6, 1)
    return report


def discover_run_dirs(root: str) -> "list[str]":
    """Run dirs at/under ``root`` — any dir containing a ``stalls/`` or ``crashes/`` child.

    Lets the CLI accept a single run dir (returns ``[root]``) OR the whole ``models/``
    tree (returns every run, incl. ``_goldens/<run>``) with one walk.
    """
    found = set()
    for dirpath, dirnames, _files in os.walk(root):
        if _STALLS[0] in dirnames or _CRASHES[0] in dirnames:
            found.add(dirpath)
    return sorted(found)


def main() -> None:
    p = argparse.ArgumentParser(
        prog="python -m agents.training.artifact_retention",
        description="Prune a run's stalls/ + crashes/ to a retention window (dry-run by "
                    "default). Accepts a single run dir or a models/ tree to sweep all runs.",
    )
    p.add_argument("path", help="a run dir, or a models/ dir to sweep every run under it")
    p.add_argument("--keep-stalls", type=int, default=KEEP_STALLS_DEFAULT,
                   help=f"keep the N most-recent stall_*.html (default {KEEP_STALLS_DEFAULT}; 0 = keep all)")
    p.add_argument("--keep-crashes", type=int, default=KEEP_CRASHES_DEFAULT,
                   help=f"keep the N most-recent restart_err_*.txt (default {KEEP_CRASHES_DEFAULT}; 0 = keep all)")
    p.add_argument("--apply", action="store_true",
                   help="actually delete (default: dry-run — only report)")
    args = p.parse_args()

    runs = discover_run_dirs(args.path)
    reports = [prune_run_artifacts(d, args.keep_stalls, args.keep_crashes, apply=args.apply)
               for d in runs]
    total = sum(r["bytes_reclaimed"] for r in reports)
    print(json.dumps({
        "applied": args.apply,
        "n_runs": len(reports),
        "total_bytes_reclaimed": total,
        "total_mb_reclaimed": round(total / 1e6, 1),
        "runs": reports,
    }, indent=2))


if __name__ == "__main__":
    main()
