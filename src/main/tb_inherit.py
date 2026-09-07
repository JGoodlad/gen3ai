"""`python -m main.tb_inherit` — inspect and BACKFILL a fork's inherited TensorBoard prefix.

The live path needs no CLI: a fork writes its parent's prefix at creation (see
`agents.training.tb_inherit`, wired in `main.train.model_build`). This tool is for the runs already
on disk, which forked before the feature existed and therefore start their charts mid-air.

    python -m main.tb_inherit --list                     # what a backfill WOULD touch
    python -m main.tb_inherit --backfill <run> [<run>…]  # DRY RUN by default
    python -m main.tb_inherit --backfill <run> --apply    # actually write
    python -m main.tb_inherit --backfill --all --apply    # every fork missing a prefix
    python -m main.tb_inherit --show <run>                # the recorded provenance

DRY RUN IS THE DEFAULT, and `--apply` is the only way past it: this writes into `models/`, which is
otherwise read-only in every agent's contract here, and a wrong parent attaches one run's history to
another. `--force` re-writes a run that already inherited (it replaces the prefix file rather than
appending a second copy).

TORCH-FREE, like `main.lineage` and `main.dose` — it reads `metadata.json` and event files, so it
works on every run in the archive regardless of architecture drift.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from agents.training import tb_inherit as TI


def _models_dir(explicit: Optional[str]) -> Optional[str]:
    if explicit:
        return explicit
    from utils.paths import main_models_dir
    d = main_models_dir()
    return str(d) if d else None


def _resolve_run(arg: str, models_dir: Optional[str]) -> Optional[str]:
    """A run given as a path, or as a bare name under `models_dir`."""
    if os.path.isdir(arg):
        return os.path.abspath(arg)
    if models_dir:
        cand = os.path.join(models_dir, arg)
        if os.path.isdir(cand):
            return os.path.abspath(cand)
    return None


def _cmd_list(models_dir: str, as_json: bool) -> int:
    rows = TI.forks_missing_prefix(models_dir)
    if as_json:
        print(json.dumps(rows, indent=2))
        return 0
    if not rows:
        print(f"No fork under {models_dir} is missing a TensorBoard prefix.")
        return 0
    ready = [r for r in rows if r["parent_tb_exists"]]
    derived = [r for r in rows if r.get("derived")]
    print(f"{len(rows)} fork(s) under {models_dir} have no inherited prefix "
          f"({len(ready)} whose parent still has a tb/ to copy):\n")
    for r in rows:
        mark = " " if r["parent_tb_exists"] else "!"
        print(f" {mark} {r['run']:<46} role={str(r['role']):<10} "
              f"fork_step={r['fork_step']:>13,}  <- {r['parent_run_name']}"
              + ("  [DERIVED parent]" if r.get("derived") else "")
              + ("" if r["parent_tb_exists"] else "   [parent has NO tb/ — cannot backfill]"))
    if derived:
        print(f"\n🚨 {len(derived)} of {len(rows)} name a DERIVED parent — the link was REGEXED out "
              f"of `original_command`, not recorded at fork time, and it can be wrong "
              f"(ai_v8_01_zarch_film_0717 claims role=fresh/fork_step=0 while its own tb opens at "
              f"step 148,401,356). Check the parent before --apply.")
    print("\nBackfill them with:")
    print("  python -m main.tb_inherit --backfill --all            # dry run")
    print("  python -m main.tb_inherit --backfill --all --apply    # write")
    return 0


def _cmd_show(runs: List[str]) -> int:
    rc = 0
    for run_dir in runs:
        prov = TI.read_provenance(run_dir)
        name = os.path.basename(os.path.normpath(run_dir))
        if prov is None:
            print(f"{name}: no inherited prefix (no {TI.PROVENANCE_BASENAME})")
            rc = 1
            continue
        print(f"{name}:")
        print(json.dumps(prov, indent=2))
    return rc


def _cmd_backfill(runs: List[str], *, apply: bool, force: bool) -> int:
    wrote = skipped = failed = 0
    for run_dir in runs:
        name = os.path.basename(os.path.normpath(run_dir))
        block = TI._read_lineage_block(run_dir)
        if not block:
            print(f"{name}: SKIP — no recorded lineage block (a legacy run; "
                  f"`python -m main.lineage --backfill` writes one first)")
            skipped += 1
            continue
        parent = block.get("fork_parent")
        if not isinstance(parent, dict) or not parent.get("run_dir"):
            print(f"{name}: SKIP — fresh run / no fork parent")
            skipped += 1
            continue
        try:
            res = TI.inherit_tb(run_dir, parent_run_dir=str(parent["run_dir"]),
                                fork_step=int(block.get("fork_step") or 0),
                                parent_run_name=parent.get("run_name"),
                                dry_run=not apply, force=force)
        except Exception as exc:  # noqa: BLE001
            print(f"{name}: FAILED — {type(exc).__name__}: {exc}")
            failed += 1
            continue
        verb = "WOULD WRITE" if (res.written and not apply) else ("WROTE" if res.written else "SKIP")
        print(f"{name}: {verb} — {res.describe().removeprefix('[tb-inherit] ')}")
        if res.written:
            wrote += 1
        else:
            skipped += 1
    print(f"\n{wrote} to write, {skipped} skipped, {failed} failed"
          + ("" if apply else "   (DRY RUN — pass --apply to write)"))
    return 1 if failed else 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m main.tb_inherit",
        description="Inspect / backfill a fork's inherited TensorBoard prefix (gen3_tb_inherit_v1).")
    p.add_argument("runs", nargs="*", help="Run dirs or bare run names under --models-dir")
    p.add_argument("--models-dir", default=None,
                   help="The run archive (default: utils.paths.main_models_dir(), i.e. the MAIN "
                        "checkout's models/ — a worktree has none of its own)")
    p.add_argument("--list", action="store_true", help="List every fork missing a prefix")
    p.add_argument("--show", action="store_true", help="Print the recorded provenance for each run")
    p.add_argument("--backfill", action="store_true", help="Write the prefix (DRY RUN unless --apply)")
    p.add_argument("--all", action="store_true", help="With --backfill: every fork missing a prefix")
    p.add_argument("--apply", action="store_true", help="Actually write. Without it, nothing changes")
    p.add_argument("--force", action="store_true", help="Re-write a run that already inherited")
    p.add_argument("--json", action="store_true", help="Machine-readable output for --list")
    args = p.parse_args(argv)

    models_dir = _models_dir(args.models_dir)

    if args.list:
        if not models_dir:
            print("No models/ archive found. Pass --models-dir (or set $GEN3AI_MODELS_DIR).",
                  file=sys.stderr)
            return 2
        return _cmd_list(models_dir, args.json)

    runs: List[str] = []
    if args.backfill and args.all:
        if not models_dir:
            print("--all needs an archive. Pass --models-dir (or set $GEN3AI_MODELS_DIR).",
                  file=sys.stderr)
            return 2
        runs = [r["run_dir"] for r in TI.forks_missing_prefix(models_dir) if r["parent_tb_exists"]]
        if not runs:
            print("Nothing to backfill.")
            return 0
    else:
        for a in args.runs:
            r = _resolve_run(a, models_dir)
            if r is None:
                print(f"Not a run directory: {a}", file=sys.stderr)
                return 2
            runs.append(r)

    if not runs:
        p.print_help()
        return 2
    if args.show:
        return _cmd_show(runs)
    if args.backfill:
        return _cmd_backfill(runs, apply=args.apply, force=args.force)
    return _cmd_show(runs)


if __name__ == "__main__":
    raise SystemExit(main())
