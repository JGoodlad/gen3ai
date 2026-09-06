"""WHO FORKED WHOM — a run's ancestry tree, offline, model-free, no torch.

```bash
export PYTHONPATH=$PYTHONPATH:src
python -m main.lineage <run> [<run2> ...]           # the tree, plus any broken links
python -m main.lineage <run> --json                 # the same, for scripts
python -m main.lineage <run> --backfill             # DRY-RUN a legacy run's derived block
python -m main.lineage <run> --backfill --apply     # actually write it
```
(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)

WHY. Every exploiter, fold, funding fork and dose arm is a fork of some parent, and every
comparison the ledger makes is a claim about that graph. Runs from `gen3_run_lineage_v1` on record
it in `metadata.json`'s immutable `lineage` block; every run before that implies it in a recorded
shell command. `agents.training.lineage` reads both through one accessor and this is its CLI.

WHAT IT CHECKS, per link, from what is on disk:
  * the parent RUN DIRECTORY is gone (a groomed/renamed/deleted run);
  * the parent CHECKPOINT's sha256 no longer matches what was recorded (it was replaced);
  * the `arch_signature` CHANGED across the link — a fork cannot have loaded a differently-shaped
    parent, so the recorded parent is wrong.

A DERIVED chain is labelled `⚠ derived` on every line it applies to. Torch is never imported and no
checkpoint is loaded, so this reads a run whose architecture drifted past current code — which is
most of `models/`.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

from agents.training.lineage import (
    ancestry_from_parent, build_lineage_from_command, check_links, fork_parent, read_block,
    read_num_timesteps, read_original_command, role_of,
)


def _resolve(name: str) -> str:
    """A run dir, or a bare run NAME resolved against the main checkout's `models/`.

    `models/` is not committed and exists only in the MAIN checkout, so a worktree must reach
    across (`utils.paths.main_models_dir`). A name that resolves nowhere is returned unchanged and
    reported as missing — a resolver that guessed would be worse than one that says so."""
    if os.path.isdir(name):
        return name
    from utils.paths import main_models_dir
    root = main_models_dir()
    if root is not None:
        cand = os.path.join(str(root), name.replace("models/", "", 1) if name.startswith("models/")
                            else name)
        if os.path.isdir(cand):
            return cand
    return name


def read_run(run_dir: str) -> Dict[str, Any]:
    """Everything the CLI knows about one run. Pure over the filesystem; unit-tested directly."""
    name = os.path.basename(os.path.normpath(run_dir))
    out: Dict[str, Any] = {
        "run": name, "dir": run_dir, "recorded": False, "derived": False, "role": None,
        "fork_step": None, "num_timesteps": None, "fork_parent": None, "teachers": [],
        "exploiter_target": None,
        "ancestry": [], "ancestry_stop": None, "checks": [], "error": None,
    }
    if not os.path.isdir(run_dir):
        out["error"] = "no such run directory"
        return out
    # HOW FAR THIS RUN TRAINED. A run that predates the key reads None => "unknown" — this is a
    # JSON-only tool and will not open a checkpoint zip to guess.
    out["num_timesteps"] = read_num_timesteps(run_dir)
    block = read_block(run_dir)
    out["recorded"] = block is not None
    # warn=True: this is THE accessor's legacy path, and its whole point is that a derived answer
    # announces itself. It goes to stderr, so `--json` stdout stays machine-readable.
    parent = fork_parent(run_dir, warn=True)
    out["derived"] = bool(parent is not None and parent.derived) or (
        block is None and read_original_command(run_dir) is not None)
    out["role"] = role_of(run_dir, warn=False)
    if block is not None:
        out["fork_step"] = block.get("fork_step")
        out["teachers"] = list(block.get("teachers") or [])
        out["exploiter_target"] = block.get("exploiter_target")
    else:
        # LEGACY: teachers/target are only in the recorded command. Derive them the same way the
        # block would have, without hashing anything.
        cmd = read_original_command(run_dir)
        if cmd:
            derived = build_lineage_from_command(cmd, model_dir=run_dir, hash_parent=False)
            if derived:
                out["teachers"] = list(derived.get("teachers") or [])
                out["exploiter_target"] = derived.get("exploiter_target")
    if parent is not None:
        out["fork_parent"] = parent.to_dict()
        chain, stop = ancestry_from_parent(parent)
        out["ancestry"], out["ancestry_stop"] = chain, stop
    out["checks"] = check_links(run_dir)
    return out


def _short(h: Optional[str], n: int = 8) -> str:
    return (h[:n] if h else "—")


def _steps(value: Optional[int]) -> str:
    """A step count, thousands-separated — or `unknown` when the run never recorded one."""
    return f"{value:,}" if isinstance(value, int) else "unknown"


def _names(entries: List[Dict[str, Any]]) -> str:
    return ", ".join((e.get("run_name") or e.get("path") or "?") for e in entries) or "—"


def render(row: Dict[str, Any]) -> str:
    """The tree for one run, plus its teachers/target and any broken links."""
    if row.get("error"):
        return f"{row['run']}: {row['error']}"
    lines = []
    tag = "recorded" if row["recorded"] else ("⚠ DERIVED from original_command" if row["derived"]
                                              else "no lineage recorded")
    lines.append(f"{row['run']}   role={row.get('role') or '—'}   [{tag}]")
    # The two step facts side by side: where this run STARTED (its fork point, from the immutable
    # lineage block) and how far it GOT (the latest `num_timesteps`). "unknown" is a real answer —
    # a legacy run recorded neither, and 0 would be a claim.
    fs, ns = row.get("fork_step"), row.get("num_timesteps")
    if fs is not None or ns is not None:
        lines.append(f"    steps: fork_step={_steps(fs)}   num_timesteps={_steps(ns)}")
    if row["teachers"]:
        lines.append(f"    teachers ({len(row['teachers'])}): {_names(row['teachers'])}")
    if row["exploiter_target"]:
        lines.append(f"    exploiter target: {_names([row['exploiter_target']])}")
    parent = row.get("fork_parent")
    if not parent:
        lines.append("    └─ (root — no fork parent)")
    for depth, node in enumerate(row["ancestry"]):
        pad = "    " + "   " * depth
        step = node.get("fork_step")
        src = node.get("source")
        mark = " ⚠ derived" if src == "original_command" else ""
        lines.append(f"{pad}└─ {node.get('run_name') or node.get('model_path') or '?'}   "
                     f"git={_short(node.get('git_hash'))}  arch={node.get('arch_signature') or '—'}  "
                     f"role={node.get('role') or '—'}  "
                     f"fork_step={step if step is not None else '—'}{mark}")
        if depth == 0 and parent:
            npad = pad + "   "
            steps = parent.get("num_timesteps")
            lines.append(f"{npad}   via {parent.get('path')}"
                         f"{f'  @{steps:,} steps' if isinstance(steps, int) else ''}"
                         f"  sha={_short(parent.get('sha256'), 12)}")
    stop = row.get("ancestry_stop")
    if stop:
        lines.append(f"    ✖ chain ends at {stop.get('at')}: {stop.get('reason')}")
    for problem in row["checks"]:
        lines.append(f"    ⚠ {problem}")
    return "\n".join(lines)


def backfill(run_dir: str, *, apply: bool = False) -> Dict[str, Any]:
    """Derive a `lineage` block for a LEGACY run and (optionally) write it into its metadata.json.

    DRY-RUN by default and REFUSES a run that already records one — the block is immutable, and a
    backfill that overwrote a recorded fork parent with a re-parsed guess would defeat the entire
    point of recording it."""
    out: Dict[str, Any] = {"run": os.path.basename(os.path.normpath(run_dir)), "dir": run_dir,
                           "action": None, "block": None, "written": False}
    if not os.path.isdir(run_dir):
        out["action"] = "SKIP — no such run directory"
        return out
    meta_path = os.path.join(run_dir, "metadata.json")
    if not os.path.exists(meta_path):
        out["action"] = "SKIP — no metadata.json"
        return out
    if read_block(run_dir) is not None:
        out["action"] = "SKIP — already records a lineage block (immutable)"
        return out
    cmd = read_original_command(run_dir)
    if not cmd:
        out["action"] = "SKIP — no original_command to derive from"
        return out
    block = build_lineage_from_command(cmd, model_dir=run_dir)
    if block is None:
        out["action"] = ("SKIP — original_command's --model is a checkpoint INSIDE this run "
                         "(a restart, not a fork)")
        return out
    out["block"] = block
    parent = block.get("fork_parent")
    who = (parent or {}).get("run_name") or "(fresh)"
    out["action"] = f"WOULD WRITE lineage: role={block['role']}, parent={who}"
    if apply:
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        meta["lineage"] = block
        tmp = meta_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        os.replace(tmp, meta_path)
        out["written"] = True
        out["action"] = f"WROTE lineage: role={block['role']}, parent={who}"
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m main.lineage", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run dirs (or bare run names under models/)")
    ap.add_argument("--json", action="store_true", help="emit the raw rows as JSON")
    ap.add_argument("--backfill", action="store_true",
                    help="derive a lineage block for a LEGACY run (DRY-RUN unless --apply)")
    ap.add_argument("--apply", action="store_true",
                    help="with --backfill: actually write the derived block into metadata.json")
    args = ap.parse_args(argv)

    dirs = [_resolve(r) for r in args.runs]
    if args.backfill:
        rows = [backfill(d, apply=args.apply) for d in dirs]
        if args.json:
            print(json.dumps({"backfill": rows}, indent=2))
            return 0
        if not args.apply:
            print("DRY RUN — nothing written. Re-run with --apply to write.\n")
        for row in rows:
            print(f"{row['run']}: {row['action']}")
            block = row.get("block")
            if block:
                print(json.dumps(block, indent=2))
        return 0

    rows = [read_run(d) for d in dirs]
    if args.json:
        print(json.dumps({"runs": rows}, indent=2))
        return 0
    for i, row in enumerate(rows):
        if i:
            print()
        print(render(row))
    return 1 if any(r["checks"] or r.get("error") for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
