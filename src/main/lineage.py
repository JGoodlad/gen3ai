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

RECORDED vs DERIVED are two INDEPENDENT facts and the header line states both. A run's lineage is
DERIVED when it was REGEXed out of `original_command` — either at read time (no block at all) or
once, by `--backfill`, into a block that says `"derived": true`. That second case is still a block
on disk, so it prints `recorded ⚠ DERIVED from original_command`; it is not a lesser kind of
recording, it is a recorded GUESS. A DERIVED ANCESTOR is a different fact again and is marked
`⚠ derived` on the node it applies to. More than one run also prints a summary line counting each.

Torch is never imported and no
checkpoint is loaded, so this reads a run whose architecture drifted past current code — which is
most of `models/`.

WHICH FILE, not just which run (`gen3_last_snapshot_resolution_v1`). A teacher / target / parent is
usually named as a run DIRECTORY, so every reference also prints the `.zip` the resolver actually
picked, its `num_timesteps`, and the rung + rule that picked it. A run recorded before that change
prints `resolved file not recorded (pre gen3_last_snapshot_resolution_v1)` — those runs went through
the OLD rule (`best_model/best_model.zip` first, which is selected on BOT win rate and for 2 of 8
R5F teachers was a ~0.93M-step export rather than the ~2.93M final), and re-resolving them today
would print a current answer as history.
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
        "run": name, "dir": run_dir, "recorded": False, "derived": False,
        "derived_self": False, "parent_derived": False, "role": None,
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
    # TWO DIFFERENT FACTS, kept apart because they answer different questions.
    #
    # `derived_self` — THIS RUN's lineage answer was REGEXed out of `original_command` rather than
    # recorded at fork time: either the block on disk says `"derived": true` (a `--backfill`
    # write), or there is no block and the command is all there is (derived at read time). The
    # first half was MISSING until 2026-09-07: the block's own key was never read, so a run whose
    # derivation concluded `fresh` — no `fork_parent`, hence no parent-side signal — printed
    # `derived: false` while its metadata said the opposite. 47 runs were invisible to the ⚠ marker
    # that way and the archive count read 115 where the census read 162 (ledger 2026-09-07).
    #
    # `parent_derived` — the parent REFERENCE this run names carries `derived`. `build_lineage`
    # never writes that into a stored `fork_parent`, so on today's archive it only ever fires
    # through `fork_parent()`'s own two paths and agrees with `derived_self`; it is kept as its own
    # term so a hand-written or future block that does carry it is not quietly dropped.
    #
    # `derived` is their union — the historic field, unchanged wherever it was already True.
    # Whether an ANCESTOR's lineage was derived is a THIRD fact, marked per node in `render`.
    out["derived_self"] = bool(block.get("derived")) if block is not None else bool(
        read_original_command(run_dir))
    out["parent_derived"] = bool(parent is not None and parent.derived)
    out["derived"] = out["derived_self"] or out["parent_derived"]
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


#: What `main.lineage` prints for a reference recorded before the resolution provenance existed.
#: `None` on the four `resolution_*`/`resolved_*` keys means NOT RECORDED — a pre-change run, whose
#: teacher went through the OLD rule (`best_model/best_model.zip` first). It is deliberately NOT
#: re-resolved under today's rule: that would print a current answer as if it were history.
LEGACY_RESOLUTION_NOTE = "resolved file not recorded (pre gen3_last_snapshot_resolution_v1)"


def _resolution(entry: Dict[str, Any]) -> str:
    """One line describing WHICH FILE a recorded reference resolved to, and HOW it was chosen."""
    rule = entry.get("resolution_rule")
    if not rule:
        return LEGACY_RESOLUTION_NOTE
    if rule == "unresolved":
        return "UNRESOLVED — the path resolved to no .zip when this run recorded it"
    steps = entry.get("resolved_num_timesteps")
    return (f"{entry.get('resolved_file') or '?'}  "
            f"@{f'{steps:,} steps' if isinstance(steps, int) else 'steps unknown'}  "
            f"[rung={entry.get('resolution_rung') or '?'} rule={rule}]")


def _reference_lines(entries: List[Dict[str, Any]], *, pad: str) -> List[str]:
    """`- <run name>` + its resolved-file line, per recorded model reference."""
    out: List[str] = []
    for e in entries:
        out.append(f"{pad}- {e.get('run_name') or e.get('path') or '?'}")
        out.append(f"{pad}    -> {_resolution(e)}")
    return out


def render(row: Dict[str, Any]) -> str:
    """The tree for one run, plus its teachers/target and any broken links."""
    if row.get("error"):
        return f"{row['run']}: {row['error']}"
    lines = []
    # RECORDED and DERIVED are two independent facts and the tag states both. A `--backfill` block
    # is on disk (recorded) AND was REGEXed out of `original_command` (derived); before 2026-09-07
    # the "recorded" branch swallowed the second half and the ⚠ never printed for such a run.
    # The tag reads `derived_self`, not the `derived` union: a run whose own block claims nothing
    # but whose stored parent reference carries the flag is not making a derived claim about
    # ITSELF, and its header is left exactly as it has always read.
    if row["recorded"]:
        tag = "recorded ⚠ DERIVED from original_command" if row["derived_self"] else "recorded"
    else:
        tag = ("⚠ DERIVED from original_command" if row["derived"] else "no lineage recorded")
    lines.append(f"{row['run']}   role={row.get('role') or '—'}   [{tag}]")
    # The two step facts side by side: where this run STARTED (its fork point, from the immutable
    # lineage block) and how far it GOT (the latest `num_timesteps`). "unknown" is a real answer —
    # a legacy run recorded neither, and 0 would be a claim.
    fs, ns = row.get("fork_step"), row.get("num_timesteps")
    if fs is not None or ns is not None:
        lines.append(f"    steps: fork_step={_steps(fs)}   num_timesteps={_steps(ns)}")
    if row["teachers"]:
        lines.append(f"    teachers ({len(row['teachers'])}):")
        lines.extend(_reference_lines(row["teachers"], pad="      "))
    if row["exploiter_target"]:
        lines.append("    exploiter target:")
        lines.extend(_reference_lines([row["exploiter_target"]], pad="      "))
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
            lines.append(f"{npad}   -> {_resolution(parent)}")
    stop = row.get("ancestry_stop")
    if stop:
        lines.append(f"    ✖ chain ends at {stop.get('at')}: {stop.get('reason')}")
    for problem in row["checks"]:
        lines.append(f"    ⚠ {problem}")
    return "\n".join(lines)


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    """How many of these runs STATE their lineage, and how many only IMPLY it.

    The four counts are deliberately not a partition — `derived` cuts ACROSS `recorded`, because a
    `--backfill` block is both on disk and a re-parse of a shell command. Reading it as a partition
    is exactly the mistake this function exists to prevent: on 2026-09-07 the backlog carried 105
    "runs with a derived parent" (the `main.tb_inherit` FORK population) against a census figure of
    162 (every run whose lineage is derived at all), and nothing printed either number."""
    return {
        "runs": len(rows),
        "recorded": sum(1 for r in rows if r.get("recorded")),
        "derived": sum(1 for r in rows if r.get("derived")),
        # `derived_self`, to agree with the ⚠ that `render` puts on those same header lines.
        "recorded_and_derived": sum(1 for r in rows
                                    if r.get("recorded") and r.get("derived_self")),
        "no_lineage": sum(1 for r in rows if not r.get("recorded") and not r.get("derived")
                          and not r.get("error")),
    }


def render_summary(s: Dict[str, int]) -> str:
    return (f"{s['runs']} runs: {s['recorded']} record a lineage block "
            f"({s['recorded_and_derived']} of them ⚠ DERIVED), "
            f"{s['derived']} ⚠ DERIVED in total, {s['no_lineage']} state no lineage at all")


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
        print(json.dumps({"runs": rows, "summary": summarize(rows)}, indent=2))
        return 0
    for i, row in enumerate(rows):
        if i:
            print()
        print(render(row))
    if len(rows) > 1:
        print()
        print(render_summary(summarize(rows)))
    return 1 if any(r["checks"] or r.get("error") for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
