"""main.baselines — READ AND CHANGE THE NAMED BASELINES (``designs/baselines.json``).

A baseline is the thing a result is read AGAINST, and until ``gen3_baselines_registry_v1`` not one
of ours was a first-class object: "production" was a hand-copied JSON nothing consumes at launch,
the untaught meter's opponent was a string literal in a module, the famine comparator was a
sentence in a ledger entry, and the curated TensorBoard set was decided by asking. The engine, the
rationale and the validation rules are in ``src/agents/training/baselines.py``; this is the CLI.

    export PYTHONPATH=$PYTHONPATH:src

    python -m main.baselines                      # every baseline, one line each
    python -m main.baselines list --verbose       # + purpose, notes, resolved file
    python -m main.baselines show production
    python -m main.baselines spec untaught_meter_opponent      # for scripts
    python -m main.baselines check                # validate everything; non-zero on drift
    python -m main.baselines set production ai_v9_21_gen17_pfspoff_0820/final_model.zip \\
        --reason "2026-09-06 · <the ledger entry title that authorises this>"

    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)

**``set`` is a PROCEDURE, not an edit.** It re-resolves the run through the run-spec choke point,
recomputes the sha256, re-reads ``config_version`` / ``arch_signature`` / the commit from the run
itself, rewrites exactly one entry, and PRINTS the ledger line to append. It never touches the
ledger: the ledger is append-only and records WHY, which no tool can author. ``--reason`` is
required for the same reason — a baseline that changed with no entry naming it is the state this
registry exists to end.

**``check`` is also a TEST** (``src/main/baselines_test.py``), unmarked and in the routine suite,
so a groomed-away baseline file or a drifted ``production_config.json`` fails the suite rather than
being discovered by whoever next reads a number against it.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

from agents.training import baselines as reg
from utils.paths import main_models_dir, models_skip_reason


def build_parser() -> argparse.ArgumentParser:
    """The parser, extracted so it can be inspected without running anything."""
    p = argparse.ArgumentParser(
        prog="python -m main.baselines",
        description="The NAMED baselines: read them, resolve them, validate them, change one.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="A baseline is resolved BY NAME, never by memory. `set` prints the ledger line to "
               "append; it never edits the ledger.")
    p.add_argument("--registry", default=None, metavar="PATH",
                   help=f"the registry JSON (default: {reg.REGISTRY_PATH}).")
    sub = p.add_subparsers(dest="cmd")

    q = sub.add_parser("list", help="every baseline, one line each (the default).")
    q.add_argument("--verbose", action="store_true", help="add purpose, notes and resolved file.")
    q.add_argument("--json", dest="json_out", action="store_true", help="emit the registry as JSON.")

    s = sub.add_parser("show", help="one baseline, in full.")
    s.add_argument("name")

    for verb, helptext in (("spec", "print the run spec a flag would take"),
                           ("path", "print the resolved absolute file path"),
                           ("describe", "print the one-line provenance consumers print")):
        v = sub.add_parser(verb, help=helptext)
        v.add_argument("name")

    c = sub.add_parser("check", help="validate everything; exit non-zero on any drift.")
    c.add_argument("--no-sha", action="store_true",
                   help="skip the sha256 re-hash (the only part that reads whole checkpoints).")
    c.add_argument("--quiet", action="store_true", help="print problems only.")

    t = sub.add_parser("set", help="re-point one baseline (a PROCEDURE — --reason required).")
    t.add_argument("name")
    t.add_argument("ref", metavar="RUN[@STEP]|RUN/FILE.zip",
                   help="the new target. EXPLICIT: a run-relative .zip, or <run>@<step>. A bare "
                        "run dir is REFUSED — the last-snapshot rule would move it silently.")
    t.add_argument("--reason", required=True, metavar="LEDGER TITLE",
                   help="the ledger entry title that authorises the change. Recorded as set_by.")
    t.add_argument("--purpose", default=None,
                   help="one sentence: what this baseline IS. Kept from the old entry if unset.")
    t.add_argument("--notes", default=None, help="free text (kept from the old entry if unset).")
    t.add_argument("--kind", choices=reg.KINDS, default=None,
                   help="checkpoint (a model .zip) or config (a model_config.json).")
    t.add_argument("--set-on", default=None, metavar="YYYY-MM-DD",
                   help="the date (default: today).")
    t.add_argument("--dry-run", action="store_true",
                   help="print the entry and the ledger line; write nothing.")
    return p


# --------------------------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------------------------

def _resolved_line(name: str, registry: Optional[str]) -> str:
    """The file a name resolves to on THIS box, or the honest reason it does not."""
    if main_models_dir() is None:
        return f"    (no archive: {models_skip_reason()})"
    try:
        b = reg.get(name, registry)
        if b.kind == "config":
            return f"    -> {reg.config_path(name, registry)}"
        r = reg.resolve(name, registry)
        return f"    -> {r.zip_path} [rung={r.rung}]"
    except reg.BaselineError as exc:
        return f"    -> UNRESOLVED: {exc}"


def cmd_list(args, registry: Optional[str]) -> int:
    doc = reg.load_registry(registry)
    if getattr(args, "json_out", False):
        print(json.dumps(doc, indent=1))
        return 0
    print(f"# {registry or reg.REGISTRY_PATH}")
    for name in reg.names(registry):
        b = reg.get(name, registry)
        print(f"  {b.describe()}")
        if args.verbose:
            print(f"    purpose: {b.purpose}")
            if b.notes:
                print(f"    notes:   {b.notes}")
            print(_resolved_line(name, registry))
    for lname in reg.list_names(registry):
        li = reg.get_list(lname, registry)
        print(f"  {li.describe()}")
        if args.verbose:
            print(f"    purpose: {li.purpose}")
    return 0


def cmd_show(args, registry: Optional[str]) -> int:
    b = reg.get(args.name, registry)
    print(b.describe())
    print(f"  purpose        {b.purpose}")
    print(f"  run            {b.run}")
    print(f"  checkpoint     {b.checkpoint}   (spec: {b.spec})")
    print(f"  commit         {b.commit}")
    print(f"  config_version {b.config_version}   arch {b.arch_signature}")
    print(f"  sha256         {b.sha256}")
    print(f"  set            {b.set_on} by {b.set_by}")
    if b.floor_elo is not None:
        print(f"  floor_elo      {b.floor_elo}")
    if b.era_checkout_only:
        print(f"  era_checkout_only — the weights are readable from commit {b.commit} only")
    if b.config_mirror_version is not None:
        print(f"  config_mirror  constructed at v{b.config_mirror_version} "
              f"(the surface run records v{b.config_version})")
    if b.config_overrides:
        print(f"  config_overrides ({len(b.config_overrides)}) {json.dumps(b.config_overrides)}")
    if b.pending:
        print(f"  pending        {json.dumps(b.pending)}")
    if b.notes:
        print(f"  notes          {b.notes}")
    print(_resolved_line(args.name, registry))
    return 0


def cmd_check(args, registry: Optional[str]) -> int:
    findings = reg.validate(registry, verify_sha=not args.no_sha)
    for f in findings:
        if args.quiet and f.level == "ok":
            continue
        print(f.line())
    bad = [f for f in findings if f.level == "error"]
    if bad:
        print(f"\n[baselines] CHECK FAILED — {len(bad)} problem(s).", file=sys.stderr)
        return 1
    warn = [f for f in findings if f.level == "warn"]
    print(f"\n[baselines] OK — {len(reg.names(registry))} baseline(s), "
          f"{len(reg.list_names(registry))} list(s)"
          + (f", {len(warn)} warning(s)" if warn else "")
          + (" (sha check SKIPPED)" if args.no_sha else "") + ".")
    return 0


# --------------------------------------------------------------------------------------------
# set — the procedure
# --------------------------------------------------------------------------------------------

def split_ref(ref: str) -> "tuple[str, str]":
    """``RUN[@STEP]`` / ``RUN/FILE`` → ``(run, checkpoint)``. Refuses a bare run directory.

    A bare dir is the ONE form the registry cannot hold: it resolves through the last-snapshot
    rungs, so the file a name points at would move the next time the run checkpoints — which is
    precisely the silence ``gen3_baselines_registry_v1`` exists to remove.
    """
    ref = ref.strip().rstrip("/")
    if ref.startswith("models/"):
        ref = ref[len("models/"):]
    if "@" in ref:
        run, step = ref.split("@", 1)
        if not step.isdigit():
            raise SystemExit(f"[baselines] {ref!r}: @<step> must be digits, got {step!r}")
        return run.rstrip("/"), "@" + step
    if "/" in ref:
        run, rest = ref.split("/", 1)
        return run, rest
    raise SystemExit(
        f"[baselines] {ref!r} is a bare run directory. A registry entry must be EXPLICIT — pass "
        f"{ref}@<step> or {ref}/<file>.zip — because a bare dir resolves through the last-snapshot "
        "rule and would silently move the moment the run writes another checkpoint.")


def build_entry(name: str, ref: str, *, reason: str, purpose: Optional[str],
                notes: Optional[str], kind: Optional[str], set_on: Optional[str],
                registry: Optional[str]) -> Dict[str, Any]:
    """Resolve the target and build the entry, every field re-read from the run itself."""
    run, checkpoint = split_ref(ref)
    models = main_models_dir()
    if models is None:
        raise SystemExit(f"[baselines] cannot set a baseline without the run archive: "
                         f"{models_skip_reason()}")
    run_dir = os.path.join(str(models), run)
    if not os.path.isdir(run_dir):
        raise SystemExit(f"[baselines] no run directory {run_dir}")

    old: Optional[reg.Baseline]
    try:
        old = reg.get(name, registry)
    except reg.BaselineError:
        old = None

    resolved_kind = kind or (old.kind if old else
                             ("config" if checkpoint.endswith(".json") else "checkpoint"))
    entry: Dict[str, Any] = {
        "kind": resolved_kind, "run": run, "checkpoint": checkpoint,
        "purpose": purpose if purpose is not None else (old.purpose if old else ""),
        "set_on": set_on or datetime.date.today().isoformat(),
        "set_by": reason,
    }
    if not entry["purpose"]:
        raise SystemExit(f"[baselines] {name!r} is new — pass --purpose: one sentence saying what "
                         "this baseline IS. An unexplained baseline is the state we are leaving.")

    cfg = _read_json(os.path.join(run_dir, "model_config.json")) or {}
    meta = _read_json(os.path.join(run_dir, "metadata.json")) or {}
    entry["commit"] = str(meta.get("git_hash") or "")
    entry["config_version"] = int(cfg.get("config_version", -1))
    entry["arch_signature"] = str(cfg.get("arch_signature", ""))

    if checkpoint.startswith("@"):
        entry["num_timesteps"] = int(checkpoint[1:])
        entry["sha256"] = _sha_of_step(run_dir, int(checkpoint[1:]))
    else:
        fpath = os.path.join(run_dir, checkpoint)
        if not os.path.isfile(fpath):
            raise SystemExit(f"[baselines] no file {fpath}")
        entry["sha256"] = reg.sha256_file(fpath)
        entry["num_timesteps"] = _num_timesteps(run_dir, checkpoint) \
            if resolved_kind == "checkpoint" else None

    text = notes if notes is not None else (old.notes if old else "")
    if text:
        entry["notes"] = text
    if old is not None:
        for key, value in (("era_checkout_only", old.era_checkout_only),
                           ("floor_elo", old.floor_elo),
                           ("config_overrides", old.config_overrides),
                           ("pending", old.pending)):
            if value:
                entry[key] = value
    return entry


def _read_json(p: str) -> Optional[Dict[str, Any]]:
    try:
        with open(p) as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def _sha_of_step(run_dir: str, step: int) -> str:
    for cand in (os.path.join(run_dir, "checkpoints", f"checkpoint_{step}_steps.zip"),
                 os.path.join(run_dir, f"checkpoint_{step}_steps.zip")):
        if os.path.isfile(cand):
            return reg.sha256_file(cand)
    raise SystemExit(f"[baselines] no checkpoint at step {step} under {run_dir}")


def _num_timesteps(run_dir: str, checkpoint: str) -> Optional[int]:
    """The step count, through the run-spec choke point — never re-derived from a filename."""
    try:
        from agents.training.fixed_opponent_pool import resolve_model_ref
        return resolve_model_ref(os.path.join(run_dir, checkpoint), warn=False).num_timesteps
    except Exception:
        return None


def ledger_line(name: str, entry: Dict[str, Any], old: Optional[reg.Baseline]) -> str:
    """The line to APPEND to the ledger. Printed, never written — the ledger is append-only and
    records WHY, which is the one field a tool cannot author."""
    was = (f"`{old.run}/{old.checkpoint}` (set {old.set_on}, {old.set_by})" if old
           else "*(new baseline)*")
    return (f"- **baseline `{name}` re-set** on {entry['set_on']}: "
            f"`{entry['run']}/{entry['checkpoint']}` "
            f"@{entry.get('num_timesteps') or '?'} steps, commit `{str(entry['commit'])[:8]}`, "
            f"config v{entry['config_version']} `{entry['arch_signature']}`, "
            f"sha256 `{entry['sha256'][:12]}…`. Was: {was}. "
            f"Reason: {entry['set_by']}.")


def cmd_set(args, registry: Optional[str]) -> int:
    path = registry or reg.REGISTRY_PATH
    doc = reg.load_registry(path)
    try:
        old: Optional[reg.Baseline] = reg.get(args.name, path)
    except reg.BaselineError:
        old = None
    entry = build_entry(args.name, args.ref, reason=args.reason, purpose=args.purpose,
                        notes=args.notes, kind=args.kind, set_on=args.set_on, registry=path)
    print(json.dumps({args.name: entry}, indent=1))
    line = ledger_line(args.name, entry, old)
    if args.dry_run:
        print("\n[baselines] --dry-run: nothing written. Ledger line would be:\n" + line)
        return 0
    doc.setdefault("baselines", {})[args.name] = entry
    with open(path, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=False)
        fh.write("\n")
    print(f"\n[baselines] wrote {path}")
    print("[baselines] APPEND this to designs/research_state/ledger.md (the tool never edits it):")
    print(line)
    findings = reg.validate(path)
    bad = [f for f in findings if f.level == "error"]
    for f in bad:
        print(f.line(), file=sys.stderr)
    return 1 if bad else 0


# --------------------------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    registry = args.registry
    cmd = args.cmd or "list"
    try:
        if cmd == "list":
            if not hasattr(args, "verbose"):
                args = parser.parse_args(["list"])
            return cmd_list(args, registry)
        if cmd == "show":
            return cmd_show(args, registry)
        if cmd == "spec":
            print(reg.spec(args.name, registry))
            return 0
        if cmd == "describe":
            print(reg.describe(args.name, registry))
            return 0
        if cmd == "path":
            b = reg.get(args.name, registry)
            print(reg.config_path(args.name, registry) if b.kind == "config"
                  else reg.resolve(args.name, registry).zip_path)
            return 0
        if cmd == "check":
            return cmd_check(args, registry)
        if cmd == "set":
            return cmd_set(args, registry)
    except reg.BaselineError as exc:
        print(f"[baselines] {exc}", file=sys.stderr)
        return 2
    parser.error(f"unknown command {cmd!r}")
    return 2


def names_for_help() -> List[str]:
    """Registry names for a `--help` epilog, degrading to `[]` rather than raising at import."""
    try:
        return reg.names()
    except reg.BaselineError:
        return []


if __name__ == "__main__":
    raise SystemExit(main())
