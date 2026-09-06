"""THE UNTAUGHT METER — ``python -m main.untaught_meter``.

The win rate of one or more checkpoints PILOTING a fixed team slice against ONE fixed opponent,
cluster-bootstrapped over teams, with **two** delta columns: against the frozen baseline, and
against a CONTINUATION CONTROL at matched depth. Engine + the full rationale:
``src/agents/training/untaught_meter.py``.

🚨 **THE SECOND COLUMN IS THE POINT.** Ledger 2026-09-06 (cell 2): a plain +1.08M-step continuation
of v8's parent — no teacher, no distillation term, no stable opponents — moved this meter
**+3.45pp [+0.46, +6.48]** on its own. A fold scored only against a FROZEN parent is therefore
credited with progress the parent would have made anyway; re-based on a continuation control, v8's
celebrated +4.64pp becomes ≈ +1.2pp and is not significant. Pass ``--control`` with the
continuation arms, or read the warning the report prints in its place.

Examples::

    export PYTHONPATH=$PYTHONPATH:src

    # the standing read: three fold arms vs the frozen parent AND vs the G5 continuation control
    python -m main.untaught_meter \\
        TC_UNF_A=ai_v9_162_TCUNFA_0903 TC_UNF_B=ai_v9_163_TCUNFB_0903 \\
        --baseline ai_v9_59_R2ACTION_0827 \\
        --control ai_v9_195_G5PLAINA_0906 ai_v9_196_G5PLAINB_0906 ai_v9_197_G5PLAINC_0906 \\
        --games-per-team 200 --workers 6 --json out.json --md out.md

    # resolve everything and exit non-zero on any miss, without playing a single battle
    python -m main.untaught_meter <refs…> --baseline <ref> --check

    # re-read committed per-team artifacts — no models, no battles
    python -m main.untaught_meter --from-rows \\
        FUND=…/untaught_TCFUNDA_end.json --baseline …/untaught_TCUNFA_end.json

**Sharding.** ``--workers N`` splits the TEAMS across N single-concurrency child processes. A cell
is a pure function of (ref, team index, battle index), so the split cannot move a number — gated by
``src/main/untaught_meter_reproducibility_integration_test.py``. Within-shard concurrency above 1 is
REFUSED: seeds pin the dice, but interleaved battles consume the shared streams in a
scheduling-dependent order.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional, Sequence, Tuple

from agents.training import untaught_meter as engine
from agents.training.untaught_meter import MeterError, ResolvedRef, TeamSlice
from utils.paths import main_models_dir, src_root


def build_parser() -> argparse.ArgumentParser:
    """The parser, extracted so it can be inspected without running a measurement."""
    p = argparse.ArgumentParser(
        prog="python -m main.untaught_meter",
        description="Untaught-meter: piloting win rate on a fixed team slice, cluster-bootstrapped "
                    "over teams, against a frozen baseline AND a continuation control.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="A delta against a FROZEN baseline alone overstates a fold by whatever a plain "
               "continuation would have gained (ledger 2026-09-06, cell 2: +3.45pp). Pass "
               "--control.")
    p.add_argument("refs", nargs="*", metavar="[LABEL=]REF",
                   help="model refs to score: a run dir, <run>@<step>, or a .zip. A bare run dir "
                        "resolves to the run's LAST SNAPSHOT, exactly as a launch resolves it. "
                        "Prefix with LABEL= to name a column.")
    p.add_argument("--baseline", metavar="[LABEL=]REF",
                   help="the frozen parent every ref is differenced against.")
    # `extend`, not the default `store`: with `nargs="+"` a repeated `--control` would silently
    # OVERWRITE the earlier group, so `--control A --control B` would score B alone and report a
    # one-arm control with no floor. Both spellings now accumulate.
    p.add_argument("--control", nargs="+", action="extend", default=[], metavar="[LABEL=]REF",
                   help="CONTINUATION arms at matched depth. Several are pooled equal-weight and "
                        "carry their own max-pairwise replicate floor.")
    p.add_argument("--teams", default=None, metavar="MANIFEST",
                   help=f"team manifest JSON, in order (default: the untaught 8, "
                        f"{engine.DEFAULT_TEAMS_MANIFEST}).")
    p.add_argument("--taught", nargs="?", const=str(engine.DEFAULT_TAUGHT_MANIFEST), default=None,
                   metavar="MANIFEST",
                   help="score the TAUGHT slice instead (default manifest: the taught 16).")
    p.add_argument("--opponent", default=engine.DEFAULT_OPPONENT, metavar="REF",
                   help="the ONE fixed opponent (default: rev-1's 24M snapshot).")
    p.add_argument("--config", default=engine.DEFAULT_CONFIG, metavar="PATH",
                   help="the model_config.json every model is loaded against; 'auto' resolves each "
                        "model's own (default: rev-1's snapshot config, what the probes used).")
    p.add_argument("--games-per-team", type=int, default=engine.DEFAULT_GAMES_PER_TEAM)
    p.add_argument("--seed", type=int, default=engine.DEFAULT_SEED,
                   help="seeds every stream; at 0 the dice reproduce the banked probes exactly.")
    p.add_argument("--workers", type=int, default=1,
                   help="shard the TEAMS over N single-concurrency processes.")
    p.add_argument("--concurrency", type=int, default=1,
                   help="battles in flight per shard. Values above 1 are REFUSED (unquotable).")
    p.add_argument("--impl", choices=("rust", "node"), default="rust")
    p.add_argument("--floor", type=float, default=None, metavar="PP",
                   help="the externally-ruled replicate floor for the BASELINE column, in pp "
                        "(e.g. 1.66 frozen / 4.27 controller-live). Regime-specific; never pooled.")
    p.add_argument("--bootstrap-draws", type=int, default=engine.DEFAULT_BOOTSTRAP_DRAWS)
    p.add_argument("--bootstrap-seed", type=int, default=engine.DEFAULT_BOOTSTRAP_SEED)
    p.add_argument("--from-rows", action="store_true",
                   help="the refs are committed per-team artifacts, not models: no battles.")
    p.add_argument("--json", dest="json_out", default=None, metavar="PATH")
    p.add_argument("--md", dest="md_out", default=None, metavar="PATH")
    p.add_argument("--check", action="store_true",
                   help="resolve every ref, team and opponent; exit non-zero on any miss. "
                        "Plays nothing.")
    p.add_argument("--dry-run", action="store_true", help="print the plan and exit 0.")
    p.add_argument("--quiet", action="store_true")
    # internal — the shard child contract, not part of the user surface
    p.add_argument("--shard-teams", default=None, help=argparse.SUPPRESS)
    p.add_argument("--shard-out", default=None, help=argparse.SUPPRESS)
    return p


# --------------------------------------------------------------------------------------------
# Resolution
# --------------------------------------------------------------------------------------------

def _split_label(spec: str) -> Tuple[Optional[str], str]:
    """``LABEL=REF`` → ``("LABEL", "REF")``. A Windows-style drive letter is not a concern here;
    a ``=`` inside a path is, so only the FIRST ``=`` splits and only when the left side has no
    path separator."""
    if "=" in spec:
        head, tail = spec.split("=", 1)
        if head and os.sep not in head and "/" not in head:
            return head, tail
    return None, spec


def _uniquify(labels: List[str]) -> List[str]:
    seen: Dict[str, int] = {}
    out: List[str] = []
    for lab in labels:
        if lab in seen:
            seen[lab] += 1
            out.append(f"{lab}#{seen[lab]}")
        else:
            seen[lab] = 0
            out.append(lab)
    return out


def _config_override(args) -> Optional[str]:
    if args.config in (None, "", "auto"):
        return None
    cand = args.config
    if not os.path.isfile(cand):
        models = main_models_dir()
        if models is not None and os.path.isfile(str(models / cand)):
            return str(models / cand)
    return cand


def resolve_all(args) -> Tuple[List[ResolvedRef], Optional[ResolvedRef], List[ResolvedRef],
                               ResolvedRef, List[TeamSlice]]:
    """Resolve teams, refs, baseline, controls and the opponent. Raises :class:`MeterError`."""
    manifest = args.taught or args.teams or str(engine.DEFAULT_TEAMS_MANIFEST)
    prefix = "T" if args.taught else "U"
    teams = engine.load_team_manifest(manifest, prefix=prefix)

    cfg = _config_override(args)
    specs = [(s, "ref") for s in args.refs]
    if args.baseline:
        specs.append((args.baseline, "baseline"))
    specs += [(s, "control") for s in args.control]

    raw = [_split_label(s) for s, _ in specs]
    labels = _uniquify([lab or engine._default_label(ref) for lab, ref in raw])

    resolved: List[ResolvedRef] = []
    problems: List[str] = []
    for (lab, (_, ref)), (_, role) in zip(zip(labels, raw), specs):
        try:
            resolved.append(engine.resolve_ref(ref, label=lab, role=role, config_override=cfg))
        except MeterError as exc:
            problems.append(str(exc))
    opponent: Optional[ResolvedRef] = None
    try:
        opponent = engine.resolve_ref(args.opponent, label="OPPONENT", role="opponent",
                                      config_override=cfg)
    except MeterError as exc:
        problems.append(str(exc))
    if problems:
        raise MeterError("\n".join(problems))
    assert opponent is not None
    refs = [r for r in resolved if r.role == "ref"]
    baseline = next((r for r in resolved if r.role == "baseline"), None)
    controls = [r for r in resolved if r.role == "control"]
    return refs, baseline, controls, opponent, teams


# --------------------------------------------------------------------------------------------
# --from-rows
# --------------------------------------------------------------------------------------------

def _rows_label(path: str) -> str:
    return os.path.basename(path).rsplit(".", 1)[0]


def load_from_rows(args) -> Tuple[Dict[str, Dict[str, engine.Cell]], List[str], List[str],
                                  Optional[str], List[str]]:
    """Ingest committed per-team artifacts. Returns (cells, team_keys, ref_labels, baseline, ctrl)."""
    specs = [(s, "ref") for s in args.refs]
    if args.baseline:
        specs.append((args.baseline, "baseline"))
    specs += [(s, "control") for s in args.control]
    raw = [_split_label(s) for s, _ in specs]
    labels = _uniquify([lab or _rows_label(p) for lab, p in raw])

    cells: Dict[str, Dict[str, engine.Cell]] = {}
    missing = [p for _, p in raw if not os.path.isfile(p)]
    if missing:
        raise MeterError("--from-rows: missing artifact(s):\n  " + "\n  ".join(missing))
    for lab, (_, path) in zip(labels, raw):
        cells[lab] = engine.cells_from_rows_artifact(path)

    # The shared teams, in the artifacts' own sorted key order — the readouts' convention.
    shared = sorted(set.intersection(*(set(c) for c in cells.values())))
    if not shared:
        raise MeterError("--from-rows: the artifacts share no team key")
    ref_labels = [lab for lab, (_, role) in zip(labels, specs) if role == "ref"]
    baseline = next((lab for lab, (_, role) in zip(labels, specs) if role == "baseline"), None)
    controls = [lab for lab, (_, role) in zip(labels, specs) if role == "control"]
    return cells, shared, ref_labels, baseline, controls


# --------------------------------------------------------------------------------------------
# Sharding
# --------------------------------------------------------------------------------------------

def _shard_teams(teams: Sequence[TeamSlice], workers: int) -> List[List[TeamSlice]]:
    """Round-robin so an expensive team does not pile onto one worker. Order within a shard is
    irrelevant to the numbers — a cell is a pure function of (ref, team index, battle index)."""
    n = max(1, min(workers, len(teams)))
    shards: List[List[TeamSlice]] = [[] for _ in range(n)]
    for i, t in enumerate(teams):
        shards[i % n].append(t)
    return [s for s in shards if s]


def _child_argv(args, team_indices: Sequence[int], out_path: str) -> List[str]:
    argv = [sys.executable, "-m", "main.untaught_meter"]
    argv += list(args.refs)
    if args.baseline:
        argv += ["--baseline", args.baseline]
    if args.control:
        argv += ["--control"] + list(args.control)
    if args.taught:
        argv += ["--taught", args.taught]
    elif args.teams:
        argv += ["--teams", args.teams]
    argv += ["--opponent", args.opponent, "--config", args.config,
             "--games-per-team", str(args.games_per_team), "--seed", str(args.seed),
             "--impl", args.impl, "--concurrency", str(args.concurrency),
             "--workers", "1",
             "--shard-teams", ",".join(str(i) for i in team_indices),
             "--shard-out", out_path]
    return argv


def _run_shards(args, refs, baseline, controls, opponent, teams, log) -> Dict[str, Dict[str, engine.Cell]]:
    shards = _shard_teams(teams, args.workers)
    if len(shards) == 1:
        return _play(args, refs, baseline, controls, opponent, teams, log)

    env = dict(os.environ)
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS"):
        env.setdefault(var, "1")
    # ABSOLUTE, via the one path-discovery module: a relative "src" would resolve against the
    # CHILD's cwd, and in a linked worktree the editable install points at MAIN's src — so a child
    # could silently import a different tree than the parent (src/packaging_gate_test.py's class).
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in [env.get("PYTHONPATH", ""), str(src_root())] if p)

    with tempfile.TemporaryDirectory(prefix="untaught_meter_") as tmp:
        procs = []
        for si, shard in enumerate(shards):
            out = os.path.join(tmp, f"shard_{si}.json")
            argv = _child_argv(args, [t.index for t in shard], out)
            log(f"  [shard {si}] teams {[t.key for t in shard]}")
            procs.append((si, out, subprocess.Popen(argv, env=env)))
        failed = [si for si, _, p in procs if p.wait() != 0]
        if failed:
            raise MeterError(f"shard(s) {failed} failed — see their output above")
        return engine.merge_cells(json.load(open(out)) for _, out, _ in procs)


def _play(args, refs, baseline, controls, opponent, teams, log) -> Dict[str, Dict[str, engine.Cell]]:
    all_refs = list(refs) + ([baseline] if baseline else []) + list(controls)
    seen: Dict[str, ResolvedRef] = {}
    for r in all_refs:
        seen[r.label] = r
    t0 = time.time()

    def progress(label, key, cell):
        log(f"    {label:24s} {key:14s} {cell.wins:4d}/{cell.finished:<4d}"
            + (f"  ({cell.timeouts} TIMEOUT)" if cell.timeouts else ""))

    cells = engine.play_cells(list(seen.values()), teams, opponent,
                              games_per_team=args.games_per_team, seed=args.seed,
                              impl=args.impl, concurrency=args.concurrency,
                              progress=None if args.quiet else progress)
    log(f"  played in {time.time() - t0:.0f}s")
    return cells


# --------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------

def _plan_lines(args, refs, baseline, controls, opponent, teams) -> List[str]:
    out = [f"teams        {len(teams)} clusters from "
           f"{args.taught or args.teams or engine.DEFAULT_TEAMS_MANIFEST}"]
    for t in teams:
        out.append(f"  [{t.index}] {t.key:14s} pin_sha {t.pin_sha}  {t.path}")
    out.append(f"opponent     {opponent.provenance()}")
    out.append(f"config       {opponent.config_path}")
    for r in refs:
        out.append(f"ref          {r.label:24s} {r.provenance()}")
    if baseline:
        out.append(f"baseline     {baseline.label:24s} {baseline.provenance()}")
    for c in controls:
        out.append(f"control      {c.label:24s} {c.provenance()}")
    if not controls:
        out.append("control      NONE — the delta vs the frozen baseline will OVERSTATE a fold by "
                   "whatever a plain continuation would have gained (ledger 2026-09-06, cell 2).")
    n = (len(refs) + (1 if baseline else 0) + len(controls)) * len(teams) * args.games_per_team
    out.append(f"battles      {n} ({args.games_per_team}/team) · concurrency {args.concurrency} · "
               f"{max(1, min(args.workers, len(teams)))} worker(s) · impl {args.impl}")
    out.append(f"seed         {args.seed}  (five env seams per team + per-battle sim/policy seeds)")
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    log = (lambda *_: None) if args.quiet else (lambda msg: print(msg, flush=True))

    if args.from_rows:
        try:
            cells, team_keys, ref_labels, baseline_label, control_labels = load_from_rows(args)
        except MeterError as exc:
            print(f"untaught_meter: {exc}", file=sys.stderr)
            return 1
        if args.check:
            print(f"untaught_meter --check: {len(cells)} artifact(s), {len(team_keys)} shared "
                  f"team key(s) — OK")
            return 0
        meta = {"mode": "from-rows", "teams_manifest": "(from the artifacts)",
                "games_per_team": None, "seed": None, "concurrency": None,
                "opponent": {"resolved_file": "(as recorded in the artifacts)"},
                "argv": list(argv if argv is not None else sys.argv[1:])}
        result = engine.aggregate(cells, team_keys, ref_labels=ref_labels,
                                  baseline_label=baseline_label, control_labels=control_labels,
                                  floor=args.floor, draws=args.bootstrap_draws,
                                  bootstrap_seed=args.bootstrap_seed)
        return _emit(args, {"_meta": meta, "result": result}, log)

    if not args.refs:
        print("untaught_meter: no refs given (see --help)", file=sys.stderr)
        return 2

    try:
        engine.check_concurrency(args.concurrency)
        refs, baseline, controls, opponent, teams = resolve_all(args)
    except MeterError as exc:
        print(f"untaught_meter: {exc}", file=sys.stderr)
        return 1

    if args.shard_teams is not None:
        wanted = {int(x) for x in args.shard_teams.split(",") if x != ""}
        teams = [t for t in teams if t.index in wanted]

    if args.check:
        print("\n".join(_plan_lines(args, refs, baseline, controls, opponent, teams)))
        print("untaught_meter --check: every ref, team and opponent resolved — OK")
        return 0
    if args.dry_run:
        print("\n".join(_plan_lines(args, refs, baseline, controls, opponent, teams)))
        print("untaught_meter --dry-run: nothing played.")
        return 0

    if args.shard_out is None:          # a shard child would just re-print the parent's plan
        log("\n".join(_plan_lines(args, refs, baseline, controls, opponent, teams)))

    try:
        cells = _run_shards(args, refs, baseline, controls, opponent, teams, log)
    except MeterError as exc:
        print(f"untaught_meter: {exc}", file=sys.stderr)
        return 1

    if args.shard_out:
        with open(args.shard_out, "w") as fh:
            json.dump({lab: {k: c.to_json() for k, c in t.items()} for lab, t in cells.items()},
                      fh, indent=1)
        return 0

    meta = {
        "mode": "play",
        "teams_manifest": args.taught or args.teams or str(engine.DEFAULT_TEAMS_MANIFEST),
        "teams": [t.to_json() for t in teams],
        "opponent": opponent.to_json(),
        "refs": [r.to_json() for r in refs],
        "baseline": baseline.to_json() if baseline else None,
        "controls": [c.to_json() for c in controls],
        "games_per_team": args.games_per_team,
        "seed": args.seed,
        "seed_convention": {
            "env_per_team": engine.team_env_seeds(args.seed, 0),
            "env_offsets": "value shown for team 0; each seam is offset + 1e6*seed + team_index",
            "sim": "[seed + team_index + 1, battle_index + 1, 3, 4]",
            "pool_sequence": "random.Random(61000 + 1e6*seed + team_index).randrange(n_pool)",
            "pilot_policy": "71000 + 1e6*seed + team_index*1000 + battle_index (per battle)",
            "opponent_policy": "72000 + 1e6*seed + team_index*1000 + battle_index (per battle)",
        },
        "concurrency": args.concurrency,
        "workers": max(1, min(args.workers, len(teams))),
        "impl": args.impl,
        "argv": list(argv if argv is not None else sys.argv[1:]),
    }
    ref_labels = [r.label for r in refs]
    result = engine.aggregate(cells, [t.key for t in teams], ref_labels=ref_labels,
                              baseline_label=baseline.label if baseline else None,
                              control_labels=[c.label for c in controls], floor=args.floor,
                              draws=args.bootstrap_draws, bootstrap_seed=args.bootstrap_seed)
    return _emit(args, {"_meta": meta, "result": result}, log)


def _emit(args, doc: dict, log) -> int:
    doc["_meta"]["volatile"] = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                "cwd": os.getcwd()}
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(doc, fh, indent=1, sort_keys=False)
        log(f"  wrote {args.json_out}")
    md = engine.render_markdown(doc)
    if args.md_out:
        with open(args.md_out, "w") as fh:
            fh.write(md)
        log(f"  wrote {args.md_out}")
    if not args.quiet:
        print()
        print(engine.render_text(doc))
    return 3 if doc["result"]["timeouts"]["inconclusive"] else 0


if __name__ == "__main__":
    sys.exit(main())
