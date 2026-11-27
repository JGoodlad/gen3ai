"""BOUNDARY-1's candidates' read against P*: the spec's builder (PREREG §18 Amendment 6 item 4, Amendment 9).

Item 4 registered the read whole except three things that did not exist: the candidates' run names, their stop steps
and the pin. A records commit writes ``read_spec_plateau_pstar.json`` from them "after the last leg and before
``plan``". This module is that writer. It holds item 4's ``anchor`` and ``decision`` blocks as constants (the test
pins them against the PREREG's own text), takes the candidates from the continuation chain's ``<run>.candidate``
markers (never a typed path), names P* through the baseline registry, applies item 4's two guards, and prints what the
read will cost. ``python -m main.reads`` plays the spec it writes; nothing here plays a game.

**A marker** is the chain's ``key=value`` record of a run that stopped: ``run``, ``tag`` (:data:`TAGS`),
``stop_step`` (the leg end), ``newest_checkpoint``, ``final`` (``models/<run>/final_model.zip``), ``final_sha256``,
``status``, ``at``.

**A candidate enters** iff its pin history is ONE entry at the run's pin from step 0 that reaches its stop step, and
the kit's arm-end check at the stop step (``<run>_to<N>M.arm_end_check.json`` beside the markers, the output of
``python -m main.b1.look check-run --arm-end``) does not read K9(b) VOID. One that fails is LEFT OUT and named in
``excluded``; two candidates still read, one refuses (the rule has no seed variance to charge P* with).

**A refusal** (:class:`PlateauReadError`, exit 2; nothing is written) is everything the registration does not decide:
a run of the cell with no marker yet, a marker that is not the chain's shape, a final that no longer hashes to its
marker, a missing arm-end check or one that names a problem other than the two guards, a ``pstar_plateau`` that is
no longer the registered file.

    python -m main.b1.plateau_read build --cell C --pin <PIN> [--markers DIR] [--out read_spec_plateau_pstar.json]
    python -m main.b1.plateau_read files-fixed --record <state>/record.json   # the REPORTED files-fixed figure
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from main.b1 import arms as A

READ = "boundary1_plateau_pstar_2026-11-16"
PREREG = "designs/research_state/measurements/boundary1_scratch_2026-11-09/PREREG.md"
SPEC_NAME = "read_spec_plateau_pstar.json"
SCHEMA = "gen3_read_spec_v1"
#: Where the continuation chain writes its markers and arm-end checks (read-only here).
MARKERS = "~/.claude/jobs/x35/segB"
STATE = "~/gen3ai_archive/boundary1_plateau_pstar"

GROUP = "kakuna"
OPPONENT = "metamon:Kakuna"
TEAMSETS: Tuple[str, ...] = ("heldout72_hpfix", "forum72", "p72")
UNIT_SEEDS = "0-35"
N_UNIT_SEEDS, UNIT_GAMES = 36, 20
QUIET_S = 300
CAND_ARM, PSTAR_ARM, PSTAR_LABEL, PSTAR_SEED = "cand", "pstar", "pstar", 2601
#: P*, as item 4 registered it. The spec names it through the registry (:func:`pstar_ref`); these are the check.
BASELINE = "pstar_plateau"
PSTAR_PATH = "models/rb_x26_s2601/checkpoints/checkpoint_280000170_steps.zip"
PSTAR_SHA256 = "814823c514fb791e3c6e671ba126f7d0fd90733098b0539bb85babe7c5ab91ed"
#: Item 4's ``decision`` block, verbatim (``plateau_read_test`` reads it back out of the PREREG).
DECISION: Dict[str, Any] = {"rule": "b1_kakuna", "params": {
    "group": GROUP,
    "contrasts": {"candidate_U": {"arms": {CAND_ARM: 1.0, PSTAR_ARM: -1.0}, "panels": list(TEAMSETS)},
                  "candidate_F": {"arms": {CAND_ARM: 1.0, PSTAR_ARM: -1.0}, "panels": ["forum72"]}},
    "reported_contrasts": ["candidate_F"], "nets": {}, "level": 0.9, "games_per_panel": N_UNIT_SEEDS * UNIT_GAMES,
    "min_games_frac": 0.95, "bootstrap": {"draws": 10000, "seed": 20261109}}}
CONFIRMING = "candidate_U"
#: §12's rate: the Kakuna row prices 25,920 games at ≈ 180 lane-hours.
GAMES_PER_LANE_HOUR = 25_920 / 180
LANES = 3

CELLS: Dict[str, str] = {"C": "c", "W": "w", "Br": "br", "WB": "wb"}
TAGS: Tuple[str, ...] = ("PLATEAU", "NOT-PLATEAUED", "PLATEAU-AT-100M (not resumed)")
MARKER_KEYS: Tuple[str, ...] = ("run", "tag", "stop_step", "newest_checkpoint", "final", "final_sha256", "status", "at")
K9B_VOID = "VOID"
_SHA = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class PlateauReadError(ValueError):
    """Something item 4 does not decide: nothing is written, and the orchestrator rules."""


def cell_runs(cell: str) -> List[Tuple[str, int]]:
    """``(run name, seed)`` of the cell's three runs, in the launch block's order."""
    if cell not in CELLS:
        raise PlateauReadError(f"unknown cell {cell!r} (known: {sorted(CELLS)})")
    return [(A.run_name(CELLS[cell], s), s) for s in A.SEEDS[CELLS[cell]]]


def read_marker(path: Path, run: str) -> Dict[str, Any]:
    """One ``<run>.candidate`` marker, checked for the chain's shape."""
    rows = [ln.split("=", 1) for ln in path.read_text().splitlines() if ln.strip()]
    if any(len(r) != 2 for r in rows) or sorted(k for k, _ in rows) != sorted(MARKER_KEYS):
        raise PlateauReadError(f"{path}: not the chain's marker (keys {sorted(MARKER_KEYS)}, one key=value a line)")
    m: Dict[str, Any] = dict(rows)
    for k in ("stop_step", "newest_checkpoint"):
        if not re.fullmatch(r"[1-9][0-9]*", m[k]):
            raise PlateauReadError(f"{path}: {k}={m[k]!r} is not a step")
        m[k] = int(m[k])
    if m["run"] != run or m["final"] != f"models/{run}/final_model.zip":
        raise PlateauReadError(f"{path}: names run {m['run']!r} and final {m['final']!r}, not {run}'s")
    if m["tag"] not in TAGS:
        raise PlateauReadError(f"{path}: tag {m['tag']!r} is none of {TAGS}")
    if not _SHA.match(m["final_sha256"]):
        raise PlateauReadError(f"{path}: final_sha256 is not a sha256")
    if not m["stop_step"] <= m["newest_checkpoint"] < m["stop_step"] + A.CHECKPOINT_EVERY:
        raise PlateauReadError(f"{path}: newest_checkpoint {m['newest_checkpoint']:,} is not within one checkpoint "
                               f"interval after stop_step {m['stop_step']:,}")
    return m


def pin_problem(history: Any, pin: str, stop_step: int) -> Optional[str]:
    """``None`` iff ``history`` (a run's ``metadata.json`` ``pin_history``) is ONE entry at ``pin`` from step 0 whose
    last save is at or after ``stop_step`` (the leg's closing save)."""
    rows = history if isinstance(history, list) else []
    if len(rows) != 1 or not isinstance(rows[0], dict) or rows[0].get("git_hash") != pin:
        seen = [(str(r.get("git_hash"))[:8], r.get("first_step"), r.get("last_step")) if isinstance(r, dict) else r
                for r in rows]
        return f"pin history {seen} is not one entry at {pin[:8]}"
    first, last = rows[0].get("first_step"), rows[0].get("last_step")
    if type(first) is not int or first != 0:
        return f"its pin entry starts at {first!r}, not step 0"
    if type(last) is not int or last < stop_step:
        return f"its pin entry ends at {last!r}, before its stop step {stop_step:,}"
    return None


def k9b_problem(check_path: Path, run: str, run_pin: str, stop_step: int) -> Optional[str]:
    """``None`` iff the kit's arm-end check of ``run`` at ``stop_step`` passed; ``"K9(b) VOID ..."`` iff its K9(b)
    read is VOID (the candidate is left out). Anything else the check says is a :class:`PlateauReadError`."""
    kit = (f"python -m main.b1.look check-run --arm <cell> --seed <seed> --pin {run_pin} --step {stop_step} --arm-end "
           f"> {check_path}  (from the kit worktree)")
    try:
        got = json.loads(check_path.read_text())
    except (OSError, ValueError) as e:
        raise PlateauReadError(f"{run}: no readable arm-end check at {check_path} ({type(e).__name__}); "
                               f"write it with: {kit}") from None
    want = {"run": run, "pin": run_pin, "step": stop_step, "arm_end": True}
    if not isinstance(got, dict) or any(got.get(k) != v for k, v in want.items()):
        raise PlateauReadError(f"{check_path}: not {run}'s arm-end check at the pin and stop step ({want})")
    k9b = got.get("k9b") if isinstance(got.get("k9b"), dict) else {}
    if k9b.get("verdict") == K9B_VOID:
        return f"K9(b) VOID through its stop step ({'; '.join(map(str, k9b.get('fires') or k9b.get('reasons') or []))})"
    if got.get("ok") is not True:
        raise PlateauReadError(f"{check_path}: the kit's check names a problem item 4's guards do not decide "
                               f"({'; '.join(map(str, got.get('problems') or ['ok is not true']))})")
    return None


def pstar_ref() -> Dict[str, Any]:
    """P*'s ``pinned`` ref, read from the baseline registry by NAME and held to the file item 4 registered."""
    from agents.training import baselines as B

    b = B.get(BASELINE)
    path = f"models/{b.run}/{b.checkpoint}"
    if (path, b.sha256) != (PSTAR_PATH, PSTAR_SHA256):
        raise PlateauReadError(f"the baseline {BASELINE!r} is now {path} ({str(b.sha256)[:12]}), not the file item 4 "
                               f"registered ({PSTAR_PATH}, {PSTAR_SHA256[:12]}): the read's P* is the registered one")
    return {"kind": "pinned", "path": path, "sha256": b.sha256}


def cost(n_models: int) -> Dict[str, Any]:
    """The read's size and its price at §12's rate."""
    units = n_models * len(TEAMSETS) * N_UNIT_SEEDS
    games = units * UNIT_GAMES
    lane_h = games / GAMES_PER_LANE_HOUR
    return {"models": n_models, "panels": len(TEAMSETS), "unit_seeds": N_UNIT_SEEDS, "unit_games": UNIT_GAMES,
            "planned_units": units, "planned_games": games, "games_per_lane_hour": GAMES_PER_LANE_HOUR,
            "lane_hours": round(lane_h, 1), "lanes": LANES, "hours_on_the_lanes": round(lane_h / LANES, 1)}


def build(cell: str, markers: Path, pin: str, models: Path, *, run_pin: Optional[str] = None,
          checks: Optional[Path] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """``(spec document, report)`` for the cell's candidates under ``markers``; see the module docstring."""
    from utils.hashing import sha256_file

    run_pin = run_pin or pin
    for name, sha in (("--pin", pin), ("--run-pin", run_pin)):
        if not _COMMIT.match(sha):
            raise PlateauReadError(f"{name} must be a full 40-hex commit sha, not {sha!r}")
    runs = cell_runs(cell)
    known = {f"{r}.candidate" for r, _ in runs}
    stray = sorted(p.name for p in markers.glob("*.candidate") if p.name not in known)
    if stray:
        raise PlateauReadError(f"{markers}: marker(s) {stray} are not of cell {cell}'s runs {[r for r, _ in runs]}")
    waiting = [r for r, _ in runs if not (markers / f"{r}.candidate").is_file()]
    if waiting:
        raise PlateauReadError(f"not ready: no candidate marker under {markers} for {waiting} (the read is planned "
                               "once the LAST candidate exists)")
    entered: List[Dict[str, Any]] = []
    excluded: List[Dict[str, Any]] = []
    for run, seed in runs:
        m = read_marker(markers / f"{run}.candidate", run)
        final = models / run / "final_model.zip"
        if not final.is_file() or sha256_file(final) != m["final_sha256"]:
            raise PlateauReadError(f"{run}: {final} is missing or no longer hashes to its marker "
                                   f"({m['final_sha256'][:12]}): the candidate is the final AT its leg end")
        try:
            history = json.loads((models / run / "metadata.json").read_text()).get("pin_history")
        except (OSError, ValueError) as e:
            raise PlateauReadError(f"{run}: metadata.json cannot be read ({type(e).__name__})") from None
        check = (checks or markers) / f"{run}_to{m['stop_step'] // 1_000_000}M.arm_end_check.json"
        why = [w for w in (pin_problem(history, run_pin, m["stop_step"]),
                           k9b_problem(check, run, run_pin, m["stop_step"])) if w]
        row = {"run": run, "seed": seed, "label": run.split("_", 1)[1], "tag": m["tag"], "stop_step": m["stop_step"],
               "newest_checkpoint": m["newest_checkpoint"], "final_sha256": m["final_sha256"]}
        (excluded if why else entered).append(dict(row, why=why) if why else row)
    if len(entered) < 2:
        raise PlateauReadError(f"{len(entered)} candidate(s) pass item 4's guards (excluded: "
                               f"{[(x['run'], x['why']) for x in excluded]}): the rule needs two, and no class is read")
    spec_models: Dict[str, Any] = {
        c["label"]: {"arm": CAND_ARM, "seed": c["seed"], "ref": {
            "kind": "final", "run": f"models/{c['run']}", "min_step": c["stop_step"], "quiet_s": QUIET_S}}
        for c in entered}
    spec_models[PSTAR_LABEL] = {"arm": PSTAR_ARM, "seed": PSTAR_SEED, "ref": pstar_ref()}
    said = "; ".join(f"{c['run']} at {c['stop_step']:,} ({c['tag']})" for c in entered)
    left = "".join(f" LEFT OUT by item 4's guards: {x['run']} ({'; '.join(x['why'])})." for x in excluded)
    note = (f"BOUNDARY-1's candidates' read against P* (PREREG section 18 Amendment 6 item 4; written by "
            f"python -m main.b1.plateau_read, Amendment 9): cell {cell}'s plateau candidates, each its run's final at "
            f"the leg end where it stopped ({said}), and P* ({BASELINE}, re-played) against Kakuna on H "
            f"(heldout72_hpfix), F (forum72) and P72 (p72), unit seeds 0-35 = 720 games a model a panel. CONFIRMING "
            f"contrast: {CONFIRMING}, the candidates' mean minus P* on U. REPORTED: candidate_F. A NOT-PLATEAUED "
            f"candidate carries that tag in every quote.{left}")
    doc = {"schema": SCHEMA, "read": READ, "prereg": PREREG, "pin": pin, "note": note, "models": spec_models,
           "anchor": {GROUP: {"opponent": OPPONENT, "teamsets": list(TEAMSETS), "unit_seeds": UNIT_SEEDS,
                              "models": list(spec_models)}},
           "decision": json.loads(json.dumps(DECISION))}
    report = {"read": READ, "cell": cell, "pin": pin, "run_pin": run_pin, "markers": str(markers),
              "candidates": entered, "excluded": excluded, "degrees_of_freedom": len(entered) - 1,
              "pstar": spec_models[PSTAR_LABEL]["ref"], **cost(len(spec_models))}
    return doc, report


def dumps(doc: Mapping[str, Any]) -> str:
    """The spec file's bytes (its sha256 is the read's identity, so one writer)."""
    return json.dumps(doc, indent=1) + "\n"


def files_fixed(record: Mapping[str, Any]) -> Dict[str, Any]:
    """Item 4's REPORTED files-fixed figure: each contrast's team-clustered standard error with no seed term, the
    square root of the record's ``v_t_pp2``, under both clusterings ("these files on other teams"). Never a class."""
    contrasts = (record.get("decision") or {}).get("contrasts") or {}
    if not contrasts:
        raise PlateauReadError("the record carries no contrast (an INCONCLUSIVE or unfinished read has no figure)")
    out: Dict[str, Any] = {}
    for name, blk in contrasts.items():
        se = {s: math.sqrt(float(x["v_t_pp2"])) for s, x in blk["schemes"].items()}
        out[name] = {"psi_pp": blk["psi_pp"], "files_fixed_se_pp": se, "wider_scheme": max(sorted(se), key=se.__getitem__)}
    return {"read": record.get("read"), "reported_only": True, "contrasts": out}


def _commands(spec: str) -> List[str]:
    return [f"python -m main.reads check-spec --spec {spec}",
            f"python -m main.reads plan --spec {spec} --state {STATE}",
            f"setsid nohup python -m main.reads run --state {STATE} --lane anchor --concurrency {LANES} --hours 24 "
            f"> {STATE}/anchor.log 2>&1 &",
            f"python -m main.reads record --state {STATE}",
            f"python -m main.b1.plateau_read files-fixed --record {STATE}/record.json"]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.b1.plateau_read", description=(__doc__ or "").split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="check the candidates, print the read's size and price, write the spec with --out")
    b.add_argument("--cell", required=True, choices=sorted(CELLS), help="the cell that continued to its plateau")
    b.add_argument("--pin", required=True, help="the commit the read plays at (item 4: the launch pin first)")
    b.add_argument("--run-pin", help="the commit the candidates trained at (default: --pin)")
    b.add_argument("--markers", default=MARKERS, help=f"the chain's marker directory (default {MARKERS})")
    b.add_argument("--checks", help="where the arm-end checks are (default: beside the markers)")
    b.add_argument("--out", help=f"write the spec here ({SPEC_NAME} in the registration folder); without it nothing "
                                 "is written")
    f = sub.add_parser("files-fixed", help="the REPORTED files-fixed figure of a played read's record")
    f.add_argument("--record", required=True)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "files-fixed":
            print(json.dumps(files_fixed(json.loads(Path(a.record).expanduser().read_text())), indent=1))
            return 0
        from utils.hashing import sha256_file
        from utils.paths import main_models_dir

        models = main_models_dir()
        if models is None:
            raise PlateauReadError("no models/ archive here (set GEN3AI_MODELS_DIR)")
        doc, report = build(a.cell, Path(a.markers).expanduser(), a.pin, Path(models), run_pin=a.run_pin,
                            checks=Path(a.checks).expanduser() if a.checks else None)
        if a.out:
            out = Path(a.out).expanduser()
            out.write_text(dumps(doc))
            report.update(spec=str(out), spec_sha256=sha256_file(out), next=_commands(str(out)))
        else:
            report.update(spec=None, next=["re-run with --out to write the spec"])
        print(json.dumps(report, indent=1))
        return 0
    except (PlateauReadError, A.ArmError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
