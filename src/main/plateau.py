"""THE PLATEAU METER, TIER 1 — the offline driver and reader (``design_evaluation.md`` §8.1; unit U9a, 2026-10-06).

    python -m main.plateau status <run> [--lineage-run DIR ...] [--json]      # reads the ledger; plays nothing
    python -m main.plateau tick   <run> [--lineage-run DIR ...] [--max-checks N] [--dry-run] [compute flags]

**What it answers.** "Does Tier 1 say PLATEAU, or not yet?" for a live or finished run. Every Δ = 10M steps the
newest node plays the node W = 50M steps back on mirrored pairs, and the registered GSPRT (``agents.training.
plateau_t1``: H0 0.50 / H1 0.52, α = β = 0.05, 40-pair batches, cap 6,000 pairs) decides GAIN / FLAT / UNDECIDED.
The run's TIER-1 status is ``TIER1_PLATEAU`` once two consecutive checks are FLAT. 🚨 **It is Tier 1 ONLY**: §8.3's
plateau also needs the cycle monitor NONE and the panel FLAT (U5 / U9, NOT BUILT) — every output says so.

**``tick``** plays every DUE check (both nodes on disk, no decision yet), oldest first: ONE ``plateau_t1`` request per
(run, check step, W), opened with its two nodes' sha256 FROZEN in the spec (a node file that changed is a refusal,
never a re-plan); one 40-pair batch at a time on ``main.h2h``'s own engine and row builder (purpose ``plateau``,
the h2h protocol and regime, a schedule key on the plateau's OWN namespace so its games never collide with another
consumer's); the rule re-read after EVERY batch, so the test stops at its first decision; ONE decision row per
request (``append_decision(unique=True)``: a second driver gets the first's decision back, never a second one).
Resumable at any batch: a re-run reads the request's rows and continues. CPU by default; ``--device cuda`` needs
the GPU LEASE (the engine takes ``utils.gpu_lock``) — never while a training run owns the GPU.

**Where it plays.** Offline, under the operator's control: the deep run's training agent runs ``tick`` after each
10M checkpoint (its 55-min cron can), on CPU beside the run (``nice``, ``scripts/ops/mem_cap.sh``) or on the GPU
between runs. The in-trainer GPU window (U4) is NOT built; a PINNED run (X26) could not carry it anyway (F-ED-19).

**The nodes** are the run's PERIODIC checkpoints (``checkpoints/checkpoint_<step>_steps.zip``): the first at or above
each multiple of Δ, within the plan's slack. ``--lineage-run`` adds an ANCESTOR's checkpoints (a fork's first W of
steps has its W-back node in the parent); the run's own file wins a tie. A check whose node is missing is reported
MISSING_NODE, never played against a substitute."""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from agents.training import eval_ledger as L
from agents.training import plateau_t1 as T1

#: This driver's rows: ONE plateau_t1 request's, at main.h2h's protocol (``main.h2h.play.PROTOCOL``; a test pins
#: the literal to it, so this module reads the ledger without importing the engine).
PROTOCOL = "gen3_eval_protocol_v1_h2h"
TIER1_ROWS = L.ReaderDecl(
    name="main.plateau.tier1", purposes=frozenset({"plateau"}),
    regime=L.RegimeFilter(protocol=PROTOCOL, play="greedy", opponent_play="greedy", mirrored=True),
    requests="own", selection="include", flags_ok=frozenset(), inference="conditional")

SCHEDULE_TAG = "gen3_plateau_t1_schedule_v1"
SCHEDULE_SEED = 0
_PERIODIC = re.compile(r"^checkpoint_(\d+)_steps\.zip$")

# per-check states beside the rule's verdicts (status only)
MISSING_NODE, NOT_STARTED, CONFLICT = "MISSING_NODE", "NOT_STARTED", "CONFLICT"


class PlateauError(RuntimeError):
    """A refusal: a frozen input changed, a check made no progress, or the ledger holds two decisions for one
    request."""


# ------------------------------------------------------------------------------------------------ the nodes
def periodic_checkpoints(run_dir: Path) -> Dict[int, Path]:
    """``{step: zip}`` of a run's periodic checkpoints (``checkpoints/`` and the legacy run root). A SIGUSR1 forced
    save is not a node: it lands off the grid."""
    out: Dict[int, Path] = {}
    for d in (run_dir, run_dir / "checkpoints"):
        if d.is_dir():
            for p in d.iterdir():
                m = _PERIODIC.match(p.name)
                if m and p.is_file():
                    out[int(m.group(1))] = p
    return out


def node_series(run_dir: Path, lineage: Sequence[Path] = ()) -> Dict[int, Path]:
    """The run's nodes plus its named ancestors' (the run's own file wins a tie)."""
    out: Dict[int, Path] = {}
    for d in reversed(list(lineage)):
        out.update(periodic_checkpoints(d))
    out.update(periodic_checkpoints(run_dir))
    return out


def node_name(nodes: Mapping[int, Path], step: Optional[int]) -> Optional[str]:
    """``<run>@<step>`` of a node (its run is the directory above ``checkpoints/``), or ``None``."""
    if step is None:
        return None
    p = nodes[step]
    run = p.parent.parent.name if p.parent.name == "checkpoints" else p.parent.name
    return f"{run}@{step}"


def schedule_key(newest_sha: str, wback_sha: str) -> str:
    """The plateau's own schedule namespace (ORDERED: the newest node is the measured side)."""
    return "plateau_t1:" + hashlib.blake2b(f"{SCHEDULE_TAG}:{newest_sha}:{wback_sha}".encode(),
                                           digest_size=8).hexdigest()


def consumer_spec(run: str, chk: T1.Check, plan: T1.CheckPlan, rule: T1.Tier1Rule, newest: Mapping[str, Any],
                  wback: Mapping[str, Any]) -> Dict[str, Any]:
    """The check's FROZEN inputs, pinned in the request's spec: a re-open with another node file is refused."""
    return {"consumer": "main.plateau", "rule": rule.rule_string(), "rule_version": T1.RULE_VERSION, "run": run,
            "grid": int(chk.grid), "plan": plan.to_json(),
            "newest": {k: newest[k] for k in ("id", "step", "sha256")},
            "wback": {k: wback[k] for k in ("id", "step", "sha256")}}


# ------------------------------------------------------------------------------------------------ the decision
def decisions_by_request(root: Path, rids: Sequence[str]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {r: [] for r in rids}
    for d in L.read_decisions(root=root, kind=T1.DECISION_KIND, request_ids=rids):
        if str(d["rule"]).startswith(T1.RULE_NAME + "("):
            out[d["request_id"]].append(d)
    return out


def _batches(rows: Sequence[Mapping[str, Any]]) -> List[int]:
    return sorted(int(r["request"]["batch"]) for r in rows)


def drive_check(writer: L.LedgerWriter, root: Path, rid: str, rule: T1.Tier1Rule, subject: str,
                play_upto: Callable[[int], None], emit: Callable[[str], None]) -> Dict[str, Any]:
    """Play ``rid``'s test to its decision and write it ONCE. ``play_upto(n)`` makes batches ``0 .. n-1`` exist (it
    plays the missing ones); the rule is re-read after every batch. Returns the decision on record."""
    while True:
        got = L.read(TIER1_ROWS, root=root, request_id=rid)
        have = _batches(got.rows)
        gap = next((i for i in range(len(have)) if i not in set(have)), None)
        if gap is None:
            res = T1.evaluate(got.rows, rule)
            if res.final:
                break
            want = len(have) + 1
        else:
            want = gap + 1                                   # fill the gap first (a voided claim, a dead driver)
        play_upto(want)
        if len(L.read(TIER1_ROWS, root=root, request_id=rid).rows) <= len(have):
            raise PlateauError(f"{rid}: batch {want - 1} was not recorded (another driver holds it, or its claim was "
                               "voided) — re-run tick later")
        emit(f"[plateau] {rid}: batch {want - 1} recorded; " + (
            "re-reading" if gap is not None else f"n_pairs {res.n_pairs + rule.batch_pairs} so far"))
    try:
        d = writer.append_decision(kind=T1.DECISION_KIND, subject=subject, consumed=got, verdict=res.verdict,
                                   rule=rule.rule_string(), rule_version=T1.RULE_VERSION, request_id=rid,
                                   unique=True)
        writer.finish_request(rid)
    except L.DecisionExistsError as e:
        d = e.decision
        if d["verdict"] != res.verdict:
            raise PlateauError(f"{rid}: the ledger's decision {d['decision_id']} says {d['verdict']!r}, the rows give "
                               f"{res.verdict!r} — run `python -m main.eval_ledger verify {d['decision_id']}`") from None
    emit(f"[plateau] {rid}: {res.verdict} ({res.reason}) at {res.n_pairs} pairs, LLR {res.llr:+.3f}; decision "
         f"{d['decision_id']}")
    return d


# ------------------------------------------------------------------------------------------------ the status
@dataclass
class CheckState:
    check: T1.Check
    rid: str
    state: str
    newest: Optional[str]
    wback: Optional[str]
    n_pairs: int = 0
    llr: Optional[float] = None
    decision_id: Optional[str] = None
    note: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {"grid": self.check.grid, "newest_step": self.check.newest, "wback_step": self.check.wback,
                "newest": self.newest, "wback": self.wback, "request_id": self.rid, "state": self.state,
                "n_pairs": self.n_pairs, "llr": None if self.llr is None else round(self.llr, 6),
                "decision_id": self.decision_id, "note": self.note}


def check_states(run: str, nodes: Mapping[int, Path], root: Path, plan: T1.CheckPlan,
                 rule: T1.Tier1Rule) -> List[CheckState]:
    """Every planned check's state, read from the ledger (plays nothing)."""
    checks = T1.plan_checks(nodes, plan)
    rids = [T1.request_id(run, c.grid, plan) for c in checks]
    decs = decisions_by_request(root, rids) if root.exists() else {r: [] for r in rids}
    out: List[CheckState] = []
    for c, rid in zip(checks, rids):
        cs = CheckState(check=c, rid=rid, state=NOT_STARTED, newest=node_name(nodes, c.newest),
                        wback=node_name(nodes, c.wback))
        if not c.playable:
            cs.state = MISSING_NODE
            cs.note = "no node at " + " and ".join(
                f"{g:,}" for g, s in ((c.grid, c.newest), (c.grid - plan.w_steps, c.wback)) if s is None)
            out.append(cs)
            continue
        rows = L.read(TIER1_ROWS, root=root, request_id=rid).rows if root.exists() else ()
        ds = decs[rid]
        try:
            res = T1.evaluate(rows, rule) if rows else None
        except T1.Tier1RowsError as e:
            cs.state, cs.note = CONFLICT, str(e)
            out.append(cs)
            continue
        if res is not None:
            cs.n_pairs, cs.llr = res.n_pairs, res.llr
        if len(ds) > 1:
            cs.state, cs.note = CONFLICT, f"{len(ds)} decisions for one request: {[d['decision_id'] for d in ds]}"
        elif ds:
            cs.state, cs.decision_id = ds[0]["verdict"], ds[0]["decision_id"]
            if res is None or res.verdict != cs.state:
                cs.state, cs.note = CONFLICT, (f"decision says {ds[0]['verdict']}, the rows give "
                                               f"{None if res is None else res.verdict}")
        elif res is not None:
            cs.state = T1.CONTINUE
            if res.final:
                cs.note = f"the rows decide {res.verdict}, no decision row yet — run tick"
        out.append(cs)
    return out


def status_report(run: str, states: Sequence[CheckState], plan: T1.CheckPlan, rule: T1.Tier1Rule,
                  root: Path) -> Dict[str, Any]:
    st = T1.tier1_status([(s.check.grid, s.state) for s in states], plan.delta_steps)
    return {"run": run, "ledger": str(root), "rule": rule.to_json(), "plan": plan.to_json(),
            "checks": [s.to_json() for s in states], "tier1": st,
            "conflicts": [s.rid for s in states if s.state == CONFLICT], "caveat": T1.TIER2_CAVEAT}


def format_report(rep: Mapping[str, Any]) -> str:
    plan, t = rep["plan"], rep["tier1"]
    lines = [f"[plateau] {rep['run']}: TIER 1 = head-to-head GSPRT newest vs W-back, Δ {plan['delta_steps']:,}, "
             f"W {plan['w_steps']:,} steps; rule {rep['rule']['rule']}{'' if rep['rule']['registered'] else ' (NOT the registered rule)'}",
             f"  ledger {rep['ledger']}",
             f"  {'check':>13}  {'newest':<34} {'w-back':<34} {'state':<13} {'pairs':>6} {'LLR':>8}  note"]
    for c in rep["checks"]:
        llr = "" if c["llr"] is None else f"{c['llr']:+.3f}"
        lines.append(f"  {c['grid']:>13,}  {c['newest'] or '-':<34} {c['wback'] or '-':<34} {c['state']:<13} "
                     f"{c['n_pairs'] or '':>6} {llr:>8}  {c['note']}")
    where = "" if t["as_of_check"] is None else f" (as of the {t['as_of_check']:,} check)"
    lines.append(f"TIER 1 STATUS: {t['status']}{where}" + (
        f"; FLAT at {', '.join(f'{g:,}' for g in t['flat_checks'])}" if t["flat_checks"] else ""))
    if rep["conflicts"]:
        lines.append(f"⚠️  CONFLICT in {len(rep['conflicts'])} check(s): {rep['conflicts']} — fix before reading")
    lines.append(f"⚠️  {rep['caveat']}")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------------------ the CLI
def _run(a: argparse.Namespace) -> tuple:
    from main.ops.run_ref import resolve_run_dir

    run_dir = resolve_run_dir(a.run)
    lineage = [resolve_run_dir(x) for x in (a.lineage_run or [])]
    plan = T1.CheckPlan(delta_steps=a.delta_steps, lag=a.lag, slack_steps=a.slack_steps)
    root = Path(L.check_write_root(a.out)) if a.out is not None else Path(L.archive_ledger_root())
    return run_dir, lineage, plan, root


def cmd_status(a: argparse.Namespace) -> int:
    run_dir, lineage, plan, root = _run(a)
    nodes = node_series(run_dir, lineage)
    rep = status_report(run_dir.name, check_states(run_dir.name, nodes, root, plan, T1.REGISTERED), plan,
                        T1.REGISTERED, root)
    print(json.dumps(rep, indent=1, sort_keys=True) if a.json else format_report(rep))
    return 2 if rep["conflicts"] else 0


def tick(run_dir: Path, lineage: Sequence[Path], plan: T1.CheckPlan, out: Optional[str], compute: Any, *,
         rule: T1.Tier1Rule = T1.REGISTERED, max_checks: Optional[int] = None, dry_run: bool = False,
         emit: Callable[[str], None] = lambda m: print(m, file=sys.stderr, flush=True)) -> Dict[str, Any]:
    """Play every DUE check of ``run_dir`` (oldest first, at most ``max_checks``) to its decision on ``main.h2h``'s
    engine (``compute``: a ``main.h2h.play.Compute``), into the ledger root ``out`` (``None`` = the archive's), and
    return the status report afterwards."""
    from main.h2h import play as PL
    from utils.rust_env import episode as EP

    run = run_dir.name
    root, writer = PL.open_writer(out, T1.PURPOSE)
    nodes = node_series(run_dir, lineage)
    due = [s for s in check_states(run, nodes, root, plan, rule) if s.state in (NOT_STARTED, T1.CONTINUE)]
    if max_checks is not None:
        due = due[:max_checks]
    emit(f"[plateau] {run}: {len(due)} due check(s): {[s.check.grid for s in due]} (ledger {root})")
    if dry_run or not due:
        writer.close()
        return status_report(run, check_states(run, nodes, root, plan, rule), plan, rule, root)
    regime = PL.regime_for(EP.stall_threshold())
    commit = PL.current_commit()
    try:
        for s in due:
            c = s.check
            new, old = PL.resolve_player(str(nodes[c.newest])), PL.resolve_player(str(nodes[c.wback]))  # type: ignore[index]
            for ref, want in ((new, c.newest), (old, c.wback)):
                if ref.num_timesteps != want:
                    raise PlateauError(f"{ref.zip_path}: num_timesteps {ref.num_timesteps} != its file name's {want}")
            spec = consumer_spec(run, c, plan, rule, new.block(), old.block())
            try:
                writer.open_request(s.rid, kind=T1.REQUEST_KIND, purpose=T1.PURPOSE, protocol=regime["protocol"],
                                    spec=PL.edge_spec(rule.batch_pairs, SCHEDULE_SEED, spec))
            except L.RequestSpecError as e:
                raise PlateauError(f"check {c.grid:,}: its FROZEN inputs changed since the request was opened (a node "
                                   f"file was replaced?) — {e}") from None
            key = schedule_key(new.sha256, old.sha256)
            with contextlib.ExitStack() as stack:
                eng: List[Any] = []

                def play_upto(n: int) -> None:
                    ep = PL.plan_edge(writer, root, regime, new, old, pairs=n * rule.batch_pairs,
                                      batch_pairs=rule.batch_pairs, schedule_seed=SCHEDULE_SEED, schedule_key=key,
                                      purpose=T1.PURPOSE, request_id=s.rid, family=None,
                                      request_kind=T1.REQUEST_KIND, consumer_spec=spec)
                    if not ep.todo:
                        return
                    if not eng:
                        stack.enter_context(PL.engine_lock(compute))
                        e = PL.H2HEngine(new, old, compute, emit)
                        stack.callback(e.close)
                        if e.regime["regime_id"] != regime["regime_id"]:
                            raise PlateauError("the engine's regime differs from the one planned")
                        eng.append(e)
                    try:
                        PL.play_planned(eng[0], writer, ep, schedule_seed=SCHEDULE_SEED, purpose=T1.PURPOSE,
                                        run_label=f"plateau_t1:{run}", commit=commit, emit=emit)
                    except L.ClaimHeldError as err:
                        raise PlateauError(f"{s.rid}: {err} — another driver is playing this check") from None

                emit(f"[plateau] check {c.grid:,}: {new.id} vs {old.id} (request {s.rid})")
                drive_check(writer, root, s.rid, rule, new.sha256, play_upto, emit)
    finally:
        writer.close()
    return status_report(run, check_states(run, nodes, root, plan, rule), plan, rule, root)


def cmd_tick(a: argparse.Namespace) -> int:
    from main.h2h import play as PL

    run_dir, lineage, plan, _root = _run(a)
    compute = PL.Compute(device=a.device, backend=a.backend, n_envs=a.n_envs, threads=a.threads,
                         torch_threads=a.torch_threads, front=a.front, profile=a.profile)
    rep = tick(run_dir, lineage, plan, a.out, compute, max_checks=a.max_checks, dry_run=a.dry_run)
    print(format_report(rep))
    return 2 if rep["conflicts"] else 0


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m main.plateau", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, hlp in (("status", "every check's state and the run's Tier-1 status (reads the ledger, plays nothing)"),
                      ("tick", "play every due check to its decision, then print the status")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("run", help="a run NAME (in the archive) or a run DIRECTORY")
        p.add_argument("--lineage-run", action="append", default=None,
                       help="an ANCESTOR run whose checkpoints are nodes too (repeatable; the run's own file wins)")
        p.add_argument("--delta-steps", type=int, default=T1.DEFAULT_PLAN.delta_steps, help="the check spacing Δ")
        p.add_argument("--lag", type=int, default=T1.DEFAULT_PLAN.lag, help="W = lag × Δ (the registered 5: 50M)")
        p.add_argument("--slack-steps", type=int, default=T1.DEFAULT_PLAN.slack_steps,
                       help="a node is the first checkpoint in [grid, grid + slack)")
        p.add_argument("--out", default=None,
                       help="the ledger ROOT (default: the run archive's _ledger; any other root under models/ is "
                            "REFUSED)")
        if name == "status":
            p.add_argument("--json", action="store_true")
        else:
            p.add_argument("--max-checks", type=int, default=None, help="play at most N due checks (oldest first)")
            p.add_argument("--dry-run", action="store_true", help="list the due checks; play nothing")
            p.add_argument("--device", default="cpu", help="cpu (default); cuda needs the GPU LEASE")
            p.add_argument("--backend", default="", choices=("", "eager", "graph", "aot"))
            p.add_argument("--n-envs", type=int, default=64)
            p.add_argument("--threads", type=int, default=4, help="the Rust core's worker threads")
            p.add_argument("--torch-threads", type=int, default=4, help="intra-op threads of a CPU forward")
            p.add_argument("--front", default="proc", choices=("proc", "ffi"))
            p.add_argument("--profile", default="release", choices=("release", "selfcheck"))
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    a = _parser().parse_args(argv)
    return {"status": cmd_status, "tick": cmd_tick}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
