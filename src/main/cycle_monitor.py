"""THE CYCLE MONITOR and THE PLATEAU'S TIER 2 — the offline driver and reader (``design_evaluation.md`` §2.5, §8.2;
units U5 + U9, the offline slice, 2026-10-19). The rules are ``agents.training.cycle_monitor`` (the four signals and
the verdict) and ``agents.training.plateau_t2`` (the panel and the two-tier signature).

    python -m main.cycle_monitor status <run> [--lineage-run DIR ...] [--panel NAME] [--json]    # reads; plays nothing
    python -m main.cycle_monitor tick   <run> [--lineage-run DIR ...] [--panel NAME] [--max-checks N] [--dry-run] ...
    python -m main.cycle_monitor ladder <run> [--from-step S] [--to-step S] [--tier1] [--json]   # DESCRIPTIVE retro

**What it answers.** At every Δ = 10M check: is the run CYCLING (NONE / SUSPECTED / CONFIRMED / INCONCLUSIVE), and —
with the Tier-1 GSPRT (``main.plateau``) and the panel — which §8.3 signature is it (``TRANSITIVE_IMPROVEMENT``,
``TRUE_PLATEAU``, ``PLATEAU_BUT_PANEL``, ``TREADMILL``, ``CONTINUE``)? ``main.plateau status`` prints the same Tier-2
columns and signature beside Tier 1. A flag COUNTS for the signature only when its SIZE on the recent window (the last
8 checks' nodes) is MATERIAL against the 1.7 pp bar; a NEGLIGIBLE one counts as NONE (``plateau_t2``, the
2026-10-19 amendment, F-ED-27). The size is computed by the reader from the rows; no decision row changes.

**``tick``** plays, oldest check first: (1) the check's ROW — the newest node against every older node at a lag in
the hybrid set, 500 mirrored pairs per edge (one ``monitor_row`` request per check, every edge a cell of it, its node
set FROZEN in the spec); (2) once every row up to the check is complete, the rule; a SUSPECTED check plays its TOP-UP
(``monitor_topup``: 500 more pairs on the edges carrying each flag) and is re-read; (3) one ``cycle_flag`` decision row
per check (unique; ``python -m main.eval_ledger verify`` re-derives it); (4) with ``--panel``, the panel cells of the
check's newest node and its W-back node (500 pairs each; 1,000 at a candidate plateau). All rows and top-ups of a
run are ONE request family (``monitor:<run>:d<Δ>``, decision kind ``cycle_flag``), so a check's read is indexed.
Games are ``main.h2h``'s (its engine, row builder and protocol) on the monitor's OWN schedule namespace. Resumable at
any batch. CPU by default (``nice`` + ``taskset`` + ``scripts/ops/mem_cap.sh`` beside a live run, rule 5b);
``--device cuda`` needs the GPU lease.

**``ladder``** is a DESCRIPTIVE retro read, never a decision: the same four signals on a run's IN-RUN snapshot ladder
(``snapshot_ladder/games.jsonl``: mirrored, 100 pairs per edge, the pool's sliding window), as of each Δ check. It is
NOT the registered instrument (a different graph, a fifth of the pairs, the edges the pool chose) — a free first look
at a pinned run that carries none of this code. It reports each check's windowed SIZE too, and with ``--tier1`` a
DESCRIPTIVE two-tier status (Tier 1 from the ledger, the ladder's monitor read, no panel).

**Cost (state it before you tick).** A row is ≤ 11 edges × 1,000 games. On rule 5b's CPU the production net plays
≈ 1.2–2 games/s (``design_x26_readiness.md`` §2.6; X26's Tier-1 rows): ≈ 1.5–2.5 h per check. The toy plays
≈ 10–20 games/s: ≈ 10–20 min. The GPU window (U4) is NOT built.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from agents.training import cycle_monitor as CM
from agents.training import eval_ledger as L
from agents.training import plateau_t1 as T1
from agents.training import plateau_t2 as T2
from main.plateau import node_name, node_series
from utils.paths import repo_path

PROTOCOL = "gen3_eval_protocol_v1_h2h"
MATRIX_ROWS = L.ReaderDecl(
    name="main.cycle_monitor.matrix", purposes=frozenset({"monitor"}),
    regime=L.RegimeFilter(protocol=PROTOCOL, play="greedy", opponent_play="greedy", mirrored=True),
    requests="family", selection="include", flags_ok=frozenset(), inference="conditional",
    decision_kind="cycle_flag")
PANEL_ROWS = L.ReaderDecl(
    name="main.cycle_monitor.panel", purposes=frozenset({"plateau"}),
    regime=L.RegimeFilter(protocol=PROTOCOL, play="greedy", opponent_play="greedy", mirrored=True),
    requests="own", selection="include", flags_ok=frozenset(), inference="conditional")

SCHEDULE_TAG = "gen3_cycle_monitor_schedule_v1"
SCHEDULE_SEED = 0
DEFAULT_INFLIGHT = 4
PANELS_FILE = ("designs", "research_state", "plateau_panels.json")
PANELS_SCHEMA = "gen3_plateau_panels_v1"

MISSING, NOT_STARTED, PARTIAL, EVALUABLE = "MISSING_NODE", "NOT_STARTED", "PARTIAL", "EVALUABLE"
WAITING = "WAITING"          # its row is complete; an EARLIER check's row is not, so it cannot be decided yet


class MonitorDriverError(RuntimeError):
    """A refusal: a frozen input changed, a panel's file moved, the ledger holds two decisions for one check."""


def family_id(run: str, plan: CM.MonitorPlan = CM.DEFAULT_PLAN) -> str:
    return f"monitor:{run}:d{plan.delta_steps}"


def schedule_key(kind: str, run: str, a_sha: str, b_sha: str, salt: str = "") -> str:
    """The monitor's own schedule namespace (ordered; the run is in it, so a fork's re-played edge is a fresh
    sample, never a duplicate seed block; a TOP-UP's salt is its check, so two checks topping up one edge play
    fresh games each)."""
    return f"{kind}:" + hashlib.blake2b(f"{SCHEDULE_TAG}:{kind}:{run}:{a_sha}:{b_sha}:{salt}".encode(),
                                        digest_size=8).hexdigest()


# ------------------------------------------------------------------------------------------------ the panel
def load_panels(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or repo_path(*PANELS_FILE)
    with open(p) as f:
        d = json.load(f)
    if d.get("schema") != PANELS_SCHEMA:
        raise MonitorDriverError(f"{p}: schema {d.get('schema')!r} is not {PANELS_SCHEMA!r}")
    return d


def panel_members(name: Optional[str], path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """The REGISTERED panel's members (``{id, path, sha256}``); ``None`` → no panel. A member file whose sha256 no
    longer matches the registry is a refusal (a moved panel is a different panel)."""
    if not name:
        return []
    panels = load_panels(path)["panels"]
    if name not in panels:
        raise MonitorDriverError(f"panel {name!r} is not registered ({sorted(panels)})")
    return [dict(m) for m in panels[name]["members"]]


def member_file(m: Mapping[str, Any]) -> str:
    """A panel member's file: an absolute path as is; a relative one in this checkout, else run-relative under the
    MAIN checkout's ``models/`` (``baselines.candidate_paths``: a member that is a named baseline's checkpoint is
    recorded by its run-relative path, so a later re-set of the baseline never moves the frozen panel)."""
    from agents.training.baselines import candidate_paths

    p = str(m["path"])
    if Path(p).is_absolute():
        return p
    for cand in [str(repo_path(p))] + candidate_paths(p):
        if Path(cand).exists():
            return cand
    raise MonitorDriverError(f"panel member {m.get('id')}: {p} does not resolve on this box")


def panel_request_id(node_id: str, panel: str) -> str:
    return f"{T2.REQUEST_KIND}:{node_id}:{panel}"


# ------------------------------------------------------------------------------------------------ the state
@dataclass
class CheckState:
    row: CM.RowPlan
    rid: str
    newest: Optional[str]
    newest_sha: Optional[str]
    edges_total: int
    edges_done: int
    state: str
    verdict: Optional[str] = None
    decision_id: Optional[str] = None
    signals: Dict[str, str] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def to_json(self) -> Dict[str, Any]:
        return {"grid": self.row.grid, "newest": self.newest, "request_id": self.rid, "edges_total": self.edges_total,
                "edges_done": self.edges_done, "state": self.state, "verdict": self.verdict,
                "decision_id": self.decision_id, "signals": self.signals, "summary": self.summary, "note": self.note}


def _sha_of(nodes: Mapping[int, Path], cache: Dict[int, str], step: int) -> str:
    if step not in cache:
        h = hashlib.sha256()
        with open(nodes[step], "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        cache[step] = h.hexdigest()
    return cache[step]


def _pairs_by_edge(rows: Sequence[Mapping[str, Any]]) -> Dict[Tuple[str, str, str], int]:
    out: Dict[Tuple[str, str, str], int] = {}
    for r in rows:
        k = (str(r["request"]["id"]), CM._side_key(r["player"]), CM._side_key(r["opponent"]))
        out[k] = out.get(k, 0) + int(r["pairs"]["n_pairs"])
    return out


def _summary(res: CM.MonitorResult) -> Dict[str, Any]:
    d = res.detail
    return {"hodge_p": d["hodge"]["p_boot"], "cyclic_share": d["hodge"]["cyclic_share"],
            "excess_rms_pp": d["hodge"].get("excess_rms_pp"),
            "lag_z_max": (d.get("lag") or {}).get("z_max"), "lag_z_crit": (d.get("lag") or {}).get("z_crit"),
            "tri_n": d["triangles"]["n"], "tri_min_p": d["triangles"].get("min_p"),
            "nash_support": d["nash"]["support"], "nash_post_support": d["nash"]["posterior_mean_support"],
            "p_newest_in_support": d["nash"]["p_newest_in_support"], "nodes": res.n_nodes, "edges": res.n_edges}


def check_states(run: str, nodes: Mapping[int, Path], root: Path, plan: CM.MonitorPlan = CM.DEFAULT_PLAN,
                 rule: Optional[CM.MonitorRule] = None, *, evaluate: bool = True,
                 shas: Optional[Dict[int, str]] = None) -> List[CheckState]:
    """Every planned check's state from the ledger (plays nothing). A check with a decision reads it; an EVALUABLE
    one (every row up to it complete, no decision yet) is evaluated when ``evaluate`` (seconds per check). A check
    whose row has one edge (the first) is played — its edge is matrix data — but never decided (NOT_APPLICABLE)."""
    rule = rule or CM.REGISTERED
    shas = {} if shas is None else shas
    rows_all = family_rows(root, family_id(run, plan))
    done = _pairs_by_edge(rows_all)
    rids = [CM.row_request_id(run, rp.grid, plan) for rp in CM.plan_rows(nodes, plan)]
    decs: Dict[str, List[Dict[str, Any]]] = {r: [] for r in rids}
    if root.exists():
        for d in L.read_decisions(root=root, kind=CM.DECISION_KIND, request_ids=rids):
            decs[d["request_id"]].append(d)
    out: List[CheckState] = []
    complete_so_far = True
    for rp, rid in zip(CM.plan_rows(nodes, plan), rids):
        cs = CheckState(row=rp, rid=rid, newest=node_name(nodes, rp.newest), newest_sha=None,
                        edges_total=len(rp.edges), edges_done=0, state=NOT_STARTED)
        if rp.newest is None:
            cs.state, cs.note = MISSING, "no node at the check"
            out.append(cs)
            continue
        cs.newest_sha = _sha_of(nodes, shas, rp.newest)
        cs.edges_done = sum(1 for _lag, s in rp.edges
                            if done.get((rid, cs.newest_sha, _sha_of(nodes, shas, s)), 0) >= rule.pairs_per_edge)
        missing_lags = [lag for lag, s in rp.opponents if s is None]
        if missing_lags:
            cs.note = f"no node at lag(s) {missing_lags}"
        complete = cs.edges_done == cs.edges_total
        if not complete:
            cs.state = PARTIAL if cs.edges_done else NOT_STARTED
        if cs.edges_total < 2:
            if complete:
                cs.state = CM.NOT_APPLICABLE
            complete_so_far = complete_so_far and complete
            out.append(cs)
            continue
        ds = decs[rid]
        if len(ds) > 1:
            raise MonitorDriverError(f"{rid}: {len(ds)} cycle_flag decisions {[d['decision_id'] for d in ds]}")
        if ds:
            cs.state, cs.verdict, cs.decision_id = ds[0]["verdict"], ds[0]["verdict"], ds[0]["decision_id"]
        elif complete and not complete_so_far:
            cs.state = WAITING
        elif complete and complete_so_far:
            cs.state = EVALUABLE
            if evaluate:
                res, _conf, verdict, final = _evaluate(run, nodes, root, rp, rid, cs.newest_sha, rule, plan, shas)
                cs.signals, cs.summary = dict(res.signals), _summary(res)
                cs.verdict = verdict
                cs.state = verdict if final else CM.CONTINUE
                if final:
                    cs.note = "the rows decide; no decision row yet — run tick"
                else:
                    cs.note = "SUSPECTED: the top-up is not complete — run tick"
        complete_so_far = complete_so_far and complete
        out.append(cs)
    return out


def family_rows(root: Path, fam: str) -> List[Dict[str, Any]]:
    """Every row of the run's monitor family ([] when the family is not registered yet)."""
    if not root.exists():
        return []
    try:
        return list(L.read(MATRIX_ROWS, root=root, family=fam).rows)
    except L.ReaderDeclError:
        return []


def _evaluate(run: str, nodes: Mapping[int, Path], root: Path, rp: CM.RowPlan, rid: str, newest_sha: str,
              rule: CM.MonitorRule, plan: CM.MonitorPlan, shas: Dict[int, str]
              ) -> Tuple[CM.MonitorResult, Optional[Dict[str, Any]], str, bool]:
    got = _check_read(run, nodes, root, rp, plan, shas)
    return CM.evaluate_rows(got.rows, newest_sha, CM.topup_request_id(run, rp.grid, plan), rule)


def _check_read(run: str, nodes: Mapping[int, Path], root: Path, rp: CM.RowPlan, plan: CM.MonitorPlan,
                shas: Dict[int, str]) -> Any:
    assert rp.newest is not None
    below = [_sha_of(nodes, shas, s) for s in sorted(nodes) if s <= rp.newest]
    return L.read(MATRIX_ROWS, root=root, family=family_id(run, plan), players=below, opponents=below)


# ------------------------------------------------------------------------------------------------ the panel state
def panel_cells(root: Path, node_id: str, panel: str) -> Dict[str, Dict[str, Any]]:
    """member sha → ``{"counts", "aborted_share"}`` of one node's panel request."""
    if not root.exists():
        return {}
    rows = L.read(PANEL_ROWS, root=root, request_id=panel_request_id(node_id, panel)).rows
    out: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        k = CM._side_key(r["opponent"])
        cell = out.setdefault(k, {"counts": [0.0] * 5, "rows": []})
        cell["counts"] = [a + float(b) for a, b in zip(cell["counts"], r["pairs"]["counts"])]
        cell["rows"].append(r)
    for cell in out.values():
        cell["aborted_share"] = CM.aborted_share(cell.pop("rows"))
    return out


def tier2_by_check(run: str, nodes: Mapping[int, Path], root: Path, t1_states: Mapping[int, str], *,
                   panel: Optional[str] = None, plan: CM.MonitorPlan = CM.DEFAULT_PLAN,
                   t1_plan: T1.CheckPlan = T1.DEFAULT_PLAN, evaluate: bool = True,
                   panel_path: Optional[Path] = None, shas: Optional[Dict[int, str]] = None,
                   rule: Optional[CM.MonitorRule] = None) -> Dict[str, Any]:
    """Per grid step: the monitor's verdict, its windowed SIZE and the COUNTED verdict (the size bar), and the
    panel's; plus the run's §8.3 signature. ``t1_states``: grid → Tier-1 state (``main.plateau``)."""
    shas = {} if shas is None else shas
    states = check_states(run, nodes, root, plan, rule, evaluate=evaluate, shas=shas)
    members = panel_members(panel, panel_path)
    msha = [m["sha256"] for m in members]
    per: List[Dict[str, Any]] = []
    for cs in states:
        g = cs.row.grid
        mon = cs.verdict if cs.verdict in CM.VERDICTS and cs.state in CM.VERDICTS else cs.state
        size: Optional[Dict[str, Any]] = None
        if evaluate and cs.verdict in CM.VERDICTS and cs.newest_sha is not None:   # its rows are complete
            size = check_size(run, nodes, root, cs.row, plan, shas)
        sst = T2.size_state(size) if size is not None else None
        counted = T2.counted_monitor(mon, sst)
        pv: Dict[str, Any] = {"verdict": T2.NOT_DECLARED}
        if members and cs.newest is not None:
            wb = T1.node_at(g - t1_plan.w_steps, sorted(nodes), t1_plan.slack_steps)
            if wb is None:
                pv = {"verdict": T2.INCOMPLETE, "short": ["no W-back node"]}
            else:
                cand = t1_states.get(g) == T1.FLAT and counted == CM.NONE      # the COUNTED monitor (size bar)
                want = T2.REGISTERED.candidate_pairs if cand else T2.REGISTERED.pairs
                pv = T2.panel_verdict(panel_cells(root, cs.newest, panel or ""),
                                      panel_cells(root, node_name(nodes, wb) or "", panel or ""), msha, want)
        per.append({"grid": g, "tier1": t1_states.get(g), "monitor": mon, "size": sst, "size_detail": size,
                    "counted": counted, "panel": pv["verdict"], "panel_detail": pv, "check": cs.to_json()})
    sig = T2.signature(per, plan.delta_steps)
    return {"checks": per, "signature": sig, "panel": panel, "family": family_id(run, plan),
            "rule": CM.REGISTERED.to_json(), "plan": plan.to_json(), "panel_rule": T2.REGISTERED.to_json(),
            "size_rule": T2.SIZE_RULE.to_json()}


def size_seed(ids: Sequence[str]) -> int:
    """The size bootstrap's seed: the monitor's seed rule over the consumed ids, salted so it never shares the
    signals' stream."""
    return CM.seed_of(["size:" + str(i) for i in ids])


def check_size(run: str, nodes: Mapping[int, Path], root: Path, rp: CM.RowPlan, plan: CM.MonitorPlan,
               shas: Dict[int, str], rule: Optional[T2.SizeRule] = None) -> Dict[str, Any]:
    """The check's non-transitive SIZE on its recent window (``plateau_t2`` docstring): the matrix of every
    ``monitor_row`` row among nodes ≤ the newest, cut to the last ``window_checks`` checks' nodes."""
    rule = rule or T2.SIZE_RULE
    got = _check_read(run, nodes, root, rp, plan, shas)
    mat = [r for r in got.rows if (r.get("request") or {}).get("kind") == CM.REQUEST_KIND_ROW]
    if not mat:
        return {"measured": False, "reason": "no monitor_row rows"}
    return CM.cyclic_size(CM.build_matrix(CM.edges_from_rows(mat)), rule.min_step(rp.grid, plan.delta_steps),
                          size_seed([r["row_id"] for r in mat]), rule.n_boot, rule.one_sided)


# ------------------------------------------------------------------------------------------------ playing
@dataclass
class Job:
    rid: str
    kind: str
    purpose: str
    family: Optional[str]
    spec: Dict[str, Any]
    cells: List[Tuple[Any, Any, int]]          # (player ref, opponent ref, pairs)
    key_kind: str
    key_salt: str = ""


def _play(job: Job, root: Path, writer: L.LedgerWriter, compute: Any, run: str, emit: Callable[[str], None],
          inflight: int, batch_pairs: int) -> None:
    from main.h2h import many as MANY
    from main.h2h import play as PL
    from utils.rust_env import episode as EP

    regime = PL.regime_for(EP.stall_threshold())
    if job.family is not None:
        writer.register_family(job.family, decision_kind=CM.DECISION_KIND, rule=CM.REGISTERED.rule_string(),
                               protocol=regime["protocol"])
    try:
        writer.open_request(job.rid, kind=job.kind, purpose=job.purpose, family=job.family,
                            protocol=None if job.family else regime["protocol"],
                            spec=PL.edge_spec(batch_pairs, SCHEDULE_SEED, job.spec))
    except L.RequestSpecError as e:
        raise MonitorDriverError(f"{job.rid}: its FROZEN inputs changed since it was opened — {e}") from None
    plans = []
    for p, o, pairs in job.cells:
        ep = PL.plan_edge(writer, root, regime, p, o, pairs=pairs, batch_pairs=batch_pairs,
                          schedule_seed=SCHEDULE_SEED, schedule_key=schedule_key(job.key_kind, run, p.sha256, o.sha256, job.key_salt),
                          purpose=job.purpose, request_id=job.rid, family=job.family, request_kind=job.kind,
                          consumer_spec=job.spec)
        if ep.todo:
            plans.append(ep)
    if not plans:
        return
    decl = MANY.preflight([(ep.player, ep.opponent) for ep in plans], emit)
    commit = PL.current_commit()
    with contextlib.ExitStack() as stack:
        stack.enter_context(PL.engine_lock(compute))
        eng = PL.H2HEngine(plans[0].player, plans[0].opponent, compute, emit, decl=decl)
        stack.callback(eng.close)
        if eng.regime["regime_id"] != regime["regime_id"]:
            raise MonitorDriverError("the engine's regime differs from the one planned")
        for i, ep in enumerate(plans):
            if i:
                eng.set_cell(ep.player, ep.opponent)
            emit(f"[monitor] {job.rid}: {ep.player.id} vs {ep.opponent.id} — {len(ep.todo)} batch(es)")
            try:
                PL.play_planned(eng, writer, ep, schedule_seed=SCHEDULE_SEED, purpose=job.purpose,
                                run_label=f"{job.kind}:{run}", commit=commit, emit=emit, inflight=inflight)
            except L.ClaimHeldError as err:
                raise MonitorDriverError(f"{job.rid}: {err} — another driver is playing this check") from None


def tick(run_dir: Path, lineage: Sequence[Path], out: Optional[str], compute: Any, *,
         plan: CM.MonitorPlan = CM.DEFAULT_PLAN, rule: Optional[CM.MonitorRule] = None, panel: Optional[str] = None,
         max_checks: Optional[int] = None, dry_run: bool = False, inflight: int = DEFAULT_INFLIGHT,
         grids: Optional[Sequence[int]] = None, panel_path: Optional[Path] = None,
         play: Optional[Callable[..., None]] = None, resolve: Optional[Callable[[str], Any]] = None,
         t1_root: Optional[Path] = None,
         emit: Callable[[str], None] = lambda m: print(m, file=sys.stderr, flush=True)) -> Dict[str, Any]:
    """Play every due check of ``run_dir`` (module docstring), oldest first; ``grids`` restricts which checks'
    ROWS and panels are played (a check is DECIDED only when every row up to it is complete). ``play`` /
    ``resolve`` replace the engine and the checkpoint resolver (tests)."""
    from main.h2h import play as PL

    play = play or _play
    rule = rule or CM.REGISTERED
    resolve = resolve or PL.resolve_player

    run = run_dir.name
    root, writer = PL.open_writer(out, CM.PURPOSE)
    nodes = node_series(run_dir, lineage)
    shas: Dict[int, str] = {}
    fam = family_id(run, plan)
    refs: Dict[int, Any] = {}

    def ref(step: int) -> Any:
        if step not in refs:
            refs[step] = resolve(str(nodes[step]))
            if refs[step].num_timesteps != step:
                raise MonitorDriverError(f"{nodes[step]}: num_timesteps {refs[step].num_timesteps} != {step}")
        return refs[step]

    members = panel_members(panel, panel_path)
    mrefs = []
    for m in members:
        r = resolve(member_file(m))
        if r.sha256 != m["sha256"]:
            raise MonitorDriverError(f"panel {panel}: {m['id']} is {r.sha256[:12]}, registered {m['sha256'][:12]}")
        mrefs.append(r)
    def decide_due() -> None:
        """Decide every check whose rows (and every earlier check's) are complete, oldest first."""
        for c in check_states(run, nodes, root, plan, rule, evaluate=False, shas=shas):
            if c.state == EVALUABLE and (grids is None or c.row.grid in grids):
                _decide(run, nodes, root, writer, c, rule, plan, shas, compute, emit, inflight, ref, play)

    states = check_states(run, nodes, root, plan, rule, evaluate=False, shas=shas)
    todo = [s for s in states if s.state in (NOT_STARTED, PARTIAL, EVALUABLE, CM.CONTINUE)
            and (grids is None or s.row.grid in grids)]
    if max_checks is not None:
        todo = todo[:max_checks]
    emit(f"[monitor] {run}: {len(todo)} check(s) to work: {[s.row.grid for s in todo]} (family {fam}, ledger {root})")
    if dry_run:
        writer.close()
        return {"due": [s.row.grid for s in todo]}
    try:
        for s in todo:
            rp = s.row
            assert rp.newest is not None
            spec = {"consumer": "main.cycle_monitor", "rule": rule.rule_string(), "rule_version": CM.RULE_VERSION,
                    "run": run, "grid": rp.grid, "plan": plan.to_json(), "newest": ref(rp.newest).block(),
                    "opponents": [{"lag": lag, **ref(st).block()} for lag, st in rp.edges]}
            spec["newest"] = {k: spec["newest"][k] for k in ("id", "step", "sha256")}
            spec["opponents"] = [{k: o[k] for k in ("lag", "id", "step", "sha256")} for o in spec["opponents"]]
            play(Job(rid=s.rid, kind=CM.REQUEST_KIND_ROW, purpose=CM.PURPOSE, family=fam, spec=spec,
                      cells=[(ref(rp.newest), ref(st), rule.pairs_per_edge) for _lag, st in rp.edges],
                      key_kind="monitor_row"), root, writer, compute, run, emit, inflight, rule.batch_pairs)
            decide_due()                          # after EVERY row: a long tick that dies keeps its decisions
        decide_due()
        if mrefs:
            t1 = _t1_states(run, nodes, t1_root or Path(L.archive_ledger_root()))
            after = check_states(run, nodes, root, plan, rule, evaluate=False, shas=shas)
            mon = {c["grid"]: c["counted"] for c in                  # the COUNTED monitor: the size bar applies
                   tier2_by_check(run, nodes, root, t1, plan=plan, shas=shas, rule=rule)["checks"]}
            for s in after:                       # every check's panel, not only the checks whose rows were due
                g = s.row.grid
                if grids is not None and g not in grids:
                    continue
                wb = T1.node_at(g - T1.DEFAULT_PLAN.w_steps, sorted(nodes), T1.DEFAULT_PLAN.slack_steps)
                if s.row.newest is None or wb is None:
                    continue
                cand = t1.get(g) == T1.FLAT and mon.get(g) == CM.NONE
                pairs = T2.REGISTERED.candidate_pairs if cand else T2.REGISTERED.pairs
                for step in (s.row.newest, wb):
                    nid = node_name(nodes, step) or ""
                    pspec = {"consumer": "main.cycle_monitor.panel", "panel": panel, "rule": T2.REGISTERED.rule_string(),
                             "node": {k: ref(step).block()[k] for k in ("id", "step", "sha256")},
                             "members": [{"id": m["id"], "sha256": m["sha256"]} for m in members]}
                    play(Job(rid=panel_request_id(nid, panel or ""), kind=T2.REQUEST_KIND, purpose=T2.PURPOSE,
                              family=None, spec=pspec, cells=[(ref(step), mr, pairs) for mr in mrefs],
                              key_kind="panel"), root, writer, compute, run, emit, inflight, T2.REGISTERED.batch_pairs)
    finally:
        writer.close()
    return {"due": [s.row.grid for s in todo]}


def _decide(run: str, nodes: Mapping[int, Path], root: Path, writer: L.LedgerWriter, s: CheckState,
            rule: CM.MonitorRule, plan: CM.MonitorPlan, shas: Dict[int, str], compute: Any,
            emit: Callable[[str], None], inflight: int, ref: Callable[[int], Any], play: Callable[..., None]) -> None:
    rp = s.row
    assert s.newest_sha is not None and rp.newest is not None
    res, conf, verdict, final = _evaluate(run, nodes, root, rp, s.rid, s.newest_sha, rule, plan, shas)
    emit(f"[monitor] check {rp.grid:,}: {res.verdict} {res.signals} ({res.reason})")
    if not final:
        by_sha = {_sha_of(nodes, shas, st): st for st in nodes if st <= rp.newest}
        edges = CM.topup_edges(res, rule)
        spec = {"consumer": "main.cycle_monitor", "rule": rule.rule_string(), "run": run, "grid": rp.grid,
                "flagged": [{k: v for k, v in it.items() if k != "edges"} | {"edges": it["edges"]}
                            for it in res.flagged_items]}
        play(Job(rid=CM.topup_request_id(run, rp.grid, plan), kind=CM.REQUEST_KIND_TOPUP, purpose=CM.PURPOSE,
                  family=family_id(run, plan), spec=spec,
                  cells=[(ref(by_sha[a]), ref(by_sha[b]), rule.topup_pairs) for a, b in edges], key_kind="monitor_topup", key_salt=str(rp.grid)),
              root, writer, compute, run, emit, inflight, rule.batch_pairs)
        res, conf, verdict, final = _evaluate(run, nodes, root, rp, s.rid, s.newest_sha, rule, plan, shas)
        if not final:
            raise MonitorDriverError(f"check {rp.grid:,}: the top-up did not complete — re-run tick")
    got = _check_read(run, nodes, root, rp, plan, shas)
    try:
        d = writer.append_decision(kind=CM.DECISION_KIND, subject=s.newest_sha, consumed=got, verdict=verdict,
                                   rule=rule.rule_string(), rule_version=CM.RULE_VERSION, request_id=s.rid,
                                   family=family_id(run, plan), unique=True)
    except L.DecisionExistsError as e:
        d = e.decision
        if d["verdict"] != verdict:
            raise MonitorDriverError(f"{s.rid}: the ledger says {d['verdict']}, the rows give {verdict}") from None
    emit(f"[monitor] check {rp.grid:,}: {verdict}; decision {d['decision_id']}"
         + (f"; confirmation {conf}" if conf else ""))


def _t1_states(run: str, nodes: Mapping[int, Path], root: Path) -> Dict[int, str]:
    from main import plateau as P

    return {s.check.grid: s.state for s in P.check_states(run, nodes, root, T1.DEFAULT_PLAN, T1.REGISTERED)}


# ------------------------------------------------------------------------------------------------ the ladder read
def ladder_edges(run_dir: Path) -> Tuple[List[Tuple[str, int, str, int, List[float]]], Dict[str, Any]]:
    """The in-run snapshot ladder's MIRRORED edges (``a_p1`` from a's side + ``b_p1`` reversed), in the one regime
    most of them share; what was dropped is counted."""
    path = run_dir / "snapshot_ladder" / "games.jsonl"
    lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    regimes: Dict[str, int] = {}
    for g in lines:
        regimes[str(g.get("regime_id"))] = regimes.get(str(g.get("regime_id")), 0) + 1
    main_regime = max(regimes, key=lambda k: (regimes[k], k)) if regimes else None
    out, dropped = [], {"no_pair_counts": 0, "other_regime": 0}
    for g in lines:
        pc = g.get("pair_counts")
        if not g.get("mirrored") or not isinstance(pc, dict) or "a_p1" not in pc or "b_p1" not in pc:
            dropped["no_pair_counts"] += 1
            continue
        if str(g.get("regime_id")) != main_regime:
            dropped["other_regime"] += 1
            continue
        c = [float(x) + float(y) for x, y in zip(pc["a_p1"], reversed(pc["b_p1"]))]
        out.append((f"snap:{int(g['a'])}", int(g["a"]), f"snap:{int(g['b'])}", int(g["b"]), c))
    return out, {"lines": len(lines), "used": len(out), "dropped": dropped, "regime_id": main_regime,
                 "regimes": regimes}


def ladder_read(run_dir: Path, *, delta_steps: int = 10_000_000, slack: int = 500_000, from_step: int = 0,
                to_step: Optional[int] = None, window_steps: Optional[int] = None, rule: Optional[CM.MonitorRule] = None,
                t1_states: Optional[Mapping[int, str]] = None,
                emit: Callable[[str], None] = lambda m: print(m, file=sys.stderr, flush=True)) -> Dict[str, Any]:
    """The four signals on the in-run ladder as of each Δ check in [from_step, to_step] (module docstring), and the
    SIZE on the recent window (``plateau_t2.SIZE_RULE``: the ladder's nodes in the last 80M of grid, whatever
    ``window_steps`` says). ``window_steps``: the SIGNALS read only nodes within that many steps of the newest (a
    CURRENT-cycling view; the registered (a) reads the whole accumulated matrix). ``t1_states`` (grid → Tier-1
    state): also a DESCRIPTIVE two-tier signature (no panel)."""
    rule = rule or CM.REGISTERED
    edges, meta = ladder_edges(run_dir)
    steps = sorted({e[1] for e in edges} | {e[3] for e in edges})
    checks = []
    if not steps:
        return {"run": run_dir.name, "meta": meta, "checks": []}
    hi = steps[-1] if to_step is None else min(to_step, steps[-1] + slack)
    g = max(delta_steps, (from_step + delta_steps - 1) // delta_steps * delta_steps)
    while g <= hi:
        below = [s for s in steps if s < g + slack]
        newest = below[-1] if below else None
        if newest is None:
            g += delta_steps
            continue
        lo = -1 if window_steps is None else newest - window_steps
        sub = [e for e in edges if lo < e[1] <= newest and lo < e[3] <= newest]
        mx = CM.build_matrix(sub)
        key = f"snap:{newest}"
        if key not in mx.keys or sum(1 for e in sub if newest in (e[1], e[3])) < 2:
            checks.append({"grid": g, "newest_step": newest, "verdict": CM.NOT_APPLICABLE})
            g += delta_steps
            continue
        seed = CM.seed_of(f"ladder:{e[0]}:{e[2]}:{','.join(str(int(x)) for x in e[4])}" for e in sub)
        try:
            res = CM.signals(mx, key, rule, seed)
        except CM.MonitorError as err:                       # e.g. a window whose graph is not connected
            checks.append({"grid": g, "newest_step": newest, "verdict": "NOT_READ", "reason": str(err)})
            g += delta_steps
            continue
        size_rule = T2.SIZE_RULE
        win = [e for e in edges if e[1] <= newest and e[3] <= newest]
        size = CM.cyclic_size(CM.build_matrix(win), size_rule.min_step(g, delta_steps),
                              CM.seed_of(f"size:ladder:{e[0]}:{e[2]}:{','.join(str(int(x)) for x in e[4])}"
                                         for e in win), size_rule.n_boot, size_rule.one_sided)
        sst = T2.size_state(size, size_rule)
        checks.append({"grid": g, "newest_step": newest, "verdict": res.verdict, "signals": res.signals,
                       "summary": _summary(res), "flagged": res.flagged_items,
                       "lag_curve": (res.detail.get("lag") or {}).get("curve"),
                       "triangles": res.detail["triangles"], "size": sst, "size_detail": size,
                       "counted": T2.counted_monitor(res.verdict, sst)})
        emit(f"[monitor/ladder] {run_dir.name} {g:,} (newest {newest:,}): {res.verdict} {res.signals}; size "
             f"{_fmt_size(size)} {sst}")
        g += delta_steps
    out = {"run": run_dir.name, "source": "snapshot_ladder (DESCRIPTIVE: not the registered instrument)",
           "window_steps": window_steps, "meta": meta, "rule": rule.to_json(),
           "size_rule": T2.SIZE_RULE.to_json(), "checks": checks}
    if t1_states is not None:
        per = [{"grid": c["grid"], "tier1": t1_states.get(c["grid"]), "monitor": c["verdict"],
                "size": c.get("size"), "panel": T2.NOT_DECLARED} for c in checks]
        out["tier1"] = {str(k): v for k, v in sorted(t1_states.items())}
        out["signature"] = {**T2.signature(per, delta_steps), "note": "DESCRIPTIVE: the ladder's monitor read, "
                            "not the registered rows; no panel"}
    return out


def _fmt_size(size: Optional[Mapping[str, Any]]) -> str:
    if not size or not size.get("measured"):
        return "not measured" + (f" ({size.get('reason')})" if size else "")
    return f"{size['size_pp']:.2f} pp [{size['lower_pp']:.2f}, {size['upper_pp']:.2f}]"


# ------------------------------------------------------------------------------------------------ the CLI
def format_tier2(rep: Mapping[str, Any]) -> List[str]:
    lines = [f"  TIER 2 = the cycle monitor (family {rep['family']}) + the panel ({rep['panel'] or 'NOT DECLARED'})",
             f"  size = the recent window's per-edge cyclic SD, pp [one-sided 95 % bounds]; bar "
             f"{T2.SIZE_RULE.bar_pp} pp ({T2.SIZE_RULE.rule_string()})",
             f"  {'check':>13}  {'tier1':<11} {'monitor':<13} {'edges':>6} {'hodge p':>8} {'lag z/crit':>12} "
             f"{'tri minp':>9} {'nash':>5}  {'size':<24} {'counted':<13} {'panel':<12} note"]
    for c in rep["checks"]:
        ch, sm = c["check"], c["check"]["summary"]
        lz = f"{sm['lag_z_max']:.2f}/{sm['lag_z_crit']:.2f}" if sm.get("lag_z_max") is not None else ""
        lines.append(f"  {c['grid']:>13,}  {str(c['tier1'] or '-'):<11} {str(c['monitor']):<13} "
                     f"{ch['edges_done']:>2}/{ch['edges_total']:<3} {sm.get('hodge_p', ''):>8} {lz:>12} "
                     f"{'' if sm.get('tri_min_p') is None else sm['tri_min_p']:>9} {sm.get('nash_support', ''):>5}  "
                     f"{_fmt_size(c.get('size_detail')) if c.get('size_detail') else '-':<24} "
                     f"{str(c.get('counted') or '-'):<13} {c['panel']:<12} {ch['note']}")
    s = rep["signature"]
    lines.append(f"TWO-TIER STATUS: {s['status']}" + (f" (as of the {s['as_of_check']:,} check)"
                                                    if s.get("as_of_check") else "") + f" — {s['why']}")
    return lines


def _run(a: argparse.Namespace) -> Tuple[Path, List[Path], Path]:
    from main.ops.run_ref import resolve_run_dir

    run_dir = resolve_run_dir(a.run)
    lineage = [resolve_run_dir(x) for x in (getattr(a, "lineage_run", None) or [])]
    root = Path(L.check_write_root(a.out)) if getattr(a, "out", None) is not None else Path(L.archive_ledger_root())
    return run_dir, lineage, root


def cmd_status(a: argparse.Namespace) -> int:
    run_dir, lineage, root = _run(a)
    nodes = node_series(run_dir, lineage)
    t1_root = Path(L.check_write_root(a.tier1_out)) if a.tier1_out else Path(L.archive_ledger_root())
    rep = tier2_by_check(run_dir.name, nodes, root, _t1_states(run_dir.name, nodes, t1_root), panel=a.panel)
    print(json.dumps(rep, indent=1, sort_keys=True, default=str) if a.json else "\n".join(
        [f"[monitor] {run_dir.name}: rule {CM.REGISTERED.rule_string()}"] + format_tier2(rep)))
    return 0


def cmd_tick(a: argparse.Namespace) -> int:
    from main.h2h import play as PL

    run_dir, lineage, _root = _run(a)
    compute = PL.Compute(device=a.device, backend=a.backend, n_envs=a.n_envs, threads=a.threads,
                         torch_threads=a.torch_threads, front=a.front, profile=a.profile)
    tick(run_dir, lineage, a.out, compute, panel=a.panel, max_checks=a.max_checks, dry_run=a.dry_run,
         inflight=a.inflight_batches, grids=a.grid,
         t1_root=Path(L.check_write_root(a.tier1_out)) if a.tier1_out else None)
    return cmd_status(a) if not a.dry_run else 0


def cmd_ladder(a: argparse.Namespace) -> int:
    run_dir, _lineage, _root = _run(a)
    t1 = None
    if a.tier1:
        t1_root = Path(L.check_write_root(a.tier1_out)) if a.tier1_out else Path(L.archive_ledger_root())
        t1 = _t1_states(run_dir.name, node_series(run_dir, []), t1_root)
    rep = ladder_read(run_dir, from_step=a.from_step, to_step=a.to_step, window_steps=a.window_steps, t1_states=t1)
    if a.json:
        print(json.dumps(rep, indent=1, sort_keys=True, default=str))
        return 0
    print(f"[monitor/ladder] {rep['run']}: {rep.get('source', '')}; {rep['meta']}")
    for c in rep["checks"]:
        sm = c.get("summary") or {}
        print(f"  {c['grid']:>13,}  newest {c['newest_step']:>11,}  {c['verdict']:<13} {c.get('signals', '')}  "
              f"nodes {sm.get('nodes')} edges {sm.get('edges')} hodge p {sm.get('hodge_p')} share "
              f"{sm.get('cyclic_share')} lag z {sm.get('lag_z_max')}/{sm.get('lag_z_crit')} tri {sm.get('tri_n')} "
              f"minp {sm.get('tri_min_p')} nash {sm.get('nash_support')} (post {sm.get('nash_post_support')})  "
              f"size {_fmt_size(c.get('size_detail'))} {c.get('size', '')} -> counted {c.get('counted', '')}")
    if "signature" in rep:
        s = rep["signature"]
        print(f"TWO-TIER STATUS (DESCRIPTIVE, ladder + Tier 1, no panel): {s['status']}"
              + (f" (as of the {s['as_of_check']:,} check)" if s.get("as_of_check") else "") + f" — {s['why']}")
    return 0


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m main.cycle_monitor", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, hlp in (("status", "every check's monitor + panel state and the two-tier signature (plays nothing)"),
                      ("tick", "play every due row / top-up / panel cell, decide, then print the status"),
                      ("ladder", "DESCRIPTIVE: the four signals on the in-run snapshot ladder, per Δ check")):
        p = sub.add_parser(name, help=hlp)
        p.add_argument("run", help="a run NAME (in the archive) or a run DIRECTORY")
        if name == "ladder":
            p.add_argument("--from-step", type=int, default=0)
            p.add_argument("--to-step", type=int, default=None)
            p.add_argument("--window-steps", type=int, default=None,
                           help="only nodes within this many steps of the check's newest (a CURRENT-cycling view)")
            p.add_argument("--tier1", action="store_true",
                           help="also read Tier 1 from the ledger and print a DESCRIPTIVE two-tier status")
            p.add_argument("--tier1-out", default=None, help="the ledger root Tier 1's rows are read from")
            p.add_argument("--json", action="store_true")
            continue
        p.add_argument("--lineage-run", action="append", default=None,
                       help="an ANCESTOR run whose checkpoints are nodes too (repeatable; the run's own file wins)")
        p.add_argument("--panel", default=None, help="a REGISTERED panel (designs/research_state/plateau_panels.json)")
        p.add_argument("--out", default=None, help="the ledger ROOT (default: the archive's; others under models/ "
                                                   "are refused)")
        p.add_argument("--json", action="store_true")
        p.add_argument("--tier1-out", default=None, help="the ledger root Tier 1's rows are read from (default: the "
                                                         "archive's, where main.plateau tick writes them)")
        if name == "tick":
            p.add_argument("--max-checks", type=int, default=None)
            p.add_argument("--grid", type=int, action="append", default=None,
                           help="work only this check (repeatable); a check is DECIDED only once every row up to "
                                "it is complete")
            p.add_argument("--dry-run", action="store_true")
            p.add_argument("--inflight-batches", type=int, default=DEFAULT_INFLIGHT)
            p.add_argument("--device", default="cpu", help="cpu (default); cuda needs the GPU LEASE")
            p.add_argument("--backend", default="", choices=("", "eager", "graph", "aot"))
            p.add_argument("--n-envs", type=int, default=64)
            p.add_argument("--threads", type=int, default=4)
            p.add_argument("--torch-threads", type=int, default=4)
            p.add_argument("--front", default="proc", choices=("proc", "ffi"))
            p.add_argument("--profile", default="release", choices=("release", "selfcheck"))
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    a = _parser().parse_args(argv)
    return {"status": cmd_status, "tick": cmd_tick, "ladder": cmd_ladder}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
