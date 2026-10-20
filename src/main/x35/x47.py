"""X47's REGISTERED readout (B-X47-READ, AM-4 of the 2026-10-19 review, ACCEPTED by the orchestrator): the comparison
point at EQUAL EFFECTIVE LEARN GPU-TIME (the PRIMARY) and the matched-step SECONDARY, with the verdict classes of
``designs/endstate/design_bot_ideas_screen.md`` §1.2-§1.4. X47 = the toy fork of T* grown from 2 to 4 attention
rounds (``kit`` arm ``toy_x47``), read against the toy C0 at the same pin and seeds (3601-3603; look 2 3604-3605).
Pure functions plus a read-only loader; no game is played here (S comes from ``main.x35.readout.population_score``).

    python -m main.x35.x47 seq --c0-run <dir> ... --x47-run <dir> ...   # the budget, each seed's s_eq, ρ (JSON)
    python -m main.x35.x47 decide <read.json>                          # the verdict (JSON; ``read`` documents it)

EFFECTIVE LEARN TIME (``design_compute_efficiency.md`` §5.4). Σ = Σ ``rollout/collect_ms`` / 1000 + Σ
``lifecycle/update_wall_s`` over the fork's OWN TensorBoard rows (``main.x35.tbrows``: the inherited file and every
step <= the fork step excluded), along the SURVIVING step path, restart-aware by ``time_to_step.py``'s method: a
process (one event file) whose first collect row is at s_1 and second at s_2 RESUMED at s_r = s_1 − (s_2 − s_1);
every row of an EARLIER process at a step > s_r is superseded (re-done work is counted once). Excluded by
construction: process start, compile, checkpoint writes and in-learn eval.

THE NODE RULE (bot screen §1.2). The budget B = the MEAN over the C0 seeds of Σ at their finals. A node is a 1M
checkpoint ``checkpoints/checkpoint_<N>_steps.zip`` (N > the fork step), or the final (Σ at the last row). s_eq of an
X47 seed = the LARGEST node whose Σ(N) <= B; a node whose Σ lies within :data:`RULE8_EPS` of B is NOT taken (the next
lower node is; rule 8). No node qualifies ⇒ INCONCLUSIVE. ρ = (mean X47 Σ per trained step at the final) / (B per
mean C0 trained step); when ρ <= 1.02 (within 1e-9 of 1.02 counts as ABOVE) the two reads coincide and only the
final is read (every s_eq := the final).

THE VERDICT (§1.4), at look 1 (3 seeds per side) and look 2 (5, AM-4). PRIMARY = R1 (``readout.decide_r1``) of each X47
seed's S at its s_eq node against the C0 seeds' S at their finals; SECONDARY = R1 of the X47 finals against the same
C0 finals. In order:
* INCONCLUSIVE: the primary is INCONCLUSIVE;
* CONTINUE: look 1's primary outcome is CONTINUE (UNDECIDED with Δ̂ > 0), whatever the secondary says: it licenses
  X47 s3604-s3605 with toy C0 s3604-s3605 (``batch.stage_status``);
* primary GAIN: ADOPT-CANDIDATE when every REQUIRED guard (G-F, G-C) PASSes, TRADE when one FAILs, INCONCLUSIVE when
  one is not read;
* primary NULL or UNDECIDED (final): PER-STEP ONLY when the secondary's class is GAIN (reported to the owner as a
  throughput-dependent result; no adoption), INCONCLUSIVE when the secondary is, else the primary's class;
* primary LOSS: LOSS.
Nothing is adopted by this rule alone.

DECLARED HERE, BEFORE ANY REAL READ (choices the design leaves open; each a FINDING of B-X47-READ):
* The two tags are summed over their OWN rows: ``update_wall_s`` is recorded after an update and dumped with the NEXT
  rollout's row (SB3 dumps before ``train()``), so a row-wise pairing would mis-attribute; Σ(N) counts every row
  LOGGED at a step <= N, which lags the weights a node holds by at most one iteration (≈ 0.1 node on the toy), the
  same convention on both arms. A segment whose two tags' row counts differ by more than 1, or a run with no row of
  a tag, is INCONCLUSIVE (absence is not zero).
* The final is a node (its Σ is every logged row), so an X47 seed whose total Σ <= B reads its final.
* The REQUIRED guards on the toy are G-F (on the PRIMARY's nodes: s_eq × C0 finals) and G-C; G-K is REPORTED only
  (whether ``main.anchors`` loads the toy is UNVERIFIED, X37c). G-E is not run on the toy.
* REPORTED, NO RULE READS THEM (review F-RV-9; the orchestrator did not rule its fixes): s_eq on the UPDATE-ONLY basis
  (Σ ``update_wall_s``), and ρ split into collect and update. A class that differs between the two bases is a
  FINDING for the orchestrator, not an UNDECIDED here.
"""
from __future__ import annotations

import glob
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from main.x35 import readout as R

COLLECT_TAG = "rollout/collect_ms"
UPDATE_TAG = "lifecycle/update_wall_s"
RULE8_EPS = R.RULE8_EPS
RHO_COINCIDE = 1.02
REQUIRED_GUARDS = ("G-F", "G-C")
PER_STEP_ONLY = "PER-STEP ONLY"
GAIN, NULL, LOSS, UNDECIDED = R.GAIN, R.NULL, R.LOSS, R.UNDECIDED
CONTINUE, INCONCLUSIVE, ADOPT, TRADE = R.CONTINUE, R.INCONCLUSIVE, R.ADOPT, R.TRADE
_CKPT = re.compile(r"checkpoint_(\d+)_steps\.zip$")

#: one process's rows: ``(collect rows, update rows)``, each ``[(step, seconds)]`` sorted by step.
SegRows = Tuple[Sequence[Tuple[int, float]], Sequence[Tuple[int, float]]]


class X47InputError(ValueError):
    """An input the registered statistic is not defined on (named; the read is INCONCLUSIVE)."""


# ------------------------------------------------------------------------------------ effective learn time
@dataclass(frozen=True)
class LearnTime:
    """One run's surviving-path rows: ``collect`` / ``update`` = ``[(step, seconds)]`` sorted by step."""

    run: str
    fork_step: int
    collect: Tuple[Tuple[int, float], ...]
    update: Tuple[Tuple[int, float], ...]
    nodes: Tuple[int, ...]

    @property
    def final_step(self) -> int:
        return max(s for s, _ in self.collect + self.update)

    def sigma(self, step: Optional[int] = None, *, basis: str = "both") -> float:
        """Σ seconds of the rows logged at steps <= ``step`` (None = every row: the final)."""
        def part(rows: Sequence[Tuple[int, float]]) -> float:
            return math.fsum(v for s, v in rows if step is None or s <= step)
        if basis == "update":
            return part(self.update)
        if basis == "collect":
            return part(self.collect)
        return part(self.collect) + part(self.update)

    def per_step(self, basis: str = "both") -> float:
        return self.sigma(basis=basis) / (self.final_step - self.fork_step)


def surviving_path(segments: Sequence[SegRows], fork_step: int) -> Tuple[List[Tuple[int, float]],
                                                                        List[Tuple[int, float]]]:
    """The restart-aware surviving path (module docs) of chronologically ordered segments; collect in SECONDS.
    Rows at steps <= ``fork_step`` are dropped first (the inherited prefix)."""
    col: List[Tuple[int, float]] = []
    upd: List[Tuple[int, float]] = []
    for i, (c_rows, u_rows) in enumerate(segments):
        c = sorted((int(s), float(v) / 1000.0) for s, v in c_rows if int(s) > fork_step)
        u = sorted((int(s), float(v)) for s, v in u_rows if int(s) > fork_step)
        if not c and not u:
            continue
        if abs(len(c) - len(u)) > 1:
            raise X47InputError(f"segment {i}: {len(c)} {COLLECT_TAG} rows vs {len(u)} {UPDATE_TAG} rows "
                                "(more than the one-row dump lag): an instrument is missing rows")
        if not c:
            raise X47InputError(f"segment {i}: no {COLLECT_TAG} row (absence is not zero)")
        s_resume = c[0][0] - (c[1][0] - c[0][0]) if len(c) > 1 else c[0][0] - 1
        col = [r for r in col if r[0] <= s_resume] + c
        upd = [r for r in upd if r[0] <= s_resume] + u
    for v in (*col, *upd):
        if not math.isfinite(v[1]) or v[1] < 0:
            raise X47InputError(f"a non-finite or negative time {v[1]!r} at step {v[0]}")
    if not col or not upd:
        raise X47InputError(f"no {COLLECT_TAG if not col else UPDATE_TAG} row after the fork step {fork_step}")
    return col, upd


def checkpoint_nodes(run_dir: str, fork_step: int) -> List[int]:
    """The run's checkpoint steps after the fork, sorted."""
    out = []
    for f in glob.glob(os.path.join(run_dir, "checkpoints", "checkpoint_*_steps.zip")):
        m = _CKPT.search(f)
        if m and int(m.group(1)) > fork_step:
            out.append(int(m.group(1)))
    return sorted(out)


def load_run(run_dir: str) -> LearnTime:
    """A real run's :class:`LearnTime` (read-only): its own TensorBoard segments and its checkpoint nodes."""
    from main.x35.tbrows import fork_step, load_segments

    fork = fork_step(run_dir)
    if fork is None:
        raise X47InputError(f"{run_dir}: the fork step is unknown (the inherited rows cannot be excluded)")
    segs = load_segments(run_dir, [COLLECT_TAG, UPDATE_TAG], after_step=fork)
    col, upd = surviving_path([(s.rows.get(COLLECT_TAG, []), s.rows.get(UPDATE_TAG, [])) for s in segs], fork)
    return LearnTime(os.path.basename(os.path.normpath(run_dir)), fork, tuple(col), tuple(upd),
                     tuple(checkpoint_nodes(run_dir, fork)))


# ------------------------------------------------------------------------------------------- the node rule
@dataclass(frozen=True)
class SEq:
    run: str
    node: Optional[int]          # None = no node qualifies (INCONCLUSIVE)
    is_final: bool
    sigma_s: Optional[float]
    skipped_rule8: Tuple[int, ...]
    node_update_basis: Optional[int]

    def as_dict(self) -> Dict[str, Any]:
        return {"run": self.run, "s_eq": self.node, "is_final": self.is_final, "sigma_s": self.sigma_s,
                "skipped_rule8": list(self.skipped_rule8), "s_eq_update_only_basis_reported": self.node_update_basis}


def _largest_under(run: LearnTime, budget: float, basis: str) -> Tuple[Optional[int], bool, List[int]]:
    cands = [(n, run.sigma(n, basis=basis), False) for n in run.nodes if n < run.final_step]
    cands.append((run.final_step, run.sigma(basis=basis), True))
    skipped: List[int] = []
    best: Optional[Tuple[int, bool]] = None
    for n, sig, fin in sorted(cands):
        if abs(sig - budget) <= RULE8_EPS:
            skipped.append(n)
            continue
        if sig < budget:
            best = (n, fin)
    return (best[0], best[1], skipped) if best else (None, False, skipped)


def s_eq(run: LearnTime, budget: float, *, budget_update: Optional[float] = None) -> SEq:
    """The node rule (module docs) for one X47 seed against the C0 budget ``budget`` (seconds)."""
    node, fin, skipped = _largest_under(run, budget, "both")
    upd = _largest_under(run, budget_update, "update")[0] if budget_update is not None else None
    return SEq(run.run, node, fin, run.sigma(node) if node is not None else None, tuple(skipped), upd)


@dataclass(frozen=True)
class Comparison:
    budget_s: float
    rho: float
    coincide: bool
    near_rho: bool
    seq: Tuple[SEq, ...]
    rho_collect: float
    rho_update: float

    def as_dict(self) -> Dict[str, Any]:
        return {"budget_s": self.budget_s, "rho": self.rho, "coincide": self.coincide, "rho_near_1_02": self.near_rho,
                "s_eq": {s.run: s.as_dict() for s in self.seq},
                "reported_only": {"rho_collect": self.rho_collect, "rho_update": self.rho_update}}


def compare(c0: Sequence[LearnTime], x47: Sequence[LearnTime]) -> Comparison:
    """The budget, ρ and every X47 seed's s_eq (module docs)."""
    if not c0 or not x47:
        raise X47InputError("the comparison needs >= 1 C0 run and >= 1 X47 run")
    budget = math.fsum(r.sigma() for r in c0) / len(c0)
    b_upd = math.fsum(r.sigma(basis="update") for r in c0) / len(c0)

    def ratio(basis: str) -> float:
        on = math.fsum(r.per_step(basis) for r in x47) / len(x47)
        off = math.fsum(r.per_step(basis) for r in c0) / len(c0)
        return on / off

    rho = ratio("both")
    near = abs(rho - RHO_COINCIDE) <= RULE8_EPS
    coincide = rho < RHO_COINCIDE and not near
    if coincide:
        seq = tuple(SEq(r.run, r.final_step, True, r.sigma(), (), r.final_step) for r in x47)
    else:
        seq = tuple(s_eq(r, budget, budget_update=b_upd) for r in x47)
    return Comparison(budget, rho, coincide, near, seq, ratio("collect"), ratio("update"))


# ---------------------------------------------------------------------------------------------- the verdict
@dataclass(frozen=True)
class X47Verdict:
    look: int
    verdict: str
    primary: R.R1Decision
    secondary: Optional[R.R1Decision]
    guards: Tuple[R.GuardResult, ...]
    reasons: Tuple[str, ...]
    reported: Dict[str, Any]

    def as_dict(self) -> Dict[str, Any]:
        return {"arm": "X47", "look": self.look, "verdict": self.verdict, "primary": self.primary.as_dict(),
                "secondary": self.secondary.as_dict() if self.secondary else None,
                "guards": {g.name: g.as_dict() for g in self.guards}, "reasons": list(self.reasons),
                "s_eq": self.reported.get("s_eq"), "rho": self.reported.get("rho"), "reported": self.reported}


def decide(look: int, primary: R.R1Decision, secondary: Optional[R.R1Decision], guards: Sequence[R.GuardResult], *,
           reported: Optional[Dict[str, Any]] = None) -> X47Verdict:
    """The registered verdict (module docs)."""
    by = {g.name: g for g in guards}
    gs = tuple(by.get(n) or R.GuardResult(n, INCONCLUSIVE, reasons=("not read",)) for n in REQUIRED_GUARDS)
    rest = tuple(g for g in guards if g.name not in REQUIRED_GUARDS)
    rep = dict(reported or {})

    def out(v: str, *why: str) -> X47Verdict:
        return X47Verdict(look, v, primary, secondary, gs + rest, tuple(why), rep)

    if primary.outcome == INCONCLUSIVE:
        return out(INCONCLUSIVE, "the primary (at s_eq) is INCONCLUSIVE", *primary.reasons)
    if primary.outcome == CONTINUE:
        return out(CONTINUE, f"the primary is UNDECIDED with Δ̂ > 0 at look {look}: X47 and toy C0 s3604-s3605")
    if primary.cls == GAIN:
        open_ = [g.name for g in gs if g.status == INCONCLUSIVE]
        if open_:
            return out(INCONCLUSIVE, f"primary GAIN, but required guard(s) {open_} INCONCLUSIVE")
        failed = [g.name for g in gs if g.status == R.FAIL]
        if failed:
            return out(TRADE, f"primary GAIN, but guard(s) {failed} FAIL")
        return out(ADOPT, "primary GAIN at equal effective learn time; every required guard PASSES")
    if primary.cls in (NULL, UNDECIDED):
        if secondary is None or secondary.outcome == INCONCLUSIVE:
            return out(INCONCLUSIVE, f"primary {primary.cls}: PER-STEP ONLY needs the secondary, which is not read",
                       *(secondary.reasons if secondary else ()))
        if secondary.cls == GAIN:
            return out(PER_STEP_ONLY, f"primary {primary.cls}, secondary (matched steps) GAIN: throughput-dependent")
        return out(str(primary.cls), f"primary {primary.cls}; secondary {secondary.cls}")
    return out(str(primary.cls), f"primary {primary.cls}")


def read(doc: Mapping[str, Any]) -> X47Verdict:
    """The whole decision from one JSON-shaped read (``decide``'s input):

    ``{"look": 1, "c0_final": {seed: S_pp}, "x47_s_eq": {seed: S_pp}, "x47_final": {seed: S_pp},
    "g_f": {"h_pp": [[...]]}, "g_c": {"b1": [e, lo, hi], "b2": [e, lo, hi]}, "g_k": {...}, "comparison": {...}}``

    ``x47_s_eq`` holds each X47 seed's S at its s_eq node (``compare``'s output, carried in ``comparison``); when ρ
    coincides they are the finals' S. G-F's matrix is the primary's (s_eq nodes × C0 finals)."""
    look = int(doc["look"])
    off = dict(doc.get("c0_final") or {})
    primary = R.decide_r1(look, dict(doc.get("x47_s_eq") or {}), off)
    fin = doc.get("x47_final")
    secondary = R.decide_r1(look, dict(fin), off) if fin else None
    gc = doc.get("g_c") or {}
    guards = [R.guard_f((doc.get("g_f") or {}).get("h_pp"), look), R.guard_c(gc.get("b1"), gc.get("b2"))]
    if doc.get("g_k") is not None:
        g = R.guard_k(doc["g_k"])
        guards.append(R.GuardResult("G-K (reported)", g.status, g.detail, g.near, g.reasons))
    rep: Dict[str, Any] = {}
    comp = doc.get("comparison") or {}
    rep["s_eq"] = comp.get("s_eq")
    rep["rho"] = comp.get("rho")
    rep["comparison"] = comp or None
    return decide(look, primary, secondary, guards, reported=rep)


def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import json
    import sys

    ap = argparse.ArgumentParser(prog="python -m main.x35.x47", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("seq", help="the budget, each X47 seed's s_eq and ρ from the runs' own TensorBoard (read-only)")
    q.add_argument("--c0-run", action="append", required=True)
    q.add_argument("--x47-run", action="append", required=True)
    d = sub.add_parser("decide", help="the verdict of one read JSON (see main.x35.x47.read)")
    d.add_argument("input")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "seq":
            comp = compare([load_run(r) for r in a.c0_run], [load_run(r) for r in a.x47_run])
            json.dump(comp.as_dict(), sys.stdout, indent=1)
        else:
            with open(a.input) as f:
                json.dump(read(json.load(f)).as_dict(), sys.stdout, indent=1, default=str)
        print()
        return 0
    except (X47InputError, R.ReadoutInputError) as exc:
        print(f"[x35 x47] INCONCLUSIVE: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_main())
