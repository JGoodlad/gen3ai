"""X47's REGISTERED readout (B-X47-READ, AM-4 of the 2026-10-19 review, ACCEPTED by the orchestrator): the comparison
point at EQUAL EFFECTIVE LEARN GPU-TIME (the PRIMARY) and the matched-step SECONDARY, with the verdict classes of
``designs/endstate/design_bot_ideas_screen.md`` §1.2-§1.4. X47 = the toy fork of T* grown from 2 to 4 attention
rounds (``kit`` arm ``toy_x47``), read against the toy C0 at the same pin and seeds (3601-3603; look 2 3604-3605).
Pure functions plus a read-only loader; no game is played here (S comes from ``main.x35.readout.population_score``).

    python -m main.x35.x47 seq --look 1 --c0-run <dir> ... --x47-run <dir> ...  # B, each seed's s_eq, ρ (JSON)
    python -m main.x35.x47 seq --look 2 ... --prior <look-1 seq.json>          # + which nodes moved (AM-X47-8)
    python -m main.x35.x47 decide <read.json>                                 # the verdict (JSON; ``read``)
    python -m main.x35.x47 norms --run <dir>                                  # the descriptor (AM-X47-10)
    python -m main.x35.x47 policy-diff --run <dir> --bank <bank_x26>          # the policy descriptor (AM-X47b-3)

EFFECTIVE LEARN TIME (``design_compute_efficiency.md`` §5.4). Σ = Σ ``rollout/collect_ms`` / 1000 + Σ
``lifecycle/update_wall_s`` over the fork's OWN TensorBoard rows (``main.x35.tbrows``: the inherited file and every
step <= the fork step excluded), along the SURVIVING step path, restart-aware by ``time_to_step.py``'s method: a
process (one event file) whose first collect row is at s_1 and second at s_2 RESUMED at s_r = s_1 − (s_2 − s_1);
every row of an EARLIER process at a step > s_r is superseded (re-done work is counted once). Excluded by
construction: process start and compile (before the first row). INCLUDED (AM-X47-4, ruled 2026-10-20): in-learn
eval, checkpoint writes and snapshot saves -- every callback runs inside the collect timer
(``instrumented_ppo/rollout_probes.py``), so Σ is the training loop's wall time (≈ 0.999 of the first-to-last-row
span on toy C0 s3601). It is NOT corrected (the measured bias is ≈ 0.25M steps of s_eq); each run's eval share
(Σ ``eval/wall_sec`` on the same surviving path ÷ Σ) is REPORTED beside ρ.

WHICH RUNS ``seq`` READS (AM-X47-3, AM-X47-12). Every run, C0 or X47, must be FINISHED and K9(b)-clean, else ``seq``
is INCONCLUSIVE (exit 2, :func:`check_finished`): ``final_model.zip`` exists; ``metadata.json`` ``num_timesteps`` = the
step of the run's last TensorBoard row; that step >= the run's own ``--steps`` (``metadata.json`` ``cli_args.steps``);
and its AM-6 record ``<stage>/k9b/<run>.json`` reads PASS. An X47 run must also keep its 1M nodes (ruled: X47 arms are
not thinned before the read): a gap > 1.5 × ``checkpoint_every_steps`` between consecutive nodes (the fork step and
the final included) is a refusal (:func:`check_nodes_dense`).

WHICH FILES ARE READ (AM-X47-11). "The final" = ``final_model.zip`` (its ``num_timesteps`` = the last row's step), NOT
the last ``checkpoint_<N>_steps.zip`` (toy C0 s3601: 360,098,229 against 360,000,067). An s_eq node =
``checkpoints/checkpoint_<N>_steps.zip``, or ``final_model.zip`` when ``is_final``; ``seq`` prints each one's ``file``.

THE NODE RULE (bot screen §1.2). The budget B = the MEAN over the C0 seeds of Σ at their finals. A node is a 1M
checkpoint ``checkpoints/checkpoint_<N>_steps.zip`` (N > the fork step), or the final (Σ at the last row). s_eq of an
X47 seed = the LARGEST node whose Σ(N) <= B; a node whose Σ lies within :data:`RULE8_EPS` of B is NOT taken (the next
lower node is; rule 8). No node qualifies ⇒ INCONCLUSIVE. ρ = (mean X47 Σ per trained step at the final) / (B per
mean C0 trained step); when ρ <= 1.02 (within 1e-9 of 1.02 counts as ABOVE) the two reads coincide and only the
final is read (every s_eq := the final).

LOOK 2 (AM-X47-8). ``seq --look 1`` reads exactly 3 C0 + 3 X47 runs, ``--look 2`` exactly 5 + 5: at look 2, B and every
X47 seed's s_eq are RECOMPUTED over all five C0 seeds. S is read at each new node; a look-1 S is reused only when its
seed's node did not move (``--prior <look-1 seq JSON>`` records both looks' nodes and which seeds are re-read,
:func:`moved_nodes`).

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
Nothing is adopted by this rule alone. Every verdict prints what it does to X47b (AM-X47-2, :data:`X47B_ACTION`); an
UNDECIDED line states its own look's detectable effect (AM-X47c-1, ``readout.r1_detectable_effect``).

REPORTED, NEVER DECIDING (registered 2026-10-20, before any read):
* AM-X47-5: a PER-STEP ONLY carries D = mean over seeds of (S_final − S_s_eq) with its two-sided 95 % t interval on
  n − 1 df (:func:`per_step_gap`): "throughput-dependent" only when the lower bound is > 0 (within 1e-9 of 0 is not;
  zero variance has no interval), else "BOUNDARY".
* AM-X47-7: each R member's Δ (X47 at s_eq − C0 finals) with its Welch interval at the look's 97.5 %
  (:func:`per_member`), the out-of-lineage member :data:`OUT_OF_LINEAGE` on its own; a primary GAIN whose
  ``pstar_x26`` Δ̂ <= 0 is labelled "in-lineage".
* AM-X47-10: the new rounds' weight norms (``python -m main.x35.x47 norms``, :func:`round_norms`).
* AM-X47b-3: the POLICY-DIFFERENCE descriptor (``python -m main.x35.x47 policy-diff``, ``x47_report.policy_diff``):
  per node, the mean KL(π_node ‖ π_zeroed) on a fixed bank, π_zeroed = the same weights with every new round's
  ``out_proj`` / ``linear2`` (weight and bias) zeroed, which makes each extra round exactly the identity.

DECLARED HERE, BEFORE ANY REAL READ (choices the design leaves open; each a FINDING of B-X47-READ):
* The two tags are summed over their OWN rows: ``update_wall_s`` is recorded after an update and dumped with the NEXT
  rollout's row (SB3 dumps before ``train()``), so a row-wise pairing would mis-attribute; Σ(N) counts every row
  LOGGED at a step <= N, which lags the weights a node holds by at most one iteration (≈ 0.1 node on the toy), the
  same convention on both arms. A segment whose two tags' row counts differ by more than 1, or a run with no row of
  a tag, is INCONCLUSIVE (absence is not zero).
* The final is a node (its Σ is every logged row), so an X47 seed whose total Σ <= B reads its final.
* The REQUIRED guards on the toy are G-F (on the PRIMARY's nodes: s_eq × C0 finals) and G-C; G-K is REPORTED only
  (whether ``main.anchors`` loads the toy is UNVERIFIED, X37c). G-E is not run on the toy.
* REPORTED, NO RULE READS THEM (review F-RV-9, CLOSED by AM-X47-9: no flip rule): s_eq on the UPDATE-ONLY basis
  (Σ ``update_wall_s``), never read for S, and ρ split into collect and update.
"""
from __future__ import annotations

import glob
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from main.x35 import readout as R
from main.x35 import x47_report as XR

COLLECT_TAG = "rollout/collect_ms"
UPDATE_TAG = "lifecycle/update_wall_s"
#: in-learn eval's wall time per cycle: INSIDE ``collect_ms`` (AM-X47-4), reported as a share of Σ only.
EVAL_TAG = "eval/wall_sec"
#: the stage directory whose ``k9b/<run>.json`` records AM-X47-3 reads (``--stage`` overrides).
STAGE_REL = "designs/research_state/measurements/x35_breakers_stage_2026-10-17"
#: AM-X47-12: an X47 run's consecutive nodes may be at most this many ``checkpoint_every_steps`` apart.
MAX_NODE_GAP_EVERY = 1.5
RULE8_EPS = R.RULE8_EPS
RHO_COINCIDE = 1.02
REQUIRED_GUARDS = ("G-F", "G-C")
PER_STEP_ONLY = "PER-STEP ONLY"
GAIN, NULL, LOSS, UNDECIDED = R.GAIN, R.NULL, R.LOSS, R.UNDECIDED
CONTINUE, INCONCLUSIVE, ADOPT, TRADE = R.CONTINUE, R.INCONCLUSIVE, R.ADOPT, R.TRADE
_CKPT = re.compile(r"checkpoint_(\d+)_steps\.zip$")
FINAL_FILE = "final_model.zip"

#: AM-X47-2 (ACCEPTED as proposed, 2026-10-20): what each FINAL class does to X47b. UNDECIDED splits on the sign of
#: the primary's Δ̂ (Δ̂ > 0 by AM-X47-2; Δ̂ <= 0, within 1e-9 of 0 included, by AM-X47b-4); a look-1 CONTINUE is not
#: final. AM-X47c-1: the UNDECIDED lines are TEMPLATES; ``{mde}`` is the CURRENT look's detectable effect
#: (``readout.r1_detectable_effect``: 4.6 pp after look 2, ≈ 6.8 at look 1), never a constant.
X47B_ACTION: Dict[str, str] = {
    ADOPT: "X47b RUNS (licensed by ADOPT-CANDIDATE)",
    TRADE: "X47b only on the orchestrator's ruling (a required guard FAILed)",
    PER_STEP_ONLY: "X47b RUNS (licensed by PER-STEP ONLY)",
    UNDECIDED + "+": ("X47b keeps its rank-9 conditional slot, unlicensed but not dropped; report: no effect >= {mde:.1f} "
                      "pp detected (look {look}, σ {sigma} pp, 80 %) -- never 'depth does not help'"),
    UNDECIDED + "-": ("X47b LEAVES its rank-9 conditional slot and is DEPRIORITISED in the backlog (AM-X47b-4); "
                      "report: no effect >= {mde:.1f} pp detected; point estimate <= 0 (look {look}, σ {sigma} pp, "
                      "80 %) -- never 'depth does not help'"),
    NULL: "X47b drops in rank, not out (the toy only ranks before calibration); a verdict on THIS growth recipe only",
    LOSS: "X47b is not run on this evidence",
    CONTINUE: "not final: look 2 (T1b) first",
    INCONCLUSIVE: "nothing: fix the named input and re-read",
}


def x47b_action(verdict: str, primary: "R.R1Decision") -> str:
    """AM-X47-2's line for a verdict (:data:`X47B_ACTION`); an UNDECIDED line states the primary's OWN look's
    detectable effect (AM-X47c-1)."""
    if verdict == UNDECIDED:
        pos = primary.stat is not None and R._clears(primary.stat.delta, 0.0, above=True)[0]
        return X47B_ACTION[UNDECIDED + ("+" if pos else "-")].format(
            mde=R.r1_detectable_effect(primary.look), look=primary.look, sigma=R.POWER_SIGMA_PP)
    return X47B_ACTION.get(verdict, X47B_ACTION[INCONCLUSIVE])

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
    run_dir: str = ""
    eval_s: Optional[float] = None          # Σ eval/wall_sec on the surviving path (None = the run logged none)

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

    def node_file(self, node: int) -> str:
        """AM-X47-11: the file a node is read from, relative to the run directory."""
        return FINAL_FILE if node == self.final_step else f"checkpoints/checkpoint_{node}_steps.zip"

    def eval_share(self) -> Optional[float]:
        return None if self.eval_s is None else self.eval_s / self.sigma()


def _s_resume(c: Sequence[Tuple[int, float]]) -> int:
    return c[0][0] - (c[1][0] - c[0][0]) if len(c) > 1 else c[0][0] - 1


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
        s_resume = _s_resume(c)
        col = [r for r in col if r[0] <= s_resume] + c
        upd = [r for r in upd if r[0] <= s_resume] + u
    for v in (*col, *upd):
        if not math.isfinite(v[1]) or v[1] < 0:
            raise X47InputError(f"a non-finite or negative time {v[1]!r} at step {v[0]}")
    if not col or not upd:
        raise X47InputError(f"no {COLLECT_TAG if not col else UPDATE_TAG} row after the fork step {fork_step}")
    return col, upd


def surviving_extra(segments: Sequence[SegRows], extra: Sequence[Sequence[Tuple[int, float]]],
                    fork_step: int) -> List[Tuple[int, float]]:
    """A third tag's rows (``extra[i]`` = segment i's, in its own units) on the SAME surviving path as
    :func:`surviving_path` (each segment's resume point is read from its collect rows)."""
    out: List[Tuple[int, float]] = []
    for (c_rows, _), e_rows in zip(segments, extra):
        c = sorted((int(s), float(v)) for s, v in c_rows if int(s) > fork_step)
        e = sorted((int(s), float(v)) for s, v in e_rows if int(s) > fork_step)
        if not c:
            continue
        out = [r for r in out if r[0] <= _s_resume(c)] + e
    return out


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
    segs = load_segments(run_dir, [COLLECT_TAG, UPDATE_TAG, EVAL_TAG], after_step=fork)
    pairs = [(s.rows.get(COLLECT_TAG, []), s.rows.get(UPDATE_TAG, [])) for s in segs]
    col, upd = surviving_path(pairs, fork)
    ev = surviving_extra(pairs, [s.rows.get(EVAL_TAG, []) for s in segs], fork)
    has_eval = any(EVAL_TAG in s.rows for s in segs)
    return LearnTime(os.path.basename(os.path.normpath(run_dir)), fork, tuple(col), tuple(upd),
                     tuple(checkpoint_nodes(run_dir, fork)), run_dir=os.path.abspath(run_dir),
                     eval_s=math.fsum(v for _, v in ev) if has_eval else None)


# ----------------------------------------------------------------------- AM-X47-3 / AM-X47-12: which runs are read
def check_finished(run: LearnTime, stage: str) -> List[str]:
    """AM-X47-3: every reason ``run`` is not a FINISHED, K9(b)-PASS run (empty = it may be read)."""
    name, d = run.run, run.run_dir
    why: List[str] = []
    if not os.path.isfile(os.path.join(d, FINAL_FILE)):
        why.append(f"{name}: no {FINAL_FILE} (the run has not finished)")
    try:
        with open(os.path.join(d, "metadata.json")) as f:
            meta = json.load(f)
    except (OSError, ValueError) as exc:
        meta = {}
        why.append(f"{name}: metadata.json unreadable ({exc})")
    nt = meta.get("num_timesteps")
    if nt is None or int(nt) != run.final_step:
        why.append(f"{name}: metadata.json num_timesteps {nt!r} != the last TensorBoard row's step {run.final_step} "
                   "(the final is not the last row)")
    target = (meta.get("cli_args") or {}).get("steps")
    if target is None:
        why.append(f"{name}: metadata.json records no cli_args.steps (the run's own target is unknown)")
    elif run.final_step < int(target):
        why.append(f"{name}: the last row's step {run.final_step} < the run's --steps {int(target)}")
    rec = os.path.join(stage, "k9b", f"{name}.json")
    try:
        with open(rec) as f:
            k = json.load(f)
    except (OSError, ValueError):
        why.append(f"{name}: no readable K9(b) arm-end record {rec} (AM-6: `python -m main.x35.kit k9b`)")
    else:
        if k.get("schema") != "x35_k9b_arm_end_v1" or k.get("run") != name:
            why.append(f"{name}: {rec} is not this run's AM-6 record (schema {k.get('schema')!r}, run {k.get('run')!r})")
        elif k.get("verdict") != "PASS":
            why.append(f"{name}: K9(b) arm end reads {k.get('verdict')!r}, not PASS (a VOID arm enters no read)")
    return why


def check_nodes_dense(run: LearnTime, every: Optional[int]) -> List[str]:
    """AM-X47-12: an X47 run keeps its 1M nodes until the read; a gap > :data:`MAX_NODE_GAP_EVERY` × ``every``
    between consecutive nodes (the fork step and the final included) means it was thinned."""
    if not every or every <= 0:
        return [f"{run.run}: checkpoint_every_steps unknown (the node grid cannot be checked)"]
    pts = [run.fork_step] + [n for n in run.nodes if n < run.final_step] + [run.final_step]
    gaps = [(a, b) for a, b in zip(pts, pts[1:]) if b - a > MAX_NODE_GAP_EVERY * every]
    return [f"{run.run}: no node between {a} and {b} (> {MAX_NODE_GAP_EVERY} x {every} steps): thinned? X47 arms keep "
            "every 1M node until the X47 read (AM-X47-12)" for a, b in gaps[:5]]


def checkpoint_every(run_dir: str) -> Optional[int]:
    try:
        with open(os.path.join(run_dir, "metadata.json")) as f:
            v = (json.load(f).get("cli_args") or {}).get("checkpoint_every_steps")
    except (OSError, ValueError):
        return None
    return int(v) if v else None


# ------------------------------------------------------------------------------------------- the node rule
@dataclass(frozen=True)
class SEq:
    run: str
    node: Optional[int]          # None = no node qualifies (INCONCLUSIVE)
    is_final: bool
    sigma_s: Optional[float]
    skipped_rule8: Tuple[int, ...]
    node_update_basis: Optional[int]
    file: Optional[str] = None           # AM-X47-11: relative to the run directory

    def as_dict(self) -> Dict[str, Any]:
        return {"run": self.run, "s_eq": self.node, "is_final": self.is_final, "file": self.file,
                "sigma_s": self.sigma_s, "skipped_rule8": list(self.skipped_rule8),
                "s_eq_update_only_basis_reported": self.node_update_basis}


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
    return SEq(run.run, node, fin, run.sigma(node) if node is not None else None, tuple(skipped), upd,
               run.node_file(node) if node is not None else None)


@dataclass(frozen=True)
class Comparison:
    budget_s: float
    rho: float
    coincide: bool
    near_rho: bool
    seq: Tuple[SEq, ...]
    rho_collect: float
    rho_update: float
    c0_finals: Tuple[Tuple[str, int, str], ...] = ()      # (run, final step, file): AM-X47-11
    eval_share: Tuple[Tuple[str, Optional[float]], ...] = ()

    def as_dict(self) -> Dict[str, Any]:
        return {"budget_s": self.budget_s, "rho": self.rho, "coincide": self.coincide, "rho_near_1_02": self.near_rho,
                "s_eq": {s.run: s.as_dict() for s in self.seq},
                "c0_final": {r: {"step": n, "file": f} for r, n, f in self.c0_finals},
                "reported_only": {"rho_collect": self.rho_collect, "rho_update": self.rho_update,
                                  "eval_share": dict(self.eval_share)}}


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
        seq = tuple(SEq(r.run, r.final_step, True, r.sigma(), (), r.final_step, FINAL_FILE) for r in x47)
    else:
        seq = tuple(s_eq(r, budget, budget_update=b_upd) for r in x47)
    return Comparison(budget, rho, coincide, near, seq, ratio("collect"), ratio("update"),
                      tuple((r.run, r.final_step, FINAL_FILE) for r in c0),
                      tuple((r.run, r.eval_share()) for r in (*c0, *x47)))


def moved_nodes(prior: Mapping[str, Any], comp: Comparison) -> Dict[str, Dict[str, Any]]:
    """AM-X47-8: each X47 seed's look-1 node (``prior`` = the look-1 ``seq`` JSON) against its look-2 node; a seed is
    RE-READ unless both nodes are the same step (a new seed, or a node that moved, is read at its new node)."""
    old = prior.get("s_eq") or {}
    out: Dict[str, Dict[str, Any]] = {}
    for s in comp.seq:
        o = (old.get(s.run) or {}).get("s_eq")
        out[s.run] = {"look1_node": o, "look2_node": s.node, "file": s.file,
                      "reread": o is None or s.node is None or int(o) != int(s.node)}
    return out


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
                "x47b": x47b_action(self.verdict, self.primary),
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
            gap = rep.get("per_step_gap") or {}
            label = gap.get("label") or "BOUNDARY (the per-seed gap D was not computed)"
            return out(PER_STEP_ONLY, f"primary {primary.cls}, secondary (matched steps) GAIN: {label}")
        return out(str(primary.cls), f"primary {primary.cls}; secondary {secondary.cls}")
    return out(str(primary.cls), f"primary {primary.cls}")


def read(doc: Mapping[str, Any]) -> X47Verdict:
    """The whole decision from one JSON-shaped read (``decide``'s input):

    ``{"look": 1, "c0_final": {seed: S_pp}, "x47_s_eq": {seed: S_pp}, "x47_final": {seed: S_pp},
    "g_f": {"h_pp": [[...]]}, "g_c": <the main.x35.gc record>, "g_k": {...}, "comparison": {...}}``

    ``g_c`` is the toy side's seed-pooled G-C record (AM-X47b-1: X47 FINAL s360x against toy C0 final s360x, the T*
    truth of AM-X47b-2); a bare triple, or a record over other seeds, is INCONCLUSIVE.

    ``x47_s_eq`` holds each X47 seed's S at its s_eq node (``compare``'s output, carried in ``comparison``); when ρ
    coincides they are the finals' S. G-F's matrix is the primary's (s_eq nodes × C0 finals).

    REPORTED, never deciding: ``per_step_gap`` (AM-X47-5, from ``x47_final`` and ``x47_s_eq``) and, when the read
    carries ``"per_member": {"c0_final": {seed: {member: pp}}, "x47_s_eq": {seed: {member: pp}}}``, ``per_member``
    (AM-X47-7) with ``lineage_label`` "in-lineage" on a primary GAIN whose :data:`~x47_report.OUT_OF_LINEAGE` Δ̂ <= 0."""
    look = int(doc["look"])
    off = dict(doc.get("c0_final") or {})
    primary = R.decide_r1(look, dict(doc.get("x47_s_eq") or {}), off)
    fin = doc.get("x47_final")
    secondary = R.decide_r1(look, dict(fin), off) if fin else None
    guards = [R.guard_f((doc.get("g_f") or {}).get("h_pp"), look),
              R.guard_c_record(doc.get("g_c"), list(doc.get("x47_s_eq") or {}), list(off))]
    if doc.get("g_k") is not None:
        g = R.guard_k(doc["g_k"])
        guards.append(R.GuardResult("G-K (reported)", g.status, g.detail, g.near, g.reasons))
    rep: Dict[str, Any] = {}
    comp = doc.get("comparison") or {}
    rep["s_eq"] = comp.get("s_eq")
    rep["rho"] = comp.get("rho")
    rep["comparison"] = comp or None
    if fin:
        rep["per_step_gap"] = XR.per_step_gap(dict(fin), dict(doc.get("x47_s_eq") or {}))
    pm = doc.get("per_member")
    if pm is not None:
        rep["per_member"] = XR.per_member(dict(pm.get("x47_s_eq") or {}), dict(pm.get("c0_final") or {}),
                                          s_on=dict(doc.get("x47_s_eq") or {}), s_off=off)
    if primary.cls == GAIN and primary.outcome != CONTINUE:
        rep["lineage_label"] = XR.lineage_label(rep.get("per_member"))
    return decide(look, primary, secondary, guards, reported=rep)


def _seq(a: Any) -> Dict[str, Any]:
    """``seq``: AM-X47-3 / -8 / -12's refusals first (every reason at once), then :func:`compare`."""
    from utils.paths import repo_path

    stage = a.stage or str(repo_path(STAGE_REL))
    want = R.LOOK_SEEDS[a.look]
    why: List[str] = []
    if len(a.c0_run) != want or len(a.x47_run) != want:
        why.append(f"look {a.look} reads exactly {want} C0 + {want} X47 runs (got {len(a.c0_run)} + {len(a.x47_run)})")
    if a.prior and a.look != 2:
        why.append("--prior is look 2's (the look-1 nodes it compares against)")
    c0 = [load_run(r) for r in a.c0_run]
    x47 = [load_run(r) for r in a.x47_run]
    for r in (*c0, *x47):
        why += check_finished(r, stage)
    for r in x47:
        why += check_nodes_dense(r, checkpoint_every(r.run_dir))
    if why:
        raise X47InputError("; ".join(why))
    comp = compare(c0, x47)
    out = comp.as_dict()
    out["look"] = a.look
    if a.prior:
        with open(a.prior) as f:
            out["nodes_by_look"] = moved_nodes(json.load(f), comp)
    return out


def _main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import sys

    ap = argparse.ArgumentParser(prog="python -m main.x35.x47", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("seq", help="the budget, each X47 seed's s_eq and ρ from the runs' own TensorBoard (read-only)")
    q.add_argument("--look", type=int, required=True, choices=sorted(R.LOOK_SEEDS),
                   help="1: exactly 3 C0 + 3 X47 runs; 2: exactly 5 + 5 (B recomputed over all five C0, AM-X47-8)")
    q.add_argument("--c0-run", action="append", required=True)
    q.add_argument("--x47-run", action="append", required=True)
    q.add_argument("--stage", default=None, help="the stage dir holding k9b/<run>.json (default: X35's)")
    q.add_argument("--prior", default=None, help="look 2 only: the look-1 seq JSON (records which nodes moved)")
    d = sub.add_parser("decide", help="the verdict of one read JSON (see main.x35.x47.read)")
    d.add_argument("input")
    n = sub.add_parser("norms", help="AM-X47-10: the new rounds' weight norms on every checkpoint of a run (read-only)")
    n.add_argument("--run", required=True)
    pd = sub.add_parser("policy-diff", help="AM-X47b-3: mean KL on a bank to the same weights with the new rounds "
                                            "zeroed (read-only; REPORTED only)")
    pd.add_argument("--run", required=True)
    pd.add_argument("--bank", required=True, help="a main.policy_spectrum bank (the registered one: bank_x26)")
    pd.add_argument("--stride", type=int, default=10, help="every k-th checkpoint (the final is always read)")
    pd.add_argument("--node", type=int, action="append", default=[], help="read only these checkpoint steps")
    pd.add_argument("--threads", type=int, default=1)
    a = ap.parse_args(argv)
    try:
        if a.cmd == "seq":
            json.dump(_seq(a), sys.stdout, indent=1)
        elif a.cmd == "norms":
            json.dump(XR.run_norms(a.run), sys.stdout, indent=1)
        elif a.cmd == "policy-diff":
            json.dump(XR.run_policy_diff(a.run, a.bank, stride=a.stride, nodes=a.node, threads=a.threads),
                      sys.stdout, indent=1)
        else:
            with open(a.input) as f:
                json.dump(read(json.load(f)).as_dict(), sys.stdout, indent=1, default=str)
        print()
        return 0
    except (X47InputError, R.ReadoutInputError) as exc:
        print(f"[x35 x47] INCONCLUSIVE: {exc}", file=sys.stderr)
        return 2
    except ValueError as exc:                   # policy-diff's refusals (a missing node, not a grown network)
        if a.cmd != "policy-diff":
            raise
        print(f"[x35 x47] policy-diff REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_main())
