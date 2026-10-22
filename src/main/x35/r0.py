"""X35 R0, the EXPLOITABILITY CURVE (registered 2026-10-20 in ``R0_REGISTRATION.md`` of the X35 stage directory;
launch sheet §0 AM-R0; the owner's direction of 2026-10-20: "train an exploiter ... against prod to see how much head
room we have"). Nothing here launches, trains or plays except ``play``, which plays CPU games between saved files.

WHAT R0 IS. One fresh BEST RESPONDER (BR) per side, forked FROM the side's frozen start node and trained against that
node ALONE: ``toy_r0`` against T* (X34 290,000,087), at the toy pin; ``r0`` against P* (X26 280,000,170), at the
production pin (which waits for FP32_ALT, AM-8). The argv is the G-E reader's (``main.x35.ge``, F-LS-19: "R0 is the
SAME argv with P* as the target") with the start node as the target, run for :data:`L_R0` instead of 5M:
``--model S --exploiter S --no-self-play --trainee-teams <offense 5> --fork-lr 5.6e-05 --fork-lr-freeze
--behaviour-check warn --steps <S + 20M> --checkpoint-every-steps 1000000 [toy extras] --device cuda --seed <seed>
--run-name ...``. Production R0's first 5M is LG-1's registered R0 (A6's trigger): :func:`prefix_cell` reads it.

THE CURVE (:func:`curve_plan`, :func:`play_curve`). At every read point k in :data:`READ_POINTS` (0, 2M, ..., 20M) the
BR's checkpoint plays :data:`GAMES_PER_POINT` fresh greedy-vs-greedy games against the target on the Rust eval core
(``untaught_meter.play_cells``: the BR pilots its recorded offense pin, the target draws from the pool: the geometry of
the run's own vs-target series and of ``main.best_response_gap --play --greedy``). k = 0 is the START FILE itself (the
BR's weights at the fork ARE the target's), so gap(0) is the offense pin's matchup against the pool, not
exploitation: the curve reports gap(k) and the learned part gap(k) - gap(0).

THE CLASS (:func:`classify`, pure, deterministic; standing rule 8 at :data:`RULE8`):
* H, the HEADROOM: the pooled gap over :data:`LATE` (16, 18, 20M); its learned part H - gap(0) (Newcombe 95 %).
* D = pooled(LATE) - pooled(:data:`EARLY` = 10, 12, 14M), Newcombe 95 %. **RISING** iff lower(D) > +RULE8;
  else **SATURATED** iff upper(D) < :data:`SAT_BAR` - RULE8; else **UNRESOLVED**.
* t90: the first k >= 2M whose gap exceeds gap(0) + 0.9 (H - gap(0)) by more than 1e-9 (within 1e-9 is not reached);
  **NO RISE** (t90 undefined) iff H - gap(0) < :data:`NO_RISE`.
* > 25 % timeouts at any point: **INCONCLUSIVE** (a timeout is never a semantic outcome).

THE STOP RULE is the fixed ``--steps``: no stop on a score. A RISING class at the cap licenses nothing by itself; the
orchestrator may register an extension to 2 x :data:`L_R0` (a resume of the same run at the same pin).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

#: kit.Arm.phase of an R0 best responder (0 single, 1 exploiters, 2 folded in, 3 a G-E reader).
R0_PHASE = 4
#: arm -> side. The toy arm's name starts ``toy_`` because ``launch_x35.sh`` reads the side from the prefix.
R0_ARMS: Dict[str, str] = {"r0": "production", "toy_r0": "toy"}
#: the kit seed IS the training seed: production trains at 3531, the G-E readers' seed (§5.6.1 "readers 3531+"), so R0
#: and ``ge_c0`` share the dice; the toy at 3631 (the toy side's 36xx block).
R0_SEEDS: Dict[str, Tuple[int, ...]] = {"r0": (3531,), "toy_r0": (3631,)}
L_R0 = 20_000_000
READ_EVERY = 2_000_000
READ_POINTS: Tuple[int, ...] = tuple(range(0, L_R0 + 1, READ_EVERY))
EARLY: Tuple[int, ...] = (10_000_000, 12_000_000, 14_000_000)
LATE: Tuple[int, ...] = (16_000_000, 18_000_000, 20_000_000)
#: 80 games per offense team x 5 teams.
GAMES_PER_POINT = 400
READ_SEED = 20261020
#: LG-1's registered R0 statistic: the in-run pooled vs-target rate over the first 5M post-fork.
A6_PREFIX = 5_000_000
SAT_BAR = 0.05
NO_RISE = 0.02
T90_FRAC = 0.9
RULE8 = 0.0005
TIMEOUT_BAR = 0.25
#: a read point's checkpoint is the first one at or past B + k and short of B + k + this, B = the fork step floored to
#: a multiple of this (saves land just past each ABSOLUTE multiple of the cadence, e.g. checkpoint_296000066_steps.zip).
CKPT_TOL = 1_000_000
CURVE_SCHEMA = "x35_r0_curve_v1"
SATURATED, RISING, UNRESOLVED, INCONCLUSIVE = "SATURATED", "RISING", "UNRESOLVED", "INCONCLUSIVE"
_CKPT = re.compile(r"^checkpoint_(\d+)_steps\.zip$")


class R0Error(ValueError):
    """R0 cannot be built, planned or read (named; a refusal, never a number)."""


def build_r0_argv(arm: str, seed: int, call: Mapping[str, Any], *, run_name: str, behaviour: str = "warn",
                  ckpt_every: str = "1000000", extras: Sequence[str] = ()) -> List[str]:
    """The trainer argv of one R0 (module docs): the G-E reader's argv with the side's start node as the target."""
    from main.x35.ge import READER_FORK_LR, offense_teams

    if arm not in R0_ARMS:
        raise R0Error(f"{arm!r} is not an R0 arm ({sorted(R0_ARMS)})")
    if call.get("side") != R0_ARMS[arm]:
        raise R0Error(f"{arm} reads the {R0_ARMS[arm]} call record, got side {call.get('side')!r}")
    start, step = str(call["pstar"]["path"]), int(call["pstar"]["step"])
    return (["--model", start, "--exploiter", start, "--no-self-play", "--trainee-teams", offense_teams(),
             "--fork-lr", READER_FORK_LR, "--fork-lr-freeze", "--behaviour-check", behaviour,
             "--steps", str(step + L_R0), "--checkpoint-every-steps", ckpt_every] + list(extras)
            + ["--device", "cuda", "--seed", str(seed), "--run-name", run_name])


# ----------------------------------------------------------------------------------------------------- the plan
def checkpoints(run_dir: str) -> Dict[int, str]:
    """``{step: path}`` of ``<run>/checkpoints/checkpoint_<step>_steps.zip``."""
    d = Path(run_dir) / "checkpoints"
    out: Dict[int, str] = {}
    if d.is_dir():
        for p in d.iterdir():
            m = _CKPT.match(p.name)
            if m:
                out[int(m.group(1))] = str(p)
    return out


def checkpoint_for(run_dir: str, fork_step: int, k: int, start_file: str,
                   final_step: Optional[int] = None) -> Tuple[str, int]:
    """``(path, step)`` of read point ``k``: the START FILE at k = 0; else the first checkpoint in [B + k, B + k +
    CKPT_TOL), B = the fork's checkpoint boundary (``fork_step`` floored to a multiple of CKPT_TOL), or, at the cap,
    ``final_model.zip`` when its step (``final_step``) is >= B + k.

    The window is anchored on B, not on ``fork_step``, because the trainer saves at the first call past each ABSOLUTE
    multiple of ``--checkpoint-every-steps`` (``main.train.constants.checkpoint_due``): the +6M save of a fork at
    290,000,087 is 296,000,066, short of fork + 6M (the first R0-toy read, 2026-10-21)."""
    if k == 0:
        return start_file, fork_step
    base = int(fork_step) - int(fork_step) % CKPT_TOL
    lo, hi = base + k, base + k + CKPT_TOL
    hit = sorted(s for s in checkpoints(run_dir) if lo <= s < hi)
    if hit:
        return checkpoints(run_dir)[hit[0]], hit[0]
    fz = Path(run_dir) / "final_model.zip"
    if k == L_R0 and fz.is_file() and final_step is not None and final_step >= lo:
        return str(fz), int(final_step)
    raise R0Error(f"{run_dir}: no checkpoint in [{lo:,}, {hi:,}) for read point +{k:,} (thinned or not reached)")


def curve_plan(run_dir: str, fork_step: int, start_file: str, final_step: Optional[int] = None
               ) -> List[Tuple[int, str, int]]:
    """``[(k, path, step)]`` for every read point; a missing one is a refusal (a thinned run cannot be read)."""
    return [(k,) + checkpoint_for(run_dir, fork_step, k, start_file, final_step) for k in READ_POINTS]


# ----------------------------------------------------------------------------------------------------- the class
@dataclass(frozen=True)
class Point:
    k: int
    wins: int
    finished: int
    ties: int = 0
    attempted: Optional[int] = None

    @property
    def gap(self) -> float:
        return self.wins / self.finished - 0.5

    @property
    def timeout_fraction(self) -> float:
        a = self.attempted if self.attempted is not None else self.finished
        return (a - self.finished) / a if a else 1.0


def _pool(points: Mapping[int, Point], ks: Sequence[int]) -> Tuple[int, int]:
    return sum(points[k].wins for k in ks), sum(points[k].finished for k in ks)


def classify(pts: Sequence[Point]) -> Dict[str, Any]:
    """The registered class of one curve (module docs). Refuses a curve without every read point."""
    from agents.training.best_response_gap import newcombe_diff_ci
    from agents.training.stats import wilson_ci

    by = {p.k: p for p in pts}
    missing = [k for k in READ_POINTS if k not in by or by[k].finished <= 0]
    if missing:
        raise R0Error(f"the curve lacks read point(s) {missing} (every registered point is required)")
    rows = []
    for k in READ_POINTS:
        p = by[k]
        lo, hi = wilson_ci(p.wins, p.finished)
        rows.append({"k": k, "wins": p.wins, "finished": p.finished, "ties": p.ties,
                     "timeout_fraction": p.timeout_fraction, "gap": p.gap, "gap_ci": [lo - 0.5, hi - 0.5]})
    bad = [r["k"] for r in rows if r["timeout_fraction"] > TIMEOUT_BAR]
    out: Dict[str, Any] = {"points": rows}
    if bad:
        out.update(cls=INCONCLUSIVE, why=f"> {TIMEOUT_BAR:.0%} timeouts at read point(s) {bad}")
        return out
    w0, n0 = by[0].wins, by[0].finished
    wl, nl = _pool(by, LATE)
    we, ne = _pool(by, EARLY)
    g0 = w0 / n0 - 0.5
    h = wl / nl - 0.5
    learned = newcombe_diff_ci(wl, nl, w0, n0)
    d = newcombe_diff_ci(wl, nl, we, ne)
    if d[1] > RULE8:
        cls = RISING
    elif d[2] < SAT_BAR - RULE8:
        cls = SATURATED
    else:
        cls = UNRESOLVED
    t90: Optional[int] = None
    if h - g0 >= NO_RISE:
        bar = g0 + T90_FRAC * (h - g0)
        t90 = next((k for k in READ_POINTS[1:] if by[k].gap - bar > 1e-9), None)
    hlo, hhi = wilson_ci(wl, nl)
    out.update(cls=cls, gap0=g0, headroom=h, headroom_ci=[hlo - 0.5, hhi - 0.5], learned=list(learned),
               late_minus_early=list(d), t90=t90, no_rise=h - g0 < NO_RISE,
               why=(f"D = {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]: " +
                    {RISING: "the lower bound is above 0: still rising at the cap",
                     SATURATED: f"the upper bound is below {SAT_BAR:+.2f}: saturated",
                     UNRESOLVED: "neither: the curve cannot say"}[cls]))
    return out


def prefix_cell(run: Any, limit: int = A6_PREFIX) -> Tuple[int, int]:
    """LG-1's registered R0 statistic (A6's trigger: gap >= 0.15): ``(wins, games)`` pooled over the run's own post-fork
    vs-target eval rows at step <= fork + ``limit`` (``run`` = ``best_response_gap.read_exploiter``'s record)."""
    if run.fork_step is None:
        raise R0Error(f"{run.run}: no recorded fork step")
    rows = [p for p in run.post_fork if p.step <= int(run.fork_step) + limit]
    if not rows:
        raise R0Error(f"{run.run}: no post-fork vs-target row at or below fork + {limit:,}")
    return sum(p.wins for p in rows), sum(p.games for p in rows)


# ------------------------------------------------------------------------------------------------------ the read
def _point_record(ledger: str, label: str) -> Path:
    return Path(ledger) / f"{label}.json"


def play_curve(run_dir: str, out: str, *, call: Mapping[str, Any], games: int = GAMES_PER_POINT,
               seed: int = READ_SEED, greedy: bool = True, progress=None, ledger: Optional[str] = None,
               max_points: Optional[int] = None) -> Dict[str, Any]:
    """Play the registered curve of one FINISHED R0 run against its target and write ``out`` ONCE (CPU; the Rust eval
    core). The run must have recorded the call record's start node as its target and piloted the offense pin.

    ``ledger`` (a directory OUTSIDE ``models/``) makes the read RESUMABLE in chunks: each read point is played as its
    own ``play_cells`` call and banked as ``<ledger>/<label>.json`` (counts + trimmed game rows); a banked point whose
    file, step, seed and regime match is reused, never replayed. The deals are identical either way (the Rust meter's
    seed is a function of (seed, team index) only: CRN across refs). ``max_points`` stops after that many NEW points
    and returns ``{"partial": True, ...}`` without writing ``out``; ``out`` is written once every point is banked."""
    from dataclasses import asdict

    from agents.training import best_response_gap as E
    from agents.training import untaught_meter as um

    if os.path.exists(out):
        raise R0Error(f"{out} exists: a curve record is written ONCE")
    run = E.read_exploiter(run_dir, E.load_teamsets())
    start = str(call["pstar"]["path"])
    if not run.target_file or os.path.realpath(run.target_file) != os.path.realpath(start):
        raise R0Error(f"{run.run}: recorded target {run.target_file!r} is not the call record's start {start}")
    if run.archetype != "offense":
        raise R0Error(f"{run.run}: archetype {run.archetype!r} ({run.membership}), not 'offense'")
    if not os.path.isfile(os.path.join(run_dir, "final_model.zip")) or (run.budget or 0) < L_R0:
        raise R0Error(f"{run.run}: not finished at +{L_R0:,} (budget {run.budget})")
    plan = curve_plan(run_dir, int(run.fork_step), start, run.num_timesteps)
    teams = um.team_slices(run.teams, prefix="X", source=f"{run.run} recorded trainee pin")
    refs = [um.resolve_ref(path, label=f"k{k // 1_000_000:02d}M") for k, path, _ in plan]
    target = um.resolve_ref(start, label="target", role="opponent")
    per_team = -(-int(games) // len(teams))
    regime = "greedy-vs-greedy" if greedy else "stochastic@1"
    if ledger is None:
        cells = um.play_cells(refs, teams, target, games_per_team=per_team, seed=seed, stochastic=not greedy,
                              progress=progress)
    else:
        cells, new = {}, 0
        Path(ledger).mkdir(parents=True, exist_ok=True)
        for (k, path, step), ref in zip(plan, refs):
            rec = _point_record(ledger, ref.label)
            want = {"k": k, "path": path, "step": step, "seed": seed, "games_per_team": per_team, "regime": regime,
                    "target_file": start}
            if rec.is_file():
                got = json.loads(rec.read_text())
                if {f: got.get(f) for f in want} != want:
                    raise R0Error(f"{rec}: banked for {({f: got.get(f) for f in want})}, this read is {want}")
                cells[ref.label] = {t: um.Cell(**c) for t, c in got["cells"].items()}
                continue
            if max_points is not None and new >= max_points:
                return {"partial": True, "banked": sorted(cells), "remaining": len(plan) - len(cells)}
            log: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
            one = um.play_cells([ref], teams, target, games_per_team=per_team, seed=seed, stochastic=not greedy,
                                progress=progress, game_log=log)
            cells[ref.label] = one[ref.label]
            body = dict(want, cells={t: asdict(c) for t, c in one[ref.label].items()},
                        games={f"{a}|{b}": rows for (a, b), rows in log.items()})
            tmp = rec.with_suffix(".tmp")
            tmp.write_text(json.dumps(body, sort_keys=True) + "\n")
            os.replace(tmp, rec)
            new += 1
    pts = []
    for (k, path, step), ref in zip(plan, refs):
        got = cells[ref.label].values()
        pts.append(Point(k, sum(c.wins for c in got), sum(c.finished for c in got), sum(c.ties for c in got),
                         sum(c.attempted for c in got)))
    doc = {"schema": CURVE_SCHEMA, "run": run.run, "target_file": start, "fork_step": run.fork_step,
           "regime": regime, "seed": seed, "games_per_team": per_team,
           "transport": um.cells_transport(cells),
           "plan": [{"k": k, "path": path, "step": step} for k, path, step in plan],
           "a6_prefix_cell": list(prefix_cell(run)) if run.post_fork else None}
    doc.update(classify(pts))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return doc


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.x35.r0", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan", help="print the read points' files of one R0 run (exit 2 = a point is missing)")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--call", required=True)
    q = sub.add_parser("play", help="play the registered curve (CPU) and write the record ONCE")
    q.add_argument("--run-dir", required=True)
    q.add_argument("--call", required=True)
    q.add_argument("--out", required=True)
    q.add_argument("--stochastic", action="store_true", help="the SECONDARY regime (stochastic@1), never the class")
    q.add_argument("--ledger", help="bank each read point here (outside models/) so the read resumes in chunks")
    q.add_argument("--max-points", type=int, help="with --ledger: stop after this many NEW points (exit 3, partial)")
    a = ap.parse_args(argv)
    try:
        call = json.loads(Path(a.call).read_text())
        if a.cmd == "plan":
            from agents.training import best_response_gap as E

            run = E.read_exploiter(a.run_dir, E.load_teamsets())
            for k, path, step in curve_plan(a.run_dir, int(run.fork_step), str(call["pstar"]["path"]),
                                            run.num_timesteps):
                print(f"+{k:>10,}  {step:>12,}  {path}")
            return 0
        if a.max_points is not None and not a.ledger:
            raise R0Error("--max-points needs --ledger (a partial read must be banked)")
        doc = play_curve(a.run_dir, a.out, call=call, greedy=not a.stochastic, ledger=a.ledger,
                         max_points=a.max_points)
        if doc.get("partial"):
            print(json.dumps(doc))
            return 3
        print(json.dumps({k: doc[k] for k in ("run", "cls", "why", "gap0", "headroom", "t90") if k in doc}, indent=1))
        return 0
    except R0Error as exc:
        print(f"[x35 r0] REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
