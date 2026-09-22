"""THE BEST-RESPONSE GAP — the POPULATION loop's meter (engine; CLI: ``main.best_response_gap``).

The exploiter flywheel's teachers turned out to be **target-specific counterplay**: the offense
5-team teacher beat its parent 0.657 head-to-head and was −1.00 pp against a third party on the
same five teams (ledger 2026-09-21, *NONE of the three 5-team teachers is admitted*), and the
redesign against a DISTRIBUTION of opponents failed the same way (−1.37 pp). The next-era design
is the AlphaStar-style POPULATION loop — exploiters enter the generalist's opponent POOL and the
generalist absorbs them by its own gradient — and that loop's meter is not "can a teacher teach".
It is the **BEST-RESPONSE GAP**:

    gap(t) = P(a fresh exploiter trained against generalist G_t beats G_t) − 0.5

read at MATCHED exploiter budget across rounds. **The loop is working iff the gap FALLS round over
round** (0.66 → toward 0.50). A gap that does not fall means the generalist is not absorbing what
the best responder found; a gap that falls means each round's best response is a smaller one, which
is the whole claim of a population loop and the same quantity `main.exploitability` folds from
admission artifacts — read there as a net EXTRACTION over a seniority-matched reference, read here
as a head-to-head rate against the target itself.

🚨 **THIS METER AND `main.exploitability` MEASURE DIFFERENT THINGS AND NEITHER PREDICTS THE OTHER.**
The training-side vs-target curve this reads plays the TARGET ITSELF; the admission gate plays a
fixed THIRD PARTY. 2026-09-21 is the measurement that made the distinction expensive: the head-to-
head winner was the most NEGATIVE arm on the third-party gate. Quote the two side by side, never
one as evidence for the other.

═══ WHAT IT READS ════════════════════════════════════════════════════════════════════════════

Per exploiter run — an ``--exploiter <target>`` run — it reads, never re-derives:

* the TARGET, through ``agents.training.lineage``'s recorded ``exploiter_target`` block (its
  ``resolved_file`` / ``resolved_num_timesteps``, i.e. WHICH weights, not just which run);
* the run's own vs-target SERIES from ``<run>/eval_results.jsonl``. The on-disk key is
  ``row["externals"]["ext_<target run name>"]`` — ``{"win_rate": r, "counts": [wins, n]}`` — written
  by ``eval_callback.record_elo`` for the fixed cross-run opponent that
  ``fixed_opponent_pool.register_exploiter_for_eval`` auto-registers from ``--exploiter``. Only
  POST-FORK rows count: a row at or below ``lineage.fork_step`` is the PARENT's cycle, not this
  run's. Both the ENDPOINT (the last post-fork row — the convention every banked number on record
  uses) and the POOLED post-fork rate are reported, each with a Wilson interval;
* the BUDGET (``metadata.json``'s ``num_timesteps`` − ``lineage.fork_step``), the DOSE
  (``main.dose.read_run``'s ``dose_rate``) and the REGIME (below);
* the pinned TEAMS, through ``matchup_spec.read_recorded_trainee_teams`` — the one provenance
  reader, which RAISES rather than answering ``[]`` for a path that is not a run directory.

🚨 **THE SERIES IS GREEDY-VS-GREEDY, WHICH IS NOT THE TRAINING REGIME.** ``main.eval_worker``'s
FIXED branch builds the cross-run opponent ``stochastic=False, temperature=1.0`` ("eval = greedy
yardstick") against a greedy ``EvalRLPlayer`` trainee. ``eval_sentinel_greedy`` does NOT govern it
— that flag moves the ``sentinel_*`` branch, and an exploiter run has no sentinels at all. So the
series is the EVAL regime; ``--play`` defaults to the TRAINING regime (stochastic@1 both sides) and
``--play --greedy`` reproduces the series' own. The two are different populations, every row states
which it is, and the gap table is built from ONE of them — never a mixture.

═══ WHAT IT REFUSES ═══════════════════════════════════════════════════════════════════════════

A gap is only comparable across rounds at a matched EXPLOITER, so four properties are checked over
every pair of exploiters in the invocation and a mismatch is a typed refusal naming the cause:

* :class:`UnmatchedBudgetError` — post-fork steps differ by more than ``budget_tol`` (2 %);
* :class:`UnmatchedDoseError`   — ``dose_rate`` differs by more than ``dose_tol`` (10 %);
* :class:`UnmatchedRegimeError` — the regime tuple differs: ``eval_sentinel_greedy``, the
  exploiter opponent mix (``exploiter_keep_bots`` / ``exploiter_bot_fraction`` / the temperature
  schedule), the per-cycle ``n_games``, and whether the TARGET pilots its own pin or the pool.

``--allow-unmatched`` downgrades each to a WARNING that is carried into the printed header AND into
the JSON, because **the era-2/era-1 comparison of 2026-09-21 was confounded by exactly a 4.5×
dose gap** (3.815e-08 against 8.392e-09) that nobody registered: same ``--lr 0.0003`` flag both
times, but ``--fork-lr`` unset, so the new parent's annealed rate was inherited. A number that
crosses that boundary must carry the reason it should not be trusted, on the same line.

The gap itself is NOT refused for being small — a flat gap is the measurement.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from agents.training.stats import wilson_ci
from utils.paths import main_models_dir, repo_path

#: The three registered 5-team archetype sets, copied out of the era-2 launch job's scratch
#: directory on 2026-09-22 so this meter depends on a COMMITTED file rather than on
#: `~/.claude/jobs/<id>/tmp/`, which is not a durable location and is not readable from a
#: different machine. Era-1's single `--trainee-team` is element 0 (the `anchor`) of one of these,
#: so era-1 is the strict 1-team special case and membership assigns it the same archetype.
DEFAULT_TEAMSETS = repo_path("designs/research_state/exploiter_teamsets_2026-09-22.json")

#: Relative tolerances for the matched-exploiter gate. Budget is nearly exact in practice (a
#: launch lands on an `n_steps` boundary), dose is a median over a live controller's trajectory.
DEFAULT_BUDGET_TOL = 0.02
DEFAULT_DOSE_TOL = 0.10

#: Bootstrap for the paired round-over-round delta. The pairing unit is the ARCHETYPE, of which
#: there are three — so the interval is wide on purpose and the census prints beside it.
DEFAULT_DRAWS = 20000
DEFAULT_BOOTSTRAP_SEED = 20260922

#: The two regimes a rate can be measured in, spelled out wherever a number is printed.
SERIES_REGIME = "greedy-vs-greedy (EVAL regime, main.eval_worker FIXED branch)"
PLAY_REGIME_STOCHASTIC = "stochastic@1 both sides (TRAINING regime)"
PLAY_REGIME_GREEDY = "greedy both sides (EVAL regime — matches the series)"


# --------------------------------------------------------------------------------------------
# Refusals — typed, each naming its cause
# --------------------------------------------------------------------------------------------

class BestResponseGapError(RuntimeError):
    """A refusal the caller surfaces verbatim. ``cause`` is the machine-readable reason."""

    cause = "best_response_gap"


class RunReadError(BestResponseGapError):
    """The run is not an exploiter run, or states nothing this meter can read."""

    cause = "run_read"


class SeriesError(BestResponseGapError):
    """The run records no post-fork vs-target cycle. A meter with no rows REFUSES; it never
    reports a zero, which is what an absent series would otherwise look like."""

    cause = "series"


class UnmatchedError(BestResponseGapError):
    """Two exploiters being compared differ on a property the comparison depends on."""

    cause = "unmatched"


class UnmatchedBudgetError(UnmatchedError):
    cause = "unmatched_budget"


class UnmatchedDoseError(UnmatchedError):
    cause = "unmatched_dose"


class UnmatchedRegimeError(UnmatchedError):
    cause = "unmatched_regime"


# --------------------------------------------------------------------------------------------
# Statistics
# --------------------------------------------------------------------------------------------

def newcombe_diff_ci(w1: int, n1: int, w2: int, n2: int,
                     z: float = 1.96) -> Tuple[float, float, float]:
    """``(p1 − p2, lo, hi)`` — Newcombe's square-and-add interval for a DIFFERENCE of two
    INDEPENDENT proportions, built from each arm's Wilson interval.

    Used rather than a normal approximation for the same reason :func:`wilson_ci` is: near 0 or 1
    the Wald interval is degenerate, and two of the six cells this meter reads sat at 0.74 on
    n = 100. The two arms here ARE independent — two different runs' eval cycles, no shared dice —
    which is what this method assumes; the round-over-round MEAN across archetypes is a different,
    PAIRED quantity and gets :func:`paired_delta_ci` instead.
    """
    p1 = (w1 / n1) if n1 else 0.0
    p2 = (w2 / n2) if n2 else 0.0
    l1, u1 = wilson_ci(w1, n1, z)
    l2, u2 = wilson_ci(w2, n2, z)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


def paired_delta_ci(pairs: Sequence[Tuple[Tuple[int, int], Tuple[int, int]]], *,
                    draws: int = DEFAULT_DRAWS, seed: int = DEFAULT_BOOTSTRAP_SEED,
                    ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """``(mean Δ, lo, hi)`` over PAIRED cells — ``pairs`` is ``[((w_later, n_later),
    (w_earlier, n_earlier)), …]``, one entry per pairing unit (here: per archetype).

    TWO-LEVEL bootstrap, because two things are uncertain and reporting one of them would
    understate the width. The outer level resamples the PAIRING UNITS with replacement (the
    between-archetype spread — the reason a mean over three archetypes is not a mean over 1,200
    battles); the inner level redraws each resampled cell's win count from
    ``Binomial(n, ŵ/n)`` (the within-cell sampling noise at n = 400). The statistic is the same
    functional as the point estimate, so the interval brackets the estimate rather than describing
    a different quantity — the defect `stats.cluster_bootstrap_diff_ci` carries its own warning
    about.

    ``(None, None, None)`` for fewer than two pairs: a one-unit "spread" is not a measurement.
    """
    if len(pairs) < 2:
        return (None, None, None)
    later = np.array([[w, n] for (w, n), _ in pairs], dtype=float)
    earlier = np.array([[w, n] for _, (w, n) in pairs], dtype=float)
    point = float(np.mean(later[:, 0] / later[:, 1] - earlier[:, 0] / earlier[:, 1]))

    rng = np.random.default_rng(seed)
    k = len(pairs)
    idx = rng.integers(0, k, (draws, k))
    lw, ln = later[idx, 0], later[idx, 1]
    ew, en = earlier[idx, 0], earlier[idx, 1]
    lb = rng.binomial(ln.astype(int), np.clip(lw / ln, 0.0, 1.0)) / ln
    eb = rng.binomial(en.astype(int), np.clip(ew / en, 0.0, 1.0)) / en
    boot = (lb - eb).mean(axis=1)
    return point, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


# --------------------------------------------------------------------------------------------
# Team archetypes
# --------------------------------------------------------------------------------------------

def load_teamsets(path: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """The registered archetype → teamset map. Raises :class:`RunReadError` on a malformed file,
    naming the key, because an archetype silently read as ``None`` would collapse two rounds'
    rows into one unlabelled bucket."""
    src = str(path or DEFAULT_TEAMSETS)
    try:
        with open(src) as fh:
            raw = json.load(fh)
    except OSError as exc:
        raise RunReadError(f"teamsets {src!r}: {exc}") from exc
    if not isinstance(raw, dict) or not raw:
        raise RunReadError(f"teamsets {src!r}: not a non-empty object of archetype -> teamset")
    for name, block in raw.items():
        if not isinstance(block, dict) or not isinstance(block.get("hashes"), list):
            raise RunReadError(f"teamsets {src!r}: {name!r} states no 'hashes' list")
    return raw


def _stem(path: str) -> str:
    return os.path.basename(path).split(".")[0]


def archetype_of(team_paths: Sequence[str], teamsets: Dict[str, Dict[str, Any]],
                 ) -> Tuple[Optional[str], str]:
    """``(archetype, membership)`` for a run's pinned teams — by CONTENT-derived file stem against
    each registered teamset's ``hashes``.

    A run is assigned the archetype whose hash set CONTAINS all of its teams, so era-1's single
    anchor team and era-2's full five both land on the same name (``membership`` says ``1/5`` or
    ``5/5``, which is a real difference and stays visible). ``(None, …)`` when no set contains them
    or more than one does — reported as ``unassigned``, never guessed, because the archetype is
    the PAIRING UNIT of the round-over-round delta and a wrong one pairs the wrong arms.
    """
    if not team_paths:
        return None, "no pinned teams"
    stems = {_stem(p) for p in team_paths}
    hits = [name for name, block in teamsets.items()
            if stems <= {str(h) for h in block.get("hashes", [])}]
    if len(hits) != 1:
        which = ", ".join(sorted(hits)) if hits else "none"
        return None, f"{len(stems)} team(s), matching sets: {which}"
    name = hits[0]
    return name, f"{len(stems)}/{len(teamsets[name].get('hashes', []))}"


# --------------------------------------------------------------------------------------------
# One exploiter run
# --------------------------------------------------------------------------------------------

@dataclass(frozen=True)
class SeriesPoint:
    """One eval cycle's vs-target cell."""
    step: int
    wins: int
    games: int
    post_fork: bool

    @property
    def rate(self) -> float:
        return self.wins / self.games if self.games else float("nan")

    def to_json(self) -> dict:
        return {"step": self.step, "wins": self.wins, "games": self.games,
                "win_rate": self.rate, "post_fork": self.post_fork}


@dataclass
class ExploiterRun:
    """Everything this meter knows about one ``--exploiter`` run. Pure over the filesystem."""
    run: str
    run_dir: str
    target_run: Optional[str]
    target_file: Optional[str]
    target_step: Optional[int]
    target_pins_own_teams: Optional[bool]
    fork_step: Optional[int]
    num_timesteps: Optional[int]
    budget: Optional[int]
    dose_rate: Optional[float]
    lr_median: Optional[float]
    teams: List[str]
    archetype: Optional[str]
    membership: str
    regime: Dict[str, Any]
    series: List[SeriesPoint]
    lineage_derived: bool
    round: Optional[int] = None

    # ---- the two rates, each with its Wilson interval ----
    @property
    def post_fork(self) -> List[SeriesPoint]:
        return [p for p in self.series if p.post_fork]

    def endpoint(self) -> Optional[SeriesPoint]:
        """The LAST post-fork cycle — the convention every banked vs-target number uses."""
        rows = self.post_fork
        return rows[-1] if rows else None

    def pooled(self) -> Tuple[int, int]:
        rows = self.post_fork
        return sum(p.wins for p in rows), sum(p.games for p in rows)

    def cell(self, stat: str) -> Tuple[int, int]:
        """``(wins, games)`` for the chosen statistic. ``stat`` is ``pooled`` or ``endpoint``."""
        if stat == "endpoint":
            ep = self.endpoint()
            return (ep.wins, ep.games) if ep else (0, 0)
        return self.pooled()

    def to_json(self) -> dict:
        ep = self.endpoint()
        pw, pn = self.pooled()
        return {
            "run": self.run, "dir": self.run_dir, "round": self.round,
            "archetype": self.archetype, "membership": self.membership,
            "target": {"run": self.target_run, "resolved_file": self.target_file,
                       "resolved_num_timesteps": self.target_step,
                       "pins_own_teams": self.target_pins_own_teams},
            "fork_step": self.fork_step, "num_timesteps": self.num_timesteps,
            "budget": self.budget, "dose_rate": self.dose_rate, "lr_median": self.lr_median,
            "lineage_derived": self.lineage_derived,
            "teams": list(self.teams), "regime": dict(self.regime),
            "series_regime": SERIES_REGIME,
            "series": [p.to_json() for p in self.series],
            "endpoint": ep.to_json() if ep else None,
            "pooled": {"wins": pw, "games": pn,
                       "win_rate": (pw / pn) if pn else None,
                       "ci": list(wilson_ci(pw, pn)) if pn else None},
        }


def _read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path) as fh:
            obj = json.load(fh)
    except (OSError, ValueError):
        return {}
    return obj if isinstance(obj, dict) else {}


def resolve_run_dir(ref: str) -> str:
    """A run dir, or a bare run NAME resolved against the MAIN checkout's ``models/``.

    ``models/`` is not committed and exists only in the main checkout, so a worktree reaches across
    through ``utils.paths.main_models_dir``. A ref that resolves nowhere is returned unchanged and
    the caller refuses by name — never silently.
    """
    if os.path.isdir(ref):
        return os.path.abspath(ref)
    root = main_models_dir()
    if root is not None:
        rel = ref[len("models/"):] if ref.startswith("models/") else ref
        for cand in (root / rel, root.parent / ref):
            if os.path.isdir(cand):
                return str(cand)
    return ref


def read_series(run_dir: str, target_run: Optional[str],
                fork_step: Optional[int]) -> List[SeriesPoint]:
    """The vs-target cycles from ``<run>/eval_results.jsonl``.

    The external label is ``ext_<target run name>`` (``fixed_opponent_pool.EXT_PREFIX``). When the
    target's run name is known the label is matched EXACTLY; a file whose rows carry exactly one
    ``ext_*`` under a different name is a target/series disagreement and REFUSES rather than
    reading whichever external happens to be there. A row at or below ``fork_step`` is marked
    ``post_fork=False`` — it is the parent's cycle. (No run in today's archive has one: every
    exploiter's eval file starts fresh after the fork. The filter is the guard for the case where
    a future fork inherits its parent's file, which would otherwise credit the parent's rate to
    the best responder.)
    """
    path = os.path.join(run_dir, "eval_results.jsonl")
    if not os.path.isfile(path):
        raise SeriesError(f"{path}: no eval_results.jsonl — this run recorded no eval cycle.")
    want = f"ext_{target_run}" if target_run else None
    out: List[SeriesPoint] = []
    seen_labels: set = set()
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            ext = row.get("externals")
            if not isinstance(ext, dict):
                continue
            seen_labels.update(ext)
            cell = ext.get(want) if want else None
            if not isinstance(cell, dict):
                continue
            counts = cell.get("counts")
            if not (isinstance(counts, (list, tuple)) and len(counts) == 2):
                raise SeriesError(
                    f"{path}: the {want!r} cell at step {row.get('step')} states no [wins, games] "
                    "'counts' — a win_rate without its denominator cannot carry an interval.")
            step = int(row.get("step") or 0)
            out.append(SeriesPoint(step=step, wins=int(counts[0]), games=int(counts[1]),
                                   post_fork=fork_step is None or step > int(fork_step)))
    if not out:
        raise SeriesError(
            f"{path}: no {want!r} cycle. The externals this file carries are: "
            f"{', '.join(sorted(seen_labels)) or '(none)'}. The exploiter target recorded in "
            "lineage and the external the run evaluated against disagree — this meter will not "
            "read a different opponent's rate as the best-response gap.")
    if not any(p.post_fork for p in out):
        raise SeriesError(
            f"{path}: all {len(out)} {want!r} cycles are at or below fork_step={fork_step} — they "
            "are the PARENT's rows, not this run's best-response curve.")
    return out


def _regime(meta: Dict[str, Any], series: Sequence[SeriesPoint],
            target_pins: Optional[bool]) -> Dict[str, Any]:
    """The properties the vs-target number depends on, as one comparable tuple.

    🚨 ``eval_sentinel_greedy`` is in here for the RECORD, not because it moves this number: the
    ``ext_`` branch is hard-coded greedy either way (`eval_worker.py`, "eval = greedy yardstick")
    and an exploiter run has no sentinels at all. What DOES move it is the exploiter's own
    training opponent mix (a run that spends half its episodes on bots is best-responding at half
    rate), the per-cycle sample size, and whether the target pilots its own pinned team or draws
    from the shared pool — "my specialist team vs your random team" is not a mirror match.
    """
    cli = meta.get("cli_args") if isinstance(meta.get("cli_args"), dict) else {}
    games = sorted({p.games for p in series})
    return {
        "eval_sentinel_greedy": cli.get("eval_sentinel_greedy"),
        "exploiter_keep_bots": cli.get("exploiter_keep_bots"),
        "exploiter_bot_fraction": cli.get("exploiter_bot_fraction"),
        "exploiter_temp_mode": cli.get("exploiter_temp_mode"),
        "exploiter_temp_end": cli.get("exploiter_temp_end"),
        "cycle_games": games[0] if len(games) == 1 else games,
        "target_pins_own_teams": target_pins,
    }


def read_exploiter(ref: str, teamsets: Dict[str, Dict[str, Any]]) -> ExploiterRun:
    """Read one exploiter run. Every fact comes from what the run WROTE DOWN.

    The target is the recorded ``lineage.exploiter_target`` block, read through
    ``agents.training.lineage``'s accessors — the same path ``main.lineage`` prints, including its
    legacy derive-from-``original_command`` fallback, which marks itself ``derived``. It is never
    re-resolved under today's rule: that would print a current answer as history.
    """
    from agents.training.lineage import build_lineage_from_command, read_block, read_num_timesteps
    from agents.training.lineage import read_original_command
    from agents.training.matchup_spec import read_recorded_trainee_teams
    from main import dose as dose_cli

    run_dir = resolve_run_dir(ref)
    if not os.path.isdir(run_dir):
        raise RunReadError(f"{ref!r}: no such run directory (tried {run_dir!r}). A bare run name "
                           "is resolved against the MAIN checkout's models/, which a worktree "
                           "reaches through utils.paths.main_models_dir().")
    block = read_block(run_dir)
    derived = False
    if block is None:
        cmd = read_original_command(run_dir)
        if not cmd:
            raise RunReadError(f"{run_dir}: no lineage block and no original_command — nothing "
                               "on disk names this run's exploiter target.")
        block = build_lineage_from_command(cmd, model_dir=run_dir, hash_parent=False) or {}
        derived = True
    derived = derived or bool(block.get("derived"))

    target = block.get("exploiter_target")
    if not isinstance(target, dict) or not target:
        raise RunReadError(
            f"{os.path.basename(run_dir)}: recorded role is {block.get('role')!r} and its lineage "
            "names NO exploiter_target — this is not an exploiter run. The best-response gap is "
            "defined only for a run that best-responded to something.")

    fork_step = block.get("fork_step")
    steps = read_num_timesteps(run_dir)
    budget = (int(steps) - int(fork_step)) if (steps is not None and fork_step is not None) else None

    target_dir = target.get("run_dir")
    target_pins: Optional[bool] = None
    if target_dir and os.path.isdir(target_dir):
        try:
            target_pins = bool(read_recorded_trainee_teams(target_dir))
        except (FileNotFoundError, ValueError):
            target_pins = None

    try:
        teams = read_recorded_trainee_teams(run_dir)
    except FileNotFoundError as exc:
        raise RunReadError(
            f"{os.path.basename(run_dir)}: {exc}\n  A run records its --trainee-team(s) as the "
            "paths it was LAUNCHED with, which are repo-relative. Run this meter from the repo "
            f"root (cwd is {os.getcwd()!r}), or the team file has genuinely moved.") from exc
    arch, membership = archetype_of(teams, teamsets)
    series = read_series(run_dir, target.get("run_name"), fork_step)
    drow = dose_cli.read_run(run_dir)

    return ExploiterRun(
        run=os.path.basename(os.path.normpath(run_dir)), run_dir=run_dir,
        target_run=target.get("run_name"),
        target_file=target.get("resolved_file") or target.get("resolved_path"),
        target_step=target.get("resolved_num_timesteps") or target.get("num_timesteps"),
        target_pins_own_teams=target_pins,
        fork_step=fork_step, num_timesteps=steps, budget=budget,
        dose_rate=drow.get("dose_rate"), lr_median=drow.get("lr_median"),
        teams=list(teams), archetype=arch, membership=membership,
        regime=_regime(_read_json(os.path.join(run_dir, "metadata.json")), series, target_pins),
        series=series, lineage_derived=derived)


# --------------------------------------------------------------------------------------------
# Grouping — by TARGET, then by ROUND
# --------------------------------------------------------------------------------------------

def target_key(run: ExploiterRun) -> str:
    """The identity of the generalist being best-responded to — the FILE, not the run.

    Two exploiters that name the same run directory at different steps are best-responding to
    different generalists, and a bare-run-dir reference resolves to the run's LAST SNAPSHOT, which
    moves as the run trains. The recorded ``resolved_file`` is what was actually loaded.
    """
    return f"{run.target_file or run.target_run or '?'}@{run.target_step}"


def assign_rounds(runs: Sequence[ExploiterRun],
                  override: Optional[Dict[str, int]] = None) -> Dict[str, int]:
    """``{target_key: round}``. Inferred from the target's STEP, ascending, 1-based.

    The population loop's rounds ARE ordered by the generalist's depth: round N+1's target is
    round N's generalist after it absorbed round N's exploiters. ``override`` maps an exploiter
    run NAME or a target run NAME to a round and wins wherever it applies — for a loop whose
    rounds are not step-ordered (a re-run of round 2 from a re-trained generalist, say).
    """
    override = dict(override or {})
    keys: Dict[str, Optional[int]] = {}
    for r in runs:
        keys.setdefault(target_key(r), r.target_step)
    ordered = sorted(keys, key=lambda k: (keys[k] is None, keys[k] or 0, k))
    rounds = {k: i + 1 for i, k in enumerate(ordered)}
    for r in runs:
        if r.run in override:
            rounds[target_key(r)] = override[r.run]
        elif r.target_run and r.target_run in override:
            rounds[target_key(r)] = override[r.target_run]
    return rounds


# --------------------------------------------------------------------------------------------
# The matched-exploiter gate
# --------------------------------------------------------------------------------------------

@dataclass
class Mismatch:
    kind: str
    a: str
    b: str
    detail: str

    def message(self) -> str:
        return f"{self.kind.upper()} MISMATCH — {self.a} vs {self.b}: {self.detail}"

    def to_json(self) -> dict:
        return {"kind": self.kind, "a": self.a, "b": self.b, "detail": self.detail,
                "message": self.message()}


_ERRORS = {"budget": UnmatchedBudgetError, "dose": UnmatchedDoseError,
           "regime": UnmatchedRegimeError}


def _rel(a: float, b: float) -> float:
    lo, hi = min(abs(a), abs(b)), max(abs(a), abs(b))
    return float("inf") if lo == 0 else (hi - lo) / lo


def check_matched(runs: Sequence[ExploiterRun], *, budget_tol: float = DEFAULT_BUDGET_TOL,
                  dose_tol: float = DEFAULT_DOSE_TOL,
                  allow_unmatched: bool = False) -> List[Mismatch]:
    """Every pair of exploiters, on budget / dose / regime. Raises unless ``allow_unmatched``.

    Returns the mismatches so a permitted comparison still carries them — into the header, into
    the JSON, and into whatever the reader pastes into the ledger. An unrecorded confound is the
    failure this gate exists for.
    """
    found: List[Mismatch] = []
    for i, a in enumerate(runs):
        for b in runs[i + 1:]:
            if a.budget is not None and b.budget is not None:
                if _rel(a.budget, b.budget) > budget_tol:
                    found.append(Mismatch("budget", a.run, b.run,
                                          f"{a.budget:,} vs {b.budget:,} post-fork steps "
                                          f"({_rel(a.budget, b.budget) * 100:.1f}% apart, "
                                          f"tolerance {budget_tol * 100:.0f}%)"))
            elif a.budget is None or b.budget is None:
                found.append(Mismatch("budget", a.run, b.run,
                                      "one side recorded no budget (num_timesteps − fork_step)"))
            if a.dose_rate and b.dose_rate:
                if _rel(a.dose_rate, b.dose_rate) > dose_tol:
                    ratio = max(a.dose_rate, b.dose_rate) / min(a.dose_rate, b.dose_rate)
                    found.append(Mismatch("dose", a.run, b.run,
                                          f"{a.dose_rate:.4g} vs {b.dose_rate:.4g} "
                                          f"({ratio:.2f}x apart, tolerance "
                                          f"{dose_tol * 100:.0f}%)"))
            elif a.dose_rate is None or b.dose_rate is None:
                found.append(Mismatch("dose", a.run, b.run,
                                      "one side recorded no dose_rate (main.dose read nothing)"))
            diff = [k for k in sorted(set(a.regime) | set(b.regime))
                    if a.regime.get(k) != b.regime.get(k)]
            if diff:
                found.append(Mismatch(
                    "regime", a.run, b.run,
                    "; ".join(f"{k}: {a.regime.get(k)!r} vs {b.regime.get(k)!r}" for k in diff)))
    if found and not allow_unmatched:
        kind = found[0].kind
        raise _ERRORS[kind](
            "REFUSING the comparison — the exploiters are not matched:\n  "
            + "\n  ".join(m.message() for m in found)
            + "\n\n  A best-response gap is a property of (the generalist, the exploiter's "
              "budget, its dose, its regime). Comparing two rounds that differ on any of those "
              "attributes to the GENERALIST what belongs to the exploiter — which is exactly how "
              "the 2026-09-21 era-2/era-1 read was confounded, by a 4.5x dose gap nobody "
              "registered (--fork-lr unset, the new parent's annealed rate inherited).\n"
              "  Pass --allow-unmatched to print it anyway; the mismatch is then carried in the "
              "header and in the JSON, and belongs in every quote of the number.")
    return found


# --------------------------------------------------------------------------------------------
# The gap table
# --------------------------------------------------------------------------------------------

#: The gap's null. A best responder that cannot beat its target scores 0.5 against it; the gap is
#: how far above that the best response sits, so `gap = rate - NULL` and a loop that is absorbing
#: drives it toward 0.
NULL_RATE = 0.5


def _gap_row(run: ExploiterRun, stat: str) -> Dict[str, Any]:
    w, n = run.cell(stat)
    lo, hi = wilson_ci(w, n)
    ep = run.endpoint()
    pw, pn = run.pooled()
    return {
        "run": run.run, "archetype": run.archetype, "membership": run.membership,
        "wins": w, "games": n,
        "rate": (w / n) if n else None,
        "endpoint_rate": ep.rate if ep else None,
        "endpoint_step": ep.step if ep else None,
        "pooled_rate": (pw / pn) if pn else None, "pooled_games": pn,
        "gap": (w / n - NULL_RATE) if n else None,
        "gap_lo": lo - NULL_RATE if n else None,
        "gap_hi": hi - NULL_RATE if n else None,
        "n_cycles": len(run.post_fork),
    }


def build_report(runs: Sequence[ExploiterRun], *, stat: str = "pooled",
                 rounds: Optional[Dict[str, int]] = None,
                 mismatches: Sequence[Mismatch] = (),
                 draws: int = DEFAULT_DRAWS,
                 bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
                 rate_regime: str = SERIES_REGIME,
                 played: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The whole report: per-round rows, and the round-over-round delta PAIRED on archetype.

    ``stat`` selects which rate the gap is computed from — ``pooled`` (every post-fork cycle, the
    higher-power read) or ``endpoint`` (the last cycle, the convention the banked numbers use).
    Both are always shown per row; only one drives the gap and its CI, and the header says which.

    The delta is computed only between CONSECUTIVE rounds and only on archetypes present in both.
    An archetype in one round and not the other is reported as UNPAIRED rather than dropped in
    silence — dropping it would shrink the pairing unit count without saying so, and the count is
    what makes the interval readable.
    """
    if stat not in ("pooled", "endpoint"):
        raise BestResponseGapError(f"stat {stat!r}: expected 'pooled' or 'endpoint'")
    rounds = dict(rounds or assign_rounds(runs))
    for r in runs:
        r.round = rounds.get(target_key(r))

    by_round: Dict[int, Dict[str, Any]] = {}
    for r in sorted(runs, key=lambda x: (x.round or 0, x.archetype or "~", x.run)):
        blk = by_round.setdefault(r.round or 0, {
            "round": r.round, "target_run": r.target_run, "target_file": r.target_file,
            "target_step": r.target_step, "budget": r.budget, "dose_rate": r.dose_rate,
            "rows": [], "unassigned": []})
        row = _gap_row(r, stat)
        blk["rows"].append(row)
        if r.archetype is None:
            blk["unassigned"].append({"run": r.run, "why": r.membership})

    caveats: List[str] = []
    deltas: List[Dict[str, Any]] = []
    order = sorted(by_round)
    for earlier, later in zip(order, order[1:]):
        a = {row["archetype"]: row for row in by_round[earlier]["rows"] if row["archetype"]}
        b = {row["archetype"]: row for row in by_round[later]["rows"] if row["archetype"]}
        shared = sorted(set(a) & set(b))
        per: List[Dict[str, Any]] = []
        pairs: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
        for name in shared:
            d, lo, hi = newcombe_diff_ci(b[name]["wins"], b[name]["games"],
                                         a[name]["wins"], a[name]["games"])
            per.append({"archetype": name, "gap_earlier": a[name]["gap"],
                        "gap_later": b[name]["gap"], "delta": d, "lo": lo, "hi": hi})
            pairs.append(((b[name]["wins"], b[name]["games"]),
                          (a[name]["wins"], a[name]["games"])))
        for name in shared:
            if a[name]["membership"] != b[name]["membership"]:
                caveats.append(
                    f"round {earlier} -> {later}, archetype {name}: the TEAMSET SIZE changed "
                    f"({a[name]['membership']} -> {b[name]['membership']}). The two gaps are "
                    "best responses inside DIFFERENT subgame restrictions, so the delta "
                    "confounds 'the loop absorbed it' with 'the later exploiter had a harder "
                    "job'. `main.exploitability`'s caveat 2 is the same one.")
        point, plo, phi = paired_delta_ci(pairs, draws=draws, seed=bootstrap_seed)
        if point is None:
            verdict = "UNREADABLE — fewer than two archetypes are present in both rounds"
        elif plo is not None and phi is not None and phi < 0:
            verdict = "THE GAP FELL — the generalist is absorbing its best responders"
        elif plo is not None and plo > 0:
            verdict = "THE GAP ROSE — the later generalist is MORE exploitable, not less"
        else:
            verdict = "NOT DETECTED — the paired interval straddles zero"
        deltas.append({
            "earlier_round": earlier, "later_round": later, "per_archetype": per,
            "unpaired": sorted(set(a) ^ set(b)),
            "mean_delta": point, "lo": plo, "hi": phi, "n_pairs": len(pairs),
            "verdict": verdict,
        })

    return {
        "stat": stat, "null_rate": NULL_RATE, "rate_regime": rate_regime,
        "rounds": [by_round[k] for k in order],
        "deltas": deltas,
        "caveats": caveats,
        "mismatches": [m.to_json() for m in mismatches],
        "unmatched": bool(mismatches),
        "runs": [r.to_json() for r in runs],
        "played": played,
    }


# --------------------------------------------------------------------------------------------
# The optional PLAY path — fresh head-to-head games on the rust bridge
# --------------------------------------------------------------------------------------------

def play_head_to_head(run: ExploiterRun, *, games: int, seed: int = 0, impl: str = "rust",
                      greedy: bool = False, concurrency: int = 1,
                      progress=None) -> Dict[str, Any]:
    """``games`` fresh exploiter-vs-target battles, offline, on the bridge. Writes NOTHING.

    It reuses ``agents.training.untaught_meter.play_cells`` — the tool of record for offline
    head-to-head battles between two saved checkpoints — with the exploiter as the PILOT on its
    own recorded ``--trainee-team(s)`` pin and the target as the opponent drawing from the full
    team pool. That is the same geometry the training-time series measures, so the two numbers
    describe the same matchup; what differs is the REGIME, and the caller must say which.

    🚨 **A PLAYED NUMBER AND A SERIES NUMBER ARE DIFFERENT POPULATIONS unless ``greedy=True``.**
    The default here is the TRAINING regime (stochastic@1 both sides); the series is greedy both
    sides. The returned block carries its regime, and the report never folds a played rate into
    the gap table's series rows.

    ``games`` is split evenly across the pinned teams and ROUNDED UP, so the realised total is
    ``ceil(games / n_teams) * n_teams`` — reported as ``games_played`` rather than silently
    reinterpreting the request. Concurrency above 1 is refused by the engine (the shared streams
    are consumed in a scheduling-dependent order, so a seeded run stops reproducing).
    """
    from agents.training import untaught_meter as um

    if games <= 0:
        raise BestResponseGapError("--play N: N must be positive.")
    if not run.teams:
        raise BestResponseGapError(
            f"{run.run}: recorded NO pinned trainee teams, so there is no team set to replay the "
            "head-to-head on. A full-pool exploiter must be played through its own harness.")
    if not run.target_file or not os.path.isfile(run.target_file):
        raise BestResponseGapError(
            f"{run.run}: the recorded exploiter target file {run.target_file!r} is not on disk — "
            "refusing to re-resolve it under today's rule, which would play a different opponent "
            "than the series measured.")
    um.check_concurrency(concurrency)

    teams = um.team_slices(run.teams, prefix="X", source=f"{run.run} recorded trainee pin")
    exploiter = um.resolve_ref(run.run_dir, label=run.run, role="ref")
    target = um.resolve_ref(run.target_file, label=run.target_run or "target", role="opponent")
    per_team = -(-int(games) // len(teams))

    cells = um.play_cells([exploiter], teams, target, games_per_team=per_team, seed=seed,
                          impl=impl, concurrency=concurrency, stochastic=not greedy,
                          progress=progress)
    got = cells[exploiter.label]
    wins = sum(c.wins for c in got.values())
    finished = sum(c.finished for c in got.values())
    attempted = sum(c.attempted for c in got.values())
    ties = sum(c.ties for c in got.values())
    lo, hi = wilson_ci(wins, finished)
    timeouts = attempted - finished
    return {
        "run": run.run, "target": run.target_run, "target_file": run.target_file,
        "regime": PLAY_REGIME_GREEDY if greedy else PLAY_REGIME_STOCHASTIC,
        "greedy": greedy, "impl": impl, "seed": seed, "concurrency": concurrency,
        "games_requested": int(games), "games_per_team": per_team,
        "games_played": attempted, "finished": finished,
        "wins": wins, "ties": ties, "losses": finished - wins - ties,
        "timeouts": timeouts,
        "timeout_fraction": (timeouts / attempted) if attempted else None,
        "inconclusive": bool(attempted and timeouts / attempted > 0.25),
        "win_rate": (wins / finished) if finished else None,
        "ci": [lo, hi] if finished else None,
        "gap": (wins / finished - NULL_RATE) if finished else None,
        "per_team": {k: {"wins": c.wins, "finished": c.finished, "attempted": c.attempted}
                     for k, c in got.items()},
        "teams": [t.to_json() for t in teams],
    }
