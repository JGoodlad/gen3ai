"""main.critic_gate — THE WIN-PROB CRITIC ARM'S PRE-REGISTERED READ, in one command.

``designs/ai_v12/design_winprob_only_critic.md`` §5.5 registers, **before** the run, three
endpoints, a control and a falsification clause. Each half already has an instrument; nobody had
composed them, so producing the read meant four invocations, four output formats and a reader
holding the §4.3 bars in their head. This is that composition and nothing more — **every statistic
here is computed by the tool that owns it**:

    ladder        <run>/snapshot_ladder/ladder.json                          (the anchored BT fit)
    calibration   main.scaffolding_gauge --reliability --reliability-reweight    (imported)
    kill (G7)     the run's own recorded eval metrics + its trace summaries
    untaught      main.untaught_meter --baseline … --control …                    (invoked)

    export PYTHONPATH=$PYTHONPATH:src
    python -m main.critic_gate models/<run> --parent models/<parent> \\
        --control models/<contA> models/<contB> --md gate.md --json gate.json
    python -m main.critic_gate models/<run> --parent models/<parent> --check   # resolve only
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)

═══ THE FOUR THINGS IT REFUSES TO GET WRONG ═══════════════════════════════════════════════════

1. **The ladder is read at matched SNAPSHOT COUNT, never at matched step**, against the parent
   **CONTINUED** — and while the run is unfinished it prints ``rating not final``, because BT
   re-solves every node on every add and the newest is systematically inflated (gen-10's 12M fell
   2089 → 2021 over 12 refits). An unconverged or absent ladder REFUSES, naming the key.
2. **RESOLUTION is the primary calibration line** (§4.1: reliability is already ~0.002, so a
   promotion that improves ECE and leaves resolution flat has moved the meter that was never the
   disease). ``bot`` and ``pool`` are gated separately and NEVER pooled — their measured
   calibration bias has opposite sign.
3. **The bars come from the committed artifact**, read at run time from
   ``designs/research_state/measurements/winprob_critic_baseline_2026-09-06/`` — never hardcoded
   here, so the bar and the record cannot drift apart. **G2 and G3 are RELATIVE bars** (owner
   ruling 2026-09-06): §4.3's absolute 0.005 / 0.05 are ALREADY BREACHED by the baseline on the
   ``pool`` stratum, so the arm is asked to be no worse than its predecessor on each gated
   stratum, never to clear a number its predecessor never cleared. The absolutes are still
   computed and printed, as ASPIRATIONAL targets that gate nothing.
4. **The falsification clause is a verdict, not a footnote.** G1 flat with G2–G4 passing prints the
   design's own sentence verbatim and the tool does NOT report a pass. ``critic_gate_test.py``
   asserts the constant still matches the design file word for word.

═══ WHAT IT CANNOT CLAIM ══════════════════════════════════════════════════════════════════════

* G5 (``sd_true_excess``), G6 (the mirror table) and G8 (``win_mask`` coverage) are §6 gaps
  M1/M2/M3 and are **not computed here**. They are listed as NOT RUNNABLE in every report, because
  a gate with unrunnable criteria silently becomes the runnable ones under time pressure.
* The baseline artifact publishes no interval for ``resolution``, so G1 compares the ARM's
  cluster-bootstrap CI against the baseline as a FIXED bar. That is what a pre-registered bar is,
  but it is an asymmetry and it is printed.
* The stall rate is read off the CAPTURED traces, which are loss-enriched by design
  (``agents.training.trace_selection``). It is labelled as a quota statistic every time it is
  printed; the episode LENGTH beside it is the run's own full-cycle recorded metric.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from main.critic_gate_design import (BASELINE_ARTIFACT, DEFAULT_BASELINE_DIR,
                                     DEFAULT_MAX_EP_LEN_RATIO, DEFAULT_MAX_STALL_RATE, DESIGN_DOC,
                                     FALSIFICATION_CLAUSE, G1_BASELINE_REDUCE, G2_MAX_RELIABILITY,
                                     G3_MAX_ECE, GATED_STRATA, NOT_RUNNABLE,
                                     OWNER_RULING_2026_09_06, RELATIVE_BARS,
                                     RELATIVE_BASELINE_REDUCE, RELATIVE_RULE, RULE_BETTER,
                                     RULE_NO_CI, RULE_NONINFERIOR, RULE_WORSE,
                                     STALL_RATE_SOURCE_NOTE, Z95)
from main.critic_gate_render import render_markdown, render_text
from agents.training import baselines
from utils.paths import src_root

#: The FAMINE PRE-TEST's comparator, BY NAME out of `designs/baselines.json`
#: (`gen3_baselines_registry_v1`). It lived only in one ledger entry until 2026-09-06, so "which
#: run is the famine comparator, and what is its floor?" was a question answerable only by reading
#: prose. The FLOOR travels with it — `floor_elo` on that registry entry — because a bar and the
#: run it is a bar against are one fact, not two.
FAMINE_COMPARATOR_BASELINE = "famine_comparator"


class GateRefusal(SystemExit):
    """A missing / malformed input. Exit 2 with the offending key or path NAMED.

    A refusal, never a plausible-looking zero — the recorded-vs-derived-key defect class has cost
    this tree enough (see `main.exploitability`'s schema refusal, the same pattern).
    """

    def __init__(self, where: str, message: str) -> None:
        super().__init__(f"\n[critic_gate] REFUSAL — {where}\n\n  {message}\n")
        self.code = 2


# --------------------------------------------------------------------------- resolution

def _resolve_ref(spec: str, *, what: str) -> Dict[str, Any]:
    """A model ref → the file it names, through the ONE choke point (the last-snapshot rule).

    `gen3_last_snapshot_resolution_v1`: a bare run dir means the run's LAST SNAPSHOT, exactly as a
    launch means it. The rung and rule are recorded so no reader has to infer WHICH FILE was read.

    `gen3_baselines_registry_v1`: the spec may also be a NAME from `designs/baselines.json`
    (`--parent v9_fold_parent`), which expands to that entry's EXPLICIT checkpoint. The name and
    its one-line provenance are carried into the report, so a reader never has to recognise a path
    to know which run was meant.
    """
    from agents.training.fixed_opponent_pool import resolve_model_ref
    given, baseline_name, provenance = spec, None, None
    if baselines.is_name(spec):
        b = baselines.get(spec)
        spec, baseline_name, provenance = b.spec, b.name, b.describe()
    last: Exception = FileNotFoundError(f"{spec!r}: nothing tried")
    for cand in baselines.candidate_paths(spec):
        try:
            r = resolve_model_ref(cand)
        except (FileNotFoundError, ValueError) as exc:
            last = exc
            continue
        return {"spec": given, "baseline": baseline_name, "baseline_provenance": provenance,
                "resolved_file": r.zip_path, "run_dir": r.run_dir,
                "run_base": r.run_base, "resolution_rung": r.rung, "resolution_rule": r.rule,
                "resolved_num_timesteps": r.num_timesteps, "config": r.config_path}
    raise GateRefusal(f"{what} {given!r}", str(last))


def _resolve_famine(args, *, explicit: Optional[bool] = None
                    ) -> Tuple[Optional[Dict[str, Any]], Optional[float]]:
    """``(comparator, floor_elo)`` for the famine pre-test, or ``(None, None)`` when it is off.

    The FLOOR comes from the comparator's own registry entry unless ``--famine-floor-elo`` names
    one — a bar and the run it is a bar against are one fact. A comparator that is a raw REF with
    no registry entry therefore REFUSES rather than inventing a floor: 38 is a measurement about
    two specific runs, not a constant of nature.

    🚨 **THE DEFAULT YIELDS; AN EXPLICIT FLAG REFUSES** — the same asymmetry `--compile-trainer`
    settles the same way. The default comparator is a registry NAME resolving under `models/`, and
    `models/` is not committed: on a fresh clone, in CI, or on any box that does not carry rev-1,
    a default that refused would take the WHOLE read down over one endpoint of five that nobody
    asked for. So an unresolvable DEFAULT is recorded as NOT READ, naming the run and the reason,
    and every other section still runs. An unresolvable comparator the caller NAMED is still a
    refusal: they asked for it, so its absence is an error rather than a fact about this box.
    """
    spec = getattr(args, "famine_comparator", None)
    if not spec or spec == "off":
        return None, None
    # EXPLICIT is read from the ARGV, not by comparing the value against the default: a caller
    # who deliberately types the default NAME has still asked for it, and a value comparison
    # cannot tell those apart. `main` passes it. A direct caller that does not gets the
    # conservative reading — treat it as explicit and REFUSE — because silently NOT READING a
    # registered endpoint is the failure that must never happen by accident.
    if explicit is None:
        explicit = spec != FAMINE_COMPARATOR_BASELINE
    try:
        comparator = _resolve_ref(spec, what="--famine-comparator")
    except (SystemExit, baselines.BaselineError):
        if explicit:
            raise
        where = baselines.spec(spec) if baselines.is_name(spec) else spec
        return {"unavailable": (
            f"the DEFAULT famine comparator ({where}) is not on this box, so the "
            "pre-test was not read. Every other section still ran. Pass --famine-comparator with "
            "a ref that IS here, or `off` to say so deliberately."), "spec": spec}, None
    if args.famine_floor_elo is not None:
        return comparator, float(args.famine_floor_elo)
    if baselines.is_name(spec):
        floor = baselines.get(spec).floor_elo
        if floor is not None:
            return comparator, float(floor)
    raise GateRefusal(
        f"--famine-comparator {spec!r}",
        "no kill floor: this comparator carries no `floor_elo` in designs/baselines.json and no "
        "--famine-floor-elo was given. The floor is a MEASUREMENT about two specific runs (the max "
        "|delta| between them at matched steps), so it cannot be defaulted for an arbitrary "
        "comparator. Pass --famine-floor-elo, name a registry baseline that records one, or pass "
        "--famine-comparator off.")


def _run_is_finished(run_dir: str) -> Tuple[bool, Optional[str]]:
    """A run is FINISHED when it wrote a final model. Anything else is live, and a live run's
    newest ladder node is systematically inflated (the third ELO reading rule)."""
    for name in ("final_model.zip", "final_model_interrupted.zip"):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            return True, p
    return False, None


# --------------------------------------------------------------------------- (1) the ladder

def load_ladder(run_dir: str, *, what: str) -> Dict[str, Any]:
    """``<run>/snapshot_ladder/ladder.json``, fully validated. Refuses, naming the key."""
    path = os.path.join(run_dir, "snapshot_ladder", "ladder.json")
    if not os.path.exists(path):
        raise GateRefusal(f"{what} ladder", (
            f"no {path!r}. The pre-registered endpoint is the DENSE ladder (+-10), not `eval/elo` "
            "(+-29), so there is no substitute. Build it with:\n"
            f"    python -m agents.training.snapshot_ladder {run_dir} --backfill"))
    try:
        with open(path) as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as exc:
        raise GateRefusal(f"{what} ladder", f"{path!r} is unreadable as JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise GateRefusal(f"{what} ladder", f"{path!r}: top level is {type(doc).__name__}.")
    for key in ("ratings", "se", "converged"):
        if key not in doc:
            raise GateRefusal(f"{what} ladder", f"{path!r} has no {key!r} key — this ladder was "
                                                "written by a different schema and cannot be read.")
    if not doc["ratings"]:
        raise GateRefusal(f"{what} ladder", f"{path!r}: 'ratings' is EMPTY — no snapshot is rated, "
                                            "so there is no node to compare at any count.")
    if not doc["converged"]:
        raise GateRefusal(f"{what} ladder", (
            f"{path!r}: 'converged' is false. An unconverged BT fit's ratings and SEs are not a "
            "reading; re-fit it (`python -m agents.training.snapshot_ladder <run> --backfill`) "
            "before quoting a number."))
    return {"path": path, **doc}


def _nodes(ladder: Dict[str, Any]) -> List[Tuple[int, float, float]]:
    """``[(step, elo, se)]`` in step order."""
    se = ladder.get("se") or {}
    out: List[Tuple[int, float, float]] = []
    for k, v in ladder["ratings"].items():
        try:
            step = int(k)
        except (TypeError, ValueError) as exc:
            raise GateRefusal("ladder", f"{ladder['path']!r}: rating key {k!r} is not a "
                                        "step number.") from exc
        out.append((step, float(v), float(se.get(k, float("nan")))))
    return sorted(out)


def ladder_section(run: Dict[str, Any], parent: Dict[str, Any],
                   at_snapshots: Optional[int]) -> Dict[str, Any]:
    """Endpoint 1 — the anchored ladder at matched SNAPSHOT COUNT, against the parent CONTINUED."""
    lr = load_ladder(run["run_dir"], what="run")
    lp = load_ladder(parent["run_dir"], what="parent")
    nr, npar = _nodes(lr), _nodes(lp)
    avail = min(len(nr), len(npar))
    n = at_snapshots or avail
    if n < 1 or n > avail:
        raise GateRefusal("--at-snapshots", (
            f"asked for {n} snapshot(s); the run rates {len(nr)} and the parent rates "
            f"{len(npar)}, so the matched count can be at most {avail}. Comparing at MATCHED "
            "COUNT (never at matched step) is the third ELO reading rule."))
    a, b = nr[n - 1], npar[n - 1]
    d = a[1] - b[1]
    se = math.sqrt((a[2] ** 2 if math.isfinite(a[2]) else 0.0)
                   + (b[2] ** 2 if math.isfinite(b[2]) else 0.0))
    finished, final_path = _run_is_finished(run["run_dir"])
    pfinished, _ = _run_is_finished(parent["run_dir"])
    anchored = bool(lr.get("anchored_to_bots")) and bool(lp.get("anchored_to_bots"))
    return {
        "at_snapshots": n,
        "available_matched": avail,
        "run": {"ladder_path": lr["path"], "n_rated": len(nr),
                "anchored_to_bots": bool(lr.get("anchored_to_bots")),
                "fit_quality": lr.get("fit_quality"),
                "nodes": [{"i": i + 1, "step": s, "elo": e, "se": v, "ci95": Z95 * v}
                          for i, (s, e, v) in enumerate(nr[:n])],
                "node_at_count": {"step": a[0], "elo": a[1], "se": a[2], "ci95": Z95 * a[2]},
                "finished": finished, "final_model": final_path},
        "parent": {"ladder_path": lp["path"], "n_rated": len(npar),
                   "anchored_to_bots": bool(lp.get("anchored_to_bots")),
                   "fit_quality": lp.get("fit_quality"),
                   "nodes": [{"i": i + 1, "step": s, "elo": e, "se": v, "ci95": Z95 * v}
                             for i, (s, e, v) in enumerate(npar[:n])],
                   "node_at_count": {"step": b[0], "elo": b[1], "se": b[2], "ci95": Z95 * b[2]},
                   "finished": pfinished},
        "delta_elo": d, "delta_se": se, "delta_ci95": [d - Z95 * se, d + Z95 * se],
        "rating_final": finished,
        "rating_note": ("rating final — the run wrote a final model" if finished else
                        "rating not final — run not finished. BT re-solves every node on every "
                        "add and the NEWEST node is systematically inflated (gen-10's 12M fell "
                        "2089 -> 2021 over 12 refits), so this number will move."),
        "comparability": (
            "matched SNAPSHOT COUNT, never matched step. Both fits are anchored to the same "
            "pinned bots, so the delta is meaningful; its SE combines two independent fits and "
            "carries NO term for the anchor uncertainty they share." if anchored else
            "AT LEAST ONE LADDER IS NOT BOT-ANCHORED — its scale is arbitrary and the delta below "
            "is NOT cross-run comparable. Run `python -m agents.training.bot_elo_calibration`."),
    }


def famine_section(run: Dict[str, Any], comparator: Dict[str, Any],
                   at_snapshots: Optional[int], floor_elo: float,
                   floor_source: str) -> Dict[str, Any]:
    """THE FAMINE PRE-TEST — does terminal-only reward learn at the INCUMBENT's rate?

    The registered rule (ledger 2026-09-06, *FAMINE PRE-TEST*): at ~5M, if the arm's anchored
    rating at matched SNAPSHOT COUNT trails the comparator by more than ``floor_elo`` **AND**
    ``win_rate_vs_bots`` is not rising cycle over cycle, terminal-only starves ⇒ kill the arm and
    launch FROZEN-φ. This computes the FIRST half; the second is `win_rate_vs_bots`, which the
    G7 section already reads off the run's own recorded metrics and which no ladder can supply.

    🚨 **A TRAIL INSIDE THE FLOOR IS NOT EVIDENCE OF EQUIVALENCE.** The incumbent had PBRS *and*
    PopArt *and* the shaped critic, so this is a rate comparison ACROSS RECIPES against the
    incumbent's own run-to-run noise — the pre-registered confound, printed with the verdict.
    """
    lad = ladder_section(run, comparator, at_snapshots)
    trail = -lad["delta_elo"]                      # positive = the arm is BEHIND
    starving = trail > floor_elo
    return {
        "comparator": comparator, "floor_elo": floor_elo, "floor_source": floor_source,
        "ladder": lad, "trail_elo": trail, "exceeds_floor": starving,
        "rule": ("at ~5M: trailing the comparator by more than the floor at matched SNAPSHOT "
                 "COUNT **AND** win_rate_vs_bots not rising ⇒ terminal-only starves ⇒ kill the "
                 "arm and launch FROZEN-φ."),
        "half_computed": "the LADDER half only — win_rate_vs_bots is the AND-gate's other half.",
        "confound": ("the incumbent had PBRS AND PopArt AND the shaped critic, so this is a rate "
                     "comparison ACROSS RECIPES and the floor is the incumbent's own run-to-run "
                     "noise, not a replicate of this arm. A trail inside the floor is NOT evidence "
                     "the two recipes are equivalent — only that starvation has not been "
                     "demonstrated."),
    }


# --------------------------------------------------------------------------- (2) calibration

def load_baseline(baseline_dir: str) -> Dict[str, Any]:
    """The committed §4.1 bars, per stratum. Read from the artifact; NEVER hardcoded here."""
    path = os.path.join(baseline_dir, BASELINE_ARTIFACT)
    if not os.path.exists(path):
        raise GateRefusal("baseline artifact", (
            f"no {path!r}. G1's bar is the committed measurement of the head this design promotes; "
            "without it there is no pre-registered bar and the gate is not the gate."))
    try:
        with open(path) as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as exc:
        raise GateRefusal("baseline artifact", f"{path!r} unreadable as JSON: {exc}") from exc
    if not isinstance(doc, dict) or "reliability" not in doc:
        raise GateRefusal("baseline artifact", (
            f"{path!r} has no 'reliability' block — it was written without --reliability, so it "
            "carries no resolution to be a bar."))
    per: Dict[str, List[Dict[str, Any]]] = {}
    for blk in doc["reliability"]:
        if not blk.get("reweighted"):
            raise GateRefusal("baseline artifact", (
                f"{path!r} step {blk.get('step')} is NOT selection-reweighted ('reweighted' is "
                "false). The capture quota is loss-enriched and the raw table INVERTS the verdict "
                "(§4.2) — a raw bar is not a bar. Re-measure with --reliability-reweight."))
        for row in blk.get("strata", []):
            if row.get("kind") not in ("all", "class"):
                continue
            for key in ("resolution", "reliability", "ece", "skill"):
                if key not in row:
                    raise GateRefusal("baseline artifact",
                                      f"{path!r} stratum {row.get('name')!r} has no {key!r}.")
            per.setdefault(str(row["name"]), []).append({"step": int(blk["step"]), **row})
    if not per:
        raise GateRefusal("baseline artifact",
                          f"{path!r}: no `all`/`bot`/`pool` strata rows to read a bar from.")
    return {"path": path, "per_stratum": per,
            "run": (doc.get("meta") or {}).get("run_name"),
            "steps": sorted({int(b["step"]) for b in doc["reliability"]})}


def baseline_bar(baseline: Dict[str, Any], stratum: str, reduce: str,
                 metric: str = "resolution", at_step: Optional[int] = None) -> Dict[str, Any]:
    """One stratum's baseline value for ONE metric — at the MATCHED checkpoint, else reduced.

    ``metric`` names the same key on both sides — the committed artifact's and the gauge's
    ``reliability_table``'s — which is what makes "matched stratum, matched metric" checkable
    rather than asserted.

    ``at_step`` asks for the baseline's own row at that exact step. **A checkpoint must be judged
    against the matched checkpoint whenever one exists**, and this is not a nicety: without it a
    generation can be reported as inferior TO ITSELF, because its step-26M row would be compared
    against the reduction's step-28M value. The reduction is what answers the ordinary case, where
    the arm's step grid and the baseline's do not coincide at all; it is a fallback, and which of
    the two happened is recorded in ``matched`` and printed.

    G1 reduces ``resolution`` with ``max`` (the STRICTEST value to beat) and does NOT step-match —
    it is a single pre-registered bar by design. The relative bars reduce their lower-is-better
    metric with ``last`` (the baseline's own final checkpoint). Both defaults live in
    ``critic_gate_design``.
    """
    rows = baseline["per_stratum"].get(stratum)
    if not rows:
        raise GateRefusal("baseline artifact",
                          f"{baseline['path']!r} has no {stratum!r} stratum, so there is no "
                          "matched-stratum bar for it. G1 compares LIKE WITH LIKE or not at all.")
    vals = [(float(r[metric]), int(r["step"])) for r in rows]
    matched = False
    if at_step is not None and any(st == int(at_step) for _, st in vals):
        v, step = next((x, st) for x, st in vals if st == int(at_step))
        matched = True
    elif reduce == "max":
        v, step = max(vals)
    elif reduce == "min":
        v, step = min(vals)
    elif reduce == "last":
        v, step = sorted(vals, key=lambda t: t[1])[-1]
    else:
        v, step = sum(x for x, _ in vals) / len(vals), -1
    return {"metric": metric, "value": v, "resolution": v, "from_step": step, "reduce": reduce,
            "matched": matched, "per_step": {str(s): x for x, s in vals}}


def relative_verdict(point: float, ci: Optional[Sequence[float]],
                     base: float) -> Tuple[bool, str]:
    """The owner ruling's NON-INFERIORITY comparison, for one lower-is-better metric.

    Returns ``(passed, which clause decided it)``. The clauses, in the order they are tried:

    * the arm's point estimate is at or below the baseline's — better or equal, nothing to argue;
    * the arm's CI CONTAINS the baseline's value — the arm is above it, but not detectably, which
      is non-inferiority and deliberately NOT a claim that the arm is better;
    * the arm's whole CI sits above it — the only FAIL.

    A non-finite CI cannot support a non-inferiority claim, so the point alone decides and the row
    says so. That is the conservative direction: a missing interval never converts a worse point
    estimate into a pass.
    """
    lo, hi = ((float(ci[0]), float(ci[1])) if ci is not None and len(ci) == 2
              else (float("nan"), float("nan")))
    if point <= base:
        return True, RULE_BETTER
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return False, RULE_NO_CI
    if lo <= base <= hi:
        return True, RULE_NONINFERIOR
    return False, RULE_WORSE


def _gauge_strata_arrays(sl: Dict[str, Any], weights) -> "Dict[str, Tuple[Any, Any, Any, Any]]":
    """``{stratum: (p, y, battles, w)}`` for `all` + the GATED strata, using the gauge's own split."""
    import numpy as np

    from main.scaffolding_gauge import opponent_class
    p = np.asarray(sl["win_probs"], dtype=np.float64)
    y = np.asarray(sl["outcomes"], dtype=np.float64)
    b = np.asarray(sl["battles"])
    cls = np.array([opponent_class(o) for o in np.asarray(sl["opponents"]).tolist()])
    out: "Dict[str, Tuple[Any, Any, Any, Any]]" = {"all": (p, y, b, weights)}
    for c in GATED_STRATA:
        sel = cls == c
        if sel.any():
            out[c] = (p[sel], y[sel], b[sel], None if weights is None else weights[sel])
    return out


def calibration_section(run_dir: str, baseline: Dict[str, Any], *, bins: int, boot: int,
                        seed: int, max_battles_per_step: Optional[int], reduce: str,
                        relative_reduce: str, max_reliability: float, max_ece: float,
                        say=lambda _m: None) -> Dict[str, Any]:
    """Endpoint 2 — §4.3's G1-G4, per checkpoint, bot and pool separately, selection-reweighted.

    Every statistic is the scaffolding gauge's. ``build_reliability`` produces the canonical rows
    (so they are directly comparable with the committed artifact's); the intervals G1 and the
    relative bars need are the one thing the gauge does not publish, and they are obtained by
    calling the gauge's OWN ``reliability_table`` under the gauge's OWN ``cluster_bootstrap_ci``
    — never a second estimator. All three metrics share ONE seed, so they are resampled over the
    SAME battle draws and their intervals are mutually coherent.

    **G2 and G3 are RELATIVE bars** (owner ruling 2026-09-06): the arm must be no worse than the
    baseline's SAME-stratum value, by ``relative_verdict``. §4.3's absolute 0.005 / 0.05 are
    carried through as ASPIRATIONAL targets — computed, printed, and gating nothing.
    """
    from agents.training.scaffolding import cluster_bootstrap_ci, reliability_table
    from main.scaffolding_gauge import (SelectionWeightError, build_reliability,
                                        build_trace_selection_block, collect_slices,
                                        selection_weights, true_win_rates, win_rate_sources)

    slices, coverage = collect_slices(run_dir, max_battles_per_step=max_battles_per_step,
                                      seed=seed, say=say)
    try:
        rw = true_win_rates(run_dir)
    except SelectionWeightError as exc:
        # SURFACE the gauge's own refusal. Falling back to the unweighted table would produce a
        # number that looks identical and answers a different question (§4.2).
        raise GateRefusal("selection reweighting", str(exc)) from exc

    blocks = build_reliability(slices, bins=bins, n_boot=boot, seed=seed, reweight=rw)

    checkpoints: List[Dict[str, Any]] = []
    for blk in blocks:
        step = int(blk["step"])
        sl = slices[step]
        w, _ = selection_weights(sl["outcomes"], sl["battles"], sl["opponents"], rw.get(step, {}))
        arrays = _gauge_strata_arrays(sl, w)
        rows_by_name = {str(r["name"]): r for r in blk["strata"] if r["kind"] in ("all", "class")}
        strata: List[Dict[str, Any]] = []
        for name, (p, y, b, wi) in arrays.items():
            row = rows_by_name.get(name)
            if row is None:
                continue

            def _stat(key, _p=p, _y=y, _w=wi):
                def _fn(idx):
                    return reliability_table(_p[idx], _y[idx], bins=bins,
                                             weights=None if _w is None else _w[idx])[key]
                return cluster_bootstrap_ci(_fn, b, n_boot=boot, seed=seed + 300)

            lo, hi = _stat("resolution")
            bar = baseline_bar(baseline, name, reduce)
            skill, skill_lo = float(row["skill"]), float(row["skill_ci_lo"])

            # ---- G2 / G3: per-stratum NON-INFERIORITY against the same-stratum baseline.
            aspirational = {"reliability": max_reliability, "ece": max_ece}
            relative: Dict[str, Any] = {}
            for spec in RELATIVE_BARS:
                arm = float(row[spec.metric])
                arm_ci = list(_stat(spec.metric))
                rbar = baseline_bar(baseline, name, relative_reduce, metric=spec.metric,
                                    at_step=step)
                ok, rule = relative_verdict(arm, arm_ci, rbar["value"])
                target = float(aspirational[spec.metric])
                relative[spec.gate] = {
                    "metric": spec.metric, "label": spec.label,
                    "arm": arm, "arm_ci": arm_ci,
                    "baseline": rbar["value"], "baseline_from_step": rbar["from_step"],
                    "baseline_per_step": rbar["per_step"], "baseline_reduce": relative_reduce,
                    "baseline_matched_step": rbar["matched"],
                    "baseline_source": ("the baseline's OWN row at this step"
                                        if rbar["matched"] else
                                        f"no baseline row at this step; reduced "
                                        f"(`{relative_reduce}`) over the artifact's steps"),
                    "delta": arm - rbar["value"], "pass": bool(ok), "decided_by": rule,
                    "aspirational": target, "aspirational_met": bool(arm <= target),
                    "baseline_meets_aspirational": bool(rbar["value"] <= target),
                }

            strata.append({
                "stratum": name, "gated": name in GATED_STRATA,
                "n_states": int(row["n"]), "n_battles": int(row["n_battles"]),
                "ess": float(row["ess"]), "base_rate": float(row["base_rate"]),
                "resolution": float(row["resolution"]),
                "resolution_ci": [lo, hi],
                "baseline_resolution": bar["resolution"],
                "baseline_from_step": bar["from_step"],
                "baseline_per_step": bar["per_step"],
                "delta_resolution": float(row["resolution"]) - bar["resolution"],
                "reliability": float(row["reliability"]),
                "reliability_ci": relative["G2"]["arm_ci"],
                "baseline_reliability": relative["G2"]["baseline"],
                "ece": float(row["ece"]), "mce": float(row["mce"]),
                "ece_ci": relative["G3"]["arm_ci"],
                "baseline_ece": relative["G3"]["baseline"],
                "brier": float(row["brier"]), "uncertainty": float(row["uncertainty"]),
                "skill": skill, "skill_ci": [skill_lo, float(row["skill_ci_hi"])],
                "decomp_residual": float(row["decomp_residual"]),
                "relative": relative,
                "G1_resolution": bool(float(row["resolution"]) > bar["resolution"]
                                      and math.isfinite(lo) and lo > bar["resolution"]),
                "G2_reliability": relative["G2"]["pass"],
                "G2_decided_by": relative["G2"]["decided_by"],
                "G3_ece": relative["G3"]["pass"],
                "G3_decided_by": relative["G3"]["decided_by"],
                "G4_skill": bool(skill > 0.0 and math.isfinite(skill_lo) and skill_lo > 0.0),
            })
        checkpoints.append({"step": step, "reweighted": True,
                            "selection": blk.get("selection"), "strata": strata})

    gated = [s for c in checkpoints for s in c["strata"] if s["gated"]]
    return {
        "artifact": baseline["path"],
        "baseline_run": baseline["run"],
        "baseline_steps": baseline["steps"],
        "baseline_reduce": reduce,
        "relative_baseline_reduce": relative_reduce,
        "bins": bins, "boot": boot, "seed": seed,
        "coverage": coverage,
        "trace_selection": build_trace_selection_block(run_dir),
        "true_win_rate_sources": {str(k): v for k, v in win_rate_sources(run_dir).items()},
        "checkpoints": checkpoints,
        "bars": {"G1": "resolution > the matched-stratum committed baseline, CI clearing it",
                 "G2": "reliability NO WORSE than the same-stratum baseline: " + RELATIVE_RULE,
                 "G3": "ECE NO WORSE than the same-stratum baseline: " + RELATIVE_RULE,
                 "G4": "skill > 0 with the cluster CI clearing 0",
                 "aspirational_only": {"G2_max_reliability": max_reliability,
                                       "G3_max_ece": max_ece},
                 "owner_ruling": OWNER_RULING_2026_09_06},
        "verdict": {
            "G1": bool(gated) and all(s["G1_resolution"] for s in gated),
            "G2": bool(gated) and all(s["G2_reliability"] for s in gated),
            "G3": bool(gated) and all(s["G3_ece"] for s in gated),
            "G4": bool(gated) and all(s["G4_skill"] for s in gated),
            "n_gated_rows": len(gated),
        },
        "not_gated_note": ("`all` is reported for context and NEVER gated: it averages two "
                           "populations whose measured calibration bias has opposite sign."),
        "asymmetry_note": ("the committed baseline publishes no interval for `resolution`, so G1 "
                           "compares the ARM's cluster CI against the baseline as a FIXED bar. "
                           "G2/G3 inherit the same asymmetry: the baseline value they are compared "
                           "against is a point, and only the ARM carries an interval."),
        "owner_ruling": OWNER_RULING_2026_09_06,
        "relative_rule": RELATIVE_RULE,
    }


# --------------------------------------------------------------------------- (3) the kill

def _recorded_ep_len(run_dir: str) -> Dict[int, Dict[str, Any]]:
    """``{step: {ep_len_bots, ep_len_pool, source}}`` from the run's OWN recorded eval metrics.

    The FULL-cycle number, not a quota statistic: ``mean_ep_len_vs_bots`` and ``pool.mean_ep_len``
    are written per eval cycle into ``metadata.json`` (top-level ``latest_eval``, plus every
    ``snapshot_history[*].latest_eval``). ``eval_results.jsonl`` does **not** carry episode length
    — it is read for the cycle inventory only, which is why both files are consulted and named.
    """
    out: Dict[int, Dict[str, Any]] = {}
    meta_path = os.path.join(run_dir, "metadata.json")
    if not os.path.exists(meta_path):
        return out
    try:
        with open(meta_path) as fh:
            meta = json.load(fh)
    except (OSError, ValueError):
        return out
    if not isinstance(meta, dict):
        return out
    blocks: List[Tuple[Dict[str, Any], str]] = []
    if isinstance(meta.get("latest_eval"), dict):
        blocks.append((meta["latest_eval"], "metadata.latest_eval"))
    hist = meta.get("snapshot_history")
    if isinstance(hist, dict):
        for name, entry in sorted(hist.items()):
            if isinstance(entry, dict) and isinstance(entry.get("latest_eval"), dict):
                blocks.append((entry["latest_eval"], f"snapshot_history[{name}].latest_eval"))
    for blk, source in blocks:
        step = blk.get("step")
        if not isinstance(step, (int, float)):
            continue
        pool = blk.get("pool") if isinstance(blk.get("pool"), dict) else {}
        out[int(step)] = {"ep_len_bots": blk.get("mean_ep_len_vs_bots"),
                          "ep_len_pool": (pool or {}).get("mean_ep_len"),
                          "win_rate_vs_bots": blk.get("win_rate_vs_bots"),
                          "source": source}
    return out


def _trace_turns(run_dir: str, stall_turns: int) -> Dict[int, Dict[str, Any]]:
    """Per step: the CAPTURED battles' turn counts and how many read as a stall.

    A stall is ``turns >= stall_turns`` (``MAX_TURNS`` is also the forfeit deadline) or a result
    that is neither a win nor a loss. Loss-enriched by construction — labelled as such wherever it
    is printed, never presented as a population rate.
    """
    from main.prober.discovery import build_trace_tree
    tree = build_trace_tree(run_dir)
    out: Dict[int, Dict[str, Any]] = {}
    for sg in tree.steps:
        turns: List[int] = []
        stalls = 0
        unreadable = 0
        outcomes: Dict[str, int] = {}
        for og in sg.opponents:
            for bt in og.battles:
                try:
                    with open(bt.summary_path) as fh:
                        m = (json.load(fh).get("meta") or {})
                except (OSError, ValueError):
                    unreadable += 1
                    continue
                t = m.get("turns")
                res = str(m.get("result", "?")).upper()
                outcomes[res] = outcomes.get(res, 0) + 1
                if isinstance(t, (int, float)):
                    turns.append(int(t))
                    if int(t) >= stall_turns:
                        stalls += 1
                        continue
                if res not in ("WIN", "LOSS"):
                    stalls += 1
        n_traces = sum(outcomes.values())
        out[sg.step] = {
            "n_traces": n_traces, "n_with_turns": len(turns), "n_unreadable": unreadable,
            "stalls": stalls,
            "stall_rate": (stalls / n_traces) if n_traces else float("nan"),
            "mean_turns": (sum(turns) / len(turns)) if turns else float("nan"),
            "max_turns": max(turns) if turns else None,
            "outcomes": outcomes,
        }
    return out


def kill_section(run: Dict[str, Any], parent: Dict[str, Any], *, stall_turns: int,
                 max_stall_rate: float, max_ep_len_ratio: float) -> Dict[str, Any]:
    """G7 — the KILL condition. §3.2/§3.3 removed two anti-stall defences; this watches for them.

    A ``[0,1]`` critic cannot rank a timeout BELOW a loss, so a run that learns to run out the
    clock pays nothing in its own value function. That is why this is a kill, not a monitor.
    """
    ep = _recorded_ep_len(run["run_dir"])
    pep = _recorded_ep_len(parent["run_dir"])
    traces = _trace_turns(run["run_dir"], stall_turns)

    def _era(key: str) -> Optional[float]:
        vals = [float(v[key]) for v in pep.values() if isinstance(v.get(key), (int, float))]
        return max(vals) if vals else None

    era_bots, era_pool = _era("ep_len_bots"), _era("ep_len_pool")
    cycles: List[Dict[str, Any]] = []
    for step in sorted(set(ep) | set(traces)):
        e = ep.get(step, {})
        t = traces.get(step, {})
        row: Dict[str, Any] = {
            "step": step,
            "ep_len_bots": e.get("ep_len_bots"), "ep_len_pool": e.get("ep_len_pool"),
            "ep_len_source": e.get("source"),
            "stall_rate_captured": t.get("stall_rate"),
            "mean_turns_captured": t.get("mean_turns"),
            "max_turns_captured": t.get("max_turns"),
            "n_traces": t.get("n_traces"), "stalls": t.get("stalls"),
            "outcomes": t.get("outcomes"),
        }
        breaches: List[str] = []
        sr = row["stall_rate_captured"]
        if isinstance(sr, (int, float)) and math.isfinite(float(sr)) and float(sr) > max_stall_rate:
            breaches.append(f"stall rate {float(sr):.4f} > {max_stall_rate:.4f}")
        for key, era in (("ep_len_bots", era_bots), ("ep_len_pool", era_pool)):
            v = row[key]
            if isinstance(v, (int, float)) and era:
                ratio = float(v) / float(era)
                row[key + "_ratio_vs_era"] = ratio
                if ratio > max_ep_len_ratio:
                    breaches.append(f"{key} {float(v):.2f} is {ratio:.2f}x the era's {era:.2f} "
                                    f"(> {max_ep_len_ratio:.2f})")
        row["breaches"] = breaches
        row["kill"] = bool(breaches)
        cycles.append(row)

    return {
        "thresholds": {"max_stall_rate": max_stall_rate, "stall_turns": stall_turns,
                       "max_ep_len_ratio": max_ep_len_ratio,
                       "threshold_provenance": STALL_RATE_SOURCE_NOTE},
        "era": {"run": parent["run_base"], "ep_len_bots": era_bots, "ep_len_pool": era_pool,
                "note": "the ERA is the parent's own recorded eval metrics, worst (max) cycle"},
        "cycles": cycles,
        "kill": any(c["kill"] for c in cycles),
        "measured": any(c["stall_rate_captured"] is not None or c["ep_len_bots"] is not None
                        for c in cycles),
        "sources": {
            "episode_length": "metadata.json latest_eval / snapshot_history[*].latest_eval "
                              "(mean_ep_len_vs_bots, pool.mean_ep_len) — the FULL cycle",
            "stall_rate": "the captured eval traces' per-battle summary meta.turns / meta.result "
                          "— the CAPTURE QUOTA, which is loss-enriched by design "
                          "(agents.training.trace_selection), so read it as an upper-ish bound, "
                          "never as a population rate",
            "eval_results.jsonl": "carries no episode-length field; consulted for the cycle "
                                  "inventory only",
        },
    }


# --------------------------------------------------------------------------- (4) the meter

def meter_argv(run: Dict[str, Any], parent: Dict[str, Any], controls: List[Dict[str, Any]], *,
               games_per_team: int, rows: Sequence[str], from_rows: bool, workers: int,
               json_out: Optional[str], check: bool, dry_run: bool,
               extra: Sequence[str] = ()) -> List[str]:
    """The ``main.untaught_meter`` invocation — endpoint 3, WITH the continuation control."""
    argv = [sys.executable, "-m", "main.untaught_meter"]
    if from_rows:
        argv += list(rows) + ["--from-rows"]
    else:
        argv += [f"ARM={run['spec']}"]
    argv += ["--baseline", parent["spec"]]
    if controls:
        argv += ["--control"] + [c["spec"] for c in controls]
    if not from_rows:
        argv += ["--games-per-team", str(games_per_team), "--workers", str(workers)]
    if json_out:
        argv += ["--json", json_out]
    if check:
        argv += ["--check"]
    if dry_run:
        argv += ["--dry-run"]
    return argv + list(extra)


def meter_section(argv: List[str], json_out: Optional[str],
                  *, timeout: Optional[float] = None) -> Dict[str, Any]:
    """Run it and read its own JSON back. Its statistics stay ITS statistics — nothing re-derived."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        p for p in [env.get("PYTHONPATH", ""), str(src_root())] if p)
    t0 = time.time()
    proc = subprocess.run(argv, env=env, capture_output=True, text=True, timeout=timeout)
    doc: Optional[Dict[str, Any]] = None
    if json_out and os.path.exists(json_out):
        try:
            with open(json_out) as fh:
                doc = json.load(fh)
        except (OSError, ValueError):
            doc = None
    return {"argv": argv, "returncode": proc.returncode, "stdout": proc.stdout,
            "stderr": proc.stderr, "result": doc, "seconds": round(time.time() - t0, 1),
            "control_note": ("re-based on the continuation control, as §5.5 requires"
                             if "--control" in argv else
                             "NO --control was given: a delta against a FROZEN baseline alone "
                             "OVERSTATES an arm by whatever a plain continuation would have "
                             "gained (ledger 2026-09-06 cell 2: +3.45pp)")}


# --------------------------------------------------------------------------- the verdict

def verdict(doc: Dict[str, Any]) -> Dict[str, Any]:
    """The whole read as one word — and the falsification clause is one of the words."""
    cal = doc.get("calibration") or {}
    cv = cal.get("verdict") or {}
    kill = bool((doc.get("kill") or {}).get("kill"))
    g1, g2, g3, g4 = cv.get("G1"), cv.get("G2"), cv.get("G3"), cv.get("G4")
    if not cv.get("n_gated_rows"):
        return {"verdict": "NOT MEASURED",
                "why": "no gated (bot/pool) calibration row was produced — there is nothing to "
                       "read, which is not the same as a failure.",
                "falsified": False}
    if kill:
        return {"verdict": "KILL",
                "why": "G7 breached: " + "; ".join(
                    b for c in doc["kill"]["cycles"] for b in c["breaches"]),
                "falsified": False}
    if (not g1) and g2 and g3 and g4:
        # §5.5's own clause, verbatim, AS THE VERDICT — never a pass with a footnote.
        return {"verdict": "FALSIFIED — the wrong-meter trap",
                "why": FALSIFICATION_CLAUSE, "falsified": True}
    if g1 and g2 and g3 and g4:
        return {"verdict": "PASS",
                "why": "G1 (resolution, the primary endpoint) clears the committed baseline in "
                       "every gated stratum, G2-G4 hold, and G7 is not breached.",
                "falsified": False}
    failed = [n for n, v in (("G1", g1), ("G2", g2), ("G3", g3), ("G4", g4)) if not v]
    return {"verdict": "MIXED", "why": "criteria not met: " + ", ".join(failed),
            "falsified": False}


# --------------------------------------------------------------------------- --check

def check_inputs(args, run: Dict[str, Any], parent: Dict[str, Any],
                 controls: List[Dict[str, Any]],
                 famine: Optional[Dict[str, Any]] = None,
                 famine_floor: Optional[float] = None) -> Tuple[List[str], List[str]]:
    """Resolve every input; return ``(ok lines, problems)``. Computes NOTHING."""
    ok: List[str] = []
    bad: List[str] = []
    ok.append(f"run      {run['resolved_file']} "
              f"[rung={run['resolution_rung']} rule={run['resolution_rule']}]")
    ok.append(f"parent   {parent['resolved_file']} "
              f"[rung={parent['resolution_rung']} rule={parent['resolution_rule']}]")
    for d, label in ((run, "run"), (parent, "parent"), (famine, "famine")):
        if d and d.get("baseline_provenance"):
            ok.append(f"baseline {label} = {d['baseline_provenance']}")
    if famine is None:
        ok.append("famine   OFF (--famine-comparator off) — the pre-test is not read")
    elif famine.get("unavailable"):
        # NOT a problem: the DEFAULT comparator simply is not on this box. `--check` answers
        # "would this read run?", and it would — with one endpoint recorded as not read.
        ok.append(f"famine   NOT READ — {famine['unavailable']}")
    else:
        ok.append(f"famine   {famine['resolved_file']} [floor {famine_floor} ELO]")
        try:
            lad = load_ladder(famine["run_dir"], what="--famine-comparator")
            ok.append(f"ladder   {lad['path']} — {len(lad['ratings'])} rated, converged")
        except SystemExit as exc:
            bad.append(str(exc).strip())
    for c in controls:
        ok.append(f"control  {c['resolved_file']} [rung={c['resolution_rung']}]")
    for label, d in (("run", run), ("parent", parent)):
        try:
            lad = load_ladder(d["run_dir"], what=label)
            ok.append(f"ladder   {lad['path']} — {len(lad['ratings'])} rated, converged")
        except SystemExit as exc:
            bad.append(str(exc).strip())
    try:
        b = load_baseline(args.baseline_dir)
        ok.append(f"baseline {b['path']} — {b['run']} steps {b['steps']}, "
                  f"strata {sorted(b['per_stratum'])}")
    except SystemExit as exc:
        bad.append(str(exc).strip())
    try:
        from main.prober.discovery import build_trace_tree
        tree = build_trace_tree(run["run_dir"])
        if tree.is_empty:
            bad.append(f"no eval traces under {run['run_dir']!r} — the calibration gate reads "
                       "eval_traces/step_*/<opponent>/*_states.npz")
        else:
            ok.append(f"traces   {len(tree.steps)} step(s), {len(tree.all_battles())} battle(s)")
    except SystemExit as exc:
        bad.append(str(exc).strip())
    from main.scaffolding_gauge import SelectionWeightError, true_win_rates
    try:
        rw = true_win_rates(run["run_dir"])
        ok.append(f"reweight resolved for {len(rw)} step(s)")
    except SelectionWeightError as exc:
        bad.append(f"selection reweighting: {exc}")
    if _recorded_ep_len(run["run_dir"]):
        ok.append("ep_len   metadata.json latest_eval / snapshot_history")
    else:
        bad.append(f"no recorded episode length for {run['run_dir']!r} — metadata.json carries no "
                   "latest_eval / snapshot_history[*].latest_eval with mean_ep_len_vs_bots, so G7 "
                   "has no episode-length half")
    if not controls:
        ok.append("control  NONE — the untaught-meter delta vs the frozen baseline will OVERSTATE "
                  "this arm (ledger 2026-09-06 cell 2: a plain continuation moved it +3.45pp)")
    return ok, bad


# --------------------------------------------------------------------------- entry point

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m main.critic_gate",
        description="The win-prob critic arm's PRE-REGISTERED READ "
                    "(designs/ai_v12/design_winprob_only_critic.md §5.5): the anchored ladder at "
                    "matched snapshot count, the §4.3 calibration gate with RESOLUTION primary, "
                    "the G7 kill condition, and the untaught meter with a continuation control. "
                    "Composes the existing tools; invents no statistics.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Every bar is READ from the committed baseline artifact, never hardcoded. G2 and "
               "G3 are PER-STRATUM RELATIVE bars (owner ruling 2026-09-06) and 4.3's absolute "
               "numbers are printed as aspirational targets only. G1 flat with G2-G4 passing "
               "prints the design's falsification clause AS THE VERDICT, never a pass with a "
               "footnote.")
    ap.add_argument("run", help="the arm's run directory (or any ref the last-snapshot rule "
                                "resolves: <run>, <run>@<step>, a .zip)")
    ap.add_argument("--parent", required=True, metavar="REF|BASELINE",
                    help="the generation being replaced — its ladder is the comparator and its "
                         "last snapshot is the untaught meter's frozen baseline. A NAME from "
                         "designs/baselines.json also works (e.g. `--parent v9_fold_parent`)")
    ap.add_argument("--famine-comparator", default=FAMINE_COMPARATOR_BASELINE,
                    metavar="REF|BASELINE",
                    help=f"the FAMINE PRE-TEST's comparator — does terminal-only reward learn at "
                         f"the incumbent's rate? (default: the {FAMINE_COMPARATOR_BASELINE!r} "
                         f"baseline). 'off' skips the pre-test.")
    ap.add_argument("--famine-floor-elo", type=float, default=None, metavar="ELO",
                    help="the kill floor in ELO (default: the comparator baseline's own recorded "
                         "floor_elo — 38, the max |delta| between two same-class runs at matched "
                         "steps, NOT the adjacent-node spread)")
    ap.add_argument("--control", nargs="+", action="extend", default=[], metavar="REF",
                    help="CONTINUATION arms of the parent at matched depth. §5.5: a FROZEN parent "
                         "is the wrong baseline — it credits an arm with progress the baseline "
                         "would have made anyway (ledger 2026-09-06 cell 2, +3.45pp).")
    ap.add_argument("--at-snapshots", type=int, default=None, metavar="N",
                    help="compare the ladders at N snapshots (default: the matched maximum)")
    ap.add_argument("--baseline-dir", default=DEFAULT_BASELINE_DIR, metavar="DIR",
                    help="the committed calibration baseline directory "
                         "(default: the winprob_critic_baseline_2026-09-06 measurement)")
    ap.add_argument("--baseline-reduce", choices=("max", "min", "mean", "last"), default=None,
                    help="how to reduce the baseline's several steps into ONE value per stratum. "
                         f"Applies to BOTH gate families when given. Unset: G1 uses "
                         f"`{G1_BASELINE_REDUCE}` (the STRICTEST resolution to beat) and G2/G3 use "
                         f"`{RELATIVE_BASELINE_REDUCE}` (the baseline's own final checkpoint)")
    ap.add_argument("--reliability-bins", type=int, default=10,
                    help="equal-width forecast bins (default 10, the gauge's own default)")
    ap.add_argument("--boot", type=int, default=400,
                    help="cluster-bootstrap resamples over BATTLES (default 400)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-battles-per-step", type=int, default=None,
                    help="seeded subsample of whole BATTLES per step (clusters stay intact)")
    ap.add_argument("--max-reliability", type=float, default=G2_MAX_RELIABILITY,
                    help=f"design 4.3's ABSOLUTE reliability number (default "
                         f"{G2_MAX_RELIABILITY}). ASPIRATIONAL: printed, never gated — G2 is the "
                         "per-stratum non-inferiority bar (owner ruling 2026-09-06)")
    ap.add_argument("--max-ece", type=float, default=G3_MAX_ECE,
                    help=f"design 4.3's ABSOLUTE ECE number (default {G3_MAX_ECE}). "
                         "ASPIRATIONAL: printed, never gated — G3 is the per-stratum "
                         "non-inferiority bar (owner ruling 2026-09-06)")
    ap.add_argument("--max-stall-rate", type=float, default=DEFAULT_MAX_STALL_RATE,
                    help=f"G7 kill threshold (default {DEFAULT_MAX_STALL_RATE}; "
                         f"{STALL_RATE_SOURCE_NOTE})")
    ap.add_argument("--stall-turns", type=int, default=None, metavar="N",
                    help="a battle at or past this many turns reads as a stall (default MAX_TURNS, "
                         "which is also the forfeit deadline)")
    ap.add_argument("--max-ep-len-ratio", type=float, default=DEFAULT_MAX_EP_LEN_RATIO,
                    help="G7: kill above this multiple of the ERA's recorded episode length "
                         f"(default {DEFAULT_MAX_EP_LEN_RATIO})")
    ap.add_argument("--games-per-team", type=int, default=200,
                    help="untaught-meter battles per team (default 200)")
    ap.add_argument("--meter-workers", type=int, default=1,
                    help="untaught-meter shards over TEAMS")
    ap.add_argument("--meter-rows", nargs="+", action="extend", default=[], metavar="[LABEL=]PATH",
                    help="committed per-team rows artifacts — runs the meter with --from-rows "
                         "(no models, no battles)")
    ap.add_argument("--meter-timeout", type=float, default=None, metavar="SEC")
    ap.add_argument("--skip-meter", action="store_true",
                    help="omit endpoint 3 entirely (the report says so in print)")
    ap.add_argument("--json", dest="json_out", default=None, metavar="PATH")
    ap.add_argument("--md", dest="md_out", default=None, metavar="PATH")
    ap.add_argument("--check", action="store_true",
                    help="resolve every input and exit non-zero on any miss. Computes nothing.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan (and the meter's) and exit 0")
    ap.add_argument("--quiet", action="store_true")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    say = (lambda _m: None) if args.quiet else (lambda m: print(f"[critic_gate] {m}", flush=True))

    run = _resolve_ref(args.run, what="run")
    parent = _resolve_ref(args.parent, what="--parent")
    controls = [_resolve_ref(c, what="--control") for c in args.control]
    raw_argv = list(argv if argv is not None else sys.argv[1:])
    famine, famine_floor = _resolve_famine(args, explicit=any(
        a == "--famine-comparator" or a.startswith("--famine-comparator=")
        for a in raw_argv))
    for d, label in ((run, "run"), (parent, "--parent"), (famine, "--famine-comparator")):
        if d and d.get("baseline_provenance"):
            print(f"[baseline] {label}: {d['baseline_provenance']}")

    if args.stall_turns is None:
        from agents.observation.constants import MAX_TURNS
        args.stall_turns = int(MAX_TURNS)

    # An explicit --baseline-reduce governs BOTH families; unset, each keeps its own default,
    # because "the strictest resolution to beat" and "the baseline's own last checkpoint" are
    # different questions and one word cannot answer both by accident.
    g1_reduce = args.baseline_reduce or G1_BASELINE_REDUCE
    rel_reduce = args.baseline_reduce or RELATIVE_BASELINE_REDUCE

    resolve_only = bool(args.check or args.dry_run)
    meter_json = (None if (resolve_only or args.skip_meter) else
                  os.path.join(tempfile.gettempdir(), f"critic_gate_meter_{os.getpid()}.json"))
    margv = meter_argv(run, parent, controls, games_per_team=args.games_per_team,
                       rows=args.meter_rows, from_rows=bool(args.meter_rows),
                       workers=args.meter_workers, json_out=meter_json,
                       check=args.check, dry_run=args.dry_run)

    if resolve_only:
        ok, bad = check_inputs(args, run, parent, controls, famine, famine_floor)
        for line in ok:
            print("  " + line)
        print("  meter    " + " ".join(margv[2:]))
        mt = meter_section(margv, None, timeout=args.meter_timeout)
        for line in (mt["stdout"] or "").rstrip().splitlines():
            print("    | " + line)
        if mt["returncode"] != 0:
            bad.append("untaught_meter refused:\n" + (mt["stderr"] or "").strip())
        if bad:
            print("\n[critic_gate] --check FAILED:", file=sys.stderr)
            for b in bad:
                print("  ✗ " + b, file=sys.stderr)
            return 1
        print(f"\n[critic_gate] --{'check' if args.check else 'dry-run'}: every input resolved — OK")
        return 0

    thresholds = {
        "G1": "resolution > the committed baseline's matched-stratum value, CI clearing it",
        "G2": "reliability no worse than the same-stratum baseline. " + RELATIVE_RULE,
        "G3": "ECE no worse than the same-stratum baseline. " + RELATIVE_RULE,
        "G4": "skill > 0, CI clearing 0",
        "aspirational_only": {"G2_max_reliability": args.max_reliability,
                              "G3_max_ece": args.max_ece,
                              "note": "design 4.3's absolute numbers. PRINTED, never gated."},
        "owner_ruling": OWNER_RULING_2026_09_06,
        "G7_max_stall_rate": args.max_stall_rate, "G7_stall_turns": args.stall_turns,
        "G7_max_ep_len_ratio": args.max_ep_len_ratio,
        "G7_threshold_provenance": STALL_RATE_SOURCE_NOTE,
        "baseline_reduce": g1_reduce, "relative_baseline_reduce": rel_reduce,
        "baseline_reduce_source": ("--baseline-reduce" if args.baseline_reduce else "per-family "
                                   "default (critic_gate_design)"),
        "reliability_bins": args.reliability_bins,
        "bootstrap_resamples": args.boot, "seed": args.seed,
        "games_per_team": args.games_per_team,
    }
    doc: Dict[str, Any] = {
        "tool": "critic_gate", "tool_version": 1,
        "_meta": {"run": run, "parent": parent, "controls": controls,
                  "design": DESIGN_DOC, "baseline_dir": args.baseline_dir,
                  "thresholds": thresholds,
                  "argv": list(argv if argv is not None else sys.argv[1:]),
                  "volatile": {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                               "cwd": os.getcwd()}},
        "not_runnable": [{"id": g, "criterion": w, "why": y} for g, w, y in NOT_RUNNABLE],
        "falsification_clause": FALSIFICATION_CLAUSE,
    }

    say("(1) ladder")
    doc["ladder"] = ladder_section(run, parent, args.at_snapshots)
    say("(2) calibration gate")
    baseline = load_baseline(args.baseline_dir)
    doc["calibration"] = calibration_section(
        run["run_dir"], baseline, bins=args.reliability_bins, boot=args.boot, seed=args.seed,
        max_battles_per_step=args.max_battles_per_step, reduce=g1_reduce,
        relative_reduce=rel_reduce, max_reliability=args.max_reliability,
        max_ece=args.max_ece, say=say)
    if famine is not None and famine.get("unavailable"):
        # The DEFAULT comparator is not on this box — recorded as NOT READ, never as `off`
        # (which means somebody chose to skip it) and never as a refusal of the whole read.
        doc["famine"] = dict(famine)
    elif famine is not None and famine_floor is not None:
        say("(2b) famine pre-test")
        try:
            doc["famine"] = famine_section(run, famine, args.at_snapshots, famine_floor,
                                           "--famine-floor-elo" if args.famine_floor_elo is not None
                                           else f"registry `{args.famine_comparator}`.floor_elo")
        except SystemExit as exc:
            # A comparator with no usable ladder must not take down the WHOLE read — the famine
            # pre-test is one endpoint beside four, and its refusal is recorded as such.
            doc["famine"] = {"unavailable": str(exc).strip(), "comparator": famine,
                             "floor_elo": famine_floor}
    else:
        doc["famine"] = None
    say("(3) G7 kill condition")
    doc["kill"] = kill_section(run, parent, stall_turns=args.stall_turns,
                               max_stall_rate=args.max_stall_rate,
                               max_ep_len_ratio=args.max_ep_len_ratio)
    if args.skip_meter:
        doc["untaught_meter"] = None
    else:
        say("(4) untaught meter")
        doc["untaught_meter"] = meter_section(margv, meter_json, timeout=args.meter_timeout)
    doc["verdict"] = verdict(doc)

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(doc, fh, indent=1)
        say(f"wrote {args.json_out}")
    md = render_markdown(doc)
    if args.md_out:
        with open(args.md_out, "w") as fh:
            fh.write(md + "\n")
        say(f"wrote {args.md_out}")
    if not args.quiet:
        print()
        print(render_text(doc))
    return 0 if doc["verdict"]["verdict"] == "PASS" else 1


if __name__ == "__main__":                                       # pragma: no cover
    sys.exit(main())
