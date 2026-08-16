"""main.endofrun — the AUTOMATED end-of-run battery: every verdict a generation needs, one command.

The generation loop turns over every ~2 days, and each turnover ends the same way: fit the dense
ladder, run the critic-route audit, run the per-family edge ablation, read the awareness/coverage
instruments, then apply the PRE-REGISTERED decision rules by hand from a runbook. This module
mechanizes that: the runbooks (`designs/research_state/gen*_endofrun_runbook.md`) remain the
pre-registration — the rules are encoded here as PURE FUNCTIONS citing them — and one invocation
produces one verdict artifact:

    export PYTHONPATH=$PYTHONPATH:src
    python -m main.endofrun models/<run> [--ref models/<prev-run>] [--max-states 6000]
        [--skip elo,audits,awareness] [--out designs/research_state/measurements]

Outputs `<out>/<run>_endofrun.json` (every measured number + every verdict with the rule that
produced it) and a Markdown report beside it. Steps are independent and fail SOFT — a step that
cannot run records WHY (the honest-status rule: an error is never an empty result):

  * **elo**       — dense offline anchored ladder for the run (and the reference when given),
                    compared by the TAIL-K convention (mean of the last K snapshot ratings at
                    matched count — the ELO reading rules: end-of-run only, never the inflated
                    newest node of a mid-run fit).
  * **audits**    — the final checkpoint over stratified eval-trace states: `critic_route_audit`
                    (route arms → the §2 deletion rule) and `edge_ablation_audit` (per-family →
                    the family-alive rule). LOADS THE MODEL: on an arch-drifted run it records
                    `needs_pinned_tree` with the exact worktree commands instead of failing.
  * **awareness** — model-free: `ProbeSession.awareness_scan` twice (losses → the blind/cap
                    verdicts; all outcomes → quantile coverage) vs the recorded gen-10 baselines.

The verdicts are DECISION-SUPPORT, not decisions: the report prints rule → number → verdict and
flags every pre-registered confound (route substitutability; the coverage loss-filter bias).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
from typing import Dict, List, Optional

# ------------------------------------------------------------------ pre-registered baselines
# Gen-10 instrument baselines (measured 2026-08-15; ledger "Dist-head instrument baselines").
AWARENESS_BASELINES = {
    "blind_loss_fraction": 0.072,
    "median_lead_time": 7.0,
    "cap_aware_ge_bar_fraction": 0.50,
    "coverage80": 0.44,
    "pit_mean": 0.396,
}

TAIL_K = 4                    # the tail-K ladder convention (the gen-8 resolution lesson)
NONINF_MARGIN = -15.0         # §5 convention: within −15 …
NONINF_HARD = -40.0           # … with the CI excluding −40
ROUTE_DV_FRAC = 0.20          # §2: single-arm |dV| < 20% of all_off …
ROUTE_FLIP_MAX = 0.02         # … AND < 2% flips ⇒ deletion candidate


# ------------------------------------------------------------------ the pure verdict rules

def tail_rating(curve: List[tuple], k: int = TAIL_K) -> Optional[dict]:
    """Mean of the last ``k`` snapshot ratings (+ pooled SE). ``curve`` = ``[(step, elo, se)]``
    from ``EloFit.snapshot_curve()``. None with fewer than ``k`` snapshots (an under-sampled
    tail is exactly the mid-run reading the convention forbids)."""
    if len(curve) < k:
        return None
    tail = curve[-k:]
    elos = [e for _s, e, _se in tail]
    ses = [se for _s, _e, se in tail]
    return {"k": k, "steps": [s for s, _e, _se in tail],
            "elo": statistics.mean(elos),
            "se": (sum(se * se for se in ses) / len(ses)) ** 0.5}


def non_inferiority(cur: Optional[dict], ref: Optional[dict],
                    margin: float = NONINF_MARGIN, hard: float = NONINF_HARD) -> dict:
    """The §1/§5 rule: NON_INFERIOR when the tail-K delta is within ``margin`` AND the 95% CI
    excludes ``hard``; INFERIOR when the whole CI sits below ``margin``; else INCONCLUSIVE."""
    if cur is None or ref is None:
        return {"verdict": "UNAVAILABLE",
                "why": "tail rating missing on one side (under-sampled ladder or no --ref)"}
    delta = cur["elo"] - ref["elo"]
    se = (cur["se"] ** 2 + ref["se"] ** 2) ** 0.5
    lo, hi = delta - 1.96 * se, delta + 1.96 * se
    if delta >= margin and lo > hard:
        verdict = "NON_INFERIOR"
    elif hi < margin:
        verdict = "INFERIOR"
    else:
        verdict = "INCONCLUSIVE"
    return {"verdict": verdict, "delta": round(delta, 1),
            "ci95": [round(lo, 1), round(hi, 1)],
            "rule": f"delta >= {margin} and CI-low > {hard}"}


def route_verdicts(arms: Dict[str, dict], dv_frac: float = ROUTE_DV_FRAC,
                   flip_max: float = ROUTE_FLIP_MAX) -> dict:
    """The §2 consolidation rule over `critic_route_audit` arms: a route whose single arm reads
    < ``dv_frac`` of the `all_off` |dV| AND < ``flip_max`` flips is a DELETION_CANDIDATE; else
    KEEP. Arms that are not deletable routes (`all_off`, `nmr`, `event_seats`,
    `hidden_opp_both/pi`) get READ-ONLY notes instead of verdicts. Carries the pre-registered
    confound: routes are partial substitutes — when `all_off` far exceeds the sum of singles,
    small single arms mean SHARED content, not unused routes."""
    out: dict = {"per_route": {}, "notes": []}
    all_off = arms.get("all_off")
    if all_off is None or all_off.get("dv_mean", 0.0) <= 0.0:
        out["notes"].append("no all_off arm / zero joint |dV| — the ratio rule cannot run")
        return out
    joint = all_off["dv_mean"]
    deletable = [a for a in ("seed", "threat", "hidden_opp_vf", "intent_reduce", "entity_pool")
                 if a in arms]
    single_sum = 0.0
    for a in deletable:
        row = arms[a]
        frac = row["dv_mean"] / joint
        single_sum += row["dv_mean"]
        out["per_route"][a] = {
            "dv_frac_of_all_off": round(frac, 3),
            "flip_rate": row["flip_rate"],
            "verdict": ("DELETION_CANDIDATE"
                        if frac < dv_frac and row["flip_rate"] < flip_max else "KEEP"),
            "rule": f"|dV| < {dv_frac:.0%} of all_off and flips < {flip_max:.0%}",
        }
    if single_sum < 0.6 * joint:
        out["notes"].append(
            f"CONFOUND (pre-registered): single arms sum to {single_sum:.3g} vs all_off "
            f"{joint:.3g} — heavy shared content; read the joint arm before deleting anything.")
    for a in ("nmr", "event_seats", "hidden_opp_pi", "hidden_opp_both"):
        if a in arms:
            out["per_route"][a] = {**{k: round(v, 4) for k, v in arms[a].items()},
                                   "verdict": "READ",
                                   "rule": "informational arm — its decision rule lives in the "
                                           "runbook prose, not the ratio rule"}
    return out


def family_verdicts(fams: Dict[str, dict], targets=("h",)) -> dict:
    """The family-alive rule (gen-12 runbook §2): a zero-init family is ALIVE when its |dV|
    reaches at least HALF the median of the other (live) families; both-null ⇒ NULL."""
    out = {}
    for t in targets:
        if t not in fams:
            out[t] = {"verdict": "ABSENT"}
            continue
        others = [v["dv_mean"] for k, v in fams.items() if k != t]
        if not others:
            out[t] = {"verdict": "UNCOMPARABLE"}
            continue
        med = statistics.median(others)
        row = fams[t]
        alive = med > 0 and row["dv_mean"] >= 0.5 * med
        out[t] = {"dv_mean": round(row["dv_mean"], 4),
                  "median_live_family_dv": round(med, 4),
                  "kl_mean": round(row.get("kl_mean", 0.0), 6),
                  "verdict": "ALIVE" if alive else "NULL",
                  "rule": "|dV| >= 0.5 x median live family"}
    return out


def awareness_verdicts(loss_agg: dict, all_agg: dict,
                       baselines: dict = AWARENESS_BASELINES) -> dict:
    """Instrument reads vs the recorded gen-10 baselines — DIRECTIONS, not pass/fail (the
    runbook treats these as instrument verdicts). Coverage is judged on the ALL-outcomes scan
    (the loss filter biases PIT low by construction — pre-registered)."""
    qc = (all_agg or {}).get("quantile_coverage") or {}
    rows = {
        "blind_loss_fraction": ((loss_agg or {}).get("blind_loss_fraction"),
                                baselines["blind_loss_fraction"]),
        "median_lead_time": ((loss_agg or {}).get("median_lead_time"),
                             baselines["median_lead_time"]),
        "cap_aware_ge_bar_fraction": ((loss_agg or {}).get("cap_aware_ge_bar_fraction"),
                                      baselines["cap_aware_ge_bar_fraction"]),
        "coverage80": (qc.get("coverage80"), baselines["coverage80"]),
        "pit_mean": (qc.get("pit_mean"), baselines["pit_mean"]),
    }
    out = {}
    better_up = {"median_lead_time", "cap_aware_ge_bar_fraction", "coverage80"}
    _EPS = 0.005          # rounding-noise floor: a delta under this is UNCHANGED, not a verdict
    for k, (cur, base) in rows.items():
        if cur is None:
            out[k] = {"verdict": "UNAVAILABLE"}
            continue
        if k == "pit_mean":
            d = abs(base - 0.5) - abs(cur - 0.5)          # >0 ⇒ moved toward calibrated
            direction = ("UNCHANGED" if abs(d) < _EPS
                         else ("IMPROVED" if d > 0 else "WORSE"))
        else:
            d = (cur - base) if k in better_up else (base - cur)
            direction = ("UNCHANGED" if abs(cur - base) < _EPS
                         else ("IMPROVED" if d > 0 else "WORSE"))
        out[k] = {"current": cur, "gen10_baseline": base, "verdict": direction}
    return out


# ------------------------------------------------------------------ the steps

def step_elo(run_dir: str, ref_run: Optional[str], anchors: str) -> dict:
    try:
        from main import elo as elo_cli
        _rows, fit, _a = elo_cli.analyze(run_dir, source="auto", anchors_path=anchors)
        cur = tail_rating(fit.snapshot_curve())
        ref = None
        if ref_run:
            _r2, fit2, _a2 = elo_cli.analyze(ref_run, source="auto", anchors_path=anchors)
            ref = tail_rating(fit2.snapshot_curve())
        return {"status": "ok",
                "current_tail": cur, "reference_tail": ref,
                "non_inferiority": non_inferiority(cur, ref),
                "curve_len": len(fit.snapshot_curve())}
    except Exception as e:  # noqa: BLE001 — a step failure is a recorded fact, not a crash
        return {"status": "error", "why": f"{type(e).__name__}: {e}"}


def _resolve_final_checkpoint(run_dir: str) -> Optional[str]:
    latest = os.path.join(run_dir, "latest.txt")
    if os.path.exists(latest):
        rel = open(latest).read().strip()
        p = os.path.join(run_dir, rel)
        if os.path.exists(p):
            return p
    for name in ("final_model.zip", "final_model_interrupted.zip"):
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            return p
    return None


def step_audits(run_dir: str, max_states: int, batch: int) -> dict:
    from agents.model.edge_ablation_audit import _collect_states
    from agents.model.edge_ablation_audit import audit as family_audit
    from agents.model.critic_route_audit import audit as route_audit

    ckpt = _resolve_final_checkpoint(run_dir)
    if ckpt is None:
        return {"status": "error", "why": "no final checkpoint (latest.txt / final_model*.zip)"}
    pattern = os.path.join(run_dir, "eval_traces", "**", "*_states.npz")
    try:
        obs, masks, coverage = _collect_states([pattern], max_states)
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "why": f"state collection: {type(e).__name__}: {e}"}
    try:
        from agents.model.snapshot import current_model_version, load_foreign_opponent
        from agents.observation.state_encoder import load_mappings
        model, _ver = load_foreign_opponent(
            ckpt, current_version=current_model_version(load_mappings()), device="cpu")
    except Exception as e:  # noqa: BLE001 — the arch-drift path, made actionable
        git_hash = "?"
        meta = os.path.join(run_dir, "metadata.json")
        if os.path.exists(meta):
            git_hash = json.load(open(meta)).get("git_hash", "?")
        return {"status": "needs_pinned_tree",
                "why": f"{type(e).__name__}: {e}",
                "how": [f"git worktree add /tmp/endofrun-pinned {git_hash}",
                        "copy src/main/endofrun.py + src/agents/model/critic_route_audit.py "
                        "into the pinned tree if it predates them",
                        f"PYTHONPATH=src python -m main.endofrun {run_dir} --skip elo,awareness"]}
    routes = route_audit(model.policy, obs, masks, batch=batch)
    fams = None
    try:
        fams = family_audit(model.policy, obs, masks, batch=batch)
    except Exception as e:  # noqa: BLE001 — families-off runs
        fams = {"error": f"{type(e).__name__}: {e}"}
    result = {"status": "ok", "checkpoint": ckpt, "n_states": int(len(obs)),
              "coverage": coverage, "route_arms": routes,
              "route_verdicts": route_verdicts(routes)}
    if isinstance(fams, dict) and "error" not in fams:
        result["family_arms"] = fams
        result["family_verdicts"] = family_verdicts(fams)
    else:
        result["family_arms"] = fams
    return result


def step_awareness(run_dir: str) -> dict:
    try:
        from main.prober.session import ProbeSession
        with ProbeSession(run_dir) as s:
            losses = s.awareness_scan(outcome="loss")
            everything = s.awareness_scan(outcome=None)
        if "error" in losses:
            return {"status": "skipped", "why": losses["error"]}
        return {"status": "ok",
                "loss_aggregate": losses["aggregate"],
                "all_aggregate": everything["aggregate"],
                "verdicts": awareness_verdicts(losses["aggregate"], everything["aggregate"])}
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "why": f"{type(e).__name__}: {e}"}


def step_mechanics(run_dir: str, ref_run: Optional[str]) -> dict:
    """G2 — the mechanic USAGE read (design_conditional_execution.md §6): pick_rate/mean_prob
    per conditional-execution mechanic, model-free over the eval-trace `actions` blocks. On a
    run that trained the v84/v85 cells this is the did-it-work readout against the ref
    generation's numbers; on any other run it is the standing pre-build baseline."""
    try:
        from agents.model.mechanic_usage_baseline import measure
        cur = measure(run_dir)
        out = {"status": "ok", "current": cur}
        if ref_run:
            try:
                out["reference"] = measure(ref_run)
            except Exception as e:  # noqa: BLE001
                out["reference_error"] = f"{type(e).__name__}: {e}"
        return out
    except Exception as e:  # noqa: BLE001
        return {"status": "error", "why": f"{type(e).__name__}: {e}"}


# ------------------------------------------------------------------ report

def render_markdown(report: dict) -> str:
    L = [f"# End-of-run battery — `{report['run']}`", ""]
    elo = report["steps"].get("elo", {})
    L.append("## 1. Ladder (tail-%d, matched-count convention)" % TAIL_K)
    if not elo:
        L.append("- skipped")
    elif elo.get("status") == "ok":
        cur, ref, ni = elo["current_tail"], elo["reference_tail"], elo["non_inferiority"]
        L.append(f"- current: **{cur['elo']:.1f}** ± {1.96 * cur['se']:.1f}" if cur
                 else "- current: tail under-sampled")
        if ref:
            L.append(f"- reference: {ref['elo']:.1f} ± {1.96 * ref['se']:.1f}")
        L.append(f"- **{ni['verdict']}**"
                 + (f" (Δ {ni['delta']}, CI {ni['ci95']}; rule: {ni['rule']})"
                    if "delta" in ni else f" — {ni.get('why', '')}"))
    else:
        L.append(f"- {elo.get('status')}: {elo.get('why')}")
    aud = report["steps"].get("audits", {})
    L.append("\n## 2. Critic routes + edge families")
    if not aud:
        L.append("- skipped")
    elif aud.get("status") == "ok":
        for r, v in aud["route_verdicts"].get("per_route", {}).items():
            L.append(f"- `{r}`: **{v['verdict']}**"
                     + (f" (|dV| {v['dv_frac_of_all_off']:.0%} of all_off, "
                        f"flips {v['flip_rate']:.2%})" if "dv_frac_of_all_off" in v else ""))
        for n in aud["route_verdicts"].get("notes", []):
            L.append(f"- ⚠️ {n}")
        for f, v in (aud.get("family_verdicts") or {}).items():
            L.append(f"- family `{f}`: **{v['verdict']}**"
                     + (f" (|dV| {v['dv_mean']} vs median {v['median_live_family_dv']})"
                        if "dv_mean" in v else ""))
    else:
        L.append(f"- {aud.get('status')}: {aud.get('why')}")
        for h in aud.get("how", []):
            L.append(f"    - `{h}`")
    aw = report["steps"].get("awareness", {})
    L.append("\n## 3. Awareness + coverage (vs gen-10 baselines)")
    if not aw:
        L.append("- skipped")
    elif aw.get("status") == "ok":
        for k, v in aw["verdicts"].items():
            cur = v.get("current")
            L.append(f"- {k}: **{v['verdict']}**"
                     + (f" ({cur} vs {v['gen10_baseline']})" if cur is not None else ""))
    else:
        L.append(f"- {aw.get('status')}: {aw.get('why')}")
    mech = report["steps"].get("mechanics", {})
    if mech:
        L.append("\n## 4. Mechanic usage (G2 — conditional execution)")
        if mech.get("status") == "ok":
            cur = mech["current"]["mechanics"]
            ref = (mech.get("reference") or {}).get("mechanics", {})
            for name, st in cur.items():
                if st["available"] == 0:
                    continue
                line = (f"- {name}: pick {st['pick_rate']:.1%} of {st['available']} "
                        f"(mean prob {st['mean_prob']:.1%})")
                r = ref.get(name)
                if r and r.get("pick_rate") is not None:
                    line += f" — ref {r['pick_rate']:.1%}"
                L.append(line)
        else:
            L.append(f"- {mech.get('status')}: {mech.get('why', '')}")
    L.append("\n*Verdicts are decision-support against the pre-registered runbook rules; "
             "the runbooks remain the registration of record.*")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir")
    ap.add_argument("--ref", default=None, help="the previous generation's run dir (ladder Δ)")
    ap.add_argument("--anchors", default="data/gen3_bot_elo_anchors.json")
    ap.add_argument("--max-states", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--skip", default="",
                    help="comma list of steps: elo,audits,awareness,mechanics")
    ap.add_argument("--out", default="designs/research_state/measurements")
    a = ap.parse_args(argv)
    skip = {s for s in a.skip.split(",") if s}

    report = {"run": os.path.basename(os.path.normpath(a.run_dir)),
              "run_dir": os.path.abspath(a.run_dir),
              "ref": a.ref, "steps": {}}
    if "elo" not in skip:
        report["steps"]["elo"] = step_elo(a.run_dir, a.ref, a.anchors)
    if "audits" not in skip:
        report["steps"]["audits"] = step_audits(a.run_dir, a.max_states, a.batch)
    if "awareness" not in skip:
        report["steps"]["awareness"] = step_awareness(a.run_dir)
    if "mechanics" not in skip:
        report["steps"]["mechanics"] = step_mechanics(a.run_dir, a.ref)

    os.makedirs(a.out, exist_ok=True)
    jpath = os.path.join(a.out, f"{report['run']}_endofrun.json")
    with open(jpath, "w") as f:
        json.dump(report, f, indent=1, default=float)
    md = render_markdown(report)
    mpath = os.path.join(a.out, f"{report['run']}_endofrun.md")
    with open(mpath, "w") as f:
        f.write(md)
    print(md)
    print(f"wrote {jpath}\nwrote {mpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
