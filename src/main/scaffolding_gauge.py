"""main.scaffolding_gauge — the SCAFFOLDING GAUGE, offline, over a run's eval traces.

How far apart are the two value readouts, and is the gap CLOSING? The critic ``V`` estimates the
**shaped** return; the win-prob head estimates the **game**. Their divergence is the reward
scaffolding still doing work, and its trajectory across checkpoints is the registered signal for
when shaping coefficients can begin annealing toward the pure game (ledger 2026-08-29).

    python -m main.scaffolding_gauge models/<run>
    python -m main.scaffolding_gauge models/<run> --plot            # + a PNG beside the JSON
    python -m main.scaffolding_gauge models/<run> --constancy       # only the db9bb5c sanity row
    python -m main.scaffolding_gauge models/<run> --opponent aggressive --boot 1000
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)

Model-FREE and read-only: it reads ``values`` and ``win_probs`` straight out of the traces the run
already wrote (``eval_traces/step_*/<opponent>/<outcome>_*_states.npz``), so it loads no checkpoint,
plays no battle, and is immune to architecture drift. A run whose checkpoints can no longer be
loaded still yields a full gauge curve.

═══ 🚨 UNITS HONESTY — the headline, restated at the CLI ═══════════════════════════════════════

Recorded ``V`` is a **PopArt-normalized shaped return**. There is no general unit conversion to a
win probability, so this tool refuses to invent one and ships TWO gauges instead:

* **RANK gauge** (Spearman ρ between V and P(win) per step-slice) — unit-free and ALWAYS valid.
  Claims ordering agreement, and nothing about magnitude.
* **AFFINE gauge** (fit a per-checkpoint affine V→outcome map on realized outcomes, then compare
  the V-implied outcome with the head) — in probability units, but the map is a per-checkpoint FIT,
  not a conversion, and it does not transport to another checkpoint or run. Part of its residual is
  the affine family being a worse outcome predictor rather than the heads disagreeing; that part is
  broken out as ``readout_penalty`` and printed beside the number.

Both are labelled again in the JSON's ``units`` block, so a reader six months out cannot get the
number without the caveat. Full statement: `agents.training.scaffolding`'s module docstring.

**Every CI is a CLUSTER bootstrap over BATTLES**, because outcome labels are per-battle and
broadcast to every state. An i.i.d. interval over states would be fabricated tightness.

**The step-to-step curve is not a controlled comparison.** Each point is whatever the eval quota
sampled at that checkpoint — a different opponent mix, a different win rate, a different number of
battles. Read the TREND with the per-step ``n_battles`` and ``base_rate`` columns in view, and take
a verdict only from an arm-vs-control read at matched step.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

#: Shipped inside the artifact: what each gauge may and may not be quoted for.
UNITS: Dict[str, Dict[str, str]] = {
    "recorded_V": {
        "what": "The critic's estimate of the SHAPED, gamma-discounted return, in the run's own "
                "PopArt-normalized units, as recorded at play time in states.npz['values'].",
        "cannot": "It is not a win probability and cannot be converted into one. Its scale moves "
                  "over training (PopArt sigma), so even the RAW value is only comparable within "
                  "a run, and only cautiously across checkpoints of one run.",
    },
    "rank_gauge": {
        "what": "(1 - Spearman rho(V, P(win))) / 2 over the step-slice's states. 0 = identical "
                "ordering, 0.5 = independent, 1 = inverted. Unit-free; invariant to PopArt and to "
                "any monotone reparameterization of either axis.",
        "cannot": "Says nothing about magnitude or calibration. Goes AMBIGUOUS as V_shaped "
                  "flattens (the PBRS constancy endpoint): a falling rho late in a shaped run "
                  "cannot be told from V running out of variance to rank with. Read it beside the "
                  "constancy row's v_std / within_frac.",
    },
    "affine_gauge": {
        "what": "RMS gap, in probability units, between the win-prob head and clip(a*V+b, 0, 1) "
                "where (a, b) is the least-squares fit of V to the REALIZED per-battle outcome on "
                "this same slice.",
        "cannot": "The map is a per-checkpoint FIT, not a unit conversion; it does not transport "
                  "to another checkpoint or run. The affine family cannot express the true "
                  "V->outcome map, so part of every residual is a worse readout rather than a "
                  "divergence — quote `readout_penalty` (Brier(readout) - Brier(head)) with it, "
                  "and treat a large rms with a large readout_penalty as a readout finding.",
    },
    "constancy": {
        "what": "Spread of V across the slice's states (ledger db9bb5c): under PBRS with a good "
                "frozen potential, all evaluative content migrates into the reward stream and "
                "V_shaped is driven toward a CONSTANT. v_std falling toward 0 in a frozen-phi arm "
                "CONFIRMS the theory; it is not a critic failure.",
        "cannot": "v_std rides PopArt and is not comparable across runs — use `dispersion` "
                  "(v_std / E|V|) for that, and prefer an arm-vs-control read at matched step. A "
                  "low v_std with within_frac near 0 is the FAILURE look-alike (V has become a "
                  "per-battle matchup lookup, not a flattened potential).",
    },
    "curve": {
        "what": "One row per checkpoint step, in step order, plus an OLS slope over the rows.",
        "cannot": "Not a controlled comparison: each point carries whatever the eval quota "
                  "sampled at that step (opponent mix, win rate, battle count all move). The "
                  "slope is descriptive; a verdict needs arm-vs-control at matched step.",
    },
}


# --------------------------------------------------------------------------- trace reading

def _npz_for(summary_path: str) -> str:
    return summary_path.replace("_summary.json", "_states.npz")


def collect_slices(
    run_or_traces: str,
    *,
    opponent: Optional[str] = None,
    max_battles_per_step: Optional[int] = None,
    seed: int = 0,
    say=lambda _m: None,
) -> "tuple[Dict[int, Dict[str, np.ndarray]], Dict[str, Any]]":
    """Read every trace into per-step parallel arrays ``{step: {values, win_probs, outcomes,
    battles, opponents}}`` plus a coverage record.

    Rows are kept only where ``has_state`` is set AND ``win_probs`` is finite — a run trained with
    ``--win-prob-mode none`` records an all-NaN column, and this is exactly the case that must
    produce "no data" rather than a curve of zeros.

    ``max_battles_per_step`` subsamples whole BATTLES (seeded), never rows: the cluster structure
    the bootstrap depends on has to survive the cap.
    """
    from main.prober.discovery import build_trace_tree

    tree = build_trace_tree(run_or_traces)
    if tree.is_empty:
        raise SystemExit(
            f"[scaffolding_gauge] no eval traces under {run_or_traces!r} — this tool reads "
            "`eval_traces/step_*/<opponent>/<outcome>_*_states.npz`, which a run writes only when "
            "battle recording is on. Point it at a run directory that has one.")

    rng = np.random.default_rng(seed)
    out: Dict[int, Dict[str, np.ndarray]] = {}
    coverage: Dict[str, Any] = {"n_traces_seen": 0, "n_traces_read": 0, "n_traces_no_npz": 0,
                                "n_traces_no_winprob": 0, "per_step": {}, "per_opponent": {}}
    for sg in tree.steps:
        rows_v: List[np.ndarray] = []
        rows_p: List[np.ndarray] = []
        rows_y: List[np.ndarray] = []
        rows_b: List[np.ndarray] = []
        rows_o: List[np.ndarray] = []
        battles = [b for og in sg.opponents for b in og.battles
                   if opponent is None or og.name == opponent]
        coverage["n_traces_seen"] += len(battles)
        if max_battles_per_step is not None and len(battles) > max_battles_per_step:
            keep = rng.choice(len(battles), size=max_battles_per_step, replace=False)
            battles = [battles[i] for i in sorted(keep)]
        for bt in battles:
            npz_path = bt.npz_path or _npz_for(bt.summary_path)
            if not os.path.exists(npz_path):
                coverage["n_traces_no_npz"] += 1
                continue
            try:
                with np.load(npz_path) as d:
                    if "values" not in d or "win_probs" not in d:
                        coverage["n_traces_no_winprob"] += 1
                        continue
                    v = np.asarray(d["values"], dtype=np.float64)
                    p = np.asarray(d["win_probs"], dtype=np.float64)
                    has = (np.asarray(d["has_state"]).astype(bool) if "has_state" in d
                           else np.ones(v.shape, dtype=bool))
            except (OSError, ValueError, KeyError) as exc:
                say(f"skipping unreadable {os.path.basename(npz_path)}: {exc}")
                coverage["n_traces_no_npz"] += 1
                continue
            keep_rows = has & np.isfinite(v) & np.isfinite(p)
            if not keep_rows.any():
                coverage["n_traces_no_winprob"] += 1
                continue
            n = int(keep_rows.sum())
            y = 1.0 if bt.outcome == "win" else 0.0
            rows_v.append(v[keep_rows])
            rows_p.append(p[keep_rows])
            rows_y.append(np.full(n, y))
            rows_b.append(np.full(n, bt.summary_path, dtype=object))
            rows_o.append(np.full(n, bt.opponent, dtype=object))
            coverage["n_traces_read"] += 1
            coverage["per_opponent"][bt.opponent] = coverage["per_opponent"].get(bt.opponent, 0) + 1
        if not rows_v:
            continue
        out[sg.step] = {
            "values": np.concatenate(rows_v), "win_probs": np.concatenate(rows_p),
            "outcomes": np.concatenate(rows_y), "battles": np.concatenate(rows_b),
            "opponents": np.concatenate(rows_o),
        }
        coverage["per_step"][str(sg.step)] = {"n_battles": len(rows_v),
                                              "n_states": int(out[sg.step]["values"].size)}
    if not out:
        raise SystemExit(
            "[scaffolding_gauge] the traces carry no usable (V, P(win)) pairs. "
            f"{coverage['n_traces_seen']} traces seen, {coverage['n_traces_no_npz']} without a "
            f"readable states.npz, {coverage['n_traces_no_winprob']} with no finite win_probs "
            "column. An all-NaN win_probs column means the run trained with `--win-prob-mode "
            "none`: there is no win-prob head to compare the critic against, so there is no gauge "
            "to read. This is a REFUSAL, not a zero.")
    return out, coverage


# --------------------------------------------------------------------------- the fold

def _ols_slope(xs: Sequence[float], ys: Sequence[float]) -> Dict[str, float]:
    """Descriptive OLS slope of a gauge against step. NaN when fewer than 3 finite points."""
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return {"slope_per_Mstep": float("nan"), "n_points": float(int(ok.sum()))}
    x, y = x[ok], y[ok]
    xm, ym = x.mean(), y.mean()
    den = float(np.sum((x - xm) ** 2))
    if den <= 0.0:
        return {"slope_per_Mstep": float("nan"), "n_points": float(x.size)}
    slope = float(np.sum((x - xm) * (y - ym)) / den)
    return {"slope_per_Mstep": slope * 1e6, "n_points": float(x.size)}


def build_report(
    slices: Dict[int, Dict[str, np.ndarray]],
    coverage: Dict[str, Any],
    *,
    run_dir: str,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    from agents.training.scaffolding import gauge_slice

    rows: List[Dict[str, Any]] = []
    for step in sorted(slices):
        s = slices[step]
        g = gauge_slice(s["values"], s["win_probs"], s["outcomes"], s["battles"],
                        n_boot=n_boot, seed=seed)
        rows.append({
            "step": int(step),
            "n_states": int(s["values"].size),
            "n_battles": int(len(set(s["battles"].tolist()))),
            "win_rate": float(np.mean(s["outcomes"])),
            **{f"rank_{k}": v for k, v in g["rank"].items()},
            **{f"affine_{k}": v for k, v in g["affine"].items()},
            **{f"const_{k}": v for k, v in g["constancy"].items()},
        })
    steps = [r["step"] for r in rows]
    trend = {
        "rank_gauge": _ols_slope(steps, [r["rank_gauge"] for r in rows]),
        "affine_rms": _ols_slope(steps, [r["affine_rms"] for r in rows]),
        "v_std": _ols_slope(steps, [r["const_v_std"] for r in rows]),
    }
    from utils.git import get_git_hash
    return {
        "tool": "scaffolding_gauge",
        "tool_version": 1,
        "meta": {
            "run_dir": os.path.abspath(run_dir),
            "run_name": os.path.basename(os.path.abspath(run_dir).rstrip("/")),
            "n_steps": len(rows),
            "n_boot": int(n_boot),
            "seed": int(seed),
            "coverage": coverage,
            "tree_git_hash": get_git_hash(),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "curve": rows,
        "trend": trend,
        "units": UNITS,
    }


# --------------------------------------------------------------------------- rendering

def _f(v: Any, spec: str = "8.3f") -> str:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    if not np.isfinite(f):
        width = int(spec.split(".")[0].lstrip("<>") or 0)
        return f"{'—':>{width}}"
    return format(f, spec)


def render(report: Dict[str, Any], *, constancy_only: bool = False) -> str:
    lines: List[str] = []
    m = report["meta"]
    lines.append("=" * 100)
    lines.append(f"SCAFFOLDING GAUGE v{report['tool_version']}   {m['run_name']}")
    lines.append(f"  {m['n_steps']} checkpoint steps   "
                 f"{m['coverage']['n_traces_read']} traces read   "
                 f"bootstrap {m['n_boot']} x cluster-over-battles   seed {m['seed']}")
    lines.append("=" * 100)

    if constancy_only:
        lines.append("\nCONSTANCY SANITY ROW (ledger db9bb5c: under PBRS with a good frozen "
                     "potential, V_shaped -> CONSTANT)")
        lines.append(f"  {'step':>12}{'n':>8}{'battles':>9}{'v_mean':>10}{'v_std':>10}"
                     f"{'v_iqr':>10}{'disp':>9}{'within%':>9}")
        for r in report["curve"]:
            lines.append(
                f"  {r['step']:>12,}{r['n_states']:>8}{r['n_battles']:>9}"
                f"{_f(r['const_v_mean'], '10.3f')}{_f(r['const_v_std'], '10.3f')}"
                f"{_f(r['const_v_iqr'], '10.3f')}{_f(r['const_dispersion'], '9.3f')}"
                f"{_f(100 * r['const_within_frac'] if np.isfinite(r['const_within_frac']) else float('nan'), '9.1f')}")
        t = report["trend"]["v_std"]
        lines.append(f"\n  v_std trend: {_f(t['slope_per_Mstep'], '.5f')} per 1M steps over "
                     f"{int(t['n_points'])} points")
        lines.append("  READ: v_std falling toward 0 in a FROZEN-phi arm CONFIRMS the theory (all")
        lines.append("  evaluative content migrated into the reward stream). v_std low with within%")
        lines.append("  near 0 is the look-alike FAILURE: V became a per-battle matchup lookup.")
        lines.append("  UNITS: v_std rides PopArt — comparable within a run, `disp` across.")
        return "\n".join(lines)

    lines.append("\n(1) RANK GAUGE — (1 - Spearman rho(V, P(win))) / 2.  UNIT-FREE; ordering only.")
    lines.append(f"  {'step':>12}{'n':>8}{'battles':>9}{'winrate':>9}{'rho':>9}{'gauge':>9}"
                 f"{'[ 95% cluster CI ]':>24}")
    for r in report["curve"]:
        ci = f"[{_f(r['rank_ci_lo'], '7.3f')},{_f(r['rank_ci_hi'], '7.3f')} ]"
        lines.append(f"  {r['step']:>12,}{r['n_states']:>8}{r['n_battles']:>9}"
                     f"{_f(r['win_rate'], '9.3f')}{_f(r['rank_rho'], '9.3f')}"
                     f"{_f(r['rank_gauge'], '9.3f')}{ci:>24}")
    t = report["trend"]["rank_gauge"]
    lines.append(f"  trend {_f(t['slope_per_Mstep'], '.5f')} / 1M steps  "
                 f"({int(t['n_points'])} points)   NEGATIVE = the readouts are converging.")

    lines.append("\n(2) CALIBRATED-AFFINE GAUGE — |clip(a*V+b) - P(win)| in PROBABILITY units.")
    lines.append(f"  {'step':>12}{'a':>10}{'b':>8}{'rms':>8}{'bias':>8}"
                 f"{'[ 95% cluster CI ]':>22}{'Brier_head':>12}{'Brier_aff':>11}{'penalty':>9}")
    for r in report["curve"]:
        ci = f"[{_f(r['affine_ci_lo'], '6.3f')},{_f(r['affine_ci_hi'], '6.3f')} ]"
        lines.append(f"  {r['step']:>12,}{_f(r['affine_a'], '10.4f')}{_f(r['affine_b'], '8.3f')}"
                     f"{_f(r['affine_rms'], '8.3f')}{_f(r['affine_bias'], '8.3f')}{ci:>22}"
                     f"{_f(r['affine_brier_head'], '12.4f')}{_f(r['affine_brier_v_affine'], '11.4f')}"
                     f"{_f(r['affine_readout_penalty'], '9.4f')}")
    t = report["trend"]["affine_rms"]
    lines.append(f"  trend {_f(t['slope_per_Mstep'], '.5f')} / 1M steps  "
                 f"({int(t['n_points'])} points)")
    lines.append("  ⚠️  `penalty` = Brier(affine readout) - Brier(head). Large penalty ⇒ most of")
    lines.append("      `rms` is the affine FAMILY being a worse outcome predictor, NOT the two")
    lines.append("      heads disagreeing. That is a readout finding, not a divergence finding.")

    lines.append("\n(3) CONSTANCY SANITY ROW (ledger db9bb5c) — does V_shaped flatten?")
    lines.append(f"  {'step':>12}{'v_mean':>10}{'v_std':>10}{'v_iqr':>10}{'disp':>9}{'within%':>9}")
    for r in report["curve"]:
        wf = r["const_within_frac"]
        lines.append(f"  {r['step']:>12,}{_f(r['const_v_mean'], '10.3f')}"
                     f"{_f(r['const_v_std'], '10.3f')}{_f(r['const_v_iqr'], '10.3f')}"
                     f"{_f(r['const_dispersion'], '9.3f')}"
                     f"{_f(100 * wf if np.isfinite(wf) else float('nan'), '9.1f')}")

    lines.append("\n" + "-" * 100)
    lines.append("UNITS: recorded V is a PopArt-normalized SHAPED return. No general unit")
    lines.append("conversion to a win probability exists — hence two gauges. (1) claims ORDERING")
    lines.append("only. (2) is in probability units but its map is a per-checkpoint FIT that does")
    lines.append("not transport, and part of its residual is the affine family, not divergence.")
    lines.append("Every CI is a CLUSTER bootstrap over BATTLES (labels are per-battle, broadcast).")
    lines.append("The step curve is NOT a controlled comparison — the eval quota moves under it.")
    return "\n".join(lines)


def write_plot(report: Dict[str, Any], path: str) -> "str | None":
    """Three stacked panels — rank gauge, affine rms, v_std — against step. Returns the path, or
    None (with a printed reason) when matplotlib is unavailable: a missing optional renderer must
    never fail the measurement."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:                                   # pragma: no cover — env-dependent
        print(f"[scaffolding_gauge] --plot skipped: matplotlib unavailable ({exc})")
        return None
    rows = report["curve"]
    steps = [r["step"] / 1e6 for r in rows]
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    panels = [
        ("rank_gauge", "rank_ci_lo", "rank_ci_hi", "rank gauge (1-rho)/2 — unit-free", "tab:blue"),
        ("affine_rms", "affine_ci_lo", "affine_ci_hi", "affine rms — probability units", "tab:red"),
        ("const_v_std", None, None, "v_std — PopArt units (constancy row)", "tab:green"),
    ]
    for ax, (key, lo, hi, title, color) in zip(axes, panels):
        ys = [r[key] for r in rows]
        ax.plot(steps, ys, "o-", color=color)
        if lo and hi:
            ax.fill_between(steps, [r[lo] for r in rows], [r[hi] for r in rows],
                            color=color, alpha=0.18, label="95% cluster CI")
            ax.legend(loc="best", fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("step (millions)")
    fig.suptitle(f"scaffolding gauge — {report['meta']['run_name']}\n"
                 "V is PopArt-normalized SHAPED return; panel 1 claims ORDERING only, "
                 "panel 2's map is a per-checkpoint fit", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- entry point

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="python -m main.scaffolding_gauge",
        description="Offline scaffolding gauge: V-implied outcome vs the win-prob head, per "
                    "checkpoint, over a run's eval traces. Model-free.")
    ap.add_argument("run", help="a run directory (or an eval_traces tree)")
    ap.add_argument("--out", default=None,
                    help="JSON path (default: <run>/scaffolding_gauge.json)")
    ap.add_argument("--plot", nargs="?", const="", default=None,
                    help="write a PNG (default: beside the JSON)")
    ap.add_argument("--opponent", default=None, help="restrict to one opponent directory")
    ap.add_argument("--max-battles-per-step", type=int, default=None,
                    help="seeded subsample of whole BATTLES per step (clusters stay intact)")
    ap.add_argument("--boot", type=int, default=400, help="cluster-bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--constancy", action="store_true",
                    help="print ONLY the db9bb5c constancy sanity row (the JSON is unchanged)")
    ap.add_argument("--quiet", action="store_true")
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    say = (lambda _m: None) if args.quiet else (lambda m: print(f"[scaffolding_gauge] {m}",
                                                               flush=True))
    t0 = time.time()
    run_dir = args.run
    slices, coverage = collect_slices(
        run_dir, opponent=args.opponent, max_battles_per_step=args.max_battles_per_step,
        seed=args.seed, say=say)
    say(f"{len(slices)} step slices, {coverage['n_traces_read']} traces read")
    report = build_report(slices, coverage, run_dir=run_dir, n_boot=args.boot, seed=args.seed)
    report["meta"]["runtime_sec"] = round(time.time() - t0, 1)

    out = args.out or os.path.join(run_dir, "scaffolding_gauge.json")
    try:
        with open(out, "w") as fh:
            json.dump(report, fh, indent=1)
        say(f"wrote {out}")
    except OSError as exc:
        say(f"could not write {out}: {exc}")
    if args.plot is not None:
        png = args.plot or os.path.splitext(out)[0] + ".png"
        written = write_plot(report, png)
        if written:
            say(f"wrote {written}")
    print(render(report, constancy_only=args.constancy))
    return 0


if __name__ == "__main__":                                       # pragma: no cover
    sys.exit(main())
