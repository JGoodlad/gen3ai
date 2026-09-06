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
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from agents.training import trace_selection

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




# --------------------------------------------------------- the RELIABILITY block (opt-in)

#: Opponent-directory name prefixes that group into an opponent CLASS. The two classes answer
#: different questions: the scripted BOTS are a fixed, stationary population the head was never
#: trained against, while the pool SENTINELS are recent selves — the ~90% self-play mixture the
#: BCE labels actually come from. `win_prob_decomposition.md` axis 3 (the ECOLOGY split) measured
#: the head's mean bias FLIPPING SIGN between them, so a pooled calibration number is a
#: population-confounded average of two different forecasters and must never be quoted alone.
_SENTINEL_PREFIX = "sentinel"


def opponent_class(name: str) -> str:
    """`sentinel_3` -> `pool`, anything else -> `bot`. The one place the split is defined."""
    return "pool" if str(name).startswith(_SENTINEL_PREFIX) else "bot"


def build_reliability(
    slices: Dict[int, Dict[str, np.ndarray]],
    *,
    bins: int = 10,
    n_boot: int = 400,
    seed: int = 0,
    reweight: Optional[Dict[int, Dict[str, float]]] = None,
) -> List[Dict[str, Any]]:
    """Per-step reliability blocks, each stratified into ``all`` / ``bot`` / ``pool`` / per-opponent.

    Every stratum carries a cluster-bootstrap CI over BATTLES for ``brier`` and for ``skill`` —
    the labels are per-battle and broadcast, so an i.i.d. interval over states would be roughly
    sqrt(states-per-battle) too tight (the recorded Simpson trap in this tree).

    ``reweight`` (``{step: {opponent: true win rate}}``, from :func:`true_win_rates`) turns on the
    SELECTION correction: rows are importance-weighted so each opponent's win/loss mix matches the
    eval cycle's rather than the capture quota's. The weights are constant within a battle, so the
    bootstrap's clusters are unchanged and the CIs stay valid.
    """
    from agents.training.scaffolding import cluster_bootstrap_ci, reliability_table

    out: List[Dict[str, Any]] = []
    for step in sorted(slices):
        s = slices[step]
        p = np.asarray(s["win_probs"], dtype=np.float64)
        y = np.asarray(s["outcomes"], dtype=np.float64)
        b = np.asarray(s["battles"])
        opp = np.asarray(s["opponents"])
        cls = np.array([opponent_class(o) for o in opp.tolist()])

        w: Optional[np.ndarray] = None
        wmeta: Optional[Dict[str, Any]] = None
        if reweight is not None:
            w, wmeta = selection_weights(y, b, opp, reweight.get(int(step), {}))

        strata: List[Tuple[str, str, np.ndarray]] = [("all", "all", np.ones(p.size, dtype=bool))]
        for c in ("bot", "pool"):
            sel = cls == c
            if sel.any():
                strata.append(("class", c, sel))
        for o in sorted(set(opp.tolist())):
            strata.append(("opponent", str(o), opp == o))

        rows: List[Dict[str, Any]] = []
        for i, (kind, name, sel) in enumerate(strata):
            pi, yi, bi = p[sel], y[sel], b[sel]
            wi = None if w is None else w[sel]

            def _stat(idx, key, _p=pi, _y=yi, _w=wi):
                return reliability_table(_p[idx], _y[idx], bins=bins,
                                         weights=None if _w is None else _w[idx])[key]

            r = reliability_table(pi, yi, bins=bins, weights=wi)
            r["brier_ci_lo"], r["brier_ci_hi"] = cluster_bootstrap_ci(
                lambda idx: _stat(idx, "brier"), bi, n_boot=n_boot, seed=seed + 100 + i)
            r["skill_ci_lo"], r["skill_ci_hi"] = cluster_bootstrap_ci(
                lambda idx: _stat(idx, "skill"), bi, n_boot=n_boot, seed=seed + 200 + i)
            r["n_battles"] = int(len(set(bi.tolist())))
            rows.append({"kind": kind, "name": name, **r})
        blk: Dict[str, Any] = {"step": int(step), "bins": int(bins), "strata": rows,
                               "reweighted": reweight is not None}
        if wmeta is not None:
            blk["selection"] = wmeta
        out.append(blk)
    return out


# ----------------------------------------------- the SELECTION reweighting (`--reliability-reweight`)

class SelectionWeightError(RuntimeError):
    """The true per-opponent win rates could not be resolved. A REFUSAL, never a silent fall-back
    to unweighted — an unweighted table looks identical and is answering a different question."""


def _sentinel_names_in_manifest_order(step_dir: str) -> List[str]:
    """The `sentinel_k` directory names, in the order `eval_manifest.json` lists them.

    The manifest's `opponents` list and `eval_results.jsonl`'s `sentinels` list are written by the
    same eval cycle from the same pool ordering, which is what licenses joining them BY POSITION.
    That join is an inference, so it is made in ONE place, it is asserted rather than assumed (the
    counts must agree), and it refuses instead of guessing.
    """
    with open(os.path.join(step_dir, "eval_manifest.json")) as fh:
        opponents = json.load(fh).get("opponents") or []
    return [str(o) for o in opponents if opponent_class(str(o)) == "pool"]


def manifest_win_rates_by_step(run_dir: str) -> Dict[int, Dict[str, float]]:
    """``{step: {opponent: win rate}}`` read from each cycle's OWN ``eval_manifest.json``.

    The PREFERRED source (`gen3_trace_selection_manifest_v1`), because the manifest states the
    cycle's per-opponent played/won counts in the same file as the traces they produced — no join,
    and in particular **no positional sentinel join** of the kind
    :func:`_sentinel_names_in_manifest_order` has to make against ``eval_results.jsonl``.

    A tree that records no selection returns ``{}``, which is what makes the fall-back below
    byte-identical on every legacy run.
    """
    out: Dict[int, Dict[str, float]] = {}
    root = os.path.join(run_dir, "eval_traces")
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        m = re.match(r"^step_(\d+)$", name)
        if not m:
            continue
        mpath = os.path.join(root, name, "eval_manifest.json")
        if not os.path.exists(mpath):
            continue
        try:
            with open(mpath) as fh:
                manifest = json.load(fh)
        except (OSError, ValueError):
            continue
        rates = trace_selection.manifest_win_rates(manifest)
        if rates:
            out[int(m.group(1))] = rates
    return out


def true_win_rates(run_dir: str) -> Dict[int, Dict[str, float]]:
    """``{step: {opponent: true win rate}}`` — the population the reweighting targets.

    The FULL eval cycle the traces were sampled out of, whose per-opponent rates the run already
    recorded. Not an outside estimate.

    TWO SOURCES, in preference order. The per-cycle **`eval_manifest.json`** selection block wins
    where it exists: it is written by the same cycle into the same directory as the traces, so it
    needs no cross-file join. Everything it does not cover falls back to **`eval_results.jsonl`**
    with the existing positional-sentinel behaviour, unchanged. A run with neither REFUSES.

    On a legacy tree the manifest half is empty and this returns exactly what the jsonl half always
    returned, value for value — the numbers this tool prints there do not move.
    """
    from_manifest = manifest_win_rates_by_step(run_dir)
    path = os.path.join(run_dir, "eval_results.jsonl")
    if not os.path.exists(path):
        if from_manifest:
            return from_manifest
        raise SelectionWeightError(
            f"--reliability-reweight needs the run's recorded per-opponent win rates and "
            f"{path!r} does not exist. Without it the true eval population is unknown, and an "
            "UNWEIGHTED table over the capture quota is not a calibration of this head — it is a "
            "calibration on a loss-enriched subsample. Point the tool at the run directory (not a "
            "bare eval_traces tree), or drop --reliability-reweight and read the raw table knowing "
            "what it is.")
    out: Dict[int, Dict[str, float]] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            step = row.get("step")
            if step is None:
                continue
            rates = {str(k): float(v) for k, v in (row.get("bots") or {}).items()}
            sent = list(row.get("sentinels") or [])
            step_dir = os.path.join(run_dir, "eval_traces", f"step_{int(step)}")
            names = (_sentinel_names_in_manifest_order(step_dir)
                     if os.path.exists(os.path.join(step_dir, "eval_manifest.json")) else [])
            if names and len(names) == len(sent):
                for name, s in zip(names, sent):
                    if s.get("win_rate") is not None:
                        rates[name] = float(s["win_rate"])
            out[int(step)] = rates
    # The manifest is PREFERRED per (step, opponent): it is the cycle's own record, in the same
    # directory as the traces. Where it says nothing, the jsonl's value survives untouched — which
    # on a legacy tree is every value, so nothing this tool prints there moves.
    for step, rates in from_manifest.items():
        out.setdefault(step, {}).update(rates)
    return out


def win_rate_sources(run_dir: str) -> Dict[int, str]:
    """``{step: source}`` — WHERE each step's true rates came from, for the report's provenance.

    ``"eval_manifest"`` (the cycle's own selection block), ``"eval_results.jsonl"`` (the legacy
    join), or ``"mixed"``. Reported rather than assumed, because the two sources can disagree —
    the jsonl route infers sentinel names BY POSITION and the manifest route does not.
    """
    from_manifest = manifest_win_rates_by_step(run_dir)
    has_jsonl = os.path.exists(os.path.join(run_dir, "eval_results.jsonl"))
    steps = set(from_manifest)
    if has_jsonl:
        try:
            steps |= set(true_win_rates(run_dir))
        except SelectionWeightError:
            pass
    out: Dict[int, str] = {}
    for step in sorted(steps):
        in_m = step in from_manifest
        out[step] = ("mixed" if (in_m and has_jsonl)
                     else "eval_manifest" if in_m else "eval_results.jsonl")
    return out


def build_trace_selection_block(run_dir: str) -> Dict[str, Any]:
    """What the traces are a SAMPLE OF, read off the trace tree itself.

    Per step: whether that cycle recorded a selection, the rule in words, the per-opponent capture
    rates, and which source the reweighting would take its true rates from. A step that records
    nothing is reported as ``known: false`` with the standing UNKNOWN label — never omitted and
    never defaulted to uniform, because "we did not record the quota" and "the quota was uniform"
    are different facts and only one of them is true here.
    """
    sources = win_rate_sources(run_dir)
    root = os.path.join(run_dir, "eval_traces")
    steps: Dict[str, Any] = {}
    n_known = 0
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            m = re.match(r"^step_(\d+)$", name)
            if not m:
                continue
            step = int(m.group(1))
            manifest = None
            mpath = os.path.join(root, name, "eval_manifest.json")
            if os.path.exists(mpath):
                try:
                    with open(mpath) as fh:
                        manifest = json.load(fh)
                except (OSError, ValueError):
                    manifest = None
            rates = trace_selection.capture_rates(manifest)
            n_known += 1 if rates else 0
            steps[str(step)] = {
                "known": bool(rates),
                "rule": trace_selection.selection_rule(manifest),
                "label": trace_selection.describe_selection(manifest),
                "per_opponent": rates,
                "true_win_rate_source": sources.get(step),
            }
    return {
        "schema": trace_selection.SELECTION_SCHEMA,
        "n_steps": len(steps),
        "n_steps_with_selection": n_known,
        "any_unknown": n_known < len(steps),
        "unknown_label": trace_selection.UNKNOWN_LABEL,
        "steps": steps,
    }


def selection_weights(
    outcomes: np.ndarray,
    battles: np.ndarray,
    opponents: np.ndarray,
    true_rates: Dict[str, float],
) -> "tuple[np.ndarray, Dict[str, Any]]":
    """Per-ROW importance weights that restore the eval population's win/loss mix per opponent.

    Within one opponent, the capture quota kept ``m`` wins of ``M`` battles while the cycle itself
    won at rate ``q``. The weight on a captured WIN is ``q / (m/M)`` and on a captured LOSS
    ``(1 − q) / (1 − m/M)``; both are constant within a battle, so the battle clustering the
    bootstrap depends on is untouched. Weights are normalized to mean 1 for readability only —
    every statistic here is a ratio of weighted sums, so the scale cancels.

    An opponent with **no recorded true rate**, or with no captured battles on one side of the
    outcome (``m == 0`` or ``m == M``), cannot be corrected: its rows are RETURNED WITH WEIGHT 0
    and named in the report, because silently leaving them at weight 1 would mix a corrected
    population with an uncorrected one and label the result corrected.
    """
    y = np.asarray(outcomes, dtype=np.float64).ravel()
    opp = np.asarray(opponents).ravel()
    w = np.zeros(y.size, dtype=np.float64)
    dropped: Dict[str, str] = {}
    per_opp: Dict[str, Dict[str, float]] = {}
    for name in sorted(set(opp.tolist())):
        sel = opp == name
        q = true_rates.get(str(name))
        # the captured mix is a property of BATTLES, not of decisions.
        seen: Dict[str, float] = {}
        for b, yy in zip(np.asarray(battles).ravel()[sel].tolist(), y[sel].tolist()):
            seen[b] = yy
        m, M = float(sum(seen.values())), float(len(seen))
        if q is None:
            dropped[str(name)] = "no recorded true win rate for this opponent"
            continue
        if M == 0 or m == 0.0 or m == M:
            dropped[str(name)] = (f"the quota captured only one outcome class "
                                  f"({int(m)} wins of {int(M)} battles) — nothing to reweight")
            continue
        captured = m / M
        w_win, w_loss = q / captured, (1.0 - q) / (1.0 - captured)
        w[sel] = np.where(y[sel] > 0.5, w_win, w_loss)
        per_opp[str(name)] = {"true_win_rate": q, "captured_win_rate": captured,
                              "w_win": w_win, "w_loss": w_loss, "n_battles": int(M)}
    total = float(w.sum())
    if total <= 0.0:
        raise SelectionWeightError(
            "--reliability-reweight resolved no usable opponent: " + json.dumps(dropped, indent=1))
    w *= w.size / total                                   # mean 1; every statistic is a ratio
    return w, {"per_opponent": per_opp, "dropped": dropped,
               "n_rows_zeroed": int((w == 0.0).sum())}


def render_reliability(report: Dict[str, Any]) -> str:
    """Section (4): the reliability / Brier / ECE table, per step and per opponent class."""
    blocks = report.get("reliability") or []
    lines: List[str] = []
    lines.append("\n(4) RELIABILITY — the win-prob head against the REALIZED OUTCOME.")
    lines.append("    skill = 1 - Brier/Brier_base. NEGATIVE = worse than always predicting the")
    lines.append("    slice's base rate. res(olution) HIGHER is better; rel(iability) LOWER is.")
    for blk in blocks:
        tag = ("SELECTION-REWEIGHTED to the eval cycle's win rates" if blk.get("reweighted")
               else "RAW capture quota — NOT the deployed population")
        lines.append(f"\n  step {blk['step']:,}   ({blk['bins']} equal-width bins)  [{tag}]")
        sel = blk.get("selection") or {}
        if sel.get("dropped"):
            for name, why in sorted(sel["dropped"].items()):
                lines.append(f"    ⚠️  {name}: WEIGHT 0 — {why}")
        lines.append(f"    {'stratum':<16}{'n':>7}{'ess':>7}{'btl':>5}{'base':>7}{'Brier':>9}"
                     f"{'[ 95% cluster CI ]':>22}{'skill':>8}{'ECE':>8}{'MCE':>8}"
                     f"{'rel':>8}{'res':>8}")
        for r in blk["strata"]:
            tag = r["name"] if r["kind"] != "class" else f"{r['name']} (class)"
            lines.append(
                f"    {tag:<16}{r['n']:>7}{_f(r.get('ess'), '7.0f')}"
                f"{r['n_battles']:>5}{_f(r['base_rate'], '7.3f')}"
                f"{_f(r['brier'], '9.4f')}"
                f"    [{_f(r['brier_ci_lo'], '7.4f')},{_f(r['brier_ci_hi'], '7.4f')} ]"
                f"{_f(r['skill'], '8.3f')}{_f(r['ece'], '8.3f')}{_f(r['mce'], '8.3f')}"
                f"{_f(r['reliability'], '8.4f')}{_f(r['resolution'], '8.4f')}")
        allrow = next((r for r in blk["strata"] if r["kind"] == "all"), None)
        if allrow:
            lines.append(f"    per-bin reliability curve (stratum `all`)  "
                         f"[decomp residual {_f(allrow['decomp_residual'], '.4f')}]")
            lines.append(f"      {'bin':<12}{'n':>7}{'p_mean':>9}{'y_rate':>9}{'gap':>9}")
            for row in allrow["table"]:
                span = "{:.1f}-{:.1f}".format(row["lo"], row["hi"])
                lines.append(f"      {span:<12}{row['n']:>7}"
                             f"{_f(row['p_mean'], '9.3f')}{_f(row['y_rate'], '9.3f')}"
                             f"{_f(row['gap'], '9.3f')}")
    lines.append("\n  ⚠️  SELECTION: these battles are the eval QUOTA's, not a random sample of")
    lines.append("      play, and the quota over-captures losses. The labels are per-BATTLE and")
    lines.append("      broadcast to every decision of that battle. Read `bot` and `pool`")
    lines.append("      separately — the head's bias is measured to FLIP SIGN between them.")
    # What this tree RECORDS about that quota — the label, per step. A step with no record is
    # named as UNKNOWN rather than left to read as uniform.
    sel = report.get("trace_selection") or {}
    for step in sorted((sel.get("steps") or {}), key=lambda s: int(s)):
        e = sel["steps"][step]
        src = e.get("true_win_rate_source") or "none"
        lines.append(f"      step {int(step):,}: {e['label']}")
        lines.append(f"        true-rate source for --reliability-reweight: {src}")
    return "\n".join(lines)
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
    ap.add_argument("--reliability", action="store_true",
                    help="ALSO compute the win-prob head's calibration against the realized "
                         "outcome — reliability curve, Brier, Brier SKILL score, ECE and the "
                         "Murphy reliability/resolution split, stratified by opponent CLASS "
                         "(bot vs pool sentinel) and by opponent. Adds a `reliability` block to "
                         "the JSON; the existing blocks are untouched.")
    ap.add_argument("--reliability-bins", type=int, default=10,
                    help="equal-width forecast bins for --reliability (default 10)")
    ap.add_argument("--reliability-reweight", action="store_true",
                    help="CORRECT THE CAPTURE QUOTA. The recorded traces are a loss-enriched "
                         "subsample (measured on ai_v9_59_R2ACTION_0827: captured outcome rate "
                         "0.46 against a recorded 0.90 vs bots), so an unweighted table scores the "
                         "head on a population it was never deployed against. This importance-"
                         "weights each opponent's rows back to the win/loss mix the cycle itself "
                         "recorded — PREFERRING each cycle's own eval_manifest.json selection "
                         "block, falling back to eval_results.jsonl where it has none. REFUSES "
                         "rather than falling back to unweighted when neither can be resolved.")
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
    if args.reliability:
        # WHAT THE TRACES ARE A SAMPLE OF — the recorder's quota, read off the tree itself
        # (`gen3_trace_selection_manifest_v1`). Present under --reliability whether or not the
        # tree records one: absent reads as SELECTION UNKNOWN, never as uniform.
        report["trace_selection"] = build_trace_selection_block(run_dir)
        rw = true_win_rates(run_dir) if args.reliability_reweight else None
        report["reliability"] = build_reliability(
            slices, bins=args.reliability_bins, n_boot=args.boot, seed=args.seed, reweight=rw)
        report["units"]["reliability"] = {
            "what": "The win-prob head scored against the REALIZED per-battle outcome: Brier, the "
                    "Brier SKILL score against the slice's own base rate, ECE/MCE, and Murphy's "
                    "reliability/resolution split, per opponent CLASS (bot vs pool sentinel) and "
                    "per opponent. Unlike both gauges above, this compares the head to the TRUTH "
                    "rather than to the other readout.",
            "cannot": "The battles are the eval QUOTA's, not a random sample of play, and the "
                      "quota over-captures losses — so the base rate is a property of the quota. "
                      "Labels are per-battle and broadcast to every decision, so `n` is not a "
                      "sample size; only the cluster CIs are. A pooled row averages two "
                      "populations whose measured bias has OPPOSITE SIGN "
                      "(designs/learning/win_prob_decomposition.md axis 3) and must never be "
                      "quoted alone.",
        }
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
    if args.reliability and not args.constancy:
        print(render_reliability(report))
    return 0


if __name__ == "__main__":                                       # pragma: no cover
    sys.exit(main())
