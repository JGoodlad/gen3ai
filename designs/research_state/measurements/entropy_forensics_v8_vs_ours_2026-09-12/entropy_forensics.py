"""Policy-entropy + late-strength-slope forensics across the v8 line and the gen/win-prob eras.

    export PYTHONPATH=$PYTHONPATH:src
    python3 entropy_forensics.py --out <dir>

ZERO GPU, zero torch. Reads only TensorBoard events (through the project's own reader,
``main.ops.tb_read.load`` — never a hand-rolled event parser), ``snapshot_ladder/ladder.json``,
``metadata.json`` and ``snapshots/``. Writes ONLY under ``--out``; nothing under ``models/``.

WHAT THE ENTROPY TAG IS. The project logs ``train/entropy_loss`` =
``-mean(entropy_per_decision)`` in NATS over the 11-way MASKED categorical
(``instrumented_ppo/ppo.py:415``, the standard UNWEIGHTED metric — independent of the
defensive/bait entropy boosts, both 1.0 in every run here). So policy entropy is

    H = -train/entropy_loss   [nats],  H_max = ln(11) = 2.3979 when all 11 actions are legal.

``ACTION_SPACE_SIZE = 11`` (``agents/action/constants.py``) and the ``[switch x6, move x4,
struggle]`` layout have been unchanged since 2026-05-22 (``f02e633a``), i.e. across every run
here, so the entropy SCALE is commensurable across the eras. What is NOT commensurable is the
mask's mean SUPPORT: the number of legal actions is a function of the opponent ecology and board
(faints, trapping, forced switches), which differs by era — so a LEVEL difference between eras
carries an ecology confound, and only WITHIN-run shapes are read as such.

TWO HAZARDS THIS SCRIPT ENCODES.

1. **A fork's TB directory carries its PARENT's history.** ``gen3_tb_inherit_v1`` writes an
   ``events.out.tfevents.0000000000.inherited.0.0`` prefix truncated at ``fork_step``, and
   ``tb_read.load()`` reads every event file in the directory — so a naive read of a fold's
   entropy returns the whole ancestral chain. Every series here is split at ``fork_step``
   (from ``metadata.json``'s lineage / the recorded ``tb_inherit`` provenance): ``own`` is
   ``step > fork_step``, ``inherited`` is ``step <= fork_step``.
2. **A fork's ``snapshots/`` and therefore its ``ladder.json`` carry the PARENT's nodes too**
   (a genuine fork auto-seeds its parent's self-play pool). ``ai_v9_59``'s 14-node ladder is 12
   inherited nodes + 2 of its own. Ladder nodes are filtered to ``step > fork_step``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from main.ops.run_ref import resolve_run_dir
from main.ops import tb_read

# --------------------------------------------------------------------------------------------
# The cohort. `era` is the recipe family; `spine` chains the segments that are one continuous
# optimisation trajectory (a fork continues its parent's weights, so its own segment is the
# tail of the parent's curve).
# --------------------------------------------------------------------------------------------
RUNS: List[Dict[str, object]] = [
    dict(name="ai_v8_01_zarch_film_0717", era="v8", spine="v8_sibling",
         label="v8 line, FiLM-heads sibling arm off the 148.4M surgery point"),
    dict(name="ai_v8_03_zarch_control_0718", era="v8", spine="v8",
         label="v8 line, the backbone control arm (v8_04's parent)"),
    dict(name="ai_v8_04_distill_4teacher_0722", era="v8", spine="v8",
         label="v8 line, 4-teacher fold — THE 277M 'strong and still learning' parent"),
    dict(name="ai_v8_14_distill3_0725", era="v8", spine="v8",
         label="v8 line, 3-teacher fold — the +69-anchored-ELO gift"),
    dict(name="ai_v9_29_rev1_0823", era="gen", spine="gen",
         label="gen era, fresh-from-zero parent rev-1 (v9_59's parent)"),
    dict(name="ai_v9_59_R2ACTION_0827", era="gen", spine="gen",
         label="gen era, R2-ACTION fold — THE 28M gen-era parent that does NOT gain"),
    dict(name="ai_v12_02_winprob_critic", era="v12", spine="v12_02",
         label="win-prob era, arm 2, fresh, 75M"),
    dict(name="ai_v12_11_ladder_ctrl10M", era="v12", spine="v12_11",
         label="win-prob era, 10M ladder CONTROL arm A, fresh"),
]

#: The SEED REPLICATES of `ai_v12_11`'s exact recipe. They cost nothing extra and they are what
#: turns an entropy DIFFERENCE into a reading: the project's floor is the MAX pairwise |delta|
#: over the replicates in hand, and without one an entropy gap is a number, not a finding.
REPLICATES: List[Dict[str, object]] = [
    dict(name="ai_v12_11_ladder_ctrl10M", era="v12", spine="v12_11", label="ctrl10M seed A"),
    dict(name="ai_v12_15_ladder_ctrl10M_b", era="v12", spine="v12_15", label="ctrl10M seed B"),
    dict(name="ai_v12_16_ladder_ctrl10M_c", era="v12", spine="v12_16", label="ctrl10M seed C"),
]

TAGS = [
    "train/entropy_loss",
    "train/approx_kl",
    "train/clip_fraction",
    "train/explained_variance",
    "train/learning_rate",
    "train/clip_range",
    "train/grad_norm",
    "train/n_epochs",
    "train/selfplay_fraction",
    "train/selfplay_promoted_steps",
    "eval/pool_snapshot_count",
    "eval/win_rate_vs_pool",
    "eval/win_rate_vs_bots",
    "hparams/ent_coef",
]

H_MAX = math.log(11.0)


# --------------------------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------------------------
def fork_step(run_dir: Path) -> int:
    """The step this run's OWN optimisation began at. 0 for a fresh run.

    Read from the recorded lineage, never re-derived from the events (a fresh run's first
    rollout is not at step 0 either, so "the first event" cannot distinguish the two).
    """
    md = json.loads((run_dir / "metadata.json").read_text())
    lin = md.get("lineage") or {}
    for key in ("fork_step", "forked_at_steps", "parent_steps"):
        v = lin.get(key)
        if isinstance(v, (int, float)):
            return int(v)
    ti = md.get("tb_inherit") or {}
    if isinstance(ti.get("fork_step"), (int, float)):
        return int(ti["fork_step"])
    inh = run_dir / "tb" / "events.out.tfevents.0000000000.inherited.0.0"
    sidecar = run_dir / "tb" / "tb_inherit.json"
    if sidecar.exists():
        return int(json.loads(sidecar.read_text())["fork_step"])
    if inh.exists():                                    # prefix present, provenance unreadable
        raise SystemExit(f"REFUSING {run_dir.name}: inherited TB prefix present but no fork_step "
                         "recorded — the own/inherited split would be a guess.")
    return 0


def _series(sr: Dict[str, List[Tuple[int, float]]], tag: str) -> Tuple[np.ndarray, np.ndarray]:
    pts = sr.get(tag) or []
    if not pts:
        return np.array([]), np.array([])
    a = np.asarray(pts, dtype=float)
    return a[:, 0], a[:, 1]


def _med_head(v: np.ndarray, k: int) -> Optional[float]:
    return float(np.median(v[:k])) if v.size else None


def _med_tail(v: np.ndarray, k: int) -> Optional[float]:
    return float(np.median(v[-k:])) if v.size else None


# --------------------------------------------------------------------------------------------
# slopes
# --------------------------------------------------------------------------------------------
def ols_slope(x: np.ndarray, y: np.ndarray) -> Dict[str, Optional[float]]:
    """Unweighted OLS slope with its residual SE. n < 3 -> refuse (None), never a point."""
    n = int(x.size)
    if n < 3:
        return dict(n=n, slope=None, se=None, intercept=None)
    X = np.vstack([np.ones(n), x]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - 2
    s2 = float(resid @ resid) / dof
    xtx_inv = np.linalg.inv(X.T @ X)
    se = math.sqrt(max(s2 * xtx_inv[1, 1], 0.0))
    return dict(n=n, slope=float(beta[1]), se=se, intercept=float(beta[0]))


def wls_slope(x: np.ndarray, y: np.ndarray, se_y: np.ndarray) -> Dict[str, Optional[float]]:
    """Inverse-variance WLS slope. Returns BOTH SEs and the WIDER as `se`.

    `se_analytic` believes the per-node SEs; `se_resid` believes the scatter. The nodes are NOT
    independent (Bradley-Terry re-solves every rating on every add), so both understate; taking
    the wider is the honest of the two available choices, and it is labelled as such.
    """
    n = int(x.size)
    if n < 3:
        return dict(n=n, slope=None, se=None, se_analytic=None, se_resid=None)
    w = 1.0 / np.clip(se_y, 1e-6, None) ** 2
    X = np.vstack([np.ones(n), x]).T
    W = np.diag(w)
    xtwx_inv = np.linalg.inv(X.T @ W @ X)
    beta = xtwx_inv @ X.T @ W @ y
    resid = y - X @ beta
    se_analytic = math.sqrt(max(xtwx_inv[1, 1], 0.0))
    chi2 = float(resid @ (w * resid)) / (n - 2)
    se_resid = se_analytic * math.sqrt(max(chi2, 0.0))
    se = max(se_analytic, se_resid)
    return dict(n=n, slope=float(beta[1]), se=se, se_analytic=se_analytic, se_resid=se_resid)


def verdict(slope: Optional[float], se: Optional[float]) -> str:
    if slope is None or se is None:
        return "REFUSED (n < 3)"
    if slope - 2 * se > 0:
        return "RISING"
    if slope + 2 * se < 0:
        return "FALLING"
    return "NOT DETECTED (interval spans zero)"


# --------------------------------------------------------------------------------------------
# part 1 — entropy
# --------------------------------------------------------------------------------------------
def read_run(spec: Dict[str, object]) -> Dict[str, object]:
    name = str(spec["name"])
    rd = resolve_run_dir(name)
    fs = fork_step(rd)
    sr = tb_read.load(rd, TAGS)

    out: Dict[str, object] = dict(
        run=name, era=spec["era"], spine=spec["spine"], label=spec["label"],
        run_dir=str(rd), fork_step=fs,
    )

    st, el = _series(sr, "train/entropy_loss")
    if st.size == 0:
        raise SystemExit(f"REFUSING {name}: NO SCALAR train/entropy_loss. Absence is not a zero.")
    own = st > fs
    out["n_entropy_points_total"] = int(st.size)
    out["n_entropy_points_own"] = int(own.sum())
    out["n_entropy_points_inherited"] = int((~own).sum())
    if own.sum() < 8:
        out["entropy"] = dict(status=f"REFUSED — only {int(own.sum())} own rollouts")
    else:
        s, H = st[own], -el[own]               # H = -entropy_loss, nats
        k = max(3, int(0.02 * H.size))
        frac = (s - s[0]) / max(float(s[-1] - s[0]), 1.0)
        h0, h1 = _med_head(H, k), _med_tail(H, k)
        i_min = int(np.argmin(H))
        # last-quarter and middle-half slopes, per 1M env steps
        q = s >= s[0] + 0.75 * (s[-1] - s[0])
        m = (s >= s[0] + 0.25 * (s[-1] - s[0])) & (s <= s[0] + 0.75 * (s[-1] - s[0]))
        late = ols_slope(s[q] / 1e6, H[q])
        mid = ols_slope(s[m] / 1e6, H[m])
        # where does H first fall below a share of its own start?
        cross = {}
        for share in (0.9, 0.75, 0.5, 0.25):
            idx = np.nonzero(H <= share * h0)[0]
            cross[f"first_step_below_{share:g}x_H0"] = int(s[idx[0]]) if idx.size else None
            cross[f"first_frac_below_{share:g}x_H0"] = round(float(frac[idx[0]]), 4) if idx.size else None
        out["entropy"] = dict(
            status="OK", tag="train/entropy_loss", sign="H = -train/entropy_loss (nats)",
            H_max_all_legal=round(H_MAX, 4),
            step_first=int(s[0]), step_last=int(s[-1]), n=int(H.size), window_k=k,
            H_start=round(h0, 4), H_end=round(h1, 4),
            H_end_over_H_start=round(h1 / h0, 4),
            H_start_over_Hmax=round(h0 / H_MAX, 4), H_end_over_Hmax=round(h1 / H_MAX, 4),
            H_min=round(float(H[i_min]), 4), H_min_step=int(s[i_min]),
            H_min_frac_of_run=round(float(frac[i_min]), 4),
            H_min_over_H_start=round(float(H[i_min]) / h0, 4),
            H_max_observed=round(float(H.max()), 4),
            late_slope_nats_per_Mstep=None if late["slope"] is None else round(late["slope"], 6),
            late_slope_se=None if late["se"] is None else round(late["se"], 6),
            late_n=late["n"],
            late_verdict=verdict(late["slope"], late["se"]),
            mid_slope_nats_per_Mstep=None if mid["slope"] is None else round(mid["slope"], 6),
            mid_slope_se=None if mid["se"] is None else round(mid["se"], 6),
            mid_n=mid["n"], mid_verdict=verdict(mid["slope"], mid["se"]),
            **cross,
        )

    # the companion PPO health scalars, own segment only
    comp: Dict[str, object] = {}
    for tag in ("train/approx_kl", "train/clip_fraction", "train/explained_variance",
                "train/learning_rate", "train/grad_norm", "train/clip_range"):
        s, v = _series(sr, tag)
        if s.size == 0:
            comp[tag] = "NO SCALAR FOUND"
            continue
        o = s > fs
        if o.sum() == 0:
            comp[tag] = "no own points"
            continue
        vv = v[o]
        k = max(3, int(0.02 * vv.size))
        comp[tag] = dict(n=int(vv.size), start=round(float(np.median(vv[:k])), 6),
                         median=round(float(np.median(vv)), 6),
                         end=round(float(np.median(vv[-k:])), 6),
                         min=round(float(vv.min()), 6), max=round(float(vv.max()), 6))
    out["companions"] = comp

    # ent_coef as the run ACTUALLY logged it (never from the argv)
    s, v = _series(sr, "hparams/ent_coef")
    if s.size:
        o = s > fs
        vv = v[o] if o.sum() else v
        out["ent_coef_logged"] = dict(distinct=sorted({round(float(x), 6) for x in vv}),
                                      n=int(vv.size), source="TB hparams/ent_coef")
    else:
        out["ent_coef_logged"] = "NO SCALAR FOUND"

    s, v = _series(sr, "train/n_epochs")
    out["n_epochs_logged"] = sorted({int(x) for x in v[s > fs]}) if s.size else "NO SCALAR FOUND"

    # --- the STALENESS leg of the collapse hypothesis -------------------------------------
    # `train/selfplay_promoted_steps` is recorded ONLY at a promotion (selfplay_callback.py:800)
    # and its VALUE is the promoted step, so the series IS the promotion ladder. A run that stops
    # promoting keeps training against a frozen pool — the operational meaning of "self-play went
    # stale". The trailing LAG (last own step - last own promotion) is the direct read.
    ps, pv = _series(sr, "train/selfplay_promoted_steps")
    stale: Dict[str, object] = {}
    if ps.size == 0:
        stale["promotions"] = "NO SCALAR FOUND"
    else:
        o = ps > fs
        promoted = sorted({int(x) for x in pv[o]})
        last_own_step = int(st[own].max()) if own.sum() else None
        gaps = [b - a for a, b in zip(promoted, promoted[1:])]
        stale["promotions"] = dict(
            n_own=len(promoted), steps=promoted,
            n_inherited_from_parent=int((~o).sum()),
            mean_gap=None if not gaps else int(sum(gaps) / len(gaps)),
            max_gap=None if not gaps else int(max(gaps)),
            last_promotion=promoted[-1] if promoted else None,
            last_own_step=last_own_step,
            trailing_lag_steps=None if (not promoted or last_own_step is None)
                               else int(last_own_step - promoted[-1]),
            trailing_lag_share_of_own_segment=None if (not promoted or last_own_step is None)
                               else round((last_own_step - promoted[-1])
                                          / max(last_own_step - int(st[own].min()), 1), 4),
        )
    for tag in ("eval/pool_snapshot_count", "eval/win_rate_vs_pool", "eval/win_rate_vs_bots",
                "train/selfplay_fraction"):
        s_, v_ = _series(sr, tag)
        if s_.size == 0:
            stale[tag] = "NO SCALAR FOUND"
            continue
        o = s_ > fs
        if o.sum() == 0:
            stale[tag] = "no own points (values present are the PARENT's, inherited)"
            continue
        vv = v_[o]
        stale[tag] = dict(n=int(vv.size), first=round(float(vv[0]), 4),
                          median=round(float(np.median(vv)), 4),
                          last=round(float(vv[-1]), 4), max=round(float(vv.max()), 4))
    try:
        mc = json.loads((rd / "model_config.json").read_text())
        stale["eval_sentinel_greedy_recorded"] = mc.get("eval_sentinel_greedy", "NOT RECORDED "
                                                        "(pre config v112) — regime UNKNOWN, "
                                                        "read the launch line")
        stale["config_version"] = mc.get("config_version")
    except Exception as exc:
        stale["eval_sentinel_greedy_recorded"] = f"unreadable: {exc}"
    out["staleness"] = stale

    try:
        cr = tb_read.selfplay_crossing_step(rd)
        if fs > 0:
            cr["HAZARD"] = ("this run INHERITED its parent's TB prefix, which carries the pool "
                            "tags — the crossing below is the PARENT's, not this fork's. A fork "
                            "starts with the parent's pool seeded, so it has no crossing of its own.")
        out["selfplay_crossing"] = cr
    except Exception as exc:                                     # descriptive, never fatal
        out["selfplay_crossing"] = f"unavailable: {type(exc).__name__}: {exc}"

    # the raw own-segment entropy curve, thinned for the plot + the json
    if out["entropy"].get("status") == "OK":                     # type: ignore[union-attr]
        s, H = st[own], -el[own]
        stride = max(1, s.size // 400)
        out["entropy_curve"] = dict(step=[int(x) for x in s[::stride]],
                                    H=[round(float(x), 5) for x in H[::stride]])
    return out


# --------------------------------------------------------------------------------------------
# part 2 — the dense ladder's late slope
# --------------------------------------------------------------------------------------------
def read_ladder(spec: Dict[str, object]) -> Dict[str, object]:
    name = str(spec["name"])
    rd = resolve_run_dir(name)
    fs = fork_step(rd)
    p = rd / "snapshot_ladder" / "ladder.json"
    if not p.exists():
        return dict(run=name, status="NO ladder.json — this run has no dense ladder")
    d = json.loads(p.read_text())
    rat, ses = d["ratings"], d["se"]
    nodes = sorted(((int(k), float(rat[k]), float(ses.get(k, float('nan')))) for k in rat),
                   key=lambda t: t[0])
    own = [t for t in nodes if t[0] > fs]
    res: Dict[str, object] = dict(
        run=name, era=spec["era"], status="OK", ladder_version=d.get("version"),
        base=d.get("base"), converged=d.get("converged"),
        anchored_to_bots=d.get("anchored_to_bots"),
        n_frozen_pairs_measured=d.get("n_frozen_pairs_measured"),
        eval_sentinel_edges_dropped=d.get("eval_sentinel_edges_dropped"),
        fork_step=fs, n_nodes_in_file=len(nodes), n_nodes_own=len(own),
        n_nodes_inherited_from_parent=len(nodes) - len(own),
        nodes=[dict(step=s, rating=round(r, 1), se=round(e, 1)) for s, r, e in own],
    )
    if len(own) < 4:
        res["slopes"] = dict(status=f"REFUSED — {len(own)} own node(s); a slope needs >= 4 "
                                   "(3 after the newest is dropped for §3.2 rule 2)")
        return res

    # §3.2 rule 2: the newest BT node is systematically inflated. Headline drops it.
    for tag, seq in (("with_newest", own), ("drop_newest", own[:-1])):
        x = np.array([t[0] for t in seq], float) / 1e6
        y = np.array([t[1] for t in seq], float)
        e = np.array([t[2] for t in seq], float)
        n = len(seq)
        third = max(3, int(math.ceil(n / 3)))
        lo, hi = max(0, n - third), n
        mlo = max(0, (n - third) // 2)
        mid_sl = slice(mlo, mlo + third)
        full = wls_slope(x, y, e)
        late = wls_slope(x[lo:hi], y[lo:hi], e[lo:hi])
        midd = wls_slope(x[mid_sl], y[mid_sl], e[mid_sl])
        diff = None
        if late["slope"] is not None and midd["slope"] is not None:
            ds = late["slope"] - midd["slope"]
            dse = math.sqrt(late["se"] ** 2 + midd["se"] ** 2)
            diff = dict(delta=round(ds, 3), se=round(dse, 3), verdict=verdict(ds, dse))
        res.setdefault("slopes", {})[tag] = dict(     # type: ignore[union-attr]
            n_nodes=n,
            step_span_M=[round(float(x[0]), 2), round(float(x[-1]), 2)],
            full=_fmt(full), late=_fmt(late), late_window=[int(seq[lo][0]), int(seq[hi - 1][0])],
            middle=_fmt(midd), middle_window=[int(seq[mid_sl][0][0]), int(seq[mid_sl][-1][0])],
            late_minus_middle=diff,
            total_elo_change=round(float(y[-1] - y[0]), 1),
        )
    return res


def _fmt(s: Dict[str, Optional[float]]) -> Dict[str, object]:
    return dict(n=s["n"],
                slope_elo_per_Mstep=None if s["slope"] is None else round(s["slope"], 3),
                se=None if s.get("se") is None else round(s["se"], 3),
                se_analytic=None if s.get("se_analytic") is None else round(s["se_analytic"], 3),
                se_resid=None if s.get("se_resid") is None else round(s["se_resid"], 3),
                verdict=verdict(s["slope"], s.get("se")))


# --------------------------------------------------------------------------------------------
# plots
# --------------------------------------------------------------------------------------------
def plot(results: Sequence[Dict[str, object]], out: Path) -> List[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colours = {"v8": "#1b6ca8", "gen": "#c0392b", "v12": "#7d3c98"}
    styles = {"ai_v8_01_zarch_film_0717": (0, (2, 2)), "ai_v8_03_zarch_control_0718": "-",
              "ai_v8_04_distill_4teacher_0722": (0, (5, 2)),
              "ai_v8_14_distill3_0725": (0, (1, 1)),
              "ai_v9_29_rev1_0823": "-", "ai_v9_59_R2ACTION_0827": (0, (5, 2)),
              "ai_v12_02_winprob_critic": "-", "ai_v12_11_ladder_ctrl10M": (0, (5, 2)),
              "ai_v12_15_ladder_ctrl10M_b": (0, (5, 2)),
              "ai_v12_16_ladder_ctrl10M_c": (0, (5, 2))}
    made: List[str] = []

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    for r in results:
        c = r.get("entropy_curve")
        if not c:
            continue
        s = np.asarray(c["step"], float)
        H = np.asarray(c["H"], float)
        era = str(r["era"])
        kw = dict(color=colours[era], linestyle=styles[str(r["run"])], lw=1.5, alpha=0.9,
                  label=f"{r['run']} ({era}, ent-coef "
                        f"{r['ent_coef_logged']['distinct'] if isinstance(r['ent_coef_logged'], dict) else '?'})")
        axes[0].plot(s / 1e6, H, **kw)
        axes[1].plot((s - s[0]) / max(s[-1] - s[0], 1.0), H, **kw)
    for ax, xl, t in ((axes[0], "absolute env step (M)", "vs ABSOLUTE step"),
                      (axes[1], "fraction of the run's OWN segment", "vs FRACTION of run")):
        ax.axhline(H_MAX, color="k", lw=0.8, ls=":")
        ax.annotate("ln(11) = 2.398, all 11 legal", (0.02, H_MAX), xycoords=("axes fraction", "data"),
                    fontsize=7, va="bottom")
        ax.set_xlabel(xl); ax.set_ylabel("policy entropy H (nats)")
        ax.set_title(f"Policy entropy {t}", fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].set_xscale("log")
    axes[1].legend(fontsize=6.5, loc="upper right")
    fig.suptitle("H = -train/entropy_loss over the 11-way masked categorical; OWN segment only "
                 "(inherited TB prefix stripped at fork_step)", fontsize=9)
    fig.tight_layout()
    f = out / "entropy_trajectories.png"
    fig.savefig(f, dpi=140); plt.close(fig); made.append(f.name)

    # the spines, chained
    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    spines: Dict[str, List[Dict[str, object]]] = {}
    for r in results:
        if r.get("entropy_curve"):
            spines.setdefault(str(r["spine"]), []).append(r)
    for sp, rs in sorted(spines.items()):
        rs = sorted(rs, key=lambda r: r["entropy"]["step_first"])   # type: ignore[index]
        xs, ys = [], []
        for r in rs:
            c = r["entropy_curve"]
            xs += list(np.asarray(c["step"], float) / 1e6)
            ys += list(c["H"])
        ax.plot(xs, ys, lw=1.4, label=f"{sp}: " + " -> ".join(str(r["run"]) for r in rs))
    ax.axhline(H_MAX, color="k", lw=0.8, ls=":")
    ax.set_xscale("log"); ax.set_xlabel("absolute env step (M), log"); ax.set_ylabel("H (nats)")
    ax.set_title("Chained optimisation SPINES — a fork continues its parent's weights", fontsize=10)
    ax.grid(alpha=0.25); ax.legend(fontsize=7)
    fig.tight_layout()
    f = out / "entropy_spines.png"
    fig.savefig(f, dpi=140); plt.close(fig); made.append(f.name)
    return made


def plot_ladders(lads: Sequence[Dict[str, object]], out: Path) -> Optional[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    usable = [l for l in lads if l.get("status") == "OK" and l.get("nodes")]
    if not usable:
        return None
    fig, axes = plt.subplots(1, len(usable), figsize=(3.4 * len(usable), 4.2), squeeze=False)
    for ax, l in zip(axes[0], usable):
        n = l["nodes"]
        x = [d["step"] / 1e6 for d in n]
        y = [d["rating"] for d in n]
        e = [d["se"] for d in n]
        ax.errorbar(x, y, yerr=e, marker="o", ms=3.5, lw=1.2, capsize=2)
        if len(n) >= 2:
            ax.plot(x[-1], y[-1], marker="o", ms=8, mfc="none", mec="crimson", mew=1.6)
        ax.set_title(f"{l['run']}\n{l['n_nodes_own']} own nodes "
                     f"({l['n_nodes_inherited_from_parent']} inherited dropped)", fontsize=7.5)
        ax.set_xlabel("env step (M)"); ax.grid(alpha=0.25)
    axes[0][0].set_ylabel("dense-ladder rating (Elo, base 1000)")
    fig.suptitle("snapshot_ladder/ladder.json — own nodes only; the ringed newest node is "
                 "systematically INFLATED (UNDERSTANDING §3.2 rule 2) and is dropped from the "
                 "headline slope", fontsize=8)
    fig.tight_layout()
    f = out / "ladder_late_slopes.png"
    fig.savefig(f, dpi=140); plt.close(fig)
    return f.name


# --------------------------------------------------------------------------------------------
# part 1b — the FORK DISCONTINUITY in entropy
# --------------------------------------------------------------------------------------------
PARENT_OF = {
    "ai_v8_04_distill_4teacher_0722": "ai_v8_03_zarch_control_0718",
    "ai_v8_14_distill3_0725": "ai_v8_04_distill_4teacher_0722",
    "ai_v9_59_R2ACTION_0827": "ai_v9_29_rev1_0823",
}


def fork_discontinuity(ent: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """H just after a fold's first rollouts minus H at its parent's last rollouts.

    A fold resumes its parent's WEIGHTS and optimizer, so a step change here is attributable to
    the fold's own objective + rollout distribution, not to re-initialisation. It is a
    DESCRIPTIVE contrast — n = 1 per fold, no floor, and the parent's own end-of-run drift is a
    live alternative account (the windows are ~2% of each segment, so that drift is small but
    not zero).
    """
    by = {str(r["run"]): r for r in ent}
    rows = []
    for child, parent in PARENT_OF.items():
        c, p_ = by.get(child), by.get(parent)
        if not c or not p_:
            continue
        ce, pe = c.get("entropy", {}), p_.get("entropy", {})
        if ce.get("status") != "OK" or pe.get("status") != "OK":
            continue
        rows.append(dict(
            fold=child, parent=parent, era=c["era"],
            parent_H_end=pe["H_end"], fold_H_start=ce["H_start"],
            delta_nats=round(ce["H_start"] - pe["H_end"], 4),
            fold_H_end=ce["H_end"],
            fold_end_minus_parent_end=round(ce["H_end"] - pe["H_end"], 4),
        ))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="output directory (created; nothing else written)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    ent = [read_run(s) for s in RUNS]
    lad = [read_ladder(s) for s in RUNS]

    # the replicate floor on the entropy endpoints (identical recipe, different seed)
    reps = [read_run(s) for s in REPLICATES if str(s["name"]) not in {str(r["run"]) for r in ent}]
    rep_all = [r for r in ent if str(r["run"]) in {str(s["name"]) for s in REPLICATES}] + reps
    floor: Dict[str, object] = dict(
        recipe="ai_v12_11 / _15_b / _16_c — the 10M win-prob ladder CONTROL, seed replicates",
        arms=[dict(run=r["run"], H_start=r["entropy"]["H_start"], H_end=r["entropy"]["H_end"],
                   H_min=r["entropy"]["H_min"], step_last=r["entropy"]["step_last"],
                   late_slope=r["entropy"]["late_slope_nats_per_Mstep"])
              for r in rep_all if r["entropy"].get("status") == "OK"],
    )
    for key in ("H_start", "H_end", "H_min"):
        vals = [a[key] for a in floor["arms"]]          # type: ignore[index]
        floor[f"floor_{key}_nats"] = round(max(vals) - min(vals), 4) if len(vals) > 1 else None
    (out / "entropy_replicate_floor.json").write_text(json.dumps(floor, indent=2))
    for r in reps:
        ent.append(r)

    disc = fork_discontinuity(ent)
    (out / "fork_entropy_discontinuity.json").write_text(json.dumps(disc, indent=2))
    (out / "entropy.json").write_text(json.dumps(ent, indent=2))
    (out / "ladder_slopes.json").write_text(json.dumps(lad, indent=2))
    pngs = plot(ent, out)
    lp = plot_ladders(lad, out)
    if lp:
        pngs.append(lp)

    print("REPLICATE FLOOR on the entropy endpoints (" + str(floor["recipe"]) + "):")
    for a in floor["arms"]:                              # type: ignore[union-attr]
        print(f"   {a['run']:30s} H0 {a['H_start']:.3f}  Hend {a['H_end']:.3f}  Hmin {a['H_min']:.3f}")
    print(f"   => floor(H_end) = {floor['floor_H_end_nats']} nats, "
          f"floor(H_start) = {floor['floor_H_start_nats']}, floor(H_min) = {floor['floor_H_min_nats']}")
    print()
    print(f"wrote {out}/entropy.json, {out}/ladder_slopes.json, " + ", ".join(pngs))
    print()
    hdr = f"{'run':32s} {'era':4s} {'H0':>6s} {'Hend':>6s} {'end/H0':>7s} {'Hend/ln11':>10s} {'lateSlope':>10s} {'±se':>8s}  verdict"
    print(hdr); print("-" * len(hdr))
    for r in ent:
        e = r["entropy"]
        if e.get("status") != "OK":
            print(f"{r['run'][:32]:32s} {str(r['era']):4s} {e.get('status')}")
            continue
        print(f"{r['run'][:32]:32s} {str(r['era']):4s} {e['H_start']:6.3f} {e['H_end']:6.3f} "
              f"{e['H_end_over_H_start']:7.3f} {e['H_end_over_Hmax']:10.3f} "
              f"{e['late_slope_nats_per_Mstep']:10.4f} {e['late_slope_se']:8.4f}  {e['late_verdict']}")
    print()
    print(f"{'fold':32s} {'era':4s} {'parent Hend':>11s} {'fold H0':>8s} {'delta':>7s} {'fold Hend':>10s} {'vs parent':>10s}")
    for d in disc:
        print(f"{d['fold'][:32]:32s} {d['era']:4s} {d['parent_H_end']:11.3f} {d['fold_H_start']:8.3f} "
              f"{d['delta_nats']:+7.3f} {d['fold_H_end']:10.3f} {d['fold_end_minus_parent_end']:+10.3f}")
    print()
    for l in lad:
        if l.get("status") != "OK":
            print(f"{l['run']:32s} {l.get('status')}")
            continue
        s = l.get("slopes", {})
        if s.get("status"):
            print(f"{l['run']:32s} {s['status']}")
            continue
        d = s["drop_newest"]
        print(f"{l['run']:32s} own n={l['n_nodes_own']:2d}  late {d['late']['slope_elo_per_Mstep']:+8.2f} "
              f"±{d['late']['se']:6.2f} {d['late']['verdict']:32s} | mid "
              f"{d['middle']['slope_elo_per_Mstep']:+8.2f} ±{d['middle']['se']:6.2f} "
              f"| late-mid {d['late_minus_middle']['delta']:+8.2f} ±{d['late_minus_middle']['se']:6.2f} "
              f"{d['late_minus_middle']['verdict']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
