"""The X14 ON arm's K9(b) flag, diagnosed from the battery's own artifacts (CPU only, read-only on every input).

Two reads, both written to ``x14_k9b_diag.json`` beside this file (``X14_K9B_DIAGNOSIS.md`` is the verdict):

1. ``tb``: the per-update K9(b) series of both arms (full precision, the fork's own rows) and X26's serial series over
   the WINDOW of updates just before the fork step (the same weights era), with the share of updates whose judged max
   reaches the 1e-4 bar and the exact binomial tail of the ON arm's count under X26's rate.
2. ``replay``: each violation's dumped snapshot weights (``behaviour_violation_u<k>_policy.pt``, written INSIDE the
   snapshot swap, so they are the version-k weights) and dumped rows (the violator + the scan's 32 worst), re-run on
   the CPU through the probe's own forward (`consistency.probe_forward`: train mode, fp32 trunk): a THIRD fp32
   program beside T2's CUDA graph (stored mu) and the learner's eager CUDA recompute, plus each row's rounding
   SENSITIVITY under a ~1-ulp weight jitter (`_jitter_spread`). (An fp64 forward is not available: the extractor
   casts its observation to fp32.)

Run from a checkout whose ``src/`` equals the battery pin's (f4b0a74f2; ``--check-pin`` verifies), with
``CUDA_VISIBLE_DEVICES=""``::

    PYTHONPATH=$PWD/src python designs/research_state/measurements/gpu_window_battery_2026-10-20/x14_k9b_diag.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
RUNS = Path.home() / "gen3ai_archive" / "x14_ab_runs_2026-10-21"
ON, OFF = RUNS / "rb_x14ab2_on_s2601", RUNS / "rb_x14ab2_off_s2601"
X26_RUN = "rb_x26_s2601"      # under the MAIN checkout's models/ (`utils.paths.main_models_dir`)
FORK = 380_000_195
ROWS_PER_UPDATE = 98_304
X26_WINDOW_UPDATES, X26_RECENT_UPDATES = 1_000, 200
PIN = "f4b0a74f2421a167b411972109147c8366b67671"
BAR = 1e-4
JITTER_REPS, JITTER_SEED = 8, 20261021
TAGS = ("behaviour/max_abs_dlogp_judged", "behaviour/p99_abs_dlogp_current", "behaviour/rows_current",
        "behaviour_snapshot/max_abs_dlogp_judged", "behaviour_snapshot/p99_abs_dlogp_current",
        "behaviour_snapshot/rows_current")


def _series(run: Path, lo: int, hi: int) -> Dict[str, List[Tuple[int, float]]]:
    from main.x35.tbrows import load_segments

    out: Dict[str, List[Tuple[int, float]]] = {t: [] for t in TAGS}
    for seg in load_segments(str(run), list(TAGS), after_step=lo):
        for t in TAGS:
            out[t] += [(s, v) for s, v in seg.rows.get(t, []) if s <= hi]
    return {t: sorted(v) for t, v in out.items()}


def _summ(vals: Sequence[float]) -> Dict[str, Any]:
    a = np.asarray([v for v in vals if math.isfinite(v)], dtype=np.float64)
    nonfinite = int(len(vals) - a.size)
    if not a.size:
        return {"n": 0, "nonfinite": nonfinite}
    return {"n": int(a.size), "nonfinite": nonfinite, "min": float(a.min()), "median": float(np.median(a)),
            "q90": float(np.quantile(a, 0.9)), "max": float(a.max()), "n_at_or_over_1e-4": int((a >= BAR).sum())}


def _binom_tail(k: int, n: int, p: float) -> float:
    """P(X >= k), X ~ Binomial(n, p), exact."""
    return float(sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1)))


def tb_read() -> Dict[str, Any]:
    hi = FORK + 10 ** 9
    arms = {"off": _series(OFF, FORK, hi), "on": _series(ON, FORK, hi)}
    from utils.paths import main_models_dir

    models = main_models_dir()
    if models is None:
        raise SystemExit("no main checkout models/ dir: X26's serial series cannot be read")
    x26 = _series(models / X26_RUN, FORK - X26_WINDOW_UPDATES * ROWS_PER_UPDATE, FORK)
    out: Dict[str, Any] = {"fork": FORK, "x26_window": [FORK - X26_WINDOW_UPDATES * ROWS_PER_UPDATE, FORK]}
    for name, ser in (("off", arms["off"]), ("on", arms["on"]), ("x26", x26)):
        rows = {t: dict(v) for t, v in ser.items()}
        # the empty-set sentinel: a p99 logged NaN at a step whose rows_current is 0 judged nothing
        for fam in ("behaviour", "behaviour_snapshot"):
            p99 = rows[f"{fam}/p99_abs_dlogp_current"]
            cur = rows[f"{fam}/rows_current"]
            empty_nan = [s for s, v in p99.items() if not math.isfinite(v) and cur.get(s) == 0.0]
            real_nan = [s for s, v in p99.items() if not math.isfinite(v) and cur.get(s) != 0.0]
            out.setdefault(name, {})[fam] = {
                "judged_max": _summ(list(rows[f"{fam}/max_abs_dlogp_judged"].values())),
                "judged_max_series": sorted(rows[f"{fam}/max_abs_dlogp_judged"].items()),
                "p99": _summ([v for s, v in p99.items() if s not in empty_nan]),
                "p99_nan_at_zero_current_rows": len(empty_nan), "p99_nan_with_current_rows": len(real_nan),
                "rows_current": _summ(list(cur.values()))}
    x = out["x26"]["behaviour"]["judged_max"]
    rate = x["n_at_or_over_1e-4"] / x["n"]
    on = out["on"]["behaviour_snapshot"]["judged_max"]
    off = out["off"]["behaviour"]["judged_max"]
    from scipy.stats import mannwhitneyu

    x26_vals = np.asarray([v for _, v in out["x26"]["behaviour"]["judged_max_series"]])
    x26_recent = x26_vals[-X26_RECENT_UPDATES:]
    on_vals = [v for _, v in out["on"]["behaviour_snapshot"]["judged_max_series"]]
    off_vals = [v for _, v in out["off"]["behaviour"]["judged_max_series"]]
    out["compare"] = {
        "x26_rate_at_or_over_1e-4": rate,
        "on_count": [on["n_at_or_over_1e-4"], on["n"]], "off_count": [off["n_at_or_over_1e-4"], off["n"]],
        "P_on_count_or_more_under_x26_rate": _binom_tail(on["n_at_or_over_1e-4"], on["n"], rate),
        "P_off_count_or_fewer_under_x26_rate": 1.0 - _binom_tail(off["n_at_or_over_1e-4"] + 1, off["n"], rate),
        # where each arm's per-update maxima sit in X26's distribution (mean percentile; 0.5 = same distribution)
        "on_mean_percentile_in_x26": float(np.mean([(x26_vals < v).mean() for _, v in
                                                     out["on"]["behaviour_snapshot"]["judged_max_series"]])),
        "off_mean_percentile_in_x26": float(np.mean([(x26_vals < v).mean() for _, v in
                                                      out["off"]["behaviour"]["judged_max_series"]])),
        # X26's LAST updates before the fork (the weights the arms start from), and two-sided Mann-Whitney U p-values
        "x26_recent_updates": int(x26_recent.size),
        "x26_recent_rate_at_or_over_1e-4": float((x26_recent >= BAR).mean()),
        "x26_recent_median": float(np.median(x26_recent)),
        "P_off_count_or_fewer_under_x26_recent_rate": 1.0 - _binom_tail(
            off["n_at_or_over_1e-4"] + 1, off["n"], float((x26_recent >= BAR).mean())),
        "mwu_p_on_vs_off": float(mannwhitneyu(on_vals, off_vals, alternative="two-sided").pvalue),
        "mwu_p_on_vs_x26_recent": float(mannwhitneyu(on_vals, x26_recent, alternative="two-sided").pvalue),
        "mwu_p_off_vs_x26_recent": float(mannwhitneyu(off_vals, x26_recent, alternative="two-sided").pvalue),
    }
    return out


def _forward(model: Any, obs: Dict[str, np.ndarray], acts: np.ndarray, masks: np.ndarray) -> np.ndarray:
    import torch as th

    from agents.training.rust_rollout.consistency import probe_forward

    new, _full = probe_forward(model, obs, th.as_tensor(acts).long(), th.as_tensor(masks))
    return new


def _jitter_spread(model: Any, obs: Dict[str, np.ndarray], acts: np.ndarray, masks: np.ndarray,
                   reps: int = JITTER_REPS) -> np.ndarray:
    """Each row's ROUNDING SENSITIVITY: the largest |log pi - log pi_0| over ``reps`` CPU fp32 forwards whose every
    weight is multiplied by (1 + 2^-23 * N(0, 1)) (a ~1-ulp jitter; the bar derivation's R used a few-ulp jitter,
    learner_gates.md), seeded; the weights are restored bit-exactly."""
    import torch as th

    base = _forward(model, obs, acts, masks)
    params = [p for p in model.policy.parameters() if p.is_floating_point()]
    keep = [p.detach().clone() for p in params]
    g = th.Generator().manual_seed(JITTER_SEED)
    worst = np.zeros_like(base)
    try:
        for _ in range(reps):
            with th.no_grad():
                for p, k in zip(params, keep):
                    p.copy_(k * (1 + 2.0 ** -23 * th.randn(k.shape, generator=g, dtype=k.dtype)))
            worst = np.maximum(worst, np.abs(_forward(model, obs, acts, masks) - base))
    finally:
        with th.no_grad():
            for p, k in zip(params, keep):
                p.copy_(k)
    return worst


def replay(u: int) -> Dict[str, Any]:
    import torch as th

    from main.capacity import load_policy

    rec = next(json.loads(l) for l in (ON / "behaviour_violations.jsonl").open() if json.loads(l)["n_updates"] == u)
    z = np.load(ON / f"behaviour_violation_u{u}_rows.npz")
    model, _cfg, _ = load_policy(str(ON / "final_model.zip"), str(ON), "cpu")
    sd = th.load(ON / f"behaviour_violation_u{u}_policy.pt", map_location="cpu")
    model.policy.load_state_dict(sd, strict=True)
    rows: Dict[str, Any] = {}
    for pre, label in (("obs__", "violator"), ("scan_obs__", "scan32")):
        obs = {k[len(pre):]: z[k] for k in z.files if k.startswith(pre)}
        acts = z["action" if pre == "obs__" else "scan_action"]
        stored = z["logp_stored" if pre == "obs__" else "scan_logp_stored"]
        recomp = z["logp_recomputed" if pre == "obs__" else "scan_logp_recomputed"]
        masks = obs["action_mask"]
        c32 = _forward(model, obs, acts, masks)
        jit = _jitter_spread(model, obs, acts, masks)
        rows[label] = {
            "n": int(acts.size),
            "stored_T2_cuda": stored.tolist(), "recomputed_learner_cuda": recomp.tolist(),
            "cpu_fp32": c32.tolist(),
            "abs_stored_minus_recomputed": np.abs(stored - recomp).tolist(),
            "abs_stored_minus_cpu32": np.abs(stored - c32).tolist(),
            "abs_recomputed_minus_cpu32": np.abs(recomp - c32).tolist(),
            "jitter_spread": jit.tolist(),
        }
    top = sorted(rec["rows"], key=lambda r: -r["abs_dlogp"])[0]
    return {"n_updates": u, "num_timesteps": rec["num_timesteps"], "where": rec["where"],
            "route_policy_version": rec["route"]["policy_version"], "row_version": top["version"],
            "max": next(c["value"] for c in rec["conditions"] if c["statistic"] == "max"),
            "over_bar_by": next(c["value"] for c in rec["conditions"] if c["statistic"] == "max") / BAR - 1.0,
            "p99": rec["p99"], "rows_judged": rec["rows_judged"], "rows_excluded": rec["rows_excluded"],
            "second_largest_judged": sorted([r["abs_dlogp"] for r in rec["rows"] if not r["excluded"]])[-2],
            "violator": {k: top[k] for k in ("index", "abs_dlogp", "action", "tie_margin", "tie_site",
                                              "p_action_stored", "margin_stored", "env", "dec_n", "slot")},
            "scan": rec.get("scan"), "rows": rows}


def check_pin() -> None:
    """The replay must run the pin's forward: refuse a checkout whose src/ (tests aside) differs from the pin's."""
    out = subprocess.run(["git", "diff", "--name-only", PIN, "--", "src"], capture_output=True, text=True,
                         check=True, cwd=HERE).stdout.split()
    code = [p for p in out if not p.endswith("_test.py")]
    if code:
        raise SystemExit(f"src/ differs from the pin {PIN[:9]} outside tests: {code}")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "x14_k9b_diag.json"))
    ap.add_argument("--skip-replay", action="store_true")
    a = ap.parse_args(argv)
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise SystemExit('CPU only: run with CUDA_VISIBLE_DEVICES=""')
    import torch as th

    th.set_num_threads(3)
    res: Dict[str, Any] = {"tb": tb_read()}
    if not a.skip_replay:
        check_pin()
        res["replay"] = [replay(u) for u in (38730, 38750)]
    Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["tb"]["compare"], indent=1))
    for r in res.get("replay", []):
        v, sc = r["rows"]["violator"], r["rows"]["scan32"]
        print(f"u{r['n_updates']}: max {r['max']:.4g} (+{100 * r['over_bar_by']:.1f}% over), rows of v{r['row_version']} "
              f"(learner at v{r['route_policy_version']}); violator |T2-cpu| {v['abs_stored_minus_cpu32'][0]:.3g} "
              f"|learner-cpu| {v['abs_recomputed_minus_cpu32'][0]:.3g} jitter {v['jitter_spread'][0]:.3g}; scan32 "
              f"median |T2-cpu| {np.median(sc['abs_stored_minus_cpu32']):.3g} max {max(sc['abs_stored_minus_cpu32']):.3g}, "
              f"median jitter {np.median(sc['jitter_spread']):.3g} max {max(sc['jitter_spread']):.3g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
