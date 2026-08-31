"""HISTORY-BLOCK OOD CHECK — how anomalous are the constructed risk-probe observations, and in
WHICH obs blocks?

Third leg of `starmie_ood_control_2026-08-31.md`. Consumes:
  * `starmie_ood_control_obs.npz`          - the constructed observations (starmie_ood_control_probe)
  * `starmie_ood_control_traces_stats.npz` - per-dim mean/std + raw subsamples of REAL eval-trace
                                             observations (starmie_ood_control_traces)
and writes `starmie_ood_control_distance_2026-08-31.json`.

Per obs BLOCK it reports three numbers, each a PERCENTILE against the trace distribution's own
spread so "how far out" is answered in the units the traces themselves supply:
  1. mean |z| and the fraction of live dims past |z| > 3   (marginal anomaly)
  2. the diagonal Mahalanobis distance, and its percentile among trace rows scored the same way
  3. the nearest-neighbour Euclidean distance to the trace sample, and its percentile among
     trace-to-trace nearest-neighbour distances (joint anomaly - the number that notices a
     combination of individually-ordinary values)

Reference sets: the POOLED trace sample, and where the trace file supplies one, the matched
faint-count stratum. A constructed point is compared to both, because "off-distribution" against
all decisions and "off-distribution against decisions at ITS OWN faint count" are different claims.

Run (from the repo root):
    python designs/research_state/measurements/starmie_ood_control_distance.py
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
CPU-only, seconds.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

OUT_DIR = Path(__file__).resolve().parent
OBS_NPZ = OUT_DIR / "starmie_ood_control_obs.npz"
TRACE_NPZ = OUT_DIR / "starmie_ood_control_traces_stats.npz"
OUT_JSON = OUT_DIR / "starmie_ood_control_distance_2026-08-31.json"

# Verified against agents.observation.constants / Gen3ObservationEncoder.get_layout().
BLOCKS = {
    "our_team": (0, 732),
    "opp_team": (732, 1464),
    "active_ctx": (1464, 1580),
    "global_env": (1580, 1600),
    "board": (1600, 1617),
    "pair_history": (1617, 1797),
    "event_window": (1797, 2501),
}
SLOT = 122            # POKEMON_FULL_DIM
LAST_ACTION = (113, 119)   # POKEMON_LAST_ACTION_OFFSET .. +6, per mon slot
RECENCY = (109, 112)       # POKEMON_RECENCY_OFFSET .. +3, per mon slot


def _derived_index_blocks() -> dict:
    """The per-entity HISTORY facts that live INSIDE the team blocks (Tier H-A1 last-action and
    the E9 recency triple) - gathered so they can be scored as history rather than as roster."""
    last, rec = [], []
    for slot in range(12):
        base = slot * SLOT
        last.extend(range(base + LAST_ACTION[0], base + LAST_ACTION[1]))
        rec.extend(range(base + RECENCY[0], base + RECENCY[1]))
    return {"per_mon_last_action": np.array(last), "per_mon_recency": np.array(rec)}


def _index(name: str) -> np.ndarray:
    lo, hi = BLOCKS[name]
    return np.arange(lo, hi)


def _score(point: np.ndarray, ref: np.ndarray, idx: np.ndarray) -> dict:
    """`point` (D,), `ref` (N, D) real trace observations, `idx` the block's dimensions."""
    p = point[idx].astype(np.float64)
    r = ref[:, idx].astype(np.float64)
    mu, sd = r.mean(0), r.std(0)
    live = sd > 1e-8                      # dims the traces actually vary in
    n_live = int(live.sum())
    out = {"n_dims": int(idx.size), "n_live_dims": n_live, "n_ref": int(r.shape[0])}

    # constant-in-traces dims the constructed point nonetheless differs on: unrepresentable values
    const = ~live
    if const.any():
        diff = np.abs(p[const] - mu[const]) > 1e-6
        out["n_const_dims_violated"] = int(diff.sum())
        out["frac_const_dims_violated"] = round(float(diff.mean()), 4)
    else:
        out["n_const_dims_violated"] = 0
        out["frac_const_dims_violated"] = 0.0

    if n_live == 0:
        return out

    z = (p[live] - mu[live]) / sd[live]
    zr = (r[:, live] - mu[live]) / sd[live]
    out["mean_abs_z"] = round(float(np.abs(z).mean()), 4)
    out["max_abs_z"] = round(float(np.abs(z).max()), 3)
    out["frac_abs_z_gt3"] = round(float((np.abs(z) > 3).mean()), 4)
    out["ref_mean_abs_z"] = round(float(np.abs(zr).mean()), 4)

    d_point = float(np.sqrt((z ** 2).mean()))
    d_ref = np.sqrt((zr ** 2).mean(1))
    out["maha_diag_rms_z"] = round(d_point, 4)
    out["maha_ref_median"] = round(float(np.median(d_ref)), 4)
    out["maha_ref_p99"] = round(float(np.percentile(d_ref, 99)), 4)
    out["maha_percentile"] = round(float((d_ref < d_point).mean() * 100), 3)

    # nearest neighbour in RAW block units (not z-scaled): the joint-anomaly reading
    d_to_ref = np.sqrt(((r - p) ** 2).sum(1))
    nn_point = float(d_to_ref.min())
    m = min(r.shape[0], 1500)
    sub = r[:m]
    dm = np.sqrt(np.maximum(
        (sub ** 2).sum(1)[:, None] + (sub ** 2).sum(1)[None, :] - 2 * sub @ sub.T, 0.0))
    np.fill_diagonal(dm, np.inf)
    nn_ref = dm.min(1)
    out["nn_dist"] = round(nn_point, 4)
    out["nn_ref_median"] = round(float(np.median(nn_ref)), 4)
    out["nn_ref_max"] = round(float(nn_ref.max()), 4)
    out["nn_percentile"] = round(float((nn_ref < nn_point).mean() * 100), 3)
    out["nn_ratio_to_ref_median"] = round(nn_point / max(1e-9, float(np.median(nn_ref))), 3)
    return out


def main() -> None:
    assert OBS_NPZ.exists(), f"missing {OBS_NPZ} - run starmie_ood_control_probe.py first"
    assert TRACE_NPZ.exists(), f"missing {TRACE_NPZ} - run starmie_ood_control_traces.py first"

    obs = {k: v for k, v in np.load(OBS_NPZ).items()}
    tr = np.load(TRACE_NPZ)
    pooled = tr["sample"]
    refs = {"pooled_traces": pooled}
    if "sample_lowfaint" in tr.files and tr["sample_lowfaint"].shape[0] >= 100:
        refs["traces_F2_F3_diagonal"] = tr["sample_lowfaint"]
    if "sample_our_faints" in tr.files:
        of, pf = tr["sample_our_faints"], tr["sample_opp_faints"]
        hi = pooled[(of >= 4) & (pf >= 4)]
        if hi.shape[0] >= 100:
            refs["traces_faints_ge4_both"] = hi

    idx_blocks = {name: _index(name) for name in BLOCKS}
    idx_blocks.update(_derived_index_blocks())
    idx_blocks["FULL_OBS"] = np.arange(pooled.shape[1])

    # which constructed points to score: the base cell of every arm + the v2 prelude
    points = {k: v for k, v in obs.items()
              if k.startswith("base_F") or k == "v2_prelude_F5"}
    if not points:
        points = dict(list(obs.items())[:4])

    results = {
        "meta": {
            "date": "2026-08-31",
            "obs_npz": OBS_NPZ.name, "trace_npz": TRACE_NPZ.name,
            "obs_dim": int(pooled.shape[1]),
            "reference_sets": {k: int(v.shape[0]) for k, v in refs.items()},
            "points_scored": sorted(points),
            "git_head": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                       text=True).stdout.strip(),
            "reading": "maha_percentile / nn_percentile are the fraction of REAL trace decisions "
                       "that sit CLOSER to the trace centre / to their own nearest neighbour than "
                       "the constructed point does. 100.0 means no trace decision in the reference "
                       "set is as far out as the constructed one on that block.",
        },
        "scores": {},
    }

    for pname, point in sorted(points.items()):
        results["scores"][pname] = {}
        for rname, ref in refs.items():
            results["scores"][pname][rname] = {
                bname: _score(point, ref, idx) for bname, idx in idx_blocks.items()}
        print(f"-- {pname}")
        for bname in idx_blocks:
            s = results["scores"][pname]["pooled_traces"][bname]
            print(f"   {bname:22s} mean|z| {s.get('mean_abs_z')!s:8s} "
                  f"maha_pct {s.get('maha_percentile')!s:7s} nn_pct {s.get('nn_percentile')!s:7s} "
                  f"nn/med {s.get('nn_ratio_to_ref_median')}")

    OUT_JSON.write_text(json.dumps(results, indent=1))
    print(f"[saved {OUT_JSON}]")


if __name__ == "__main__":
    main()
