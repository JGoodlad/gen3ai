"""Summarise replay.json into the decisive numbers the README quotes (writes analysis.json).

    python3 analyse.py replay.json analysis.json
"""
from __future__ import annotations

import json
import sys

import numpy as np


def main() -> None:
    d = json.load(open(sys.argv[1]))
    rows = d["rows"]
    ref = np.array([r["ref64"] for r in rows])
    paths = {
        "stored_T2_gpu": np.array([r["stored"] for r in rows]),
        "learner_gpu": np.array([r["gpu_learner"] for r in rows]),
        "learner32_cpu_b32": np.array([r["learn32_cpu_b32"] for r in rows]),
        "learner32_cpu_b1": np.array([r["learn32_cpu_b1"] for r in rows]),
        "decision32_cpu_b8": np.array([r["dec32_cpu"]["8"] for r in rows]),
        "decision32_cpu_b64": np.array([r["dec32_cpu"]["64"] for r in rows]),
        "decision32_cpu_b256": np.array([r["dec32_cpu"]["256"] for r in rows]),
    }
    if rows[0].get("compiled32_cpu_b256") is not None:
        paths["compiled32_cpu_b256"] = np.array([r["compiled32_cpu_b256"] for r in rows])
    err = {k: v - ref for k, v in paths.items()}
    pert_std = np.array([r["pert_std"] for r in rows])
    gate_d = np.abs(paths["stored_T2_gpu"] - paths["learner_gpu"])
    # the per-row fp32 error scale: the rms over every fp32 path of (path - exact)
    fp32_scale = np.sqrt(np.mean(np.stack([e ** 2 for e in err.values()]), 0))
    out = {
        "row0_violator": {
            "error_vs_fp64_by_path": {k: float(v[0]) for k, v in err.items()},
            "gate_abs_d": float(gate_d[0]),
            "fp32_error_scale_rms": float(fp32_scale[0]),
            "pert_std": float(pert_std[0]), "pert_maxdev": rows[0]["pert_maxdev"],
            "raw_logits_legal_fp64": rows[0]["raw_logits_legal"],
            "raw_legal_err32": rows[0]["raw_legal_err32"],
            "p_action": rows[0]["p_action"], "n_legal": rows[0]["n_legal"],
            "stale_d": rows[0]["stale_d"],
        },
        "others_1_31": {
            "median_abs_error_vs_fp64_by_path": {k: float(np.median(np.abs(v[1:]))) for k, v in err.items()},
            "max_abs_error_vs_fp64_by_path": {k: float(np.max(np.abs(v[1:]))) for k, v in err.items()},
            "median_fp32_error_scale_rms": float(np.median(fp32_scale[1:])),
            "median_pert_std": float(np.median(pert_std[1:])),
            "median_gate_abs_d": float(np.median(gate_d[1:])),
            "min_abs_stale_d": float(np.min(np.abs([r["stale_d"] for r in rows]))),
            "median_abs_stale_d": float(np.median(np.abs([r["stale_d"] for r in rows]))),
        },
        "rank_of_row0": {
            "fp32_error_scale": int((fp32_scale > fp32_scale[0]).sum()) + 1,
            "pert_std": int((pert_std > pert_std[0]).sum()) + 1,
            "abs_learner_gpu_error": int((np.abs(err["learner_gpu"]) > abs(err["learner_gpu"][0])).sum()) + 1,
        },
        "spearman_gate_d_vs_fp32_scale": float(np.corrcoef(np.argsort(np.argsort(gate_d)),
                                                           np.argsort(np.argsort(fp32_scale)))[0, 1]),
        "spearman_gate_d_vs_pert_std": float(np.corrcoef(np.argsort(np.argsort(gate_d)),
                                                         np.argsort(np.argsort(pert_std)))[0, 1]),
        "gate_d_over_pert_std": {"row0": float(gate_d[0] / pert_std[0]),
                                 "others_median": float(np.median(gate_d[1:] / pert_std[1:])),
                                 "others_max": float(np.max(gate_d[1:] / pert_std[1:]))},
        "checks": {k: d[k] for k in ("weights_dump_equals_final_model_exception", "dump_rows_align_with_json",
                                     "dump_masks_match_json", "stored_dist_legal_set_equals_mask")},
        "fp64_two_paths_max_abs_diff": float(max(abs(r["dec64"] - r["ref64"]) for r in rows)),
        "compiled_error": d.get("compiled_error"),
        "k_perturb": d.get("k_perturb"),
    }
    merr = d.get("module_relerr_fp32_vs_fp64", {})
    ratios = []
    for name, v in merr.items():
        v = np.array(v, dtype=float)
        if v.size == len(rows) and np.isfinite(v).all() and np.median(v[1:]) > 0:
            ratios.append((float(v[0] / np.median(v[1:])), name, float(v[0]), float(np.median(v[1:]))))
    ratios.sort(reverse=True)
    out["module_relerr_row0_over_others_top"] = [
        {"module": n, "ratio": r, "row0": a, "others_median": b} for r, n, a, b in ratios[:15]]
    out["module_relerr_selected"] = {n: {"row0": a, "others_median": b} for r, n, a, b in ratios
                                     if n in ("features_extractor", "mlp_extractor", "pointer_head",
                                              "features_extractor.team_transformer", "features_extractor.projection",
                                              "features_extractor.hypothesis_builder.delta_out")}
    with open(sys.argv[2], "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
