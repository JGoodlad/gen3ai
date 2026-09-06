"""CROSS-ERA READOUT v2 — joins the corrected gen-era and v8-era artifacts.

Reads only committed JSON; no models, no battles. Every number is RECOMPUTED from the per-team
KL vectors rather than copied from a summary block, so a summary that drifted from its own data
shows up here as a disagreement.

Two things differ from content_locality/combine.py:

  * the GEN side is read under BOTH references — REF-A (the fold parent `ai_v9_59_R2ACTION_0827`,
    what the fold sees) and REF-B (`ai_v9_29_rev1_0823/final_model.zip`, the exploiters' true fork
    origin). The v8 side has one column because its parent IS its teachers' origin.
  * every CI resamples over the clusters of its own array (`boot.Boot`), which cannot
    under-sample the way v1's named index matrices could.

The cross-era contrast is UNPAIRED by construction — the eras share no team, no pool and no
parent — so each era's clusters are resampled independently and the difference carries both eras'
noise. It is a difference of RATIOS (unit-free) because the two eras' absolute KL levels are not
comparable.

Run: python combine_v2.py <gen_v2.json> <v8_v2.json> [out.json]
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from boot import Boot   # noqa: E402


def _gen_arrays(gen, ref):
    """R (per taught team) and L (per teacher), per half, from the gen artifact's KL vectors."""
    gk = {k: np.array(v) for k, v in gen[f"per_team_kl_fwd_ref{ref}"].items()}
    teams = gen["_meta"]["teams"]
    gidx = {t["basename"]: t["i"] for t in teams}
    n_unt = sum(1 for t in teams if t["kind"] == "untaught")
    pairs = sorted({k[3:] for k in gk if k.startswith("UNF")})
    taught_of = {p: gen[f"primary_A_per_teacher_ref{ref}"]["UNF" + p]["taught_teams"] for p in pairs}
    R, L = {}, {}
    for half, pref in (("unfunded", "UNF"), ("funded", "FUND")):
        own, sib = [], []
        for p in pairs:
            for b in taught_of[p]:
                i = gidx[b]
                own.append(gk[pref + p][i])
                sib.append(np.mean([gk[pref + q][i] for q in pairs if q != p]))
        R[half] = np.array(own) / np.array(sib)
        L[half] = np.array([gk[pref + p][[gidx[b] for b in taught_of[p]]].mean()
                            / gk[pref + p][:n_unt].mean() for p in pairs])
    return R, L


def main(gen_path, v8_path, out_path=None):
    gen = json.load(open(gen_path))
    v8 = json.load(open(v8_path))
    boot = Boot()

    vb = v8["primary_B_sibling_control"]
    vR = np.array(vb["per_team_own"]) / np.array(vb["per_team_siblings"])
    vL = np.array([v8["primary_A_per_teacher"][t]["L"] for t in v8["primary_A_per_teacher"]])

    out = {"_meta": {
        "gen_artifact": os.path.basename(gen_path), "v8_artifact": os.path.basename(v8_path),
        "statistic": "R = KL(own teacher||reference) / mean KL(sibling teachers||reference), "
                     "per taught team, on the SAME parent-piloted states",
        "references": {
            "gen_REF_A": gen["_meta"]["ref_A_parent"],
            "gen_REF_B": gen["_meta"]["ref_B_origin"],
            "v8": v8["_meta"]["parent_and_origin"] + "  (parent == origin; one column)"},
        "gen_reference_gap_kl_parent_given_origin":
            gen["_meta"]["ref_gap_kl_parent_given_origin"],
        "resolver": "agents.training.fixed_opponent_pool._resolve_zip_and_config(run_dir, None)",
        "cross_era_contrast_is_UNPAIRED": "the eras share no team, no pool, no parent",
        "n_gen_taught_teams": 16, "n_v8_taught_teams": int(len(vR)),
        "gen_states": gen["_meta"]["n_states"], "v8_states": v8["_meta"]["n_states"]}}

    print(f"  states: gen {out['_meta']['gen_states']}  v8 {out['_meta']['v8_states']}")
    print(f"  gen reference gap KL(parent||origin) "
          f"{out['_meta']['gen_reference_gap_kl_parent_given_origin']:.4f}")

    m, lo, hi = boot.mean_ci(vR)
    out["v8"] = {"R": m, "ci95": [lo, hi], "local_beyond_1": lo > 1.0,
                 "L": float(vL.mean()), "L_ci95": list(boot.ci(vL)),
                 "kl_own_mean": vb["kl_own_mean"], "kl_siblings_mean": vb["kl_siblings_mean"]}

    for ref in ("A", "B"):
        R, L = _gen_arrays(gen, ref)
        blk = {}
        for half in ("unfunded", "funded"):
            m, lo, hi = boot.mean_ci(R[half])
            lm, llo, lhi = boot.mean_ci(L[half])
            g = gen[f"primary_A_groups_ref{ref}"][half]
            blk[half] = {"R": m, "R_ci95": [lo, hi], "local_beyond_1": lo > 1.0,
                         "L": lm, "L_ci95": [llo, lhi],
                         "kl_taught_mean": g["kl_taught_mean"],
                         "kl_taught_ci95": g["kl_taught_ci95"],
                         "kl_untaught_mean": g["kl_untaught_mean"],
                         "kl_untaught_ci95": g["kl_untaught_ci95"]}
        # within-gen, PAIRED on the same 16 taught teams and the same states
        d = R["funded"] - R["unfunded"]
        m, lo, hi = boot.mean_ci(d)
        blk["within_gen_funded_minus_unfunded_R"] = {
            "delta": m, "ci95": [lo, hi], "separates_from_zero": not (lo <= 0 <= hi),
            "paired_on": "the same 16 taught teams and the same states"}
        # cross-era, UNPAIRED
        for half in ("unfunded", "funded"):
            dd = boot.dist(vR) - boot.dist(R[half])
            lo2, hi2 = float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))
            blk[f"cross_era_v8_minus_gen_{half}_R"] = {
                "delta": float(vR.mean() - R[half].mean()), "ci95": [lo2, hi2],
                "separates_from_zero": not (lo2 <= 0 <= hi2)}
        out[f"ref{ref}"] = blk

    out["floor"] = {"gen": gen["floor"], "v8": v8["floor"]}

    def verdict(lo, hi):
        return "NOT DETECTED" if lo <= 0 <= hi else "SIGNIFICANT"

    print("\n  SIBLING-CONTROL LOCALITY  R = own / siblings  (1.00 = perfectly GLOBAL)")
    v = out["v8"]
    print(f"    v8_all                      R {v['R']:.4f}  CI [{v['ci95'][0]:.4f},"
          f"{v['ci95'][1]:.4f}]  "
          f"{'LOCAL (CI excludes 1)' if v['local_beyond_1'] else 'CI includes 1 -> GLOBAL not excluded'}")
    for ref, label in (("A", "REF-A fold parent"), ("B", "REF-B true origin")):
        for half in ("unfunded", "funded"):
            b = out[f"ref{ref}"][half]
            print(f"    gen_{half:9s} [{label:17s}] R {b['R']:.4f}  "
                  f"CI [{b['R_ci95'][0]:.4f},{b['R_ci95'][1]:.4f}]  "
                  f"{'LOCAL (CI excludes 1)' if b['local_beyond_1'] else 'CI includes 1 -> GLOBAL not excluded'}")

    print("\n  RAW L = KL_taught / KL_untaught  (confounded by the team set -- see floor)")
    print(f"    v8_all                      L {v['L']:.4f}  CI [{v['L_ci95'][0]:.4f},"
          f"{v['L_ci95'][1]:.4f}]")
    for ref, label in (("A", "REF-A fold parent"), ("B", "REF-B true origin")):
        for half in ("unfunded", "funded"):
            b = out[f"ref{ref}"][half]
            print(f"    gen_{half:9s} [{label:17s}] L {b['L']:.4f}  "
                  f"CI [{b['L_ci95'][0]:.4f},{b['L_ci95'][1]:.4f}]   "
                  f"taught {b['kl_taught_mean']:.4f}  untaught {b['kl_untaught_mean']:.4f}")

    print("\n  MATCHED-NOISE FLOOR")
    for tag, f in out["floor"]["gen"].items():
        t = f.get("taught16_mean", f.get("taught_mean"))
        print(f"    gen {tag:15s} (ref {f.get('reference','A')}) untaught {f['untaught_mean']:.4f}"
              f"  taught {t:.4f}  floor L {t/f['untaught_mean']:.4f}")
    for tag, f in out["floor"]["v8"].items():
        t = f.get("taught_mean")
        print(f"    v8  {tag:15s}            untaught {f['untaught_mean']:.4f}  taught {t:.4f}"
              f"  floor L {t/f['untaught_mean']:.4f}")

    for ref, label in (("A", "REF-A fold parent"), ("B", "REF-B true origin")):
        print(f"\n  CROSS-ERA (unpaired), gen side under {label}:")
        for half in ("unfunded", "funded"):
            c = out[f"ref{ref}"][f"cross_era_v8_minus_gen_{half}_R"]
            print(f"    v8_minus_gen_{half}_R: {c['delta']:+.4f} "
                  f"CI [{c['ci95'][0]:+.4f},{c['ci95'][1]:+.4f}]  {verdict(*c['ci95'])}")
        w = out[f"ref{ref}"]["within_gen_funded_minus_unfunded_R"]
        print(f"    WITHIN-GEN (paired) funded-unfunded R {w['delta']:+.4f} "
              f"CI [{w['ci95'][0]:+.4f},{w['ci95'][1]:+.4f}]  {verdict(*w['ci95'])}")

    if out_path:
        json.dump(out, open(out_path, "w"), indent=1)
        print(f"\n  wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
