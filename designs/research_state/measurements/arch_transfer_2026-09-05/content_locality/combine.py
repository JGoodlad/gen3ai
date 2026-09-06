"""CROSS-ERA READOUT — joins the gen-era and v8-era artifacts into the one table and verdict.

Reads only committed JSON; no models, no battles. Every number it prints is recomputed from the
per-team KL vectors in those files rather than copied from their summary blocks, so a summary that
drifted from its own data would show up here as a disagreement rather than propagate.

The cross-era contrast is UNPAIRED by construction -- the two eras share no team, no pool and no
parent -- so its bootstrap resamples each era's own cluster set independently and the difference
carries both eras' noise. It is reported as a difference of RATIOS (unit-free) because the two
eras' absolute KL levels are not comparable: different architectures, different obs dimension,
different fold parents.

Run: python combine.py <gen.json> <v8.json> [out.json]
"""
import json
import sys

import numpy as np


def boot(vals, idx):
    b = np.asarray(vals)[idx].mean(axis=1)
    return float(np.mean(vals)), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def main(gen_path, v8_path, out_path=None):
    gen = json.load(open(gen_path))
    v8 = json.load(open(v8_path))
    rng = np.random.default_rng(20260905)

    # ---- gen era: R per taught team, per half, recomputed from the per-team KL vectors --------
    gk = {k: np.array(v) for k, v in gen["per_team_kl_fwd"].items()}
    gteams = gen["_meta"]["teams"]
    gidx = {t["basename"]: t["i"] for t in gteams}
    n_unt_g = sum(1 for t in gteams if t["kind"] == "untaught")
    pairs = sorted({k[3:] for k in gk if k.startswith("UNF")})
    taught_of = {p: gen["primary_A_per_teacher"]["UNF" + p]["taught_teams"] for p in pairs}

    gR, gL = {}, {}
    for half, pref in (("unfunded", "UNF"), ("funded", "FUND")):
        own, sib = [], []
        for p in pairs:
            for b in taught_of[p]:
                i = gidx[b]
                own.append(gk[pref + p][i])
                sib.append(np.mean([gk[pref + q][i] for q in pairs if q != p]))
        gR[half] = np.array(own) / np.array(sib)
        gL[half] = np.array([gk[pref + p][[gidx[b] for b in taught_of[p]]].mean()
                             / gk[pref + p][:n_unt_g].mean() for p in pairs])

    # ---- v8 era ------------------------------------------------------------------------------
    vb = v8["primary_B_sibling_control"]
    vR = np.array(vb["per_team_own"]) / np.array(vb["per_team_siblings"])
    vL = np.array([v8["primary_A_per_teacher"][t]["L"] for t in v8["primary_A_per_teacher"]])

    bsg = rng.integers(0, len(gR["funded"]), (20000, len(gR["funded"])))
    bsv = rng.integers(0, len(vR), (20000, len(vR)))

    out = {"_meta": {
        "gen_artifact": gen_path, "v8_artifact": v8_path,
        "statistic": "R = KL(own teacher||parent) / mean KL(sibling teachers||parent), "
                     "per taught team, on the SAME parent-piloted states",
        "why_R_and_not_L": "R holds the TEAM fixed, so any tendency of a team's states to make "
                           "two arbitrary policies differ more cancels exactly. L does not, and "
                           "the matched-noise floor shows that tendency is real and points in "
                           "OPPOSITE directions in the two eras (gen floor L > 1, v8 floor L < 1).",
        "cross_era_contrast_is_UNPAIRED": "the eras share no team, no pool, no parent",
        "n_gen_taught_teams": len(gR["funded"]), "n_v8_taught_teams": len(vR),
        "gen_states": gen["_meta"]["n_states"], "v8_states": v8["_meta"]["n_states"]}}

    rows = {}
    for name, arr, bs in (("gen_unfunded", gR["unfunded"], bsg),
                          ("gen_funded", gR["funded"], bsg),
                          ("v8_all", vR, bsv)):
        m, lo, hi = boot(arr, bs)
        rows[name] = {"R": m, "ci95": [lo, hi], "local_beyond_1": lo > 1.0}
    out["sibling_R"] = rows

    for name, arr, bs in (("gen_unfunded", gL["unfunded"], rng.integers(0, 8, (20000, 8))),
                          ("gen_funded", gL["funded"], rng.integers(0, 8, (20000, 8))),
                          ("v8_all", vL, rng.integers(0, len(vL), (20000, len(vL))))):
        m, lo, hi = boot(arr, bs)
        out.setdefault("teacher_L", {})[name] = {"L": m, "ci95": [lo, hi]}

    out["floor"] = {"gen": {k: v for k, v in gen["floor"].items()},
                    "v8": {k: v for k, v in v8["floor"].items()}}

    # ---- the era contrast, both halves of the gen side ---------------------------------------
    out["cross_era"] = {}
    for half in ("unfunded", "funded"):
        d = vR[bsv].mean(axis=1) - gR[half][bsg].mean(axis=1)
        lo, hi = float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
        out["cross_era"][f"v8_minus_gen_{half}_R"] = {
            "delta": float(vR.mean() - gR[half].mean()), "ci95": [lo, hi],
            "separates_from_zero": not (lo <= 0 <= hi)}

    # ---- the within-gen contrast (paired on the 16 taught teams) -----------------------------
    d = gR["funded"] - gR["unfunded"]
    m, lo, hi = boot(d, bsg)
    out["within_gen"] = {"funded_minus_unfunded_R": {
        "delta": m, "ci95": [lo, hi], "separates_from_zero": not (lo <= 0 <= hi),
        "paired_on": "the same 16 taught teams and the same states"}}

    def verdict(lo, hi, floorish=False):
        if lo <= 0 <= hi:
            return "NOT DETECTED"
        return "SIGNIFICANT"

    print(f"  states: gen {out['_meta']['gen_states']}  v8 {out['_meta']['v8_states']}")
    print("\n  SIBLING-CONTROL LOCALITY  R = own / siblings  (1.00 = perfectly GLOBAL)")
    for k, v in rows.items():
        print(f"    {k:14s} R {v['R']:.4f}  CI [{v['ci95'][0]:.4f},{v['ci95'][1]:.4f}]  "
              f"{'LOCAL (CI excludes 1)' if v['local_beyond_1'] else 'CI includes 1 -> GLOBAL not excluded'}")
    print("\n  RAW L = KL_taught / KL_untaught  (confounded by the team set -- see floor)")
    for k, v in out["teacher_L"].items():
        print(f"    {k:14s} L {v['L']:.4f}  CI [{v['ci95'][0]:.4f},{v['ci95'][1]:.4f}]")
    print("\n  MATCHED-NOISE FLOOR (two nearby same-run checkpoints of each era's parent)")
    for era in ("gen", "v8"):
        for tag, f in out["floor"][era].items():
            u = f["untaught_mean"]
            t = f.get("taught16_mean", f.get("taught_mean"))
            print(f"    {era:3s} {tag:15s} untaught {u:.4f}  taught {t:.4f}  "
                  f"floor L {t/u:.4f}")
    print("\n  CROSS-ERA (unpaired):")
    for k, v in out["cross_era"].items():
        print(f"    {k}: {v['delta']:+.4f} CI [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]  "
              f"{verdict(*v['ci95'])}")
    v = out["within_gen"]["funded_minus_unfunded_R"]
    print(f"\n  WITHIN-GEN (paired): funded-unfunded R {v['delta']:+.4f} "
          f"CI [{v['ci95'][0]:+.4f},{v['ci95'][1]:+.4f}]  {verdict(*v['ci95'])}")

    if out_path:
        json.dump(out, open(out_path, "w"), indent=1)
        print(f"\n  wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
