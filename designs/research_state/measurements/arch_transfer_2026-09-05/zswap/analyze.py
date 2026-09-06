"""H7 z-swap readout. Pure post-processing of era_zswap.py's JSON -- no models, no battles.

Runs in EITHER tree (numpy only). Team is the cluster; cluster bootstrap over teams, 20000
resamples, seeded 20260905 (the same seed content_locality used).

  python analyze.py <zswap_nN.json> [out_analysis.json]
"""
import json, sys
import numpy as np

B = 20000
SEED = 20260905
# Registered rails (PREREGISTRATION.md sec.3)
RAIL_HI, RAIL_LO = 0.50, 0.20
# content_locality's matched-noise floor, the CONSERVATIVE (larger) member of each pair.
FLOOR_TAUGHT, FLOOR_UNTAUGHT = 0.0535, 0.0664
ALL_CONDS = ["a", "b", "c1", "c2", "d0", "dmu", "zsens_T", "zsens0_T", "zsensmu_T"]
# f = 1 - cond/a, i.e. the fraction of the baseline divergence the intervention removes.
FRACS = [("f_b", "b", "PRE-REGISTERED PRIMARY"),
         ("f_dmu", "dmu", "secondary descriptive"),
         ("f_d0", "d0", "secondary descriptive (OFF-MANIFOLD: z=0 is never a real code)")]


def ci(x, idx):
    b = np.asarray(x)[idx].mean(axis=1)
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def frac_ci(a, b, idx):
    """f = 1 - mean(b)/mean(a), paired -- the SAME cells resampled for both."""
    a, b = np.asarray(a), np.asarray(b)
    f = 1.0 - b[idx].mean(axis=1) / a[idx].mean(axis=1)
    return float(np.percentile(f, 2.5)), float(np.percentile(f, 97.5))


def verdict_f(lo, hi):
    if lo > RAIL_HI:
        return "z CARRIES THE SPECIALISATION (CI clears the 0.50 rail)"
    if hi < RAIL_LO:
        return "SHARED WEIGHTS (CI below the 0.20 rail)"
    return "PARTIAL / no verdict (CI spans a rail)"


def main(path, out_path=None):
    R = json.load(open(path))
    M = R["_meta"]
    teams, n_unt = M["teams"], M["n_untaught"]
    taught_of, shared = M["taught_of"], set(M["shared_teams"])
    IDX = {t["sha10"]: t["i"] for t in teams}
    names = list(taught_of)
    kl = {t: {c: np.array(v) for c, v in d.items()} for t, d in R["per_team_kl"].items()}
    floors = [k for k in kl if k.startswith("FLOOR_")]
    CONDS = [c for c in ALL_CONDS if c in kl[names[0]]]

    rng = np.random.default_rng(SEED)
    unt_i = list(range(n_unt))
    bsU = rng.integers(0, n_unt, (B, n_unt))

    print(f"\n=== H7 Z-SWAP — v8 era, per_team={M['per_team']}, {M['n_states']} states, "
          f"{len(teams)} teams ({n_unt} untaught + {len(teams)-n_unt} taught) ===")
    shim_ok = all(v["disarmed_max_abs_delta"] == 0 and v["own_z_max_abs_delta"] == 0
                  for v in M["acid_shim"].values())
    print(f"    ACID shim faithful on all {len(M['acid_shim'])} models (bit-identical): {shim_ok}")
    print(f"    ACID distinct baseline KL vectors: {M['acid_all_distinct']}")

    out = {"_meta": {"source": path, "per_team": M["per_team"], "n_states": M["n_states"],
                     "bootstrap": B, "seed": SEED, "conditions_present": CONDS,
                     "acid_shim_faithful": shim_ok,
                     "rails": {"z_carries": RAIL_HI, "shared_weights": RAIL_LO},
                     "floor": {"taught": FLOOR_TAUGHT, "untaught": FLOOR_UNTAUGHT}}}

    # -------------------------------------------------------------- does FiLM do anything at all?
    if "film_magnitude" in R:
        print("\n--- FiLM MAGNITUDE: how much does the code modulate the head features? ---")
        print("    ||h*dg + db|| / ||h||, averaged over states (h = pre-FiLM pi_pre / vf_pre)")
        for t, d in R["film_magnitude"].items():
            print(f"  {t:16s} pi {d['pi']['mean_relative_modulation']*100:6.2f}%   "
                  f"vf {d['vf']['mean_relative_modulation']*100:6.2f}%")
        out["film_magnitude"] = R["film_magnitude"]

    # ------------------------------------------------------------------ z geometry + param split
    print("\n--- Z GEOMETRY (does a swap have anything to swap?) ---")
    zg = R["z_geometry"]
    print(f"  parent           |z| {zg['parent']['mean_norm']:.3f}  centred RMS "
          f"{zg['parent']['centred_rms']:.3f}  top-dir share of centred energy "
          f"{zg['parent']['top_dir_energy_share']*100:.1f}%")
    for t in [*floors, *names]:
        g = zg[t]
        print(f"  {t:16s} |z| {g['mean_norm']:.3f}  centred RMS {g['centred_rms']:.3f}  "
              f"top-dir {g['top_dir_energy_share']*100:.1f}%  RMS dist to z_P "
              f"{g['rms_dist_to_ref']:.3f} (= {g['rel_dist_to_ref']*100:.1f}% of |z|)")
    out["z_geometry"] = zg

    print("\n--- PARAMETER-MASS SPLIT: how much of |theta_T - theta_P| lives in the z path? ---")
    print("    behavioural z path = z_encoder + film generators (recon_head EXCLUDED: aux-only,")
    print("    never fed forward).  enrichment = displacement share / parameter share (1.0 = fair)")
    ps = R["param_split"]
    for t in [*floors, *names]:
        p = ps[t]
        print(f"  {t:16s} params {p['param_frac_z']*100:6.3f}%   ||dtheta||^2 share "
              f"{p['disp_frac_z_sq']*100:6.3f}%   L1 share {p['disp_frac_z_l1']*100:6.3f}%   "
              f"enrichment {p['enrichment_sq']:.3f}x   ||dtheta|| {p['l2_norm_total']:.3f}")
    if "groups" in ps[names[0]]:
        print("\n    per-group (param share -> ||dtheta||^2 share, enrichment):")
        gnames = ["z_encoder", "film_generators", "recon_head", "shared_trunk_and_heads"]
        print(f"      {'model':16s} " + "  ".join(f"{g[:15]:>22s}" for g in gnames))
        for t in [*floors, *names]:
            cells = []
            for g in gnames:
                e = ps[t]["groups"][g]
                cells.append(f"{e['param_frac']*100:5.2f}%->{e['disp_frac_sq']*100:5.2f}% "
                             f"({e['enrichment_sq']:4.2f}x)")
            print(f"      {t:16s} " + "  ".join(f"{c:>22s}" for c in cells))
    out["param_split"] = ps

    # -------------------------------------------------------------------------- slices per teacher
    print("\n--- PER-TEACHER KL BY CONDITION AND SLICE ---")
    print("    a=T[z_T]||P[z_P]  b=T[z_P]||P[z_P]  c1=P[z_T]||P[z_P]  c2=T[z_T]||P[z_T]")
    print("    d0=T[0]||P[0]  dmu=T[zbar]||P[zbar]")
    print("    zsens_T=T[z_P]||T[z_T]   zsens0_T=T[0]||T[z_T]   zsensmu_T=T[zbar]||T[z_T]")
    per_teacher = {}
    for nm in names:
        own_i = [IDX[s] for s in taught_of[nm]]
        sib_i = sorted({IDX[s] for o in names if o != nm for s in taught_of[o]} - set(own_i))
        slices = {"taught_own": own_i, "taught_sibling": sib_i, "untaught": unt_i}
        row = {}
        for sl, ii in slices.items():
            bs = rng.integers(0, len(ii), (B, len(ii)))
            d = {c: {"mean": float(kl[nm][c][ii].mean()),
                     "ci95": list(ci(kl[nm][c][ii], bs))} for c in CONDS}
            for key, cond, _lab in FRACS:
                if cond in CONDS:
                    fa, fb = kl[nm]["a"][ii], kl[nm][cond][ii]
                    d[key] = {"point": float(1 - fb.mean() / fa.mean()),
                              "ci95": list(frac_ci(fa, fb, bs))}
            d["n_teams"] = len(ii)
            row[sl] = d
        per_teacher[nm] = row
        print(f"\n  {nm}  (own {len(own_i)} teams, sibling {len(sib_i)}, untaught {n_unt})")
        for sl in slices:
            d = row[sl]
            print(f"    {sl:15s} " + "  ".join(f"{c} {d[c]['mean']:.4f}" for c in CONDS))
            print(f"    {'':15s} f_b {d['f_b']['point']:+.4f} "
                  f"CI [{d['f_b']['ci95'][0]:+.4f},{d['f_b']['ci95'][1]:+.4f}]")
    out["per_teacher"] = per_teacher

    # ------------------------------------------------------------------------------ POOLED (own)
    cells = [(nm, IDX[s]) for nm in names for s in taught_of[nm]]
    own = {c: np.array([kl[nm][c][i] for nm, i in cells]) for c in CONDS}
    bsO = rng.integers(0, len(cells), (B, len(cells)))
    untp = {c: np.array([np.mean([kl[nm][c][i] for nm in names]) for i in unt_i]) for c in CONDS}

    print(f"\n=== POOLED ({len(cells)} teacher-own-team cells; untaught = {n_unt} teams "
          f"averaged over the 3 teachers) ===")
    pooled = {}
    for sl, vals, bs in (("taught_own", own, bsO), ("untaught", untp, bsU)):
        fl = FLOOR_TAUGHT if sl == "taught_own" else FLOOR_UNTAUGHT
        d = {c: {"mean": float(vals[c].mean()), "ci95": list(ci(vals[c], bs))} for c in CONDS}
        print(f"\n  {sl}  (n={len(vals['a'])}; floor on this slice {fl:.4f})")
        for c in CONDS:
            print(f"    {c:10s} {d[c]['mean']:.4f}  CI [{d[c]['ci95'][0]:.4f},{d[c]['ci95'][1]:.4f}]"
                  f"{'   <- WITHIN FLOOR' if d[c]['mean'] < fl else ''}")
        for key, cond, lab in FRACS:
            if cond not in CONDS:
                continue
            lo, hi = frac_ci(vals["a"], vals[cond], bs)
            pt = float(1 - vals[cond].mean() / vals["a"].mean())
            rem = vals["a"] - vals[cond]
            rlo, rhi = ci(rem, bs)
            d[key] = {"point": pt, "ci95": [lo, hi], "label": lab,
                      "kl_removed": {"mean": float(rem.mean()), "ci95": [rlo, rhi],
                                     "floor": fl, "within_floor": bool(abs(rem.mean()) < fl)}}
            if key == "f_b":
                d[key]["verdict"] = verdict_f(lo, hi)
            print(f"    {key:10s} {pt:+.4f}  CI [{lo:+.4f},{hi:+.4f}]   ({lab})")
            print(f"    {'':10s} KL removed {rem.mean():+.4f} CI [{rlo:+.4f},{rhi:+.4f}]  -> "
                  f"{'WITHIN FLOOR' if abs(rem.mean()) < fl else 'above floor'}")
        if "verdict" in d.get("f_b", {}):
            print(f"    ==> PRIMARY VERDICT: {d['f_b']['verdict']}")
        d["n_cells"] = len(vals["a"])
        pooled[sl] = d
    out["pooled"] = pooled

    fo = 1 - own["b"][bsO].mean(1) / own["a"][bsO].mean(1)
    fu = 1 - untp["b"][bsU].mean(1) / untp["a"][bsU].mean(1)
    dd = fo - fu
    e = {"point": float(pooled["taught_own"]["f_b"]["point"]
                        - pooled["untaught"]["f_b"]["point"]),
         "ci95": [float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))]}
    e["verdict"] = "NOT DETECTED" if e["ci95"][0] <= 0 <= e["ci95"][1] else "SIGNIFICANT"
    out["f_b_taught_minus_untaught"] = e
    print(f"\n  f_b(taught-own) - f_b(untaught) = {e['point']:+.4f} "
          f"CI [{e['ci95'][0]:+.4f},{e['ci95'][1]:+.4f}]  -> {e['verdict']}")

    c1 = own["c1"]; rem = own["a"] - own["b"]
    dc = c1[bsO].mean(1) - rem[bsO].mean(1)
    out["c1_vs_removed"] = {"c1_mean": float(c1.mean()), "removed_mean": float(rem.mean()),
                            "diff": float(c1.mean() - rem.mean()),
                            "ci95": [float(np.percentile(dc, 2.5)),
                                     float(np.percentile(dc, 97.5))]}
    print(f"  c1 (P[z_T]||P[z_P]) {c1.mean():.4f}  vs  KL removed by the swap {rem.mean():.4f}  "
          f"diff {c1.mean()-rem.mean():+.4f} CI [{out['c1_vs_removed']['ci95'][0]:+.4f},"
          f"{out['c1_vs_removed']['ci95'][1]:+.4f}]")

    # ------------------------------------------------------------------ SIBLING CONTROL R by cond
    print("\n=== SIBLING-CONTROL LOCALITY R = own / siblings, per condition ===")
    print("    (content_locality PRIMARY B, same singly-taught teams. 1.00 = perfectly GLOBAL)")
    used = [(nm, IDX[s]) for nm in names for s in taught_of[nm] if s not in shared]
    bsR = rng.integers(0, len(used), (B, len(used)))
    Rout = {}
    for c in CONDS:
        o = np.array([kl[nm][c][i] for nm, i in used])
        s_ = np.array([np.mean([kl[x][c][i] for x in names if x != nm]) for nm, i in used])
        r = o / s_
        lo, hi = ci(r, bsR)
        dl, dh = ci(o - s_, bsR)
        Rout[c] = {"R": float(r.mean()), "R_ci95": [lo, hi], "own": float(o.mean()),
                   "siblings": float(s_.mean()), "delta": float((o - s_).mean()),
                   "delta_ci95": [dl, dh], "excludes_1": not (lo <= 1.0 <= hi),
                   "n_teams": len(used)}
        print(f"  R_{c:10s} {r.mean():.4f}  CI [{lo:.4f},{hi:.4f}]   own {o.mean():.4f} "
              f"siblings {s_.mean():.4f}  -> "
              f"{'LOCAL' if not (lo <= 1 <= hi) else 'GLOBAL not excluded'}")
    def _ratio(c):
        o = np.array([kl[nm][c][i] for nm, i in used])
        s = np.array([np.mean([kl[x][c][i] for x in names if x != nm]) for nm, i in used])
        return o / s

    ra = _ratio("a")
    print("\n  PAIRED R_a - R_<ablation>  (does the ablation ERASE the locality?)")
    for c in ("b", "dmu", "d0"):
        if c not in CONDS:
            continue
        dR = ra[bsR].mean(1) - _ratio(c)[bsR].mean(1)
        q = {"point": float(ra.mean() - _ratio(c).mean()),
             "ci95": [float(np.percentile(dR, 2.5)), float(np.percentile(dR, 97.5))]}
        q["verdict"] = "NOT DETECTED" if q["ci95"][0] <= 0 <= q["ci95"][1] else "SIGNIFICANT"
        Rout[f"a_minus_{c}_paired"] = q
        print(f"    R_a - R_{c:4s} {q['point']:+.4f} CI [{q['ci95'][0]:+.4f},"
              f"{q['ci95'][1]:+.4f}]  -> {q['verdict']}")
    out["sibling_control_R"] = Rout

    # ------------------------------------------------------------------------------------ floors
    print("\n=== MATCHED-NOISE FLOOR, measured on THIS run's states ===")
    flo = {}
    tt = list(range(n_unt, len(teams)))
    bsT = rng.integers(0, len(tt), (B, len(tt)))
    for f in floors:
        d = {}
        for sl, ii, bs in (("taught", tt, bsT), ("untaught", unt_i, bsU)):
            d[sl] = {c: float(kl[f][c][ii].mean()) for c in CONDS}
            d[sl]["a_ci95"] = list(ci(kl[f]["a"][ii], bs))
        flo[f] = d
        print(f"  {f:16s} taught a {d['taught']['a']:.4f} b {d['taught']['b']:.4f} "
              f"c1 {d['taught']['c1']:.4f}  |  untaught a {d['untaught']['a']:.4f} "
              f"b {d['untaught']['b']:.4f}")
    out["floor_this_run"] = flo

    # ------------------------------------------------------- MECHANISM: own-network z-dependence
    print("\n=== MECHANISM PRECHECK: how much does z move a SINGLE network? (taught-own slice) ===")
    mech = {}
    rows = [("teacher: swap in P's z   T[z_P]||T[z_T]", "zsens_T"),
            ("teacher: zero z          T[0]||T[z_T]", "zsens0_T"),
            ("teacher: mean z          T[zbar]||T[z_T]", "zsensmu_T"),
            ("parent:  swap in T's z   P[z_T]||P[z_P]", "c1")]
    for lab, c in rows:
        if c not in CONDS:
            continue
        v = own[c]
        lo, hi = ci(v, bsO)
        mech[c] = {"mean": float(v.mean()), "ci95": [lo, hi],
                   "within_floor": bool(float(v.mean()) < FLOOR_TAUGHT), "label": lab}
        print(f"  {lab:42s} {v.mean():.4f} CI [{lo:.4f},{hi:.4f}]  -> "
              f"{'WITHIN FLOOR' if v.mean() < FLOOR_TAUGHT else 'above floor'}")
    if "parent_self_kl" in R:
        for k, v in R["parent_self_kl"].items():
            a = np.array(v)[n_unt:]
            mech[k] = {"mean": float(a.mean()), "taught_teams": len(a),
                       "within_floor": bool(float(a.mean()) < FLOOR_TAUGHT)}
            print(f"  parent self:  {k:26s} {a.mean():.4f}  -> "
                  f"{'WITHIN FLOOR' if a.mean() < FLOOR_TAUGHT else 'above floor'}")
    mech["floor_taught"] = FLOOR_TAUGHT
    out["mechanism"] = mech

    if out_path:
        json.dump(out, open(out_path, "w"), indent=1)
        print(f"\n  wrote {out_path}")
    return out


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
