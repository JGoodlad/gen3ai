"""Re-derive every number the README quotes, straight from the committed artifacts.

A README is prose and prose drifts; this script is the gate that says it has not. It prints the
artifact value beside the value claimed in the text and FAILS on any disagreement.

  python verify_readme.py
"""
import json, os, sys

D = os.path.dirname(os.path.abspath(__file__))


def J(n):
    return json.load(open(os.path.join(D, n)))


A9, A3 = J("analysis_n9.json"), J("analysis_n3.json")
S9, S3 = J("analysis_finalint_n9.json"), J("analysis_finalint_n3.json")
Z9, Z3 = J("zswap_n9.json"), J("zswap_n3.json")
ZS9 = J("zswap_finalint_n9.json")

rows, bad = [], 0


def chk(label, claimed, actual, tol=5e-4):
    global bad
    ok = abs(claimed - actual) <= tol
    bad += (not ok)
    rows.append((label, claimed, actual, "OK" if ok else "MISMATCH"))


def chk_ci(label, claimed, actual, tol=5e-4):
    chk(label + " lo", claimed[0], actual[0], tol)
    chk(label + " hi", claimed[1], actual[1], tol)


# --- sec 7.1 primary -----------------------------------------------------------------------
p9, p3 = A9["pooled"]["taught_own"], A3["pooled"]["taught_own"]
u9, u3 = A9["pooled"]["untaught"], A3["pooled"]["untaught"]
chk("f_b taught n9", 0.0383, p9["f_b"]["point"])
chk_ci("f_b taught n9 CI", (0.0212, 0.0554), p9["f_b"]["ci95"])
chk("f_b taught n3", 0.0356, p3["f_b"]["point"])
chk_ci("f_b taught n3 CI", (0.0108, 0.0577), p3["f_b"]["ci95"])
chk("f_b untaught n9", 0.0069, u9["f_b"]["point"])
chk_ci("f_b untaught n9 CI", (-0.0032, 0.0188), u9["f_b"]["ci95"])
chk("f_b untaught n3", 0.0083, u3["f_b"]["point"])
chk("KL removed taught n9", 0.0149, p9["f_b"]["kl_removed"]["mean"])
chk_ci("KL removed taught n9 CI", (0.0081, 0.0218), p9["f_b"]["kl_removed"]["ci95"])
assert p9["f_b"]["kl_removed"]["within_floor"], "on-slice removal must be WITHIN FLOOR"
chk("floor taught", 0.0535, p9["f_b"]["kl_removed"]["floor"])
d9 = A9["f_b_taught_minus_untaught"]
chk("f_b taught-untaught n9", 0.0314, d9["point"])
chk_ci("f_b taught-untaught n9 CI", (0.0109, 0.0514), d9["ci95"])

# --- sec 7.2 condition table (n9) ----------------------------------------------------------
for c, t_, u_ in (("a", 0.3893, 0.2329), ("b", 0.3744, 0.2313), ("c1", 0.0149, 0.0031),
                  ("c2", 0.3922, 0.2339), ("dmu", 0.3566, 0.2281), ("d0", 0.2728, 0.1694)):
    chk(f"n9 taught {c}", t_, p9[c]["mean"])
    chk(f"n9 untaught {c}", u_, u9[c]["mean"])
chk("f_dmu taught n9", 0.0840, p9["f_dmu"]["point"])
chk_ci("f_dmu taught n9 CI", (0.0423, 0.1393), p9["f_dmu"]["ci95"])
chk("f_dmu KL removed n9", 0.0327, p9["f_dmu"]["kl_removed"]["mean"])
assert p9["f_dmu"]["kl_removed"]["within_floor"], "f_dmu removal must be WITHIN FLOOR"
chk("f_d0 taught n9", 0.2994, p9["f_d0"]["point"])
chk_ci("f_d0 taught n9 CI", (0.2623, 0.3412), p9["f_d0"]["ci95"])

# --- sec 7.3 c1 vs removed -----------------------------------------------------------------
cv = A9["c1_vs_removed"]
chk("c1 n9", 0.0149, cv["c1_mean"])
chk("removed n9", 0.0149, cv["removed_mean"])
chk("c1-removed diff n9", 0.0000, cv["diff"])
chk_ci("c1-removed CI n9", (-0.0064, 0.0063), cv["ci95"])

# --- sec 7.4 R ------------------------------------------------------------------------------
R9, R3 = A9["sibling_control_R"], A3["sibling_control_R"]
for c, v9, v3 in (("a", 1.8316, 1.7940), ("b", 1.8327, 1.8136), ("c2", 1.8431, 1.8165),
                  ("dmu", 1.6637, 1.6184), ("d0", 1.7246, 1.6690)):
    chk(f"R_{c} n9", v9, R9[c]["R"])
    chk(f"R_{c} n3", v3, R3[c]["R"])
chk_ci("R_a n9 CI", (1.5349, 2.1782), R9["a"]["R_ci95"])
chk_ci("R_a n3 CI", (1.4814, 2.1584), R3["a"]["R_ci95"])
chk_ci("R_b n9 CI", (1.5351, 2.1735), R9["b"]["R_ci95"])
chk("R_a-R_b n9", -0.0011, R9["a_minus_b_paired"]["point"])
chk_ci("R_a-R_b n9 CI", (-0.0379, 0.0318), R9["a_minus_b_paired"]["ci95"])
chk("R_a-R_b n3", -0.0197, R3["a_minus_b_paired"]["point"])
chk("R_a-R_dmu n9", 0.1679, R9["a_minus_dmu_paired"]["point"])
chk_ci("R_a-R_dmu n9 CI", (0.0532, 0.3451), R9["a_minus_dmu_paired"]["ci95"])
chk("R_a-R_dmu n3", 0.1756, R3["a_minus_dmu_paired"]["point"])
chk("R_a-R_d0 n9", 0.1070, R9["a_minus_d0_paired"]["point"])
assert R9["a_minus_b_paired"]["verdict"] == "NOT DETECTED"
assert R9["a_minus_dmu_paired"]["verdict"] == "SIGNIFICANT"
assert R9["a_minus_d0_paired"]["verdict"] == "NOT DETECTED"
# sibling means quoted in the prose
chk("R_a own n9", 0.3969, R9["a"]["own"])
chk("R_a siblings n9", 0.2303, R9["a"]["siblings"])
chk("finalint R_a siblings n9", 0.2959, S9["sibling_control_R"]["a"]["siblings"])

# --- sec 5 teacher-file table ----------------------------------------------------------------
for t, sl, h, s in (("pool10", "taught_own", 0.4722, 0.4719),
                    ("pool10", "untaught", 0.3176, 0.3223),
                    ("semistall3", "taught_own", 0.2156, 0.4478),
                    ("semistall3", "untaught", 0.1036, 0.2190),
                    ("defensive10", "taught_own", 0.3586, 0.3529),
                    ("defensive10", "untaught", 0.2775, 0.2807)):
    chk(f"{t}/{sl} best_model", h, A9["per_teacher"][t][sl]["a"]["mean"])
    chk(f"{t}/{sl} finalint", s, S9["per_teacher"][t][sl]["a"]["mean"])
chk("finalint R_a n9", 1.4498, S9["sibling_control_R"]["a"]["R"])
chk_ci("finalint R_a n9 CI", (1.2728, 1.6722), S9["sibling_control_R"]["a"]["R_ci95"], tol=2e-3)
chk("finalint R_a n3", 1.4601, S3["sibling_control_R"]["a"]["R"])

# --- sec 6 mechanism ------------------------------------------------------------------------
fm9 = A9["film_magnitude"]
for m, pi, vf in (("parent", 174.53, 190.73), ("FLOOR_c277178", 178.84, 194.88),
                  ("FLOOR_c275758", 180.42, 196.13), ("pool10", 231.75, 232.96),
                  ("semistall3", 223.68, 238.59), ("defensive10", 243.48, 246.69)):
    chk(f"film pi {m}", pi, fm9[m]["pi"]["mean_relative_modulation"] * 100, tol=0.02)
    chk(f"film vf {m}", vf, fm9[m]["vf"]["mean_relative_modulation"] * 100, tol=0.02)
zg = A9["z_geometry"]
chk("parent |z| n9", 16.232, zg["parent"]["mean_norm"], tol=2e-3)
chk("parent centred rms n9", 10.659, zg["parent"]["centred_rms"], tol=2e-3)
for m, d, pct in (("FLOOR_c277178", 1.334, 8.2), ("FLOOR_c275758", 1.911, 11.7),
                  ("pool10", 7.928, 46.5), ("semistall3", 6.339, 37.2),
                  ("defensive10", 7.808, 45.0)):
    chk(f"z dist {m}", d, zg[m]["rms_dist_to_ref"], tol=2e-3)
    chk(f"z dist %% {m}", pct, zg[m]["rel_dist_to_ref"] * 100, tol=0.06)
mech = A9["mechanism"]
chk("zsens_T", 0.0237, mech["zsens_T"]["mean"])
chk_ci("zsens_T CI", (0.0189, 0.0286), mech["zsens_T"]["ci95"])
chk("zsensmu_T", 0.0449, mech["zsensmu_T"]["mean"])
chk("zsens0_T", 0.1112, mech["zsens0_T"]["mean"])
chk("c1 mech", 0.0149, mech["c1"]["mean"])
chk("zsensmu_P", 0.0214, mech["zsensmu_P"]["mean"])
chk("zsens0_P", 0.0530, mech["zsens0_P"]["mean"])
for k in ("zsens_T", "zsensmu_T", "c1", "zsensmu_P", "zsens0_P"):
    assert mech[k]["within_floor"], f"{k} should be WITHIN FLOOR"
assert not mech["zsens0_T"]["within_floor"], "zsens0_T should be ABOVE floor"
# the >= 12.24 lower bound on ||zbar||
lb = (zg["parent"]["mean_norm"] ** 2 - zg["parent"]["centred_rms"] ** 2) ** 0.5
chk("||zbar|| lower bound", 12.24, lb, tol=5e-3)

# --- sec 7.5 param split ---------------------------------------------------------------------
ps = Z9["param_split"]
chk("z path param share", 2.166, ps["pool10"]["param_frac_z"] * 100, tol=2e-3)
for m, sq in (("FLOOR_c277178", 0.026), ("FLOOR_c275758", 0.104), ("pool10", 1.097),
              ("semistall3", 0.351), ("defensive10", 0.676)):
    chk(f"z disp {m}", sq, ps[m]["disp_frac_z_sq"] * 100, tol=2e-3)
for m, e in (("pool10", 0.506), ("semistall3", 0.162), ("defensive10", 0.312),
             ("FLOOR_c277178", 0.012), ("FLOOR_c275758", 0.048)):
    chk(f"enrichment {m}", e, ps[m]["enrichment_sq"], tol=2e-3)
for m, n in (("pool10", 16.485), ("semistall3", 11.050), ("defensive10", 14.736),
             ("FLOOR_c277178", 6.970), ("FLOOR_c275758", 8.330)):
    chk(f"||dtheta|| {m}", n, ps[m]["l2_norm_total"], tol=2e-3)
g = ps["pool10"]["groups"]
chk("z_encoder param share", 0.24, g["z_encoder"]["param_frac"] * 100, tol=6e-3)
chk("film gen param share", 1.92, g["film_generators"]["param_frac"] * 100, tol=6e-3)
chk("recon_head param share", 0.38, g["recon_head"]["param_frac"] * 100, tol=6e-3)
chk("trunk param share", 97.46, g["shared_trunk_and_heads"]["param_frac"] * 100, tol=6e-3)

# --- sec 4 wall clock / states ----------------------------------------------------------------
for nm, art, st, ws, tw in (("best n3", Z3, 4180, 254, 368), ("best n9", Z9, 11650, 618, 913),
                            ("fi n9", ZS9, 11650, 628, 920)):
    chk(f"{nm} states", st, art["_meta"]["n_states"], tol=0)
    chk(f"{nm} states wall", ws, art["_meta"]["wall_s_states"], tol=1.0)
    chk(f"{nm} total wall", tw, art["_meta"]["wall_s_total"], tol=1.0)

# --- structural claims -------------------------------------------------------------------------
for art in (Z3, Z9):
    for m, v in art["_meta"]["acid_shim"].items():
        assert v["disarmed_max_abs_delta"] == 0.0 and v["own_z_max_abs_delta"] == 0.0, m
    assert art["_meta"]["acid_all_distinct"]
n_shim = sum(len(a["_meta"]["acid_shim"]) for a in (Z3, Z9)) * 2
print(f"  shim ACID checks across the two headline cells: {n_shim} (README says 24)")
assert n_shim == 24, n_shim
assert A9["pooled"]["taught_own"]["f_b"]["verdict"].startswith("SHARED WEIGHTS")
assert A3["pooled"]["taught_own"]["f_b"]["verdict"].startswith("SHARED WEIGHTS")
assert Z9["_meta"]["teacher_rung"] == {k: "best_model/best_model.zip"
                                       for k in ("pool10", "semistall3", "defensive10")}
assert ZS9["_meta"]["teacher_rung"] == {k: "final_model_interrupted.zip"
                                        for k in ("pool10", "semistall3", "defensive10")}

w = max(len(r[0]) for r in rows)
for lab, c, a, v in rows:
    print(f"  {lab:{w}s}  README {c:>10.4f}   artifact {a:>10.4f}   {v}")
print(f"\n  {len(rows)} numeric claims checked, {bad} mismatch(es)")
sys.exit(1 if bad else 0)
