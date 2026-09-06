"""Re-derive every number README.md quotes, from the JSON artefacts, and FAIL on any mismatch.

Same contract as ../exploiter_drift/verify_readme.py and ../content_locality_v2/verify_readme.py:
the README is prose over machine output, so the prose is checked against the machine output rather
than trusted. Every assertion below names the README claim it pins.

Run: nice -n 10 python verify_readme.py
(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
A = json.load(open(os.path.join(HERE, "analysis.json")))
DG = json.load(open(os.path.join(HERE, "drift_gen.json")))
DV = json.load(open(os.path.join(HERE, "drift_v8.json")))
PG = json.load(open(os.path.join(HERE, "popart_split_gen.json")))
PV = json.load(open(os.path.join(HERE, "popart_split_v8.json")))
README = open(os.path.join(HERE, "README.md")).read()

fails: list[str] = []
checks = 0


def eq(claim, got, want, tol=5e-5):
    global checks
    checks += 1
    if abs(float(got) - float(want)) > tol:
        fails.append(f"{claim}: README says {want}, artefact says {got}")


def is_(claim, got, want):
    global checks
    checks += 1
    if got != want:
        fails.append(f"{claim}: README says {want!r}, artefact says {got!r}")


def in_readme(claim, s):
    global checks
    checks += 1
    if s not in README:
        fails.append(f"{claim}: the string {s!r} is not in README.md")


# --- provenance ------------------------------------------------------------------------------
is_("parent step v8", DV["_meta"]["parent_steps"], 277_583_267)
is_("parent step gen", DG["_meta"]["parent_steps"], 28_115_184)
is_("P v8", DV["_meta"]["n_params"], 3_512_397)
is_("P gen", DG["_meta"]["n_params"], 3_147_887)
is_("n states v8", DV["_meta"]["n_states"], 456)
is_("n states gen", DG["_meta"]["n_states"], 456)
is_("grouping imported", DG["_meta"]["grouping_source"],
    "../sharing_kernel/kernel.py (imported)")
is_("gen KL source is the real one", DG["_meta"]["kl_source"],
    "agents.training.instrumented_ppo.distill_anchor.masked_kl_rows")
is_("v8 KL source is the era copy", DV["_meta"]["kl_source"],
    "../content_locality/era_kl.masked_kl_rows_era")
in_readme("v8 P quoted", "3,512,397")
in_readme("gen P quoted", "3,147,887")

# --- depth grid ------------------------------------------------------------------------------
for key, t, steps, l2, kl in [
    ("A@d1", 416_750, 278_000_017, 1.7659, 0.0337),
    ("A@d3", 1_080_663, 278_663_930, 2.7901, 0.0530),
    ("B@d1", 416_747, 278_000_014, 1.7699, 0.0342),
    ("B@d3", 1_081_344, 278_664_611, 2.7496, 0.0546),
    ("C@d1", 416_747, 278_000_014, 1.7623, 0.0311),
    ("C@d3", 1_086_364, 278_669_631, 2.9342, 0.0626),
]:
    r = DV["arms"][key]
    is_(f"v8 {key} t", r["t"], t)
    is_(f"v8 {key} step", r["steps"], steps)
    eq(f"v8 {key} |d|", r["l2"], l2, 5e-5)
    eq(f"v8 {key} KL", r["kl_all"], kl, 5e-5)

for key, t, steps, l2, kl in [
    ("A@d1", 500_016, 28_615_200, 1.9054, 0.0710),
    ("A@d2", 1_000_032, 29_115_216, 2.6385, 0.1090),
    ("A@d3", 1_179_648, 29_294_832, 2.8949, 0.1215),
    ("B@d1", 500_016, 28_615_200, 1.9024, 0.0852),
    ("B@d2", 1_000_032, 29_115_216, 2.6151, 0.1246),
    ("B@d3", 1_179_648, 29_294_832, 2.8493, 0.1410),
    ("C@d1", 500_016, 28_615_200, 1.9116, 0.0763),
    ("C@d2", 1_000_032, 29_115_216, 2.5998, 0.1141),
    ("C@d3", 1_179_648, 29_294_832, 2.8499, 0.1433),
]:
    r = DG["arms"][key]
    is_(f"gen {key} t", r["t"], t)
    is_(f"gen {key} step", r["steps"], steps)
    eq(f"gen {key} |d|", r["l2"], l2, 5e-5)
    eq(f"gen {key} KL", r["kl_all"], kl, 5e-5)

# same-depth agreement claims
ag = A["v8_same_depth_agreement"]
is_("A step gap 29", ag["A"]["step_gap"], 29)
is_("C step gap 5019", ag["C"]["step_gap"], 5_019)
is_("A bit-identical", ag["A"]["identical_to_1e9"], True)
is_("C bit-identical", ag["C"]["identical_to_1e9"], True)
is_("B NOT bit-identical", ag["B"]["identical_to_1e9"], False)
eq("B rel diff 2.36e-03", ag["B"]["rel_diff"], 2.36e-3, 5e-6)

# --- P1 --------------------------------------------------------------------------------------
P1 = A["P1_displacement_exponent"]
for era_key, rows in {
    "v8_2pt": {"ALL": (0.4800, 0.4620, 0.5321, 0.4914),
               "action_head": (0.4961, 0.4746, 0.5486, 0.5064),
               "encoders": (0.5322, 0.4903, 0.5729, 0.5318),
               "team_transformer": (0.4799, 0.4620, 0.5174, 0.4865),
               "projection_mlp": (0.4804, 0.4654, 0.5389, 0.4949),
               "belief_op": (0.5873, 0.5353, 0.6364, 0.5863),
               "critic": (0.1597, 0.1766, 0.1639, 0.1667)},
    "gen_2pt_outer": {"ALL": (0.4873, 0.4706, 0.4652, 0.4744),
                      "action_head": (0.5841, 0.5811, 0.5667, 0.5773),
                      "encoders": (0.4896, 0.4755, 0.4702, 0.4784),
                      "team_transformer": (0.4772, 0.4634, 0.4523, 0.4643),
                      "projection_mlp": (0.5408, 0.5279, 0.5271, 0.5319),
                      "belief_op": (0.4301, 0.4233, 0.4169, 0.4235),
                      "critic": (0.4862, 0.3939, 0.3748, 0.4183)},
}.items():
    for g, (a_, b_, c_, m_) in rows.items():
        pa = P1[era_key][g]["per_arm"]
        eq(f"P1 {era_key} {g} A", pa["A"], a_)
        eq(f"P1 {era_key} {g} B", pa["B"], b_)
        eq(f"P1 {era_key} {g} C", pa["C"], c_)
        eq(f"P1 {era_key} {g} mean", P1[era_key][g]["mean"], m_)

eq("P1 gen 3pt ALL 0.4694", P1["gen_3pt"]["ALL"]["mean"], 0.4694)
eq("P1 v8 ALL CI lo", P1["v8_2pt"]["ALL"]["ci"][0], 0.4620)
eq("P1 v8 ALL CI hi", P1["v8_2pt"]["ALL"]["ci"][1], 0.5321)
eq("P1 gen ALL CI lo", P1["gen_2pt_outer"]["ALL"]["ci"][0], 0.4652)
eq("P1 gen ALL CI hi", P1["gen_2pt_outer"]["ALL"]["ci"][1], 0.4873)

C1 = A["P1_contrast_matched_2pt"]["ALL"]
eq("P1 ALL diff +0.0170", C1["diff_v8_minus_gen"], 0.0170)
eq("P1 ALL arm-spread floor 0.0701", C1["max_within_cell_spread"], 0.0701)
eq("P1 ALL perm p 0.700", C1["perm_p"], 0.700, 1e-9)
is_("P1 ALL verdict WITHIN FLOOR", C1["verdict"], "WITHIN FLOOR")
is_("P1 ALL CIs overlap", C1["ci_disjoint"], False)
is_("P1 permutation arrangements = 20", C1["perm_arrangements"], 20)
# the pre-registered SIGNIFICANT rule requires BOTH; ALL meets neither
if C1["complete_separation"] or C1["ci_disjoint"]:
    fails.append("P1 ALL: README claims NOT DETECTED but a separation criterion fired")
checks += 1
# 0.4914 is 0.0086 from 0.5 and 0.51 from 1.0, as the README says
eq("P1 v8 ALL distance from 0.5", abs(P1["v8_2pt"]["ALL"]["mean"] - 0.5), 0.0086, 1e-4)

# group sizes quoted in the composition caveat
LV, LG = DV["_meta"]["group_layout"], DG["_meta"]["group_layout"]
for g, want_v8, want_gen in [("action_head", 5_643, 55_683),
                             ("belief_op", 204_065, 512_267),
                             ("critic", 1_261_891, 742_650)]:
    is_(f"v8 {g} size", LV[g][1] - LV[g][0], want_v8)
    is_(f"gen {g} size", LG[g][1] - LG[g][0], want_gen)

# --- PopArt follow-up -------------------------------------------------------------------------
is_("PopArt layer = 513 params (v8)", PV["n_popart"], 513)
is_("PopArt layer = 513 params (gen)", PG["n_popart"], 513)
is_("v8 critic remainder", PV["n_rest"], 1_261_378)
is_("gen critic remainder", PG["n_rest"], 742_137)
for arm, want in {"A": 0.2654, "B": 0.3150, "C": 0.2846}.items():
    eq(f"v8 critic-ex-PopArt {arm}", PV["b_critic_excluding_popart"][arm], want)
for arm, want in {"A": 0.4889, "B": 0.3938, "C": 0.3763}.items():
    eq(f"gen critic-ex-PopArt {arm}", PG["b_critic_excluding_popart"][arm], want)
eq("v8 critic-ex-PopArt mean 0.2883",
   float(np.mean(list(PV["b_critic_excluding_popart"].values()))), 0.2883)
eq("gen critic-ex-PopArt mean 0.4196",
   float(np.mean(list(PG["b_critic_excluding_popart"].values()))), 0.4196)
# complete separation of the CORRECTED critic exponent
if not (min(PG["b_critic_excluding_popart"].values())
        > max(PV["b_critic_excluding_popart"].values())):
    fails.append("corrected critic: README claims complete separation, artefact disagrees")
checks += 1
eq("corrected critic min gen arm 0.376",
   min(PG["b_critic_excluding_popart"].values()), 0.3763)
eq("corrected critic max v8 arm 0.315",
   max(PV["b_critic_excluding_popart"].values()), 0.3150)
# the quoted PopArt share bands
v8_d1 = [PV["arms"][f"{a}@d1"]["popart_share_of_critic_sq"] for a in "ABC"]
v8_d3 = [PV["arms"][f"{a}@d3"]["popart_share_of_critic_sq"] for a in "ABC"]
eq("v8 PopArt share d1 min 25.2%", min(v8_d1) * 100, 25.23, 0.01)
eq("v8 PopArt share d1 max 27.6%", max(v8_d1) * 100, 27.64, 0.01)
eq("v8 PopArt share d3 min 5.8%", min(v8_d3) * 100, 5.78, 0.01)
eq("v8 PopArt share d3 max 8.6%", max(v8_d3) * 100, 8.56, 0.01)
gen_sh = [PG["arms"][f"{a}@{d}"]["popart_share_of_critic_sq"]
          for a in "ABC" for d in ("d1", "d2", "d3")]
if max(gen_sh) * 100 > 0.82 + 1e-6:
    fails.append(f"gen PopArt share exceeds the quoted 0.82% ceiling: {max(gen_sh)*100:.3f}%")
checks += 1
eq("PopArt-only b, v8 mean -0.53",
   float(np.mean(list(PV["b_critic_popart_only"].values()))), -0.5317, 5e-4)

# --- P2 --------------------------------------------------------------------------------------
P2 = A["P2_replicate_cosine"]
for era, want in {
    "v8": {"ALL": (0.2219, 0.2221, 0.2153, 0.2198),
           "action_head": (0.1647, 0.2126, 0.1878, 0.1884),
           "encoders": (0.3656, 0.3619, 0.3693, 0.3656),
           "team_transformer": (0.4161, 0.4120, 0.4020, 0.4100),
           "projection_mlp": (0.1025, 0.1065, 0.0981, 0.1024),
           "belief_op": (0.8058, 0.7881, 0.7782, 0.7907),
           "critic": (0.5138, 0.5388, 0.5642, 0.5389)},
    "gen": {"ALL": (0.4102, 0.3995, 0.4020, 0.4039),
            "action_head": (0.1687, 0.1510, 0.1556, 0.1584),
            "encoders": (0.4370, 0.4361, 0.4323, 0.4351),
            "team_transformer": (0.5154, 0.5103, 0.5101, 0.5119),
            "projection_mlp": (0.0717, 0.0606, 0.0641, 0.0655),
            "belief_op": (0.7299, 0.7239, 0.7253, 0.7263),
            "critic": (0.5823, 0.5599, 0.5734, 0.5719)},
}.items():
    for g, (ab, ac, bc, m_) in want.items():
        vals = P2[era]["d3"][g]["values"]
        eq(f"P2 {era} {g} A.B", vals[0], ab)
        eq(f"P2 {era} {g} A.C", vals[1], ac)
        eq(f"P2 {era} {g} B.C", vals[2], bc)
        eq(f"P2 {era} {g} mean", P2[era]["d3"][g]["mean"], m_)

CP2 = A["P2_contrast_at_end"]["ALL"]
eq("P2 ALL diff -0.1841", CP2["diff_v8_minus_gen"], -0.1841)
is_("P2 ALL complete separation", CP2["complete_separation"], True)
eq("P2 ALL perm p 0.100", CP2["perm_p"], 0.100, 1e-9)
if not (CP2["v8_mean"] < CP2["gen_mean"]):
    fails.append("P2: README says OUR replicates agree MORE; artefact disagrees")
checks += 1
eq("P2 rw floor v8 5.34e-04", A["P2_random_walk_floor"]["v8"], 5.34e-4, 5e-7)
eq("P2 rw floor gen 5.64e-04", A["P2_random_walk_floor"]["gen"], 5.64e-4, 5e-7)
# the "22-40%" band the README claims for both cells' ALL cosine
allc = P2["v8"]["d3"]["ALL"]["values"] + P2["gen"]["d3"]["ALL"]["values"]
if not (0.21 <= min(allc) and max(allc) <= 0.42):
    fails.append(f"P2 22-40% band: observed range {min(allc):.4f}-{max(allc):.4f}")
checks += 1

# --- P3 --------------------------------------------------------------------------------------
P3 = A["P3_output_twin"]
for era, want in {
    "v8": {"kl_all": (0.4762, 0.4917, 0.7311, 0.5663),
           "kl_taught": (0.4994, 0.3950, 0.6892, 0.5278),
           "kl_untaught": (0.4318, 0.6963, 0.8137, 0.6473)},
    "gen": {"kl_all": (0.6259, 0.5872, 0.7347, 0.6493),
            "kl_taught": (0.7396, 0.5239, 0.7192, 0.6609),
            "kl_untaught": (0.3884, 0.7512, 0.7821, 0.6406)},
}.items():
    for s, (a_, b_, c_, m_) in want.items():
        pa = P3[era][s]["per_arm"]
        eq(f"P3 {era} {s} A", pa["A"], a_)
        eq(f"P3 {era} {s} B", pa["B"], b_)
        eq(f"P3 {era} {s} C", pa["C"], c_)
        eq(f"P3 {era} {s} mean", P3[era][s]["mean"], m_)
eq("P3 v8 kl_all CI lo", P3["v8"]["kl_all"]["ci"][0], 0.4762)
eq("P3 v8 kl_all CI hi", P3["v8"]["kl_all"]["ci"][1], 0.7311)
eq("P3 gen kl_all CI lo", P3["gen"]["kl_all"]["ci"][0], 0.5872)
eq("P3 gen kl_all CI hi", P3["gen"]["kl_all"]["ci"][1], 0.7347)
CP3 = A["P3_contrast"]["kl_all"]
eq("P3 diff -0.0829", CP3["diff_v8_minus_gen"], -0.0829)
is_("P3 verdict WITHIN FLOOR", CP3["verdict"], "WITHIN FLOOR")
eq("P3 taught floor 0.2942", A["P3_contrast"]["kl_taught"]["max_within_cell_spread"], 0.2942)
eq("c/b v8 1.15", A["P3_c_over_b"]["v8"], 1.153, 5e-3)
eq("c/b gen 1.37", A["P3_c_over_b"]["gen"], 1.369, 5e-3)
for era, arm, m_, lo, hi in [("v8", "A", 0.0530, 0.0457, 0.0605),
                             ("v8", "B", 0.0546, 0.0464, 0.0634),
                             ("v8", "C", 0.0626, 0.0536, 0.0719),
                             ("gen", "A", 0.1215, 0.1057, 0.1383),
                             ("gen", "B", 0.1410, 0.1235, 0.1587),
                             ("gen", "C", 0.1433, 0.1197, 0.1688)]:
    lv = P3[era]["levels"]["d3"][arm]
    eq(f"P3 level {era} {arm} mean", lv["mean"], m_)
    eq(f"P3 level {era} {arm} lo", lv["lo"], lo)
    eq(f"P3 level {era} {arm} hi", lv["hi"], hi)

# --- E1 ---------------------------------------------------------------------------------------
E1 = A["E1_within_arm_direction_persistence"]
GRPS = ["ALL", "action_head", "encoders", "team_transformer", "projection_mlp", "belief_op",
        "critic"]
E1_WANT = {
    ("v8", "A"): (0.6210, 0.6472, 0.6783, 0.6671, 0.6564, 0.6440, 0.6716, 0.6292),
    ("v8", "B"): (0.6208, 0.6555, 0.6587, 0.6751, 0.6669, 0.6546, 0.7024, 0.5156),
    ("v8", "C"): (0.6194, 0.6043, 0.6200, 0.6263, 0.6246, 0.6003, 0.6407, 0.5336),
    ("gen", "A"): (0.6511, 0.6533, 0.6599, 0.6582, 0.6864, 0.6352, 0.6497, 0.7081),
    ("gen", "B"): (0.6511, 0.6511, 0.6499, 0.6660, 0.6976, 0.6410, 0.6445, 0.6543),
    ("gen", "C"): (0.6511, 0.6502, 0.6541, 0.6583, 0.6870, 0.6394, 0.6444, 0.6777),
}
for (era, arm), want in E1_WANT.items():
    r = E1[era][arm]
    eq(f"E1 {era} {arm} sqrt(t1/t2)", r["diffusive_prediction"], want[0])
    for g, w in zip(GRPS, want[1:]):
        eq(f"E1 {era} {arm} {g}", r[g], w)
# the README's E1 claims, each checked as stated
all_dev = {(e, a): E1[e][a]["ALL"] - E1[e][a]["diffusive_prediction"]
           for e in ("v8", "gen") for a in "ABC"}
eq("E1 gen ALL within 0.0022",
   max(abs(v) for (e, _), v in all_dev.items() if e == "gen"), 0.0022, 5e-4)
eq("E1 v8 ALL within 0.035",
   max(abs(v) for (e, _), v in all_dev.items() if e == "v8"), 0.0347, 5e-4)
dev = {(e, a, g): E1[e][a][g] - E1[e][a]["diffusive_prediction"]
       for e in ("v8", "gen") for a in "ABC" for g in GRPS}
eq("E1 per-group min deviation -0.105", min(dev.values()), -0.1052, 5e-4)
eq("E1 per-group max deviation +0.082", max(dev.values()), 0.0816, 5e-4)
eq("E1 v8 per-group range low", min(v for (e, _, _), v in dev.items() if e == "v8"), -0.1052, 5e-4)
eq("E1 v8 per-group range high", max(v for (e, _, _), v in dev.items() if e == "v8"), 0.0816, 5e-4)
eq("E1 gen per-group range low", min(v for (e, _, _), v in dev.items() if e == "gen"), -0.0159, 5e-4)
eq("E1 gen per-group range high", max(v for (e, _, _), v in dev.items() if e == "gen"), 0.0570, 5e-4)
cos_max = max(E1[e][a][g] for e in ("v8", "gen") for a in "ABC" for g in GRPS)
eq("E1 largest cosine anywhere 0.7081", cos_max, 0.7081, 5e-4)
worst_to_pred = max(abs(v) for v in dev.values())
closest_to_one = 1.0 - cos_max
eq("E1 worst distance to sqrt(t1/t2) = 0.105", worst_to_pred, 0.1052, 5e-4)
eq("E1 smallest distance to 1.0 = 0.292", closest_to_one, 0.2919, 5e-4)
if closest_to_one / worst_to_pred < 2.7:
    fails.append(f"E1: README claims >=2.7x closer to diffusive than to directed; "
                 f"ratio is {closest_to_one / worst_to_pred:.2f}")
checks += 1

# --- buffers ----------------------------------------------------------------------------------
for era, D in (("v8", DV), ("gen", DG)):
    for k, r in D["arms"].items():
        got = sorted(r["buffers_changed"])
        if got != ["popart.mu", "popart.nu", "popart.sigma"]:
            fails.append(f"{era} {k}: buffers changed = {got}, README says only the PopArt three")
        checks += 1

# --- the four VERDICT-row claims, as strings --------------------------------------------------
for s in ["NOT DETECTED", "NOT SUPPORTED", "WITHIN FLOOR",
          "THE NOISE-VS-DRIFT ACCOUNT IS DEAD FOR THIS CONTRAST"]:
    in_readme("verdict string", s)

print(f"[v] {checks} checks")
if fails:
    print(f"[v] ❌ {len(fails)} MISMATCH(ES):")
    for f_ in fails:
        print("   -", f_)
    sys.exit(1)
print("[v] ✅ every number README.md quotes is reproduced from the artefacts")
