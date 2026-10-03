"""Print the headline tables from out/tier0<tag>.json (used to write the README)."""
import json, sys
tag = sys.argv[1] if len(sys.argv) > 1 else ""
d = json.load(open(f"out/tier0{tag}.json"))
print("counts", d["counts"], "r", d["r_hist"])
F = d["feats"]; P = d["proxies"]
for bel in d["cells"]:
    print(f"\n=== {bel}: mean |err| (pp; bulk = rel %) and flip rate % [excluded %], pooled")
    print(f"{'rep':10s} " + " ".join(f"{f:>8s}" for f in F) + " | " + " ".join(f"{p:>12s}" for p in P))
    order = ["R0"] + [f"{x}_{b}" for b in d["budgets"] for x in ("R1", "R2", "OP", "OO")] + ["Rinf"]
    for rn in order:
        c = d["cells"][bel][rn]["all"]
        print(f"{rn:10s} " + " ".join(f"{c[f]['mean']:8.2f}" for f in F) + " | " +
              " ".join(f"{100*c[p]['flip']:5.1f} [{100*c[p]['excluded']:4.1f}]" for p in P))

def ci(c):
    return f"{c['mean']:.2f} [{c['ci'][0]:.2f}, {c['ci'][1]:.2f}]"
print("\n=== CIs, prior, pooled (mean |err|, battle-clustered 95%)")
for rn in ["R0", "R1_1:1", "R1_k+6", "R1_12", "R1_18", "R2_1:1", "OO_1:1", "Rinf"]:
    c = d["cells"]["prior"][rn]["all"]
    print(f"{rn:8s} " + " | ".join(f"{f} {ci(c[f])}" for f in ("out", "in_mean", "in_max", "in_emax", "margin")))
    print(" " * 9 + " | ".join(f"{p} {100*c[p]['flip']:.1f} [{100*c[p]['ci'][0]:.1f}, {100*c[p]['ci'][1]:.1f}] n={c[p]['n']}" for p in P))
print("\n=== derived, prior, pooled: OTHER share / Jensen share / recovered of R0 (pt, lo, hi) / of gap")
for f in F:
    dd = d["derived"]["prior"]["all"][f]
    print(f, "Rinf rec of R0", [round(x, 3) if x is not None else None for x in dd["Rinf_recovered_of_R0"]])
    for bn in d["budgets"]:
        x = dd[bn]
        fm = lambda v: "[" + ", ".join("None" if y is None else f"{y:.2f}" for y in v) + "]"
        print(f"   {bn:5s} other(OP) {fm(x['other_share'])} oracle(OO) {fm(x['oracle_other_share'])} jensen {fm(x['jensen_share'])} recR0 {fm(x['recovered_of_R0'])} recR0(d) {fm(x['recovered_of_R0_d'])} gap {x['recovered_of_gap_R0_to_Rinf']}")
print("\n=== per r, prior: in_mean / P2e / P3 / out by rep")
for rr in ["1", "2", "3", "4", "5"]:
    row = []
    for rn in ["R0", "R1_1:1", "R1_k+6", "R1_12", "R1_18", "Rinf", "OO_1:1"]:
        c = d["cells"]["prior"][rn][rr]
        row.append(f"{rn}: {c['in_mean']['mean']:.1f}/{100*c['P2e']['flip']:.1f}/{100*c['P3']['flip']:.1f}/{c['out']['mean']:.1f}")
    print(f"r={rr} n={d['cells']['prior']['R0'][rr]['n']}: " + "  ".join(row))
print("\n=== belief mass (OTHER share of hidden mass · recall), real states")
for bel, m in d["belief_mass"].items():
    for bn, byr in m.items():
        print(bel, bn, " | ".join(f"r{rr} {v['other_share']:.0%} {v['recall']:.2f} t{v['near_ties']}" for rr, v in byr.items() if rr != "frac_other_mass_gt1"), "| mass>1:", round(byr["frac_other_mass_gt1"], 3))

print("\n=== flips on the COMMON included set (%), prior")
cm = d["flips_common_set"]["prior"]
for p in P:
    print(p, "n", cm[p]["n"], " ".join(f"{rn} {100*cm[p][rn][0]:.1f}" for rn in ["R0", "R1_1:1", "R1_k+6", "R1_12", "R1_18", "Rinf", "R2_1:1", "OP_1:1", "OO_1:1"]))
print("P1 structural exclusion (fewer than 2 usable damaging moves):", d["p1_structural_exclusion"])

print("\n=== paired deltas vs R1_1:1 (common set flips, pp; feature |err| pp), prior")
for p in P:
    dd = cm[p]["delta_vs_R1_1:1"]
    print(p, " ".join(f"{rn} {100*dd[rn][0]:+.2f} [{100*dd[rn][1]:+.2f},{100*dd[rn][2]:+.2f}]" for rn in ["R0", "R1_k+6", "R1_12", "R1_18", "Rinf", "OP_1:1"]))
fd = cm["feature_delta_vs_R1_1:1"]
for f in F:
    print(f, " ".join(f"{rn} {fd[f][rn][0]:+.2f} [{fd[f][rn][1]:+.2f},{fd[f][rn][2]:+.2f}]" for rn in ["R0", "R1_k+6", "R1_12", "R1_18", "Rinf", "OP_1:1"]))
