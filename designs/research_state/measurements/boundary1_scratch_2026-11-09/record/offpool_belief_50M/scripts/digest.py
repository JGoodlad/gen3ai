"""Compact markdown digest of the tables (reads tables/*.json; prints to stdout)."""
import json, sys
T = sys.argv[1]
J = lambda n: json.load(open(f"{T}/{n}.json"))
cells, br, w, bm, wm, bw, wbr, per, seed = (J("cells_vs_C"), J("breadth_at_256"), J("width_at_breadth"), J("breadth_main"),
                                          J("width_main"), None, None, J("per_run"), J("seed_contrasts"))
PAN = [("home72", "home72 (in pool)"), ("heldout72_hpfix", "H"), ("forum72", "F"), ("p72", "P72")]
MET = [("species_logp", "species"), ("moves_ll", "moves_ll"), ("items_logp", "items")]
RUNS = {"C": ["b1c_s5101", "b1c_s5102", "b1c_s5110"], "W": ["b1w_s5103", "b1w_s5104", "b1w_s5111"],
        "Br": ["b1br_s5105", "b1br_s5106", "b1br_s5112"], "WB": ["b1wb_s5107", "b1wb_s5108", "b1wb_s5113"]}
f = lambda d, k="mean": f"{d[k]:+.2f}"
iv = lambda d: f"{d['mean']:+.2f} [{d['lo']:+.2f}, {d['hi']:+.2f}]"
print("## per-cell belief - prior, team-clustered 95% interval [seed range of the three runs]\n")
print("| metric | panel | prior level | " + " | ".join(RUNS) + " |\n|---|---|---|" + "---|" * 4)
for m, mn in MET:
    for p, pn in PAN:
        a = cells["panels"][p]["metrics"][m]["arms"]
        row = []
        for c in RUNS:
            vs = [per["panels"][p]["metrics"][m]["arms"][r]["post_minus_prior"]["mean"] for r in RUNS[c]]
            row.append(f"{iv(a[c]['post_minus_prior'])} [{min(vs):+.2f}..{max(vs):+.2f}]")
        print(f"| {mn} | {pn} | {a['C']['prior']['mean']:+.2f} | " + " | ".join(row) + " |")
print("\n## factor contrasts: meter team-bootstrap 95% interval | seed-level 95% t interval (8 df)\n")
print("| metric | panel | breadth ON-OFF (Br+WB vs C+W) | seed t | width ON-OFF (W+WB vs C+Br) | seed t | breadth @128 Br-C | seed t | breadth @256 WB-W | seed t |\n|---|---|---|---|---|---|---|---|---|---|")
for m, mn in MET:
    for p, pn in PAN:
        sc = seed["panels"][p][m]["post_minus_prior"]["contrasts"]
        g = lambda t, ref, arm: t["panels"][p]["metrics"][m]["arms"][arm][f"minus_{ref}"]
        s = lambda k: f"{sc[k]['est']:+.2f} [{sc[k]['lo']:+.2f}, {sc[k]['hi']:+.2f}]"
        print(f"| {mn} | {pn} | {iv(g(bm,'OFF','ON'))} | {s('breadth_main (Br+WB)-(C+W)')} | {iv(g(wm,'OFF','ON'))} | {s('width_main (W+WB)-(C+Br)')} | "
              f"{iv(g(cells,'C','Br'))} | {s('breadth @128 Br-C')} | {iv(g(br,'W','WB'))} | {s('breadth @256 WB-W')} |")
