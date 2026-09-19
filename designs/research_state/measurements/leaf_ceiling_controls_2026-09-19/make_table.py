"""Render score.json as the README's tables — so no number in the prose is retyped by hand."""
import json
import sys

d = json.load(open(sys.argv[1]))
COLS = ["POOLED", "rand|top1", "rand|top2", "top1|top2"]
# a "|" inside a markdown cell ends the cell — escape it in every HEADER we print
HDR = [c.replace("|", "\\|") for c in COLS]


def row(name, get):
    return "| " + name + " | " + " | ".join(get(c) for c in COLS) + " |"


print("### single-rollout labels — pairwise accuracy [95% CI over forks]\n")
print("| scorer | " + " | ".join(HDR) + " |")
print("|---" * (len(COLS) + 1) + "|")
for nm in d["single_label"]["POOLED"]["heads"]:
    print(row(nm, lambda c, nm=nm: "%.4f [%.4f, %.4f]" % (
        d["single_label"][c]["heads"][nm]["acc"],
        *d["single_label"][c]["heads"][nm]["ci"])))

print("\n### paired Δ vs arm W's head, same pairs\n")
print("| scorer | " + " | ".join(HDR) + " |")
print("|---" * (len(COLS) + 1) + "|")
for nm in d["single_label"]["POOLED"]["heads"]:
    if nm == "headW":
        continue
    def g(c, nm=nm):
        x = d["single_label"][c]["heads"][nm]["delta_vs_headW"]
        return "%+.4f [%+.4f, %+.4f] %s" % (x["delta"], *x["ci"],
                                            "**DET**" if x["detected"] else "ND")
    print(row(nm, g))

if "averaged_label" in d:
    print("\n### K-averaged labels\n")
    print("| scorer | " + " | ".join(HDR) + " |")
    print("|---" * (len(COLS) + 1) + "|")
    for nm in d["averaged_label"]["POOLED"]["heads"]:
        print(row(nm, lambda c, nm=nm: "%.4f [%.4f, %.4f]" % (
            d["averaged_label"][c]["heads"][nm]["acc"],
            *d["averaged_label"][c]["heads"][nm]["ci"])))
    print("\n### the MATCHED-NOISE half-label read (one K/2 label, two scorers)\n")
    print("| scorer | " + " | ".join(HDR) + " |")
    print("|---" * (len(COLS) + 1) + "|")
    for nm in d["averaged_label"]["POOLED"]["half_label"]["heads"]:
        print(row(nm, lambda c, nm=nm: "%.4f [%.4f, %.4f]" % (
            d["averaged_label"][c]["half_label"]["heads"][nm]["acc"],
            *d["averaged_label"][c]["half_label"]["heads"][nm]["ci"])))
    print(row("**ROLLOUT − head**", lambda c: "%+.4f [%+.4f, %+.4f] %s" % (
        d["averaged_label"][c]["half_label"]["rollout_minus_head"]["delta"],
        *d["averaged_label"][c]["half_label"]["rollout_minus_head"]["ci"],
        "**DET**" if d["averaged_label"][c]["half_label"]["rollout_minus_head"]["detected"]
        else "ND")))
    print("\n### ceilings and the recovered gap\n")
    print("| quantity | " + " | ".join(HDR) + " |")
    print("|---" * (len(COLS) + 1) + "|")
    for lab, f in (
        ("split-half agreement (K/2 vs K/2)", lambda c: "%.4f" % d["averaged_label"][c]["split_half_ceiling"]["agreement"]),
        ("→ implied oracle acc @K/2", lambda c: "%.4f" % d["averaged_label"][c]["split_half_ceiling"]["oracle_acc"]),
        ("recovered true-gap sd", lambda c: "%.4f" % d["averaged_label"][c]["gap"]["true_gap_sd"]),
        ("E|gap|", lambda c: "%.4f" % d["averaged_label"][c]["gap"]["E_abs_gap"]),
        ("model oracle @K=1", lambda c: "%.4f" % d["averaged_label"][c]["gap"]["acc_oracle"]["K1"]["acc"]),
        ("single-vs-averaged order agreement", lambda c: "%.4f (n=%d)" % (
            d["averaged_label"][c]["order_agreement_single_vs_avg"]["agree"],
            d["averaged_label"][c]["order_agreement_single_vs_avg"]["n"])),
    ):
        print(row(lab, f))
    print("\n### the K-mean as a SCORER on the banked single labels (same subset)\n")
    print("| scorer | " + " | ".join(HDR) + " |")
    print("|---" * (len(COLS) + 1) + "|")
    for nm in d["averaged_label"]["POOLED"]["single_on_subset"]["heads"]:
        print(row(nm, lambda c, nm=nm: "%.4f [%.4f, %.4f]" % (
            d["averaged_label"][c]["single_on_subset"]["heads"][nm]["acc"],
            *d["averaged_label"][c]["single_on_subset"]["heads"][nm]["ci"])))
