"""B5 (the ride-along A head vs ground truth) and B8 (RND coverage / saturation): tabulate ``main.ridealong_read``
JSONs of X26 checkpoints (the Lane S bank's first 200 battles, `truth_v2`'s turns inside them).

    python3 b5_b8_summary.py <ridealong dir> <labels comma-separated> <out.json> <out.md>

B5, per truth continuation (K2 / N0 / C_fix), the reader's battle-clustered 95 % (B = 1,000):
- (ii) Spearman of A's member-mean against the truth's per-action values, within turn, and A minus the policy
  logits' own Spearman (the reference row): does A know MORE than the policy about the values?
- (ii) ``starved_top2``: on turns with a starved near-best action, how often A ranks one in its top 2 (chance given);
- (iii) A's member spread on starved near-best actions vs every other legal action (AUROC): does A's disagreement
  point at the moves the policy starves?
B8 (Amendment 1 of the ride-along registration, descriptive here, NOT its registered verdict read):
- (b) coverage: each RND key's AUROC for b1 exploiter rows vs the rest and b2 late-game rows vs the rest;
- (c) saturation: each key's raw-error relative spread (IQR / median) on the bank.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

KEYS = ("rnd", "rndv_fast", "rndv_decay", "rndv_small", "rndv_feat")
CONTS = ("K2final", "N0final75M", "C_fixfinal")


def main(d: str, labels: str, out_json: str, out_md: str) -> int:
    res = {}
    for lab in labels.split(","):
        j = json.load(open(Path(d) / f"{lab}.json"))
        b5 = {}
        for c in CONTS:
            t = j["truth"][c]
            b5[c] = {"A_spearman": t["ii_A_mean"]["spearman"], "A_minus_logits": t["ii_spearman_A_minus_logits"],
                     "A_starved_top2": t["ii_A_mean"]["starved_top2"],
                     "logits_starved_top2": t["ii_policy_logits_reference"]["starved_top2"],
                     "A_std_starved_near_auroc": t["iii_adv_std_on_starved_near_best"]["auroc_starved_near_vs_other_legal"]}
        cov = j["rnd_variants"]["b_coverage"]["contrasts"]
        sat = j["rnd_variants"]["c_saturation"]["keys"]
        res[lab] = {"heads": j["heads"], "bank": j["bank"], "b5": b5,
                    "b8_coverage": {k: {"b1_exploiter": cov["b1_exploiter_vs_rest"]["auroc"][k],
                                        "b2_late_game": cov["b2_late_game_vs_rest"]["auroc"][k]} for k in KEYS},
                    "b8_saturation_rel_spread": {k: sat[k]["rel_spread"] for k in KEYS},
                    "b8_median_raw_err": {k: sat[k]["median"] for k in KEYS},
                    "ens_std_auroc_v_err": j["i_uncertainty_vs_v_error"]["all"]["ens_std"]["auroc"]}
    Path(out_json).write_text(json.dumps({"schema": "x26_collapse_battery_b5_b8_v1", "rows": res}, indent=1))

    def ci(x, k="mean"):
        return f"{x[k]:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}]"
    L = ["### B5: A vs the truth (`truth_v2`, three continuations K2 / N0 / C_fix)", "",
         "| X26 @ | A Spearman vs truth | A − logits Spearman | A ranks a starved near-best in its top 2 (logits) | A spread AUROC, starved near-best vs other legal |",
         "|---|---|---|---|---|"]
    for lab, r in res.items():
        b = r["b5"]
        L.append(f"| {lab} | " + " / ".join(f"{b[c]['A_spearman']['mean']:.3f}" for c in CONTS) + " | "
                 + " / ".join(ci(b[c]["A_minus_logits"], "delta") for c in CONTS) + " | "
                 + " / ".join(f"{b[c]['A_starved_top2']['mean']:.3f} ({b[c]['logits_starved_top2']['mean']:.3f})" for c in CONTS)
                 + f" (chance {b['K2final']['A_starved_top2']['chance']:.2f}) | "
                 + " / ".join(f"{b[c]['A_std_starved_near_auroc']['auroc']:.3f}" for c in CONTS) + " |")
    L += ["", "### B8: RND coverage (AUROC; b1 exploiter rows vs rest / b2 late game vs rest) and saturation (IQR / median of the raw error)", "",
          "| X26 @ | " + " | ".join(KEYS) + " | saturation rel. spread (" + ", ".join(KEYS) + ") |",
          "|---|" + "---|" * len(KEYS) + "---|"]
    for lab, r in res.items():
        c = r["b8_coverage"]
        L.append(f"| {lab} | " + " | ".join(f"{c[k]['b1_exploiter']['value']:.3f} / {c[k]['b2_late_game']['value']:.3f}" for k in KEYS)
                 + " | " + ", ".join(f"{r['b8_saturation_rel_spread'][k]:.3f}" for k in KEYS) + " |")
    Path(out_md).write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:]))
