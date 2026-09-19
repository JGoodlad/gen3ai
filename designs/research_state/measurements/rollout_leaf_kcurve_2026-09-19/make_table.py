"""Render the JSON reads into the README's tables. NO NUMBER IN THE README IS TYPED BY HAND.

    python3 make_table.py --kcurve kcurve.json --rule playoff_rule.json --battery battery.json
"""
from __future__ import annotations

import argparse
import json

KS = (1, 2, 4, 8, 16)
COLS = ("POOLED", "rand|top1", "rand|top2", "top1|top2")


def f(x, n=4):
    return "—" if x is None else f"{x:.{n}f}"


def ci(c, n=4):
    return "—" if not c else f"[{c[0]:.{n}f}, {c[1]:.{n}f}]"


def lvl(s):
    return f"**{f(s['acc'])}** {ci(s['ci'])}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kcurve")
    ap.add_argument("--rule")
    ap.add_argument("--battery")
    a = ap.parse_args()

    if a.kcurve:
        K = json.load(open(a.kcurve))
        cur, cost = K["curve"], K["cost"]
        tpr = cost["mean_remaining_turns_per_rollout_fresh"]
        print("### the K-curve — pairwise accuracy against ONE K′ = 8 label "
              "[95 % CI over forks]\n")
        head = "| scorer | rollouts | sim turns / leaf eval | " + " | ".join(
            c.replace("|", "\\|") for c in COLS) + " |"
        print(head)
        print("|" + "---|" * (3 + len(COLS)))
        row = ["**headW** (the 75M critic)", "**0**", "**0**"]
        for c in COLS:
            row.append(lvl(cur[c]["scorers"]["headW"]))
        print("| " + " | ".join(row) + " |")
        for k in KS:
            row = [f"ROLLOUT_{k}", str(k), f"{k * tpr:.0f}"]
            for c in COLS:
                row.append(lvl(cur[c]["scorers"][f"ROLLOUT_{k}"]))
            print("| " + " | ".join(row) + " |")
        print()
        print("### Δ vs the head, on IDENTICAL pairs\n")
        print("| scorer | " + " | ".join(c.replace("|", "\\|") for c in COLS) + " |")
        print("|" + "---|" * (1 + len(COLS)))
        for k in KS:
            row = [f"ROLLOUT_{k}"]
            for c in COLS:
                d = cur[c]["scorers"][f"ROLLOUT_{k}"]["delta_vs_head"]
                tag = "**DET**" if d["detected"] else "ND"
                row.append(f"{d['delta']:+.4f} {ci(d['ci'])} {tag}")
            print("| " + " | ".join(row) + " |")
        print()
        print("### the LABEL CEILING beside every level, and the bars\n")
        print("| quantity | " + " | ".join(c.replace("|", "\\|") for c in COLS) + " |")
        print("|" + "---|" * (1 + len(COLS)))
        rows = [
            ("**measured ceiling (model-free LOWER bound) — `ROLLOUT_16`**",
             lambda c: f"**{f(cur[c]['ceiling_measured_lower_bound']['acc'])}** "
                       f"{ci(cur[c]['ceiling_measured_lower_bound']['ci'])}"),
            ("split-half agreement of the LABEL (two K = 4 halves)",
             lambda c: f(cur[c]["ceiling_split_half_of_label"]["agreement"])),
            ("→ implied oracle acc *(cross-check only)*",
             lambda c: f(cur[c]["ceiling_split_half_of_label"]["implied_oracle_acc"])),
            ("parametric `ACC_oracle(K′=8)` *(cross-check only)*",
             lambda c: f(cur[c]["ceiling_parametric"]["acc_oracle_at_K8"])),
            ("label-noise share of the observed gap variance",
             lambda c: f(cur[c]["ceiling_parametric"]["label_noise_share"])),
            ("recovered **E\\|true gap\\|**",
             lambda c: f(cur[c]["ceiling_parametric"]["E_abs_gap"])),
            ("**BAR K1** — smallest K whose CI lower bound clears the head's point",
             lambda c: f"**{cur[c]['bar_K1_smallest_K_clearing_head_point'] or 'NONE ≤ 16'}**"),
            ("**BAR K2** — smallest K whose CI lower bound clears 0.60",
             lambda c: f"**{cur[c]['bar_K2_smallest_K_clearing_060'] or 'NONE ≤ 16'}**"),
            ("**BAR K4** — the knee (marginal < average gain per rollout)",
             lambda c: f"**K = {cur[c]['bar_K4_knee'] or '> 8'}**"),
            ("n non-tied pairs", lambda c: str(cur[c]["n_nontied"])),
        ]
        for name, fn in rows:
            print("| " + name + " | " + " | ".join(fn(c) for c in COLS) + " |")
        print()
        print("### the MARGINAL table (BAR K3 / K4)\n")
        print("| step | " + " | ".join(c.replace("|", "\\|") for c in COLS) + " |")
        print("|" + "---|" * (1 + len(COLS)))
        for i, k in enumerate(KS[:-1]):
            row = [f"K {k} → {2*k}: Δacc"]
            for c in COLS:
                m = cur[c]["marginal"][i]
                row.append(f"{m['d_acc']:+.4f} ({m['d_acc_per_rollout']:+.4f}/rollout)")
            print("| " + " | ".join(row) + " |")
        print()
        print("### the COST column, measured\n")
        for k, v in sorted(cost.items()):
            print(f"* `{k}` = "
                  f"{v if isinstance(v, (int, str)) else round(float(v), 4)}")
        print()
        print("### the depth split on `top1|top2` (post-hoc, hazard 2 of the control)\n")
        ds = cur["top1|top2"].get("depth_split") or {}
        if ds:
            print("| scorer | " + " | ".join(f"{k} (n={v['n_nontied']})"
                                             for k, v in ds.items()) + " |")
            print("|" + "---|" * (1 + len(ds)))
            for nm in ["headW"] + [f"ROLLOUT_{k}" for k in KS]:
                print("| " + nm + " | " + " | ".join(
                    lvl(v["scorers"][nm]) for v in ds.values()) + " |")
        print()
        print("### join / instrument\n")
        print("```\n" + json.dumps(K["join"], indent=1) + "\n```")
        print("\n### what the policy's own preference is worth, on these forks\n")
        print("```\n" + json.dumps(K["policy_ordering_value"], indent=1) + "\n```")

    if a.rule:
        R = json.load(open(a.rule))
        print(f"\n### the PLAYOFF's OWN decision rule, applied offline to {R['n_forks']} forks' "
              f"CRN-paired rollouts\n")
        print("| R (paired rollouts) | rollouts / decision | CONCLUSIVE rate [95 % CI] | "
              "n conclusive | keeps the policy's action | agrees with the K′ = 8 label |")
        print("|---|---|---|---|---|---|")
        for k in ("1", "2", "4", "8", "16"):
            v = R["R"][k]
            print(f"| **{k}** | {v['rollouts_per_decision']} | "
                  f"**{f(v['conclusive_rate'])}** {ci(v['conclusive_rate_ci'])} | "
                  f"{v['n_conclusive']} | {f(v['kept_policy_when_conclusive'])} | "
                  f"**{f(v['agrees_with_K8_label_when_conclusive'])}** "
                  f"(n={v['n_scored_against_label']}) |")

    if a.battery:
        B = json.load(open(a.battery))
        print("\n### the BATTERY — the rollout-leaf cell and its contemporaneous controls\n")
        print("| cell | battles | pairs | **L2 paired** [normal CI] | [bootstrap CI] | "
              "unpaired Wilson | changed / decision | s / battle |")
        print("|---|---|---|---|---|---|---|---|")
        for nm, c in sorted(B["cells"].items()):
            L = c["L2_paired_normal"]
            W = c["L2_unpaired_wilson"]
            print(f"| **{nm}** | {c['n_battles']} | {L['n']} | **{f(L['mean'])}** "
                  f"{ci(L['ci_normal'])} | {ci(L['ci_boot'])} | "
                  f"{f(W['rate'])} {ci(W['ci'])} | "
                  f"{f(c['rates']['changed_per_decision'])} | "
                  f"{c['mean_wall_s_per_battle']:.0f} |")
        print("\n### the MECHANISM row — the primary read (AMENDMENT §2)\n")
        print("| cell | decisions | screen_decisive | playoffs PLAYED | INCONCLUSIVE | "
              "no_budget | realized R | s / adjudicated decision |")
        print("|---|---|---|---|---|---|---|---|")
        for nm, c in sorted(B["cells"].items()):
            m, r = c["mechanism"], c["rates"]
            print(f"| **{nm}** | {m['n_decisions']} | {m['n_screen_decisive']} | "
                  f"{m['n_playoff']} | {m['n_playoff_inconclusive']} | "
                  f"{m['n_playoff_no_budget']} | {r['mean_realized_R']:.2f} | "
                  f"{r['playoff_wall_s_per_ran_decision']:.1f} |")
        print("\n### paired deltas on shared indices\n")
        for k, v in sorted(B["paired_deltas"].items()):
            if v.get("delta") is None:
                continue
            tag = "**DETECTED**" if v.get("detected") else "NOT DETECTED"
            print(f"* `{k}` = {v['delta']:+.4f} {ci(v['ci_boot'])} {tag} "
                  f"(n = {v['n_shared_pairs']} shared pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
