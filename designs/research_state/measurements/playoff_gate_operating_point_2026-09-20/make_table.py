"""Renders `gate_curve.json` + `battery.json` into the README's tables.

    python3 make_table.py --curve gate_curve.json --battery battery.json > tables.md

NO NUMBER IN THE README IS RETYPED BY HAND. That is the whole job of this file, and it is the same
discipline the 2026-09-19 read used: a table transcribed by hand is a table that drifts from its
JSON the first time either is edited.
"""
from __future__ import annotations

import argparse
import json


def f(x, n=4, sign=False):
    if x is None:
        return "—"
    return f"{x:+.{n}f}" if sign else f"{x:.{n}f}"


def ci(c, n=4, sign=True):
    return "—" if not c else f"[{f(c[0], n, sign)}, {f(c[1], n, sign)}]"


def curve_tables(j) -> str:
    o = j["oracle_gain_per_decision"]
    L = []
    L.append("### the GATE'S OPERATING CURVE — the production rule, imported, swept "
             "on the 665 banked forks\n")
    L.append(f"*The label is the banked K′ = 8 mean, DISJOINT from the gate's fresh dice. "
             f"E\\|label gap\\| = **{f(j['e_abs_label_gap'])}**, "
             f"{f(j['frac_label_tied'], 3)} of pairs label-TIED, so the "
             f"**ORACLE CEILING on the per-decision gain is {f(o, 5, True)}** "
             f"{ci(j['oracle_gain_per_decision_ci'], 5)} — every EV below is quoted as a "
             f"fraction of it.*\n")
    L.append("| R | SE mult | resolve rate [95 % CI] | overrules / resolved | agrees w/ K′=8 label"
             " | agrees w/ 16-roll ⚠️ | EV gain / RESOLVED | **EV gain / DECISION** [95 % CI] |"
             " × oracle |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for R in (4, 8, 16):
        for k in (2.0, 1.5, 1.0, 0.5):
            c = j["gated"][f"R{R}_k{k}_mp4"]
            tag = " **DET**" if c["ev_detected"] else " ND"
            star = " ⬅ **CHOSEN**" if (R == 4 and k == 0.5) else (
                " *(production)*" if (R == 4 and k == 2.0) else "")
            L.append(f"| **{R}** | **{k:g}**{star} | {f(c['resolve_rate'])} "
                     f"{ci(c['resolve_rate_ci'], 4, False)} | "
                     f"{f(c['overrule_rate_of_resolved'])} | "
                     f"{f(c['agree_label_when_resolved'])} (n={c['n_agree_label']}) | "
                     f"{f(c['agree_r16_when_resolved'])} | "
                     f"{f(c['ev_gain_per_resolved'], 4, True)} | "
                     f"**{f(c['ev_gain_per_decision'], 5, True)}** "
                     f"{ci(c['ev_gain_per_decision_ci'], 5)}{tag} | "
                     f"{f(c['frac_of_oracle'], 3)}× |")
        a = j["always"][f"R{R}"]
        L.append(f"| **{R}** | **NO GATE** *(always re-rank)* | {f(a['resolve_rate'])} "
                 f"{ci(a['resolve_rate_ci'], 4, False)} | "
                 f"{f(a['overrule_rate_of_resolved'])} | "
                 f"{f(a['agree_label_when_resolved'])} (n={a['n_agree_label']}) | "
                 f"{f(a['agree_r16_when_resolved'])} | "
                 f"{f(a['ev_gain_per_resolved'], 4, True)} | "
                 f"**{f(a['ev_gain_per_decision'], 5, True)}** "
                 f"{ci(a['ev_gain_per_decision_ci'], 5)}"
                 f"{' **DET**' if a['ev_detected'] else ' ND'} | "
                 f"{f(a['frac_of_oracle'], 3)}× |")

    L.append("\n### the MIN_PAIRS axis — INERT at R ≥ 4 by construction, and what it does where "
             "it bites\n")
    L.append("| R | SE mult | resolve, MIN_PAIRS = 2 | resolve, MIN_PAIRS = 4 | identical? |")
    L.append("|---|---|---|---|---|")
    for R in j["grid"]["R"]:
        for k in j["grid"]["se_multiple"]:
            a = j["gated"][f"R{R}_k{k}_mp2"]
            b = j["gated"][f"R{R}_k{k}_mp4"]
            same = j["min_pairs_identical"][f"R{R}_k{k}"]
            L.append(f"| {R} | {k:g} | {f(a['resolve_rate'])} | {f(b['resolve_rate'])} | "
                     f"{'**yes**' if same else 'no — MIN_PAIRS BITES'} |")

    L.append("\n### the two points the instruction asked for, named\n")
    best = max((c for c in j["gated"].values() if c["R"] == 4 and c["min_pairs"] == 4),
               key=lambda c: c["ev_gain_per_decision"])
    bestany = max((c for c in j["gated"].values() if c["min_pairs"] == 4),
                  key=lambda c: c["ev_gain_per_decision"])
    ok80 = [c for c in j["gated"].values()
            if c["min_pairs"] == 4 and (c["agree_label_when_resolved"] or 0) >= 0.80]
    L.append("| question | answer |")
    L.append("|---|---|")
    L.append(f"| the point maximising **EV gain × resolve rate** (= EV gain per DECISION) at "
             f"R = 4 — **the point the live battery ran** | **R = 4, SE mult = "
             f"{best['se_multiple']:g}**: {f(best['ev_gain_per_decision'], 5, True)} per decision "
             f"{ci(best['ev_gain_per_decision_ci'], 5)}, resolve {f(best['resolve_rate'])}, "
             f"{f(best['frac_of_oracle'], 3)}× the oracle |")
    L.append(f"| the same, over the WHOLE grid (cost ignored) | **R = {bestany['R']}, SE mult = "
             f"{bestany['se_multiple']:g}**: {f(bestany['ev_gain_per_decision'], 5, True)} per "
             f"decision, resolve {f(bestany['resolve_rate'])}, "
             f"{f(bestany['frac_of_oracle'], 3)}× the oracle |")
    L.append("| the point(s) where **agreement with the label stays ≥ 0.80** | "
             + (", ".join(f"**R = {c['R']}, SE mult = {c['se_multiple']:g}** "
                          f"({f(c['agree_label_when_resolved'])} at resolve "
                          f"{f(c['resolve_rate'])}, EV/decision "
                          f"{f(c['ev_gain_per_decision'], 5, True)})"
                          for c in sorted(ok80, key=lambda c: (c["R"], -c["se_multiple"])))
                 or "none") + " |")
    return "\n".join(L)


def battery_tables(j) -> str:
    L = []
    L.append("### the BATTERY — the playoff cell at the chosen point and its two "
             "contemporaneous controls\n")
    L.append("| cell | battles | unfin | pairs | **L2 paired** [normal CI] | [bootstrap CI] | "
             "unpaired Wilson | changed / decision | s / battle |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for name, c in sorted(j["cells"].items()):
        p = c["L2_paired_normal"]
        w = c["L2_unpaired_wilson"]
        L.append(f"| **{name}** | {c['n_battles']} | {c['n_unfinished']} | {p['n']} | "
                 f"**{f(p['mean'])}** {ci(p['ci_normal'], 4, False)} | "
                 f"{ci(p['ci_boot'], 4, False)} | "
                 f"{f(w['rate'])} {ci(w['ci'], 4, False)} | "
                 f"{f(c['rates']['changed_per_decision'])} | "
                 f"{c['mean_wall_s_per_battle']:.0f} |")

    L.append("\n### the MECHANISM row — the primary read\n")
    L.append("| cell | decisions | screen_decisive | playoffs RUN | RESOLVED (played) | "
             "inconclusive | no_budget | failed rollouts | **live resolve rate** | realized R | "
             "s / adjudicated decision |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, c in sorted(j["cells"].items()):
        m, r = c["mechanism"], c["rates"]
        L.append(f"| **{name}** | {m['n_decisions']} | {m['n_screen_decisive']} | "
                 f"{m['n_playoff_ran']} | {m['n_playoff']} | {m['n_playoff_inconclusive']} | "
                 f"{m['n_playoff_no_budget']} | {m['n_playoff_failed']} | "
                 f"**{f(r['playoff_resolve_rate_of_run'])}** | {r['mean_realized_R']:.2f} | "
                 f"{r['playoff_wall_s_per_ran_decision']:.1f} |")

    L.append("\n### paired deltas on SHARED game indices\n")
    for k, d in sorted(j["paired_deltas"].items()):
        if d.get("delta") is None:
            L.append(f"* `{k}` — fewer than 2 shared pairs (n = {d['n_shared_pairs']})")
        else:
            L.append(f"* `{k}` = {f(d['delta'], 4, True)} {ci(d['ci_boot'])} "
                     f"{'**DETECTED**' if d['detected'] else 'NOT DETECTED'} "
                     f"(n = {d['n_shared_pairs']} shared pairs)")
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--curve", required=True)
    ap.add_argument("--battery", default="")
    args = ap.parse_args(argv)
    print(curve_tables(json.load(open(args.curve))))
    if args.battery:
        print()
        print(battery_tables(json.load(open(args.battery))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
