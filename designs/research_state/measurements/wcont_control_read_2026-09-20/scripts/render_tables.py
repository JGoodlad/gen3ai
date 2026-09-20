#!/usr/bin/env python3
"""Render every markdown table in README.md straight from out/*.json — nothing is hand-transcribed.

Usage: render_tables.py <out_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

TEAMS = ["U_61590463", "U_92832108", "U_ce35b736", "U_9909f2e9",
         "U_9d5f8458", "U_f7ba5702", "U_90b94599", "U_dbf81d8e"]
BIG5, DDTAR = "U_f6229d2c", "U_9eb3abdc"
DEPTHS = ["+3M", "+6M", "+12M"]


def ci(v):
    return f"[{v[0]:+.2f}, {v[1]:+.2f}]"


def ci4(v):
    return f"[{v[0]:+.4f}, {v[1]:+.4f}]"


def main() -> int:
    out = Path(sys.argv[1])
    u = json.loads((out / "wcont_untaught_delta.json").read_text())
    s = json.loads((out / "wcont_slice_read.json").read_text())
    a = json.loads((out / "wcont_away_anchor.json").read_text())
    r = json.loads((out / "wcont_run_rows.json").read_text())

    P = print
    P("### THE 2 x 3 x 3 TABLE — path x depth x row\n")
    P("| depth | path | untaught pp | untaught Δ vs arm W | Big-5 rate | Big-5 Δ | DDTar rate | DDTar Δ |")
    P("|---|---|---:|---|---:|---|---:|---|")
    for d in DEPTHS:
        for tag, name in (("control", "**CONTROL** `ai_v13_09_wcont`"),
                          ("fold_path", "fold path")):
            uu = u["delta_path"][d].get(tag)
            tr = s["trajectory"][d][tag]
            b, t = tr[BIG5], tr[DDTAR]
            P(f"| **{d}** | {name} | {uu['level_pp']:.2f} | **{uu['delta_pp']:+.2f}** {ci(uu['ci95_pp'])} "
              f"| {b['win_rate']:.4f} | **{b['delta_vs_armW']:+.4f}** {ci4(b['newcombe95'])} "
              f"| {t['win_rate']:.4f} | **{t['delta_vs_armW']:+.4f}** {ci4(t['newcombe95'])} |")
    aw = u["levels_pp"]["armW"]
    lb = s["levels"][BIG5]["armW"]["win_rate"]
    lt = s["levels"][DDTAR]["armW"]["win_rate"]
    P(f"| +0 | arm W — the FROZEN PARENT | {aw:.2f} | — | {lb:.4f} | — | {lt:.4f} | — |")
    wb = u["levels_pp"]["armWb"]
    P(f"| +0 | W_b — the seed-floor arm | {wb:.2f} | — | {s['levels'][BIG5]['armWb']['win_rate']:.4f} "
      f"| — | {s['levels'][DDTAR]['armWb']['win_rate']:.4f} | — |")
    P(f"\n*floors: untaught **{u['imported_floor_pp']} pp** (IMPORTED) · Big-5 "
      f"**{abs(s['per_team'][BIG5]['contrasts']['THE_FLOOR_armW_minus_armWb']['delta']):.4f}** "
      f"(measured on the cell) · DDTar "
      f"**{abs(s['per_team'][DDTAR]['contrasts']['THE_FLOOR_armW_minus_armWb']['delta']):.4f}** "
      f"(measured on the cell)*")

    P("\n### THE VERDICT TABLE\n")
    P("| # | row | finding | floor | (a) | (b) | verdict |")
    P("|---|---|---|---:|---|---|---|")
    n = 0
    for d in DEPTHS:
        v = u["delta_path"][d]["control"]
        n += 1
        P(f"| **1{chr(96+n)}** | untaught, CONTROL, {d} | **{v['delta_pp']:+.2f} pp** {ci(v['ci95_pp'])}, "
          f"{v['teams_favouring_over_armW']} of 8 | {v['floor_pp']} | "
          f"{'✅' if v['clause_a_abs_delta_gt_floor'] else '❌'} | "
          f"{'✅' if v['clause_b_ci_excludes_floor_point'] else '❌'} | **{v['verdict']}** |")
    for d in DEPTHS:
        v = u["delta_extract"][d]
        P(f"| **X{d}** | 🚨 EXTRACTION, untaught, {d} (fold − control) | **{v['delta_pp']:+.2f} pp** "
          f"{ci(v['ci95_pp'])}, {v['teams_favouring_fold_path']} of 8 | {v['floor_pp']} | "
          f"{'✅' if v['clause_a_abs_delta_gt_floor'] else '❌'} | "
          f"{'✅' if v['clause_b_ci_excludes_floor_point'] else '❌'} | **{v['verdict']}** |")
    for team, lab in ((BIG5, "BIG-5"), (DDTAR, "DDTAR")):
        for d, ref in zip(DEPTHS, ("wcont_p3M", "wcont_p6M", "wcont_p12M")):
            v = s["per_team"][team]["verdicts"][f"{ref}_minus_armW"]
            P(f"| **2** | per-slice {lab}, CONTROL, {d} | **{v['delta']:+.4f}** {ci4(v['newcombe95'])} "
              f"| {v['floor_on_this_cell']} | {'✅' if v['clause_a_abs_delta_gt_floor'] else '❌'} | "
              f"{'✅' if v['clause_b_ci_excludes_floor_point'] else '❌'} | **{v['verdict']}** |")
        for d in DEPTHS:
            v = s["per_team"][team]["verdicts"][f"EXTRACT_{d}"]
            P(f"| **X** | 🚨 EXTRACTION, {lab}, {d} (fold − control) | **{v['delta']:+.4f}** "
              f"{ci4(v['newcombe95'])} | {v['floor_on_this_cell']} | "
              f"{'✅' if v['clause_a_abs_delta_gt_floor'] else '❌'} | "
              f"{'✅' if v['clause_b_ci_excludes_floor_point'] else '❌'} | **{v['verdict']}** |")
    c = a["contrasts"]["wcont_p12M_minus_armW"]
    P(f"| **3** | `SmallRL` greedy away, 100 games | **{a['wcont_p12M']['win_rate']:+.3f}** ⇒ Δ "
      f"**{c['delta']:+.3f}** [{c['newcombe95'][0]:+.3f}, {c['newcombe95'][1]:+.3f}] | "
      f"{a['floor']['value']} | {'✅' if c['clause_a_abs_delta_gt_floor'] else '❌'} | "
      f"{'✅' if c['clause_b_ci_excludes_floor_point'] else '❌'} | **{c['verdict']}** |")

    P("\n### ROW 1 — every team\n")
    hdr = " | ".join(f"`{t}`" for t in TEAMS)
    P(f"| ref | level | wins/finished | {hdr} |")
    P("|---|---:|---:|" + "---:|" * 8)
    order = ["wcont_p12M", "wcont_p6M", "wcont_p3M", "cont_p12M", "fold_p3M", "fold_p6M",
             "cont_p9M", "cont_p7_5M", "fold_p1M", "armWb", "armW"]
    for ref in order:
        if ref not in u["levels_pp"]:
            continue
        row = " | ".join(f"{u['per_team_pp'][ref][t]:.2f}" for t in TEAMS)
        P(f"| **{ref}** | **{u['levels_pp'][ref]:.2f}** | {u['wins_over_finished'][ref]} | {row} |")

    P("\n### ROW 2 — every cell\n")
    for team, lab in ((BIG5, "BIG-5 (BALANCE), teacher t1"), (DDTAR, "DDTar (OFFENSE), teacher t2")):
        P(f"\n**{lab}** — `{team}`\n")
        P("| ref | win rate | Wilson 95 % | wins/800 | source |")
        P("|---|---:|---|---:|---|")
        for ref, v in sorted(s["levels"][team].items(), key=lambda kv: -kv[1]["win_rate"]):
            P(f"| {'**' + ref + '**' if ref.startswith('wcont') else ref} | {v['win_rate']:.4f} | "
              f"[{v['wilson95'][0]:.4f}, {v['wilson95'][1]:.4f}] | {v['wins']} | "
              f"{'IMPORTED' if v['imported'] else 'measured here'} |")

    P("\n### THE SPLIT\n")
    P("| depth | path | Big-5 Δ | DDTar Δ | Big-5 − DDTar | sign pattern = the fold's? |")
    P("|---|---|---:|---:|---:|---|")
    for d in DEPTHS:
        for tag, name in (("control", "**CONTROL**"), ("fold_path", "fold path")):
            e = s["THE_SPLIT"]["by_depth"][d][tag]
            P(f"| {d} | {name} | {e['big5_delta']:+.4f} | {e['ddtar_delta']:+.4f} | "
              f"**{e['big5_minus_ddtar']:+.4f}** | {'YES' if e['sign_pattern_matches_the_fold'] else '**NO — both POSITIVE**'} |")

    P("\n### ROW 4 / ROW 5 — the descriptors\n")
    e = r["row4_entropy"]
    P(f"* **entropy** `H_end` control **{e['H_end_here_median_last_20']}** (banked "
      f"{e['H_end_banked_ledger_e77963c4']}, reproduces={e['reproduces_banked']}) vs fold path "
      f"**{e['fold_path_H_end_banked']}** ⇒ gap **{e['gap_control_minus_fold_path']:+.4f}** nats = "
      f"**{e['multiple_of_the_floor']}x** the 75M run-level floor {e['run_level_entropy_floor_75M_armW_vs_Wb']}")
    pc = r["row5_promotions"]["control_ai_v13_09_wcont"]
    pf = r["row5_promotions"]["fold_ai_v13_07_fold1"]
    P(f"* **promotions** control **{pc['own_post_fork_promotions']}** post-fork "
      f"({pc['own_promotion_steps']}), `snapshot_ladder/` present={pc['snapshot_ladder_present']}; "
      f"fold path **{pf['own_post_fork_promotions']}**, present={pf['snapshot_ladder_present']}")
    P("\n| step | +steps | control `win_rate_vs_pool` | control `win_rate_vs_bots` |")
    P("|---:|---:|---:|---:|")
    wp = dict(r["series_post_fork"]["eval/win_rate_vs_pool"])
    wb2 = dict(r["series_post_fork"]["eval/win_rate_vs_bots"])
    for st in sorted(wp):
        P(f"| {st:,} | +{(st - r['fork_step_the_cut']) / 1e6:.1f}M | **{wp[st]:.3f}** | {wb2[st]:.4f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
