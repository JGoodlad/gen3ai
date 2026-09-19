#!/usr/bin/env python3
"""Render this read's markdown tables straight from the JSON artifacts.

It exists so that no number in ``README.md`` is hand-transcribed: every figure in the rendered
tables is read out of ``out/*.json``, which are the tools' own outputs. Run it and paste.

The headline table it renders is **THE WITHIN-FOLD TRAJECTORY** — the untaught row and both taught
slices at +1M / +3M / +6.09M / +7.59M / +9.09M / +12.09M in one frame, with the FORK CROSSING
marked between +6.09M and +7.59M.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("out")
BIG5, DDTAR = "U_f6229d2c", "U_9eb3abdc"


def pp(x):
    return f"{x:+.2f}"


def trajectory():
    u = json.loads((OUT / "cont_untaught_delta.json").read_text())
    s = json.loads((OUT / "cont_slice_read.json").read_text())
    print("### THE WITHIN-FOLD TRAJECTORY\n")
    print("| depth (cumulative post-fork) | run | untaught level pp | untaught Δ vs arm W | "
          "Big-5 rate | Big-5 Δ | DDTar rate | DDTar Δ |")
    print("|---|---|---:|---|---:|---|---:|---|")
    for r, row in u["trajectory"].items():
        st = s["trajectory"].get(r, {})
        b, dd = st.get(BIG5, {}), st.get(DDTAR, {})
        if row.get("fork_crossing_immediately_before"):
            print("| — | 🚨 **FORK CROSSING** `ai_v13_07_fold1` → `ai_v13_08_fold1_cont` "
                  "@81,100,800 | — | — | — | — | — | — |")
        ci = row.get("ci95_pp")
        print(f"| **{row['depth_cumulative_post_fork']}** | `{row['run']}` | "
              f"**{row['level_pp']:.2f}** | **{pp(row['delta_vs_armW_pp'])}** "
              f"[{pp(ci[0])}, {pp(ci[1])}] | "
              f"{b.get('win_rate')} | {_d(b)} | {dd.get('win_rate')} | {_d(dd)} |")
    print(f"\n*arm W (the frozen PARENT, +0): untaught "
          f"{u['levels_pp']['armW']:.2f} pp · Big-5 "
          f"{s['levels'][BIG5]['armW']['win_rate']} · DDTar "
          f"{s['levels'][DDTAR]['armW']['win_rate']}.*")
    print(f"*W_b (the seed floor arm, +0): untaught {u['levels_pp']['armWb']:.2f} pp · Big-5 "
          f"{s['levels'][BIG5]['armWb']['win_rate']} · DDTar "
          f"{s['levels'][DDTAR]['armWb']['win_rate']}.*")


def _d(cell):
    if not cell or cell.get("delta_vs_armW") is None:
        return "—"
    lo, hi = cell["newcombe95"]
    return f"{cell['delta_vs_armW']:+.4f} [{lo:+.4f}, {hi:+.4f}]"


def untaught():
    d = json.loads((OUT / "cont_untaught_delta.json").read_text())
    print("\n### ROW 1 levels (pp), per team\n")
    order = ["fold_p1M", "fold_p3M", "fold_p6M", "cont_p7_5M", "cont_p9M", "cont_p12M",
             "armW", "armWb"]
    teams = list(next(iter(d["per_team_pp"].values())).keys())
    print("| ref | level pp | wins/finished | " + " | ".join(f"`{t}`" for t in teams) + " |")
    print("|---|---:|---:|" + "---:|" * len(teams))
    for r in order:
        if r not in d["levels_pp"]:
            continue
        row = d["per_team_pp"][r]
        print(f"| `{r}` | **{d['levels_pp'][r]:.2f}** | {d['wins_over_finished'][r]} | "
              + " | ".join(f"{row[t]:.2f}" for t in teams) + " |")
    print("\n### ROW 1 contrasts\n")
    print("| contrast | Δ pp | CI95 | teams favouring first | verdict |")
    print("|---|---:|---|---:|---|")
    for k, c in d["contrasts"].items():
        v = d["verdicts"].get(k.split("_minus_")[0], {})
        print(f"| {c['label']} | **{pp(c['delta_pp'])}** | [{pp(c['ci95_pp'][0])}, "
              f"{pp(c['ci95_pp'][1])}] | {c['teams_favouring_first']} of {c['n_teams']} | "
              f"{v.get('verdict', '—')} |")


def slice_rows():
    d = json.loads((OUT / "cont_slice_read.json").read_text())
    for t, lab in ((BIG5, "BIG-5 (BALANCE)"), (DDTAR, "DDTAR (OFFENSE)")):
        print(f"\n### ROW 2 — the {lab} slice\n")
        print("| ref | win rate | Wilson 95 % | wins/800 | source |")
        print("|---|---:|---|---:|---|")
        for r, c in sorted(d["levels"][t].items(), key=lambda kv: -kv[1]["win_rate"]):
            src = "IMPORTED" if c["imported"] else "measured here"
            print(f"| `{r}` | **{c['win_rate']:.4f}** | [{c['wilson95'][0]:.4f}, "
                  f"{c['wilson95'][1]:.4f}] | {c['wins']} | {src} |")
        print("\n| contrast | Δ | Newcombe 95 % | floor | verdict |")
        print("|---|---:|---|---:|---|")
        for k, c in d["per_team"][t]["contrasts"].items():
            if not c:
                continue
            v = d["per_team"][t].get("verdicts", {}).get(k, {})
            print(f"| {c['label']} | **{c['delta']:+.4f}** | [{c['newcombe95'][0]:+.4f}, "
                  f"{c['newcombe95'][1]:+.4f}] | {v.get('floor_on_this_cell', '—')} | "
                  f"{v.get('verdict', '—')} |")


if __name__ == "__main__":
    trajectory()
    untaught()
    slice_rows()
