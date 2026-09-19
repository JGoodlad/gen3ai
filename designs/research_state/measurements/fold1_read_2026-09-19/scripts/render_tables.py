#!/usr/bin/env python3
"""Render this read's markdown tables straight from the JSON artifacts.

It exists so that no number in ``README.md`` is hand-transcribed: every figure in the rendered
tables is read out of ``out/*.json``, which are the tools' own outputs. Run it and paste.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("out")


def pp(x):
    return f"{x:+.2f}"


def untaught():
    d = json.loads((OUT / "fold_untaught_delta.json").read_text())
    print("### levels (pp)\n")
    order = ["fold_p1M", "fold_p3M", "fold_p6M", "armW", "armWb"]
    teams = list(next(iter(d["per_team_pp"].values())).keys())
    print("| ref | level pp | wins/finished | " + " | ".join(teams) + " |")
    print("|---|---:|---:|" + "---:|" * len(teams))
    for r in order:
        if r not in d["levels_pp"]:
            continue
        row = d["per_team_pp"][r]
        print(f"| `{r}` | **{d['levels_pp'][r]:.2f}** | {d['wins_over_finished'][r]} | "
              + " | ".join(f"{row[t]:.2f}" for t in teams) + " |")
    print("\n### contrasts\n")
    print("| contrast | Δ pp | CI95 | teams favouring first | floor | verdict |")
    print("|---|---:|---|---:|---:|---|")
    for k, c in d["contrasts"].items():
        v = d["verdicts"].get(k.split("_minus_")[0], {}) if k.endswith("_minus_armW") else {}
        fl = v.get("floor_pp", "")
        ver = v.get("verdict", "")
        print(f"| {c['label']} | **{pp(c['delta_pp'])}** | [{pp(c['ci95_pp'][0])}, "
              f"{pp(c['ci95_pp'][1])}] | {c['teams_favouring_first']} of {c['n_teams']} | {fl} | {ver} |")
    print("\n### reproduction\n")
    print(json.dumps(d["reproduction"], indent=1))
    print("\n### branch clauses\n")
    print(json.dumps(d["row1_branch_clauses"], indent=1))


def slice_():
    d = json.loads((OUT / "fold_slice_read.json").read_text())
    for t, rows in d["levels"].items():
        print(f"\n### {t} — {d['team_labels'].get(t, '')}\n")
        print("| ref | win rate | Wilson 95 % | wins/finished | timeouts |")
        print("|---|---:|---|---:|---:|")
        for r in ("t1_big5", "t2_ddtar", "fold_p6M", "fold_p3M", "armW", "armWb"):
            if r not in rows:
                continue
            c = rows[r]
            print(f"| `{r}` | **{c['win_rate']:.4f}** | [{c['wilson95'][0]:.4f}, "
                  f"{c['wilson95'][1]:.4f}] | {c['wins']}/{c['finished']} | {c['timeouts']} |")
        print("\n| contrast | Δ | Newcombe 95 % |")
        print("|---|---:|---|")
        for k, c in d["per_team"][t]["contrasts"].items():
            if not c:
                continue
            print(f"| {c['label']} | **{c['delta']:+.4f}** | [{c['newcombe95'][0]:+.4f}, "
                  f"{c['newcombe95'][1]:+.4f}] |")
        print("\n**verdicts**")
        print(json.dumps(d["per_team"][t].get("verdicts", {}), indent=1))
    print("\n### the divergence\n")
    print(json.dumps(d.get("the_divergence", {}), indent=1))
    print("\n### pooled (SECONDARY)\n")
    print(json.dumps(d["pooled_SECONDARY"]["levels"], indent=1))


if __name__ == "__main__":
    which = sys.argv[2] if len(sys.argv) > 2 else "all"
    if which in ("all", "untaught") and (OUT / "fold_untaught_delta.json").exists():
        untaught()
    if which in ("all", "slice") and (OUT / "fold_slice_read.json").exists():
        slice_()
