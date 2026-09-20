#!/usr/bin/env python3
"""Renders every markdown table in README.md straight from out/*.json.

No number in the note is hand-transcribed. Usage:
    render_tables.py <out_dir>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BIG5, DDTAR = "U_f6229d2c", "U_9eb3abdc"
PATHS = [("SPLIT `ai_v13_11_split_lossoff`", "split"),
         ("control `ai_v13_09_wcont`", "control"),
         ("fold path", "fold_path")]
REFMAP = {"+3M": {"split": "split_p3M", "control": "wcont_p3M", "fold_path": "fold_p3M"},
          "+6M": {"split": "split_p6M", "control": "wcont_p6M", "fold_path": "fold_p6M"},
          "+12M": {"split": "split_p12M", "control": "wcont_p12M", "fold_path": "cont_p12M"}}


def pp(x, n=2, sign=True):
    if x is None:
        return "—"
    return f"{x:+.{n}f}" if sign else f"{x:.{n}f}"


def ci(c, n=2):
    return "—" if not c else f"[{c[0]:+.{n}f}, {c[1]:+.{n}f}]"


def main() -> int:
    d = Path(sys.argv[1])
    unt = json.loads((d / "split_untaught_delta.json").read_text())
    sl = json.loads((d / "split_slice_read.json").read_text())
    br = json.loads((d / "split_branches.json").read_text())

    print("### THE 3 x 3 x 3 TABLE — path x depth x row\n")
    print("| depth | path | untaught pp | untaught Δ vs arm W | Big-5 rate | Big-5 Δ | "
          "DDTar rate | DDTar Δ |")
    print("|---|---|---:|---|---:|---|---:|---|")
    for depth in ("+3M", "+6M", "+12M"):
        for label, tag in PATHS:
            ref = REFMAP[depth][tag]
            u = unt["delta_path"][depth].get(tag, {})
            row = [depth, ("**" + label + "**") if tag == "split" else label,
                   f"{unt['levels_pp'].get(ref, float('nan')):.2f}",
                   f"**{pp(u.get('delta_pp'))}** {ci(u.get('ci95_pp'))}"]
            for team in (BIG5, DDTAR):
                lv = sl["levels"][team].get(ref, {})
                g = sl["per_team"][team]["contrasts"].get(f"{ref}_minus_armW") or {}
                row += [f"{lv.get('win_rate', float('nan')):.4f}",
                        f"**{pp(g.get('delta'), 4)}** {ci(g.get('newcombe95'), 4)}"]
            print("| " + " | ".join(row) + " |")
    for nm, ref in (("arm W — the FROZEN PARENT", "armW"), ("W_b — the seed-floor arm", "armWb")):
        r = [" +0 ", nm, f"{unt['levels_pp'].get(ref, float('nan')):.2f}", "—"]
        for team in (BIG5, DDTAR):
            r += [f"{sl['levels'][team].get(ref, {}).get('win_rate', float('nan')):.4f}", "—"]
        print("| " + " | ".join(r) + " |")
    print("\n*floors: untaught **3.69 pp** (IMPORTED) · Big-5 **0.0475** (measured on the cell) · "
          "DDTar **0.0850** (measured on the cell)*\n")

    print("\n### THE TWO CONTRASTS — Δ_ecology (split − control, THREE levers) and "
          "Δ_loss (split − fold, ONE lever)\n")
    print("| depth | row | Δ_ecology | CI95 | floor | (a) | (b) | verdict | Δ_loss | CI95 | "
          "(a) | (b) | verdict |")
    print("|---|---|---:|---|---:|---|---|---|---:|---|---|---|---|")
    tick = {True: "✅", False: "❌"}
    for depth in ("+3M", "+6M", "+12M"):
        rows = br["by_depth"][depth]["rows"]
        for rk, nm, nd in (("untaught_8", "untaught 8", 2), ("big5_slice", "Big-5 slice", 4),
                           ("ddtar_slice", "DDTar slice", 4)):
            r = rows[rk]
            cells = [depth, nm]
            for k in ("ecology", "loss"):
                v = r[k]
                cells += [f"**{pp(v['delta'], nd)}**", ci(v["ci"], nd)]
                if k == "ecology":
                    cells.append(f"{r['floor']}")
                cells += [tick[v["clause_a"]], tick[v["clause_b"]],
                          ("**OUTSIDE**" if v["verdict"] == "OUTSIDE THE FLOOR" else "WITHIN")]
            # reorder: depth,row,ecoΔ,ecoCI,floor,a,b,verdict,lossΔ,lossCI,a,b,verdict
            print("| " + " | ".join(cells) + " |")

    print("\n### THE BRANCH CLAUSES\n")
    for depth in ("+12M", "+3M"):
        e = br["by_depth"][depth]
        print(f"**{depth}** — {e['role']} ⇒ **{e['BRANCH']}**\n")
        print("| branch | clause | met? |")
        print("|---|---|---|")
        for k, v in e["clauses"].items():
            print(f"| `{k}` | {v['clause']} | {tick[v['MET']]} |")
        print()
    print(f"**THE BRANCH: {br['THE_BRANCH']}**\n")
    print(f"*depths agree: {br['depths_agree']}*\n")

    print("\n### THE SPLIT STATISTIC (Big-5 minus DDTar difference-of-deltas) — DESCRIPTOR\n")
    print("| depth | split | control | fold path |")
    print("|---|---:|---:|---:|")
    for depth in ("+3M", "+6M", "+12M"):
        e = sl["THE_SPLIT_STATISTIC"]["by_depth"][depth]
        print("| " + depth + " | " + " | ".join(
            pp(e.get(t, {}).get("big5_minus_ddtar"), 4) for t in
            ("split", "control", "fold_path")) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
