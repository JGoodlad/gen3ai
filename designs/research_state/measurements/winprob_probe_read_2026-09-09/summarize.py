"""Fold the tmp outputs into the SMALL committed artifacts + print README-ready tables.

Only summary JSON is committed — never the feature arrays (295 MB of raw obs per substrate) and
never the per-state table. `run.sh` regenerates everything from `models/` in ~40 min.

The READING RULE is applied here rather than by eye, so the verdict per (substrate, target) is a
function of the numbers and not of the author:

  DECODES(fset, bucket) := the score's battle-clustered CI lower bound is above the permutation
                           null's p95 for that cell.
  (i)   DISCARDED       raw DECODES at t1_3 and pooled does NOT, and delta(pooled-raw) CI is
                        entirely below zero.
  (ii)  NOT OBSERVABLE  nothing DECODES at t1_3, and at least one set DECODES late.
  (iii) HEAD IGNORES IT pooled DECODES at t1_3 (the win head's own input carries it) while V's
                        own decode of the same target is at or below its null.
"""
from __future__ import annotations

import argparse
import json
import os

FSETS = ("raw", "pi", "vf", "pooled", "V")
BUCKET_ORDER = ("t1", "t1_3", "t4_10", "t11_24", "t25p")
BUCKET_LABEL = {"t1": "turn 1", "t1_3": "turns 1–3", "t4_10": "turns 4–10",
                "t11_24": "turns 11–24", "t25p": "turns ≥25"}


def decodes(cell, k):
    lo = (cell.get("ci") or {}).get(k, [None, None])[0]
    p95 = ((cell.get("null") or {}).get(k) or {}).get("p95")
    return bool(lo is not None and p95 is not None and lo > p95)


def read(cells, bucket, target):
    for c in cells:
        if c.get("bucket") == bucket and c.get("target") == target and "score" in c:
            return c
    return None


def verdict(cells, target):
    c = read(cells, "t1_3", target)
    if c is None:
        return "INCONCLUSIVE (no turn-1–3 cell)"
    d_raw, d_pool, d_V = decodes(c, "raw"), decodes(c, "pooled"), decodes(c, "V")
    dlo, dhi = (c.get("delta_ci") or {}).get("pooled-raw", [None, None])
    late = any(decodes(x, k) for k in FSETS
               for x in [read(cells, b, target) for b in ("t11_24", "t25p")] if x)
    if d_pool and not d_V:
        return "(iii) THE HEAD HAS IT AND DOES NOT USE IT"
    if d_raw and not d_pool and dhi is not None and dhi < 0:
        return "(i) THE NETWORK DISCARDS IT"
    if not d_raw and not d_pool and late:
        return "(ii) NOT OBSERVABLE EARLY"
    if d_pool and d_V:
        return "(iii-partial) both the value path AND V carry it"
    return "MIXED — read the table"


def table(cells, target, title):
    out = [f"\n### {title}\n",
           "| bucket | n states | n battles | RAW | TRUNK (pi) | VALUE (vf) | POOLED | V | "
           "Δ(pooled−raw) | Δ(pooled−vf) |", "|---|---|---|---|---|---|---|---|---|---|"]
    for b in BUCKET_ORDER:
        c = read(cells, b, target)
        if c is None:
            continue

        def cell(k):
            s, ci = c["score"][k], c["ci"][k]
            mark = "**" if decodes(c, k) else ""
            null = c["null"][k]["p95"]
            return f"{mark}{s}{mark} [{ci[0]}, {ci[1]}] (null≤{null})"
        d1 = c["delta_ci"].get("pooled-raw", [None, None])
        d2 = c["delta_ci"].get("pooled-vf", [None, None])
        out.append(f"| {BUCKET_LABEL[b]} | {c['n_states']} | {c['n_battles']} | "
                   + " | ".join(cell(k) for k in FSETS)
                   + f" | [{d1[0]}, {d1[1]}] | [{d2[0]}, {d2[1]}] |")
    return "\n".join(out)


def main(a):
    small = {"generated": "run.sh", "substrates": {}}
    md = []
    for name, tag in (("A", "arm A — ai_v12_02_winprob_critic @ 74M"),
                      ("CTRL", "ctrl — ai_v12_11_ladder_ctrl10M @ 10M")):
        dec = json.load(open(os.path.join(a.tmp, f"decode_{name}.json")))
        ext = json.load(open(os.path.join(a.tmp, name, "extract_meta.json")))
        mlp_path = os.path.join(a.tmp, f"mlp_{name}.json")
        mlp = json.load(open(mlp_path)) if os.path.exists(mlp_path) else None
        cells = dec["cells"]
        sub = {"label": tag, "run": ext["run"], "snapshot": ext["snapshot"],
               "n_states": ext["n_states"], "n_battles": ext["n_battles"],
               "n_teams": ext["n_teams"], "n_draw_excluded": ext["n_draw_excluded"],
               "qc_exact_max_abs_Vfwd_minus_Vrec": ext.get("qc_exact_max_abs_Vfwd_minus_Vrec"),
               "refusals": ext["refusals"], "cells": cells,
               "verdict": {t: verdict(cells, t)
                           for t in ("opp_elo", "opp_class", "own_team_wr", "own_team_id")},
               "mlp": (mlp or {}).get("cells")}
        small["substrates"][name] = sub
        md.append(f"\n## {tag}\n")
        md.append(f"states {ext['n_states']} · battles {ext['n_battles']} · teams "
                  f"{ext['n_teams']} · draws excluded {ext['n_draw_excluded']} · "
                  f"QC max|V_fwd−V_rec| on the exact cycle "
                  f"{ext.get('qc_exact_max_abs_Vfwd_minus_Vrec'):.2e}")
        for t, ttl in (("opp_elo", "opponent Elo (R²)"), ("opp_class", "opponent class (AUC)"),
                       ("own_team_wr", "own team's LOO win rate (R²)"),
                       ("own_team_id", "own team identity, macro one-vs-rest (AUC)")):
            md.append(table(cells, t, ttl))
            md.append(f"\n**Reading: {sub['verdict'][t]}**\n")
        if mlp:
            md.append("\n### MLP check (turns 1–3)\n")
            md.append("| target | RAW | TRUNK | VALUE | POOLED |")
            md.append("|---|---|---|---|---|")
            for c in mlp["cells"]:
                md.append(f"| {c['target']} | " + " | ".join(
                    f"{c['score'][k]} (null≤{c['null'][k]['p95']})"
                    for k in ("raw", "pi", "vf", "pooled")) + " |")
    with open(os.path.join(a.out_dir, "probe_stats.json"), "w") as f:
        json.dump(small, f, indent=1)
    with open(os.path.join(a.out_dir, "tables.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))
    print("\nwrote probe_stats.json + tables.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmp", required=True)
    ap.add_argument("--out-dir", default=".")
    main(ap.parse_args())
