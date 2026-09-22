"""Audit every committed `snapshot_ladder/ladder.json` under `models/` against a refit.

READ-ONLY on `models/`: `fit_ladder(..., write=False)` never touches the archive. One JSON per
run lands beside this script; `README.md` carries the table.

    export PYTHONPATH=$PYTHONPATH:src
    nice -n 15 python3 designs/research_state/measurements/ladder_refit_audit_2026-09-22/refit_audit.py

The refit uses the COMMITTED file's own rated steps as the node set (not the pool on disk), so
the comparison is node-for-node: Bradley-Terry re-solves every node on every add, so a fit over a
different node set is a different object and its deltas would conflate two effects.
"""
from __future__ import annotations

import json
import os
import sys
import glob
import traceback

from agents.training import snapshot_ladder as sl

OUT = os.path.dirname(os.path.abspath(__file__))
MODELS = os.environ.get("GEN3AI_MODELS_DIR", "/home/goodlad/dev/gen3ai/models")


def audit_one(run_dir: str) -> dict:
    name = os.path.basename(run_dir)
    lp = sl.ladder_json_path(run_dir)
    gp = sl.games_log_path(run_dir)
    rec: dict = {"run": name,
                 "ladder_json": os.path.exists(lp),
                 "games_jsonl": os.path.exists(gp),
                 "games_lines": sum(1 for _ in open(gp)) if os.path.exists(gp) else 0}
    if not rec["ladder_json"]:
        rec["error"] = "no ladder.json"
        return rec
    committed = json.load(open(lp))
    status, detail = sl.recipe_status(committed)
    rec["recipe_status"] = status
    rec["recipe_detail"] = detail
    rec["committed_computed_at"] = committed.get("computed_at")
    rec["committed_sentinel_count_key"] = committed.get("eval_sentinel_edges_dropped")
    cr = committed.get("ratings") or {}
    rec["n_nodes"] = len(cr)
    rec["refit_possible"] = bool(rec["games_jsonl"] and cr)
    if not rec["refit_possible"]:
        rec["error"] = "no games.jsonl" if not rec["games_jsonl"] else "no rated nodes"
        return rec
    steps = sorted(int(k) for k in cr)
    refit = sl.fit_ladder(run_dir, write=False, steps=steps)
    rr = refit.get("ratings") or {}
    rec["refit_sentinel_edges_dropped"] = refit.get("eval_sentinel_edges_dropped")
    rec["refit_n_frozen_pairs"] = refit.get("n_frozen_pairs_measured")
    rec["committed_n_frozen_pairs"] = committed.get("n_frozen_pairs_measured")
    rec["refit_converged"] = refit.get("converged")
    rec["committed_converged"] = committed.get("converged")
    nodes = []
    for k in map(str, steps):
        c = cr.get(k)
        n = rr.get(k)
        nodes.append({"step": int(k), "committed": c, "refit": n,
                      "delta": None if (c is None or n is None) else round(n - c, 1)})
    rec["nodes"] = nodes
    rec["missing_in_refit"] = [n["step"] for n in nodes if n["refit"] is None]
    ds = [n["delta"] for n in nodes if n["delta"] is not None]
    rec["max_abs_delta"] = max((abs(d) for d in ds), default=0.0)
    rec["mean_delta"] = round(sum(ds) / len(ds), 2) if ds else 0.0
    newest = nodes[-1]
    rec["newest_step"] = newest["step"]
    rec["newest_committed"] = newest["committed"]
    rec["newest_refit"] = newest["refit"]
    rec["newest_delta"] = newest["delta"]
    rec["newest_delta_sign"] = ("-" if (newest["delta"] or 0) < 0
                                else "+" if (newest["delta"] or 0) > 0 else "0")
    # BEST node can move too — the headline rule says newest at run end, but a curve read quotes
    # the max, so record whether the ARGMAX node changes.
    if all(n["committed"] is not None for n in nodes) and not rec["missing_in_refit"]:
        rec["argmax_committed"] = max(nodes, key=lambda n: n["committed"])["step"]
        rec["argmax_refit"] = max(nodes, key=lambda n: n["refit"])["step"]
        rec["within_run_order_changes"] = int(rec["argmax_committed"] != rec["argmax_refit"])
        # Kendall-style: how many ADJACENT within-run node pairs swap order on refit
        swaps = 0
        for i in range(len(nodes) - 1):
            a, b = nodes[i], nodes[i + 1]
            if (a["committed"] < b["committed"]) != (a["refit"] < b["refit"]):
                swaps += 1
        rec["adjacent_node_swaps"] = swaps
    return rec


# ── does the refit change the ORDERING a cross-run claim rests on? ──────────────────────────
#
# A rating is only ever used in two ways: quoted, or COMPARED. A uniform shift would be harmless
# to the second use; these deltas are not uniform (they scale with how much of a node's evidence
# came from eval sentinel edges), so the ordering has to be checked directly. Sibling groups are
# the campaigns that were actually compared against each other in the ledger.
import itertools
import re

SIBLING_GROUPS = {
    "v9 generation ladder (gen1..gen17)": lambda n: bool(re.search(r"_gen\d+", n)),
    "E-substrate + baitbot (E1..E4)": lambda n: bool(re.search(r"_E[1-4]_", n)),
    "fold-dial fd* (A/B/C/E/F, F-p1c, F-p2c)": lambda n: bool(re.search(r"_fd[A-F]_|_fdF_p", n)),
    "R2 rung (CTRL/ACTION/TOPK/KL/PLAIN)": lambda n: "_R2" in n,
    "R3/R4 + DOSE": lambda n: "_R3" in n or "_R4" in n,
    "teacher-content TC* (09-03/09-04)": lambda n: "_TC" in n,
    "09-01 factorial (B2/C1/N2/G1SHORT)": lambda n: bool(re.search(r"_(B2|C1|N2|G1SHORT)_", n)),
    "tdaux rung2 (lam00/lam10/lam30)": lambda n: "tdaux_rung2" in n,
    "G-gate (G1_action/G2_advgate/G1p_matched)":
        lambda n: bool(re.search(r"_G1_action|_G2_advgate|_G1p_matched", n)),
    "v12 ladder cells (10..29)": lambda n: bool(re.match(r"ai_v12_[12]\d_ladder", n)),
    "v13 flywheel family": lambda n: n.startswith("ai_v13_"),
    "winprob critic pair (v12_01 / v12_02)": lambda n: n.endswith("_winprob_critic"),
}


def ordering(records: list[dict]) -> dict:
    by = {r["run"]: r for r in records if r.get("newest_committed") is not None
          and r.get("newest_refit") is not None}
    out = {"groups": [], "global": {}}
    for label, pred in SIBLING_GROUPS.items():
        names = [n for n in by if pred(n)]
        if len(names) < 2:
            continue
        rank = lambda key: sorted(names, key=lambda n: -by[n][key])   # noqa: E731
        flips = [(x, y) for x, y in itertools.combinations(names, 2)
                 if ((by[x]["newest_committed"] > by[y]["newest_committed"]) !=
                     (by[x]["newest_refit"] > by[y]["newest_refit"]))]
        out["groups"].append({
            "group": label, "n": len(names),
            "order_changes": rank("newest_committed") != rank("newest_refit"),
            "flipped_pairs": len(flips),
            "n_pairs": len(names) * (len(names) - 1) // 2,
            "committed_order": rank("newest_committed"),
            "refit_order": rank("newest_refit"),
            "flips": [{"a": x, "b": y,
                       "committed_delta": round(by[x]["newest_committed"]
                                                - by[y]["newest_committed"], 1),
                       "refit_delta": round(by[x]["newest_refit"] - by[y]["newest_refit"], 1)}
                      for x, y in flips],
        })
    names = list(by)
    oc = sorted(names, key=lambda n: -by[n]["newest_committed"])
    orf = sorted(names, key=lambda n: -by[n]["newest_refit"])
    out["global"] = {
        "n": len(names),
        "flipped_pairs": sum(1 for x, y in itertools.combinations(names, 2)
                             if ((by[x]["newest_committed"] > by[y]["newest_committed"]) !=
                                 (by[x]["newest_refit"] > by[y]["newest_refit"]))),
        "n_pairs": len(names) * (len(names) - 1) // 2,
        "biggest_rank_moves": sorted(
            ({"run": n, "committed_rank": oc.index(n) + 1, "refit_rank": orf.index(n) + 1,
              "places": abs(oc.index(n) - orf.index(n))} for n in names),
            key=lambda d: -d["places"])[:15],
    }
    return out


def main() -> int:
    runs = sorted(os.path.dirname(p) for p in glob.glob(os.path.join(MODELS, "*", "snapshot_ladder")))
    out = []
    for r in runs:
        try:
            rec = audit_one(r)
        except Exception as e:                       # noqa: BLE001 — one bad run must not stop the sweep
            rec = {"run": os.path.basename(r), "error": f"{type(e).__name__}: {e}",
                   "traceback": traceback.format_exc()}
        out.append(rec)
        with open(os.path.join(OUT, f"{rec['run']}.json"), "w") as f:
            json.dump(rec, f, indent=2)
        print(f"{rec['run']:<46} {rec.get('recipe_status','-'):<7} "
              f"n={rec.get('n_nodes','-'):<3} maxΔ={rec.get('max_abs_delta','-'):<7} "
              f"newestΔ={rec.get('newest_delta','-')}", flush=True)
    with open(os.path.join(OUT, "_all_runs.json"), "w") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(OUT, "_ordering.json"), "w") as f:
        json.dump(ordering(out), f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
