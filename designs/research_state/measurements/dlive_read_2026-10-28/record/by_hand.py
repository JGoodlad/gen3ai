"""By-hand reproduction of the dlive read's decision (frozen record; not a tool). Reads the state's inputs.json (the
rule's own input document, branch rows after AM-DL-PEER) and recomputes, without main.reads or live_branch:
the per-panel and pooled-U mean D, the game-clustered bootstrap (its own loop, the registered rng order), the class,
and the teeth counts. Writes by_hand.json next to it.

    python by_hand.py /home/goodlad/gen3ai_archive/dlive_read_2026-10-28
"""
import json
import sys
from pathlib import Path

import numpy as np

U = ["heldout72_hpfix", "forum72", "p72"]
HOME = "home72"


def main(state: str) -> int:
    st = Path(state)
    doc = json.loads((st / "inputs.json").read_text())
    dec = json.loads((st / "decision.json").read_text())
    rows = doc["branches"]
    br = [r for r in rows if r["kind"] == "branch" and r["status"] == "counted"]
    clusters = {}
    for r in br:
        clusters.setdefault(r["panel"], {}).setdefault(r["deal"], []).append(r["y_branch"] - r["y_raw"])
    sums = {p: np.array([sum(v) for _, v in sorted(clusters[p].items())]) for p in clusters}
    ns = {p: np.array([len(v) for _, v in sorted(clusters[p].items())]) for p in clusters}
    per = {p: float(sums[p].sum() / ns[p].sum()) for p in clusters}
    point = float(np.mean([per[p] for p in U]))
    rng = np.random.default_rng(20261028)
    bs = []
    for _ in range(2000):
        vals = []
        for p in U:
            i = rng.integers(0, len(sums[p]), len(sums[p]))
            vals.append(sums[p][i].sum() / ns[p][i].sum())
        bs.append(float(np.mean(vals)))
    lo, hi = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
    cls = "GO" if lo > 0.0 + 0.0005 else "STOP"
    nulls = [r for r in rows if r["kind"] == "null"]
    teeth = {
        "branches_planned": sum(r["kind"] == "branch" for r in rows),
        "branches_counted": len(br),
        "branches_dropped": {p: sum(r["kind"] == "branch" and r["status"] == "dropped" and r["panel"] == p for r in rows)
                             for p in [*U, HOME]},
        "branches_pending": sum(r["status"] == "pending" for r in rows),
        "nulls_counted": sum(r["status"] == "counted" for r in nulls), "nulls": len(nulls),
        "nulls_changed": sum(r["status"] == "counted" and (r["y_branch"] != r["y_raw"]
                                                             or r["branch_turns"] != r["raw_turns"]) for r in nulls),
        "g_branch_fails_left": sum(bool(r["g_branch_fails"]) for r in rows),
        "g_branch_peer_units": sorted(r["unit"] for r in rows if r.get("g_branch_peer")),
        "capture_fails": sum(r["status"] == "counted" and not r["capture"]["ok"] for r in rows),
        "forced_wrong": sum(r["status"] == "counted" and (r.get("forced") != r["action"]
                                                         or r.get("branch_index") != r["game"]) for r in rows),
        "engines": sorted({r.get(k) for r in rows if r["status"] == "counted" for k in ("raw_engine", "branch_engine")}),
    }
    pr = dec["primary"]
    out = {"per_panel_mean_d": per, "pooled_u": {"est": point, "lo": lo, "hi": hi, "n": int(sum(ns[p].sum() for p in U))},
           "class": cls, "teeth": teeth,
           "equal_to_rule": {"est": abs(point - pr["est"]) < 1e-12, "lo": abs(lo - pr["lo"]) < 1e-12,
                             "hi": abs(hi - pr["hi"]) < 1e-12, "class": cls == dec["verdict"]}}
    Path(__file__).with_name("by_hand.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(out["equal_to_rule"]), json.dumps(out["pooled_u"]))
    return 0 if all(out["equal_to_rule"].values()) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
