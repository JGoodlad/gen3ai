"""Summarise ``validate_sources.py``'s rows → ``validate_sources_summary.json`` (this folder). Pure Python.

    python -I validate_read.py end2 prod
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t1_common as T  # noqa: E402


def rate(a):
    return {"rate": round(a[0] / a[1], 4) if a[1] else None, "n": a[1]}


def main() -> None:
    out = {}
    for cp in sys.argv[1:]:
        # the first run's rows, then re-runs of some arms (validate_sources_<tag>.jsonl) replace those arms
        by: dict = {}
        for p in [T.cp_dir(cp) / "validate_sources.jsonl"] + sorted(T.cp_dir(cp).glob("validate_sources_*.jsonl")):
            for r in T.read_rows(p):
                if r["id"] in by:
                    by[r["id"]]["arms"].update(r["arms"])
                    if p.name != "validate_sources.jsonl":
                        by[r["id"]].setdefault("rerun", []).append(p.name)
                else:
                    by[r["id"]] = r
        rows = list(by.values())
        res: dict = {"states": len(rows), "k": sorted(r["k"] for r in rows),
                     "k_net_equals_k": sum(r["k"] == r["k_net"] for r in rows), "arms": {}}
        ge = [r["gate_equality"] for r in rows if "gate_equality" in r]
        res["pool_gate_equality_vs_builder"] = {"teams": sum(g["teams"] for g in ge),
                                                "builder_pass": sum(g["builder_pass"] for g in ge),
                                                "agree": sum(g["agree"] for g in ge)}
        arms = sorted({a for r in rows for a in r["arms"]})
        for arm in arms:
            recs = [(r, r["arms"][arm]) for r in rows if arm in r["arms"]]
            ok = [(r, x) for r, x in recs if x["ok"]]
            acc = {k: [0, 0] for k in ("species_hit", "set_recall", "new_set_recall")}
            for _, x in ok:
                for k in acc:
                    acc[k][0] += x["truth"][k][0]
                    acc[k][1] += x["truth"][k][1]
            aud: dict = {}
            for _, x in ok:
                for c, v in x["audit"].items():
                    aud.setdefault(c, [0, 0])
                    aud[c][0] += bool(v["pass"])
                    aud[c][1] += 1
            cpu = [x["cpu_s"] for _, x in recs]
            a = {"states": len(recs), "built": len(ok),
                 "excluded": [{"id": r["id"], "k": r["k"], "error": x.get("error", "")[:300],
                               "reason": x.get("reason")} for r, x in recs if not x["ok"]],
                 "cpu_s_mean": round(statistics.mean(cpu), 2) if cpu else None,
                 "cpu_s_median": round(statistics.median(cpu), 2) if cpu else None,
                 "cpu_s_max": max(cpu) if cpu else None,
                 "tries_mean": round(statistics.mean(t for _, x in ok for t in x["tries"]), 3) if ok else None,
                 "fails": {}, "truth": {k: rate(v) for k, v in acc.items()},
                 "audit_pass": {c: f"{p}/{n}" for c, (p, n) in aud.items()}}
            for _, x in ok:
                for k, v in x["fails"].items():
                    a["fails"][k] = a["fails"].get(k, 0) + v
            if arm == "FN":
                cnt: dict = {}
                for _, x in ok:
                    for k, v in x["fn"]["counts"].items():
                        cnt[k] = cnt.get(k, 0) + v
                    for k in ("species_draws", "species_fallback"):
                        cnt["draws:" + k] = cnt.get("draws:" + k, 0) + x["fn"]["draw_counts"][k]
                    rc = x["fn"]["row_check"]
                    cnt["seen_no_presence"] = cnt.get("seen_no_presence", 0) + len(rc["seen_no_presence"])
                    cnt["seen_unfillable"] = cnt.get("seen_unfillable", 0) + len(rc["seen_unfillable"])
                    cnt["num_collisions"] = cnt.get("num_collisions", 0) + len(rc["num_collisions"])
                    cnt["screen_rebuilt"] = cnt.get("screen_rebuilt", 0) + x["fn"]["screen_rebuilt"]
                a["provenance"] = cnt
                a["species_mass_outside_pool_max"] = max((x["fn"]["row_check"]["species_mass_outside_pool"]
                                                          for _, x in ok), default=None)
            if arm == "FNfresh":
                a["fresh_vs_shared"] = {k: sum(x.get(k, 0) for _, x in ok) for k in
                                        ("samplers_equal_F", "samplers_differ_F", "samplers_missing_in_F",
                                         "screen_rebuilt", "hp_any_reset")}
                a["fresh_vs_shared"]["teams_equal_FN"] = sum(bool(x.get("teams_equal_FN")) for _, x in ok)
                a["fresh_vs_shared"]["seen_facts_equal_FN"] = sum(bool(x.get("seen_facts_equal_FN")) for _, x in ok)
            if arm == "POOLSETS":
                c: dict = {}
                for _, x in ok:
                    for k, v in x["poolsets"]["counts"].items():
                        c[k] = c.get(k, 0) + v
                a["slot_tables"] = c
            if arm == "POOL":
                pm = [x["pool"] for _, x in ok]
                a["pool"] = {
                    "candidates": [p["candidates"] for p in pm], "passed": [p["passed"] for p in pm],
                    "passed_distinct_teams": [p["passed_distinct_teams"] for p in pm],
                    "true_in_candidates": sum(p["true_in_candidates"] for p in pm),
                    "true_gated": sum(p["true_gated"] for p in pm),
                    "true_in_passed": sum(p["true_in_passed"] for p in pm),
                    "identified (one passing team = the truth)": sum(p["passed_distinct_teams"] == 1
                                                                    and p["true_in_passed"] for p in pm),
                    "true_drawn_worlds": f"{sum(p['true_drawn'] for p in pm)}/{8 * len(pm)}",
                    "excluded_reasons": [x.get("reason") for _, x in recs if not x["ok"]]}
            res["arms"][arm] = a
        out[cp] = res
    pilot = json.loads((T.PILOT / "move_accuracy_read.json").read_text())
    out["pilot_C_prime_reference"] = {cp: pilot[cp]["det_c"] for cp in pilot}
    out["pilot_smogon_reference"] = {cp: pilot[cp]["det"] for cp in pilot}
    (HERE / "validate_sources_summary.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(out, indent=1, sort_keys=True)[:6000])


if __name__ == "__main__":
    main()
