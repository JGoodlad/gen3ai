"""Fold the per-arm `<arm>_<weights>.json` files of `measure.py` into `result.json`: per arm and weights, the
excluded rows by CUTOFF FAMILY and flip verdict (value-identical vs genuinely distinct) under the rule
before, and the excluded share under the rule after (`gen3_behaviour_tie_identity_v1`).

    python summarize.py        # writes result.json beside it
"""
from __future__ import annotations

import collections
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARMS = ("blob", "oracle_species", "oracle_full", "fixed_mass")
WEIGHTS = ("fresh", "perturbed")


def family(sites: str) -> str:
    """The cutoff family a row's tied sites belong to (a row can sit at several)."""
    parts = {s.split(" ")[0] + " " + s.split(" ")[1] for s in sites.split(" + ")}
    fams = set()
    for p in parts:
        if p.startswith("damage_op.py:") and p.endswith("argmax"):
            fams.add("dominant-move argmax (damage_op)")
        elif "argsort" in p:
            fams.add("X5 stable order (hypothesis_set argsort)")
        elif p.startswith("damage_kinds.py"):
            fams.add("fixed damage >= current HP (damage_kinds)")
        else:
            fams.add("belief-candidate top-K / its cutoff (pointer_head, damage_op_pairwise, damage_op_blocks)")
    return " + ".join(sorted(fams))


def main() -> None:
    out = {"schema": "k9_tie_identity_v1", "eps": None, "arms": {}}
    for a in ARMS:
        for w in WEIGHTS:
            r = json.loads((HERE / f"{a}_{w}.json").read_text())
            out["eps"] = r["eps"]
            fam = collections.Counter()
            for row in r["by_site"]:
                fam[(family(row["sites"]), row["verdict"])] += row["rows"]
            out["arms"][f"{a}/{w}"] = {
                "rows": r["rows"], "excluded_before": r["excluded_before"],
                "excluded_frac_before": r["excluded_frac_before"],
                "flip_identical": r["cleared_identical"], "flip_distinct": r["excluded_before"] - r["cleared_identical"],
                "excluded_after_payload_rule": r["new_rule"]["excluded_payload_only"],
                "excluded_after": r["new_rule"]["excluded"], "excluded_frac_after": r["new_rule"]["excluded_frac"],
                "after_payload_rule_by_flip_verdict": r["new_rule"]["payload_only_rows_by_flip_verdict"],
                "distinct_rows_cleared": r["excluded_before"] - r["cleared_identical"]
                - r["new_rule"]["payload_only_rows_by_flip_verdict"].get("distinct", 0),
                "before_by_family": [{"family": k[0], "verdict": k[1], "rows": v} for k, v in fam.most_common()],
                "determinism_failures": r["determinism_failures_untouched_rows"], "flip_noop_rows": r["flip_noop_rows"],
                "max_abs_dlogp_vs_stored": r["max_abs_dlogp_vs_stored"]}
    (HERE / "result.json").write_text(json.dumps(out, indent=1) + "\n")
    for k, v in out["arms"].items():
        print(f"{k:28s} before {v['excluded_frac_before']:.3f} (identical {v['flip_identical']}, distinct "
              f"{v['flip_distinct']})  after {v['excluded_frac_after']:.3f}  distinct cleared {v['distinct_rows_cleared']}")
        for f in v["before_by_family"]:
            print(f"      {f['rows']:4d} {f['verdict']:9s} {f['family']}")


if __name__ == "__main__":
    main()
