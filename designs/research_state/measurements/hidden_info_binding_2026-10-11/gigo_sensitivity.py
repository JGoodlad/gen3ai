"""POST-HOC SENSITIVITY to the audit findings (NOT registered; written after the audits reported): the pilot's
oracle − fair gap G (LEAF-K8, Phase 2 Smogon worlds) and its D1 split, (a) on all states, (b) EXCLUDING every state
with ANY played world the A2 follow-up found inconsistent with the searched side's PROTOCOL
(`audit_a2_obs_inconsistent_<cp>.json`: gender tokens, Focus Punch tightening, Beat Up hits) or any A2 unexpected
cell (`audit_a2_field_diff_<cp>.json`), and (c) A4's clause-illegal teams are NOT excluded (A4 reports counts and examples, no per-state list).
Bounds how much the instrument's known defects can move G.

    python -I gigo_sensitivity.py  →  gigo_sensitivity.json
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "value_disc_pilot_2026-10-11"))
import descriptive_reads as DR  # noqa: E402
import score as S1  # noqa: E402


def ids_from(obj, key="id"):
    out = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key and isinstance(v, str) and "#p" in v:
                out.add(v)
            else:
                out |= ids_from(v, key)
    elif isinstance(obj, list):
        for v in obj:
            out |= ids_from(v, key)
    elif isinstance(obj, str) and "#p" in obj and obj.count("/") >= 3:
        out.add(obj)
    return out


def main():
    res = {}
    for cp in DR.CPS:
        _st, recs, _ex = DR.per_state(cp, "det")
        inc = json.loads((HERE / f"audit_a2_obs_inconsistent_{cp}.json").read_text())
        bad_proto = {s["id"] for s in inc["by_dir"]["det"]["states"]}
        a2 = json.loads((HERE / f"audit_a2_field_diff_{cp}.json").read_text())["by_dir"]["det"]
        bad_a2 = {ex["id"] for cat, exs in a2["examples"].items() if cat.startswith("UNEXPECTED") for ex in exs}
        if len(bad_a2) != a2["states_with_unexpected"]:
            raise SystemExit(f"{cp}: the A2 examples list {len(bad_a2)} states, the count is {a2['states_with_unexpected']}")
        # A4's clause violations are reported as counts + examples only (no per-state list): not excluded here (named)
        bad_a4 = set()
        out = {"n_all": len(recs), "excluded_protocol": len(bad_proto), "excluded_a2_cells": len(bad_a2),
               "excluded_a4_listed": len(bad_a4)}
        for lab, drop in (("all", set()), ("minus_protocol_and_a2", bad_proto | bad_a2),
                          ("minus_protocol_a2_a4", bad_proto | bad_a2 | bad_a4)):
            rs = [r for r in recs if r["id"] not in drop]
            bt = S1.Boot([r["battle"] for r in rs], [1.0] * len(rs))
            g = np.array([r["imp_o"]["ARGMAX|LEAF-K8"] - r["imp_p"]["ARGMAX|LEAF-K8"] for r in rs])
            ip = np.array([r["imp_p"]["ARGMAX|LEAF-K8"] for r in rs])
            d = {"n": len(rs), "G": bt.stat(g), "fair_IMP": bt.stat(ip)}
            d.update(DR.strata(bt, rs, g, "D1", lambda r: r["unrevealed"] if r["unrevealed"] < 2 else "2+"))
            out[lab] = d
        res[cp] = out
        print(cp, {k: (v["n"], v["G"]["mean"], v["G"]["ci"], v["fair_IMP"]["mean"]) for k, v in out.items()
                   if isinstance(v, dict)}, "excluded", out["excluded_protocol"], out["excluded_a2_cells"],
              out["excluded_a4_listed"])
    (HERE / "gigo_sensitivity.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
