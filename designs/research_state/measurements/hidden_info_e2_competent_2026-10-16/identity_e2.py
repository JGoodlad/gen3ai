"""E2 §5 VALIDITY (no arm outcome is printed: only match counts):

  1. the CROSS-COMMIT identity: `oracle` worlds built at the E2 HEAD (`build_e2_worlds.py --validation oracle`) and played
     at the PAD pin must reproduce Phase 1 (rows/O) exactly — every legal action's outcomes on worlds 0..7 = Phase 1's
     leaf seeds 0..7, value traces within Tier 1's tolerances, the one-ply critic's 8 successors = Phase 1's seeds 40..47
     (Tier 1's `identity_read.compare`, imported);
  2. the k = 6 IDENTITIES: on every k = 6 state, F+U ≡ F and O−U ≡ O outcome for outcome (O = Phase 1's leaf seeds 0..7)
     and the same value traces (`compare` again). A k = 6 state where F and F+U are BOTH exclusions is consistent (the
     same honest screen fails on both: at k = 6 F+U tells nothing F does not hold) and is counted, not compared.

    python -I identity_e2.py [--cp end2,prod] [--out identity_e2.json]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2_common as E  # noqa: E402

E.add_paths()


def tally(pairs, compare, allow_both_excluded: bool = False) -> dict:
    tot = {"states": 0, "refused": 0, "both_not_ok": 0, "actions": 0, "out_mismatch": 0, "trace_mismatch": 0,
           "crit_mismatch": 0, "max_dv": 0.0, "max_dq": 0.0, "examples": []}
    for did, ra, rb in pairs:
        tot["states"] += 1
        a_ok, b_ok = bool((ra or {}).get("ok")), bool((rb or {}).get("ok"))
        if not a_ok and not b_ok and allow_both_excluded:
            tot["both_not_ok"] += 1             # the same exclusion on both sides: consistent (no row to compare)
            continue
        if not a_ok or not b_ok:
            tot["refused"] += 1
            tot["examples"].append(f"{did}: one side not ok ({a_ok} / {b_ok})")
            continue
        c = compare(ra, rb)
        for k in ("actions", "out_mismatch", "trace_mismatch", "crit_mismatch"):
            tot[k] += c[k]
        tot["max_dv"] = max(tot["max_dv"], c["max_dv"])
        tot["max_dq"] = max(tot["max_dq"], c["max_dq"])
        tot["examples"] += [f"{did}: {e}" for e in c["examples"]][: max(0, 10 - len(tot["examples"]))]
    tot["verdict"] = "PASS" if (tot["states"] > tot["both_not_ok"] and not tot["refused"] and not tot["out_mismatch"]
                                and not tot["trace_mismatch"] and not tot["crit_mismatch"]) else \
        ("EMPTY" if not tot["states"] else "FAIL")
    return tot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cp", default=",".join(E.ORDER))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    from identity_read import compare          # Tier 1's (pure)

    doc = {}
    for cp in a.cp.split(","):
        st = E.e2_states(cp)
        truth = E.read_dir(E.rows_dir(cp, "O"), "shard_*.jsonl")
        rows = {s: E.read_dir(E.cp_dir(cp) / "rows" / s, "rows_*.jsonl") for s in ("V_oracle", "F", "FpU", "OmU")}
        orc = [(d, truth.get(d), r) for d, r in sorted(rows["V_oracle"].items()) if d in truth]
        k6 = [d for d, e in st.items() if int(e["k"]) == 6]
        fu = [(d, rows["F"].get(d), rows["FpU"].get(d)) for d in k6 if d in rows["F"] and d in rows["FpU"]]
        ou = [(d, truth.get(d), rows["OmU"].get(d)) for d in k6 if d in truth and d in rows["OmU"]]
        doc[cp] = {"oracle = Phase 1": tally(orc, compare),
                   "F+U = F at k = 6": tally(fu, compare, allow_both_excluded=True),
                   "O-U = O at k = 6": tally(ou, compare), "k6_states_in_bank": len(k6)}
    s = json.dumps(doc, indent=1, sort_keys=True)
    print(s)
    if a.out:
        (E.HERE / a.out).write_text(s + "\n")


if __name__ == "__main__":
    main()
