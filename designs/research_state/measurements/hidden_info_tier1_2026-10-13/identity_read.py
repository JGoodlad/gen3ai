"""VALIDITY (no arm outcome read): the whole cross-commit pipeline on the TRUE team. Worlds built by the HEAD builder
under the ``oracle`` spec (`build_worlds.py --validation oracle`: the true team and prefix), stored in the det format,
played at the PAD pin by `play_worlds.py`, must reproduce the pilot's Phase 1 rows exactly: for every legal action,
world j's outcome and value trace = Phase 1's leaf seed j (j = 0..7), and the one-ply critic's 8 successors = Phase 1's
critic seeds 40..47. Also, for a code given with ``--alias-of-truth``, the same comparison (an arm that equals the truth by
construction: O-U at k = 6).

    python -I identity_read.py [--code V_oracle] [--states registered] [--out identity_<code>.json]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import t1_common as T  # noqa: E402


def phase1_rows(cp: str, states: str) -> dict:
    d = (T.PILOT_ARCH / cp / "rows") if states == "registered" else (T.cp_dir(cp) / "rows_supp")
    out = {}
    for p in sorted(d.glob("shard_*.jsonl")):
        for r in T.read_rows(p):
            out[r["id"]] = r
    return out


#: the forward batches rows differently in a lockstep of 8 worlds than in Phase 1's one-world playouts, so a value can
#: move by float re-association (the pilot's note: "up to float reassociation"); values are stored rounded (trace V to
#: 5 decimals, critic q to 6). Outcomes, turns and switch flags must be EXACT; a value within these bounds is equal.
V_TOL = 5e-5
Q_TOL = 1e-5


def compare(r1: dict, rw: dict) -> dict:
    res = {"actions": 0, "out_mismatch": 0, "trace_mismatch": 0, "crit_mismatch": 0, "max_dv": 0.0, "max_dq": 0.0,
           "examples": []}
    if sorted(r1["out"], key=int) != sorted(rw["out"], key=int):
        res["examples"].append("action sets differ")
        res["out_mismatch"] += 1
        return res
    for a in r1["out"]:
        res["actions"] += 1
        if r1["out"][a][: T.J] != rw["out"][a]:
            res["out_mismatch"] += 1
            if len(res["examples"]) < 5:
                res["examples"].append(f"{a}: out {r1['out'][a][:T.J]} vs {rw['out'][a]}")
        for j in range(T.J):
            t1, tw = r1["trace"][a][j], rw["trace"][a][j]
            bad = t1["t"] != tw["t"] or t1["s"] != tw["s"] or len(t1["v"]) != len(tw["v"])
            if not bad and t1["v"]:
                dv = max(abs(float(x) - float(y)) for x, y in zip(t1["v"], tw["v"]))
                res["max_dv"] = max(res["max_dv"], dv)
                bad = dv > V_TOL
            if bad:
                res["trace_mismatch"] += 1
                if len(res["examples"]) < 5:
                    res["examples"].append(f"{a}/{j}: trace differs")
        q1 = r1["crit"][a]["q"]
        qw = rw["crit"][a]["q"]
        dq = max(abs(float(x) - float(y)) for x, y in zip(q1, qw))
        res["max_dq"] = max(res["max_dq"], dq)
        if dq > Q_TOL or r1["crit"][a]["src"] != rw["crit"][a]["src"]:
            res["crit_mismatch"] += 1
            if len(res["examples"]) < 5:
                res["examples"].append(f"{a}: crit {q1} vs {qw}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="V_oracle")
    ap.add_argument("--states", default="registered", choices=("registered", "supp"))
    ap.add_argument("--k6-only", action="store_true", help="compare only states at k = 6 (an arm equal to O there)")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    doc = {"code": a.code, "states": a.states, "cps": {}}
    ok_all = True
    for cp in T.CPS:
        rows = T.read_dir(T.rows_dir(cp, a.code), "rows_*.jsonl")
        if not rows:
            continue
        r1 = phase1_rows(cp, a.states)
        st = T.registered_states(cp) if a.states == "registered" else T.supplement_states(cp)
        tot = {"states": 0, "refused": 0, "actions": 0, "out_mismatch": 0, "trace_mismatch": 0, "crit_mismatch": 0,
               "examples": []}
        for did, rw in sorted(rows.items()):
            if rw.get("states") != a.states:
                continue
            if a.k6_only and 6 - int(st[did]["unrevealed"]) != 6:
                continue
            tot["states"] += 1
            if not rw.get("ok") or not r1.get(did, {}).get("ok"):
                tot["refused"] += 1
                tot["examples"].append(f"{did}: refused {rw.get('error')}")
                continue
            c = compare(r1[did], rw)
            for k in ("actions", "out_mismatch", "trace_mismatch", "crit_mismatch"):
                tot[k] += c[k]
            tot["max_dv"] = max(tot.get("max_dv", 0.0), c["max_dv"])
            tot["max_dq"] = max(tot.get("max_dq", 0.0), c["max_dq"])
            tot["examples"] += [f"{did}: {e}" for e in c["examples"]][: max(0, 10 - len(tot["examples"]))]
        tot["verdict"] = "PASS" if (tot["states"] and not tot["refused"] and not tot["out_mismatch"]
                                    and not tot["trace_mismatch"] and not tot["crit_mismatch"]) else "FAIL"
        ok_all &= tot["verdict"] == "PASS"
        doc["cps"][cp] = tot
    doc["verdict"] = "PASS" if ok_all and doc["cps"] else "FAIL"
    s = json.dumps(doc, indent=1, sort_keys=True)
    print(s)
    if a.out:
        (T.HERE / a.out).write_text(s + "\n")


if __name__ == "__main__":
    main()
