"""T38: each scratch ledger's replay of the recorded check vs the archive's record (fixture)."""
import json
import sys

from agents.training import eval_ledger as L
from agents.training import plateau_t1 as T
from main import plateau as P

fx = {c["request_id"]: c for c in json.load(open("/tmp/t38/fixture.json"))["checks"]}
RID = sys.argv[1]
for root in sys.argv[2:]:
    rid = RID
    rows = list(L.read(P.TIER1_ROWS, root=root, request_id=rid).rows)
    rec = fx.get(rid)
    res = T.evaluate(rows, T.REGISTERED)
    ds = L.read_decisions(root=root, kind="plateau")
    by = {int(r["request"]["batch"]): r for r in rows}
    out = {"ledger": root, "request": rid, "rows": len(rows), "decision": [d["verdict"] for d in ds],
           "verdict": res.verdict, "stop_batch": res.stop_batch, "n_pairs": res.n_pairs, "llr": round(res.llr, 9),
           "surplus": res.surplus_batches,
           "games_per_s": sorted({r["compute"].get("games_per_s") for r in rows})[:3],
           "inflight": sorted({r["compute"].get("inflight_batches", 1) for r in rows})}
    if rec is not None:
        rb = {b["batch"]: b for b in rec["batches"]}
        common = sorted(set(rb) & set(by))
        out["recorded"] = {"verdict": rec["decision"]["verdict"], **rec["recorded"]}
        out["same_games_as_recorded"] = all(
            by[b]["compute"]["outcome_digest_all"] == rb[b]["outcome_digest_all"]
            and by[b]["pairs"]["counts"] == rb[b]["pair_counts"] and by[b]["seed"]["cycle_seed"] == rb[b]["cycle_seed"]
            for b in common)
        out["batches_compared"] = len(common)
        out["same_decision_as_recorded"] = (res.verdict, res.stop_batch, res.n_pairs, round(res.llr, 9)) == (
            rec["decision"]["verdict"], rec["recorded"]["stop_batch"], rec["recorded"]["n_pairs"],
            rec["recorded"]["llr"])
    print(json.dumps(out))
