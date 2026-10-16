import json, sys
from agents.training import eval_ledger as L
from agents.training import plateau_t1 as T
from main import plateau as P
root = L.archive_ledger_root()
out = {"source": "models/_ledger (the archive ledger), plateau_t1 requests of the T-ENT toy (T33, 2026-10-15)",
       "checks": []}
for run, grid in (("rb_toy_ent_s4001", 25_000_000), ("rb_toy_ent_s4002", 65_000_000), ("rb_toy_ent_s4002", 90_000_000)):
    plan = T.CheckPlan(delta_steps=5_000_000, lag=4)
    rid = T.request_id(run, grid, plan)
    rows = list(L.read(P.TIER1_ROWS, root=root, request_id=rid).rows)
    ds = [d for d in L.read_decisions(root=root, kind="plateau", request_ids=[rid])]
    assert len(ds) == 1, ds
    d = ds[0]
    res = T.evaluate(rows, T.Tier1Rule.parse(d["rule"]))
    assert res.verdict == d["verdict"]
    rows.sort(key=lambda r: int(r["request"]["batch"]))
    out["checks"].append({
        "request_id": rid, "player": rows[0]["player"]["id"], "opponent": rows[0]["opponent"]["id"],
        "decision": {"decision_id": d["decision_id"], "verdict": d["verdict"], "rule": d["rule"],
                     "consumed_count": d["consumed"]["count"]},
        "recorded": {"n_pairs": res.n_pairs, "llr": round(res.llr, 9), "stop_batch": res.stop_batch,
                     "surplus_batches": res.surplus_batches},
        "batches": [{"batch": int(r["request"]["batch"]), "row_id": r["row_id"], "pair_counts": r["pairs"]["counts"],
                     "counts": r["counts"], "cycle_seed": r["seed"]["cycle_seed"],
                     "outcome_digest_all": r["compute"].get("outcome_digest_all")} for r in rows]})
json.dump(out, sys.stdout, indent=1)
print()
