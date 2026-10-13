"""THE SUPPLEMENT's PHASE 1 TRUTH (arm O on the supplement, §12.13.3): the pilot's own Phase 1 driver, unchanged, over
the supplement bank. Per state: every legal action x the pilot's 40 shared playout seeds (`truth.turn_seeds(id, 48)`:
leaf 0..15, truth 16..39) played to the END by the checkpoint greedy on both sides in the oracle world, the searched
side's value trace, and the critic's one-ply successors on seeds 40..47. The scoring is the pilot's
`run_playouts.one_state`, IMPORTED (never copied).

    run_pin.sh <end2|prod> 2 6 24h phase1_supp.py --shard i --of W [--limit N] [--cost-only]
    run_pin.sh <end2|prod> 2 <GB> 24h phase1_supp.sh <W>          # W shards in one capped job

Rows: ~/gen3ai_archive/hidden_info_tier1/<cp>/rows_supp/shard_<i>_of_<W>.jsonl, one fsync'd row per state; resumable
(done = an id with a row in ANY shard file, so a relaunch may change W). `--cost-only` prints the cost and writes
NOTHING (no outcome is printed either: id, ok, error and cost only). Runs at the UNMODIFIED pin only.
"""
import argparse
import json
import time

import t1_supp as S

T = S.T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cost-only", action="store_true")
    a = ap.parse_args()
    c = S.cp()
    S.check_plain_pin(c)
    import common as C  # the pilot's
    import run_playouts as RP  # the pilot's Phase 1 driver

    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.reader import inference_globals
    from utils.rust_env import ffi as F
    from utils.rust_env.successors import SearchCore

    bank = load_bank(T.bank_path(c, "supp"))
    sup = json.loads((S.HERE / f"states_supp_{c}.json").read_text())
    if bank.manifest["content_sha256"] != sup["bank_content_sha256"]:
        raise SystemExit("the supplement bank is not the one states_supp_<cp>.json describes")
    ids = sorted(d["id"] for d in bank.decisions)
    if ids != sorted(sup["states"]):
        raise SystemExit("bank ids != states_supp ids")
    ids = T.shard(ids, a.shard, a.of)
    if a.limit:
        ids = ids[:a.limit]
    rdir = T.cp_dir(c) / "rows_supp"
    path = rdir / f"shard_{a.shard}_of_{a.of}.jsonl"
    done = set() if a.cost_only else {r["id"] for q in sorted(rdir.glob("shard_*.jsonl")) for r in T.read_rows(q)}
    todo = [x for x in ids if x not in done]
    print(f"[phase1_supp] {c} shard {a.shard}/{a.of}: {len(done)} done, {len(todo)} to go", flush=True)
    off = C.clock_offset()
    lib = F.load(F.default_path("release"))
    core = SearchCore(lib=lib)
    t0 = time.monotonic()
    tot = {"cpu": 0.0, "wall": 0.0, "play": 0, "rows": 0, "n": 0}
    with inference_globals(C.threads()):
        model = C.load_model(c)
        fwd = RP.Fwd(model)
        f = None
        if not a.cost_only:
            rdir.mkdir(parents=True, exist_ok=True)
            f = open(path, "a")
        try:
            for i, did in enumerate(todo):
                r = RP.one_state(core, fwd, bank, did, off)
                r["cp"] = c
                r["set"] = "supp"
                r["pin"] = T.CPS[c]["pin"]
                if a.cost_only:
                    if r.get("ok"):
                        tot["cpu"] += r["cost"]["play"]["cpu_s"] + r["cost"]["crit"]["cpu_s"]
                        tot["wall"] += r["cost"]["play"]["wall_s"] + r["cost"]["crit"]["wall_s"]
                        tot["play"] += r["cost"]["play"]["playouts"]
                        tot["rows"] += r["cost"]["play"]["rows"]
                        tot["n"] += 1
                    print(json.dumps({"id": did, "ok": r.get("ok"), "error": r.get("error"),
                                      "cost": r.get("cost")}), flush=True)
                    continue
                T.write_row(f, r)
                if not r.get("ok"):
                    print(f"[phase1_supp] REFUSED {did}: {r.get('error')}", flush=True)
                el = time.monotonic() - t0
                print(f"[phase1_supp] {i + 1}/{len(todo)} {el / 60:.1f} min, eta "
                      f"{el / (i + 1) * (len(todo) - i - 1) / 60:.1f} min", flush=True)
        finally:
            if f is not None:
                f.close()
            core.close()
    if a.cost_only and tot["n"]:
        print(json.dumps({"states": tot["n"], "threads": C.threads(),
                          "cpu_s_per_state": round(tot["cpu"] / tot["n"], 1),
                          "wall_s_per_state": round(tot["wall"] / tot["n"], 1),
                          "playouts_per_state": round(tot["play"] / tot["n"], 1),
                          "cpu_s_per_playout": round(tot["cpu"] / tot["play"], 4),
                          "rows_per_playout": round(tot["rows"] / tot["play"], 2),
                          "dedup_unique_share": round(fwd.unique / max(1, fwd.rows), 3)}), flush=True)


if __name__ == "__main__":
    main()
