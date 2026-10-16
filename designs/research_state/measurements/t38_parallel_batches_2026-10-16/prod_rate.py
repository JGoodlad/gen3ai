"""T38: the production-network game rate of the plateau meter's batches at several in-flight counts. The SAME 8 batches
of 40 pairs (rb_bf_eb_s2001 15M vs 10M, the plateau's schedule namespace) into a fresh scratch ledger per K, on ONE
engine, so every K plays the same games on the same process."""
import json
import sys
import time

from main import plateau as P
from main.h2h import play as PL
from utils.rust_env import episode as EP

M = "/home/goodlad/dev/gen3ai/models/rb_bf_eb_s2001/checkpoints"
KS = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [1, 4, 8]
NB = int(sys.argv[2]) if len(sys.argv) > 2 else 8
new = PL.resolve_player(f"{M}/checkpoint_15000029_steps.zip")
old = PL.resolve_player(f"{M}/checkpoint_10000194_steps.zip")
compute = PL.Compute(device="cpu", n_envs=64, threads=4, torch_threads=2)
regime = PL.regime_for(EP.stall_threshold())
key = P.schedule_key(new.sha256, old.sha256)
emit = lambda m: print(m, file=sys.stderr, flush=True)  # noqa: E731
eng = PL.H2HEngine(new, old, compute, emit)
out = []
try:
    for k in KS:
        root, writer = PL.open_writer(f"/tmp/t38/ledger_prodrate_k{k}", "plateau")
        ep = PL.plan_edge(writer, root, regime, new, old, pairs=NB * 40, batch_pairs=40, schedule_seed=0,
                          schedule_key=key, purpose="plateau", request_id=f"t38_rate:k{k}", family=None,
                          request_kind="adhoc")
        t0, c0 = time.perf_counter(), time.process_time()
        load0 = open("/proc/loadavg").read().split()[:3]
        PL.play_planned(eng, writer, ep, schedule_seed=0, purpose="plateau", run_label="t38_rate", commit="t38",
                        emit=emit, inflight=k)
        wall, cpu = time.perf_counter() - t0, time.process_time() - c0
        writer.close()
        rows = sorted(PL.read_own(regime["protocol"], root=root, request_id=ep.rid).rows,
                      key=lambda r: r["request"]["batch"])
        rec = {"k": k, "batches": NB, "games": NB * 80, "wall_s": round(wall, 1), "cpu_s": round(cpu, 1),
               "games_per_s": round(NB * 80 / wall, 3), "cpu_s_per_game": round(cpu / (NB * 80), 3),
               "load_start": load0, "load_end": open("/proc/loadavg").read().split()[:3],
               "pair_counts": [r["pairs"]["counts"] for r in rows],
               "digests": [r["compute"]["outcome_digest"] for r in rows],
               "near_tie_games": [r["compute"]["near_tie_games"] for r in rows]}
        out.append(rec)
        print(json.dumps({k2: v for k2, v in rec.items() if k2 not in ("pair_counts", "digests", "near_tie_games")}),
              flush=True)
finally:
    eng.close()
same = all(o["digests"] == out[0]["digests"] and o["pair_counts"] == out[0]["pair_counts"] for o in out)
print(json.dumps({"same_games_every_k": same}), flush=True)
json.dump(out, open("/tmp/t38/prod_rate.json", "w"), indent=1)
