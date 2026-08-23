"""End-to-end `better_line` wall clock, node vs rust — the consumer-level number.

The per-op benchmark isolates driver latency; this measures what a caller actually waits for:
one full depth-2 beam search over a real decision (open_root + every expand + the obs
materialization + the value forwards). Interleaved node/rust with the order flipping each rep,
so a drifting box load hits both arms equally.

CAVEAT THAT MATTERS FOR READING THE RATIO: the model here is the integration test's
`V = obs.sum()` stub, so the value forward is ~free and this is close to a pure driver+
materializer measurement. With a REAL extractor the model share is constant across impls, so the
end-to-end speedup is strictly LOWER than what this prints. Treat it as the upper bound on what
switching the driver buys `better_line`, and the per-op table as the mechanism.

    python src/rust_sim/harness/better_line_bench.py [reps]
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
from __future__ import annotations

import asyncio
import json
import statistics
import sys
import tempfile
import time

import numpy as np

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.training.obs_roundtrip_fuzz_test import RecordingFuzzPlayer
from main.prober.better_line import better_line_decision
from main.prober.better_line_integration_test import _SumModel
from main.prober.engine import _has_state
from utils.bridge.local_battle_runner import run_local_battles
from utils.bridge.reconstruction import ReconstructionRecord
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder


def record_one(out_dir, tag):
    pool = TeamLoader().get_all_teams()
    trainee = RecordingFuzzPlayer(
        out_dir=out_dir, rng_seed=tag, battle_format="gen3ou", team=Gen3Teambuilder(pool),
        account_configuration=AccountConfiguration(f"BBt{tag}", "pw"),
        server_configuration=LocalhostServerConfiguration,
        start_listening=False, max_concurrent_battles=1)
    opp = RandomPlayer(
        battle_format="gen3ou", team=Gen3Teambuilder(pool),
        account_configuration=AccountConfiguration(f"BBo{tag}", "pw"),
        server_configuration=LocalhostServerConfiguration,
        start_listening=False, max_concurrent_battles=1)
    asyncio.run(run_local_battles(trainee, opp, 1))
    p = trainee.trace_prefixes[0]
    rec = ReconstructionRecord.load(f"{p}_reconstruction.json")
    with open(f"{p}_summary.json") as f:
        summary = json.load(f)
    with np.load(f"{p}_states.npz") as z:
        npz = {k: z[k] for k in z.files}
    return rec, summary, npz


def mid_anchor(summary, npz):
    invs = summary["invocations"]
    c = [i for i, inv in enumerate(invs)
         if inv.get("phase") == "move_selection" and i + 1 < len(invs)
         and _has_state(npz, i) and _has_state(npz, i + 1)]
    return c[len(c) // 2] if c else None


def main():
    reps = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    from utils.contention import warn_if_contended, describe_contention
    warn_if_contended("better_line end-to-end")
    print(describe_contention())

    with tempfile.TemporaryDirectory(prefix="bl_bench_") as td:
        rec, summary, npz = record_one(td, int(time.time() * 1000) % 100000)
    anchor = mid_anchor(summary, npz)
    assert anchor is not None
    print(f"anchor decision {anchor} (turn {summary['invocations'][anchor]['turn']})", flush=True)

    times = {"node": [], "rust": []}
    for r in range(reps):
        order = ("node", "rust") if r % 2 == 0 else ("rust", "node")
        for impl in order:
            t0 = time.perf_counter()
            out = better_line_decision(_SumModel(), rec, summary, npz, anchor,
                                       depth=2, beam=3, top_k=4,
                                       opp_model=_SumModel(), impl=impl)
            times[impl].append((time.perf_counter() - t0) * 1e3)
        print(f"  rep {r + 1}/{reps}: node {times['node'][-1]:.0f} ms  "
              f"rust {times['rust'][-1]:.0f} ms  ({len(out['candidates'])} candidates)", flush=True)

    n, u = statistics.median(times["node"]), statistics.median(times["rust"])
    print("\n" + "=" * 64)
    print("better_line depth=2 beam=3 top_k=4 — END-TO-END wall clock (ms)")
    print("=" * 64)
    for impl in ("node", "rust"):
        v = times[impl]
        print(f"  {impl:<5} median {statistics.median(v):8.1f}   min {min(v):8.1f}   "
              f"max {max(v):8.1f}")
    print(f"\n  speedup (median): {n / u:.2f}x   [disjoint ranges: "
          f"{max(times['rust']) < min(times['node'])}]")
    print("\n  NOTE: V = obs.sum() here, so the model forward is ~free. With a real extractor")
    print("  the model share is impl-invariant, so the real end-to-end gain is LOWER than this.")


if __name__ == "__main__":
    main()
