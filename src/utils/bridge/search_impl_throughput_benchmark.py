"""A/B the Node vs Rust OFFLINE SEARCH driver — the latency gate for `better_line`.

`gen3_rust_search_driver_v1` / `gen3_rust_replay_driver_v1` gave the clone-and-branch search a
second engine (`src/rust_sim/src/bin/search_driver.rs`). Equivalence is already pinned bit-for-bit
(`src/rust_sim/harness/search_impl_parity.py`, `src/rust_sim/harness/replay_impl_parity.py`, `search_clone_parity_fuzz_test --impl
rust`, the cross-impl `better_line_integration_test`). **This measures whether it is FASTER**, which
is the only remaining reason to prefer one.

WHAT IS ACTUALLY ON THE CLOCK. A `better_line` call is dominated by three driver operations:

  * `open_root`   — rebuild the recorded battle from `>start` to the start of turn T. Cost grows
                    with T (it replays the prefix), and it is paid ONCE per search.
  * `expand_many` — clone the parent node, apply one joint turn, return the suffix. Paid
                    `beam x top_k` times per ply. THE hot path, and the reason the warm search
                    server exists at all: node clones via `State.serializeBattle` (~1.7 ms),
                    rust via a derived deep `Clone` of `BridgeSession`.
  * `reroll_many` — the independent oracle: rebuild from turn 1 PER ARM. Deliberately not a clone,
                    so it is the "what the clone saved you" baseline.

Also reported: child RSS (the search-teacher runs `--teacher-workers` of these concurrently, and
the live bridge child is ~224 MB node vs ~9 MB rust, so this is worth knowing), and process spawn
+ first-response latency (paid per `better_line` when no warm session is injected).

CONTENTION. This project has a recorded incident where a node-vs-rust throughput result was taken
on a saturated box and had to be superseded **with the conclusion reversed**. Two defences here:

1. `warn_if_contended()` at entry — a benchmark's output IS the measurement, so bounds are never
   scaled, only announced.
2. **Per-rep INTERLEAVING** (`node, rust, node, rust, ...` rather than all-node-then-all-rust).
   A back-to-back A/B is only load-stable if the load is constant across the two halves; a live
   training run's load is not. Interleaving puts both arms under the same drifting load, so the
   RATIO stays meaningful even mid-run. The reported spread across reps tells you whether to
   believe it: a tight IQR under load is a real signal, a wide one is not.

Absolute ms scale with whatever else the box is doing. **The ratio and the spread are the
load-stable signal.** Not a pytest target; run as a script. No server.

    python src/utils/bridge/search_impl_throughput_benchmark.py
    python src/utils/bridge/search_impl_throughput_benchmark.py --battles 3 --reps 8 --turns 5,15,30
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import tempfile
import time

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from utils.bridge.local_battle_runner import run_local_battles
from utils.bridge.reconstruction import ReconstructionRecord, reroll_many
from utils.bridge.search_session import SearchSession
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

IMPLS = ("node", "rust")


# ---------------------------------------------------------------------------
# Fixture: real bridge battles (the records a real `better_line` would search)
# ---------------------------------------------------------------------------

def _record_battles(n: int, record_impl: str):
    """Play `n` real gen3ou battles in-process and return their reconstruction records."""
    out = []
    pool = TeamLoader().get_all_teams()
    with tempfile.TemporaryDirectory(prefix="search_bench_") as td:
        from agents.training.obs_roundtrip_fuzz_test import RecordingFuzzPlayer
        for i in range(n):
            tag = (int(time.time() * 1000) + i * 7919) % 100000
            trainee = RecordingFuzzPlayer(
                out_dir=td, rng_seed=tag, battle_format="gen3ou", team=Gen3Teambuilder(pool),
                account_configuration=AccountConfiguration(f"SBt{tag}", "pw"),
                server_configuration=LocalhostServerConfiguration,
                start_listening=False, max_concurrent_battles=1)
            opp = RandomPlayer(
                battle_format="gen3ou", team=Gen3Teambuilder(pool),
                account_configuration=AccountConfiguration(f"SBo{tag}", "pw"),
                server_configuration=LocalhostServerConfiguration,
                start_listening=False, max_concurrent_battles=1)
            asyncio.run(run_local_battles(trainee, opp, 1, impl=record_impl))
            prefix = trainee.trace_prefixes[0]
            rec = ReconstructionRecord.load(f"{prefix}_reconstruction.json")
            out.append(rec)
    return out


def _arms(node_id, n_arms: int):
    """A realistic beam batch: one CRN anchor + explicit candidates under fresh dice."""
    arms = [{"node_id": node_id, "label": 0, "recorded_exact": True}]
    for k in range(1, n_arms):
        arms.append({"node_id": node_id, "label": k,
                     "p1_action": f"move {1 + (k % 4)}", "p2_action": "random",
                     "seed": f"sodium,{k:016x}"})
    return arms


def _rss_kb(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1])
    except Exception:
        pass
    return 0


# ---------------------------------------------------------------------------
# One measured rep: spawn a session, time each operation on one record
# ---------------------------------------------------------------------------

def _one_rep(impl: str, record, turns, n_arms: int, warm_turn: int) -> dict:
    """One session's worth of measurements.

    COLD vs WARM is separated deliberately. `subprocess.Popen` returns immediately, so timing it
    measures nothing — the child's real startup (node: V8 boot + `require`-ing the Showdown dist;
    rust: an execve and a ~16 MB dex load) lands on whatever the FIRST request happens to be. A
    first cut folded it into `open_root(turn 5)` and produced the giveaway that turn 5 looked
    slower than turn 15, which is backwards since the prefix replay is shorter. So:

      * `cold_ms`  = spawn -> first `open_root` returns. What a `better_line` pays when no warm
                     session is injected. Reported on its own; it is a real cost, not an artifact.
      * everything else is measured AFTER a discarded warmup request, so it is steady state.
    """
    m = {"cold_ms": None, "open_root_ms": {}, "expand_ms": {}, "rss_kb": 0}

    t0 = time.perf_counter()
    ss = SearchSession(record, impl=impl)
    try:
        # --- COLD: spawn -> first response (includes interpreter/dex startup) ---
        warmed = False
        try:
            root = ss.open_root(warm_turn)
            m["cold_ms"] = (time.perf_counter() - t0) * 1e3
            ss.expand_many(_arms(root.node_id, n_arms))   # warm the expand path too
            warmed = True
        except Exception:
            pass
        if not warmed:
            return m                            # this record never reached the warmup turn

        # --- WARM: steady-state per-operation cost ---
        for T in turns:
            t1 = time.perf_counter()
            try:
                root = ss.open_root(T)
            except Exception:
                continue                       # battle never reached turn T — skip, don't crash
            m["open_root_ms"][T] = (time.perf_counter() - t1) * 1e3

            t2 = time.perf_counter()
            ss.expand_many(_arms(root.node_id, n_arms))
            m["expand_ms"][T] = (time.perf_counter() - t2) * 1e3 / n_arms   # PER ARM

        m["rss_kb"] = _rss_kb(ss._proc.pid) if ss._proc else 0
    finally:
        ss.close()
    return m


def _reroll_rep(impl: str, record, turn: int, n_arms: int):
    """The rebuild-per-arm oracle — what a clone-less search would cost."""
    arms = [{"p1_action": f"move {1 + (k % 4)}", "p2_action": "random",
             "seed": f"sodium,{k:016x}", "label": k} for k in range(n_arms)]
    t0 = time.perf_counter()
    try:
        reroll_many(record, turn, arms, impl=impl)
    except Exception:
        return None
    return (time.perf_counter() - t0) * 1e3 / n_arms


def _fmt(vals):
    if not vals:
        return "     n/a"
    return f"{statistics.median(vals):8.2f}"


def _spread(vals):
    if len(vals) < 4:
        return ""
    q = statistics.quantiles(vals, n=4)
    return f"  [IQR {q[0]:.2f}–{q[2]:.2f}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--battles", type=int, default=2, help="records to search over")
    ap.add_argument("--reps", type=int, default=6, help="interleaved node/rust reps per record")
    ap.add_argument("--turns", default="5,15,30",
                    help="decision turns to open a root at (prefix cost grows with turn)")
    ap.add_argument("--arms", type=int, default=8, help="arms per expand_many batch")
    ap.add_argument("--record-impl", choices=IMPLS, default="node",
                    help="engine that PLAYS the fixture battles (not under test)")
    ap.add_argument("--json", default=None, help="also write the raw numbers here")
    a = ap.parse_args()

    from utils.contention import warn_if_contended
    warn_if_contended("search-impl latency")

    turns = [int(x) for x in a.turns.split(",")]
    print(f"recording {a.battles} fixture battle(s) on {a.record_impl} ...", flush=True)
    records = _record_battles(a.battles, a.record_impl)

    warm_turn = min(turns)
    acc = {i: {"cold": [], "open": {T: [] for T in turns}, "expand": {T: [] for T in turns},
               "rss": [], "reroll": []} for i in IMPLS}

    print(f"interleaving {a.reps} reps x {len(records)} records x {len(turns)} turns "
          f"({a.arms} arms/expand) ...", flush=True)
    for rep in range(a.reps):
        for rec in records:
            # INTERLEAVED, and the order FLIPS every rep so neither impl systematically gets the
            # warmer page cache / the quieter slice of a drifting load.
            order = IMPLS if rep % 2 == 0 else tuple(reversed(IMPLS))
            for impl in order:
                m = _one_rep(impl, rec, turns, a.arms, warm_turn)
                if m["cold_ms"] is not None:
                    acc[impl]["cold"].append(m["cold_ms"])
                for T, v in m["open_root_ms"].items():
                    acc[impl]["open"][T].append(v)
                for T, v in m["expand_ms"].items():
                    acc[impl]["expand"][T].append(v)
                if m["rss_kb"]:
                    acc[impl]["rss"].append(m["rss_kb"])
                r = _reroll_rep(impl, rec, turns[len(turns) // 2], a.arms)
                if r is not None:
                    acc[impl]["reroll"].append(r)
        print(f"  rep {rep + 1}/{a.reps} done", flush=True)

    print("\n" + "=" * 78)
    print("SEARCH DRIVER LATENCY — node vs rust (medians, ms; lower is better)")
    print("=" * 78)
    print("(everything below COLD is steady state — measured after a discarded warmup request)")
    print(f"{'operation':<28}{'node':>10}{'rust':>10}{'speedup':>10}")
    print("-" * 78)

    def row(label, nv, rv, note=""):
        if not nv or not rv:
            print(f"{label:<28}{_fmt(nv):>10}{_fmt(rv):>10}{'n/a':>10}")
            return
        n, r = statistics.median(nv), statistics.median(rv)
        sp = f"{n / r:.2f}x" if r > 0 else "n/a"
        print(f"{label:<28}{n:>10.2f}{r:>10.2f}{sp:>10}  {note}")

    row("COLD spawn -> 1st root", acc["node"]["cold"], acc["rust"]["cold"],
        "paid once, no warm session")
    for T in turns:
        row(f"open_root(turn {T})", acc["node"]["open"][T], acc["rust"]["open"][T],
            "once per search")
    for T in turns:
        row(f"expand_many/arm(turn {T})", acc["node"]["expand"][T], acc["rust"]["expand"][T],
            "THE hot path")
    row("reroll_many/arm", acc["node"]["reroll"], acc["rust"]["reroll"], "clone-less oracle")

    print("-" * 78)
    nr = acc["node"]["rss"]
    rr = acc["rust"]["rss"]
    if nr and rr:
        print(f"{'child RSS (MB)':<28}{statistics.median(nr) / 1024:>10.1f}"
              f"{statistics.median(rr) / 1024:>10.1f}"
              f"{statistics.median(nr) / max(1, statistics.median(rr)):>9.1f}x  smaller is better")

    mid = turns[len(turns) // 2]
    print("\nspread (is the ratio believable under this load?):")
    for T in (mid,):
        print(f"  expand/arm node{_spread(acc['node']['expand'][T])}")
        print(f"  expand/arm rust{_spread(acc['rust']['expand'][T])}")

    if a.json:
        with open(a.json, "w") as f:
            json.dump({i: {"cold": acc[i]["cold"], "rss_kb": acc[i]["rss"],
                           "reroll": acc[i]["reroll"],
                           "open": {str(T): acc[i]["open"][T] for T in turns},
                           "expand": {str(T): acc[i]["expand"][T] for T in turns}}
                       for i in IMPLS}, f)
        print(f"\nraw -> {a.json}")


if __name__ == "__main__":
    main()
