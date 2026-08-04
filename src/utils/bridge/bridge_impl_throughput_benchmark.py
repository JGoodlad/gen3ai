"""A/B the Node vs Rust bridge child at PRODUCTION env-worker scale — the throughput gate.

``--use-bridge=rust`` swaps the bridge child binary for the std-only Rust ``sim_bridge``. It is
protocol-compatible (``harness/gen_sim_bridge_diff.js``) and byte-for-byte with Node, but nothing
measured whether it is as FAST — and the only numbers on record (node 798 fps vs rust 427 fps at
8 envs) were taken on a contended box, so they are not an A/B.

This is that A/B. It reproduces the training rollout's transport shape *without* the policy or the
GPU: ``--workers`` independent PROCESSES (the ``SubprocVecEnv`` env workers), each owning one
``Gen3Env`` + one persistent bridge child, stepping random legal actions through the real
obs/reward stack. Aggregate env-steps/sec is the number ``--use-bridge`` actually buys.

Why processes and not threads: the per-env cost is CPU-bound Python (obs build) plus a blocking
pipe round-trip to the child. Only separate processes reproduce the real contention — at
``--workers 48`` on a 16-core box, 48 python workers AND 48 bridge children compete, which is
exactly where the transport choice starts to matter.

``both`` runs the two impls back-to-back in the same invocation (same teams, same seeds, same box
state) so the ratio is the load-stable signal — absolute fps scales with whatever else the box is
doing. Run on an otherwise-idle box for a clean baseline.

Not a pytest target; run as a script. In-process bridge, no server.

    export PYTHONPATH=$PYTHONPATH:src
    python src/utils/bridge/bridge_impl_throughput_benchmark.py --workers 8 --seconds 60
    python src/utils/bridge/bridge_impl_throughput_benchmark.py --workers 48 --seconds 120
    python src/utils/bridge/bridge_impl_throughput_benchmark.py --impl rust --workers 1 --seconds 30
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time

# Match the launcher's thread pinning (src/main/launcher/child.py). Without it torch spins a
# default-sized thread pool PER worker and a 48-worker run thrashes a 16-core box into
# meaninglessness — measuring the scheduler, not the bridge. Must precede the torch import.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

_MAX_EPISODE_STEPS = 1500


def _build_env(impl, teams, worker_id):
    """One Gen3Env wired to a persistent bridge child of ``impl``, plus a random opponent."""
    import numpy as np  # noqa: F401  (imported for the caller's rng; keeps child imports lazy)
    from poke_env import AccountConfiguration
    from poke_env.player import RandomPlayer
    from poke_env.environment.single_agent_wrapper import SingleAgentWrapper

    from agents.observation.state_encoder import load_mappings
    from agents.training.gen3_env import Gen3Env
    from utils.bridge.bridge_session import attach_bridge_transport
    from utils.teambuilder import Gen3Teambuilder

    env = Gen3Env(
        load_mappings(), battle_format="gen3ou", team=Gen3Teambuilder(teams),
        account_configuration1=AccountConfiguration(f"ThruB{worker_id}", None),
        start_listening=False,
    )
    # recycle_every=0: a recycle mid-measurement would charge one worker a spawn the others
    # never pay, which shows up as variance rather than signal.
    attach_bridge_transport(env, battle_format="gen3ou", persistent=True, recycle_every=0, impl=impl)
    opp = RandomPlayer(
        battle_format="gen3ou", team=Gen3Teambuilder(teams),
        account_configuration=AccountConfiguration(f"ThruBOpp{worker_id}", None),
        start_listening=False,
    )
    w = SingleAgentWrapper(env, opp)
    w.action_space = env.action_space
    w.observation_space = env.observation_space
    return w


def _worker(impl, worker_id, seconds, warmup_steps, barrier, result_q):
    """Play random-action episodes for ``seconds`` and report (steps, battles, spawn_s).

    ``barrier`` makes every worker start its timed window together — a staggered start would
    let early workers bank throughput on an idle box and read as a speedup.
    """
    try:
        import numpy as np
        from utils.team_loader.loader import TeamLoader

        rng = np.random.default_rng(1000 + worker_id)
        teams = TeamLoader().get_sample_teams() or TeamLoader().get_all_teams()

        t_spawn = time.perf_counter()
        w = _build_env(impl, teams, worker_id)
        spawn_s = time.perf_counter() - t_spawn

        def play(deadline=None, step_budget=None):
            """Step until the deadline / budget; returns (steps, completed battles)."""
            steps = battles = 0
            obs, _ = w.reset()
            ep_step = 0
            while True:
                if deadline is not None and time.perf_counter() >= deadline:
                    return steps, battles
                if step_budget is not None and steps >= step_budget:
                    return steps, battles
                mask = np.asarray(obs["action_mask"]).astype(bool)
                legal = np.flatnonzero(mask)
                action = int(rng.choice(legal)) if legal.size else 0
                obs, _r, term, trunc, _i = w.step(action)
                steps += 1
                ep_step += 1
                if term or trunc or ep_step >= _MAX_EPISODE_STEPS:
                    battles += 1
                    ep_step = 0
                    obs, _ = w.reset()

        # Warm up OUTSIDE the timed window: the first steps pay lazy data-table loads (and, for
        # node, V8 JIT warmup) that a production worker amortizes over hours.
        play(step_budget=warmup_steps)

        barrier.wait()
        t0 = time.perf_counter()
        steps, battles = play(deadline=t0 + seconds)
        elapsed = time.perf_counter() - t0

        child_rss = _bridge_child_rss_mb(w)
        try:
            w.env.close()
        except Exception:
            pass
        result_q.put({
            "worker": worker_id, "steps": steps, "battles": battles,
            "elapsed": elapsed, "spawn_s": spawn_s, "child_rss_mb": child_rss,
        })
    except Exception as e:  # a dead worker must be LOUD, never a silently-missing sample
        import traceback
        result_q.put({"worker": worker_id, "error": f"{type(e).__name__}: {e}",
                      "traceback": traceback.format_exc()[-2000:]})


def _bridge_child_rss_mb(w):
    """RSS of this worker's bridge child, or None if it can't be read."""
    try:
        proc = w.env._bridge_session._proc
        with open(f"/proc/{proc.pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        return None
    return None


def _run_impl(impl, workers, seconds, warmup_steps):
    """Run one impl's worker fleet; returns the aggregate result dict."""
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(workers)
    result_q = ctx.Queue()
    procs = [
        ctx.Process(target=_worker, args=(impl, i, seconds, warmup_steps, barrier, result_q))
        for i in range(workers)
    ]
    print(f"\n=== impl={impl} workers={workers} — spawning + warming up "
          f"({warmup_steps} steps/worker) ===", flush=True)
    t_wall = time.perf_counter()
    for p in procs:
        p.start()

    results, errors = [], []
    # Generous timeout: spawn + warmup + the timed window + teardown.
    deadline = time.perf_counter() + seconds + 900
    while len(results) + len(errors) < workers and time.perf_counter() < deadline:
        try:
            r = result_q.get(timeout=5)
        except Exception:
            if not any(p.is_alive() for p in procs):
                break
            continue
        (errors if "error" in r else results).append(r)

    for p in procs:
        p.join(timeout=30)
        if p.is_alive():
            p.terminate()
    wall = time.perf_counter() - t_wall

    for e in errors:
        print(f"  ❌ worker {e['worker']}: {e['error']}\n{e.get('traceback','')}", flush=True)
    if not results:
        raise RuntimeError(f"impl={impl}: every worker failed — no throughput sample")

    total_steps = sum(r["steps"] for r in results)
    total_battles = sum(r["battles"] for r in results)
    mean_elapsed = sum(r["elapsed"] for r in results) / len(results)
    fps = total_steps / mean_elapsed
    per_worker = fps / len(results)
    rss = [r["child_rss_mb"] for r in results if r["child_rss_mb"]]
    spawn = [r["spawn_s"] for r in results]
    out = {
        "impl": impl, "ok_workers": len(results), "failed_workers": len(errors),
        "total_steps": total_steps, "battles": total_battles, "fps": fps,
        "fps_per_worker": per_worker, "ms_per_step": 1000.0 * len(results) / fps,
        "child_rss_mb": (sum(rss) / len(rss)) if rss else None,
        "spawn_s_max": max(spawn), "wall_s": wall,
    }
    print(f"  {impl}: {fps:8.1f} steps/s aggregate | {per_worker:6.1f}/worker | "
          f"{out['ms_per_step']:5.2f} ms/step | {total_battles} battles | "
          f"child RSS {out['child_rss_mb'] or float('nan'):.0f} MB | "
          f"spawn≤{out['spawn_s_max']:.1f}s | {len(errors)} failed", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impl", default="both", choices=["node", "rust", "both"])
    ap.add_argument("--workers", type=int, default=8, help="parallel env workers (processes)")
    ap.add_argument("--seconds", type=float, default=60.0, help="timed window per impl")
    ap.add_argument("--warmup-steps", type=int, default=200, help="untimed steps per worker")
    args = ap.parse_args()

    try:
        load = os.getloadavg()[0]
    except OSError:
        load = float("nan")
    print(f"[throughput] workers={args.workers} seconds={args.seconds:.0f} "
          f"cores={os.cpu_count()} load1={load:.2f}", flush=True)
    if load > args.workers * 0.5:
        print("  ⚠️  the box is ALREADY loaded — absolute fps will be depressed; trust the "
              "node-vs-rust RATIO, not the numbers.", flush=True)

    impls = ["node", "rust"] if args.impl == "both" else [args.impl]
    results = {}
    for impl in impls:
        results[impl] = _run_impl(impl, args.workers, args.seconds, args.warmup_steps)

    if len(results) == 2:
        n, r = results["node"]["fps"], results["rust"]["fps"]
        verdict = ("rust FASTER" if r > n * 1.02 else
                   "rust SLOWER" if r < n * 0.98 else "PARITY")
        print(f"\n=== VERDICT @ workers={args.workers}: {verdict} — "
              f"rust/node = {r / n:.3f}x  (node {n:.0f} vs rust {r:.0f} steps/s) ===", flush=True)
    return 0


if __name__ == "__main__":
    # A benchmark on a busy box reports a confidently wrong number — say so up front.
    from utils.contention import warn_if_contended
    warn_if_contended("bridge-impl throughput")
    sys.exit(main())
