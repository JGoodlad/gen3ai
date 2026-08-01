"""Long-running fuzz + multi-env soak for the ``BridgeSession`` env transport — the RL path.

Unlike the protocol/obs-parity fuzzes (which drive whole battles via the *synchronous*
``run_local_battles``), this hammers the **async env transport**: a real ``Gen3Env`` driven the
gym way (action in via ``step()``, the background pump feeds battle state) over thousands of
episodes on a **persistent** bridge child. It targets the risky surface — child reuse across
battles (no state leak / tag collision), the single long-lived reader, the
``__END__``→``_battle_ended`` handoff, the force-switch path, long stall battles, and mid-battle
forfeit-resets — and asserts the invariants on every episode, failing immediately with detail.

Two extras over a plain fuzz:
  * **Per-phase timing.** Each episode is split into reset (incl. the inter-battle ``__END__``
    wait) vs the step loop, with the slowest single step tracked. Slow episodes (> ``--slow-ms``)
    print a breakdown so an outlier-slow episode is *attributable* (child descheduled under load
    vs a genuine transport stall) rather than a mystery.
  * **Multi-env soak (``--workers N``).** N parallel single-env loops → N concurrent persistent
    children, reproducing production-scale load (the conditions under which an outlier appears).

Not a pytest target (no ``test_*`` → collected-but-empty); run as a script. Real battles via the
in-process bridge, **no server** — never touches the :8001 training server.

    export PYTHONPATH=$PYTHONPATH:src
    python src/utils/bridge/bridge_session_fuzz_test.py 2000               # 2000 episodes, 1 env
    python src/utils/bridge/bridge_session_fuzz_test.py 90m                # 90 min, 1 env
    python src/utils/bridge/bridge_session_fuzz_test.py 30m --workers 64   # 64 concurrent children
    python src/utils/bridge/bridge_session_fuzz_test.py 500 --spawn        # spawn-per-battle mode
    python src/utils/bridge/bridge_session_fuzz_test.py 30m --slow-ms 3000 # report episodes > 3s
    python src/utils/bridge/bridge_session_fuzz_test.py 2000 --impl rust   # the Rust sim_bridge
    python src/utils/bridge/bridge_session_fuzz_test.py 30m --impl rust --workers 48

``--impl`` picks the bridge child binary (``node`` default, ``rust`` for the pokesim
``sim_bridge``). Both must pass: this is the ONLY gate on the persistent-child reuse and
forfeit-reset paths, which the protocol-level diff harness (spawn-per-battle) never exercises.
"""

from __future__ import annotations

import multiprocessing as mp
import sys
import time

import numpy as np

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper

from agents.observation.state_encoder import load_mappings
from agents.training.gen3_env import Gen3Env
from utils.bridge.bridge_session import attach_bridge_transport
from utils.team_loader.loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

# A gen3ou battle that hasn't ended within this many decisions is a hang/loop, not a long game
# (the trainer forfeits stalls at ~250 turns). Tripping this is a real failure.
_MAX_STEPS_PER_EPISODE = 1500
# A single episode past this wall-clock is a deadlock (a healthy bridge episode is < ~2s).
_EPISODE_WALL_BUDGET_S = 120.0


def _teams():
    loader = TeamLoader()
    return loader.get_sample_teams() or loader.get_all_teams()


def _build(teams, idx: int, persistent: bool, impl: str = "node"):
    env = Gen3Env(
        load_mappings(), battle_format="gen3ou", team=Gen3Teambuilder(teams),
        account_configuration1=AccountConfiguration(f"Fuzz{idx}", None),
        start_listening=False,
    )
    attach_bridge_transport(env, battle_format="gen3ou", persistent=persistent, impl=impl)
    opponent = RandomPlayer(
        battle_format="gen3ou", team=Gen3Teambuilder(teams),
        account_configuration=AccountConfiguration(f"FuzzOpp{idx}", None),
        start_listening=False,
    )
    w = SingleAgentWrapper(env, opponent)
    w.action_space = env.action_space
    w.observation_space = env.observation_space
    return w


def _check_obs(obs, expected_dim: int):
    o = np.asarray(obs["observation"], dtype=np.float64)
    if o.shape != (expected_dim,):
        raise AssertionError(f"obs shape {o.shape} != ({expected_dim},)")
    if not np.all(np.isfinite(o)):
        bad = np.where(~np.isfinite(o))[0][:8]
        raise AssertionError(f"non-finite obs values at indices {bad.tolist()}")
    m = np.asarray(obs["action_mask"])
    if m.shape != (11,):
        raise AssertionError(f"mask shape {m.shape} != (11,)")
    if int(m.sum()) == 0:
        raise AssertionError("action mask has no legal action while still playing")


def _play_episode(w, rng, expected_dim, early_reset_at):
    """Play one episode; return (result, stats). result is 'finished' or 'early'.

    stats keys: steps, turns, total_ms, reset_ms, end_wait_ms (inter-battle __END__ wait),
    step_loop_ms, slowest_step_ms, slowest_step_idx.
    `early_reset_at` (or None): stop mid-battle after that many steps WITHOUT finishing, so the
    caller's next reset() exercises the forfeit / unfinished-battle path on the persistent child.
    """
    t_start = time.perf_counter()
    obs, _ = w.reset()
    reset_ms = (time.perf_counter() - t_start) * 1000.0
    end_wait_ms = getattr(w.env._bridge_session, "_last_end_wait_s", 0.0) * 1000.0

    slowest_step_ms = 0.0
    slowest_step_idx = -1
    loop0 = time.perf_counter()
    result = None
    steps = 0
    for step in range(1, _MAX_STEPS_PER_EPISODE + 1):
        _check_obs(obs, expected_dim)
        mask = np.asarray(obs["action_mask"]).astype(bool)
        legal = np.flatnonzero(mask)
        action = int(rng.choice(legal)) if legal.size else 0
        s0 = time.perf_counter()
        obs, _r, term, trunc, _i = w.step(action)
        s_ms = (time.perf_counter() - s0) * 1000.0
        if s_ms > slowest_step_ms:
            slowest_step_ms, slowest_step_idx = s_ms, step
        steps = step
        if term or trunc:
            result = "finished"
            break
        if early_reset_at is not None and step >= early_reset_at:
            result = "early"
            break
        if time.perf_counter() - t_start > _EPISODE_WALL_BUDGET_S:
            raise AssertionError(
                f"episode wall-clock exceeded {_EPISODE_WALL_BUDGET_S}s at step {step} "
                f"(turn {w.env.battle1.turn}) — deadlock?"
            )
    else:
        raise AssertionError(
            f"episode exceeded {_MAX_STEPS_PER_EPISODE} steps without finishing — hang/loop?"
        )

    step_loop_ms = (time.perf_counter() - loop0) * 1000.0
    stats = {
        "steps": steps, "turns": w.env.battle1.turn,
        "total_ms": (time.perf_counter() - t_start) * 1000.0,
        "reset_ms": reset_ms, "end_wait_ms": end_wait_ms, "step_loop_ms": step_loop_ms,
        "slowest_step_ms": slowest_step_ms, "slowest_step_idx": slowest_step_idx,
    }
    return result, stats


_VALUE_FLAGS = {"--workers", "--slow-ms"}


def _parse_budget(argv):
    # First bare (non-flag) token is the budget: "2000" → 2000 episodes, "90m"/"30s" → time.
    # Skip tokens that are VALUES of --workers / --slow-ms (also bare numbers).
    skip = {i + 1 for i, a in enumerate(argv) if a in _VALUE_FLAGS}
    arg = next((a for i, a in enumerate(argv)
                if not a.startswith("-") and i not in skip), None)
    if arg is None:
        return "count", 1000
    if arg.endswith("m"):
        return "time", float(arg[:-1]) * 60.0
    if arg.endswith("s"):
        return "time", float(arg[:-1])
    return "count", int(arg)


def _fmt_slow(idx_prefix, ep, stats):
    return (f"{idx_prefix}⏱  ep {ep}: total {stats['total_ms']:.0f}ms = reset "
            f"{stats['reset_ms']:.0f}ms (end_wait {stats['end_wait_ms']:.0f}ms) + step_loop "
            f"{stats['step_loop_ms']:.0f}ms over {stats['steps']} steps; slowest step "
            f"{stats['slowest_step_ms']:.0f}ms @ {stats['slowest_step_idx']} | turns {stats['turns']}")


def _run_single(idx, mode, budget, persistent, slow_ms, seed, prefix="", impl="node"):
    """One env's soak loop. Returns a summary dict; raises on any invariant failure."""
    teams = _teams()
    rng = np.random.default_rng(seed)
    w = _build(teams, idx=idx, persistent=persistent, impl=impl)
    expected_dim = w.env.observation_encoder.dimension

    start = time.perf_counter()
    finished = early = total_steps = max_turns = 0
    slowest = {"total_ms": 0.0}
    worst_end_wait = 0.0
    pids = set()
    ep = 0
    try:
        while True:
            if mode == "time" and time.perf_counter() - start >= budget:
                break
            if mode == "count" and ep >= int(budget):
                break
            ep += 1
            early_reset_at = int(rng.integers(2, 8)) if ep % 9 == 0 else None
            res, stats = _play_episode(w, rng, expected_dim, early_reset_at)

            total_steps += stats["steps"]
            max_turns = max(max_turns, stats["turns"])
            worst_end_wait = max(worst_end_wait, stats["end_wait_ms"])
            if stats["total_ms"] > slowest["total_ms"]:
                slowest = {**stats, "ep": ep}
            if stats["total_ms"] >= slow_ms:
                print(_fmt_slow(prefix, ep, stats), flush=True)

            if res == "finished":
                finished += 1
                b = w.env.battle1
                if not b.finished:
                    raise AssertionError(f"ep {ep}: terminated but battle not finished")
                if b.turn <= 0:
                    raise AssertionError(f"ep {ep}: finished battle has turn {b.turn}")
                if b.player_role != "p1":
                    raise AssertionError(f"ep {ep}: player_role {b.player_role!r} != 'p1'")
            else:
                early += 1

            proc = w.env._bridge_session._proc
            if proc is not None:
                pids.add(proc.pid)
            if persistent and len(pids) > 1:
                raise AssertionError(f"ep {ep}: persistent child respawned — PIDs {pids}")

            if ep % 200 == 0:
                elapsed = time.perf_counter() - start
                rate = ep / elapsed if elapsed else 0
                print(f"{prefix}ep {ep:6d} | finished {finished} early {early} | "
                      f"steps {total_steps} | max_turns {max_turns} | "
                      f"slowest_ep {slowest['total_ms']:.0f}ms | worst_end_wait "
                      f"{worst_end_wait:.0f}ms | {rate:.1f} ep/s | pids {len(pids)}", flush=True)
    finally:
        w.close()

    return {
        "idx": idx, "episodes": ep, "finished": finished, "early": early,
        "steps": total_steps, "max_turns": max_turns, "pids": len(pids),
        "elapsed_s": time.perf_counter() - start, "slowest": slowest,
        "worst_end_wait_ms": worst_end_wait,
    }


def _worker_entry(idx, mode, budget, persistent, slow_ms, seed, impl="node"):
    prefix = f"[w{idx:02d}] "
    try:
        s = _run_single(idx, mode, budget, persistent, slow_ms, seed, prefix=prefix, impl=impl)
        sl = s["slowest"]
        print(f"{prefix}✅ {s['episodes']} eps ({s['finished']} fin, {s['early']} early), "
              f"slowest_ep {sl.get('total_ms', 0):.0f}ms (end_wait {sl.get('end_wait_ms', 0):.0f}ms, "
              f"slowest_step {sl.get('slowest_step_ms', 0):.0f}ms), pids {s['pids']}", flush=True)
        sys.exit(0)
    except Exception as e:
        print(f"{prefix}❌ FUZZ FAILED: {type(e).__name__}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    persistent = "--spawn" not in sys.argv
    slow_ms = 5000.0
    if "--slow-ms" in sys.argv:
        slow_ms = float(sys.argv[sys.argv.index("--slow-ms") + 1])
    workers = 1
    if "--workers" in sys.argv:
        workers = int(sys.argv[sys.argv.index("--workers") + 1])
    # The bridge child binary under test. This fuzz only ever exercised `node`, which is exactly
    # how the Rust bridge reached a training run with its persistent-child / forfeit-reset path
    # ungated — `--impl rust` closes that hole.
    impl = "node"
    if "--impl" in sys.argv:
        impl = sys.argv[sys.argv.index("--impl") + 1]
    mode, budget = _parse_budget(sys.argv[1:])
    label = f"persistent, {impl}" if persistent else f"spawn-per-battle, {impl}"
    desc = f"{budget/60:.0f} min" if mode == "time" else f"{int(budget)} episodes"

    if workers <= 1:
        print(f"[fuzz] bridge env ({label}) — {desc}, 1 env, slow>{slow_ms:.0f}ms", flush=True)
        try:
            s = _run_single(1, mode, budget, persistent, slow_ms, seed=0, impl=impl)
        except Exception as e:
            print(f"\n❌ FUZZ FAILED: {type(e).__name__}: {e}", flush=True)
            raise
        sl = s["slowest"]
        print(f"\n✅ [fuzz] OK — {s['episodes']} eps ({s['finished']} fin, {s['early']} early) in "
              f"{s['elapsed_s']:.0f}s | {s['steps']} steps | max_turns {s['max_turns']} | "
              f"slowest_ep {sl.get('total_ms', 0):.0f}ms (end_wait {sl.get('end_wait_ms', 0):.0f}ms, "
              f"slowest_step {sl.get('slowest_step_ms', 0):.0f}ms @ ep {sl.get('ep', '?')}) | "
              f"worst_end_wait {s['worst_end_wait_ms']:.0f}ms | "
              f"{'reused 1 child' if persistent and s['pids'] == 1 else label}", flush=True)
        return

    # Multi-env soak: N independent processes, each its own env + persistent child.
    print(f"[soak] bridge env ({label}) — {desc} × {workers} concurrent children, "
          f"slow>{slow_ms:.0f}ms", flush=True)
    ctx = mp.get_context("spawn")
    procs = []
    for i in range(workers):
        p = ctx.Process(target=_worker_entry,
                        args=(i, mode, budget, persistent, slow_ms, i + 1, impl))
        p.start()
        procs.append(p)
        # Stagger starts so N simultaneous spawn-imports (torch/agents/...) don't thundering-herd.
        if workers > 8:
            time.sleep(0.2)
    failures = 0
    for p in procs:
        p.join()
        if p.exitcode != 0:
            failures += 1
    if failures:
        print(f"\n❌ [soak] {failures}/{workers} workers FAILED", flush=True)
        sys.exit(1)
    print(f"\n✅ [soak] OK — all {workers} workers passed", flush=True)


if __name__ == "__main__":
    main()
