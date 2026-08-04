"""Measure a PERSISTENT bridge child's memory growth over many battles → pick a recycle interval.

The launcher restarts the Python training child every ~3h, which tears down and respawns every
bridge child (fresh heap). So the only question is: does ONE bridge child's RSS grow fast enough
to matter WITHIN a 3h window? If it's flat, no recycle is needed; if it climbs, set
``recycle_every`` (or fix the JS) to bound it.

This runs one persistent bridge env with recycling DISABLED (recycle_every=0), plays battles, and
samples the Node child's RSS (``/proc/<pid>/VmRSS``) every ``--sample-every`` episodes. It reports
the growth rate (MB per 1000 battles, from a linear fit past a warmup) and extrapolates to a 3h
window at the *production* per-env battle rate (derived from a measured FPS and mean episode
length), then recommends a recycle interval.

Not a pytest target; run as a script. In-process bridge, no server.

    export PYTHONPATH=$PYTHONPATH:src
    python src/utils/bridge/bridge_heap_growth_benchmark.py 25m                # 25 min, sample/50
    python src/utils/bridge/bridge_heap_growth_benchmark.py 3000 --sample-every 100
    python src/utils/bridge/bridge_heap_growth_benchmark.py 20m --prod-fps 1200 --prod-envs 64
"""

from __future__ import annotations

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

_MAX_STEPS = 1500


def _teams():
    loader = TeamLoader()
    return loader.get_sample_teams() or loader.get_all_teams()


def _rss_mb(pid):
    """Resident set size of `pid` in MB, or None if it's gone."""
    try:
        with open(f"/proc/{pid}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except (FileNotFoundError, ProcessLookupError, ValueError):
        return None
    return None


def _build(teams):
    env = Gen3Env(
        load_mappings(), battle_format="gen3ou", team=Gen3Teambuilder(teams),
        account_configuration1=AccountConfiguration("HeapGrow", None),
        start_listening=False,
    )
    # recycle_every=0 → never recycle, so we observe the RAW growth of one child.
    attach_bridge_transport(env, battle_format="gen3ou", persistent=True, recycle_every=0)
    opp = RandomPlayer(
        battle_format="gen3ou", team=Gen3Teambuilder(teams),
        account_configuration=AccountConfiguration("HeapGrowOpp", None),
        start_listening=False,
    )
    w = SingleAgentWrapper(env, opp)
    w.action_space = env.action_space
    w.observation_space = env.observation_space
    return w


def _play_episode(w, rng):
    obs, _ = w.reset()
    for step in range(1, _MAX_STEPS + 1):
        mask = np.asarray(obs["action_mask"]).astype(bool)
        legal = np.flatnonzero(mask)
        action = int(rng.choice(legal)) if legal.size else 0
        obs, _r, term, trunc, _i = w.step(action)
        if term or trunc:
            return step
    return _MAX_STEPS


def _parse_args(argv):
    budget = next((a for a in argv if not a.startswith("-")), "3000")
    mode, val = ("time", float(budget[:-1]) * 60) if budget.endswith("m") else ("count", int(budget))
    def opt(name, default, cast=float):
        return cast(argv[argv.index(name) + 1]) if name in argv else default
    return mode, val, int(opt("--sample-every", 50, int)), opt("--prod-fps", 1200.0), int(opt("--prod-envs", 64, int))


def main():
    mode, budget, sample_every, prod_fps, prod_envs = _parse_args(sys.argv[1:])
    teams = _teams()
    rng = np.random.default_rng(0)
    w = _build(teams)
    session = w.env._bridge_session

    print(f"[heap] persistent child, recycle OFF — "
          f"{'%.0f min' % (budget/60) if mode=='time' else '%d battles' % budget}, "
          f"sample/{sample_every}", flush=True)

    samples = []  # (battles, rss_mb)
    total_steps = 0
    start = time.perf_counter()
    ep = 0
    # Prime the child (first episode spawns it).
    total_steps += _play_episode(w, rng)
    ep += 1
    base_pid = session._proc.pid
    base_rss = _rss_mb(base_pid)
    samples.append((ep, base_rss))
    print(f"[heap] child pid={base_pid} | start RSS={base_rss:.1f} MB", flush=True)

    while True:
        if mode == "time" and time.perf_counter() - start >= budget:
            break
        if mode == "count" and ep >= budget:
            break
        total_steps += _play_episode(w, rng)
        ep += 1
        if session._proc.pid != base_pid:
            print(f"❌ child respawned at ep {ep} (recycle was off!) — aborting", flush=True)
            break
        if ep % sample_every == 0:
            rss = _rss_mb(base_pid)
            samples.append((ep, rss))
            elapsed = time.perf_counter() - start
            print(f"[heap] ep {ep:5d} | RSS {rss:7.1f} MB (Δ{rss-base_rss:+6.1f}) | "
                  f"{ep/elapsed:.1f} ep/s | mean_steps {total_steps/ep:.0f}", flush=True)

    w.close()

    # --- analysis ---
    elapsed = time.perf_counter() - start
    eps = np.array([s[0] for s in samples], dtype=float)
    rss = np.array([s[1] for s in samples], dtype=float)
    mean_steps = total_steps / ep
    # The first ~tens of battles are V8 warmup (heap jumps then plateaus); the STEADY-STATE slope
    # is what matters for a 3h window. Fit the back HALF of the samples (clearly post-warmup) and
    # use that for the recommendation; also report the warmup-contaminated full fit for contrast.
    def _slope(e, r):
        return np.polyfit(e, r, 1)[0] if len(e) >= 2 else 0.0
    full_slope = _slope(eps, rss)
    h = len(eps) // 2
    slope = _slope(eps[h:], rss[h:]) if len(eps) - h >= 2 else full_slope  # steady-state
    mb_per_1k = slope * 1000.0
    warmup_mb = rss[min(h, len(rss) - 1)] - rss[0]  # heap gained during warmup (one-time)

    # Production per-env battle rate: total fps / envs = steps/s/env; ÷ mean steps = battles/s/env.
    steps_s_env = prod_fps / prod_envs
    battles_s_env = steps_s_env / mean_steps if mean_steps else 0
    battles_3h = battles_s_env * 3 * 3600
    growth_3h = slope * battles_3h

    print("\n=== heap growth summary ===")
    print(f"  battles played    : {ep}  ({elapsed:.0f}s, {ep/elapsed:.1f} ep/s single-env)")
    print(f"  mean episode steps: {mean_steps:.0f}")
    print(f"  RSS start / end   : {rss[0]:.1f} / {rss[-1]:.1f} MB  (Δ {rss[-1]-rss[0]:+.1f} MB)")
    print(f"  one-time warmup   : ~{warmup_mb:+.0f} MB (first half, V8 heap settling — bounded)")
    print(f"  STEADY-STATE rate : {mb_per_1k:+.2f} MB / 1000 battles  (2nd-half fit)")
    print(f"  (full-run fit, warmup-contaminated, for contrast: {full_slope*1000:+.1f} MB/1k)")
    print(f"\n  --- extrapolation to the 3h launcher window (prod {prod_envs} envs @ {prod_fps:.0f} fps) ---")
    print(f"  per-env battle rate : {battles_s_env*3600:.0f} battles/hour → ~{battles_3h:.0f} battles in 3h")
    print(f"  projected RSS growth over 3h: {growth_3h:+.0f} MB per child"
          f"  (× {prod_envs} envs = {growth_3h*prod_envs/1024:+.2f} GB box-wide)")
    # Recommendation
    if abs(mb_per_1k) < 5:
        print(f"\n  ➜ RECOMMENDATION: heap is ~flat ({mb_per_1k:+.1f} MB/1k). The 3h launcher restart "
              f"already bounds it; recycle_every can be HIGH or 0 (off).")
    else:
        for budget_mb in (200, 500):
            rec = int(budget_mb / slope) if slope > 0 else 0
            print(f"  ➜ to cap a child at +{budget_mb} MB: recycle_every ≈ {rec} battles")


if __name__ == "__main__":
    # A benchmark on a busy box reports a confidently wrong number — say so up front.
    from utils.contention import warn_if_contended
    warn_if_contended("bridge heap growth")
    main()
