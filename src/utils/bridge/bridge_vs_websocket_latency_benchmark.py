"""Transport-latency A/B: websocket Showdown server vs in-process bridge, single env.

Isolates exactly what the bridge changes — the per-decision step round-trip and the per-episode
reset/start overhead — by driving ONE `Gen3Env` with a trivial `RandomPlayer` opponent (no GPU,
no policy net) the identical way under each transport. Everything above the transport (obs build,
reward, mask, the wrapper) is byte-for-byte the same, so the wall-clock delta is the transport.

Not a pass/fail test (no `test_*`; pytest collects nothing) — a profiler, like the other
`*_benchmark.py`. Absolute ms scale with machine load (run on an idle box for a clean baseline);
the **ratio** between transports is the load-stabler signal. The websocket arm needs a live server
(default :9001 — pass --port); the bridge arm needs none.

    export PYTHONPATH=$PYTHONPATH:src
    python src/utils/bridge/bridge_vs_websocket_latency_benchmark.py [--steps 600] [--port 9001]
                                                                     [--mode both|bridge|websocket]
                                                                     [--no-persistent]
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.ps_client.server_configuration import localhost_server_configuration
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper

from agents.observation.state_encoder import load_mappings
from agents.training.gen3_env import Gen3Env
from utils.bridge.bridge_session import attach_bridge_transport
from utils.team_loader.loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder


def _teams():
    loader = TeamLoader()
    return loader.get_sample_teams() or loader.get_all_teams()


def _wrap(env, teams, idx):
    opponent = RandomPlayer(
        battle_format="gen3ou", team=Gen3Teambuilder(teams),
        account_configuration=AccountConfiguration(f"LatOpp{idx}", None),
        start_listening=False,
    )
    w = SingleAgentWrapper(env, opponent)
    w.action_space = env.action_space
    w.observation_space = env.observation_space
    return w


def _build_websocket(teams, idx, port):
    env = Gen3Env(
        load_mappings(), battle_format="gen3ou", team=Gen3Teambuilder(teams),
        account_configuration1=AccountConfiguration(f"LatWs{idx}", "password"),
        server_configuration=localhost_server_configuration(port),
    )
    return _wrap(env, teams, idx)


def _build_bridge(teams, idx, persistent):
    env = Gen3Env(
        load_mappings(), battle_format="gen3ou", team=Gen3Teambuilder(teams),
        account_configuration1=AccountConfiguration(f"LatBr{idx}", None),
        start_listening=False,
    )
    attach_bridge_transport(env, battle_format="gen3ou", persistent=persistent)
    return _wrap(env, teams, idx)


def _bench(wrapped, steps, rng):
    """Drive `steps` decisions with random legal actions; time only the step/reset calls.
    Returns (total_seconds, n_episodes). Stops on an episode boundary so the env can close
    cleanly (poke-env refuses reset_battles() while a battle is still running)."""
    obs, _ = wrapped.reset()
    t0 = time.perf_counter()
    n_eps = 0
    done_at_boundary = False
    for i in range(steps):
        mask = np.asarray(obs["action_mask"]).astype(bool)
        legal = np.flatnonzero(mask)
        action = int(rng.choice(legal)) if legal.size else 0
        obs, _r, term, trunc, _i = wrapped.step(action)
        if term or trunc:
            n_eps += 1
            if i >= steps - 1:
                done_at_boundary = True
                break
            obs, _ = wrapped.reset()
    dt = time.perf_counter() - t0
    # Drain to the next episode boundary so close() doesn't hit a live battle.
    if not done_at_boundary:
        while True:
            mask = np.asarray(obs["action_mask"]).astype(bool)
            legal = np.flatnonzero(mask)
            action = int(rng.choice(legal)) if legal.size else 0
            obs, _r, term, trunc, _i = wrapped.step(action)
            if term or trunc:
                n_eps += 1
                break
    return dt, n_eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--port", type=int, default=9001, help="Showdown port for the websocket arm")
    ap.add_argument("--mode", choices=["both", "bridge", "websocket"], default="both")
    ap.add_argument("--no-persistent", action="store_true",
                    help="Bridge arm uses spawn-per-battle instead of one reused child")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    teams = _teams()
    results = {}

    if args.mode in ("websocket", "both"):
        print(f"[websocket] server :{args.port} — building env…", flush=True)
        w = _build_websocket(teams, 1, args.port)
        try:
            dt, eps = _bench(w, args.steps, np.random.default_rng(args.seed))
            results["websocket"] = (dt, eps)
            print(f"[websocket] {args.steps} steps in {dt:.2f}s → "
                  f"{1000*dt/args.steps:.2f} ms/step, {eps} episodes", flush=True)
        finally:
            w.close()

    if args.mode in ("bridge", "both"):
        persistent = not args.no_persistent
        label = f"bridge/{'persistent' if persistent else 'spawn-per-battle'}"
        print(f"[{label}] building env (no server)…", flush=True)
        w = _build_bridge(teams, 1, persistent)
        try:
            dt, eps = _bench(w, args.steps, np.random.default_rng(args.seed))
            results["bridge"] = (dt, eps)
            print(f"[{label}] {args.steps} steps in {dt:.2f}s → "
                  f"{1000*dt/args.steps:.2f} ms/step, {eps} episodes", flush=True)
        finally:
            w.close()

    if "websocket" in results and "bridge" in results:
        ws_ms = 1000 * results["websocket"][0] / args.steps
        br_ms = 1000 * results["bridge"][0] / args.steps
        print(f"\n=== transport latency (single env, RandomPlayer opponent) ===")
        print(f"  websocket : {ws_ms:6.2f} ms/step")
        print(f"  bridge    : {br_ms:6.2f} ms/step")
        if br_ms > 0:
            print(f"  speedup   : {ws_ms / br_ms:.2f}x  ({(1 - br_ms/ws_ms)*100:+.0f}% per step)")
        print("  (absolute ms scale with machine load; the ratio is the load-stabler signal)")


if __name__ == "__main__":
    main()
