"""M1 measurement: the cost of the Rust `parse` per decision, as a ratio to one env step's wall time.

For a pending OWNER decision (not decided here): should training's OBSERVATION path always go
text -> parse -> view (one path, continuously validated by training) and leave the typed-at-source
path to search?

Interleaved, one process, `nice 10`, the load average printed beside every block:

* **Rust parse** (`core_events --bench-parse`): one side's stream read INCREMENTALLY, decision by
  decision (a decision = the lines since the previous `|request|`, the request included), through
  `Line::parse` + the reading fold — exactly what a text-driven observation path would pay per
  decision. Pure Rust, in-process: NO IPC/FFI cost is included (M5 decides the binding). Corpus:
  the COMMIT tier's 8 recorded battles (6 seeded-random, 2 production-policy), both viewers.
* **Env step** (`Gen3Env.step` wall): the production env on the production rust bridge (persistent
  child), a RandomPlayer opponent, random LEGAL actions for the trainee — so it EXCLUDES the policy
  forward and the PPO batch (both would make a step longer, i.e. the ratio smaller). A `reset()`'s
  first step is excluded.

    export PYTHONPATH=$PYTHONPATH:src
    python designs/research_state/measurements/rust_core_m1_2026-09-23/parse_cost.py --rounds 5
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np


def load() -> float:
    return os.getloadavg()[0]


def rust_block(script: str, rounds: int) -> dict:
    from utils.bridge.sim_bridge_bin import resolve_core_events_bin

    p = subprocess.run([resolve_core_events_bin(), "--bench-parse", str(rounds)], input=script,
                       capture_output=True, text=True, check=True)
    return json.loads(p.stdout)


def env_block(wrapped, rng, steps: int) -> list:
    out = []
    obs, _ = wrapped.reset()
    while len(out) < steps:
        mask = np.asarray(obs["action_mask"]).astype(bool)
        legal = np.flatnonzero(mask)
        action = int(rng.choice(legal)) if legal.size else 0
        t0 = time.perf_counter()
        obs, _r, term, trunc, _i = wrapped.step(action)
        out.append(time.perf_counter() - t0)
        if term or trunc:
            obs, _ = wrapped.reset()
    return out


def build_env():
    from poke_env import AccountConfiguration
    from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
    from poke_env.player import RandomPlayer

    from agents.observation.state_encoder import load_mappings
    from agents.training.gen3_env import Gen3Env
    from utils.bridge.bridge_session import attach_bridge_transport
    from utils.team_loader.loader import TeamLoader
    from utils.teambuilder import Gen3Teambuilder

    teams = TeamLoader().get_all_teams()
    env = Gen3Env(load_mappings(), battle_format="gen3ou", team=Gen3Teambuilder(teams),
                  account_configuration1=AccountConfiguration("ParseCost", None),
                  start_listening=False)
    attach_bridge_transport(env, battle_format="gen3ou", persistent=True, recycle_every=10000)
    opp = RandomPlayer(battle_format="gen3ou", team=Gen3Teambuilder(teams),
                       account_configuration=AccountConfiguration("ParseCostOpp", None),
                       start_listening=False)
    wrapped = SingleAgentWrapper(env, opp)
    wrapped.action_space = env.action_space
    wrapped.observation_space = env.observation_space
    return wrapped


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=200, help="env steps per round")
    ap.add_argument("--seed", type=int, default=0, help="the env's random-action seed")
    a = ap.parse_args()
    from agents.battle.rust_core_parity import load_commit_fixture

    script = "\n".join(line for b in load_commit_fixture() for line in b.script()) + "\n"
    wrapped = build_env()
    rng = np.random.default_rng(a.seed)
    env_block(wrapped, rng, 30)                             # warm-up (imports, child spawn, caches)
    rows = []
    for r in range(a.rounds):
        l0 = load()
        steps = env_block(wrapped, rng, a.steps)
        l1 = load()
        rust = rust_block(script, 3)
        l2 = load()
        env_ms = statistics.median(steps) * 1e3
        parse_us = rust["ns_per_decision_median"] / 1e3
        rows.append({"round": r + 1, "load_env": round((l0 + l1) / 2, 1), "load_rust": round((l1 + l2) / 2, 1),
                     "env_step_ms_median": round(env_ms, 3), "env_step_ms_mean": round(statistics.mean(steps) * 1e3, 3),
                     "parse_us_per_decision": round(parse_us, 2),
                     "request_us_per_decision": round(rust["request_ns_per_decision_median"] / 1e3, 2),
                     "ratio_parse_to_env_step": round(parse_us / (env_ms * 1e3), 5)})
        print(json.dumps(rows[-1]), flush=True)
    med = statistics.median(x["ratio_parse_to_env_step"] for x in rows)
    print(json.dumps({"rounds": len(rows), "decisions_per_round_rust": rust["decisions"],
                      "lines_per_decision": round(rust["lines"] / rust["decisions"], 2),
                      "median_ratio": med,
                      "median_parse_us": statistics.median(x["parse_us_per_decision"] for x in rows),
                      "median_env_step_ms": statistics.median(x["env_step_ms_median"] for x in rows)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
