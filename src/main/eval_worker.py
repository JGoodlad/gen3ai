"""Subprocess eval worker.

Loads a frozen model snapshot, runs the per-opponent eval gather for the opponents
it work-steals from the shared pool against the (shared) Showdown server, and
writes one result JSON per opponent that the trainer's ``PerOpponentEvalCallback``
picks up. Spawned by that callback as::

    python -m main.eval_worker <config.json>

Running eval in a fresh process means all its memory is returned to the OS on
exit (no fragmentation in the trainer), and the frozen snapshot lets eval run in
parallel with training — the worker reads a static copy, not the mutating model.

Config JSON keys: snapshot, port, model_dir, step, n_games, opponent_pool,
claim_dir, result_dir, concurrency, device, worker_id, cycle_tag.

Work stealing: all workers share `opponent_pool` + `claim_dir`; each repeatedly
claims the next unclaimed opponent (atomic O_EXCL lock) and writes its result to
`result_dir/result__<opponent>.json`, until the pool is exhausted. A worker that
finishes a cheap opponent immediately grabs the next, so uneven per-opponent cost
self-balances across the (default 3) workers.
"""
import os

# CPU eval shares the box with training — keep BLAS/OMP from spawning a thread per
# core in this process (mirrors the trainer's SubprocVecEnv workers).
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import sys
import json
import asyncio
import traceback

from sb3_contrib import MaskablePPO
from poke_env.ps_client import LocalhostServerConfiguration
from poke_env.ps_client.server_configuration import localhost_server_configuration

from agents.observation.state_encoder import load_mappings
from agents.training.eval_callback import (
    build_eval_opponents, build_eval_players, run_eval, claim_next_opponent,
)
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder


def _run(cfg: dict) -> None:
    # Rebuild the stateless eval infrastructure deterministically from the data dir,
    # exactly as train_rl_agent does — nothing is passed in-memory across the process
    # boundary except the config (and the snapshot zip on disk).
    mappings = load_mappings()
    loader = TeamLoader()
    all_teams = loader.get_all_teams()
    sample_teams = loader.get_sample_teams()
    trainee_tb = Gen3Teambuilder(all_teams, bias_teams=sample_teams, bias_prob=0.1)
    opp_tb = Gen3Teambuilder(all_teams)

    port = cfg.get("port")
    server_config = localhost_server_configuration(port) if port else LocalhostServerConfiguration

    # Frozen weights — inference only, so the base algorithm + env=None is enough
    # (the policy class and its extractor kwargs are restored from the zip).
    model = MaskablePPO.load(cfg["snapshot"], env=None, device=cfg.get("device", "cpu"))

    pool = cfg["opponent_pool"]
    claim_dir = cfg["claim_dir"]
    result_dir = cfg["result_dir"]
    wid = cfg["worker_id"]
    cycle_tag = cfg["cycle_tag"]

    claim_seq = 0
    while True:
        name = claim_next_opponent(claim_dir, pool)
        if name is None:
            break  # every opponent claimed by some worker → this one is done
        # Unique account suffix per (cycle, worker, claim) so the lingering
        # connection from a prior claim can't collide on the shared server.
        tag = f"{cycle_tag}{wid}{claim_seq}"
        claim_seq += 1
        opponents = build_eval_opponents(server_config, opp_tb, [name], tag)
        players = build_eval_players(
            model, [name], trainee_tb, mappings, server_config, cfg["concurrency"], tag,
        )
        m = asyncio.run(run_eval(
            players, opponents, cfg["n_games"], cfg.get("model_dir"), cfg["step"],
        ))
        result = {
            "win_rate": m["win_rates"][name],
            "reward_mean": m["reward_means"][name],
            "ep_len": m["ep_lens"][name],
            "duration_sec": m["durations_sec"][name],
            "worker_id": wid,
        }
        # Write atomically (tmp + rename) so the parent never reads a half-written file.
        out = os.path.join(result_dir, f"result__{name}.json")
        tmp = out + ".tmp"
        with open(tmp, "w") as f:
            json.dump(result, f)
        os.replace(tmp, out)


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m main.eval_worker <config.json>", file=sys.stderr)
        return 2
    with open(sys.argv[1]) as f:
        cfg = json.load(f)
    try:
        _run(cfg)
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
