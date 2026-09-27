#!/usr/bin/env python3
"""ONE untaught-8 unit: (ref, team, 25-battle chunk), one fsynced JSON row per battle.

The per-battle body is the round-2 G-U driver's (``../../population_loop_r2_2026-09-24/read/scripts/
gu_driver.py`` ``worker``), which is ``agents.training.untaught_meter.play_cells``' body VERBATIM but
for the battle range. A cell is a PURE FUNCTION of (ref, team index, battle index): every battle
re-seeds the sim, both players' sampling generators and the opponent-team draw (prefix-consistent
``pool_sequence``), so the unit split and the worker count change no number.

Registered cell (learner_battery_2026-09-26.md §4.2): ``--opponent untaught_meter_opponent_v14
--config auto --seed 0``, concurrency 1, 600 games per team. The opponent is passed as its FILE
(the registry name is absent at registration — a FINDING in the README); ``--config auto`` resolves
each model's own ``model_config.json``.

    gu_unit.py --label C --ref <zip> --opponent <zip> --team-index 3 --chunk 7 --rows <dir>

RESUME: battle indices already on disk are skipped; a torn last line is truncated.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict

GAMES_PER_TEAM = 600
CHUNK = 25
SEED = 0


def read_rows(path: Path) -> Dict[int, dict]:
    if not path.exists():
        return {}
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        cut = raw.rfind(b"\n") + 1
        with open(path, "r+b") as fh:
            fh.truncate(cut)
        raw = raw[:cut]
    return {int(d["j"]): d for d in (json.loads(x) for x in raw.decode().splitlines() if x.strip())}


def unit_path(rows: Path, label: str, team_index: int, chunk: int) -> Path:
    return rows / label / f"t{team_index}" / f"c{chunk:02d}.jsonl"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--opponent", required=True)
    ap.add_argument("--team-index", type=int, required=True)
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--rows", required=True)
    a = ap.parse_args()

    import torch as th
    from poke_env.ps_client import AccountConfiguration
    from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

    from agents.inference.player import RLPlayer
    from agents.model.snapshot import current_model_version, load_foreign_opponent
    from agents.observation.state_encoder import load_mappings
    from agents.training import untaught_meter as engine
    from main import untaught_meter as cli
    from utils.bridge.local_battle_runner import run_local_battles
    from utils.team_loader import TeamLoader

    args = cli.build_parser().parse_args(
        [f"{a.label}={a.ref}", "--opponent", a.opponent, "--config", "auto", "--seed", str(SEED)])
    cli.apply_baseline_defaults(args)
    refs_r, baseline, controls, opponent, teams = cli.resolve_all(args)
    assert baseline is None and not controls and len(refs_r) == 1
    ref = refs_r[0]
    team = {t.index: t for t in teams}[a.team_index]

    path = unit_path(Path(a.rows), a.label, a.team_index, a.chunk)
    path.parent.mkdir(parents=True, exist_ok=True)
    done = read_rows(path)
    todo = [j for j in range(a.chunk * CHUNK, (a.chunk + 1) * CHUNK) if j not in done]
    if not todo:
        print(f"unit {a.label} t{a.team_index} c{a.chunk:02d}: complete on disk")
        return 0

    engine.check_concurrency(1)
    th.set_num_threads(1)
    PinnedTeam, PairedPool = engine._teambuilders()
    maps = load_mappings()
    cv = current_model_version(maps)
    opp_model = engine._strip_debugger(load_foreign_opponent(
        opponent.zip_path, current_version=cv, device="cpu", config_path=opponent.config_path)[0])
    model = engine._strip_debugger(load_foreign_opponent(
        ref.zip_path, current_version=cv, device="cpu", config_path=ref.config_path)[0])
    pool = PairedPool(TeamLoader().get_all_teams())
    seq = engine.pool_sequence(SEED, team.index, GAMES_PER_TEAM, len(pool.packed_teams))
    commit = os.environ.get("N0Q_TREE_COMMIT", "?")
    ti = team.index
    pilot = RLPlayer(model=model, team=PinnedTeam(team.path), battle_format="gen3ou",
                     server_configuration=LocalhostServerConfiguration, mappings=maps,
                     account_configuration=AccountConfiguration(f"UM{ti}a", "pw"),
                     stochastic=True, start_listening=False)
    opp = RLPlayer(model=opp_model, team=pool, battle_format="gen3ou",
                   server_configuration=LocalhostServerConfiguration, mappings=maps,
                   account_configuration=AccountConfiguration(f"UM{ti}b", "pw"),
                   stochastic=True, start_listening=False)
    pool.set_sequence(seq)
    with open(path, "a") as fh:
        for j in todo:
            t0 = time.time()
            ps, os_ = engine.policy_seeds(SEED, ti, j)
            engine._reseed_player(pilot, ps)
            engine._reseed_player(opp, os_)
            pool.at(j)
            pilot.reset_battles()
            opp.reset_battles()
            asyncio.run(run_local_battles(pilot, opp, 1, concurrency=1, impl="rust",
                                          seed=engine.sim_seed(SEED, ti, j)))
            finished = int(pilot.n_finished_battles == 1)
            row = {"label": a.label, "ref": ref.zip_path, "opponent": opponent.zip_path,
                   "team_key": team.key, "team_index": ti, "j": j, "chunk": a.chunk,
                   "finished": finished,
                   "won": int(pilot.n_won_battles) if finished else 0,
                   "tied": int(pilot.n_tied_battles) if finished else 0,
                   "opp_team": seq[j], "wall_s": round(time.time() - t0, 2), "tree": commit}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    print(f"unit {a.label} t{ti} c{a.chunk:02d}: {len(todo)} battles written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
