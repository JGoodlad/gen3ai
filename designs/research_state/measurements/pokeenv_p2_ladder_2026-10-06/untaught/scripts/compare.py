"""P2 paired read: the untaught level of ONE ref vs the default opponent on the untaught 8, OLD stack (poke-env
RLPlayers over the bridge, the Python encoder — HEAD's play_cells, snapshotted to /tmp/old_um) vs NEW stack (the
Rust eval core, untaught_rust). Usage:
  compare.py old <team_index> <games> <out.json>     # one team, one process (the old path is serial)
  compare.py new all <games> <out.json>              # all 8 teams on one engine
Counts only; timings are contaminated (loaded box)."""
import json
import os
import sys
import time

sys.path.insert(0, "/tmp/old_um")
from agents.training import untaught_meter as um  # noqa: E402
from agents.training import baselines  # noqa: E402
from utils.paths import main_models_dir  # noqa: E402

REF = "rb_x5ab_blob_s1001"
mode, which, games, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
models = main_models_dir()
ref = um.resolve_ref(str(models / REF), label="REF")
opp = um.resolve_ref(str(models / baselines.spec(um.DEFAULT_OPPONENT_BASELINE)), label="OPPONENT", role="opponent")
teams = um.load_team_manifest(str(um.DEFAULT_TEAMS_MANIFEST))
t0 = time.time()
if mode == "old":
    import untaught_meter_old as old

    # the worktree holds only the EMISSION SELF-CHECK build of the bridge (same sim; it also self-checks)
    os.environ["POKESIM_SIM_BRIDGE_BIN"] = ("/home/goodlad/dev/gen3ai/.claude/worktrees/agent-a95ca950ad27679ff/"
                                            "src/rust_sim/target/selfcheck/sim_bridge")
    sel = [t for t in teams if t.index == int(which)]
    cells = old.play_cells([ref], sel, opp, games_per_team=games, seed=0, stochastic=True)
else:
    from main.h2h.play import Compute

    info = {}
    cells = um.play_cells([ref], teams, opp, games_per_team=games, seed=0, stochastic=True, info=info,
                          compute=Compute(device="cpu", backend="eager", n_envs=32, threads=2, torch_threads=4,
                                          front="proc", profile="release"))
doc = {"mode": mode, "ref": ref.to_json(), "opponent": opp.to_json(), "games_per_team": games, "seed": 0,
       "wall_s": round(time.time() - t0, 1),
       "cells": {k: c.to_json() for k, c in cells["REF"].items()}}
if mode == "new":
    doc["engine"] = {k: info.get(k) for k in ("transport", "encoder", "core_stamp", "eval_regime", "compute")}
with open(out, "w") as fh:
    json.dump(doc, fh, indent=1)
print(mode, which, {k: (c.wins, c.ties, c.finished, c.attempted) for k, c in cells["REF"].items()},
      f"{time.time() - t0:.0f}s")
