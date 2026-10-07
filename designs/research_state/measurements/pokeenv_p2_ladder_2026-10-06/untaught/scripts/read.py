"""Read the P2 untaught paired comparison: OLD (python_bridge) vs NEW (rust_eval) levels of one ref, through the
meter's own aggregation (cluster bootstrap over the 8 teams, paired index set), read across the transport boundary
ON PURPOSE (``allow_transport_mix=True``). Writes ``summary.json`` beside the inputs and prints the table."""
import json
import os
import sys

from agents.training import untaught_meter as um

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(games: int):
    new_name = "new_rust_eval.json" if games == 20 else f"new_rust_eval_g{games}.json"
    with open(os.path.join(HERE, new_name)) as fh:
        new = json.load(fh)
    old_cells = {}
    for t in range(8):
        name = f"old_python_bridge_t{t}.json" if games == 20 else f"old_python_bridge_g{games}_t{t}.json"
        with open(os.path.join(HERE, name)) as fh:
            old_cells.update(json.load(fh)["cells"])
    cells = {"NEW_rust_eval": {k: um.cell_from_json(v) for k, v in new["cells"].items()},
             "OLD_python_bridge": {k: um.cell_from_json(v) for k, v in old_cells.items()}}
    return cells, new


out = {}
for games in (20, 60):
    cells, new = load(games)
    teams = list(new["cells"])
    res = um.aggregate(cells, teams, ref_labels=["NEW_rust_eval"], baseline_label="OLD_python_bridge",
                       allow_transport_mix=True)
    lv = res["levels"]
    c = res["contrasts"][0]["vs_baseline"]
    out[f"g{games}"] = {
        "games_per_team": games, "teams": teams, "transport": res["transport"],
        "new_level_pp": lv["NEW_rust_eval"]["cluster_mean_pp"], "new_ci95_pp": lv["NEW_rust_eval"]["cluster_ci95_pp"],
        "old_level_pp": lv["OLD_python_bridge"]["cluster_mean_pp"],
        "old_ci95_pp": lv["OLD_python_bridge"]["cluster_ci95_pp"],
        "delta_new_minus_old_pp": c["delta_pp"], "delta_ci95_pp": c["ci95_pp"], "verdict_no_floor": c["verdict"],
        "new_wins_ties_finished": [lv["NEW_rust_eval"]["wins"], lv["NEW_rust_eval"]["ties"],
                                   lv["NEW_rust_eval"]["finished"]],
        "old_wins_ties_finished": [lv["OLD_python_bridge"]["wins"], lv["OLD_python_bridge"]["ties"],
                                   lv["OLD_python_bridge"]["finished"]],
        "old_timeouts": lv["OLD_python_bridge"]["timeouts"],
        "new_turn_limit_draws": sum(int(x.turn_limit_draws or 0) for x in cells["NEW_rust_eval"].values()),
        "per_team_new": lv["NEW_rust_eval"]["per_team_win_rate"],
        "per_team_old": lv["OLD_python_bridge"]["per_team_win_rate"],
    }
    o = out[f"g{games}"]
    print(f"{games}/team: NEW {o['new_level_pp']:.2f}pp {o['new_ci95_pp']}  OLD {o['old_level_pp']:.2f}pp "
          f"{o['old_ci95_pp']}  delta {o['delta_new_minus_old_pp']:+.2f} {o['delta_ci95_pp']}  "
          f"old timeouts {o['old_timeouts']}  new turn-limit draws {o['new_turn_limit_draws']}")
with open(os.path.join(HERE, "summary.json"), "w") as fh:
    json.dump(out, fh, indent=1)
sys.exit(0)
