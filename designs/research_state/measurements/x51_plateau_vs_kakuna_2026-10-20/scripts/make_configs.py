"""X51 — write the per-cell anchors configs (2026-10-20).

Each config is X37a's stage-2 symmetric config (`x37a_eval_matrix_2026-10-18/configs/anchors_x37a_forum72.json`)
with the team set swapped: BOTH sides pilot one set — our side from X51's sha-checked byte copy, Kakuna from the
Metamon set of the same name. The pair cell's config adds ONE agent, `Kakuna28` (Metamon's `Kakuna` policy at
checkpoint 28), which the build commit's `main.anchors` resolves through the agent's `policy` key.

    python make_configs.py <out dir>      # writes anchors_x51_<cell>.json and prints each sha256
"""
import copy
import hashlib
import json
import sys
from pathlib import Path

MEAS = Path("/home/goodlad/dev/gen3ai/designs/research_state/measurements")
BASE = MEAS / "x37a_eval_matrix_2026-10-18" / "configs" / "anchors_x37a_forum72.json"
STATE = Path("/home/goodlad/dev/gen3ai-reads/x51_2026-10-20")
SETS = {"home72": "gen3ai_home72", "heldout72_hpfix": "gen3ai_heldout72_hpfix",
        "forum72": "gen3ai_forum72", "p72": "gen3ai_p72"}


def build(team_set, pair=False):
    cfg = copy.deepcopy(json.loads(BASE.read_text()))
    mm = cfg["opponents"]["metamon"]
    mm["team_sets"]["home"] = SETS[team_set]
    ours = str(STATE / f"teams_{team_set}")
    cfg["our_team_sources"]["home"] = {"kind": "dir", "path": ours}
    cfg["opponents"]["foulplay"]["team_dirs"]["home"] = ours
    if pair:
        mm["agents"]["Kakuna28"] = {"policy": "Kakuna", "checkpoint": 28,
                                    "params": mm["agents"]["Kakuna"]["params"]}
        cfg["_README"] = [f"X51 pair cell: Kakuna ckpt 28 (side A, reported) vs Kakuna ckpt 34, both piloting "
                          f"{team_set} ({SETS[team_set]}). Needs the agent `policy` key (main.anchors)."]
    else:
        cfg["_README"] = [f"X51 symmetric cell {team_set}: both sides pilot {team_set} (our byte copy; Metamon set "
                          f"{SETS[team_set]}), sha256-checked against its committed manifest before every unit."]
    return cfg


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    specs = {f"sym_{s}": build(s) for s in SETS}
    specs["pair_home72"] = build("home72", pair=True)
    for name, cfg in specs.items():
        p = out / f"anchors_x51_{name}.json"
        p.write_text(json.dumps(cfg, indent=1) + "\n")
        print(name, hashlib.sha256(p.read_bytes()).hexdigest())


if __name__ == "__main__":
    main(Path(sys.argv[1]))
