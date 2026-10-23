"""X52 read — write the two REVERSE-cell anchors configs (2026-10-22).

Each is X51's `anchors_x51_sym_home72.json` (X37a's stage-2 config, home72 on Metamon's side: `gen3ai_home72`) with
ONE change: OUR side pilots one X52 half (a sha-checked byte copy of its 36 files), Kakuna keeps home72. So our model
pilots the half and Kakuna ckpt 34 pilots home72 (AM-X52-e).

    python make_configs.py <out dir>      # writes anchors_x52_<half>.json and prints each sha256
"""
import copy
import hashlib
import json
import sys
from pathlib import Path

BASE = Path("/home/goodlad/dev/gen3ai/designs/research_state/measurements/x51_plateau_vs_kakuna_2026-10-20/configs/"
            "anchors_x51_sym_home72.json")
STATE = Path("/home/goodlad/gen3ai_archive/x52_read")


def build(half):
    cfg = copy.deepcopy(json.loads(BASE.read_text()))
    assert cfg["opponents"]["metamon"]["team_sets"]["home"] == "gen3ai_home72"
    ours = str(STATE / f"teams_{half}")
    cfg["our_team_sources"]["home"] = {"kind": "dir", "path": ours}
    cfg["opponents"]["foulplay"]["team_dirs"]["home"] = ours
    cfg["_README"] = [f"X52 REVERSE cell, {half} half: our side pilots the X52 {half} half (36 files, byte copy, "
                      "sha256-checked against x52_teams_manifest.json before every unit); Kakuna pilots home72 "
                      "(Metamon set gen3ai_home72)."]
    return cfg


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    for half in ("practised", "unpractised"):
        p = out / f"anchors_x52_{half}.json"
        p.write_text(json.dumps(build(half), indent=1) + "\n")
        print(half, hashlib.sha256(p.read_bytes()).hexdigest())


if __name__ == "__main__":
    main(Path(sys.argv[1]))
