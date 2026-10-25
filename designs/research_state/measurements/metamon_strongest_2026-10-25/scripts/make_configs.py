"""Write the read's per-panel anchors configs (2026-10-25).

Each config is X51's committed symmetric config for that panel
(`../x51_plateau_vs_kakuna_2026-10-20/configs/anchors_x51_sym_<panel>.json`, both sides pilot the panel: our side
from X51's sha-checked byte copy, the peer from the Metamon set of the same name) with ONE agent added,
`PastaMittens`, exactly as `designs/ops/anchors.json` pins it. The identity unit (PREREG precondition 4) runs on
X51's own `sym_home72` config, unchanged.

    python make_configs.py <out dir>      # writes anchors_mm_<panel>.json and prints each sha256
"""
import hashlib
import json
import sys
from pathlib import Path

MAIN = Path("/home/goodlad/dev/gen3ai")
X51 = MAIN / "designs" / "research_state" / "measurements" / "x51_plateau_vs_kakuna_2026-10-20" / "configs"
PANELS = ("home72", "heldout72_hpfix", "forum72", "p72")


def build(panel, agents_json):
    cfg = json.loads((X51 / f"anchors_x51_sym_{panel}.json").read_text())
    pm = json.loads(agents_json.read_text())["opponents"]["metamon"]["agents"]["PastaMittens"]
    cfg["opponents"]["metamon"]["agents"]["PastaMittens"] = pm
    cfg["_README"] = [f"metamon_strongest symmetric cell {panel}: X51's sym_{panel} config + the PastaMittens agent "
                      "(the Kakuna router, preset pinned by sha256)."]
    return cfg


def main(out, agents_json):
    out.mkdir(parents=True, exist_ok=True)
    for panel in PANELS:
        p = out / f"anchors_mm_{panel}.json"
        p.write_text(json.dumps(build(panel, agents_json), indent=1) + "\n")
        print(panel, hashlib.sha256(p.read_bytes()).hexdigest())


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else MAIN / "designs" / "ops" / "anchors.json")
