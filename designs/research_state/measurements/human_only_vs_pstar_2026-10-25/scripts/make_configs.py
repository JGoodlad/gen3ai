"""Write FR-3's per-panel anchors configs and the registered sha256 of every config the queue passes (2026-10-25).

FR-3's config for a panel is X51's committed symmetric config for it
(`../../x51_plateau_vs_kakuna_2026-10-20/configs/anchors_x51_sym_<panel>.json`: both sides pilot the panel, our side
from X51's sha-checked byte copy, the peer from the Metamon set of the same name) with TWO agents added, `LargeRL`
and `LargeIL`, exactly as `designs/ops/anchors.json` pins them. FR-1 (SmallRL) runs on X51's configs unchanged.

    python make_configs.py <out dir>      # writes anchors_hr_<panel>.json and CONFIG_SHA.json beside this file
"""
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN = HERE.parents[4]
X51 = MAIN / "designs" / "research_state" / "measurements" / "x51_plateau_vs_kakuna_2026-10-20" / "configs"
PANELS = ("home72", "heldout72_hpfix", "forum72", "p72")
AGENTS = ("LargeRL", "LargeIL")


def build(panel, agents_json):
    cfg = json.loads((X51 / f"anchors_x51_sym_{panel}.json").read_text())
    pinned = json.loads(agents_json.read_text())["opponents"]["metamon"]["agents"]
    for a in AGENTS:
        cfg["opponents"]["metamon"]["agents"][a] = pinned[a]
    cfg["_README"] = [f"human_only_vs_pstar FR-3 symmetric cell {panel}: X51's sym_{panel} config + the LargeRL and "
                      "LargeIL agents (ckpt 40, hub revision ac8e88e7)."]
    return cfg


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(out, agents_json):
    out.mkdir(parents=True, exist_ok=True)
    reg = {}
    for panel in PANELS:
        p = out / f"anchors_hr_{panel}.json"
        p.write_text(json.dumps(build(panel, agents_json), indent=1) + "\n")
        reg[f"hr_{panel}"] = sha(p)
        reg[f"x51_sym_{panel}"] = sha(X51 / f"anchors_x51_sym_{panel}.json")
    (HERE / "CONFIG_SHA.json").write_text(json.dumps(reg, indent=1, sort_keys=True) + "\n")
    print(json.dumps(reg, indent=1, sort_keys=True))


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]) if len(sys.argv) > 2 else MAIN / "designs" / "ops" / "anchors.json")
