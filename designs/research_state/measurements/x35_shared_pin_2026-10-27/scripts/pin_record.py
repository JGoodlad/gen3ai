"""The X35 shared play pin's record (git objects + the built stamps): usage pin_record.py <pin> <tag> (cwd = the pin's
worktree, PYTHONPATH = its src)."""
import json
import subprocess
import sys
from pathlib import Path

from main.anchors.server import engine_stamp
from main.reads import playpath as PP
from main.reads.engine import E1, engine_at, lock_blob

pin, tag = PP.resolve(sys.argv[1]), sys.argv[2]
GAME_FROM = PP.resolve("f4b25238d")
RESTORED = ["src/rust_sim", "src/rust_env", "src/agents/model", "src/agents/observation", "src/agents/inference",
            "src/utils/team_loader", "src/utils/team_packing.py", "src/utils/teambuilder.py", "src/utils/gen3_utils.py"]
git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout.strip()
base = git("rev-parse", f"{pin}^")
pf = PP.play_files(pin)
pre = lambda s: s.split(" ", 1)[1]
vs = {}
for o in ("6c428453a", "5e389ddac", "f4b25238d", "c4a5c37b6", base):
    d = PP.differences(PP.play_files(PP.resolve(o)), pf)
    vs[o[:9]] = {"differ": len(d), "game_files_differing": sum(pre(x).startswith(tuple(RESTORED)) for x in d),
                 "bridge": [x for x in d if pre(x).startswith("src/utils/bridge/")]}
rec = {
    "schema": "x35_shared_play_pin_v1", "pin": pin, "tag": tag, "base": base, "base_is": "main at the build",
    "game_from": GAME_FROM,
    "restored": {p: {"tree_or_blob": git("rev-parse", f"{pin}:{p}"),
                     "equals_game_from": git("rev-parse", f"{pin}:{p}") == git("rev-parse", f"{GAME_FROM}:{p}")}
                 for p in RESTORED},
    "kept_from_base": {"src/utils/bridge": git("rev-parse", f"{pin}:src/utils/bridge"),
                       "equals_base": git("rev-parse", f"{pin}:src/utils/bridge") == git("rev-parse", f"{base}:src/utils/bridge")},
    "engine": {"engine_at_pin": engine_at(pin), "is_E1": engine_at(pin) == E1, "built_engine_stamp": engine_stamp(),
               "cargo_lock_blobs": {c: lock_blob(Path(f"src/{c}/Cargo.lock")) for c in ("rust_sim", "rust_env")}},
    "play_path": {"files": len(pf), "digest": PP.digest(pf)[:16], "vs": vs},
}
print(json.dumps(rec, indent=1))
