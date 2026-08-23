"""Capture the NODE search_driver.js wire output on a real bridge battle record.

Scratch tool (tmp/, not shipped): produces the GOLDEN a Rust `search_driver` binary
must reproduce byte-for-byte. It talks to `search_driver.js` over raw stdin/stdout
JSON — deliberately NOT through `SearchSession`, so it stays valid while the Python
seam is being reworked.

    python src/rust_sim/harness/search_golden.py [n_battles] [--out tmp/search_golden_node.json]

Writes {"cases": [{record, turn, open_root, expand_many}, ...]}.
"""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.training.obs_roundtrip_fuzz_test import RecordingFuzzPlayer
from utils.bridge.local_battle_runner import run_local_battles
from utils.bridge.reconstruction import ReconstructionRecord
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

# parents[3] is the REPO ROOT: this file is <root>/src/rust_sim/harness/. It was
# parents[1] while the harness lived in <root>/tmp/, and the move (`ede4c79`) left the
# index behind — resolving to a path that does not exist.
DRIVER = Path(__file__).resolve().parents[3] / "src/utils/bridge/search_driver.js"
BATTLE_FORMAT = "gen3ou"


def record_battle(out_dir: str, tag: int):
    pool = TeamLoader().get_all_teams()
    trainee = RecordingFuzzPlayer(
        out_dir=out_dir, rng_seed=tag, battle_format=BATTLE_FORMAT,
        team=Gen3Teambuilder(pool),
        account_configuration=AccountConfiguration(f"SGt{tag}", "pw"),
        server_configuration=LocalhostServerConfiguration,
        start_listening=False, max_concurrent_battles=1)
    opp = RandomPlayer(
        battle_format=BATTLE_FORMAT, team=Gen3Teambuilder(pool),
        account_configuration=AccountConfiguration(f"SGo{tag}", "pw"),
        server_configuration=LocalhostServerConfiguration,
        start_listening=False, max_concurrent_battles=1)
    asyncio.run(run_local_battles(trainee, opp, 1))
    prefix = trainee.trace_prefixes[0]
    rec = ReconstructionRecord.load(f"{prefix}_reconstruction.json")
    return rec


class NodeDriver:
    def __init__(self):
        self.p = subprocess.Popen(
            ["node", str(DRIVER)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1)
        self.seq = 0

    def call(self, **payload):
        self.seq += 1
        payload["id"] = self.seq
        self.p.stdin.write(json.dumps(payload) + "\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("driver died: " + self.p.stderr.read())
        return json.loads(line)

    def close(self):
        try:
            self.call(cmd="close")
        except Exception:
            pass
        self.p.kill()


# The arm shapes the search actually uses: the CRN anchor (realized dice, recorded
# choices), an explicit candidate under original dice, an explicit candidate under a
# fresh seed, and a fully random arm (exercises auxRngFromSeed + randomChoice).
def arms_for(root):
    node_id = root["node_id"]
    rec_p2 = root["recorded_choices"].get("p2")
    out = [dict(node_id=node_id, label=0, recorded_exact=True)]
    for k, mv in enumerate(["move 1", "move 2", "move 3", "switch 2"]):
        out.append(dict(node_id=node_id, label=1 + k, p1_action=mv,
                        p2_action=rec_p2 or "random", seed="original"))
    for k, seed in enumerate(["sodium,0011223344556677", "gen5,1234567890abcdef"]):
        out.append(dict(node_id=node_id, label=10 + k, p1_action="move 1",
                        p2_action="random", seed=seed))
    out.append(dict(node_id=node_id, label=20, p1_action="random",
                    p2_action="random", seed="sodium,deadbeefcafe", followup="random"))
    out.append(dict(node_id=node_id, label=21, p1_action="move 1",
                    p2_action="random", seed="sodium,00ff00ff", followup="default"))
    return out


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else 3
    out_path = "tmp/search_golden_node.json"
    if "--out" in sys.argv:
        out_path = sys.argv[sys.argv.index("--out") + 1]

    cases = []
    with tempfile.TemporaryDirectory() as td:
        for b in range(n):
            tag = (int(time.time() * 1000) + b * 7919) % 100000
            rec = record_battle(td, tag)
            rd = rec.to_dict() if hasattr(rec, "to_dict") else None
            if rd is None:  # fall back to the on-disk shape
                rd = {"v": rec.v, "format_id": rec.format_id,
                      "prng_seed": rec.prng_seed,
                      "input_log": list(rec.input_log),
                      "commands": [list(c) for c in rec.commands]}
            # A few turns spread across the battle, so switches/faints/forced
            # switches all appear somewhere in the golden.
            #
            # TURN 1 is in the set deliberately (`gen3_search_turn1_open_v1`): the rust driver
            # could not open the first decision of any battle, and this golden — which only ever
            # sampled turns 2/5/9 — is why the cross-impl gate never saw it. A capability the
            # golden does not sample is a capability the gate cannot speak to.
            turns = sorted({1, 2, 5, 9})
            drv = NodeDriver()
            try:
                for T in turns:
                    root = drv.call(cmd="open_root", record=rd, turn=T)
                    if not root.get("ok"):
                        continue
                    arms = arms_for(root)
                    exp = drv.call(cmd="expand_many", arms=arms)
                    # depth-2: branch off the first surviving child
                    child = next((a["node_id"] for a in exp.get("arms", [])
                                  if a.get("node_id")), None)
                    exp2 = None
                    if child:
                        exp2 = drv.call(cmd="expand_many", arms=[
                            dict(node_id=child, label=100, p1_action="move 1",
                                 p2_action="random", seed="sodium,abc123"),
                        ])
                    cases.append({"record": rd, "turn": T, "open_root": root,
                                  "expand_many": exp, "expand_depth2": exp2})
                    print(f"battle {b} turn {T}: {len(exp.get('arms', []))} arms", flush=True)
            finally:
                drv.close()

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"cases": cases}, f)
    print(f"wrote {len(cases)} cases -> {out_path}")


if __name__ == "__main__":
    main()
