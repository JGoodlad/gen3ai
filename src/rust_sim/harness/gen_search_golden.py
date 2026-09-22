"""Capture the NODE search-driver golden that `search_impl_parity.py` replays.

It records REAL gen3ou battles, opens a search root at several turns of each, and saves node
`search_driver.js`'s verbatim responses to `open_root` / `expand_many` / a depth-2
`expand_many`, keyed exactly as `search_impl_parity.arm_spec` rebuilds them — the two files are
one tool and the ARM SET here is the contract between them.

🚨 **This used to live in the gitignored `tmp/`, and it was GONE when the gate was next needed.**
The harness beside it documented `tmp/search_golden.py` as its producer for months while nothing
on disk answered to that name, so running the strongest cross-impl gate started with rewriting
its generator. A gate whose input cannot be regenerated is a gate nobody runs.

    export PYTHONPATH=$PYTHONPATH:src
    python src/rust_sim/harness/gen_search_golden.py [--battles N] [--out PATH]
    python src/rust_sim/harness/search_impl_parity.py --golden tmp/search_golden_node.json

🚨 **Two FRESH goldens before calling the gate green** — the contract's own rule
(`designs/rust_sim/one_sided_view.md` §4, `src/rust_sim/CLAUDE.md`): the battles are random, and
across seven freshly generated goldens the per-golden divergence count ran 1/0/0/6/8/0/6.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import tempfile
import time

import numpy as np

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.training.obs_roundtrip_fuzz_test import RecordingFuzzPlayer
from utils.bridge.local_battle_runner import run_local_battles
from utils.bridge.reconstruction import ReconstructionRecord
from utils.bridge.sim_bridge_bin import search_driver_spawn_argv
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder


class NodeDriver:
    def __init__(self):
        self.p = subprocess.Popen(search_driver_spawn_argv("node"), stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                  bufsize=1)

    def call(self, req):
        self.p.stdin.write(json.dumps(req) + "\n")
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError("node driver died: " + self.p.stderr.read()[:4000])
        return json.loads(line)

    def close(self):
        try:
            self.call({"id": -1, "cmd": "close"})
        except Exception:                                            # noqa: BLE001
            pass
        try:
            self.p.wait(timeout=5)
        except Exception:                                            # noqa: BLE001
            self.p.kill()


def arms_for(root):
    """The arm set `search_impl_parity.arm_spec` reconstructs, by label."""
    nid, rec_p2 = root["node_id"], (root.get("recorded_choices") or {}).get("p2")
    out = [{"node_id": nid, "label": 0, "recorded_exact": True}]
    for i, mv in enumerate(["move 1", "move 2", "move 3", "switch 2"], start=1):
        out.append({"node_id": nid, "label": i, "p1_action": mv,
                    "p2_action": rec_p2 or "random", "seed": "original"})
    for label, seed in ((10, "sodium,0011223344556677"), (11, "gen5,1234567890abcdef")):
        out.append({"node_id": nid, "label": label, "p1_action": "move 1",
                    "p2_action": "random", "seed": seed})
    out.append({"node_id": nid, "label": 20, "p1_action": "random", "p2_action": "random",
                "seed": "sodium,deadbeefcafe", "followup": "random"})
    out.append({"node_id": nid, "label": 21, "p1_action": "move 1", "p2_action": "random",
                "seed": "sodium,00ff00ff", "followup": "default"})
    return out


def record_one(out_dir, impl="rust"):
    ts = int(time.time() * 1000) % 100000
    pool = TeamLoader().get_all_teams()
    trainee = RecordingFuzzPlayer(
        out_dir=out_dir, rng_seed=ts, battle_format="gen3ou", team=Gen3Teambuilder(pool),
        account_configuration=AccountConfiguration(f"SGt{ts}", "pw"),
        server_configuration=LocalhostServerConfiguration,
        start_listening=False, max_concurrent_battles=1)
    opp = RandomPlayer(
        battle_format="gen3ou", team=Gen3Teambuilder(pool),
        account_configuration=AccountConfiguration(f"SGo{ts}", "pw"),
        server_configuration=LocalhostServerConfiguration,
        start_listening=False, max_concurrent_battles=1)
    asyncio.run(run_local_battles(trainee, opp, 1, impl=impl))
    prefix = trainee.trace_prefixes[0]
    record = ReconstructionRecord.load(f"{prefix}_reconstruction.json")
    with open(f"{prefix}_summary.json") as f:
        summary = json.load(f)
    return record, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--battles", type=int, default=3)
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--out", default="tmp/search_golden_node.json")
    a = ap.parse_args()

    cases = []
    call_id = 0
    for _ in range(a.battles):
        with tempfile.TemporaryDirectory() as td:
            record, summary = record_one(td)
        invs = summary["invocations"]
        cand = [i for i in range(len(invs))
                if invs[i].get("phase") == "move_selection" and int(invs[i]["turn"]) > 1]
        if not cand:
            continue
        turns = sorted({int(invs[cand[int(len(cand) * f)]]["turn"])
                        for f in (0.25, 0.5, 0.75)})[:a.turns]
        drv = NodeDriver()
        try:
            for turn in turns:
                call_id += 1
                root = drv.call({"id": call_id, "cmd": "open_root",
                                 "record": record.to_dict(), "turn": int(turn)})
                if not root.get("ok", True) or "node_id" not in root:
                    continue
                arms = arms_for(root)
                call_id += 1
                exp = drv.call({"id": call_id, "cmd": "expand_many", "arms": arms})
                child = next((x["node_id"] for x in exp.get("arms", []) if x.get("node_id")),
                             None)
                d2 = None
                if child:
                    call_id += 1
                    d2 = drv.call({"id": call_id, "cmd": "expand_many", "arms": [
                        {"node_id": child, "label": 100, "p1_action": "move 1",
                         "p2_action": "random", "seed": "sodium,abc123"}]})
                cases.append({"record": record.to_dict(), "turn": int(turn),
                              "open_root": root, "expand_many": exp,
                              **({"expand_depth2": d2} if d2 else {})})
        finally:
            drv.close()
    with open(a.out, "w") as f:
        json.dump({"cases": cases}, f)
    print(f"wrote {len(cases)} cases -> {a.out}")


if __name__ == "__main__":
    main()
