"""Flags-off identity for gen3_endstate_facts_v1: the dynamo graph / state_dict / output shas of five configs, at the
parent tree and at this build, through `../static_recovery_2026-10-09/graph_sha.py`.

    python shas.py <parent src dir> <build src dir>        # the parent: `git archive 58f8149a src data designs/production_config.json`

Prints one JSON line per (config, tree)."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
GRAPH_SHA = os.path.join(HERE, "..", "static_recovery_2026-10-09", "graph_sha.py")
TREES = {"parent": sys.argv[1], "build": sys.argv[2]}
SR = {"token_encoding": "static", "mon_hazard_cost": "on", "move_actor_state": "on", "trunk_layers": 3,
      "switch_hazard_cost": "on", "eot_residual": "on"}
E = {**SR, "move_resolution": "on", "speed_physics": "on", "value_threat_inject": False, "op_reduction": "principled",
     "obs_facts": "v1"}
END = {**E, "move_resolution_facts": "full", "status_facts": "exact", "ko_ramp": "exact", "drop_progress_clock": "on",
       "g_ledger": "eot"}
CONFIGS = [("production", {}, ("parent", "build")), ("static", {"token_encoding": "static"}, ("parent", "build")),
           ("static_recovery", SR, ("parent", "build")), ("E (bundle + static_recovery)", E, ("parent", "build")),
           ("endstate", END, ("build",))]
for name, ov, trees in CONFIGS:
    for t in trees:
        code = (f"import runpy,sys; sys.argv=['graph_sha.py', {json.dumps(json.dumps(ov))}]; "
                f"runpy.run_path({GRAPH_SHA!r}, run_name='__main__')")
        out = subprocess.run([sys.executable, "-c", code], cwd=TREES[t], capture_output=True, text=True,
                             timeout=1800).stdout.strip().splitlines()
        d = json.loads(out[-1])
        print(json.dumps({"config": name, "tree": t, **{k: d[k] for k in ("n_graphs", "graph_lines", "graph_sha",
                                                                          "state_sha", "out_sha", "n_params")}}),
              flush=True)
