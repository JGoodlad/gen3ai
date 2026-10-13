"""python shas2.py <tree root>  -> production + endstate graph/state/out sha with that tree's src first."""
import json
import os
import subprocess
import sys

TREE = os.path.abspath(sys.argv[1])
SRC = os.path.join(TREE, "src")
GRAPH_SHA = os.path.join(TREE, "designs/research_state/measurements/static_recovery_2026-10-09/graph_sha.py")
sys.path.insert(0, SRC)
from main.train.arch_arms import arm_overlay  # noqa: E402

END = arm_overlay("endstate")
print(json.dumps({"tree": TREE, "endstate_overlay": END}), flush=True)
for name, ov in [("production", {}), ("endstate", END)]:
    code = (f"import runpy,sys; sys.argv=['graph_sha.py', {json.dumps(json.dumps(ov))}]; "
            f"runpy.run_path({GRAPH_SHA!r}, run_name='__main__')")
    env = dict(os.environ, PYTHONPATH=SRC)
    r = subprocess.run([sys.executable, "-c", code], cwd=SRC, capture_output=True, text=True, timeout=3000, env=env)
    lines = r.stdout.strip().splitlines()
    if r.returncode != 0 or not lines:
        print(json.dumps({"config": name, "error": r.stderr[-3000:]}), flush=True)
        continue
    d = json.loads(lines[-1])
    print(json.dumps({"config": name, **{k: d[k] for k in ("src", "n_graphs", "graph_lines", "graph_sha",
                                                          "state_sha", "out_sha", "n_params")}}), flush=True)
