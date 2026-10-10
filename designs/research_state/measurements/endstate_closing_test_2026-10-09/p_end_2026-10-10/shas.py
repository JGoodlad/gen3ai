"""P_end identity: graph / state / outputs sha via the registered graph_sha.py, run with cwd = <tree>/src.

    python pend_shas.py <tree root>
"""
import json
import os
import subprocess
import sys

TREE = os.path.abspath(sys.argv[1])
SRC = os.path.join(TREE, "src")
GRAPH_SHA = os.path.join(TREE, "designs", "research_state", "measurements", "static_recovery_2026-10-09", "graph_sha.py")
sys.path.insert(0, SRC)
from main.train.arch_arms import arm_overlay  # noqa: E402

END = arm_overlay("endstate")
END_95 = {k: v for k, v in END.items() if k != "move_set_closure"}
CONFIGS = [("production", {}), ("endstate (new P_end overlay)", END), ("endstate @95d014fa overlay (control)", END_95)]
for name, ov in CONFIGS:
    code = (f"import runpy,sys; sys.argv=['graph_sha.py', {json.dumps(json.dumps(ov))}]; "
            f"runpy.run_path({GRAPH_SHA!r}, run_name='__main__')")
    r = subprocess.run([sys.executable, "-c", code], cwd=SRC, capture_output=True, text=True, timeout=3000)
    lines = r.stdout.strip().splitlines()
    if r.returncode != 0 or not lines:
        print(json.dumps({"config": name, "error": r.stderr[-2000:]}), flush=True)
        continue
    d = json.loads(lines[-1])
    print(json.dumps({"config": name, "src": d["src"], **{k: d[k] for k in ("n_graphs", "graph_lines", "graph_sha",
                                                                         "state_sha", "out_sha", "n_params")}}),
          flush=True)
