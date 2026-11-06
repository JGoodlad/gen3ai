"""S'-HONEST stage 1, the AGREEMENT NOISE FLOOR (2026-11-05, OUTCOME-BLIND): r0s4_rolls against ITSELF under another
key (other world draws, other playout dice) at the first N states of the timing check's sample, so the two presets'
agreement (timing.py) is read against the agreement two runs of one preset reach. The live searcher's `_search`, P*
greedy on CPU, one thread; resumable (<out>/noise.jsonl).

    PYTHONPATH=src python3 noise_floor.py --timing DIR --selected S --model ZIP --model-sha256 H --n 100
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timing", required=True)
    ap.add_argument("--selected", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-sha256", required=True)
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch

    torch.set_num_threads(1)
    import numpy as np
    from agents.battle.core_obs import wrap_row
    from main.anchors.core_side import CorePolicy, load_policy
    from main.anchors.search_side import LiveSearcher
    from main.anchors.session import OurSideState
    from main.value_search.sign_gate import GateScorer

    if hashlib.sha256(Path(a.model).read_bytes()).hexdigest() != a.model_sha256:
        sys.exit("the model's bytes are not the registered ones")
    rows = [json.loads(x) for x in (Path(a.timing) / "rows.jsonl").read_text().splitlines() if x.strip()]
    rows = sorted(rows, key=lambda r: hashlib.sha256(f"sprime_honest_stage1:{r['key']}".encode()).hexdigest())[:a.n]
    want = {r["key"] for r in rows}
    sel = {}
    for x in open(a.selected):
        r = json.loads(x)
        if r["key"] in want:
            sel[r["key"]] = r
    model = load_policy(str(a.model), "cpu", "bare", OurSideState())
    s = LiveSearcher(model, CorePolicy(model, stochastic=False, temperature=1.0), "r0s4_rolls@10", Path("/dev/null"))
    guard = GateScorer(a.model, searcher=s)
    out = Path(a.timing) / "noise.jsonl"
    done = {json.loads(x)["key"] for x in out.read_text().splitlines() if x.strip()} if out.exists() else set()
    for r in rows:
        if r["key"] in done:
            continue
        rec = sel[r["key"]]
        _t, d = guard.root(rec)
        row = np.array(wrap_row(d["frame"]), dtype=np.float32)
        got = s._search(rec["log"], int(rec["at"]), rec["side"], row, np.asarray(rec["mask"], dtype=np.int8), d,
                        rec["key"] + "|noise", int(rec["pi_argmax"]))
        line = {"key": r["key"], "panel": r["panel"], "searched": got["searched"], "played": got["played"],
                "cpu_s": got["cpu_s"]}
        with open(out, "a") as f:
            f.write(json.dumps(line, sort_keys=True) + "\n")
        print(json.dumps(line), flush=True)


if __name__ == "__main__":
    main()
