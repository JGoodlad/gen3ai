"""S'-HONEST play PREREG input (2026-11-06, OUTCOME-BLIND): THE STAGE-1 TOOTH. Is THIS checkout's r0s4_honest@10, at
its DEFAULT budget, the builder stage 1 measured at budget B3 (`sprime_honest_stage1_2026-11-05/`)?

The states: stage 1's B3 re-runs (`curve_b3_rows.jsonl.gz`: the 44 states the base budget limited and six it did not,
every one run with `--honest-budget 1800,4000000,8`), the first N in the order sha256("sprime_honest_play_tooth:" +
key), their kept logs from dlive's `selected.jsonl` (never a referee field or an outcome). Each runs through the live
searcher's own `_search` behind `sign_gate.GateScorer`'s root guards, P* greedy on CPU, one thread, with the preset's
DEFAULT sampler (no budget override). Every non-timing field stage 1 kept must be equal: searched, why, error, played,
deviation, q, world_tries, and the honest build record (the chain's counts, the fails, each world's feeds and
re-rolls). Resumable: one row a state in <out>/rows.jsonl.

    PYTHONPATH=src python3 stage1_tooth.py --rows R.jsonl.gz --selected S --model ZIP --model-sha256 H --out DIR [--n 24]
"""
import argparse
import gzip
import hashlib
import json
import os
import sys
import time
from pathlib import Path

PRESET = "r0s4_honest@10"
FIELDS = ("searched", "why", "error", "played", "deviation", "q", "world_tries")


def order(key):
    return hashlib.sha256(f"sprime_honest_play_tooth:{key}".encode()).hexdigest()


def kept(r):
    """Stage 1's kept non-timing fields of one honest record (its own row format)."""
    out = {k: r.get(k) for k in FIELDS}
    h = r.get("honest")
    out["honest"] = None if h is None else {"chain": h["chain"], "fails": h["fails"], "world_feeds": h["world_feeds"],
                                            "world_rerolls": h["world_rerolls"]}
    return json.loads(json.dumps(out, sort_keys=True))


def main(a):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import torch

    torch.set_num_threads(1)
    from agents.battle.core_obs import wrap_row
    from main.anchors.server import engine_stamp
    from main.value_search.sign_gate import GateScorer

    if hashlib.sha256(Path(a.model).read_bytes()).hexdigest() != a.model_sha256:
        sys.exit("the model's bytes are not the registered ones")
    rows = [json.loads(x) for x in gzip.open(a.rows, "rt")]
    todo = sorted(rows, key=lambda r: order(r["key"]))[:a.n]
    want = {r["key"] for r in todo}
    sel = {}
    for x in open(a.selected):
        r = json.loads(x)
        if r["key"] in want:
            sel[r["key"]] = r
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_f = out / "rows.jsonl"
    done = {json.loads(x)["key"] for x in rows_f.read_text().splitlines() if x.strip()} if rows_f.exists() else set()
    g = GateScorer(a.model, PRESET)                     # the preset's own default sampler: the budget is the code's
    eng = engine_stamp()
    try:
        for r in todo:
            if r["key"] in done:
                continue
            rec = sel[r["key"]]
            t0 = time.monotonic()
            _truth, d = g.root(rec)
            row = np.array(wrap_row(d["frame"]), dtype=np.float32)
            s = g.s._search(rec["log"], int(rec["at"]), rec["side"], row, np.asarray(rec["mask"], dtype=np.int8), d,
                            rec["key"], int(rec["pi_argmax"]))
            h = s.get("honest")
            got = {k: s.get(k) for k in FIELDS}
            got["honest"] = None if h is None else {"chain": h["chain"], "fails": h["fails"],
                                                    "world_feeds": [w.get("feeds") for w in h["worlds"]],
                                                    "world_rerolls": [w.get("rerolls") for w in h["worlds"]]}
            got = json.loads(json.dumps(got, sort_keys=True, default=str))
            ref = kept(r["res"][PRESET])
            diff = sorted(k for k in ref if ref[k] != got.get(k))
            line = {"key": r["key"], "panel": r["panel"], "turn": r["turn"], "engine": eng, "diff": diff,
                    "stage1": ref, "here": got, "cpu_s": s.get("cpu_s"), "wall_s": round(time.monotonic() - t0, 3)}
            with open(rows_f, "a") as f:
                f.write(json.dumps(line, sort_keys=True) + "\n")
            print(json.dumps({"key": r["key"][-16:], "panel": r["panel"], "turn": r["turn"], "diff": diff,
                              "cpu_s": s.get("cpu_s")}), flush=True)
    finally:
        g.close()
    res = [json.loads(x) for x in rows_f.read_text().splitlines() if x.strip()]
    doc = {"schema": "sprime_honest_stage1_tooth_v1", "preset": PRESET, "n": len(res), "engine": sorted({x["engine"] for x in res}),
           "mismatched": [x["key"] for x in res if x["diff"]], "by_panel": {}, "searched": sum(1 for x in res if x["here"]["searched"]),
           "excluded": sum(1 for x in res if x["here"]["why"] == "excluded"),
           "verdict": "PASS" if res and not any(x["diff"] for x in res) else "FAIL"}
    for x in res:
        doc["by_panel"][x["panel"]] = doc["by_panel"].get(x["panel"], 0) + 1
    (out / "tooth.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps(doc, sort_keys=True))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--selected", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-sha256", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=24)
    main(ap.parse_args())
