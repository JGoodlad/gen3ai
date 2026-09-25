"""Build STAND-IN run dirs (no weights) shaped like the convergence side-check's RB+ / RC+, so the
§4.4 invocations of `main.best_response_gap` can be exercised before either extension exists.

Each stand-in copies its round-1 reader's small files (metadata.json, model_config.json,
checkpoints/*.json sidecars), re-points its fork parent at that reader's final (111,280,128), keeps
its recorded exploiter target (B's or C's REAL final @103,219,200), sets num_timesteps to the
registered landing 115,310,592, and writes TWO eval cycles (the ones an extension from 111,280,128
reaches on the 2M absolute grid: ~112M and ~114M) whose vs-target win counts are a SCENARIO — not a
prediction. Nothing is written under models/.

    python brgap_ext_standin_sim.py <outdir>
"""
import json
import os
import shutil
import sys

M = "/home/goodlad/dev/gen3ai/models"
OUT = sys.argv[1]
FORK = 111280128
LANDS = 115310592
GRID_SHIFT = 8000000               # the reader's cycles 1-2 (104M, 106M) -> the extension's (112M, 114M)

# (stand-in name, source reader, per-cycle wins out of 100 at ~112M and ~114M) — SCENARIO values
SCEN = {
    "sim_ext_RBX": ("ai_v13_24_popr1_read_loop", [62, 63]),
    "sim_ext_RCX": ("ai_v13_25_popr1_read_ctrl", [68, 67]),
}

os.makedirs(OUT, exist_ok=True)
for name, (src_run, wins) in SCEN.items():
    src = os.path.join(M, src_run)
    d = os.path.join(OUT, name)
    os.makedirs(os.path.join(d, "checkpoints"), exist_ok=True)
    shutil.copy2(os.path.join(src, "model_config.json"), os.path.join(d, "model_config.json"))
    for f in os.listdir(os.path.join(src, "checkpoints")):
        if f.endswith(".json"):
            shutil.copy2(os.path.join(src, "checkpoints", f), os.path.join(d, "checkpoints", f))
    meta = json.load(open(os.path.join(src, "metadata.json")))
    lin = meta["lineage"]
    fp = lin["fork_parent"]
    fp.update(run_name=src_run, run_dir=src,
              path=f"models/{src_run}/final_model.zip",
              resolved_path=os.path.join(src, "final_model.zip"),
              resolved_file=os.path.join(src, "final_model.zip"),
              num_timesteps=FORK, resolved_num_timesteps=FORK)
    lin["fork_step"] = FORK            # the exploiter_target block is left EXACTLY as recorded
    meta["num_timesteps"] = LANDS
    json.dump(meta, open(os.path.join(d, "metadata.json"), "w"))
    tgt = lin["exploiter_target"]["run_name"]
    rows = [json.loads(line) for line in open(os.path.join(src, "eval_results.jsonl")) if line.strip()]
    with open(os.path.join(d, "eval_results.jsonl"), "w") as fh:
        for row, w in zip(rows, wins):
            row["step"] = int(row["step"]) + GRID_SHIFT
            row["externals"] = {f"ext_{tgt}": {"win_rate": w / 100, "counts": [w, 100]}}
            fh.write(json.dumps(row) + "\n")
    print(name, "target", tgt, "fork", FORK, "lands", LANDS, "cycles",
          [json.loads(line)["step"] for line in open(os.path.join(d, "eval_results.jsonl"))])
