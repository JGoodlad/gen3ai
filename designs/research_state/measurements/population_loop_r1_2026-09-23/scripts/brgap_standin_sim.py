"""Build STAND-IN run dirs (no weights) that look exactly like round-1's reader exploiters, so
`main.best_response_gap` can be exercised on the design before any arm exists.

Every reader is a copy of ai_v13_18_teach5_offense_hidose's small files (metadata.json,
model_config.json, eval_results.jsonl, checkpoints/*.json sidecars) with its exploiter target
re-pointed at a stand-in generalist (B or C) whose final sits at 103,219,200, the steps shifted by
+8,060,928, and the vs-target win counts replaced by the scenario's numbers.
Nothing is written under models/.
"""
import json, os, shutil, sys

M = "/home/goodlad/dev/gen3ai/models"
SRC = os.path.join(M, "ai_v13_18_teach5_offense_hidose")
OUT = sys.argv[1]
SHIFT = 103219200 - 95158272  # 8,060,928: a round-1 generalist's final vs G0

# (reader name, target name, per-cycle wins out of 100) — scenario values, NOT predictions
SCEN = {
    "sim_r1_reader_B": ("sim_r1_loop_B", [55, 54, 58, 57]),
    "sim_r1_reader_C": ("sim_r1_ctrl_C", [65, 60, 66, 66]),
}

os.makedirs(OUT, exist_ok=True)
# the two stand-in generalists: plateau's metadata (unpinned) under a new name
for tgt in ("sim_r1_loop_B", "sim_r1_ctrl_C"):
    d = os.path.join(OUT, tgt)
    os.makedirs(d, exist_ok=True)
    for f in ("metadata.json", "model_config.json"):
        shutil.copy2(os.path.join(M, "ai_v13_12_plateau", f), os.path.join(d, f))

for rd, (tgt, wins) in SCEN.items():
    d = os.path.join(OUT, rd)
    os.makedirs(os.path.join(d, "checkpoints"), exist_ok=True)
    shutil.copy2(os.path.join(SRC, "model_config.json"), os.path.join(d, "model_config.json"))
    for f in os.listdir(os.path.join(SRC, "checkpoints")):
        if f.endswith(".json"):
            shutil.copy2(os.path.join(SRC, "checkpoints", f), os.path.join(d, "checkpoints", f))
    meta = json.load(open(os.path.join(SRC, "metadata.json")))
    tdir = os.path.join(OUT, tgt)
    for blk in (meta["lineage"]["exploiter_target"], meta["lineage"]["fork_parent"]):
        blk.update(run_name=tgt, run_dir=tdir,
                   resolved_file=os.path.join(tdir, "final_model.zip"),
                   resolved_path=os.path.join(tdir, "final_model.zip"),
                   path=f"models/{tgt}/final_model.zip",
                   num_timesteps=103219200, resolved_num_timesteps=103219200)
    meta["lineage"]["fork_step"] = 103219200
    meta["num_timesteps"] = 111280128
    json.dump(meta, open(os.path.join(d, "metadata.json"), "w"))
    rows = [json.loads(l) for l in open(os.path.join(SRC, "eval_results.jsonl")) if l.strip()]
    with open(os.path.join(d, "eval_results.jsonl"), "w") as fh:
        for row, w in zip(rows, wins):
            row["step"] = int(row["step"]) + SHIFT
            row["externals"] = {f"ext_{tgt}": {"win_rate": w / 100, "counts": [w, 100]}}
            fh.write(json.dumps(row) + "\n")
print("built", sorted(os.listdir(OUT)))
