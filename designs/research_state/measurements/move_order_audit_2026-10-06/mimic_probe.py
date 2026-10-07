"""Mimic / Transform: does the request block still resolve every choosable move to a per-mon slot?"""
import json
import numpy as np
import torch as th

from agents.battle.core_obs import wrap_row
from agents.battle.rust_core_parity import RecordedBattle, run_core
from agents.training import learner_golden as LG
from main.train.production_args import production_args

CLAYDOL = "Claydol||leftovers|levitate|rapidspin,mimic,earthquake,toxic|Sassy|252,100,112,,16,28|||||"
DITTO = "Ditto||leftovers|limber|transform|Relaxed|252,,252,,4,|||||"
SNORLAX = "Snorlax||leftovers|thickfat|bodyslam,curse,rest,shadowball|Careful|252,,4,,252,|||||"
SKARM = "Skarmory||leftovers|keeneye|spikes,toxic,drillpeck,roar|Impish|252,,252,,4,|||||"
battles = [
    RecordedBattle(label="mimic", format_id="gen3ou", seed="sodium,00000000000000000000000000000001",
                   p1={"name": "A", "team": CLAYDOL + "]" + SKARM},
                   p2={"name": "B", "team": SNORLAX + "]" + SKARM},
                   commands=[["p1", "move earthquake"], ["p2", "move bodyslam"],
                             ["p1", "move mimic"], ["p2", "move bodyslam"],
                             ["p1", "move toxic"], ["p2", "move curse"],
                             ["p1", "move earthquake"], ["p2", "move curse"]]),
    RecordedBattle(label="transform", format_id="gen3ou", seed="sodium,00000000000000000000000000000002",
                   p1={"name": "A", "team": DITTO + "]" + SKARM},
                   p2={"name": "B", "team": SNORLAX + "]" + SKARM},
                   commands=[["p1", "move transform"], ["p2", "move curse"],
                             ["p1", "move curse"], ["p2", "move curse"],
                             ["p1", "move bodyslam"], ["p2", "move curse"]]),
]
m = LG.build_learner(args=production_args())
fe = m.policy.features_extractor
for res in run_core(battles, trackers=True, obs=True):
    print("=====", res["label"], "ok", res["ok"], res.get("error"))
    if not res["ok"]:
        continue
    chunks = res["chunks"]
    for entry in res["trackers"][0]:
        if "obs" not in entry:
            continue
        req = None
        for side, lines in chunks[: int(entry["after"]) + 1]:
            if side == 0:
                for ln in lines:
                    if ln.startswith("|request|"):
                        req = json.loads(ln[len("|request|"):])
        row = th.as_tensor(np.array(wrap_row(entry["obs"])))[None]
        ctx = fe.unpack({"observation": row})
        srt = ctx.all_move_ids[0, ctx.our_active_idx[0]].tolist()
        rids = ctx.our_active_req_move_ids[0].tolist()
        leg = ctx.our_active_req_move_legal[0].tolist()
        rmoves = [(mv["id"], mv.get("disabled")) for mv in (req.get("active") or [{}])[0].get("moves", [])]
        print("request", rmoves, "| req ids", rids, "legal", leg, "| sorted slot ids", srt,
              "| mask", entry["mask"], "| tokens", entry["tokens"], "| choice", entry.get("choice"))
