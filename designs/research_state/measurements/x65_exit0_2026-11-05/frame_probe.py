"""Feasibility probe: rebuild S'-LIVE searched-decision frames from the game capture alone; compare frame_sha256."""
import json, sys, time
from pathlib import Path
from main.anchors.input_log import frame_sha256
from main.value_search import worlds as W

cells = [Path(p) for p in sys.argv[1:]]
ok = bad = err = 0
t0 = time.process_time()
for cell in cells:
    cap = json.load(open(next((cell / "capture").glob("*.json"))))
    start = json.loads(cap["commands"][0][len("START "):])
    cmds = cap["commands"][1:]
    for line in open(cell / "search_decisions.jsonl"):
        d = json.loads(line)
        if not d.get("searched"):
            continue
        log = {"format_id": start["formatid"], "seed": ",".join(str(x) for x in cap["seed"]),
               "names": [start["p1"]["name"], start["p2"]["name"]],
               "teams": [start["p1"]["team"], start["p2"]["team"]], "cmds": cmds[: int(d["at"])]}
        if d.get("rescore_sample") and d["log"]["cmds"] != log["cmds"]:
            print("LOG ORDER DIFFERS at", d["key"]); 
        try:
            truth = W.prepare_root(log, int(d["at"]), d["side"], d["key"], root_choice_in_log=False)
            if truth.ref and frame_sha256(truth.ref[-1]) == d["frame_sha256"]:
                ok += 1
            else:
                bad += 1
                print("BAD", cell.name, d["key"], d["at"], "sampled" if d.get("rescore_sample") else "")
        except Exception as e:
            err += 1
            if err < 3: print("ERR", type(e).__name__, str(e)[:300])
print(json.dumps({"ok": ok, "bad": bad, "err": err, "cpu_s": round(time.process_time() - t0, 2)}))
