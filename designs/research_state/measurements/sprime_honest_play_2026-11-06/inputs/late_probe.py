"""S'-HONEST play PREREG input (2026-11-06, OUTCOME-BLIND): r0s4_honest@10 against r0s4_rolls@10 on S'-LIVE's own
searched-game states at EVERY turn, the late game included (stage 1 measured dlive's raw states, turns 10-30 only).

The states: S'-LIVE's re-score sample (`rescore_sample: true` records of `search_decisions.jsonl`, which keep the
honest log, the mask and the tokens), from complete attempt dirs only (never a `.partial.<epoch>` dir), stratified by
panel and turn bucket, the first N of a stratum in the order sha256("sprime_honest_play_inputs:" + key). Never read: a
game result, a summary, a capture, `games.jsonl`. Each state runs both presets through the live searcher's own
`_search` behind `sign_gate.GateScorer`'s root guards (the identity re-score's own path), P* greedy on CPU, one thread,
in one process, so their CPU is compared at the same contention. Resumable: one row a state in <out>/rows.jsonl.
`--shard i/k` runs the i-th of k interleaved slices.

    PYTHONPATH=src python3 late_probe.py --cells C --model ZIP --model-sha256 H --out DIR [--shard 0/2]
    PYTHONPATH=src python3 late_probe.py --summarize --out DIR --cells C
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

PANELS = {"heldout72_hpfix": "H", "forum72": "F", "p72": "P72"}
PRESETS = ("r0s4_rolls@10", "r0s4_honest@10")
#: (bucket, lo, hi, states a panel): stage 1 covered turns 10-30 on dlive's states; the late buckets are the new input
BUCKETS = (("10-19", 10, 19, 6), ("20-29", 20, 29, 6), ("30-63", 30, 63, 18), ("64+", 64, 10 ** 6, 18))
FIELDS = ("searched", "why", "error", "played", "deviation", "cpu_s", "world_cpu_s", "playout_cpu_s", "wall_s", "q",
          "log_pi", "root_v", "spent", "world_tries", "search_key")


def bucket(turn):
    return next(b for b, lo, hi, _n in BUCKETS if lo <= turn <= hi)


def order(key):
    return hashlib.sha256(f"sprime_honest_play_inputs:{key}".encode()).hexdigest()


def sampled_records(cells):
    """Every sampled search-path record of a complete attempt dir: (panel, record)."""
    out = []
    for d in sorted(os.listdir(cells)):
        if ".partial." in d:
            continue
        panel = d.split("__")[1]
        p = Path(cells) / d / "search_decisions.jsonl"
        if panel not in PANELS or not p.is_file():
            continue
        for x in open(p):
            r = json.loads(x)
            if r.get("rescore_sample") is True and "log" in r:
                out.append((panel, r))
    return out


def sample(cells):
    by = {}
    for panel, r in sampled_records(cells):
        by.setdefault((panel, bucket(int(r["turn"]))), []).append(r)
    todo = []
    for panel in PANELS:
        for b, _lo, _hi, n in BUCKETS:
            rs = sorted(by.get((panel, b), []), key=lambda r: order(r["key"]))[:n]
            todo += [(panel, b, r) for r in rs]
    return todo


def validator_cpu():
    """The CPU-s of the process's shared Showdown validator child so far (stage 1's FINDING S'H-9)."""
    from main.value_search import world_legality as WL

    v = WL._SHARED.get(WL.FORMAT)
    p = getattr(v, "_proc", None)
    if p is None or p.poll() is not None:
        return 0.0
    try:
        f = open(f"/proc/{p.pid}/stat").read().rsplit(")", 1)[1].split()
    except OSError:
        return 0.0
    return (int(f[11]) + int(f[12])) / os.sysconf("SC_CLK_TCK")


def run(a):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import numpy as np
    import torch

    torch.set_num_threads(1)
    from agents.battle.core_obs import wrap_row
    from main.anchors.core_side import CorePolicy, load_policy
    from main.anchors.search_side import LiveSearcher
    from main.anchors.server import engine_stamp
    from main.anchors.session import OurSideState
    from main.value_search.sign_gate import GateScorer

    if hashlib.sha256(Path(a.model).read_bytes()).hexdigest() != a.model_sha256:
        sys.exit("the model's bytes are not the registered ones")
    model = load_policy(str(a.model), "cpu", "bare", OurSideState())
    policy = CorePolicy(model, stochastic=False, temperature=1.0)
    ss = {p: LiveSearcher(model, policy, p, Path("/dev/null")) for p in PRESETS}
    guard = GateScorer(a.model, searcher=ss[PRESETS[0]])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_f = out / "rows.jsonl"
    done = set()
    if rows_f.exists():
        done = {json.loads(x)["key"] for x in rows_f.read_text().splitlines() if x.strip()}
    i, k = (int(x) for x in a.shard.split("/"))
    todo = [x for j, x in enumerate(sample(a.cells)) if j % k == i]
    eng = engine_stamp()
    for panel, b, rec in todo:
        if rec["key"] in done:
            continue
        t0 = time.monotonic()
        _truth, d = guard.root(rec)
        row = np.array(wrap_row(d["frame"]), dtype=np.float32)
        mask = np.asarray(rec["mask"], dtype=np.int8)
        argmax = int(rec["pi_argmax"])
        res = {}
        for p in PRESETS:
            v0 = validator_cpu()
            r = ss[p]._search(rec["log"], int(rec["at"]), rec["side"], row, mask, d, rec["key"], argmax)
            res[p] = {kk: r.get(kk) for kk in FIELDS}
            res[p]["validator_cpu_s"] = round(validator_cpu() - v0, 3)
            if "honest" in r:
                h = r["honest"]
                res[p]["honest"] = {"chain": h["chain"], "fails": h["fails"],
                                    "world_feeds": [w.get("feeds") for w in h["worlds"]],
                                    "world_rerolls": [w.get("rerolls") for w in h["worlds"]]}
        live = {kk: rec.get(kk) for kk in ("searched", "why", "played", "cpu_s", "world_cpu_s", "playout_cpu_s")}
        line = {"key": rec["key"], "panel": PANELS[panel], "bucket": b, "turn": int(rec["turn"]),
                "n_legal": int(rec["n_legal"]), "pi_argmax": argmax, "engine": eng, "live_rolls": live, "res": res,
                "wall_s": round(time.monotonic() - t0, 3)}
        with open(rows_f, "a") as f:
            f.write(json.dumps(line, sort_keys=True) + "\n")
        print(json.dumps({"key": rec["key"][-16:], "panel": line["panel"], "turn": line["turn"],
                          **{p.split("@")[0]: (res[p]["searched"], res[p]["cpu_s"]) for p in PRESETS}}), flush=True)


def summarize(a):
    """Per panel and bucket: both presets' exclusion rate and CPU-s a decision, the honest / rolls CPU ratio on the
    same states, and S'-LIVE's own search-path decision count and live rolls CPU by bucket (every record of the
    complete attempt dirs: its `searched`, `why`, `cpu_s` and `turn` only)."""
    import statistics as S

    rows = [json.loads(x) for x in (Path(a.out) / "rows.jsonl").read_text().splitlines() if x.strip()]
    live = {}
    for d in sorted(os.listdir(a.cells)):
        if ".partial." in d:
            continue
        panel = d.split("__")[1]
        p = Path(a.cells) / d / "search_decisions.jsonl"
        if panel not in PANELS or not p.is_file():
            continue
        for x in open(p):
            r = json.loads(x)
            if "key" not in r:
                continue
            e = live.setdefault((PANELS[panel], bucket(int(r["turn"]))), {"n": 0, "excluded": 0, "cpu_s": 0.0})
            e["n"] += 1
            e["excluded"] += int(r.get("why") == "excluded")
            e["cpu_s"] += float(r.get("cpu_s") or 0.0)
        live.setdefault(("games", PANELS[panel]), {"n": 0})["n"] += 1
    out = {"n_states": len(rows), "engine": sorted({r["engine"] for r in rows}), "by": {}}
    for panel in PANELS.values():
        for b, _lo, _hi, _n in BUCKETS:
            rs = [r for r in rows if r["panel"] == panel and r["bucket"] == b]
            if not rs:
                continue
            e = {}
            for p in PRESETS:
                xs = [r["res"][p] for r in rs]
                cpu = [x["cpu_s"] + x["validator_cpu_s"] for x in xs]
                e[p.split("@")[0]] = {"excluded": sum(1 for x in xs if x["why"] == "excluded"), "n": len(xs),
                                      "cpu_s_mean": round(S.mean(cpu), 2), "cpu_s_median": round(S.median(cpu), 2),
                                      "cpu_s_max": round(max(cpu), 2)}
            hon = [r["res"]["r0s4_honest@10"] for r in rs]
            rol = [r["res"]["r0s4_rolls@10"] for r in rs]
            e["cpu_ratio_honest_over_rolls"] = round(sum(x["cpu_s"] + x["validator_cpu_s"] for x in hon)
                                                     / sum(x["cpu_s"] + x["validator_cpu_s"] for x in rol), 3)
            both = [(h, r) for h, r in zip(hon, rol) if h["searched"] and r["searched"]]
            e["same_pick_both_searched"] = [sum(1 for h, r in both if h["played"] == r["played"]), len(both)]
            e["rolls_reproduces_live"] = sum(1 for r in rs if r["res"]["r0s4_rolls@10"]["played"]
                                             == r["live_rolls"]["played"]
                                             and r["res"]["r0s4_rolls@10"]["why"] == r["live_rolls"]["why"])
            lv = live.get((panel, b), {})
            e["slive"] = {"search_path_decisions": lv.get("n"), "excluded": lv.get("excluded"),
                          "excl_rate": round(lv["excluded"] / lv["n"], 4) if lv.get("n") else None,
                          "live_rolls_cpu_s_mean": round(lv["cpu_s"] / lv["n"], 2) if lv.get("n") else None}
            out["by"][f"{panel}/{b}"] = e
    out["slive_games"] = {k[1]: v["n"] for k, v in live.items() if k[0] == "games"}
    print(json.dumps(out, indent=1, sort_keys=True))
    (Path(a.out) / "summary.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--model")
    ap.add_argument("--model-sha256")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args()
    summarize(a) if a.summarize else run(a)
