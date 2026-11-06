"""S'-HONEST stage 1, B8's TIMING AND EXCLUSION CHECK (2026-11-05, OUTCOME-BLIND): r0s4_rolls and r0s4_honest at the
SAME banked U states of the dlive read, the live searcher's own `_search` (its root guards, P* greedy on CPU, one
thread), CPU per decision, the exclusion rate and the two presets' agreement, per panel (H, F, P72).

The states: dlive's SCORED states of the three U panels (`scored.jsonl`, never a referee field or an outcome; the
selected row carries the kept log, its frame digest, mask and tokens), in the order sha256("sprime_honest_stage1:" +
key), the first N a panel. Resumable: one row a state in <out>/rows.jsonl, a done key skipped. `--shard i/k` runs
the i-th of k interleaved slices (two processes share the file by appending whole lines).

    PYTHONPATH=src python3 timing.py --selected S --scored C --model ZIP --model-sha256 H --per-panel 100 \
        --out DIR [--shard 0/2]
    PYTHONPATH=src python3 timing.py --summarize --out DIR
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
#: S'-LIVE's searched units (design_sprime_honest.md §4): 6.2 CPU-min a searched game, 103,616 search-path
#: decisions over its 2,698 counted searched units
SLIVE_GAME_CPU_MIN, SLIVE_DECISIONS_PER_GAME = 6.2, 103616 / 2698
VOLATILE = ("cpu_s", "wall_s", "process_cpu_s", "world_cpu_s", "playout_cpu_s")


def validator_cpu():
    """The CPU-s of the process's shared Showdown validator child so far (a persistent Node process the searcher's
    own `cpu_s` does not count: the record's FINDING), or 0 before it starts."""
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


def order(key):
    return hashlib.sha256(f"sprime_honest_stage1:{key}".encode()).hexdigest()


def sample(selected, scored, per_panel):
    sel = {}
    for x in open(selected):
        r = json.loads(x)
        sel[r["key"]] = r
    by = {p: [] for p in PANELS}
    for x in open(scored):
        r = json.loads(x)
        if r["panel"] in by and r["key"] in sel:
            by[r["panel"]].append(r["key"])
    out = []
    for p, keys in by.items():
        out += [(p, k) for k in sorted(keys, key=order)[:per_panel]]
    return [(p, sel[k]) for p, k in out]


def run(a):
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
    model = load_policy(str(a.model), "cpu", "bare", OurSideState())
    policy = CorePolicy(model, stochastic=False, temperature=1.0)
    presets = tuple(a.presets.split(",")) if a.presets else PRESETS
    smp = {}
    if a.honest_budget:
        from main.value_search.honest_worlds import HonestSampler

        st, fd, at = (int(x) for x in a.honest_budget.split(","))
        smp = {p: HonestSampler(max_steps=st, chain_feeds=fd, attempts=at) for p in presets if "honest" in p}
    ss = {p: LiveSearcher(model, policy, p, Path("/dev/null"), sampler=smp.get(p)) for p in presets}
    guard = GateScorer(a.model, searcher=ss[presets[0]])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows_f = out / "rows.jsonl"
    done = set()
    if rows_f.exists():
        done = {json.loads(x)["key"] for x in rows_f.read_text().splitlines() if x.strip()}
    i, k = (int(x) for x in a.shard.split("/"))
    todo = [x for j, x in enumerate(sample(a.selected, a.scored, a.per_panel)) if j % k == i]
    if a.keys:
        want = set(Path(a.keys).read_text().split())
        todo = [x for x in todo if x[1]["key"] in want]
    for panel, rec in todo:
        if rec["key"] in done:
            continue
        t0 = time.monotonic()
        _truth, d = guard.root(rec)
        row = np.array(wrap_row(d["frame"]), dtype=np.float32)
        mask = np.asarray(rec["mask"], dtype=np.int8)
        argmax = int(rec["pi_argmax"])
        res = {}
        for p in presets:
            v0 = validator_cpu()
            r = ss[p]._search(rec["log"], int(rec["at"]), rec["side"], row, mask, d, rec["key"], argmax)
            vcpu = round(validator_cpu() - v0, 3)
            res[p] = {kk: r.get(kk) for kk in ("searched", "why", "error", "played", "deviation", "cpu_s",
                                               "world_cpu_s", "playout_cpu_s", "wall_s", "q", "world_tries")}
            res[p]["validator_cpu_s"] = vcpu
            if "honest" in r:
                h = r["honest"]
                res[p]["honest"] = {"chain": h["chain"], "fails": h["fails"],
                                    "world_feeds": [w.get("feeds") for w in h["worlds"]],
                                    "world_rerolls": [w.get("rerolls") for w in h["worlds"]]}
        line = {"key": rec["key"], "panel": PANELS[panel], "turn": int(rec["turn"]), "pi_argmax": argmax,
                "n_legal": int(rec["n_legal"]), "res": res, "wall_s": round(time.monotonic() - t0, 3)}
        with open(rows_f, "a") as f:
            f.write(json.dumps(line, sort_keys=True) + "\n")
        print(json.dumps({"key": rec["key"][-16:], "panel": line["panel"], "turn": line["turn"],
                          **{p.split("@")[0]: (res[p]["searched"], res[p]["cpu_s"]) for p in presets}}), flush=True)


def summarize(a):
    import numpy as np

    rows = [json.loads(x) for x in (Path(a.out) / "rows.jsonl").read_text().splitlines() if x.strip()]
    seen, uniq = set(), []
    for r in rows:
        if r["key"] not in seen:
            seen.add(r["key"])
            uniq.append(r)
    out = {}
    for panel in ("H", "F", "P72", "U"):
        rs = [r for r in uniq if panel == "U" or r["panel"] == panel]
        if not rs:
            continue
        d = {"n": len(rs), "mean_turn": round(float(np.mean([r["turn"] for r in rs])), 2)}
        have = [p for p in PRESETS if all(p in r["res"] for r in rs)]
        for p in have:
            nm = p.split("@")[0]
            x = [r["res"][p] for r in rs]
            ex = [not v["searched"] for v in x]
            cpu = np.array([v["cpu_s"] for v in x], dtype=float)
            d[nm] = {"excluded": int(sum(ex)), "exclusion_rate": round(float(np.mean(ex)), 4),
                     "cpu_s_mean": round(float(cpu.mean()), 3), "cpu_s_median": round(float(np.median(cpu)), 3),
                     "cpu_s_p90": round(float(np.quantile(cpu, 0.9)), 3),
                     "world_cpu_s_mean": round(float(np.mean([v["world_cpu_s"] for v in x])), 3),
                     "playout_cpu_s_mean": round(float(np.mean([v["playout_cpu_s"] for v in x])), 3),
                     "deviation_rate": round(float(np.mean([bool(v.get("deviation")) for v in x])), 4)}
            if all("validator_cpu_s" in v for v in x):
                vc = np.array([v["validator_cpu_s"] for v in x], dtype=float)
                d[nm]["validator_cpu_s_mean"] = round(float(vc.mean()), 3)
                d[nm]["cpu_with_validator_s_mean"] = round(float((cpu + vc).mean()), 3)
        if len(have) < 2:
            out[panel] = d
            continue
        a_, b_ = (p.split("@")[0] for p in PRESETS)
        both = [r for r in rs if r["res"][PRESETS[0]]["searched"] and r["res"][PRESETS[1]]["searched"]]
        d["both_searched"] = len(both)
        d["agreement"] = round(float(np.mean([r["res"][PRESETS[0]]["played"] == r["res"][PRESETS[1]]["played"]
                                              for r in both])), 4) if both else None
        d["agreement_on_rolls_deviations"] = None
        dev = [r for r in both if r["res"][PRESETS[0]]["deviation"]]
        if dev:
            d["agreement_on_rolls_deviations"] = round(float(np.mean([r["res"][PRESETS[0]]["played"] ==
                                                                      r["res"][PRESETS[1]]["played"] for r in dev])), 4)
        noise_f = Path(a.out) / "noise.jsonl"
        if noise_f.exists():
            nz = {}
            for x in noise_f.read_text().splitlines():
                if x.strip():
                    q = json.loads(x)
                    nz[q["key"]] = q
            sub = [r for r in rs if r["key"] in nz and nz[r["key"]]["searched"]
                   and r["res"][PRESETS[0]]["searched"] and r["res"][PRESETS[1]]["searched"]]
            if sub:
                d["noise_floor"] = {
                    "n": len(sub),
                    "rolls_vs_rolls_other_key": round(float(np.mean([nz[r["key"]]["played"] ==
                                                                     r["res"][PRESETS[0]]["played"] for r in sub])), 4),
                    "honest_vs_rolls_same_states": round(float(np.mean([r["res"][PRESETS[1]]["played"] ==
                                                                        r["res"][PRESETS[0]]["played"] for r in sub])),
                                                         4)}
        ratio = d[b_]["cpu_s_mean"] / d[a_]["cpu_s_mean"]
        d["cpu_ratio_honest_over_rolls"] = round(ratio, 3)
        d["projected_cpu_min_per_searched_game"] = {
            "ratio_to_slive": round(SLIVE_GAME_CPU_MIN * ratio, 2),
            "absolute": round(SLIVE_DECISIONS_PER_GAME * d[b_]["cpu_s_mean"] / 60.0, 2)}
        out[panel] = d
    (Path(a.out) / "summary.json").write_text(json.dumps(out, indent=1, sort_keys=True))
    print(json.dumps(out, indent=1, sort_keys=True))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected")
    ap.add_argument("--scored")
    ap.add_argument("--model")
    ap.add_argument("--model-sha256")
    ap.add_argument("--per-panel", type=int, default=100)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--presets", help="a comma list (default: both)")
    ap.add_argument("--honest-budget", help="steps,chain_feeds,attempts for r0s4_honest's builder (default: its own)")
    ap.add_argument("--keys", help="a file of keys: run only these states")
    a = ap.parse_args()
    summarize(a) if a.summarize else run(a)
