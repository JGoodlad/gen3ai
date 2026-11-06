"""S'-HONEST design input 3 (2026-11-05, OUTCOME-BLIND): the per-SEGMENT FRESH-DICE match rate a leak-honest world
builder faces (design_sprime_honest.md D3, §4), measured on S'-LIVE's kept honest logs with the TRUE teams.

For every command of a decision's prefix that RESOLVES a segment alone (a turn's second choice, a forced switch: the
other side has no open decision once it is fed), the playout tool opens the true prefix there (``rust_env_playout_open``),
plays that recorded command as the branch's forced action under the battle's OWN dice (``seeds[0] = null``) and under K
fresh seeds, steps once (to the next pending decision), and compares the searched side's protocol lines since the branch
point (its requests included) with the null branch's. A segment's rate = the share of fresh seeds whose lines are
byte-identical. Damage rolls are NOT controlled (the pessimistic case); every earlier segment keeps the true dice, so
a persisted counter is the truth's here (its cost is not in this number). Also: the CPU per branch and whether the
segment's lines hold a ``|-damage|`` line.

    PYTHONPATH=src python3 segment_match.py <S'-LIVE state>/anchor/u/pstar_s/cells 64 40 segment_match.json
"""
import ctypes
import glob
import hashlib
import json
import re
import sys
import time

import numpy as np

from main.value_search.playout import ACT, release_lib
from utils.rust_env.successors import SearchCore, _raise


def toid(s):
    return re.sub(r"[^a-z0-9]", "", s.lower())


def tok_index(tokens, tok):
    for i, t in tokens.items():
        if t == tok:
            return int(i)
    want = toid(tok)
    hits = [int(i) for i, t in tokens.items()
            if toid(t) == want or (want.startswith("movehiddenpower") and toid(t).startswith(want))]
    return hits[0] if len(hits) == 1 else None


def main(cells, K, nmax, out):
    core = SearchCore(lib=release_lib())

    def open_(req):
        return core._cstr(core.lib.rust_env_playout_open, json.dumps(req).encode())

    def segment(log, i, viewer):
        c = log["cmds"][i]
        if c.startswith("FORCELOSE"):
            return None
        _, s, tok = c.split(" ", 2)
        base = {"log": log, "at": i, "side": s, "actions": None, "seeds": [None], "stall": None, "max_turns": 999,
                "keep_cmds": False}
        root = open_(base)
        if root["other_open"]:
            return None                   # the other side still chooses: this command does not resolve alone
        a = tok_index(root["tokens"], tok)
        if a is None:
            return "unmatched"
        seeds = [None] + [f"sodium,{hashlib.sha256(f'sprime_honest_segm_v1:{i}:{k}'.encode()).hexdigest()[:32]}"
                          for k in range(K)]
        t0 = time.process_time()
        open_(dict(base, actions=[a], seeds=seeds, text=viewer))
        cap = 2 * len(seeds)
        rows = np.empty((cap, core.obs_dim), dtype=np.float32)
        masks = np.empty((cap, ACT), dtype=np.uint8)
        who = np.empty(cap, dtype=np.uint32)
        acts = np.zeros(cap, dtype=np.int32)
        with core._lock:
            k = core.lib.rust_env_playout_step(core._h, acts.ctypes.data, 0, rows.ctypes.data, masks.ctypes.data,
                                               who.ctypes.data, cap)
            if k == ctypes.c_size_t(-1).value:
                _raise(core.lib)
        res = core._cstr(core.lib.rust_env_playout_results)
        cpu = time.process_time() - t0
        br = res["branches"]
        ref = br[0]["text"]
        return sum(1 for b in br[1:] if b["text"] == ref), cpu, any(ln.startswith("|-damage|") for ln in ref)

    files = sorted(f for f in glob.glob(cells + "/*/search_decisions.jsonl") if ".partial." not in f)
    rng = np.random.default_rng(20261105)
    picked = []
    for f in rng.permutation(files):
        for line in open(f):
            if '"rescore_sample": true' in line:
                d = json.loads(line)
                if d.get("searched"):
                    picked.append(d)
                    break
        if len(picked) >= nmax:
            break
    rows_out = []
    for d in picked:
        log, at, side = d["log"], d["at"], d["side"]
        lg = dict(log, cmds=log["cmds"][:at])
        segs = [r for r in (segment(lg, i, side) for i in range(at)) if r is not None]
        ok = [s for s in segs if isinstance(s, tuple)]
        rates = [h / K for h, _c, _d in ok]
        rows_out.append({"key": d["key"], "turn": d["turn"], "segments": len(ok),
                         "unmatched": sum(1 for s in segs if s == "unmatched"),
                         "rates": [round(r, 4) for r in rates], "damage": [bool(dm) for _h, _c, dm in ok],
                         "cpu_per_branch_ms": round(1000 * float(np.mean([c / (K + 1) for _h, c, _d in ok])), 4),
                         "log10_whole_prefix_one_seed": round(float(np.sum(np.log10(np.maximum(rates, 1e-12)))), 2)})
        print(json.dumps({k: v for k, v in rows_out[-1].items() if k not in ("rates", "damage")}), flush=True)
    r = np.array([x for row in rows_out for x in row["rates"]])
    dm = np.array([x for row in rows_out for x in row["damage"]])

    def tries(x):
        return float(np.mean(1.0 / np.maximum(x, 1.0 / (2 * K))))

    summary = {"K": K, "decisions": len(rows_out), "segments": int(len(r)),
               "mean_turn": round(float(np.mean([row["turn"] for row in rows_out])), 2),
               "rate_mean": round(float(r.mean()), 4), "rate_mean_damage": round(float(r[dm].mean()), 4),
               "rate_mean_no_damage": round(float(r[~dm].mean()), 4), "damage_share": round(float(dm.mean()), 4),
               "share_zero_of_K": round(float((r == 0).mean()), 4),
               "tries_per_segment_damage": round(tries(r[dm]), 2), "tries_per_segment_no_damage": round(tries(r[~dm]), 2),
               "tries_per_prefix_median": round(float(np.median([sum(1.0 / max(x, 1.0 / (2 * K)) for x in row["rates"])
                                                                  for row in rows_out])), 1),
               "cpu_per_branch_ms_mean": round(float(np.mean([row["cpu_per_branch_ms"] for row in rows_out])), 4),
               "log10_whole_prefix_one_seed_median": float(np.median([row["log10_whole_prefix_one_seed"]
                                                                      for row in rows_out]))}
    json.dump({"summary": summary, "decisions": rows_out}, open(out, "w"), indent=1, sort_keys=True)
    print(json.dumps(summary, indent=1, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
