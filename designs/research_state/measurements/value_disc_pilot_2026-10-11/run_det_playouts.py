"""PHASE 2's DRIVER (README §8): per state, every legal action played to the END in each of the 16
DETERMINIZED worlds (``det_worlds.py``; world j with Phase 1's dice seed j), CP greedy on both sides,
with the searched side's value trace; plus the HONEST critic: the one-ply successors in worlds 0–7 on
Phase 1's critic seeds 40–47 (one each). One fsync'd JSONL row per state; resumable.

    run_at_pin.sh <cp> 2 8 24h shards.sh W --det        (shards.sh passes --det through)
    run_at_pin.sh <cp> 2 8 24h run_det_playouts.py --shard i --of W [--limit N]

The worlds of one state are stepped in LOCKSTEP (one core handle each) and every pending row of
every world goes through ONE forward per step (identical rows deduplicated, as Phase 1).
"""
import argparse
import ctypes
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from run_playouts import ACT, MOVE_START, Fwd, greedy  # noqa: E402

#: worlds played per state (Amendment 3: 8, the first 8 of the 16 built; K ≤ 8 on the det rows)
J = 8
J_CRIT = 8


def world_log(b, team_w, cmds_w, opp_key):
    """The bank battle's core input log with the opponent's team and the root prefix replaced."""
    from main.policy_spectrum.truth import to_log

    log = to_log(b)
    teams = list(log["teams"])
    teams[0 if opp_key == "p1" else 1] = team_w
    pre = [f"CHOOSE {s} {t}" for s, t in cmds_w]
    return dict(log, teams=teams, cmds=pre + log["cmds"][len(pre):])


def lockstep(cores, fwd, logs, at, side, seeds, off):
    """Open one playout per core (all actions, ONE seed each) and step them together."""
    from utils.rust_env.successors import _raise, production_stall

    me = 0 if side == "p1" else 1
    st = []
    for core, log, seed in zip(cores, logs, seeds):
        req = {"log": log, "at": int(at), "side": side, "actions": None, "seeds": [seed],
               "stall": production_stall(side), "max_turns": 999, "keep_cmds": False}
        root = core._cstr(core.lib.rust_env_playout_open, json.dumps(req).encode())
        nb = int(root["branches"])
        cap = 2 * nb
        st.append({"root": root, "nb": nb, "cap": cap, "n": 0, "live": True,
                   "rows": np.empty((cap, core.obs_dim), dtype=np.float32),
                   "masks": np.empty((cap, ACT), dtype=np.uint8), "who": np.empty(cap, dtype=np.uint32),
                   "acts": np.zeros(cap, dtype=np.int32), "tr": [{"t": [], "v": [], "s": ""} for _ in range(nb)]})
    steps = 0
    while True:
        pend = []
        for core, s in zip(cores, st):
            if not s["live"]:
                continue
            with core._lock:
                k = core.lib.rust_env_playout_step(core._h, s["acts"].ctypes.data, s["n"], s["rows"].ctypes.data,
                                                   s["masks"].ctypes.data, s["who"].ctypes.data, s["cap"])
                if k == ctypes.c_size_t(-1).value:
                    _raise(core.lib)
            if k == 0:
                s["live"] = False
                continue
            pend.append((s, int(k)))
        if not pend:
            break
        steps += 1
        out = fwd(np.concatenate([s["rows"][:k] for s, k in pend]), np.concatenate([s["masks"][:k] for s, k in pend]))
        o = 0
        for s, k in pend:
            oo = out[o:o + k]
            o += k
            a = greedy(oo, s["masks"][:k])
            mine = np.flatnonzero((s["who"][:k] % 2) == me)
            if len(mine):
                turns = C.decode_turn(s["rows"][mine], off)
                for jj, i in enumerate(mine):
                    t = s["tr"][int(s["who"][i] // 2)]
                    t["t"].append(int(turns[jj]))
                    t["v"].append(round(float(oo[i, ACT]), 5))
                    t["s"] += "0" if s["masks"][i, MOVE_START:].any() else "1"
            s["acts"][:k] = a
            s["n"] = k
    res = [core._cstr(core.lib.rust_env_playout_results) for core in cores]
    return st, res, steps


def one_state(cores, fwd, bank, did, wrow, off):
    from main.policy_spectrum.qhat import end_value, one_ply
    from main.policy_spectrum.truth import cmd_index, turn_seeds

    d = next(x for x in bank.decisions if x["id"] == did)
    b = bank.battles[bank.battle_index[d["battle"]]]
    side = d["side"]
    opp_key = "p2" if side == "p1" else "p1"
    at = cmd_index(b, side, d["n"])
    seeds = turn_seeds(did, C.CRIT_SEEDS[1])
    logs = [world_log(b, wrow["teams"][j], wrow["cmds"][j], opp_key) for j in range(J)]
    row = {"id": did, "at": at, "side": side, "turn": d["turn"], "n_worlds": J}
    c0, w0, r0 = C.cpu_seconds(), time.monotonic(), fwd.rows
    st, res, steps = lockstep(cores, fwd, logs, at, side, [seeds[j] for j in range(J)], off)
    me = 0 if side == "p1" else 1
    out, trace, oo = {}, {}, []
    for j, (s, r) in enumerate(zip(st, res)):
        tok = {str(k): v for k, v in sorted((int(k), v) for k, v in s["root"]["tokens"].items())}
        if tok != d["tokens"]:
            row.update({"ok": False, "error": f"world {j}: root tokens {tok} != banked {d['tokens']}"})
            return row
        oo.append(bool(s["root"].get("other_open")))
        for bi, br in enumerate(r["branches"]):
            a = str(int(br["action"]))
            e = br["end"]
            val = 0.0 if (e["truncated"] or e["winner"] is None) else (1.0 if int(e["winner"]) == me else -1.0)
            out.setdefault(a, [None] * J)[j] = {1.0: "+", -1.0: "-", 0.0: "0"}[val]
            trace.setdefault(a, [None] * J)[j] = s["tr"][bi]
    for a in out:
        if any(x is None for x in out[a]):
            raise RuntimeError(f"{did}: action {a} is missing a world")
        out[a] = "".join(out[a])
    row.update({"out": out, "trace": trace, "other_open": oo})
    cost_play = {"cpu_s": round(C.cpu_seconds() - c0, 3), "wall_s": round(time.monotonic() - w0, 3),
                 "rows": fwd.rows - r0, "playouts": sum(s["nb"] for s in st), "steps": steps}
    c1 = C.cpu_seconds()

    def greedy_answer(r_, m_, j_):
        return greedy(fwd(r_, m_.astype(np.uint8)), m_)

    crit = {}
    for j in range(J_CRIT):
        op = one_ply(cores[j], logs[j], at, side, [seeds[C.CRIT_SEEDS[0] + j]], greedy_answer)
        idx = [i for i, br in enumerate(op["branches"]) if br["row"] is not None]
        v = np.full(len(op["branches"]), np.nan)
        if idx:
            o = fwd(np.stack([op["branches"][i]["row"] for i in idx]),
                    np.stack([op["branches"][i]["mask"] for i in idx]).astype(np.uint8))
            v[idx] = o[:, ACT]
        for i, br in enumerate(op["branches"]):
            a = crit.setdefault(str(br["action"]), {"q": [None] * J_CRIT, "src": [None] * J_CRIT})
            if br["row"] is None:
                q = end_value(br["end"], side)
                a["q"][j], a["src"][j] = q, {1.0: "+", -1.0: "-", 0.0: "0"}[q]
            else:
                a["q"][j], a["src"][j] = round(2.0 * float(v[i]) - 1.0, 6), ("S" if br["switch_only"] else "V")
    for a in crit.values():
        a["src"] = "".join(a["src"])
    row["crit"] = crit
    row["cost"] = {"play": cost_play, "crit": {"cpu_s": round(C.cpu_seconds() - c1, 3)}}
    row["ok"] = True
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    c = C.cp()
    C.check_pin(c)
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.reader import inference_globals
    from utils.rust_env import ffi as F
    from utils.rust_env.successors import SearchCore, SuccessorsError

    od = C.out_dir(c)
    bank = load_bank(od / "bank")
    worlds = {}
    for q in sorted((od / "det").glob("worlds_*.jsonl")):
        for r in C.read_rows(q):
            worlds[r["id"]] = r
    ids = sorted(d["id"] for d in bank.decisions)
    ids = [x for i, x in enumerate(ids) if i % a.of == a.shard][: a.limit]
    path = od / "det" / f"rows_{a.shard}_of_{a.of}.jsonl"
    done = {r["id"] for q in sorted(path.parent.glob("rows_*.jsonl")) for r in C.read_rows(q)}
    todo = [x for x in ids if x not in done]
    print(f"[det] {c} shard {a.shard}/{a.of}: {len(done)} done, {len(todo)} to go", flush=True)
    off = C.clock_offset()
    lib = F.load(F.default_path("release"))
    cores = [SearchCore(lib=lib) for _ in range(J)]
    t0 = time.monotonic()
    with inference_globals(C.threads()):
        fwd = Fwd(C.load_model(c))
        with open(path, "a") as f:
            for i, did in enumerate(todo):
                w = worlds.get(did)
                if w is None or not w.get("ok") or any(t is None for t in w["teams"]):
                    r = {"id": did, "ok": False, "error": "no complete world set (det_worlds)"}
                else:
                    try:
                        r = one_state(cores, fwd, bank, did, w, off)
                    except SuccessorsError as exc:
                        r = {"id": did, "ok": False, "error": str(exc)[:500]}
                        for k, core in enumerate(cores):       # a refused open leaves no state; reopen all
                            core.close()
                            cores[k] = SearchCore(lib=lib)
                r["cp"] = c
                C.write_row(f, r)
                if not r.get("ok"):
                    print(f"[det] REFUSED {did}: {r.get('error')}", flush=True)
                el = time.monotonic() - t0
                print(f"[det] {i + 1}/{len(todo)} {el / 60:.1f} min, eta {el / (i + 1) * (len(todo) - i - 1) / 60:.1f} min",
                      flush=True)
    for core in cores:
        core.close()


if __name__ == "__main__":
    main()
