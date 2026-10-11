"""THE PLAYOUT DRIVER (README §3, B1): per sampled state, every legal action × the 40 shared playout
seeds played to the END by CP greedy on both sides (oracle world), with the searched side's VALUE
TRACE (CP's V at every one of its decisions, with the decision's turn and a switch-only flag), plus
the critic's one-ply successors on 8 separate seeds. One fsync'd JSONL row per state; resumable.

    run_at_pin.sh <cp> 2 6 24h run_playouts.py --shard i --of W [--limit N] [--cost-only]

The continuation is the same as ``main.policy_spectrum.truth.branch_turn`` (``play_out``'s request:
``stall`` = production, ``max_turns`` 999, the decision's command index as the root, the other side's
open root decision answered by CP's greedy policy identically across siblings). The loop is
``play_out``'s own, re-written here only to see each pending row's ``who`` and V.

Identical pending rows of one step (same obs bytes and mask) share ONE forward (exact: a row's
forward does not depend on the others in the batch up to float reassociation).
``--cost-only`` prints the cost and writes NOTHING (the size-rule throughput pilot never reads an
outcome).
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

ACT = 11
MOVE_START = 6


def uniq(rows: np.ndarray, masks: np.ndarray):
    key = np.concatenate([rows.view(np.uint8).reshape(len(rows), -1), masks.astype(np.uint8)], axis=1)
    v = np.ascontiguousarray(key).view(np.dtype((np.void, key.shape[1]))).ravel()
    _, first, inv = np.unique(v, return_index=True, return_inverse=True)
    return first, inv.ravel()


class Fwd:
    def __init__(self, model):
        from main.policy_spectrum.qhat import logits_and_values

        self.f = lambda r, m: logits_and_values(model, r, m, batch=1024)
        self.rows = 0
        self.unique = 0
        self.cpu = 0.0

    def __call__(self, rows, masks):
        first, inv = uniq(rows, masks)
        t = C.cpu_seconds()
        out = self.f(np.ascontiguousarray(rows[first]), np.ascontiguousarray(masks[first]))
        self.cpu += C.cpu_seconds() - t
        self.rows += len(rows)
        self.unique += len(first)
        return out[inv]


def greedy(out, masks):
    lg = np.where(masks.astype(bool), out[:, :ACT].astype(np.float64), -np.inf)
    return lg.argmax(axis=1).astype(np.int32)


def playouts(core, fwd, log, at, side, seeds, off):
    from utils.rust_env.successors import _raise, production_stall

    me = 0 if side == "p1" else 1
    req = {"log": log, "at": int(at), "side": side, "actions": None, "seeds": list(seeds),
           "stall": production_stall(side), "max_turns": 999, "keep_cmds": False}
    root = core._cstr(core.lib.rust_env_playout_open, json.dumps(req).encode())
    nb = int(root["branches"])
    cap = 2 * nb
    rows = np.empty((cap, core.obs_dim), dtype=np.float32)
    masks = np.empty((cap, ACT), dtype=np.uint8)
    who = np.empty(cap, dtype=np.uint32)
    acts = np.zeros(cap, dtype=np.int32)
    tr = [{"t": [], "v": [], "s": ""} for _ in range(nb)]
    n, steps, sim = 0, 0, 0.0
    while True:
        ts = time.perf_counter()
        with core._lock:
            k = core.lib.rust_env_playout_step(core._h, acts.ctypes.data, n, rows.ctypes.data,
                                               masks.ctypes.data, who.ctypes.data, cap)
            if k == ctypes.c_size_t(-1).value:
                _raise(core.lib)
        sim += time.perf_counter() - ts
        if k == 0:
            break
        steps += 1
        out = fwd(rows[:k], masks[:k])
        a = greedy(out, masks[:k])
        mine = np.flatnonzero((who[:k] % 2) == me)
        if len(mine):
            turns = C.decode_turn(rows[mine], off)
            for j, i in enumerate(mine):
                b = int(who[i] // 2)
                tr[b]["t"].append(int(turns[j]))
                tr[b]["v"].append(round(float(out[i, ACT]), 5))
                tr[b]["s"] += "0" if masks[i, MOVE_START:].any() else "1"
        acts[:k] = a
        n = k
    res = core._cstr(core.lib.rust_env_playout_results)
    return root, res["branches"], tr, {"steps": steps, "sim_s": round(sim, 3)}


def one_state(core, fwd, bank, did, off):
    from main.policy_spectrum.qhat import end_value, one_ply
    from main.policy_spectrum.truth import cmd_index, to_log, turn_seeds
    from utils.rust_env.successors import SuccessorsError

    d = next(x for x in bank.decisions if x["id"] == did)
    b = bank.battles[bank.battle_index[d["battle"]]]
    side = d["side"]
    at = cmd_index(b, side, d["n"])
    if b.commands[at][1] != d["played"]:
        raise RuntimeError(f"{did}: command {at} is {b.commands[at]!r}, the bank played {d['played']!r}")
    seeds = turn_seeds(did, C.CRIT_SEEDS[1])
    row = {"id": did, "at": at, "side": side, "turn": d["turn"], "n_seeds": C.N_PLAY,
           "crit_seeds": list(C.CRIT_SEEDS)}
    log = to_log(b)
    c0, w0, r0 = C.cpu_seconds(), time.monotonic(), fwd.rows
    try:
        root, branches, tr, st = playouts(core, fwd, log, at, side, seeds[:C.N_PLAY], off)
    except SuccessorsError as exc:
        row.update({"ok": False, "error": str(exc)[:500]})
        return row
    tokens = {str(k): v for k, v in sorted((int(k), v) for k, v in root["tokens"].items())}
    if tokens != d["tokens"]:
        row.update({"ok": False, "error": f"root tokens {tokens} != banked {d['tokens']}"})
        return row
    row["other_open"] = bool(root.get("other_open"))
    out, ends, trace = {}, {}, {}
    me = 0 if side == "p1" else 1
    for bi, br in enumerate(branches):
        a, s = str(int(br["action"])), int(br["seed"])
        e = br["end"]
        val = 0.0 if (e["truncated"] or e["winner"] is None) else (1.0 if int(e["winner"]) == me else -1.0)
        out.setdefault(a, [None] * C.N_PLAY)[s] = {1.0: "+", -1.0: "-", 0.0: "0"}[val]
        ends.setdefault(a, [None] * C.N_PLAY)[s] = int(e["turn"])
        trace.setdefault(a, [None] * C.N_PLAY)[s] = tr[bi]
    for a in out:
        if any(x is None for x in out[a]):
            raise RuntimeError(f"{did}: action {a} is missing a seed")
        out[a] = "".join(out[a])
    row.update({"out": out, "end_turn": ends, "trace": trace})
    cost_play = {"cpu_s": round(C.cpu_seconds() - c0, 3), "wall_s": round(time.monotonic() - w0, 3),
                 "rows": fwd.rows - r0, "playouts": len(branches), **st}
    # the critic's one-ply successors on seeds 40..47 (qhat.one_ply; the opponent answered greedily)
    c1, w1 = C.cpu_seconds(), time.monotonic()

    def greedy_answer(r, m, j):
        return greedy(fwd(r, m.astype(np.uint8)), m)

    try:
        op = one_ply(core, log, at, side, seeds[C.CRIT_SEEDS[0]:C.CRIT_SEEDS[1]], greedy_answer)
    except SuccessorsError as exc:
        row.update({"ok": False, "error": "one_ply: " + str(exc)[:500]})
        return row
    if op["tokens"] != d["tokens"]:
        row.update({"ok": False, "error": "one_ply root tokens differ"})
        return row
    idx = [i for i, br in enumerate(op["branches"]) if br["row"] is not None]
    v = np.full(len(op["branches"]), np.nan)
    if idx:
        o = fwd(np.stack([op["branches"][i]["row"] for i in idx]),
                np.stack([op["branches"][i]["mask"] for i in idx]).astype(np.uint8))
        v[idx] = o[:, ACT]
    crit = {}
    for i, br in enumerate(op["branches"]):
        a = crit.setdefault(str(br["action"]), {"q": [None] * 8, "src": [None] * 8})
        s = int(br["seed"])
        if br["row"] is None:
            q = end_value(br["end"], side)
            a["q"][s], a["src"][s] = q, {1.0: "+", -1.0: "-", 0.0: "0"}[q]
        else:
            a["q"][s], a["src"][s] = round(2.0 * float(v[i]) - 1.0, 6), ("S" if br["switch_only"] else "V")
    for a in crit.values():
        a["src"] = "".join(a["src"])
    row["crit"] = crit
    row["cost"] = {"play": cost_play, "crit": {"cpu_s": round(C.cpu_seconds() - c1, 3),
                                               "wall_s": round(time.monotonic() - w1, 3)},
                   "fwd_rows": fwd.rows, "fwd_unique": fwd.unique}
    row["ok"] = True
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--cost-only", action="store_true")
    a = ap.parse_args()
    c = C.cp()
    C.check_pin(c)
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.reader import inference_globals
    from utils.rust_env import ffi as F
    from utils.rust_env.successors import SearchCore

    od = C.out_dir(c)
    bank = load_bank(od / "bank")
    ids = sorted(d["id"] for d in bank.decisions)
    ids = [x for i, x in enumerate(ids) if i % a.of == a.shard]
    if a.limit:
        ids = ids[:a.limit]
    path = od / "rows" / f"shard_{a.shard}_of_{a.of}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    # done = a row in ANY shard file (so a relaunch may change the shard count)
    done = set() if a.cost_only else {r["id"] for q in sorted(path.parent.glob("shard_*.jsonl"))
                                      for r in C.read_rows(q)}
    todo = [x for x in ids if x not in done]
    print(f"[playouts] {c} shard {a.shard}/{a.of}: {len(done)} done, {len(todo)} to go", flush=True)
    off = C.clock_offset()
    lib = F.load(F.default_path("release"))
    core = SearchCore(lib=lib)
    t0 = time.monotonic()
    tot = {"cpu": 0.0, "play": 0, "rows": 0}
    with inference_globals(C.threads()):
        model = C.load_model(c)
        fwd = Fwd(model)
        f = None if a.cost_only else open(path, "a")
        try:
            for i, did in enumerate(todo):
                r = one_state(core, fwd, bank, did, off)
                r["cp"] = c
                if a.cost_only:
                    if r.get("ok"):
                        tot["cpu"] += r["cost"]["play"]["cpu_s"] + r["cost"]["crit"]["cpu_s"]
                        tot["play"] += r["cost"]["play"]["playouts"]
                        tot["rows"] += r["cost"]["play"]["rows"]
                    print(json.dumps({"id": did, "ok": r.get("ok"), "error": r.get("error"),
                                      "cost": r.get("cost")}), flush=True)
                    continue
                C.write_row(f, r)
                if not r.get("ok"):
                    print(f"[playouts] REFUSED {did}: {r.get('error')}", flush=True)
                el = time.monotonic() - t0
                print(f"[playouts] {i + 1}/{len(todo)} {el / 60:.1f} min, eta "
                      f"{el / (i + 1) * (len(todo) - i - 1) / 60:.1f} min, fwd unique "
                      f"{fwd.unique}/{fwd.rows}", flush=True)
        finally:
            if f is not None:
                f.close()
            core.close()
    if a.cost_only and tot["play"]:
        print(json.dumps({"states": len(todo), "cpu_s_per_playout": round(tot["cpu"] / tot["play"], 4),
                          "rows_per_playout": round(tot["rows"] / tot["play"], 2),
                          "playouts_per_state": round(tot["play"] / len(todo), 1),
                          "cpu_s_per_state": round(tot["cpu"] / len(todo), 1),
                          "dedup_unique_share": round(fwd.unique / max(1, fwd.rows), 3)}), flush=True)


if __name__ == "__main__":
    main()
