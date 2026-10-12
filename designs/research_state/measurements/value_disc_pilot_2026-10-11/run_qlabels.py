"""PHASE 3, step 2 (Amendment 7): ORACLE per-action labels on the Q head's TRAINING states (``make_qstates.py``):
every legal action × ``K_Q`` seeds played to the END from the TRUE state, CP greedy on both sides (Phase 1's
continuation exactly: ``run_playouts.playouts``, production stall, ``max_turns`` 999, the other side's open root
decision answered greedily). No value trace, no critic successors. One fsync'd JSONL row per state; resumable.

    run_at_pin.sh end2 1 6 12h shards.sh W --qlabels        (each shard single-threaded)

→ archive ``<cp>/qrows/shard_<i>_of_<W>.jsonl``: {id, ok, out: {action: "+-"}, end_turn, other_open, cost}.
These are TRAINING labels only: no evaluation state (the registered 600) is in the training bank.
"""
import argparse
import ctypes
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import numpy as np  # noqa: E402

from run_playouts import Fwd, playouts  # noqa: E402

K_Q = 2


def one_state(core, fwd, bank, dmap, did, off):
    from main.policy_spectrum.truth import cmd_index, to_log, turn_seeds
    from utils.rust_env.successors import SuccessorsError

    d = dmap[did]
    b = bank.battles[bank.battle_index[d["battle"]]]
    side = d["side"]
    at = cmd_index(b, side, d["n"])
    if b.commands[at][1] != d["played"]:
        raise RuntimeError(f"{did}: command {at} is {b.commands[at]!r}, the bank played {d['played']!r}")
    seeds = turn_seeds("q:" + did, K_Q)
    row = {"id": did, "at": at, "side": side, "turn": d["turn"], "k": K_Q}
    c0, r0 = C.cpu_seconds(), fwd.rows
    try:
        root, branches, _tr, st = playouts(core, fwd, to_log(b), at, side, seeds, off)
    except SuccessorsError as exc:
        row.update({"ok": False, "error": str(exc)[:500]})
        return row
    tokens = {str(k): v for k, v in sorted((int(k), v) for k, v in root["tokens"].items())}
    if tokens != d["tokens"]:
        row.update({"ok": False, "error": f"root tokens {tokens} != banked {d['tokens']}"})
        return row
    me = 0 if side == "p1" else 1
    out, ends = {}, {}
    for br in branches:
        a, s = str(int(br["action"])), int(br["seed"])
        e = br["end"]
        val = 0.0 if (e["truncated"] or e["winner"] is None) else (1.0 if int(e["winner"]) == me else -1.0)
        out.setdefault(a, [None] * K_Q)[s] = {1.0: "+", -1.0: "-", 0.0: "0"}[val]
        ends.setdefault(a, [None] * K_Q)[s] = int(e["turn"])
    for a in out:
        if any(x is None for x in out[a]):
            raise RuntimeError(f"{did}: action {a} is missing a seed")
        out[a] = "".join(out[a])
    row.update({"ok": True, "out": out, "end_turn": ends, "other_open": bool(root.get("other_open")),
                "cost": {"cpu_s": round(C.cpu_seconds() - c0, 3), "rows": fwd.rows - r0, **st}})
    return row


def group_states(cores, fwd, bank, dmap, dids, off):
    """``one_state``'s rows for several states at once, their playouts stepped in LOCKSTEP (one core each) with every
    pending row of every state in ONE forward per step: the same playouts (same request, seeds and greedy policy),
    batched for throughput (a forward costs about the same at 15 rows as at 120)."""
    from main.policy_spectrum.truth import cmd_index, to_log, turn_seeds
    from run_playouts import ACT, greedy
    from utils.rust_env.successors import _raise, production_stall

    st = []
    for core, did in zip(cores, dids):
        d = dmap[did]
        b = bank.battles[bank.battle_index[d["battle"]]]
        side = d["side"]
        at = cmd_index(b, side, d["n"])
        if b.commands[at][1] != d["played"]:
            raise RuntimeError(f"{did}: command {at} is {b.commands[at]!r}, the bank played {d['played']!r}")
        req = {"log": to_log(b), "at": int(at), "side": side, "actions": None, "seeds": turn_seeds("q:" + did, K_Q),
               "stall": production_stall(side), "max_turns": 999, "keep_cmds": False}
        row = {"id": did, "at": at, "side": side, "turn": d["turn"], "k": K_Q}
        try:
            root = core._cstr(core.lib.rust_env_playout_open, json.dumps(req).encode())
        except Exception as exc:      # a refused open: this state's row says so, the others go on
            row.update({"ok": False, "error": str(exc)[:500]})
            st.append({"row": row, "live": False, "bad": True})
            continue
        nb = int(root["branches"])
        cap = 2 * nb
        st.append({"row": row, "d": d, "root": root, "nb": nb, "cap": cap, "n": 0, "live": True, "bad": False,
                   "rows": np.empty((cap, core.obs_dim), dtype=np.float32),
                   "masks": np.empty((cap, ACT), dtype=np.uint8), "who": np.empty(cap, dtype=np.uint32),
                   "acts": np.zeros(cap, dtype=np.int32)})
    c0, r0 = C.cpu_seconds(), fwd.rows
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
        out = fwd(np.concatenate([s["rows"][:k] for s, k in pend]), np.concatenate([s["masks"][:k] for s, k in pend]))
        o = 0
        for s, k in pend:
            s["acts"][:k] = greedy(out[o:o + k], s["masks"][:k])
            s["n"] = k
            o += k
    cpu = (C.cpu_seconds() - c0) / max(1, len(dids))
    rows_ = []
    for core, s in zip(cores, st):
        row = s["row"]
        if s["bad"]:
            rows_.append(row)
            continue
        res = core._cstr(core.lib.rust_env_playout_results)
        tokens = {str(k): v for k, v in sorted((int(k), v) for k, v in s["root"]["tokens"].items())}
        if tokens != s["d"]["tokens"]:
            row.update({"ok": False, "error": f"root tokens {tokens} != banked {s['d']['tokens']}"})
            rows_.append(row)
            continue
        me = 0 if row["side"] == "p1" else 1
        outc, ends = {}, {}
        for br in res["branches"]:
            a_, sd = str(int(br["action"])), int(br["seed"])
            e = br["end"]
            val = 0.0 if (e["truncated"] or e["winner"] is None) else (1.0 if int(e["winner"]) == me else -1.0)
            outc.setdefault(a_, [None] * K_Q)[sd] = {1.0: "+", -1.0: "-", 0.0: "0"}[val]
            ends.setdefault(a_, [None] * K_Q)[sd] = int(e["turn"])
        if any(x is None for v in outc.values() for x in v):
            raise RuntimeError(f"{row['id']}: an action is missing a seed")
        row.update({"ok": True, "out": {a_: "".join(v) for a_, v in outc.items()}, "end_turn": ends,
                    "other_open": bool(s["root"].get("other_open")),
                    "cost": {"cpu_s_share": round(cpu, 3), "group": len(dids)}})
        rows_.append(row)
    return rows_


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--group", type=int, default=8, help="states stepped in lockstep per forward")
    a = ap.parse_args()
    c = C.cp()
    C.check_pin(c)
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.reader import inference_globals
    from utils.rust_env import ffi as F
    from utils.rust_env.successors import SearchCore, SuccessorsError

    od = C.out_dir(c)
    bank = load_bank(od / "qbank")
    dmap = {d["id"]: d for d in bank.decisions}
    ids = sorted(dmap)
    ids = [x for i, x in enumerate(ids) if i % a.of == a.shard][: a.limit]
    path = od / "qrows" / f"shard_{a.shard}_of_{a.of}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    done = {r["id"] for q in sorted(path.parent.glob("shard_*.jsonl")) for r in C.read_rows(q)}
    todo = [x for x in ids if x not in done]
    print(f"[qlabels] {c} shard {a.shard}/{a.of}: {len(done)} done, {len(todo)} to go", flush=True)
    off = C.clock_offset()
    lib = F.load(F.default_path("release"))
    G = max(1, a.group)
    cores = [SearchCore(lib=lib) for _ in range(G)]
    t0 = time.monotonic()
    with inference_globals(C.threads()):
        fwd = Fwd(C.load_model(c))
        with open(path, "a") as f:
            for i in range(0, len(todo), G):
                part = todo[i:i + G]
                try:
                    rows_ = group_states(cores[:len(part)], fwd, bank, dmap, part, off)
                    if any(not r.get("ok") for r in rows_):      # a refused open leaves no state: fresh cores
                        for k, core in enumerate(cores):
                            core.close()
                            cores[k] = SearchCore(lib=lib)
                except SuccessorsError:
                    # a step failure: reopen the cores and label this group one state at a time (each refusal its row)
                    for k, core in enumerate(cores):
                        core.close()
                        cores[k] = SearchCore(lib=lib)
                    rows_ = []
                    for did in part:
                        try:
                            rows_.append(one_state(cores[0], fwd, bank, dmap, did, off))
                        except SuccessorsError as exc:
                            rows_.append({"id": did, "ok": False, "error": str(exc)[:500]})
                            cores[0].close()
                            cores[0] = SearchCore(lib=lib)
                for r in rows_:
                    r["cp"] = c
                    C.write_row(f, r)
                    if not r.get("ok"):
                        print(f"[qlabels] REFUSED {r['id']}: {r.get('error')}", flush=True)
                n = min(i + G, len(todo))
                el = time.monotonic() - t0
                print(f"[qlabels] {n}/{len(todo)} {el / 60:.1f} min, eta {el / n * (len(todo) - n) / 60:.1f} min",
                      flush=True)
    for core in cores:
        core.close()


if __name__ == "__main__":
    main()
