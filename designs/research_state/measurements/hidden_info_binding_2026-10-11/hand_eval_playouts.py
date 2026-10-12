"""THE HAND-EVALUATOR DRIVER (design doc §6.5 / R-EVAL §12.6): truncated re-plays that record, at every
searched-side decision, BOTH Foul Play's hand evaluation of the row (`hand_eval_pin` → the HEAD source,
`src/main/hidden_info_study/hand_eval.py`, by file path) and the checkpoint critic's V.

    run_at_pin.sh <cp> 2 8 24h hand_eval_playouts.py --world O [--shard i --of W] [--limit N]
    run_at_pin.sh <cp> 2 8 24h hand_eval_playouts.py --world P                # the pilot's Phase 2 worlds
    run_at_pin.sh <cp> 2 8 24h hand_eval_playouts.py --world F --worlds-dir <dir with worlds_*.jsonl>

Per registered state (the checkpoint's bank, `hib_common.bank_dir`) and EVERY legal action:

- **HE-TRUNC** (d ∈ {1, 2, 4}): 8 playouts per action, CP greedy on both sides (the pilot's continuation: production
  stall, the decision's command index as the root, the other side's open root decision answered greedily),
  TRUNCATED once the battle reaches turn T + 5 (T = the root's turn; the core's ``max_turns``), so every searched-side
  decision with turn ≤ T + 4 is seen. Oracle world **O**: the TRUE battle, Phase 1's leaf seeds 0..7. Fair worlds
  **P** / **F**: archived world j (``worlds_*.jsonl`` rows, the first 8 built) with dice seed j (Phase 1's leaf seed j),
  stepped in lockstep as ``run_det_playouts`` does. Each branch's TRACE is every searched-side decision's turn ``t``,
  the critic's raw V ``v`` (win probability, 5 decimals, as Phase 1), the hand eval ``h`` (3 decimals) and a
  switch-only flag ``s``; its END is ``+`` / ``-`` / ``0`` (a game end, from the searched side) or ``T`` (our
  truncation), with the turn. The scorer reads X_d = the first trace entry with t ≥ T + d, else the end.
- **HE-1PLY**: the one-ply successors (``qhat.one_ply``) on Phase 1's critic seeds 40..47 (oracle: all eight in the
  true battle; fair: world j with seed 40 + j), each captured row's V and hand eval, or the end.
- ``h_root``: the hand eval of the ROOT row (identical in every gate-passed world), so a reader can apply Foul Play's
  own MCTS leaf scale ``sigmoid(0.0125 · (h − h_root))`` with terminals 1 / 0 (``hand_eval.foul_play_leaf``).

OUTPUTS are durable (one fsync'd JSONL row per state), resumable (a state in any ``rows_*`` file of its directory is
done) and CODED: the directory is ``<out-root>/<code>/`` with ``code = "H" + sha256(hib_arm_v1: + "HE@<world>")[:8]``
(``hand_eval_arm_key.json``); a row names its code, never its world. Default out-root:
``~/gen3ai_archive/hidden_info_binding/<cp>/hand_eval/``.

``--identity`` (a VALIDITY check, no outcome read): compares each branch's (t, v) trace with the pilot's archived
trace of the same (state, action, seed / world) prefix — the truncated re-play must reproduce it — and prints only
the match counts. Nothing here computes or prints an IMP, an accuracy or a comparison of scorers.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import hib_common as HC  # noqa: E402

HC.pin_paths()
import common as C  # noqa: E402  (the pilot's)
import hand_eval_pin as HP  # noqa: E402
from run_det_playouts import world_log  # noqa: E402
from run_playouts import ACT, MOVE_START, Fwd, greedy  # noqa: E402

D_MAX = 4                      # the deepest truncation read (TRUNC-4); the core truncates at T + D_MAX + 1
J = 8                          # playouts per action (seeds 0..7 / worlds 0..7)
WORLDS = ("O", "P", "F")
KEY_PATH = HERE / "hand_eval_arm_key.json"


def world_code(world: str) -> str:
    return "H" + hashlib.sha256((HC.ARM_SALT + f"HE@{world}").encode()).hexdigest()[:8]


def write_key() -> Dict[str, str]:
    key = {world_code(w): f"HE@{w}" for w in WORLDS}
    KEY_PATH.write_text(json.dumps({"salt": HC.ARM_SALT, "scheme": 'code = "H" + sha256(salt + label)[:8]',
                                    "codes": key}, indent=1, sort_keys=True) + "\n")
    return key


def _end_code(e: dict, me: int) -> str:
    if e["truncated"]:
        return "T"
    if e["winner"] is None:
        return "0"
    return "+" if int(e["winner"]) == me else "-"


def play(cores, fwd, logs, at, side, seed_lists, max_turns, off, he, L, tally):
    """Open one playout per core (all actions × its seeds, truncated at ``max_turns``) and step them in lockstep;
    every pending row of every core through ONE forward per step. Returns per core (root, results, traces)."""
    from utils.rust_env.successors import _raise, production_stall

    me = 0 if side == "p1" else 1
    st = []
    for core, log, seeds in zip(cores, logs, seed_lists):
        req = {"log": log, "at": int(at), "side": side, "actions": None, "seeds": list(seeds),
               "stall": production_stall(side), "max_turns": int(max_turns), "keep_cmds": False}
        root = core._cstr(core.lib.rust_env_playout_open, json.dumps(req).encode())
        nb = int(root["branches"])
        cap = 2 * nb
        st.append({"root": root, "nb": nb, "cap": cap, "n": 0, "live": True,
                   "rows": np.empty((cap, core.obs_dim), dtype=np.float32),
                   "masks": np.empty((cap, ACT), dtype=np.uint8), "who": np.empty(cap, dtype=np.uint32),
                   "acts": np.zeros(cap, dtype=np.int32),
                   "tr": [{"t": [], "v": [], "h": [], "s": ""} for _ in range(nb)]})
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
                t0 = C.cpu_seconds()
                for jj, i in enumerate(mine):
                    t = s["tr"][int(s["who"][i] // 2)]
                    t["t"].append(int(turns[jj]))
                    t["v"].append(round(float(oo[i, ACT]), 5))
                    t["h"].append(round(he.hand_eval(s["rows"][i], L), 3))
                    t["s"] += "0" if s["masks"][i, MOVE_START:].any() else "1"
                tally["he_cpu"] += C.cpu_seconds() - t0
                tally["he_rows"] += len(mine)
            s["acts"][:k] = a
            s["n"] = k
    res = [core._cstr(core.lib.rust_env_playout_results) for core in cores]
    return st, res, steps


def one_state(world, cores, fwd, bank, did, wrow, root_row, off, he, L) -> dict:
    from main.policy_spectrum.qhat import end_value, one_ply
    from main.policy_spectrum.truth import cmd_index, to_log, turn_seeds

    d = next(x for x in bank.decisions if x["id"] == did)
    b = bank.battles[bank.battle_index[d["battle"]]]
    side = d["side"]
    me = 0 if side == "p1" else 1
    opp_key = "p2" if side == "p1" else "p1"
    at = cmd_index(b, side, d["n"])
    seeds = turn_seeds(did, C.CRIT_SEEDS[1])
    T = int(d["turn"])
    max_turns = T + D_MAX + 1
    if world == "O":
        logs = [to_log(b)]
        seed_lists = [seeds[:J]]
        crit_calls = [(0, seeds[C.CRIT_SEEDS[0]:C.CRIT_SEEDS[0] + J])]
    else:
        logs = [world_log(b, wrow["teams"][j], wrow["cmds"][j], opp_key) for j in range(J)]
        seed_lists = [[seeds[j]] for j in range(J)]
        crit_calls = [(j, [seeds[C.CRIT_SEEDS[0] + j]]) for j in range(J)]
    row = {"id": did, "code": world_code(world), "at": at, "side": side, "turn": T, "max_turns": max_turns,
           "he_sha256": HP.source_sha(), "n": J}
    tally = {"he_cpu": 0.0, "he_rows": 0}
    c0, w0, r0 = C.cpu_seconds(), time.monotonic(), fwd.rows
    t0 = C.cpu_seconds()
    row["h_root"] = round(he.hand_eval(root_row, L), 3)
    tally["he_cpu"] += C.cpu_seconds() - t0
    st, res, steps = play(cores[:len(logs)], fwd, logs, at, side, seed_lists, max_turns, off, he, L, tally)
    trace: Dict[str, List[Optional[dict]]] = {}
    end: Dict[str, List[Optional[list]]] = {}
    oo = []
    for j, (s, r) in enumerate(zip(st, res)):
        tok = {str(k): v for k, v in sorted((int(k), v) for k, v in s["root"]["tokens"].items())}
        if tok != d["tokens"]:
            return dict(row, ok=False, error=f"core {j}: root tokens {tok} != banked {d['tokens']}")
        oo.append(bool(s["root"].get("other_open")))
        for bi, br in enumerate(r["branches"]):
            a = str(int(br["action"]))
            slot = int(br["seed"]) if world == "O" else j
            trace.setdefault(a, [None] * J)[slot] = s["tr"][bi]
            end.setdefault(a, [None] * J)[slot] = [_end_code(br["end"], me), int(br["end"]["turn"])]
    for a in trace:
        if any(x is None for x in trace[a]):
            raise RuntimeError(f"{did}: action {a} is missing a playout")
    row.update({"trace": trace, "end": end, "other_open": oo})
    cost_play = {"cpu_s": round(C.cpu_seconds() - c0, 3), "wall_s": round(time.monotonic() - w0, 3),
                 "fwd_rows": fwd.rows - r0, "playouts": sum(s["nb"] for s in st), "steps": steps}

    # HE-1PLY / CRITIC: the one-ply successors on seeds 40..47
    c1 = C.cpu_seconds()

    def greedy_answer(r_, m_, j_):
        return greedy(fwd(r_, m_.astype(np.uint8)), m_)

    op1: Dict[str, dict] = {}
    for j, sd in crit_calls:
        op = one_ply(cores[j], logs[j], at, side, sd, greedy_answer)
        idx = [i for i, br in enumerate(op["branches"]) if br["row"] is not None]
        v = np.full(len(op["branches"]), np.nan)
        if idx:
            o = fwd(np.stack([op["branches"][i]["row"] for i in idx]),
                    np.stack([op["branches"][i]["mask"] for i in idx]).astype(np.uint8))
            v[idx] = o[:, ACT]
        for i, br in enumerate(op["branches"]):
            a = op1.setdefault(str(br["action"]), {"v": [None] * J, "h": [None] * J, "src": [None] * J})
            slot = int(br["seed"]) if world == "O" else j
            if br["row"] is None:
                a["src"][slot] = {1.0: "+", -1.0: "-", 0.0: "0"}[end_value(br["end"], side)]
            else:
                t0 = C.cpu_seconds()
                a["h"][slot] = round(he.hand_eval(np.asarray(br["row"], dtype=np.float32), L), 3)
                tally["he_cpu"] += C.cpu_seconds() - t0
                tally["he_rows"] += 1
                a["v"][slot] = round(float(v[i]), 6)
                a["src"][slot] = "S" if br["switch_only"] else "V"
    for a in op1.values():
        if any(x is None for x in a["src"]):
            raise RuntimeError(f"{did}: a one-ply slot is missing")
        a["src"] = "".join(a["src"])
    row["one_ply"] = op1
    row["cost"] = {"play": cost_play, "one_ply": {"cpu_s": round(C.cpu_seconds() - c1, 3)},
                   "he": {"cpu_s": round(tally["he_cpu"], 3), "rows": tally["he_rows"]},
                   "total_cpu_s": round(C.cpu_seconds() - c0, 3)}
    row["ok"] = True
    return row


def _archived_traces(world: str, cp: str, worlds_dir: Path) -> Dict[str, dict]:
    """The pilot's own rows for the identity check: Phase 1 (O) or the worlds' Phase 2 rows (P / F)."""
    src = HC.phase1_rows_dir(cp).glob("shard_*.jsonl") if world == "O" else worlds_dir.glob("rows_*.jsonl")
    out: Dict[str, dict] = {}
    for q in sorted(src):
        for r in C.read_rows(q):
            if r.get("ok"):
                out[r["id"]] = r
    return out


def identity(row: dict, ref: dict, world: str) -> Dict[str, int]:
    """Branch-level reproduction of the archived (t, v) trace prefix and of the one-ply V (no outcome read)."""
    n = {"branches": 0, "trace_match": 0, "one_ply": 0, "one_ply_match": 0}
    for a, trs in row["trace"].items():
        for j, tr in enumerate(trs):
            old = ref["trace"][a][j]
            n["branches"] += 1
            k = len(tr["t"])
            if old["t"][:k] == tr["t"] and all(abs(x - y) <= 2e-5 for x, y in zip(old["v"][:k], tr["v"])):
                n["trace_match"] += 1
    for a, op in row["one_ply"].items():
        q = ref.get("crit", {}).get(a, {}).get("q")
        if q is None:
            continue
        for j, v in enumerate(op["v"]):
            if v is None:
                continue
            n["one_ply"] += 1
            if q[j] is not None and abs((2.0 * v - 1.0) - q[j]) <= 2e-5:
                n["one_ply_match"] += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", choices=WORLDS, required=True)
    ap.add_argument("--worlds-dir", default=None, help="P: default the pilot's <states>/det; F: required")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--identity", action="store_true")
    a = ap.parse_args()
    cp = HC.hib_cp()
    HC.check_pin(cp)
    write_key()
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.reader import inference_globals
    from utils.rust_env import ffi as F
    from utils.rust_env.successors import SearchCore, SuccessorsError

    he = HP.load()
    L = HP.layout()
    worlds_dir: Optional[Path] = None
    worlds: Dict[str, dict] = {}
    if a.world != "O":
        if a.worlds_dir:
            worlds_dir = Path(a.worlds_dir)
        elif a.world == "P":
            worlds_dir = HC.PILOT_ARCHIVE / HC.CPS[cp]["states"] / "det"
        else:
            raise SystemExit("--world F needs --worlds-dir")
        for q in sorted(worlds_dir.glob("worlds_*.jsonl")):
            for r in C.read_rows(q):
                worlds[r["id"]] = r
        if not worlds:
            raise SystemExit(f"no worlds_*.jsonl rows under {worlds_dir}")
    code = world_code(a.world)
    out_root = Path(a.out_root) if a.out_root else HC.out_dir(cp) / "hand_eval"
    od = out_root / code
    od.mkdir(parents=True, exist_ok=True)
    path = od / f"rows_{a.shard}_of_{a.of}.jsonl"
    bank = load_bank(HC.bank_dir(cp))
    ids = sorted(d["id"] for d in bank.decisions)
    ids = [x for i, x in enumerate(ids) if i % a.of == a.shard][: a.limit]
    done = {r["id"] for q in sorted(od.glob("rows_*.jsonl")) for r in C.read_rows(q)}
    todo = [x for x in ids if x not in done]
    print(f"[hand_eval] {cp} code {code} shard {a.shard}/{a.of}: {len(done)} done, {len(todo)} to go -> {od}",
          flush=True)
    if not todo:
        return
    t_root = C.cpu_seconds()
    roots = HC.roots(bank, todo)
    print(f"[hand_eval] root rows replayed: {len(roots)} states, {C.cpu_seconds() - t_root:.1f} CPU-s", flush=True)
    ref = _archived_traces(a.world, cp, worlds_dir or Path()) if a.identity else {}
    off = C.clock_offset()
    lib = F.load(F.default_path("release"))
    n_cores = 1 if a.world == "O" else J
    cores = [SearchCore(lib=lib) for _ in range(n_cores)]
    ident = {"branches": 0, "trace_match": 0, "one_ply": 0, "one_ply_match": 0}
    t0 = time.monotonic()
    cpu = []
    with inference_globals(C.threads()):
        fwd = Fwd(HC.load_model(cp))
        with open(path, "a") as f:
            for i, did in enumerate(todo):
                w = worlds.get(did)
                if a.world != "O" and (w is None or not w.get("ok") or any(t is None for t in w["teams"][:J])):
                    r = {"id": did, "code": code, "ok": False, "error": f"no complete set of the first {J} worlds"}
                else:
                    try:
                        r = one_state(a.world, cores, fwd, bank, did, w, roots[did]["row"], off, he, L)
                    except (SuccessorsError, he.HandEvalError) as exc:
                        r = {"id": did, "code": code, "ok": False, "error": f"{type(exc).__name__}: {str(exc)[:500]}"}
                        for k, core in enumerate(cores):      # a refused open leaves no state; reopen all
                            core.close()
                            cores[k] = SearchCore(lib=lib)
                r["cp"] = cp
                C.write_row(f, r)
                if r.get("ok"):
                    cpu.append(r["cost"]["total_cpu_s"])
                    if a.identity and did in ref:
                        for k, v in identity(r, ref[did], a.world).items():
                            ident[k] += v
                else:
                    print(f"[hand_eval] REFUSED {did}: {r.get('error')}", flush=True)
                el = time.monotonic() - t0
                cost = r.get("cost", {})
                print(f"[hand_eval] {i + 1}/{len(todo)} {el / 60:.1f} min, eta {el / (i + 1) * (len(todo) - i - 1) / 60:.1f}"
                      f" min; this state: {cost.get('total_cpu_s')} CPU-s (play {cost.get('play', {}).get('cpu_s')},"
                      f" one-ply {cost.get('one_ply', {}).get('cpu_s')}, hand eval {cost.get('he', {}).get('cpu_s')}"
                      f" over {cost.get('he', {}).get('rows')} rows; {cost.get('play', {}).get('playouts')} playouts)",
                      flush=True)
    for core in cores:
        core.close()
    if cpu:
        print(json.dumps({"states": len(cpu), "cpu_s_per_state": round(float(np.mean(cpu)), 2),
                          "cpu_s_per_state_max": round(float(np.max(cpu)), 2)}), flush=True)
    if a.identity:
        print("[hand_eval] identity vs the archived pilot rows:", json.dumps(ident), flush=True)


if __name__ == "__main__":
    main()
