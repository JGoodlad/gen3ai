"""STAGE 1c, PHASE B — THE PLAYOUTS (``PREREGISTRATION.md`` §6), run ONLY when phase A's registered bar is met. Every
played arm's 8 worlds (F8 / C8: the first 8 of their set; a resampled arm: the 8 proposals its importance resample
kept, each with ITS OWN rolls) are played with X26's 225M checkpoint at the PIN checkout (X26's pin ``9c033489`` + the
damage-roll control ``fc010d7d3``: stage 1's play recipe, ``play_s1.check_pin``) by the value-discrimination pilot's
own driver (``run_det_playouts.one_state``, imported unchanged): every legal action to the END in each world (world j
on dice seed j), greedy on both sides, and the honest one-ply critic (seeds 40 + j). A world's rolls are added to the
log the pilot rebuilds (``play_s1.RollsLog``, imported unchanged), and ``one_state`` REFUSES a world whose root tokens
differ from the banked root's.

    run_play.sh play_1c.py --identity                    # THE IDENTITY TOOTH (12 banked F worlds; must be 12 / 12)
    run_play.sh play_1c.py --one-chunk [--budget-min 40]  # the next states in the registered order, every arm

Exit codes of ``--one-chunk``: 0 = a chunk done; 3 = every state played in every arm (``B_DONE``); 4 = the registered
stop passed (``B_STOPPED``). Rows: ``<archive>/play/<arm>/rows.jsonl``, one fsync'd line per (state, arm).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import c1c_common as K  # noqa: E402

os.environ["WS1_ARCHIVE"] = str(K.ARCH / "ws1")
sys.path.insert(1, str(K.STAGE1))
import s1_common as E  # noqa: E402

for _p in (E.T1, E.BRC, E.PILOT):
    if str(_p) not in sys.path:
        sys.path.insert(2, str(_p))


def arm_world_row(arm: str, wrow: dict, row: dict) -> Optional[dict]:
    """The 8 worlds an arm plays (``c1c_common.play_indices``: a corpus arm from its STRICT weights, AMENDMENT A1), in
    the det format the pilot's driver reads, each with its own rolls; None when the arm has no valid worlds here."""
    idx = K.play_indices(row, arm)
    if idx is None:
        return None
    sets = K.arm_sets(arm)
    teams = K.arm_slice({s: wrow[s]["teams"] for s in sets}, arm)
    cmds = K.arm_slice({s: wrow[s]["cmds"] for s in sets}, arm)
    rolls = K.arm_slice({s: wrow[s]["rolls"] for s in sets}, arm)
    return {"id": wrow["id"], "side": wrow["side"], "at": wrow["at"], "teams": [teams[i] for i in idx],
            "cmds": [cmds[i] for i in idx], "rolls": [rolls[i] for i in idx], "src": idx}


def identity_ids() -> List[str]:
    return sorted(E.common_ids(), key=lambda x: hashlib.sha256((K.IDENTITY_SALT + x).encode()).hexdigest()
                  )[:K.N_IDENTITY]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--identity", action="store_true")
    ap.add_argument("--one-chunk", action="store_true")
    ap.add_argument("--budget-min", type=float, default=40.0)
    a = ap.parse_args()
    import play_s1 as PS                                 # stage 1's pin check and roll wrapper (unchanged)

    head = PS.check_pin()
    import common as C                                   # the pilot's: clock offset
    import run_det_playouts as RDP
    from run_playouts import Fwd

    if RDP.J != K.J:
        raise SystemExit(f"the pilot driver plays {RDP.J} worlds, stage 1c registers {K.J}")
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.reader import inference_globals
    from main.policy_spectrum.truth import cmd_index
    from utils.rust_env import ffi as F
    from utils.rust_env.successors import SearchCore, SuccessorsError

    bank = load_bank(E.MAT_ARCH / "bank")
    dec = {x["id"]: x for x in bank.decisions}
    jobs: List[tuple] = []
    if a.identity:
        banked = E.read_by_id(E.MAT_ARCH / "worlds" / "shared" / "F" / "worlds_0_of_1.jsonl", first=True)
        out_path = K.d("play", "identity") / "rows.jsonl"
        done_i = K.read_by_id(out_path)
        jobs = [("identity", did, banked[did]) for did in identity_ids() if did not in done_i]
        files = {"identity": open(out_path, "a")}
    else:
        ra = json.loads((HERE / "result_1c_a.json").read_text())
        if not ra.get("bar", {}).get("met"):
            raise SystemExit("phase A's bar was not met: no playout is registered")
        arms = list(ra["bar"]["play"])
        rows = K.read_by_id(K.ARCH / "rows" / "rows.jsonl")
        worlds = K.read_by_id(K.ARCH / "worlds" / "worlds.jsonl")
        order = [i for i in E.ordered_ids() if i in rows]
        paths = {arm: K.d("play", arm) / "rows.jsonl" for arm in arms}
        done = {arm: K.read_by_id(paths[arm]) for arm in arms}
        todo_states = [i for i in order if any(i not in done[arm] for arm in arms)]
        if not todo_states:
            (K.ARCH / "B_DONE").write_text(time.strftime("%F %T") + "\n")
            print("[1c play] every state played in every arm: B_DONE", flush=True)
            sys.exit(3)
        sb = K.ARCH / "STOP_B"
        if sb.exists() and time.time() >= float(sb.read_text().split()[0]):
            (K.ARCH / "B_STOPPED").write_text(time.strftime("%F %T") + "\n")
            sys.exit(4)
        for did in todo_states:
            for arm in arms:
                if did in done[arm]:
                    continue
                jobs.append((arm, did, arm_world_row(arm, worlds[did], rows[did])))
        files = {arm: open(p, "a") for arm, p in paths.items()}
    print(f"[1c play] {len(jobs)} (arm, state) jobs at {head[:8]}", flush=True)
    off = C.clock_offset()
    lib = F.load(F.default_path("release"))
    cores = [SearchCore(lib=lib) for _ in range(K.J)]
    rl = PS.RollsLog(RDP.world_log)
    RDP.world_log = rl
    t0 = time.monotonic()
    try:
        with inference_globals(1):
            model, ck = PS.load_model()
            fwd = Fwd(model)
            last_state = None
            for n_i, (arm, did, w) in enumerate(jobs):
                if a.one_chunk and did != last_state and (time.monotonic() - t0) / 60.0 >= a.budget_min:
                    break                                # chunks end at a STATE boundary
                last_state = did
                d = dec[did]
                if w is None:
                    r = {"id": did, "ok": False, "excluded": True, "error": "the arm has no valid worlds here"}
                elif cmd_index(bank.battles[bank.battle_index[d["battle"]]], w["side"], d["n"]) != int(w["at"]):
                    r = {"id": did, "ok": False, "defect": True, "error": "the pin's root command index differs"}
                else:
                    rl.arm(w)
                    try:
                        r = RDP.one_state(cores, fwd, bank, did, w, off)
                        rl.check_done()
                        if not r.get("ok"):
                            r["defect"] = True           # a refused world (root tokens differ): a defect
                    except SuccessorsError as exc:
                        rl.row = None
                        r = {"id": did, "ok": False, "defect": True, "error": str(exc)[:500]}
                        for k, core in enumerate(cores):
                            core.close()
                            cores[k] = SearchCore(lib=lib)
                    r["src"] = w.get("src")
                r.update({"arm": arm, "step": ck["step"], "pin_head": head})
                K.write_row(files[arm], r)
                if not r.get("ok"):
                    print(f"[1c play] {arm} not played {did}: {str(r.get('error'))[:200]}", flush=True)
                print(f"[1c play] {n_i + 1}/{len(jobs)} {(time.monotonic() - t0) / 60:.1f} min", flush=True)
    finally:
        for f in files.values():
            f.close()
        for core in cores:
            core.close()
    if a.identity:
        banked_rows = E.read_by_id(E.MAT_ARCH / "rows" / E.ANCHOR / "F" / "rows_0_of_1.jsonl")
        got = K.read_by_id(K.d("play", "identity") / "rows.jsonl")
        ids = identity_ids()
        same = sum(1 for did in ids if did in got and did in banked_rows
                   and all(json.dumps(got[did].get(k), sort_keys=True) == json.dumps(banked_rows[did].get(k),
                                                                                     sort_keys=True)
                           for k in ("out", "trace", "crit", "other_open")))
        res = {"states": len(ids), "identical": same, "ids_sha256": K.ids_sha(ids), "play_head": head}
        (K.d("play", "identity") / "identity.json").write_text(json.dumps(res, indent=1) + "\n")
        print(f"[1c identity] {same} / {len(ids)} identical", flush=True)
        if same != len(ids):
            raise SystemExit("THE IDENTITY TOOTH FAILED: this checkout does not play unarmed worlds as the pin did")
        return
    rows_done = {arm: K.read_by_id(K.ARCH / "play" / arm / "rows.jsonl") for arm in files}
    order = [i for i in E.ordered_ids() if i in K.read_by_id(K.ARCH / "rows" / "rows.jsonl")]
    if all(i in rows_done[arm] for i in order for arm in rows_done):
        (K.ARCH / "B_DONE").write_text(time.strftime("%F %T") + "\n")
        sys.exit(3)


if __name__ == "__main__":
    main()
