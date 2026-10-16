"""E2 §2: PLAY one arm's worlds at the checkpoint's PAD pin (the pin + the engine's gender draw pad) — Tier 1's
`play_worlds.py` on this bank: the pilot's own Phase 2 driver (`run_det_playouts.one_state`, imported unchanged). Per
state with a built world row: every legal action played to the END in each of the 8 worlds (world j on dice seed j), the
checkpoint GREEDY on both sides, with the searched side's value trace, plus the honest one-ply critic (world j, critic
seed 40 + j).

    run_e2.sh <cp>_pad 1 8 24h play_e2_worlds.py --arm F [--shard i --of W] [--limit N]   (--arm V:oracle for the check)

One fsync'd JSONL row per state in ``<archive>/<cp>/rows/<arm>/rows_<i>_of_<W>.jsonl``; resumable (done = an id in ANY
``rows_*`` file of the arm's directory). A state whose world row is an exclusion or a defect gets an ``ok: false`` row
carrying the builder's reason. No outcome is printed.
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2_common as E  # noqa: E402

E.add_paths()


def arm_dirname(arm: str) -> str:
    return ("V_" + E.arm_dir(arm[2:])) if arm.startswith("V:") else E.arm_dir(arm)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    cp = os.environ.get("VDP_CP")
    if cp not in E.CPS:
        raise SystemExit("run through run_e2.sh <cp>_pad")
    if os.environ.get("HIB_E2_KIND") != "pad" or not os.environ.get("HIB_E2_CHECKOUT_HEAD", "").startswith(E.CPS[cp]["pad"]):
        raise SystemExit("worlds are played at the PAD pin only (run_e2.sh <cp>_pad)")
    if a.arm not in E.ARMS and not a.arm.startswith("V:"):
        raise SystemExit(f"unknown arm {a.arm}")
    import main as _m

    if not str(Path(_m.__file__).resolve()).startswith(os.environ["HIB_E2_CHECKOUT_DIR"]):
        raise SystemExit(f"main imported from {_m.__file__}, not the pad pin")
    import common as C  # the pilot's: clock, threads, model loader
    from run_det_playouts import J as PILOT_J, one_state
    from run_playouts import Fwd

    if PILOT_J != E.J:
        raise SystemExit(f"the pilot driver plays {PILOT_J} worlds, E2 registers {E.J}")
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.reader import inference_globals
    from main.policy_spectrum.truth import cmd_index
    from utils.rust_env import ffi as F
    from utils.rust_env.successors import SearchCore, SuccessorsError

    bank = load_bank(E.bank_path(cp))
    dec = {d["id"]: d for d in bank.decisions}
    sub = arm_dirname(a.arm)
    worlds = E.read_dir(E.cp_dir(cp) / "worlds" / sub, "worlds_*.jsonl")
    od = E.cp_dir(cp) / "rows" / sub
    od.mkdir(parents=True, exist_ok=True)
    done = set(E.read_dir(od, "rows_*.jsonl"))
    ids = E.shard([i for i in worlds if i not in done], a.shard, a.of)[: a.limit]
    path = od / f"rows_{a.shard}_of_{a.of}.jsonl"
    print(f"[e2 play] {cp} {a.arm} shard {a.shard}/{a.of}: {len(done)} done, {len(ids)} to go", flush=True)
    off = C.clock_offset()
    lib = F.load(F.default_path("release"))
    cores = [SearchCore(lib=lib) for _ in range(E.J)]
    t0 = time.monotonic()
    with inference_globals(C.threads()):
        fwd = Fwd(C.load_model(cp))
        with open(path, "a") as f:
            for n_i, did in enumerate(ids):
                w = worlds[did]
                d = dec[did]
                if not w.get("ok"):
                    r = {"id": did, "ok": False, "excluded": not w.get("crash"), "defect": bool(w.get("crash")),
                         "error": "world build: " + str(w.get("error"))[:500]}
                elif len(w["teams"]) != E.J or any(t is None for t in w["teams"]):
                    r = {"id": did, "ok": False, "error": f"world row holds {len(w['teams'])} worlds, not {E.J}"}
                elif cmd_index(bank.battles[bank.battle_index[d["battle"]]], w["side"], d["n"]) != int(w["at"]):
                    r = {"id": did, "ok": False, "error": f"the pin's root command index != the builder's {w['at']}"}
                else:
                    try:
                        r = one_state(cores, fwd, bank, did, w, off)
                    except SuccessorsError as exc:
                        r = {"id": did, "ok": False, "error": str(exc)[:500]}
                        for k, core in enumerate(cores):       # a refused open leaves no state; reopen all
                            core.close()
                            cores[k] = SearchCore(lib=lib)
                r.update({"cp": cp, "arm": a.arm, "pin_head": os.environ["HIB_E2_CHECKOUT_HEAD"],
                          "world_head": w.get("head")})
                E.write_row(f, r)
                if not r.get("ok"):
                    print(f"[e2 play] REFUSED {did}: {str(r.get('error'))[:200]}", flush=True)
                el = time.monotonic() - t0
                print(f"[e2 play] {n_i + 1}/{len(ids)} {el / 60:.1f} min, eta "
                      f"{el / (n_i + 1) * (len(ids) - n_i - 1) / 60:.1f} min", flush=True)
    for core in cores:
        core.close()


if __name__ == "__main__":
    main()
