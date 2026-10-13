"""TIER 1, step 2: PLAY one arm's worlds at the checkpoint's PAD pin (the pin + the engine's gender draw pad).

    run_pin.sh <cp>_pad 2 8 24h play_worlds.py --code Tnn [--states registered|supp|both] [--shard i --of W] [--limit N]

Per state with a built world row (`build_worlds.py`): every legal action played to the END in each of the 8 worlds
(world j with the pilot's dice seed j), the checkpoint GREEDY on both sides, with the searched side's value trace; plus
the HONEST one-ply critic (world j, critic seed 40 + j). This is the pilot's own Phase 2 driver
(`value_disc_pilot_2026-10-11/run_det_playouts.one_state`, imported unchanged), pointed at this unit's worlds and
archive. One fsync'd JSONL row per state in ``<archive>/<cp>/rows/<code>/rows_<states>_<i>_of_<W>.jsonl``; resumable
(done = an id in ANY ``rows_*`` file of the code's directory). A state whose world row is an exclusion gets a row
``ok: false`` carrying the builder's reason. No outcome is printed.
"""
import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t1_common as T  # noqa: E402

sys.path.insert(1, str(T.PILOT))
import common as C  # noqa: E402  (the pilot's: clock, threads, model loader)
from run_det_playouts import J as PILOT_J, one_state  # noqa: E402

if PILOT_J != T.J:
    raise SystemExit(f"the pilot driver plays {PILOT_J} worlds, this unit registers {T.J}")


def check_pad_pin(cp: str) -> None:
    if os.environ.get("HIB_T1_PIN_KIND") != "pad":
        raise SystemExit("worlds are played at a PAD pin only (run_pin.sh <cp>_pad)")
    if not os.environ.get("HIB_T1_PIN_HEAD", "").startswith(T.CPS[cp]["pad"]):
        raise SystemExit(f"pad pin HEAD {os.environ.get('HIB_T1_PIN_HEAD')} != {T.CPS[cp]['pad']}")
    import main  # noqa: F401

    if not str(Path(main.__file__).resolve()).startswith(os.environ["HIB_T1_PIN_DIR"]):
        raise SystemExit(f"main imported from {main.__file__}, not the pad pin")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", required=True)
    ap.add_argument("--states", default="both", choices=("registered", "supp", "both"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    cp = C.cp()
    check_pad_pin(cp)
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.reader import inference_globals
    from main.policy_spectrum.truth import cmd_index
    from run_playouts import Fwd

    def w_dec(bank, did):
        return next(x for x in bank.decisions if x["id"] == did)
    from utils.rust_env import ffi as F
    from utils.rust_env.successors import SearchCore, SuccessorsError

    which = ("registered", "supp") if a.states == "both" else (a.states,)
    banks = {w: load_bank(T.bank_path(cp, w)) for w in which if T.bank_path(cp, w).exists()}
    wd = T.worlds_dir(cp, a.code)
    worlds = {}
    for w in banks:
        worlds.update({i: r for i, r in T.read_dir(wd, f"worlds_{w}_*.jsonl").items()})
    od = T.rows_dir(cp, a.code)
    done = set(T.read_dir(od, "rows_*.jsonl"))
    ids = T.shard([i for i in worlds if i not in done], a.shard, a.of)[: a.limit]
    path = od / f"rows_{a.states}_{a.shard}_of_{a.of}.jsonl"
    print(f"[play] {cp} {a.code} shard {a.shard}/{a.of}: {len(done)} done, {len(ids)} to go", flush=True)
    off = C.clock_offset()
    lib = F.load(F.default_path("release"))
    cores = [SearchCore(lib=lib) for _ in range(T.J)]
    t0 = time.monotonic()
    with inference_globals(C.threads()):
        fwd = Fwd(C.load_model(cp))
        with open(path, "a") as f:
            for n_i, did in enumerate(ids):
                w = worlds[did]
                bank = banks[w["states"]]
                if not w.get("ok"):
                    r = {"id": did, "ok": False, "excluded": True, "error": "world build: " + str(w.get("error"))[:500]}
                elif len(w["teams"]) != T.J or any(t is None for t in w["teams"]):
                    r = {"id": did, "ok": False, "error": f"world row holds {len(w['teams'])} worlds, not {T.J}"}
                elif cmd_index(bank.battles[bank.battle_index[w_dec(bank, did)["battle"]]], w["side"],
                               w_dec(bank, did)["n"]) != int(w["at"]):
                    r = {"id": did, "ok": False, "error": f"the pin's root command index != the builder's {w['at']}"}
                else:
                    try:
                        r = one_state(cores, fwd, bank, did, w, off)
                    except SuccessorsError as exc:
                        r = {"id": did, "ok": False, "error": str(exc)[:500]}
                        for k, core in enumerate(cores):       # a refused open leaves no state; reopen all
                            core.close()
                            cores[k] = SearchCore(lib=lib)
                r.update({"cp": cp, "code": a.code, "states": w["states"], "pin_head": os.environ["HIB_T1_PIN_HEAD"]})
                T.write_row(f, r)
                if not r.get("ok"):
                    print(f"[play] REFUSED {did}: {str(r.get('error'))[:200]}", flush=True)
                el = time.monotonic() - t0
                print(f"[play] {n_i + 1}/{len(ids)} {el / 60:.1f} min, eta "
                      f"{el / (n_i + 1) * (len(ids) - n_i - 1) / 60:.1f} min", flush=True)
    for core in cores:
        core.close()


if __name__ == "__main__":
    main()
