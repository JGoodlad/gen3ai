"""PREREGISTRATION §3.4: PLAY one arm's worlds with one anchor's checkpoint at the series' pin (X26's pin holds the gender
draw pad, so no pad pin): the pilot's own Phase 2 driver (`run_det_playouts.one_state`, imported unchanged). Per state
with a built world row: every legal action played to the END in each of the 8 worlds (world j on dice seed j), the
checkpoint GREEDY on both sides, with the searched side's value trace, plus the honest one-ply critic (world j, critic
seed 40 + j).

    run_mat.sh pin 1 8 6h play_mat_worlds.py --cp x26_025M --arm F|FN|F+M|F+U|O-M|O-U|V:oracle \
        --mode lean [--limit N] [--bank interim|plateau]

One fsync'd JSONL row per state in ``<archive>/rows/<label>/<arm>/rows_0_of_1.jsonl``; resumable; ``--limit`` caps NEW
work. A state whose world row is an exclusion or a defect gets an ``ok: false`` row carrying the builder's reason. No
outcome is printed.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mat_common as E  # noqa: E402

E.add_paths()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cp", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--mode", choices=tuple(E.MODE_RANK), required=True)
    ap.add_argument("--bank", default="interim", choices=sorted(E.BANKS))
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    E.set_bank(a.bank)
    E.check_pin()
    if a.cp not in E.registered_ckpts():
        raise SystemExit(f"{a.cp}: not a registered anchor")
    if a.arm not in E.ARMS and not a.arm.startswith("V:"):
        raise SystemExit(f"unknown arm {a.arm}")
    if a.arm.startswith("V:") and a.bank != "interim":
        raise SystemExit("the identity is played on the interim bank only (A1)")
    import common as C  # the pilot's: clock, cpu seconds
    from run_det_playouts import J as PILOT_J, one_state
    from run_playouts import Fwd

    if PILOT_J != E.J:
        raise SystemExit(f"the pilot driver plays {PILOT_J} worlds, the battery registers {E.J}")
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.reader import inference_globals
    from main.policy_spectrum.truth import cmd_index
    from utils.rust_env import ffi as F
    from utils.rust_env.successors import SearchCore, SuccessorsError

    bank = load_bank(E.bank_path())
    dec = {d["id"]: d for d in bank.decisions}
    worlds = E.read_dir(E.worlds_dir(a.cp, a.arm), "worlds_*.jsonl")
    want = set(E.identity_ids()) if a.arm.startswith("V:") and not E.PILOT_MODE else set(E.mode_ids(a.mode))
    od = E.rows_dir(a.cp, a.arm)
    done = E.read_dir(od, "rows_*.jsonl")
    ids = E.src_module("maturity_series").select_new_work(sorted(x for x in worlds if x in want), done, a.limit)
    print(f"[mat play] {a.cp} {a.arm}: {len(done)} done, {len(ids)} to go", flush=True)
    if not ids:
        return
    off = C.clock_offset()
    lib = F.load(F.default_path("release"))
    cores = [SearchCore(lib=lib) for _ in range(E.J)]
    step = E.registered_ckpts()[a.cp]["step"]
    t0 = time.monotonic()
    with inference_globals(E.threads()):
        fwd = Fwd(E.load_model(a.cp))
        with open(od / "rows_0_of_1.jsonl", "a") as f:
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
                r.update({"cp": a.cp, "step": step, "arm": a.arm, "pin_head": E.series().pin,
                          "world_head": w.get("head")})
                E.write_row(f, r)
                if not r.get("ok"):
                    print(f"[mat play] REFUSED {did}: {str(r.get('error'))[:200]}", flush=True)
                el = time.monotonic() - t0
                print(f"[mat play] {n_i + 1}/{len(ids)} {el / 60:.1f} min, eta "
                      f"{el / (n_i + 1) * (len(ids) - n_i - 1) / 60:.1f} min", flush=True)
    for core in cores:
        core.close()


if __name__ == "__main__":
    main()
