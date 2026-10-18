"""PREREGISTRATION §3.2, arm O: PHASE 1, the TRUTH, at one anchor: the pilot's own Phase 1 driver
(`run_playouts.one_state`, IMPORTED, never copied) with the anchor's checkpoint copy. Per state: every legal action x
the pilot's 40 shared seeds (`truth.turn_seeds(id, 48)`: leaf 0..15, referee 16..39) played to the END by the checkpoint
greedy on both sides in the true world, the searched side's value trace, and the critic's one-ply successors on seeds
40..47.

    run_mat.sh pin 1 6 6h phase1_mat.py --cp x26_025M --mode lean [--limit N] [--ids FILE]

Rows: ``<archive>/rows/<label>/O/shard_0_of_1.jsonl``, one fsync'd row per state; resumable; ``--limit`` caps NEW work
(``maturity_series.select_new_work``). No outcome is printed (progress counts only).
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
    ap.add_argument("--mode", choices=("lean", "full"), required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", default=None)
    a = ap.parse_args()
    E.check_pin()
    if a.cp not in E.registered_ckpts():
        raise SystemExit(f"{a.cp}: not a registered anchor")
    import common as C  # the pilot's
    import run_playouts as RP  # the pilot's Phase 1 driver

    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.reader import inference_globals
    from utils.rust_env import ffi as F
    from utils.rust_env.successors import SearchCore

    bank = load_bank(E.bank_path())
    ids = E.mode_ids(a.mode)
    if not set(ids) <= {d["id"] for d in bank.decisions}:
        raise SystemExit("the mode's ids are not all in the bank")
    if a.ids:
        keep = set(Path(a.ids).read_text().split())
        ids = [x for x in ids if x in keep]
    rdir = E.rows_dir(a.cp, "O")
    done = E.read_dir(rdir, "shard_*.jsonl")
    todo = E.src_module("maturity_series").select_new_work(ids, done, a.limit)
    print(f"[phase1_mat] {a.cp}: {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        return
    off = C.clock_offset()
    lib = F.load(F.default_path("release"))
    core = SearchCore(lib=lib)
    step = E.registered_ckpts()[a.cp]["step"]
    t0 = time.monotonic()
    with inference_globals(E.threads()):
        fwd = RP.Fwd(E.load_model(a.cp))
        with open(rdir / "shard_0_of_1.jsonl", "a") as f:
            try:
                for i, did in enumerate(todo):
                    r = RP.one_state(core, fwd, bank, did, off)
                    r.update({"cp": a.cp, "step": step, "set": "mat", "pin": E.series().pin})
                    E.write_row(f, r)
                    if not r.get("ok"):
                        print(f"[phase1_mat] REFUSED {did}: {str(r.get('error'))[:200]}", flush=True)
                    el = time.monotonic() - t0
                    print(f"[phase1_mat] {i + 1}/{len(todo)} {el / 60:.1f} min, eta "
                          f"{el / (i + 1) * (len(todo) - i - 1) / 60:.1f} min", flush=True)
            finally:
                core.close()


if __name__ == "__main__":
    main()
