"""E2 §2, arm O: PHASE 1, the TRUTH on the E2 bank — the pilot's own Phase 1 driver (`run_playouts.one_state`, IMPORTED,
never copied), at the UNMODIFIED pin, exactly as Tier 1's `phase1_supp.py` ran it on the supplement. Per state: every
legal action x the pilot's 40 shared seeds (`truth.turn_seeds(id, 48)`: leaf 0..15, referee 16..39) played to the END by
the checkpoint greedy on both sides in the true world, the searched side's value trace, and the critic's one-ply
successors on seeds 40..47.

    run_e2.sh <end2|prod> 1 6 24h phase1_e2.py [--shard i --of W] [--limit N] [--ids FILE]

Rows: ~/gen3ai_archive/hidden_info_e2/<cp>/rows/O/shard_<i>_of_<W>.jsonl, one fsync'd row per state; resumable (done = an
id with a row in ANY shard file). No outcome is printed (progress counts only).
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import e2_common as E  # noqa: E402

E.add_paths()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--ids", default=None, help="a file of decision ids (one per line) to restrict to")
    a = ap.parse_args()
    c = os.environ.get("VDP_CP")
    if c not in E.CPS:
        raise SystemExit("run through run_e2.sh <end2|prod>")
    import make_e2_bank as MB

    MB.check_plain_pin(c)
    import common as C  # the pilot's
    import run_playouts as RP  # the pilot's Phase 1 driver

    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.reader import inference_globals
    from utils.rust_env import ffi as F
    from utils.rust_env.successors import SearchCore

    bank = load_bank(E.bank_path(c))
    st = E.e2_states(c)
    doc_sha = __import__("json").loads(E.states_path(c).read_text())["bank_content_sha256"]
    if bank.manifest["content_sha256"] != doc_sha:
        raise SystemExit("the bank is not the one states_e2_<cp>.json describes")
    ids = sorted(d["id"] for d in bank.decisions)
    if ids != sorted(st):
        raise SystemExit("bank ids != states_e2 ids")
    if a.ids:
        keep = {x.strip() for x in Path(a.ids).read_text().splitlines() if x.strip()}
        ids = [x for x in ids if x in keep]
    ids = E.shard(ids, a.shard, a.of)
    rdir = E.rows_dir(c, "O")
    path = rdir / f"shard_{a.shard}_of_{a.of}.jsonl"
    done = set(E.read_dir(rdir, "shard_*.jsonl"))
    todo = [x for x in ids if x not in done]
    if a.limit:                                  # cap the NEW work; capping ids first re-picked the done ones forever
        todo = todo[:a.limit]
    print(f"[phase1_e2] {c} shard {a.shard}/{a.of}: {len(done)} done, {len(todo)} to go", flush=True)
    off = C.clock_offset()
    lib = F.load(F.default_path("release"))
    core = SearchCore(lib=lib)
    t0 = time.monotonic()
    with inference_globals(C.threads()):
        model = C.load_model(c)
        fwd = RP.Fwd(model)
        with open(path, "a") as f:
            try:
                for i, did in enumerate(todo):
                    r = RP.one_state(core, fwd, bank, did, off)
                    r.update({"cp": c, "set": "e2", "pin": E.CPS[c]["pin"]})
                    E.write_row(f, r)
                    if not r.get("ok"):
                        print(f"[phase1_e2] REFUSED {did}: {str(r.get('error'))[:200]}", flush=True)
                    el = time.monotonic() - t0
                    print(f"[phase1_e2] {i + 1}/{len(todo)} {el / 60:.1f} min, eta "
                          f"{el / (i + 1) * (len(todo) - i - 1) / 60:.1f} min", flush=True)
            finally:
                core.close()


if __name__ == "__main__":
    main()
