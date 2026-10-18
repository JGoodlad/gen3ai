"""PREREGISTRATION §3.3, FN pin side: the anchor network's OWN belief at each root, one CPU forward at the series' pin
(the pilot's Phase 2c read, `det_worlds.root_belief(moves=True)` + `move_belief.py`, IMPORTED; Tier 1's `fn_belief.py`
row format, so Tier 1's `fn_worlds.fn_sampler` reads it unchanged on the HEAD side).

    run_mat.sh pin 1 8 2h fn_belief_mat.py --cp x26_025M --mode lean [--limit N]

Rows: ``<archive>/fn_belief/<label>.jsonl``: ``{"id", "cp", "pin", "species": {species id: π}, "moves": {"seen": {num:
{move id: π}}, "hyp": {num: {move id: π}}}}``. Resumable; ``--limit`` caps NEW work.
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
    a = ap.parse_args()
    E.check_pin()
    if a.cp not in E.registered_ckpts():
        raise SystemExit(f"{a.cp}: not a registered anchor")
    import common as PC  # the pilot's
    import det_worlds as DW

    from main.policy_spectrum.bank import load_bank

    E.install_pilot_loader(a.cp)                 # root_belief calls common.load_model(cp): the anchor's copy
    bank = load_bank(E.bank_path())
    out = E.d("fn_belief") / f"{a.cp}.jsonl"
    done = E.read_dir(out.parent, out.name)
    todo = E.src_module("maturity_series").select_new_work(E.mode_ids(a.mode), done, a.limit)
    print(f"[fn_belief_mat] {a.cp}: {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        return
    t0, c0 = time.monotonic(), PC.cpu_seconds()
    species, moves = DW.root_belief(a.cp, bank, todo_ids=todo, moves=True)
    miss = [x for x in todo if x not in species or x not in moves]
    if miss:
        raise SystemExit(f"root_belief returned no belief for {len(miss)} ids, e.g. {miss[:3]}")
    with open(out, "a") as f:
        for did in todo:
            E.write_row(f, {"id": did, "cp": a.cp, "pin": E.series().pin, "species": species[did],
                            "moves": {k: {str(n): w for n, w in v.items()} for k, v in moves[did].items()}})
    print(f"[fn_belief_mat] wrote {len(todo)} rows in {time.monotonic() - t0:.0f} s wall, "
          f"{PC.cpu_seconds() - c0:.0f} CPU-s", flush=True)


if __name__ == "__main__":
    main()
