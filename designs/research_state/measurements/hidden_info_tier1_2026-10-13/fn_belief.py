"""FN, PIN SIDE: the network's OWN belief at each root, read by ONE CPU forward at the checkpoint's UNMODIFIED pin
(the value-discrimination pilot's Phase 2c read, `value_disc_pilot_2026-10-11/det_worlds.root_belief(moves=True)` +
`move_belief.py`, IMPORTED, not re-implemented).

    run_pin.sh <end2|prod> 2 8 <timeout> fn_belief.py --states registered|supp [--limit N]

Per state, one row of ``~/gen3ai_archive/hidden_info_tier1/<cp>/fn_belief_<states>.jsonl``:
``{"id", "cp", "pin", "species": {species id: π} (the X5 species presence, every species with π > 0),
"moves": {"seen": {num: {move id: π}}, "hyp": {num: {move id: π}}}}`` — the per-mon fixed-mass move presence over the
mon's UNREVEALED candidates (Σ π = 4 − r), keyed by the species NUM as the pilot stored it (a formes-sharing num is
resolved on the HEAD side by the root's own species). The HEAD side (`fn_worlds.py`) reads these rows.
Resumable: ids already in the file are skipped.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import t1_common as T  # noqa: E402

sys.path.insert(0, str(T.PILOT))
import common as PC  # noqa: E402  (the pilot's common: VDP_* env, load_model, check_pin)
import det_worlds as DW  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", choices=("registered", "supp"), required=True)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    if os.environ.get("HIB_T1_PIN_KIND") != "pin":
        raise SystemExit("fn_belief runs at the UNMODIFIED pin (run_pin.sh end2|prod), not a pad pin")
    cp = PC.cp()
    PC.check_pin(cp)
    from main.policy_spectrum.bank import load_bank

    bp = T.bank_path(cp, a.states)
    if not (bp / "decisions.jsonl.gz").exists():
        raise SystemExit(f"{bp}: no bank (the supplement bank is built by another unit)")
    bank = load_bank(bp)
    out = T.cp_dir(cp) / f"fn_belief_{a.states}.jsonl"
    done = {r["id"] for r in T.read_rows(out)}
    ids = sorted(d["id"] for d in bank.decisions)
    if a.states == "registered":
        reg = set(T.registered_states(cp))
        if set(ids) != reg:
            raise SystemExit(f"the bank holds {len(ids)} ids, the registered set {len(reg)}: they differ")
    todo = [x for x in ids if x not in done][: a.limit]
    print(f"[fn_belief] {cp} {a.states}: {len(done)} done, {len(todo)} to go", flush=True)
    if not todo:
        return
    t0 = time.monotonic()
    c0 = PC.cpu_seconds()
    species, moves = DW.root_belief(cp, bank, todo_ids=todo, moves=True)
    miss = [x for x in todo if x not in species or x not in moves]
    if miss:
        raise SystemExit(f"root_belief returned no belief for {len(miss)} ids, e.g. {miss[:3]}")
    with open(out, "a") as f:
        for did in todo:
            T.write_row(f, {"id": did, "cp": cp, "pin": os.environ["VDP_PIN_SHA"], "states": a.states,
                            "species": species[did],
                            "moves": {k: {str(n): w for n, w in v.items()} for k, v in moves[did].items()}})
    print(f"[fn_belief] wrote {len(todo)} rows to {out} in {time.monotonic() - t0:.0f} s wall, "
          f"{PC.cpu_seconds() - c0:.0f} CPU-s", flush=True)


if __name__ == "__main__":
    main()
