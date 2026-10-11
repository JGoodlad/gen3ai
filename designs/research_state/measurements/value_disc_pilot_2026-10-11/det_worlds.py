"""PHASE 2, step 1: build the determinized worlds for every sampled state (no playout, no outcome):
J = 16 gate-passed opponent teams per state (``determinize.build_worlds``), one durable row per state.

    run_at_pin.sh <cp> 1 8 6h det_worlds.py [--limit N] [--J 16] [--shard i --of W]

→ archive ``<cp>/det/worlds_<shard>.jsonl``.
"""
import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import determinize as D  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--J", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    a = ap.parse_args()
    c = C.cp()
    C.check_pin(c)
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.truth import cmd_index

    od = C.out_dir(c)
    bank = load_bank(od / "bank")
    P = D.Priors(Path(os.environ["VDP_PIN"]) / "data" / "pokemon")
    ids = sorted(d["id"] for d in bank.decisions)
    ids = [x for i, x in enumerate(ids) if i % a.of == a.shard][: a.limit]
    path = od / "det" / f"worlds_{a.shard}_of_{a.of}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Phase 1's root: did the log already hold the opponent's root-turn choice (other_open False)?
    other_open = {}
    for q in sorted((od / "rows").glob("shard_*.jsonl")):
        for r in C.read_rows(q):
            if r.get("ok"):
                other_open[r["id"]] = bool(r["other_open"])
    done = {r["id"] for q in sorted(path.parent.glob("worlds_*.jsonl")) for r in C.read_rows(q)}
    todo = [x for x in ids if x not in done]
    print(f"[worlds] {c}: {len(done)} done, {len(todo)} to go", flush=True)
    t0 = time.monotonic()
    with open(path, "a") as f:
        for i, did in enumerate(todo):
            d = next(x for x in bank.decisions if x["id"] == did)
            b = bank.battles[bank.battle_index[d["battle"]]]
            at = cmd_index(b, d["side"], d["n"])
            c0 = C.cpu_seconds()
            try:
                if did not in other_open:
                    raise RuntimeError("no Phase-1 row for this state (other_open unknown)")
                w = D.build_worlds(b, d["side"], at, d["n"], did, P, a.J,
                                   root_choice_in_log=not other_open[did])
                row = {"id": did, "ok": True, "other_open": other_open[did], **w}
            except Exception as exc:          # counted, never dropped
                row = {"id": did, "ok": False, "error": f"{type(exc).__name__}: {str(exc)[:400]}"}
            row["cpu_s"] = round(C.cpu_seconds() - c0, 2)
            C.write_row(f, row)
            if (i + 1) % 10 == 0 or i + 1 == len(todo):
                el = time.monotonic() - t0
                print(f"[worlds] {i + 1}/{len(todo)} {el / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
