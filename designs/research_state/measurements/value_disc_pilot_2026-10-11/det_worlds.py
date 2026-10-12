"""PHASE 2, step 1: build the determinized worlds for every sampled state (no playout, no outcome):
J = 16 gate-passed opponent teams per state (``determinize.build_worlds``), one durable row per state.

    run_at_pin.sh <cp> 1 8 6h det_worlds.py [--limit N] [--J 16] [--shard i --of W] [--sampler smogon|belief|moves]

``--sampler belief`` (Phase 2b): never-seen species from the network's X5 species presence; ``moves`` (Phase 2c,
Amendment 6): that AND the hidden moves from the network's own per-mon move presence (`move_belief.py`);
``truemoves`` (Phase 2c-T, the ceiling arm): as ``moves`` but every SEEN mon keeps its TRUE set.

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


def root_belief(c, bank, todo_ids, moves=False):
    """({decision id: {species id: π}}, {decision id: move_w}): the network's X5 species presence at each root (one
    CPU forward) and, with ``moves`` (Phase 2c), its per-mon move presence in `determinize.sample_world`'s ``move_w``
    form ({"seen": {num: {move: π}}, "hyp": {num: {move: π}}}; `move_belief.root_move_beliefs`, the same forward)."""
    from contextlib import nullcontext

    import numpy as np

    import move_belief as MB
    from agents.inference.belief_decode import _num_to_species_id
    from main.policy_spectrum.qhat import logits_and_values
    from main.policy_spectrum.reader import inference_globals
    from main.policy_spectrum.replay import replay

    want = set(todo_ids)
    decs = [d for d in sorted(bank.decisions, key=lambda x: x["id"]) if d["id"] in want]
    bids = sorted({d["battle"] for d in decs})
    reps = dict(zip(bids, replay([bank.battles[bank.battle_index[b]].recorded() for b in bids], workers=1)))
    num2id = _num_to_species_id()
    n2m = MB.move_ids(Path(os.environ["VDP_PIN"]) / "data" / "pokemon") if moves else {}
    out, mv = {}, {}
    with inference_globals(C.threads()), (MB.capture_roster() if moves else nullcontext({})) as box:
        model = C.load_model(c)
        ext = model.policy.features_extractor
        for i in range(0, len(decs), 64):
            part = decs[i:i + 64]
            dd = [reps[d["battle"]].decisions[0 if d["side"] == "p1" else 1][d["n"]] for d in part]
            box.clear()
            logits_and_values(model, np.stack([np.array(x.row, dtype=np.float32) for x in dd]),
                              np.stack([np.asarray(x.mask, dtype=np.uint8) for x in dd]), batch=64)
            pi = ext.last_hypothesis.species.pi.float().cpu().numpy()
            slots = MB.slots_from_box(box, n2m) if moves else None
            for j, d in enumerate(part):
                out[d["id"]] = {num2id[int(n)]: float(pi[j, n]) for n in np.flatnonzero(pi[j] > 0) if int(n) in num2id}
                if moves:
                    m = {"seen": {}, "hyp": {}}
                    for s in slots[j]:
                        if s["num"] > 0:
                            m["hyp" if s["hyp"] else "seen"][s["num"]] = s["w"]
                    mv[d["id"]] = m
    return out, mv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--J", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--sampler", choices=("smogon", "belief", "moves", "truemoves"), default="smogon")
    ap.add_argument("--det-dir", default="det")
    ap.add_argument("--max-tries", type=int, default=D.MAX_TRIES)
    ap.add_argument("--builder", choices=("v1", "v2"), default="v2", help="v2 = the fixed builder (Amendment 8)")
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
    path = od / a.det_dir / f"worlds_{a.shard}_of_{a.of}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    # Phase 1's root: did the log already hold the opponent's root-turn choice (other_open False)?
    other_open = {}
    belief, moves = (root_belief(c, bank, todo_ids=ids, moves=a.sampler in ("moves", "truemoves"))
                     if a.sampler in ("belief", "moves", "truemoves") else ({}, {}))
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
                w = D.build_worlds(b, d["side"], at, d["n"], did, P, a.J, max_tries=a.max_tries, builder=a.builder,
                                   root_choice_in_log=not other_open[did],
                                   species_w=belief.get(did) if a.sampler != "smogon" else None,
                                   move_w=({**moves[did], "true_seen": a.sampler == "truemoves"}
                                           if a.sampler in ("moves", "truemoves") else None))
                row = {"id": did, "ok": True, "other_open": other_open[did], **w}
                if a.sampler in ("moves", "truemoves"):
                    row["move_w"] = {k: {str(n): v for n, v in x.items()} for k, x in moves[did].items()}
            except Exception as exc:          # counted, never dropped
                row = {"id": did, "ok": False, "error": f"{type(exc).__name__}: {str(exc)[:400]}"}
            row["cpu_s"] = round(C.cpu_seconds() - c0, 2)
            C.write_row(f, row)
            if (i + 1) % 10 == 0 or i + 1 == len(todo):
                el = time.monotonic() - t0
                print(f"[worlds] {i + 1}/{len(todo)} {el / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
