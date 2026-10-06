"""The PAIRED speed benchmark's read (X5 Amendment 1 / 5, §7.4's reader rule), x5-perf-encode 2026-10-05.

    s_read.py --dir <scratch> --blocks sB_blob_1 sB_fm_1 ... --out s.json

Per block (one fresh launch into a seeded 20-snapshot pool, regime A): the update-CYCLE walls = the wall
between consecutive `train/train_ms` TensorBoard records (rollout + update). A cycle is EXCLUDED when its
window holds an eval cycle (an `eval` / `eval_play` phase marker), a restart (none can occur: one trainer
process per block, checked), the compile-canary update (the 10th train call, `CANARY_UPDATE`), or any
CPU-sampler row with a windowed contention factor >= 1.05 (the quiet rule, standing rule 8). The first TB
record is the first REAL update; the dry update writes none, so every cycle starts after it.

s = median(fixed_mass cycles, pooled over blocks) / median(blob cycles) - 1; also per block pair and the
block-level spread (each block's median), and the matched-wall-time checkpoint floor_1M(15M / (1 + s)).
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

CANARY_UPDATE = 10      # the 10th train call (i = 10 counting the dry update as 0) carries the canary


def tb_walls(tb_dir: Path, tag: str = "train/train_ms"):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    ea.Reload()
    return [(e.step, e.wall_time, e.value) for e in ea.Scalars(tag)]


def read_block(d: Path, tag: str) -> dict:
    ph = [json.loads(x) for x in (d / f"{tag}.phases.jsonl").read_text().splitlines() if x.strip()]
    cpu = [json.loads(x) for x in (d / f"{tag}.cpu.jsonl").read_text().splitlines()[1:] if x.strip()]
    evals = [(p["t0"], p.get("t1", p["t0"])) for p in ph if p["phase"] in ("eval", "eval_play")]
    upd = [p for p in ph if p["phase"] == "update"]
    tbd = d / "models" / tag / "tb"
    runs = [p for p in tbd.rglob("events.out.tfevents.*")]
    assert len({r.parent for r in runs}) == 1, f"{tag}: more than one TB run dir (a restart?) {runs}"
    recs = tb_walls(runs[0].parent)
    cycles = []
    for k in range(1, len(recs)):
        w0, w1 = recs[k - 1][1], recs[k][1]
        upd_i = k + 1                    # record k is the update with train-call index k + 1 (dry = 0)
        why = None
        if upd_i == CANARY_UPDATE:
            why = "compile-canary update"
        elif upd_i == CANARY_UPDATE + 1:
            why = "after the compile-canary update (its record is written after the canary, so this cycle reads short)"
        elif any(a < w1 and b > w0 for a, b in evals):
            why = "eval cycle in window"
        else:
            f = [s["factor"] for s in cpu if w0 <= s["t"] - s["dt"] and s["t"] <= w1]
            if not f:
                why = "no sampler rows"
            elif max(f) >= 1.05:
                why = f"contended (max factor {max(f):.3f})"
        cycles.append({"k": k, "update": upd_i, "wall_s": round(w1 - w0, 3), "excluded": why})
    kept = [c["wall_s"] for c in cycles if c["excluded"] is None]
    return {"tag": tag, "tb_records": len(recs), "update_phases": len(upd), "cycles": cycles,
            "kept_n": len(kept), "kept": kept, "median": st.median(kept) if kept else None,
            "train_ms_median": st.median([r[2] for r in recs[1:]]) / 1000 if len(recs) > 1 else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--blocks", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d = Path(a.dir)
    blocks = [read_block(d, t) for t in a.blocks]
    arm = lambda b: "fixed_mass" if "_fm_" in b["tag"] else "blob"       # noqa: E731
    pooled = {k: [x for b in blocks if arm(b) == k for x in b["kept"]] for k in ("blob", "fixed_mass")}
    med = {k: st.median(v) for k, v in pooled.items()}
    s = med["fixed_mass"] / med["blob"] - 1
    bmed = {k: [b["median"] for b in blocks if arm(b) == k] for k in ("blob", "fixed_mass")}
    pairs = [bf["median"] / bb["median"] - 1 for bb, bf in zip([b for b in blocks if arm(b) == "blob"],
                                                              [b for b in blocks if arm(b) == "fixed_mass"])]
    ck = math.floor(15_000_000 / (1 + s) / 1_000_000) * 1_000_000
    res = {"s": s, "median_cycle_s": med, "n_cycles": {k: len(v) for k, v in pooled.items()},
           "block_medians": bmed, "s_per_block_pair": pairs,
           "s_block_range": [min(pairs), max(pairs)],
           "matched_wall_time_steps": 15_000_000 / (1 + s), "matched_wall_time_checkpoint": ck,
           "train_ms_block_medians_s": {k: [b["train_ms_median"] for b in blocks if arm(b) == k]
                                        for k in ("blob", "fixed_mass")},
           "blocks": blocks}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "blocks"}, indent=1))


if __name__ == "__main__":
    main()
