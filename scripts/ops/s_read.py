"""The PAIRED speed benchmark's cost read (X5 Amendment 1 / 5, §7.4's reader rule; the static-token screen's `s`).

    python3 scripts/ops/s_read.py --dir <scratch> --blocks sB_legacy_1 sB_static_1 ... --out s.json

Per block (one fresh launch into a seeded 20-snapshot pool, regime A): the update-CYCLE walls = the wall
between consecutive `train/train_ms` TensorBoard records (rollout + update). A cycle is EXCLUDED when its
window holds an eval cycle (an `eval` / `eval_play` phase marker), a restart (none can occur: one trainer
process per block, checked), the compile-canary update (the 10th train call, `CANARY_UPDATE`), or any
CPU-sampler row with a windowed contention factor >= 1.05 (the quiet rule, standing rule 8), or the STOP: a
record written after the driver's SIGTERM (`drive.log`) is the abort path's dump, whose window holds an update
but NO rollout (it reads ~10-15 s short), so that cycle is excluded. The first TB record is the first REAL
update; the dry update writes none, so every cycle starts after it.

s = median(alt-arm cycles, pooled over blocks) / median(base-arm cycles) - 1; also per block pair and the
block-level spread (each block's median), and the matched-wall-time checkpoint floor_1M(15M / (1 + s)).

INSUFFICIENT DATA IS A RESULT, NOT A CRASH (2026-10-08). A block whose every cycle was excluded (a contended
box leaves ZERO quiet cycles) has no median. The first version divided `None` by `None` and died with a
TypeError, losing the pairs that DID read. Now such a pair reports `status: "insufficient quiet cycles"` with
the kept counts of both sides; the pooled `s` is `null` (with its reason) when either arm has fewer than
`--min-cycles` kept cycles pooled over its blocks; the exit status is 0 when `s` was computed and 3 when it was
not (the JSON is written either way).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics as st
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

CANARY_UPDATE = 10      # the 10th train call (i = 10 counting the dry update as 0) carries the canary
INSUFFICIENT = "insufficient quiet cycles"


def tb_walls(tb_dir: Path, tag: str = "train/train_ms"):
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    ea.Reload()
    return [(e.step, e.wall_time, e.value) for e in ea.Scalars(tag)]


def classify_cycles(recs: Sequence[tuple], evals: Sequence[tuple], cpu: Sequence[dict], t_stop: float) -> List[dict]:
    """One row per cycle (consecutive TB records): its wall and why it is excluded, or None. Pure."""
    cycles = []
    for k in range(1, len(recs)):
        w0, w1 = recs[k - 1][1], recs[k][1]
        upd_i = k + 1                    # record k is the update with train-call index k + 1 (dry = 0)
        why = None
        if w1 >= t_stop:
            why = "the stop: the abort dump after SIGTERM (an update, no rollout)"
        elif upd_i == CANARY_UPDATE:
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
    return cycles


def block_summary(tag: str, recs: Sequence[tuple], cycles: List[dict], update_phases: int) -> dict:
    kept = [c["wall_s"] for c in cycles if c["excluded"] is None]
    return {"tag": tag, "tb_records": len(recs), "update_phases": update_phases, "cycles": cycles,
            "kept_n": len(kept), "kept": kept, "median": st.median(kept) if kept else None,
            "train_ms_median": st.median([r[2] for r in recs[1:]]) / 1000 if len(recs) > 1 else None}


def read_block(d: Path, tag: str) -> dict:
    ph = [json.loads(x) for x in (d / f"{tag}.phases.jsonl").read_text().splitlines() if x.strip()]
    cpu = [json.loads(x) for x in (d / f"{tag}.cpu.jsonl").read_text().splitlines()[1:] if x.strip()]
    m = re.search(r"SIGTERM -> \d+ at ([0-9.]+)", (d / f"{tag}.drive.log").read_text())
    t_stop = float(m.group(1)) if m else float("inf")
    evals = [(p["t0"], p.get("t1", p["t0"])) for p in ph if p["phase"] in ("eval", "eval_play")]
    upd = [p for p in ph if p["phase"] == "update"]
    tbd = d / "models" / tag / "tb"
    runs = [p for p in tbd.rglob("events.out.tfevents.*")]
    assert len({r.parent for r in runs}) == 1, f"{tag}: more than one TB run dir (a restart?) {runs}"
    recs = tb_walls(runs[0].parent)
    return block_summary(tag, recs, classify_cycles(recs, evals, cpu, t_stop), len(upd))


def _ratio(num: Optional[float], den: Optional[float]) -> Optional[float]:
    return None if num is None or den is None else num / den - 1


def summarize(blocks: List[dict], arm: Callable[[dict], str], base: str, alt: str, min_cycles: int = 1) -> dict:
    """The pooled `s`, the per-pair `s` and the block spread; a side without data is REPORTED, never divided."""
    pooled = {k: [x for b in blocks if arm(b) == k for x in b["kept"]] for k in (base, alt)}
    n_cycles = {k: len(v) for k, v in pooled.items()}
    short = [k for k in (base, alt) if n_cycles[k] < max(1, min_cycles)]
    med: Dict[str, Optional[float]] = {k: (st.median(v) if v else None) for k, v in pooled.items()}
    s = None if short else _ratio(med[alt], med[base])
    base_blocks = [b for b in blocks if arm(b) == base]
    alt_blocks = [b for b in blocks if arm(b) == alt]
    pair_rows = []
    for i, (bb, ba) in enumerate(zip(base_blocks, alt_blocks)):
        ok = bb["kept_n"] >= max(1, min_cycles) and ba["kept_n"] >= max(1, min_cycles)
        pair_rows.append({"pair": i, "base": bb["tag"], "alt": ba["tag"], "kept_n": [bb["kept_n"], ba["kept_n"]],
                          "s": _ratio(ba["median"], bb["median"]) if ok else None,
                          "status": "ok" if ok else INSUFFICIENT})
    pairs = [r["s"] for r in pair_rows if r["s"] is not None]
    unpaired = abs(len(base_blocks) - len(alt_blocks))
    res: Dict[str, Any] = {
        "s": s,
        "s_status": "ok" if s is not None else
        f"{INSUFFICIENT} in the pooled {' and '.join(short)} arm (kept: {n_cycles}, need >= {max(1, min_cycles)})",
        "median_cycle_s": med, "n_cycles": n_cycles,
        "block_medians": {k: [b["median"] for b in blocks if arm(b) == k] for k in (base, alt)},
        "pairs": pair_rows, "s_per_block_pair": pairs,
        "s_block_range": [min(pairs), max(pairs)] if pairs else None,
        "n_pairs_read": len(pairs), "n_pairs_insufficient": sum(r["status"] == INSUFFICIENT for r in pair_rows),
        "unpaired_blocks": unpaired,
        "matched_wall_time_steps": None if s is None else 15_000_000 / (1 + s),
        "matched_wall_time_checkpoint": None if s is None else math.floor(15_000_000 / (1 + s) / 1_000_000) * 1_000_000,
        "train_ms_block_medians_s": {k: [b["train_ms_median"] for b in blocks if arm(b) == k] for k in (base, alt)},
    }
    return res


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--blocks", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base-arm", default="legacy")
    ap.add_argument("--alt-arm", default="static")
    ap.add_argument("--alt-marker", default="_static_", help="a block whose tag contains this is the alt arm")
    ap.add_argument("--min-cycles", type=int, default=1, help="quiet cycles an arm (or a pair side) needs to read")
    a = ap.parse_args(argv)
    d = Path(a.dir)
    blocks = [read_block(d, t) for t in a.blocks]
    res = summarize(blocks, lambda b: a.alt_arm if a.alt_marker in b["tag"] else a.base_arm,
                    a.base_arm, a.alt_arm, a.min_cycles)
    res["blocks"] = blocks
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "blocks"}, indent=1))
    return 0 if res["s"] is not None else 3


if __name__ == "__main__":
    sys.exit(main())
