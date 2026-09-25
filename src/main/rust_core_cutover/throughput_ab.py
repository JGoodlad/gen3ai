"""The cutover tier's THROUGHPUT non-regression (program §3 target 10): `trainer_turn_benchmark.py
--pin-battles` over the RUST bridge, `--obs-source python` vs `--obs-source core`, INTERLEAVED
(A B, B A, A B, …) so a drifting box load lands on both arms alike.

The measured quantity is the per-decision FULL CYCLE (entry to entry between the trainee's
decisions, one in-process battle): our Python CPU plus the child's sim advance and — under `core`
— the child's parse / trackers / encode, all serialized, so it is the total CPU a decision costs.
The benchmark's "our controllable CPU" is reported beside it (the Python side alone).

Verdict (pre-registered): NON-REGRESSION iff the upper end of the 95% bootstrap CI of the mean
per-pair relative delta ``(core - python) / python`` of the cycle median is <= +3%.

    python -m main.rust_core_cutover.throughput_ab --pairs 6 --decisions 400 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

BAR = 0.03
_CYCLE = re.compile(r"full decision cycle\s*:\s*([\d.]+) ms")
_OURS = re.compile(r"our controllable CPU\s*:\s*([\d.]+) ms")


def parse(text: str) -> Dict[str, Optional[float]]:
    c, o = _CYCLE.search(text), _OURS.search(text)
    return {"cycle_ms": float(c.group(1)) if c else None, "ours_ms": float(o.group(1)) if o else None}


def bootstrap_ci(xs: List[float], n: int = 10_000, seed: int = 0) -> List[float]:
    rng = random.Random(seed)
    means = sorted(statistics.fmean(rng.choice(xs) for _ in xs) for _ in range(n))
    return [means[int(0.025 * n)], means[int(0.975 * n) - 1]]


def verdict(pairs: List[dict], bar: float = BAR) -> dict:
    rel = [(p["core"]["cycle_ms"] - p["python"]["cycle_ms"]) / p["python"]["cycle_ms"] for p in pairs]
    ours = [(p["core"]["ours_ms"] - p["python"]["ours_ms"]) / p["python"]["ours_ms"] for p in pairs]
    ci = bootstrap_ci(rel)
    return {"pairs": len(pairs), "cycle_rel_mean": statistics.fmean(rel), "cycle_rel_ci95": ci,
            "ours_rel_mean": statistics.fmean(ours), "ours_rel_ci95": bootstrap_ci(ours),
            "bar": bar, "non_regression": ci[1] <= bar}


def run_arm(obs_source: str, seed: int, decisions: int, log: Path) -> dict:
    from utils.paths import src_path

    argv = [sys.executable, str(src_path("agents", "training", "trainer_turn_benchmark.py")),
            "--pin-battles", "--seed", str(seed), "--decisions", str(decisions),
            "--bridge", "rust", "--obs-source", obs_source]
    t0 = time.time()
    p = subprocess.run(argv, capture_output=True, text=True, check=False)
    log.write_text(p.stdout + "\n" + p.stderr)
    row = parse(p.stdout)
    if p.returncode != 0 or row["cycle_ms"] is None:
        raise RuntimeError(f"{obs_source} arm failed (exit {p.returncode}); see {log}")
    return {**row, "wall_s": round(time.time() - t0, 1), "load1": __import__("os").getloadavg()[0]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--pairs", type=int, default=6)
    ap.add_argument("--decisions", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0, help="the SAME pinned battles in every run")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    pairs = []
    for k in range(a.pairs):
        order = ("python", "core") if k % 2 == 0 else ("core", "python")
        pair = {"k": k, "order": list(order)}
        for arm in order:
            pair[arm] = run_arm(arm, a.seed, a.decisions, a.out / f"pair{k}_{arm}.log")
        pairs.append(pair)
        print(json.dumps(pair), flush=True)
        (a.out / "pairs.jsonl").write_text("".join(json.dumps(p) + "\n" for p in pairs))
    v = verdict(pairs)
    (a.out / "verdict.json").write_text(json.dumps(v, indent=1) + "\n")
    print(json.dumps(v, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
