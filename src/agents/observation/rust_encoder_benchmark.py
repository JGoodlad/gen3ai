"""THE ENCODER BENCHMARK — the Rust encoder's cost per decision on the obs golden's banked battles (P6 of the
poke-env retirement: the row every training run reads is the core's, so this replaces ``obs_build_benchmark.py``
as the MANDATORY before / after benchmark of any observation change).

For a seeded sample of the golden battles' p1 decisions (``agents.training.golden_obs_core``: banked input logs,
the same battles every run), ``core_events --obs --obs-bench 0 K REPS`` replays the battle and times, at decision K,
REPS encodes of the version (its view memoized — the production shape, ``encode_us``) and REPS of ``present()`` +
encode (the cold shape, ``present_encode_us``), asserting the two rows byte-equal. Reported: per-decision medians,
then the median / p90 / max over decisions of each.

A benchmark's output IS the measurement: it WARNS on a contended box and never rescales (root ``CLAUDE.md``).
The binary is the RELEASE build (``utils.bridge.sim_bridge_bin``; never the self-check build, which also checks
every emission). Run it before and after an obs change, on the same box, and compare the two lines::

    python -m agents.observation.rust_encoder_benchmark --decisions 40 --reps 300 | tee /tmp/enc_before.txt
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from typing import Dict, List


def decision_counts(battles) -> List[int]:
    """p1's decision count per banked battle (from one plain ``--obs`` replay)."""
    from agents.battle.core_replay import run_core

    return [sum(1 for c in res["trackers"][0] if "obs" in c) for res in run_core(battles, trackers=True, obs=True)]


def bench_one(binary: str, battle, k: int, reps: int) -> Dict:
    stdin = "\n".join(battle.script()) + "\n"
    p = subprocess.run([binary, "--obs", "--trackers", "--obs-bench", "0", str(k), str(reps)], input=stdin,
                       capture_output=True, text=True, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"core_events --obs-bench failed (exit {p.returncode}): {p.stderr.strip()[-1000:]}")
    res = json.loads(p.stdout.splitlines()[0])
    if not res.get("ok") or "obs_bench" not in res:
        raise RuntimeError(f"no obs_bench in the core's answer for {battle.label} decision {k}: {res.get('error')}")
    return res["obs_bench"]


def pct(xs: List[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--decisions", type=int, default=40, help="how many decisions to time (seeded sample)")
    ap.add_argument("--reps", type=int, default=300, help="encodes per decision, per shape")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, help="also write the per-decision results here")
    a = ap.parse_args(argv)

    from agents.training.golden_obs_core import load_battles
    from utils.bridge.sim_bridge_bin import resolve_core_events_bin
    from utils.contention import warn_if_contended

    if os.environ.get("POKESIM_EMISSION_SELFCHECK") == "1":
        print("[encoder-bench] ⚠️ POKESIM_EMISSION_SELFCHECK=1 — timing the SELF-CHECK build, not production's",
              file=sys.stderr)
    warn_if_contended()
    binary = resolve_core_events_bin()
    battles = load_battles()
    counts = decision_counts(battles)
    pool = [(b, k) for b, n in zip(battles, counts) for k in range(n)]
    sample = random.Random(a.seed).sample(pool, min(a.decisions, len(pool)))
    rows = []
    for b, k in sample:
        r = bench_one(binary, b, k, a.reps)
        rows.append({"battle": b.label, "k": k, "turn": r["turn"], "encode_us": r["encode_us_median"],
                     "present_encode_us": r["present_encode_us_median"], "nan_poison": r["nan_poison"]})
    if any(r["nan_poison"] for r in rows):
        print("[encoder-bench] ⚠️ the binary NaN-poisons rows (a test build): its timings are not production's",
              file=sys.stderr)
    for key in ("encode_us", "present_encode_us"):
        xs = [r[key] for r in rows]
        print(f"[encoder-bench] {key:>18}: median {pct(xs, 0.5):8.1f} µs   p90 {pct(xs, 0.9):8.1f} µs   "
              f"max {max(xs):8.1f} µs   ({len(xs)} decisions × {a.reps} reps)")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"binary": binary, "reps": a.reps, "seed": a.seed, "rows": rows}, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
