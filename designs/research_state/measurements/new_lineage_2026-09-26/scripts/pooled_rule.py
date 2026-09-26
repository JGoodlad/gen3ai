"""The NEW LINEAGE population loop's POOLED read (registration §5.3), mechanised — plus its power table.

    # the registered read, once all R rounds' readers have finished (R = 4 by registration)
    python pooled_rule.py --pair RB1:RC1 --pair RB2:RC2 --pair RB3:RC3 --pair RB4:RC4 --json OUT
    # the power statement the registration quotes (no runs read)
    python pooled_rule.py --power

Each --pair is LOOP_READER:CONTROL_READER (run names or run dirs) for one round. Every reader is read
through `agents.training.best_response_gap.read_exploiter` — the meter's own reader, so the
vs-target series is exactly what `main.best_response_gap` sees — and ALL readers of the read go
through the meter's `check_matched` (budget / dose / regime) together: one mismatch and the read is
VOID (exit 3). Nothing is written unless --json is given (finding K-1 of round 2: never write into cwd).

Estimand (per round r): d_r = rate(RB_r) - rate(RC_r), each rate POOLED over the reader's post-fork
eval cycles (4 cycles x 200 games = 800 per reader). Pooled: D = mean_r d_r (equal weights: every
round has the same registered n). CI: fixed-effect Wald, Var(D) = sum_r Var(d_r) / R^2 with
Var(d_r) = pB(1-pB)/nB + pC(1-pC)/nC. Per-round rows carry their Newcombe 95 % CI (the round-1/2
statistic) as DESCRIPTORS. Heterogeneity: Cochran's Q over the d_r (descriptor, never folded).
Rule (bar = max(F', 5.0 pp)): OUTSIDE iff |D| > bar AND the CI excludes the bar point on that side;
EQUIVALENT iff the whole CI lies inside [-bar, +bar]; otherwise NOT DETECTED.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

Z = 1.959963984540054


def _pp(x: float) -> str:
    return f"{100 * x:+.2f}"


def verdict(d: float, lo: float, hi: float, bar: float) -> str:
    if abs(d) > bar and (hi < -bar if d < 0 else lo > bar):
        return "OUTSIDE, BELOW" if d < 0 else "OUTSIDE, ABOVE"
    if lo > -bar and hi < bar:
        return "EQUIVALENT"
    return "NOT DETECTED"


def power_table(rates=(0.55, 0.60), n_per_reader=(400, 800), rounds=(1, 2, 3, 4, 5, 6),
                effects=(-0.08, -0.10), bar=0.05) -> list:
    """Detection threshold and power of the pooled rule. Readers' rates near p on both sides."""
    from statistics import NormalDist
    nd = NormalDist()
    rows = []
    for p in rates:
        for n in n_per_reader:
            se1 = math.sqrt(2 * p * (1 - p) / n)
            for R in rounds:
                se = se1 / math.sqrt(R)
                hw = Z * se
                thr = -(bar + hw)            # D must sit below this for the CI's top to clear -bar
                pw = {f"{e:+.2f}": nd.cdf((thr - e) / se) for e in effects}
                rows.append({"p": p, "n_per_reader": n, "R": R, "games_per_reader_total": n * R,
                             "half_width_pp": 100 * hw, "detect_if_D_below_pp": 100 * thr,
                             "power_at": pw})
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", action="append", default=[], metavar="LOOP_READER:CONTROL_READER")
    ap.add_argument("--bar", type=float, default=0.05, help="max(F', 0.05); F' from the round-0 pair")
    ap.add_argument("--expect-cycle-games", type=int, default=200)
    ap.add_argument("--expect-rounds", type=int, default=4)
    ap.add_argument("--standin", action="store_true",
                    help="mechanics check only: relax the cycle-games / round-count expectations")
    ap.add_argument("--power", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    if a.power:
        rows = power_table(bar=a.bar)
        print(f"POWER of the pooled rule (bar {100 * a.bar:.1f} pp; two readers per round at rate p)")
        print("  p     n/reader  R  half-width  detect iff D <   P(detect | -8)  P(detect | -10)")
        for r in rows:
            print(f"  {r['p']:.2f}  {r['n_per_reader']:8d}  {r['R']}  {r['half_width_pp']:9.2f}  "
                  f"{r['detect_if_D_below_pp']:+13.2f}   {r['power_at']['-0.08']:13.2f}  "
                  f"{r['power_at']['-0.10']:14.2f}")
        if a.json:
            with open(a.json, "w") as fh:
                json.dump(rows, fh, indent=2)
        return 0

    from agents.training.best_response_gap import (check_matched, load_teamsets, newcombe_diff_ci,
                                                   read_exploiter)
    if not a.pair:
        ap.error("give --pair LOOP_READER:CONTROL_READER per round, or --power")
    if not a.standin and len(a.pair) != a.expect_rounds:
        print(f"VOID: {len(a.pair)} round(s) given, the registration fixes R = {a.expect_rounds}")
        return 3
    ts = load_teamsets()
    pairs = [tuple(read_exploiter(x, ts) for x in p.split(":")) for p in a.pair]
    everyone = [r for p in pairs for r in p]
    mism = check_matched(everyone, allow_unmatched=True)
    if mism:
        for m in mism:
            print(f"VOID: unmatched {m.kind}: {m.a} vs {m.b}: {m.detail}")
        return 3
    rounds = []
    for i, (b, c) in enumerate(pairs, 1):
        if b.target_file == c.target_file:
            print(f"VOID: round {i}'s two readers share a target ({b.target_file})")
            return 3
        cyc = {p.games for r in (b, c) for p in r.post_fork}
        if not a.standin and cyc != {a.expect_cycle_games}:
            print(f"VOID: round {i} cycle sizes {sorted(cyc)} != registered {a.expect_cycle_games}")
            return 3
        wb, nb = sum(p.wins for p in b.post_fork), sum(p.games for p in b.post_fork)
        wc, nc = sum(p.wins for p in c.post_fork), sum(p.games for p in c.post_fork)
        pb, pc = wb / nb, wc / nc
        _, lo, hi = newcombe_diff_ci(wb, nb, wc, nc)
        var = pb * (1 - pb) / nb + pc * (1 - pc) / nc
        rounds.append({"round": i, "loop_reader": b.run, "control_reader": c.run,
                       "loop": [wb, nb], "control": [wc, nc], "d": pb - pc,
                       "newcombe_ci": [lo, hi], "var": var,
                       "curves": {"loop": [[p.step, p.wins, p.games] for p in b.post_fork],
                                  "control": [[p.step, p.wins, p.games] for p in c.post_fork]}})
    R = len(rounds)
    D = sum(r["d"] for r in rounds) / R
    se = math.sqrt(sum(r["var"] for r in rounds)) / R
    lo, hi = D - Z * se, D + Z * se
    w = [1 / r["var"] for r in rounds]
    d_iv = sum(wi * r["d"] for wi, r in zip(w, rounds)) / sum(w)
    Q = sum(wi * (r["d"] - d_iv) ** 2 for wi, r in zip(w, rounds))
    v = verdict(D, lo, hi, a.bar)
    print("POOLED READ (new-lineage registration §5.3)" + ("  — STAND-IN, a mechanics check, NOT a read"
                                                         if a.standin else ""))
    for r in rounds:
        print(f"  round {r['round']}: {r['loop_reader']} {r['loop'][0]}/{r['loop'][1]}  vs  "
              f"{r['control_reader']} {r['control'][0]}/{r['control'][1]}   d = {_pp(r['d'])} pp "
              f"[{_pp(r['newcombe_ci'][0])}, {_pp(r['newcombe_ci'][1])}] (Newcombe, descriptor)")
    print(f"  POOLED D = {_pp(D)} pp  [{_pp(lo)}, {_pp(hi)}]  (fixed-effect Wald, R = {R}, SE {100 * se:.2f})")
    print(f"  heterogeneity: Cochran Q = {Q:.2f} on {R - 1} df (descriptor)")
    print(f"  bar {100 * a.bar:.2f} pp  ->  VERDICT: {v}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump({"standin": a.standin, "bar": a.bar, "rounds": rounds, "D": D, "ci": [lo, hi],
                       "se": se, "cochran_q": Q, "verdict": v}, fh, indent=2)
        print(f"  wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
