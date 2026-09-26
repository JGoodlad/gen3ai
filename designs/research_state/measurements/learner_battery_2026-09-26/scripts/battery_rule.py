"""The LEARNER BATTERY's decision rule (registration §4), mechanised — plus the power table (§4.5).

    python battery_rule.py --power
    python battery_rule.py rule --arm E5 --kind speed --du -0.6 -2.9 1.7 --s 0.141 0.132 0.150 \
        --ga 0.012 -0.027 0.051 --stall-ratio 1.04
    python battery_rule.py rule --arm L95 --kind quality --du 4.1 1.6 6.4 --ga ... --stall-ratio ...

All strength numbers in PERCENTAGE POINTS, as (point, CI low, CI high):
  --du : untaught-8 Δ = U(arm final) − U(C final), the meter's own cluster bootstrap (§4.2)
  --ga : SmallRL greedy Δ = rate(arm) − rate(C), pooled 1,200 games each, Newcombe 95 % (§4.4); in pp
  --s  : the projected speed gain S (fraction, not pp) and its bootstrap CI (speed_read.py)
  --stall-ratio : [STALL LOGGED] per 1M steps, arm / C (§4.4)

Rule (bars fixed now): untaught bar 3.69 pp (the G-U replicate floor, carried provisionally from the
old lineage); speed bar S >= 0.05 with CI low > 0; G-A floor 11.0 pp; stall ratio <= 1.5.
  speed arm (E5, T32): ADOPT iff SPEED passes AND untaught NON-INFERIOR (CI low > −3.69) AND guards clean.
  quality arm (L95)  : ADOPT iff untaught OUTSIDE ABOVE (point > 3.69 AND CI low > 3.69) AND guards clean.
  guards: G-A is a KILL iff OUTSIDE BELOW (point < −11.0 AND CI high < −11.0); stall ratio > 1.5 is a KILL.
"""
from __future__ import annotations

import argparse
import math
from statistics import NormalDist

BAR_U = 3.69
BAR_S = 0.05
FLOOR_GA = 11.0
STALL_MAX = 1.5
Z = 1.959963984540054


def classify(d, lo, hi, bar):
    if abs(d) > bar and (hi < -bar if d < 0 else lo > bar):
        return "OUTSIDE, BELOW" if d < 0 else "OUTSIDE, ABOVE"
    if lo > -bar and hi < bar:
        return "EQUIVALENT"
    return "NOT DETECTED"


def rule(a) -> int:
    du, dlo, dhi = a.du
    u_cls = classify(du, dlo, dhi, BAR_U)
    ni = dlo > -BAR_U
    kills = []
    if a.ga is not None:
        g, glo, ghi = a.ga
        if g < -FLOOR_GA and ghi < -FLOOR_GA:
            kills.append(f"G-A OUTSIDE BELOW the {FLOOR_GA} pp floor ({g:+.2f} [{glo:+.2f}, {ghi:+.2f}])")
    else:
        kills.append("G-A NOT READ — the rule cannot adopt without it")
    if a.stall_ratio is None or a.stall_ratio > STALL_MAX:
        kills.append(f"stall ratio {a.stall_ratio} > {STALL_MAX} (or not read)")
    print(f"== {a.arm} ({a.kind})  untaught Δ vs C = {du:+.2f} [{dlo:+.2f}, {dhi:+.2f}] pp -> {u_cls}"
          f"; non-inferior at −{BAR_U}: {'YES' if ni else 'NO'}")
    if a.kind == "speed":
        s, slo, shi = a.s
        s_ok = s >= BAR_S and slo > 0
        print(f"   speed S = {100 * s:+.2f} % [{100 * slo:+.2f}, {100 * shi:+.2f}] -> "
              f"{'PASSES' if s_ok else 'FAILS'} the {100 * BAR_S:.0f} % bar")
        adopt = s_ok and ni and not kills
        why = ("speed passes, per-step non-inferior, guards clean" if adopt else
               "; ".join(([] if s_ok else ["speed below the bar"]) +
                         ([] if ni else [f"per-step non-inferiority not shown ({u_cls})"]) + kills))
    else:
        adopt = u_cls == "OUTSIDE, ABOVE" and not kills
        why = ("untaught OUTSIDE ABOVE, guards clean" if adopt else
               "; ".join(([] if u_cls == "OUTSIDE, ABOVE" else [f"untaught {u_cls} (a change needs "
                                                               "evidence; the status quo holds)"]) + kills))
    print(f"   VERDICT: {'ADOPT' if adopt else 'NOT ADOPTED'} — {why}")
    return 0


def power() -> int:
    nd = NormalDist()
    print("Untaught-8 Δ between two refs, Wald on game noise alone (conservative vs the meter's cluster")
    print("bootstrap, which read half-widths of 2.2–2.9 pp at 200 games/team in the old lineage).")
    print(f"{'games/team':>10} {'per ref':>8} {'SE pp':>6} {'hw pp':>6} | NON-INFERIORITY (CI low > −3.69) "
          f"P(pass | true Δ) | SUPERIORITY P(OUTSIDE ABOVE | true Δ) | EQUIV P(| 0)")
    for gpt in (200, 400, 600):
        n = 8 * gpt
        for p in (0.60,):
            se = 100 * math.sqrt(2 * p * (1 - p) / n)
            hw = Z * se
            ni = {d: 1 - nd.cdf((-BAR_U + hw - d) / se) for d in (0.0, -1.0, -2.0, -3.69)}
            sup = {d: 1 - nd.cdf((BAR_U + hw - d) / se) for d in (4.0, 6.0, 8.0, 10.0)}
            eq = nd.cdf((BAR_U - hw) / se) - nd.cdf((-BAR_U + hw) / se) if hw < BAR_U else 0.0
            print(f"{gpt:>10} {n:>8} {se:6.2f} {hw:6.2f} | "
                  + " ".join(f"{d:+.2f}:{v:.2f}" for d, v in ni.items()) + " | "
                  + " ".join(f"{d:+.0f}:{v:.2f}" for d, v in sup.items()) + f" | {eq:.2f}")
    print("\nSmallRL G-A guard, 1,200 games per model at p ~ 0.5: Δ SE "
          f"{100 * math.sqrt(2 * 0.25 / 1200):.2f} pp, hw {Z * 100 * math.sqrt(2 * 0.25 / 1200):.2f}; "
          f"the KILL needs Δ < −{FLOOR_GA} with the CI's top below it, i.e. a point below "
          f"≈ −{FLOOR_GA + Z * 100 * math.sqrt(2 * 0.25 / 1200):.1f} pp — a guard against a gross loss only.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--power", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    r = sub.add_parser("rule")
    r.add_argument("--arm", required=True)
    r.add_argument("--kind", choices=("speed", "quality"), required=True)
    r.add_argument("--du", type=float, nargs=3, required=True)
    r.add_argument("--s", type=float, nargs=3, default=None)
    r.add_argument("--ga", type=float, nargs=3, default=None)
    r.add_argument("--stall-ratio", type=float, default=None)
    a = ap.parse_args(argv)
    if a.power:
        return power()
    if a.cmd == "rule":
        if a.kind == "speed" and a.s is None:
            ap.error("a speed arm needs --s")
        return rule(a)
    ap.error("--power or rule")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
