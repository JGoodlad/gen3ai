"""THE BOT-SIDE CALIBRATION DECAY — printed per read (registered 2026-09-07, ledger 18bfc1c0).

    python -m main.ops.calib_trend <critic_gate-report.txt>

G1's resolution and G4's skill, per checkpoint, BOT and POOL kept apart. On the arm this was
written for the bot stratum's resolution has decayed while pool holds, and that split is
invisible in the 'all' row and in the tool's across-steps aggregate verdict.

🚨 THE AGGREGATE VERDICT IS NOT THE PER-STEP TRUTH. critic_gate's "criteria not met: G2, G3"
is an ACROSS-STEPS aggregate. At one step specifically both GATED strata can PASS (bot: point
<= base on reliability and ECE; pool: CI covers base) while only the ungated 'all' row fails.
State the per-step truth beside the aggregate, never instead of it.

Reads a critic_gate TEXT REPORT — the file, not the run. Redirect `python -m main.critic_gate
<run> …` to a file and pass that here.

Promoted 2026-09-07 from the Training Run session's ``calib_trend.py``. ONE LITERAL REMOVED and
recorded here: the session copy closed with the fixed sentence "The decay is BOT-SIDE. Pool
holds." — printed unconditionally, whatever the two strata's endpoints said. That is the same
defect the sibling stall exhibit carries a 🚨 about (a direction word from a literal rather than
from the numbers), and it had already produced one known-unreliable "falling" read there. The
numbers above it are UNCHANGED; only the pre-written conclusion is gone.
"""
from __future__ import annotations

import os
import re
import sys
from typing import Sequence


def parse(path: str):
    """The (step, stratum, battles, resolution, base, skill, G1, G4) rows of the report."""
    rows = []
    sect = False
    for line in open(path):
        if "CALIBRATION GATE" in line:
            sect = True
            continue
        if sect and "G2/G3" in line:
            break
        if sect:
            f = line.split()
            if len(f) >= 9 and f[1] in ("bot", "pool") and re.match(r"^[\d,]+$", f[0]):
                # 0-indexed: 0=step 1=stratum 2=btl 3=resolution 4=[CI] 5=base 6=skill 7=G1 8=G4
                rows.append((int(f[0].replace(",", "")), f[1], f[2], f[3], f[5], f[6], f[7],
                             f[8] if len(f) > 8 else "?"))
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if args else 2
    path = args[0]
    if not os.path.exists(path):
        print(f"REFUSING: no such report file: {path}")
        return 2
    rows = parse(path)
    if not rows:
        print("  NO calibration rows parsed  <- absence is not a zero")
        return 2

    print("  G1 RESOLUTION (primary) and G4 SKILL, bot vs pool — NEVER pooled")
    print(f"    {'step':>12}  {'btl':>4}  {'bot res':>8} {'bot skill':>9} {'G1':>3}{'G4':>3}   "
          f"{'pool res':>8} {'pool skill':>10} {'G1':>3}{'G4':>3}")
    by: dict = {}
    for st_, stratum, btl, res, base, skill, g1, g4 in rows:
        by.setdefault(st_, {})[stratum] = (btl, res, skill, g1, g4)
    for st_ in sorted(by):
        b = by[st_].get("bot")
        p = by[st_].get("pool")
        bs = f"{b[1]:>8} {b[2]:>9} {b[3]:>3}{b[4]:>3}" if b else " " * 24
        ps = f"{p[1]:>8} {p[2]:>10} {p[3]:>3}{p[4]:>3}" if p else ""
        print(f"    {st_:>12,}  {(b or p)[0]:>4}  {bs}   {ps}")

    bots = [(s, float(v["bot"][1]), float(v["bot"][2])) for s, v in sorted(by.items())
            if "bot" in v]
    pools = [(s, float(v["pool"][1]), float(v["pool"][2])) for s, v in sorted(by.items())
             if "pool" in v]
    if len(bots) >= 2 and len(pools) >= 2:
        print(f"\n  BOT  resolution {bots[0][1]:.4f} -> {bots[-1][1]:.4f}   "
              f"skill {bots[0][2]:+.3f} -> {bots[-1][2]:+.3f}")
        print(f"  POOL resolution {pools[0][1]:.4f} -> {pools[-1][1]:.4f}   "
              f"skill {pools[0][2]:+.3f} -> {pools[-1][2]:+.3f}")
        print("  Read the two strata SEPARATELY — a divergence here is a POPULATION difference,")
        print("  not a single-metric artefact. No account is asserted.")
    else:
        print("\n  fewer than 2 steps in one of the strata — the endpoints are NOT COMPARED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
