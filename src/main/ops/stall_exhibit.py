"""THE STALL EXHIBIT — ep_len and draw_rate per 1M bucket, on the SAME rows.

    python -m main.ops.stall_exhibit <run-name|run-dir> [--draw-bar 0.01]

A stall is a MONOTONE ep_len rise WITH draw rate rising alongside. A competence sawtooth is
ep_len jumping at each pool promotion and decaying between, while draw rate stays flat or low.
One table with both columns is what separates them; two tables in two messages is not.

🚨 EVERY DIRECTION WORD HERE IS DERIVED FROM THE TWO WINDOW MEDIANS, NEVER A LITERAL.
An earlier version of this check printed "draw_rate falling" from a HARDCODED label while the
numbers underneath read 0.0024 -> 0.0031 (rising). Any "falling" read of draw_rate reported
before 2026-09-07 00:15 came from that literal and is KNOWN-UNRELIABLE.

Promoted 2026-09-07 from the Training Run session's ``stall_exhibit.py``.
"""
from __future__ import annotations

import os
import statistics as st
import sys
from typing import Sequence

from main.ops.run_ref import event_dirs, resolve_run_dir

DEFAULT_DRAW_BAR = 0.01


def direction(a: float, b: float, eps: float = 0.0) -> str:
    """DERIVED from the two values. Never a literal."""
    if b > a + eps:
        return "rising"
    if b < a - eps:
        return "falling"
    return "flat"


def main(argv: Sequence[str] | None = None) -> int:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if args else 2
    run = resolve_run_dir(args[0])
    draw_bar = float(args[args.index("--draw-bar") + 1]) if "--draw-bar" in args \
        else DEFAULT_DRAW_BAR

    ep, dr, sp = [], [], []
    for d in event_dirs(run):
        ea = EventAccumulator(d, size_guidance={"scalars": 0})
        ea.Reload()
        t = ea.Tags()["scalars"]
        if "rollout/ep_len_mean" in t:
            ep += [(e.step, e.value) for e in ea.Scalars("rollout/ep_len_mean")]
        if "signal/draw_rate" in t:
            dr += [(e.step, e.value) for e in ea.Scalars("signal/draw_rate")]
        if "train/selfplay_fraction" in t:
            sp += [(e.step, e.value) for e in ea.Scalars("train/selfplay_fraction")]
    ep, dr, sp = sorted(set(ep)), sorted(set(dr)), sorted(set(sp))
    if not ep:
        print("NO SCALAR rollout/ep_len_mean  <- absence is not a zero")
        return 2

    regs = []
    for s, v in sp:
        if not regs or v != regs[-1][1]:
            regs.append((s, v))
    cur_start = regs[-1][0] if regs else 0

    promos = set()
    snapdir = run / "snapshots"
    if snapdir.is_dir():
        for f in os.listdir(snapdir):
            if f.endswith(".zip"):
                dg = "".join(c for c in f if c.isdigit())
                if dg:
                    promos.add(int(dg) // 1_000_000)

    be, bd = {}, {}
    for s, v in ep:
        be.setdefault(s // 1_000_000, []).append(v)
    for s, v in dr:
        bd.setdefault(s // 1_000_000, []).append(v)

    print("=== STALL EXHIBIT: ep_len and draw_rate on the SAME rows ===")
    print(f"  run {run}")
    print("  a STALL = monotone ep_len rise WITH draw rate rising alongside")
    print("  a COMPETENCE SAWTOOTH = ep_len jumps at each pool promotion, decays between\n")
    print(f"  {'bucket':<10} {'ep_len':>7}  {'draw_rate':>10}   n   note")
    for k in sorted(be):
        e = st.mean(be[k])
        d = st.median(bd[k]) if k in bd else float("nan")
        note = []
        if k in promos:
            note.append("POOL PROMOTION")
        if k * 1_000_000 < cur_start <= (k + 1) * 1_000_000:
            note.append("regime step")
        elif (k + 1) * 1_000_000 <= cur_start:
            note.append("pool-empty")
        print(f"  {k}-{k+1}M{'':<4} {e:7.2f}  {d:10.4f}  {len(be[k]):2d}   {' · '.join(note)}")

    sp_ks = [k for k in sorted(be) if k * 1_000_000 >= cur_start]
    if len(sp_ks) >= 2:
        e0, e1 = st.mean(be[sp_ks[0]]), st.mean(be[sp_ks[-1]])
        d0v, d1v = st.median(bd[sp_ks[0]]), st.median(bd[sp_ks[-1]])
        print(f"\n  across the self-play regime ({sp_ks[0]}-{sp_ks[0]+1}M -> "
              f"{sp_ks[-1]}-{sp_ks[-1]+1}M):")
        print(f"    ep_len    {e0:.2f} -> {e1:.2f}   {direction(e0, e1)}")
        print(f"    draw_rate {d0v:.4f} -> {d1v:.4f}   {direction(d0v, d1v)}   [both words DERIVED]")
        # monotonicity: a stall's ep_len rise is monotone, a sawtooth's is not
        seq = [st.mean(be[k]) for k in sp_ks]
        ups = sum(1 for a, b in zip(seq, seq[1:]) if b > a)
        print(f"    ep_len monotone? {ups}/{len(seq)-1} bucket steps up "
              f"-> {'MONOTONE' if ups == len(seq)-1 else 'NOT monotone (sawtooth)'}")
        print(f"    draw_rate vs the bar {draw_bar}: {d1v/draw_bar:.0%} of it")
        both = (direction(e0, e1) == "rising" and direction(d0v, d1v) == "rising"
                and ups == len(seq) - 1)
        print(f"\n  STALL SIGNATURE (monotone ep_len rise AND draw rate rising): "
              f"{'PRESENT' if both else 'ABSENT'}")
    else:
        print("\n  fewer than 2 buckets inside the current regime — the across-regime")
        print("  comparison is NOT COMPUTED. This is not an ABSENT signature.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
