"""vf_coef's logratio under ALL THREE FRAMINGS, with the REGISTERED one named.

    python -m main.ops.vf_framings <run-name|run-dir>

The registered statistic (ledger cfc72ad0) is the MEDIAN OF THE LAST 20 ROLLOUTS. It is
unchanged and it is what the verdict rests on. The other two framings are printed beside it
because that window can STRADDLE an opponent-regime boundary (train/selfplay_fraction 0.0 ->
0.9), and a reader is entitled to see whether the verdict depends on the straddle.

The straddle is SELF-CLEARING: 20 rollouts is ~2M steps, so the window sits fully inside the
new regime about 2M steps after the boundary.

Bar: |med| <= 0.5 KEEP · 0.5 < |med| < 1.0 KEEP+FLAG, re-read at the 2nd restart ·
     |med| >= 1.0 NEW ARM at --vf-coef 0.5*10^-med. vf_coef is RESUME-IMMUTABLE, so a NEW ARM
     verdict means a new run, never a change to this one. Never act on a single sample.

Promoted 2026-09-07 from the Training Run session's ``vf_framings.py``.
"""
from __future__ import annotations

import statistics as st
import sys
from typing import Sequence

from main.ops.run_ref import event_dirs, resolve_run_dir

COMPONENT_TAGS = ["grad/value_norm_shared", "grad/policy_norm_shared", "grad/value_share",
                  "train/value_loss", "win_prob/critic_brier"]


def verdict(m: float) -> str:
    a = abs(m)
    if a <= 0.5:
        return "KEEP vf_coef 0.5"
    if a < 1.0:
        return "KEEP + FLAG, re-read at the 2nd restart"
    return f"NEW ARM at --vf-coef {0.5*(10**-m):.4g} (resume-immutable: a NEW RUN)"


def main(argv: Sequence[str] | None = None) -> int:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if args else 2
    run = resolve_run_dir(args[0])
    subs = event_dirs(run)
    lr, sp = [], []
    for d in subs:
        ea = EventAccumulator(d, size_guidance={"scalars": 0})
        ea.Reload()
        t = ea.Tags()["scalars"]
        if "grad/value_policy_logratio" in t:
            lr += [(e.step, e.value) for e in ea.Scalars("grad/value_policy_logratio")]
        if "train/selfplay_fraction" in t:
            sp += [(e.step, e.value) for e in ea.Scalars("train/selfplay_fraction")]
    lr, sp = sorted(set(lr)), sorted(set(sp))
    if not lr:
        print("NO SCALAR grad/value_policy_logratio  <- absence is not a zero")
        return 2

    regs = []
    for s, v in sp:
        if not regs or v != regs[-1][1]:
            regs.append((s, v))
    cur_start, cur_val = (regs[-1] if regs else (0, None))

    print("=== vf_coef  grad/value_policy_logratio  (ledger cfc72ad0; standing rule 15 = "
          "ledger 62d4948e) ===")
    print(f"  run {run}")
    if regs:
        print("  regimes: " + " -> ".join(f"{v:.2f}@{s:,}" for s, v in regs))
        print(f"  CURRENT regime {cur_val:.2f} since step {cur_start:,}")

    last20 = [v for _, v in lr[-20:]]
    w_lo = lr[-20][0] if len(lr) >= 20 else lr[0][0]
    straddles = cur_val is not None and w_lo < cur_start
    m1 = st.median(last20)
    print(f"\n  [REGISTERED]  median of the LAST 20 rollouts (steps {w_lo:,}..{lr[-1][0]:,})")
    print(f"                {m1:+.3f} log10 = {10**m1:.2f}x   -> {verdict(m1)}")
    print(f"                window spans {min(last20):+.3f}..{max(last20):+.3f}"
          f"  ({10**max(last20)/10**min(last20):.1f}x end to end)")
    print(f"                {'STRADDLES the regime boundary' if straddles else 'fully inside the current regime'}")

    post = [v for s, v in lr if s >= cur_start] if cur_val is not None else []
    if post:
        m2 = st.median(post)
        print(f"\n  [context]     current regime only ({cur_val:.2f}, steps >= {cur_start:,}), "
              f"n={len(post)}")
        print(f"                {m2:+.3f} log10 = {10**m2:.2f}x   -> {verdict(m2)}")

    b = {}
    for s, v in lr:
        b.setdefault(s // 1_000_000, []).append(v)
    k = max(b)
    m3 = st.median(b[k])
    print(f"\n  [context]     latest 1M bucket ({k}-{k+1}M), n={len(b[k])}")
    print(f"                {m3:+.3f} log10 = {10**m3:.2f}x   -> {verdict(m3)}")

    print("\n  per-1M buckets:")
    for kk in sorted(b):
        mm = st.median(b[kk])
        # A bucket is tagged by where it SITS relative to the boundary, not by its start
        # alone: a bucket CONTAINING the boundary step was once called pool-empty by a
        # start-only test. A bucket that spans the boundary is neither regime and says so.
        lo_b, hi_b = kk * 1_000_000, (kk + 1) * 1_000_000
        if cur_val is None:
            tag = ""
        elif lo_b >= cur_start:
            tag = f"  <- regime {cur_val:.2f}"
        elif hi_b > cur_start:
            tag = f"  <- STRADDLES the boundary ({cur_start:,})"
        else:
            tag = "  <- pool-empty"
        print(f"    {kk}-{kk+1}M  {mm:+.3f}  = {10**mm:5.2f}x   n={len(b[kk])}{tag}")

    if cur_val is not None:
        pre = [v for s, v in lr if s < cur_start]
        if pre and post:
            print(f"\n  FINDING (UNVERIFIED cause): the ratio SHIFTED "
                  f"{st.median(post)-st.median(pre):+.3f} log10 "
                  f"({10**st.median(pre)/10**st.median(post):.1f}x) at the regime boundary --")
            print(f"    pool-empty median {st.median(pre):+.3f} (n={len(pre)}) -> "
                  f"self-play median {st.median(post):+.3f} (n={len(post)}).")
            print("    The value gradient shrank relative to the policy gradient once the opponent")
            print("    became a neural snapshot. This is a property of the ARM, not of the window.")
            print("    Cause UNVERIFIED -- no explanation is asserted here.")
            print("    It is also why the +/-0.5 band should be read PER REGIME from here: the band")
            print("    was set on pool-empty data. Standing rule 15, ledger 62d4948e.")
    print("\n  The straddle is SELF-CLEARING: the 20-rollout window sits fully inside the")
    print("  new regime about 2M steps after the boundary.")

    # ── COMPONENTS (registered at restart 2): a RATIO WITHOUT ITS COMPONENTS is what let a
    # walk in the DENOMINATOR look like the critic's doing. Print them at every read, per regime.
    C: dict = {}
    for d in subs:
        ea = EventAccumulator(d, size_guidance={"scalars": 0})
        ea.Reload()
        tg = ea.Tags()["scalars"]
        for t in COMPONENT_TAGS:
            if t in tg:
                C.setdefault(t, []).extend((e.step, e.value) for e in ea.Scalars(t))
    miss = [t for t in COMPONENT_TAGS if t not in C]
    print("\n  COMPONENTS — the ratio is value_norm / policy_norm; watch the DENOMINATOR")
    if miss:
        print(f"    MISSING: {miss}  <- absence is not a zero")
    CB: dict = {}
    for t, v in C.items():
        for s_, x in sorted(set(v)):
            CB.setdefault(t, {}).setdefault(s_ // 1_000_000, []).append(x)
    kk_all = sorted({k for t in CB for k in CB[t]})
    print(f"    {'bucket':<9}{'val_norm':>9}{'pol_norm':>9}{'val_share':>10}{'val_loss':>9}"
          f"{'brier':>8}  regime")
    for k in kk_all:
        cells = []
        for t in COMPONENT_TAGS:
            bb = CB.get(t, {}).get(k)
            cells.append(f"{st.median(bb):>9.3f}" if bb else "        -")
        reg = ("pool-empty" if (k + 1) * 1_000_000 <= cur_start
               else ("STRADDLE" if k * 1_000_000 < cur_start else f"{cur_val:.2f}"
                     if cur_val is not None else "?"))
        print(f"    {k}-{k+1}M    " + "".join(cells[:3]) + "".join(cells[3:]) + f"  {reg}")
    print("    READ THIS FIRST: a falling ratio with a FLAT value_loss and FLAT brier is the")
    print("    POLICY term growing, not the critic sharpening. A sharpening critic would show")
    print("    loss and brier FALLING with it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
