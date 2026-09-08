"""Evaluate an arm's kill bars from TENSORBOARD. READ AMENDMENT 3 + 4.

    python -m main.ops.killbar <run-name|run-dir> [--bar 5M|10M] [--freeze]
                               [--frozen-refs PATH]

CLAUSE 2 IS A FIXED EFFECT SIZE WITHIN ONE OPPONENT REGIME.
  AMENDMENT 3 (ledger 0e2bacc7): "ep_len rising" = test mean - frozen reference >= +3.0 turns.
  AMENDMENT 4 (ledger 38bfca94): the reference AND the test window must lie ENTIRELY inside the
  same opponent regime, where a REGIME is a constant value of train/selfplay_fraction.

WHY 4 EXISTS. Amendment 3's reference was the 1-3M window -- the POOL-EMPTY regime, every
battle against scripted bots. selfplay_fraction stepped 0.0 -> 0.9 at 4.00M and per-1M ep_len
means went  1-2M -0.05 · 2-3M +0.05 · 3-4M -1.06 · 4-5M +3.59  against that reference. The
4-5M bucket ALREADY CLEARS the +3.0 bar. At 10M the test window (8-10M) is entirely self-play,
so clause 2 would have read MET permanently for a reason with nothing to do with famine, and
the gate would have collapsed onto clause 1 alone -- the very failure the fixed effect size
was adopted to prevent. Two competent neural players simply take longer to resolve a game
than a scripted bot does.

THE RULE, applied automatically here so the next regime step does not need anyone to notice:
  reference     mean of rollout/ep_len_mean over the FIRST 2M-step window lying fully inside
                the CURRENT regime, frozen ONCE per regime and named with the regime it was
                measured in. A frozen entry is never rewritten.
  test window   the last 2M before the read.
  NOT EVALUABLE when: the reference window for the current regime is not yet complete; a
                regime step falls INSIDE the test window; or reference and test sit in
                different regimes. Then clause 1 and the famine read decide, and the report
                SAYS SO. Declared in advance, never decided at the read.
  bar           +3.0 turns. Welch 95% CI reported beside it, descriptively.

TWO THINGS EVERY REPORT USING THIS MUST STATE (registered with the amendment):
  DIRECTION -- this is again HARDER to fire and favours the arm being run. It was registered
  before 8M, i.e. before the data it judges existed.
  RESIDUAL CONFOUND -- under self-play, COMPETENCE ITSELF lengthens games, and the pool gains
  a stronger snapshot every 2M. So a genuine +3.0 inside the self-play regime can still be
  improvement rather than stall. That is why the AND with draw rate is the load-bearing
  structure, and why the famine verdict is the LADDER, never clause 2 alone.

WHERE THE FROZEN REFERENCES LIVE. Beside the RUN (``<run>/ops_frozen_refs.json``), not beside
this module. A reference is a fact about one arm's opponent regime; a repo-level file would let
two arms overwrite each other's, and this tool's whole contract is that a frozen entry is never
rewritten. ``--frozen-refs PATH`` moves it. It is only ever WRITTEN under an explicit
``--freeze``; a read never creates it.

Promoted 2026-09-07 from the Training Run session's ``killbar.py``.
"""
from __future__ import annotations

import datetime
import json
import math
import os
import statistics as st
import sys
from pathlib import Path
from typing import Sequence

from main.ops.run_ref import event_dirs, resolve_run_dir

RISING_BAR_TURNS = 3.0
WINDOW = 2_000_000
FROZEN_BASENAME = "ops_frozen_refs.json"
# The pool-empty reference, kept in the record and NEVER used against self-play data.
POOL_EMPTY_REF = {"regime": 0.0, "mean": 30.0395,
                  "window": "1,081,344..2,949,120", "n": 20,
                  "note": "AMENDMENT 3 reference; superseded for self-play by AMENDMENT 4"}


# ── PURE CLAUSE LOGIC + PLANTED-CASE SELF-CHECK ────────────────────────────────────────────
# Registered 2026-09-07 after the sibling plateau tool's clause 1 false-fired: a broken clause
# can ride inside correct verdicts indefinitely when it is ANDed with a second one. The fix is
# not a better condition, it is a PLANTED CASE PER CLAUSE run before every real read.

def clause1_draw_rate_met(median_draw_rate: float, threshold: float) -> bool:
    """True iff the draw-rate clause is MET (>= its bar)."""
    return median_draw_rate >= threshold


def clause2_rising(test_mean: float, reference: float, bar: float = RISING_BAR_TURNS) -> bool:
    """True iff the FIXED EFFECT SIZE is met: test mean - frozen reference >= bar turns."""
    return (test_mean - reference) >= bar


def clause2_evaluable(latest_step: int, regime_start: int, window: int, ref_frozen: bool) -> bool:
    """AMENDMENT 4: evaluable only if the reference window is complete, no regime step falls
    inside the test window, and a reference has been frozen for THIS regime."""
    if not ref_frozen:
        return False
    if latest_step < regime_start + window:      # reference window not complete
        return False
    if (latest_step - window) < regime_start:    # a regime step falls inside the test window
        return False
    return True


def self_check() -> None:
    fails = []
    # clause 1 — only the threshold decides
    if clause1_draw_rate_met(0.009, 0.01):
        fails.append("clause1: 0.009 read as MET against a 0.01 bar")
    if not clause1_draw_rate_met(0.011, 0.01):
        fails.append("clause1: 0.011 not read as MET against a 0.01 bar")
    # clause 2 — only the effect size decides, against the FROZEN reference
    if clause2_rising(34.0, 30.0) is not True:
        fails.append("clause2: +4.0 turns not read as RISING against a +3.0 bar")
    if clause2_rising(31.0, 30.0):
        fails.append("clause2: +1.0 turns read as RISING against a +3.0 bar")
    if clause2_rising(29.0, 30.0):
        fails.append("clause2: a FALLING series read as RISING")
    # evaluability — each refusal condition must bite on its own
    if clause2_evaluable(5_000_000, 4_000_000, 2_000_000, True):
        fails.append("evaluability: an INCOMPLETE reference window read as evaluable")
    if clause2_evaluable(6_500_000, 4_000_000, 2_000_000, True) is not True:
        fails.append("evaluability: a complete, non-straddling window read as NOT evaluable")
    if clause2_evaluable(6_500_000, 6_000_000, 2_000_000, True):
        fails.append("evaluability: a regime step INSIDE the test window read as evaluable")
    if clause2_evaluable(9_000_000, 4_000_000, 2_000_000, False):
        fails.append("evaluability: NO frozen reference read as evaluable")
    if fails:
        print("=== KILL BAR — REFUSING: SELF-CHECK FAILED ===")
        for f in fails:
            print(f"  {f}")
        print("  The clause logic does not behave on planted cases; the real read cannot be")
        print("  trusted. Nothing is concluded.")
        raise SystemExit(3)


def series(run_dir: Path, tags: Sequence[str]) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    out: dict = {}
    for d in event_dirs(run_dir):
        ea = EventAccumulator(d, size_guidance={"scalars": 0})
        ea.Reload()
        for t in tags:
            if t in ea.Tags()["scalars"]:
                out.setdefault(t, []).extend((e.step, e.value) for e in ea.Scalars(t))
    return {t: sorted(set(v)) for t, v in out.items()}


def regimes(sp):
    """[(start_step, value)] -- one entry per distinct selfplay_fraction value, in order."""
    out = []
    for s, v in sp:
        if not out or v != out[-1][1]:
            out.append((s, v))
    return out


def load_frozen(path: Path) -> dict:
    return json.load(open(path)) if os.path.exists(path) else {}


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if args else 2
    self_check()

    run = resolve_run_dir(args[0])
    bar = args[args.index("--bar") + 1] if "--bar" in args else "5M"
    do_freeze = "--freeze" in args
    frozen_path = Path(args[args.index("--frozen-refs") + 1]) if "--frozen-refs" in args \
        else run / FROZEN_BASENAME
    thresh = 0.03 if bar == "5M" else 0.01

    s = series(run, ["rollout/ep_len_mean", "signal/draw_rate",
                     "reward/untracked_abs_mean", "train/selfplay_fraction"])
    ep, dr, sp = (s.get("rollout/ep_len_mean", []), s.get("signal/draw_rate", []),
                  s.get("train/selfplay_fraction", []))
    if not ep or not dr:
        print("NO SCALAR for ep_len_mean and/or draw_rate  <- absence is not a zero")
        return 2

    latest = ep[-1][0]
    lo = latest - WINDOW
    regs = regimes(sp) if sp else []
    cur_start, cur_val = (regs[-1] if regs else (0, None))

    print(f"=== {bar} KILL BAR (AND-gate; BOTH clauses must hold) ===")
    print(f"    run {run}")
    print(f"    latest step {latest:,}")
    print("    clause 2: READ AMENDMENT 3 (ledger 0e2bacc7) + AMENDMENT 4 (ledger 38bfca94) "
          "-- fixed effect size")
    print(f"              of +{RISING_BAR_TURNS} turns, WITHIN ONE OPPONENT REGIME")
    if regs:
        print("    regimes (train/selfplay_fraction): "
              + " -> ".join(f"{v:.2f}@{s0:,}" for s0, v in regs))
        print(f"    CURRENT regime {cur_val:.2f} since step {cur_start:,}")

    d_test = [v for st_, v in dr if st_ > lo]
    d_med = st.median(d_test)
    c1 = clause1_draw_rate_met(d_med, thresh)      # the SELF-CHECKED path
    print(f"\n  clause 1  draw_rate median {d_med:.4f} over the last {WINDOW/1e6:.0f}M"
          f"  (max {max(d_test):.4f})   bar >= {thresh}")
    print(f"            {'MET' if c1 else 'not met'} -- {d_med/thresh:.0%} of the bar")

    # --- clause 2, with the AMENDMENT 4 evaluability gate ---------------------------------
    frozen = load_frozen(frozen_path)
    key = f"selfplay_{cur_val:.2f}" if cur_val is not None else "unknown"
    ref_end = cur_start + WINDOW
    reasons = []
    if cur_val is None:
        reasons.append("no train/selfplay_fraction scalar -- regime unknown")
    else:
        if latest < ref_end:
            reasons.append(f"the reference window for regime {cur_val:.2f} "
                           f"({cur_start:,}..{ref_end:,}) is NOT COMPLETE "
                           f"(latest {latest:,})")
        if lo < cur_start:
            reasons.append(f"a regime step falls INSIDE the test window "
                           f"(window opens {lo:,}, regime {cur_val:.2f} starts {cur_start:,})")

    ref_entry = frozen.get(key)
    if not reasons and ref_entry is None:
        reasons.append(f"no frozen reference for regime {cur_val:.2f} "
                       f"(run with --freeze once the window is complete; refs at {frozen_path})")

    print("\n  clause 2  ", end="")
    if reasons:
        print("NOT EVALUABLE")
        for r in reasons:
            print(f"              - {r}")
        print("            => clause 1 and the FAMINE READ decide. Declared in advance by")
        print("               AMENDMENT 4, not chosen at the read.")
        c2 = False
        if cur_val is not None and latest >= ref_end:
            w = [v for st_, v in ep if cur_start <= st_ < ref_end]
            if w:
                print(f"            (for information, the regime-{cur_val:.2f} reference window "
                      f"would be {st.mean(w):.4f}, n={len(w)})")
    else:
        REF = ref_entry["mean"]
        test = [v for st_, v in ep if st_ > lo]
        tmean = st.mean(test)
        delta = tmean - REF
        c2 = clause2_rising(tmean, REF)               # the SELF-CHECKED path
        print(f"reference {REF:.4f} turns  [regime {ref_entry['regime']:.2f}, "
              f"window {ref_entry['window']}, n={ref_entry['n']}, frozen {ref_entry['frozen_on']}]")
        print(f"            test window (steps > {lo:,}): mean {tmean:.2f}, n={len(test)}")
        print(f"            delta {delta:+.2f} turns   bar >= +{RISING_BAR_TURNS}")
        print(f"            {'RISING -- clause 2 MET' if c2 else 'NOT RISING'}"
              f"  ({delta/RISING_BAR_TURNS:.0%} of the bar)")
        refv = [v for st_, v in ep if cur_start <= st_ < ref_end]
        if len(refv) > 1 and len(test) > 1:
            va, vb = st.variance(refv), st.variance(test)
            se = math.sqrt(va / len(refv) + vb / len(test))
            l, h = delta - 2 * se, delta + 2 * se
            note = "  (CI includes zero -- descriptive; the bar is the point estimate)" \
                   if l <= 0 <= h else ""
            print(f"            Welch 95% CI [{l:+.2f}, {h:+.2f}]{note}")

    print(f"\n  for the record, the POOL-EMPTY (regime 0.00) reference is "
          f"{POOL_EMPTY_REF['mean']:.4f} turns")
    print(f"    ({POOL_EMPTY_REF['note']}) -- it is NOT used against self-play data")

    unt = s.get("reward/untracked_abs_mean", [])
    u_ok = bool(unt) and max(abs(v) for _, v in unt[-20:]) == 0.0
    print(f"\n  precondition  reward/untracked_abs_mean == 0 over last 20: "
          f"{'ok' if u_ok else 'VIOLATED' if unt else 'NO SCALAR <- not a zero'}")

    fires = c1 and c2
    # READ AMENDMENT 5 (ledger 08de50d4): this AND-gate was the bar of TWO reads -- the 5M
    # smoke and the 10M read. Both have been taken and passed, so the gate is DISCHARGED,
    # not standing. The NUMBERS above still matter and are still printed; the GATE VERDICT
    # is retired, because a train draw_rate crossing 0.01 tomorrow cannot fire a kill on its
    # own. THE STANDING KILL IS G7 (eval stall_rate <= 0.05, ep_len <= 1.25x era), which
    # `python -m main.critic_gate` computes at every snapshot. This tool does not compute it.
    print("\n  => GATE DISCHARGED after its two reads (READ AMENDMENT 5, ledger 08de50d4)")
    print(f"     numbers stand, verdict retired: clause1={'MET' if c1 else 'no'}, "
          f"clause2={'MET' if c2 else 'no/NOT EVALUABLE'}")
    print("     THE STANDING KILL IS G7 — see `python -m main.critic_gate`.")
    print("     MONITOR (report, not kill): train draw_rate > 0.01 in TWO consecutive 1M buckets.")
    print("\n  DIRECTION: AMENDMENT 4 makes this HARDER to fire and favours the arm being run.")
    print("    Registered before 8M, i.e. before the data it judges existed.")
    print("  RESIDUAL CONFOUND: under self-play, COMPETENCE lengthens games and the pool gains")
    print("    a stronger snapshot every 2M -- so a genuine +3.0 here can still be improvement,")
    print("    not stall. The AND with draw rate is the load-bearing structure, and the famine")
    print("    verdict is the LADDER, never clause 2 alone.")
    if bar == "10M":
        print("     10M additionally requires the FAMINE READ to fail. This tool does not run it.")

    if do_freeze:
        if cur_val is None:
            print("\n[freeze] no regime known -- refusing")
            return 2
        if key in frozen:
            print(f"\n[freeze] regime {cur_val:.2f} ALREADY FROZEN at "
                  f"{frozen[key]['mean']:.4f} -- refusing to overwrite (frozen once per regime)")
        elif latest < ref_end:
            print(f"\n[freeze] window {cur_start:,}..{ref_end:,} incomplete "
                  f"(latest {latest:,}) -- refusing")
        else:
            w = [v for st_, v in ep if cur_start <= st_ < ref_end]
            frozen[key] = {"regime": cur_val, "mean": st.mean(w), "n": len(w),
                           "window": f"{cur_start:,}..{ref_end:,}",
                           "frozen_on": datetime.date.today().isoformat(),
                           "amendment": "READ AMENDMENT 4"}
            json.dump(frozen, open(frozen_path, "w"), indent=2)
            print(f"\n[freeze] regime {cur_val:.2f} reference FROZEN at "
                  f"{frozen[key]['mean']:.4f} turns (n={len(w)}, window {frozen[key]['window']}) "
                  f"-> {frozen_path}")
    return 1 if fires else 0


if __name__ == "__main__":
    raise SystemExit(main())
