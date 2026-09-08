"""Measure a restart's STARTUP COST, against a reading PRE-REGISTERED before the number existed.

    python -m main.ops.restart_startup <run-name|run-dir> --baseline PATH
                                       [--cadence-from-step N]

Orchestrator 1, 2026-09-06, before the 23:27 restart:

  EXPECTATION: the restart should NOT pay the 4M promotion's 363 s.
  --compile-opponents-preload traces the opponent extractor ONCE in the forkserver at env
  construction and every worker inherits the graph by fork. At a restart the pool snapshots
  EXIST at construction, so they are exactly what the preload covers. The original launch
  could not use it for pool opponents because the pool was empty.

  READING (fixed in advance):
    excess < ~120 s                      -> preload covered it. The ETA keeps its 262 s per
                                            restart. Nothing to write.
    excess >= ~300 s AND 48 fresh traces -> the preload is NOT covering pool snapshots on a
                                            restart. That is a DEFECT, not a tax: it gets a
                                            backlog row with the log lines as evidence, and is
                                            NOT folded into the ETA as a cost of doing business.
    between                              -> report the number, claim neither reading.

METHOD (same as the promotions): TB wall time from the last pre-restart rollout to the first
post-restart one, minus the steady rollout cadence, plus a fresh-vs-reused compile-line count
taken against the pre-restart baseline file.

THE BASELINE FILE is ``key=value`` lines captured BEFORE the restart, and it is REQUIRED:
``last_rollout_step``, ``last_rollout_wall``, ``captured_at``, ``compile_fresh_total``,
``compile_reused_total``, ``log_inode``, ``log_size``. Without it there is nothing to difference
and the section is not computable — which is not the same as a cost of zero.

``--cadence-from-step`` bounds which pre-restart rollouts define the steady cadence. It defaults
to the FIRST ROLLOUT STRICTLY INSIDE THE CURRENT OPPONENT REGIME, because a cadence measured
across a regime boundary is a mixture of two cadences. (In the session copy this was the literal
``4_128_768`` — that arm's first self-play rollout; the default reproduces it.)

Promoted 2026-09-07 from the Training Run session's ``restart_startup.py``.
"""
from __future__ import annotations

import os
import statistics as st
import sys
from typing import Sequence

from main.ops.run_ref import event_dirs, refuse, resolve_run_dir

REQUIRED_BASELINE_KEYS = ("last_rollout_step", "last_rollout_wall", "captured_at",
                          "compile_fresh_total", "compile_reused_total")
#: A restart on the arm this was written for costs 312-317 s. An excess below this is the
#: boundary NOT BEING IN THE DATA YET, not a free startup.
MIN_PLAUSIBLE_RESTART_S = 100.0


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901 - one linear read, kept whole
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if args else 2
    run = resolve_run_dir(args[0])
    if "--baseline" not in args:
        refuse("REFUSING: --baseline is required.",
               "  The startup cost is a DIFFERENCE against a baseline captured BEFORE the",
               "  restart. Without it there is nothing to difference, and a missing baseline is",
               "  not a cost of zero. Nothing is concluded.")
    base_path = args[args.index("--baseline") + 1]
    cadence_from = int(args[args.index("--cadence-from-step") + 1]) \
        if "--cadence-from-step" in args else None
    log = os.path.join(run, "launcher_child.log")

    if not os.path.exists(base_path):
        refuse(f"REFUSING: baseline file {base_path} does not exist. Nothing is concluded.")
    base = {}
    for line in open(base_path):
        if "=" in line:
            k, v = line.strip().split("=", 1)
            base[k] = v
    missing_keys = [k for k in REQUIRED_BASELINE_KEYS if k not in base]
    if missing_keys:
        refuse(f"REFUSING: baseline {base_path} is missing {missing_keys}.",
               "  A partial baseline yields a partial difference that still prints a number.")

    ev = []
    sp = []
    for d in event_dirs(run):
        ea = EventAccumulator(d, size_guidance={"scalars": 0})
        ea.Reload()
        tags = ea.Tags()["scalars"]
        if "rollout/ep_len_mean" in tags:
            ev += [(e.step, e.wall_time) for e in ea.Scalars("rollout/ep_len_mean")]
        if "train/selfplay_fraction" in tags:
            sp += [(e.step, e.value) for e in ea.Scalars("train/selfplay_fraction")]
    ev = sorted(set(ev))
    if not ev:
        print("NO SCALAR rollout/ep_len_mean  <- absence is not a zero")
        return 2

    pre_step = int(base["last_rollout_step"])
    pre_wall = float(base["last_rollout_wall"])
    # Filter by STEP, never by wall time. The baseline records wall rounded to the second,
    # so `w > pre_wall` matches the BASELINE'S OWN rollout and yields a 0 s gap -- which the
    # reading below then scores as "preload covered it". A dry run produced exactly that false
    # PASS before the restart had happened. Step is exact and monotone; use it.
    after = [(s, w) for s, w in ev if s > pre_step]

    if cadence_from is None:
        # The first rollout STRICTLY INSIDE the current opponent regime: a cadence measured
        # across a regime boundary is a mixture of two cadences.
        regs = []
        for s, v in sorted(set(sp)):
            if not regs or v != regs[-1][1]:
                regs.append((s, v))
        regime_start = regs[-1][0] if regs else 0
        inside = [s for s, _ in ev if s >= regime_start]
        cadence_from = inside[0] if inside else 0

    print("=== RESTART STARTUP COST (reading pre-registered before the number existed) ===")
    print(f"  run {run}")
    print(f"  baseline captured {base['captured_at']}  at step {pre_step:,}")
    print(f"  cadence measured from rollouts after step {cadence_from:,} "
          "(current opponent regime only)")

    cad = [w1 - w0 for (s0, w0), (s1, w1) in zip(ev, ev[1:])
           if cadence_from < s1 and w1 <= pre_wall]
    cadence = st.median(cad) if cad else float("nan")
    print(f"  steady cadence {cadence:.0f} s (n={len(cad)})")
    if not cad:
        print("\n  NO pre-restart rollouts inside the current regime -- the cadence is")
        print("  UNMEASURABLE and no excess can be computed. Nothing is concluded.")
        return 2

    if not after:
        print("\n  NO post-baseline rollout yet -- the restart has not produced one.")
        print("  This is NOT a reading of zero; re-run once a rollout lands after the restart.")
        return 0

    # the restart gap is the largest gap that spans the restart; find it among post-baseline gaps
    seq = [(pre_step, pre_wall)] + after
    gaps = [(s1, w1 - w0) for (s0, w0), (s1, w1) in zip(seq, seq[1:])]
    # 🚨 A BASELINE THAT SPANS MORE THAN ONE RESTART reports the OLDEST restart's gap, not the
    # newest. Measured 2026-09-07: a baseline captured before restart 3 was still in place when
    # restart 4 fired, and the tool returned restart 3's 578 s / +412 s as if it were restart 4's
    # (whose real cost was 513 s / +346 s). The largest gap is not the latest one.
    big = [(s_, g_) for s_, g_ in gaps if g_ - cadence >= 100.0]
    if len(big) > 1:
        print(f"\n  REFUSING: the baseline spans {len(big)} restart-sized gaps, not one:")
        for s_, g_ in big:
            print(f"    step -> {s_:>12,}   gap {g_:6.0f} s   excess {g_-cadence:+6.0f} s")
        print("  The largest is not necessarily the LATEST. Capture a fresh baseline before each")
        print("  restart, or measure the intended gap directly. Nothing is concluded.")
        return 2
    gs, gap = max(gaps, key=lambda x: x[1])
    excess = gap - cadence
    # An excess near zero does NOT mean "the preload covered it perfectly" -- it means the
    # boundary is NOT IN THE DATA YET, because the new child has not logged a rollout and every
    # gap found is pre-restart. Reporting a small excess as a measurement is the same error as
    # reading a count from a ring buffer or a trend from one point: the number came back clean
    # and describes nothing.
    if 0 < excess < MIN_PLAUSIBLE_RESTART_S:
        print(f"\n  NOT YET OBSERVED: largest gap {gap:.0f} s, excess {excess:+.0f} s.")
        print(f"  A restart costs several hundred seconds, so an excess under "
              f"{MIN_PLAUSIBLE_RESTART_S:.0f} s means the new child has not logged a rollout")
        print("  and every gap here is PRE-restart. This is NOT a reading of 'startup was free'.")
        print("  Re-run once a post-restart rollout lands.")
        return 0
    if gap <= 0 or excess < -1.0:
        # A restart gap cannot be shorter than the cadence. If it looks that way, the boundary
        # was not found and the number is an artifact -- refuse rather than emit a verdict.
        print(f"\n  REFUSING: gap {gap:.0f} s / excess {excess:+.0f} s is not a restart boundary.")
        print("  A restart cannot be FASTER than the steady cadence. Nothing is concluded.")
        return 2
    print("\n  first post-baseline rollouts: " + ", ".join(f"{s:,}" for s, _ in after[:3]))
    print(f"  LARGEST gap across the boundary: {gap:.0f} s  ->  step {gs:,}")
    print(f"  EXCESS over cadence: {excess:+.0f} s")

    fresh_now = reused_now = 0
    if os.path.exists(log):
        txt = open(log, errors="replace").read()
        fresh_now = sum(1 for line in txt.splitlines()
                        if "CompileExtractor" in line and "median" in line)
        reused_now = sum(1 for line in txt.splitlines()
                         if "CompileExtractor" in line and "reused" in line)
        # THE CHILD LOG IS A RING BUFFER (~1024 KiB). Its first line says so, and its SIZE
        # SHRANK 716,367 -> 578,513 across one restart while the INODE stayed constant -- so
        # an inode check does NOT establish that history survived. Counting historical lines
        # in it is invalid: old entries are silently trimmed. A count taken now is a count of
        # whatever still fits, not of what happened.
        trimmed = ("ring buffer" in txt[:400]
                   or os.stat(log).st_size < int(base.get("log_size", 0)))
        same_file = (str(os.stat(log).st_ino) == base.get("log_inode")) and not trimmed
        print(f"\n  child log {'SAME file' if same_file else 'NEW file (rotated on restart)'}"
              f"  -- fresh traces {base['compile_fresh_total']} -> {fresh_now},"
              f"  reused {base['compile_reused_total']} -> {reused_now}")
        if trimmed:
            d_fresh = None
            print("  🚨 RING BUFFER TRIMMED — historical compile-line counts are UNMEASURABLE")
            print("     from this log. The fresh-vs-reused half of the pre-registered reading")
            print("     CANNOT be evaluated. This is not a count of zero.")
        else:
            d_fresh = fresh_now - int(base["compile_fresh_total"]) if same_file else fresh_now
        print(f"  NEW fresh traces since the baseline: {d_fresh}")
    else:
        d_fresh = -1
        print("\n  child log MISSING -- fresh-trace count unavailable, NOT read as zero")

    print("\n  VERDICT against the pre-registered reading:")
    if d_fresh is None:
        print(f"    excess {excess:.0f} s is VALID (TB wall time is not trimmed), but the")
        print("    fresh-trace clause is UNMEASURABLE, so the registered reading cannot be")
        print("    completed. Report the duration; assert NO mechanism.")
    elif excess < 120:
        print(f"    excess {excess:.0f} s < 120 s -> PRELOAD COVERED IT.")
        print("    The ETA keeps its 262 s per restart. Nothing to write.")
    elif excess >= 300 and d_fresh >= 40:
        print(f"    excess {excess:.0f} s >= 300 s AND {d_fresh} new fresh traces")
        print("    -> DEFECT: --compile-opponents-preload is not covering pool snapshots on a")
        print("       restart. Backlog row with the log lines as evidence.")
        print("       This is NOT folded into the ETA as a cost of doing business.")
    else:
        print(f"    excess {excess:.0f} s with {d_fresh} new fresh traces -- BETWEEN the registered")
        print("    bands. Report the number; claim neither reading.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
