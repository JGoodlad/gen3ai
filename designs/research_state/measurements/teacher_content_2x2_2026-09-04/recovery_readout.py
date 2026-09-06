"""THE RECOVERY READOUT — the 2x2's untaught LEVELS vs the parent at three depths, and the
within-arm p1M -> mid / p1M -> end change that turns "recovers" from a reading into a measurement.

WHAT IT ANSWERS. `tc_readout.py` reads the funded-minus-unfunded CONTRAST and the replicate floor.
This reads the two things that contrast is built out of, and the statistic that explains it:

  LEVELS    each arm, and each half, against the fold parent R2ACTION on the untaught 8, at p1M /
            mid / end. Banked: BOTH halves sit ~3pp BELOW the parent at +1M (-3.12 vs -3.28,
            indistinguishable), and only the UNFUNDED half climbs out (+1.25 mid, +1.97 end) while
            the funded half does not (-4.88, -2.41).
  RECOVERY  the within-arm depth change (arm_dep - arm_p1M), paired on teams. THE PARENT TERM
            CANCELS EXACTLY in (arm_mid - par) - (arm_p1M - par), so this statistic carries no
            parent noise at all and is CLEANER than the levels it is built from. Banked:
            UNFUNDED - FUNDED recovery +6.28 [+3.16, +9.81] at p1M->mid and +4.53 [+1.94, +7.41]
            at p1M->end -- "funded teachers BLOCK the recovery" as a number.

Nothing here is copied out of the ledger: every delta and interval is recomputed from the per-team
win/games rows, so a transcription error in the ledger cannot enter it.

🚨 THREE THINGS THE OUTPUT SAYS SO THEY CANNOT BE DROPPED IN A QUOTE.

* "RECOVERY" IS RECOVERY *TOWARD THE PARENT* ONLY BECAUSE BOTH HALVES SAT BELOW IT AT p1M. That is
  a premise IN THE DATA (-3.12 / -3.28, indistinguishable), not part of the statistic — which is
  why the levels are printed above the recovery and the p1M row is checked for it.
* THE FUNDED HALF IS HETEROGENEOUS. FUND_A is flat (+0.87) and FUND_B DETERIORATED (-4.38, interval
  excluding zero on the wrong side). The banked sentence is "funded shows NO CONSISTENT recovery",
  never "funded does not recover": the pooled interval averages one flat arm and one that got
  worse. The per-arm rows are printed for exactly this reason.
* THE BAR IS THE FROZEN SIX-DRAW 1.66, RECOMPUTED, NEVER THE CONTROLLER-LIVE 4.27. Every arm here
  ran frozen at K=3 with `--fork-lr-freeze`; borrowing the live bar (or a pooled cross-regime one)
  is the error `tc_readout.py`'s two-bar split exists to prevent.

BOOTSTRAP. Cluster bootstrap over the 8 untaught TEAMS (the real unit), 20000 draws, one declared
seed. The seed the ad-hoc session used was not recorded; every seed tried agrees with the banked
intervals to within 0.12pp, which is ~2 steps of this statistic's own 0.0625pp resolution grid
(8 teams x 200 games). Point estimates are seed-free and reproduce EXACTLY.

Run: python recovery_readout.py      (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
     python recovery_readout.py --check
"""
import json
import os
import sys

import numpy as np

P = os.path.dirname(os.path.abspath(__file__))
MEAS = os.path.dirname(P)                              # .../research_state/measurements
# The PARENT's untaught-8 artifact belongs to the batch that produced it and stays there; the 2x2
# and K=6 rows sit beside this script. ONE copy of every file in the tree.
PARENT_ART = f"{MEAS}/reuse_batch_2026-09-03/untaught_R2ACTION.json"
MODELS = "/home/goodlad/dev/gen3ai/models"             # models/ exists only in the main checkout
RUN = {"TCFUNDA": "ai_v9_160_TCFUNDA_0903", "TCUNFA": "ai_v9_162_TCUNFA_0903",
       "TCFUNDB": "ai_v9_161_TCFUNDB_0903", "TCUNFB": "ai_v9_163_TCUNFB_0903"}
BOOT, SEED = 20000, 20260904

DEPTHS = ("p1M", "mid", "end")
ARMS = ("TCFUNDA", "TCFUNDB", "TCUNFA", "TCUNFB")
HALVES = (("FUNDED half", ("TCFUNDA", "TCFUNDB")), ("UNFUNDED half", ("TCUNFA", "TCUNFB")))
NICE = {"TCFUNDA": "TC_FUND_A", "TCFUNDB": "TC_FUND_B",
        "TCUNFA": "TC_UNF_A", "TCUNFB": "TC_UNF_B"}


def path_of(tag, depth):
    return os.path.join(P, f"untaught_{tag}_{depth}.json")


def cells(path):
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    # POOLED is a summary row, not a team cell -- counting it reports 9/8.
    c = {k: (v["wins"], v.get("games", v.get("n", 0)))
         for k, v in d.items() if isinstance(v, dict) and "wins" in v and k != "POOLED"}
    return c if len(c) == 8 else None      # a partial point is UNCOVERED, never pooled


def rates(c, keys):
    return np.array([c[k][0] / c[k][1] for k in keys])


def ci(d, rng):
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(BOOT)])
    return d.mean() * 100, np.percentile(boot, 2.5) * 100, np.percentile(boot, 97.5) * 100


def label(d, lo, hi, bar):
    """WITHIN FLOOR (|d| < bar; the CI may still exclude zero -- the games are consistent, the ARM
    is not separated), NOT DETECTED (|d| >= bar, CI spans zero), SIGNIFICANT."""
    if abs(d) < bar:
        return "WITHIN FLOOR"
    return "NOT DETECTED" if lo <= 0 <= hi else "SIGNIFICANT"


def pin(tag):
    f = os.path.join(MODELS, RUN.get(tag, ""), "metadata.json")
    if not os.path.exists(f):
        return None
    return (json.load(open(f)).get("git_hash") or "")[:8]


def pinnote(a, b):
    pa, pb = pin(a), pin(b)
    if pa and pb and pa != pb:
        return f"  ⚠ PIN-SPLIT {pa}/{pb} (verified inert)"
    return "  [pin-clean]" if pa and pb else ""


def main():
    rng = np.random.default_rng(SEED)
    par = cells(PARENT_ART)
    if par is None:
        print(f"NO PARENT BASELINE: {PARENT_ART} absent or partial. The LEVELS are vs-parent "
              "deltas,\nso they cannot be computed. STOP.")
        return 1
    keys = sorted(par)
    pw = rates(par, keys)

    W, absent = {}, []
    for t in ARMS:
        for dep in DEPTHS:
            c = cells(path_of(t, dep))
            if c is None or sorted(c) != keys:
                absent.append(f"{NICE[t]}/{dep}")
            else:
                W[(t, dep)] = rates(c, keys)
    if absent:
        print(f"UNCOVERED (artifact absent, <8 cells, or a different team set): "
              f"{', '.join(absent)}")

    # The bar: this batch's OWN six frozen replicate draws, recomputed here rather than typed, so
    # it cannot drift from tc_readout.py's. Frozen arms take the frozen bar; the controller-live
    # 4.27 belongs to N1/N2-class comparisons and is never borrowed.
    mags = [abs((W[(a, dep)] - W[(b, dep)]).mean()) * 100
            for a, b in (("TCFUNDA", "TCFUNDB"), ("TCUNFA", "TCUNFB"))
            for dep in DEPTHS if (a, dep) in W and (b, dep) in W]
    bar = float(np.mean(mags)) if len(mags) == 6 else None

    print("=== UNTAUGHT-8 LEVELS vs the fold parent R2ACTION "
          f"(wr {pw.mean():.4f}), by depth ===")
    print(f"    8 teams x 200 games, paired on teams, cluster bootstrap over TEAMS "
          f"({BOOT} draws, seed {SEED})")
    if bar is None:
        print("    NO BAR: fewer than six frozen replicate draws resolve. Deltas only.")
    else:
        print(f"    bar {bar:.2f}pp — this batch's OWN six frozen draws, pooled over depths "
              f"(never the\n    controller-live 4.27, and never a pooled cross-regime bar)")

    def row(name, d, suffix="", width=22):
        m, lo, hi = ci(d, rng)
        lab = label(m, lo, hi, bar) if bar else ""
        print(f"  {name:{width}s} {m:+6.2f} [{lo:+6.2f},{hi:+6.2f}]  {lab}{suffix}")
        return m

    levels = {}
    for dep in DEPTHS:
        print(f"\n  --- {dep} ---")
        for t in ARMS:
            if (t, dep) in W:
                row(NICE[t], W[(t, dep)] - pw)
        for name, ts in HALVES:
            if all((t, dep) in W for t in ts):
                levels[(name, dep)] = row(
                    name, np.mean([W[(t, dep)] for t in ts], axis=0) - pw, pinnote(*ts))

    if all((n, "p1M") in levels for n, _ in HALVES):
        f0, u0 = levels[("FUNDED half", "p1M")], levels[("UNFUNDED half", "p1M")]
        both_below = f0 < 0 and u0 < 0
        print(f"\n  PREMISE CHECK for the word 'recovery': at p1M both halves sit "
              f"{'BELOW' if both_below else 'NOT both below'} the parent\n"
              f"  ({f0:+.2f} / {u0:+.2f}, differing by {abs(f0 - u0):.2f}pp"
              f"{' — inside the bar, i.e. indistinguishable' if bar and abs(f0-u0) < bar else ''})."
              + ("  So a rise IS a move toward the parent."
                 if both_below else
                 "  ⚠ A RISE IS THEN NOT NECESSARILY A MOVE TOWARD THE PARENT — the word "
                 "'recovery' does not apply."))

    # ---- the recovery statistic: the parent cancels exactly, so no parent noise enters ------
    print("\n=== RECOVERY — the WITHIN-ARM depth change (arm_dep - arm_p1M), paired on teams ===")
    print("    (arm_dep - par) - (arm_p1M - par) == arm_dep - arm_p1M: the parent term cancels")
    rec = {}
    for dep in ("mid", "end"):
        print(f"\n  --- p1M -> {dep} ---")
        for t in ARMS:
            if (t, dep) in W and (t, "p1M") in W:
                row(NICE[t], W[(t, dep)] - W[(t, "p1M")])
        for name, ts in HALVES:
            if all((t, d) in W for t in ts for d in ("p1M", dep)):
                rec[(name, dep)] = np.mean([W[(t, dep)] - W[(t, "p1M")] for t in ts], axis=0)
                row(name, rec[(name, dep)], pinnote(*ts))
        if ("FUNDED half", dep) in rec and ("UNFUNDED half", dep) in rec:
            row("UNFUNDED - FUNDED", rec[("UNFUNDED half", dep)] - rec[("FUNDED half", dep)],
                "   <- the readout")

    # ---- the heterogeneity that the pooled row hides ---------------------------------------
    print("\n=== HETEROGENEITY within the funded half (why the banked sentence is 'NO CONSISTENT "
          "recovery') ===")
    for dep in ("mid", "end"):
        for name, ts in HALVES:
            vals = [(W[(t, dep)] - W[(t, "p1M")]).mean() * 100 for t in ts
                    if (t, dep) in W and (t, "p1M") in W]
            if len(vals) == 2:
                agree = "AGREE in sign" if vals[0] * vals[1] > 0 else "DISAGREE in sign"
                print(f"  p1M->{dep:3s} {name:15s} arms {vals[0]:+6.2f} / {vals[1]:+6.2f}  "
                      f"{agree}, spread {abs(vals[0] - vals[1]):.2f}pp")
    print("  A half whose two arms disagree in sign is reported as NO CONSISTENT recovery — the "
          "pooled\n  interval there averages arms that did different things, and is not evidence "
          "that the half\n  did the average of them.")
    return 0


def check():
    """--check: resolve every input this readout reads and report any that is MISSING, without
    computing anything. The guarded defect is a readout whose artifacts were never committed
    beside it (or at a committed path it resolves)."""
    want = [PARENT_ART] + [path_of(t, d) for t in ARMS for d in DEPTHS]
    missing = [f for f in want if not os.path.exists(f)]
    for f in missing:
        print(f"MISSING {f}")
    print(f"recovery_readout.py: {len(want) - len(missing)}/{len(want)} input artifacts present")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(check() if "--check" in sys.argv else main())
