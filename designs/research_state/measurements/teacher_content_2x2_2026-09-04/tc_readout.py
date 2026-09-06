"""THE 2x2 READOUT — the funded-vs-unfunded teacher-content contrast, each comparison against the
floor of ITS OWN REGIME.

WHAT THE BATCH BUYS. Four arms: FUND_A/FUND_B (teachers = the 8 funded R5FUND forks) and
UNF_A/UNF_B (teachers = their 8 unfunded R5F parents), SAME 16 teams both halves. Within a half the
two arms differ in exactly one argv token (--run-name), so each half's spread is a DRAW, not an
effect. That yields two things at once:

  CONTRAST  FUND - UNF, the question: does a teacher trained on more per-team budget hand down
            more? Read at matched depth, paired on teams.
  FLOOR     SIX independent frozen-regime replicate draws (FUND_A-FUND_B and UNF_A-UNF_B, each at
            three depths). Until 2026-09-05 every fold verdict rested on ONE draw.

TWO REGIMES, TWO BARS — NEVER A POOLED ONE. N1/N2 ran controller-live at grad_accum_steps=2; the
2x2 arms ran frozen at K=3 (--fork-lr-freeze). This script pooled all nine draws into one 2.53pp bar
until 2026-09-06 and applied it to both families; that is RETRACTED (ledger, 2026-09-05 retraction
+ the 2026-09-06 bookkeeping entry, finding 1). The regimes differ by ~3x, their draw intervals
overlap heavily, and pooling them lets a frozen comparison borrow a live floor's slack and a live
comparison borrow a frozen floor's strictness. So:

  FROZEN bar 1.66pp   the six frozen draws (pooled over depths, never one depth) — applies to the
                      2x2 contrast, both of whose arms are frozen.
  LIVE bar 4.27pp     the three N1-N2 draws (the ruled pooled fold floor, 2026-09-04 00:50) —
                      applies to C1-vs-B2 and every other N1/N2-class controller-live comparison.

Neither bar is computed from the other, and no comparison is read at a cross-regime bar. NO VERDICT
IN THE LEDGER MOVES under this correction: the 2x2 contrast is WITHIN FLOOR at p1M and SIGNIFICANT
at mid/end under either bar. C1-vs-B2 does move, back to what is banked — at 4.27 only the +1M leg
clears robustly, mid and end sit INSIDE the floor's own interval [+1.23, +6.92] and are
bar-uncertain. That re-read is STILL OWED and this frozen batch does not discharge it; it needs more
CONTROLLER-LIVE draws.

Depths are comparable across both sets: all six runs share fork 28,115,184 -> target 32,567,760
(span 4,452,576), so p1M/mid/end denote the same training depth everywhere.

Labels, as ruled: WITHIN FLOOR (|d| < bar; the CI may still exclude zero -- that says the games are
consistent, not that the ARM differs), NOT DETECTED (|d| >= bar, CI spans zero), SIGNIFICANT, and
SIGNIFICANT? (bar-uncertain) where |d| clears the bar's point estimate but lands inside the bar's
OWN interval, so whether it clears depends on where in that interval the true floor lies.

Run: python tc_readout.py            (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
import json, os
import numpy as np

P = os.path.dirname(os.path.abspath(__file__))
MEAS = os.path.dirname(P)                              # .../research_state/measurements
MODELS = "/home/goodlad/dev/gen3ai/models"             # models/ exists only in the main checkout
# WHERE EACH ARM'S ARTIFACT LIVES. An artifact belongs to the batch that PRODUCED it, so the 2x2 and
# K=6 rows sit beside this script and the reuse-batch rows (N1/N2/C1/B2) stay in their own directory
# -- ONE copy of every file in the tree, and no path is relative to a cwd. This script read every
# tag from its OWN directory until 2026-09-06, where four of the eight tags have never been; the
# 2x2/K=6 half additionally existed only in a session job dir until it was rescued on 2026-09-05.
DIR = {t: f"{MEAS}/reuse_batch_2026-09-03" for t in ("N1", "N2", "C1", "B2")}
RUN = {"TCFUNDA": "ai_v9_160_TCFUNDA_0903", "TCUNFA": "ai_v9_162_TCUNFA_0903",
       "TCFUNDB": "ai_v9_161_TCFUNDB_0903", "TCUNFB": "ai_v9_163_TCUNFB_0903",
       "N1": "ai_v9_142_N1_0901", "N2": "ai_v9_143_N2_0901"}
TAGS = ("TCFUNDA", "TCFUNDB", "TCUNFA", "TCUNFB", "N1", "N2", "C1", "B2")


def pin(tag):
    """The commit an arm actually recorded. A comparison across two pins is PIN-SPLIT and says so —
    on this batch main was frozen only AFTER arm 1, so FUND_A sits on 0c76e2ee and everything after
    on 52ab5914. The difference was verified inert, but a reader must be told which cells carry it
    rather than have it silently averaged in."""
    f = os.path.join(MODELS, RUN.get(tag, ""), "metadata.json")
    if not os.path.exists(f):
        return None
    return (json.load(open(f)).get("git_hash") or "")[:8]


def pinnote(a, b):
    pa, pb = pin(a), pin(b)
    if pa and pb and pa != pb:
        return f"  ⚠ PIN-SPLIT {pa}/{pb} (verified inert)"
    return "  [pin-clean]" if pa and pb else ""
DEPTHS = ("p1M", "mid", "end")
PAIRS = {"FUND": ("TCFUNDA", "TCFUNDB"), "UNF": ("TCUNFA", "TCUNFB")}
BOOT, SEED = 20000, 20260904
# The live floor's own interval is bootstrapped on a SEPARATE stream (the ruling's seed, 2026-09-05)
# so that adding it does not shift the draw order of the main rng and silently move every interval
# this readout has already banked.
FLOOR_SEED = 20260905


def path_of(tag, depth=None):
    return os.path.join(DIR.get(tag, P),
                        f"untaught_{tag}_{depth}.json" if depth else f"untaught_{tag}.json")


def load(tag, depth=None):
    f = path_of(tag, depth)
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    # POOLED is a summary row, not a team cell -- counting it reports 9/8.
    cells = {k: (v["wins"], v.get("games", v.get("n", 0)))
             for k, v in d.items() if isinstance(v, dict) and "wins" in v and k != "POOLED"}
    return cells if len(cells) == 8 else None          # a partial point is UNCOVERED, never pooled


def per_team(a, b):
    """Per-team rate difference a-b over the shared teams, or None."""
    if not a or not b:
        return None
    teams = sorted(set(a) & set(b))
    if len(teams) < 8:
        return None
    return np.array([a[t][0] / a[t][1] - b[t][0] / b[t][1] for t in teams])


def ci(d, rng):
    boot = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(BOOT)])
    return d.mean() * 100, np.percentile(boot, 2.5) * 100, np.percentile(boot, 97.5) * 100


def label(d, lo, hi, bar, bar_hi=None):
    """bar_hi, when given, is the UPPER end of the bar's own interval: a delta above `bar` but below
    `bar_hi` clears the floor's point estimate and not the floor's uncertainty."""
    if abs(d) < bar:
        return "WITHIN FLOOR"
    if lo <= 0 <= hi:
        return "NOT DETECTED"
    if bar_hi is not None and abs(d) < bar_hi:
        return "SIGNIFICANT? (bar-uncertain: inside the floor's own interval)"
    return "SIGNIFICANT"


def main():
    rng = np.random.default_rng(SEED)

    # ---- the floor draws -------------------------------------------------------------------
    draws, missing = {}, []
    for half, (a, b) in PAIRS.items():
        for dep in DEPTHS:
            d = per_team(load(a, dep), load(b, dep))
            if d is None:
                missing.append(f"{half}/{dep}")
            else:
                draws[f"{half}/{dep}"] = d
    n1n2 = {dep: per_team(load("N1", dep), load("N2", dep)) for dep in DEPTHS}
    n1n2 = {k: v for k, v in n1n2.items() if v is not None}

    print("=== REPLICATE DRAWS (each is one pair differing in --run-name alone) ===")
    for k, d in list(draws.items()) + [(f"N1N2/{k}", v) for k, v in n1n2.items()]:
        m, lo, hi = ci(d, rng)
        regime = "controller-live gas=2" if k.startswith("N1N2") else "frozen K=3"
        half = k.split("/")[0]
        note = pinnote(*PAIRS[half]) if half in PAIRS else ""
        print(f"  {k:14s} {m:+7.2f}pp [{lo:+7.2f},{hi:+7.2f}]   ({regime}){note}")
    if missing:
        print(f"  UNCOVERED (arm absent or <8 cells): {', '.join(missing)} — NOT pooled")

    if not draws:
        print("\nNo frozen draw available yet — the 2x2 has not produced a complete pair. STOP.")
        return

    # TWO BARS, ONE PER REGIME. A floor is a MAGNITUDE, so it pools |mean| over its own draws.
    frozen_mags = np.abs([d.mean() for d in draws.values()]) * 100
    frozen_bar = float(frozen_mags.mean())
    live_bar = live_hi = None
    if n1n2:
        live_bar = float(np.abs([d.mean() for d in n1n2.values()]).mean() * 100)
        # The three live draws are ONE pair at three depths and all share sign (N2 sits below N1
        # everywhere), so the ruling pooled them SIGNED and took the CI over teams. The six frozen
        # draws are two pairs whose signs differ, so no such signed pooling exists for them and
        # their spread is reported as min/max instead of a fabricated interval.
        frng = np.random.default_rng(FLOOR_SEED)
        pooled_live = np.mean(list(n1n2.values()), axis=0)
        _, live_lo, live_hi = ci(pooled_live, frng)

    print(f"\n  FROZEN bar  {frozen_bar:.2f}pp  ({len(draws)} draws, K=3 frozen; spread "
          f"{frozen_mags.min():.2f}..{frozen_mags.max():.2f}, signs differ so there is no pooled "
          f"interval) — the bar for the 2x2 contrast")
    if live_bar is not None:
        print(f"  LIVE bar    {live_bar:.2f}pp [{live_lo:+.2f},{live_hi:+.2f}]  "
              f"({len(n1n2)} draws, controller-live gas=2; ONE fold draw read at three depths) "
              f"— the bar for C1-vs-B2")
    print("  The two are NEVER pooled: the 2.53pp nine-draw bar this script used until 2026-09-06 "
          "is RETRACTED (ledger 2026-09-05 / 2026-09-06 finding 1).")

    # ---- the contrast (both arms frozen -> the FROZEN bar) ---------------------------------
    bar = frozen_bar
    print(f"\n=== FUNDED - UNFUNDED, paired on teams, FROZEN bar {bar:.2f}pp ===")
    for dep in DEPTHS:
        legs = []
        for a, b in (("TCFUNDA", "TCUNFA"), ("TCFUNDB", "TCUNFB")):
            d = per_team(load(a, dep), load(b, dep))
            if d is not None:
                legs.append((f"{a}-{b}", d))
        if not legs:
            print(f"  {dep:4s} UNCOVERED"); continue
        for name, d in legs:
            m, lo, hi = ci(d, rng)
            a_, b_ = name.split("-")
            print(f"  {dep:4s} {name:20s} {m:+7.2f} [{lo:+7.2f},{hi:+7.2f}]  "
                  f"{label(m,lo,hi,bar)}{pinnote(a_, b_)}")
        if len(legs) == 2:
            d = (legs[0][1] + legs[1][1]) / 2          # the two legs share teams; average, then CI
            m, lo, hi = ci(d, rng)
            # A pooled bar can be cleared by a magnitude smaller than a single draw inside it, so
            # say when the readout clears even the LARGEST frozen draw — that is the strong reading.
            strong = (f"  (exceeds even the largest single frozen draw {frozen_mags.max():.2f})"
                      if abs(m) > frozen_mags.max() else "")
            print(f"  {dep:4s} {'BOTH LEGS':20s} {m:+7.2f} [{lo:+7.2f},{hi:+7.2f}]  "
                  f"{label(m,lo,hi,bar)}   <- the readout{strong}")

    # ---- C1 - B2: CONTROLLER-LIVE arms, so the LIVE bar, and the re-read is STILL OWED ------
    if live_bar is None:
        print("\n=== C1 - B2: no N1/N2 draw available, so no live bar. NOT read. ===")
    else:
        print(f"\n=== C1 - B2 at the CONTROLLER-LIVE bar {live_bar:.2f}pp "
              f"[{live_lo:+.2f},{live_hi:+.2f}] ===")
        for dep in DEPTHS:
            d = per_team(load("C1", dep), load("B2", dep))
            if d is None:
                print(f"  {dep:4s} UNCOVERED"); continue
            m, lo, hi = ci(d, rng)
            print(f"  {dep:4s} {m:+7.2f} [{lo:+7.2f},{hi:+7.2f}]  "
                  f"{label(m,lo,hi,live_bar,live_hi)}")
        print(f"\n  STILL OWED — THIS BATCH DOES NOT DISCHARGE IT. C1 and B2 both ran "
              f"controller-live, so their bar is\n  the live {live_bar:.2f}, and that bar's own "
              f"uncertainty [{live_lo:+.2f},{live_hi:+.2f}] is what leaves the mid and end legs\n"
              f"  bar-uncertain. The frozen draws this batch adds say nothing about a "
              f"controller-live spread; the\n  re-read needs more CONTROLLER-LIVE draws. Reading "
              f"these legs at a pooled nine-draw bar — as this\n  script did until 2026-09-06 — "
              f"printed SIGNIFICANT at all three depths, which is exactly the\n  cross-regime "
              f"borrowing the two-bar split exists to prevent.")


def check():
    """--check: resolve every input this readout reads and report any that is MISSING, without
    computing anything. Exists so a cheap test can prove the script still runs AS COMMITTED --
    the defect it guards is a readout whose artifacts were never committed beside it."""
    missing = [f for t in TAGS for dep in DEPTHS
               if not os.path.exists(f := path_of(t, dep))]
    for f in missing:
        print(f"MISSING {f}")
    print(f"tc_readout.py: {len(TAGS) * len(DEPTHS) - len(missing)}/{len(TAGS) * len(DEPTHS)} "
          f"input artifacts present")
    return 1 if missing else 0


if __name__ == "__main__":
    import sys
    sys.exit(check() if "--check" in sys.argv else (main() or 0))
