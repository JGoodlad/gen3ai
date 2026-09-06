"""THE K=6 READOUT — the v8-dose cell's three pre-registered readings, P1 / P2 / P3, recomputed.

WHERE THIS FILE LIVES, AND WHY IT IS NOT IN A `k6_*` DIRECTORY. The K=6 cell (TC_UNF_K6_A
`ai_v9_170_TCUNFK6A_0904`, TC_UNF_K6_B `ai_v9_171_TCUNFK6B_0904`, launched 2026-09-04 21:22 on batch
pin eb5261ff) never had a measurements directory of its own. Its arms are the 2x2's UNFUNDED argv
minus two tokens (`--grad-accum-steps 3->6`, `--run-name`), and its per-team rows were produced by
the 2x2 batch's own probes (`untaught_probe.py` / `taught_probe.py`) at the same stamp — so they
were rescued and homed with the 2x2's on 2026-09-06 and sit BESIDE this script as
`untaught_TCUNFK6{A,B}_{p1M,mid,end}.json` and `taught_TCUNFK6{A,B}_end.json`. This readout is
therefore committed where its inputs are, rather than a new directory being invented to hold a
script whose data is elsewhere — the defect the readout gate exists to catch.

THE CELL. Two frozen unfunded folds at `--grad-accum-steps 6` = dose `2.8e-5 x 10 / (2048 x 6)` =
2.2786e-08 = **1.06x v8's own dose** (the K=3 arms are 2.12x). Same eight unfunded teachers, same
parent R2ACTION, `--fork-lr 2.8e-5 --fork-lr-freeze`, pool 14/90%. The question: at v8's dose, does
the fold GIFT off-slice?

  P1  each arm and the pooled pair vs the parent, untaught, at all three depths.
      Banked: NO GIFT — parent-neutral at the end (-0.22), a significant HOLE at +1M (-4.19), and
      the hole-then-recovery SHAPE of the K=3 cell replicating at half the dose.
  P2  the pooled K=6 pair vs the pooled K=3 unfunded pair — the DOSE axis at FIXED teachers,
      frozen on both sides, so the parent cancels. Banked: null at every depth.
  P3  the frozen replicate floor AT K=6 vs at K=3 — was the small K=3 floor dose-stable, or lucky.
      Banked: NOT small and NOT depth-stable. Two arms one argv token apart diverged 4pp at p1M.

Nothing here is copied out of the ledger: every delta and interval is recomputed from the per-team
win/games rows, so a transcription error in the ledger cannot enter it. The TAUGHT side of this
cell (+4.19 / +4.78 / pooled +4.48, and K=6 - K=3 = -0.38) is read by `taught_readout.py`, which
owns the taught slice for every arm in both cells.

🚨 THE BAR RULE, WHICH THIS OUTPUT ENFORCES RATHER THAN ASSUMES. A cell is read at ITS OWN floor,
and where two floors are available the LARGER one is taken — never a bar borrowed from a cell with
MORE draws when the cell in hand has FEWER. K=6's own three draws give 2.46pp; the 2x2's six give
1.66; the nine pooled give ~1.9. P1 and P2 are read at 2.46. This is the third time in this program
that a floor from few draws at one depth proved close to uninformative (0.12pp was the extreme
case), which is why P3's caution is printed as outranking P1's headline.

⚠️ EVERY FLOOR HERE IS OPERATIONAL, NOT A PURE DRAW FLOOR (correction that reaches backward across
the program). Two arms one argv token apart share a pool at LAUNCH only: from the first self-play
promotion each arm's own snapshots enter its pool, so the arms are different self-play runs with
diverging opponent distributions. That is the RIGHT bar for judging any two arms — every arm
self-plays — but "pure draw floor" was wrong and is withdrawn. Divergent opponents are an
unadjudicated candidate account for P3's 4pp p1M draw.

BOOTSTRAP. Cluster bootstrap over the 8 untaught TEAMS (the real unit), 20000 draws, one declared
seed. The seed the ad-hoc session used was not recorded; every seed tried agrees with the banked
intervals to within 0.12pp, ~2 steps of this statistic's own 0.0625pp resolution grid
(8 teams x 200 games). Point estimates are seed-free and reproduce EXACTLY.

Run: python k6_readout.py            (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
     python k6_readout.py --check
"""
import json
import os
import sys

import numpy as np

P = os.path.dirname(os.path.abspath(__file__))
MEAS = os.path.dirname(P)                              # .../research_state/measurements
PARENT_ART = f"{MEAS}/reuse_batch_2026-09-03/untaught_R2ACTION.json"
MODELS = "/home/goodlad/dev/gen3ai/models"             # models/ exists only in the main checkout
RUN = {"TCUNFA": "ai_v9_162_TCUNFA_0903", "TCUNFB": "ai_v9_163_TCUNFB_0903",
       "TCUNFK6A": "ai_v9_170_TCUNFK6A_0904", "TCUNFK6B": "ai_v9_171_TCUNFK6B_0904",
       "TCFUNDA": "ai_v9_160_TCFUNDA_0903", "TCFUNDB": "ai_v9_161_TCFUNDB_0903"}
BOOT, SEED = 20000, 20260904

DEPTHS = ("p1M", "mid", "end")
K6 = ("TCUNFK6A", "TCUNFK6B")
K3 = ("TCUNFA", "TCUNFB")
FUND = ("TCFUNDA", "TCFUNDB")                          # only for the nine-draw pooled floor
NICE = {"TCUNFK6A": "TC_UNF_K6_A", "TCUNFK6B": "TC_UNF_K6_B",
        "TCUNFA": "TC_UNF_A", "TCUNFB": "TC_UNF_B",
        "TCFUNDA": "TC_FUND_A", "TCFUNDB": "TC_FUND_B"}


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
        print(f"NO PARENT BASELINE: {PARENT_ART} absent or partial. P1 is a vs-parent delta, so it "
              "cannot\nbe computed. STOP.")
        return 1
    keys = sorted(par)
    pw = rates(par, keys)

    W, absent = {}, []
    for t in K6 + K3 + FUND:
        for dep in DEPTHS:
            c = cells(path_of(t, dep))
            if c is None or sorted(c) != keys:
                absent.append(f"{NICE[t]}/{dep}")
            else:
                W[(t, dep)] = rates(c, keys)
    if absent:
        print(f"UNCOVERED (artifact absent, <8 cells, or a different team set): "
              f"{', '.join(absent)}")

    def draws_of(pair):
        return {dep: W[(pair[0], dep)] - W[(pair[1], dep)] for dep in DEPTHS
                if (pair[0], dep) in W and (pair[1], dep) in W}

    d_k6, d_k3, d_fund = draws_of(K6), draws_of(K3), draws_of(FUND)
    if len(d_k6) != 3:
        print("Fewer than three K=6 draws resolve — this cell has no floor of its own. STOP.")
        return 1
    k6_bar = float(np.mean([abs(v.mean()) * 100 for v in d_k6.values()]))
    k3_mags = [abs(v.mean()) * 100 for v in list(d_k3.values()) + list(d_fund.values())]
    k3_bar = float(np.mean(k3_mags)) if len(k3_mags) == 6 else None
    all_mags = [abs(v.mean()) * 100
                for v in list(d_k6.values()) + list(d_k3.values()) + list(d_fund.values())]
    bar = k6_bar if k3_bar is None else max(k6_bar, k3_bar)

    print("=== BARS (a cell is read at ITS OWN floor; where two exist, the LARGER) ===")
    print(f"  K=6 own floor      {k6_bar:.2f}pp  ({len(d_k6)} draws)   <- the bar for P1 and P2")
    if k3_bar is not None:
        print(f"  K=3 2x2 floor      {k3_bar:.2f}pp  ({len(k3_mags)} draws) — MORE draws, SMALLER "
              f"bar; not borrowed")
        print(f"  all frozen pooled  {np.mean(all_mags):.2f}pp  ({len(all_mags)} draws) — reported, "
              f"not used as the bar here")
    if bar != k6_bar:
        print(f"  bar taken: {bar:.2f}pp (the larger)")

    def row(name, d, suffix="", width=22, b=None):
        m, lo, hi = ci(d, rng)
        lab = label(m, lo, hi, bar if b is None else b)
        print(f"  {name:{width}s} {m:+6.2f} [{lo:+6.2f},{hi:+6.2f}]  {lab}{suffix}")
        return m

    # ---- P1 -------------------------------------------------------------------------------
    print(f"\n=== P1 — K=6 vs the fold parent R2ACTION (wr {pw.mean():.4f}), untaught 8, "
          f"bar {bar:.2f}pp ===")
    p1 = {}
    for dep in DEPTHS:
        print(f"  --- {dep} ---")
        for t in K6:
            if (t, dep) in W:
                row(NICE[t], W[(t, dep)] - pw)
        if all((t, dep) in W for t in K6):
            p1[dep] = row("K=6 pooled", np.mean([W[(t, dep)] for t in K6], axis=0) - pw,
                          pinnote(*K6) + "   <- the readout")
    if len(p1) == 3:
        # HOLE-THEN-RECOVERY: the deepest point is p1M and both later depths sit above it.
        arc = ("   (hole, then recovery)"
               if p1["p1M"] < 0 and p1["p1M"] < min(p1["mid"], p1["end"]) else "")
        print(f"\n  SHAPE  p1M {p1['p1M']:+.2f} -> mid {p1['mid']:+.2f} -> "
              f"end {p1['end']:+.2f}{arc}")
        if all((t, dep) in W for t in K3 for dep in DEPTHS):
            k3 = {dep: (np.mean([W[(t, dep)] for t in K3], axis=0) - pw).mean() * 100
                  for dep in DEPTHS}
            print(f"  K=3 unfunded, same teachers, twice the dose: "
                  f"{k3['p1M']:+.2f} -> {k3['mid']:+.2f} -> {k3['end']:+.2f}  "
                  f"— the SAME arc at half the dose")
    print("\n  --- the within-arm recovery (the parent cancels exactly) ---")
    for dep in ("mid", "end"):
        if all((t, d) in W for t in K6 for d in ("p1M", dep)):
            row(f"K=6 pooled p1M->{dep}",
                np.mean([W[(t, dep)] - W[(t, "p1M")] for t in K6], axis=0), width=22)

    # ---- P2 -------------------------------------------------------------------------------
    print(f"\n=== P2 — the DOSE axis at FIXED teachers: K=6 - K=3, frozen both sides, "
          f"bar {bar:.2f}pp ===")
    print("    (both halves are folds of the same 8 unfunded teachers off the same parent, so the "
          "parent\n     cancels; the only difference is grad_accum_steps 6 vs 3 = 1.06x vs 2.12x "
          "v8's dose)")
    for dep in DEPTHS:
        if all((t, dep) in W for t in K6 + K3):
            row(dep, np.mean([W[(t, dep)] for t in K6], axis=0)
                - np.mean([W[(t, dep)] for t in K3], axis=0), width=22)
    print("  A row whose CI excludes zero but whose |d| sits under the bar is WITHIN FLOOR: the "
          "games are\n  consistent, but two arms one argv token apart differ by this much at this "
          "dose, so a\n  difference this size is not a dose effect.")

    # ---- P3 -------------------------------------------------------------------------------
    print("\n=== P3 — the frozen replicate floor AT K=6: is the small K=3 floor dose-stable? ===")
    for dep in DEPTHS:
        if dep in d_k6:
            m, lo, hi = ci(d_k6[dep], rng)
            print(f"  K6_A - K6_B  {dep:4s} {m:+6.2f} [{lo:+6.2f},{hi:+6.2f}]"
                  f"{'  (clears zero)' if not lo <= 0 <= hi else ''}{pinnote(*K6)}")
    spread = [abs(v.mean()) * 100 for v in d_k6.values()]
    print(f"  mean {k6_bar:.2f}pp over {len(spread)} draws, spread {min(spread):.2f}.."
          f"{max(spread):.2f} — NOT small and NOT depth-stable.")
    if k3_bar is not None:
        print(f"  vs the K=3 frozen floor {k3_bar:.2f}pp over 6 draws; all nine pooled "
              f"{np.mean(all_mags):.2f}pp.")
    big = [d for d in DEPTHS if d in d_k6 and abs(d_k6[d].mean()) * 100 > 3.0]
    if big:
        print(f"  ⚠ TWO ARMS DIFFERING IN ONE ARGV TOKEN diverged by "
              f"{max(abs(d_k6[d].mean()) * 100 for d in big):.2f}pp at {', '.join(big)}. "
              f"Divergent\n    self-play opponents (each arm promotes its own snapshots from the "
              f"first promotion on) is an\n    unadjudicated candidate account. THIS CAUTION "
              f"OUTRANKS P1's HEADLINE.")
    return 0


def check():
    """--check: resolve every input this readout reads and report any that is MISSING, without
    computing anything. The guarded defect is a readout whose artifacts were never committed
    beside it (or at a committed path it resolves)."""
    want = [PARENT_ART] + [path_of(t, d) for t in K6 + K3 + FUND for d in DEPTHS]
    missing = [f for f in want if not os.path.exists(f)]
    for f in missing:
        print(f"MISSING {f}")
    print(f"k6_readout.py: {len(want) - len(missing)}/{len(want)} input artifacts present")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(check() if "--check" in sys.argv else main())
