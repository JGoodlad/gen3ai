"""THE 2x2 READOUT — the funded-vs-unfunded teacher-content contrast, against a 3-draw fold floor.

WHAT THE BATCH BUYS. Four arms: FUND_A/FUND_B (teachers = the 8 funded R5FUND forks) and
UNF_A/UNF_B (teachers = their 8 unfunded R5F parents), SAME 16 teams both halves. Within a half the
two arms differ in exactly one argv token (--run-name), so each half's spread is a DRAW, not an
effect. That yields two things at once:

  CONTRAST  FUND - UNF, the question: does a teacher trained on more per-team budget hand down
            more? Read at matched depth, paired on teams.
  FLOOR     two INDEPENDENT frozen-regime replicate draws (FUND_A-FUND_B, UNF_A-UNF_B), which with
            the controller-live N1-N2 draw makes THREE. Until now every fold verdict rested on ONE.

THE POOLING SPANS TWO REGIMES and the output says so on every line: N1/N2 ran controller-live at
grad_accum_steps=2, the 2x2 arms frozen at K=3 (--fork-lr-freeze). They are pooled because three
draws beat one, not because the regimes are known to be equivalent — the script reports the
frozen-only floor beside the 3-draw floor so a reader can see what the regime assumption is worth.

Depths are comparable across both sets: all six runs share fork 28,115,184 -> target 32,567,760
(span 4,452,576), so p1M/mid/end denote the same training depth everywhere.

Labels, as ruled: WITHIN FLOOR (|d| < bar; the CI may still exclude zero -- that says the games are
consistent, not that the ARM differs), NOT DETECTED (|d| >= bar, CI spans zero), SIGNIFICANT.

Run: python tc_readout.py            (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
import json, os
import numpy as np

P = os.path.dirname(os.path.abspath(__file__))
MODELS = "/home/goodlad/dev/gen3ai/models"
RUN = {"TCFUNDA": "ai_v9_160_TCFUNDA_0903", "TCUNFA": "ai_v9_162_TCUNFA_0903",
       "TCFUNDB": "ai_v9_161_TCFUNDB_0903", "TCUNFB": "ai_v9_163_TCUNFB_0903",
       "N1": "ai_v9_142_N1_0901", "N2": "ai_v9_143_N2_0901"}


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


def load(tag, depth=None):
    f = os.path.join(P, f"untaught_{tag}_{depth}.json" if depth else f"untaught_{tag}.json")
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


def label(d, lo, hi, bar):
    if abs(d) < bar:
        return "WITHIN FLOOR"
    return "NOT DETECTED" if lo <= 0 <= hi else "SIGNIFICANT"


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

    frozen_bar = float(np.abs([d.mean() for d in draws.values()]).mean() * 100)
    alldraws = list(draws.values()) + list(n1n2.values())
    pooled_bar = float(np.abs([d.mean() for d in alldraws]).mean() * 100)
    print(f"\n  FROZEN-ONLY floor  {frozen_bar:.2f}pp  ({len(draws)} draw(s), K=3 frozen)")
    print(f"  3-DRAW floor       {pooled_bar:.2f}pp  ({len(alldraws)} draws, SPANS the "
          f"controller-live/frozen regimes — the regime assumption is what the gap above costs)")
    bar = pooled_bar

    # ---- the contrast ----------------------------------------------------------------------
    print(f"\n=== FUNDED - UNFUNDED, paired on teams, bar {bar:.2f}pp ===")
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
            print(f"  {dep:4s} {'BOTH LEGS':20s} {m:+7.2f} [{lo:+7.2f},{hi:+7.2f}]  "
                  f"{label(m,lo,hi,bar)}   <- the readout")

    # ---- the two bar-uncertain C1-vs-B2 legs, re-read ONCE against the wider floor ----------
    print(f"\n=== C1 - B2 re-read at the {bar:.2f}pp floor (was bar-uncertain at 4.27) ===")
    for dep in DEPTHS:
        d = per_team(load("C1", dep), load("B2", dep))
        if d is None:
            print(f"  {dep:4s} UNCOVERED"); continue
        m, lo, hi = ci(d, rng)
        print(f"  {dep:4s} {m:+7.2f} [{lo:+7.2f},{hi:+7.2f}]  {label(m,lo,hi,bar)}")
    print("\nNOTE: the 4.27 bar rested on ONE draw; this one rests on "
          f"{len(alldraws)}. If a leg's verdict moves, say so BY NAME.")


if __name__ == "__main__":
    main()
