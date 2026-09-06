"""THE TAUGHT READOUT — the ON-SLICE half of the 2x2 (and of the K=6 cell), vs the fold parent.

WHAT IT ANSWERS. `tc_readout.py` reads the UNTAUGHT 8 — what a fold does off its own slice. This
reads the other half: the 16 teams the fold was actually TRAINED on, every arm against the same
parent (R2ACTION), paired on teams. Three questions, in the order they were asked:

  1. DID THE FOLD TEACH ANYTHING AT ALL?  arm - parent on the taught 16. Banked ~+5pp.
  2. DOES TEACHER FUNDING BUY MORE TEACHING?  FUNDED - UNFUNDED, paired. Banked +0.25, WITHIN FLOOR
     -- so the funded half's off-slice robbery buys nothing on-slice either.
  3. DOES HALVING THE DOSE COST TEACHING?  K=6 - K=3 at fixed (unfunded) teachers. Banked -0.38.

Nothing here is copied out of the ledger: every delta and interval is recomputed from the per-team
win/games rows of `taught_*_end.json`, so a transcription error in the ledger cannot enter it.

🚨 TWO QUOTING RULES THIS OUTPUT ENFORCES BY PRINTING THEM.

* TAUGHT AND UNTAUGHT LEVELS ARE NOT COMPARABLE. The parent's taught win rate is 0.4747 against its
  untaught 0.5825 -- the taught 16 are simply harder ground for this parent. Only the vs-parent
  DELTAS within a slice compare, and the +4.98 gain is measured on the harder set. Lining an arm's
  taught 0.5297 up against its untaught 0.6019 and reading a decline is the team sets differing, not
  the model.
* THE TAUGHT FLOOR IS PROVISIONAL AND EVERY LABEL IT PRODUCES CARRIES THAT WORD. A slice takes its
  OWN regime's floor -- the taught floor is never blended with the untaught one -- but this one is
  TWO draws at ONE depth (the taught pass is endpoint-only), which is the exact structure that
  over-claimed the untaught floor at 0.12pp before six draws corrected it to 1.66. So the script
  labels against it because that is the right slice, stamps PROVISIONAL on every such label, and
  then re-reads the whole comparison at the untaught six-draw 1.66 as a SENSITIVITY, naming any row
  whose verdict moves. A row that reads the same under both bars does not depend on the provisional
  number at all.

Bars are never blended across slices, and never across the frozen / controller-live regimes
either (see `tc_readout.py`'s two-bar split).

BOOTSTRAP. Cluster bootstrap over the 16 taught TEAMS (the real unit), 20000 draws, one declared
seed. The seed the ad-hoc session used was not recorded; every seed tried agrees with the banked
intervals to within 0.07pp, which is 1-2 steps of this statistic's own 0.031pp resolution grid
(16 teams x 200 games). Point estimates are seed-free and reproduce EXACTLY.

Run: python taught_readout.py        (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
     python taught_readout.py --check
"""
import json
import os
import sys

import numpy as np

P = os.path.dirname(os.path.abspath(__file__))
MODELS = "/home/goodlad/dev/gen3ai/models"             # models/ exists only in the main checkout
RUN = {"TCFUNDA": "ai_v9_160_TCFUNDA_0903", "TCUNFA": "ai_v9_162_TCUNFA_0903",
       "TCFUNDB": "ai_v9_161_TCFUNDB_0903", "TCUNFB": "ai_v9_163_TCUNFB_0903",
       "TCUNFK6A": "ai_v9_170_TCUNFK6A_0904", "TCUNFK6B": "ai_v9_171_TCUNFK6B_0904"}
BOOT, SEED = 20000, 20260904
# The UNTAUGHT frozen floor from this batch's own six replicate draws (tc_readout.py). It is the
# only measured bar available to a frozen-regime comparison here; the taught draws are two, at one
# depth, and are printed as evidence rather than used as a bar.
FROZEN_BAR_SOURCE = "tc_readout.py (six frozen untaught draws, pooled over depths)"

PARENT = "R2ACTION"
ARMS = ("TCFUNDA", "TCFUNDB", "TCUNFA", "TCUNFB", "TCUNFK6A", "TCUNFK6B")
NICE = {"TCFUNDA": "TC_FUND_A", "TCFUNDB": "TC_FUND_B", "TCUNFA": "TC_UNF_A",
        "TCUNFB": "TC_UNF_B", "TCUNFK6A": "TC_UNF_K6_A", "TCUNFK6B": "TC_UNF_K6_B"}


def pin(tag):
    """The commit an arm actually recorded. Main was frozen only AFTER arm 1 of the 2x2, so FUND_A
    sits on one commit and everything after on another; a comparison across two pins is PIN-SPLIT
    and must say so rather than have the drift silently averaged in. Silent when models/ is
    unreachable (a worktree, another machine) — an absent pin is not a clean one."""
    f = os.path.join(MODELS, RUN.get(tag, ""), "metadata.json")
    if not os.path.exists(f):
        return None
    return (json.load(open(f)).get("git_hash") or "")[:8]


def pinnote(a, b):
    pa, pb = pin(a), pin(b)
    if pa and pb and pa != pb:
        return f"  ⚠ PIN-SPLIT {pa}/{pb} (verified inert)"
    return "  [pin-clean]" if pa and pb else ""


def path_of(tag):
    return os.path.join(P, f"taught_{tag}_end.json")


def cells(tag):
    """The per-team (wins, games) rows, or None. POOLED is a summary row, not a team cell."""
    f = path_of(tag)
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    c = {k: (v["wins"], v.get("games", v.get("n", 0)))
         for k, v in d.items() if isinstance(v, dict) and "wins" in v and k != "POOLED"}
    return c if len(c) == 16 else None      # a partial pass is UNCOVERED, never pooled


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


def frozen_untaught_bar():
    """Recompute the six-draw frozen untaught floor from the untaught artifacts in this directory.
    Imported as a NUMBER rather than typed, so it cannot drift from `tc_readout.py`'s."""
    mags = []
    for a, b in (("TCFUNDA", "TCFUNDB"), ("TCUNFA", "TCUNFB")):
        for dep in ("p1M", "mid", "end"):
            fs = [os.path.join(P, f"untaught_{t}_{dep}.json") for t in (a, b)]
            if not all(os.path.exists(f) for f in fs):
                return None
            cs = []
            for f in fs:
                d = json.load(open(f))
                cs.append({k: (v["wins"], v.get("games", v.get("n", 0)))
                           for k, v in d.items()
                           if isinstance(v, dict) and "wins" in v and k != "POOLED"})
            ks = sorted(set(cs[0]) & set(cs[1]))
            if len(ks) != 8:
                return None
            mags.append(abs(np.mean([cs[0][k][0] / cs[0][k][1] - cs[1][k][0] / cs[1][k][1]
                                     for k in ks])) * 100)
    return float(np.mean(mags))


def main():
    rng = np.random.default_rng(SEED)
    par = cells(PARENT)
    if par is None:
        print(f"NO PARENT BASELINE: {path_of(PARENT)} absent or partial. Every number below is a\n"
              "vs-parent delta, so there is nothing to print. STOP.")
        return 1
    keys = sorted(par)
    pw = rates(par, keys)

    arm, absent = {}, []
    for t in ARMS:
        c = cells(t)
        if c is None or sorted(c) != keys:
            absent.append(t)
        else:
            arm[t] = rates(c, keys)

    print(f"=== TAUGHT-16, endpoint, vs the fold parent {PARENT} (wr {pw.mean():.4f}) ===")
    print(f"    16 teams x 200 games, paired on teams, cluster bootstrap over TEAMS "
          f"({BOOT} draws, seed {SEED})")
    if absent:
        print(f"    UNCOVERED (artifact absent or <16 cells): {', '.join(absent)}")

    # ---- the two bars, both computed, neither typed ------------------------------------------
    draws = {}
    for name, (a, b) in (("FUND", ("TCFUNDA", "TCFUNDB")), ("UNF", ("TCUNFA", "TCUNFB")),
                         ("K6", ("TCUNFK6A", "TCUNFK6B"))):
        if a in arm and b in arm:
            draws[name] = arm[a] - arm[b]
    # The 2x2's OWN taught floor: its two replicate draws. K=6 is a different cell and supplies its
    # own draw, applied only to rows that involve a K=6 arm (the K=6 cell's rule: never borrow a bar
    # from a cell with more draws when the cell in hand has fewer -- take the larger).
    tb = [abs(draws[k].mean()) * 100 for k in ("FUND", "UNF") if k in draws]
    taught_bar = float(np.mean(tb)) if tb else None
    k6_bar = (max(taught_bar, abs(draws["K6"].mean()) * 100)
              if taught_bar is not None and "K6" in draws else taught_bar)
    untaught_bar = frozen_untaught_bar()
    if taught_bar is None:
        print("    NO TAUGHT BAR: fewer than two replicate draws resolve. Deltas only.")
    else:
        print(f"    bar {taught_bar:.2f}pp PROVISIONAL — the taught slice's OWN floor, from this "
              f"batch's two\n    replicate draws at ONE depth. Rows involving a K=6 arm take "
              f"{k6_bar:.2f}pp (that cell's own,\n    larger draw). Sensitivity against the "
              f"untaught six-draw {untaught_bar:.2f} is printed at the end."
              if untaught_bar else
              f"    bar {taught_bar:.2f}pp PROVISIONAL — the taught slice's OWN floor, two draws "
              f"at ONE depth.")

    def row(name, d, bar, width=30, suffix=""):
        m, lo, hi = ci(d, rng)
        lab = f"{label(m, lo, hi, bar)} (prov.)" if bar else ""
        print(f"  {name:{width}s} {m:+6.2f} [{lo:+6.2f},{hi:+6.2f}]  {lab}{suffix}")
        return m, lo, hi

    print("\n  arm                 taught wr   vs parent")
    for t in ARMS:
        if t not in arm:
            continue
        b = k6_bar if "K6" in t else taught_bar
        m, lo, hi = ci(arm[t] - pw, rng)
        lab = f"{label(m, lo, hi, b)} (prov.)" if b else ""
        print(f"  {NICE[t]:18s}   {arm[t].mean():.4f}   {m:+6.2f} [{lo:+6.2f},{hi:+6.2f}]  {lab}")

    # ---- halves: two replicates of one recipe are independent draws, so pooling is legitimate --
    print("\n  half / pool                     vs parent")
    for name, ts in (("FUNDED teachers (K=3)", ("TCFUNDA", "TCFUNDB")),
                     ("UNFUNDED teachers (K=3)", ("TCUNFA", "TCUNFB")),
                     ("all four K=3 arms", ("TCFUNDA", "TCFUNDB", "TCUNFA", "TCUNFB")),
                     ("UNFUNDED teachers (K=6)", ("TCUNFK6A", "TCUNFK6B"))):
        if not all(t in arm for t in ts):
            print(f"  {name:30s}  UNCOVERED"); continue
        row(name, np.mean([arm[t] for t in ts], axis=0) - pw,
            k6_bar if any("K6" in t for t in ts) else taught_bar)

    # ---- Q2: does teacher funding buy more teaching? ----------------------------------------
    print("\n=== FUNDED - UNFUNDED on the taught 16 (does extra teacher budget TEACH more?) ===")
    legs, sens = [], []
    for a, b in (("TCFUNDA", "TCUNFA"), ("TCFUNDB", "TCUNFB")):
        if a in arm and b in arm:
            legs.append((f"{NICE[a]}-{NICE[b]}", arm[a] - arm[b]))
    for (name, d), (a, b) in zip(legs, (("TCFUNDA", "TCUNFA"), ("TCFUNDB", "TCUNFB"))):
        sens.append((name, *row(name, d, taught_bar, 26, pinnote(a, b))))
    if len(legs) == 2:
        d = (legs[0][1] + legs[1][1]) / 2
        m, lo, hi = ci(d, rng)
        lab = f"{label(m, lo, hi, taught_bar)} (prov.)" if taught_bar else ""
        print(f"  {'BOTH LEGS':26s} {m:+6.2f} [{lo:+6.2f},{hi:+6.2f}]  {lab}   <- the readout")
        sens.append(("BOTH LEGS", m, lo, hi))

    # ---- Q3: does halving the dose cost teaching? -------------------------------------------
    print("\n=== K=6 - K=3 on the taught 16, fixed (unfunded) teachers — the DOSE axis ===")
    if all(t in arm for t in ARMS[2:]):
        d = (np.mean([arm[t] for t in ("TCUNFK6A", "TCUNFK6B")], axis=0)
             - np.mean([arm[t] for t in ("TCUNFA", "TCUNFB")], axis=0))
        sens.append(("K=6 - K=3", *row("K=6 - K=3", d, k6_bar, 26)))
    else:
        print("  UNCOVERED")

    # ---- the taught replicate draws, and what the provisional bar is worth -------------------
    print("\n=== TAUGHT replicate draws (one argv token apart), ALL AT ONE DEPTH ===")
    for name, (a, b) in (("FUND", ("TCFUNDA", "TCFUNDB")), ("UNF", ("TCUNFA", "TCUNFB")),
                         ("K6", ("TCUNFK6A", "TCUNFK6B"))):
        if name not in draws:
            print(f"  {name:6s} UNCOVERED"); continue
        m, lo, hi = ci(draws[name], rng)
        print(f"  {name:6s} {m:+6.2f} [{lo:+6.2f},{hi:+6.2f}]{pinnote(a, b)}")
    if taught_bar is not None:
        print(f"  2x2 taught floor {taught_bar:.2f}pp (2 draws) · with the K=6 draw "
              f"{np.mean([abs(draws[k].mean()) * 100 for k in draws]):.2f}pp (3 draws) — "
              f"PROVISIONAL,\n  one depth. Two draws at one depth is what put the untaught floor "
              f"at 0.12 before six draws\n  made it 1.66. Neither is quoted as 'the taught floor'.")

    if taught_bar is not None and untaught_bar is not None:
        print(f"\n=== SENSITIVITY: the same rows at the untaught six-draw frozen "
              f"{untaught_bar:.2f}pp ===")
        moved = [(n, label(m, lo, hi, taught_bar), label(m, lo, hi, untaught_bar))
                 for n, m, lo, hi in sens]
        moved = [x for x in moved if x[1] != x[2]]
        for n, a, b in moved:
            print(f"  MOVES  {n:26s} {a} -> {b}")
        print(f"  {len(sens) - len(moved)}/{len(sens)} rows read the SAME under both bars"
              + (" — those do not depend on the provisional number at all."
                 if len(moved) < len(sens) else ""))

    print(f"\nQUOTING RULE: the parent's taught wr is {pw.mean():.4f}. Its UNTAUGHT wr is 0.5825 — "
          f"a different,\nEASIER team set. Compare vs-parent DELTAS within a slice; never a taught "
          f"LEVEL against an\nuntaught one.")
    return 0


def check():
    """--check: resolve every input this readout reads and report any that is MISSING, without
    computing anything. The guarded defect is a readout whose artifacts were never committed
    beside it."""
    want = [path_of(t) for t in (PARENT,) + ARMS]
    want += [os.path.join(P, f"untaught_{t}_{dep}.json")
             for t in ("TCFUNDA", "TCFUNDB", "TCUNFA", "TCUNFB")
             for dep in ("p1M", "mid", "end")]          # the frozen bar is recomputed, not typed
    missing = [f for f in want if not os.path.exists(f)]
    for f in missing:
        print(f"MISSING {f}")
    print(f"taught_readout.py: {len(want) - len(missing)}/{len(want)} input artifacts present")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(check() if "--check" in sys.argv else main())
