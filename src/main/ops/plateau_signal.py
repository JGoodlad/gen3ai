"""THE PLATEAU SIGNAL — a REPORT trigger on the snapshot ladder. Registered ledger ae8e9fa9.

    python -m main.ops.plateau_signal <run-name|run-dir> [--dip-step N] [--fixed-fit]

Fires ONLY when BOTH hold at a new snapshot:
  (1) the new node is BELOW its predecessor on the ladder, for the SECOND CONSECUTIVE add;
  (2) pooled head-to-head vs its THREE most recent ancestors is below 0.50 with the Wilson
      95% interval EXCLUDING 0.50 — again.

Firing is a REPORT TRIGGER, never a kill. THE KILL STAYS G7.

If the new node comes in ABOVE the pre-dip node, or the pooled interval includes 0.50, the dip
node is recorded as a ONE-OFF and nothing further is said about it.

Also prints, per the registration: the head-to-head profile, and the new node against the DIP
node specifically — a node that beats the dip but not the pre-dip node is a different shape from
one that beats neither.

Promoted 2026-09-07 from the Training Run session's ``plateau_signal.py`` (the run directory and
the dip step were literals; ``--dip-step`` is now REQUIRED rather than defaulting to one arm's).
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Sequence

from main.ops.run_ref import refuse, resolve_run_dir


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


# ── PURE CLAUSE LOGIC + PLANTED-CASE SELF-CHECK ────────────────────────────────────────────
# Registered 2026-09-07 after clause 1 false-fired at 30M: it compared against the DIP node
# five adds back instead of the immediately preceding add, so it was destined to fire on the
# next dip regardless of consecutiveness. The AND with clause 2 hid it — a broken clause can
# ride inside correct verdicts indefinitely. The general fix is not a better condition, it is
# a PLANTED CASE PER CLAUSE, run before every real read, refusing if either misreads.

def clause1_second_consecutive_dip(vals) -> bool:
    """vals: ladder ratings in step order. True iff the last TWO adds were both below."""
    if len(vals) < 3:
        return False
    return vals[-1] < vals[-2] and vals[-2] < vals[-3]


def clause2_pooled_below(wins: int, games: int) -> bool:
    """True iff the Wilson 95% interval for wins/games lies ENTIRELY below 0.50."""
    if games <= 0:
        return False
    return wilson(wins, games)[1] < 0.5


def self_check() -> None:
    """Planted cases. Each isolates ONE clause; a misread REFUSES the whole run."""
    fails = []
    # clause 1 — only the last two adds may decide
    if clause1_second_consecutive_dip([100.0, 110.0, 105.0]):
        fails.append("clause1: a SINGLE dip (100,110,105) read as the second consecutive")
    if not clause1_second_consecutive_dip([110.0, 105.0, 100.0]):
        fails.append("clause1: TWO consecutive dips (110,105,100) not recognised")
    if clause1_second_consecutive_dip([100.0, 105.0, 110.0]):
        fails.append("clause1: a RISING ladder read as a dip")
    # the exact shape that false-fired: an old dip, then up, then one small dip
    if clause1_second_consecutive_dip([100.0, 90.0, 120.0, 119.0]):
        fails.append("clause1: an OLD dip five adds back still decides (the 30M defect)")
    # clause 2 — only the interval may decide
    if not clause2_pooled_below(100, 300):
        fails.append("clause2: 100/300 (clearly below) not detected")
    if clause2_pooled_below(150, 300):
        fails.append("clause2: 150/300 (covers 0.50) read as below")
    if clause2_pooled_below(200, 300):
        fails.append("clause2: 200/300 (clearly ABOVE) read as below")
    if fails:
        print("=== PLATEAU SIGNAL — REFUSING: SELF-CHECK FAILED ===")
        for f in fails:
            print(f"  {f}")
        print("  The clause logic does not behave on planted cases, so the real read cannot")
        print("  be trusted. Nothing is concluded.")
        raise SystemExit(3)


#: 🚨 DETECT THE LADDER FIX BY AN ARTIFACT THE NEW SEMANTICS EMIT, never by a capability that
#: may predate it and never by source text. A first attempt gated on ``write=``, which fit_ladder
#: has carried since 462e9cdb — it ACCEPTED and labelled a still-biased refit "sentinel edges
#: dropped". A second scanned the source, but a docstring phrase is not a contract. The fix
#: (f9a3ddf6) is an UNCONDITIONAL body change with NO new parameter; its positive artifact is a
#: key in the returned dict holding the COUNT of snap-vs-snap eval rows excluded. A pre-fix tree
#: cannot emit it.
FIX_KEY = "eval_sentinel_edges_dropped"


def main(argv: Sequence[str] | None = None) -> int:  # noqa: C901 - one registered read, kept whole
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if args else 2
    self_check()

    run = resolve_run_dir(args[0])
    if "--dip-step" not in args:
        refuse("REFUSING: --dip-step is required.",
               "  The registered read names the DIP NODE explicitly; a default would silently",
               "  score one arm's dip against another arm's ladder. Nothing is concluded.")
    dip = int(args[args.index("--dip-step") + 1])
    use_fixed = "--fixed-fit" in args

    # ── WHICH FIT TO READ ──────────────────────────────────────────────────────────────────
    # 🚨 THE COMMITTED ladder.json CARRIES A KNOWN BIAS (ledger f9a3ddf6): fit_ladder folded the
    # eval cycles' greedy-vs-STOCHASTIC sentinel edges into the greedy-vs-greedy frozen matrix,
    # inflating the newest nodes by ~21-29 Elo. The inflation is NOT UNIFORM (-21, -23, -23, -29
    # measured across four nodes), so a WITHIN-FIT difference does not cancel it: an add can move
    # by ~6 Elo and a NEAR-ZERO add can CHANGE SIGN. Clause 1's threshold is a comparison against
    # zero, so its verdicts are affected.
    if use_fixed:
        from agents.training.snapshot_ladder import fit_ladder
        lad = fit_ladder(str(run), write=False)
        if FIX_KEY not in lad:
            print("=== PLATEAU SIGNAL — REFUSING ===")
            print(f"  --fixed-fit requested, but fit_ladder's result carries no '{FIX_KEY}'.")
            print("  The f9a3ddf6 fix is REGISTERED but NOT LANDED here, so this refit still folds")
            print("  in the greedy-vs-stochastic sentinel edges (+8.9pp per pair) while looking")
            print("  corrected. Refusing.")
            print(f"  keys present: {sorted(lad)}")
            print("  If the landing named the key differently, update FIX_KEY — do NOT relax the")
            print("  check to a source-text or parameter test; both have already failed here.")
            return 2
        bias_note = (f"FIXED FIT (refit from games.jsonl; {lad[FIX_KEY]} eval sentinel edge(s) "
                     f"dropped, per '{FIX_KEY}')")
    else:
        ladder_json = run / "snapshot_ladder" / "ladder.json"
        if not ladder_json.exists():
            refuse(f"REFUSING: {ladder_json} does not exist — there is no ladder to read.")
        lad = json.load(open(ladder_json))
        bias_note = ("committed ladder.json — CARRIES THE f9a3ddf6 BIAS (~21-29 Elo newest-node "
                     "inflation, non-uniform, so near-zero adds may change sign under the fix)")
    r = {int(k): v for k, v in lad["ratings"].items()}
    se = {int(k): v for k, v in lad.get("se", {}).items()}
    nodes = sorted(r)

    games = {}
    with open(run / "snapshot_ladder" / "games.jsonl") as f:
        for line in f:
            g = json.loads(line)
            games[(g["a"], g["b"])] = (g["wins_a"], g["games"])

    def h2h(x, y):
        """wins of x over y, games."""
        if (x, y) in games:
            w, n = games[(x, y)]
            return w, n
        if (y, x) in games:
            w, n = games[(y, x)]
            return n - w, n
        return None

    new = nodes[-1]

    # 🚨 THE LADDER MUST BE CAUGHT UP WITH THE SNAPSHOTS ON DISK. The run's updater plays each
    # new node on promotion and takes minutes; run before it finishes and this tool reports the
    # PREVIOUS node's verdict while looking entirely current. That is the same failure shape as
    # reading a compile count from a ring buffer or a restart gap before the boundary lands:
    # every precondition unmet, every number clean.
    snapdir = run / "snapshots"
    snap_steps = sorted(
        int("".join(c for c in f if c.isdigit()))
        for f in os.listdir(snapdir) if f.endswith(".zip")
    ) if snapdir.is_dir() else []
    missing = [s_ for s_ in snap_steps if s_ not in r]
    if missing:
        print("=== PLATEAU SIGNAL — REFUSING ===")
        print(f"  {len(missing)} snapshot(s) on disk have NO ladder node yet: "
              + ", ".join(f"{m/1e6:.0f}M" for m in missing))
        print(f"  ladder newest = {new/1e6:.0f}M, snapshots newest = {snap_steps[-1]/1e6:.0f}M")
        print("  The run's own updater is still playing them. Running now would report the")
        print("  PREVIOUS node's verdict as if it were the new one. Wait for the updater;")
        print("  never --backfill concurrently.")
        return 2

    print("=== PLATEAU SIGNAL (registered 2026-09-07, ledger ae8e9fa9) ===")
    print(f"  run {run}")
    print(f"  fit source: {bias_note}")
    print(f"  newest node {new:,}   ladder {r[new]:.1f} ± {se.get(new, 0):.1f}")
    print(f"  dip node    {dip:,}   ladder {r.get(dip, float('nan')):.1f}\n")

    print("  head-to-head profile of the newest node:")
    for nd in nodes[:-1]:
        hh = h2h(new, nd)
        if hh:
            w, n = hh
            lo, hi = wilson(w, n)
            mark = "  <- BELOW" if hi < 0.5 else ("  <- below .50" if w / n < 0.5 else "")
            print(f"    {new/1e6:5.0f}M vs {nd/1e6:5.0f}M   {w:>3}/{n}  = {w/n:.2f}  "
                  f"[{lo:.3f}, {hi:.3f}]{mark}")

    # clause 1 — below predecessor for a SECOND consecutive add
    # 🚨 "SECOND CONSECUTIVE" MEANS TWO ADDS IN A ROW: the newest node below ITS predecessor
    # AND that predecessor below ITS OWN predecessor. An earlier version compared against the DIP
    # node five adds back, which false-fired at 30M (28M was ABOVE its predecessor, so 30M was the
    # FIRST below-predecessor add, not the second). The signal did not fire only because clause 2
    # is ANDed and happened to miss.
    prev = nodes[-2] if len(nodes) >= 2 else None
    prev2 = nodes[-3] if len(nodes) >= 3 else None
    below_pred = prev is not None and r[new] < r[prev]
    prev_was_below = prev2 is not None and r[prev] < r[prev2]
    c1 = clause1_second_consecutive_dip([r[n_] for n_ in nodes])   # the SELF-CHECKED path
    if prev is not None:
        print(f"\n  clause 1  newest {r[new]:.1f} vs predecessor {r[prev]:.1f} "
              f"({'BELOW' if below_pred else 'above'});")
    if prev2 is not None:
        print(f"            predecessor {r[prev]:.1f} vs ITS predecessor "
              f"{r[prev2]:.1f} ({'BELOW' if prev_was_below else 'above'})")
    else:
        print("            (no third node yet)")
    print(f"            SECOND CONSECUTIVE below-predecessor add: {'YES' if c1 else 'no'}")
    if not use_fixed:
        print("            ⚠️  read off the BIASED fit — a near-zero add may change sign under the")
        print("               f9a3ddf6 fix. A clause-1 YES here is NOT meaningful without a")
        print("               fixed-fit confirmation.")

    # clause 2 — pooled vs the three most recent ancestors
    anc = [n for n in nodes if n < new][-3:]
    W = N = 0
    for nd in anc:
        hh = h2h(new, nd)
        if hh:
            W += hh[0]
            N += hh[1]
    lo, hi = wilson(W, N)
    c2 = clause2_pooled_below(W, N)                                # the SELF-CHECKED path
    print(f"\n  clause 2  pooled vs the recent three "
          f"({', '.join(f'{a/1e6:.0f}M' for a in anc)}): {W}/{N} = "
          f"{W/N if N else float('nan'):.3f}  Wilson [{lo:.3f}, {hi:.3f}]")
    print(f"            interval EXCLUDES 0.50: {'YES' if c2 else 'no'}")

    # the registered extra: new vs the dip node specifically
    if dip in r and new != dip:
        hh = h2h(new, dip)
        if hh:
            w, n = hh
            l2, h2 = wilson(w, n)
            print(f"\n  vs the DIP node ({dip/1e6:.0f}M): {w}/{n} = {w/n:.2f}  [{l2:.3f}, {h2:.3f}]")
            print("    a node that beats the dip but not the pre-dip node is a DIFFERENT SHAPE")
            print("    from one that beats neither — the registration asks for both.")

    # The registered ONE-OFF condition names the PRE-DIP node explicitly, so make that
    # determination mechanical too rather than leaving it to a reading at the time.
    predip = None
    if dip in r:
        i = nodes.index(dip)
        if i > 0:
            predip = nodes[i - 1]
    if predip is not None and new != dip:
        above_predip = r[new] > r[predip]
        print(f"\n  ONE-OFF CONDITION (registered): newest {r[new]:.1f} vs the PRE-DIP node "
              f"{predip/1e6:.0f}M {r[predip]:.1f} -> {'ABOVE' if above_predip else 'below'}")
        if above_predip or not c2:
            print(f"    => the {dip/1e6:.0f}M node is recorded as a ONE-OFF; nothing further is")
            print("       said about it. (Registered: fires if newest is ABOVE the pre-dip node,")
            print("       OR the pooled interval includes 0.50.)")

    fires = c1 and c2
    print(f"\n  => PLATEAU SIGNAL {'FIRES — REPORT to the orchestrator' if fires else 'does not fire'}")
    print("     This is a REPORT trigger, never a kill. THE KILL STAYS G7.")
    if not fires and new != dip:
        print(f"     Not firing means the {dip/1e6:.0f}M node is recorded as a ONE-OFF and")
        print("     nothing further is said about it.")
    return 1 if fires else 0


if __name__ == "__main__":
    raise SystemExit(main())
