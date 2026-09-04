"""Re-read every untaught delta against the MEASURED replicate floor.

The floor is the spread between two byte-identical arms (differing only in --run-name), so it is
what the instrument cannot separate from a re-run of the same recipe. Deltas are cluster-bootstrapped
over TEAMS (8 clusters) because between-team variance dominates and more games per team does not
shrink it.

Three labels, and keeping them apart is the point:
  SIGNIFICANT   |d| >= floor AND the CI excludes zero.
  WITHIN FLOOR  |d| <  floor. The CI may still exclude zero -- that says the GAMES are consistent,
                not that the ARM differs. Two byte-identical folds differ by the floor.
  NOT DETECTED  |d| >= floor but the CI spans zero.

Depth-specific: each depth gets its OWN floor from the N1/N2 pair at that depth. Applying the
endpoint floor across depths is over-conservative when interior floors are smaller.

WHAT THIS FLOOR IS A FLOOR FOR. N1/N2 ran CONTROLLER-LIVE at grad_accum_steps=2 (no --fork-lr, no
freeze), so 5.94pp is the controller-live fold floor and it carries the KL controller's own
step-size wander inside it. A frozen-dose arm (--fork-lr-freeze, K=3) is a different regime and
this floor is not measured on it -- the 2x2 replicate pairs measure that one directly. Do not
quote 5.94 as the floor for a frozen arm.

Run: python floor_reread.py [depth ...]   (default: every depth with a complete 8/8 N1/N2 pair)
     (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""
import json, os, sys, numpy as np

P = os.path.dirname(os.path.abspath(__file__))
PARENT = 'R2ACTION'          # the fold parent; baseline 932/1600 = 0.5825, NEVER re-measured
NOFOLD_FLOOR = 4.19          # from two byte-identical NO-FOLD runs
# THE DEFAULT BAR (Model Review 2 ruling 2026-09-03): the POOLED floor over all three depths of the
# N1/N2 pair, 4.27pp CI [+1.23,+6.92]. Own-depth floors were set aside because two of the three were
# single-pair bars whose CI spans zero (clause 3 forbids the smaller bar on one pair) and the three
# depths are not distinguishable from each other, so clause 5's premise did not fire.
# CAVEAT that rides with every quote of 4.27: pooling three depths of the SAME two runs adds games
# and checkpoints, NOT training draws. The independent fold draws behind it are still ONE, and N2
# sits below N1 at all three depths. The interval is over TEAMS, not over draws.
POOLED_FLOOR = 4.27
# Every arm of the fold experiment takes max(NOFOLD_FLOOR, fold floor), coefficient irrelevant
# (Model Review 2 ruling 2026-09-03): a coef-0 arm has no measured replicate spread of its own, so
# it gets no smaller bar BY ASSUMPTION. FOLDS is kept only to mark which pairs are fold-on-BOTH-
# sides, which is the one case clause 1 lets take the bare fold floor.
FOLDS = {'R4ACTION', 'R4DOSE12', 'R4DOSE6', 'R4DOSE3', 'B2', 'N1', 'N2'}   # C1 is coef-0

def load(tag):
    f = os.path.join(P, f'untaught_{tag}.json')
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    return {k: (v['wins'], v.get('games', v.get('n', 0))) for k, v in d.items()
            if isinstance(v, dict) and 'wins' in v and k != 'POOLED'}   # POOLED is not a cell

def delta_ci(a, b, B=20000, seed=20260903):
    rng = np.random.default_rng(seed)
    teams = sorted(set(a) & set(b))
    d = np.array([a[t][0]/a[t][1] - b[t][0]/b[t][1] for t in teams])
    boot = np.array([d[rng.integers(0, len(teams), len(teams))].mean() for _ in range(B)])
    return d.mean()*100, np.percentile(boot, 2.5)*100, np.percentile(boot, 97.5)*100, len(teams)

def label(d, lo, hi, bar):
    if abs(d) < bar:
        return 'WITHIN FLOOR'
    return 'NOT DETECTED' if lo <= 0 <= hi else 'SIGNIFICANT'

def floor_at(depth):
    """The fold floor at one depth, or None if the N1/N2 pair is not 8/8 there."""
    a, b = load(f'N1_{depth}'), load(f'N2_{depth}')
    if not a or not b or len(a) < 8 or len(b) < 8:
        return None, (len(a) if a else 0, len(b) if b else 0)
    d, lo, hi, _ = delta_ci(a, b)
    return (abs(d), d, lo, hi), (8, 8)

def main():
    depths = sys.argv[1:] or ['p1M', 'mid', 'end']
    use_own = os.environ.get('FLOOR_OWN_DEPTH') == '1'   # the set-aside policy, for re-checking only
    for depth in depths:
        f, cells = floor_at(depth)
        print(f'\n=== depth {depth} ===')
        if f is None:
            print(f'  N1/N2 pair INCOMPLETE ({cells[0]}/8, {cells[1]}/8 cells) — no floor, '
                  f'every verdict at this depth is UNCOVERED, never borrowed from another depth')
            continue
        own, d, lo, hi = f
        fl = own if use_own else POOLED_FLOOR
        print(f'  bar {fl:.2f}pp  ({"own-depth (SET ASIDE policy)" if use_own else "POOLED, the ruling"})'
              f'   [own-depth floor at this depth: {own:.2f}, N1−N2 {d:+.2f} [{lo:+.2f},{hi:+.2f}], 8 teams]')
        par = load(PARENT)
        rows = []
        for arm in ('R4ACTION', 'R4DOSE12', 'R4DOSE6', 'R4DOSE3', 'B2', 'N1', 'N2', 'C1'):
            a = load(f'{arm}_{depth}') or (load(arm) if arm == 'R4ACTION' else None)
            if not a:
                print(f'  {arm}_{depth}: UNCOVERED'); continue
            bar = max(NOFOLD_FLOOR, fl)          # ruling: same bar whatever the coefficient
            rows.append((f'{arm} vs {PARENT}', *delta_ci(a, par)[:3], bar))
        for a, b in (('C1', 'B2'), ('B2', 'N1'), ('B2', 'N2'), ('R4DOSE3', 'R4DOSE12'),
                     ('R4DOSE6', 'R4DOSE12'), ('R4DOSE3', 'R4DOSE6'), ('B2', 'R4DOSE3')):
            x, y = load(f'{a}_{depth}'), load(f'{b}_{depth}')
            if not x or not y:
                continue
            both = a in FOLDS and b in FOLDS
            rows.append((f'{a} vs {b}', *delta_ci(x, y)[:3], fl if both else max(NOFOLD_FLOOR, fl)))
        for name, dd, l, h, bar in rows:
            print(f'  {name:26s} {dd:+7.2f} [{l:+7.2f},{h:+7.2f}]  bar {bar:5.2f}  {label(dd,l,h,bar)}')

if __name__ == '__main__':
    main()
