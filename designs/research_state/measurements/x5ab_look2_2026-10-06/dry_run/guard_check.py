"""The plan / read guards on FAKE ledger cells (the real ledger refused both tamper attempts before they could be
planted: a same-seed replay is DuplicateBatchError, another schedule seed under the request is a spec conflict)."""
import sys
from pathlib import Path
from types import SimpleNamespace as NS

D = Path(__file__).resolve().parent.parent   # the look-2 measurement directory
sys.path.insert(0, str(D))
import plan_look2 as P  # noqa: E402
import read_look2 as R  # noqa: E402

S = Path("/home/goodlad/.cache/gen3ai/x5look2/dry")
M = S / "models"
got, fails = P.resolve(M, [1001, 1002])
assert not fails, fails
rows, cols = P.rows_cols(got, "steps")
sh = {s: (P.sha(rows[s]), P.sha(cols[s])) for s in (1001, 1002)}


def cell(r, c, n=1000):
    sc = {(1001, 1001): 0.48, (1001, 1002): 0.51, (1002, 1001): 0.47, (1002, 1002): 0.53}[(r, c)]
    return NS(player=sh[r][0], opponent=sh[c][1], player_id=f"fm{r}", opponent_id=f"blob{c}", n_pairs=n,
              verdict="OK", reasons=(), score=sc, w=1, l=1, d=0, aborted=0, pairs=(0, 0, n, 0, 0))


def run(l1, l2, label):
    P.look_cells = lambda root, fam, req: l1 if req.startswith("x5ab_look1") else l2
    try:
        cells, _ = P.plan(S / "ledger", M, "steps", [1001, 1002], [1001])
        out = f"PLANNED {len(cells)}"
    except P.Refused as e:
        out = f"REFUSED: {e}"
    rd = R.read_one(S / "ledger", "steps", got, [1001, 1002], [1001], [], 1000)
    print(f"{label}\n  plan: {out}\n  read: {rd['verdict']} {rd['decision']['reasons']}")


L1 = [cell(1001, 1001)]
NEW = [cell(1001, 1002), cell(1002, 1001), cell(1002, 1002)]
run(L1, NEW, "A. clean")
run(L1, NEW + [cell(1001, 1001)], "B. a look-1 cell replayed under look 2")
run(L1 + [cell(1002, 1002)], NEW, "C. look 1 holds a cell outside its registered 1 x 1")
run([], NEW, "D. look 1 missing")
run(L1, NEW[:2], "E. look 2 incomplete (one new cell missing)")
