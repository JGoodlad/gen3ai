"""H1's non-saturating read: the PAIRED-SEED diagonal (static seed i vs legacy seed i) at ~5M and ~10M, played
on CPU by `main.h2h play-many` at the pin into a ledger root OUTSIDE models/ (this study's own, purpose `audit`).
Seeds do not pair runs, so the diagonal is just a fixed, balanced subset of the cross; its 15M counterpart is the
look-1/2 cross's own diagonal (47.33 / 50.52 / 48.15 / 48.95 / 44.40). DESCRIPTIVE.
Writes the cells JSON for --cells (argv[1]: 5M | 10M | both)."""
import json
import sys
from pathlib import Path

A = Path("/home/goodlad/gen3ai_archive/static_diag_2026-10-09")
plan = dict(reversed(l.split()) for l in (A / "plan.txt").read_text().splitlines() if l.strip())   # label -> zip
which = sys.argv[1]
stages = ("5M", "10M") if which == "both" else (which,)
cells = [[plan[f"S{i}_{st}"], plan[f"L{i}_{st}"]] for st in stages for i in range(1, 6)]
print(json.dumps(cells, indent=1))
