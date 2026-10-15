"""T33 s4002: each Tier-1 check's newest-node score, read from a tick_stderr.log.

score = (W + D/2) / games, pooled over the check's `[h2h] batch` W/L/D lines (W = the newest node's wins).
Usage: python3 -I scores.py <tick_stderr.log>  ->  TSV: check, pairs, W, L, D, score %
"""
import re
import sys

cur = None
acc: dict[int, list[int]] = {}
with open(sys.argv[1], encoding="utf-8") as fh:
    for line in fh:
        m = re.search(r"\[plateau\] check ([\d,]+):", line)
        if m:
            cur = int(m.group(1).replace(",", ""))
            acc[cur] = [0, 0, 0]
            continue
        m = re.search(r"W/L/D (\d+)/(\d+)/(\d+)", line)
        if m and cur is not None:
            for i in range(3):
                acc[cur][i] += int(m.group(i + 1))
print("check\tpairs\tW\tL\tD\tscore_pct")
for k, (w, l, d) in sorted(acc.items()):
    n = w + l + d
    print(f"{k}\t{n // 2}\t{w}\t{l}\t{d}\t{100 * (w + d / 2) / n:.1f}")
