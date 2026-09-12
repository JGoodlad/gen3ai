"""Guard against the parent measurement's HAZARD 6 — a hand-typed number in a README.

Every 4-decimal number that appears in README.md must exist somewhere in `frame_check.json` or
`branch_test.json` (as a value, or as a value rounded to 3-4 places, or as a simple ratio the
tables already carry). Prints the ones it cannot find so they can be checked by hand or removed.
Not committed as a gate — a one-shot pre-commit check, run and its output recorded in §7.
"""
import json
import re
import sys

txt = open(sys.argv[1]).read()
pool = set()


def walk(o):
    if isinstance(o, dict):
        for v in o.values():
            walk(v)
    elif isinstance(o, list):
        for v in o:
            walk(v)
    elif isinstance(o, (int, float)) and not isinstance(o, bool):
        for nd in (2, 3, 4):
            pool.add(f"{abs(o):.{nd}f}".lstrip("0") if abs(o) < 1 else f"{abs(o):.{nd}f}")
        pool.add(str(abs(o)))


for f in sys.argv[2:]:
    walk(json.load(open(f)))

missing = []
for m in re.finditer(r"(?<![\w.])(\d+\.\d{2,4})(?![\w])", txt):
    s = m.group(1)
    if s in pool or s.lstrip("0") in pool or ("0" + s) in pool:
        continue
    missing.append(s)
print(f"{len(set(missing))} distinct numbers in {sys.argv[1]} not found in the JSONs:")
for s in sorted(set(missing)):
    print("  ", s)
