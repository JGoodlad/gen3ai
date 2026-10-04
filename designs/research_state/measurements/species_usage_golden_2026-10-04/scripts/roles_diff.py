"""Diff two `python -m main.belief_roles roles` outputs (the parent tree's and this tree's): the role-set sha,
roles added / dropped, every role's expected carriers per team before -> after, substitute-pair counts and the
rule-8 exclusions.

    python roles_diff.py roles_before.json roles_after.json > out/roles_diff.txt
"""
import json
import sys

b, a = (json.load(open(p)) for p in sys.argv[1:3])
rb = {r["move"]: r["carriers"] for r in b["roles"]}
ra = {r["move"]: r["carriers"] for r in a["roles"]}
print(f"role set sha {b['sha256'][:12]} ({len(rb)} roles) -> {a['sha256'][:12]} ({len(ra)} roles)")
print(f"ADDED   {sorted(set(ra) - set(rb))}")
print(f"DROPPED {sorted(set(rb) - set(ra))}")
print(f"substitute pairs {len(b['pairs'])} -> {len(a['pairs'])}")
print(f"excluded before {b['excluded']}\nexcluded after  {a['excluded']}")
print("expected carriers per team (role: before -> after, ratio):")
for m in sorted(set(rb) | set(ra), key=lambda m: -ra.get(m, rb.get(m, 0.0))):
    x, y = rb.get(m), ra.get(m)
    print(f"  {m:14s} {x if x is None else round(x, 4)} -> {y if y is None else round(y, 4)}"
          + (f"  (x{y / x:.3f})" if x and y else ""))
