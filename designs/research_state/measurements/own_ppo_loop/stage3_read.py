"""E3 for U4: base (pre-U4) vs U4, the root smoke at seed 42. Identity bar on every base point; U4 may add
ONLY points at the final step (P3's final dump)."""
import sys, math
exec(open(__import__("pathlib").Path(__file__).with_name("tb_compare.py")).read().split("a, b = load")[0])
a, b = load(sys.argv[1]), load(sys.argv[2])
WALL = ("time/fps", "time/time_elapsed")
def wall(t): return t in WALL or t.endswith("_ms") or t.endswith("_s") or "_ms_per_" in t or t.endswith("decisions_per_s")
def eq(x, y): return x[0] == y[0] and (x[1] == y[1] or (x[1] is not None and y[1] is not None and math.isnan(x[1]) and math.isnan(y[1])))
last = max(s for v in a.values() for s, _ in v)
ok = extra_tags = 0; bad = []
for t in sorted(set(a) | set(b)):
    A, B = a.get(t, []), b.get(t, [])
    if len(B) < len(A) or not (wall(t) or all(eq(x, y) for x, y in zip(A, B))):
        bad.append(t); continue
    tail = B[len(A):]
    if any(s != last for s, _ in tail):
        bad.append(t); continue
    ok += 1; extra_tags += bool(tail)
print(f"final step {last}; {ok} tags consistent ({extra_tags} carry the extra final-step point); BAD {len(bad)}: {bad[:10]}")
print("tags only in U4:", sorted(set(b) - set(a)))
