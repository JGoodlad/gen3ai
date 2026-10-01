import sys, glob, math
sys.argv = sys.argv
exec(open(__import__("pathlib").Path(__file__).with_name("tb_compare.py")).read().split("a, b = load")[0])
a, b = load(sys.argv[1]), load(sys.argv[2])
def eq(x, y):
    return x[0] == y[0] and (x[1] == y[1] or (x[1] is not None and y[1] is not None and math.isnan(x[1]) and math.isnan(y[1])))
bad = []
for t in sorted(set(a) | set(b)):
    A, B = a.get(t, []), b.get(t, [])
    if len(A) == len(B) and all(eq(x, y) for x, y in zip(A, B)): continue
    bad.append(t)
print(len(set(a)|set(b)), "tags;", len(bad), "differ (incl. wall clocks):")
print(" ", bad)
