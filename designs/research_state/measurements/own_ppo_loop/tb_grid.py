import sys
exec(open(__import__("pathlib").Path(__file__).with_name("tb_compare.py")).read().split("a, b = load")[0])
def grid(d): return {t: [s for s, _ in v] for t, v in load(d).items()}
for x, y in [(sys.argv[1], sys.argv[2]), (sys.argv[3], sys.argv[4])]:
    a, b = grid(x), grid(y)
    print(x.split("/")[-1], y.split("/")[-1], "tags equal:", set(a) == set(b), "| tags whose step grid differs:", [t for t in a if a[t] != b.get(t)][:10])
