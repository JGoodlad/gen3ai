"""Render results.json (analyze.py) into markdown tables: levels and contrasts, per family/metric/stratum.

    python render.py <results.json> > tables.md
"""
import json
import sys

ARMS = ("pool", "ladder", "procedural")
SHORT = {"pool": "a POOL", "ladder": "b LADDER", "procedural": "c PROC"}


def f(v, nd):
    return f"{v:.{nd}f}"


def cell(x, nd=3):
    pt, lo, hi = x
    return f"{f(pt, nd)} [{f(lo, nd)}, {f(hi, nd)}]"


def flag(x):
    _, lo, hi = x
    return "" if lo <= 0 <= hi else (" ▲" if lo > 0 else " ▼")


def main(path):
    r = json.load(open(path))
    arms = tuple(r.get("arms", ARMS))
    print(f"n paired battles = {r['n_paired_battles']}\n")
    for fam, strata in r["families"].items():
        mets = [m for m in strata["all"] if m != "n_slots"]
        for m in mets:
            nd = 4 if m in ("bce",) else 3
            print(f"### {fam} · {m}\n")
            print("| k | slots " + "/".join(arms) + " | " + " | ".join(f"head {a}" for a in arms) + " | "
                  + " | ".join(f"prior {a}" for a in arms) + " |")
            print("|" + "---|" * (2 + 2 * len(arms)))
            for st, d in strata.items():
                ns = "/".join(str(d["n_slots"][a]) for a in arms)
                row = [st, ns] + [cell(d[m][a]["head"], nd) for a in arms] + [cell(d[m][a]["prior"], nd) for a in arms]
                print("| " + " | ".join(row) + " |")
            print()
            base = arms[0]
            print("| k | " + " | ".join(f"Δ {a}−{base} (head)" for a in arms[1:]) + " | "
                  + " | ".join(f"A {a}" for a in arms) + " | "
                  + " | ".join(f"D {a} = A {base} − A {a}" for a in arms[1:]) + " |")
            print("|" + "---|" * (1 + 2 * (len(arms) - 1) + len(arms)))
            for st, d in strata.items():
                x = d[m]
                row = ([st] + [cell(x[f"delta_{a}"], nd) + flag(x[f"delta_{a}"]) for a in arms[1:]]
                       + [cell(x[a]["A"], nd) + flag(x[a]["A"]) for a in arms]
                       + [cell(x[f"D_{a}"], nd) + flag(x[f"D_{a}"]) for a in arms[1:]])
                print("| " + " | ".join(row) + " |")
            print()


if __name__ == "__main__":
    main(sys.argv[1])
