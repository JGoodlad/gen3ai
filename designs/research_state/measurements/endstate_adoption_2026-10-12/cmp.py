import json
import sys

a = json.load(open(sys.argv[1]))["resolved"]
b = json.load(open(sys.argv[2]))["resolved"]
print(f"A = {sys.argv[1]}\nB = {sys.argv[2]}")
for k in sorted(set(a) | set(b)):
    if k not in a:
        print(f"  only in B: {k} = {b[k]!r}")
    elif k not in b:
        print(f"  only in A: {k} = {a[k]!r}")
    elif a[k] != b[k]:
        print(f"  DIFF {k}: A={a[k]!r} B={b[k]!r}")
print("compared", len(set(a) | set(b)), "keys")
