"""Compare two census dumps (A vs B): per seed, per column, how many ops moved; for OBS the moved (op, env, side) rows and the named slices of the moved cells."""
import sys, os, collections
import numpy as np
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.observation.schema import build_schema
A, B = sys.argv[1], sys.argv[2]
L = Gen3ObservationEncoder(load_mappings()).get_layout()
sl = build_schema(L).slices()
names = sorted(((s.start, s.stop, n) for n, s in sl.items()), key=lambda t: (t[0], t[1]))
def name_of(i):
    best = None
    for a, b, n in names:
        if a <= i < b and (best is None or (b - a) < (best[1] - best[0])):
            best = (a, b, n)
    return best[2] if best else f"cell{i}"
OBS_DIM = int(L["total_dim"]) if "total_dim" in L else 2845
n_env, sides = 8, 2
tot = collections.Counter()
for seed in (11, 12, 13):
    cols = sorted(f.split(".", 1)[1] for f in os.listdir(A) if f.startswith(f"s{seed}."))
    for c in cols:
        a = open(f"{A}/s{seed}.{c}", "rb").read(); b = open(f"{B}/s{seed}.{c}", "rb").read()
        if a == b: continue
        if c == "OBS":
            fa = np.frombuffer(a, dtype="<f4").reshape(-1, n_env, sides, OBS_DIM); fb = np.frombuffer(b, dtype="<f4").reshape(-1, n_env, sides, OBS_DIM)
            m = np.any(fa.view("<u4") != fb.view("<u4"), axis=3)
            cells = collections.Counter(name_of(int(i)) for i in np.flatnonzero(np.any((fa.view("<u4") != fb.view("<u4")).reshape(-1, OBS_DIM), axis=0)))
            print(f"seed {seed} OBS: {int(m.sum())} of {m.size} (op, env, side) rows moved; ops with any move {int(m.any(axis=(1,2)).sum())}/{m.shape[0]}; slices: {dict(cells)}")
            # the moved cells' first op per env
        else:
            fa = np.frombuffer(a, dtype=np.uint8); fb = np.frombuffer(b, dtype=np.uint8)
            n = min(len(fa), len(fb))
            print(f"seed {seed} {c}: bytes differ ({int((fa[:n] != fb[:n]).sum())} of {n}; lengths {len(fa)} {len(fb)})")
