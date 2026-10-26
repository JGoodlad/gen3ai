import numpy as np
from collections import Counter
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.observation.schema import build_schema
from agents.observation import gen3_effects as G
a = np.load("/tmp/gigo_audit/me/golden_head.npy"); b = np.load("/tmp/gigo_audit/me/golden_fix.npy")
print("shapes", a.shape, b.shape)
L = Gen3ObservationEncoder(load_mappings()).get_layout()
sl = build_schema(L).slices()
names = {}
for k, s in sl.items():
    for i in range(s.start, s.stop):
        names.setdefault(i, k)
ctx = L["parts"]["active_context"] if "active_context" in L["parts"] else None
rows = np.flatnonzero((a != b).any(1))
print("rows differing", len(rows))
cells = Counter()
for r in rows:
    for i in np.flatnonzero(a[r] != b[r]):
        cells[(names.get(int(i), "?"), int(i), float(a[r, i]), float(b[r, i]))] += 1
for k, n in sorted(cells.items(), key=lambda x: -x[1])[:40]:
    print(n, k)
