"""The per-species magnitude of F-X5-47's fix: the normalized species-usage marginal (the share
`build_species_usage_prior` emits, before its 1e-6 floor) under the PARENT's Raw count table vs this tree's
W table, plus the slot prior itself (`build_species_usage_prior`, floor + renormalization) both ways, and
Smogon's own latest-month weighted `usage` as an outside reference.

    PYTHONPATH=<tree>/src python shift.py --parent out/parent_species_usage.json > out/usage_shift.txt
"""
import argparse
import json
import math
import statistics
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--parent", required=True)
a = ap.parse_args()

from agents.gen3_data import priors as P  # noqa: E402
from agents.model import dex_ids as DX  # noqa: E402
from agents import gen3_data  # noqa: E402

old = json.loads(Path(a.parent).read_text())
new = P.species_usage()
assert old.keys() == new.keys(), "species sets differ"
to, tn = sum(old.values()), sum(new.values())
so = {s: v / to for s, v in old.items()}
sn = {s: v / tn for s, v in new.items()}
lr = {s: math.log(sn[s] / so[s]) for s in sn}
latest = {P._species_id(k): float(r.get("usage", 0.0)) for k, r in P.smogon_stats_raw()["data"].items()}

print(f"species: {len(sn)}; Raw count total {to:.6g}, W total {tn:.6g} (W / Raw count overall {tn / to:.4f})")
v = sorted(lr.values())
print(f"log(new share / old share) over all {len(v)}: min {v[0]:+.3f} median {statistics.median(v):+.3f} "
      f"max {v[-1]:+.3f}; ratio min x{math.exp(v[0]):.3f} max x{math.exp(v[-1]):.3f}")
top = sorted(sn, key=lambda s: -sn[s])[:25]
vt = sorted(lr[s] for s in top)
print(f"over the top-25 by new share: log ratio {vt[0]:+.3f} .. {vt[-1]:+.3f}")
print("\ntop-25 by new share: species  old_share -> new_share  (ratio)   6*new_share vs Smogon latest-month usage")
for s in top:
    print(f"  {s:12s} {so[s]:.5f} -> {sn[s]:.5f}  (x{sn[s] / so[s]:.3f})   {6 * sn[s]:.4f} vs {latest[s]:.4f}")
print("\nbiggest movers DOWN (all species):")
for s in sorted(lr, key=lambda s: lr[s])[:10]:
    print(f"  {s:12s} {so[s]:.6f} -> {sn[s]:.6f}  (x{sn[s] / so[s]:.3f})")
print("biggest movers UP (all species):")
for s in sorted(lr, key=lambda s: -lr[s])[:10]:
    print(f"  {s:12s} {so[s]:.6f} -> {sn[s]:.6f}  (x{sn[s] / so[s]:.3f})")

# the slot prior the op / T0 / roles read (floor + renormalization), both ways
n = max(sd.num for sd in (gen3_data.species.get(i) for i in gen3_data.species.base_form_ids())) + 1
p_new = DX.build_species_usage_prior(n)
real = P.species_usage
P.species_usage = lambda: old
try:
    p_old = DX.build_species_usage_prior(n)
finally:
    P.species_usage = real
d = (p_new.double() - p_old.double()).abs()
print(f"\nbuild_species_usage_prior({n}): max |d| {float(d.max()):.4g} over {int((d > 0).sum())} of {n} nums; "
      f"L1 {float(d.sum()):.4f}; Tyranitar {float(p_old[gen3_data.species.get('tyranitar').num]):.5f} -> "
      f"{float(p_new[gen3_data.species.get('tyranitar').num]):.5f}")
# agreement with Smogon's own weighted usage (latest month) — the outside reference
for name, share in (("old", so), ("new", sn)):
    rel = [abs(6 * share[s] - latest[s]) / latest[s] for s in top if latest[s] > 0]
    print(f"top-25 |6*{name}_share - latest usage| / latest usage: median {statistics.median(rel):.3f}, "
          f"max {max(rel):.3f}")
