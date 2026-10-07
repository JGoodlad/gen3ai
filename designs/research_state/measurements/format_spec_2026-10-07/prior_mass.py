"""Measure the prior mass the format-spec filter moves (on the CURRENT committed priors)."""
import json
import sys

sys.path.insert(0, 'src')
from agents import gen3_data  # noqa: E402
from agents.gen3_data import format_spec as fs  # noqa: E402

D = 'data/pokemon/'
spec = fs.active()
it = json.load(open(D + 'gen3_item_priors.json'))
ab = json.load(open(D + 'gen3_ability_priors.json'))
mv = json.load(open(D + 'gen3_move_priors.json'))
tm = json.load(open(D + 'gen3_teammate_priors.json'))
st = json.load(open(D + 'gen3_smogon_stats.json'))['data']
W = {fs.to_id(k): sum(v.get('Abilities', {}).values()) for k, v in st.items()}
TW = sum(W.values())

print("== ITEMS (Smogon item prior, sum-1 per species)")
for item in sorted(spec.banned_items):
    rows = {s: r[item] for s, r in it.items() if r.get(item, 0) > 0}
    wmass = sum(W.get(s, 0) * p for s, p in rows.items()) / TW
    top = sorted(rows.items(), key=lambda kv: -kv[1])[:8]
    print(f"  {item}: {len(rows)} species carry mass; usage-weighted share of ALL sets = {wmass:.4%}; "
          f"max {max(rows.values(), default=0):.4f}; top {[(s, round(p, 3)) for s, p in top]}")
# top species by usage: their QC mass
top_usage = sorted(W.items(), key=lambda kv: -kv[1])[:15]
print("  top-15-usage species' quickclaw P:", [(s, round(it.get(s, {}).get('quickclaw', 0.0), 4)) for s, _ in top_usage])

print("== ABILITIES (dex-anchored Smogon ability prior)")
for s, r in sorted(ab.items()):
    b = {k: v for k, v in r.items() if k in spec.banned_abilities}
    if b:
        rest = 1 - sum(b.values())
        print(f"  {s}: banned {b} -> {'ROW DROPPED (no legal ability)' if rest <= 0 else {k: round(v / rest, 4) for k, v in r.items() if k not in b}}"
              f"  (W={W.get(s, 0):.0f})")

print("== MOVES (P(move in set))")
nb = 0
for s, r in mv.items():
    for m in r:
        if spec.is_banned('move', m, species=s):
            nb += 1
            print('  ', s, m, r[m])
print(f"  banned (species, move) entries with mass: {nb}")

print("== TEAMMATES / SPECIES USAGE")
print("  banned species with a chaos record:", [s for s in W if s in spec.banned_species and W[s] > 0])
print("  teammate entries on a banned species:", sum(1 for r in tm.values() for k in r if k in spec.banned_species))

print("== MODEL-TABLE FLOORS (legal-unobserved floors that sit on banned entities)")
cells = 0
for sid in gen3_data.species.base_form_ids():
    legal = gen3_data.learnset.get_legal_moves(sid)
    if legal is None:
        continue
    for m in legal:
        if spec.is_banned('move', m, species=sid) and m not in mv.get(sid, {}):
            cells += 1
print(f"  move-prior cells (species, banned move) legal per learnset, at the 0.02 floor: {cells}")
print(f"  item-prior columns at the 1e-5 floor per species row: {len(spec.banned_items)} (every species row)")
print(f"  species-usage floor 1e-6 / co-occurrence floor 1e-4 on banned species: "
      f"{len([s for s in spec.banned_species if gen3_data.species.get(s) is not None and gen3_data.species.get(s).base_species is None])} base species")
