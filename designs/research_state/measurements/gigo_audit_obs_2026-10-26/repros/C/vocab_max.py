"""Max dex num per vocabulary vs the layout's table sizes."""
from agents import gen3_data
from agents.observation.state_encoder import load_mappings, Gen3ObservationEncoder
m = load_mappings(); L = Gen3ObservationEncoder(m).get_layout()
for cat, key in [("species","max_species"),("moves","max_moves"),("items","max_items"),("abilities","max_abilities")]:
    d = m[cat]; nums = [(v["num"], k) for k, v in d.items() if isinstance(v, dict) and "num" in v]
    nums.sort()
    neg = [x for x in nums if x[0] < 0]; zero=[x for x in nums if x[0]==0]
    print(cat, "n=", len(nums), "min", nums[0], "max", nums[-1], "table", L[key], "neg", neg[:5], "zero", zero[:8], "n_zero", len(zero))
    over = [x for x in nums if x[0] >= L[key]]
    print("  >= table:", over[:10])
    # duplicates among non-forme
    from collections import Counter
    c = Counter(v["num"] for k,v in d.items() if isinstance(v,dict) and "num" in v and not v.get("baseSpecies"))
    print("  dup nums (non-forme):", [(n, [k for k,v in d.items() if isinstance(v,dict) and v.get('num')==n and not v.get('baseSpecies')]) for n,k in c.items() if k>1][:10])
print("max_types", L["max_types"])
