"""Which observation columns the prior filter moved on the golden battles: the ability priors at 5e8d2956 (before
the filter) vs the working tree. Run from the repo root."""
import json
import subprocess
import sys

import numpy as np

sys.path.insert(0, 'src')
from agents.gen3_data import priors  # noqa: E402
from agents.training import golden_obs_capture as G  # noqa: E402

new = np.array(G.capture_vectors())
old_ab = json.loads(subprocess.run(["git", "show", "5e8d2956:data/pokemon/gen3_ability_priors.json"],
                                   capture_output=True, text=True, check=True).stdout)
priors.ability_raw = lambda: old_ab  # noqa: E731
import agents.observation.state_encoder as SE  # noqa: E402
for name in dir(SE):
    f = getattr(SE, name)
    if hasattr(f, "cache_clear"):
        f.cache_clear()
old = np.array(G.capture_vectors())
assert old.shape == new.shape, (old.shape, new.shape)
diff = old != new
rows = diff.any(1)
cols = np.where(diff.any(0))[0]
print("rows changed", int(rows.sum()), "of", len(rows), "; columns changed", len(cols))
layout = SE.get_observation_encoder(SE.load_mappings()).get_layout() if hasattr(SE, "get_observation_encoder") else {}


def name(c):
    best = None
    for k, v in layout.items():
        if isinstance(v, dict) and "offset" in v and "dim" in v and v["offset"] <= c < v["offset"] + v["dim"]:
            if best is None or v["dim"] < best[1]:
                best = (k, v["dim"], c - v["offset"])
    return best


for c in cols:
    print(c, name(c), sorted(set(np.round(old[diff[:, c], c], 4)))[:4], "->", sorted(set(np.round(new[diff[:, c], c], 4)))[:4])
