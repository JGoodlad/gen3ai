"""Silent-0 defaults in the Rust encoder: (1) every Smogon ability-prior name resolves to an ability num (data.rs
`abilities.get(a).unwrap_or(0.0)`); (2) on fixture rows, MOVE-type event rows whose move id is 0 or actor species 0."""
import json, sys
import numpy as np
from agents import gen3_data
from agents.observation.constants import EventCol, OFFSET_EVENT_WINDOW
import agents.observation.constants as C
from agents.observation.state_encoder import load_mappings, Gen3ObservationEncoder
L = Gen3ObservationEncoder(load_mappings()).get_layout()
ab = gen3_data.abilities.raw(); pri = gen3_data.priors.ability_raw()
miss = sorted({(sp, a) for sp, d in pri.items() for a in d if a not in ab})
print("ability-prior names with no ability num:", len(miss), miss[:10])
sp = gen3_data.species.raw(); print("prior species not in species table:", [s for s in pri if s not in sp][:10])
from utils.rust_env.fixture_battles import play_rows
X = np.concatenate([g[0] for g in play_rows(8, seed=7, n_teams=None)])
N = L['event_window_n']; D = len(EventCol)
E = X[:, OFFSET_EVENT_WINDOW:OFFSET_EVENT_WINDOW + N * D].reshape(-1, D)
E = E[E[:, EventCol.VALID] > 0.5]
mvT = C.EVENT_T_MOVE
mrows = E[E[:, EventCol.TYPE] == mvT]
print("event rows", len(E), "MOVE rows", len(mrows), "MOVE rows with move id 0:", int((mrows[:, EventCol.MOVE] == 0).sum()),
      "with actor species 0:", int((mrows[:, EventCol.ACTOR_SPECIES] == 0).sum()))
print("any valid row with actor species 0, by type:", {int(t): int(((E[:, EventCol.TYPE] == t) & (E[:, EventCol.ACTOR_SPECIES] == 0)).sum()) for t in np.unique(E[:, EventCol.TYPE])})
