"""Every categorical id column the model decodes with `.long()`: over real fixture rows, is the float an EXACT
integer in [0, vocab)? Prints per column: n nonzero, min, max, non-integer count, out-of-range count, vocab."""
import sys, json
import numpy as np
from agents.observation.state_encoder import load_mappings, Gen3ObservationEncoder
from agents.observation.schema import build_schema
from agents.observation.constants import (EventCol, OFFSET_EVENT_WINDOW, OFFSET_OBS_FACTS, OFFSET_OUR_TEAM,
    POKEMON_FULL_DIM, POKEMON_LAST_ACTION_OFFSET, N_EVENT_TYPES, N_ITEM_TRANSITIONS, N_ENTRY, N_DENIAL, N_EVENT_STAT,
    N_EVENT_STATUS, FACTS_CHOICE_OFFSET)
from agents.observation.gen3_effects import CANT_DIM_LIVE
from agents.battle.faint_causes import FAINT_CAUSE_DIM_LIVE
from utils.rust_env.fixture_battles import play_rows

L = Gen3ObservationEncoder(load_mappings()).get_layout()
sl = build_schema(L).slices()
pk = L['pokemon']; mv = pk['moves']['layout']
V = dict(species=L['max_species'], move=L['max_moves'], item=L['max_items'], ability=L['max_abilities'], type=L['max_types'])
TYPE_SCATTER = 19   # move_resolution's type_rev.scatter width (the T2 assert's)
cols = {}   # name -> (flat column indices, vocab)
def mon_cols(off):
    return [OFFSET_OUR_TEAM + s * POKEMON_FULL_DIM + off for s in range(12)]
cols['species'] = (mon_cols(pk['species']['offset']), V['species'])
cols['item'] = (mon_cols(pk['items']['offset']), V['item'])
cols['ability1'] = (mon_cols(pk['abilities']['offset']), V['ability'])
cols['ability2'] = (mon_cols(pk['abilities']['offset'] + 1), V['ability'])
cols['type1'] = (mon_cols(pk['types']['offset']), TYPE_SCATTER)
cols['type2'] = (mon_cols(pk['types']['offset'] + 1), TYPE_SCATTER)
cols['move_id'] = (sum((mon_cols(pk['moves']['offset'] + s['offset']) for s in mv['slots']), []), V['move'])
cols['move_type'] = (sum((mon_cols(pk['moves']['offset'] + s['offset'] + mv['slot_layout']['type']['offset']) for s in mv['slots']), []), TYPE_SCATTER)
cols['last_move'] = (mon_cols(POKEMON_LAST_ACTION_OFFSET), V['move'])
arm = sl['reactive.active_req_moves']; per = (arm.stop - arm.start) // 3
cols['req_move_id'] = (list(range(arm.start, arm.start + per)), V['move'])
cols['req_move_type'] = (list(range(arm.start + per, arm.start + 2 * per)), TYPE_SCATTER)
N = L['event_window_n']; D = len(EventCol)
def ev(c): return [OFFSET_EVENT_WINDOW + k * D + int(c) for k in range(N)]
for c, voc in [(EventCol.TYPE, N_EVENT_TYPES), (EventCol.ACTOR_SPECIES, V['species']), (EventCol.TARGET_SPECIES, V['species']),
               (EventCol.REL_SPECIES, V['species']), (EventCol.MOVE, V['move']), (EventCol.CALLER, V['move']),
               (EventCol.STATUS, N_EVENT_STATUS), (EventCol.CANT, CANT_DIM_LIVE + 1), (EventCol.FAINT_CAUSE, FAINT_CAUSE_DIM_LIVE + 1),
               (EventCol.ITEM_TRANSITION, N_ITEM_TRANSITIONS), (EventCol.ENTRY, N_ENTRY), (EventCol.DENIAL, N_DENIAL), (EventCol.STAT, N_EVENT_STAT)]:
    cols['ev.' + c.name] = (ev(c), voc)
cols['facts.choice_first_move'] = ([OFFSET_OBS_FACTS + FACTS_CHOICE_OFFSET + 2], V['move'])
w = sl['global_env.weather']; hz = sl['global_env.hazards']
gl = L['global_layout']
print("global layout:", json.dumps(gl, default=str)[:600])

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 7
teams = None
if len(sys.argv) > 2:
    teams = json.load(open(sys.argv[2]))
games = play_rows(int(sys.argv[3]) if len(sys.argv) > 3 else 8, seed=seed, n_teams=None, teams=teams)
X = np.concatenate([g[0] for g in games]).astype(np.float32)
print(f"rows {X.shape[0]} from {len(games)} games, seed {seed}, teams {'pool' if teams is None else sys.argv[2]}")
bad = 0
for name, (idx, voc) in cols.items():
    v = X[:, idx].reshape(-1)
    nonint = int((v != np.trunc(v)).sum())
    oor = int(((v < 0) | (v >= voc)).sum())
    bad += nonint + oor
    print(f"{name:28s} vocab {voc:4d}  nnz {int((v != 0).sum()):7d}  min {v.min():8.3f}  max {v.max():8.3f}  "
          f"distinct {len(np.unique(v)):4d}  non-integer {nonint}  out-of-[0,vocab) {oor}")
for name, s, scale, top in [("weather_turns", w, None, None), ("spikes", hz, 3.0, 3)]:
    print(name, "slice", s, "distinct values", np.unique(X[:, s])[:20])
print("TOTAL VIOLATIONS", bad)
