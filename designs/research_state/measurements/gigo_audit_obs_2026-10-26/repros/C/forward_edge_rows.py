"""Run the PRODUCTION extractor (designs/production_config.json) eagerly on CPU over the crafted edge-id rows (Deoxys
386, typed HP 355-370, Psycho Boost 354, Curse '???' type 18, Castform under sun, Ditto Transform) and the pool rows:
CPU indexing raises on an index >= its table, so a clean pass = every id is inside every table it reaches. Then the
crafted WORST CASE: every species / move / item / ability / type cell set to its table's last legal row."""
import json, sys
import numpy as np, torch
from utils.rust_env.fixture_battles import play_rows
from agents.model.delivery_graph import build_extractor
from agents.model.extra_obs_keys import zero_extra_obs
from agents.observation.constants import OFFSET_OUR_TEAM, POKEMON_FULL_DIM
fe, cfg, L = build_extractor()
teams = json.load(open(sys.argv[1]))
X = np.concatenate([g[0] for g in play_rows(6, seed=21, teams=teams)] + [g[0] for g in play_rows(4, seed=22, n_teams=None)])
def fwd(x):
    with torch.no_grad():
        out = fe({"observation": torch.from_numpy(x.astype(np.float32)), **zero_extra_obs(fe, x.shape[0])})
    t = out if isinstance(out, torch.Tensor) else out[0]
    return t
t = fwd(X); print("real edge rows", X.shape, "forward ok, finite:", bool(torch.isfinite(t).all()))
pk = L['pokemon']; mv = pk['moves']['layout']
W = X[:64].copy()
for s in range(12):
    b = OFFSET_OUR_TEAM + s * POKEMON_FULL_DIM
    if W[0, b] == 0: continue
    W[:, b + pk['species']['offset']] = L['max_species'] - 1
    W[:, b + pk['items']['offset']] = L['max_items'] - 1
    W[:, b + pk['abilities']['offset']] = L['max_abilities'] - 1
    W[:, b + pk['types']['offset']] = 18; W[:, b + pk['types']['offset'] + 1] = 18
    for sl in mv['slots']:
        W[:, b + pk['moves']['offset'] + sl['offset']] = L['max_moves'] - 1
        W[:, b + pk['moves']['offset'] + sl['offset'] + mv['slot_layout']['type']['offset']] = 18
try:
    t = fwd(W); print("worst-case last-row ids: forward ok, finite:", bool(torch.isfinite(t).all()))
except Exception as e:
    print("worst-case last-row ids RAISED:", type(e).__name__, str(e)[:300])
for name, val in (("type 19 (== scatter width)", 19), ("type -1", -1)):
    Z = X[:8].copy(); b = OFFSET_OUR_TEAM + 6 * POKEMON_FULL_DIM
    Z[:, b + pk['types']['offset'] + 1] = val
    try:
        fwd(Z); print(name, "-> forward ok (SILENT)")
    except Exception as e:
        print(name, "-> RAISED", type(e).__name__, str(e)[:160])
