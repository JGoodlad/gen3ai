"""Design input only (no policy forward): how many side-0 rows carry an ELIGIBLE target mon per group."""
import numpy as np
import n2common as N
from agents.observation import constants as K
from static_probe import SLOT, signature, turn1_slots
from utils.team_sources import packed_teams

lib = N.lib()
pool = turn1_slots(packed_teams("pool"), lib).reshape(-1, SLOT)
pool_set = {signature(s) for s in pool}
for panel in ("home72", "F", "H", "P72"):
    z = N.load_rows(panel)
    rows = np.flatnonzero(z["side"] == 0)
    n_rows = n_mons = 0
    games = set()
    for r in rows:
        o = z["obs"][r]
        slots = o[:6 * SLOT].reshape(6, SLOT)
        act = int(np.argmax(slots[:, K.POKEMON_ACTIVE_OFFSET]))
        seen = o[K.OFFSET_OBS_FACTS:K.OFFSET_OBS_FACTS + 42].reshape(6, 7)
        k = 0
        for b in range(6):
            if b == act or slots[b, K.POKEMON_HP_OFFSET] < 1.0 or seen[b].any():
                continue
            if slots[b, K.POKEMON_CONDITION_OFFSET + 1:K.POKEMON_CONDITION_OFFSET + K.CONDITION_DIM].any():
                continue
            inpool = signature(slots[b]) in pool_set
            if inpool == (panel == "home72"):
                k += 1
        if k:
            n_rows += 1
            n_mons += k
            games.add(int(z["game"][r]))
    print(panel, "side0 rows", len(rows), "eligible rows", n_rows, "target mons", n_mons, "games", len(games))
