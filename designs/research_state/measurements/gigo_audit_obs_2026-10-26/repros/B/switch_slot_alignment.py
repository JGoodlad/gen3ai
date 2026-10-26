"""GIGO audit B: does switch action i name the mon in OBS our_team slot i? At every p1 row (fixture battles,
uniform p1): mask[i] = 1 must imply obs slot i is NOT our active, is alive (hp > 0); our active's slot must be
masked off; and every alive non-active slot must be switchable unless the active is trapped (obs trapped flag)
or the row is not a switch-capable request. Offsets from agents.observation.constants (no literals)."""
import sys

import numpy as np

from agents.observation import constants as K
from utils.rust_env.fixture_battles import play_rows

n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 3
games = play_rows(n, seed=seed, n_teams=None)
mon = K.POKEMON_FULL_DIM
team0 = K.OFFSET_OUR_TEAM
hp_off = K.POKEMON_HP_OFFSET if hasattr(K, "POKEMON_HP_OFFSET") else None
print("POKEMON_FULL_DIM", mon, "OFFSET_OUR_TEAM", team0, "hp offset", hp_off, "active", K.POKEMON_ACTIVE_OFFSET,
      "trapped", K.POKEMON_TRAPPED_OFFSET)
bad = {"switch to active": 0, "switch to hp0": 0, "alive bench not switchable (not trapped)": 0}
rows = 0
ex = []
for obs, mask in games:
    for o, m in zip(obs, mask):
        rows += 1
        slots = o[team0:team0 + 6 * mon].reshape(6, mon)
        active = slots[:, K.POKEMON_ACTIVE_OFFSET] > 0.5
        hp = slots[:, hp_off] if hp_off is not None else np.ones(6)
        trapped = bool(slots[active, K.POKEMON_TRAPPED_OFFSET].max()) if active.any() else False
        for i in range(6):
            if m[i] and active[i]:
                bad["switch to active"] += 1; ex.append(("active", i, m.astype(int).tolist(), active.tolist()))
            if m[i] and hp[i] <= 0:
                bad["switch to hp0"] += 1; ex.append(("hp0", i, m.astype(int).tolist(), hp.tolist()))
        moves_offered = m[6:].any()
        if moves_offered and not trapped:
            for i in range(6):
                if not active[i] and hp[i] > 0 and not m[i]:
                    bad["alive bench not switchable (not trapped)"] += 1
                    ex.append(("bench", i, m.astype(int).tolist(), hp.tolist(), active.tolist()))
print("rows", rows, bad)
for e in ex[:5]:
    print("  ", e)
