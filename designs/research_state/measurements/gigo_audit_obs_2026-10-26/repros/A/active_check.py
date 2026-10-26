"""ACTIVE-flag alignment (investigator A): the slot carrying the ACTIVE flag must hold the species of the side's
most recent SWITCH_IN event in the event window (when one is in the window and no later FAINT of that mon)."""
import sys
import numpy as np
import agents.observation.constants as K
from utils.rust_env import ffi
from utils.rust_env.fixture_battles import play_rows

E = K.EVENT_COL
games = play_rows(int(sys.argv[1]), seed=int(sys.argv[2]), n_teams=None, lib=ffi.load(ffi.default_path("release")))
checked = bad = 0
ex = []
for g, (obs, _m) in enumerate(games):
    for d, row in enumerate(obs):
        ew = row[K.OFFSET_EVENT_WINDOW:K.OFFSET_EVENT_WINDOW + K.EVENT_WINDOW_DIM].reshape(K.EVENT_WINDOW_N, K.EVENT_TOKEN_DIM)
        for side_val, base in ((1.0, K.OFFSET_OUR_TEAM), (-1.0, K.OFFSET_OPP_TEAM)):
            last = None
            for r in ew:
                if r[E.VALID] != 1 or r[E.ACTOR_SIDE] != side_val:
                    continue
                if int(r[E.TYPE]) == 2:  # SWITCH_IN
                    last = r[E.ACTOR_SPECIES]
            if last is None:
                continue
            act = [base + i * K.POKEMON_FULL_DIM for i in range(6) if row[base + i * K.POKEMON_FULL_DIM + K.POKEMON_ACTIVE_OFFSET] == 1]
            checked += 1
            if len(act) != 1 or row[act[0] + K.POKEMON_SPECIES_OFFSET] != last:
                bad += 1
                if len(ex) < 5:
                    ex.append((g, d, side_val, last, [row[a] for a in act]))
print(f"rows checked {checked}, misaligned {bad}", ex)
