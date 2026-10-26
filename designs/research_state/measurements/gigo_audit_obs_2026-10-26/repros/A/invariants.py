"""Row invariants over seeded fixture battles (investigator A): leak-shaped and range-shaped checks on p1's rows.

1. opponent HP cells are k/100 (the public percent), never an exact fraction;
2. the opponent spread block (IV/EV/known/nature) is all-zero (oracle off);
3. at most one ACTIVE flag per team; the trapped bits only on OUR active slot; last_action only on active slots;
4. categorical ids within the model's embedding sizes (species / move / item / ability / type);
5. the opponent's revealed-move count per slot <= 4, and a move 'known' bit implies an id;
6. stale-after-switch: the slot that carries a last_action / protect<1 is the active slot.
"""
from __future__ import annotations

import sys
from collections import Counter

import numpy as np

import agents.observation.constants as K
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from utils.rust_env.fixture_battles import play_rows

LAY = Gen3ObservationEncoder(load_mappings()).get_layout()
FD = K.POKEMON_FULL_DIM


def slot(row, team, i):
    base = K.OFFSET_OUR_TEAM if team == "our" else K.OFFSET_OPP_TEAM
    return row[base + i * FD: base + (i + 1) * FD]


def main(n_battles: int, seed: int) -> int:
    from utils.rust_env import ffi
    games = play_rows(n_battles, seed=seed, n_teams=None, lib=ffi.load(ffi.default_path("release")))
    bad = Counter()
    ex = {}
    maxid = Counter()
    n = 0
    for g, (obs, _mask) in enumerate(games):
        for d, row in enumerate(obs):
            n += 1
            for team in ("our", "opp"):
                act = 0
                for i in range(6):
                    s = slot(row, team, i)
                    if s[K.POKEMON_SPECIES_KNOWN_OFFSET] == 0:
                        if np.any(s != 0):
                            bad[f"{team} empty slot non-zero"] += 1
                        continue
                    act += int(s[K.POKEMON_ACTIVE_OFFSET] == 1)
                    sp = s[K.POKEMON_SPECIES_OFFSET]
                    maxid["species"] = max(maxid["species"], sp)
                    maxid["item"] = max(maxid["item"], s[K.POKEMON_ITEMS_OFFSET])
                    maxid["ability1"] = max(maxid["ability1"], s[K.POKEMON_ABILITIES_OFFSET])
                    maxid["ability2"] = max(maxid["ability2"], s[K.POKEMON_ABILITIES_OFFSET + 1])
                    maxid["type"] = max(maxid["type"], *s[K.POKEMON_TYPES_OFFSET:K.POKEMON_TYPES_OFFSET + 2])
                    for m in range(4):
                        mb = K.POKEMON_MOVES_OFFSET + m * K.MOVE_SLOT_DIM
                        maxid["move"] = max(maxid["move"], s[mb])
                        maxid["movetype"] = max(maxid["movetype"], s[mb + 4])
                        if s[mb + 6] == 1 and s[mb] == 0:
                            bad[f"{team} move known w/o id"] += 1
                    maxid["last_move"] = max(maxid["last_move"], s[K.POKEMON_LAST_ACTION_OFFSET])
                    if team == "opp":
                        hp = float(s[K.POKEMON_HP_OFFSET])
                        if abs(hp * 100 - round(hp * 100)) > 1e-4:
                            bad["opp hp not k/100"] += 1
                            ex.setdefault("opp hp not k/100", (g, d, i, hp))
                        sp_blk = s[K.POKEMON_SPREAD_OFFSET:K.POKEMON_SPREAD_OFFSET + K.POKEMON_SPREAD_DIM]
                        if np.any(sp_blk != 0):
                            bad["opp spread non-zero"] += 1
                            ex.setdefault("opp spread non-zero", (g, d, i, sp_blk.tolist()))
                        if s[K.POKEMON_TRAPPED_OFFSET] or s[K.POKEMON_MAYBE_TRAPPED_OFFSET]:
                            bad["opp trapped bit"] += 1
                    if s[K.POKEMON_ACTIVE_OFFSET] == 0:
                        la = s[K.POKEMON_LAST_ACTION_OFFSET:K.POKEMON_LAST_ACTION_OFFSET + K.POKEMON_LAST_ACTION_DIM]
                        if np.any(la != 0):
                            bad[f"{team} last_action on bench"] += 1
                        if s[K.POKEMON_TRAPPED_OFFSET] or s[K.POKEMON_MAYBE_TRAPPED_OFFSET]:
                            bad[f"{team} trapped on bench"] += 1
                        if s[K.POKEMON_PROTECT_OFFSET] != 1.0:
                            bad[f"{team} protect<1 on bench"] += 1
                            ex.setdefault(f"{team} protect<1 on bench", (g, d, i, float(s[K.POKEMON_PROTECT_OFFSET])))
                if act > 1:
                    bad[f"{team} >1 active"] += 1
                    ex.setdefault(f"{team} >1 active", (g, d))
            ew = row[K.OFFSET_EVENT_WINDOW:K.OFFSET_EVENT_WINDOW + K.EVENT_WINDOW_DIM].reshape(K.EVENT_WINDOW_N, K.EVENT_TOKEN_DIM)
            E = K.EVENT_COL
            for col, name in ((E.ACTOR_SPECIES, "ev_actor"), (E.TARGET_SPECIES, "ev_target"), (E.MOVE, "ev_move"),
                              (E.CALLER, "ev_caller"), (E.REL_SPECIES, "ev_rel"), (E.TYPE, "ev_type"), (E.STATUS, "ev_status"), (E.CANT, "ev_cant"), (E.FAINT_CAUSE, "ev_faint"), (E.ITEM_TRANSITION, "ev_itemtr"), (E.ENTRY, "ev_entry"), (E.DENIAL, "ev_denial"), (E.STAT, "ev_stat")):
                maxid[name] = max(maxid[name], ew[:, col].max())
            rq = row[K.OFFSET_REACTIVE:K.OFFSET_REACTIVE + K.REACTIVE_DIM]
    print(f"battles {len(games)} rows {n}")
    print("max ids:", {k: float(v) for k, v in maxid.items()})
    print("vocab:", {k: LAY[k] for k in ("max_species", "max_moves", "max_items", "max_abilities", "max_types")})
    print("violations:", dict(bad))
    for k, v in ex.items():
        print("  example", k, v)
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]), int(sys.argv[2])))
