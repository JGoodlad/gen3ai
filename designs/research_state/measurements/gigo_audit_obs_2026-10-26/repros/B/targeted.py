"""GIGO audit B: TARGETED legality battles — teams built to reach the lock / trap / disable surface the pool
rarely reaches (recharge, two-turn, Outrage/Thrash/Petal Dance, Rollout, Uproar, Bide, Choice Band, Encore,
Taunt, Disable, Torment, Imprison, Mean Look / Spider Web / Block, partial traps, Arena Trap / Shadow Tag /
Magnet Pull, Perish Song, Baton Pass, Pursuit, Focus Punch). Both sides EXTERNAL, answered uniformly over
their masks (self-play). Writes a games json in mask_vs_engine.py's shape for mask_vs_request.py.
"""
import json
import sys
from collections import Counter

import numpy as np

from utils.rust_env import ffi
from utils.rust_env import protocol as P

N_BATTLES = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 1
OUT = sys.argv[3] if len(sys.argv) > 3 else f"/tmp/gigo_audit/B/games_targeted_{SEED}_{N_BATTLES}.json"


def mon(species, item, ability, moves, nature="Hardy", evs="252,252,4,,,"):
    return f"{species}||{item}|{ability}|{','.join(moves)}|{nature}|{evs}||||| "[:-1]


A = "]".join([
    mon("Snorlax", "leftovers", "thickfat", ["hyperbeam", "solarbeam", "bodyslam", "rollout"]),
    mon("Salamence", "choiceband", "intimidate", ["outrage", "earthquake", "dragonclaw", "rockslide"]),
    mon("Skarmory", "leftovers", "keeneye", ["drillpeck", "spikes", "toxic", "whirlwind"]),
    mon("Vileplume", "leftovers", "chlorophyll", ["petaldance", "solarbeam", "sleeppowder", "sludgebomb"]),
    mon("Exploud", "leftovers", "soundproof", ["uproar", "thrash", "batonpass", "focuspunch"]),
    mon("Golem", "leftovers", "sturdy", ["rollout", "earthquake", "explosion", "pursuit"]),
])
B = "]".join([
    mon("Smeargle", "leftovers", "owntempo", ["encore", "taunt", "disable", "torment"]),
    mon("Umbreon", "leftovers", "synchronize", ["meanlook", "perishsong", "imprison", "wrap"]),
    mon("Dugtrio", "choiceband", "arenatrap", ["earthquake", "rockslide", "aerialace", "substitute"]),
    mon("Magneton", "leftovers", "magnetpull", ["thunderbolt", "thunderwave", "substitute", "toxic"]),
    mon("Wobbuffet", "leftovers", "shadowtag", ["counter", "mirrorcoat", "encore", "safeguard"]),
    mon("Gyarados", "leftovers", "intimidate", ["firespin", "whirlpool", "dragondance", "hyperbeam"]),
])
teams = [A, B]

lib = ffi.load(ffi.default_path("release"))
n_envs = 2
spec = P.spec_json(n=n_envs, threads=1, teams=teams, names=("fixone", "fixtwo"), turn_limit=250,
                   refusal_budget=10_000, bank_dir=None, opponents=[{"kind": "external"}])
core = ffi.FfiCore(spec, lib=lib)
c = core.cols
rng = np.random.default_rng(SEED)
flip = [0]


def stage(envs):
    for e in envs:
        # alternate which side holds which team, so both the lock team and the trap team play p1
        c["ep_team"][e] = [0, 1] if flip[0] % 2 == 0 else [1, 0]
        flip[0] += 1
        c["ep_seed"][e] = rng.integers(0, 65536, 4)
        c["ep_opp"][e] = 0


stage(range(n_envs)); core.reset(); stage(range(n_envs))
cur = {e: [] for e in range(n_envs)}
games = []
quarantined = 0
while len(games) < N_BATTLES:
    rec = {}
    for e in range(n_envs):
        for s in (0, 1):
            if c["need"][e, s]:
                m = c["mask"][e, s].astype(bool)
                a = int(rng.choice(np.flatnonzero(m)))
                c["action"][e, s] = a
                if s == 0:
                    rec[e] = dict(turn=int(c["turn"][e]), dec_n=int(c["dec_n"][e, 0]), mask=m.astype(int).tolist(),
                                  a=a, episode=int(c["episode"][e]))
    core.step()
    for e in range(n_envs):
        if e in rec:
            r = rec[e]
            r.update(reward=float(c["reward"][e]), done=int(c["done"][e]), term=int(c["terminated"][e]),
                     trunc=int(c["truncated"][e]), refused=int(c["refused"][e]))
            cur[e].append(r)
        if c["done"][e] == 1 and c["refused"][e] == 1 and not (c["terminated"][e] or c["truncated"][e]):
            quarantined += 1
            cur[e] = []
    for f in core.finished():
        e = f["env"]
        games.append(dict(fin=f, rows=cur[e]))
        cur[e] = []
    if (c["done"] == 1).any():
        stage([e for e in range(n_envs) if c["done"][e] == 1])
bank = core.bank()
core.close()
print("games", len(games), "quarantined", quarantined, "banked", len(bank))
for b in bank[:5]:
    print("  BANK", {k: (str(v)[:300]) for k, v in b.items() if k != "script"})
errs = Counter()
for g in games:
    t = ffi.trace(lib, g["fin"]["script"], commit="a", label="x")
    for ln in t["records"]["p1"].splitlines()[1:]:
        text = json.loads(ln)["text"]
        if text.startswith("|error|"):
            errs[text[:100]] += 1
print("p1 |error| lines:")
for k, v in errs.most_common():
    print("  ", v, k)
json.dump(dict(games=games), open(OUT, "w"))
print("wrote", OUT)
