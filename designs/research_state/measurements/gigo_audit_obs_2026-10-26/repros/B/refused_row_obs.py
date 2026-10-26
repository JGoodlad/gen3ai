"""GIGO audit B: the obs of a REFUSED hidden-trap switch row vs the re-request row that follows it (same turn):
which layout fields differ (does the model SEE that the switch was refused / that it is trapped?)."""
import json
from collections import Counter

import numpy as np

from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from utils.rust_env import ffi
from utils.rust_env import protocol as P
from utils.team_sources import packed_teams

L = Gen3ObservationEncoder(load_mappings()).get_layout()
R = [("our_team", 0, 732), ("opp_team", 732, 1464), ("context", 1464, 1584), ("global", 1584, 1604),
     ("reactive", 1604, 1621), ("pair_history", 1621, 1801), ("event_window", 1801, 2761), ("obs_facts", 2761, 2845)]
assert L["event_window_offset"] == 1801 and L["obs_facts_offset"] == 2761 and L["total_dim"] == 2845


def name(i):
    for n, a, b in R:
        if a <= i < b:
            if n == "event_window":
                return f"event_window[token {(i - a) // L['event_token_dim']}]"
            return f"{n}[{i - a}]"
    return f"?{i}"


lib = ffi.load(ffi.default_path("release"))
teams = list(packed_teams("pool"))
dug = [t for t in teams if "arenatrap" in t or "magnetpull" in t or "shadowtag" in t]
spec = P.spec_json(n=1, threads=1, teams=dug + teams[:40], names=("fixone", "fixtwo"), turn_limit=250, refusal_budget=4,
                   bank_dir=None, opponents=[{"kind": "bot", "bot": "heuristic", "seed": 3, "streams": "episode"}])
core = ffi.FfiCore(spec, lib=lib)
c = core.cols
rng = np.random.default_rng(5)


def stage():
    c["ep_team"][0] = [int(rng.integers(len(dug), len(dug) + 40)), int(rng.integers(0, len(dug)))]  # p2 holds the trapper
    c["ep_seed"][0] = rng.integers(0, 65536, 4)
    c["ep_opp"][0] = 0


stage(); core.reset(); stage()
prev = None
diffs = Counter()
pairs = 0
games = 0
while games < 30 and pairs < 25:
    if c["need"][0, 0]:
        m = c["mask"][0, 0].astype(bool)
        obs = c["obs"][0, 0].copy()
        turn = int(c["turn"][0])
        if prev is not None and prev[0] == turn and prev[2] < 6 and not m[:6].any() and m[6:].any():
            pairs += 1
            d = np.flatnonzero(obs != prev[1])
            for k in sorted({name(int(i)).split('[')[0] for i in d}):
                diffs[k] += 1
            if pairs <= 2:
                print("pair", pairs, "turn", turn, "differing cols:", [(name(int(i)), float(prev[1][i]), float(obs[i])) for i in d][:40])
        legal = np.flatnonzero(m)
        sw = legal[legal < 6]
        a = int(rng.choice(sw)) if sw.size and rng.random() < 0.7 else int(rng.choice(legal))
        c["action"][0, 0] = a
        prev = (turn, obs, a)
    core.step()
    if c["done"][0] == 1:
        games += 1
        prev = None
        stage()
core.close()
print("refused pairs", pairs, "games", games)
print("columns differing (count over pairs):")
for k, v in diffs.most_common(40):
    print("  ", v, k)
