"""GIGO audit B: mask vs the engine's own acceptance, plus terminal/reward columns.

Plays seeded battles on the Rust env core (uniform-over-mask p1, heuristic bot p2 — the fixture setup),
logs every p1 decision (turn, dec_n, mask, chosen action, reward/done/term/trunc after the step), then
replays each finished script through ffi.trace and scans p1's record for |error| lines (a mask-allowed
choice the engine refused) and the |request| JSON at each decision.
"""
import json
import sys
from collections import Counter

import numpy as np

from utils.rust_env import ffi
from utils.rust_env import protocol as P
from utils.team_sources import packed_teams

N_BATTLES = int(sys.argv[1]) if len(sys.argv) > 1 else 40
SEED = int(sys.argv[2]) if len(sys.argv) > 2 else 7
POLICY = sys.argv[3] if len(sys.argv) > 3 else "uniform"   # uniform | switchy
N_TEAMS = int(sys.argv[4]) if len(sys.argv) > 4 else 0
TURN_LIMIT = int(sys.argv[5]) if len(sys.argv) > 5 else 250

lib = ffi.load(ffi.default_path("release"))
teams = list(packed_teams("pool"))
if N_TEAMS:
    teams = teams[:N_TEAMS]
n_envs = 2
spec = P.spec_json(n=n_envs, threads=1, teams=teams, names=("fixone", "fixtwo"), turn_limit=TURN_LIMIT,
                   refusal_budget=4, bank_dir=None,
                   opponents=[{"kind": "bot", "bot": "heuristic", "seed": 3, "streams": "episode"}])
core = ffi.FfiCore(spec, lib=lib)
c = core.cols
rng = np.random.default_rng(SEED)


def stage(envs):
    for e in envs:
        c["ep_team"][e] = rng.choice(len(teams), 2, replace=False)
        c["ep_seed"][e] = rng.integers(0, 65536, 4)
        c["ep_opp"][e] = 0


stage(range(n_envs)); core.reset(); stage(range(n_envs))
cur = {e: [] for e in range(n_envs)}
games = []
step_log = []   # (env, episode_before, need0, turn, reward, done, term, trunc, refused, episode_after)
while len(games) < N_BATTLES:
    pend = [e for e in range(n_envs) if c["need"][e, 0]]
    ep_before = c["episode"].copy()
    rec = {}
    for e in pend:
        m = c["mask"][e, 0].astype(bool)
        legal = np.flatnonzero(m)
        if POLICY == "switchy" and legal[legal < 6].size:
            a = int(rng.choice(legal[legal < 6])) if rng.random() < 0.6 else int(rng.choice(legal))
        else:
            a = int(rng.choice(legal))
        c["action"][e, 0] = a
        rec[e] = dict(turn=int(c["turn"][e]), dec_n=int(c["dec_n"][e, 0]), mask=m.astype(int).tolist(), a=a,
                      episode=int(c["episode"][e]))
    core.step()
    for e in range(n_envs):
        if e in rec:
            r = rec[e]
            r.update(reward=float(c["reward"][e]), done=int(c["done"][e]), term=int(c["terminated"][e]),
                     trunc=int(c["truncated"][e]), refused=int(c["refused"][e]))
            cur[e].append(r)
        step_log.append((e, int(ep_before[e]), int(e in rec), int(c["reward"][e] != 0), int(c["done"][e]),
                         int(c["refused"][e])))
    for f in core.finished():
        e = f["env"]
        games.append(dict(fin=f, rows=cur[e]))
        cur[e] = []
    if (c["done"] == 1).any():
        stage([e for e in range(n_envs) if c["done"][e] == 1])
core.close()

errs = Counter()
err_examples = {}
n_dec = 0
outcome = Counter()
anomalies = []
for gi, g in enumerate(games):
    f, rows = g["fin"], g["rows"]
    n_dec += len(rows)
    last = rows[-1] if rows else None
    outcome[(f["winner"], f["forfeit"], last and last["reward"], last and last["term"], last and last["trunc"])] += 1
    # every non-last row must carry reward 0 / done 0
    for r in rows[:-1]:
        if r["done"] or r["reward"] != 0.0:
            anomalies.append(("non-last row done/reward", gi, r))
    if last and not last["done"]:
        anomalies.append(("last row not done", gi, last))
    t = ffi.trace(lib, f["script"], commit="audit", label=f"g{gi}")
    txt = t["records"]["p1"]
    if not isinstance(txt, str):
        txt = json.dumps(txt)
    for line in txt.splitlines():
        if "|error|" in line:
            i = line.index("|error|")
            msg = line[i:i + 120]
            key = msg.split("\\n")[0][:90]
            errs[key] += 1
            err_examples.setdefault(key, (gi, f["script"].count("CHOOSE p1")))
print(f"games {len(games)}  p1 decisions {n_dec}")
print("outcomes (winner, forfeit, last_reward, term, trunc):")
for k, v in sorted(outcome.items(), key=lambda kv: -kv[1]):
    print("  ", k, v)
print("p1 |error| lines in the replayed records:")
for k, v in errs.most_common():
    print("  ", v, k, "example game", err_examples[k])
print("anomalies:", len(anomalies))
for a in anomalies[:10]:
    print("  ", a)
json.dump(dict(games=[dict(fin={k: v for k, v in g["fin"].items()}, rows=g["rows"]) for g in games]),
          open(f"/tmp/gigo_audit/B/games_{POLICY}_{SEED}_{N_BATTLES}.json", "w"))
