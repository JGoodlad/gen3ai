"""Trace env steps (turn, need, p1/p2 actions, p1 intent label) for one battle of labels_vs_truth's run, + the finished() record."""
import sys, json, numpy as np
sys.argv = [sys.argv[0]] + sys.argv[1:]
import labels_vs_truth as LV
from utils.rust_env import ffi, protocol as P
from utils.rust_env.columns import LABEL_FAMILIES
from utils.team_sources import packed_teams
seed = int(sys.argv[1]); want_env = int(sys.argv[2]); want_ep = int(sys.argv[3])
teams = ([l.strip() for l in open(sys.argv[4]) if l.strip()] if len(sys.argv) > 4 else list(packed_teams("pool"))[:12])
spec = P.spec_json(n=2, threads=1, teams=teams, names=("audone", "audtwo"), turn_limit=250, refusal_budget=4,
                   bank_dir=None, labels=tuple(LABEL_FAMILIES), opponents=[{"kind": "external"}])
core = ffi.FfiCore(spec, lib=ffi.load(ffi.default_path("release"))); c = core.cols
rng = np.random.default_rng(seed); staged = {}; cur = {}; epi = {0: 0, 1: 0}
def stage(e):
    staged[e] = tuple(int(x) for x in rng.choice(len(teams), 2, replace=False))
    c["ep_team"][e] = staged[e]; c["ep_seed"][e] = rng.integers(0, 65536, 4); c["ep_opp"][e] = 0
for e in range(2): stage(e)
core.reset()
for e in range(2): cur[e] = staged[e]; stage(e)
done = 0
while epi[want_env] <= want_ep:
    e = want_env
    if epi[e] == want_ep:
        lab = [int(c[k][e, 0][0]) for k in ("opp_action_kind", "opp_action_num", "opp_switch_slot", "opp_switch_species")]
        print(f"turn {int(c['turn'][e])} need {c['need'][e].tolist()} dec_n {c['dec_n'][e].tolist()} p1label {lab if c['need'][e,0] else '-'}")
    acts = {}
    for ee in range(2):
        for s in (0, 1):
            if c["need"][ee, s]:
                m = np.flatnonzero(c["mask"][ee, s].astype(bool)); a = int(rng.choice(m)); c["action"][ee, s] = a
                if ee == e: acts[s] = a
    if epi[e] == want_ep:
        o = c["obs"][e, 1]
        req = [int(round(o[LV.REACT0 + LV.ARM + j])) for j in range(4)]
        own = [int(round(o[LV.OUR0 + k * LV.PD + LV.SP])) for k in range(6)]
        desc = {s: (f"move {req[a-6]}" if 6 <= a < 10 else (f"switch slot{a} sp{own[a]}" if a < 6 else "struggle")) if s == 1 else a for s, a in acts.items()}
        print("   submit", desc)
    core.step()
    for f in core.finished():
        ee = int(f["env"])
        if ee == want_env and epi[ee] == want_ep:
            open(f"/tmp/gigo_audit/D/script_s{seed}_e{want_env}_ep{want_ep}.txt","w").write(f["script"]); print("SCRIPT SAVED")
        epi[ee] += 1; cur[ee] = staged[ee]; stage(ee)
core.close()
