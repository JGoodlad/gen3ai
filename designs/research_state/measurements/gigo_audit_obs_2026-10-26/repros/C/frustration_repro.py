"""CONFIRM: a paste's `Happiness: 0` is dropped by OUR packer (team_packing `self.happiness or ""`), so the Rust
engine unpacks happiness 255 and Frustration hits at BP 1 instead of 102. Plays the pool team
teams/others/yak_attack/88a79879fa283313.txt (Snorlax lead, Frustration) on the Rust env core twice: as OUR training
path packs it, and with the misc field carrying '0' (Showdown's own pack). p1 always clicks Frustration when legal."""
import json, sys
import numpy as np
from utils.team_loader import TeamLoader
from utils.team_packing import Teambuilder
from utils.gen3_utils import fix_gen3_hp_ivs
from utils.rust_env import ffi, protocol as P
from utils.rust_env.build import ensure_built
from agents import gen3_data
from agents.observation.state_encoder import load_mappings, Gen3ObservationEncoder
from agents.observation.constants import EventCol, OFFSET_EVENT_WINDOW
from agents.observation.schema import build_schema

L = Gen3ObservationEncoder(load_mappings()).get_layout()
sl = build_schema(L).slices()
arm = sl['reactive.active_req_moves']; per = (arm.stop - arm.start) // 3
FRUS = gen3_data.moves.raw()['frustration']['num']
N_EV = L['event_window_n']; EVD = len(EventCol)
tl = TeamLoader()
txt = dict(tl._loaded)['teams/others/yak_attack/88a79879fa283313.txt']
mons = Teambuilder.parse_showdown_team(txt)
snor = [m for m in mons if 'Frustration' in m.moves][0]
mons = [snor] + [m for m in mons if m is not snor]
ours = Teambuilder.join_team(fix_gen3_hp_ivs(mons))
lead = ours.split(']')[0]
print("OUR packed lead :", lead)
fixed_lead = lead[:lead.rfind('|') + 1] + "0"          # Showdown's pack writes happiness 0 in the misc field
fixed = ']'.join([fixed_lead] + ours.split(']')[1:])
print("FIXED packed lead:", fixed_lead)
opp = Teambuilder.join_team(fix_gen3_hp_ivs(Teambuilder.parse_showdown_team(tl._loaded[0][1])))
ensure_built("release"); lib = ffi.load(ffi.default_path("release"))

def run(team, seed):
    spec = P.spec_json(n=1, threads=1, teams=[team, opp], names=("fixone", "fixtwo"), turn_limit=60, refusal_budget=4,
                       bank_dir=None, opponents=[{"kind": "bot", "bot": "heuristic", "seed": 3, "streams": "episode"}])
    core = ffi.FfiCore(spec, lib=lib); c = core.cols
    def stage():
        c["ep_team"][0] = [0, 1]; c["ep_seed"][0] = [seed, 2, 3, 4]; c["ep_opp"][0] = 0
    stage(); core.reset(); stage()
    mags = []
    try:
        for _ in range(400):
            if c["need"][0, 0]:
                o = c["obs"][0, 0].copy(); m = c["mask"][0, 0].astype(bool)
                ev = o[OFFSET_EVENT_WINDOW:OFFSET_EVENT_WINDOW + N_EV * EVD].reshape(N_EV, EVD)
                for r in ev:
                    if r[EventCol.VALID] > 0.5 and int(r[EventCol.MOVE]) == FRUS and r[EventCol.ACTOR_SIDE] > 0:
                        mags.append(round(float(r[EventCol.MAGNITUDE]), 4))
                ids = o[arm.start:arm.start + per].astype(int)
                a = next((6 + k for k in range(per) if ids[k] == FRUS and m[6 + k]), None)
                if a is None:
                    a = next((6 + k for k in range(per) if m[6 + k]), int(np.flatnonzero(m)[0]))
                c["action"][0, 0] = a
            core.step()
            if core.finished():
                break
    finally:
        getattr(core, "close", lambda: None)()
    return mags
for name, team in (("OURS (happiness dropped)", ours), ("FIXED (happiness 0)", fixed)):
    for seed in (11, 12):
        print(name, "seed", seed, "Frustration event MAGNITUDE rows (dedup by window, oldest first):", sorted(set(run(team, seed))))
