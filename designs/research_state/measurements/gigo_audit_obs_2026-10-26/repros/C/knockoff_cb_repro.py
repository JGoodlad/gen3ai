"""PROBE: after Knock Off removes an opponent's Choice Band, the obs writes item id = Choice Band WITH consumed = 1
(present layer '-enditem' -> consumed_item; encoder item() consumed branch). The op's CB reads compare the id
alone (damage_op.py `revealed_cb = (opp_item == cb_item_num)`), so a knocked-off Band reads P(CB) = 1."""
import numpy as np
from utils.rust_env import ffi, protocol as P
from utils.rust_env.build import ensure_built
from agents import gen3_data
from agents.observation.state_encoder import load_mappings, Gen3ObservationEncoder
from agents.observation.schema import build_schema
from agents.observation.constants import OFFSET_OPP_TEAM, POKEMON_FULL_DIM, POKEMON_ITEMS_OFFSET, POKEMON_ACTIVE_OFFSET, POKEMON_HP_OFFSET
L = Gen3ObservationEncoder(load_mappings()).get_layout(); sl = build_schema(L).slices()
arm = sl['reactive.active_req_moves']; per = (arm.stop - arm.start) // 3
KO = gen3_data.moves.raw()['knockoff']['num']; CB = gen3_data.items.raw()['choiceband']['num']
p1 = ']'.join(["Dusclops||leftovers|pressure|knockoff,seismictoss,willowisp,rest|Impish|252,,252,,4,|||||",
               "Snorlax||leftovers|thickfat|knockoff,bodyslam,rest,sleeptalk|Careful|252,,4,,252,|||||",
               "Skarmory||leftovers|keeneye|knockoff,spikes,roar,toxic|Impish|252,,252,,4,|||||"])
p2 = ']'.join(["Tauros||choiceband|intimidate|doubleedge,earthquake,rockslide,return|Jolly|,252,,,4,252|||||",
               "Aerodactyl||choiceband|rockhead|rockslide,doubleedge,earthquake,hiddenpowerflying|Jolly|4,252,,,,252||30,30,30,30,30,|||",
               "Heracross||choiceband|guts|megahorn,brickbreak,rockslide,hiddenpowerghost|Adamant|,252,,,4,252|||||"])
ensure_built("release"); lib = ffi.load(ffi.default_path("release"))
spec = P.spec_json(n=1, threads=1, teams=[p1, p2], names=("fixone", "fixtwo"), turn_limit=60, refusal_budget=4,
                   bank_dir=None, opponents=[{"kind": "bot", "bot": "heuristic", "seed": 3, "streams": "episode"}])
core = ffi.FfiCore(spec, lib=lib); c = core.cols
def stage(): c["ep_team"][0] = [0, 1]; c["ep_seed"][0] = [5, 6, 7, 8]; c["ep_opp"][0] = 0
stage(); core.reset(); stage()
hits = 0
saved = []
try:
    for _ in range(600):
        if c["need"][0, 0]:
            o = c["obs"][0, 0].copy(); m = c["mask"][0, 0].astype(bool)
            for s in range(6):
                b = OFFSET_OPP_TEAM + s * POKEMON_FULL_DIM
                iid, known, cons = o[b + POKEMON_ITEMS_OFFSET: b + POKEMON_ITEMS_OFFSET + 3]
                if int(iid) == CB and cons > 0.5:
                    hits += 1
                    saved.append((o.copy(), b))
                    if hits <= 4:
                        print(f"opp slot {s} species {int(o[b])} item id {int(iid)} (Choice Band={CB}) known {known} consumed {cons} "
                              f"active {o[b + POKEMON_ACTIVE_OFFSET]} hp {o[b + POKEMON_HP_OFFSET]:.2f} -> op revealed_cb = {float(int(iid) == CB)}")
            ids = o[arm.start:arm.start + per].astype(int)
            a = next((6 + k for k in range(per) if ids[k] == KO and m[6 + k]), None)
            c["action"][0, 0] = a if a is not None else int(np.flatnonzero(m)[0])
        core.step()
        if core.finished():
            break
finally:
    getattr(core, "close", lambda: None)()
print("rows x opp slots with a knocked-off Choice Band read as item id = CB:", hits)

# --- the production extractor's op on a saved row: P(CB) for the knocked-off Band vs the same row with a revealed non-CB item
import torch
from agents.model.delivery_graph import build_extractor
from agents.model.extra_obs_keys import zero_extra_obs
fe, cfg, _ = build_extractor()
o, b = saved[0]
alt = o.copy(); alt[b + POKEMON_ITEMS_OFFSET] = gen3_data.items.raw()['leftovers']['num']; alt[b + POKEMON_ITEMS_OFFSET + 2] = 0.0
none = o.copy(); none[b + POKEMON_ITEMS_OFFSET:b + POKEMON_ITEMS_OFFSET + 3] = 0.0
for name, row in (("knocked-off Band (id CB, consumed 1)", o), ("revealed Leftovers (id 234, known 1)", alt), ("all-zero item (unrevealed)", none)):
    obs = {"observation": torch.from_numpy(row[None].astype(np.float32)), **zero_extra_obs(fe, 1)}
    with torch.no_grad():
        fe(obs)
    t = fe.damage_op.last_raw_tensors
    print(f"{name:40s} op p_cb = {float(t.p_cb[0, 0]):.4f}  cb_high[active] = {t.cb_high[0].tolist()}")
