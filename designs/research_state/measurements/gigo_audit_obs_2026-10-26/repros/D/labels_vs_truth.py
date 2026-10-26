"""SCOPE D repro: every core label family declared, self-play (both sides answered here, seeded uniform over the
mask), and every p1 label compared at every p1 decision to an INDEPENDENT truth: p2's packed team text (species,
moves, item, nature/EVs/IVs, L100 stat formula) and the action p2 actually submitted (intent).

usage: labels_vs_truth.py <n_battles> <seed> [teams_file]   (teams_file: one packed team per line; default pool[:12])
"""
import json
import sys
from collections import Counter, defaultdict

import numpy as np

from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from utils.rust_env import ffi
from utils.rust_env import protocol as P
from utils.rust_env.build import ensure_built
from utils.rust_env.columns import LABEL_FAMILIES
from utils.paths import repo_path

L = Gen3ObservationEncoder(load_mappings()).get_layout()
OPP0 = L["parts"]["opp_team"]["start"]
OUR0 = L["parts"]["our_team"]["start"]
PD = L["parts"]["opp_team"]["reshape"][1]
SP = L["pokemon"]["species"]["offset"] + L["pokemon"]["species"]["layout"]["species_id"]["offset"]
SK = L["pokemon"]["species_known"]["offset"]
REACT0 = L["parts"]["reactive"]["start"]
ARM = L["reactive_layout"]["active_req_moves"]["offset"]

D = repo_path("data", "pokemon")
SPECIES = json.load(open(D / "gen3_species.json"))
MOVES = json.load(open(D / "gen3_moves.json"))
CAT = {int(v["num"]): v.get("category", "?") for v in MOVES.values()}
NAME = {int(v["num"]): k for k, v in MOVES.items()}
ITEMS = json.load(open(D / "gen3_items.json"))
NATURES = json.load(open(D / "gen3_natures.json"))
ALIASES = json.load(open(D / "gen3_move_aliases.json"))
HP_TYPES = ["bug", "dark", "dragon", "electric", "fighting", "fire", "flying", "ghost", "grass", "ground", "ice",
            "poison", "psychic", "rock", "steel", "water"]
STATS = ["hp", "atk", "def", "spa", "spd", "spe"]


def to_id(s):
    return "".join(ch for ch in s.lower() if ch.isalnum())


def parse_packed(team):
    mons = []
    for m in team.split("]"):
        f = m.split("|")
        species = to_id(f[1] or f[0])
        evs = [int(x) if x else 0 for x in (f[6].split(",") if f[6] else [""] * 6)]
        ivs = [int(x) if x else 31 for x in (f[8].split(",") if f[8] else [""] * 6)]
        lvl = int(f[10]) if len(f) > 10 and f[10] else 100
        misc = f[11].split(",") if len(f) > 11 and f[11] else []
        hpt = to_id(misc[1]) if len(misc) > 1 and misc[1] else ""
        mv = [ALIASES.get(to_id(x), to_id(x)) for x in f[4].split(",") if x]
        if hpt:
            mv = [("hiddenpower" + hpt if x == "hiddenpower" else x) for x in mv]
        elif "hiddenpower" in mv:
            iv6 = [int(x) if x else 31 for x in (f[8].split(",") if f[8] else [""] * 6)]
            a = sum(((iv6[i] & 1) << k) for k, i in enumerate([0, 1, 2, 5, 3, 4]))
            order = ["fighting", "flying", "poison", "ground", "rock", "bug", "ghost", "steel", "fire", "water", "grass",
                     "electric", "psychic", "ice", "dragon", "dark"]
            mv = [("hiddenpower" + order[a * 15 // 63] if x == "hiddenpower" else x) for x in mv]
        mons.append(dict(species=species, item=to_id(f[2]), moves=mv,
                         nature=to_id(f[5]) or "serious", evs=evs, ivs=ivs, level=lvl))
    return mons


def stat(base, ev, iv, mult, level=100):
    pre = (2 * base + iv + ev // 4) * level // 100 + 5
    return pre * 11 // 10 if mult > 1 else (pre * 9 // 10 if mult < 1 else pre)


def truth_of(mon):
    sd = SPECIES[mon["species"]]
    nat = NATURES[mon["nature"]]
    st = [stat(sd["baseStats"][s], mon["evs"][i + 1], mon["ivs"][i + 1], nat.get(s, 1.0), mon["level"])
          for i, s in enumerate(STATS[1:])]
    hp = next((HP_TYPES.index(m[11:]) for m in mon["moves"] if m.startswith("hiddenpower") and m[11:] in HP_TYPES), None)
    return dict(num=int(sd["num"]), moves=sorted(int(MOVES[m]["num"]) for m in mon["moves"]),
                item=int(ITEMS[mon["item"]]["num"]) if mon["item"] else 0, stats=st, nature=int(nat["num"]),
                ev=[mon["evs"][i] // 4 * 4 for i in range(1, 6)], hp=hp)


def main():
    n_battles, seed = int(sys.argv[1]), int(sys.argv[2])
    if len(sys.argv) > 3:
        teams = [ln.strip() for ln in open(sys.argv[3]) if ln.strip()]
    else:
        from utils.team_sources import packed_teams
        teams = list(packed_teams("pool"))[:12]
    T = [[truth_of(m) for m in parse_packed(t)] for t in teams]
    fams = tuple(LABEL_FAMILIES)
    ensure_built("release")
    n_envs = 2
    spec = P.spec_json(n=n_envs, threads=1, teams=teams, names=("audone", "audtwo"), turn_limit=250,
                       refusal_budget=4, bank_dir=None, labels=fams, opponents=[{"kind": "external"}])
    core = ffi.FfiCore(spec, lib=ffi.load(ffi.default_path("release")))
    c = core.cols
    rng = np.random.default_rng(seed)
    staged, cur = {}, {}

    def stage(e):
        staged[e] = tuple(int(x) for x in rng.choice(len(teams), 2, replace=False))
        c["ep_team"][e] = staged[e]
        c["ep_seed"][e] = rng.integers(0, 65536, 4)
        c["ep_opp"][e] = 0

    for e in range(n_envs):
        stage(e)
    core.reset()
    for e in range(n_envs):
        cur[e] = staged[e]
        stage(e)
    win = {e: [] for e in range(n_envs)}       # p2 submissions since p1's last decision
    prev_frame = {e: None for e in range(n_envs)}   # p1's previous row opp species per slot (revealed)
    cnt = Counter()
    bad = defaultdict(list)
    done = 0
    epc = {e: 0 for e in range(n_envs)}
    hist = {e: [] for e in range(n_envs)}
    while done < n_battles:
        for e in range(n_envs):
            if c["need"][e, 0]:
                check(e, c, T[cur[e][1]], win[e], prev_frame[e], cnt, bad, (e, epc[e]), hist[e])
                obs = c["obs"][e, 0]
                prev_frame[e] = [int(round(obs[OPP0 + k * PD + SP])) if obs[OPP0 + k * PD + SK] >= 0.5 else None
                                 for k in range(6)]
                win[e] = []
        for e in range(n_envs):
            for s in (0, 1):
                if c["need"][e, s]:
                    m = np.flatnonzero(c["mask"][e, s].astype(bool))
                    a = int(rng.choice(m))
                    c["action"][e, s] = a
                    if s == 1:
                        o = c["obs"][e, 1]
                        req = [int(round(o[REACT0 + ARM + j])) for j in range(4)]
                        own = [int(round(o[OUR0 + k * PD + SP])) for k in range(6)]
                        win[e].append(dict(a=a, n_legal=len(m), req=req, own=own, forced=not c['mask'][e, 1][6:].any()))
                    o1 = c['obs'][e, s]
                    hist[e].append((int(c['turn'][e]), s, a, len(m)))
        core.step()
        for f in core.finished():
            e = int(f["env"])
            done += 1
            epc[e] += 1
            hist[e] = []
            cur[e] = staged[e]
            stage(e)
            win[e] = []
            prev_frame[e] = None
    core.close()
    print(json.dumps({"counts": dict(sorted(cnt.items()))}, indent=0))
    for k, v in sorted(bad.items()):
        print(f"MISMATCH {k}: {len(v)}  e.g. {v[:4]}")


def check(e, c, truth, window, prev_frame, cnt, bad, eid=None, hist=None):
    obs = c["obs"][e, 0]
    known = [obs[OPP0 + k * PD + SK] >= 0.5 for k in range(6)]
    sp = [int(round(obs[OPP0 + k * PD + SP])) for k in range(6)]
    by_num = {}
    for t in truth:
        by_num[t["num"]] = t  # a later same-species mon overrides (as the writer); species clause makes it moot
    cnt["rows"] += 1
    dec = int(c["dec_n"][e, 0])
    turn = int(c["turn"][e])
    tag = (eid, dec, turn)
    forced = not c['mask'][e, 0][6:].any()
    cnt['forced_rows'] += int(forced)
    bs = c["belief_species"][e, 0]
    bm = c["belief_moves"][e, 0]
    km = c["known_moves"][e, 0]
    revealed_nums = []
    for k in range(6):
        if not known[k]:
            continue
        cnt["revealed_slots"] += 1
        t = by_num.get(sp[k])
        if t is None:
            bad["obs_species_not_on_truth_team"].append((tag, k, sp[k]))
            continue
        revealed_nums.append(sp[k])
        if sorted(int(x) for x in km[k] if x >= 0) != t["moves"]:
            bad["known_moves"].append((tag, k, sorted(int(x) for x in km[k]), t["moves"]))
        il, im = int(c["item_label"][e, 0][k]), float(c["item_mask"][e, 0][k])
        if im != 1.0:
            bad["item_mask_not_1"].append((tag, k))
        elif il != t["item"]:
            cnt["item_differs_from_initial"] += 1
            bad["item_vs_initial(info)"].append((tag, k, il, t["item"]))
        hl, hm = int(c["hp_type_label"][e, 0][k]), float(c["hp_type_mask"][e, 0][k])
        if (t["hp"] is None) != (hm == 0.0) or (t["hp"] is not None and hl != t["hp"]):
            bad["hp_type"].append((tag, k, hl, hm, t["hp"]))
        spr = [int(x) for x in c["belief_spread"][e, 0][k]]
        if spr != t["stats"] or c["belief_spread_mask"][e, 0][k] != 1.0:
            bad["spread"].append((tag, k, spr, t["stats"]))
        if int(c["belief_nature"][e, 0][k]) != t["nature"] or c["belief_nature_mask"][e, 0][k] != 1.0:
            bad["nature"].append((tag, k, int(c["belief_nature"][e, 0][k]), t["nature"]))
        if [int(x) for x in c["belief_ev"][e, 0][k]] != t["ev"] or c["belief_ev_mask"][e, 0][k] != 1.0:
            bad["ev"].append((tag, k, [int(x) for x in c["belief_ev"][e, 0][k]], t["ev"]))
        if bs[k] != -1 or any(bm[k] != -1):
            bad["belief_on_revealed_slot"].append((tag, k))
    for k in range(6):
        if not known[k]:
            for key in ("item_mask", "hp_type_mask", "belief_spread_mask", "belief_nature_mask", "belief_ev_mask"):
                if c[key][e, 0][k] != 0:
                    bad[f"{key}_on_unrevealed"].append((tag, k))
            if any(km[k] != -1):
                bad["known_moves_on_unrevealed"].append((tag, k))
    hidden = sorted(t["num"] for t in truth if t["num"] not in revealed_nums)
    believed = [k for k in range(6) if not known[k]]
    exp_bs = [-1] * 6
    exp_bm = [None] * 6
    for slot, num in zip(believed, hidden):
        exp_bs[slot] = num
        exp_bm[slot] = by_num[num]["moves"]
    if [int(x) for x in bs] != exp_bs:
        bad["belief_species"].append((tag, [int(x) for x in bs], exp_bs))
    for k in believed:
        got = sorted(int(x) for x in bm[k] if x >= 0)
        if (exp_bm[k] or []) != got:
            bad["belief_moves"].append((tag, k, got, exp_bm[k]))
    # ---- intent
    kind, num = int(c["opp_action_kind"][e, 0][0]), int(c["opp_action_num"][e, 0][0])
    slot, spc = int(c["opp_switch_slot"][e, 0][0]), int(c["opp_switch_species"][e, 0][0])
    cnt[f"intent_kind_{kind}"] += 1
    if dec == 0:
        if (kind, num, slot, spc) != (2, 0, -100, 0):
            bad["intent_at_reset"].append((tag, kind, num, slot, spc))
        return
    if window and 6 <= window[0]["a"] < 10:
        nm = NAME.get(window[0]["req"][window[0]["a"] - 6])
        if nm in ("explosion", "selfdestruct", "memento", "destinybond", "batonpass"):
            cnt[f"chosen_{nm}"] += 1
            cnt[f"chosen_{nm}_labelled_kind{kind}"] += 1
    if len(window) == 2 and 6 <= window[0]["a"] < 10 and window[1]["a"] < 6 and window[1]["forced"]:
        cnt[f"w2_move_then_forced_replacement_kind{kind}"] += 1
        bad["w2_move_then_replacement_example(info)"].append((tag, kind, num, NAME.get(window[0]["req"][window[0]["a"] - 6])))
    if len(window) != 1:
        cnt[f"intent_window_len_{len(window)}"] += 1
        if kind != 2:
            bad[f"intent_label_on_window_len_{len(window)}(info)"].append((tag, kind, num, slot, spc, [w["a"] for w in window], hist[-6:]))
        return
    w = window[0]
    a = w["a"]
    if 6 <= a < 10:
        exp_num = w["req"][a - 6]
        if kind == 0:
            cnt["intent_move_labelled"] += 1
            cnt["cat_labelled_" + str(CAT.get(exp_num))] += 1
            hp_ok = exp_num == 237 and 355 <= num <= 370
            if num != exp_num and not hp_ok:
                bad["intent_move_num_vs_chosen(check called/encore)"].append((tag, num, exp_num))
        elif kind == 1:
            bad["intent_switch_label_for_chosen_move"].append((tag, num, slot, spc, exp_num))
        else:
            cnt["intent_move_masked"] += 1
            cnt["intent_move_masked_on_forced_row" if forced else "intent_move_masked_on_normal_row"] += 1
            cnt[("cat_maskforced_" if forced else "cat_masknormal_") + str(CAT.get(exp_num))] += 1
            if not forced:
                cnt["masknormal_move_" + NAME.get(exp_num, str(exp_num))] += 1
                bad["masknormal_example_" + NAME.get(exp_num, str(exp_num)) + "(info)"].append(tag)
    elif a < 6:
        exp_sp = w["own"][a]
        if kind == 1:
            cnt["intent_switch_labelled"] += 1
            if spc != exp_sp:
                bad["intent_switch_species"].append((tag, spc, exp_sp))
            exp_slot = (prev_frame.index(exp_sp) if (prev_frame and exp_sp in prev_frame) else -100)
            if slot != exp_slot:
                bad["intent_switch_slot"].append((tag, slot, exp_slot, prev_frame))
        elif kind == 0:
            bad["intent_move_label_for_chosen_switch"].append((tag, num, exp_sp, w["n_legal"]))
        else:
            cnt["intent_switch_masked"] += 1
    else:
        if kind == 0 and num != int(MOVES["struggle"]["num"]):
            bad["intent_struggle"].append((tag, num))


if __name__ == "__main__":
    main()
