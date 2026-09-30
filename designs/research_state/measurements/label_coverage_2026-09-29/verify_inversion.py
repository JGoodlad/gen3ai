import sys, collections
from main.rust_core_cutover.envs import packed_teams
from poke_env.teambuilder.teambuilder import Teambuilder
from poke_env.data.normalize import to_id_str
from agents import gen3_data
from old_belief_tables import invert_nature_evs  # `git show 74393cd6:src/agents/model/belief_tables.py > old_belief_tables.py`
ORDER = ("atk", "def", "spa", "spd", "spe"); IDX = {"hp":0,"atk":1,"def":2,"spa":3,"spd":4,"spe":5}
nat = gen3_data.natures.raw()
def stat(b, iv, ev, lvl, m):
    pre = (2*b + iv + ev//4) * lvl // 100 + 5
    return pre*11//10 if m > 1 else pre*9//10 if m < 1 else pre
for src in sys.argv[1:]:
    teams = packed_teams(src)
    c = collections.Counter()
    for t in teams:
        for tb in Teambuilder.parse_packed_team(t):
            sid = to_id_str(tb.species or tb.nickname)
            sd = gen3_data.species.get(sid)
            if sd is None: c["nodex"] += 1; continue
            lvl = tb.level or 100
            nname = (tb.nature or "serious").lower(); nv = nat.get(nname) or nat["serious"]
            ivs = tb.ivs or [31]*6; evs = tb.evs or [0]*6
            base = [sd.base_stats[k] for k in ORDER]
            der = [stat(base[j], ivs[IDX[k]], evs[IDX[k]], lvl, float(nv.get(k, 1.0))) for j, k in enumerate(ORDER)]
            res = invert_nature_evs(der, base, species_id=sid)
            iv31 = all(ivs[IDX[k]] == 31 for k in ORDER)
            c["mons"] += 1; c[f"iv31={iv31}"] += 1; c[f"lvl100={lvl==100}"] += 1
            if res is None:
                c[f"FAIL iv31={iv31} lvl100={lvl==100}"] += 1
            else:
                truev = [evs[IDX[k]]//4*4 for k in ORDER]
                ok_n = int(nv["num"]) == res[0]; ok_e = list(res[1]) == truev
                c[f"inv ok iv31={iv31} nature_eq={ok_n} ev_eq={ok_e}"] += 1
            if any(e % 4 for e in evs): c["ev_not_mult4"] += 1
            if any(e > 252 for e in evs): c["ev_gt252"] += 1
    print(src, len(teams), dict(sorted(c.items())))
