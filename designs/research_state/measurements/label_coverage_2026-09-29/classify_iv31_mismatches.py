import sys, collections
from main.rust_core_cutover.envs import packed_teams
from poke_env.teambuilder.teambuilder import Teambuilder
from poke_env.data.normalize import to_id_str
from agents import gen3_data
from old_belief_tables import invert_nature_evs  # `git show 74393cd6:src/agents/model/belief_tables.py > old_belief_tables.py`
ORDER = ("atk", "def", "spa", "spd", "spe"); IDX = {"hp":0,"atk":1,"def":2,"spa":3,"spd":4,"spe":5}
nat = gen3_data.natures.raw(); bynum = {int(v["num"]): k for k, v in nat.items()}
def mults(n): return tuple(float(nat[n].get(k,1.0)) for k in ORDER)
def stat(b, iv, ev, lvl, m):
    pre = (2*b + iv + ev//4) * lvl // 100 + 5
    return pre*11//10 if m > 1 else pre*9//10 if m < 1 else pre
src = sys.argv[1]; c = collections.Counter(); ex = []
for t in packed_teams(src):
    for tb in Teambuilder.parse_packed_team(t):
        sid = to_id_str(tb.species or tb.nickname); sd = gen3_data.species.get(sid)
        ivs = tb.ivs or [31]*6; evs = tb.evs or [0]*6
        if any(ivs[IDX[k]] != 31 for k in ORDER): continue
        nname = (tb.nature or "serious").lower()
        base = [sd.base_stats[k] for k in ORDER]
        der = [stat(base[j], 31, evs[IDX[k]], 100, mults(nname)[j]) for j, k in enumerate(ORDER)]
        num, ev = invert_nature_evs(der, base, species_id=sid)
        if bynum[num] != nname:
            same = mults(bynum[num]) == mults(nname)
            c[f"nature differs, same_mults={same}"] += 1
            if not same and len(ex) < 5: ex.append((sid, nname, evs, bynum[num], ev))
            if same and [evs[IDX[k]]//4*4 for k in ORDER] != list(ev): c["same mults but ev differ"] += 1
        elif [evs[IDX[k]]//4*4 for k in ORDER] != list(ev):
            c["ev differs"] += 1
            if len(ex) < 8: ex.append(("EV", sid, nname, evs, ev))
print(src, dict(c)); [print(e) for e in ex]
