"""E10 offline read: hidden-slot move recall@4, the flat sentinel row vs the parameter-free mixture,
over pool / ladder teams (team-level, no battles): reveal the first k species, score each hidden
mon's true moves against the top-4 of each prior."""
import random, sys, torch, numpy as np
from agents import gen3_data
from agents.model.belief_heads import MoveBelief
from agents.model.t0_species import T0SpeciesPrior
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.model.belief_tables import _belief_num
from utils import team_sources
from poke_env.teambuilder.teambuilder import Teambuilder
from poke_env.data.normalize import to_id_str
lay = Gen3ObservationEncoder(load_mappings()).get_layout()
mb = MoveBelief(lay["max_moves"], lay["move_embedding_dim"], prior_fusion=True, n_species=lay["max_species"], move_candidate_floor=0.02)
t0 = T0SpeciesPrior(lay["max_species"])
def team_of(packed):
    out=[]
    mons = Teambuilder.parse_packed_team(packed) if '|' in packed.split('\n')[0] else Teambuilder.parse_showdown_team(packed)
    for m in mons:
        sp = gen3_data.species.get(to_id_str(m.species or m.nickname))
        if sp is None: return None
        mv=set()
        for x in m.moves:
            md = gen3_data.moves.get(to_id_str(x))
            if md is not None: mv.add(_belief_num(to_id_str(x), md))
        out.append((sp.num, mv))
    return out if len(out)==6 else None
res={}
for source in sys.argv[1:]:
    flat=[];mix=[]
    for key in range(0, 300, 2):
        try: a,_=team_sources.pair(source,key)
        except Exception as e: continue
        t=team_of(a)
        if t is None: continue
        for k in range(1,6):
            ids=torch.zeros(1,6,dtype=torch.long)
            for i in range(k): ids[0,i]=t[i][0]
            bel=ids==0
            probs=t0(ids,bel)
            hid=mb.hidden_slot_prior_logits(probs)[0]
            flatrow=mb.move_prior_logits[0]
            top_mix=set(torch.topk(hid,4).indices.tolist()); top_flat=set(torch.topk(flatrow,4).indices.tolist())
            for i in range(k,6):
                true=t[i][1]
                if not true: continue
                mix.append(len(true&top_mix)/min(4,len(true))); flat.append(len(true&top_flat)/min(4,len(true)))
    print(source, "hidden slots", len(mix), "recall@4 flat %.3f mixture %.3f" % (np.mean(flat), np.mean(mix)))
