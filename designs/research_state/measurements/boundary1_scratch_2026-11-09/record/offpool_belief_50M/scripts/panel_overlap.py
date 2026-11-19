import json,sys
from main import heldout_panels as HP
from main.anchors.team_overlap import Reference, parse_items, overlap
from utils.team_corpora import load_corpus
cp=load_corpus("n1_proc_v1","ALL")
ref=Reference(parse_items(list(zip(cp.ids,cp.texts))))
out={"corpus":cp.ident(),"n_corpus":len(cp)}
for p in ("home72","heldout72_hpfix","forum72","p72"):
    e=HP.panel_entry(p)
    out[p]={"held_out_from":e["held_out_from"],"content_digest":e["content_digest"],"overlap":overlap(parse_items(HP.panel_team_texts(p)),ref)}
json.dump(out,open(sys.argv[1],"w"),indent=1)
for p in ("home72","heldout72_hpfix","forum72","p72"):
    o=out[p]["overlap"]; print(p,{k:o[k] for k in("n_read_teams","exact_team","same_species6","mons_seen","n_read_mons","teams_sharing_ge4","max_shared_mons_hist")})
