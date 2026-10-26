"""Run the pack oracle over the pool and both corpora through the real teambuilder paths; time it."""
import gzip, json, time
from utils.team_sources import team_list
from utils.team_corpora import CORPORA, DATA_NAME
from utils.teambuilder import Gen3Teambuilder
t = time.time()
tb = Gen3Teambuilder(team_list("pool"))
print("pool ok", len(tb.packed_teams), f"{time.time()-t:.1f}s")
for name, d in CORPORA.items():
    texts = [json.loads(l)["text"] for l in gzip.open(d / DATA_NAME, "rt")]
    t = time.time()
    tb2 = Gen3Teambuilder(team_list("pool")[:3], mix_teams=texts, mix_prob=0.5)
    print(name, "ok", len(tb2.mix_packed_teams), f"{time.time()-t:.1f}s")
