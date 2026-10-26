"""X61 review check: how much of each panel the N1 corpus and the pool already hold, per TEAM, per mon SET and per
full set. CPU, read-only, deterministic. Run from a checkout with PYTHONPATH=src:
    python designs/research_state/measurements/x61_n1_breadth_2026-10-25/review/coverage.py
mon set = species + sorted moves, Hidden Power typed (gen3_team_overlap_v1 ``key``); full set = + item, ability,
nature, EVs (``strict_key``)."""
import collections

from main import heldout_panels as HPN
from main.anchors.team_overlap import Reference, parse_items
from utils.paths import repo_path
from utils.team_corpora import load_corpus
from utils.team_loader import TeamLoader

corp = parse_items([(f"c{i}", t) for i, t in enumerate(load_corpus("n1_proc_v1", "ALL").texts)])
pool = parse_items([(f"p{i}", t) for i, t in enumerate(TeamLoader(str(repo_path("data", "teams"))).get_all_teams())])
cref, pref = Reference(corp), Reference(pool)
cmon = collections.Counter(m.key for t in corp for m in t.mons)
pmon = {m.key for t in pool for m in t.mons}
cfull = {m.strict_key for t in corp for m in t.mons}
pfull = {m.strict_key for t in pool for m in t.mons}
print("panel | mon sets in corpus (>=5 copies) / in pool | full sets in corpus / in pool | max shared with one corpus "
      "team | with one pool team")
for p in ("heldout72_hpfix", "forum72", "p72", "home72"):
    teams = parse_items(HPN.panel_team_texts(p))
    mons = [m for t in teams for m in t.mons]
    n = len(mons)
    f = lambda c: round(c / n, 3)
    print(p, n, f(sum(m.key in cmon for m in mons)), f(sum(cmon.get(m.key, 0) >= 5 for m in mons)),
          f(sum(m.key in pmon for m in mons)), "|", f(sum(m.strict_key in cfull for m in mons)),
          f(sum(m.strict_key in pfull for m in mons)), "|",
          dict(sorted(collections.Counter(cref.max_shared(t)[0] for t in teams).items())),
          dict(sorted(collections.Counter(pref.max_shared(t)[0] for t in teams).items())))
